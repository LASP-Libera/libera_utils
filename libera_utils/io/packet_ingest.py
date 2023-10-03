"""Module for packet ingest"""
# Standard
import argparse
from datetime import datetime
import logging
import os
import json
# Installed
from cloudpathlib import AnyPath
from sqlalchemy import func
from psycopg2 import DataError, ProgrammingError, OperationalError
from bitstring import ReadError
import boto3
# Local
from libera_utils.db import getdb
from libera_utils.db.database import DatabaseException
from libera_utils.io.construction_record import ConstructionRecord, PDSRecord
from libera_utils.io.manifest import Manifest
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


class IngestDuplicateError(Exception):
    """Custom Exception for ingesting a duplicate into the DB"""

    def __init__(self, message, file=None):
        self.file = file
        super().__init__(message)


def ingest(parsed_args: argparse.Namespace):
    # TODO fix this
    # pylint: disable=too-many-statements, too-many-branches
    """Ingest and update records into database using manifest
    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    output_manifest_path : str
        Path of output manifest
    """
    now = datetime.utcnow().strftime("%Y%m%dt%H%M%S")
    configure_task_logging(f'l0_packet_ingester_{now}',
                           app_package_name='libera_utils',
                           console_log_level=logging.DEBUG if parsed_args.verbose else None)

    processing_path = AnyPath(os.environ['PROCESSING_DROPBOX'])
    logger.debug(f"Processing dropbox set to {processing_path}")

    # Retrieves secrets to allow DB access
    secret_name = str(os.environ['SECRET_NAME'])
    logger.debug(f"Secret Name: {secret_name}")
    set_db_credentials_from_secret_manager(secret_name)

    # read json information
    logger.debug("Reading Manifest file")
    m = Manifest.from_file(parsed_args.manifest_filepath)
    m.validate_checksums()

    logger.info("Starting L0 packet ingester...")
    logger.debug(f"CLI args: {parsed_args}")

    pds_from_db = {}
    ingested_pds = {}
    ingested_records = []

    for file in m.files:
        # TODO: Use our filenaming.L0Filename to find valid CRs and PDS files.
        # is there a next cr in the manifest
        if 'CONS' in file['filename']:
            try:
                db_pds_dict, ingested_cons = cr_ingest(file, processing_path)
                pds_from_db.update(db_pds_dict)
            except IngestDuplicateError as error:
                logger.error(error)
            if ingested_cons:
                ingested_records.append(ingested_cons)

    for file in m.files:
        # is there a next pds in the manifest
        if 'PDS' in file['filename']:
            try:
                ingested_pds = pds_ingest(file, processing_path)
            except IngestDuplicateError as error:
                logger.error(error)
            if ingested_pds:
                ingested_records.append(ingested_pds)

    logger.debug(f"Files found in manifest: {ingested_records}")

    logger.info("Inserting files (CRs and PDS files) into database")
    # insert cr_id for pds files in the db associated with the current cr
    if pds_from_db:
        try:
            with getdb().session() as s:
                for cr_filename in pds_from_db.items():
                    # query cr_id that has been inserted
                    cr_query = s.query(Cr).filter(Cr.file_name == str(cr_filename[0])).all()
                    # query all pds associated with cr
                    pds_query = s.query(PdsFile).filter(
                        PdsFile.file_name == func.any(cr_filename[1])).all()
                    # assign cr_id
                    for pds in pds_query:
                        pds.cr_id = cr_query[0].id
        except (DatabaseException, DataError, ProgrammingError, OperationalError) as error:
            logger.error(error)

    # Create output manifest file containing a list of the product files that the processing created
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest=parsed_args.manifest_filepath)

    logger.info("Moving files from receiver bucket to dropbox in preparation for archiving")
    # move files over
    incoming_path = AnyPath(os.path.dirname(m.files[0]["filename"]))
    for file in ingested_records:
        # TODO: figure out what to do with duplicate files (delete, rename, etc)
        # this could be fasly if the ingested dictionary this is returned is empty
        if not file:
            logger.info("Duplicate files.")
        else:
            current_file_location = incoming_path / os.path.basename(file['filename'])
            destination_location = processing_path / os.path.basename(file['filename'])
            smart_copy_file(current_file_location, destination_location,
                            delete=parsed_args.delete)

        output_manifest.add_files(file["filename"])

    # write output manifest to L0 ingest dropbox
    output_dir = processing_path
    logger.info(f"Writing resulting output manifest to {output_dir}")

    output_manifest.write(output_dir)

    logger.info("L0 ingest algorithm complete. Exiting.")
    return str(output_manifest.filename.path)


def cr_ingest(file: dict, output_dir: str):
    """Ingest cr records into database
    Parameters
    ----------
    file : Dictionary
        Dictionary containing path and checksum of cr
    output_dir : str
        Directory for output data

    Returns
    -------
    db_pds_dict : Dictionary
        Dictionary that associates the pds file in the db with the current cr
    ingested_dict : Dictionary
        Dictionary of records that have been ingested
    """
    filename = AnyPath(os.path.basename(file['filename']))
    logger.info(f"Ingesting construction record {file}")
    pds_files_from_db = []
    ingested_dict = {}
    db_pds_dict = {}

    try:
        with getdb().session() as s:

            cr_query = s.query(Cr).filter(
                Cr.file_name == str(filename)).all()

            # check if cr is in the db
            if not cr_query:
                logger.debug(f"Detected a new CR file {filename}. Parsing and inserting data.")
                # parse cr into nested orm objects
                cr = ConstructionRecord.from_file(file['filename'])

                if not cr.pds_files_list:
                    curr_pds_from_db = {}
                else:
                    curr_pds_from_db = {f.pds_filename: f for f in cr.pds_files_list}

                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == func.any(list(curr_pds_from_db.keys()))).all()

                # if there are some pds records from the current cr in the db
                # associate them with current cr, but do not set pds ingest time
                if pds_query:

                    for pds_object in pds_query:
                        pds_files_from_db.append(pds_object.file_name)
                        logger.info("In database: %s", pds_object.file_name)

                        if pds_object.file_name in list(curr_pds_from_db):
                            cr.pds_files_list.remove(
                                curr_pds_from_db[pds_object.file_name])

                cr_orm = cr.to_orm()
                s.merge(cr_orm)

                # create ingested dictionary
                ingested_dict = {"filename": output_dir / filename,
                                 "checksum": file['checksum']}

            else:
                raise IngestDuplicateError(f"Duplicate Construction record: {filename}", file)

            # for the pds files that were already in the db,
            # associate the pds file in the db with the current cr
            if pds_files_from_db:
                # TODO: This filename is a PosixPath. Should it be a string?
                db_pds_dict = {filename: pds_files_from_db}
            else:
                db_pds_dict = {}

    except OperationalError as error:
        logger.error(error)
        raise error
    except ReadError as error:
        logger.error(f'Bad construction record read. Is this a bad CNS record? Error given was: {error}')
        raise error

    return db_pds_dict, ingested_dict


def pds_ingest(file: dict, output_dir: AnyPath):
    """Ingest pd records into database that do not have an associated cr
    Parameters
    ----------
    file : Dictionary
        Dictionary containing path and checksum of pd
    output_dir : str
        Directory for output data

    Returns
    -------
    ingested_dict : Dictionary
        Dictionary of records that have been ingested
    """
    logger.info(f"Ingesting PDS file {file}")
    filename = os.path.basename(file['filename'])
    ingested_dict = {}

    try:
        with getdb().session() as s:

            # check to see if pds is in db
            pds_query = s.query(PdsFile).filter(
                PdsFile.file_name == filename).all()

            # if pds is not in db then insert the pds file into the db
            # without associating it with a cr; set the ingest time
            if not pds_query:
                logger.debug(f"{filename} not found in DB. Inserting new record")
                # parse pds into nested orm objects
                pds = PDSRecord(filename)
                pds_orm = pds.to_orm()
                s.add(pds_orm)

                # create ingested dictionary
                ingested_dict = {"filename": output_dir / filename,
                                 "checksum": file['checksum']}
            # if pds is in db but does not have ingest time, update the ingest time
            elif pds_query[0].ingested is None:
                logger.debug(f"{filename} found in the DB but it is lacking an ingest time. This is likely because "
                             "it was listed in a previous CR file.")
                pds_query[0].ingested = datetime.utcnow()

                # create ingested dictionary
                ingested_dict = {"filename": output_dir / filename,
                                 "checksum": file['checksum']}
            # TODO: for future endeavors, check the archive time. But it is not needed currently
            # create elif statements
            # elif pds_query[0].ingested and not pds_query[0].archived:
            #     ingested_dict = {"filename": os.path.join(output_dir, filename),
            #                      "checksum": file['checksum']}
            else:
                raise IngestDuplicateError(f"Duplicate PDS file: {filename}", file)

    except OperationalError as error:
        logger.error(error)
        raise error

    return ingested_dict


def set_db_credentials_from_secret_manager(secret_name: str):
    """Set Environment Variables for RDS access
    Parameters
    ----------
    secret_name : str
        The name of the secret in the Secrets Manager to access.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name="us-west-2")
    secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )
    secret_object = json.loads(secret_value_response['SecretString'])

    os.environ["LIBERA_DB_HOST"] = secret_object["host"]
    os.environ["LIBERA_DB_USER"] = secret_object["username"]
    os.environ["LIBERA_DB_NAME"] = secret_object["dbname"]
    os.environ["PGPASSWORD"] = secret_object["password"]
    logger.debug("Secret loaded and stored as environment variables.")
