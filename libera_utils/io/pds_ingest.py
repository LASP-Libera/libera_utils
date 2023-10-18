"""Module for L0 file ingest"""
# Standard
import argparse
from datetime import datetime
import logging
import os
# Installed
from cloudpathlib import AnyPath
from sqlalchemy import func
from psycopg2 import OperationalError
# Local
from libera_utils.db import getdb
from libera_utils.io.pds import ConstructionRecord, PDSRecord
from libera_utils.io.manifest import Manifest
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging
from libera_utils.io.filenaming import L0Filename
from libera_utils.db.database_utils import set_db_credentials_from_secret_manager

logger = logging.getLogger(__name__)


class IngestDuplicateError(Exception):
    """Custom Exception for ingesting a duplicate into the DB"""

    def __init__(self, message, filename=None):
        self.filename = filename
        super().__init__(message)


def ingest(parsed_args: argparse.Namespace):
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
    configure_task_logging(f'l0_ingest_{now}',
                           app_package_name='libera_utils',
                           console_log_level=logging.DEBUG if parsed_args.verbose else None)
    logger.debug(f"CLI args: {parsed_args}")

    processing_dropbox_path = AnyPath(os.environ['PROCESSING_DROPBOX'])
    logger.debug(f"Processing dropbox set to {processing_dropbox_path}")

    # Retrieves secrets to allow DB access
    db_secret_name = str(os.environ['DB_SECRET_NAME'])
    logger.debug(f"Database Secret Name: {db_secret_name}")
    set_db_credentials_from_secret_manager(db_secret_name)

    # read json information
    logger.debug("Reading Manifest file")
    m = Manifest.from_file(parsed_args.manifest_filepath)
    m.validate_checksums()

    logger.info("Starting L0 ingest...")

    ingested_files = []
    for file in m.files:
        try:
            # TODO: Consider more error handling
            filepath = AnyPath(file["filename"])
            l0_file_name = L0Filename(filepath)
            if not filepath.is_absolute():
                raise ValueError(f"File path {filepath} is not an absolute filepath")
            if l0_file_name.filename_parts.file_number == 0:
                cr_ingest(filepath)
                ingested_files.append(filepath)
            else:
                pds_ingest(filepath)
                ingested_files.append(filepath)
        except IngestDuplicateError as error:
            # TODO what should we do with these files? Move them to the l0 dropbox? Keep them in the receiver bucket?
            logger.debug(f"The file {error.filename} already exists in the the DB and will not be included")
        except Exception as unhandled:
            logger.exception(unhandled)
            raise

    logger.debug(f"Files ingested from manifest: {ingested_files}")

    # Create output manifest file containing a list of the product files that the processing created
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest=parsed_args.manifest_filepath)

    logger.info("Moving files from receiver bucket to dropbox as output data products")
    # move files to output directory
    for filepath in ingested_files:
        destination_location = processing_dropbox_path / os.path.basename(filepath)
        smart_copy_file(filepath, destination_location,
                        delete=parsed_args.delete)

        output_manifest.add_files(destination_location)

    # write output manifest to L0 ingest dropbox
    logger.info(f"Writing resulting output manifest to {processing_dropbox_path}")

    output_manifest.write(processing_dropbox_path)

    logger.info("L0 ingest algorithm complete. Exiting.")
    return str(output_manifest.filename.path.absolute())


def cr_ingest(filename: str or AnyPath):
    """Ingest cr records into Postgres database
    Parameters
    ----------
    filename : str or AnyPath
        Filename of the construction record to be ingested
    """
    filename_only = AnyPath(filename).name
    logger.info(f"Ingesting construction record with filename: {filename_only}")
    try:
        with getdb().session() as s:
            cr_query = s.query(Cr).filter(
                Cr.file_name == str(filename_only)).all()

            # check if cr is in the db
            if not cr_query:
                logger.debug(f"Detected a new CR file {filename}. Parsing and inserting data.")
                # parse cr into nested orm objects
                cr = ConstructionRecord.from_file(filename)
                cr_orm = cr.to_orm()

                # If there are some pds records from the current cr in the db
                # associate them with current cr
                pds_filenames = [x.file_name for x in cr_orm.pds_files]
                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == func.any(pds_filenames)).all()

                for query_pds_file in pds_query:
                    logger.info(f"PDS file {query_pds_file.file_name} is already in the database")
                    # As the PDS file exists in the database then the PDSRecord needs to be linked by the foreign
                    # key to the construction record. The search below matches the pds file entry in the ORM model
                    # to the PDS query that was done and resets the ORM object id to match the query id.
                    # When the merge is done later this connection of id's will ensure new orm model for the
                    # construction record back populates the cr_id in the pds record already in the database.
                    for orm_pds_file in cr_orm.pds_files:
                        if orm_pds_file.file_name == query_pds_file.file_name:
                            orm_pds_file.id = query_pds_file.id

                s.merge(cr_orm)
            else:
                raise IngestDuplicateError(f"Duplicate Construction record: {filename}", filename)

    except OperationalError as error:
        logger.error(error)
        raise error


def pds_ingest(filename: str or AnyPath):
    """Ingest pd records into database that do not have an associated cr
    Parameters
    ----------
    filename : str or Any Path
        Filename of the PDS file to be ingested
    """
    logger.info(f"Ingesting PDS file {filename}")
    filename_only = AnyPath(filename).name

    try:
        with getdb().session() as s:
            # check to see if pds is in db
            pds_query = s.query(PdsFile).filter(
                PdsFile.file_name == filename_only).all()

            # if pds is not in db then insert the pds file into the db
            # without associating it with a cr; set the ingest time
            if not pds_query:
                logger.debug(f"{filename} not found in DB. Inserting new record")
                # parse pds into nested orm objects
                pds = PDSRecord.from_filename(filename_only)
                pds_orm = pds.to_orm()
                s.add(pds_orm)

            # if pds is in db but does not have ingest time, update the ingest time
            elif pds_query[0].ingested is None:
                logger.debug(f"{filename} found in the DB but it is lacking an ingest time. This is because "
                             "it was listed in a previous CR file.")
                pds_query[0].ingested = datetime.utcnow()
            # TODO this doesn't work with our unit test framework. Think more about this and testing it.
            #elif pds_query[0].ingested and not pds_query[0].archived:
            #    logger.debug(f"{filename} found in DB with an ingest time and no archive time."
            #                 f"this is an issue but not the responsibility of this pds ingest to raise.")
            else:
                logger.info(f"Duplicate PDS file {filename} (in the DB and has an ingest time). Skipping DB insert.")
                raise IngestDuplicateError(f"Duplicate PDS file: {filename}", filename)

    except OperationalError as error:
        logger.exception(error)
        raise
