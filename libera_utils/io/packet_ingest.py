"""Module for packet ingest"""
# Standard
import argparse
from datetime import datetime
import logging
import os
# Installed
from cloudpathlib import AnyPath
from sqlalchemy import func
# Local
from libera_utils.db import getdb
from libera_utils.io.construction_record import ConstructionRecord, PDSRecord
from libera_utils.io.manifest import Manifest, ManifestType, ManifestFilename
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging


logger = logging.getLogger(__name__)


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
    configure_task_logging(f'l0_packet_ingester_{now}',
                           app_package_name='libera_utils',
                           console_log_level=logging.DEBUG if parsed_args.verbose else None)

    logger.info("Starting L0 packet ingester...")
    logger.debug(f"CLI args: {parsed_args}")

    processing_dropbox = os.environ['PROCESSING_DROPBOX']
    logger.debug(f"Processing dropbox: {processing_dropbox}")

    # read json information
    m = Manifest.from_file(parsed_args.manifest_filepath)
    m.validate_checksums()

    db_pds_dict_all = {}
    output_files = []

    for file in m.files:
        # TODO: Use our filenaming.L0Filename to find valid CRs and PDS files.
        # is there a next cr in the manifest
        if 'CONS' in file['filename']:
            db_pds_dict, con_ingested_dict = cr_ingest(file, processing_dropbox)
            db_pds_dict_all.update(db_pds_dict)
            if con_ingested_dict:
                output_files.append(con_ingested_dict)

    for file in m.files:
        # is there a next pds in the manifest
        if 'PDS' in file['filename']:
            pds_ingested_dict = pds_ingest(file, processing_dropbox)
            if pds_ingested_dict:
                output_files.append(pds_ingested_dict)

    logger.debug(f"Files found in manifest: {output_files}")

    logger.info("Inserting files (CRs and PDS files) into database")
    # insert cr_id for pds files in the db associated with the current cr
    if db_pds_dict_all:
        with getdb().session() as s:
            for cr_filename in db_pds_dict_all.items():
                # query cr_id that has been inserted
                cr_query = s.query(Cr).filter(Cr.file_name == cr_filename[0]).all()
                # query all pds associated with cr
                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == func.any(cr_filename[1])).all()
                # assign cr_id
                for pds in pds_query:
                    pds.cr_id = cr_query[0].id

    logger.info("Moving files from receiver bucket to dropbox in preparation for archiving")
    # move files over
    for file in output_files:

        # TODO: figure out what to do with duplicate files (delete, rename, etc)
        # TODO: Gavin wonders how file could ever be falsy?
        if not file:
            logger.info("Duplicate files.")
        else:
            input_dir = os.path.join(os.path.dirname(m.files[0]['filename']),
                                     os.path.basename(file['filename']))
            smart_copy_file(input_dir, os.path.join(processing_dropbox, os.path.basename(file['filename'])),
                            delete=parsed_args.delete)

    # write output manifest to L0 ingest dropbox
    output_manifest_filename = ManifestFilename.from_filename_parts(
        manifest_type=ManifestType.OUTPUT,
        created_time=m.filename.filename_parts.created_time)

    # Write output manifest file containing a list of the product files that the processing created
    output_manifest = Manifest(manifest_type=ManifestType.OUTPUT,
                               filename=output_manifest_filename,
                               files=output_files,
                               configuration={})
    output_manifest_path = output_manifest.write(outpath=AnyPath(processing_dropbox))
    logger.info("L0 ingest algorithm complete. Exiting.")

    return str(output_manifest_path)


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
    logger.info(f"Ingesting construction record {file}")
    filename = os.path.basename(file['filename'])
    db_pds = []

    with getdb().session() as s:

        cr_query = s.query(Cr).filter(
            Cr.file_name == filename).all()

        # check if cr is in the db
        if not cr_query:
            logger.debug(f"Detected a new CR file {filename}. Parsing and inserting data.")
            # parse cr into nested orm objects
            cr = ConstructionRecord.from_file(file['filename'])

            if not cr.pds_files_list:
                all_pds_dict = {}
            else:
                all_pds_dict = {f.pds_filename: f for f in cr.pds_files_list}

            pds_query = s.query(PdsFile).filter(
                PdsFile.file_name == func.any(list(all_pds_dict.keys()))).all()

            # if there are some pds records from the current cr in the db
            # associate them with current cr, but do not set pds ingest time
            if pds_query:

                for i, pds_object in enumerate(pds_query):
                    db_pds.append(pds_query[i].file_name)
                    logger.info("In database: %s", pds_object.file_name)

                    if pds_query[i].file_name in list(all_pds_dict):
                        cr.pds_files_list.remove(
                            all_pds_dict[pds_query[i].file_name])

            cr_orm = cr.to_orm()
            s.merge(cr_orm)

            # create ingested dictionary
            ingested_dict = {"filename": os.path.join(output_dir, filename),
                             "checksum": file['checksum']}
        else:
            logger.info(f"Duplicate CR {filename} (in DB and has ingest time). Skipping insert.")
            ingested_dict = {}

    # for the pds files that were already in the db,
    # associate the pds file in the db with the current cr
    if db_pds:
        db_pds_dict = {filename: db_pds}
    else:
        db_pds_dict = {}

    return db_pds_dict, ingested_dict


def pds_ingest(file: dict, output_dir: str):
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
            ingested_dict = {"filename": os.path.join(output_dir, filename),
                             "checksum": file['checksum']}
        # if pds is in db but does not have ingest time, update the ingest time
        elif pds_query[0].ingested is None:
            logger.debug(f"{filename} found in the DB but it is lacking an ingest time. This is likely because "
                         "it was listed in a previous CR file.")
            pds_query[0].ingested = datetime.utcnow()

            # create ingested dictionary
            ingested_dict = {"filename": os.path.join(output_dir, filename),
                             "checksum": file['checksum']}
        else:
            logger.info(f"Duplicate PDS file {filename} (in the DB and has an ingest time). Skipping DB insert.")
            ingested_dict = {}

    return ingested_dict
