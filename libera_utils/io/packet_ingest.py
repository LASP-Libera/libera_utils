"""Module for packet ingest"""
# Standard
import argparse
import datetime
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

    processing_dropbox = os.environ['PROCESSING_DROPBOX']

    # read json information
    m = Manifest.from_file(parsed_args.manifest_filepath)
    m.validate_checksums()

    mfn = ManifestFilename.from_filename_parts(
        manifest_type=ManifestType.OUTPUT,
        created_time=datetime.datetime.utcnow())

    db_pds_dict_all = {}
    output_files = []

    for file in m.files:
        # is there a next cr in the manifest
        if 'CONS' in file['filename']:
            db_pds_dict, con_ingested_dict = cr_ingest(file, processing_dropbox)
            db_pds_dict_all.update(db_pds_dict)
            output_files.append(con_ingested_dict)

    for file in m.files:
        # is there a next pds in the manifest
        if 'PDS' in file['filename']:
            pds_ingested_dict = pds_ingest(file, processing_dropbox)
            if pds_ingested_dict:
                output_files.append(pds_ingested_dict)

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

    # write output manifest to L0 ingest dropbox
    output_dir = AnyPath(processing_dropbox)
    logger.info("Writing resulting output manifest to %s", output_dir)

    # Write output manifest file containing a list of the product files that the processing created
    output_manifest_path = os.path.join(output_dir, str(mfn))
    output_manifest = Manifest(manifest_type=ManifestType.OUTPUT,
                               filename=output_manifest_path,
                               files=output_files,
                               configuration={})

    # move files over
    for file in output_files:
        input_dir = os.path.join(os.path.dirname(m.files[0]['filename']),
                                 os.path.basename(file['filename']))
        smart_copy_file(input_dir, output_dir, delete=parsed_args.delete)

    output_manifest.write(output_manifest_path)
    logger.info("Algorithm complete. Exiting.")

    return output_manifest_path


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
    filename = os.path.basename(file['filename'])
    db_pds = []

    with getdb().session() as s:

        cr_query = s.query(Cr).filter(
            Cr.file_name == filename).all()

        # check if cr is in the db
        if not cr_query:

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

            # output filename
            output_dir = os.path.join(output_dir, filename)

            # create ingested dictionary
            ingested_dict = {"filename": output_dir,
                             "checksum": file['checksum']}
        else:
            logger.info("Duplicate cr: %s", filename)
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
    filename = os.path.basename(file['filename'])

    with getdb().session() as s:

        # check to see if pds is in db
        pds_query = s.query(PdsFile).filter(
            PdsFile.file_name == filename).all()

        # if pds is not in db then insert the pds file into the db
        # without associating it with a cr; set the ingest time
        if not pds_query:
            # parse pds into nested orm objects
            pds = PDSRecord(filename)
            pds_orm = pds.to_orm()
            s.add(pds_orm)

            # output filename
            output_dir = os.path.join(output_dir, filename)

            # create ingested dictionary
            ingested_dict = {"filename": output_dir,
                             "checksum": file['checksum']}
        # if pds is in db but does not have ingest time, update the ingest time
        elif pds_query[0].ingested is None:
            pds_query[0].ingested = datetime.datetime.utcnow()

            # output filename
            output_dir = os.path.join(output_dir, filename)

            # create ingested dictionary
            ingested_dict = {"filename": output_dir,
                             "checksum": file['checksum']}
        else:
            logger.info("Duplicate pd: %s", filename)
            ingested_dict = {}

    return ingested_dict
