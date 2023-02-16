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
from libera_utils.io.construction_record import ConstructionRecord, PDSFiles
from libera_utils.io.manifest import Manifest, ManifestType, ManifestFilename
from libera_utils.db.models import *


logger = logging.getLogger(__name__)


def ingest(parsed_args: argparse.Namespace):
    """Ingest and update records into database using manifest
    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """

    # read json information
    m = Manifest.from_file(parsed_args.manifest_filepath)

    mfn = ManifestFilename.from_filename_parts(
        manifest_type=ManifestType.OUTPUT,
        created_time=datetime.datetime.utcnow())

    db_pds_dict={}
    output_files=[]

    for file in m.files:

        # is there a next cr in the manifest
        if 'CONS' in file['filename']:
            dicts,con_ingested_dict = cr_ingest(file)
            db_pds_dict.update(dicts)
            output_files.append(con_ingested_dict)

        # is there a next pds in the manifest
        if 'PDS' in file['filename']:
            pds_ingested_dict = pds_ingest(file)
            output_files.append(pds_ingested_dict)

    # insert cr_id for pds files in the db associated with the current cr
    if db_pds_dict:
        with getdb().session() as s:
            for cr_filename in db_pds_dict.keys():
                # query cr_id that has been inserted
                cr_query = s.query(Cr).filter(Cr.file_name == cr_filename).all()
                # query all pds associated with cr
                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == func.any(
                        db_pds_dict[cr_filename])).all()
                # assign cr_id
                for pds in pds_query:
                    pds.cr_id = cr_query[0].id

    # write output manifest to L0 ingest dropbox
    output_dir = AnyPath(parsed_args.outdir)
    logger.info("Writing resulting output manifest to %s", output_dir)

    # Write output manifest file containing a list of the product files that the processing created
    output_manifest_path = os.path.join(output_dir, str(mfn))
    output_manifest = Manifest(manifest_type=ManifestType.OUTPUT,
                               filename=output_manifest_path,
                               files=output_files,
                               configuration={})
    output_manifest.write(output_manifest_path)
    logger.info("Algorithm complete. Exiting.")


def cr_ingest(file: str):
    """Ingest cr records into database using manifest
    Parameters
    ----------
    file : Dictionary
        Dictionary containing path and checksum of cr

    Returns
    -------
    None
    """
    filename = os.path.basename(file['filename'])
    dicts = {}

    with getdb().session() as s:

        cr_query = s.query(Cr).filter(
            Cr.file_name == filename).all()

        # check if cr is in the db
        if not cr_query:
            pds_filename = []

            # parse cr into nested orm objects
            cr = ConstructionRecord.from_file(file['filename'])

            # search db for next pds record contained in cr
            for pds_file in cr.pds_files_list:
                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == pds_file.pds_filename).all()

                # pds records missing from the db
                if pds_query:
                    pds_filename.append(pds_file.pds_filename)

            # if there are some pds records from the current cr in the db
            # create all pds file records and associate them with current cr,
            # but do not set pds ingest time
            cr = ConstructionRecord.from_file(
                file['filename'], pds_excluded=pds_filename)
            cr_orm = cr.to_orm()
            s.merge(cr_orm)

            # create ingested dictionary
            ingested_dict = {"filename": file['filename'],
                             "checksum": file['checksum']}
        else:
            logger.info("Duplicate cr: %s", filename)
            ingested_dict = {}

    # for the pds files that were already in the db,
    # associate the pds file in the db with the current cr
    if pds_filename:
        dicts[filename] = pds_filename

    return dicts, ingested_dict


def pds_ingest(file: str):
    """Ingest pd records into database using manifest
    Parameters
    ----------
    file : Dictionary
        Dictionary containing path and checksum of pd

    Returns
    -------
    None
    """
    filename = os.path.basename(file['filename'])

    with getdb().session() as s:

        # check to see if pds is in db
        pds_query = s.query(PdsFile).filter(
            PdsFile.file_name == filename).all()

        # if pds is not in db insert then insert the pds file into the db
        # without associating it with a cr; set the ingest time
        if not pds_query:
            # parse pds into nested orm objects
            pds = PDSFiles(filename)
            pds_orm = pds.to_orm()
            s.add(pds_orm)

            #create ingested dictionary
            ingested_dict = {"filename": file['filename'],
                             "checksum": file['checksum']}
        # if pds is in db but does not have ingest time, update the ingest time
        elif pds_query[0].ingested is None:
            pds_query[0].ingested = datetime.datetime.utcnow()

            # create ingested dictionary
            ingested_dict = {"filename": file['filename'],
                             "checksum": file['checksum']}
        else:
            logger.info("Duplicate pd: %s", filename)
            ingested_dict = {}

    return ingested_dict
