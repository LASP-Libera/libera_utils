"""Tests for the pds_ingest module"""
# Installed
import os
import pytest
from cloudpathlib import AnyPath
# Local
from libera_utils.db import getdb
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.manifest import Manifest
from libera_utils.io.pds_ingest import ingest, cr_ingest, pds_ingest
from libera_utils.io.pds import ConstructionRecord, PDSRecord
from libera_utils.io.smart_open import smart_open


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
def test_manifest_ingest_cr_and_pds(clean_local_db, test_type,
                                    setup_l0_ingest_environment_with_manifest,
                                    test_construction_record_1, test_construction_record_2,
                                    test_pds_file_1, test_pds_file_2):
    """Test that cr_id was assigned properly to pds records and pds ingest
    time was assigned for records ingested separately from a manifest with 2
    construction records each with 1 associated pds file."""
    parsed_args = setup_l0_ingest_environment_with_manifest
    # insert
    ouput_manifest_filepath = ingest(parsed_args)

    cr_1 = ConstructionRecord.from_file(test_construction_record_1)
    cr_2 = ConstructionRecord.from_file(test_construction_record_2)
    pds_1 = PDSRecord.from_filename(test_pds_file_1)
    pds_2 = PDSRecord.from_filename(test_pds_file_2)

    with getdb().session() as s:
        cr_query_1 = s.query(Cr).filter(Cr.file_name == cr_1.file_name).all()
        cr_query_2 = s.query(Cr).filter(Cr.file_name == cr_2.file_name).all()

    with getdb().session() as s:
        pds_query_1 = s.query(PdsFile).filter(
            PdsFile.file_name == pds_1.filename).all()
        pds_query_2 = s.query(PdsFile).filter(
            PdsFile.file_name == pds_2.filename).all()

    # Cr ingested successfully
    assert cr_query_1[0].ingested is not None
    assert cr_query_2[0].ingested is not None

    # Proper cr_id assignment
    assert cr_query_1[0].id == pds_query_1[0].cr_id
    assert cr_query_2[0].id == pds_query_2[0].cr_id

    # Pds records ingested separately
    assert pds_query_1[0].ingested is not None
    assert pds_query_2[0].ingested is not None

    # Ensure output manifest exists using smart open
    assert smart_open(ouput_manifest_filepath) is not None


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
def test_output_manifest_from_input_manifest(clean_local_db, test_type,
                                             setup_l0_ingest_environment_with_manifest):
    """Test output manifest files are a list of the files from the input manifest"""
    parsed_args = setup_l0_ingest_environment_with_manifest
    m_input = Manifest.from_file(parsed_args.manifest_filepath)

    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    # Check that the output manifest has the same created time as the input manifest
    assert m_output.filename.filename_parts.created_time == m_input.filename.filename_parts.created_time

    # Check that input and output files match
    output_files = []
    input_files = []
    for file in m_input.files:
        input_files.append(os.path.basename(file['filename']))
    for file in m_output.files:
        output_files.append(os.path.basename(file['filename']))

    assert input_files == output_files


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
def test_output_manifest_duplicate_pds(clean_local_db, test_type, test_pds_file_1,
                                       setup_l0_ingest_environment_with_manifest):
    """Test output manifest file created does not contain pds records already inserted
    """
    # Start by ingesting a single PDS file into the DB
    pds_ingest(test_pds_file_1)

    # Ingest from a base manifest with 2 CR and 2 PDS (1 is a duplicate)
    parsed_args = setup_l0_ingest_environment_with_manifest
    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    # Check that the ouput files don't include the duplicate file
    file_list = []
    for file in m_output.files:
        file_list.append(os.path.basename(file['filename']))
    assert len(file_list) == 3
    assert test_pds_file_1.name not in file_list
    # TODO make sure the output manifest has an error logged
    # assert m_output.errors is not None


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
def test_output_manifest_duplicate_cr(clean_local_db, test_type, test_construction_record_1,
                                      setup_l0_ingest_environment_with_manifest):
    """Test output manifest file created does not contain pds records already inserted
    """
    # Start by ingesting a single PDS file into the DB
    cr_ingest(test_construction_record_1)

    # Ingest from a base manifest with 2 CR and 2 PDS (1 is a duplicate)
    parsed_args = setup_l0_ingest_environment_with_manifest
    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    # Check that the ouput files don't include the duplicate file
    file_list = []
    for file in m_output.files:
        file_list.append(os.path.basename(file['filename']))
    assert len(file_list) == 3
    assert test_construction_record_1.name not in file_list
    # TODO make sure the output manifest has an error logged for the duplicate file
    # assert m_output.errors is not None


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
def test_output_manifest_path(clean_local_db, test_type, setup_l0_ingest_environment_with_manifest):
    """Test that the ingested files are moved to the dropbox and the manifest path for files is in the processing
    dropbox path."""
    parsed_args = setup_l0_ingest_environment_with_manifest
    m_input = Manifest.from_file(parsed_args.manifest_filepath)
    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    for input_file in m_input.files:
        for file in m_output.files:
            output_file_path = AnyPath(file["filename"])
            assert "processing" in str(output_file_path.parent)
            assert input_file["filename"] != file["filename"]
