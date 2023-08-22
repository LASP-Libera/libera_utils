"""Tests for the packet_ingest module"""
# Installed
from argparse import Namespace
from cloudpathlib import S3Path, AnyPath
import os
import sys
import pytest
# Local
from libera_utils.db import getdb
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.packet_ingest import ingest, cr_ingest
from libera_utils.io.construction_record import ConstructionRecord, PDSRecord


class MockParsedArgsNamespace(Namespace):
    """ Generates dummy parser """
    def __init__(self, manifest_filepath, short_tmp_path=None):
        super().__init__()
        self.manifest_filepath = str(manifest_filepath)
        self.outdir = str(short_tmp_path)
        self.delete = False
        self.verbose = False


@pytest.fixture
def generate_input_manifest_local(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data"""

    filenames = (test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS",
                 test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.CONS",
                 test_data_path / "P1590011AAAAAAAAAAAAAT21099051420500.PDS",
                 test_data_path / "P1590011AAAAAAAAAAAAAT21099051420501.PDS")

    input_manifest = Manifest(ManifestType.INPUT, files=[], configuration={})

    input_manifest.add_files(*filenames)

    os.mkdir(tmp_path / "processing")
    input_manifest_file_path = input_manifest.write(outpath=tmp_path / "processing",
                                                    filename='libera_input_manifest_20230102t112233.json')

    return input_manifest_file_path


@pytest.fixture
def generate_input_manifest_s3(test_data_path, create_mock_bucket, write_file_to_s3):
    """Generating test manifest from the data in test_data"""
    r_bucket = create_mock_bucket()

    input_manifest = Manifest(ManifestType.INPUT, files=[], configuration={})

    filenames = ("J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS",
                 "J01_G011_LZ_2021-04-09T02-00-00Z_V01.CONS",
                 "P1590011AAAAAAAAAAAAAT21099051420500.PDS",
                 "P1590011AAAAAAAAAAAAAT21099051420501.PDS")

    for filename in filenames:
        s3_file_path = f"s3://{r_bucket.name}/{filename}"
        local_path = test_data_path / filename
        write_file_to_s3(local_path, s3_file_path)
        input_manifest.add_files(s3_file_path)

    d_bucket = create_mock_bucket()
    input_manifest_file_path = input_manifest.write(outpath=f"s3://{d_bucket.name}/processing")

    return input_manifest_file_path


@pytest.fixture
def insert_single_pds_from_each_cr(clean_local_db,
                                   test_construction_record_09t00,
                                   test_construction_record_09t02):
    """
    insert single pds from each cr
    """

    cr_0 = ConstructionRecord.from_file(test_construction_record_09t00)
    cr_1 = ConstructionRecord.from_file(test_construction_record_09t02)

    # insert P1590011AAAAAAAAAAAAAT21099051420500.PDS
    pds_0 = PDSRecord(cr_0.pds_files_list[0].pds_filename)
    pds_0_orm = pds_0.to_orm()

    # insert P1590011AAAAAAAAAAAAAT21099065436900.PDS
    pds_1 = PDSRecord(cr_1.pds_files_list[0].pds_filename)
    pds_1_orm = pds_1.to_orm()

    with getdb().session() as s:
        s.add(pds_0_orm)
        s.add(pds_1_orm)


@pytest.fixture
def insert_multiple_pds_from_single_cr(clean_local_db,
                                       test_construction_record_09t00):
    """
    Insert multiple pds from a single cr
    """

    cr = ConstructionRecord.from_file(test_construction_record_09t00)

    # insert P1590011AAAAAAAAAAAAAT21099051420500.PDS
    pds = PDSRecord(cr.pds_files_list[0].pds_filename)
    pds_0_orm = pds.to_orm()

    # insert P1590011AAAAAAAAAAAAAT21099051420501.PDS
    pds = PDSRecord(cr.pds_files_list[1].pds_filename)
    pds_1_orm = pds.to_orm()

    with getdb().session() as s:
        s.add(pds_0_orm)
        s.add(pds_1_orm)


@pytest.fixture
def insert_output(clean_local_db, test_construction_record_09t00,
                  test_construction_record_09t02):
    """
    Insert output records
    """
    cr_0 = ConstructionRecord.from_file(test_construction_record_09t00)
    cr_1 = ConstructionRecord.from_file(test_construction_record_09t02)

    cr_orm_0 = cr_0.to_orm()
    cr_orm_1 = cr_1.to_orm()

    with getdb().session() as s:
        s.add(cr_orm_0)
        s.add(cr_orm_1)


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"]
)
def test_pds_ingest(clean_local_db, test_construction_record_09t00,
                    test_construction_record_09t02, tmp_path, monkeypatch,
                    insert_single_pds_from_each_cr, generate_input_manifest_s3,
                    generate_input_manifest_local, test_type):
    """Test that cr_id was assigned properly to pds records and pds ingest
    time was assigned for records ingested separately"""

    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3
    else:
        input_manifest_path = generate_input_manifest_local

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)

    # insert
    ingest(parsed_args)

    cr_0 = ConstructionRecord.from_file(test_construction_record_09t00)
    cr_1 = ConstructionRecord.from_file(test_construction_record_09t02)

    with getdb().session() as s:
        cr_query_0 = s.query(Cr).filter(Cr.file_name == cr_0.file_name).all()
        cr_query_1 = s.query(Cr).filter(Cr.file_name == cr_1.file_name).all()

    with getdb().session() as s:
        pds_query_0 = s.query(PdsFile).filter(
            PdsFile.file_name == cr_0.pds_files_list[0].pds_filename).all()
        pds_query_1 = s.query(PdsFile).filter(
            PdsFile.file_name == cr_1.pds_files_list[0].pds_filename).all()

    # proper cr_id assignment
    assert cr_query_0[0].id == pds_query_0[0].cr_id
    assert cr_query_1[0].id == pds_query_1[0].cr_id
    # pds records ingested separately
    assert pds_query_0[0].ingested is not None
    assert pds_query_1[0].ingested is not None


def test_cr_ingest_pds_values(clean_local_db, test_construction_record_09t00,
                              monkeypatch, generate_input_manifest_local, generate_input_manifest_s3,
                              insert_multiple_pds_from_single_cr):
    """Test that cr_ingest returns expected values when multiple pds records
    from a single cr are already present in the database"""

    # read json information
    parsed_args = MockParsedArgsNamespace(AnyPath(generate_input_manifest_local))
    m = Manifest.from_file(parsed_args.manifest_filepath)
    db_pds_dict, _ = cr_ingest(
        m.files[0], parsed_args.outdir)

    cr = ConstructionRecord.from_file(test_construction_record_09t00)

    with getdb().session() as s:
        cr_query_0 = s.query(Cr).filter(Cr.file_name == cr.file_name).all()
        assert str(cr_query_0).__contains__("J01_G011_LZ_2021-04-09T00-00-00Z_V01")
        assert [cr.pds_files_list[0].pds_filename, cr.pds_files_list[1].pds_filename] \
               in db_pds_dict.values()



@pytest.mark.parametrize(
    "test_type", ["S3", "Local"]
)
def test_pds_ingest_time(clean_local_db, tmp_path, monkeypatch, generate_input_manifest_s3,
                         generate_input_manifest_local, insert_single_pds_from_each_cr, test_type):
    """Test that pds ingest time is listed for records listed in manifest
    """

    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3
    else:
        input_manifest_path = generate_input_manifest_local

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)

    ingest(parsed_args)
    m = Manifest.from_file(input_manifest_path)

    for file in m.files:
        if 'PDS' in file['filename']:
            filename = os.path.basename(file['filename'])

            with getdb().session() as s:
                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == filename).all()

            assert pds_query[0].ingested is not None


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"]
)
def test_output_manifest_to_input_manifest(clean_local_db, tmp_path, monkeypatch,
                                           generate_input_manifest_local, generate_input_manifest_s3, test_type):
    """Test output manifest file created contains a list of the
    product files that the processing created
    """
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3
    else:
        input_manifest_path = generate_input_manifest_local

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)
    m = Manifest.from_file(input_manifest_path)

    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    # Check that the output manifest path exists
    assert m_output.filename.path.exists()

    # Check that the output manifest has the same created time as the input manifest
    assert m_output.filename.filename_parts.created_time == m.filename.filename_parts.created_time

    input_files = []
    output_files = []

    for file in m.files:
        input_files.append(os.path.basename(file['filename']))
    for file in m_output.files:
        output_files.append(os.path.basename(file['filename']))

    assert input_files == output_files


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"]
)
def test_output_manifest_correct_pds(clean_local_db, tmp_path, monkeypatch, generate_input_manifest_s3,
                                     generate_input_manifest_local, insert_single_pds_from_each_cr, test_type):
    """Test output manifest file created does not contain pds records already inserted
    """
    parsed_args = MockParsedArgsNamespace(str(generate_input_manifest_local))
    monkeypatch.setenv("PROCESSING_DROPBOX", str(tmp_path))

    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3
    else:
        input_manifest_path = generate_input_manifest_local

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)
    m = Manifest.from_file(input_manifest_path)

    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    file_list = []

    for file in m_output.files:
        file_list.append(os.path.basename(file['filename']))

    assert 'P1590011AAAAAAAAAAAAAT21099051420500.PDS' not in file_list


def test_print_ingest_results(clean_local_db, test_construction_record_09t00, tmp_path, monkeypatch,
                              generate_input_manifest_local, insert_single_pds_from_each_cr):
    """Test that after calling ingest a query result prints as a representative of the entry."""

    # insert
    parsed_args = MockParsedArgsNamespace(str(generate_input_manifest_local))
    monkeypatch.setenv("PROCESSING_DROPBOX", "/".join([str(tmp_path),'']))
    ingest(parsed_args)

    cr_0 = ConstructionRecord.from_file(test_construction_record_09t00)

    with getdb().session() as s:
        cr_query_0 = s.query(Cr).filter(Cr.file_name == cr_0.file_name).all()
        print(cr_query_0)
        assert str(cr_query_0).__contains__("J01_G011_LZ_2021-04-09T00-00-00Z_V01")
