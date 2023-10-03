"""Tests for the packet_ingest module"""
# Installed
from argparse import Namespace
import os
from cloudpathlib import AnyPath
import pytest
from sqlalchemy.exc import OperationalError
from bitstring import ReadError
# Local
from libera_utils.db import getdb
from libera_utils.db.database import DatabaseException
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.manifest import Manifest
from libera_utils.io.packet_ingest import ingest, cr_ingest, pds_ingest, IngestDuplicateError
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
                    create_mock_secret_manager, insert_single_pds_from_each_cr, generate_input_manifest_s3,
                    generate_input_manifest_local, test_type):
    """Test that cr_id was assigned properly to pds records and pds ingest
    time was assigned for records ingested separately"""

    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3()
    else:
        input_manifest_path = generate_input_manifest_local()

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)

    monkeypatch.setenv("SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")
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
    parsed_args = MockParsedArgsNamespace(AnyPath(generate_input_manifest_local()))
    m = Manifest.from_file(parsed_args.manifest_filepath)

    db_pds_dict, _ = cr_ingest(
        m.files[0], parsed_args.outdir)

    cr = ConstructionRecord.from_file(test_construction_record_09t00)

    with getdb().session() as s:
        cr_query_0 = s.query(Cr).filter(Cr.file_name == cr.file_name).all()
        assert "J01_G011_LZ_2021-04-09T00-00-00Z_V01" in str(cr_query_0)
        assert cr.pds_files_list[0].pds_filename in db_pds_dict.get \
            (AnyPath("J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS"))
        assert cr.pds_files_list[1].pds_filename in db_pds_dict.get \
            (AnyPath("J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS"))


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"]
)
def test_pds_ingest_time(clean_local_db, tmp_path, monkeypatch, generate_input_manifest_s3, create_mock_secret_manager,
                         generate_input_manifest_local, insert_single_pds_from_each_cr, test_type):
    """Test that pds ingest time is listed for records listed in manifest
    """
    filenames = ("P1590011AAAAAAAAAAAAAT21099051420500.PDS",
                 "P1590011AAAAAAAAAAAAAT21099051420501.PDS")
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3(*filenames)
    else:
        input_manifest_path = generate_input_manifest_local(*filenames)

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)

    monkeypatch.setenv("SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")

    ingest(parsed_args)
    m = Manifest.from_file(input_manifest_path)

    for file in m.files:
        filename = os.path.basename(file['filename'])

        with getdb().session() as s:
            pds_query = s.query(PdsFile).filter(
                PdsFile.file_name == filename).all()

        assert pds_query[0].ingested is not None


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"]
)
def test_output_manifest_to_input_manifest(clean_local_db, tmp_path, monkeypatch, create_mock_secret_manager,
                                           generate_input_manifest_local, generate_input_manifest_s3, test_type):
    """Test output manifest file created contains a list of the
    product files that the processing created
    """
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3()
    else:
        input_manifest_path = generate_input_manifest_local()

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)
    m = Manifest.from_file(input_manifest_path)

    monkeypatch.setenv("SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")

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
                                     generate_input_manifest_local, insert_single_pds_from_each_cr, test_type,
                                     create_mock_secret_manager):
    """Test output manifest file created does not contain pds records already inserted
    """
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3()
    else:
        input_manifest_path = generate_input_manifest_local()

    parsed_args = MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", processing_path)

    monkeypatch.setenv("SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")

    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    file_list = []

    for file in m_output.files:
        file_list.append(os.path.basename(file['filename']))

    assert 'P1590011AAAAAAAAAAAAAT21099051420500.PDS' not in file_list


def test_print_ingest_results(clean_local_db, test_construction_record_09t00, tmp_path, monkeypatch,
                              generate_input_manifest_local, insert_single_pds_from_each_cr,
                              create_mock_secret_manager):
    """Test that after calling ingest a query result prints as a representative of the entry."""

    # insert
    parsed_args = MockParsedArgsNamespace(str(generate_input_manifest_local()))
    monkeypatch.setenv("PROCESSING_DROPBOX", "/".join([str(tmp_path), '']))

    monkeypatch.setenv("SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")

    ingest(parsed_args)

    cr_0 = ConstructionRecord.from_file(test_construction_record_09t00)

    with getdb().session() as s:
        cr_query_0 = s.query(Cr).filter(Cr.file_name == cr_0.file_name).all()
        print(cr_query_0)
        assert "J01_G011_LZ_2021-04-09T00-00-00Z_V01" in str(cr_query_0)

# TODO Matt to fix the mock secret manager
@pytest.mark.xfail
def test_wrong_credentials(clean_local_db, tmp_path, monkeypatch_session, test_data_path,
                           generate_input_manifest_local, monkeypatch, create_mock_secret_manager):
    """Test that if connecting to db with wrong credentials, ingest will  throw errors"""

    # If this runs then all the subsequent calls to secret manager get this secret not the good one
    # That's weird and bad, but beyond the scope of this ticket and the secret manager makes this test
    # need to be different anyways.
    #monkeypatch.setenv("SECRET_NAME", "bad-test-secret")
    #create_mock_secret_manager("bad-test-secret", bad_login=True)

    parsed_args = MockParsedArgsNamespace(str(generate_input_manifest_local()))
    monkeypatch.setenv("PROCESSING_DROPBOX", "/".join([str(tmp_path), '']))
    processing_dropbox = os.environ['PROCESSING_DROPBOX']
    corrupt_cns = test_data_path / 'bad_record.CONS'
    corrupt = {'filename': corrupt_cns,
               'checksum': 12345678910}

    with pytest.raises((DatabaseException, OperationalError)):
        ingest(parsed_args)
    with pytest.raises((DatabaseException, OperationalError)):
        cr_ingest(corrupt, processing_dropbox)
    with pytest.raises((DatabaseException, OperationalError)):
        pds_ingest(corrupt, processing_dropbox)


def test_duplicate_file(clean_local_db, tmp_path, monkeypatch,
                        generate_input_manifest_local, create_mock_secret_manager):
    """Test that duplicate files are handled correctly"""

    m = Manifest.from_file(generate_input_manifest_local())
    parsed_args = MockParsedArgsNamespace(str(m.filename))
    monkeypatch.setenv("PROCESSING_DROPBOX", "/".join([str(tmp_path), '']))
    processing_dropbox = os.environ['PROCESSING_DROPBOX']

    monkeypatch.setenv("SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")

    ingest(parsed_args)

    for file in m.files:
        if 'CONS' in file['filename']:
            with pytest.raises(IngestDuplicateError):
                cr_ingest(file, processing_dropbox)

    for file in m.files:
        # is there a next pds in the manifest
        if 'PDS' in file['filename']:
            with pytest.raises(IngestDuplicateError):
                pds_ingest(file, processing_dropbox)


def test_corrupt_cons(clean_local_db, test_data_path, tmp_path):
    """Test that corrupt construction records will not be ingested"""

    corrupt_cons = test_data_path / 'bad_record.CONS'
    corrupt = {'filename': corrupt_cons,
               'checksum': 12345678910}

    with pytest.raises(ReadError):
        cr_ingest(file=corrupt, output_dir=tmp_path)
