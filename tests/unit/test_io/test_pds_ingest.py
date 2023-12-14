"""Tests for the pds_ingest module"""
import os
# Installed
import pytest
from bitstring import ReadError
from sqlalchemy.exc import OperationalError
# Local
from libera_utils.db import getdb
from libera_utils.db.database import DatabaseException
from libera_utils.db.models import Cr, PdsFile
from libera_utils.io.pds_ingest import cr_ingest, pds_ingest, set_db_credentials_from_secret_manager
from libera_utils.io.pds_ingest import IngestDuplicateError
from libera_utils.io.pds import ConstructionRecord, PDSRecord


def test_cr_ingest(clean_local_db, test_construction_record_1):
    """Test that a construction record (CR) was put into the DB with the PDS file entry for the CR removed.
    Check this by ensuring the PDS file entry doesn't exist for this CR file and the additional PDS file entry has no
    ingest time"""
    cr_ingest(test_construction_record_1)

    # A reference object made from the same file
    cr = ConstructionRecord.from_file(test_construction_record_1)

    # As long as this query comes back with a result then the cr filename matches the db entry
    with getdb().session() as s:
        cr_query = s.query(Cr).filter(Cr.file_name == cr.file_name).all()
        # This should be an empty query as this entry is removed by the cr_ingest method
        # however cr object direct from the ConstructionRecord class will still have this entry
        pds_query_0 = s.query(PdsFile).filter(PdsFile.file_name == cr.pds_files_list[0].filename).all()
        pds_query_1 = s.query(PdsFile).filter(PdsFile.file_name == cr.pds_files_list[1].filename).all()

    # Check the PDS files associated with the CR is only 1 and the CR has an ingest time
    assert cr_query[0].n_pds_files == 1
    assert cr_query[0].ingested is not None

    # Confirm the empty DB PDS entry for the CR itself
    assert len(pds_query_0) == 0
    # Confirm details of the PDS file entry associated with this CR
    assert len(pds_query_1) == 1
    assert pds_query_1[0].ingested is None
    assert pds_query_1[0].cr_id == cr_query[0].id


def test_pds_ingest_no_cr(clean_local_db, test_pds_file_1):
    """Test that an ingest of a PDS file that is not a construction record makes an appropriate entry"""
    pds_ingest(test_pds_file_1)

    with getdb().session() as s:
        pds_query = s.query(PdsFile).filter(PdsFile.file_name == test_pds_file_1.name)

    # Check the PDS file has an ingested time and no associated CR
    assert pds_query[0].ingested is not None
    assert pds_query[0].cr_id is None


def test_pds_ingest_with_existing_cr(clean_local_db, test_pds_file_1,
                                     test_construction_record_1):
    """Test that pds_ingest returns expected values when a single cr is already present in the database"""
    # Put a CR into the DB
    cr_ingest(test_construction_record_1)

    # Validate the state of the PDS DB entry before ingesting the PDS file itself
    pds = PDSRecord.from_filename(test_pds_file_1)
    with getdb().session() as s:
        pds_query_before = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
    assert pds_query_before[0].ingested is None

    # Ingest a PDS file
    pds_ingest(test_pds_file_1)

    with getdb().session() as s:
        pds_query_after = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
    # Check that the PDS entry was ingested and is still associated with the CR
    assert pds_query_after[0].ingested is not None
    assert pds_query_after[0].cr_id == pds_query_before[0].cr_id


def test_cr_ingest_with_existing_pds(clean_local_db,
                                     test_construction_record_1, test_pds_file_1):
    # Put a PDS into the DB
    pds_ingest(test_pds_file_1)

    # Validate PDS status before CR ingest
    pds = PDSRecord.from_filename(test_pds_file_1)
    with getdb().session() as s:
        pds_query_before = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
    assert pds_query_before[0].cr_id is None
    assert pds_query_before[0].ingested is not None

    cr_ingest(test_construction_record_1)

    cr = ConstructionRecord.from_file(test_construction_record_1)
    with getdb().session() as s:
        cr_query = s.query(Cr).filter(Cr.file_name == cr.file_name).all()
        pds_query_after = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()

    # Ensure the PDS file is associated with newly added construction record
    assert pds_query_after[0].cr_id == cr_query[0].id
    # Ensure the ingested time was not overwritten
    assert pds_query_before[0].ingested == pds_query_after[0].ingested


def test_print_cr_ingest_results(clean_local_db, test_construction_record_1):
    """Test that after calling ingest a query result prints as a representative of the entry."""
    cr_ingest(test_construction_record_1)

    cr_0 = ConstructionRecord.from_file(test_construction_record_1)

    with getdb().session() as s:
        cr_query_0 = s.query(Cr).filter(Cr.file_name == cr_0.file_name).all()
        print(cr_query_0)
        assert str(cr_query_0).__contains__(test_construction_record_1.name.split(".")[0])


def test_duplicate_cr_file(clean_local_db, test_construction_record_1):
    """Test that duplicate construction record files are handled with an error of the correct type and
    that none of the attributes existing in the database are changed. This test starts by ingesting
    a single CR file into the database.
    """
    # Put a single CR into the DB and store the DB information before a duplicate is attempted to be ingested
    cr_ingest(test_construction_record_1)
    cr = ConstructionRecord.from_file(test_construction_record_1)
    with getdb().session() as s:
        cr_query = s.query(Cr).filter(Cr.file_name == cr.file_name).all()

    # Try ingesting a duplicate and should error out
    with pytest.raises(IngestDuplicateError):
        cr_ingest(test_construction_record_1)

    # Ensure the ingested time was not overwritten
    with getdb().session() as s:
        cr_query_after = s.query(Cr).filter(Cr.file_name == cr.file_name).all()
    assert cr_query_after[0].ingested == cr_query[0].ingested


def test_duplicate_solo_pds_file(clean_local_db, test_pds_file_1):
    """Test that duplicate pds files are handled with an error of the correct type and
    that none of the attributes existing in the database are changed. This test begins with
    ingesting a single PDS file into the database.
    """
    pds_ingest(test_pds_file_1)

    pds = PDSRecord.from_filename(test_pds_file_1)
    with getdb().session() as s:
        pds_query_before = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
    assert pds_query_before[0].cr_id is None
    assert pds_query_before[0].ingested is not None

    with pytest.raises(IngestDuplicateError):
        pds_ingest(test_pds_file_1)

    with getdb().session() as s:
        pds_query_after = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
    assert pds_query_after[0].cr_id is None
    assert pds_query_after[0].ingested == pds_query_before[0].ingested


def test_duplicate_cr_associated_pds_file(clean_local_db, test_construction_record_1, test_pds_file_1):
    """Test that duplicate pds files are handled with an error of the correct type and
    that none of the attributes existing in the database are changed. This test begins with
    ingesting both a PDS and a CR file into the DB.
    """
    cr_ingest(test_construction_record_1)
    pds_ingest(test_pds_file_1)

    pds = PDSRecord.from_filename(test_pds_file_1)
    cr = ConstructionRecord.from_file(test_construction_record_1)
    with getdb().session() as s:
        pds_query_before = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
        cr_query = s.query(Cr).filter(Cr.file_name == cr.file_name).all()

    with pytest.raises(IngestDuplicateError):
        pds_ingest(test_pds_file_1)

    with getdb().session() as s:
        pds_query_after = s.query(PdsFile).filter(PdsFile.file_name == pds.filename).all()
    assert pds_query_after[0].cr_id == cr_query[0].id
    assert pds_query_after[0].ingested == pds_query_before[0].ingested


def test_corrupt_cons(clean_local_db, test_data_path):
    """Test that corrupt construction records will not be ingested"""
    corrupt_cons = test_data_path / 'bad_record.PDS'

    # This will not pass an L0 filename and throw the Value Error
    with pytest.raises(ValueError):
        cr_ingest(corrupt_cons)



def test_wrong_credentials(clean_local_db, test_data_path, monkeypatch):
    """Test that if connecting to db with wrong credentials, ingest will  throw errors"""
    corrupt_cns = test_data_path / 'bad_record.PDS'
    monkeypatch.setenv("LIBERA_DB_NAME", 'FAIL')
    monkeypatch.setenv("LIBERA_DB_USER", 'FAIL')
    monkeypatch.setenv("PGPASSWORD", 'FAIL')

    with pytest.raises((DatabaseException, OperationalError)):
        cr_ingest(corrupt_cns)
    with pytest.raises((DatabaseException, OperationalError)):
        pds_ingest(corrupt_cns)


def test_set_db_secret_credentials(create_mock_secret_manager):
    create_mock_secret_manager("test-secret")
    set_db_credentials_from_secret_manager("test-secret")

    assert os.environ["LIBERA_DB_USER"] == "libera_unit_tester"
    assert os.environ["LIBERA_DB_NAME"] == "libera"
    assert os.environ["PGPASSWORD"] == "testerpass"


def test_two_different_db_secret_credentials(monkeypatch, create_mock_secret_manager):
    create_mock_secret_manager("test_secret_0")
    set_db_credentials_from_secret_manager("test_secret_0")

    assert os.environ["LIBERA_DB_USER"] == "libera_unit_tester"

    create_mock_secret_manager("test_secret_1", username="another_tester")
    set_db_credentials_from_secret_manager("test_secret_1")

    assert os.environ["LIBERA_DB_USER"] == "another_tester"
