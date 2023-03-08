"""Tests for the packet_ingest module"""
# Installed
import pytest
import os
# Local
from libera_utils.db import getdb
from libera_utils.db.models import *
from libera_utils.io.manifest import Manifest
from libera_utils.io.packet_ingest import ingest, cr_ingest
from libera_utils.io.construction_record import ConstructionRecord, PDSFiles


class DummyParser:
    """ Generates dummy parser """
    def __init__(self, manifest_filepath, short_tmp_path=None):
        self.manifest_filepath = str(manifest_filepath)
        self.outdir = str(short_tmp_path)


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
    pds_0 = PDSFiles(cr_0.pds_files_list[0].pds_filename)
    pds_0_orm = pds_0.to_orm()

    # insert P1590011AAAAAAAAAAAAAT21099065436900.PDS
    pds_1 = PDSFiles(cr_1.pds_files_list[0].pds_filename)
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
    pds = PDSFiles(cr.pds_files_list[0].pds_filename)
    pds_0_orm = pds.to_orm()

    # insert P1590011AAAAAAAAAAAAAT21099051420501.PDS
    pds = PDSFiles(cr.pds_files_list[1].pds_filename)
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


@pytest.mark.usefixtures('insert_single_pds_from_each_cr')
def test_pds_assigned_single(clean_local_db, test_input_manifest,
                             test_construction_record_09t00,
                             test_construction_record_09t02):
    """Test that cr_id was assigned properly to pds records and pds ingest
    time was assigned for records ingested separately"""

    # insert
    parsed_args = DummyParser(test_input_manifest)
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


@pytest.mark.usefixtures('insert_multiple_pds_from_single_cr')
def test_pds_assigned_mult(clean_local_db, test_input_manifest,
                           test_construction_record_09t00):
    """Test that cr_ingest returns expected values when multiple pds records
    from a single cr are already present in the database"""

    # read json information
    parsed_args = DummyParser(test_input_manifest)
    m = Manifest.from_file(parsed_args.manifest_filepath)
    db_pds_dict, con_ingested_dict = cr_ingest(
        m.files[0], parsed_args.outdir)

    cr = ConstructionRecord.from_file(test_construction_record_09t00)

    assert [cr.pds_files_list[0].pds_filename, cr.pds_files_list[1].pds_filename] \
           in db_pds_dict.values()


@pytest.mark.usefixtures('insert_single_pds_from_each_cr')
def test_manifest_assigned(clean_local_db, test_input_manifest):
    """Test that pds ingest time is listed for records listed in manifest
    """
    parsed_args = DummyParser(test_input_manifest)
    ingest(parsed_args)

    m = Manifest.from_file(test_input_manifest)

    for file in m.files:
        if 'PDS' in file['filename']:
            filename = os.path.basename(file['filename'])

            with getdb().session() as s:
                pds_query = s.query(PdsFile).filter(
                    PdsFile.file_name == filename).all()

            assert pds_query[0].ingested is not None


def test_output_manifest_all(clean_local_db, test_input_manifest):
    """Test output manifest file created contains a list of the
    product files that the processing created
    """
    parsed_args = DummyParser(test_input_manifest)

    m = Manifest.from_file(test_input_manifest)

    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    input_files = []
    output_files = []

    for file in m.files:
        input_files.append(os.path.basename(file['filename']))
    for file in m_output.files:
        output_files.append(os.path.basename(file['filename']))

    assert input_files == output_files


@pytest.mark.usefixtures('insert_single_pds_from_each_cr')
def test_output_manifest_partial(clean_local_db, test_input_manifest):
    """Test output manifest file created does not contain pds records already inserted
    """
    parsed_args = DummyParser(test_input_manifest)

    file_list = []

    output_manifest_path = ingest(parsed_args)
    m_output = Manifest.from_file(output_manifest_path)

    for file in m_output.files:
        file_list.append(os.path.basename(file['filename']))

    assert 'P1590011AAAAAAAAAAAAAT21099051420500.PDS' not in file_list
