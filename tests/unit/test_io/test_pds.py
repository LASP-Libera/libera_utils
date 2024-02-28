"""Tests for manifest module"""
# Local
from libera_utils.io.pds import ConstructionRecord, PDSRecord


def test_construction_reader_from_file(test_construction_record_1):
    """
    Test construction reader from file
    """
    cr = ConstructionRecord.from_file(test_construction_record_1)

    # For an initial test, check that the final entry date is correct. This will at least show all the data was read
    # into the correct locations.
    # TODO reevaluate this test when we have construction records of the type we will use

    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.year == 2021
    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.month == 4
    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.day == 9
    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.hour == 1
    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.minute == 59
    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.second == 59
    assert cr.pds_files_list[1].apids[0].apid_last_packet_utc.microsecond == 5260


def test_construction_record_properties(test_construction_record_1):
    """
    Test construction record properties
    """
    cr = ConstructionRecord.from_file(test_construction_record_1)

    assert cr.edos_version_major == 8
    assert cr.edos_version_release == 1

    assert cr.apid_data_list[0].apid == 11
    assert cr.apid_data_list[0].scid == 159

    assert cr.apid_data_list[0].vcids_list[0].vcid == 0
    assert cr.apid_data_list[0].vcids_list[0].scid == 159

    assert cr.pds_files_list[1].apids[0].apid == 11
    assert cr.pds_files_list[1].apids[0].scid == 159


def test_construction_orm_creation(test_construction_record_1, clean_local_db):
    """
    Test construction orm creation
    """
    cr = ConstructionRecord.from_file(test_construction_record_1)
    cr_orm = cr.to_orm()

    # Check that the CR object has 2 PDS file entries, itself + 1 PDS data file
    assert cr.pds_file_count == 2
    # Check that the PDS file record of the construction record itself was removed from the orm representation.
    assert cr_orm.n_pds_files == 1

    # Check that the ORM models have the correct values
    assert len(cr_orm.scs_start_stop_times) == 4
    assert len(cr_orm.pds_files[0].apids) == 1
    assert len(cr_orm.apids) == 1
    assert len(cr_orm.apids[0].vcids) == 1
    assert len(cr_orm.apids[0].ssc_gaps) == 0
    assert len(cr_orm.apids[0].ssc_length_discrepancies) == 0
    assert len(cr_orm.apids[0].edos_fill_data) == 0

    assert cr_orm.pds_files[0].apids[0].last_packet_utc_time.year == 2021
    assert cr_orm.pds_files[0].apids[0].last_packet_utc_time.hour == 1
    assert cr_orm.pds_files[0].apids[0].last_packet_utc_time.second == 59
    assert cr_orm.pds_files[0].apids[0].apid == 11
    assert cr_orm.pds_files[0].apids[0].scid == 159


def test_pds_reader_from_file(test_pds_file_1):
    """
    Test pds reader from file
    """
    pds = PDSRecord.from_filename(test_pds_file_1)
    pds_orm = pds.to_orm()
    # TODO reevaluate this test when we have construction records of the type we will use

    assert pds_orm.file_name is not None
    assert pds_orm.ingested is not None
    assert len(pds_orm.apids) == 0


def test_construction_ddb_items(test_construction_record_1):
    """
    Test construction ddb items
    """
    cr = ConstructionRecord.from_file(test_construction_record_1)
    cr_ddb_items = cr.to_ddb_items()

    for item in cr_ddb_items:
        assert item["PK"] == str(test_construction_record_1.name)
        if item["SK"] == "#L0#APID11":
            assert item["applicable-date"] == "2021-04-09"
        if item["SK"] == "#CR":
            assert item["last_packet_utc_time"] == "2021-04-09 01:59:59.005260+00:00"
            # This is the number of PDS files associated with the construction record making sure the
            # construction record is not included in the PDS file count
            assert cr.pds_file_count == 2
            assert item["n_pds_files"] == 1

def test_pds_ddb_items(test_pds_file_1):
    """
    Test pds ddb items
    """
    pds = PDSRecord.from_filename(test_pds_file_1)
    pds_ddb_item = pds.to_ddb_pds_file_item()

    assert pds_ddb_item["PK"] == str(test_pds_file_1.name)
    assert pds_ddb_item["SK"] == "#"
    assert pds_ddb_item["ingested"] is not None
    #TODO have this check the libera_utils version (does Gavin know how?)
    assert pds_ddb_item["algorithm-version"] == "1.0.0"

