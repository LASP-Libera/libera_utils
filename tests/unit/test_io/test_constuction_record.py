"""Tests for manifest module"""
# Standard
from datetime import datetime, timedelta

# Local
from libera_utils.io.construction_record import ConstructionRecord


def test_construction_reader_from_file(text_construction_record):
    cr = ConstructionRecord(text_construction_record)

    # For an initial test, check that the final entry date is correct. This will at least show all the data was read
    # into the correct locations.
    # TODO reevaluate this test when we have construction records of the type we will use

    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.year == 2021
    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.month == 4
    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.day == 9
    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.hour == 1
    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.minute == 59
    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.second == 59
    assert cr.PDS_files_list[1].APID_this_file[0].APID_CCSDS_time_code_non_CR_datetime.microsecond == 5260


