"""Integration tests for libera_utils.packets module"""

import pytest

from libera_utils import packets

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def test_read_sc_packet_data(test_pds_file_1):
    """Test reading spacecraft packet data from real PDS file

    Uses JPSS geolocation packet definition and verifies APID 11 parsing.
    """
    result_df = packets.read_sc_packet_data([test_pds_file_1])

    # Basic assertions
    assert not result_df.empty, "DataFrame should not be empty"
    assert len(result_df) > 0, "Should have parsed some packets"

    # Verify APID filtering worked correctly
    assert "PKT_APID" in result_df.columns, "DataFrame should contain PKT_APID column"
    assert all(result_df["PKT_APID"] == 11), "All packets should have APID 11"


def test_read_azel_packet_data(test_azel_ccsds_2025_218_18_37_32, test_azel_ccsds_2025_218_18_41_30):
    """Test reading Az/El packet data from real CCSDS files

    Uses AZEL packet definition and verifies APID 1048 parsing with
    restructuring to 50 samples per packet. Tests with both new data files
    that should have no duplicate timestamps.
    """
    result_df = packets.read_azel_packet_data([test_azel_ccsds_2025_218_18_37_32, test_azel_ccsds_2025_218_18_41_30])

    # Basic assertions
    assert not result_df.empty, "DataFrame should not be empty"
    assert len(result_df) > 0, "Should have parsed some samples"

    # Verify expected columns from restructuring
    expected_columns = {"SAMPLE_SEC", "SAMPLE_USEC", "AZ_ANGLE_RAD", "EL_ANGLE_RAD"}
    assert expected_columns.issubset(set(result_df.columns)), (
        f"Missing expected columns: {expected_columns - set(result_df.columns)}"
    )

    # Verify sample count relationship (50 samples per packet)
    # With the new test data files, there should be no duplicate timestamps
    assert len(result_df) % 50 == 0, (
        f"Sample count ({len(result_df)}) should be a multiple of 50 (50 samples per packet)"
    )
    assert len(result_df) >= 100, "Should have at least 100 samples (2 packets worth from 2 files)"

    # Verify no duplicate timestamps (combination of SAMPLE_SEC and SAMPLE_USEC)
    assert result_df[["SAMPLE_SEC", "SAMPLE_USEC"]].duplicated().sum() == 0, "Should have no duplicate timestamps"
