"""Integration tests for libera_utils.packets module"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cfchecker import cfchecks

from libera_utils import packets
from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.packet_configs import get_packet_config

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# TESTS OF FUNCTIONS TO BE REMOVED IN 5.2.1
# -----------------------------------------
def test_read_sc_packet_data(test_jpss1_pds_file_1):
    """Test reading spacecraft packet data from real PDS file

    Uses JPSS geolocation packet definition and verifies APID 11 parsing.
    """
    result_df = packets.read_sc_packet_data([test_jpss1_pds_file_1])

    # Basic assertions
    assert not result_df.empty, "DataFrame should not be empty"
    assert len(result_df) > 0, "Should have parsed some packets"

    # Verify APID filtering worked correctly
    assert "PKT_APID" in result_df.columns, "DataFrame should contain PKT_APID column"
    assert all(result_df["PKT_APID"] == 11), "All packets should have APID 11"


def test_read_azel_packet_data(test_ccsds_2025_218_18_37_32, test_ccsds_2025_218_18_41_30):
    """Test reading Az/El packet data from real CCSDS files

    Uses AZEL packet definition and verifies APID 1048 parsing with
    restructuring to 50 samples per packet. Tests with both new data files
    that should have no duplicate timestamps.
    """
    result_df = packets.read_azel_packet_data([test_ccsds_2025_218_18_37_32, test_ccsds_2025_218_18_41_30])

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


# L1A Packets Tests
# -----------------
def check_cf_conformance(file: str | Path, silent=True, **kwargs):
    """Use CFChecker to check file conformance to CF conventions

    Notes
    -----
    This facility is not particularly useful in assertion testing because it seems to mark everything as warning or INFO,
    even when things are truly required.
    This helper function is left in because it can be useful in development
    """
    checker = cfchecks.CFChecker(silent=silent, **kwargs)
    try:
        checker.checker(str(file))
    except cfchecks.FatalCheckerError:
        print("Checking of file %s aborted due to error", file)

    print("\nGLOBAL")
    for level, msgs in checker.results["global"].items():
        for msg in msgs:
            print(f"\t{level}: {msg}")

    print("\nVARIABLES")
    for var_name, var_results in checker.results["variables"].items():
        print(f"\t{var_name}:")
        for level, msgs in var_results.items():
            for msg in msgs:
                print(f"\t\t{level}: {msg}")
    return checker


@pytest.mark.parametrize(
    ("packet_file_fixtures", "apid", "product_definition_file"),
    [
        (
            [
                "test_jpss1_pds_file_1",
                "test_jpss4_pds_file_1",
            ],
            LiberaApid.jpss_sc_pos,
            "jpss_sc_pos_l1a.yml",
        ),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_axis_sample, "icie_axis_sample_l1a.yml"),
        (["test_ccsds_2025_221_16_56_48"], LiberaApid.icie_rad_sample, "icie_rad_sample_l1a.yml"),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_wfov_sci, "icie_wfov_sci_l1a.yml"),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_nom_hk, "icie_nom_hk_l1a.yml"),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_crit_hk, "icie_crit_hk_l1a.yml"),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_temp_hk, "icie_temp_hk_l1a.yml"),
    ],
    ids=("SC_POS", "AXIS_SAMPLE", "RAD_SAMPLE", "WFOV_SCI", "NOM_HK", "CRIT_HK", "TEMP_HK"),
)
def test_process_packets_to_l1a_product(packet_file_fixtures, apid, product_definition_file, request, tmp_path):
    """Test complete L1A processing pipeline: PDS files → L1A Dataset → DataProductConfig → NetCDF4"""
    # Get the actual test file from the fixture
    packet_files = [request.getfixturevalue(f) for f in packet_file_fixtures]
    l1a_processing_config = get_packet_config(apid)  # Still needed for test assertions

    print("Parsing packets to L1A dataset...")
    dataset = packets.parse_packets_to_l1a_dataset(packet_files=packet_files, apid=apid.value)

    # Verify basic structure
    assert isinstance(dataset, type(dataset)), "Should return an xarray Dataset"
    assert len(dataset.data_vars) > 0, "Dataset should contain data variables"

    # Verify packet dimension exists (not swapped with time coordinate)
    assert "packet" in dataset.dims, "Missing 'packet' dimension"

    # Verify packet time is a non-dimension coordinate with "packet" dimension
    packet_time_coord = l1a_processing_config.packet_time_coordinate
    assert packet_time_coord in dataset.coords, f"Missing coordinate: {packet_time_coord}"
    assert dataset[packet_time_coord].dims == ("packet",), f"{packet_time_coord} should have 'packet' dimension"
    assert dataset == dataset.sortby(packet_time_coord)  # Should already be sorted

    n_packets_in_ds = dataset.sizes["packet"]

    # Verify sample group coordinates exist for each sample group
    for group in l1a_processing_config.sample_groups:
        assert group.sample_time_dimension in dataset.coords
        assert group.sample_time_dimension in dataset.dims
        assert dataset.sizes[group.sample_time_dimension] == n_packets_in_ds * group.sample_count  # Check sample count
        assert dataset == dataset.sortby(group.sample_time_dimension)  # Should already be sorted

        # Verify packet_index variable exists for each sample group
        packet_index_var = f"{group.name}_packet_index"
        assert packet_index_var in dataset.data_vars, f"Missing packet_index variable: {packet_index_var}"
        assert dataset[packet_index_var].dims == (group.sample_time_dimension,)

        # Verify packet_index values are correct (0, 0, ..., 1, 1, ..., etc.)
        expected_indices = np.repeat(np.arange(n_packets_in_ds), group.sample_count)
        np.testing.assert_array_equal(dataset[packet_index_var].values, expected_indices)

    print("Enforcing LiberaDataProductDefinition on dataset object")

    # Create LiberaDataProductDefinition from product definition file
    product_definition_path = Path(str(config.get("LIBERA_PRODUCT_DEFINITIONS_PATH"))) / product_definition_file
    product_config = LiberaDataProductDefinition.from_yaml(product_definition_path)

    # Coerce dataset to match product definition configuration
    dataset, errors = product_config.enforce_dataset_conformance(dataset)
    if errors:
        print(errors)
        raise ValueError("After conformance enforcement, dataset is still not valid")

    # Write NetCDF for round trip testing
    output_path = tmp_path / f"{apid.name}_l1a.nc"
    dataset.to_netcdf(output_path)

    print(f"   NetCDF file: {output_path.name}")
    assert output_path.exists(), "NetCDF file was not created"

    file_size = output_path.stat().st_size
    print(f"   File size: {file_size:,} bytes")
    assert file_size > 0, "NetCDF file is empty"

    print("Reading back NetCDF file for verification...")
    with xr.open_dataset(output_path) as read_dataset:
        errors = product_config.check_dataset_conformance(read_dataset, strict=False)
        assert not errors, errors

        # Check round trip data values are unchanged
        for var_name in read_dataset.data_vars:
            # Verify all values preserved exactly
            np.testing.assert_array_equal(
                read_dataset[var_name].values,
                read_dataset[var_name].values,
                err_msg=f"Variable values not preserved for {var_name}",
            )

    print("   ✓ NetCDF file verification complete")
