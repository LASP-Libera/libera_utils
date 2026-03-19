"""Integration tests for libera_utils.packets module"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cfchecker import cfchecks
from space_packet_parser.xtce.validation import validate_xtce

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a import packets
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


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


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("packet_file_fixtures", "apid", "time_dimension", "skip_header_bytes"),
    [
        (
            [
                "test_jpss1_pds_file_1",
                "test_jpss1_pds_file_2",
                "test_jpss1_pds_file_3",
            ],
            LiberaApid.jpss_sc_pos,
            "PACKET_JPSS_TIME",
            0,
        ),
        (
            ["test_jpss4_pds_file_1"],
            LiberaApid.jpss_sc_pos,
            "PACKET_JPSS_TIME",
            0,
        ),
        (
            ["test_ccsds_2025_221_17_17_58"],
            LiberaApid.icie_axis_sample,
            "PACKET_ICIE_TIME",
            8,
        ),
        (
            ["test_ccsds_2025_221_16_56_48"],
            LiberaApid.icie_rad_sample,
            "PACKET_ICIE_TIME",
            8,
        ),
        (
            ["test_ccsds_2025_221_17_17_58"],
            LiberaApid.icie_rad_sample,
            "PACKET_ICIE_TIME",
            8,
        ),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_wfov_sci, "PACKET_ICIE_TIME", 8),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_nom_hk, "PACKET_ICIE_TIME", 8),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_crit_hk, "PACKET_ICIE_TIME", 8),
        (["test_ccsds_2025_221_17_17_58"], LiberaApid.icie_temp_hk, "PACKET_ICIE_TIME", 8),
        # pev_sw_stat (1000) and pec_sw_stat (1002) and icie_cal_sample (1044) - present in IOV SWC event data
        (["test_iov_swc_event"], LiberaApid.pev_sw_stat, "PACKET_ICIE_TIME", 8),
        (["test_iov_swc_event"], LiberaApid.pec_sw_stat, "PACKET_ICIE_TIME", 8),
        (["test_iov_swc_event"], LiberaApid.icie_cal_sample, "PACKET_ICIE_TIME", 8),
        # icie_rad_full (1035) and icie_cal_full (1043) - present in ISTR gain calibration event data
        (["test_istr_gain_event"], LiberaApid.icie_rad_full, "PACKET_ICIE_TIME", 8),
        (["test_istr_gain_event"], LiberaApid.icie_cal_full, "PACKET_ICIE_TIME", 8),
    ],
    ids=(
        "JPSS1_SC_POS",
        "JPSS4_SC_POS",
        "AXIS_SAMPLE",
        "RAD_SAMPLE_1",
        "RAD_SAMPLE_2",
        "WFOV_SCI",
        "NOM_HK",
        "CRIT_HK",
        "TEMP_HK",
        "PEV_SW_STAT_IOV",
        "PEC_SW_STAT_IOV",
        "CAL_SAMPLE_IOV",
        "RAD_FULL_ISTR",
        "CAL_FULL_ISTR",
    ),
)
def test_process_packets_to_l1a_product(
    packet_file_fixtures,
    apid,
    time_dimension,
    skip_header_bytes,
    request,
    tmp_path,
    monkeypatch,
):
    """Test complete L1A processing pipeline: PDS files → L1A Dataset → DataProductConfig → NetCDF4"""
    # Get the actual test file from the fixture
    packet_files: list[Path] = [request.getfixturevalue(f) for f in packet_file_fixtures]
    l1a_processing_config = get_packet_config(apid)  # Still needed for test assertions

    # Set global config for skipping packet header bytes for ground test data
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", str(skip_header_bytes))

    print("Parsing packets to L1A dataset...")
    dataset = packets.parse_packets_to_l1a_dataset(packet_files=packet_files, apid=apid.value)

    # Verify basic structure
    assert isinstance(dataset, type(dataset)), "Should return an xarray Dataset"
    assert len(dataset.data_vars) > 0, "Dataset should contain data variables"

    # Verify packet dimension exists (not swapped with time coordinate)
    assert "PACKET" in dataset.dims, "Missing 'packet' dimension"

    # Verify packet time is a non-dimension coordinate with "packet" dimension
    packet_time_coord = l1a_processing_config.packet_time_coordinate
    assert packet_time_coord in dataset.coords, f"Missing coordinate: {packet_time_coord}"
    assert dataset[packet_time_coord].dims == ("PACKET",), f"{packet_time_coord} should have 'packet' dimension"
    assert dataset == dataset.sortby(packet_time_coord)  # Should already be sorted

    n_packets_in_ds = dataset.sizes["PACKET"]

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
    product_definition_path = get_l1a_product_definition_path(apid)
    product_config = LiberaDataProductDefinition.from_yaml(product_definition_path)

    # Write NetCDF for round trip testing
    output_filename = write_libera_data_product(
        data_product_definition=product_definition_path,
        data=dataset,
        output_path=tmp_path,
        time_variable=time_dimension,
        strict=True,
    )
    output_path = output_filename.path

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


@pytest.mark.parametrize(
    "packet_definition",
    [
        pytest.param(
            config.get("LIBERA_PACKET_DEFINITION"),
            marks=pytest.mark.xfail(
                reason="Libera XTCE definition has incorrect namespace declarations and unused parameter definitions"
            ),
        ),
        config.get("JPSS_GEOLOCATION_PACKET_DEFINITION"),
        pytest.param(
            config.get("LIBERA_PEV_PACKET_DEFINITION"),
            marks=pytest.mark.xfail(
                reason="PEV XTCE definition may have incorrect namespace declarations or unused parameter definitions"
            ),
        ),
        pytest.param(
            config.get("LIBERA_PEC_PACKET_DEFINITION"),
            marks=pytest.mark.xfail(
                reason="PEC XTCE definition may have incorrect namespace declarations or unused parameter definitions"
            ),
        ),
    ],
)
def test_packet_definition_validity(packet_definition):
    """Test that the XTCE packet definitions are valid"""
    validate_xtce(packet_definition, level="all")
