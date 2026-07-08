"""Integration tests for libera_utils.packets module"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cfchecker import cfchecks
from space_packet_parser.xtce.validation import validate_xtce

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a import packets
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config
from libera_utils.l1a.wfov_image_metadata import (
    BLOB_BYTE_COORD,
    CAMERA_TIME_COORD,
    WFOV_IMAGE_BLOB_LENGTH_VAR,
    WFOV_IMAGE_BLOB_VAR,
)

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
        (["test_ditl_camera_with_duplicate_packet"], LiberaApid.icie_wfov_sci, CAMERA_TIME_COORD, 8),
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
    if apid == LiberaApid.icie_wfov_sci:
        with pytest.warns(UserWarning, match=r"Detected 1 duplicate PACKET_ICIE_TIME"):
            dataset = packets.parse_packets_to_l1a_dataset(packet_files=packet_files, apid=apid.value)
    else:
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

    if apid == LiberaApid.icie_nom_hk:
        assert "ICIE__SW_FP_WP_ST_WP" in dataset.data_vars
        assert dataset["ICIE__SW_FP_WP_ST_WP"].dims == ("PACKET", "ARRAY_128")
        assert dataset["ICIE__SW_FP_WP_ST_WP"].shape[1] == 128

    if apid == LiberaApid.icie_wfov_sci:
        assert CAMERA_TIME_COORD in dataset.coords
        assert CAMERA_TIME_COORD in dataset.dims
        assert BLOB_BYTE_COORD in dataset.dims
        assert "CAMERA_PACKET_INDEX" in dataset.data_vars
        assert "PACKET_IMAGE_ID" in dataset.data_vars
        assert "WFOV_FSW_PARSE_VALID" in dataset.data_vars
        assert "WFOV_FPGA_PARSE_VALID" in dataset.data_vars
        assert WFOV_IMAGE_BLOB_VAR in dataset.data_vars
        assert WFOV_IMAGE_BLOB_LENGTH_VAR in dataset.data_vars
        assert dataset.sizes[CAMERA_TIME_COORD] > 0
        assert dataset.sizes[CAMERA_TIME_COORD] == dataset.attrs["n_complete_images"]
        assert "n_missing_sop_or_eop" in dataset.attrs
        assert "n_bad_images" in dataset.attrs

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


@pytest.mark.filterwarnings("error")
def test_wfov_sci_filename_uses_image_time_bounds(
    test_ditl_camera_with_duplicate_packet,
    monkeypatch,
    tmp_path,
):
    """WFOV L1A filenames must reflect FSW image times, not CCSDS packet telemetry times."""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    apid = LiberaApid.icie_wfov_sci
    packet_config = get_packet_config(apid)

    with pytest.warns(UserWarning, match=r"Detected 1 duplicate PACKET_ICIE_TIME"):
        dataset = packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ditl_camera_with_duplicate_packet],
            apid=apid.value,
        )

    expected_first = np.datetime64("2028-02-14T04:23:03.681622", "us")
    expected_last = np.datetime64("2028-02-14T04:23:13.564655", "us")

    camera_times = dataset[CAMERA_TIME_COORD].values
    np.testing.assert_equal(camera_times[0], expected_first)
    np.testing.assert_equal(camera_times[-1], expected_last)

    packet_first = dataset[packet_config.packet_time_coordinate].values[0]
    packet_last = dataset[packet_config.packet_time_coordinate].values[-1]
    assert expected_first != packet_first
    assert expected_last != packet_last

    product_definition_path = get_l1a_product_definition_path(apid)
    output_filename = write_libera_data_product(
        data_product_definition=product_definition_path,
        data=dataset,
        output_path=tmp_path,
        time_variable=CAMERA_TIME_COORD,
        strict=True,
    )
    parsed_filename = LiberaDataProductFilename.from_file_path(output_filename.path)
    filename = parsed_filename.path.name
    assert "20280214T042303" in filename
    assert "20280214T042313" in filename
    assert "20280215T131631" not in filename
    assert "20280215T131727" not in filename


@pytest.mark.filterwarnings("error")
def test_ditl_camera_wfov_image_blob_round_trip_and_decompress(
    test_ditl_camera_with_duplicate_packet,
    monkeypatch,
    tmp_path,
):
    """DITL WFOV SCI: parse, write, read back, and smoke-test JPEG-LS decompression."""
    from io import BytesIO

    import pillow_jpls  # noqa: F401 - registers JPEG-LS plugin with Pillow
    from PIL import Image

    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    apid = LiberaApid.icie_wfov_sci

    with pytest.warns(UserWarning, match=r"Detected 1 duplicate PACKET_ICIE_TIME"):
        dataset = packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ditl_camera_with_duplicate_packet],
            apid=apid.value,
        )

    n_complete = dataset.sizes[CAMERA_TIME_COORD]
    assert n_complete == dataset.attrs["n_complete_images"]
    assert n_complete < dataset.sizes["PACKET"]

    pre_write_payloads = []
    for image_index in range(n_complete):
        length = int(dataset[WFOV_IMAGE_BLOB_LENGTH_VAR].values[image_index])
        pre_write_payloads.append(dataset[WFOV_IMAGE_BLOB_VAR].values[image_index, :length].tobytes())

    product_definition_path = get_l1a_product_definition_path(apid)
    output_filename = write_libera_data_product(
        data_product_definition=product_definition_path,
        data=dataset,
        output_path=tmp_path,
        time_variable=CAMERA_TIME_COORD,
        strict=True,
    )

    with xr.open_dataset(output_filename.path) as read_dataset:
        product_config = LiberaDataProductDefinition.from_yaml(product_definition_path)
        errors = product_config.check_dataset_conformance(read_dataset, strict=False)
        assert not errors, errors

        assert read_dataset.sizes[CAMERA_TIME_COORD] == n_complete
        for image_index, expected_payload in enumerate(pre_write_payloads):
            length = int(read_dataset[WFOV_IMAGE_BLOB_LENGTH_VAR].values[image_index])
            round_tripped = read_dataset[WFOV_IMAGE_BLOB_VAR].values[image_index, :length].tobytes()
            assert round_tripped == expected_payload

        for image_index in range(min(3, n_complete)):
            length = int(read_dataset[WFOV_IMAGE_BLOB_LENGTH_VAR].values[image_index])
            payload = read_dataset[WFOV_IMAGE_BLOB_VAR].values[image_index, :length].tobytes()
            with Image.open(BytesIO(payload)) as img:
                assert img.size == (2048, 2048)


@pytest.mark.filterwarnings("error")
def test_ditl_camera_duplicate_packet_timestamp_deduplicated(
    test_ditl_camera_with_duplicate_packet,
    monkeypatch,
):
    """DITL WFOV SCI data includes one duplicate packet timestamp

    L1A processing must warn, drop the duplicate packet, and leave unique packet times.
    """
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    apid = LiberaApid.icie_wfov_sci
    packet_time_coordinate = get_packet_config(apid).packet_time_coordinate

    with pytest.warns(
        UserWarning,
        match=r"Detected 1 duplicate PACKET_ICIE_TIME in dataset",
    ):
        dataset = packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ditl_camera_with_duplicate_packet],
            apid=apid.value,
        )

    # Fixture has 1510 WFOV SCI packets with one repeated PACKET_ICIE_TIME.
    assert dataset.sizes["PACKET"] == 1509
    _, counts = np.unique(dataset[packet_time_coordinate].values, return_counts=True)
    assert not np.any(counts > 1)


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
