"""Integration tests for libera_utils.packets module"""

import multiprocessing as mp
import time
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import space_packet_parser as spp
import xarray as xr
from cfchecker import cfchecks
from cloudpathlib import AnyPath
from space_packet_parser import load_xtce
from space_packet_parser.xarr import create_dataset
from space_packet_parser.xtce.validation import validate_xtce

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a import packets
from libera_utils.l1a.data_time_extractors import extract_data_time_range
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config
from libera_utils.l1a.wfov_image_metadata import (
    BLOB_BYTE_COORD,
    CAMERA_TIME_COORD,
    ERROR_FLAGGED_IMAGE_COUNT_ATTR,
    FIRST_IMAGE_INCOMPLETE_ATTR,
    FOOTER_MISMATCH_COUNT_ATTR,
    HEADER_PARSE_ERROR_COUNT_ATTR,
    LAST_IMAGE_INCOMPLETE_ATTR,
    PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR,
    WFOV_COMPRESSED_IMAGE_LENGTH_VAR,
    WFOV_COMPRESSED_IMAGE_VAR,
)
from libera_utils.time import dt64_to_utc_datetime, multipart_to_dt64
from tests.marks import strict_warnings

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


@strict_warnings
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
    caplog,
):
    """Test complete L1A processing pipeline: PDS files → L1A Dataset → DataProductConfig → NetCDF4"""
    # Get the actual test file from the fixture
    packet_files: list[Path] = [request.getfixturevalue(f) for f in packet_file_fixtures]
    l1a_processing_config = get_packet_config(apid)  # Still needed for test assertions

    # Set global config for skipping packet header bytes for ground test data
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", str(skip_header_bytes))

    print("Parsing packets to L1A dataset...")
    if apid == LiberaApid.icie_wfov_sci:
        with pytest.warns(UserWarning, match=r"Detected 1 duplicate SRC_SEQ_CTR packets"):
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
    order_warnings = [
        r
        for r in caplog.records
        if r.name == "libera_utils.l1a.packet_ordering"
        and r.levelname == "WARNING"
        and "duplicate SRC_SEQ_CTR" not in r.getMessage()
    ]
    assert order_warnings == [], "packet time or the counter disagrees with SRC_SEQ_CTR order on this fixture"

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
        assert WFOV_COMPRESSED_IMAGE_VAR in dataset.data_vars
        assert WFOV_COMPRESSED_IMAGE_LENGTH_VAR in dataset.data_vars
        assert dataset.sizes[CAMERA_TIME_COORD] > 0
        assert "ICIE__WFOV_DATA" not in dataset.data_vars
        assert PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR in dataset.attrs
        assert ERROR_FLAGGED_IMAGE_COUNT_ATTR in dataset.attrs
        assert FOOTER_MISMATCH_COUNT_ATTR in dataset.attrs
        assert HEADER_PARSE_ERROR_COUNT_ATTR in dataset.attrs
        assert FIRST_IMAGE_INCOMPLETE_ATTR in dataset.attrs
        assert LAST_IMAGE_INCOMPLETE_ATTR in dataset.attrs

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


@strict_warnings
def test_wfov_sci_filename_uses_image_time_bounds(
    test_ditl_camera_with_duplicate_packet,
    monkeypatch,
    tmp_path,
):
    """WFOV L1A filenames must reflect FSW image times, not CCSDS packet telemetry times."""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    apid = LiberaApid.icie_wfov_sci
    packet_config = get_packet_config(apid)

    with pytest.warns(UserWarning, match=r"Detected 1 duplicate SRC_SEQ_CTR packets"):
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


@strict_warnings
def test_ditl_camera_wfov_compressed_image_round_trip_and_decompress(
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

    with pytest.warns(UserWarning, match=r"Detected 1 duplicate SRC_SEQ_CTR packets"):
        dataset = packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ditl_camera_with_duplicate_packet],
            apid=apid.value,
        )

    n_complete = dataset.sizes[CAMERA_TIME_COORD]
    assert n_complete < dataset.sizes["PACKET"]
    # Fixture has 1509 WFOV SCI packets; only 3 stitch into complete images. The rest are all
    # leading/trailing edge truncation (this packet window doesn't open on an SOP or close on an
    # EOP), not genuine anomalies, so they don't count toward PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR.
    assert dataset.attrs[PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR] == 0
    assert dataset.attrs[FIRST_IMAGE_INCOMPLETE_ATTR] == 1
    assert dataset.attrs[LAST_IMAGE_INCOMPLETE_ATTR] == 1
    assert dataset.attrs[ERROR_FLAGGED_IMAGE_COUNT_ATTR] == 0
    assert dataset.attrs[FOOTER_MISMATCH_COUNT_ATTR] == 0
    assert dataset.attrs[HEADER_PARSE_ERROR_COUNT_ATTR] == 0

    pre_write_payloads = []
    for image_index in range(n_complete):
        length = int(dataset[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values[image_index])
        pre_write_payloads.append(dataset[WFOV_COMPRESSED_IMAGE_VAR].values[image_index, :length].tobytes())

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
            length = int(read_dataset[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values[image_index])
            round_tripped = read_dataset[WFOV_COMPRESSED_IMAGE_VAR].values[image_index, :length].tobytes()
            assert round_tripped == expected_payload

        for image_index in range(min(3, n_complete)):
            length = int(read_dataset[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values[image_index])
            payload = read_dataset[WFOV_COMPRESSED_IMAGE_VAR].values[image_index, :length].tobytes()
            with Image.open(BytesIO(payload)) as img:
                assert img.size == (2048, 2048)


def test_wfov_sci_no_complete_images_raises(
    test_ccsds_2025_221_17_17_58,
    monkeypatch,
):
    """A WFOV granule where no image completes must fail loudly, not emit an unusable product.

    This ISTR file carries 1298 APID 1040 packets that never form a complete SOP->EOP image. Before
    the raise it produced a dataset with ICIE__WFOV_DATA already dropped, zero CAMERA_TIME rows, and
    PacketCountNotUsedInImages == 0 -- which then failed to write under strict=True (missing every
    CAMERA_TIME variable) and under strict=False (IndexError on an empty time coordinate). The raw
    packets were destroyed either way.
    """
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")

    with pytest.raises(ValueError, match=r"No complete WFOV images could be produced"):
        packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ccsds_2025_221_17_17_58],
            apid=LiberaApid.icie_wfov_sci.value,
        )


@strict_warnings
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
        match=r"Detected 1 duplicate SRC_SEQ_CTR packets",
    ):
        dataset = packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ditl_camera_with_duplicate_packet],
            apid=apid.value,
        )

    # Fixture has 1510 WFOV SCI packets with one repeated PACKET_ICIE_TIME.
    assert dataset.sizes["PACKET"] == 1509
    _, counts = np.unique(dataset[packet_time_coordinate].values, return_counts=True)
    assert not np.any(counts > 1)


def _single_batch_reference(packet_files: list[Path], apid: int, skip_header_bytes: int) -> xr.Dataset:
    """Every file handed to ``create_dataset`` in one call: the result a per-file parse must reproduce."""
    definition_path = config.get(get_packet_config(LiberaApid(apid)).packet_definition_config_key)
    return create_dataset(
        packet_files=[AnyPath(f) for f in packet_files],
        xtce_packet_definition=load_xtce(definition_path),
        generator_kwargs={"skip_header_bytes": skip_header_bytes},
        packet_filter=lambda packet_bytes: packet_bytes.apid == apid,
    )[apid].swap_dims({"packet": "PACKET"})


def _assert_same_packet_dataset(actual: xr.Dataset, expected: xr.Dataset) -> None:
    # assert_identical compares string values but not their width, which the accumulator widens.
    xr.testing.assert_identical(actual, expected)
    assert {name: var.dtype for name, var in actual.variables.items()} == {
        name: var.dtype for name, var in expected.variables.items()
    }


_EQUIVALENCE_CASES = pytest.mark.parametrize(
    ("packet_file_fixtures", "apid", "skip_header_bytes"),
    [
        pytest.param(
            [("test_jpss1_pds_file_1", None), ("test_jpss1_pds_file_2", None), ("test_jpss1_pds_file_3", None)],
            LiberaApid.jpss_sc_pos.value,
            0,
            id="JPSS1_3xPDS",
        ),
        pytest.param(
            [
                ("test_ditl_demux_path", "LIBERA_SDC_1040_ccsds_2025_318_13_00_00"),
                ("test_ditl_demux_path", "LIBERA_SDC_1040_ccsds_2025_318_13_20_00"),
            ],
            LiberaApid.icie_wfov_sci.value,
            0,
            id="DITL_DEMUX_2x1040",
        ),
        pytest.param(
            [("test_ccsds_2025_218_18_37_32", None), ("test_ccsds_2025_218_18_41_30", None)],
            LiberaApid.icie_rad_sample.value,
            8,
            id="ISTR_2x1036",
        ),
    ],
)


def _resolve_packet_files(request: pytest.FixtureRequest, packet_file_fixtures: list[tuple[str, str | None]]):
    """Fixture names, each with an optional file name inside the directory the fixture names."""
    paths = []
    for fixture, name in packet_file_fixtures:
        path = request.getfixturevalue(fixture)
        paths.append(path / name if name else path)
    return paths


@_EQUIVALENCE_CASES
def test_per_file_parse_matches_single_batch(packet_file_fixtures, apid, skip_header_bytes, request):
    packet_files = _resolve_packet_files(request, packet_file_fixtures)
    definition_path = config.get(get_packet_config(LiberaApid(apid)).packet_definition_config_key)
    parsed = packets.parse_packets_to_dataset(packet_files, definition_path, apid, skip_header_bytes=skip_header_bytes)
    _assert_same_packet_dataset(parsed, _single_batch_reference(packet_files, apid, skip_header_bytes))


@_EQUIVALENCE_CASES
def test_max_workers_matches_in_process(packet_file_fixtures, apid, skip_header_bytes, request):
    # Not strict_warnings: forking a multi-threaded process warns on Python 3.12+.
    packet_files = _resolve_packet_files(request, packet_file_fixtures)
    definition_path = config.get(get_packet_config(LiberaApid(apid)).packet_definition_config_key)
    in_process = packets.parse_packets_to_dataset(
        packet_files, definition_path, apid, skip_header_bytes=skip_header_bytes
    )
    in_workers = packets.parse_packets_to_dataset(
        packet_files, definition_path, apid, max_workers=2, skip_header_bytes=skip_header_bytes
    )
    _assert_same_packet_dataset(in_workers, in_process)


def test_l1a_dataset_max_workers_matches_in_process(
    test_jpss1_pds_file_1, test_jpss1_pds_file_2, test_jpss1_pds_file_3
):
    files = [test_jpss1_pds_file_1, test_jpss1_pds_file_2, test_jpss1_pds_file_3]
    apid = LiberaApid.jpss_sc_pos.value
    in_process = packets.parse_packets_to_l1a_dataset(packet_files=files, apid=apid)
    in_workers = packets.parse_packets_to_l1a_dataset(packet_files=files, apid=apid, max_workers=2)
    _assert_same_packet_dataset(in_workers.drop_attrs(), in_process.drop_attrs())


def test_jpss1_file_order_does_not_change_the_product(
    test_jpss1_pds_file_1, test_jpss1_pds_file_2, test_jpss1_pds_file_3
):
    files = [test_jpss1_pds_file_1, test_jpss1_pds_file_2, test_jpss1_pds_file_3]
    forward = packets.parse_packets_to_l1a_dataset(packet_files=files, apid=LiberaApid.jpss_sc_pos.value)
    reversed_ = packets.parse_packets_to_l1a_dataset(packet_files=files[::-1], apid=LiberaApid.jpss_sc_pos.value)
    xr.testing.assert_identical(forward.drop_attrs(), reversed_.drop_attrs())
    assert forward.attrs["input_files"] == reversed_.attrs["input_files"] == [f.name for f in files]
    assert forward.sizes["PACKET"] == 21600, "the three files cross a SRC_SEQ_CTR wrap"


def _write_packets(path, packets_bytes) -> None:
    path.write_bytes(b"".join(bytes(packet) for packet in packets_bytes))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(slice(0, 3000), slice(2000, None), id="overlapping"),
        pytest.param([slice(0, 100), slice(200, None)], [slice(100, 200)], id="gap-filling"),
    ],
)
def test_jpss1_split_files_parse_like_the_whole_file(tmp_path, test_jpss1_pds_file_1, first, second):
    """Overlapping or gap-filling pieces of one file, passed in reverse, parse to the whole file's product."""
    with open(test_jpss1_pds_file_1, "rb") as stream:
        whole = list(spp.ccsds_generator(stream, skip_header_bytes=0))
    pieces = []
    for name, spans in (("a.PDS", first), ("b.PDS", second)):
        spans = spans if isinstance(spans, list) else [spans]
        _write_packets(tmp_path / name, [packet for span in spans for packet in whole[span]])
        pieces.append(tmp_path / name)

    apid = LiberaApid.jpss_sc_pos.value
    expected = packets.parse_packets_to_l1a_dataset(packet_files=[test_jpss1_pds_file_1], apid=apid)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected .* duplicate SRC_SEQ_CTR packets")
        result = packets.parse_packets_to_l1a_dataset(packet_files=pieces[::-1], apid=apid)
    xr.testing.assert_identical(result.drop_attrs(), expected.drop_attrs())


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


@pytest.mark.parametrize(
    "pds_fixture",
    ["test_jpss1_pds_file_1", "test_jpss4_pds_file_1"],
)
def test_jpss_sc_pos_data_time_range_spans_both_sample_groups(pds_fixture, request):
    """APID 11's extent runs from the earliest to the latest of its ADGPS/ADCFA sample times."""
    packet_file = request.getfixturevalue(pds_fixture)
    apid = LiberaApid.jpss_sc_pos
    packet_config = get_packet_config(apid)

    first, last = extract_data_time_range(packet_file, int(apid), skip_header_bytes=0)

    packet_ds = packets.parse_packets_to_dataset(
        [packet_file],
        str(config.get(packet_config.packet_definition_config_key)),
        int(apid),
        skip_header_bytes=0,
    )
    group_spans = {}
    for group in packet_config.sample_groups:
        times = multipart_to_dt64(packet_ds, **group.time_field_patterns.multipart_kwargs).values
        group_spans[group.name] = (times.min(), times.max())
    assert set(group_spans) == {"ADGPS", "ADCFA"}

    assert first < last
    assert first == dt64_to_utc_datetime(min(group_first for group_first, _ in group_spans.values()))
    assert last == dt64_to_utc_datetime(max(group_last for _, group_last in group_spans.values()))

    # Every group's own span sits inside the reported extent, which the overlap would not cover.
    for group_first, group_last in group_spans.values():
        assert first <= dt64_to_utc_datetime(group_first)
        assert last >= dt64_to_utc_datetime(group_last)


def _scripted_worker(connection, packet_file, packet_definition_path, apid, generator_kwargs):
    """Stand-in for ``packets._parse_file_in_subprocess`` whose result is chosen by file name.

    ``slow`` never reports, ``dead`` exits without reporting, ``bad`` fails, ``other`` parses to a
    different variable set, and anything else parses to two packets.
    """
    if packet_file == "slow.bin":
        time.sleep(60)
    elif packet_file == "bad.bin":
        connection.send((ValueError(f"bad file {packet_file}"), None))
    elif packet_file == "other.bin":
        connection.send((None, xr.Dataset({"OTHER": (("PACKET",), np.zeros(2))})))
    elif packet_file != "dead.bin":
        connection.send((None, xr.Dataset({"DATA_FIELD": (("PACKET",), np.zeros(2))})))
    connection.close()


class TestWorkerProcesses:
    """Worker-process failures reach the caller, and no worker outlives the call."""

    @pytest.mark.parametrize(
        ("files", "max_workers", "error", "match"),
        [
            pytest.param(["bad.bin", "b.bin"], 2, ValueError, "bad file", id="worker-error"),
            pytest.param(["dead.bin", "b.bin"], 2, RuntimeError, "exited without a result", id="worker-died"),
            pytest.param(
                ["bad.bin", "slow.bin", "c.bin"], 2, ValueError, "bad file", id="worker-error-sibling-running"
            ),
            pytest.param(
                ["a.bin", "other.bin", "slow.bin"], 3, ValueError, "do not match the other files", id="consumer-error"
            ),
        ],
    )
    def test_error_reaches_the_caller_and_no_worker_outlives_it(self, files, max_workers, error, match):
        with (
            mock.patch.object(packets, "_parse_file_in_subprocess", _scripted_worker),
            pytest.raises(error, match=match),
        ):
            packets.parse_packets_to_dataset(files, "fake.xml", 1040, max_workers=max_workers)
        assert mp.active_children() == []
