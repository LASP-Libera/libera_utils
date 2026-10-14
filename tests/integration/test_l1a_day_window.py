"""Integration tests for the L1A day window: gate, trim and measurement on recorded packet files."""

from datetime import date, timedelta

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a import packets
from libera_utils.l1a.data_time_extractors import extract_data_time_range
from libera_utils.l1a.day_coverage import evaluate_day_coverage, measure_time_axis_coverage
from libera_utils.l1a.day_window import trim_l1a_to_day_window
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config
from libera_utils.l1a.wfov_image_metadata import (
    CAMERA_PACKET_INDEX_VAR,
    CAMERA_TIME_COORD,
    MEM_DUMP_FLAGS_VAR,
    PACKET_IMAGE_ID_VAR,
)

pytestmark = pytest.mark.integration

JPSS1_FIXTURES = ("test_jpss1_pds_file_1", "test_jpss1_pds_file_2", "test_jpss1_pds_file_3")
JPSS1_SAMPLE_DIMS = ("ADGPS_JPSS_TIME", "ADCFA_JPSS_TIME")


@pytest.fixture(scope="module")
def jpss1_files(request):
    return [request.getfixturevalue(name) for name in JPSS1_FIXTURES]


@pytest.fixture(scope="module")
def jpss1_dataset(jpss1_files):
    """Three JPSS-1 PDS files, 2021-04-08T23:59:59.93 to 2021-04-09T05:59:59 at 1 Hz."""
    return packets.parse_packets_to_l1a_dataset(packet_files=jpss1_files, apid=LiberaApid.jpss_sc_pos.value)


def _write_and_check(dataset: xr.Dataset, apid: LiberaApid, time_variable: str, tmp_path) -> None:
    definition_path = get_l1a_product_definition_path(apid)
    output = write_libera_data_product(
        data_product_definition=definition_path,
        data=dataset,
        output_path=tmp_path,
        time_variable=time_variable,
        strict=True,
    )
    definition = LiberaDataProductDefinition.from_yaml(definition_path)
    with xr.open_dataset(output.path) as reread:
        assert not definition.check_dataset_conformance(reread, strict=False)


@pytest.mark.parametrize(
    ("day", "n_packets"),
    [(date(2021, 4, 8), 601), (date(2021, 4, 9), 21600), (date(2021, 4, 10), 0)],
    ids=["day_before_keeps_its_right_buffer", "day_keeps_everything", "day_after_keeps_nothing"],
)
def test_jpss1_trim_across_midnight(jpss1_dataset, day, n_packets):
    trimmed = trim_l1a_to_day_window(jpss1_dataset, day=day)
    assert trimmed.sizes["PACKET"] == n_packets
    for sample_dim in JPSS1_SAMPLE_DIMS:
        assert trimmed.sizes[sample_dim] == n_packets
    for index_var in ("ADGPS_packet_index", "ADCFA_packet_index"):
        index = trimmed[index_var].values
        if index.size:
            assert index.min() == 0
            assert index.max() == n_packets - 1


def test_jpss1_trimmed_day_writes_a_conforming_product(jpss1_dataset, tmp_path):
    trimmed = trim_l1a_to_day_window(jpss1_dataset, day=date(2021, 4, 8))
    _write_and_check(trimmed, LiberaApid.jpss_sc_pos, "PACKET_JPSS_TIME", tmp_path)


def test_jpss1_measured_coverage(jpss1_dataset):
    trimmed = trim_l1a_to_day_window(jpss1_dataset, day=date(2021, 4, 9))
    coverage = measure_time_axis_coverage(trimmed["ADGPS_JPSS_TIME"].values, day=date(2021, 4, 9))
    assert coverage.day_frac == pytest.approx(0.25, abs=1e-3)
    assert coverage.median_cadence == timedelta(seconds=1)
    assert abs(coverage.max_gap - timedelta(seconds=1)) < timedelta(milliseconds=1)


def test_jpss1_gate_on_l0_spans(jpss1_files):
    """The extractor's aware-UTC spans go straight into the gate; 0.9 s file seams merge."""
    day = date(2021, 4, 9)
    spans = [extract_data_time_range(path, LiberaApid.jpss_sc_pos.value) for path in jpss1_files]
    policy = get_packet_config(LiberaApid.jpss_sc_pos).day_coverage

    result = evaluate_day_coverage(spans, day=day, policy=policy)
    assert result.day_frac == pytest.approx(0.25, abs=1e-3)
    assert result.left_frac == pytest.approx(0.000115, abs=1e-5)
    assert result.right_frac == 0.0
    assert not result.is_complete

    abutting_only = evaluate_day_coverage(spans, day=day, policy=policy, seam_tolerance=timedelta(0))
    assert abutting_only.day_frac < result.day_frac


def test_wfov_trims_on_camera_time(test_ditl_camera_with_duplicate_packet, monkeypatch, tmp_path):
    """Images were taken on 2028-02-14 but their packets carry 2028-02-15 packet times."""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    with pytest.warns(UserWarning, match=r"Detected 1 duplicate PACKET_ICIE_TIME"):
        dataset = packets.parse_packets_to_l1a_dataset(
            packet_files=[test_ditl_camera_with_duplicate_packet], apid=LiberaApid.icie_wfov_sci.value
        )

    assert trim_l1a_to_day_window(dataset, day=date(2028, 2, 15)).sizes["PACKET"] == 0

    trimmed = trim_l1a_to_day_window(dataset, day=date(2028, 2, 14))
    assert trimmed.sizes["PACKET"] == 1078
    assert trimmed.sizes[CAMERA_TIME_COORD] == 3
    image_ids = trimmed[PACKET_IMAGE_ID_VAR].values
    assert np.all(image_ids != -1)
    sop_packets = trimmed[CAMERA_PACKET_INDEX_VAR].values
    assert set(image_ids[sop_packets]) == set(image_ids)
    flags = trimmed[MEM_DUMP_FLAGS_VAR].values
    assert all(flag.startswith(b"SOP") for flag in flags[sop_packets])
    _write_and_check(trimmed, LiberaApid.icie_wfov_sci, CAMERA_TIME_COORD, tmp_path)


def test_measure_sees_a_gap_inside_one_iov_file(test_iov_swc_event, monkeypatch):
    """One L0 file whose span looks continuous, with a 67 s hole in its samples."""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    dataset = packets.parse_packets_to_l1a_dataset(
        packet_files=[test_iov_swc_event], apid=LiberaApid.icie_cal_sample.value
    )
    coverage = measure_time_axis_coverage(dataset["CAL_SAMPLE_FPE_TIME"].values, day=date(2025, 12, 12))
    assert abs(coverage.max_gap - timedelta(seconds=67.25)) < timedelta(milliseconds=10)
    assert abs(coverage.median_cadence - timedelta(milliseconds=5)) < timedelta(microseconds=50)
    assert coverage.day_frac == pytest.approx(0.000778, abs=1e-5)
