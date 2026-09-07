"""Integration tests for ground-test CCSDS scanning and data-time extraction."""

import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import extract_data_time_range
from libera_utils.l1a.ground_ccsds import scan_ground_ccsds_file

pytestmark = pytest.mark.integration


# Demuxed fixtures are one APID per file with no record header; see
# test_data/packets/libera_ditl_demux/notes.md. ``data_span`` is "interval" when the file holds
# more than one distinct data time, "point" for a single SOP, and None when data times are
# unavailable. A point span anywhere else would mean timestamps were dropped after the min/max
# collapse instead of before it.
@pytest.mark.parametrize(
    ("basename", "apid", "packet_date", "data_date", "data_span"),
    [
        ("LIBERA_SDC_1036_ccsds_2025_318_13_00_00", LiberaApid.icie_rad_sample, "2028-02-15", "2028-02-15", "interval"),
        (
            "LIBERA_SDC_1048_ccsds_2025_318_13_00_00",
            LiberaApid.icie_axis_sample,
            "2028-02-15",
            "2028-02-15",
            "interval",
        ),
        ("LIBERA_SDC_1057_ccsds_2025_318_13_00_00", LiberaApid.icie_nom_hk, "2028-02-15", None, None),
        ("LIBERA_SDC_1040_ccsds_2025_318_13_00_00", LiberaApid.icie_wfov_sci, "2028-02-15", "2028-02-14", "point"),
        ("LIBERA_SDC_1040_ccsds_2025_318_13_20_00", LiberaApid.icie_wfov_sci, "2028-02-15", None, None),
    ],
    ids=("rad_sample", "axis_sample", "nom_hk_no_data_times", "wfov_with_sop", "wfov_no_sop"),
)
def test_scan_demuxed_ground_ccsds_file(basename, apid, packet_date, data_date, data_span, test_ditl_demux_path):
    """Scanning a demuxed file resolves its APID from the name and returns its time spans."""
    span = scan_ground_ccsds_file(test_ditl_demux_path / basename)

    assert span.apid == apid
    assert span.first_packet_time <= span.last_packet_time
    assert span.first_packet_time.date().isoformat() == packet_date

    if data_date is None:
        assert span.first_data_time is None
        assert span.last_data_time is None
    else:
        assert span.first_data_time.date().isoformat() == data_date
        assert span.last_data_time.date().isoformat() == data_date
        if data_span == "interval":
            assert span.first_data_time < span.last_data_time
        else:
            assert span.first_data_time == span.last_data_time


def test_scan_demuxed_wfov_without_sop_is_degraded_not_failed(test_ditl_demux_path):
    """A WFOV file with no SOP keeps its packet span and explains the missing data times."""
    span = scan_ground_ccsds_file(test_ditl_demux_path / "LIBERA_SDC_1040_ccsds_2025_318_13_20_00")

    assert span.first_packet_time is not None
    assert span.first_data_time is None
    assert LiberaApid.icie_wfov_sci.name in span.degraded_reason


def test_demuxed_wfov_data_time_precedes_its_packet_time(test_ditl_demux_path):
    """Camera images are downlinked well after capture, so data time can precede packet time."""
    span = scan_ground_ccsds_file(test_ditl_demux_path / "LIBERA_SDC_1040_ccsds_2025_318_13_00_00")

    assert span.first_data_time < span.first_packet_time


# ``multi_time`` says the fixture supplies more than one distinct data time for that APID, so the
# span must be an interval. A WFOV span comes only from SOP packets, and test_ccsds_2025_221_17_17_58
# holds exactly one SOP, so a point span is correct there. Everywhere else a point span means
# timestamps were dropped after the min/max instead of before it.
@pytest.mark.parametrize(
    ("fixture_name", "apid", "expected_date", "multi_time"),
    [
        ("test_ditl_camera_with_duplicate_packet", LiberaApid.icie_wfov_sci, "2028-02-14", True),
        ("test_ditl_camera_with_duplicate_packet", LiberaApid.icie_rad_sample, "2028-02-15", True),
        ("test_ditl_camera_with_duplicate_packet", LiberaApid.icie_axis_sample, "2028-02-15", True),
        ("test_istr_gain_event", LiberaApid.icie_rad_full, "2025-08-06", True),
        ("test_istr_gain_event", LiberaApid.icie_cal_full, "2025-08-06", True),
        ("test_istr_gain_event", LiberaApid.icie_cal_sample, "2025-08-06", True),
        ("test_iov_swc_event", LiberaApid.icie_cal_sample, "2025-12-12", True),
        ("test_iov_swc_event", LiberaApid.icie_rad_sample, "2025-12-12", True),
        ("test_iov_swc_event", LiberaApid.icie_axis_sample, "2025-12-12", True),
        ("test_ccsds_2025_221_17_17_58", LiberaApid.icie_wfov_sci, "2025-08-09", False),  # single SOP
        ("test_ccsds_2025_221_17_17_58", LiberaApid.icie_rad_sample, "2025-08-09", True),
        ("test_ccsds_2025_218_18_41_30", LiberaApid.icie_rad_sample, "2025-08-06", True),
    ],
    ids=(
        "ditl_wfov",
        "ditl_rad_sample",
        "ditl_axis_sample",
        "istr_rad_full",
        "istr_cal_full",
        "istr_cal_sample",
        "iov_cal_sample",
        "iov_rad_sample",
        "iov_axis_sample",
        "istr_wfov",
        "istr_rad_sample",
        "istr41_rad_sample",
    ),
)
def test_extract_data_time_range_from_ground_ccsds(fixture_name, apid, expected_date, multi_time, request):
    """Data-time extractors return science spans for all DATA_TIME_INDEXED_APIDS in fixtures."""
    packet_file = request.getfixturevalue(fixture_name)
    first, last = extract_data_time_range(packet_file, apid, skip_header_bytes=8)

    if multi_time:
        assert first < last
    else:
        assert first == last
    assert first.date().isoformat() == expected_date
    assert last.date().isoformat() == expected_date
