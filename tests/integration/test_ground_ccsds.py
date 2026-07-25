"""Integration tests for ground-test CCSDS discovery and data-time extraction."""

import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import DATA_TIME_INDEXED_APIDS, extract_data_time_range
from libera_utils.l1a.ground_ccsds import scan_ground_ccsds_file
from libera_utils.l1a.l1a_packet_configs import get_packet_config

pytestmark = pytest.mark.integration


def _has_packet_config(apid: LiberaApid) -> bool:
    """Return True if ``apid`` has an L1A packet configuration."""
    try:
        get_packet_config(apid)
    except KeyError:
        return False
    return True


@pytest.mark.parametrize(
    ("fixture_name", "expected_configured", "expected_data_time_apids", "sample_unknowns"),
    [
        (
            "test_ditl_camera_with_duplicate_packet",
            {
                LiberaApid.pev_sw_stat,
                LiberaApid.pec_sw_stat,
                LiberaApid.icie_rad_sample,
                LiberaApid.icie_wfov_sci,
                LiberaApid.icie_axis_sample,
                LiberaApid.icie_crit_hk,
                LiberaApid.icie_nom_hk,
                LiberaApid.icie_temp_hk,
            },
            {LiberaApid.icie_rad_sample, LiberaApid.icie_wfov_sci},
            (105, 116, 215, 1006, 1008, 1058, 1200),
        ),
        (
            "test_iov_swc_event",
            {
                LiberaApid.pev_sw_stat,
                LiberaApid.pec_sw_stat,
                LiberaApid.icie_rad_sample,
                LiberaApid.icie_cal_sample,
                LiberaApid.icie_axis_sample,
                LiberaApid.icie_crit_hk,
                LiberaApid.icie_nom_hk,
                LiberaApid.icie_temp_hk,
            },
            {LiberaApid.icie_rad_sample, LiberaApid.icie_cal_sample},
            (112, 115, 116, 212, 215, 1018, 1058, 1200),
        ),
        (
            "test_istr_gain_event",
            {
                LiberaApid.pev_sw_stat,
                LiberaApid.pec_sw_stat,
                LiberaApid.icie_rad_full,
                LiberaApid.icie_cal_full,
                LiberaApid.icie_cal_sample,
                LiberaApid.icie_axis_sample,
                LiberaApid.icie_crit_hk,
                LiberaApid.icie_nom_hk,
                LiberaApid.icie_temp_hk,
            },
            {LiberaApid.icie_rad_full, LiberaApid.icie_cal_full, LiberaApid.icie_cal_sample},
            (217, 218, 412, 1006, 1008, 1058, 1200),
        ),
        (
            "test_ccsds_2025_221_17_17_58",
            {
                LiberaApid.pev_sw_stat,
                LiberaApid.pec_sw_stat,
                LiberaApid.icie_rad_sample,
                LiberaApid.icie_wfov_sci,
                LiberaApid.icie_axis_sample,
                LiberaApid.icie_crit_hk,
                LiberaApid.icie_nom_hk,
                LiberaApid.icie_temp_hk,
            },
            {LiberaApid.icie_rad_sample, LiberaApid.icie_wfov_sci},
            (215, 216, 217, 218, 1006, 1008, 1200),
        ),
        (
            "test_ccsds_2025_218_18_41_30",
            {
                LiberaApid.pev_sw_stat,
                LiberaApid.pec_sw_stat,
                LiberaApid.icie_rad_sample,
                LiberaApid.icie_axis_sample,
                LiberaApid.icie_crit_hk,
                LiberaApid.icie_nom_hk,
                LiberaApid.icie_temp_hk,
            },
            {LiberaApid.icie_rad_sample},
            (112, 212, 217, 218, 1006, 1008, 1058, 1200),
        ),
    ],
    ids=("ditl", "iov_swc", "istr_gain", "istr_wfov", "istr_unused"),
)
def test_scan_ground_ccsds_file_across_captures(
    fixture_name,
    expected_configured,
    expected_data_time_apids,
    sample_unknowns,
    request,
):
    """Scan multi-APID ground captures for known/unknown APIDs and time spans."""
    packet_file = request.getfixturevalue(fixture_name)
    result = scan_ground_ccsds_file(packet_file, skip_header_bytes=8)

    assert set(result.known_apids).issubset(set(LiberaApid))
    assert all(int(apid) in result.all_apids for apid in result.known_apids)
    assert expected_configured.issubset(set(result.known_apids))
    assert expected_configured.issubset(set(result.time_spans))

    for unknown in sample_unknowns:
        assert unknown in result.all_apids
        assert unknown not in {int(apid) for apid in result.known_apids}

    # Known APIDs without packet config appear in known_apids but not time_spans
    for apid in result.known_apids:
        if _has_packet_config(apid):
            assert apid in result.time_spans
        else:
            assert apid not in result.time_spans

    assert set(result.time_spans).issubset(set(result.known_apids))
    assert expected_data_time_apids.issubset(set(result.time_spans))

    for apid, span in result.time_spans.items():
        assert span.first_packet_time <= span.last_packet_time
        if apid in DATA_TIME_INDEXED_APIDS:
            assert span.first_data_time is not None
            assert span.last_data_time is not None
            assert span.first_data_time <= span.last_data_time
        else:
            assert span.first_data_time is None
            assert span.last_data_time is None


@pytest.mark.parametrize(
    ("fixture_name", "apid", "expected_date"),
    [
        ("test_ditl_camera_with_duplicate_packet", LiberaApid.icie_wfov_sci, "2028-02-14"),
        ("test_ditl_camera_with_duplicate_packet", LiberaApid.icie_rad_sample, "2028-02-15"),
        ("test_istr_gain_event", LiberaApid.icie_rad_full, "2025-08-06"),
        ("test_istr_gain_event", LiberaApid.icie_cal_full, "2025-08-06"),
        ("test_istr_gain_event", LiberaApid.icie_cal_sample, "2025-08-06"),
        ("test_iov_swc_event", LiberaApid.icie_cal_sample, "2025-12-12"),
        ("test_iov_swc_event", LiberaApid.icie_rad_sample, "2025-12-12"),
        ("test_ccsds_2025_221_17_17_58", LiberaApid.icie_wfov_sci, "2025-08-09"),
        ("test_ccsds_2025_221_17_17_58", LiberaApid.icie_rad_sample, "2025-08-09"),
        ("test_ccsds_2025_218_18_41_30", LiberaApid.icie_rad_sample, "2025-08-06"),
    ],
    ids=(
        "ditl_wfov",
        "ditl_rad_sample",
        "istr_rad_full",
        "istr_cal_full",
        "istr_cal_sample",
        "iov_cal_sample",
        "iov_rad_sample",
        "istr_wfov",
        "istr_rad_sample",
        "istr41_rad_sample",
    ),
)
def test_extract_data_time_range_from_ground_ccsds(fixture_name, apid, expected_date, request):
    """Data-time extractors return science spans for all DATA_TIME_INDEXED_APIDS in fixtures."""
    packet_file = request.getfixturevalue(fixture_name)
    first, last = extract_data_time_range(packet_file, apid, skip_header_bytes=8)

    assert first <= last
    assert first.date().isoformat() == expected_date
    assert last.date().isoformat() == expected_date
