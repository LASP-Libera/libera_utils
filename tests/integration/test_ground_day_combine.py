"""Integration: multi-file ground CCSDS day assembly primitives (coverage + parse + trim)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_coverage import coverage_policy_for_apid, evaluate_day_coverage, measure_time_axis_coverage
from libera_utils.l1a.day_window import assert_data_times_unique_monotonic, trim_l1a_to_day_window
from libera_utils.l1a.ground_ccsds import GROUND_CCSDS_SKIP_HEADER_BYTES, scan_ground_ccsds_file
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import parse_packets_to_l1a_dataset
from libera_utils.l1a.wfov_image_metadata import CAMERA_TIME_COORD

pytestmark = pytest.mark.integration

# UTC calendar day for DOY 193 in 2026
_GROUND_DAY = date(2026, 7, 12)
_APID = LiberaApid.icie_nom_hk

# Muxed captures still carry an 8-byte record header before each CCSDS primary header; the
# demuxed DITL2 set does not.
_MUXED_SKIP_HEADER_BYTES = 8

# Full-day demuxed DITL2 captures. Not in the repo (~24 MB per two-hour granule), so the
# full-day gate test skips unless the directory is present.
_DITL2_ENV_VAR = "LIBERA_DITL2_PREPROCESSED_DIR"
_DITL2_DEFAULT_DIR = Path("/Users/mawa7160/dev/data/LIBERA/DITL2/SDC_Preprocessed")
_DITL2_APID = LiberaApid.icie_rad_sample


def _sorted_day_files(day_dir: Path) -> list[Path]:
    return sorted(p for p in day_dir.iterdir() if p.is_file() and p.name.startswith("ccsds_"))


def _ditl2_dir() -> Path:
    return Path(os.environ.get(_DITL2_ENV_VAR, _DITL2_DEFAULT_DIR))


@pytest.mark.integration
def test_ground_day_coverage_parse_trim(test_ground_day_ccsds_dir: Path) -> None:
    """Scan DOY-193 ground set, gate coverage, then multi-file decode/trim with ground_data=True.

    Captures are short (~1 min) chunks spaced ~2 hr apart, so default dense coverage for a full
    UTC day is expected to be incomplete. Production would skip or force; this test still runs the
    shared parse → trim → uniqueness path used when combining is forced.
    """
    files = _sorted_day_files(test_ground_day_ccsds_dir)
    assert len(files) == 14

    intervals: list[tuple] = []
    for path in files:
        span = scan_ground_ccsds_file(path, apid=int(_APID), skip_header_bytes=_MUXED_SKIP_HEADER_BYTES)
        assert span is not None
        assert span.apid == _APID
        intervals.append((span.first_packet_time, span.last_packet_time))

    coverage = evaluate_day_coverage(intervals, day=_GROUND_DAY, policy=coverage_policy_for_apid(_APID))
    assert coverage.n_intervals == len(files)
    # Sparse short chunks do not meet dense day/buffer fraction gates.
    assert not coverage.is_complete

    l1a_ds = parse_packets_to_l1a_dataset(
        [str(p) for p in files],
        int(_APID),
        ground_data=True,
        skip_header_bytes=_MUXED_SKIP_HEADER_BYTES,
    )
    time_coord = get_packet_config(_APID).packet_time_coordinate
    assert time_coord in l1a_ds

    trimmed = trim_l1a_to_day_window(l1a_ds, day=_GROUND_DAY, time_coord=time_coord)
    assert trimmed.sizes.get(trimmed[time_coord].dims[0], 0) > 0
    assert_data_times_unique_monotonic(trimmed, time_coord, ground_data=True)


@pytest.mark.integration
@pytest.mark.skipif(not _ditl2_dir().is_dir(), reason=f"DITL2 captures not available; set ${_DITL2_ENV_VAR}")
def test_contiguous_ditl2_day_gates_complete() -> None:
    """A contiguous UTC day of demuxed DITL2 RAD captures must gate complete.

    This is the assertion that would have caught both halves of the combine defect: spans
    clamped to their applicable date drove ``left_frac`` to 0.0, and requiring exact abutment
    across the two-hour file seams made ``right_ok`` unreachable.
    """
    day_dir = _ditl2_dir()
    files = sorted(day_dir.glob(f"LIBERA_SDC_{int(_DITL2_APID)}_ccsds_2026_19[234]_*"))
    assert files, f"No APID {int(_DITL2_APID)} captures under {day_dir}"

    intervals: list[tuple] = []
    for path in files:
        span = scan_ground_ccsds_file(path, skip_header_bytes=GROUND_CCSDS_SKIP_HEADER_BYTES)
        if span is None or span.first_data_time is None or span.last_data_time is None:
            continue
        intervals.append((span.first_data_time, span.last_data_time))

    coverage = evaluate_day_coverage(intervals, day=_GROUND_DAY, policy=coverage_policy_for_apid(_DITL2_APID))
    assert coverage.left_frac == 1.0
    assert coverage.day_frac == 1.0
    assert coverage.right_frac == 1.0
    assert coverage.is_complete


# WFOV ground captures whose CAMERA_TIME runs on a different clock than their packet time.
_WFOV_ENV_VAR = "LIBERA_WFOV_DAY_DIR"
_WFOV_DEFAULT_DIR = Path("/Users/mawa7160/dev/data/LIBERA/CCSDS/L1A_24Hr_testing/1040")


def _wfov_dir() -> Path:
    return Path(os.environ.get(_WFOV_ENV_VAR, _WFOV_DEFAULT_DIR))


@pytest.mark.integration
@pytest.mark.skipif(not _wfov_dir().is_dir(), reason=f"WFOV captures not available; set ${_WFOV_ENV_VAR}")
def test_wfov_packet_and_camera_axes_disagree():
    """The two recorded coverage bases must be able to disagree, because on WFOV they do.

    In these captures CAMERA_TIME lags PACKET_ICIE_TIME by hours and the lag grows, so a two-hour
    granule of packets carries only about an hour of camera time. One coverage number would hide
    which clock it described; this is why the granule records both.
    """
    granule = _wfov_dir() / "LIBERA_SDC_1040_ccsds_2026_193_04_00_00"
    if not granule.is_file():
        pytest.skip(f"{granule.name} not present")

    apid = int(LiberaApid.icie_wfov_sci)
    l1a_ds = parse_packets_to_l1a_dataset([str(granule)], apid, ground_data=True, skip_header_bytes=0)
    packet_times = l1a_ds[get_packet_config(apid).packet_time_coordinate].values
    camera_times = l1a_ds[CAMERA_TIME_COORD].values

    packet_coverage = measure_time_axis_coverage(packet_times, day=_GROUND_DAY)
    camera_coverage = measure_time_axis_coverage(camera_times, day=_GROUND_DAY)

    assert packet_coverage.day_frac > camera_coverage.day_frac
    # The camera clock is contiguous at its ~5 s cadence; it is offset, not gappy.
    assert camera_coverage.median_cadence.total_seconds() == pytest.approx(5.0, abs=0.1)
    packet_span = packet_times.max() - packet_times.min()
    camera_span = camera_times.max() - camera_times.min()
    assert camera_span < packet_span
