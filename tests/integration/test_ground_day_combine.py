"""Integration: multi-file ground CCSDS day assembly primitives (coverage + parse + trim)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_coverage import evaluate_day_coverage
from libera_utils.l1a.day_window import assert_data_times_unique_monotonic, trim_l1a_to_day_window
from libera_utils.l1a.ground_ccsds import scan_ground_ccsds_file
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import parse_packets_to_l1a_dataset

pytestmark = pytest.mark.integration

# UTC calendar day for DOY 193 in 2026
_GROUND_DAY = date(2026, 7, 12)
_APID = LiberaApid.icie_nom_hk


def _sorted_day_files(day_dir: Path) -> list[Path]:
    return sorted(p for p in day_dir.iterdir() if p.is_file() and p.name.startswith("ccsds_"))


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
        scan = scan_ground_ccsds_file(path, skip_header_bytes=8)
        assert _APID in scan.time_spans
        span = scan.time_spans[_APID]
        intervals.append((span.first_packet_time, span.last_packet_time))

    coverage = evaluate_day_coverage(intervals, day=_GROUND_DAY)
    assert coverage.n_intervals == len(files)
    # Sparse short chunks do not meet dense day/buffer fraction gates.
    assert not coverage.is_complete

    l1a_ds = parse_packets_to_l1a_dataset(
        [str(p) for p in files],
        int(_APID),
        ground_data=True,
        skip_header_bytes=8,
    )
    time_coord = get_packet_config(_APID).packet_time_coordinate
    assert time_coord in l1a_ds

    trimmed = trim_l1a_to_day_window(l1a_ds, day=_GROUND_DAY, time_coord=time_coord)
    assert trimmed.sizes.get(trimmed[time_coord].dims[0], 0) > 0
    assert_data_times_unique_monotonic(trimmed, time_coord, ground_data=True)
