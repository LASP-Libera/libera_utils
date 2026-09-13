"""Unit tests for L1A day-coverage completeness gates."""

from datetime import UTC, date, datetime, timedelta

import pytest

from libera_utils.l1a.day_coverage import (
    DEFAULT_BUFFER_COVERAGE_FRAC,
    DEFAULT_DAY_COVERAGE_FRAC,
    DEFAULT_SEAM_TOLERANCE,
    evaluate_day_coverage,
)


def test_full_day_and_buffers_complete():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 16, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day)
    assert result.is_complete
    assert result.left_frac == 1.0
    assert result.day_frac == 1.0
    assert result.right_frac == 1.0


def test_missing_right_buffer_incomplete():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 16, 0, 0, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day)
    assert result.left_ok
    assert result.day_ok
    assert not result.right_ok
    assert not result.is_complete


def test_partial_day_below_default_frac():
    day = date(2028, 2, 15)
    # Only 12 hours of the day core
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 0, 0, tzinfo=UTC), datetime(2028, 2, 15, 12, 0, tzinfo=UTC)),
        (datetime(2028, 2, 16, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day)
    assert result.day_frac == 0.5
    assert not result.day_ok
    assert not result.is_complete


def test_sparse_day_any_overlap_passes_core():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 12, 0, tzinfo=UTC), datetime(2028, 2, 15, 12, 5, tzinfo=UTC)),
        (datetime(2028, 2, 16, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    dense = evaluate_day_coverage(intervals, day=day)
    assert not dense.day_ok
    sparse = evaluate_day_coverage(intervals, day=day, require_any_day_overlap=True)
    assert sparse.day_ok
    assert sparse.is_complete


def test_overlapping_intervals_merged():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 15, 0, 0, tzinfo=UTC), datetime(2028, 2, 15, 14, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 12, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 5, tzinfo=UTC)),
        (datetime(2028, 2, 15, 23, 55, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day)
    assert result.day_frac == 1.0
    assert result.is_complete


def test_defaults_exported():
    assert DEFAULT_DAY_COVERAGE_FRAC == 0.9
    assert DEFAULT_BUFFER_COVERAGE_FRAC == 0.99
    assert DEFAULT_SEAM_TOLERANCE == timedelta(seconds=1)
    assert timedelta(minutes=10).total_seconds() == 600


# Seams measured in DITL2 APID 1036: one sample period, the common FPE dead-time step, and the
# largest intra-file gap seen anywhere in the campaign.
_MEASURED_SEAMS_MS = (5, 15, 25)


def _day_spanning_intervals(day: date, seam: timedelta) -> list[tuple[datetime, datetime]]:
    """Three back-to-back files covering both buffers and the whole core, split by ``seam``."""
    left_start = datetime.combine(day, datetime.min.time()) - timedelta(minutes=10)
    right_end = datetime.combine(day + timedelta(days=1), datetime.min.time()) + timedelta(minutes=10)
    midnight = datetime.combine(day, datetime.min.time())
    next_midnight = datetime.combine(day + timedelta(days=1), datetime.min.time())
    # Seams land inside each buffer, where DITL2 two-hour chunk boundaries actually fall.
    first_seam = midnight + timedelta(seconds=24)
    second_seam = next_midnight + timedelta(seconds=45)
    return [
        (left_start, first_seam),
        (first_seam + seam, second_seam),
        (second_seam + seam, right_end),
    ]


@pytest.mark.parametrize("seam_ms", _MEASURED_SEAMS_MS)
def test_measured_seams_reach_full_buffer_coverage(seam_ms):
    """A seam at any measured size must still score 1.0, not merely clear the 0.99 gate.

    Demanding ``buffer_coverage_frac=1.0`` isolates the tolerance: before it existed, every
    midnight crossing in DITL2 put a seam inside the right buffer, so ``right_ok`` was
    unreachable for every day of the campaign.
    """
    day = date(2026, 7, 12)
    result = evaluate_day_coverage(
        _day_spanning_intervals(day, timedelta(milliseconds=seam_ms)),
        day=day,
        buffer_coverage_frac=1.0,
    )
    assert result.is_complete
    assert result.left_frac == 1.0
    assert result.right_frac == 1.0


def test_real_gap_in_right_buffer_fails():
    day = date(2026, 7, 12)
    result = evaluate_day_coverage(_day_spanning_intervals(day, timedelta(seconds=30)), day=day)
    assert result.left_ok
    assert not result.right_ok
    assert not result.is_complete


def test_gap_just_over_seam_tolerance_is_not_merged():
    day = date(2026, 7, 12)
    over = DEFAULT_SEAM_TOLERANCE + timedelta(milliseconds=1)
    result = evaluate_day_coverage(_day_spanning_intervals(day, over), day=day, buffer_coverage_frac=1.0)
    assert not result.right_ok
    under = DEFAULT_SEAM_TOLERANCE - timedelta(milliseconds=1)
    merged = evaluate_day_coverage(_day_spanning_intervals(day, under), day=day, buffer_coverage_frac=1.0)
    assert merged.is_complete


def test_seam_tolerance_is_tunable_per_caller():
    day = date(2026, 7, 12)
    intervals = _day_spanning_intervals(day, timedelta(milliseconds=15))
    strict = evaluate_day_coverage(intervals, day=day, buffer_coverage_frac=1.0, seam_tolerance=timedelta(0))
    assert not strict.is_complete


def test_buffer_frac_absorbs_a_small_real_dropout():
    """0.99 of a 600 s buffer allows a 6 s outage, which the tolerance alone would not."""
    day = date(2026, 7, 12)
    six_seconds = evaluate_day_coverage(_day_spanning_intervals(day, timedelta(seconds=6)), day=day)
    assert six_seconds.is_complete
    seven_seconds = evaluate_day_coverage(_day_spanning_intervals(day, timedelta(seconds=7)), day=day)
    assert not seven_seconds.right_ok


# The 14 APID-1036 File Metadata spans for 2026-07-12 from the DITL2 campaign, as read from the
# demuxed ground captures. The incident that motivated the seam tolerance logged
# left=0.0 day=0.917100486261574 right=0.9999916666666667 against these same files, because
# searchable rows were clamped to their applicable date and only the last row per file survived
# PK de-duplication.
_DITL2_APID_1036_2026_07_12 = [
    ("2026-07-11T22:00:31.700565", "2026-07-12T00:00:23.883009"),
    ("2026-07-12T00:00:23.888009", "2026-07-12T02:00:19.666594"),
    ("2026-07-12T02:00:19.671594", "2026-07-12T04:00:40.270054"),
    ("2026-07-12T04:00:40.275054", "2026-07-12T06:00:41.815796"),
    ("2026-07-12T06:00:41.820796", "2026-07-12T08:00:39.811309"),
    ("2026-07-12T08:00:39.816309", "2026-07-12T10:00:28.813907"),
    ("2026-07-12T10:00:28.818907", "2026-07-12T12:00:41.996197"),
    ("2026-07-12T12:00:42.011205", "2026-07-12T14:00:54.019009"),
    ("2026-07-12T14:00:54.034010", "2026-07-12T16:00:45.230194"),
    ("2026-07-12T16:00:45.245194", "2026-07-12T18:00:40.215743"),
    ("2026-07-12T18:00:40.230747", "2026-07-12T20:00:31.673691"),
    ("2026-07-12T20:00:31.678691", "2026-07-12T22:00:37.577026"),
    ("2026-07-12T22:00:37.582026", "2026-07-13T00:00:44.973576"),
    ("2026-07-13T00:00:44.978576", "2026-07-13T02:00:00.311886"),
]


def test_ditl2_full_day_gates_complete():
    intervals = [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in _DITL2_APID_1036_2026_07_12]
    result = evaluate_day_coverage(intervals, day=date(2026, 7, 12))
    assert result.left_frac == 1.0
    assert result.day_frac == 1.0
    assert result.right_frac == 1.0
    assert result.is_complete


def test_ditl2_day_clamped_to_applicable_date_reproduces_the_incident():
    """Guard the other half of the fix: clamped spans still fail, so unclamping is load-bearing."""
    clamped = []
    for a, b in _DITL2_APID_1036_2026_07_12:
        start, end = datetime.fromisoformat(a), datetime.fromisoformat(b)
        # Only the final applicable date's fragment survived selected[record.PK].
        last_day = datetime.combine(end.date(), datetime.min.time())
        clamped.append((max(start, last_day), end))
    result = evaluate_day_coverage(clamped, day=date(2026, 7, 12))
    assert result.left_frac == 0.0
    assert not result.is_complete
