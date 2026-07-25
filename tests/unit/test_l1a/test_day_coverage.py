"""Unit tests for L1A day-coverage completeness gates."""

from datetime import UTC, date, datetime, timedelta

from libera_utils.l1a.day_coverage import (
    DEFAULT_BUFFER_COVERAGE_FRAC,
    DEFAULT_DAY_COVERAGE_FRAC,
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
    assert DEFAULT_BUFFER_COVERAGE_FRAC == 1.0
    assert timedelta(minutes=10).total_seconds() == 600
