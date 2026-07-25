"""Day-coverage evaluation for L1A combine completeness gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from libera_utils.l1a.day_window import DEFAULT_DAY_BUFFER

# Default fractions of each interval that must be covered by the union of L0 spans.
DEFAULT_DAY_COVERAGE_FRAC = 0.9
DEFAULT_BUFFER_COVERAGE_FRAC = 1.0


@dataclass(frozen=True, slots=True)
class DayCoverageResult:
    """Coverage fractions and pass/fail for left buffer, day core, and right buffer."""

    left_frac: float
    day_frac: float
    right_frac: float
    left_ok: bool
    day_ok: bool
    right_ok: bool
    n_intervals: int

    @property
    def is_complete(self) -> bool:
        """True when all three gates pass."""
        return self.left_ok and self.day_ok and self.right_ok


def _as_naive_utc(value: datetime) -> datetime:
    """Normalize aware/naive datetime to naive UTC."""
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _clip_interval(
    start: datetime,
    end: datetime,
    window_start: datetime,
    window_end: datetime,
) -> tuple[datetime, datetime] | None:
    """Return intersection of [start, end] with half-open [window_start, window_end), or None."""
    clipped_start = max(start, window_start)
    clipped_end = min(end, window_end)
    if clipped_start < clipped_end:
        return clipped_start, clipped_end
    return None


def _merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    """Merge overlapping/adjacent half-open intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: list[tuple[datetime, datetime]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _coverage_fraction(
    intervals: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
) -> float:
    """Fraction of [window_start, window_end) covered by the union of intervals."""
    window_seconds = (window_end - window_start).total_seconds()
    if window_seconds <= 0:
        return 0.0
    clipped: list[tuple[datetime, datetime]] = []
    for start, end in intervals:
        piece = _clip_interval(start, end, window_start, window_end)
        if piece is not None:
            clipped.append(piece)
    covered = sum((end - start).total_seconds() for start, end in _merge_intervals(clipped))
    return min(1.0, covered / window_seconds)


def day_core_and_buffer_bounds(
    day: date,
    buffer: timedelta = DEFAULT_DAY_BUFFER,
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime], tuple[datetime, datetime]]:
    """Return (left_buffer, day_core, right_buffer) as naive-UTC half-open intervals."""
    day_start = datetime.combine(day, datetime.min.time())
    day_end = datetime.combine(day + timedelta(days=1), datetime.min.time())
    left = (day_start - buffer, day_start)
    core = (day_start, day_end)
    right = (day_end, day_end + buffer)
    return left, core, right


def evaluate_day_coverage(
    intervals: list[tuple[datetime, datetime]],
    *,
    day: date,
    buffer: timedelta = DEFAULT_DAY_BUFFER,
    day_coverage_frac: float = DEFAULT_DAY_COVERAGE_FRAC,
    buffer_coverage_frac: float = DEFAULT_BUFFER_COVERAGE_FRAC,
    require_any_day_overlap: bool = False,
) -> DayCoverageResult:
    """Evaluate L0 time-span coverage against day core and midnight buffers.

    Parameters
    ----------
    intervals : list of (start, end)
        Effective L0 time ranges (data times preferred). Endpoints may be aware or naive UTC.
    day : date
        Applicable UTC calendar day.
    buffer : timedelta, optional
        Midnight buffer on each side (default 10 minutes).
    day_coverage_frac : float, optional
        Minimum fraction of ``[D, D+1)`` that must be covered for dense APIDs.
        Ignored when ``require_any_day_overlap`` is True.
    buffer_coverage_frac : float, optional
        Minimum fraction of each buffer interval that must be covered.
    require_any_day_overlap : bool, optional
        If True (sparse products such as WFOV), day core passes when any overlap with
        ``[D, D+1)`` exists, instead of requiring ``day_coverage_frac``.

    Returns
    -------
    DayCoverageResult
        Fractions and boolean gates for left buffer, day core, and right buffer.
    """
    normalized = [
        (_as_naive_utc(start), _as_naive_utc(end)) for start, end in intervals if start is not None and end is not None
    ]
    # Drop inverted/empty intervals
    normalized = [(s, e) for s, e in normalized if s < e]

    left, core, right = day_core_and_buffer_bounds(day, buffer)
    left_frac = _coverage_fraction(normalized, *left)
    day_frac = _coverage_fraction(normalized, *core)
    right_frac = _coverage_fraction(normalized, *right)

    left_ok = left_frac >= buffer_coverage_frac
    right_ok = right_frac >= buffer_coverage_frac
    if require_any_day_overlap:
        day_ok = day_frac > 0.0
    else:
        day_ok = day_frac >= day_coverage_frac

    return DayCoverageResult(
        left_frac=left_frac,
        day_frac=day_frac,
        right_frac=right_frac,
        left_ok=left_ok,
        day_ok=day_ok,
        right_ok=right_ok,
        n_intervals=len(normalized),
    )
