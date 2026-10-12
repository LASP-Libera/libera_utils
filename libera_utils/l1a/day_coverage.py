"""Day-coverage evaluation for L1A combine completeness gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import numpy as np

from libera_utils.l1a.day_window import day_core_and_buffer_bounds, day_window_bounds
from libera_utils.l1a.l1a_packet_configs import CoverageMode, DayCoveragePolicy
from libera_utils.l1a.packet_slicing import assert_no_missing_times

# Spacing at or below this is treated as continuous rather than as a gap. It absorbs two
# effects that are not missing data. First, a file's span is recorded over sample *timestamps*,
# ``[t_first, t_last]``, while coverage is about time occupied: a sample occupies
# ``[t_k, t_k + sample_period)``, so every file under-reports by one period and N concatenated
# files lose N periods. Second, the RAD FPE leaves ~10 ms of dead time between packets on ~26%
# of packets, which appears as a 15 ms step between the last sample of one packet and the first
# of the next. One second is 40x the largest intra-file sample gap measured in DITL2 (25 ms) and
# four orders of magnitude below the smallest real dropout the gate should catch (a missing
# 2-hour chunk).
DEFAULT_SEAM_TOLERANCE = timedelta(seconds=1)


@dataclass(frozen=True, slots=True)
class DayCoverageResult:
    """Coverage fractions and pass/fail for left buffer, day core, and right buffer.

    Attributes
    ----------
    mode : CoverageMode
        Mode of the policy the result was judged under.
    left_frac, day_frac, right_frac : float
        Covered fraction, in ``[0, 1]``, of the left midnight buffer, the day core ``[D, D+1)``
        and the right midnight buffer.
    left_ok, day_ok, right_ok : bool
        Whether each region passes the policy. Under ``EVENT_DRIVEN`` all three equal presence.
    """

    mode: CoverageMode
    left_frac: float
    day_frac: float
    right_frac: float
    left_ok: bool
    day_ok: bool
    right_ok: bool

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


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]],
    tolerance: timedelta,
) -> list[tuple[datetime, datetime]]:
    """Merge overlapping/adjacent half-open intervals, joining across gaps up to ``tolerance``."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: list[tuple[datetime, datetime]] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + tolerance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _coverage_fraction(
    intervals: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
    tolerance: timedelta,
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
    covered = sum((end - start).total_seconds() for start, end in _merge_intervals(clipped, tolerance))
    return min(1.0, covered / window_seconds)


def overlaps_day_window(start: datetime, end: datetime, *, day: date) -> bool:
    """Return True if the closed span ``[start, end]`` touches the closed day window of ``day``.

    The window is :func:`~libera_utils.l1a.day_window.day_window_bounds`, the same one
    :func:`~libera_utils.l1a.day_window.trim_l1a_to_day_window` keeps. Endpoints may be aware or
    naive UTC.
    """
    window_start, window_end = day_window_bounds(day)
    return _as_naive_utc(start) <= window_end and _as_naive_utc(end) >= window_start


def evaluate_day_coverage(
    intervals: list[tuple[datetime | None, datetime | None]],
    *,
    day: date,
    policy: DayCoveragePolicy,
    seam_tolerance: timedelta = DEFAULT_SEAM_TOLERANCE,
) -> DayCoverageResult:
    """Evaluate L0 time-span coverage against day core and midnight buffers.

    Pairs with a ``None`` endpoint and inverted pairs (``start > end``) are ignored. A
    zero-length span counts toward presence but adds nothing to the fractions.

    Parameters
    ----------
    intervals : list of (start, end)
        Effective L0 time ranges (data times preferred). Endpoints may be aware or naive UTC.
    day : date
        Applicable UTC calendar day.
    policy : DayCoveragePolicy
        Completeness rule for the APID being combined, ``get_packet_config(apid).day_coverage``.
    seam_tolerance : timedelta, optional
        Spacing between two spans that still counts as continuous. Default
        ``DEFAULT_SEAM_TOLERANCE`` (1 s). Pass ``timedelta(0)`` to require exact abutment.

    Returns
    -------
    DayCoverageResult
        Fractions and boolean gates for left buffer, day core, and right buffer. Under
        ``EVENT_DRIVEN`` all three ``*_ok`` equal presence: whether any span overlaps the day
        window (:func:`overlaps_day_window`).
    """
    spans = [
        (_as_naive_utc(start), _as_naive_utc(end)) for start, end in intervals if start is not None and end is not None
    ]
    spans = [(s, e) for s, e in spans if s <= e]
    nonempty = [(s, e) for s, e in spans if s < e]

    left, core, right = day_core_and_buffer_bounds(day)
    left_frac = _coverage_fraction(nonempty, *left, tolerance=seam_tolerance)
    day_frac = _coverage_fraction(nonempty, *core, tolerance=seam_tolerance)
    right_frac = _coverage_fraction(nonempty, *right, tolerance=seam_tolerance)

    if policy.mode is CoverageMode.EVENT_DRIVEN:
        # Presence anywhere in the buffered window is the whole test. Gating the buffers on a
        # fraction as well would make an event-driven day unreachable: an APID that is only on
        # for an event almost never has data at both midnights.
        present = any(overlaps_day_window(start, end, day=day) for start, end in spans)
        left_ok = day_ok = right_ok = present
    else:
        left_ok = left_frac >= policy.buffer_coverage_frac
        right_ok = right_frac >= policy.buffer_coverage_frac
        day_ok = day_frac >= policy.day_coverage_frac

    return DayCoverageResult(
        mode=policy.mode,
        left_frac=left_frac,
        day_frac=day_frac,
        right_frac=right_frac,
        left_ok=left_ok,
        day_ok=day_ok,
        right_ok=right_ok,
    )


# A spacing this many times the axis's own median spacing is treated as a real gap rather than
# jitter. The axis calibrates the threshold itself, so one number covers a 1 Hz housekeeping APID
# and a 5 s camera cadence alike.
DEFAULT_GAP_FACTOR = 5.0


@dataclass(frozen=True, slots=True)
class TimeAxisCoverage:
    """What a granule's own time axis says about its occupancy of a day and its buffers.

    Attributes
    ----------
    day_frac, left_frac, right_frac : float
        Occupied fraction, in ``[0, 1]``, of the day core ``[D, D+1)`` and the left and right
        midnight buffers.
    median_cadence : timedelta
        Median spacing between consecutive samples; ``DEFAULT_SEAM_TOLERANCE`` for a single
        sample and zero for an empty axis.
    max_gap : timedelta
        Largest spacing between consecutive samples. Equals ``median_cadence`` on an evenly
        spaced axis, and is zero for an empty or single-sample axis.
    n_times : int
        Number of samples on the axis.
    """

    day_frac: float
    left_frac: float
    right_frac: float
    median_cadence: timedelta
    max_gap: timedelta
    n_times: int


def measure_time_axis_coverage(
    times: np.ndarray,
    *,
    day: date,
) -> TimeAxisCoverage:
    """Measure how much of a day a granule's time axis actually occupies.

    This is a measurement, not a gate. :func:`evaluate_day_coverage` works from whole-file L0
    spans and so cannot see a gap *inside* a file; this works from the samples themselves.

    Samples count as continuous until their spacing exceeds ``DEFAULT_GAP_FACTOR`` times the
    axis's median spacing (never less than ``DEFAULT_SEAM_TOLERANCE``), and the last sample of
    each run is credited with one median cadence of occupancy.

    Parameters
    ----------
    times : numpy.ndarray
        Datetime64 (or datetime-like) values from one science time axis. Need not be sorted.
    day : date
        Applicable UTC calendar day.

    Returns
    -------
    TimeAxisCoverage
        Fractions for the day core and both buffers, plus the axis's median cadence and largest
        gap. An empty axis returns all zeros. A single sample has ``median_cadence`` equal to
        ``DEFAULT_SEAM_TOLERANCE`` and occupies that long.

    Raises
    ------
    MissingDataTimeError
        If any value is ``NaT``. A data-time axis carries a time for every sample, so a missing
        one means the producer is broken.
    """
    values = np.asarray(times).astype("datetime64[us]").ravel()
    assert_no_missing_times(values, "time axis")
    values.sort()
    if values.size == 0:
        return TimeAxisCoverage(0.0, 0.0, 0.0, timedelta(0), timedelta(0), 0)

    if values.size == 1:
        cadence_us = int(DEFAULT_SEAM_TOLERANCE.total_seconds() * 1e6)
        max_gap_us = 0
        run_bounds = [(values[0], values[0])]
    else:
        spacings_us = np.diff(values).astype("int64")
        cadence_us = int(np.median(spacings_us))
        max_gap_us = int(spacings_us.max())
        threshold_us = max(cadence_us * DEFAULT_GAP_FACTOR, DEFAULT_SEAM_TOLERANCE.total_seconds() * 1e6)
        # Split at real gaps; each remaining run is one continuously occupied interval.
        break_positions = np.flatnonzero(spacings_us > threshold_us) + 1
        run_bounds = [(run[0], run[-1]) for run in np.split(values, break_positions)]

    occupancy = timedelta(microseconds=max(cadence_us, 1))
    intervals = [(first.astype(datetime), last.astype(datetime) + occupancy) for first, last in run_bounds]

    left, core, right = day_core_and_buffer_bounds(day)
    return TimeAxisCoverage(
        day_frac=_coverage_fraction(intervals, *core, tolerance=timedelta(0)),
        left_frac=_coverage_fraction(intervals, *left, tolerance=timedelta(0)),
        right_frac=_coverage_fraction(intervals, *right, tolerance=timedelta(0)),
        median_cadence=timedelta(microseconds=cadence_us),
        max_gap=timedelta(microseconds=max_gap_us),
        n_times=int(values.size),
    )
