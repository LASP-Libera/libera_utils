"""Day-coverage evaluation for the L1A combine completeness check."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np

from libera_utils.l1a.day_window import day_bounds
from libera_utils.l1a.l1a_packet_configs import CoverageMode, DayCoveragePolicy

# Spacing at or below this counts as continuous: a span runs from a file's first to its last
# sample, so it under-reports the time the file occupies by one sample period.
# TODO[LIBSDC-876]: confirm the inter-packet timing with ops and set this from it.
DEFAULT_SEAM_TOLERANCE = timedelta(seconds=1)

# A spacing more than this many times an axis's median spacing is a gap. The axis sets its own
# threshold, so one number serves a 1 Hz housekeeping APID and a 5 s camera cadence alike.
DEFAULT_GAP_FACTOR = 5.0


@dataclass(frozen=True, slots=True)
class DayCoverageResult:
    """Whether L0 spans reach both edges of the buffered day window, and how much of it they cover.

    Attributes
    ----------
    mode : CoverageMode
        Mode of the policy the result was judged under.
    covers_window : bool
        Whether the spans that touch the buffered day window ``[D - B, D + 1 + B]`` reach both of
        its edges. Under ``PRESENCE``, equals presence.
    window_frac : float
        Covered fraction, in ``[0, 1]``, of the buffered day window.
    window_frac_ok : bool
        Whether ``window_frac`` meets the policy's ``window_coverage_frac``. Under ``PRESENCE``,
        equals presence.
    """

    mode: CoverageMode
    covers_window: bool
    window_frac: float
    window_frac_ok: bool

    @property
    def is_complete(self) -> bool:
        """True when both checks pass."""
        return self.covers_window and self.window_frac_ok


def _coverage_fraction(
    spans: list[tuple[datetime, datetime]],
    window_start: datetime,
    window_end: datetime,
    tolerance: timedelta,
) -> float:
    """Fraction of ``[window_start, window_end)`` covered by ``spans``, joining gaps up to ``tolerance``."""
    covered = timedelta(0)
    run: tuple[datetime, datetime] | None = None
    for start, end in sorted((max(s, window_start), min(e, window_end)) for s, e in spans):
        if start >= end:
            continue
        if run is not None and start <= run[1] + tolerance:
            run = (run[0], max(run[1], end))
            continue
        if run is not None:
            covered += run[1] - run[0]
        run = (start, end)
    if run is not None:
        covered += run[1] - run[0]
    return min(1.0, covered / (window_end - window_start))


def overlaps_day_window(start: datetime, end: datetime, *, day: date) -> bool:
    """Return True if the closed span ``[start, end]`` touches the closed day window of ``day``.

    The window is the one :func:`~libera_utils.l1a.day_window.trim_l1a_to_day_window` keeps.

    Parameters
    ----------
    start, end : datetime
        Span endpoints, timezone-aware.
    day : datetime.date
        Applicable UTC calendar day.

    Returns
    -------
    bool
        Whether the span and the window share at least one instant.

    Raises
    ------
    TypeError
        If ``start`` or ``end`` is timezone-naive.
    """
    bounds = day_bounds(day)
    starts_by_window_end = start <= bounds.window_end
    ends_after_window_start = end >= bounds.window_start
    return starts_by_window_end and ends_after_window_start


def evaluate_day_coverage(
    intervals: list[tuple[datetime | None, datetime | None]],
    *,
    day: date,
    policy: DayCoveragePolicy,
    seam_tolerance: timedelta = DEFAULT_SEAM_TOLERANCE,
) -> DayCoverageResult:
    """Evaluate L0 time-span coverage of the buffered day window ``[D - B, D + 1 + B]``.

    Under ``FRACTION`` a day passes two checks: the spans that touch the window start by its
    left edge and end by its right edge (``covers_window``), and the spans cover at least
    ``policy.window_coverage_frac`` of the window (``window_frac_ok``). A gap counts the same
    against the fraction wherever it falls in the window.

    Pairs with a ``None`` endpoint and inverted pairs (``start > end``) are ignored. A span whose
    start equals its end, such as a file holding one time, counts toward presence and toward
    reaching a window edge but adds nothing to the fraction.

    Parameters
    ----------
    intervals : list of tuple of datetime
        Effective L0 time ranges (data times preferred), timezone-aware.
    day : datetime.date
        Applicable UTC calendar day.
    policy : DayCoveragePolicy
        Completeness rule for the APID being combined, ``get_packet_config(apid).day_coverage``.
    seam_tolerance : datetime.timedelta, optional
        Spacing between two spans that still counts as continuous, and the distance by which a
        span may fall short of a window edge and still reach it. Default
        ``DEFAULT_SEAM_TOLERANCE`` (1 s). Pass ``timedelta(0)`` to require exact abutment.

    Returns
    -------
    DayCoverageResult
        The window fraction and both checks. Under ``PRESENCE`` both checks equal presence:
        whether any span overlaps the day window (:func:`overlaps_day_window`).

    Raises
    ------
    TypeError
        If an endpoint is timezone-naive.
    """
    # TODO[LIBSDC-878]: measure the time each L0 file occupies; its span cannot show a gap inside it.
    spans = [(start, end) for start, end in intervals if start is not None and end is not None and start <= end]

    bounds = day_bounds(day)
    in_window = [(start, end) for start, end in spans if overlaps_day_window(start, end, day=day)]
    window_frac = _coverage_fraction(spans, bounds.window_start, bounds.window_end, seam_tolerance)

    if policy.mode is CoverageMode.PRESENCE:
        # An APID that is on only for an event almost never has data at both midnights.
        covers_window = window_frac_ok = bool(in_window)
    else:
        covers_window = bool(in_window) and (
            min(start for start, _ in in_window) <= bounds.window_start + seam_tolerance
            and max(end for _, end in in_window) >= bounds.window_end - seam_tolerance
        )
        window_frac_ok = window_frac >= policy.window_coverage_frac

    return DayCoverageResult(
        mode=policy.mode,
        covers_window=covers_window,
        window_frac=window_frac,
        window_frac_ok=window_frac_ok,
    )


@dataclass(frozen=True, slots=True)
class TimeAxisCoverage:
    """What a granule's own time axis says about its occupancy of a day and its buffers.

    Attributes
    ----------
    day_frac : float
        Occupied fraction, in ``[0, 1]``, of the day core ``[D, D+1)``.
    left_frac : float
        Occupied fraction, in ``[0, 1]``, of the left midnight buffer.
    right_frac : float
        Occupied fraction, in ``[0, 1]``, of the right midnight buffer.
    median_cadence : datetime.timedelta
        Median spacing between consecutive samples; zero for an empty or single-sample axis.
    max_gap : datetime.timedelta
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


def measure_time_axis_coverage(times: np.ndarray, *, day: date) -> TimeAxisCoverage:
    """Measure how much of a day and its buffers a time axis occupies.

    Each sample occupies the spacing to the next one. A spacing larger than ``DEFAULT_GAP_FACTOR``
    times the median spacing, and larger than ``DEFAULT_SEAM_TOLERANCE``, is a gap: the sample
    before it, like the last sample, occupies one median spacing.

    Parameters
    ----------
    times : numpy.ndarray
        Datetime64 values from one data-time axis, in any order.
    day : datetime.date
        Applicable UTC calendar day.

    Returns
    -------
    TimeAxisCoverage
        Fractions for the day core and both buffers, plus the axis's median cadence and largest
        gap. An empty axis returns all zeros, and a single sample occupies no time.
    """
    t = np.sort(np.asarray(times, dtype="datetime64[us]").ravel())
    spacing = np.diff(t)
    if spacing.size == 0:
        return TimeAxisCoverage(0.0, 0.0, 0.0, timedelta(0), timedelta(0), int(t.size))

    cadence = np.median(spacing)
    threshold = max(cadence * DEFAULT_GAP_FACTOR, np.timedelta64(DEFAULT_SEAM_TOLERANCE, "us"))
    ends = t + np.append(np.where(spacing <= threshold, spacing, cadence), cadence)

    def occupied_fraction(start: datetime, end: datetime) -> float:
        # The axis is datetime64, which carries no timezone; the bounds are UTC by construction.
        s = np.datetime64(start.replace(tzinfo=None), "us")
        e = np.datetime64(end.replace(tzinfo=None), "us")
        return float((np.clip(ends, s, e) - np.clip(t, s, e)).sum() / (e - s))

    bounds = day_bounds(day)
    return TimeAxisCoverage(
        day_frac=occupied_fraction(bounds.day_start, bounds.day_end),
        left_frac=occupied_fraction(bounds.window_start, bounds.day_start),
        right_frac=occupied_fraction(bounds.day_end, bounds.window_end),
        median_cadence=cadence.astype("timedelta64[us]").item(),
        max_gap=spacing.max().astype("timedelta64[us]").item(),
        n_times=int(t.size),
    )
