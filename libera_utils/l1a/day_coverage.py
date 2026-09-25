"""Day-coverage evaluation for L1A combine completeness gates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

import numpy as np

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_window import DEFAULT_DAY_BUFFER

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


class CoverageMode(StrEnum):
    """How a day's completeness is judged for one APID."""

    CONTINUOUS = "continuous"
    EVENT_DRIVEN = "event_driven"


@dataclass(frozen=True, slots=True)
class DayCoveragePolicy:
    """Completeness rule for one APID.

    Parameters
    ----------
    mode : CoverageMode
        ``CONTINUOUS`` gates on the fractions below. ``EVENT_DRIVEN`` passes every gate as soon
        as any data overlaps the buffered day window, and takes no fractions.
    day_coverage_frac : float | None
        Minimum covered fraction of ``[D, D+1)``. Required for ``CONTINUOUS``.
    buffer_coverage_frac : float | None
        Minimum covered fraction of each midnight buffer. Required for ``CONTINUOUS``.

    Raises
    ------
    ValueError
        If a ``CONTINUOUS`` policy omits either fraction, an ``EVENT_DRIVEN`` policy carries
        one, or a fraction falls outside ``(0, 1]``.
    """

    mode: CoverageMode
    day_coverage_frac: float | None = None
    buffer_coverage_frac: float | None = None

    def __post_init__(self) -> None:
        fractions = {
            "day_coverage_frac": self.day_coverage_frac,
            "buffer_coverage_frac": self.buffer_coverage_frac,
        }
        if self.mode is CoverageMode.CONTINUOUS:
            missing = [name for name, value in fractions.items() if value is None]
            if missing:
                raise ValueError(f"A {self.mode} policy requires {', '.join(missing)}.")
            for name, value in fractions.items():
                if not 0.0 < value <= 1.0:
                    raise ValueError(f"{name} must be in (0, 1], got {value}.")
        else:
            supplied = [name for name, value in fractions.items() if value is not None]
            if supplied:
                raise ValueError(f"A {self.mode} policy gates on presence alone; drop {', '.join(supplied)}.")


# Every LiberaApid, explicitly. An APID added to the enum without a row here fails the
# completeness check below at import rather than at combine time inside the Lambda.
APID_COVERAGE_POLICIES: dict[LiberaApid, DayCoveragePolicy] = {
    LiberaApid.jpss_sc_pos: DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.99, 0.99),
    LiberaApid.pev_sw_stat: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.pec_sw_stat: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_sw_stat: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_seq_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_fp_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_log_msg: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_rad_full: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_rad_sample: DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.90, 0.95),
    LiberaApid.icie_axis_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_wfov_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_wfov_sci: DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.60, 0.60),
    LiberaApid.icie_cal_full: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_cal_sample: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_axis_sample: DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.90, 0.95),
    LiberaApid.icie_wfov_resp: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_crit_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_nom_hk: DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.99, 0.99),
    LiberaApid.icie_ana_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
    LiberaApid.icie_temp_hk: DayCoveragePolicy(CoverageMode.EVENT_DRIVEN),
}

_unlisted_apids = sorted(apid.name for apid in LiberaApid if apid not in APID_COVERAGE_POLICIES)
if _unlisted_apids:
    raise ValueError(f"APID_COVERAGE_POLICIES is missing a policy for: {', '.join(_unlisted_apids)}")


def coverage_policy_for_apid(apid: LiberaApid | int) -> DayCoveragePolicy:
    """Return the completeness policy for ``apid``.

    Parameters
    ----------
    apid : LiberaApid | int
        APID being combined.

    Returns
    -------
    DayCoveragePolicy
        The policy registered in :data:`APID_COVERAGE_POLICIES`.

    Raises
    ------
    KeyError
        If ``apid`` is not a ``LiberaApid``, or has no registered policy.
    """
    try:
        return APID_COVERAGE_POLICIES[LiberaApid(apid)]
    except ValueError as ve:
        raise KeyError(f"No coverage policy for APID {apid}: not a LiberaApid member.") from ve


@dataclass(frozen=True, slots=True)
class DayCoverageResult:
    """Coverage fractions and pass/fail for left buffer, day core, and right buffer."""

    mode: CoverageMode
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


def _merge_intervals(
    intervals: list[tuple[datetime, datetime]],
    tolerance: timedelta = DEFAULT_SEAM_TOLERANCE,
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
    tolerance: timedelta = DEFAULT_SEAM_TOLERANCE,
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
    policy: DayCoveragePolicy,
    buffer: timedelta = DEFAULT_DAY_BUFFER,
    seam_tolerance: timedelta = DEFAULT_SEAM_TOLERANCE,
) -> DayCoverageResult:
    """Evaluate L0 time-span coverage against day core and midnight buffers.

    Parameters
    ----------
    intervals : list of (start, end)
        Effective L0 time ranges (data times preferred). Endpoints may be aware or naive UTC.
    day : date
        Applicable UTC calendar day.
    policy : DayCoveragePolicy
        Completeness rule for the APID being combined, from :func:`coverage_policy_for_apid`.
    buffer : timedelta, optional
        Midnight buffer on each side (default 10 minutes).
    seam_tolerance : timedelta, optional
        Spacing between two spans that still counts as continuous. Default
        ``DEFAULT_SEAM_TOLERANCE`` (1 s). Pass ``timedelta(0)`` to require exact abutment.

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
    left_frac = _coverage_fraction(normalized, *left, tolerance=seam_tolerance)
    day_frac = _coverage_fraction(normalized, *core, tolerance=seam_tolerance)
    right_frac = _coverage_fraction(normalized, *right, tolerance=seam_tolerance)

    if policy.mode is CoverageMode.EVENT_DRIVEN:
        # Presence anywhere in the buffered window is the whole test. Gating the buffers on a
        # fraction as well would make an event-driven day unreachable: an APID that is only on
        # for an event almost never has data at both midnights.
        window_start, window_end = left[0], right[1]
        present = any(start < window_end and end > window_start for start, end in normalized)
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
        n_intervals=len(normalized),
    )


# A spacing this many times the axis's own median spacing is treated as a real gap rather than
# jitter. The axis calibrates the threshold itself, so one number covers a 1 Hz housekeeping APID
# and a 5 s camera cadence alike.
DEFAULT_GAP_FACTOR = 5.0


@dataclass(frozen=True, slots=True)
class TimeAxisCoverage:
    """What a granule's own time axis says about its occupancy of a day and its buffers."""

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
    buffer: timedelta = DEFAULT_DAY_BUFFER,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    seam_tolerance: timedelta = DEFAULT_SEAM_TOLERANCE,
) -> TimeAxisCoverage:
    """Measure how much of a day a granule's time axis actually occupies.

    This is a measurement, not a gate. :func:`evaluate_day_coverage` works from whole-file L0
    spans and so cannot see a gap *inside* a file; this works from the samples themselves.

    Samples count as continuous until their spacing exceeds ``gap_factor`` times the axis's
    median spacing (never less than ``seam_tolerance``), and the last sample of each run is
    credited with one median cadence of occupancy.

    Parameters
    ----------
    times : numpy.ndarray
        Datetime64 (or datetime-like) values from one science time axis. Need not be sorted.
    day : date
        Applicable UTC calendar day.
    buffer : timedelta, optional
        Midnight buffer on each side (default 10 minutes).
    gap_factor : float, optional
        Multiple of the median spacing above which a spacing is a gap. Default 5.
    seam_tolerance : timedelta, optional
        Floor on the gap threshold, so a near-zero median cadence cannot make every spacing a gap.

    Returns
    -------
    TimeAxisCoverage
        Fractions for the day core and both buffers, plus the axis's median cadence and largest
        gap. An empty axis returns all zeros.
    """
    values = np.asarray(times).astype("datetime64[us]").ravel()
    values = values[~np.isnat(values)]
    values.sort()
    if values.size == 0:
        return TimeAxisCoverage(0.0, 0.0, 0.0, timedelta(0), timedelta(0), 0)

    if values.size == 1:
        cadence_us = int(seam_tolerance.total_seconds() * 1e6)
        max_gap_us = 0
        run_bounds = [(values[0], values[0])]
    else:
        spacings_us = np.diff(values).astype("int64")
        cadence_us = int(np.median(spacings_us))
        max_gap_us = int(spacings_us.max())
        threshold_us = max(cadence_us * gap_factor, seam_tolerance.total_seconds() * 1e6)
        # Split at real gaps; each remaining run is one continuously occupied interval.
        break_positions = np.flatnonzero(spacings_us > threshold_us) + 1
        run_bounds = [(run[0], run[-1]) for run in np.split(values, break_positions) if run.size]

    occupancy = timedelta(microseconds=max(cadence_us, 1))
    intervals = [(first.astype(datetime), last.astype(datetime) + occupancy) for first, last in run_bounds]

    left, core, right = day_core_and_buffer_bounds(day, buffer)
    return TimeAxisCoverage(
        day_frac=_coverage_fraction(intervals, *core, tolerance=timedelta(0)),
        left_frac=_coverage_fraction(intervals, *left, tolerance=timedelta(0)),
        right_frac=_coverage_fraction(intervals, *right, tolerance=timedelta(0)),
        median_cadence=timedelta(microseconds=cadence_us),
        max_gap=timedelta(microseconds=max_gap_us),
        n_times=int(values.size),
    )
