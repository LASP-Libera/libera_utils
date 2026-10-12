"""Unit tests for L1A day-coverage completeness gates."""

from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_coverage import (
    DEFAULT_SEAM_TOLERANCE,
    CoverageMode,
    DayCoveragePolicy,
    coverage_policy_for_apid,
    evaluate_day_coverage,
    measure_time_axis_coverage,
    overlaps_day_window,
)

# Stands in for the continuous APIDs in the tests below that are about the coverage arithmetic
# rather than about any one APID's thresholds.
CONTINUOUS_POLICY = DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.9, 0.99)
EVENT_DRIVEN_POLICY = DayCoveragePolicy(CoverageMode.EVENT_DRIVEN)


def test_full_day_and_buffers_complete():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 16, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY)
    assert result.is_complete
    assert result.left_frac == 1.0
    assert result.day_frac == 1.0
    assert result.right_frac == 1.0


def test_missing_right_buffer_incomplete():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 16, 0, 0, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY)
    assert result.left_ok
    assert result.day_ok
    assert not result.right_ok
    assert not result.is_complete


def test_half_day_fails_continuous_policy():
    day = date(2028, 2, 15)
    # Only 12 hours of the day core
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 0, 0, tzinfo=UTC), datetime(2028, 2, 15, 12, 0, tzinfo=UTC)),
        (datetime(2028, 2, 16, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY)
    assert result.day_frac == 0.5
    assert not result.day_ok
    assert not result.is_complete


def test_short_span_passes_event_driven_but_not_continuous():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 12, 0, tzinfo=UTC), datetime(2028, 2, 15, 12, 5, tzinfo=UTC)),
        (datetime(2028, 2, 16, 0, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    continuous = evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY)
    assert not continuous.day_ok
    event_driven = evaluate_day_coverage(intervals, day=day, policy=EVENT_DRIVEN_POLICY)
    assert event_driven.day_ok
    assert event_driven.is_complete


def test_overlapping_intervals_merged():
    day = date(2028, 2, 15)
    intervals = [
        (datetime(2028, 2, 15, 0, 0, tzinfo=UTC), datetime(2028, 2, 15, 14, 0, tzinfo=UTC)),
        (datetime(2028, 2, 15, 12, 0, tzinfo=UTC), datetime(2028, 2, 16, 0, 0, tzinfo=UTC)),
        (datetime(2028, 2, 14, 23, 50, tzinfo=UTC), datetime(2028, 2, 15, 0, 5, tzinfo=UTC)),
        (datetime(2028, 2, 15, 23, 55, tzinfo=UTC), datetime(2028, 2, 16, 0, 10, tzinfo=UTC)),
    ]
    result = evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY)
    assert result.day_frac == 1.0
    assert result.is_complete


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

    Demanding ``buffer_coverage_frac=1.0`` isolates the tolerance: every midnight crossing in
    DITL2 puts a seam inside the right buffer.
    """
    day = date(2026, 7, 12)
    result = evaluate_day_coverage(
        _day_spanning_intervals(day, timedelta(milliseconds=seam_ms)),
        day=day,
        policy=DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.9, 1.0),
    )
    assert result.is_complete
    assert result.left_frac == 1.0
    assert result.right_frac == 1.0


def test_real_gap_in_right_buffer_fails():
    day = date(2026, 7, 12)
    result = evaluate_day_coverage(
        _day_spanning_intervals(day, timedelta(seconds=30)), day=day, policy=CONTINUOUS_POLICY
    )
    assert result.left_ok
    assert not result.right_ok
    assert not result.is_complete


def test_gap_just_over_seam_tolerance_is_not_merged():
    day = date(2026, 7, 12)
    over = DEFAULT_SEAM_TOLERANCE + timedelta(milliseconds=1)
    strict_buffer = DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.9, 1.0)
    result = evaluate_day_coverage(_day_spanning_intervals(day, over), day=day, policy=strict_buffer)
    assert not result.right_ok
    under = DEFAULT_SEAM_TOLERANCE - timedelta(milliseconds=1)
    merged = evaluate_day_coverage(_day_spanning_intervals(day, under), day=day, policy=strict_buffer)
    assert merged.is_complete


def test_seam_tolerance_is_tunable_per_caller():
    day = date(2026, 7, 12)
    intervals = _day_spanning_intervals(day, timedelta(milliseconds=15))
    strict = evaluate_day_coverage(
        intervals,
        day=day,
        policy=DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.9, 1.0),
        seam_tolerance=timedelta(0),
    )
    assert not strict.is_complete


def test_buffer_frac_absorbs_a_small_real_dropout():
    """0.99 of a 600 s buffer allows a 6 s outage, which the tolerance alone would not."""
    day = date(2026, 7, 12)
    six_seconds = evaluate_day_coverage(
        _day_spanning_intervals(day, timedelta(seconds=6)), day=day, policy=CONTINUOUS_POLICY
    )
    assert six_seconds.is_complete
    seven_seconds = evaluate_day_coverage(
        _day_spanning_intervals(day, timedelta(seconds=7)), day=day, policy=CONTINUOUS_POLICY
    )
    assert not seven_seconds.right_ok


# The 14 APID-1036 File Metadata spans for 2026-07-12 from the DITL2 campaign, as read from the
# demuxed ground captures.
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
    result = evaluate_day_coverage(intervals, day=date(2026, 7, 12), policy=CONTINUOUS_POLICY)
    assert result.left_frac == 1.0
    assert result.day_frac == 1.0
    assert result.right_frac == 1.0
    assert result.is_complete


def test_event_driven_passes_buffers_it_does_not_cover():
    """Buffers with no data at all do not fail an event-driven APID."""
    day = date(2026, 7, 12)
    intervals = [(datetime(2026, 7, 12, 14, 0), datetime(2026, 7, 12, 14, 3))]
    result = evaluate_day_coverage(intervals, day=day, policy=EVENT_DRIVEN_POLICY)
    assert result.left_frac == 0.0
    assert result.right_frac == 0.0
    assert result.is_complete
    assert result.mode is CoverageMode.EVENT_DRIVEN


def test_event_driven_with_no_overlapping_data_is_incomplete():
    day = date(2026, 7, 12)
    intervals = [(datetime(2026, 7, 10, 14, 0), datetime(2026, 7, 10, 14, 3))]
    assert not evaluate_day_coverage(intervals, day=day, policy=EVENT_DRIVEN_POLICY).is_complete


def test_event_driven_data_only_in_a_buffer_still_passes():
    """Data inside the buffer belongs to this day's combine window, so it must trigger one."""
    day = date(2026, 7, 12)
    intervals = [(datetime(2026, 7, 11, 23, 55), datetime(2026, 7, 11, 23, 58))]
    result = evaluate_day_coverage(intervals, day=day, policy=EVENT_DRIVEN_POLICY)
    assert result.day_frac == 0.0
    assert result.is_complete


def test_continuous_just_under_its_threshold_fails():
    day = date(2026, 7, 12)
    start = datetime(2026, 7, 12, 0, 0)
    # 0.985 of the core, above the 0.90 policy and below the 0.99 one.
    intervals = [
        (start - timedelta(minutes=10), start),
        (start, start + timedelta(seconds=round(86400 * 0.985))),
        (start + timedelta(days=1), start + timedelta(days=1, minutes=10)),
    ]
    strict = evaluate_day_coverage(intervals, day=day, policy=DayCoveragePolicy(CoverageMode.CONTINUOUS, 0.99, 0.99))
    assert not strict.day_ok
    assert not strict.is_complete
    assert evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY).is_complete


def test_unusable_intervals_are_ignored():
    day = date(2028, 2, 15)
    full = (datetime(2028, 2, 14, 23, 50), datetime(2028, 2, 16, 0, 10))
    baseline = evaluate_day_coverage([full], day=day, policy=CONTINUOUS_POLICY)
    noisy = evaluate_day_coverage(
        [
            full,
            (None, datetime(2028, 2, 15, 12)),
            (datetime(2028, 2, 15, 12), None),
            (datetime(2028, 2, 15, 13), datetime(2028, 2, 15, 12)),
        ],
        day=day,
        policy=CONTINUOUS_POLICY,
    )
    assert noisy == baseline
    assert not evaluate_day_coverage(
        [(datetime(2028, 2, 15, 13), datetime(2028, 2, 15, 12))], day=day, policy=EVENT_DRIVEN_POLICY
    ).is_complete


def test_non_utc_aware_intervals_are_converted_to_utc():
    """19:00-05:00 on the 14th is 00:00 UTC on the 15th, so the span covers the whole core."""
    day = date(2028, 2, 15)
    minus_five = timezone(timedelta(hours=-5))
    intervals = [(datetime(2028, 2, 14, 19, 0, tzinfo=minus_five), datetime(2028, 2, 15, 19, 0, tzinfo=minus_five))]
    result = evaluate_day_coverage(intervals, day=day, policy=CONTINUOUS_POLICY)
    assert result.day_frac == 1.0
    assert result.left_frac == 0.0
    assert result.right_frac == 0.0


def test_zero_length_span_is_present_for_event_driven():
    day = date(2028, 2, 15)
    instant = datetime(2028, 2, 15, 12, 0)
    result = evaluate_day_coverage([(instant, instant)], day=day, policy=EVENT_DRIVEN_POLICY)
    assert result.is_complete
    assert result.day_frac == 0.0


def test_zero_length_span_leaves_continuous_fractions_unchanged():
    day = date(2028, 2, 15)
    half_day = (datetime(2028, 2, 15, 0, 0), datetime(2028, 2, 15, 12, 0))
    instant = datetime(2028, 2, 15, 18, 0)
    without = evaluate_day_coverage([half_day], day=day, policy=CONTINUOUS_POLICY)
    with_instant = evaluate_day_coverage([half_day, (instant, instant)], day=day, policy=CONTINUOUS_POLICY)
    assert with_instant == without


@pytest.mark.parametrize(
    "instant",
    [datetime(2028, 2, 14, 23, 50), datetime(2028, 2, 16, 0, 10)],
    ids=["window_start", "window_end"],
)
def test_span_touching_only_a_window_edge_is_present(instant):
    day = date(2028, 2, 15)
    assert overlaps_day_window(instant, instant, day=day)
    assert evaluate_day_coverage([(instant, instant)], day=day, policy=EVENT_DRIVEN_POLICY).is_complete


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (datetime(2028, 2, 14, 23, 0), datetime(2028, 2, 14, 23, 49, 59, 999999), False),
        (datetime(2028, 2, 16, 0, 10, 0, 1), datetime(2028, 2, 16, 1, 0), False),
        (datetime(2028, 2, 14, 0, 0), datetime(2028, 2, 17, 0, 0), True),
        (datetime(2028, 2, 15, 0, 0, tzinfo=timezone(timedelta(hours=1))), datetime(2028, 2, 15, 0, 0), True),
        (
            datetime(2028, 2, 14, 18, 0, tzinfo=timezone(timedelta(hours=-5))),
            datetime(2028, 2, 14, 18, 49, tzinfo=timezone(timedelta(hours=-5))),
            False,
        ),
    ],
    ids=["before", "after", "enclosing", "aware_inside", "aware_before"],
)
def test_overlaps_day_window(start, end, expected):
    assert overlaps_day_window(start, end, day=date(2028, 2, 15)) is expected


@pytest.mark.parametrize(
    ("apid", "mode", "day_frac", "buffer_frac"),
    [
        (LiberaApid.icie_nom_hk, CoverageMode.CONTINUOUS, 0.99, 0.99),
        (LiberaApid.icie_rad_sample, CoverageMode.CONTINUOUS, 0.90, 0.95),
        (LiberaApid.icie_wfov_sci, CoverageMode.CONTINUOUS, 0.60, 0.60),
        (LiberaApid.icie_rad_full, CoverageMode.EVENT_DRIVEN, None, None),
        (LiberaApid.icie_cal_sample, CoverageMode.EVENT_DRIVEN, None, None),
    ],
)
def test_registered_policies(apid, mode, day_frac, buffer_frac):
    policy = coverage_policy_for_apid(apid)
    assert policy.mode is mode
    assert policy.day_coverage_frac == day_frac
    assert policy.buffer_coverage_frac == buffer_frac


def test_policy_lookup_accepts_int_and_rejects_unknown():
    assert coverage_policy_for_apid(1057) is coverage_policy_for_apid(LiberaApid.icie_nom_hk)
    with pytest.raises(KeyError):
        coverage_policy_for_apid(4321)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": CoverageMode.CONTINUOUS}, "requires day_coverage_frac, buffer_coverage_frac"),
        ({"mode": CoverageMode.CONTINUOUS, "day_coverage_frac": 0.9}, "requires buffer_coverage_frac"),
        (
            {"mode": CoverageMode.CONTINUOUS, "day_coverage_frac": 1.5, "buffer_coverage_frac": 0.9},
            r"day_coverage_frac must be in \(0, 1\], got 1.5",
        ),
        (
            {"mode": CoverageMode.CONTINUOUS, "day_coverage_frac": 0.0, "buffer_coverage_frac": 0.9},
            r"day_coverage_frac must be in \(0, 1\], got 0.0",
        ),
        ({"mode": CoverageMode.EVENT_DRIVEN, "day_coverage_frac": 0.9}, "drop day_coverage_frac"),
    ],
)
def test_malformed_policies_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DayCoveragePolicy(**kwargs)


def _axis(start: str, stop: str, step_s: int) -> np.ndarray:
    return np.arange(
        np.datetime64(start, "us"),
        np.datetime64(stop, "us"),
        np.timedelta64(step_s * 1_000_000, "us"),
    )


def test_measure_full_day_axis():
    coverage = measure_time_axis_coverage(_axis("2026-07-12T00:00:00", "2026-07-13T00:00:00", 5), day=date(2026, 7, 12))
    assert coverage.day_frac == 1.0
    assert coverage.median_cadence == timedelta(seconds=5)
    assert coverage.max_gap == timedelta(seconds=5)
    assert coverage.n_times == 17280


def test_measure_sees_a_gap_inside_the_axis():
    """The whole point of measuring the axis: a gap a file-span gate cannot see."""
    axis = _axis("2026-07-12T00:00:00", "2026-07-13T00:00:00", 5)
    without_two_hours = np.concatenate([axis[:1440], axis[2880:]])
    coverage = measure_time_axis_coverage(without_two_hours, day=date(2026, 7, 12))
    assert coverage.day_frac == pytest.approx(1 - 2 / 24, abs=1e-4)
    # The spacing across the hole is the two missing hours plus the one cadence that bridges it.
    assert coverage.max_gap == timedelta(hours=2, seconds=5)
    # The same data as whole-file spans reads as a covered day, because the gap is inside a file.
    spans = [(axis[0].astype(datetime), axis[-1].astype(datetime))]
    assert evaluate_day_coverage(spans, day=date(2026, 7, 12), policy=CONTINUOUS_POLICY).day_frac == pytest.approx(
        1.0, abs=1e-4
    )


def test_measure_wfov_cadence_is_not_read_as_gaps():
    """A 5 s camera cadence is the signal, not a dropout; only the outlier gap counts."""
    axis = _axis("2026-07-12T01:43:30", "2026-07-12T03:05:19", 5)
    with_outage = np.concatenate([axis[:300], axis[300:] + np.timedelta64(120, "s")])
    coverage = measure_time_axis_coverage(with_outage, day=date(2026, 7, 12))
    assert coverage.median_cadence == timedelta(seconds=5)
    assert coverage.max_gap == timedelta(seconds=125)
    # Only the single outage is lost, not every inter-sample space.
    occupied = timedelta(seconds=5) * coverage.n_times
    assert coverage.day_frac == pytest.approx(occupied.total_seconds() / 86400, abs=1e-4)


def test_measure_empty_axis():
    coverage = measure_time_axis_coverage(np.array([], dtype="datetime64[us]"), day=date(2026, 7, 12))
    assert coverage.day_frac == 0.0
    assert coverage.n_times == 0


def test_measure_single_sample_occupies_one_seam_tolerance():
    coverage = measure_time_axis_coverage(
        np.array(["2026-07-12T12:00:00"], dtype="datetime64[us]"), day=date(2026, 7, 12)
    )
    assert coverage.n_times == 1
    assert coverage.median_cadence == DEFAULT_SEAM_TOLERANCE
    assert coverage.max_gap == timedelta(0)
    assert coverage.day_frac == pytest.approx(DEFAULT_SEAM_TOLERANCE.total_seconds() / 86400)


def test_measure_ignores_nat_and_order():
    axis = _axis("2026-07-12T00:00:00", "2026-07-13T00:00:00", 5)
    shuffled = np.random.default_rng(0).permutation(
        np.concatenate([axis, np.array(["NaT"] * 3, dtype="datetime64[us]")])
    )
    assert measure_time_axis_coverage(shuffled, day=date(2026, 7, 12)) == measure_time_axis_coverage(
        axis, day=date(2026, 7, 12)
    )


def test_measure_buffer_fractions():
    """An axis from 23:55 on D-1 to 00:05 on D+1 fills half of each 10-minute buffer."""
    coverage = measure_time_axis_coverage(_axis("2026-07-11T23:55:00", "2026-07-13T00:05:00", 5), day=date(2026, 7, 12))
    assert coverage.left_frac == pytest.approx(0.5)
    assert coverage.right_frac == pytest.approx(0.5)
    assert coverage.day_frac == 1.0
