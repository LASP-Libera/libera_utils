"""Unit tests for acquisition ordering and its sequence-counter cross-check."""

from datetime import timedelta
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.packet_ordering import (
    DEFAULT_PROBE_PACKETS,
    SEQUENCE_COUNTER_MODULUS,
    check_packet_acquisition_order,
    find_segment_breaks,
    order_packet_files,
    summarize_packet_order,
    unwrap_sequence_counter,
)

_START = np.datetime64("2026-07-13T08:00:00", "us")


def _times(offsets_us, start=_START):
    return start + np.asarray(offsets_us, dtype=np.int64).astype("timedelta64[us]")


def _cadence_times(n, period_us=250_000):
    return _times(np.arange(n) * period_us)


class TestUnwrapSequenceCounter:
    def test_clean_run(self):
        unwrapped, segment = unwrap_sequence_counter(np.arange(100, 110))
        assert list(unwrapped) == list(range(10))
        assert set(segment.tolist()) == {0}

    def test_rollover_costs_one_not_a_modulus(self):
        """16383 -> 0 is a nominal step, so the unwrapped count advances by 1."""
        unwrapped, _ = unwrap_sequence_counter(np.array([16382, 16383, 0, 1]))
        assert list(np.diff(unwrapped)) == [1, 1, 1]

    def test_two_wraps_in_one_segment(self):
        counter = np.arange(0, 2 * SEQUENCE_COUNTER_MODULUS + 5) % SEQUENCE_COUNTER_MODULUS
        unwrapped, _ = unwrap_sequence_counter(counter)
        assert list(unwrapped[:3]) == [0, 1, 2]
        assert unwrapped[-1] == counter.size - 1
        assert np.all(np.diff(unwrapped) == 1)

    def test_packets_lost_across_the_rollover_are_a_gap_not_a_decrease(self):
        """16382 -> 1 is one missing packet, not a decrease of 16381."""
        unwrapped, _ = unwrap_sequence_counter(np.array([16381, 16382, 1, 2]))
        assert list(np.diff(unwrapped)) == [1, 3, 1]

    def test_is_monotonic_by_construction(self):
        """Unwrapping along the input axis cannot reorder it: argsort is always the identity."""
        rng = np.random.default_rng(0)
        counter = rng.integers(0, SEQUENCE_COUNTER_MODULUS, size=500)
        unwrapped, _ = unwrap_sequence_counter(counter)
        assert np.all(np.diff(unwrapped) >= 0)
        assert np.array_equal(np.argsort(unwrapped, kind="stable"), np.arange(counter.size))

    def test_segments_restart_the_count(self):
        breaks = np.array([True, False, False, True, False])
        unwrapped, segment = unwrap_sequence_counter(np.array([10, 11, 12, 900, 901]), segment_break=breaks)
        assert list(unwrapped) == [0, 1, 2, 0, 1]
        assert list(segment) == [0, 0, 0, 1, 1]

    def test_empty(self):
        unwrapped, segment = unwrap_sequence_counter(np.array([], dtype=np.int64))
        assert unwrapped.size == 0
        assert segment.size == 0


class TestFindSegmentBreaks:
    def test_backward_jitter_does_not_break_a_segment(self):
        """A -1.8 s step is the FSW defect, and is exactly what must stay inside one segment."""
        times = _times([0, 250_000, -1_550_000, 500_000, 750_000])
        breaks = find_segment_breaks(times)
        assert list(breaks) == [True, False, False, False, False]

    def test_large_backward_step_breaks(self):
        times = _times([0, 250_000, -10_000_000, 500_000])
        breaks = find_segment_breaks(times)
        assert breaks[2]

    def test_long_outage_breaks(self):
        times = _times([0, 250_000, 120_000_000])
        breaks = find_segment_breaks(times)
        assert breaks[2]

    def test_unusable_times_break_on_both_sides(self):
        times = _cadence_times(5)
        valid = np.array([True, True, False, True, True])
        breaks = find_segment_breaks(times, valid_time_mask=valid)
        assert breaks[2]
        assert breaks[3]

    def test_empty_and_single(self):
        assert find_segment_breaks(_times([])).size == 0
        assert list(find_segment_breaks(_times([0]))) == [True]


class TestSummarizePacketOrder:
    def test_clean_axis_is_nominal(self):
        diagnostics = summarize_packet_order(_cadence_times(10), np.arange(100, 110))
        assert diagnostics.is_nominal
        assert diagnostics.n_time_inversions == 0
        assert diagnostics.n_packets_displaced == 0
        assert diagnostics.n_segments == 1
        assert diagnostics.sequence_counter_available

    @pytest.mark.parametrize("gap", [2, 8, 21])
    def test_measured_gaps_count_as_missing_not_out_of_order(self, gap):
        counter = np.array([100, 101, 101 + gap, 102 + gap])
        diagnostics = summarize_packet_order(_cadence_times(4), counter)
        assert diagnostics.n_sequence_gaps == 1
        assert diagnostics.n_missing_packets == gap - 1
        assert diagnostics.n_order_violations == 0

    def test_repeated_counter_counted(self):
        diagnostics = summarize_packet_order(_cadence_times(4), np.array([100, 101, 101, 102]))
        assert diagnostics.n_repeated_counters == 1
        assert diagnostics.n_missing_packets == 0
        assert not diagnostics.is_nominal

    def test_huge_step_is_an_order_violation(self):
        diagnostics = summarize_packet_order(_cadence_times(3), np.array([100, 9000, 9001]))
        assert diagnostics.n_order_violations == 1
        assert diagnostics.n_sequence_gaps == 0

    def test_time_inversion_is_counted_and_displacement_measured(self):
        """The -1.8 s inversion measured in DITL2: counted, and the sort it avoided quantified."""
        times = _times([0, 250_000, -1_550_000, 500_000, 750_000])
        diagnostics = summarize_packet_order(times, np.arange(100, 105))
        assert diagnostics.n_time_inversions == 1
        assert diagnostics.max_time_inversion_us == 1_800_000
        # A stable time sort would have pulled row 2 to the front, moving three rows.
        assert diagnostics.n_packets_displaced == 3
        assert diagnostics.n_order_violations == 0
        assert diagnostics.n_missing_packets == 0

    def test_instrument_reset_isolated_by_a_segment_break(self):
        """NOM-HK's real reset: SSC 5584 -> 1 with a pre-sync 1958 timestamp."""
        times = np.array(
            ["2026-07-13T08:00:00", "2026-07-13T08:00:01", "1958-01-01T00:00:02", "1958-01-01T00:00:03"],
            dtype="datetime64[us]",
        )
        diagnostics = summarize_packet_order(times, np.array([5583, 5584, 1, 2]))
        assert diagnostics.n_segments == 2
        # The reset step itself is across the break, so it is not scored as a violation.
        assert diagnostics.n_order_violations == 0

    def test_missing_sequence_counter_still_reports_time_facts(self):
        times = _times([0, 250_000, -1_550_000, 500_000])
        diagnostics = summarize_packet_order(times, None)
        assert not diagnostics.sequence_counter_available
        assert diagnostics.n_time_inversions == 1
        assert diagnostics.n_missing_packets == 0

    def test_uncorroborated_gap_flagged_when_cadence_known(self):
        """A counter gap of 8 at 1 Hz must show 8 s of elapsed time, or it is worth surfacing."""
        corroborated = summarize_packet_order(
            _times([0, 1_000_000, 9_000_000, 10_000_000]),
            np.array([100, 101, 109, 110]),
            nominal_cadence=timedelta(seconds=1),
        )
        assert corroborated.n_sequence_gaps == 1
        assert corroborated.n_uncorroborated_gaps == 0

        uncorroborated = summarize_packet_order(
            _times([0, 1_000_000, 2_000_000, 3_000_000]),
            np.array([100, 101, 109, 110]),
            nominal_cadence=timedelta(seconds=1),
        )
        assert uncorroborated.n_uncorroborated_gaps == 1

    def test_empty_axis(self):
        diagnostics = summarize_packet_order(_times([]), np.array([], dtype=np.int64))
        assert diagnostics.n_packets == 0
        assert diagnostics.is_nominal


def _packet_dataset(times, counter=None):
    data = {}
    if counter is not None:
        data["SRC_SEQ_CTR"] = ("PACKET", np.asarray(counter, dtype=np.uint16))
    return xr.Dataset(data, coords={"PACKET_ICIE_TIME": ("PACKET", times)})


class TestCheckPacketAcquisitionOrder:
    def test_dataset_is_returned_unchanged(self):
        times = _times([0, 250_000, -1_550_000, 500_000])
        dataset = _packet_dataset(times, np.arange(100, 104))
        with pytest.warns(UserWarning, match="steps backward"):
            out, diagnostics = check_packet_acquisition_order(dataset, "PACKET_ICIE_TIME")
        assert np.array_equal(out["PACKET_ICIE_TIME"].values, times)
        assert diagnostics.n_time_inversions == 1

    def test_missing_sequence_counter_warns_and_falls_back_to_input_order(self, caplog):
        times = _cadence_times(4)
        dataset = _packet_dataset(times)
        with caplog.at_level("WARNING"):
            out, diagnostics = check_packet_acquisition_order(dataset, "PACKET_ICIE_TIME")
        assert np.array_equal(out["PACKET_ICIE_TIME"].values, times)
        assert not diagnostics.sequence_counter_available
        assert "no SRC_SEQ_CTR variable" in caplog.text

    def test_clean_axis_emits_no_warning(self, recwarn):
        dataset = _packet_dataset(_cadence_times(10), np.arange(100, 110))
        _, diagnostics = check_packet_acquisition_order(dataset, "PACKET_ICIE_TIME")
        assert diagnostics.is_nominal
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]

    def test_summary_is_one_line_regardless_of_event_count(self, caplog):
        """~390 inversions per granule is the measured rate, so output must not scale with it."""
        rng = np.random.default_rng(1)
        n = 2000
        offsets = np.arange(n) * 250_000
        inverted = rng.choice(np.arange(1, n), size=200, replace=False)
        offsets[inverted] -= 1_800_000
        dataset = _packet_dataset(_times(offsets), np.arange(n) % SEQUENCE_COUNTER_MODULUS)
        with caplog.at_level("WARNING"), pytest.warns(UserWarning, match="steps backward"):
            _, diagnostics = check_packet_acquisition_order(dataset, "PACKET_ICIE_TIME")
        assert diagnostics.n_time_inversions > 100
        assert len(caplog.records) == 1

    def test_verbose_adds_detail(self, caplog):
        times = _times([0, 250_000, -1_550_000, 500_000])
        dataset = _packet_dataset(times, np.arange(100, 104))
        with caplog.at_level("WARNING"), pytest.warns(UserWarning, match="steps backward"):
            check_packet_acquisition_order(dataset, "PACKET_ICIE_TIME", verbose=True)
        assert len(caplog.records) == 2

    def test_sequence_counter_off_the_packet_dimension_is_ignored(self, caplog):
        dataset = _packet_dataset(_cadence_times(3), np.arange(100, 103))
        dataset = dataset.assign(SRC_SEQ_CTR=(("OTHER",), np.array([1, 2], dtype=np.uint16)))
        with caplog.at_level("WARNING"):
            _, diagnostics = check_packet_acquisition_order(dataset, "PACKET_ICIE_TIME")
        assert not diagnostics.sequence_counter_available
        assert "not on the PACKET dimension" in caplog.text


class TestOrderPacketFiles:
    @staticmethod
    def _definition():
        return mock.MagicMock(name="xtce_definition")

    def test_single_file_does_no_io(self):
        with mock.patch("libera_utils.l1a.packet_ordering._probe_packet_time_us") as probe:
            assert order_packet_files(["a.bin"], self._definition(), 1036, multipart_kwargs={}) == ["a.bin"]
            assert order_packet_files([], self._definition(), 1036, multipart_kwargs={}) == []
        probe.assert_not_called()

    def test_reverse_shuffled_input_comes_back_in_acquisition_order(self):
        files = ["c.bin", "a.bin", "b.bin"]
        probes = {"a.bin": 100.0, "b.bin": 200.0, "c.bin": 300.0}
        with mock.patch(
            "libera_utils.l1a.packet_ordering._probe_packet_time_us",
            side_effect=lambda f, *a, **k: probes[f],
        ):
            assert order_packet_files(files, self._definition(), 1036, multipart_kwargs={}) == [
                "a.bin",
                "b.bin",
                "c.bin",
            ]

    def test_unprobeable_file_is_kept_at_the_end_in_caller_order(self):
        files = ["good_b.bin", "bad.bin", "good_a.bin"]
        probes = {"good_a.bin": 100.0, "good_b.bin": 200.0, "bad.bin": None}
        with mock.patch(
            "libera_utils.l1a.packet_ordering._probe_packet_time_us",
            side_effect=lambda f, *a, **k: probes[f],
        ):
            ordered = order_packet_files(files, self._definition(), 1036, multipart_kwargs={})
        assert ordered == ["good_a.bin", "good_b.bin", "bad.bin"]

    def test_probe_count_default_is_passed_through(self):
        with mock.patch("libera_utils.l1a.packet_ordering._probe_packet_time_us", return_value=1.0) as probe:
            order_packet_files(["a.bin", "b.bin"], self._definition(), 1036, multipart_kwargs={})
        assert probe.call_args.kwargs["n_probe"] == DEFAULT_PROBE_PACKETS
