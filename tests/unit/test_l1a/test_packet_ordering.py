"""Unit tests for ordering the PACKET axis by SRC_SEQ_CTR and checking it against packet time."""

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.packet_ordering import (
    SEQUENCE_COUNTER_MODULUS,
    _unwrap_sequence_counter,
    check_packet_order,
    out_of_order_positions,
    sort_by_sequence_counter,
)

_START = np.datetime64("2026-07-13T08:00:00", "us")
_LOGGER = "libera_utils.l1a.packet_ordering"


def _packets(counters, offsets_s=None) -> xr.Dataset:
    if offsets_s is None:
        offsets_s = np.arange(len(counters))
    times = _START + (np.asarray(offsets_s, dtype=float) * 1e6).astype("timedelta64[us]")
    return xr.Dataset(
        {"SRC_SEQ_CTR": ("PACKET", np.asarray(counters, dtype=np.uint16))},
        coords={"PACKET_ICIE_TIME": ("PACKET", times)},
    )


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER and r.levelname == "WARNING"]


@pytest.fixture
def order_log(caplog):
    caplog.set_level("WARNING", logger=_LOGGER)
    return caplog


class TestCheckPacketOrder:
    def test_clean_axis_warns_nothing(self, order_log):
        check_packet_order(_packets([10, 11, 12, 13]), "PACKET_ICIE_TIME")
        assert _warnings(order_log) == []

    def test_backward_packet_time_with_advancing_counter_warns_with_detail(self, order_log):
        ds = _packets([10, 11, 12, 13], offsets_s=[0, 1, -1, 3])
        with pytest.warns(UserWarning, match="PACKET_ICIE_TIME steps backward at 1 position") as caught:
            check_packet_order(ds, "PACKET_ICIE_TIME")
        assert len(caught) == 1
        (message,) = _warnings(order_log)
        assert "by up to 2000000 microseconds" in message
        assert "first at position 2: SRC_SEQ_CTR 11 -> 12" in message
        assert "2026-07-13T08:00:01.000000 -> 2026-07-13T07:59:59.000000" in message
        assert "kept in SRC_SEQ_CTR order" in message

    def test_backward_packet_time_reports_the_largest_step(self, order_log):
        ds = _packets([10, 11, 12, 13, 14], offsets_s=[0, 1, 0.5, 3, -2])
        with pytest.warns(UserWarning, match="steps backward at 2 position") as caught:
            check_packet_order(ds, "PACKET_ICIE_TIME")
        assert "by up to 5000000 microseconds" in str(caught[0].message)
        assert "first at position 2: SRC_SEQ_CTR 11 -> 12" in str(caught[0].message)

    @pytest.mark.parametrize("missing", ["SRC_SEQ_CTR", "PACKET_ICIE_TIME"])
    def test_missing_variable_raises(self, missing):
        with pytest.raises(ValueError, match=f"no '{missing}' variable"):
            check_packet_order(_packets([10, 11]).drop_vars(missing), "PACKET_ICIE_TIME")


class TestOutOfOrderPositions:
    @pytest.mark.parametrize("counters", [[16382, 16383, 0, 1], [16379, 16380, 2, 3], [10, 11, 15]])
    def test_forward_steps_and_wraps_are_in_order(self, counters):
        assert out_of_order_positions(_packets(counters), "PACKET_ICIE_TIME").tolist() == []

    @pytest.mark.parametrize("counters", [[10, 11, 10, 12], [10, 11, 11, 12]])
    def test_backward_or_repeated_counter_within_the_gap_is_out_of_order(self, counters):
        assert out_of_order_positions(_packets(counters), "PACKET_ICIE_TIME").tolist() == [1]

    def test_backward_counter_with_packet_time_back_within_the_gap_is_out_of_order(self):
        ds = _packets([10, 11, 5, 6], offsets_s=[100, 101, 60, 61])
        assert out_of_order_positions(ds, "PACKET_ICIE_TIME").tolist() == [1]

    @pytest.mark.parametrize("offsets_s", [[0, 1, 62, 63], [100, 101, 0, 1]], ids=["gap-forward", "gap-backward"])
    def test_counter_restart_across_a_gap_is_not_out_of_order(self, offsets_s):
        assert out_of_order_positions(_packets([10, 11, 0, 1], offsets_s), "PACKET_ICIE_TIME").tolist() == []

    @pytest.mark.parametrize("missing", ["SRC_SEQ_CTR", "PACKET_ICIE_TIME"])
    def test_missing_variable_raises(self, missing):
        with pytest.raises(ValueError, match=f"no '{missing}' variable"):
            out_of_order_positions(_packets([10, 11]).drop_vars(missing), "PACKET_ICIE_TIME")


def _order(counters, offsets_s=None) -> list[int]:
    ds = _packets(counters, offsets_s)
    keys = _unwrap_sequence_counter(ds["SRC_SEQ_CTR"].values.astype(np.int64), ds["PACKET_ICIE_TIME"].values)
    np.testing.assert_array_equal(keys % SEQUENCE_COUNTER_MODULUS, counters)
    return np.asarray(counters)[np.argsort(keys, kind="stable")].tolist()


class TestUnwrapSequenceCounter:
    @pytest.mark.parametrize(
        ("counters", "offsets_s", "expected"),
        [
            pytest.param([10, 11, 12, 13], None, [10, 11, 12, 13], id="clean"),
            pytest.param([16382, 16383, 0, 1], None, [16382, 16383, 0, 1], id="wrap"),
            pytest.param([16379, 16380, 2, 3], None, [16379, 16380, 2, 3], id="wrap-with-loss"),
            pytest.param([10, 11, 11, 12], None, [10, 11, 11, 12], id="repeated-counter"),
            pytest.param([10, 8010], [0, 59], [10, 8010], id="forward-step-inside-the-segment-gap"),
            pytest.param([12, 13, 10, 11], [120, 121, 0, 1], [10, 11, 12, 13], id="segments-placed-by-time"),
            pytest.param([0, 1, 5000, 5001], [120, 121, 0, 1], [5000, 5001, 0, 1], id="reset-across-a-segment-gap"),
            pytest.param([100, 101, 102, 103], [0, 2, 1, 3], [100, 101, 102, 103], id="time-disagrees"),
            pytest.param([102, 100, 101], [5, 5, 5], [100, 101, 102], id="equal-times"),
            pytest.param(
                list(range(10)) + list(range(5, 15)),
                list(range(10)) + list(range(5, 15)),
                list(range(5)) + [n for n in range(5, 10) for _ in (0, 1)] + list(range(10, 15)),
                id="overlapping-files",
            ),
            pytest.param(
                [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7],
                [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7],
                list(range(12)),
                id="gap-fill",
            ),
            pytest.param([7], None, [7], id="single-packet"),
            pytest.param([], None, [], id="no-packets"),
        ],
    )
    def test_orders_by_counter(self, counters, offsets_s, expected):
        assert _order(counters, offsets_s) == expected

    def test_segment_is_placed_by_time_even_where_its_counter_continues_another(self):
        assert _order([12, 13, 10, 11], [0, 1, 300, 301]) == [12, 13, 10, 11]

    def test_segments_covering_the_whole_counter_range_stay_in_order(self):
        half = SEQUENCE_COUNTER_MODULUS // 2
        counters = list(range(SEQUENCE_COUNTER_MODULUS))
        offsets_s = [n * 1e-3 for n in range(half)] + [120 + n * 1e-3 for n in range(half)]
        assert _order(counters, offsets_s) == counters

    def test_order_does_not_depend_on_input_order(self):
        n_packets = 20000
        counters = np.arange(n_packets) % SEQUENCE_COUNTER_MODULUS
        shuffle = np.random.default_rng(0).permutation(n_packets)
        ds = _packets(counters[shuffle], np.arange(n_packets)[shuffle])
        keys = _unwrap_sequence_counter(ds["SRC_SEQ_CTR"].values.astype(np.int64), ds["PACKET_ICIE_TIME"].values)
        np.testing.assert_array_equal(np.argsort(keys), np.argsort(shuffle))


class TestSortBySequenceCounter:
    @staticmethod
    def _with_data(counters, offsets_s=None, data=None) -> xr.Dataset:
        ds = _packets(counters, offsets_s)
        ds["DATA"] = ("PACKET", np.asarray(data if data is not None else counters, dtype=np.int64))
        return ds

    def test_ordered_axis_is_returned_unchanged(self):
        ds = self._with_data([10, 11, 12])
        assert sort_by_sequence_counter(ds, "PACKET_ICIE_TIME") is ds

    def test_reorders_every_packet_variable_and_the_time_coordinate(self):
        ds = self._with_data([102, 100, 101], offsets_s=[0, 2, 1], data=[3, 1, 2])
        result = sort_by_sequence_counter(ds, "PACKET_ICIE_TIME")
        assert result["SRC_SEQ_CTR"].values.tolist() == [100, 101, 102]
        assert result["DATA"].values.tolist() == [1, 2, 3]
        assert result["PACKET_ICIE_TIME"].values.tolist() == (_START + np.array([2, 1, 0]) * 1_000_000).tolist()
        assert "PACKET_ICIE_TIME" in result.coords

    def test_identical_repeat_is_dropped_with_one_warning(self):
        ds = self._with_data([10, 11, 11, 12], offsets_s=[0, 1, 1, 2])
        with pytest.warns(UserWarning, match="Detected 1 duplicate SRC_SEQ_CTR packets") as caught:
            result = sort_by_sequence_counter(ds, "PACKET_ICIE_TIME")
        assert len(caught) == 1
        assert result["SRC_SEQ_CTR"].values.tolist() == [10, 11, 12]

    def test_distinct_packets_sharing_a_packet_time_are_kept(self):
        ds = self._with_data([10, 11, 12], offsets_s=[0, 1, 1])
        assert sort_by_sequence_counter(ds, "PACKET_ICIE_TIME").sizes["PACKET"] == 3

    def test_differing_repeat_raises_with_counter_and_times(self):
        ds = self._with_data([10, 11, 11, 12], offsets_s=[0, 1, 1.5, 2], data=[0, 1, 2, 3])
        with pytest.raises(ValueError, match=r"2 packets share SRC_SEQ_CTR 11 .* differ in \['DATA'\]") as error:
            sort_by_sequence_counter(ds, "PACKET_ICIE_TIME")
        assert "pass ground_data=True" in str(error.value)

    def test_counter_repeated_three_times_raises_naming_all_three(self):
        ds = self._with_data([10, 11, 11, 11, 12], offsets_s=[0, 1, 1, 1.5, 2], data=[0, 1, 1, 2, 3])
        with pytest.raises(ValueError, match="3 packets share SRC_SEQ_CTR 11"):
            sort_by_sequence_counter(ds, "PACKET_ICIE_TIME")

    def test_separate_differing_repeats_with_ground_data_keep_each_first(self):
        ds = self._with_data([10, 10, 11, 12, 12], offsets_s=[0, 0.5, 1, 2, 2.5], data=[0, 1, 2, 3, 4])
        with pytest.warns(UserWarning, match="SRC_SEQ_CTR") as caught:
            result = sort_by_sequence_counter(ds, "PACKET_ICIE_TIME", ground_data=True, verbose=True)
        assert len(caught) == 3
        assert "Detected 2 duplicate SRC_SEQ_CTR packets" in str(caught[-1].message)
        assert result["DATA"].values.tolist() == [0, 2, 3]

    @pytest.mark.parametrize(("verbose", "n_warnings"), [(False, 1), (True, 2)])
    def test_differing_repeat_with_ground_data_keeps_the_first(self, verbose, n_warnings):
        ds = self._with_data([10, 11, 11, 12], offsets_s=[0, 1, 1.5, 2], data=[0, 1, 2, 3])
        with pytest.warns(UserWarning, match="SRC_SEQ_CTR") as caught:
            result = sort_by_sequence_counter(ds, "PACKET_ICIE_TIME", ground_data=True, verbose=verbose)
        assert len(caught) == n_warnings
        assert result["DATA"].values.tolist() == [0, 1, 3]

    def test_counter_running_on_across_a_packet_clock_jump_raises(self):
        # The clock steps back 300 s between counters 11 and 12, which sit next to each other in the file.
        ds = self._with_data([10, 11, 12, 13], offsets_s=[300, 301, 0, 1])
        with pytest.raises(ValueError, match="placed a counter wrap apart") as excinfo:
            sort_by_sequence_counter(ds, "PACKET_ICIE_TIME")
        assert "1 pair(s)" in str(excinfo.value)

    def test_packet_a_wrap_away_by_time_from_its_counter_neighbours_raises(self):
        # The middle packet's clock reads 200 s late: its counter continues its neighbours in the file,
        # but its time puts it past a gap of more than SEGMENT_GAP, into the next wrap.
        ds = self._with_data([10, 11, 12], offsets_s=[0, 200, 1])
        with pytest.raises(ValueError, match="placed a counter wrap apart") as excinfo:
            sort_by_sequence_counter(ds, "PACKET_ICIE_TIME")
        assert "2 pair(s)" in str(excinfo.value)
        assert "first at input position 1: SRC_SEQ_CTR 10 -> 11" in str(excinfo.value)
