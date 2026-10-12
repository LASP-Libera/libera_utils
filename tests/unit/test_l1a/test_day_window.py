"""Unit tests for L1A day-window trim and uniqueness checks."""

from datetime import date, datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_window import (
    DataTimeUniquenessError,
    assert_data_times_unique_monotonic,
    day_core_and_buffer_bounds,
    day_window_bounds,
    trim_l1a_to_day_window,
)
from libera_utils.l1a.packet_slicing import MissingDataTimeError

DAY = date(2028, 2, 15)
WINDOW_START = datetime(2028, 2, 14, 23, 50)
WINDOW_END = datetime(2028, 2, 16, 0, 10)
ONE_US = timedelta(microseconds=1)


def _dt64(times: list[datetime], unit: str = "us") -> np.ndarray:
    return np.array([np.datetime64(t.replace(tzinfo=None), unit) for t in times])


def _packet_dataset(
    times: list[datetime], *, apid: LiberaApid = LiberaApid.icie_nom_hk, packet_time_var: str = "PACKET_ICIE_TIME"
) -> xr.Dataset:
    """Build a dataset with no sample axis, of an APID whose packet time is ``packet_time_var``."""
    return xr.Dataset(
        {"VALUE": ("PACKET", np.arange(len(times))), "PKT_APID": ("PACKET", np.full(len(times), int(apid)))},
        coords={packet_time_var: ("PACKET", _dt64(times))},
    )


def _sample_dataset(
    groups: dict[str, tuple[list[datetime], list[int]]],
    packet_times: list[datetime],
    *,
    unit: str = "us",
) -> xr.Dataset:
    """Build a decoded L1A-shaped dataset: a ``PACKET`` axis plus one sample axis per group.

    ``groups`` maps a sample group name to its sample times and the packet each sample came from.
    """
    data_vars = {"PKT_VAL": ("PACKET", np.arange(len(packet_times), dtype=np.int32))}
    coords = {"PACKET_ICIE_TIME": ("PACKET", _dt64(packet_times, unit))}
    for name, (times, packet_indices) in groups.items():
        dim = f"{name}_TIME"
        data_vars[f"{name}_VALUE"] = (dim, np.arange(len(times)))
        data_vars[f"{name}_packet_index"] = (dim, np.asarray(packet_indices, dtype=np.int32))
        coords[dim] = (dim, _dt64(times, unit))
    return xr.Dataset(data_vars, coords=coords)


def _axis_dataset(times: list[datetime]) -> xr.Dataset:
    """Build a dataset with one sample axis, ``RAD_SAMPLE_TIME``, one sample per packet."""
    return _sample_dataset({"RAD_SAMPLE": (times, list(range(len(times))))}, times)


def test_day_window_bounds_are_naive_datetimes():
    assert day_window_bounds(DAY) == (WINDOW_START, WINDOW_END)


def test_day_core_and_buffer_bounds_abut():
    left, core, right = day_core_and_buffer_bounds(DAY)
    assert left == (WINDOW_START, datetime(2028, 2, 15))
    assert core == (datetime(2028, 2, 15), datetime(2028, 2, 16))
    assert right == (datetime(2028, 2, 16), WINDOW_END)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_trim_keeps_both_window_edges_and_drops_one_microsecond_outside(unit):
    sample_times = [WINDOW_START - ONE_US, WINDOW_START, datetime(2028, 2, 15, 12), WINDOW_END, WINDOW_END + ONE_US]
    ds = _sample_dataset({"RAD_SAMPLE": (sample_times, [0, 1, 2, 3, 4])}, sample_times, unit=unit)
    out = trim_l1a_to_day_window(ds, day=DAY)
    assert out["PKT_VAL"].values.tolist() == [1, 2, 3]
    assert out["RAD_SAMPLE_packet_index"].values.tolist() == [0, 1, 2]


def test_uniqueness_ok():
    assert_data_times_unique_monotonic(_axis_dataset([datetime(2028, 2, 15, 1, 0), datetime(2028, 2, 15, 2, 0)]))


def test_uniqueness_empty_passes():
    assert_data_times_unique_monotonic(_packet_dataset([]))


def test_uniqueness_duplicate_raises_and_lists_duplicates():
    t = datetime(2028, 2, 15, 1, 0)
    ds = _axis_dataset([t, t, datetime(2028, 2, 15, 2, 0)])
    with pytest.raises(
        DataTimeUniquenessError,
        match=r"'RAD_SAMPLE_TIME' are not unique: 1 duplicate value\(s\): \['2028-02-15T01:00:00.000000'\]",
    ):
        assert_data_times_unique_monotonic(ds)


def test_uniqueness_lists_at_most_ten_duplicates():
    times = [datetime(2028, 2, 15, hour) for hour in range(12) for _ in range(2)]
    with pytest.raises(DataTimeUniquenessError, match="12 duplicate") as excinfo:
        assert_data_times_unique_monotonic(_axis_dataset(times))
    assert "T09:00" in str(excinfo.value)
    assert "T10:00" not in str(excinfo.value)


def test_uniqueness_out_of_order_raises():
    ds = _axis_dataset([datetime(2028, 2, 15, 2, 0), datetime(2028, 2, 15, 1, 0)])
    with pytest.raises(DataTimeUniquenessError, match="monotonic"):
        assert_data_times_unique_monotonic(ds)


def test_uniqueness_out_of_order_duplicates_are_listed():
    t = datetime(2028, 2, 15, 1, 0)
    ds = _axis_dataset([t, datetime(2028, 2, 15, 2, 0), t])
    with pytest.raises(DataTimeUniquenessError, match="1 duplicate"):
        assert_data_times_unique_monotonic(ds)


def test_uniqueness_checks_every_sample_axis():
    """ADGPS is clean; a duplicate on ADCFA alone still raises and is named."""
    times = [datetime(2028, 2, 15, 1, 0), datetime(2028, 2, 15, 2, 0)]
    ds = _sample_dataset({"ADGPS": (times, [0, 1]), "ADCFA": ([times[0], times[0]], [0, 1])}, times)
    with pytest.raises(DataTimeUniquenessError, match="'ADCFA_TIME'"):
        assert_data_times_unique_monotonic(ds)


def test_uniqueness_packet_only_product_checks_packet_time():
    t = datetime(2028, 2, 15, 1, 0)
    with pytest.raises(DataTimeUniquenessError, match="'PACKET_ICIE_TIME'"):
        assert_data_times_unique_monotonic(_packet_dataset([t, t]))


def test_uniqueness_packet_only_product_without_its_time_variable_raises():
    ds = _packet_dataset([datetime(2028, 2, 15, 12)], apid=LiberaApid.jpss_sc_pos)
    with pytest.raises(ValueError, match="PACKET_JPSS_TIME"):
        assert_data_times_unique_monotonic(ds)


@pytest.mark.parametrize("n_nat", [1, 2])
@pytest.mark.parametrize("ground_data", [False, True])
def test_uniqueness_nat_raises(n_nat, ground_data):
    """NaT is a missing data time, not a duplicate, and ground_data does not soften it."""
    ds = _axis_dataset([datetime(2028, 2, 15, 1, 0)] * (n_nat + 1))
    ds["RAD_SAMPLE_TIME"] = np.concatenate(
        [_dt64([datetime(2028, 2, 15, 1, 0)]), np.full(n_nat, np.datetime64("NaT", "us"))]
    )
    with pytest.raises(MissingDataTimeError, match=f"RAD_SAMPLE_TIME holds {n_nat} NaT"):
        assert_data_times_unique_monotonic(ds, ground_data=ground_data)


def test_uniqueness_ground_data_warns_on_duplicates():
    t = datetime(2028, 2, 15, 1, 0)
    with pytest.warns(UserWarning, match="not unique"):
        assert_data_times_unique_monotonic(_axis_dataset([t, t]), ground_data=True)


def test_uniqueness_ground_data_warns_on_out_of_order():
    ds = _axis_dataset([datetime(2028, 2, 15, 2, 0), datetime(2028, 2, 15, 1, 0)])
    with pytest.warns(UserWarning, match="monotonic"):
        assert_data_times_unique_monotonic(ds, ground_data=True)
