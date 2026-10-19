"""Unit tests for L1A day-window trim and data-time checks."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_window import (
    DayBounds,
    check_data_times,
    day_bounds,
    trim_l1a_to_day_window,
)

DAY = date(2028, 2, 15)
WINDOW_START = datetime(2028, 2, 14, 23, 50, tzinfo=UTC)
WINDOW_END = datetime(2028, 2, 16, 0, 10, tzinfo=UTC)
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


def test_day_bounds_are_aware_utc():
    bounds = day_bounds(DAY)
    assert bounds == DayBounds(
        WINDOW_START, datetime(2028, 2, 15, tzinfo=UTC), datetime(2028, 2, 16, tzinfo=UTC), WINDOW_END
    )
    assert all(value.tzinfo is UTC for value in bounds)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_trim_keeps_both_window_edges_and_drops_one_microsecond_outside(unit):
    sample_times = [WINDOW_START - ONE_US, WINDOW_START, datetime(2028, 2, 15, 12), WINDOW_END, WINDOW_END + ONE_US]
    ds = _sample_dataset({"RAD_SAMPLE": (sample_times, [0, 1, 2, 3, 4])}, sample_times, unit=unit)
    out = trim_l1a_to_day_window(ds, day=DAY)
    assert out["PKT_VAL"].values.tolist() == [1, 2, 3]
    assert out["RAD_SAMPLE_packet_index"].values.tolist() == [0, 1, 2]


def _warnings(caplog) -> list[str]:
    return [r.message for r in caplog.records if r.name == "libera_utils.l1a.day_window" and r.levelname == "WARNING"]


@pytest.fixture
def day_window_log(caplog):
    caplog.set_level("WARNING", logger="libera_utils.l1a.day_window")
    return caplog


def test_check_ok_logs_nothing(day_window_log):
    check_data_times(_axis_dataset([datetime(2028, 2, 15, 1, 0), datetime(2028, 2, 15, 2, 0)]))
    assert _warnings(day_window_log) == []


def test_check_empty_logs_nothing(day_window_log):
    check_data_times(_packet_dataset([]))
    assert _warnings(day_window_log) == []


def test_check_duplicate_warns_and_lists_duplicates(day_window_log):
    t = datetime(2028, 2, 15, 1, 0)
    check_data_times(_axis_dataset([t, t, datetime(2028, 2, 15, 2, 0)]))
    assert _warnings(day_window_log) == [
        "Data times on 'RAD_SAMPLE_TIME' are not unique: 1 duplicate value(s): ['2028-02-15T01:00:00.000000']"
    ]


def test_check_lists_at_most_ten_duplicates(day_window_log):
    times = [datetime(2028, 2, 15, hour) for hour in range(12) for _ in range(2)]
    check_data_times(_axis_dataset(times))
    (message,) = _warnings(day_window_log)
    assert "12 duplicate" in message
    assert "T09:00" in message
    assert "T10:00" not in message


def test_check_backwards_step_warns(day_window_log):
    check_data_times(_axis_dataset([datetime(2028, 2, 15, 2, 0), datetime(2028, 2, 15, 1, 0)]))
    assert _warnings(day_window_log) == ["Data times on 'RAD_SAMPLE_TIME' step backwards at 1 of 2 position(s)"]


def test_check_reports_duplicates_and_backwards_steps_together(day_window_log):
    t = datetime(2028, 2, 15, 1, 0)
    check_data_times(_axis_dataset([t, datetime(2028, 2, 15, 2, 0), t]))
    messages = _warnings(day_window_log)
    assert len(messages) == 2
    assert "1 duplicate" in messages[0]
    assert "step backwards" in messages[1]


def test_check_covers_every_sample_axis(day_window_log):
    """ADGPS is clean; a duplicate on ADCFA alone still warns and is named."""
    times = [datetime(2028, 2, 15, 1, 0), datetime(2028, 2, 15, 2, 0)]
    ds = _sample_dataset({"ADGPS": (times, [0, 1]), "ADCFA": ([times[0], times[0]], [0, 1])}, times)
    check_data_times(ds)
    (message,) = _warnings(day_window_log)
    assert "'ADCFA_TIME'" in message


def test_check_packet_only_product_checks_packet_time(day_window_log):
    t = datetime(2028, 2, 15, 1, 0)
    check_data_times(_packet_dataset([t, t]))
    (message,) = _warnings(day_window_log)
    assert "'PACKET_ICIE_TIME'" in message


def test_check_packet_only_product_without_its_time_variable_raises():
    ds = _packet_dataset([datetime(2028, 2, 15, 12)], apid=LiberaApid.jpss_sc_pos)
    with pytest.raises(ValueError, match="PACKET_JPSS_TIME"):
        check_data_times(ds)
