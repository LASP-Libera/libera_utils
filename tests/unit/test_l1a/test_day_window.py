"""Unit tests for L1A day-window trim and uniqueness checks."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.day_window import (
    DataTimeUniquenessError,
    assert_data_times_unique_monotonic,
    day_core_and_buffer_bounds,
    day_window_bounds,
    trim_l1a_to_day_window,
)

DAY = date(2028, 2, 15)
WINDOW_START = datetime(2028, 2, 14, 23, 50)
WINDOW_END = datetime(2028, 2, 16, 0, 10)
ONE_US = timedelta(microseconds=1)


def _dt64(times: list[datetime], unit: str = "us") -> np.ndarray:
    return np.array([np.datetime64(t.replace(tzinfo=None), unit) for t in times])


def _packet_dataset(times: list[datetime], packet_time_var: str = "PACKET_ICIE_TIME") -> xr.Dataset:
    return xr.Dataset(
        {"VALUE": ("PACKET", np.arange(len(times)))},
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


def _wfov_dataset(camera_times: list[datetime], packet_image_ids: list[int], packet_times: list[datetime]):
    """Build a WFOV SCI-shaped dataset. Each image's SOP is its first packet in ``packet_image_ids``."""
    image_ids = sorted({i for i in packet_image_ids if i >= 0})
    sop_indices = [packet_image_ids.index(i) for i in image_ids]
    return xr.Dataset(
        {
            "PKT_VAL": ("PACKET", np.arange(len(packet_times))),
            "PACKET_IMAGE_ID": ("PACKET", np.asarray(packet_image_ids, dtype=np.int32)),
            "CAMERA_PACKET_INDEX": ("CAMERA_TIME", np.asarray(sop_indices, dtype=np.int32)),
            "IMG": ("CAMERA_TIME", np.asarray(image_ids)),
        },
        coords={
            "PACKET_ICIE_TIME": ("PACKET", _dt64(packet_times)),
            "CAMERA_TIME": ("CAMERA_TIME", _dt64(camera_times)),
        },
    )


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


def test_trim_packet_only_product_on_packet_time():
    ds = _packet_dataset(
        [
            datetime(2028, 2, 14, 23, 40, tzinfo=UTC),
            WINDOW_START,
            datetime(2028, 2, 15, 12, 0, tzinfo=UTC),
            WINDOW_END,
            datetime(2028, 2, 16, 0, 20, tzinfo=UTC),
        ],
        packet_time_var="PACKET_JPSS_TIME",
    )
    out = trim_l1a_to_day_window(ds, day=DAY, packet_time_var="PACKET_JPSS_TIME")
    assert out["VALUE"].values.tolist() == [1, 2, 3]


def test_trim_packet_only_product_without_its_time_variable_raises():
    ds = _packet_dataset([datetime(2028, 2, 15, 12)], packet_time_var="PACKET_JPSS_TIME")
    with pytest.raises(ValueError, match="PACKET_ICIE_TIME"):
        trim_l1a_to_day_window(ds, day=DAY)


def test_trim_empty_window():
    out = trim_l1a_to_day_window(_packet_dataset([datetime(2028, 2, 14, 12, 0, tzinfo=UTC)]), day=DAY)
    assert out.sizes["PACKET"] == 0


def test_trim_keeps_whole_packets_for_straddling_samples():
    """If any sample of a packet is in the window, all samples of that packet are kept."""
    sample_times = [
        datetime(2028, 2, 14, 23, 40),
        datetime(2028, 2, 14, 23, 41),
        datetime(2028, 2, 14, 23, 49),
        datetime(2028, 2, 14, 23, 55),
        datetime(2028, 2, 15, 12, 0),
        datetime(2028, 2, 15, 12, 0, 5),
    ]
    ds = _sample_dataset({"RAD_SAMPLE": (sample_times, [0, 0, 1, 1, 2, 2])}, sample_times[::2])
    out = trim_l1a_to_day_window(ds, day=DAY)
    assert out["RAD_SAMPLE_VALUE"].values.tolist() == [2, 3, 4, 5]
    assert out["PKT_VAL"].values.tolist() == [1, 2]
    assert out["RAD_SAMPLE_packet_index"].values.tolist() == [0, 0, 1, 1]


def test_trim_covers_every_sample_group():
    """ADGPS and ADCFA are trimmed together, and both indices stay within the trimmed PACKET axis."""
    packet_times = [datetime(2028, 2, 14, 23, 0), datetime(2028, 2, 15, 12, 0), datetime(2028, 2, 16, 1, 0)]
    ds = _sample_dataset(
        {
            "ADGPS": (
                [datetime(2028, 2, 14, 23, 0), datetime(2028, 2, 15, 12, 0), datetime(2028, 2, 16, 1, 0)],
                [0, 1, 2],
            ),
            # ADCFA of packet 0 is late enough to fall in the window; ADGPS of packet 0 is not.
            "ADCFA": (
                [datetime(2028, 2, 14, 23, 55), datetime(2028, 2, 15, 12, 0, 1), datetime(2028, 2, 16, 1, 0, 1)],
                [0, 1, 2],
            ),
        },
        packet_times,
    )
    out = trim_l1a_to_day_window(ds, day=DAY)
    assert out["PKT_VAL"].values.tolist() == [0, 1]
    assert out["ADGPS_VALUE"].values.tolist() == [0, 1]
    assert out["ADCFA_VALUE"].values.tolist() == [0, 1]
    for index_var in ("ADGPS_packet_index", "ADCFA_packet_index"):
        assert out[index_var].values.tolist() == [0, 1]
        assert out[index_var].values.max() < out.sizes["PACKET"]


def test_trim_keeps_every_packet_of_a_kept_wfov_image():
    """Image 1's SOP packet time is outside the window but its camera time is in it."""
    ds = _wfov_dataset(
        camera_times=[datetime(2028, 2, 14, 23, 0), datetime(2028, 2, 14, 23, 52), datetime(2028, 2, 16, 0, 20)],
        packet_image_ids=[0, 0, 0, 1, 1, 1, -1, 2, 2, 2],
        packet_times=[
            datetime(2028, 2, 14, 23, 0),
            datetime(2028, 2, 14, 23, 0, 1),
            datetime(2028, 2, 14, 23, 0, 2),
            datetime(2028, 2, 14, 23, 49),
            datetime(2028, 2, 14, 23, 50, 1),
            datetime(2028, 2, 14, 23, 50, 2),
            datetime(2028, 2, 15, 12),
            datetime(2028, 2, 16, 0, 20),
            datetime(2028, 2, 16, 0, 20, 1),
            datetime(2028, 2, 16, 0, 20, 2),
        ],
    )
    out = trim_l1a_to_day_window(ds, day=DAY)
    assert out["PKT_VAL"].values.tolist() == [3, 4, 5, 6]
    assert out["IMG"].values.tolist() == [1]
    assert out["CAMERA_PACKET_INDEX"].values.tolist() == [0]


def test_uniqueness_ok():
    ds = _packet_dataset([datetime(2028, 2, 15, 1, 0), datetime(2028, 2, 15, 2, 0)])
    assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME")


def test_uniqueness_empty_passes():
    assert_data_times_unique_monotonic(_packet_dataset([]), "PACKET_ICIE_TIME")


def test_uniqueness_missing_coordinate_raises():
    with pytest.raises(KeyError, match="CAMERA_TIME"):
        assert_data_times_unique_monotonic(_packet_dataset([datetime(2028, 2, 15)]), "CAMERA_TIME")


def test_uniqueness_duplicate_raises_and_lists_duplicates():
    t = datetime(2028, 2, 15, 1, 0)
    ds = _packet_dataset([t, t, datetime(2028, 2, 15, 2, 0)])
    with pytest.raises(DataTimeUniquenessError, match=r"1 duplicate value\(s\): \['2028-02-15T01:00:00.000000'\]"):
        assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME")


def test_uniqueness_lists_at_most_ten_duplicates():
    times = [datetime(2028, 2, 15, hour) for hour in range(12) for _ in range(2)]
    with pytest.raises(DataTimeUniquenessError, match="12 duplicate") as excinfo:
        assert_data_times_unique_monotonic(_packet_dataset(times), "PACKET_ICIE_TIME")
    assert "T09:00" in str(excinfo.value)
    assert "T10:00" not in str(excinfo.value)


def test_uniqueness_out_of_order_raises():
    ds = _packet_dataset([datetime(2028, 2, 15, 2, 0), datetime(2028, 2, 15, 1, 0)])
    with pytest.raises(DataTimeUniquenessError, match="monotonic"):
        assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME")


def test_uniqueness_ground_data_warns_on_duplicates():
    t = datetime(2028, 2, 15, 1, 0)
    with pytest.warns(UserWarning, match="not unique"):
        assert_data_times_unique_monotonic(_packet_dataset([t, t]), "PACKET_ICIE_TIME", ground_data=True)


def test_uniqueness_ground_data_warns_on_out_of_order():
    ds = _packet_dataset([datetime(2028, 2, 15, 2, 0), datetime(2028, 2, 15, 1, 0)])
    with pytest.warns(UserWarning, match="monotonic"):
        assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME", ground_data=True)
