"""Unit tests for L1A day-window trim and uniqueness checks."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.day_window import (
    DEFAULT_DAY_BUFFER,
    DataTimeUniquenessError,
    assert_data_times_unique_monotonic,
    day_window_bounds,
    sync_packet_dim_to_index,
    trim_l1a_to_day_window,
)


def _packet_dataset(times: list[datetime]) -> xr.Dataset:
    t = np.array([np.datetime64(tt.replace(tzinfo=None), "us") for tt in times])
    return xr.Dataset(
        {"VALUE": ("PACKET", np.arange(len(times)))},
        coords={"PACKET_ICIE_TIME": ("PACKET", t)},
    )


def _sample_dataset(sample_times: list[datetime], packet_indices: list[int]) -> xr.Dataset:
    t = np.array([np.datetime64(tt.replace(tzinfo=None), "us") for tt in sample_times])
    return xr.Dataset(
        {
            "SAMPLE": ("RAD_SAMPLE_FPE_TIME", np.arange(len(sample_times))),
            "RAD_SAMPLE_packet_index": ("RAD_SAMPLE_FPE_TIME", np.asarray(packet_indices)),
        },
        coords={"RAD_SAMPLE_FPE_TIME": t},
    )


def test_day_window_bounds_default_buffer():
    start, end = day_window_bounds(date(2028, 2, 15))
    assert start == np.datetime64("2028-02-14T23:50:00", "us")
    assert end == np.datetime64("2028-02-16T00:10:00", "us")
    assert DEFAULT_DAY_BUFFER == timedelta(minutes=10)


def test_trim_in_day_only():
    day = date(2028, 2, 15)
    ds = _packet_dataset(
        [
            datetime(2028, 2, 15, 1, 0, tzinfo=UTC),
            datetime(2028, 2, 15, 12, 0, tzinfo=UTC),
            datetime(2028, 2, 15, 23, 0, tzinfo=UTC),
        ]
    )
    out = trim_l1a_to_day_window(ds, day=day, time_coord="PACKET_ICIE_TIME")
    assert out.sizes["PACKET"] == 3


def test_trim_left_and_right_buffer():
    day = date(2028, 2, 15)
    ds = _packet_dataset(
        [
            datetime(2028, 2, 14, 23, 40, tzinfo=UTC),  # outside left
            datetime(2028, 2, 14, 23, 55, tzinfo=UTC),  # in left buffer
            datetime(2028, 2, 15, 12, 0, tzinfo=UTC),
            datetime(2028, 2, 16, 0, 5, tzinfo=UTC),  # in right buffer
            datetime(2028, 2, 16, 0, 20, tzinfo=UTC),  # outside right
        ]
    )
    out = trim_l1a_to_day_window(ds, day=day, time_coord="PACKET_ICIE_TIME")
    assert out.sizes["PACKET"] == 3
    assert out["VALUE"].values.tolist() == [1, 2, 3]


def test_trim_empty_window():
    day = date(2028, 2, 15)
    ds = _packet_dataset([datetime(2028, 2, 14, 12, 0, tzinfo=UTC)])
    out = trim_l1a_to_day_window(ds, day=day, time_coord="PACKET_ICIE_TIME")
    assert out.sizes["PACKET"] == 0


def test_trim_keep_whole_packets_for_straddling_samples():
    """If any sample of a packet is in-window, keep all samples of that packet."""
    day = date(2028, 2, 15)
    # Packet 0: both samples before left buffer
    # Packet 1: first sample outside left, second sample in left buffer -> keep both
    # Packet 2: both in day
    ds = _sample_dataset(
        [
            datetime(2028, 2, 14, 23, 40, tzinfo=UTC),
            datetime(2028, 2, 14, 23, 41, tzinfo=UTC),
            datetime(2028, 2, 14, 23, 49, tzinfo=UTC),
            datetime(2028, 2, 14, 23, 55, tzinfo=UTC),
            datetime(2028, 2, 15, 12, 0, tzinfo=UTC),
            datetime(2028, 2, 15, 12, 0, 5, tzinfo=UTC),
        ],
        packet_indices=[0, 0, 1, 1, 2, 2],
    )
    out = trim_l1a_to_day_window(
        ds,
        day=day,
        time_coord="RAD_SAMPLE_FPE_TIME",
        keep_whole_groups=True,
        packet_index_var="RAD_SAMPLE_packet_index",
    )
    assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 4  # packets 1 and 2
    assert set(out["RAD_SAMPLE_packet_index"].values.tolist()) == {1, 2}


def test_trim_multiple_camera_times_in_buffer():
    day = date(2028, 2, 15)
    ds = xr.Dataset(
        {"IMG": ("CAMERA_TIME", np.arange(4))},
        coords={
            "CAMERA_TIME": (
                "CAMERA_TIME",
                np.array(
                    [
                        np.datetime64("2028-02-14T23:52:00", "us"),
                        np.datetime64("2028-02-14T23:58:00", "us"),
                        np.datetime64("2028-02-15T00:01:00", "us"),
                        np.datetime64("2028-02-16T00:20:00", "us"),
                    ]
                ),
            )
        },
    )
    out = trim_l1a_to_day_window(ds, day=day, time_coord="CAMERA_TIME", keep_whole_groups=True)
    assert out.sizes["CAMERA_TIME"] == 3


def test_sync_packet_dim_to_index_densifies_and_preserves_dtype():
    """Orphan PACKET rows are dropped; indices remap to 0..n-1 with source dtype."""
    ds = xr.Dataset(
        {
            "PKT_VAL": ("PACKET", np.array([10, 20, 30, 40], dtype=np.int16)),
            "SAMPLE": ("RAD_SAMPLE_FPE_TIME", np.arange(3)),
            "RAD_SAMPLE_packet_index": (
                "RAD_SAMPLE_FPE_TIME",
                np.array([1, 1, 3], dtype=np.int32),
            ),
        },
        coords={
            "RAD_SAMPLE_FPE_TIME": (
                "RAD_SAMPLE_FPE_TIME",
                np.array(
                    [
                        np.datetime64("2028-02-15T12:00:00", "us"),
                        np.datetime64("2028-02-15T12:00:01", "us"),
                        np.datetime64("2028-02-15T12:00:02", "us"),
                    ]
                ),
            )
        },
    )
    out = sync_packet_dim_to_index(ds, "RAD_SAMPLE_packet_index")
    assert out.sizes["PACKET"] == 2
    assert out["PKT_VAL"].values.tolist() == [20, 40]
    assert out["RAD_SAMPLE_packet_index"].dtype == np.int32
    assert out["RAD_SAMPLE_packet_index"].values.tolist() == [0, 0, 1]


def test_trim_syncs_orphan_packets_for_sample_dim():
    """After science-dim trim, PACKET length matches unique remaining packet indices."""
    day = date(2028, 2, 15)
    sample_times = [
        datetime(2028, 2, 14, 23, 40, tzinfo=UTC),  # out
        datetime(2028, 2, 14, 23, 41, tzinfo=UTC),  # out
        datetime(2028, 2, 15, 12, 0, tzinfo=UTC),  # in
        datetime(2028, 2, 15, 12, 0, 5, tzinfo=UTC),  # in
    ]
    ds = xr.Dataset(
        {
            "PKT_VAL": ("PACKET", np.array([0, 1, 2], dtype=np.int32)),
            "SAMPLE": ("RAD_SAMPLE_FPE_TIME", np.arange(4)),
            "RAD_SAMPLE_packet_index": (
                "RAD_SAMPLE_FPE_TIME",
                np.array([0, 0, 2, 2], dtype=np.int32),
            ),
        },
        coords={
            "RAD_SAMPLE_FPE_TIME": (
                "RAD_SAMPLE_FPE_TIME",
                np.array([np.datetime64(tt.replace(tzinfo=None), "us") for tt in sample_times]),
            )
        },
    )
    out = trim_l1a_to_day_window(
        ds,
        day=day,
        time_coord="RAD_SAMPLE_FPE_TIME",
        keep_whole_groups=True,
        packet_index_var="RAD_SAMPLE_packet_index",
    )
    assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 2
    assert out.sizes["PACKET"] == 1
    assert out["RAD_SAMPLE_packet_index"].values.tolist() == [0, 0]
    assert out["PKT_VAL"].values.tolist() == [2]


def test_uniqueness_ok():
    ds = _packet_dataset(
        [
            datetime(2028, 2, 15, 1, 0, tzinfo=UTC),
            datetime(2028, 2, 15, 2, 0, tzinfo=UTC),
        ]
    )
    assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME")


def test_uniqueness_duplicate_raises():
    t = datetime(2028, 2, 15, 1, 0, tzinfo=UTC)
    ds = _packet_dataset([t, t])
    with pytest.raises(DataTimeUniquenessError, match="not unique"):
        assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME")


def test_uniqueness_out_of_order_raises():
    ds = _packet_dataset(
        [
            datetime(2028, 2, 15, 2, 0, tzinfo=UTC),
            datetime(2028, 2, 15, 1, 0, tzinfo=UTC),
        ]
    )
    with pytest.raises(DataTimeUniquenessError, match="monotonic"):
        assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME")


def test_uniqueness_ground_data_warns():
    t = datetime(2028, 2, 15, 1, 0, tzinfo=UTC)
    ds = _packet_dataset([t, t])
    with pytest.warns(UserWarning, match="not unique"):
        assert_data_times_unique_monotonic(ds, "PACKET_ICIE_TIME", ground_data=True)
