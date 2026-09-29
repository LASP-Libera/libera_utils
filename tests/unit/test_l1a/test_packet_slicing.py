"""Unit tests for libera_utils.l1a.packet_slicing."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from libera_utils.l1a.packet_slicing import (
    find_sample_dims,
    sample_to_packet_index,
    select_packets,
    slice_l1a_dataset_to_time_window,
)


def _sample_product(
    n_packets: int = 6,
    samples_per_packet: int = 4,
    *,
    with_packet_index: bool = True,
    start: str = "2028-02-13T02:00:00",
    sample_dim: str = "RAD_SAMPLE_FPE_TIME",
    sample_offset_s: int = 0,
) -> xr.Dataset:
    """Build a decoded-L1A-shaped Dataset with one sample group.

    ``sample_offset_s`` shifts sample times relative to packet times, standing in for FPE/ICIE
    clock skew.
    """
    packet_times = np.datetime64(start) + np.arange(n_packets) * np.timedelta64(1, "s")
    n_samples = n_packets * samples_per_packet
    sample_step = np.timedelta64(1000 // samples_per_packet, "ms")
    sample_times = np.datetime64(start) + np.timedelta64(sample_offset_s, "s") + np.arange(n_samples) * sample_step

    data_vars = {
        "PKT_APID": ("PACKET", np.full(n_packets, 1408, dtype=np.uint16)),
        "SRC_SEQ_CTR": ("PACKET", np.arange(n_packets, dtype=np.uint16)),
        "ICIE__RAD_SAMPLE_0": (sample_dim, np.arange(n_samples, dtype=np.float32)),
    }
    if with_packet_index:
        data_vars["RAD_SAMPLE_packet_index"] = (
            sample_dim,
            np.repeat(np.arange(n_packets, dtype=np.int64), samples_per_packet),
        )

    return xr.Dataset(
        data_vars,
        coords={
            "PACKET_ICIE_TIME": ("PACKET", packet_times),
            sample_dim: (sample_dim, sample_times),
        },
    )


def _interleave_packet_pair(dataset: xr.Dataset, packet: int, samples_per_packet: int) -> xr.Dataset:
    """Interleave the sample blocks of ``packet`` and ``packet + 1``.

    Stands in for the ground-data case where two adjacent packets' sample clocks skew by less than
    a sample interval: the axis is stored in sample-time order, so their samples alternate and
    ``RAD_SAMPLE_packet_index`` steps backwards once per pair. Only the sample-indexed variables
    are permuted; the sample time coordinate stays sorted, as it is on disk.
    """
    block = slice(packet * samples_per_packet, (packet + 2) * samples_per_packet)
    interleaved = np.empty(2 * samples_per_packet, dtype=np.int64)
    interleaved[0::2] = np.arange(samples_per_packet)
    interleaved[1::2] = np.arange(samples_per_packet) + samples_per_packet

    dataset = dataset.copy(deep=True)
    for name, variable in dataset.variables.items():
        if variable.dims == ("RAD_SAMPLE_FPE_TIME",) and str(name) != "RAD_SAMPLE_FPE_TIME":
            dataset[name].values[block] = variable.values[block][interleaved]
    return dataset


def _packet_only_product(n_packets: int = 5, *, start: str = "2028-02-13T02:00:00") -> xr.Dataset:
    """Build a PEC/PEV-SW-STAT-shaped Dataset: PACKET plus an array dimension, no sample group."""
    packet_times = np.datetime64(start) + np.arange(n_packets) * np.timedelta64(1, "s")
    return xr.Dataset(
        {
            "PKT_APID": ("PACKET", np.full(n_packets, 1300, dtype=np.uint16)),
            "SOME_ARRAY": (("PACKET", "ARRAY_8"), np.zeros((n_packets, 8), dtype=np.int32)),
        },
        coords={
            "PACKET_ICIE_TIME": ("PACKET", packet_times),
            "ARRAY_8": ("ARRAY_8", np.arange(8, dtype=np.int64)),
        },
    )


def _wfov_product() -> xr.Dataset:
    """Build a WFOV SCI-shaped Dataset: three 3-packet images with one unowned packet between images 0 and 1.

    Packets are 1 s apart from 02:00:00. Camera times are 10 s after each image's SOP packet time.
    """
    packet_image_id = np.array([0, 0, 0, -1, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    sop_index = np.array([0, 4, 7], dtype=np.int32)
    packet_times = np.datetime64("2028-02-13T02:00:00", "us") + np.arange(10) * np.timedelta64(1, "s")
    return xr.Dataset(
        {
            "PKT_APID": ("PACKET", np.full(10, 1040, dtype=np.uint16)),
            "SRC_SEQ_CTR": ("PACKET", np.arange(10, dtype=np.uint16)),
            "PACKET_IMAGE_ID": ("PACKET", packet_image_id),
            "CAMERA_PACKET_INDEX": ("CAMERA_TIME", sop_index, {"long_name": "SOP packet"}),
            "WFOV_COMPRESSED_IMAGE": (("CAMERA_TIME", "BLOB_BYTE"), np.zeros((3, 4), dtype=np.uint8)),
        },
        coords={
            "PACKET_ICIE_TIME": ("PACKET", packet_times),
            "CAMERA_TIME": ("CAMERA_TIME", packet_times[sop_index] + np.timedelta64(10, "s")),
            "BLOB_BYTE": ("BLOB_BYTE", np.arange(4)),
        },
    )


class TestWfovCameraTime:
    """CAMERA_TIME rows own their SOP..EOP packet range via PACKET_IMAGE_ID."""

    def test_camera_time_is_a_sample_dim_and_blob_byte_is_not(self):
        assert find_sample_dims(_wfov_product()) == {"CAMERA_TIME"}

    def test_camera_time_found_via_camera_packet_index_when_times_are_not_decoded(self):
        ds = _wfov_product()
        ds = ds.assign_coords(CAMERA_TIME=("CAMERA_TIME", ds["CAMERA_TIME"].values.astype(np.int64)))
        assert find_sample_dims(ds) == {"CAMERA_TIME"}

    def test_sample_to_packet_index_is_the_sop(self):
        np.testing.assert_array_equal(sample_to_packet_index(_wfov_product(), "CAMERA_TIME"), [0, 4, 7])

    def test_select_packets_renumbers_camera_packet_index(self):
        out = select_packets(_wfov_product(), np.arange(4, 10))
        np.testing.assert_array_equal(out["CAMERA_PACKET_INDEX"].values, [0, 3])
        assert out["CAMERA_PACKET_INDEX"].attrs == {"long_name": "SOP packet"}
        np.testing.assert_array_equal(out["PACKET_IMAGE_ID"].values, [1, 1, 1, 2, 2, 2])

    def test_image_in_window_keeps_all_its_packets(self):
        """Image 1's camera time (02:00:14) is in the window; its later packets' times are not."""
        out = slice_l1a_dataset_to_time_window(
            _wfov_product(), np.datetime64("2028-02-13T02:00:14"), np.datetime64("2028-02-13T02:00:15")
        )
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [4, 5, 6])
        np.testing.assert_array_equal(out["CAMERA_PACKET_INDEX"].values, [0])
        assert out.sizes["CAMERA_TIME"] == 1

    def test_image_out_of_window_drops_all_its_packets(self):
        """Packets 4-6 have packet times in the window, but image 1's camera time does not."""
        out = slice_l1a_dataset_to_time_window(
            _wfov_product(), np.datetime64("2028-02-13T02:00:04"), np.datetime64("2028-02-13T02:00:06")
        )
        assert out.sizes["PACKET"] == 0
        assert out.sizes["CAMERA_TIME"] == 0

    def test_unowned_packet_selected_on_packet_time(self):
        out = slice_l1a_dataset_to_time_window(
            _wfov_product(), np.datetime64("2028-02-13T02:00:03"), np.datetime64("2028-02-13T02:00:03")
        )
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [3])
        assert out.sizes["CAMERA_TIME"] == 0

    def test_nat_camera_time_is_placed_by_its_sop_packet_time(self):
        ds = _wfov_product()
        camera_times = ds["CAMERA_TIME"].values.copy()
        camera_times[2] = np.datetime64("NaT", "us")
        ds = ds.assign_coords(CAMERA_TIME=("CAMERA_TIME", camera_times))
        out = slice_l1a_dataset_to_time_window(
            ds, np.datetime64("2028-02-13T02:00:07"), np.datetime64("2028-02-13T02:00:07")
        )
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [7, 8, 9])
        np.testing.assert_array_equal(out["CAMERA_PACKET_INDEX"].values, [0])


class TestFindSampleDims:
    """Sample-axis identification."""

    def test_finds_datetime_sample_dim(self):
        assert find_sample_dims(_sample_product()) == {"RAD_SAMPLE_FPE_TIME"}

    def test_packet_only_product_has_none(self):
        assert find_sample_dims(_packet_only_product()) == set()

    def test_array_dimension_is_not_a_sample_dim(self):
        assert "ARRAY_8" not in find_sample_dims(_packet_only_product())

    def test_found_via_packet_index_when_times_are_not_decoded(self):
        """decode_times=False leaves sample coords as integers; the index variable still names them."""
        ds = _sample_product()
        ds = ds.assign_coords(RAD_SAMPLE_FPE_TIME=("RAD_SAMPLE_FPE_TIME", np.arange(ds.sizes["RAD_SAMPLE_FPE_TIME"])))
        assert find_sample_dims(ds) == {"RAD_SAMPLE_FPE_TIME"}


class TestSampleToPacketIndex:
    """Sample-to-packet mapping and its validation."""

    def test_uses_stored_index(self):
        ds = _sample_product(n_packets=3, samples_per_packet=2)
        np.testing.assert_array_equal(sample_to_packet_index(ds, "RAD_SAMPLE_FPE_TIME"), [0, 0, 1, 1, 2, 2])

    def test_falls_back_to_positional_without_index_variable(self):
        ds = _sample_product(n_packets=3, samples_per_packet=2, with_packet_index=False)
        np.testing.assert_array_equal(sample_to_packet_index(ds, "RAD_SAMPLE_FPE_TIME"), [0, 0, 1, 1, 2, 2])

    def test_out_of_range_index_raises(self):
        ds = _sample_product(n_packets=3, samples_per_packet=2)
        ds["RAD_SAMPLE_packet_index"].values[-1] = 99
        with pytest.raises(ValueError, match="outside the range of the PACKET axis"):
            sample_to_packet_index(ds, "RAD_SAMPLE_FPE_TIME")

    def test_interleaved_index_is_returned_as_stored(self):
        """Clock skew between adjacent packets interleaves their sample blocks; that is not an error."""
        ds = _sample_product(n_packets=3, samples_per_packet=2)
        ds["RAD_SAMPLE_packet_index"].values[:] = [0, 0, 1, 2, 1, 2]
        np.testing.assert_array_equal(sample_to_packet_index(ds, "RAD_SAMPLE_FPE_TIME"), [0, 0, 1, 2, 1, 2])

    def test_inexact_ratio_without_index_variable_raises(self):
        ds = _sample_product(n_packets=3, samples_per_packet=2, with_packet_index=False)
        ds = ds.isel(RAD_SAMPLE_FPE_TIME=slice(0, 5))
        with pytest.raises(ValueError, match="not an exact multiple"):
            sample_to_packet_index(ds, "RAD_SAMPLE_FPE_TIME")


class TestSelectPackets:
    """Packet-driven subsetting and packet_index renumbering."""

    def test_keeps_whole_packets_and_renumbers_from_zero(self):
        ds = _sample_product(n_packets=6, samples_per_packet=4)
        out = select_packets(ds, np.array([2, 3, 4]))

        assert out.sizes["PACKET"] == 3
        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 12
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, np.repeat(np.arange(3), 4))
        assert out["RAD_SAMPLE_packet_index"].dtype == np.int64
        # Sample values follow their packets
        np.testing.assert_array_equal(out["ICIE__RAD_SAMPLE_0"].values, np.arange(8, 20, dtype=np.float32))

    def test_accepts_a_slice(self):
        ds = _sample_product(n_packets=6, samples_per_packet=4)
        out = select_packets(ds, slice(1, 4))
        assert out.sizes["PACKET"] == 3
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, np.repeat(np.arange(3), 4))

    def test_accepts_a_boolean_mask(self):
        ds = _sample_product(n_packets=4, samples_per_packet=2)
        out = select_packets(ds, np.array([True, False, True, False]))
        assert out.sizes["PACKET"] == 2
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [0, 2])
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, [0, 0, 1, 1])

    def test_non_contiguous_selection_renumbers_contiguously(self):
        ds = _sample_product(n_packets=6, samples_per_packet=2)
        out = select_packets(ds, np.array([0, 3, 5]))
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, [0, 0, 1, 1, 2, 2])

    def test_preserves_packet_order(self):
        """An unsorted indexer must not reorder the packet axis."""
        ds = _sample_product(n_packets=5, samples_per_packet=2)
        out = select_packets(ds, np.array([4, 0, 2]))
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [0, 2, 4])

    def test_empty_selection_returns_empty_dataset(self):
        ds = _sample_product(n_packets=4, samples_per_packet=2)
        out = select_packets(ds, np.zeros(4, dtype=bool))
        assert out.sizes["PACKET"] == 0
        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 0

    def test_packet_only_product_is_subset_normally(self):
        out = select_packets(_packet_only_product(n_packets=5), slice(1, 3))
        assert out.sizes["PACKET"] == 2
        assert out.sizes["ARRAY_8"] == 8

    def test_without_packet_index_variable(self):
        ds = _sample_product(n_packets=4, samples_per_packet=3, with_packet_index=False)
        out = select_packets(ds, np.array([1, 2]))
        assert out.sizes["PACKET"] == 2
        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 6

    def test_missing_packet_dimension_raises(self):
        with pytest.raises(ValueError, match="missing required dimension 'PACKET'"):
            select_packets(xr.Dataset(), slice(0, 1))

    def test_interleaved_packets_keep_their_own_samples(self):
        """A packet whose samples interleave with its neighbour's still takes exactly its own."""
        ds = _interleave_packet_pair(_sample_product(n_packets=6, samples_per_packet=4), 2, 4)
        out = select_packets(ds, np.array([2]))

        assert out.sizes["PACKET"] == 1
        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 4
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, np.zeros(4))
        np.testing.assert_array_equal(out["ICIE__RAD_SAMPLE_0"].values, np.arange(8, 12, dtype=np.float32))

    def test_interleaved_packets_renumber_together(self):
        ds = _interleave_packet_pair(_sample_product(n_packets=6, samples_per_packet=4), 2, 4)
        out = select_packets(ds, np.array([2, 3]))

        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 8
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, [0, 1, 0, 1, 0, 1, 0, 1])

    def test_preserves_packet_index_attributes(self):
        ds = _sample_product(n_packets=4, samples_per_packet=2)
        ds["RAD_SAMPLE_packet_index"].attrs["long_name"] = "Packet index for radiometer sample data"
        out = select_packets(ds, slice(0, 2))
        assert out["RAD_SAMPLE_packet_index"].attrs["long_name"] == "Packet index for radiometer sample data"


class TestSliceL1aDatasetToTimeWindow:
    """Time-window selection semantics."""

    def test_selects_on_sample_time_when_sample_dims_exist(self):
        ds = _sample_product(n_packets=6, samples_per_packet=4)
        t0 = np.datetime64("2028-02-13T02:00:02")
        t1 = np.datetime64("2028-02-13T02:00:03.500")
        out = slice_l1a_dataset_to_time_window(ds, t0, t1)
        assert out.sizes["PACKET"] == 2
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [2, 3])

    def test_keeps_samples_outside_the_window_from_selected_packets(self):
        """Whole packets survive, so an edge packet contributes samples on both sides of t1."""
        ds = _sample_product(n_packets=6, samples_per_packet=4)
        t0 = np.datetime64("2028-02-13T02:00:02")
        t1 = np.datetime64("2028-02-13T02:00:03.250")
        out = slice_l1a_dataset_to_time_window(ds, t0, t1)
        assert out.sizes["PACKET"] == 2
        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 8
        assert out["RAD_SAMPLE_FPE_TIME"].values.max() > t1

    def test_sample_time_selection_differs_from_packet_time_under_skew(self):
        """Under packet/sample clock skew, sample time and packet time select different packets."""
        ds = _sample_product(n_packets=6, samples_per_packet=4, sample_offset_s=3)
        t0 = np.datetime64("2028-02-13T02:00:03")
        t1 = np.datetime64("2028-02-13T02:00:05")
        # Packet time would select packets 3-5, but their samples are skewed 3 s later and fall
        # outside the window; the packets whose *samples* land in it are 0-2.
        packet_times = ds["PACKET_ICIE_TIME"].values
        np.testing.assert_array_equal(ds["SRC_SEQ_CTR"].values[(packet_times >= t0) & (packet_times <= t1)], [3, 4, 5])

        out = slice_l1a_dataset_to_time_window(ds, t0, t1)
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [0, 1, 2])

    def test_interleaving_outside_the_window_does_not_fail_the_slice(self):
        """A clock-skew anomaly elsewhere in the granule must not fail this window."""
        ds = _interleave_packet_pair(_sample_product(n_packets=8, samples_per_packet=4), 0, 4)
        out = slice_l1a_dataset_to_time_window(
            ds, np.datetime64("2028-02-13T02:00:05"), np.datetime64("2028-02-13T02:00:06.500")
        )
        np.testing.assert_array_equal(out["SRC_SEQ_CTR"].values, [5, 6])

    def test_packet_only_product_selects_on_packet_time(self):
        ds = _packet_only_product(n_packets=5)
        out = slice_l1a_dataset_to_time_window(
            ds, np.datetime64("2028-02-13T02:00:01"), np.datetime64("2028-02-13T02:00:03")
        )
        assert out.sizes["PACKET"] == 3

    def test_packet_only_product_missing_time_variable_raises(self):
        ds = _packet_only_product(n_packets=3).drop_vars("PACKET_ICIE_TIME")
        with pytest.raises(ValueError, match="no sample dimensions and is missing"):
            slice_l1a_dataset_to_time_window(
                ds, np.datetime64("2028-02-13T02:00:00"), np.datetime64("2028-02-13T02:00:01")
            )

    def test_window_outside_data_returns_empty(self):
        ds = _sample_product(n_packets=4, samples_per_packet=2)
        out = slice_l1a_dataset_to_time_window(
            ds, np.datetime64("2029-01-01T00:00:00"), np.datetime64("2029-01-02T00:00:00")
        )
        assert out.sizes["PACKET"] == 0

    def test_multiple_sample_groups_take_the_union(self):
        """A packet is kept if any sample group puts it in the window; neither group loses data."""
        ds = _sample_product(n_packets=6, samples_per_packet=2)
        second_dim = "ADCFA_SC_TIME"
        # Second group's clock runs 3 s behind the first
        second_times = ds["RAD_SAMPLE_FPE_TIME"].values - np.timedelta64(3, "s")
        ds = ds.assign_coords({second_dim: (second_dim, second_times)})
        ds["ADCFA_packet_index"] = (second_dim, np.repeat(np.arange(6, dtype=np.int64), 2))
        ds["ADCFA_VALUE"] = (second_dim, np.arange(12, dtype=np.float32))

        t0 = np.datetime64("2028-02-13T02:00:04")
        t1 = np.datetime64("2028-02-13T02:00:05")
        out = slice_l1a_dataset_to_time_window(ds, t0, t1)

        # Group one contributes packets 4-5, group two (3 s behind) contributes none in range,
        # but the union keeps both groups' rows for every selected packet.
        assert out.sizes["PACKET"] == 2
        assert out.sizes["RAD_SAMPLE_FPE_TIME"] == 4
        assert out.sizes[second_dim] == 4
        np.testing.assert_array_equal(out["RAD_SAMPLE_packet_index"].values, [0, 0, 1, 1])
        np.testing.assert_array_equal(out["ADCFA_packet_index"].values, [0, 0, 1, 1])

    def test_non_contiguous_selection_warns(self, caplog):
        ds = _sample_product(n_packets=6, samples_per_packet=2)
        # Drop the middle packets' samples out of the window by pushing their times far forward
        skewed = ds["RAD_SAMPLE_FPE_TIME"].values.copy()
        skewed[4:8] += np.timedelta64(1, "h")
        ds = ds.assign_coords(RAD_SAMPLE_FPE_TIME=("RAD_SAMPLE_FPE_TIME", skewed))

        with caplog.at_level("WARNING"):
            out = slice_l1a_dataset_to_time_window(
                ds, np.datetime64("2028-02-13T02:00:00"), np.datetime64("2028-02-13T02:00:06")
            )

        assert out.sizes["PACKET"] == 4
        assert any("disjoint runs" in record.message for record in caplog.records)
