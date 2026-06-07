"""Unit tests for WFOV image timestamp extraction."""

import struct

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_utils.l1a.wfov_image_time import (
    FSW_HEADER_MIN_SIZE,
    TIMESTAMP_SECONDS_OFFSET,
    TIMESTAMP_SUBSECONDS_OFFSET,
    extract_fsw_timestamps_from_blob,
    extract_wfov_filename_time_bounds,
)
from libera_utils.time import multipart_to_dt64


def _build_fsw_blob(timestamp_seconds: int, timestamp_subseconds: int) -> bytes:
    blob = bytearray(b"\x00" * FSW_HEADER_MIN_SIZE)
    struct.pack_into(">I", blob, TIMESTAMP_SECONDS_OFFSET, timestamp_seconds)
    struct.pack_into(">I", blob, TIMESTAMP_SUBSECONDS_OFFSET, timestamp_subseconds)
    return bytes(blob)


def _expected_datetime64(timestamp_seconds: int, timestamp_subseconds: int) -> np.datetime64:
    dt = multipart_to_dt64(
        {"timestamp_seconds": timestamp_seconds, "timestamp_subseconds": timestamp_subseconds},
        s_field="timestamp_seconds",
        us_field="timestamp_subseconds",
    )
    return np.datetime64(pd.Timestamp(dt).to_datetime64(), "us")


def _make_wfov_packet_dataset(
    rows: list[tuple[str, int, int, bytes]],
) -> xr.Dataset:
    flags = np.array([row[0] for row in rows], dtype="S8")
    offsets = np.array([row[1] for row in rows], dtype=np.uint32)
    lengths = np.array([row[2] for row in rows], dtype=np.uint32)
    data = np.array([row[3] for row in rows], dtype="S972")
    n_packets = len(rows)
    base_time = np.datetime64("2028-01-01T00:00:00", "us")
    packet_times = np.array([base_time + np.timedelta64(i, "s") for i in range(n_packets)], dtype="datetime64[us]")

    return xr.Dataset(
        {
            "ICIE__MEM_DUMP_FLAGS_WFOV": (("PACKET",), flags),
            "ICIE__MEM_DUMP_OFFSET_WFOV": (("PACKET",), offsets),
            "ICIE__MEM_DUMP_LENGTH_WFOV": (("PACKET",), lengths),
            "ICIE__WFOV_DATA": (("PACKET",), data),
        },
        coords={"PACKET_ICIE_TIME": ("PACKET", packet_times)},
    )


class TestExtractFswTimestampsFromBlob:
    def test_extracts_big_endian_fields(self):
        blob = _build_fsw_blob(2212630896, 49631)
        assert extract_fsw_timestamps_from_blob(blob) == (2212630896, 49631)

    def test_rejects_short_blob(self):
        with pytest.raises(ValueError, match="Blob too small"):
            extract_fsw_timestamps_from_blob(b"\x00" * (FSW_HEADER_MIN_SIZE - 1))


class TestExtractWfovFilenameTimeBounds:
    def test_first_and_last_sop_only(self):
        first_blob = _build_fsw_blob(100, 1)
        middle_blob = _build_fsw_blob(200, 2)
        last_blob = _build_fsw_blob(300, 3)
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(first_blob), first_blob.ljust(972, b"\x00")),
                ("MOP", 100, len(middle_blob), middle_blob.ljust(972, b"\x00")),
                ("SOP", 0, len(last_blob), last_blob.ljust(972, b"\x00")),
            ]
        )

        first, last = extract_wfov_filename_time_bounds(ds)
        np.testing.assert_equal(first, _expected_datetime64(100, 1))
        np.testing.assert_equal(last, _expected_datetime64(300, 3))

    def test_single_sop_returns_identical_bounds(self):
        blob = _build_fsw_blob(500, 7)
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), blob.ljust(972, b"\x00"))])
        first, last = extract_wfov_filename_time_bounds(ds)
        np.testing.assert_equal(first, last)

    def test_missing_sop_raises(self):
        blob = _build_fsw_blob(1, 1)
        ds = _make_wfov_packet_dataset([("MOP", 0, len(blob), blob.ljust(972, b"\x00"))])
        with pytest.raises(ValueError, match="No SOP packets"):
            extract_wfov_filename_time_bounds(ds)

    def test_ignores_sop_with_nonzero_offset(self):
        blob = _build_fsw_blob(900, 4)
        ds = _make_wfov_packet_dataset([("SOP", 512, len(blob), blob.ljust(972, b"\x00"))])
        with pytest.raises(ValueError, match="No SOP packets"):
            extract_wfov_filename_time_bounds(ds)
