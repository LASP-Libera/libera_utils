"""WFOV camera image timestamp extraction for L1A filename bounds."""

import struct

import numpy as np
import pandas as pd
import xarray as xr

from libera_utils.time import multipart_to_dt64

FSW_HEADER_MIN_SIZE = 20
TIMESTAMP_SECONDS_OFFSET = 12
TIMESTAMP_SUBSECONDS_OFFSET = 16

FIRST_IMAGE_UTC_TIME_ATTR = "first_image_utc_time"
LAST_IMAGE_UTC_TIME_ATTR = "last_image_utc_time"
WFOV_FILENAME_TIME_VARIABLE = "WFOV_FILENAME_TIME"
WFOV_FILENAME_TIME_DIMENSION = "WFOV_FILENAME_TIME_BOUND"

MEM_DUMP_FLAGS_VAR = "ICIE__MEM_DUMP_FLAGS_WFOV"
MEM_DUMP_OFFSET_VAR = "ICIE__MEM_DUMP_OFFSET_WFOV"
MEM_DUMP_LENGTH_VAR = "ICIE__MEM_DUMP_LENGTH_WFOV"
WFOV_DATA_VAR = "ICIE__WFOV_DATA"


def extract_fsw_timestamps_from_blob(blob_bytes: bytes) -> tuple[int, int]:
    """Extract FSW timestamp fields from the start of a WFOV image blob.

    Byte offsets match ``read_fsw_metadata`` in libera_cam ``metadata_parser.py``.
    """
    if len(blob_bytes) < FSW_HEADER_MIN_SIZE:
        raise ValueError(
            f"Blob too small for FSW timestamp fields: {len(blob_bytes)} bytes (minimum {FSW_HEADER_MIN_SIZE})"
        )

    try:
        timestamp_seconds = struct.unpack(">I", blob_bytes[TIMESTAMP_SECONDS_OFFSET : TIMESTAMP_SECONDS_OFFSET + 4])[0]
        timestamp_subseconds = struct.unpack(
            ">I", blob_bytes[TIMESTAMP_SUBSECONDS_OFFSET : TIMESTAMP_SUBSECONDS_OFFSET + 4]
        )[0]
    except struct.error as e:
        raise ValueError(f"Failed to parse FSW timestamps from blob: {e}") from e

    return timestamp_seconds, timestamp_subseconds


def _fsw_timestamps_to_datetime64(timestamp_seconds: int, timestamp_subseconds: int) -> np.datetime64:
    meta = {"timestamp_seconds": timestamp_seconds, "timestamp_subseconds": timestamp_subseconds}
    dt = multipart_to_dt64(meta, s_field="timestamp_seconds", us_field="timestamp_subseconds")
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    return np.datetime64(pd.Timestamp(dt).to_datetime64(), "us")


def extract_wfov_filename_time_bounds(packet_ds: xr.Dataset) -> tuple[np.datetime64, np.datetime64]:
    """Return first/last image acquisition times from SOP packets in a WFOV L1A packet dataset."""
    required_vars = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required_vars if name not in packet_ds]
    if missing:
        raise ValueError(f"Missing required WFOV variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values.astype(str)
    offsets = packet_ds[MEM_DUMP_OFFSET_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    packet_data = packet_ds[WFOV_DATA_VAR].values

    sop_indices = np.flatnonzero((flags == "SOP") & (offsets == 0))
    if sop_indices.size == 0:
        raise ValueError("No SOP packets with zero offset found in WFOV dataset")

    first_index = int(sop_indices[0])
    last_index = int(sop_indices[-1])

    first_blob = bytes(packet_data[first_index])[: lengths[first_index]]
    last_blob = bytes(packet_data[last_index])[: lengths[last_index]]

    first_seconds, first_subseconds = extract_fsw_timestamps_from_blob(first_blob)
    last_seconds, last_subseconds = extract_fsw_timestamps_from_blob(last_blob)

    return (
        _fsw_timestamps_to_datetime64(first_seconds, first_subseconds),
        _fsw_timestamps_to_datetime64(last_seconds, last_subseconds),
    )
