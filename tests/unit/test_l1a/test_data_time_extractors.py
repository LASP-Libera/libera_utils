"""Unit tests for lightweight data-time extractors."""

from pathlib import Path

import numpy as np
import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import (
    DATA_TIME_INDEXED_APIDS,
    DataTimeUndeterminedError,
    extract_data_time_range,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.wfov_image_metadata import (
    FSW_HEADER_SIZE,
    _fsw_timestamps_to_datetime64,
    extract_fsw_metadata_from_blob,
)


def test_data_time_indexed_apid_set():
    assert is_data_time_indexed_apid(LiberaApid.icie_wfov_sci)
    assert is_data_time_indexed_apid(LiberaApid.icie_rad_sample)
    assert not is_data_time_indexed_apid(LiberaApid.icie_nom_hk)
    assert LiberaApid.icie_cal_full in DATA_TIME_INDEXED_APIDS


def test_extract_rejects_packet_time_apid(tmp_path: Path):
    dummy = tmp_path / "empty.bin"
    dummy.write_bytes(b"")
    with pytest.raises(DataTimeUndeterminedError, match="packet-time indexed"):
        extract_data_time_range(dummy, LiberaApid.icie_nom_hk)


def test_fsw_timestamp_helpers_roundtrip():
    """Synthetic FSW header encodes known seconds/subseconds."""
    header = bytearray(FSW_HEADER_SIZE)
    # timestamp_seconds at offset 12, timestamp_subseconds at offset 16 (big-endian)
    header[12:16] = (2_000_000_000).to_bytes(4, "big")
    header[16:20] = (123456).to_bytes(4, "big")
    meta = extract_fsw_metadata_from_blob(bytes(header))
    assert meta["timestamp_seconds"] == 2_000_000_000
    assert meta["timestamp_subseconds"] == 123456
    dt = _fsw_timestamps_to_datetime64(meta["timestamp_seconds"], meta["timestamp_subseconds"])
    assert isinstance(dt, np.datetime64)
    assert not np.isnat(dt)
