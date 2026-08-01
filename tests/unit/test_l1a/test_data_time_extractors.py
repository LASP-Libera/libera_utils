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
    WFOV_HEADER_SIZE,
    _extract_wfov_header_metadata_from_blob,
    _fsw_timestamps_to_datetime64,
)
from tests.unit.test_l1a.test_wfov_image_metadata import _build_fsw_blob, _encode_fpga_block


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
    """Synthetic full WFOV header (FSW + FPGA) encodes known seconds/subseconds."""
    blob = _build_fsw_blob(2_000_000_000, 123456) + _encode_fpga_block()
    assert len(blob) == WFOV_HEADER_SIZE
    meta = _extract_wfov_header_metadata_from_blob(blob)
    assert meta["timestamp_seconds"] == 2_000_000_000
    assert meta["timestamp_subseconds"] == 123456
    dt = _fsw_timestamps_to_datetime64(meta["timestamp_seconds"], meta["timestamp_subseconds"])
    assert isinstance(dt, np.datetime64)
    assert not np.isnat(dt)
