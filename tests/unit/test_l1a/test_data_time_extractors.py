"""Unit tests for lightweight data-time extractors."""

from pathlib import Path

import numpy as np
import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import (
    DATA_TIME_INDEXED_APIDS,
    DataTimeUndeterminedError,
    _camera_sop_time_span,
    extract_data_time_range,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.wfov_image_metadata import (
    WFOV_HEADER_SIZE,
    _extract_wfov_header_metadata_from_blob,
    _fsw_timestamps_to_datetime64,
)
from tests.unit.test_l1a.test_wfov_image_metadata import _build_fsw_blob, _encode_fpga_block, _make_wfov_packet_dataset


def test_data_time_indexed_apid_set():
    assert is_data_time_indexed_apid(LiberaApid.icie_wfov_sci)
    assert is_data_time_indexed_apid(LiberaApid.icie_rad_sample)
    assert not is_data_time_indexed_apid(LiberaApid.icie_nom_hk)
    assert LiberaApid.icie_cal_full in DATA_TIME_INDEXED_APIDS


def test_data_time_indexed_apid_unknown_int_returns_false():
    assert not is_data_time_indexed_apid(-1)


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


def test_camera_sop_time_span_returns_none_when_no_sop_in_window():
    """A mem-dump chunk with no SOP packet (SOP landed in an earlier file/pass) is not an error."""
    rows = [("MOP", 0, 10, b"\x00" * 10) for _ in range(3)]
    packet_ds = _make_wfov_packet_dataset(rows)
    assert _camera_sop_time_span(packet_ds) is None


def test_camera_sop_time_span_extracts_valid_sop():
    """A window containing a parseable SOP header returns its FSW image time."""
    header_blob = _build_fsw_blob(2_000_000_000, 123456) + _encode_fpga_block()
    rows = [("SOP", 0, len(header_blob), header_blob)]
    packet_ds = _make_wfov_packet_dataset(rows)
    result = _camera_sop_time_span(packet_ds)
    assert result is not None
    first, last = result
    expected = _fsw_timestamps_to_datetime64(2_000_000_000, 123456)
    assert first == last == expected


def test_extract_data_time_range_returns_none_when_no_sop_in_window(monkeypatch):
    """extract_data_time_range surfaces the no-SOP case as None, not DataTimeUndeterminedError."""
    from libera_utils.l1a import data_time_extractors as mod

    rows = [("MOP", 0, 10, b"\x00" * 10)]
    fake_ds = _make_wfov_packet_dataset(rows)
    monkeypatch.setattr(mod, "parse_packets_to_dataset", lambda *a, **k: fake_ds)

    result = mod.extract_data_time_range("dummy.pds", int(LiberaApid.icie_wfov_sci))
    assert result is None
