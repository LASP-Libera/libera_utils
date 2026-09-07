"""Unit tests for lightweight data-time extractors."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import (
    DATA_TIME_INDEXED_APIDS,
    DataTimeUndeterminedError,
    _camera_sop_time_span,
    _sample_group_time_span,
    extract_data_time_range,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.wfov_image_metadata import (
    WFOV_HEADER_SIZE,
    _extract_wfov_header_metadata_from_blob,
    _fsw_timestamps_to_datetime64,
)
from libera_utils.time import CCSDS_EPOCH
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


# --- sample-group time spans (RAD/CAL epoch+period, AXIS per-sample) ---------------------

CCSDS_EPOCH_US = np.datetime64(CCSDS_EPOCH.isoformat(), "us")
RAD_SAMPLE_PERIOD_US = 5_000  # icie_rad_sample: 50 samples at 5 ms
RAD_SAMPLE_SPAN_US = 49 * RAD_SAMPLE_PERIOD_US  # epoch -> last sample of the same packet


def _ccsds_seconds(iso: str) -> int:
    """Whole seconds from CCSDS_EPOCH to an ISO timestamp, as FSW reports sample epochs."""
    return int((np.datetime64(iso, "us") - CCSDS_EPOCH_US).astype("timedelta64[s]").astype(np.int64))


def _rad_sample_dataset(epoch_seconds: list[int]) -> xr.Dataset:
    """icie_rad_sample packet dataset carrying one sample epoch per packet."""
    return xr.Dataset(
        {
            "ICIE__RAD_SAMP_START_HI": ("PACKET", np.asarray(epoch_seconds, dtype=np.int64)),
            "ICIE__RAD_SAMP_START_LO": ("PACKET", np.zeros(len(epoch_seconds), dtype=np.int64)),
        }
    )


def _axis_sample_dataset(sample_seconds: list[list[int]]) -> xr.Dataset:
    """icie_axis_sample packet dataset with per-sample time fields, one row per packet.

    ``sample_seconds[p][i]`` is the time of sample ``i`` in packet ``p``. Only the sample
    indices supplied get ``%i`` fields; ``expand_sample_times`` skips the rest.
    """
    n_samples = len(sample_seconds[0])
    data = {}
    for i in range(n_samples):
        data[f"ICIE__AXIS_SAMPLE_TM_SEC{i}"] = ("PACKET", np.asarray([p[i] for p in sample_seconds], dtype=np.int64))
        data[f"ICIE__AXIS_SAMPLE_TM_SUB{i}"] = ("PACKET", np.zeros(len(sample_seconds), dtype=np.int64))
    return xr.Dataset(data)


def test_sample_group_time_span_epoch_period_spans_all_packets():
    """Baseline: the span runs from the first epoch to the last sample of the last packet."""
    epochs = [_ccsds_seconds(f"2025-11-14T10:00:0{n}") for n in range(3)]
    first, last = _sample_group_time_span(_rad_sample_dataset(epochs), LiberaApid.icie_rad_sample)

    assert first == np.datetime64("2025-11-14T10:00:00", "us")
    assert last == np.datetime64("2025-11-14T10:00:02", "us") + np.timedelta64(RAD_SAMPLE_SPAN_US, "us")


def test_sample_group_time_span_drops_pre_floor_epoch_without_collapsing():
    """A pre-time-sync epoch is dropped before the min/max collapse, not after.

    Filtering after the collapse leaves only ``last``, degenerating the span to a single
    point at the end of the data instead of reporting the real first valid epoch.
    """
    epochs = [3] + [_ccsds_seconds(f"2025-11-14T10:00:0{n}") for n in range(3)]
    packet_ds = _rad_sample_dataset(epochs)

    first, last = _sample_group_time_span(packet_ds, LiberaApid.icie_rad_sample)

    assert first == np.datetime64("2025-11-14T10:00:00", "us")
    assert last == np.datetime64("2025-11-14T10:00:02", "us") + np.timedelta64(RAD_SAMPLE_SPAN_US, "us")
    assert first < last


def test_sample_group_time_span_raises_when_every_epoch_is_pre_floor():
    packet_ds = _rad_sample_dataset([1, 2, 3])
    with pytest.raises(DataTimeUndeterminedError, match="No usable sample times"):
        _sample_group_time_span(packet_ds, LiberaApid.icie_rad_sample)


def test_sample_group_time_span_per_sample_path_drops_pre_floor_times():
    """The per-sample (``time_field_patterns``) path applies the same floor."""
    valid = [_ccsds_seconds(f"2025-11-14T10:00:0{n}") for n in (1, 2, 3)]
    packet_ds = _axis_sample_dataset([[3, valid[0], valid[1]], valid])

    first, last = _sample_group_time_span(packet_ds, LiberaApid.icie_axis_sample)

    assert first == np.datetime64("2025-11-14T10:00:01", "us")
    assert last == np.datetime64("2025-11-14T10:00:03", "us")
