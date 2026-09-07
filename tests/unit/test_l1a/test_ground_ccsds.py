"""Unit tests for demuxed ground-test CCSDS scanning."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import DataTimeUndeterminedError
from libera_utils.l1a.ground_ccsds import (
    GROUND_CCSDS_SKIP_HEADER_BYTES,
    GroundCcsdsScanError,
    apid_from_ground_ccsds_filename,
    scan_ground_ccsds_file,
)

PACKET_SPAN = (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, 1, tzinfo=UTC))
DATA_SPAN = (datetime(2025, 1, 1, 0, 30, tzinfo=UTC), datetime(2025, 1, 1, 0, 45, tzinfo=UTC))


def test_demuxed_files_need_no_header_skip():
    """Demuxed files start at the CCSDS primary header, unlike raw ground captures."""
    assert GROUND_CCSDS_SKIP_HEADER_BYTES == 0


def test_apid_from_ground_ccsds_filename():
    """The APID is read off the basename rather than discovered by scanning the file."""
    assert apid_from_ground_ccsds_filename("/some/dir/LIBERA_SDC_1057_ccsds_2026_191_14_00_00") == 1057


def test_apid_from_ground_ccsds_filename_rejects_other_naming():
    """A name that is not the demuxed convention is not silently accepted."""
    with pytest.raises(ValueError, match="failed validation"):
        apid_from_ground_ccsds_filename("ccsds_2026_191_14_00_00")


def _patch_scan(monkeypatch, *, packet_span=PACKET_SPAN, data_span=DATA_SPAN, data_exc=None):
    """Stub out parsing and both span extractors on the ground_ccsds module."""
    from libera_utils.l1a import ground_ccsds as mod

    monkeypatch.setattr(mod, "_parse_ground_ccsds", lambda *a, **k: xr.Dataset())
    monkeypatch.setattr(mod, "_extract_packet_time_span", lambda *a, **k: packet_span)

    def _data(*_a, **_k):
        if data_exc is not None:
            raise data_exc
        return data_span

    monkeypatch.setattr(mod, "extract_data_time_range_from_dataset", _data)
    return mod


def test_scan_uses_apid_from_filename(tmp_path: Path, monkeypatch):
    """With no explicit apid, the scan resolves it from the basename."""
    _patch_scan(monkeypatch)
    f = tmp_path / "LIBERA_SDC_1036_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    span = scan_ground_ccsds_file(f)
    assert span.apid == LiberaApid.icie_rad_sample
    assert (span.first_packet_time, span.last_packet_time) == PACKET_SPAN
    assert (span.first_data_time, span.last_data_time) == DATA_SPAN
    assert span.degraded_reason is None


def test_scan_explicit_apid_overrides_filename(tmp_path: Path, monkeypatch):
    """An explicit apid lets a caller scan a file whose name is not the demuxed convention."""
    _patch_scan(monkeypatch)
    f = tmp_path / "some_other_name"
    f.write_bytes(b"")

    assert scan_ground_ccsds_file(f, apid=1036).apid == LiberaApid.icie_rad_sample


def test_scan_packet_time_only_apid_has_no_data_times(tmp_path: Path, monkeypatch):
    """A packet-time-indexed APID gets no data times and is not marked degraded."""
    _patch_scan(monkeypatch)
    f = tmp_path / "LIBERA_SDC_1057_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    span = scan_ground_ccsds_file(f)
    assert span.apid == LiberaApid.icie_nom_hk
    assert span.first_data_time is None
    assert span.last_data_time is None
    assert span.degraded_reason is None


def test_scan_records_packet_time_when_data_time_undetermined(tmp_path: Path, monkeypatch):
    """A data-time failure keeps the packet-time span and records why."""
    _patch_scan(monkeypatch, data_exc=DataTimeUndeterminedError("no usable sample times"))
    f = tmp_path / "LIBERA_SDC_1036_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    span = scan_ground_ccsds_file(f)
    assert (span.first_packet_time, span.last_packet_time) == PACKET_SPAN
    assert span.first_data_time is None
    assert span.degraded_reason == "no usable sample times"


def test_scan_degraded_reason_names_the_apid_that_returned_none(tmp_path: Path, monkeypatch):
    """A None data span records a reason for its own APID, not a hard-coded WFOV message."""
    _patch_scan(monkeypatch, data_span=None)
    f = tmp_path / "LIBERA_SDC_1036_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    span = scan_ground_ccsds_file(f)
    assert LiberaApid.icie_rad_sample.name in span.degraded_reason


def test_scan_rejects_apid_outside_libera_apid(tmp_path: Path):
    """An APID with no LiberaApid member cannot produce searchable metadata."""
    f = tmp_path / "LIBERA_SDC_9_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    with pytest.raises(GroundCcsdsScanError, match="not a known LiberaApid"):
        scan_ground_ccsds_file(f)


def test_scan_raises_when_apid_has_no_packet_config(tmp_path: Path):
    """A known APID with no L1A packet configuration yields no times at all."""
    # 1013 icie_sw_stat is a LiberaApid member with no entry in the L1A processing configs.
    f = tmp_path / "LIBERA_SDC_1013_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    with pytest.raises(GroundCcsdsScanError, match="No packet configuration"):
        scan_ground_ccsds_file(f)


def test_scan_wraps_parse_failure(tmp_path: Path):
    """A file with nothing parseable for its APID fails as a scan error, not an empty span."""
    f = tmp_path / "LIBERA_SDC_1036_ccsds_2025_318_13_00_00"
    f.write_bytes(b"")

    with pytest.raises(GroundCcsdsScanError, match="Failed to parse packets for APID 1036"):
        scan_ground_ccsds_file(f)
