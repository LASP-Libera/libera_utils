"""Unit tests for ground CCSDS APID discovery and time-span helpers."""

from pathlib import Path

import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import DATA_TIME_INDEXED_APIDS, DataTimeUndeterminedError
from libera_utils.l1a.ground_ccsds import (
    GroundCcsdsApidAbsentError,
    discover_ground_ccsds_apids,
    scan_ground_ccsds_file,
)


def test_discover_ground_ccsds_apids_ditl(test_ditl_camera_with_duplicate_packet):
    """DITL fixture exposes known Libera APIDs plus unknowns outside LiberaApid."""
    all_apids = discover_ground_ccsds_apids(test_ditl_camera_with_duplicate_packet, skip_header_bytes=8)
    assert 1040 in all_apids  # icie_wfov_sci
    assert 1036 in all_apids  # icie_rad_sample
    assert 1200 in all_apids  # unknown
    assert 105 in all_apids  # unknown
    # Unknowns must not coerce to LiberaApid
    for unknown in (105, 116, 215, 1006, 1008, 1058, 1200):
        assert unknown in all_apids
        with pytest.raises(ValueError, match=str(unknown)):
            LiberaApid(unknown)


def test_scan_ground_ccsds_file_ditl(test_ditl_camera_with_duplicate_packet):
    """Full scan returns known subset with packet times; cam/rad get data times."""
    result = scan_ground_ccsds_file(test_ditl_camera_with_duplicate_packet, skip_header_bytes=8)

    assert set(result.known_apids).issubset(set(LiberaApid))
    assert all(a in result.all_apids for a in (int(apid) for apid in result.known_apids))
    # Unknowns only on all_apids
    for unknown in (105, 116, 215, 1006, 1008, 1058, 1200):
        assert unknown in result.all_apids
        assert unknown not in {int(a) for a in result.known_apids}

    assert LiberaApid.icie_wfov_sci in result.known_apids
    assert LiberaApid.icie_rad_sample in result.known_apids
    # time_spans is a subset of known_apids (some known APIDs lack packet configs)
    assert set(result.time_spans).issubset(set(result.known_apids))
    assert LiberaApid.icie_wfov_sci in result.time_spans
    assert LiberaApid.icie_rad_sample in result.time_spans
    assert LiberaApid.icie_nom_hk in result.time_spans

    cam_span = result.time_spans[LiberaApid.icie_wfov_sci]
    assert cam_span.first_packet_time <= cam_span.last_packet_time
    assert cam_span.first_data_time is not None
    assert cam_span.last_data_time is not None
    assert cam_span.first_data_time <= cam_span.last_data_time

    rad_span = result.time_spans[LiberaApid.icie_rad_sample]
    assert rad_span.first_data_time is not None
    assert rad_span.last_data_time is not None

    hk_span = result.time_spans[LiberaApid.icie_nom_hk]
    assert hk_span.first_data_time is None
    assert hk_span.last_data_time is None
    assert hk_span.first_packet_time <= hk_span.last_packet_time

    # Data-time indexed APIDs with spans have data times set
    for apid, span in result.time_spans.items():
        if apid in DATA_TIME_INDEXED_APIDS:
            assert span.first_data_time is not None
            assert span.last_data_time is not None
        else:
            assert span.first_data_time is None
            assert span.last_data_time is None


def test_scan_skips_known_apid_without_packet_config(tmp_path: Path, monkeypatch):
    """Known LiberaApid without XTCE/packet config is listed but omitted from time_spans."""
    from libera_utils.l1a import ground_ccsds as mod

    dummy = tmp_path / "ccsds_2025_001_00_00_00"
    dummy.write_bytes(b"")

    # icie_seq_hk is a LiberaApid but has no packet configuration
    monkeypatch.setattr(mod, "discover_ground_ccsds_apids", lambda *a, **k: (1017,))
    result = scan_ground_ccsds_file(dummy, skip_header_bytes=8)
    assert result.known_apids == (LiberaApid.icie_seq_hk,)
    assert result.time_spans == {}


def test_scan_raises_when_known_apid_unparseable(tmp_path: Path, monkeypatch):
    """XTCE parse failure for a configured known APID raises GroundCcsdsApidAbsentError."""
    from libera_utils.l1a import ground_ccsds as mod
    from libera_utils.l1a.l1a_packet_configs import get_packet_config

    dummy = tmp_path / "ccsds_2025_001_00_00_00"
    dummy.write_bytes(b"not-ccsds")

    # Ensure cam APID has a config, then force parse failure
    get_packet_config(LiberaApid.icie_wfov_sci)
    monkeypatch.setattr(mod, "discover_ground_ccsds_apids", lambda *a, **k: (1040,))

    def _boom(*_a, **_k):
        raise ValueError("xtce explode")

    monkeypatch.setattr(mod, "parse_packets_to_dataset", _boom)
    with pytest.raises(GroundCcsdsApidAbsentError, match="1040"):
        scan_ground_ccsds_file(dummy, skip_header_bytes=8)


def test_extract_packet_time_span_drops_unsynced_clock_packet(tmp_path: Path, monkeypatch):
    """A single pre-time-sync (near CCSDS_EPOCH) packet must not poison the APID's packet-time span.

    Regression test for a real ground-test capture where the very first HK packet of a test
    session had a near-zero clock counter (decoding to 1958-01-01), which previously produced a
    first_packet_time 68 years before the real span and blew up the File Metadata day-walk.
    """
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from libera_utils.l1a import ground_ccsds as mod

    dummy = tmp_path / "ccsds_2025_001_00_00_00"
    dummy.write_bytes(b"")

    monkeypatch.setattr(mod, "discover_ground_ccsds_apids", lambda *a, **k: (1057,))
    monkeypatch.setattr(mod, "parse_packets_to_dataset", lambda *a, **k: SimpleNamespace(sizes={"PACKET": 3}))
    monkeypatch.setattr(
        mod,
        "multipart_to_dt64",
        lambda *a, **k: pd.Series(
            np.array(
                ["1958-01-01T00:00:02", "2026-07-10T15:13:56", "2026-07-10T15:22:28"],
                dtype="datetime64[us]",
            )
        ),
    )

    result = scan_ground_ccsds_file(dummy, skip_header_bytes=8)
    span = result.time_spans[LiberaApid.icie_nom_hk]
    assert span.first_packet_time.year == 2026
    assert span.last_packet_time.year == 2026


def test_scan_propagates_data_time_undetermined(tmp_path: Path, monkeypatch):
    """Data-time extract failure for cam/rad propagates DataTimeUndeterminedError."""
    from datetime import UTC, datetime

    from libera_utils.l1a import ground_ccsds as mod

    dummy = tmp_path / "ccsds_2025_001_00_00_00"
    dummy.write_bytes(b"")

    monkeypatch.setattr(mod, "discover_ground_ccsds_apids", lambda *a, **k: (1040,))
    monkeypatch.setattr(
        mod,
        "_extract_packet_time_span",
        lambda *a, **k: (
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 1, tzinfo=UTC),
        ),
    )
    monkeypatch.setattr(
        mod,
        "extract_data_time_range",
        lambda *a, **k: (_ for _ in ()).throw(DataTimeUndeterminedError("no SOP times")),
    )
    with pytest.raises(DataTimeUndeterminedError, match="no SOP times"):
        scan_ground_ccsds_file(dummy, skip_header_bytes=8)
