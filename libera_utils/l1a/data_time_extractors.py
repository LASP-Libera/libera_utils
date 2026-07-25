"""Lightweight data-time range extraction for L0 packet files.

Used by the Data Ingester to assign applicable dates from science data times
(camera image times, radiometer sample times) without full L1A NetCDF assembly.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from os import PathLike

import numpy as np
from cloudpathlib import AnyPath

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import DATETIME_USEC_DTYPE, drop_unsynced_clock_times, parse_packets_to_dataset
from libera_utils.l1a.wfov_image_metadata import (
    FSW_HEADER_SIZE,
    MEM_DUMP_FLAGS_VAR,
    MEM_DUMP_LENGTH_VAR,
    MEM_DUMP_OFFSET_VAR,
    WFOV_DATA_VAR,
    _fsw_timestamps_to_datetime64,
    extract_fsw_metadata_from_blob,
)
from libera_utils.time import multipart_to_dt64

logger = logging.getLogger(__name__)


class DataTimeUndeterminedError(Exception):
    """Raised when data times cannot be determined from a packet file."""


# APIDs whose File Metadata applicable date should be based on internal data times,
# not Construction Record packet times.
DATA_TIME_INDEXED_APIDS: frozenset[LiberaApid] = frozenset(
    {
        LiberaApid.icie_wfov_sci,
        LiberaApid.icie_rad_sample,
        LiberaApid.icie_rad_full,
        LiberaApid.icie_cal_sample,
        LiberaApid.icie_cal_full,
    }
)


def is_data_time_indexed_apid(apid: LiberaApid | int) -> bool:
    """Return True if applicable-date indexing should use data times for this APID."""
    return LiberaApid(int(apid)) in DATA_TIME_INDEXED_APIDS


def extract_data_time_range(
    packet_file: PathLike | str,
    apid: int,
    *,
    skip_header_bytes: int | None = None,
) -> tuple[datetime, datetime]:
    """Extract the min/max science data time span from a single packet file.

    This is intentionally cheaper than ``parse_packets_to_l1a_dataset``: it uses
    XTCE packet parsing only (no sample-field expansion into L1A product form,
    no WFOV image stitching / NetCDF assembly).

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a PDS or ground CCSDS packet file.
    apid : int
        Application Process Identifier.
    skip_header_bytes : int | None, optional
        Bytes to skip before each CCSDS primary header. When ``None``, uses
        ``SKIP_PACKET_HEADER_BYTES`` from config (default ``0`` for flight PDS;
        pass ``8`` for ground CCSDS).

    Returns
    -------
    tuple[datetime, datetime]
        ``(first_data_time, last_data_time)`` as timezone-aware UTC datetimes.

    Notes
    -----
    Ground CCSDS with an 8-byte record header is handled via ``SKIP_PACKET_HEADER_BYTES``
    (same as ``parse_packets_to_l1a_dataset``) or the ``skip_header_bytes`` argument.

    Raises
    ------
    DataTimeUndeterminedError
        If the APID is not data-time indexed, or no usable data times are found.
    """
    libera_apid = LiberaApid(apid)
    if libera_apid not in DATA_TIME_INDEXED_APIDS:
        raise DataTimeUndeterminedError(
            f"APID {apid} ({libera_apid.name}) is packet-time indexed; data-time extraction is not defined"
        )

    packet_config = get_packet_config(libera_apid)
    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    # Ground test data: set SKIP_PACKET_HEADER_BYTES=8 (see l1a_processing user docs)
    # or pass skip_header_bytes=8 explicitly.
    if skip_header_bytes is None:
        skip_header_bytes = config.get("SKIP_PACKET_HEADER_BYTES")

    try:
        packet_ds = parse_packets_to_dataset(
            [AnyPath(packet_file)],
            packet_definition_path,
            apid,
            skip_header_bytes=skip_header_bytes,
        )
    except Exception as exc:
        raise DataTimeUndeterminedError(f"Failed to parse packets for APID {apid} from {packet_file}: {exc}") from exc

    if packet_ds.sizes.get("PACKET", 0) == 0:
        raise DataTimeUndeterminedError(f"No packets found for APID {apid} in {packet_file}")

    if libera_apid == LiberaApid.icie_wfov_sci:
        first_dt64, last_dt64 = _camera_sop_time_span(packet_ds)
    else:
        first_dt64, last_dt64 = _sample_group_time_span(packet_ds, libera_apid)

    return _dt64_to_utc_datetime(first_dt64), _dt64_to_utc_datetime(last_dt64)


def _dt64_to_utc_datetime(value: np.datetime64) -> datetime:
    """Convert numpy datetime64[us] to timezone-aware UTC datetime."""
    if np.isnat(value):
        raise DataTimeUndeterminedError("Encountered NaT in data time span")
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=UTC)
    return ts.tz_convert("UTC").to_pydatetime()


def _normalize_flag(flag) -> bytes:
    """Normalize MEM_DUMP flag values to ASCII bytes."""
    if isinstance(flag, bytes | np.bytes_):
        return bytes(flag).rstrip(b"\x00")
    if isinstance(flag, str):
        return flag.encode("ascii", errors="ignore")
    return bytes(str(flag), "ascii", errors="ignore")


def _packet_blob_bytes(raw) -> bytes:
    """Convert a packet WFOV data field value to bytes."""
    if isinstance(raw, bytes | np.bytes_):
        return bytes(raw)
    if isinstance(raw, str):
        return raw.encode("latin1")
    return bytes(raw)


def _camera_sop_time_span(packet_ds) -> tuple[np.datetime64, np.datetime64]:
    """Return min/max FSW image times from SOP packets in a WFOV packet dataset."""
    required = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required if name not in packet_ds]
    if missing:
        raise DataTimeUndeterminedError(f"WFOV packet dataset missing required variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values
    offsets = packet_ds[MEM_DUMP_OFFSET_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    packet_data = packet_ds[WFOV_DATA_VAR].values

    camera_times: list[np.datetime64] = []
    for i, flag in enumerate(flags):
        if _normalize_flag(flag) != b"SOP":
            continue
        blob = _packet_blob_bytes(packet_data[i])
        offset = int(offsets[i])
        length = int(lengths[i])
        if length > 0:
            slice_bytes = blob[offset : offset + length]
        else:
            slice_bytes = blob
        if len(slice_bytes) < FSW_HEADER_SIZE:
            logger.warning("SOP packet %s has fewer than %s bytes; skipping", i, FSW_HEADER_SIZE)
            continue
        try:
            meta = extract_fsw_metadata_from_blob(slice_bytes[:FSW_HEADER_SIZE])
            camera_times.append(_fsw_timestamps_to_datetime64(meta["timestamp_seconds"], meta["timestamp_subseconds"]))
        except ValueError as exc:
            logger.warning("Failed to parse FSW header on SOP packet %s: %s", i, exc)
            continue

    if not camera_times:
        raise DataTimeUndeterminedError("No valid SOP FSW image timestamps found in WFOV packet file")

    arr = np.asarray(camera_times, dtype=DATETIME_USEC_DTYPE)
    try:
        arr = drop_unsynced_clock_times(arr, context="WFOV SOP FSW image timestamps")
    except ValueError as exc:
        raise DataTimeUndeterminedError(str(exc)) from exc
    return arr.min(), arr.max()


def _sample_group_time_span(packet_ds, apid: LiberaApid) -> tuple[np.datetime64, np.datetime64]:
    """Return min/max sample times using epoch + period (or per-sample times) from config."""
    packet_config = get_packet_config(apid)
    if not packet_config.sample_groups:
        raise DataTimeUndeterminedError(f"APID {apid} has no sample_groups for data-time extraction")

    all_times: list[np.ndarray] = []
    for group in packet_config.sample_groups:
        if group.epoch_time_fields and group.sample_period:
            epoch_times = multipart_to_dt64(packet_ds, **group.epoch_time_fields.multipart_kwargs)
            epoch_us = epoch_times.values.astype(DATETIME_USEC_DTYPE)
            if epoch_us.size == 0:
                continue
            period_us = np.timedelta64(int(group.sample_period.total_seconds() * 1e6), "us")
            # Span is first epoch through last sample of last packet
            first = epoch_us.min()
            last = epoch_us.max() + (group.sample_count - 1) * period_us
            all_times.append(np.asarray([first, last], dtype=DATETIME_USEC_DTYPE))
        elif group.time_field_patterns:
            from libera_utils.l1a.packets import _expand_sample_times

            sample_times = _expand_sample_times(packet_ds, group.time_field_patterns, group.sample_count)
            if sample_times.size:
                all_times.append(sample_times.astype(DATETIME_USEC_DTYPE))
        else:
            raise DataTimeUndeterminedError(
                f"Sample group {group.name} on APID {apid} has no epoch or per-sample time fields"
            )

    if not all_times:
        raise DataTimeUndeterminedError(f"No sample times found for APID {apid}")

    combined = np.concatenate(all_times)
    try:
        combined = drop_unsynced_clock_times(combined, context=f"APID {apid} ({apid.name}) sample times")
    except ValueError as exc:
        raise DataTimeUndeterminedError(str(exc)) from exc
    return combined.min(), combined.max()
