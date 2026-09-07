"""Lightweight data-time range extraction for L0 packet files.

Used by the Data Ingester to assign applicable dates from science data times
(camera image times, radiometer sample times) without full L1A NetCDF assembly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from os import PathLike

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import (
    DATETIME_USEC_DTYPE,
    drop_implausible_telemetry_times,
    expand_sample_times,
    parse_packets_to_dataset,
)
from libera_utils.l1a.wfov_image_metadata import (
    MEM_DUMP_FLAGS_VAR,
    MEM_DUMP_LENGTH_VAR,
    MEM_DUMP_OFFSET_VAR,
    WFOV_DATA_VAR,
    WFOV_HEADER_SIZE,
    _extract_wfov_header_metadata_from_blob,
    _fsw_timestamps_to_datetime64,
)
from libera_utils.time import dt64_to_utc_datetime, multipart_to_dt64

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
        LiberaApid.icie_axis_sample,
    }
)


def is_data_time_indexed_apid(apid: LiberaApid | int) -> bool:
    """Return True if applicable-date indexing should use data times for this APID.

    Returns False for any int that does not correspond to a known ``LiberaApid`` member,
    rather than raising, so this is safe to use as a predicate over arbitrary APID values.
    """
    try:
        return LiberaApid(int(apid)) in DATA_TIME_INDEXED_APIDS
    except ValueError:
        return False


def extract_data_time_range(
    packet_file: PathLike | str,
    apid: int,
    *,
    skip_header_bytes: int | None = None,
) -> tuple[datetime, datetime] | None:
    """Extract the min/max science data time span from a single packet file.

    Uses XTCE packet parsing only: no sample-field expansion into L1A product form and no
    WFOV image stitching or NetCDF assembly.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a single-APID PDS or demuxed ground CCSDS packet file.
    apid : int
        Application Process Identifier.
    skip_header_bytes : int | None, optional
        Bytes to skip before each CCSDS primary header. When ``None``, uses
        ``SKIP_PACKET_HEADER_BYTES`` from config (default ``0``, which is correct for
        flight PDS and for demuxed ground CCSDS).

    Returns
    -------
    tuple[datetime, datetime] | None
        As ``extract_data_time_range_from_dataset``. A ``None`` return means the caller
        should fall back to packet time, not that extraction failed.

    Raises
    ------
    DataTimeUndeterminedError
        As ``extract_data_time_range_from_dataset``, and additionally if the file cannot be
        parsed or holds no packets for ``apid``.
    """
    libera_apid = LiberaApid(apid)
    if libera_apid not in DATA_TIME_INDEXED_APIDS:
        raise DataTimeUndeterminedError(
            f"APID {apid} ({libera_apid.name}) is packet-time indexed; data-time extraction is not defined"
        )

    packet_config = get_packet_config(libera_apid)
    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
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

    return extract_data_time_range_from_dataset(packet_ds, apid)


def extract_data_time_range_from_dataset(packet_ds: xr.Dataset, apid: int) -> tuple[datetime, datetime] | None:
    """Extract the min/max science data time span from an already-parsed packet dataset.

    ``extract_data_time_range`` is the file-taking wrapper around this.

    Parameters
    ----------
    packet_ds : xr.Dataset
        Single-APID packet dataset, as returned by ``parse_packets_to_dataset``.
    apid : int
        Application Process Identifier the dataset was parsed for.

    Returns
    -------
    tuple[datetime, datetime] | None
        ``(first_data_time, last_data_time)`` as timezone-aware UTC datetimes, or ``None``
        for a WFOV dataset with no ``SOP`` packet to recover an image time from.

    Raises
    ------
    DataTimeUndeterminedError
        If the APID is not data-time indexed, or usable data times cannot be determined for
        a reason other than a missing in-window ``SOP``.
    """
    libera_apid = LiberaApid(apid)
    if libera_apid not in DATA_TIME_INDEXED_APIDS:
        raise DataTimeUndeterminedError(
            f"APID {apid} ({libera_apid.name}) is packet-time indexed; data-time extraction is not defined"
        )

    if libera_apid == LiberaApid.icie_wfov_sci:
        span = _camera_sop_time_span(packet_ds)
        if span is None:
            return None
        first_dt64, last_dt64 = span
    else:
        first_dt64, last_dt64 = _sample_group_time_span(packet_ds, libera_apid)

    try:
        return dt64_to_utc_datetime(first_dt64), dt64_to_utc_datetime(last_dt64)
    except ValueError as exc:
        raise DataTimeUndeterminedError("Encountered NaT in data time span") from exc


def _normalize_flag(flag: bytes | np.bytes_ | str | object) -> bytes:
    """Normalize MEM_DUMP flag values to ASCII bytes."""
    if isinstance(flag, bytes | np.bytes_):
        return bytes(flag).rstrip(b"\x00")
    if isinstance(flag, str):
        return flag.encode("ascii", errors="ignore")
    return bytes(str(flag), "ascii", errors="ignore")


def _camera_sop_time_span(packet_ds: xr.Dataset) -> tuple[np.datetime64, np.datetime64] | None:
    """Return min/max FSW image times from SOP packets in a WFOV packet dataset.

    Returns ``None`` rather than raising when the dataset holds no ``SOP``-flagged packet: a
    mem-dump chunk starting and ending mid-image is expected, not a failure.

    Every ``SOP`` in the window contributes, including one whose image is truncated at the end
    of the file, so this span will not match the L1A product's ``CAMERA_TIME`` range for a
    chunked file — ``CAMERA_TIME`` covers only images completing ``SOP``-to-``EOP``.
    """
    required = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required if name not in packet_ds]
    if missing:
        raise DataTimeUndeterminedError(f"WFOV packet dataset missing required variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    packet_data_var = packet_ds[WFOV_DATA_VAR]
    # Row bytes come from a uint8 view rather than converting a single element of the fixed-width
    # |S array to Python bytes: that conversion silently strips ALL trailing null bytes (not just
    # the dtype's padding), truncating any SOP payload whose real header/footer bytes end in zero.
    # Same caveat documented in wfov_image_metadata.enhance_wfov_l1a_dataset.
    packet_width = packet_data_var.dtype.itemsize
    packet_data_u8 = packet_data_var.values.view(np.uint8).reshape(-1, packet_width)

    sop_present = False
    camera_times: list[np.datetime64] = []
    for i, flag in enumerate(flags):
        if _normalize_flag(flag) != b"SOP":
            continue
        sop_present = True
        blob = packet_data_u8[i].tobytes()
        length = int(lengths[i])
        # MEM_DUMP_OFFSET is the packet's offset within the reassembled image, not into this row,
        # so the payload starts at byte 0 here — matching _stitch_wfov_images.
        slice_bytes = blob[:length] if length > 0 else blob
        if len(slice_bytes) < WFOV_HEADER_SIZE:
            logger.warning("SOP packet %s has fewer than %s bytes; skipping", i, WFOV_HEADER_SIZE)
            continue
        try:
            meta = _extract_wfov_header_metadata_from_blob(slice_bytes[:WFOV_HEADER_SIZE])
            camera_times.append(_fsw_timestamps_to_datetime64(meta["timestamp_seconds"], meta["timestamp_subseconds"]))
        except ValueError as exc:
            logger.warning("Failed to parse FSW header on SOP packet %s: %s", i, exc)
            continue

    if not camera_times:
        if not sop_present:
            logger.info("No SOP packet in WFOV packet file; no in-window image time to extract")
            return None
        raise DataTimeUndeterminedError("No valid SOP FSW image timestamps found in WFOV packet file")

    arr = np.asarray(camera_times, dtype=DATETIME_USEC_DTYPE)
    try:
        arr = drop_implausible_telemetry_times(arr, context="WFOV SOP FSW image timestamps")
    except ValueError as exc:
        raise DataTimeUndeterminedError(str(exc)) from exc
    return arr.min(), arr.max()


def _drop_implausible_group_times(times_us: np.ndarray, apid: LiberaApid, group_name: str, kind: str) -> np.ndarray:
    """Apply the plausibility window to one sample group's times.

    Returns an empty array when nothing survives, so the caller can skip that group; only an
    APID whose every group is empty is a failure.
    """
    if times_us.size == 0:
        return times_us
    try:
        return drop_implausible_telemetry_times(times_us, context=f"APID {int(apid)} ({apid.name}) {group_name} {kind}")
    except ValueError as exc:
        logger.warning("Sample group %s on APID %s yielded no usable times: %s", group_name, int(apid), exc)
        return times_us[:0]


def _sample_group_time_span(packet_ds: xr.Dataset, apid: LiberaApid) -> tuple[np.datetime64, np.datetime64]:
    """Return min/max sample times using epoch + period (or per-sample times) from config.

    Implausible times must be dropped per group, before the min/max collapse below: an
    epoch-and-period group is reduced to its first and last sample here, so filtering
    afterwards would collapse the span to a single point.
    """
    packet_config = get_packet_config(apid)
    if not packet_config.sample_groups:
        raise DataTimeUndeterminedError(f"APID {apid} has no sample_groups for data-time extraction")

    all_times: list[np.ndarray] = []
    for group in packet_config.sample_groups:
        if group.epoch_time_fields and group.sample_period:
            epoch_times = multipart_to_dt64(packet_ds, **group.epoch_time_fields.multipart_kwargs)
            epoch_us = _drop_implausible_group_times(
                epoch_times.values.astype(DATETIME_USEC_DTYPE), apid, group.name, "epochs"
            )
            if epoch_us.size == 0:
                continue
            period_us = np.timedelta64(round(group.sample_period.total_seconds() * 1e6), "us")
            first = epoch_us.min()
            last = epoch_us.max() + (group.sample_count - 1) * period_us
            all_times.append(np.asarray([first, last], dtype=DATETIME_USEC_DTYPE))
        elif group.time_field_patterns:
            sample_times = _drop_implausible_group_times(
                expand_sample_times(packet_ds, group.time_field_patterns, group.sample_count).astype(
                    DATETIME_USEC_DTYPE
                ),
                apid,
                group.name,
                "sample times",
            )
            if sample_times.size:
                all_times.append(sample_times)
        else:
            raise DataTimeUndeterminedError(
                f"Sample group {group.name} on APID {apid} has no epoch or per-sample time fields"
            )

    if not all_times:
        raise DataTimeUndeterminedError(f"No usable sample times found for APID {apid}")

    combined = np.concatenate(all_times)
    return combined.min(), combined.max()
