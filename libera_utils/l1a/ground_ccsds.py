"""Helpers for scanning demuxed ground-test CCSDS files.

Ground captures (TVAC/DITL/ISTR/IOV) are demuxed outside the pipeline into one file per
APID, named ``LIBERA_SDC_<apid>_ccsds_<yyyy>_<doy>_<hh>_<mm>_<ss>`` with the record header
already stripped. These helpers return the packet and science data time spans that File
Metadata ingest needs for one such file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from os import PathLike

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.filenaming import LiberaGroundCcsdsFilename
from libera_utils.l1a.data_time_extractors import (
    DataTimeUndeterminedError,
    extract_data_time_range_from_dataset,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import DATETIME_USEC_DTYPE, drop_implausible_telemetry_times, parse_packets_to_dataset
from libera_utils.time import dt64_to_utc_datetime, multipart_to_dt64

logger = logging.getLogger(__name__)

# Demuxed ground files carry no record header before the CCSDS primary header.
GROUND_CCSDS_SKIP_HEADER_BYTES = 0


@dataclass(frozen=True)
class GroundCcsdsTimeSpan:
    """Packet and optional science data time span for one demuxed ground CCSDS file."""

    apid: LiberaApid
    first_packet_time: datetime
    last_packet_time: datetime
    first_data_time: datetime | None  # set only for DATA_TIME_INDEXED_APIDS
    last_data_time: datetime | None
    # Why data times are absent despite a packet-time span; None when nothing is missing.
    degraded_reason: str | None = None


def _log_no_packet_span(apid: LiberaApid | int, packet_file: PathLike | str, reason: str) -> None:
    """Log why a file yields no packet-time span, and so no searchable File Metadata."""
    libera_apid = apid if isinstance(apid, LiberaApid) else None
    logger.warning(
        {
            "msg": "No packet time span available; no searchable metadata can be written",
            "apid": int(apid),
            "libera_apid": libera_apid.name if libera_apid is not None else None,
            "file": str(packet_file),
            "reason": reason,
        }
    )


def apid_from_ground_ccsds_filename(packet_file: PathLike | str) -> int:
    """Return the APID encoded in a demuxed ground CCSDS basename.

    Raises
    ------
    ValueError
        If the basename is not a valid ``LiberaGroundCcsdsFilename``.
    """
    return LiberaGroundCcsdsFilename(AnyPath(packet_file)).apid


def _parse_ground_ccsds(
    packet_file: PathLike | str,
    apid: LiberaApid,
    *,
    skip_header_bytes: int,
) -> xr.Dataset | None:
    """Parse a demuxed ground CCSDS file using the APID's configured XTCE definition.

    Returns ``None``, having logged the reason, if no usable dataset can be produced.
    """
    try:
        packet_config = get_packet_config(apid)
    except KeyError:
        _log_no_packet_span(
            apid,
            packet_file,
            f"No packet configuration for APID {int(apid)} ({apid.name})",
        )
        return None

    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    try:
        packet_ds = parse_packets_to_dataset(
            [AnyPath(packet_file)],
            packet_definition_path,
            int(apid),
            skip_header_bytes=skip_header_bytes,
        )
    except Exception as exc:
        _log_no_packet_span(apid, packet_file, f"Failed to parse packets: {exc}")
        return None

    if packet_ds.sizes.get("PACKET", 0) == 0:
        _log_no_packet_span(apid, packet_file, "File holds no packets for this APID")
        return None

    return packet_ds


def _extract_packet_time_span(
    packet_ds: xr.Dataset,
    apid: LiberaApid,
    *,
    packet_file: PathLike | str,
) -> tuple[datetime, datetime] | None:
    """Return min/max packet times from an already-parsed dataset.

    ``packet_file`` is used only for log context. Returns ``None``, having logged the reason,
    if no span can be produced.
    """
    packet_config = get_packet_config(apid)
    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype(DATETIME_USEC_DTYPE)
    try:
        packet_times_us = drop_implausible_telemetry_times(
            packet_times_us, context=f"APID {int(apid)} ({apid.name}) packet times in {packet_file}"
        )
    except ValueError as exc:
        _log_no_packet_span(apid, packet_file, str(exc))
        return None
    try:
        return dt64_to_utc_datetime(np.min(packet_times_us)), dt64_to_utc_datetime(np.max(packet_times_us))
    except ValueError as exc:
        _log_no_packet_span(apid, packet_file, f"Packet times are not representable: {exc}")
        return None


def scan_ground_ccsds_file(
    packet_file: PathLike | str,
    *,
    apid: int | None = None,
    skip_header_bytes: int = GROUND_CCSDS_SKIP_HEADER_BYTES,
) -> GroundCcsdsTimeSpan | None:
    """Scan one demuxed ground CCSDS file for its packet and science data time spans.

    A data-time-indexed APID also gets science data times via
    ``extract_data_time_range_from_dataset``. When those cannot be determined the packet-time
    span is still returned, with ``first_data_time``/``last_data_time`` ``None`` and
    ``degraded_reason`` set.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a demuxed ground CCSDS file.
    apid : int | None, optional
        APID of the packets in the file. When ``None``, taken from the basename.
    skip_header_bytes : int
        Bytes to skip before each CCSDS primary header. Demuxed files need ``0``.

    Returns
    -------
    GroundCcsdsTimeSpan | None
        Time spans for File Metadata ingest, or ``None`` if no packet-time span can be produced
        at all: the APID is not a ``LiberaApid`` member, has no packet configuration, the file
        does not parse, holds no packets for the APID, or every packet time is implausible. The
        reason is logged; a caller with nothing to write should surface that None as its own
        error, since no searchable metadata is possible for the file.

    Raises
    ------
    ValueError
        If ``apid`` is omitted and the basename is not a valid ground CCSDS filename.
    """
    resolved_apid = apid_from_ground_ccsds_filename(packet_file) if apid is None else apid
    try:
        libera_apid = LiberaApid(resolved_apid)
    except ValueError:
        _log_no_packet_span(resolved_apid, packet_file, "APID is not a known LiberaApid")
        return None

    packet_ds = _parse_ground_ccsds(packet_file, libera_apid, skip_header_bytes=skip_header_bytes)
    if packet_ds is None:
        return None
    packet_span = _extract_packet_time_span(packet_ds, libera_apid, packet_file=packet_file)
    if packet_span is None:
        return None
    first_pkt, last_pkt = packet_span

    first_data: datetime | None = None
    last_data: datetime | None = None
    degraded_reason: str | None = None
    if is_data_time_indexed_apid(libera_apid):
        try:
            data_span = extract_data_time_range_from_dataset(packet_ds, int(libera_apid))
        except DataTimeUndeterminedError as exc:
            data_span = None
            reason = str(exc)
        else:
            # A None return means the APID's own no-data-time condition was met, which carries
            # no exception message. Today only WFOV (no SOP in window) returns it.
            reason = f"No in-window data time available for {libera_apid.name}"

        if data_span is None:
            degraded_reason = reason
            logger.warning(
                {
                    "msg": "No science data time determined; recording packet time only",
                    "apid": int(libera_apid),
                    "libera_apid": libera_apid.name,
                    "file": str(packet_file),
                    "error": reason,
                }
            )
        else:
            first_data, last_data = data_span
            logger.info(
                {
                    "msg": "Extracted ground CCSDS data time span",
                    "apid": int(libera_apid),
                    "libera_apid": libera_apid.name,
                    "file": str(packet_file),
                    "first_data_time": first_data.isoformat(),
                    "last_data_time": last_data.isoformat(),
                }
            )

    return GroundCcsdsTimeSpan(
        apid=libera_apid,
        first_packet_time=first_pkt,
        last_packet_time=last_pkt,
        first_data_time=first_data,
        last_data_time=last_data,
        degraded_reason=degraded_reason,
    )
