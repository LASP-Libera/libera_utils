"""Helpers for scanning multi-APID ground-test CCSDS captures.

Ground files (TVAC/DITL/ISTR/IOV) contain many APIDs in one stream and use an
8-byte record header before each CCSDS primary header. These helpers return only
what File Metadata / L1A combining needs: the full APID list, the known
``LiberaApid`` subset, and per-known-APID packet/data time spans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from os import PathLike

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath
from space_packet_parser.generators.ccsds import ccsds_generator

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import (
    DataTimeUndeterminedError,
    extract_data_time_range_from_dataset,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import DATETIME_USEC_DTYPE, drop_implausible_telemetry_times, parse_packets_to_dataset
from libera_utils.time import dt64_to_utc_datetime, multipart_to_dt64

logger = logging.getLogger(__name__)

# Ground captures prefix each CCSDS primary header with an 8-byte record header.
GROUND_CCSDS_SKIP_HEADER_BYTES = 8


class _PacketTimeSpanUnavailable(Exception):
    """No packet time span can be produced for a known APID, for any reason."""


@dataclass(frozen=True)
class GroundCcsdsTimeSpan:
    """Packet and optional science data time span for one known APID."""

    first_packet_time: datetime
    last_packet_time: datetime
    first_data_time: datetime | None  # set only for DATA_TIME_INDEXED_APIDS
    last_data_time: datetime | None


@dataclass(frozen=True)
class GroundCcsdsScanResult:
    """APID discovery and per-known-APID time spans from a ground CCSDS file.

    ``time_spans`` determines what searchable metadata can be written; the two reason dicts
    map each APID missing from it, or degraded within it, to why.
    """

    all_apids: tuple[int, ...]  # sorted unique, known + unknown
    known_apids: tuple[LiberaApid, ...]  # intersection with LiberaApid
    time_spans: dict[LiberaApid, GroundCcsdsTimeSpan]
    failed_apids: dict[LiberaApid, str]  # no span at all; keys disjoint from time_spans
    degraded_apids: dict[LiberaApid, str]  # packet times only; keys a subset of time_spans


def _is_known_libera_apid(apid: int) -> bool:
    """Return True if ``apid`` is a defined ``LiberaApid`` member value."""
    try:
        LiberaApid(apid)
    except ValueError:
        return False
    return True


def discover_ground_ccsds_apids(
    packet_file: PathLike | str,
    *,
    skip_header_bytes: int = GROUND_CCSDS_SKIP_HEADER_BYTES,
) -> tuple[int, ...]:
    """Return sorted unique APID integers present in a ground CCSDS file.

    Does not validate that ``packet_file`` is ground-format. The caller must have already
    dispatched by filename type, as ``libera_cdk``'s ``record_handler`` does; passing a
    flight PDS file here decodes garbage rather than failing.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a ground CCSDS capture.
    skip_header_bytes : int
        Bytes to skip before each CCSDS primary header (``GROUND_CCSDS_SKIP_HEADER_BYTES``).

    Returns
    -------
    tuple[int, ...]
        Sorted unique APID values (known and unknown).
    """
    path = AnyPath(packet_file)
    apids: set[int] = set()
    with path.open("rb") as handle:
        for pkt in ccsds_generator(handle, skip_header_bytes=skip_header_bytes):
            apids.add(int(pkt.apid))
    return tuple(sorted(apids))


def _parse_known_apid(
    packet_file: PathLike | str,
    apid: LiberaApid,
    *,
    skip_header_bytes: int,
) -> xr.Dataset:
    """Parse one known APID out of a ground capture using its configured XTCE definition.

    Reads the whole file, keeping only ``apid``, so a scan costs one pass per known APID.
    Both the packet-time and data-time spans are derived from the returned dataset.

    Raises ``_PacketTimeSpanUnavailable`` if no usable dataset can be produced.
    """
    try:
        packet_config = get_packet_config(apid)
    except KeyError as exc:
        raise _PacketTimeSpanUnavailable(
            f"No packet configuration for known APID {int(apid)} ({apid.name}); "
            "cannot extract packet times for searchable File Metadata"
        ) from exc

    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    try:
        packet_ds = parse_packets_to_dataset(
            [AnyPath(packet_file)],
            packet_definition_path,
            int(apid),
            skip_header_bytes=skip_header_bytes,
        )
    except Exception as exc:
        raise _PacketTimeSpanUnavailable(
            f"Failed to parse packets for known APID {int(apid)} ({apid.name}) from {packet_file}: {exc}"
        ) from exc

    if packet_ds.sizes.get("PACKET", 0) == 0:
        raise _PacketTimeSpanUnavailable(f"No packets found for known APID {int(apid)} ({apid.name}) in {packet_file}")

    return packet_ds


def _extract_packet_time_span(
    packet_ds: xr.Dataset,
    apid: LiberaApid,
    *,
    packet_file: PathLike | str,
) -> tuple[datetime, datetime]:
    """Return min/max packet times from an already-parsed known-APID dataset.

    ``packet_file`` is used only for log and error context.

    Raises ``_PacketTimeSpanUnavailable`` for every reason no span can be produced.
    """
    packet_config = get_packet_config(apid)
    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype(DATETIME_USEC_DTYPE)
    try:
        packet_times_us = drop_implausible_telemetry_times(
            packet_times_us, context=f"known APID {int(apid)} ({apid.name}) packet times in {packet_file}"
        )
    except ValueError as exc:
        raise _PacketTimeSpanUnavailable(str(exc)) from exc
    try:
        return dt64_to_utc_datetime(np.min(packet_times_us)), dt64_to_utc_datetime(np.max(packet_times_us))
    except ValueError as exc:
        raise _PacketTimeSpanUnavailable(
            f"Packet times for known APID {int(apid)} ({apid.name}) in {packet_file} are not representable: {exc}"
        ) from exc


def scan_ground_ccsds_file(
    packet_file: PathLike | str,
    *,
    skip_header_bytes: int = GROUND_CCSDS_SKIP_HEADER_BYTES,
) -> GroundCcsdsScanResult:
    """Scan a ground CCSDS file for APIDs and per-known-APID time spans.

    Unknown APIDs (not in ``LiberaApid``) appear only in ``all_apids``. See
    ``GroundCcsdsScanResult`` for how ``time_spans``, ``failed_apids``, and
    ``degraded_apids`` partition the known APIDs.

    Per-APID failures are isolated: a packet-time failure records the APID in
    ``failed_apids`` and scanning continues. Only a failure to discover APIDs at all
    (an unreadable or unparsable file) fails the whole scan.

    Costs one pass over the file to discover APIDs plus one more per known APID.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a ground CCSDS capture.
    skip_header_bytes : int
        Bytes to skip before each CCSDS primary header (``GROUND_CCSDS_SKIP_HEADER_BYTES``).

    Returns
    -------
    GroundCcsdsScanResult
        Discovery, time-span, and per-APID failure results for File Metadata ingest.
    """
    all_apids = discover_ground_ccsds_apids(packet_file, skip_header_bytes=skip_header_bytes)
    known_apids = tuple(LiberaApid(a) for a in all_apids if _is_known_libera_apid(a))

    time_spans: dict[LiberaApid, GroundCcsdsTimeSpan] = {}
    failed_apids: dict[LiberaApid, str] = {}
    degraded_apids: dict[LiberaApid, str] = {}
    for apid in known_apids:
        try:
            packet_ds = _parse_known_apid(packet_file, apid, skip_header_bytes=skip_header_bytes)
            first_pkt, last_pkt = _extract_packet_time_span(packet_ds, apid, packet_file=packet_file)
        except _PacketTimeSpanUnavailable as exc:
            logger.warning(
                {
                    "msg": "Known APID has no packet time span; no searchable metadata will be written",
                    "apid": int(apid),
                    "libera_apid": apid.name,
                    "file": str(packet_file),
                    "error": str(exc),
                }
            )
            failed_apids[apid] = str(exc)
            continue

        first_data: datetime | None = None
        last_data: datetime | None = None
        if is_data_time_indexed_apid(apid):
            try:
                data_span = extract_data_time_range_from_dataset(packet_ds, int(apid))
            except DataTimeUndeterminedError as exc:
                data_span = None
                reason = str(exc)
            else:
                # A None return means the APID's own no-data-time condition was met, which
                # carries no exception message. Today only WFOV (no SOP in window) returns it.
                reason = f"No in-window data time available for {apid.name}"

            if data_span is None:
                logger.warning(
                    {
                        "msg": "No science data time determined for known APID; recording packet time only",
                        "apid": int(apid),
                        "libera_apid": apid.name,
                        "file": str(packet_file),
                        "error": reason,
                    }
                )
                degraded_apids[apid] = reason
            else:
                first_data, last_data = data_span
                logger.info(
                    {
                        "msg": "Extracted ground CCSDS data time span",
                        "apid": int(apid),
                        "libera_apid": apid.name,
                        "file": str(packet_file),
                        "first_data_time": first_data.isoformat(),
                        "last_data_time": last_data.isoformat(),
                    }
                )

        time_spans[apid] = GroundCcsdsTimeSpan(
            first_packet_time=first_pkt,
            last_packet_time=last_pkt,
            first_data_time=first_data,
            last_data_time=last_data,
        )

    return GroundCcsdsScanResult(
        all_apids=all_apids,
        known_apids=known_apids,
        time_spans=time_spans,
        failed_apids=failed_apids,
        degraded_apids=degraded_apids,
    )
