"""Helpers for scanning multi-APID ground-test CCSDS captures.

Ground files (TVAC/DITL/ISTR/IOV) contain many APIDs in one stream and use an
8-byte record header before each CCSDS primary header. These helpers return only
what File Metadata / L1A combining needs: the full APID list, the known
``LiberaApid`` subset, and per-known-APID packet/data time spans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike

import numpy as np
from cloudpathlib import AnyPath
from space_packet_parser.generators.ccsds import ccsds_generator

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import (
    extract_data_time_range,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import DATETIME_USEC_DTYPE, drop_unsynced_clock_times, parse_packets_to_dataset
from libera_utils.time import multipart_to_dt64

logger = logging.getLogger(__name__)


class GroundCcsdsApidAbsentError(Exception):
    """Raised when a requested known APID is not present in a ground CCSDS file."""


@dataclass(frozen=True)
class GroundCcsdsTimeSpan:
    """Packet and optional science data time span for one known APID."""

    first_packet_time: datetime
    last_packet_time: datetime
    first_data_time: datetime | None  # set only for DATA_TIME_INDEXED_APIDS
    last_data_time: datetime | None


@dataclass(frozen=True)
class GroundCcsdsScanResult:
    """APID discovery and per-known-APID time spans from a ground CCSDS file."""

    all_apids: tuple[int, ...]  # sorted unique, known + unknown
    known_apids: tuple[LiberaApid, ...]  # intersection with LiberaApid
    time_spans: dict[LiberaApid, GroundCcsdsTimeSpan]


def _dt64_to_utc_datetime(value: np.datetime64) -> datetime:
    """Convert numpy datetime64[us] to timezone-aware UTC datetime."""
    if np.isnat(value):
        raise ValueError("Encountered NaT in packet time span")
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=UTC)
    return ts.to_pydatetime().astimezone(UTC)


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
    skip_header_bytes: int = 8,
) -> tuple[int, ...]:
    """Return sorted unique APID integers present in a ground CCSDS file.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a ground CCSDS capture.
    skip_header_bytes : int
        Bytes to skip before each CCSDS primary header (ground default ``8``).

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


def _extract_packet_time_span(
    packet_file: PathLike | str,
    apid: LiberaApid,
    *,
    skip_header_bytes: int,
) -> tuple[datetime, datetime]:
    """Return min/max packet times for a known APID via XTCE parse."""
    try:
        packet_config = get_packet_config(apid)
    except KeyError as exc:
        raise GroundCcsdsApidAbsentError(
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
        raise GroundCcsdsApidAbsentError(
            f"Failed to parse packets for known APID {int(apid)} ({apid.name}) from {packet_file}: {exc}"
        ) from exc

    if packet_ds.sizes.get("PACKET", 0) == 0:
        raise GroundCcsdsApidAbsentError(f"No packets found for known APID {int(apid)} ({apid.name}) in {packet_file}")

    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype(DATETIME_USEC_DTYPE)
    try:
        packet_times_us = drop_unsynced_clock_times(
            packet_times_us, context=f"known APID {int(apid)} ({apid.name}) packet times in {packet_file}"
        )
    except ValueError as exc:
        raise GroundCcsdsApidAbsentError(str(exc)) from exc
    return _dt64_to_utc_datetime(np.min(packet_times_us)), _dt64_to_utc_datetime(np.max(packet_times_us))


def scan_ground_ccsds_file(
    packet_file: PathLike | str,
    *,
    skip_header_bytes: int = 8,
) -> GroundCcsdsScanResult:
    """Scan a ground CCSDS file for APIDs and per-known-APID time spans.

    Unknown APIDs (not in ``LiberaApid``) appear only in ``all_apids``. Known
    APIDs are listed in ``known_apids``. Time spans are produced only for known
    APIDs that have an L1A packet configuration (XTCE + time fields); other
    known APIDs are logged and omitted from ``time_spans`` (no searchable
    metadata can be written without times). Data-time-indexed APIDs also get
    science data-time spans via ``extract_data_time_range``.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a ground CCSDS capture.
    skip_header_bytes : int
        Bytes to skip before each CCSDS primary header (ground default ``8``).

    Returns
    -------
    GroundCcsdsScanResult
        Discovery and time-span results for File Metadata ingest.

    Raises
    ------
    GroundCcsdsApidAbsentError
        If a known APID with a packet config cannot be XTCE-parsed for packet times.
    DataTimeUndeterminedError
        If data-time extraction fails for a data-time-indexed known APID.
    """
    all_apids = discover_ground_ccsds_apids(packet_file, skip_header_bytes=skip_header_bytes)
    known_apids = tuple(LiberaApid(a) for a in all_apids if _is_known_libera_apid(a))

    time_spans: dict[LiberaApid, GroundCcsdsTimeSpan] = {}
    for apid in known_apids:
        try:
            get_packet_config(apid)
        except KeyError:
            logger.warning(
                {
                    "msg": "Skipping searchable time span for known APID without packet config",
                    "apid": int(apid),
                    "libera_apid": apid.name,
                    "file": str(packet_file),
                }
            )
            continue

        first_pkt, last_pkt = _extract_packet_time_span(packet_file, apid, skip_header_bytes=skip_header_bytes)

        first_data: datetime | None = None
        last_data: datetime | None = None
        if is_data_time_indexed_apid(apid):
            first_data, last_data = extract_data_time_range(packet_file, int(apid), skip_header_bytes=skip_header_bytes)
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
    )
