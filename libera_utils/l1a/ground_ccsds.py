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
from cloudpathlib import AnyPath
from space_packet_parser.generators.ccsds import ccsds_generator

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.l1a.data_time_extractors import (
    DataTimeUndeterminedError,
    extract_data_time_range,
    is_data_time_indexed_apid,
)
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.l1a.packets import DATETIME_USEC_DTYPE, drop_unsynced_clock_times, parse_packets_to_dataset
from libera_utils.time import dt64_to_utc_datetime, multipart_to_dt64

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
    # Known, time-span-eligible APIDs that raised during scanning. A GroundCcsdsApidAbsentError
    # (no packet time) excludes the APID from time_spans entirely; a DataTimeUndeterminedError
    # (no science data time) still leaves a packet-time-only entry in time_spans, so an APID can
    # appear in both dicts at once.
    failed_apids: dict[LiberaApid, str]


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

    This function does not validate that ``packet_file`` is actually ground-format;
    callers are expected to have already dispatched by filename type (as
    ``libera_cdk``'s ``record_handler`` does) before calling this with the ground
    ``skip_header_bytes`` default.

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
    return dt64_to_utc_datetime(np.min(packet_times_us)), dt64_to_utc_datetime(np.max(packet_times_us))


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

    A packet-time failure (``GroundCcsdsApidAbsentError``) drops the whole APID:
    it is recorded in ``failed_apids`` and scanning continues with the remaining
    known APIDs. A science data-time failure for a data-time-indexed APID
    (``extract_data_time_range`` returning ``None`` or raising
    ``DataTimeUndeterminedError``) is narrower: ``first_data_time``/
    ``last_data_time`` are left ``None`` on that APID's ``GroundCcsdsTimeSpan``
    while its packet time is still recorded — only the exception case also adds
    an entry to ``failed_apids``. Only a failure to discover APIDs at all
    (``discover_ground_ccsds_apids``, e.g. an unreadable or unparsable file) is
    a hard failure for the whole scan.

    Like ``discover_ground_ccsds_apids``, this function does not validate that
    ``packet_file`` is actually ground-format; correctness of ``skip_header_bytes``
    depends on the caller having already dispatched by filename type.

    Parameters
    ----------
    packet_file : PathLike | str
        Path to a ground CCSDS capture.
    skip_header_bytes : int
        Bytes to skip before each CCSDS primary header (ground default ``8``).

    Returns
    -------
    GroundCcsdsScanResult
        Discovery, time-span, and per-APID failure results for File Metadata ingest.
    """
    all_apids = discover_ground_ccsds_apids(packet_file, skip_header_bytes=skip_header_bytes)
    known_apids = tuple(LiberaApid(a) for a in all_apids if _is_known_libera_apid(a))

    time_spans: dict[LiberaApid, GroundCcsdsTimeSpan] = {}
    failed_apids: dict[LiberaApid, str] = {}
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

        try:
            first_pkt, last_pkt = _extract_packet_time_span(packet_file, apid, skip_header_bytes=skip_header_bytes)
        except GroundCcsdsApidAbsentError as exc:
            logger.warning(
                {
                    "msg": "Known APID failed packet-time extraction; skipping",
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
                data_span = extract_data_time_range(packet_file, int(apid), skip_header_bytes=skip_header_bytes)
            except DataTimeUndeterminedError as exc:
                logger.warning(
                    {
                        "msg": "Known APID failed science data-time extraction; recording packet time only",
                        "apid": int(apid),
                        "libera_apid": apid.name,
                        "file": str(packet_file),
                        "error": str(exc),
                    }
                )
                failed_apids[apid] = str(exc)
                data_span = None

            if data_span is not None:
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
            else:
                logger.info(
                    {
                        "msg": "No science data time determined for known APID; recording packet time only",
                        "apid": int(apid),
                        "libera_apid": apid.name,
                        "file": str(packet_file),
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
    )
