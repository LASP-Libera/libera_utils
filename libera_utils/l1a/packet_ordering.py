"""Acquisition ordering for L1A packet axes, cross-checked against ``SRC_SEQ_CTR``.

Instrument packet secondary-header times periodically step backward while ``SRC_SEQ_CTR``
marches on (LIBSDC-830). Sorting the PACKET axis on those times therefore permutes packets away
from the order in which they were acquired. Acquisition order is instead the order the packets
occupy in the byte stream, across files ordered by time; these helpers establish that order and
report how far the telemetry departs from it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import timedelta
from os import PathLike

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath

logger = logging.getLogger(__name__)

# CCSDS packet sequence counters are 14 bits.
SEQUENCE_COUNTER_MODULUS = 1 << 14

# A forward jump larger than this is read as a genuine ordering violation rather than as lost
# packets. Two orders of magnitude above the largest gap measured in DITL2 (21) and one order
# below the modulus, so a real wrap cannot be mistaken for one.
DEFAULT_MAX_FORWARD_GAP = 1000

# A time step outside these bounds means sequence-counter continuity across it is unknowable, so
# the counter check restarts. The break is far below the shortest interval in which the counter
# could wrap (~68 min for RAD at 250 ms per packet, ~4.5 h for NOM-HK at 1 Hz), so a wrap can
# never hide inside a non-break. The jitter tolerance is well outside the largest backward step
# measured (-2.0 s), which must not break a segment: those steps are exactly what this module
# exists to tolerate.
DEFAULT_SEGMENT_BREAK = timedelta(seconds=60)
DEFAULT_JITTER_TOLERANCE = timedelta(seconds=5)

# Packets read from the head of each file to place that file in time. A median over this many
# packets cannot be moved by one unsynced-clock packet.
DEFAULT_PROBE_PACKETS = 64

SEQUENCE_COUNTER_VARIABLE = "SRC_SEQ_CTR"


@dataclass(frozen=True, slots=True)
class PacketOrderDiagnostics:
    """What the acquisition-order check found on one packet axis.

    Attributes
    ----------
    n_packets : int
        Packets on the axis.
    n_time_inversions : int
        Adjacent pairs where packet time decreases along acquisition order. The FSW bug's
        event count.
    max_time_inversion_us : int
        Largest backward step, in microseconds, as a positive magnitude. Zero when there are
        no inversions.
    n_packets_displaced : int
        Rows a stable sort on packet time would have moved. The count of packets the
        correction keeps in place.
    n_missing_packets : int
        Packets absent according to sequence-counter gaps.
    n_sequence_gaps : int
        Distinct gaps those missing packets fall into.
    n_repeated_counters : int
        Adjacent pairs whose sequence counter did not advance.
    n_order_violations : int
        Sequence-counter steps too large to read as lost packets. Nonzero means acquisition
        order is genuinely in doubt.
    n_uncorroborated_gaps : int
        Sequence-counter gaps whose size is not supported by the elapsed packet time. Only
        counted where a nominal packet cadence is known.
    n_segments : int
        Runs of packets over which counter continuity was assumed.
    sequence_counter_available : bool
        False when the axis carries no ``SRC_SEQ_CTR``, in which case every counter-derived
        field is zero and input order was trusted unverified.
    """

    n_packets: int
    n_time_inversions: int
    max_time_inversion_us: int
    n_packets_displaced: int
    n_missing_packets: int
    n_sequence_gaps: int
    n_repeated_counters: int
    n_order_violations: int
    n_uncorroborated_gaps: int
    n_segments: int
    sequence_counter_available: bool

    @property
    def is_nominal(self) -> bool:
        """True when nothing departed from a clean, fully corroborated acquisition order."""
        return (
            self.n_time_inversions == 0
            and self.n_missing_packets == 0
            and self.n_repeated_counters == 0
            and self.n_order_violations == 0
        )


def find_segment_breaks(
    times: np.ndarray,
    *,
    segment_break: timedelta = DEFAULT_SEGMENT_BREAK,
    jitter_tolerance: timedelta = DEFAULT_JITTER_TOLERANCE,
    valid_time_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mark packets that begin a new run of assumed sequence-counter continuity.

    A break is declared where the elapsed time to the previous packet is longer than
    ``segment_break`` (an outage, across which the counter may have wrapped any number of
    times), where it runs backward by more than ``jitter_tolerance`` (beyond the known FSW
    jitter), or where either packet's time is unusable.

    Parameters
    ----------
    times : np.ndarray
        ``datetime64`` packet times in acquisition order.
    segment_break : timedelta, optional
        Forward step at or above which continuity is abandoned.
    jitter_tolerance : timedelta, optional
        Backward step tolerated before continuity is abandoned. Must stay well outside the
        FSW's own backward steps, which are ~2 s at worst.
    valid_time_mask : np.ndarray or None, optional
        Boolean mask of packets whose times are usable. When None, all times are used.

    Returns
    -------
    np.ndarray
        Boolean array, same length as ``times``. Element 0 is always True.
    """
    n = times.size
    breaks = np.zeros(n, dtype=bool)
    if n == 0:
        return breaks
    breaks[0] = True
    if n == 1:
        return breaks

    times_us = times.astype("datetime64[us]").astype(np.int64)
    deltas = np.diff(times_us)
    forward_limit = int(segment_break.total_seconds() * 1_000_000)
    backward_limit = -int(jitter_tolerance.total_seconds() * 1_000_000)
    breaks[1:] = (deltas > forward_limit) | (deltas < backward_limit)

    if valid_time_mask is not None:
        unusable = ~valid_time_mask.astype(bool)
        breaks |= unusable
        breaks[1:] |= unusable[:-1]
        breaks[0] = True
    return breaks


def unwrap_sequence_counter(
    sequence_counter: np.ndarray,
    *,
    segment_break: np.ndarray | None = None,
    modulus: int = SEQUENCE_COUNTER_MODULUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Unwrap a rolling sequence counter into a monotonic count, per segment.

    The unwrapped value is the running sum of ``(ssc[i] - ssc[i-1]) % modulus`` along the input
    axis, restarted at each segment break. A clean rollover ``16383 -> 0`` contributes 1, as
    does any nominal step, and a packet lost across the rollover contributes its true gap. The
    result is non-decreasing by construction, so it verifies the input order rather than
    supplying a new one.

    Parameters
    ----------
    sequence_counter : np.ndarray
        Counter values in acquisition order.
    segment_break : np.ndarray or None, optional
        Boolean array marking packets that start a new segment, from
        :func:`find_segment_breaks`. When None, the whole axis is one segment.
    modulus : int, optional
        Counter modulus. Default 16384 for a 14-bit CCSDS counter.

    Returns
    -------
    unwrapped : np.ndarray
        ``int64`` monotonic count, restarting at 0 in each segment.
    segment_id : np.ndarray
        ``int64`` segment index for each packet.
    """
    counter = np.asarray(sequence_counter).astype(np.int64, copy=False)
    n = counter.size
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    if segment_break is None:
        breaks = np.zeros(n, dtype=bool)
        breaks[0] = True
    else:
        breaks = np.asarray(segment_break, dtype=bool).copy()
        breaks[0] = True

    segment_id = np.cumsum(breaks) - 1
    deltas = np.zeros(n, dtype=np.int64)
    deltas[1:] = (counter[1:] - counter[:-1]) % modulus
    # A break's own step belongs to no segment, so it contributes nothing to the running count.
    deltas[breaks] = 0
    unwrapped = np.cumsum(deltas)
    # Rebase each segment to start at zero.
    starts = np.flatnonzero(breaks)
    unwrapped -= np.repeat(unwrapped[starts], np.diff(np.append(starts, n)))
    return unwrapped, segment_id


def summarize_packet_order(
    times: np.ndarray,
    sequence_counter: np.ndarray | None,
    *,
    segment_break: timedelta = DEFAULT_SEGMENT_BREAK,
    jitter_tolerance: timedelta = DEFAULT_JITTER_TOLERANCE,
    max_forward_gap: int = DEFAULT_MAX_FORWARD_GAP,
    modulus: int = SEQUENCE_COUNTER_MODULUS,
    nominal_cadence: timedelta | None = None,
) -> PacketOrderDiagnostics:
    """Measure how far a packet axis in acquisition order departs from clean telemetry.

    Parameters
    ----------
    times : np.ndarray
        ``datetime64`` packet times in acquisition order.
    sequence_counter : np.ndarray or None
        ``SRC_SEQ_CTR`` values in the same order, or None when the axis carries none.
    segment_break : timedelta, optional
        Passed to :func:`find_segment_breaks`.
    jitter_tolerance : timedelta, optional
        Passed to :func:`find_segment_breaks`.
    max_forward_gap : int, optional
        Counter step above which a step is an ordering violation rather than lost packets.
    modulus : int, optional
        Counter modulus.
    nominal_cadence : timedelta or None, optional
        Expected time between consecutive packets. When given, each counter gap is checked
        against the elapsed time it should account for; gaps time does not support are counted
        in ``n_uncorroborated_gaps``.

    Returns
    -------
    PacketOrderDiagnostics
        Counts describing the axis.
    """
    times_us = np.asarray(times).astype("datetime64[us]").astype(np.int64)
    n = times_us.size
    if n == 0:
        return PacketOrderDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sequence_counter is not None)

    time_deltas = np.diff(times_us) if n > 1 else np.zeros(0, dtype=np.int64)
    inversions = time_deltas < 0
    n_time_inversions = int(inversions.sum())
    max_inversion = int(-time_deltas[inversions].min()) if n_time_inversions else 0

    # Rows a stable sort on packet time would have moved.
    displaced = int((np.argsort(times_us, kind="stable") != np.arange(n)).sum())

    if sequence_counter is None:
        breaks = find_segment_breaks(times, segment_break=segment_break, jitter_tolerance=jitter_tolerance)
        return PacketOrderDiagnostics(
            n_packets=n,
            n_time_inversions=n_time_inversions,
            max_time_inversion_us=max_inversion,
            n_packets_displaced=displaced,
            n_missing_packets=0,
            n_sequence_gaps=0,
            n_repeated_counters=0,
            n_order_violations=0,
            n_uncorroborated_gaps=0,
            n_segments=int(breaks.sum()),
            sequence_counter_available=False,
        )

    breaks = find_segment_breaks(times, segment_break=segment_break, jitter_tolerance=jitter_tolerance)
    counter = np.asarray(sequence_counter).astype(np.int64, copy=False)
    deltas = np.zeros(n, dtype=np.int64)
    deltas[1:] = (counter[1:] - counter[:-1]) % modulus
    # Steps across a segment break carry no information about continuity.
    scored = ~breaks

    repeated = scored & (deltas == 0)
    gaps = scored & (deltas >= 2) & (deltas <= max_forward_gap)
    violations = scored & (deltas > max_forward_gap)
    n_missing = int((deltas[gaps] - 1).sum())

    n_uncorroborated = 0
    if nominal_cadence is not None and gaps.any():
        cadence_us = int(nominal_cadence.total_seconds() * 1_000_000)
        if cadence_us > 0:
            gap_positions = np.flatnonzero(gaps)
            elapsed = times_us[gap_positions] - times_us[gap_positions - 1]
            expected = deltas[gap_positions] * cadence_us
            # Half a cadence of slack each way absorbs normal packet-time jitter.
            n_uncorroborated = int((np.abs(elapsed - expected) > cadence_us // 2).sum())

    return PacketOrderDiagnostics(
        n_packets=n,
        n_time_inversions=n_time_inversions,
        max_time_inversion_us=max_inversion,
        n_packets_displaced=displaced,
        n_missing_packets=n_missing,
        n_sequence_gaps=int(gaps.sum()),
        n_repeated_counters=int(repeated.sum()),
        n_order_violations=int(violations.sum()),
        n_uncorroborated_gaps=n_uncorroborated,
        n_segments=int(breaks.sum()),
        sequence_counter_available=True,
    )


def _log_packet_order(
    diagnostics: PacketOrderDiagnostics,
    *,
    context: str,
    ground_data: bool,
    verbose: bool,
) -> None:
    """Emit one summary per granule, and per-occurrence detail only under ``verbose``.

    Inversions are reported at every rate they occur at, including zero, so the FSW defect
    stays trended rather than only noticed when it is bad. They are never raised on: nothing is
    dropped and nothing is lost, and the ordering has already been established from the byte
    stream. An ordering violation the sequence counter cannot explain is different, and warns
    loudly.
    """
    detail = {
        "msg": "packet_acquisition_order",
        "context": context,
        "ground_data": ground_data,
        "n_packets": diagnostics.n_packets,
        "n_time_inversions": diagnostics.n_time_inversions,
        "max_time_inversion_us": diagnostics.max_time_inversion_us,
        "n_packets_displaced": diagnostics.n_packets_displaced,
        "n_missing_packets": diagnostics.n_missing_packets,
        "n_sequence_gaps": diagnostics.n_sequence_gaps,
        "n_repeated_counters": diagnostics.n_repeated_counters,
        "n_order_violations": diagnostics.n_order_violations,
        "n_uncorroborated_gaps": diagnostics.n_uncorroborated_gaps,
        "n_segments": diagnostics.n_segments,
        "sequence_counter_available": diagnostics.sequence_counter_available,
    }
    if diagnostics.is_nominal:
        logger.info(detail)
        return

    logger.warning(detail)
    if diagnostics.n_time_inversions:
        fraction = diagnostics.n_time_inversions / diagnostics.n_packets
        warnings.warn(
            f"Packet time steps backward {diagnostics.n_time_inversions} time(s) "
            f"({fraction:.3%} of packets, largest {diagnostics.max_time_inversion_us} us) in "
            f"{context}. The PACKET axis is in acquisition order, so no packet moved; sorting on "
            f"packet time would have displaced {diagnostics.n_packets_displaced}. "
            f"Use verbose=True to see per-occurrence detail.",
            stacklevel=3,
        )
    if diagnostics.n_order_violations:
        warnings.warn(
            f"{diagnostics.n_order_violations} sequence-counter step(s) in {context} are too "
            f"large to be lost packets. Acquisition order for this granule is not corroborated.",
            stacklevel=3,
        )
    if verbose:
        logger.warning(
            {
                "msg": "packet_acquisition_order_detail",
                "context": context,
                **{k: v for k, v in detail.items() if k not in {"msg", "context"}},
            }
        )


def check_packet_acquisition_order(
    packet_ds: xr.Dataset,
    packet_time_coordinate: str,
    *,
    packet_dimension: str = "PACKET",
    ground_data: bool = False,
    verbose: bool = False,
    nominal_cadence: timedelta | None = None,
) -> tuple[xr.Dataset, PacketOrderDiagnostics]:
    """Verify that a packet axis already in acquisition order is self-consistent, and report.

    The dataset is returned unchanged: its row order is the order the packets occupied in the
    byte stream, which is acquisition order. This measures that order against ``SRC_SEQ_CTR``
    and logs the result. An axis with no ``SRC_SEQ_CTR`` is accepted on input order alone, with
    a warning.

    Parameters
    ----------
    packet_ds : xr.Dataset
        Packet dataset with rows in acquisition order.
    packet_time_coordinate : str
        Name of the packet time coordinate.
    packet_dimension : str, optional
        Name of the packet dimension. Default ``"PACKET"``.
    ground_data : bool, optional
        Ground-test mode, recorded on the log record. Default False.
    verbose : bool, optional
        Emit per-occurrence detail in addition to the summary. Default False.
    nominal_cadence : timedelta or None, optional
        Expected time between packets, used to corroborate counter gaps against elapsed time.

    Returns
    -------
    dataset : xr.Dataset
        ``packet_ds``, unchanged.
    diagnostics : PacketOrderDiagnostics
        What the check found.
    """
    times = packet_ds[packet_time_coordinate].values
    sequence_counter = None
    if SEQUENCE_COUNTER_VARIABLE in packet_ds.variables:
        candidate = packet_ds[SEQUENCE_COUNTER_VARIABLE]
        if candidate.dims == (packet_dimension,):
            sequence_counter = candidate.values
        else:
            logger.warning(
                "%s is not on the %s dimension (dims=%s); ordering is trusted from input order "
                "without a sequence-counter cross-check.",
                SEQUENCE_COUNTER_VARIABLE,
                packet_dimension,
                candidate.dims,
            )
    else:
        logger.warning(
            "Packet dataset has no %s variable; the PACKET axis is trusted in input order "
            "without a sequence-counter cross-check.",
            SEQUENCE_COUNTER_VARIABLE,
        )

    diagnostics = summarize_packet_order(times, sequence_counter, nominal_cadence=nominal_cadence)
    _log_packet_order(
        diagnostics,
        context=f"{packet_time_coordinate} ({diagnostics.n_packets} packets)",
        ground_data=ground_data,
        verbose=verbose,
    )
    return packet_ds, diagnostics


def _probe_packet_time_us(
    packet_file: PathLike | str,
    packet_definition,
    apid: int,
    *,
    skip_header_bytes: int,
    n_probe: int,
    multipart_kwargs: dict[str, str],
) -> float | None:
    """Median packet time, in microseconds since the CCSDS epoch, of a file's first packets.

    A median over many packets cannot be moved by one packet whose clock has not yet been
    synced, which a first-packet-wins rule would let misplace a whole file. Returns None when
    no packet of ``apid`` could be read.
    """
    # Deferred so this module can be imported without pulling in the parsing stack.
    import space_packet_parser as spp

    from libera_utils.time import CCSDS_EPOCH

    epoch_us = np.datetime64(CCSDS_EPOCH, "us").astype(np.int64)
    samples: list[int] = []
    try:
        with AnyPath(packet_file).open("rb") as stream:
            for packet_bytes in spp.ccsds_generator(
                stream,
                skip_header_bytes=skip_header_bytes,
                buffer_read_size_bytes=1 << 16,
            ):
                if packet_bytes.apid != apid:
                    continue
                parsed = packet_definition.parse_bytes(packet_bytes)
                microseconds = 0
                for unit, field in multipart_kwargs.items():
                    value = int(parsed[field])
                    if unit == "day_field":
                        microseconds += value * 86_400_000_000
                    elif unit == "s_field":
                        microseconds += value * 1_000_000
                    elif unit == "ms_field":
                        microseconds += value * 1_000
                    else:
                        microseconds += value
                samples.append(microseconds)
                if len(samples) >= n_probe:
                    break
    except Exception as exc:
        logger.warning("Could not probe packet times from %s: %s", packet_file, exc)
        return None

    if not samples:
        logger.warning("No APID %d packets found while probing %s", apid, packet_file)
        return None
    return float(np.median(samples)) + float(epoch_us)


def order_packet_files(
    packet_files: list[PathLike | str],
    packet_definition,
    apid: int,
    *,
    skip_header_bytes: int = 0,
    n_probe: int = DEFAULT_PROBE_PACKETS,
    multipart_kwargs: dict[str, str],
) -> list[PathLike | str]:
    """Order packet files by the time of the data they hold, oldest first.

    Within a file, byte order is acquisition order. Across files it is not, so files must be
    placed in time before their packets are concatenated. Placement uses a coarse median packet
    time rather than a filename: ``L0Filename`` encodes an EDOS creation time and a file number
    within a data set, not a data time, and ``parse_packets_to_l1a_dataset`` accepts arbitrary
    paths. The margin is three orders of magnitude — the timing anomaly this module exists for
    is under 2 s, while files are separated by minutes to hours.

    A file whose head cannot be probed keeps its caller-supplied position and is warned about.

    Parameters
    ----------
    packet_files : list of PathLike or str
        Files to order.
    packet_definition : XtcePacketDefinition
        Loaded XTCE definition used to decode the probed packets.
    apid : int
        APID to probe for.
    skip_header_bytes : int, optional
        Bytes preceding each CCSDS primary header.
    n_probe : int, optional
        Packets to read from each file's head. Default 64.
    multipart_kwargs : dict
        Packet time field mapping, as ``TimeFieldMapping.multipart_kwargs``.

    Returns
    -------
    list
        ``packet_files`` in acquisition order. One file or fewer is returned untouched, with
        no I/O.
    """
    if len(packet_files) <= 1:
        return list(packet_files)

    keyed: list[tuple[float, int, PathLike | str]] = []
    for position, packet_file in enumerate(packet_files):
        probed = _probe_packet_time_us(
            packet_file,
            packet_definition,
            apid,
            skip_header_bytes=skip_header_bytes,
            n_probe=n_probe,
            multipart_kwargs=multipart_kwargs,
        )
        # An unprobeable file sorts by its caller-supplied position, keeping it where it was.
        keyed.append((probed if probed is not None else float("inf"), position, packet_file))

    ordered = [entry[2] for entry in sorted(keyed, key=lambda entry: (entry[0], entry[1]))]
    if [str(f) for f in ordered] != [str(f) for f in packet_files]:
        logger.info(
            {
                "msg": "Reordered packet files into acquisition order",
                "apid": apid,
                "input_order": [str(f) for f in packet_files],
                "acquisition_order": [str(f) for f in ordered],
            }
        )
    return ordered
