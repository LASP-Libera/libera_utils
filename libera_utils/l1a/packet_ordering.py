"""Order of the L1A ``PACKET`` axis by ``SRC_SEQ_CTR``, and its check against packet time.

``SRC_SEQ_CTR`` is the packet sequence counter: it advances by one for each packet of an APID and
wraps from 16383 back to 0. The order the instrument built its packets in is the counter's order once
its wraps are counted. Packet time is not an order: it can step backward while the counter advances.
It is used only to tell which wrap of the counter a packet falls in.

:func:`~libera_utils.l1a.packets.parse_packets_to_l1a_dataset` calls :func:`sort_by_sequence_counter` to put
each APID's ``PACKET`` axis in counter order and drop packets received twice, then
:func:`check_packet_order` to warn where packet time steps backward along that order.
:func:`~libera_utils.l1a.nom_hk_trim.find_obsid_runs` calls :func:`out_of_order_positions` and refuses a
product whose axis is not in counter order.

The order is found by taking packets in packet-time order and reading each counter step modulo 16384, so a
packet whose time is slightly out of order still lands by its counter. Where the counter repeats or steps
backward between packets more than :data:`REORDER_WINDOW` apart in packet time, the packets are cut into
segments, and segments are stacked in packet-time order, each starting at the smallest forward counter step
from the one before it. The user guide's packet ordering page describes the rule and its limits.
"""

import logging
import warnings

import numpy as np
import xarray as xr

from libera_utils.l1a.packet_slicing import PACKET_APID_VAR, PACKET_DIM

logger = logging.getLogger(__name__)

SEQUENCE_COUNTER_VAR = "SRC_SEQ_CTR"

# CCSDS 133.0-B-2, Space Packet Protocol: the Packet Sequence Count is 14 bits and wraps to 0.
SEQUENCE_COUNTER_MODULUS = 1 << 14

#: The counter reorders packets only within this window of packet time. Taking packets in packet-time order,
#: a counter step that repeats or goes backward between packets more than this apart cuts them into segments
#: placed by packet time: the counter restarted, more than 8,191 packets were lost, or a packet clock is wrong,
#: and the counter alone cannot tell these apart. A forward step is read as forward across a gap of any length.
#:
#: A change to the window must keep both of these limits:
#:
#: - Lower: it must exceed the largest backward packet-time step between neighbours in counter order, or
#:   those packets are placed by packet time instead of by counter. In the second Day-in-the-Life ground test
#:   (DITL2) that step is at most 1.99 s across the 33 WFOV SCI files and 3.7 ms across NOM-HK.
#: - Upper: a counter restart that resumes within the window is placed by its counter, ahead of the packets
#:   built before it, so a longer window misorders more restarts. Losing 8,192 or more packets within the
#:   window reads as a backward step and is placed by counter, so an APID must send fewer than 8,192 packets
#:   in it: about 819 packets/s at 10 s, against ~85 packets/s for WFOV SCI (APID 1040) in DITL2.
REORDER_WINDOW = np.timedelta64(10, "s")


def _counters_and_times(packet_ds: xr.Dataset, packet_time_coordinate: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``SRC_SEQ_CTR`` values as int64 and the packet times as ``datetime64[us]``.

    Raises
    ------
    ValueError
        If ``packet_ds`` has no ``SRC_SEQ_CTR`` or no ``packet_time_coordinate`` variable.
    """
    for name in (SEQUENCE_COUNTER_VAR, packet_time_coordinate):
        if name not in packet_ds.variables:
            raise ValueError(
                f"Dataset has no {name!r} variable, so the order of its PACKET axis cannot be checked. "
                f"Decode the L0 files with parse_packets_to_l1a_dataset, which writes it."
            )
    counters = packet_ds[SEQUENCE_COUNTER_VAR].values.astype(np.int64)
    times = packet_ds[packet_time_coordinate].values.astype("datetime64[us]")
    return counters, times


def _signed_counter_steps(counters: np.ndarray) -> np.ndarray:
    """Return each step between neighbouring counters, read modulo the counter as -8192 to 8191."""
    half = SEQUENCE_COUNTER_MODULUS // 2
    return (np.diff(counters) + half) % SEQUENCE_COUNTER_MODULUS - half


def _unwrap_sequence_counter(counters: np.ndarray, times: np.ndarray, *, log_prefix: str = "") -> np.ndarray:
    """Return a sort key per packet: its ``SRC_SEQ_CTR`` with the counter's wraps counted.

    Packets are taken in packet-time order and cut into segments wherever the counter repeats or steps
    backward between packets more than :data:`REORDER_WINDOW` apart in packet time. Within a segment, each
    packet's key is the key of the packet before it in time plus the counter step between them, read modulo
    the counter as -8192 to 8191, so a packet whose time is slightly out of order still lands by its counter.
    Segments are stacked in packet-time order, each starting just above the previous one, at the next key
    whose counter value is its own lowest packet's counter. Each cut is logged at INFO with the counters and
    packet times either side of it.

    Parameters
    ----------
    counters : np.ndarray
        ``SRC_SEQ_CTR`` values in axis order.
    times : np.ndarray
        Packet times in axis order, ``datetime64``.
    log_prefix : str, optional
        Text that starts each cut's log message, such as the APID. Default is no prefix.

    Returns
    -------
    np.ndarray
        int64 key per packet in axis order, with ``key % SEQUENCE_COUNTER_MODULUS == counters``. Equal keys
        are the same packet counter; sorting by the key orders the axis by ``SRC_SEQ_CTR``.
    """
    n_packets = counters.size
    if n_packets == 0:
        return np.zeros(0, dtype=np.int64)
    by_time = np.argsort(times, kind="stable")
    sorted_counters = counters[by_time]

    steps = _signed_counter_steps(sorted_counters)
    sorted_times = times[by_time]
    gaps = np.flatnonzero((steps <= 0) & (np.diff(sorted_times) > REORDER_WINDOW)) + 1
    one_second = np.timedelta64(1, "s")
    for cut in gaps:
        apart_s = (sorted_times[cut] - sorted_times[cut - 1]) / one_second
        logger.info(
            f"{log_prefix}{SEQUENCE_COUNTER_VAR} goes from {sorted_counters[cut - 1]} at {sorted_times[cut - 1]} "
            f"to {sorted_counters[cut]} at {sorted_times[cut]}, {apart_s:.1f} s later, beyond REORDER_WINDOW "
            f"({REORDER_WINDOW / one_second:.0f} s). The counter restarted, more than 8,191 packets were lost, or a "
            f"packet clock is wrong; the packets from {sorted_times[cut]} on are placed by packet time."
        )
    steps[gaps - 1] = 0
    offsets = np.concatenate(([0], np.cumsum(steps)))

    starts = np.concatenate(([0], gaps))
    ends = np.concatenate((gaps, [n_packets]))
    lows = np.minimum.reduceat(offsets, starts)
    highs = np.maximum.reduceat(offsets, starts)
    lowest_counters = (sorted_counters[starts] - offsets[starts] + lows) % SEQUENCE_COUNTER_MODULUS

    sorted_keys = np.empty(n_packets, dtype=np.int64)
    next_key = int(lowest_counters[0])
    for start, end, low, high, lowest_counter in zip(starts, ends, lows, highs, lowest_counters, strict=True):
        base = next_key + (int(lowest_counter) - next_key) % SEQUENCE_COUNTER_MODULUS
        sorted_keys[start:end] = base + offsets[start:end] - low
        next_key = base + int(high - low) + 1

    keys = np.empty(n_packets, dtype=np.int64)
    keys[by_time] = sorted_keys
    return keys


def sort_by_sequence_counter(
    packet_ds: xr.Dataset, packet_time_coordinate: str, *, ground_data: bool = False, verbose: bool = False
) -> xr.Dataset:
    """Order the ``PACKET`` axis by ``SRC_SEQ_CTR`` and drop repeated packets.

    The key is each packet's ``SRC_SEQ_CTR`` with its wraps counted, as ``_unwrap_sequence_counter`` returns it. Packets with the same key are the same packet counter:
    when every ``PACKET`` data variable agrees they are one packet received twice, and the first is
    kept. The rows are compared here rather than by ``_drop_duplicates`` because the key is not a
    coordinate of the dataset and the error names the raw counter and packet times.

    ``packet_ds`` is consumed: each ``PACKET`` variable is removed from it as its reordered copy is made,
    so at most one variable is held twice. Use the returned dataset.

    A packet whose packet time would place it a counter wrap away from both of its neighbours in the input,
    whose counters continue its own, has a wrong packet time: it is dropped, with a ``UserWarning`` and a
    logged warning.

    Parameters
    ----------
    packet_ds : xr.Dataset
        Single-APID packet dataset with ``SRC_SEQ_CTR`` and ``packet_time_coordinate`` on ``PACKET``.
    packet_time_coordinate : str
        Name of the packet time coordinate.
    ground_data : bool, optional
        If True, a repeated counter whose packets differ keeps its first packet instead of raising.
        Default is False.
    verbose : bool, optional
        If True and ``ground_data`` is True, warn for each repeated counter whose packets differ.
        Default is False.

    Returns
    -------
    xr.Dataset
        The packets in ``SRC_SEQ_CTR`` order, one per counter.

    Raises
    ------
    ValueError
        If ``packet_ds`` has no ``SRC_SEQ_CTR`` or no ``packet_time_coordinate`` variable, if packets
        adjacent in the input with consecutive counters would be placed a counter wrap apart other than
        around one such dropped packet, or, with ``ground_data`` False, if packets with the same counter
        differ.
    """
    counters, times = _counters_and_times(packet_ds, packet_time_coordinate)
    apid = f"APID {int(packet_ds[PACKET_APID_VAR].values[0])}: " if PACKET_APID_VAR in packet_ds.variables else ""
    keys = _unwrap_sequence_counter(counters, times, log_prefix=apid)

    # Packets next to each other in a file whose counters differ by one were built one after the other;
    # keys further apart mean packet time put one of them in another wrap.
    wrong_wraps = np.flatnonzero((_signed_counter_steps(counters) == 1) & (np.diff(keys) != 1))
    # A packet a wrap away from both of its neighbours is the one whose packet time is wrong.
    bad_times = np.intersect1d(wrong_wraps, wrong_wraps + 1)
    if bad_times.size:
        first = int(bad_times[0])
        message = (
            f"{apid}Dropped {bad_times.size} packet(s) whose packet time puts them a counter wrap away from both "
            f"neighbours in the input, which continue their {SEQUENCE_COUNTER_VAR}; first at input position "
            f"{first + 1}: {SEQUENCE_COUNTER_VAR} {counters[first]} at {times[first]}, between {times[first - 1]} "
            f"and {times[first + 1]}. The packet's time field is wrong, so it cannot be placed."
        )
        warnings.warn(message)
        logger.warning(message)
        # TODO[LIBSDC-830]: carry the count of packets dropped for a bad packet time on the granule.
        packet_ds = packet_ds.isel({PACKET_DIM: np.setdiff1d(np.arange(counters.size), bad_times)})
        counters, times = _counters_and_times(packet_ds, packet_time_coordinate)
        keys = _unwrap_sequence_counter(counters, times, log_prefix=apid)
        wrong_wraps = np.flatnonzero((_signed_counter_steps(counters) == 1) & (np.diff(keys) != 1))
    if wrong_wraps.size:
        first = int(wrong_wraps[0])
        raise ValueError(
            f"{apid}{wrong_wraps.size} pair(s) of packets adjacent in the input with consecutive {SEQUENCE_COUNTER_VAR} "
            f"were placed a counter wrap apart, because their packet times differ by more than the counter can "
            f"resolve; first at input position {first + 1}: {SEQUENCE_COUNTER_VAR} {counters[first]} -> "
            f"{counters[first + 1]}, packet time {times[first]} -> {times[first + 1]}. The packet clock is wrong "
            f"for these packets, so they cannot be placed in counter order and this APID's data cannot be "
            f"processed from these files. Report the APID, the input files and the counters and packet times "
            f"above to the Libera SDC."
        )

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    repeats = np.flatnonzero(np.diff(sorted_keys) == 0) + 1
    if repeats.size == 0 and np.array_equal(order, np.arange(order.size)):
        return packet_ds

    if repeats.size:
        run_starts = repeats[np.concatenate(([True], np.diff(repeats) > 1))] - 1
        packet_vars = [name for name, var in packet_ds.data_vars.items() if PACKET_DIM in var.dims]
        n_differing = 0
        for start in run_starts:
            end = start + 1
            while end < order.size and sorted_keys[end] == sorted_keys[start]:
                end += 1
            rows = order[start:end]
            differing = [
                name
                for name in packet_vars
                if not np.all(packet_ds[name].values[rows] == packet_ds[name].values[rows[0]])
            ]
            if not differing:
                continue
            detail = (
                f"{end - start} packets share {SEQUENCE_COUNTER_VAR} {counters[rows[0]]} (packet times "
                f"{', '.join(str(t) for t in times[rows])}) but differ in {differing}."
            )
            if not ground_data:
                raise ValueError(
                    f"{detail} A repeated counter with different contents in flight data means a counter reset "
                    f"or corrupt L0: check the input files for overlapping captures of different sessions. "
                    f"For a ground capture, pass ground_data=True to keep the first packet."
                )
            same_as_first = np.ones(end - start - 1, dtype=bool)
            for name in differing:
                values = packet_ds[name].values
                same_as_first &= (values[rows[1:]] == values[rows[0]]).reshape(end - start - 1, -1).all(axis=1)
            n_differing += int(np.count_nonzero(~same_as_first))
            if verbose:
                warnings.warn(f"{detail} Keeping the first because ground_data=True.")

        if n_differing:
            message = (
                f"Dropped {repeats.size} packet(s) that repeat a kept packet's {SEQUENCE_COUNTER_VAR}: "
                f"{repeats.size - n_differing} with the same contents, {n_differing} with different contents, "
                f"kept first because ground_data=True."
            )
        else:
            message = (
                f"Dropped {repeats.size} packet(s) received twice: same {SEQUENCE_COUNTER_VAR} and contents as a "
                f"kept packet."
            )
        warnings.warn(message)
        logger.warning(message)

    keep = np.delete(order, repeats)
    coord_names = set(packet_ds.coords)
    moved: dict[str, xr.Variable] = {}
    for name in [name for name, var in packet_ds.variables.items() if PACKET_DIM in var.dims]:
        var = packet_ds.variables[name]
        moved[name] = xr.Variable(
            var.dims, np.take(var.values, keep, axis=var.dims.index(PACKET_DIM)), var.attrs, var.encoding
        )
        del packet_ds[name]
    return xr.Dataset(
        data_vars={name: var for name, var in moved.items() if name not in coord_names} | dict(packet_ds.data_vars),
        coords={name: var for name, var in moved.items() if name in coord_names} | dict(packet_ds.coords),
        attrs=packet_ds.attrs,
    )


def out_of_order_positions(packet_ds: xr.Dataset, packet_time_coordinate: str) -> np.ndarray:
    """Return the ``PACKET`` positions ``i`` where ``SRC_SEQ_CTR`` repeats or steps backward from ``i`` to ``i + 1``.

    Only steps whose packet times are within :data:`REORDER_WINDOW` of each other count, since further
    apart the counter may have restarted. An axis in ``SRC_SEQ_CTR`` order, as
    ``parse_packets_to_l1a_dataset`` writes it, has none.

    Parameters
    ----------
    packet_ds : xr.Dataset
        Dataset with ``SRC_SEQ_CTR`` and ``packet_time_coordinate`` on its ``PACKET`` axis.
    packet_time_coordinate : str
        Name of the packet time variable.

    Returns
    -------
    np.ndarray
        The positions ``i``, in ascending order; empty when the axis is in ``SRC_SEQ_CTR`` order.

    Raises
    ------
    ValueError
        If ``packet_ds`` has no ``SRC_SEQ_CTR`` or no ``packet_time_coordinate`` variable.
    """
    counters, times = _counters_and_times(packet_ds, packet_time_coordinate)
    within_window = np.abs(np.diff(times)) <= REORDER_WINDOW
    return np.flatnonzero(within_window & (_signed_counter_steps(counters) <= 0))


def check_packet_order(packet_ds: xr.Dataset, packet_time_coordinate: str) -> None:
    """Warn where packet time steps backward along the ``PACKET`` axis.

    Along an axis in ``SRC_SEQ_CTR`` order, a backward packet-time step is where packet time disagrees
    with the counter. This emits a ``UserWarning`` and logs a warning with the number of such steps, the
    largest, and the first one's counters and times. Nothing is raised for it, and the dataset is not
    changed.

    Parameters
    ----------
    packet_ds : xr.Dataset
        Dataset with ``SRC_SEQ_CTR`` and ``packet_time_coordinate`` on its ``PACKET`` axis.
    packet_time_coordinate : str
        Name of the packet time variable.

    Raises
    ------
    ValueError
        If ``packet_ds`` has no ``SRC_SEQ_CTR`` or no ``packet_time_coordinate`` variable.
    """
    counters, times = _counters_and_times(packet_ds, packet_time_coordinate)
    time_steps = np.diff(times)
    backward = np.flatnonzero(time_steps < np.timedelta64(0, "us"))
    if not backward.size:
        return
    # TODO[LIBSDC-830]: report each backward step and the missing-packet count (counter steps above 1
    #  between neighbours on the PACKET axis), and carry both counts on the granule.
    first = int(backward[0])
    message = (
        f"{packet_time_coordinate} steps backward at {backward.size} position(s) along {SEQUENCE_COUNTER_VAR} "
        f"order, by up to {-time_steps[backward].min()}; first at position {first + 1}: "
        f"{SEQUENCE_COUNTER_VAR} {counters[first]} -> {counters[first + 1]}, {times[first]} -> {times[first + 1]}. "
        f"Packet time disagrees with the counter there; packets are kept in {SEQUENCE_COUNTER_VAR} order."
    )
    warnings.warn(message)
    logger.warning(message)
