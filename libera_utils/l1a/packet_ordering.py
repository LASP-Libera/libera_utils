"""Order of the L1A ``PACKET`` axis by ``SRC_SEQ_CTR``, and its check against packet time.

``SRC_SEQ_CTR`` is the packet sequence counter: it advances by one for each packet of an APID and
wraps from 16383 back to 0. The order the instrument built its packets in is the counter's order once
its wraps are counted. Packet time is not an order: it can step backward while the counter advances.
It is used only to tell which wrap of the counter a packet falls in.
"""

import logging
import warnings

import numpy as np
import xarray as xr

from libera_utils.l1a.packet_slicing import PACKET_DIM

logger = logging.getLogger(__name__)

SEQUENCE_COUNTER_VAR = "SRC_SEQ_CTR"

# CCSDS 133.0-B-2, Space Packet Protocol: the Packet Sequence Count is 14 bits and wraps to 0.
SEQUENCE_COUNTER_MODULUS = 1 << 14

#: Packets in packet-time order are cut into segments wherever packet time jumps by more than this,
#: either way: across such a gap the counter may have wrapped unseen, or restarted. Within a segment a
#: counter step is read modulo the counter, which is unambiguous only while neighbours in packet time are
#: fewer than half the modulus (8,192 packets) apart. WFOV SCI (APID 1040), the fastest APID at ~85
#: packets/s in the WFOV captures of the second Day-in-the-Life ground test (DITL2), takes ~96 s to send
#: that many, so this must stay below 96 s. It must also exceed the largest backward packet-time step
#: between neighbours in counter order, or those packets are split into segments ordered by packet time:
#: across the 33 DITL2 WFOV SCI files that step is at most 1.99 s, and across DITL2 NOM-HK at most 3.7 ms.
SEGMENT_GAP = np.timedelta64(60, "s")


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


def _unwrap_sequence_counter(counters: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return a sort key per packet: its ``SRC_SEQ_CTR`` with the counter's wraps counted.

    Packets are taken in packet-time order and cut into segments wherever packet time jumps by more than
    :data:`SEGMENT_GAP`. Within a segment, each packet's key is the key of the packet before it in time plus
    the counter step between them, read modulo the counter as -8192 to 8191, so a packet whose time is
    slightly out of order still lands by its counter. Segments are stacked in packet-time order, each starting
    just above the previous one, at the next key whose counter value is its own lowest packet's counter.

    Parameters
    ----------
    counters : np.ndarray
        ``SRC_SEQ_CTR`` values in axis order.
    times : np.ndarray
        Packet times in axis order, ``datetime64``.

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
    gaps = np.flatnonzero(np.abs(np.diff(times[by_time])) > SEGMENT_GAP) + 1
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
        adjacent in the input with consecutive counters would be placed a counter wrap apart, or, with
        ``ground_data`` False, if packets with the same counter differ.
    """
    counters, times = _counters_and_times(packet_ds, packet_time_coordinate)
    keys = _unwrap_sequence_counter(counters, times)

    # Packets next to each other in a file whose counters differ by one were built one after the other;
    # keys further apart mean packet time put one of them in another wrap.
    wrong_wraps = np.flatnonzero((_signed_counter_steps(counters) == 1) & (np.diff(keys) != 1))
    if wrong_wraps.size:
        first = int(wrong_wraps[0])
        raise ValueError(
            f"{wrong_wraps.size} pair(s) of packets adjacent in the input with consecutive {SEQUENCE_COUNTER_VAR} "
            f"were placed a counter wrap apart, because their packet times differ by more than the counter can "
            f"resolve; first at input position {first + 1}: {SEQUENCE_COUNTER_VAR} {counters[first]} -> "
            f"{counters[first + 1]}, packet time {times[first]} -> {times[first + 1]}. The packet clock is wrong "
            f"for these packets. Check the packet time fields of the input files around those counters."
        )

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    repeats = np.flatnonzero(np.diff(sorted_keys) == 0) + 1
    if repeats.size == 0 and np.array_equal(order, np.arange(order.size)):
        return packet_ds

    if repeats.size:
        run_starts = repeats[np.concatenate(([True], np.diff(repeats) > 1))] - 1
        packet_vars = [name for name, var in packet_ds.data_vars.items() if PACKET_DIM in var.dims]
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
            if verbose:
                warnings.warn(f"{detail} Keeping the first because ground_data=True.")

        message = f"Detected {repeats.size} duplicate {SEQUENCE_COUNTER_VAR} packets; the first of each is kept."
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

    Only steps whose packet times are within :data:`SEGMENT_GAP` of each other count, since across a larger
    gap the counter may have wrapped unseen or restarted. An axis in ``SRC_SEQ_CTR`` order, as
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
    within_gap = np.abs(np.diff(times)) <= SEGMENT_GAP
    return np.flatnonzero(within_gap & (_signed_counter_steps(counters) <= 0))


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
    # TODO[LIBSDC-830]: report each backward step and carry the counts on the granule.
    first = int(backward[0])
    message = (
        f"{packet_time_coordinate} steps backward at {backward.size} position(s) along {SEQUENCE_COUNTER_VAR} "
        f"order, by up to {-time_steps[backward].min()}; first at position {first + 1}: "
        f"{SEQUENCE_COUNTER_VAR} {counters[first]} -> {counters[first + 1]}, {times[first]} -> {times[first + 1]}. "
        f"Packet time disagrees with the counter there; packets are kept in {SEQUENCE_COUNTER_VAR} order."
    )
    warnings.warn(message)
    logger.warning(message)
