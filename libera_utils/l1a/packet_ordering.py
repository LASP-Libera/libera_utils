"""Order of the L1A ``PACKET`` axis by ``SRC_SEQ_CTR``, and its check against packet time.

``SRC_SEQ_CTR`` advances by one per packet of an APID and wraps from 16383 to 0, so the order the
instrument built packets in is the order of the counter once its wraps are counted. Packet time is
not an order: it can step backward while the counter advances. It is used only to tell which wrap a
packet falls in.
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

# A packet-time step larger than this, either way, ends a segment: across an outage the counter may
# wrap unseen, and across a reset it restarts. Within a segment a counter step is read as signed modulo
# the counter, which is unambiguous only while neighbours in packet time are fewer than half the modulus
# (8,192 packets) apart. WFOV SCI, the fastest APID at ~85 packets/s in DITL2 WFOV captures, takes ~96 s
# to send that many, so this must stay below 96 s. It must also exceed the largest backward packet-time
# step between neighbours in counter order, or those packets are split into segments placed by time:
# across the 33 DITL2 WFOV SCI files that step is at most 1.99 s, and across DITL2 NOM-HK at most 3.7 ms.
OUTAGE = np.timedelta64(60, "s")


def _order_inputs(packet_ds: xr.Dataset, packet_time_coordinate: str) -> tuple[np.ndarray, np.ndarray]:
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
    """Return a sort key that counts ``SRC_SEQ_CTR`` across its wraps.

    Packets are visited in packet-time order only to pair each with a neighbour fewer than half the
    modulus away in counter terms. Summing the signed counter steps between those neighbours gives each
    packet's counter with its wraps counted, whatever the order of the input. Where packet time steps by
    more than :data:`OUTAGE`, a new segment starts, since the counter may have wrapped unseen or reset.
    A segment whose lowest counter is one more than another segment's highest continues it, as across a
    time sync with no lost packets; where two could continue the same one, the nearer in time does.
    Otherwise segments are placed by their latest packet time, each above
    the keys before it.

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
    sorted_times = times[by_time]

    steps = _signed_counter_steps(sorted_counters)
    breaks = np.flatnonzero(np.abs(np.diff(sorted_times)) > OUTAGE) + 1
    steps[breaks - 1] = 0
    relative = np.concatenate(([0], np.cumsum(steps)))

    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n_packets]))
    lows = np.minimum.reduceat(relative, starts)
    highs = np.maximum.reduceat(relative, starts)
    first_counters = (sorted_counters[starts] - relative[starts] + lows) % SEQUENCE_COUNTER_MODULUS
    last_counters = (first_counters + highs - lows) % SEQUENCE_COUNTER_MODULUS
    latest_times = sorted_times[ends - 1]

    # Link each segment to the one it continues by counter, accepting the links nearest in time first.
    n_segments = starts.size
    links = sorted(
        (abs(sorted_times[starts[later]] - latest_times[earlier]), earlier, later)
        for later in range(n_segments)
        for earlier in range(n_segments)
        if earlier != later and (first_counters[later] - last_counters[earlier]) % SEQUENCE_COUNTER_MODULUS == 1
    )
    successor = np.full(n_segments, -1)
    predecessor = np.full(n_segments, -1)
    for _, earlier, later in links:
        if successor[earlier] != -1 or predecessor[later] != -1:
            continue
        head = earlier
        while predecessor[head] != -1:
            head = predecessor[head]
        if head == later:
            continue
        successor[earlier] = later
        predecessor[later] = earlier

    chains = []
    for head in np.flatnonzero(predecessor == -1):
        chain = [int(head)]
        while successor[chain[-1]] != -1:
            chain.append(int(successor[chain[-1]]))
        chains.append(chain)
    chains.sort(key=lambda chain: max(latest_times[segment] for segment in chain))

    sorted_keys = np.empty(n_packets, dtype=np.int64)
    next_key = None
    for chain in chains:
        for position, segment in enumerate(chain):
            if next_key is None:
                base = int(first_counters[segment])
            elif position > 0:
                base = next_key
            else:
                base = next_key + (int(first_counters[segment]) - next_key) % SEQUENCE_COUNTER_MODULUS
            span = slice(starts[segment], ends[segment])
            sorted_keys[span] = base + relative[span] - lows[segment]
            next_key = base + int(highs[segment] - lows[segment]) + 1

    keys = np.empty(n_packets, dtype=np.int64)
    keys[by_time] = sorted_keys
    return keys


def _warn_on_wrong_wraps(counters: np.ndarray, times: np.ndarray, keys: np.ndarray) -> None:
    """Warn where packets adjacent in the input with consecutive counters got keys a wrap apart.

    Packets next to each other in a file whose counters differ by one were built one after the other.
    When their keys differ by more, packet time placed one of them in another wrap: a packet-time error
    of more than ~96 s at WFOV SCI's rate.
    """
    consecutive = _signed_counter_steps(counters) == 1
    wrong = consecutive & (np.diff(keys) != 1)
    n_wrong = int(np.count_nonzero(wrong))
    if not n_wrong:
        return
    first = int(np.flatnonzero(wrong)[0])
    message = (
        f"{n_wrong} pair(s) of packets adjacent in the input with consecutive {SEQUENCE_COUNTER_VAR} were "
        f"placed a counter wrap apart, because their packet times differ by more than the counter can resolve; "
        f"first at input position {first + 1}: {SEQUENCE_COUNTER_VAR} {counters[first]} -> {counters[first + 1]}, "
        f"packet time {times[first]} -> {times[first + 1]}. The packet clock is wrong for these packets; "
        f"the PACKET axis is ordered by the wrap packet time gives them."
    )
    warnings.warn(message)
    logger.warning(message)


def _sort_by_sequence_counter(
    packet_ds: xr.Dataset, packet_time_coordinate: str, *, ground_data: bool = False, verbose: bool = False
) -> xr.Dataset:
    """Order the ``PACKET`` axis by ``SRC_SEQ_CTR`` and drop repeated packets.

    The key is :func:`_unwrap_sequence_counter`. Packets with the same key are the same packet counter:
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
        If ``packet_ds`` has no ``SRC_SEQ_CTR`` or no ``packet_time_coordinate`` variable, or, with
        ``ground_data`` False, if packets with the same counter differ.
    """
    counters, times = _order_inputs(packet_ds, packet_time_coordinate)
    keys = _unwrap_sequence_counter(counters, times)
    _warn_on_wrong_wraps(counters, times, keys)

    order = np.argsort(keys, kind="stable")
    repeats = np.flatnonzero(np.diff(keys[order]) == 0) + 1
    if repeats.size == 0 and np.array_equal(order, np.arange(order.size)):
        return packet_ds

    if repeats.size:
        run_starts = repeats[np.concatenate(([True], np.diff(repeats) > 1))] - 1
        packet_vars = [name for name, var in packet_ds.data_vars.items() if PACKET_DIM in var.dims]
        for start in run_starts:
            end = start + 1
            while end < order.size and keys[order[end]] == keys[order[start]]:
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

        n_duplicates = int(repeats.size)
        message = f"Detected {n_duplicates} duplicate {SEQUENCE_COUNTER_VAR} packets; the first of each is kept."
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


def check_packet_order(packet_ds: xr.Dataset, packet_time_coordinate: str) -> None:
    """Log how ``SRC_SEQ_CTR`` and packet time step along the ``PACKET`` axis.

    The axis is split into runs wherever packet time steps by more than :data:`OUTAGE` either way.
    Within a run, a counter step of ``k > 1`` counts ``k - 1`` missing packets, including across a
    wrap (``16383 -> 0`` is one step, ``16380 -> 2`` counts 5 missing). A counter that repeats or
    steps backward within a run means the axis is not in ``SRC_SEQ_CTR`` order, and logs a warning.
    Packet time stepping backward along the axis, where packet time disagrees with the counter, emits
    a ``UserWarning`` and logs a warning. Nothing is raised for either, and the dataset is not changed.

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
    counters, times = _order_inputs(packet_ds, packet_time_coordinate)
    if times.size < 2:
        return

    time_steps = np.diff(times)
    counter_steps = _signed_counter_steps(counters)
    in_run = np.abs(time_steps) <= OUTAGE
    out_of_order = in_run & (counter_steps <= 0)
    inversions = time_steps < np.timedelta64(0, "us")

    n_inversions = int(np.count_nonzero(inversions))
    logger.info(
        "Packet order on %r: %d packets in %d run(s), %d missing, %d packet-time inversion(s) of up to %s",
        packet_time_coordinate,
        times.size,
        int(np.count_nonzero(~in_run)) + 1,
        int(np.sum(counter_steps[in_run & (counter_steps > 1)] - 1)),
        n_inversions,
        -time_steps[inversions].min() if n_inversions else np.timedelta64(0, "us"),
    )
    if n_inversions:
        # TODO[LIBSDC-830]: report each inversion and carry the counts on the granule.
        first = int(np.flatnonzero(inversions)[0])
        message = (
            f"{packet_time_coordinate} steps backward at {n_inversions} position(s) along {SEQUENCE_COUNTER_VAR} "
            f"order, by up to {-time_steps[inversions].min()}; first at position {first + 1}: "
            f"{SEQUENCE_COUNTER_VAR} {counters[first]} -> {counters[first + 1]}, {times[first]} -> {times[first + 1]}. "
            f"Packet time disagrees with the counter there; packets are kept in {SEQUENCE_COUNTER_VAR} order."
        )
        warnings.warn(message)
        logger.warning(message)
    n_out_of_order = int(np.count_nonzero(out_of_order))
    if n_out_of_order:
        first = int(np.flatnonzero(out_of_order)[0])
        logger.warning(
            "%s repeats or steps backward at %d position(s) with packet time within %s, so the PACKET axis "
            "is not in %s order; first at position %d: counter %d -> %d, %s -> %s. A decoded product whose "
            "PACKET axis was sorted by packet time shows this; decode its L0 files again.",
            SEQUENCE_COUNTER_VAR,
            n_out_of_order,
            OUTAGE,
            SEQUENCE_COUNTER_VAR,
            first + 1,
            counters[first],
            counters[first + 1],
            times[first],
            times[first + 1],
        )
