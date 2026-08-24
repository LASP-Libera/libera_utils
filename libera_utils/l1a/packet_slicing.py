"""Subset decoded L1A Datasets along ``PACKET``, carrying sample axes with their packets.

Decoded L1A products carry two kinds of axis: a ``PACKET`` axis holding one row per CCSDS
packet, and one sample axis per sample group (e.g. ``RAD_FULL_FPE_TIME``) holding the samples
expanded out of those packets. Each sample belongs to exactly one packet, and that mapping is
recorded explicitly in the ``{sample_group}_packet_index`` variable written by
:func:`~libera_utils.l1a.packets.create_l1a_dataset`.

Subsetting either axis on its own silently breaks the correspondence: ``PACKET`` shrinks while
the sample axis keeps rows belonging to packets that are no longer present, and the stored
packet indices go on pointing at the pre-subset numbering. Any consumer that needs to slice a
decoded product must therefore go through this module, which treats the packet axis as the
driver:

- packets are selected first, and every sample of a selected packet is kept while every sample
  of an unselected packet is dropped, so whole packets survive or none of them does;
- ``{sample_group}_packet_index`` is renumbered from 0 against the surviving packet axis, which
  keeps it usable as the sample-to-packet validation link in derived products.

The sample axis is stored in sample-time order, which is *almost* but not exactly packet order:
where two adjacent packets' sample clocks skew by less than a sample interval their sample blocks
interleave, and ``{sample_group}_packet_index`` steps backwards at those few positions. Nothing
here assumes packet-major blocks — the index is applied element-wise — so an interleaved run is
sliced correctly and only logged. In particular, a skew anomaly anywhere in a granule must not
fail a slice of a window that does not contain it.

Selecting a time window (:func:`slice_l1a_dataset_to_time_window`) is driven by *sample* time
where sample axes exist, because the samples are the science data being windowed. Whole packets
are still the unit that survives, so a selected packet contributes samples on both sides of the
window boundary.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a.l1a_packet_configs import get_packet_config

logger = logging.getLogger(__name__)

PACKET_DIM = "PACKET"
PACKET_APID_VAR = "PKT_APID"
PACKET_INDEX_SUFFIX = "_packet_index"
DEFAULT_PACKET_TIME_VAR = "PACKET_ICIE_TIME"


def find_sample_dims(dataset: xr.Dataset) -> set[str]:
    """Identify the sample axes of a decoded L1A Dataset.

    A sample axis is any dimension other than ``PACKET`` that either carries its own
    ``datetime64`` dimension coordinate (e.g. ``RAD_FULL_FPE_TIME``) or owns a
    ``*_packet_index`` variable. The second rule matters for Datasets opened with
    ``decode_times=False``, where a real sample coordinate is a plain integer array and the
    dtype rule alone would misclassify the product as packet-only.

    Array-index dimensions such as ``ARRAY_128`` are excluded by both rules.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset.

    Returns
    -------
    set of str
        Names of the sample dimensions.
    """
    sample_dims: set[str] = set()
    for dim in dataset.dims:
        dim_name = str(dim)
        if dim_name == PACKET_DIM:
            continue
        coord = dataset.coords.get(dim_name)
        if coord is not None and np.issubdtype(coord.dtype, np.datetime64):
            sample_dims.add(dim_name)

    for name, variable in dataset.variables.items():
        if not str(name).endswith(PACKET_INDEX_SUFFIX):
            continue
        if len(variable.dims) == 1 and str(variable.dims[0]) != PACKET_DIM:
            sample_dims.add(str(variable.dims[0]))

    return sample_dims


def _packet_index_var_name(dataset: xr.Dataset, sample_dim: str) -> str | None:
    """Return the ``*_packet_index`` variable owned by ``sample_dim``, if any."""
    for name, variable in dataset.variables.items():
        if str(name).endswith(PACKET_INDEX_SUFFIX) and variable.dims == (sample_dim,):
            return str(name)
    return None


def _configured_sample_count(dataset: xr.Dataset, sample_dim: str) -> int | None:
    """Look up the configured samples-per-packet for ``sample_dim`` from the packet config.

    Uses the ``PKT_APID`` variable carried by every decoded L1A product to recover the packet
    configuration, then finds the sample group owning ``sample_dim``. This is the authoritative
    count, independent of the Dataset's current axis lengths.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset.
    sample_dim : str
        Name of the sample dimension.

    Returns
    -------
    int or None
        Configured samples per packet, or ``None`` when the APID or sample group cannot be
        resolved (e.g. a synthetic Dataset with no ``PKT_APID``, or a renamed packet axis).
    """
    if PACKET_APID_VAR not in dataset or dataset.sizes.get(PACKET_DIM, 0) == 0:
        return None
    try:
        apid = LiberaApid(int(dataset[PACKET_APID_VAR].isel({PACKET_DIM: 0}).values))
        packet_config = get_packet_config(apid)
    except (ValueError, KeyError, IndexError):
        logger.debug("Could not resolve a packet configuration for sample dimension %s", sample_dim)
        return None

    for sample_group in packet_config.sample_groups:
        if sample_group.sample_time_dimension == sample_dim:
            return int(sample_group.sample_count)
    return None


def _samples_per_packet(dataset: xr.Dataset, sample_dim: str) -> int:
    """Return the samples-per-packet ratio for ``sample_dim``.

    Only used when the Dataset carries no ``*_packet_index`` variable for this sample axis.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset.
    sample_dim : str
        Name of the sample dimension.

    Returns
    -------
    int
        Samples per packet.

    Raises
    ------
    ValueError
        If the sample axis length is not an exact multiple of the packet axis length, or if it
        disagrees with the configured sample count.
    """
    n_packets = dataset.sizes[PACKET_DIM]
    n_samples = dataset.sizes[sample_dim]
    configured = _configured_sample_count(dataset, sample_dim)

    if n_packets == 0:
        return configured if configured is not None else 0

    if n_samples % n_packets:
        configured_note = f" (packet config says {configured}/packet)" if configured is not None else ""
        raise ValueError(
            f"{sample_dim} has {n_samples} samples for {n_packets} packets, which is not an exact "
            f"multiple{configured_note}. Samples cannot be matched to packets positionally and the "
            f"Dataset carries no {sample_dim} packet index variable. This usually means duplicate "
            f"sample timestamps were dropped during decode; re-decode from the source packet files."
        )

    ratio = n_samples // n_packets
    if configured is not None and ratio != configured:
        raise ValueError(
            f"{sample_dim} has {ratio} samples per packet but the packet config specifies "
            f"{configured}. One of the packet or sample axes has already been subset independently "
            f"of the other."
        )
    return ratio


def sample_to_packet_index(dataset: xr.Dataset, sample_dim: str) -> np.ndarray:
    """Map every row of ``sample_dim`` to its originating row of ``PACKET``.

    Prefers the stored ``{sample_group}_packet_index`` variable, which stays correct when
    duplicate samples have been dropped. Falls back to positional arithmetic
    (``sample i -> packet i // samples_per_packet``) only when no index variable is present.

    The returned mapping is *not* required to be monotonically non-decreasing. Sample axes are
    sorted by sample time in :func:`~libera_utils.l1a.packets.create_l1a_dataset`, so a few
    microseconds of skew between two adjacent packets' sample clocks is enough to interleave
    their sample blocks. That is a real property of the telemetry, not corruption, and every
    operation in this module indexes through the mapping element-wise rather than assuming
    packet-major blocks, so interleaving is carried through faithfully. It is logged, not raised.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset.
    sample_dim : str
        Name of the sample dimension.

    Returns
    -------
    np.ndarray
        ``int64`` array of length ``dataset.sizes[sample_dim]``.

    Raises
    ------
    ValueError
        If a stored packet index is out of range of the ``PACKET`` axis, which means the packet
        and sample axes describe different sets of packets.
    """
    n_packets = dataset.sizes[PACKET_DIM]
    index_var = _packet_index_var_name(dataset, sample_dim)

    if index_var is None:
        samples_per_packet = _samples_per_packet(dataset, sample_dim)
        return (np.arange(dataset.sizes[sample_dim], dtype=np.int64) // samples_per_packet).astype(np.int64)

    packet_index = np.asarray(dataset[index_var].values, dtype=np.int64)
    if packet_index.size and (packet_index.min() < 0 or packet_index.max() >= n_packets):
        raise ValueError(
            f"{index_var} contains packet indices outside the range of the {PACKET_DIM} axis: "
            f"[{packet_index.min()}, {packet_index.max()}] against {n_packets} packets. The packet "
            f"and sample axes describe different sets of packets."
        )
    n_inversions = int(np.count_nonzero(np.diff(packet_index) < 0))
    if n_inversions:
        logger.info(
            "%s steps backwards at %d of %d sample positions, so %s is not strictly packet-major. "
            "This is expected where adjacent packets' sample clocks skew enough to interleave their "
            "sample blocks; the mapping is used element-wise and stays correct.",
            index_var,
            n_inversions,
            packet_index.size,
            sample_dim,
        )
    return packet_index


def select_packets(dataset: xr.Dataset, keep_packets: slice | np.ndarray) -> xr.Dataset:
    """Subset a decoded L1A Dataset along ``PACKET``, carrying its sample axes along.

    Every sample of a selected packet is kept and every sample of an unselected packet is
    dropped, so whole packets survive. Each ``*_packet_index`` variable is renumbered from 0
    against the surviving packet axis.

    Packet order is preserved: the selection is applied as a sorted integer indexer, never a
    sort, so downstream filename generation (which reads the first and last time value
    positionally) stays correct.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset with a ``PACKET`` dimension.
    keep_packets : slice or np.ndarray
        Packets to keep, as a ``slice``, a boolean mask over ``PACKET``, or an integer array of
        packet indices.

    Returns
    -------
    xr.Dataset
        Subset Dataset. May be empty along ``PACKET`` if nothing was selected; callers decide
        whether that is an error.

    Raises
    ------
    ValueError
        If ``dataset`` has no ``PACKET`` dimension, or if a sample axis cannot be mapped back to
        the packet axis.
    """
    if PACKET_DIM not in dataset.dims:
        raise ValueError(f"Dataset is missing required dimension {PACKET_DIM!r}")

    n_packets = dataset.sizes[PACKET_DIM]
    # Sorting the indexer makes the isel order-preserving even when the caller passes an unordered
    # integer array; a reordered packet axis would corrupt positional filename time extraction.
    keep = np.unique(np.arange(n_packets)[keep_packets])
    packet_mask = np.zeros(n_packets, dtype=bool)
    packet_mask[keep] = True
    # Maps an old packet row to its row in the subset axis; only valid where packet_mask is True
    renumber = np.cumsum(packet_mask) - 1

    indexers: dict[str, np.ndarray] = {PACKET_DIM: keep}
    renumbered_indices: dict[str, tuple[str, np.ndarray]] = {}
    for sample_dim in sorted(find_sample_dims(dataset)):
        old_packet_index = sample_to_packet_index(dataset, sample_dim)
        sample_mask = packet_mask[old_packet_index]
        indexers[sample_dim] = np.flatnonzero(sample_mask)
        index_var = _packet_index_var_name(dataset, sample_dim)
        if index_var is not None:
            renumbered_indices[sample_dim] = (
                index_var,
                renumber[old_packet_index[sample_mask]].astype(np.int64),
            )

    selected = dataset.isel(indexers)

    for sample_dim, (index_var, new_values) in renumbered_indices.items():
        original = selected[index_var]
        selected[index_var] = xr.DataArray(data=new_values, dims=[sample_dim], attrs=dict(original.attrs))
        selected[index_var].encoding = {}

    return selected


def slice_l1a_dataset_to_time_window(
    dataset: xr.Dataset,
    t0: np.datetime64,
    t1: np.datetime64,
    *,
    packet_time_var: str = DEFAULT_PACKET_TIME_VAR,
) -> xr.Dataset:
    """Subset a decoded L1A Dataset to the packets covering the time window ``[t0, t1]``.

    When the Dataset has sample axes, a packet is selected if any of its samples falls inside
    the window; the union across sample groups is taken so no group loses data another group
    kept. When the Dataset has no sample axes (e.g. PEC-SW-STAT), the window is applied to
    *packet_time_var* instead.

    Whole packets always survive, so a selected packet contributes all of its samples even where
    some of them fall outside the window. Sample time drives the *selection* because the samples
    are the science data being windowed; where a packet's sample timestamps disagree with its
    packet timestamp that is an instrument anomaly, and the product carries it through
    deliberately rather than papering over it.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset with a ``PACKET`` dimension.
    t0 : np.datetime64
        Window start time (inclusive).
    t1 : np.datetime64
        Window end time (inclusive).
    packet_time_var : str
        Packet-level time coordinate, used only for Datasets with no sample axes.

    Returns
    -------
    xr.Dataset
        Time-windowed Dataset.

    Raises
    ------
    ValueError
        If ``dataset`` has no ``PACKET`` dimension, or if it has no sample axes and is missing
        *packet_time_var*.
    """
    if PACKET_DIM not in dataset.dims:
        raise ValueError(f"Dataset is missing required dimension {PACKET_DIM!r}")

    n_packets = dataset.sizes[PACKET_DIM]
    sample_dims = sorted(find_sample_dims(dataset))

    if not sample_dims:
        if packet_time_var not in dataset:
            raise ValueError(
                f"Dataset has no sample dimensions and is missing {packet_time_var!r}, so no time window can be applied"
            )
        packet_times = dataset[packet_time_var].values
        keep_mask = (packet_times >= t0) & (packet_times <= t1)
        return select_packets(dataset, keep_mask)

    keep_mask = np.zeros(n_packets, dtype=bool)
    for sample_dim in sample_dims:
        sample_times = dataset[sample_dim].values
        in_window = (sample_times >= t0) & (sample_times <= t1)
        keep_mask[np.unique(sample_to_packet_index(dataset, sample_dim)[in_window])] = True

    _log_selection(dataset, keep_mask, t0, t1, packet_time_var)
    return select_packets(dataset, keep_mask)


def _log_selection(
    dataset: xr.Dataset,
    keep_mask: np.ndarray,
    t0: np.datetime64,
    t1: np.datetime64,
    packet_time_var: str,
) -> None:
    """Report the sample-time selection and how it differs from a packet-time selection."""
    n_selected = int(keep_mask.sum())
    if packet_time_var in dataset:
        packet_times = dataset[packet_time_var].values
        n_packet_time = int(((packet_times >= t0) & (packet_times <= t1)).sum())
        logger.info(
            "Selected %d/%d packets on sample time for window [%s — %s]; packet time would have "
            "selected %d (a difference indicates packet/sample clock skew)",
            n_selected,
            keep_mask.size,
            t0,
            t1,
            n_packet_time,
        )
    else:
        logger.info("Selected %d/%d packets on sample time for window [%s — %s]", n_selected, keep_mask.size, t0, t1)

    if n_selected:
        # Count transitions from unselected to selected to detect gaps in the retained run
        n_runs = int(np.count_nonzero(np.diff(np.concatenate(([False], keep_mask)).astype(np.int8)) == 1))
        if n_runs > 1:
            logger.warning(
                "Selected packets form %d disjoint runs within the time window; the subset Dataset "
                "will contain gaps (unexpected in normal operations)",
                n_runs,
            )
