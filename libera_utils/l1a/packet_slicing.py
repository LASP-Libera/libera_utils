"""Subset decoded L1A Datasets along ``PACKET``, carrying sample axes with their packets.

Decoded L1A products carry a ``PACKET`` axis holding one row per CCSDS packet and one sample
axis per sample group (e.g. ``RAD_FULL_FPE_TIME``) holding the samples expanded out of those
packets. Each sample belongs to exactly one packet, recorded in the
``{sample_group}_packet_index`` variable written by
:func:`~libera_utils.l1a.packets.create_l1a_dataset`.

This module is the supported way to subset such a product; subsetting either axis directly
leaves the two out of correspondence. The packet axis is the driver:

- packets are selected first, and every sample of a selected packet is kept while every sample
  of an unselected packet is dropped, so whole packets survive or none of them does;
- ``{sample_group}_packet_index`` is renumbered from 0 against the surviving packet axis, which
  keeps it usable as the sample-to-packet validation link in derived products.

The sample axis is stored in sample-time order, which is almost but not exactly packet order:
where two adjacent packets' sample clocks skew by less than a sample interval their sample blocks
interleave, and ``{sample_group}_packet_index`` steps backwards at those few positions. The index
is applied element-wise throughout, so an interleaved run is sliced correctly and logged.

WFOV SCI (APID 1040) carries a ``CAMERA_TIME`` axis with one row per stitched image. Each row
records its SOP packet in ``CAMERA_PACKET_INDEX`` and owns every packet whose ``PACKET_IMAGE_ID``
matches that SOP's, SOP through EOP. ``CAMERA_PACKET_INDEX`` is renumbered like a
``*_packet_index`` variable; ``PACKET_IMAGE_ID`` values are kept as they are.

:func:`slice_l1a_dataset_to_time_window` selects on sample time where sample axes exist and on
the packet time coordinate otherwise. Whole packets survive either way, so a selected packet
contributes samples on both sides of the window boundary. A packet in no image
(``PACKET_IMAGE_ID == -1``) has no data time and is never selected.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a.l1a_packet_configs import PacketConfiguration, get_packet_config
from libera_utils.l1a.wfov_image_metadata import CAMERA_PACKET_INDEX_VAR, CAMERA_TIME_COORD, PACKET_IMAGE_ID_VAR

logger = logging.getLogger(__name__)

PACKET_DIM = "PACKET"
PACKET_APID_VAR = "PKT_APID"
PACKET_INDEX_SUFFIX = "_packet_index"


def _packet_config(dataset: xr.Dataset) -> PacketConfiguration:
    """Return the packet configuration of the APID in a decoded L1A Dataset's ``PKT_APID``.

    Raises
    ------
    ValueError
        If ``dataset`` has no ``PKT_APID`` variable, has no packets, or its APID has no packet
        configuration.
    """
    if PACKET_APID_VAR not in dataset:
        raise ValueError(
            f"The Dataset has no {PACKET_APID_VAR!r} variable, so its APID is unknown. Decode it with "
            f"libera_utils.l1a.packets.parse_packets_to_l1a_dataset, which writes it."
        )
    if dataset.sizes.get(PACKET_DIM, 0) == 0:
        raise ValueError(f"The Dataset has no packets, so its APID cannot be read from {PACKET_APID_VAR!r}.")
    apid = int(dataset[PACKET_APID_VAR].isel({PACKET_DIM: 0}).values)
    try:
        return get_packet_config(LiberaApid(apid))
    except (ValueError, KeyError) as err:
        raise ValueError(
            f"APID {apid} has no entry in l1a_processing_configs.yml. Check that {PACKET_APID_VAR} holds the "
            f"dataset's APID, or add a configuration entry for it."
        ) from err


def packet_time_coordinate(dataset: xr.Dataset) -> str:
    """Return the name of a decoded L1A Dataset's packet time coordinate.

    The name comes from the packet configuration of the APID in the Dataset's ``PKT_APID``
    variable, e.g. ``PACKET_JPSS_TIME`` for ``jpss_sc_pos``.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset of one APID, with at least one packet.

    Returns
    -------
    str
        Packet time coordinate name.

    Raises
    ------
    ValueError
        If ``dataset`` has no ``PKT_APID`` variable, has no packets, or its APID has no packet
        configuration.
    """
    return _packet_config(dataset).packet_time_coordinate


def data_time_axes(dataset: xr.Dataset) -> list[str]:
    """Return the axes that place a decoded L1A Dataset's packets in time.

    These are the sample axes (:func:`find_sample_dims`; ``CAMERA_TIME`` for WFOV SCI). A Dataset
    with no sample axis is placed by its packet time coordinate (:func:`packet_time_coordinate`).

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset with a ``PACKET`` dimension.

    Returns
    -------
    list of str
        Axis names, sorted. Empty for a Dataset with no sample axis and no packets.

    Raises
    ------
    ValueError
        If ``dataset`` has no ``PACKET`` dimension, or it has no sample axis and its packet time
        coordinate cannot be resolved or is absent.
    """
    if PACKET_DIM not in dataset.dims:
        raise ValueError(f"Dataset is missing required dimension {PACKET_DIM!r}")
    sample_dims = sorted(find_sample_dims(dataset))
    if sample_dims:
        return sample_dims
    if dataset.sizes[PACKET_DIM] == 0:
        return []
    packet_time_var = packet_time_coordinate(dataset)
    if packet_time_var not in dataset:
        raise ValueError(
            f"Dataset has no sample axis and is missing {packet_time_var!r}, the packet time coordinate its APID's "
            f"configuration names. Decode it with libera_utils.l1a.packets.parse_packets_to_l1a_dataset, which "
            f"writes it."
        )
    return [packet_time_var]


def find_sample_dims(dataset: xr.Dataset) -> set[str]:
    """Identify the sample axes of a decoded L1A Dataset.

    A sample axis is any dimension other than ``PACKET`` that either carries its own
    ``datetime64`` dimension coordinate (e.g. ``RAD_FULL_FPE_TIME``) or owns a
    ``*_packet_index`` variable (or, for ``CAMERA_TIME``, ``CAMERA_PACKET_INDEX``). The second
    rule also catches sample coordinates in Datasets opened with ``decode_times=False``, where the
    coordinate is a plain integer array.

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
        if not _is_packet_index_var(str(name)):
            continue
        if len(variable.dims) == 1 and str(variable.dims[0]) != PACKET_DIM:
            sample_dims.add(str(variable.dims[0]))

    return sample_dims


def _is_packet_index_var(name: str) -> bool:
    return name.endswith(PACKET_INDEX_SUFFIX) or name == CAMERA_PACKET_INDEX_VAR


def _packet_index_var_name(dataset: xr.Dataset, sample_dim: str) -> str | None:
    """Return the packet index variable owned by ``sample_dim``, if any."""
    for name, variable in dataset.variables.items():
        if _is_packet_index_var(str(name)) and variable.dims == (sample_dim,):
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
    try:
        packet_config = _packet_config(dataset)
    except ValueError:
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
            f"multiple{configured_note}, and the Dataset carries no {sample_dim} packet index "
            f"variable, so samples cannot be matched to packets."
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
    duplicate samples have been dropped. For ``CAMERA_TIME`` this is ``CAMERA_PACKET_INDEX``, each
    image's SOP packet. Falls back to positional arithmetic
    (``sample i -> packet i // samples_per_packet``) only when no index variable is present.

    The mapping is not necessarily monotonically non-decreasing: sample axes are sorted by
    sample time, so skew between two adjacent packets' sample clocks interleaves their sample
    blocks and the mapping steps backwards at those positions. Interleaving is logged, not
    raised.

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
            "%s steps backwards at %d of %d sample positions, so %s is not strictly packet-major; "
            "adjacent packets' sample clocks skew enough to interleave their sample blocks.",
            index_var,
            n_inversions,
            packet_index.size,
            sample_dim,
        )
    return packet_index


def select_packets(dataset: xr.Dataset, keep_packets: slice | np.ndarray) -> xr.Dataset:
    """Subset a decoded L1A Dataset along ``PACKET``, carrying its sample axes along.

    Every sample of a selected packet is kept and every sample of an unselected packet is
    dropped, so whole packets survive. Each ``*_packet_index`` variable, and
    ``CAMERA_PACKET_INDEX``, is renumbered from 0 against the surviving packet axis. A
    ``CAMERA_TIME`` row survives when its SOP packet does; keeping an image's other packets is the
    caller's job, as :func:`slice_l1a_dataset_to_time_window` does.

    Packet order is preserved; an unordered integer indexer is sorted rather than applied as a
    reordering.

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
    packet_index_by_dim = {dim: sample_to_packet_index(dataset, dim) for dim in find_sample_dims(dataset)}
    return _select_packets(dataset, keep_packets, packet_index_by_dim)


def _select_packets(
    dataset: xr.Dataset,
    keep_packets: slice | np.ndarray,
    packet_index_by_dim: dict[str, np.ndarray],
) -> xr.Dataset:
    """Apply :func:`select_packets` given each sample axis's :func:`sample_to_packet_index` mapping."""
    n_packets = dataset.sizes[PACKET_DIM]
    # Sort the indexer so isel preserves packet order even for an unordered integer array
    keep = np.unique(np.arange(n_packets)[keep_packets])
    packet_mask = np.zeros(n_packets, dtype=bool)
    packet_mask[keep] = True
    # Maps an old packet row to its row in the subset axis; only valid where packet_mask is True
    renumber = np.cumsum(packet_mask) - 1

    indexers: dict[str, np.ndarray] = {PACKET_DIM: keep}
    renumbered_indices: dict[str, tuple[str, np.ndarray]] = {}
    for sample_dim in sorted(packet_index_by_dim):
        old_packet_index = packet_index_by_dim[sample_dim]
        sample_mask = packet_mask[old_packet_index]
        indexers[sample_dim] = np.flatnonzero(sample_mask)
        index_var = _packet_index_var_name(dataset, sample_dim)
        if index_var is not None:
            renumbered_indices[sample_dim] = (
                index_var,
                renumber[old_packet_index[sample_mask]].astype(dataset[index_var].dtype),
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
) -> xr.Dataset:
    """Subset a decoded L1A Dataset to the packets covering the time window ``[t0, t1]``.

    When the Dataset has sample axes, a packet is selected if any of its samples falls inside
    the window; the union across sample groups is taken so no group loses data another group
    kept. When the Dataset has no sample axes (e.g. PEC-SW-STAT), the window is applied to its
    packet time coordinate (:func:`packet_time_coordinate`) instead.

    On ``CAMERA_TIME``, an image in the window keeps all of its packets, SOP through EOP. A packet
    in no image (``PACKET_IMAGE_ID == -1``) has no data time and is not selected.

    Whole packets always survive, so a selected packet contributes all of its samples even where
    some of them fall outside the window.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A Dataset with a ``PACKET`` dimension.
    t0 : np.datetime64
        Window start time (inclusive).
    t1 : np.datetime64
        Window end time (inclusive).

    Returns
    -------
    xr.Dataset
        Time-windowed Dataset.

    Raises
    ------
    ValueError
        As :func:`data_time_axes` raises it, or if ``dataset`` has a ``CAMERA_TIME`` axis but no
        ``PACKET_IMAGE_ID``.
    """
    axes = data_time_axes(dataset)
    n_packets = dataset.sizes[PACKET_DIM]
    if n_packets == 0:
        return dataset

    packet_index_by_dim = {dim: sample_to_packet_index(dataset, dim) for dim in find_sample_dims(dataset)}
    keep_mask = np.zeros(n_packets, dtype=bool)
    for axis in axes:
        times = dataset[axis].values
        in_window = (times >= t0) & (times <= t1)
        if axis not in packet_index_by_dim:
            keep_mask |= in_window
        elif axis == CAMERA_TIME_COORD:
            if PACKET_IMAGE_ID_VAR not in dataset:
                raise ValueError(
                    f"Dataset has a {CAMERA_TIME_COORD} axis but no {PACKET_IMAGE_ID_VAR!r}, so an image's packets "
                    f"cannot be found. Decode WFOV SCI with "
                    f"libera_utils.l1a.wfov_image_metadata.enhance_wfov_l1a_dataset."
                )
            packet_image_id = dataset[PACKET_IMAGE_ID_VAR].values
            sop_packets = packet_index_by_dim[axis][in_window]
            keep_mask |= np.isin(packet_image_id, packet_image_id[sop_packets])
        else:
            keep_mask[packet_index_by_dim[axis][in_window]] = True

    _log_selection(keep_mask, t0, t1, criterion="sample time" if packet_index_by_dim else "packet time")
    return _select_packets(dataset, keep_mask, packet_index_by_dim)


def _log_selection(keep_mask: np.ndarray, t0: np.datetime64, t1: np.datetime64, criterion: str) -> None:
    """Report how many packets a time window selected, and warn if the retained run has gaps."""
    n_selected = int(keep_mask.sum())
    logger.info("Selected %d/%d packets on %s for window [%s — %s]", n_selected, keep_mask.size, criterion, t0, t1)

    if n_selected:
        # Count transitions from unselected to selected to detect gaps in the retained run
        n_runs = int(np.count_nonzero(np.diff(np.concatenate(([False], keep_mask)).astype(np.int8)) == 1))
        if n_runs > 1:
            logger.warning(
                "Selected packets form %d disjoint runs within the time window; the subset Dataset "
                "will contain gaps (unexpected in normal operations)",
                n_runs,
            )
