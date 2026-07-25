"""Day-window trimming and data-time uniqueness checks for L1A datasets."""

from __future__ import annotations

import logging
import warnings
from datetime import UTC, date, datetime, timedelta

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

DEFAULT_DAY_BUFFER = timedelta(minutes=10)


class DataTimeUniquenessError(ValueError):
    """Raised when data timestamps are not unique or not monotonic."""


def day_window_bounds(
    day: date,
    buffer: timedelta = DEFAULT_DAY_BUFFER,
) -> tuple[np.datetime64, np.datetime64]:
    """Return half-open window ``[day 00:00 − buffer, day+1 00:00 + buffer)`` as datetime64[us]."""
    start = datetime.combine(day, datetime.min.time(), tzinfo=UTC) - buffer
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC) + buffer
    return (
        np.datetime64(start.replace(tzinfo=None), "us"),
        np.datetime64(end.replace(tzinfo=None), "us"),
    )


def sync_packet_dim_to_index(dataset: xr.Dataset, packet_index_var: str) -> xr.Dataset:
    """Keep only PACKET rows referenced by ``packet_index_var`` and densify indices.

    Dual-dimension L1A products (sample or camera dims plus ``PACKET``) can retain orphan
    PACKET rows after science-dimension selection. This subsets ``PACKET`` to the unique
    indices still referenced and remaps the index variable to dense ``0..n-1``, preserving
    the source dtype (compute in int64, cast back).

    Parameters
    ----------
    dataset : xr.Dataset
        L1A-like dataset that may contain a ``PACKET`` dimension.
    packet_index_var : str
        Variable mapping science rows to PACKET indices (e.g. ``RAD_SAMPLE_packet_index``,
        ``CAMERA_PACKET_INDEX``).

    Returns
    -------
    xr.Dataset
        Dataset with PACKET aligned to ``packet_index_var``, or unchanged if the index
        variable or PACKET dimension is absent.
    """
    if packet_index_var not in dataset or "PACKET" not in dataset.dims:
        return dataset

    index = dataset[packet_index_var]
    dtype = index.dtype
    remaining = np.unique(index.values.astype(np.int64, copy=False))
    remaining.sort()
    out = dataset.isel(PACKET=remaining)
    # Map old packet indices -> dense 0..n-1 positions after isel; keep product dtype.
    new_index = np.searchsorted(remaining, index.values.astype(np.int64, copy=False)).astype(dtype, copy=False)
    return out.assign({packet_index_var: (index.dims, new_index)})


def trim_l1a_to_day_window(
    dataset: xr.Dataset,
    *,
    day: date,
    time_coord: str,
    buffer: timedelta = DEFAULT_DAY_BUFFER,
    keep_whole_groups: bool = False,
    group_dim: str | None = None,
    packet_index_var: str | None = None,
) -> xr.Dataset:
    """Trim an L1A dataset to the applicable-day window including a timed buffer.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A-like dataset.
    day : date
        Applicable UTC calendar day.
    time_coord : str
        Name of the time coordinate used for selection (packet, sample, or CAMERA_TIME).
    buffer : timedelta, optional
        Buffer on each side of midnight. Default 10 minutes.
    keep_whole_groups : bool, optional
        If True, keep entire packets/images when any member falls in the window.
        Requires ``group_dim`` and usually ``packet_index_var`` for sample dims.
    group_dim : str | None, optional
        Dimension along which ``time_coord`` lives when ``keep_whole_groups`` is False
        or when selecting by that dim directly. Inferred from ``time_coord`` when possible.
    packet_index_var : str | None, optional
        Variable mapping sample/image indices to packet indices. Used to keep whole
        packets for sample-dimension products (e.g. ``RAD_SAMPLE_packet_index``) and,
        when present with a ``PACKET`` dimension, to drop orphan PACKET rows after trim
        via :func:`sync_packet_dim_to_index`.

    Returns
    -------
    xr.Dataset
        Dataset restricted to the day window (may be empty). When ``packet_index_var`` is
        set and a ``PACKET`` dimension exists, PACKET is synced to remaining indices.
    """
    if time_coord not in dataset.coords and time_coord not in dataset.variables:
        raise KeyError(f"time_coord '{time_coord}' not found in dataset")

    window_start, window_end = day_window_bounds(day, buffer)
    times = dataset[time_coord].values.astype("datetime64[us]")

    if times.ndim != 1:
        raise ValueError(f"time_coord '{time_coord}' must be 1-dimensional, got shape {times.shape}")

    in_window = (times >= window_start) & (times < window_end)

    # Resolve selection dimension
    coord = dataset[time_coord]
    if group_dim is None:
        if coord.dims:
            group_dim = coord.dims[0]
        else:
            raise ValueError(f"Cannot infer dimension for time_coord '{time_coord}'")

    if not keep_whole_groups:
        trimmed = dataset.isel({group_dim: np.nonzero(in_window)[0]})
    elif packet_index_var is None:
        # time_coord entries are the groups (e.g. PACKET or CAMERA_TIME)
        trimmed = dataset.isel({group_dim: np.nonzero(in_window)[0]})
    else:
        if packet_index_var not in dataset:
            raise KeyError(f"packet_index_var '{packet_index_var}' not found in dataset")

        packet_indices = dataset[packet_index_var].values
        packets_to_keep = set(packet_indices[in_window].tolist())
        if not packets_to_keep:
            trimmed = dataset.isel({group_dim: []})
        else:
            keep_mask = np.isin(packet_indices, list(packets_to_keep))
            trimmed = dataset.isel({group_dim: np.nonzero(keep_mask)[0]})

    if packet_index_var is not None:
        return sync_packet_dim_to_index(trimmed, packet_index_var)
    return trimmed


def assert_data_times_unique_monotonic(
    dataset: xr.Dataset,
    time_coord: str,
    *,
    ground_data: bool = False,
    verbose: bool = False,
) -> None:
    """Assert that ``time_coord`` values are unique and non-decreasing.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to check.
    time_coord : str
        Time coordinate / variable name.
    ground_data : bool, optional
        If True, emit warnings instead of raising on violations.
    verbose : bool, optional
        If True with ``ground_data``, include sample offending values in warnings.

    Raises
    ------
    DataTimeUniquenessError
        If uniqueness or monotonicity is violated and ``ground_data`` is False.
    """
    if time_coord not in dataset.coords and time_coord not in dataset.variables:
        raise KeyError(f"time_coord '{time_coord}' not found in dataset")

    times = dataset[time_coord].values.astype("datetime64[us]")
    if times.size == 0:
        return

    # Uniqueness
    unique_vals, counts = np.unique(times, return_counts=True)
    duplicates = unique_vals[counts > 1]
    if duplicates.size:
        msg = f"Data times on '{time_coord}' are not unique: {duplicates.size} duplicate value(s)"
        if verbose:
            msg = f"{msg}: {duplicates[:10]}"
        if ground_data:
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise DataTimeUniquenessError(msg)

    # Monotonic non-decreasing (after uniqueness, equal adjacent are duplicates)
    if times.size >= 2 and np.any(times[1:] < times[:-1]):
        msg = f"Data times on '{time_coord}' are not monotonic non-decreasing"
        if ground_data:
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise DataTimeUniquenessError(msg)
