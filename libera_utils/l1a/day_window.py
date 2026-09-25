"""Day-window trimming and data-time uniqueness checks for L1A datasets."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import numpy as np
import xarray as xr

from libera_utils.l1a.packet_slicing import DEFAULT_PACKET_TIME_VAR, slice_l1a_dataset_to_time_window

DAY_BUFFER = timedelta(minutes=10)


class DataTimeUniquenessError(ValueError):
    """Raised when data timestamps are not unique or not monotonic."""


def day_core_and_buffer_bounds(
    day: date,
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime], tuple[datetime, datetime]]:
    """Return ``(left_buffer, day_core, right_buffer)`` for ``day`` as naive-UTC ``(start, end)`` pairs.

    The three pairs abut: ``[D − DAY_BUFFER, D]``, ``[D, D + 1]``, ``[D + 1, D + 1 + DAY_BUFFER]``.
    """
    day_start = datetime.combine(day, datetime.min.time())
    day_end = day_start + timedelta(days=1)
    return (day_start - DAY_BUFFER, day_start), (day_start, day_end), (day_end, day_end + DAY_BUFFER)


def day_window_bounds(day: date) -> tuple[datetime, datetime]:
    """Return the closed window ``[D − DAY_BUFFER, D + 1 + DAY_BUFFER]`` for ``day`` as naive-UTC datetimes."""
    (window_start, _), _, (_, window_end) = day_core_and_buffer_bounds(day)
    return window_start, window_end


def trim_l1a_to_day_window(
    dataset: xr.Dataset,
    *,
    day: date,
    packet_time_var: str = DEFAULT_PACKET_TIME_VAR,
) -> xr.Dataset:
    """Trim a decoded L1A dataset to the packets inside the day window of ``day``.

    Selection follows :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window`
    over the closed window from :func:`day_window_bounds`: on the union of every sample axis
    where the product has them (a WFOV image keeps all of its packets), else on
    ``packet_time_var``. Whole packets survive, and every packet index variable is renumbered
    against the trimmed ``PACKET`` axis.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A dataset with a ``PACKET`` dimension.
    day : date
        Applicable UTC calendar day.
    packet_time_var : str, optional
        Packet time coordinate of the APID, from its packet configuration's
        ``packet_time_coordinate``.

    Returns
    -------
    xr.Dataset
        Trimmed dataset; empty along ``PACKET`` when nothing falls in the window.

    Raises
    ------
    ValueError
        If ``dataset`` has no ``PACKET`` dimension, or has no sample axes and no ``packet_time_var``.
    """
    start, end = day_window_bounds(day)
    return slice_l1a_dataset_to_time_window(
        dataset, np.datetime64(start, "us"), np.datetime64(end, "us"), packet_time_var=packet_time_var
    )


def assert_data_times_unique_monotonic(
    dataset: xr.Dataset,
    time_coord: str,
    *,
    ground_data: bool = False,
) -> None:
    """Assert that ``time_coord`` values are unique, and optionally non-decreasing.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to check.
    time_coord : str
        Time coordinate or variable name.
    ground_data : bool, optional
        If True, warn instead of raising on violations.

    Raises
    ------
    KeyError
        If ``time_coord`` is not in ``dataset``.
    DataTimeUniquenessError
        If ``ground_data`` is False and a time repeats or steps backwards. The message lists up to
        10 duplicated values.

    Warns
    -----
    UserWarning
        If ``ground_data`` is True and a time repeats or steps backwards.
    """
    if time_coord not in dataset.coords and time_coord not in dataset.variables:
        raise KeyError(f"time_coord '{time_coord}' not found in dataset")

    times = dataset[time_coord].values.astype("datetime64[us]")
    if times.size == 0:
        return

    unique_vals, counts = np.unique(times, return_counts=True)
    duplicates = unique_vals[counts > 1]
    if duplicates.size:
        msg = f"Data times on '{time_coord}' are not unique: {duplicates.size} duplicate value(s): {duplicates[:10]}"
        if ground_data:
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise DataTimeUniquenessError(msg)

    if np.any(times[1:] < times[:-1]):
        msg = f"Data times on '{time_coord}' are not monotonic non-decreasing"
        if ground_data:
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            raise DataTimeUniquenessError(msg)
