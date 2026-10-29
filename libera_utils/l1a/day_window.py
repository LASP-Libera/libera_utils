"""Day-window trimming and data-time checks for L1A datasets."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time, timedelta
from typing import NamedTuple

import numpy as np
import xarray as xr

from libera_utils.l1a.packet_slicing import data_time_axes, slice_l1a_dataset_to_time_window

logger = logging.getLogger(__name__)

DAY_BUFFER = timedelta(minutes=10)


class DayBounds(NamedTuple):
    """Ends of a UTC day and of its buffered day window, as timezone-aware UTC datetimes.

    The left buffer is ``[window_start, day_start)``, the day core ``[day_start, day_end)`` and the
    right buffer ``[day_end, window_end)``.
    """

    window_start: datetime
    day_start: datetime
    day_end: datetime
    window_end: datetime


def day_bounds(day: date) -> DayBounds:
    """Return the bounds of ``day`` and of its day window, ``DAY_BUFFER`` either side.

    Parameters
    ----------
    day : datetime.date
        Applicable UTC calendar day.

    Returns
    -------
    DayBounds
        ``(D − DAY_BUFFER, D, D + 1, D + 1 + DAY_BUFFER)`` as timezone-aware UTC datetimes.
    """
    day_start = datetime.combine(day, time(), tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    return DayBounds(day_start - DAY_BUFFER, day_start, day_end, day_end + DAY_BUFFER)


def trim_l1a_to_day_window(dataset: xr.Dataset, *, day: date) -> xr.Dataset:
    """Trim a decoded L1A dataset to the packets inside the closed day window of ``day``.

    The window is ``[D − DAY_BUFFER, D + 1 + DAY_BUFFER]`` (:func:`day_bounds`); selection is
    :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window`'s. Dataset attributes
    are carried over unchanged.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A dataset with a ``PACKET`` dimension.
    day : datetime.date
        Applicable UTC calendar day.

    Returns
    -------
    xr.Dataset
        Trimmed dataset; empty along ``PACKET`` when nothing falls in the window.

    Raises
    ------
    ValueError
        As :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window` raises it.
    """
    bounds = day_bounds(day)
    # The data axes are datetime64, which carries no timezone; the bounds are UTC by construction.
    return slice_l1a_dataset_to_time_window(
        dataset,
        np.datetime64(bounds.window_start.replace(tzinfo=None), "us"),
        np.datetime64(bounds.window_end.replace(tzinfo=None), "us"),
    )


def check_data_times(dataset: xr.Dataset) -> None:
    """Log a warning for each data-time axis that repeats a time or steps backwards.

    The axes are :func:`~libera_utils.l1a.packet_slicing.data_time_axes`, the ones the day trim
    selects on. Nothing is raised for a repeated or backwards time.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A dataset to check.

    Raises
    ------
    ValueError
        As :func:`~libera_utils.l1a.packet_slicing.data_time_axes` raises it.
    """
    # TODO[LIBSDC-877]: report repeated and backwards data times per granule, not only in the log.
    for axis in data_time_axes(dataset):
        times = np.asarray(dataset[axis].values)
        if times.size < 2:
            continue
        steps = np.diff(times)
        zero = np.zeros((), dtype=steps.dtype)
        backwards = int(np.count_nonzero(steps < zero))
        if backwards:
            unique_vals, counts = np.unique(times, return_counts=True)
            duplicates = unique_vals[counts > 1]
        else:
            duplicates = np.unique(times[1:][steps == zero])

        if duplicates.size:
            logger.warning(
                "Data times on %r are not unique: %d duplicate value(s): %s", axis, duplicates.size, duplicates[:10]
            )
        if backwards:
            logger.warning("Data times on %r step backwards at %d of %d position(s)", axis, backwards, times.size)
