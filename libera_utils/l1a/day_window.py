"""Day-window trimming and data-time uniqueness checks for L1A datasets."""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta

import numpy as np
import xarray as xr

from libera_utils.l1a.packet_slicing import (
    PACKET_DIM,
    assert_no_missing_times,
    find_sample_dims,
    packet_time_coordinate,
    slice_l1a_dataset_to_time_window,
)

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


def trim_l1a_to_day_window(dataset: xr.Dataset, *, day: date) -> xr.Dataset:
    """Trim a decoded L1A dataset to the packets inside the day window of ``day``.

    Selection follows :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window`
    over the closed window from :func:`day_window_bounds`: on the union of every sample axis
    where the product has them (a WFOV image keeps all of its packets), else on the APID's packet
    time coordinate. Whole packets survive, and every packet index variable is renumbered against
    the trimmed ``PACKET`` axis.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A dataset with a ``PACKET`` dimension.
    day : date
        Applicable UTC calendar day.

    Returns
    -------
    xr.Dataset
        Trimmed dataset; empty along ``PACKET`` when nothing falls in the window.

    Raises
    ------
    MissingDataTimeError
        If the time axis selected on holds ``NaT``.
    ValueError
        As :func:`~libera_utils.l1a.packet_slicing.slice_l1a_dataset_to_time_window` raises it.
    """
    start, end = day_window_bounds(day)
    return slice_l1a_dataset_to_time_window(dataset, np.datetime64(start, "us"), np.datetime64(end, "us"))


def assert_data_times_unique_monotonic(dataset: xr.Dataset, *, ground_data: bool = False) -> None:
    """Assert that every data-time axis of a decoded L1A dataset is unique and non-decreasing.

    The data-time axes are the dataset's sample axes
    (:func:`~libera_utils.l1a.packet_slicing.find_sample_dims`, ``CAMERA_TIME`` for WFOV SCI), the
    same axes the day trim selects on. A dataset with no sample axis is checked on its packet time
    coordinate, its only data time.

    Parameters
    ----------
    dataset : xr.Dataset
        Decoded L1A dataset to check.
    ground_data : bool, optional
        If True, warn instead of raising on a repeated or backwards-stepping time. A ``NaT`` raises
        either way.

    Raises
    ------
    MissingDataTimeError
        If an axis holds ``NaT``.
    DataTimeUniquenessError
        If ``ground_data`` is False and a time repeats or steps backwards on any axis. The message
        names the axis and lists up to 10 duplicated values.
    ValueError
        If the dataset has packets but no sample axis, and its packet time coordinate cannot be
        resolved or is absent.

    Warns
    -----
    UserWarning
        If ``ground_data`` is True and a time repeats or steps backwards.
    """
    axes = sorted(find_sample_dims(dataset))
    if not axes:
        if dataset.sizes.get(PACKET_DIM, 0) == 0:
            return
        packet_time_var = packet_time_coordinate(dataset)
        if packet_time_var not in dataset:
            raise ValueError(
                f"Dataset has no sample dimensions and is missing {packet_time_var!r}, the packet time coordinate "
                f"its APID's configuration names. Decode it with "
                f"libera_utils.l1a.packets.parse_packets_to_l1a_dataset, which writes it."
            )
        axes = [packet_time_var]

    for axis in axes:
        times = np.asarray(dataset[axis].values)
        assert_no_missing_times(times, axis)
        if times.size < 2:
            continue
        steps = np.diff(times)
        zero = np.zeros((), dtype=steps.dtype)
        inverted = bool(np.any(steps < zero))
        if inverted:
            unique_vals, counts = np.unique(times, return_counts=True)
            duplicates = unique_vals[counts > 1]
        else:
            duplicates = np.unique(times[1:][steps == zero])

        if duplicates.size:
            _report(
                f"Data times on '{axis}' are not unique: {duplicates.size} duplicate value(s): {duplicates[:10]}",
                ground_data,
            )
        if inverted:
            _report(f"Data times on '{axis}' are not monotonic non-decreasing", ground_data)


def _report(msg: str, ground_data: bool) -> None:
    if ground_data:
        warnings.warn(msg, UserWarning, stacklevel=3)
    else:
        raise DataTimeUniquenessError(msg)
