"""Helpers shared by the reader tests.

A grid reader serves a :class:`~libera_utils.footprint_matching.types.SampleSet` of
the grid points inside a box, flattened row-major. :func:`region` folds that back into
the ``(data, lats, lons)`` grid view the value assertions are written against;
:func:`sample_at` picks out the one sample served at a given location. For the
real-granule tests of the swath readers, :func:`raw_and_usable` reads a variable as
the file stores it and :func:`served_at` finds the samples served at one footprint.
"""

import netCDF4
import numpy as np

from libera_utils.footprint_matching.readers.base import AncillaryReader
from libera_utils.footprint_matching.types import BoundingBox, SampleSet


def region(
    reader: AncillaryReader, bbox: BoundingBox, time: np.datetime64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(data, lats, lons)`` with ``data`` shaped ``(n_var, n_lat, n_lon)`` for a grid query."""
    samples = reader.samples_in(bbox, time)
    n_var = samples.values.shape[0]
    if samples.n_samples == 0:
        return np.empty((n_var, 0, 0), dtype=np.float32), np.empty(0), np.empty(0)
    lats = np.unique(samples.lats)
    lons = np.unique(samples.lons)
    data = samples.values.reshape(n_var, lats.size, lons.size)
    return data, lats, lons


def sample_at(samples: SampleSet, lat: float, lon: float) -> np.ndarray:
    """Return the ``(n_var,)`` values of the single sample served at ``(lat, lon)``."""
    match = np.flatnonzero(np.isclose(samples.lats, lat, atol=1e-6) & np.isclose(samples.lons, lon, atol=1e-6))
    assert match.size == 1, f"expected one sample at ({lat}, {lon}), found {match.size}"
    return samples.values[:, match[0]]


def raw_and_usable(path, group: str | None, name: str) -> tuple[np.ndarray, np.ndarray]:
    """A variable's raw stored values and a mask of the entries that are neither fill nor outside ``valid_range``."""
    with netCDF4.Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        var = (ds.groups[group] if group else ds).variables[name]
        raw = np.asarray(var[:])
        low, high = sorted(float(v) for v in var.valid_range)
        usable = (raw != var._FillValue) & (raw >= low) & (raw <= high)
    return raw, usable


def served_at(samples: SampleSet, lat: float, lon: float, time: np.datetime64) -> np.ndarray:
    """Indices of the served samples at exactly ``(lat, lon, time)``."""
    return np.flatnonzero((samples.lats == lat) & (samples.lons == lon) & (samples.times == time))
