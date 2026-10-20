"""Helpers shared by the native-grid reader tests.

A grid reader serves a :class:`~libera_utils.footprint_matching.types.SampleSet` of
the grid points inside a box, flattened row-major. :func:`region` folds that back into
the ``(data, lats, lons)`` grid view the value assertions are written against.
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.base import AncillaryReader
from libera_utils.footprint_matching.types import BoundingBox


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
