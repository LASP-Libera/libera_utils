"""Longitude-convention helper for the readers.

- :func:`normalize_longitude` -- convert a 0..360 longitude convention to -180..180
  (the ERA5 readers' grids arrive on 0..360).
"""

import numpy as np


def normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from the 0..360 convention to -180..180.

    ERA5 grids store longitude in degrees-east on a 0..360 scale, whereas the
    footprint-matching readers use -180..180. The transform
    ``((lon + 180) % 360) - 180`` maps both conventions onto -180..180 and is a
    no-op for values already in that range.

    Parameters
    ----------
    lon : np.ndarray
        Longitude values in degrees (any convention).

    Returns
    -------
    np.ndarray
        Longitudes wrapped to the half-open interval [-180, 180).
    """
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
