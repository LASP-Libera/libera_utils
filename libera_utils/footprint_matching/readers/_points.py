"""Shared decoding helpers for the readers.

- :func:`normalize_longitude` -- convert a 0..360 longitude convention to -180..180
  (the ERA5 readers' grids arrive on 0..360).
- :func:`apply_fill_and_valid_range` -- convert a raw masked/integer array to a float
  array with fill values and out-of-range values replaced by ``NaN``, compared in the
  raw dtype before the cast.
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


def apply_fill_and_valid_range(
    raw: np.ndarray,
    fill_value: float | int | None = None,
    valid_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return ``raw`` as float64 with fill / out-of-range values set to ``NaN``.

    netCDF variables flag missing data in two ways that must both be honored: an
    explicit ``_FillValue`` sentinel (e.g. ``-999.0`` for the Deep Blue aerosol
    floats, ``32767`` for int16) *and* a ``valid_range`` attribute outside of which
    values are invalid.
    The fill comparison is made on the raw array, in its own dtype, before any
    cast, so an integer sentinel that float32 cannot represent exactly still
    matches.

    Parameters
    ----------
    raw : np.ndarray
        Raw values read from the file. May be a ``numpy.ma.MaskedArray``.
    fill_value : float or int, optional
        Sentinel value to treat as missing, compared in the raw dtype. ``None``
        skips the fill check.
    valid_range : tuple of (float, float), optional
        Inclusive ``(low, high)`` bounds; values outside are set to ``NaN``.
        The order is normalized internally, so ``(1100, 10)`` is treated the same
        as ``(10, 1100)``. ``None`` skips the range check.

    Returns
    -------
    np.ndarray
        float64 array, same shape as ``raw``, with invalid entries as ``NaN``.
    """
    if isinstance(raw, np.ma.MaskedArray):
        masked = np.ma.getmaskarray(raw)
        raw_data = np.ma.getdata(raw)
    else:
        masked = np.zeros(np.shape(raw), dtype=bool)
        raw_data = np.asarray(raw)

    invalid = masked.copy()
    if fill_value is not None:
        # Compare in the raw dtype: casting the declared sentinel to the array's dtype
        # reproduces the stored bit pattern exactly (a float32 fill such as the float32
        # maximum included), and an integer sentinel compares exactly in its own dtype.
        invalid |= raw_data == np.asarray(fill_value).astype(raw_data.dtype)

    arr = raw_data.astype(np.float64)
    arr[invalid] = np.nan

    if valid_range is not None:
        low, high = float(valid_range[0]), float(valid_range[1])
        if low > high:
            low, high = high, low
        arr[(arr < low) | (arr > high)] = np.nan

    return arr
