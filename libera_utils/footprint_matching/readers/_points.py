"""Shared decoding helpers for the readers.

- :func:`field_fill_value` -- read a field's own ``_FillValue``, raising when it has none.
- :func:`field_valid_range` -- read a field's own ``valid_range``, raising when it has none.
- :func:`apply_fill_and_valid_range` -- convert a raw masked/integer array to a float
  array with fill values and out-of-range values replaced by ``NaN`` (the CERES SSF and
  CLDPIX readers apply each field's own ``_FillValue`` and ``valid_range``).
- :func:`decode_cf_times` -- turn a netCDF time variable with CF ``units`` into
  ``datetime64[ns]`` sample times (CERES SSF).
- :func:`decode_hours_of_day` -- turn a fractional-hour-of-day time variable (CERES
  CLDPIX) into ``datetime64[ns]`` sample times, each dated to the day nearest the
  granule's start.
- :func:`granule_time_coverage` -- read the span a granule covers from its
  ``time_coverage_start`` / ``time_coverage_end`` attributes.
- :func:`neighbor_spacing_km` and :func:`scan_to_scan_spacing_km` -- measure a swath's
  or track's native sample spacing, which the coverage measure credits each sample with.
"""

from typing import Any

import netCDF4
import numpy as np


def _fill_in_dtype(fill_value: float | int, dtype: np.dtype) -> np.ndarray | None:
    """Return ``fill_value`` in ``dtype``, or ``None`` when no value of that dtype can equal it.

    An integer dtype holds the fill only when it is a whole number within the dtype's range; any other fill
    would wrap or truncate onto a valid value, so no stored value is fill. A float fill is cast to the float
    dtype, which reproduces the stored bit pattern of a sentinel such as the float32 maximum.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if not np.isfinite(fill_value) or fill_value != int(fill_value) or not info.min <= int(fill_value) <= info.max:
            return None
        return np.asarray(int(fill_value), dtype=dtype)
    return np.asarray(fill_value).astype(dtype)


def field_fill_value(variable: netCDF4.Variable, *, source: str) -> float | int:
    """Return a field's own ``_FillValue`` attribute.

    Readers take each field's fill sentinel from the file rather than hard-coding one.
    The CERES SSF and CLDPIX readers, which turn netCDF4's automatic masking off (some
    fields declare a descending ``valid_range``), pass it to
    :func:`apply_fill_and_valid_range`; the VIIRS cloud-property reader calls it only to
    require the attribute that netCDF4's automatic masking applies.

    Parameters
    ----------
    variable : netCDF4.Variable
        The field being read.
    source : str
        Name used in the error message (reader key and file).

    Returns
    -------
    float or int
        The ``_FillValue`` attribute, as stored.

    Raises
    ------
    ValueError
        If the field has no ``_FillValue`` attribute.
    """
    if "_FillValue" not in variable.ncattrs():
        raise ValueError(
            f"{source}: {variable.name} has no _FillValue attribute. The reader takes each field's fill value "
            "from the file; stage the granule as distributed, which carries one on every field."
        )
    return np.asarray(variable.getncattr("_FillValue")).item()


def field_valid_range(variable: netCDF4.Variable, *, source: str) -> tuple[float, float]:
    """Return a field's own ``valid_range`` attribute.

    Readers take each field's valid range from the file rather than hard-coding one, so
    a range the producer revises is applied as distributed. The pair is returned in the
    file's order; :func:`apply_fill_and_valid_range` accepts a descending range such as
    CLDPIX ``Eff_Cld_Pressure``'s ``[1100, 10]``.

    Parameters
    ----------
    variable : netCDF4.Variable
        The field being read.
    source : str
        Name used in the error message (reader key and file).

    Returns
    -------
    tuple[float, float]
        The two ``valid_range`` values, as floats.

    Raises
    ------
    ValueError
        If the field has no ``valid_range`` attribute, or it does not hold two values.
    """
    if "valid_range" not in variable.ncattrs():
        raise ValueError(
            f"{source}: {variable.name} has no valid_range attribute. The reader takes each field's valid range "
            "from the file; stage the granule as distributed, which carries one on every field."
        )
    values = np.ravel(variable.getncattr("valid_range"))
    if values.size != 2:
        raise ValueError(
            f"{source}: {variable.name} declares valid_range {values.tolist()!r}, which is not a (low, high) pair. "
            "Stage the granule as distributed, or update the reader for this product version."
        )
    return float(values[0]), float(values[1])


def apply_fill_and_valid_range(
    raw: np.ndarray,
    fill_value: float | int | None = None,
    valid_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return ``raw`` as float64 with fill / out-of-range values set to ``NaN``.

    netCDF variables flag missing data in two ways that must both be honored: an
    explicit ``_FillValue`` sentinel (e.g. ``-999.0`` for a float field, ``32767``
    for int16) *and* a ``valid_range`` attribute outside of which values are invalid.
    The fill comparison is made on the raw array, in its own dtype, before any
    cast, so an integer sentinel that float32 cannot represent exactly still
    matches. A fill the raw integer dtype cannot hold (``-999`` on ``uint8``,
    ``1.5`` on ``int16``) matches nothing, rather than wrapping onto a valid value.

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
        fill = _fill_in_dtype(fill_value, raw_data.dtype)
        if fill is not None:
            invalid |= raw_data == fill

    arr = raw_data.astype(np.float64)
    arr[invalid] = np.nan

    if valid_range is not None:
        low, high = float(valid_range[0]), float(valid_range[1])
        if low > high:
            low, high = high, low
        arr[(arr < low) | (arr > high)] = np.nan

    return arr


def _invalid_time_mask(variable: netCDF4.Variable, data: np.ndarray, mask: np.ndarray, *, source: str) -> np.ndarray:
    """Mark fill (``_FillValue``) and out-of-``valid_range`` entries of a time variable, both read from the file."""
    invalid = mask.copy()
    fill = _fill_in_dtype(field_fill_value(variable, source=source), data.dtype)
    if fill is not None:
        invalid |= data == fill
    low, high = sorted(field_valid_range(variable, source=source))
    invalid |= (data < low) | (data > high)
    return invalid


def _raw_and_mask(variable: netCDF4.Variable) -> tuple[np.ndarray, np.ndarray]:
    raw = variable[:]
    if isinstance(raw, np.ma.MaskedArray):
        return np.ma.getdata(raw), np.ma.getmaskarray(raw)
    return np.asarray(raw), np.zeros(np.shape(raw), dtype=bool)


def decode_cf_times(variable: netCDF4.Variable, *, source: str) -> np.ndarray:
    """Decode a netCDF4 time variable with CF ``units`` into ``datetime64[ns]``.

    Parameters
    ----------
    variable : netCDF4.Variable
        A time variable carrying a CF ``units`` attribute such as
        ``"days since 1970-01-01 00:00:00"`` (any calendar netCDF4 supports). Entries
        equal to its ``_FillValue`` or outside its ``valid_range`` decode to ``NaT``;
        the reader base class drops such samples, since a sample with no time cannot
        be matched to a footprint.
    source : str
        Name used in the error message (reader key and variable path).

    Returns
    -------
    np.ndarray
        ``datetime64[ns]`` array the shape of the variable, ``NaT`` where the
        variable is masked, fill or out of range.

    Raises
    ------
    ValueError
        If the variable has no ``units`` attribute, or its units are not a CF time
        unit netCDF4 can decode; if it has no ``_FillValue`` or ``valid_range``
        attribute, or its ``valid_range`` is not a (low, high) pair.
    """
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError(
            f"{source}: the time variable has no 'units' attribute, so its values cannot be decoded. "
            "The reader needs a CF time unit such as 'seconds since 1970-01-01'."
        )
    data, mask = _raw_and_mask(variable)
    invalid = _invalid_time_mask(variable, data, mask, source=source)
    calendar = getattr(variable, "calendar", "standard")
    # Decode a safe stand-in where the value is invalid, then overwrite with NaT: the
    # fill sentinel (1e308 days) would overflow the calendar arithmetic.
    safe = np.where(invalid, 0, data)
    try:
        decoded = netCDF4.num2date(
            safe, units=units, calendar=calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{source}: cannot decode the time variable with units {units!r} (calendar {calendar!r}): {exc}. "
            "The reader needs a CF time unit such as 'seconds since 1970-01-01'."
        ) from exc
    times = np.asarray(decoded).astype("datetime64[ns]")
    if np.any(invalid):
        times = times.copy()
        times[invalid] = np.datetime64("NaT")
    return times


def decode_hours_of_day(variable: netCDF4.Variable, reference: np.datetime64, *, source: str) -> np.ndarray:
    """Decode a time variable holding fractional UTC hours of the day into ``datetime64[ns]``.

    CERES CLDPIX stores ``Observation_Time`` this way (units ``hours``, valid range
    0 to 24), with the date only in the granule's ``time_coverage_start`` attribute.
    Each hour is dated to the day that puts it nearest ``reference``: a granule
    starting just after midnight can hold scanlines from just before it (stored as
    about 23.9999 h), and one starting just before midnight can hold scanlines from
    just after it.

    Parameters
    ----------
    variable : netCDF4.Variable
        The fractional-hour variable. Entries equal to its ``_FillValue`` or outside
        its ``valid_range`` decode to ``NaT``.
    reference : np.datetime64
        The instant the hours are dated against, the granule's ``time_coverage_start``.
        Each decoded time lies within 12 h of it.
    source : str
        Name used in error messages (reader key and variable path).

    Returns
    -------
    np.ndarray
        ``datetime64[ns]`` array the shape of the variable.

    Raises
    ------
    ValueError
        If the variable's ``units`` attribute is present and is not ``hours``; if it has
        no ``_FillValue`` or ``valid_range`` attribute, or its ``valid_range`` is not a
        (low, high) pair.
    """
    units = getattr(variable, "units", None)
    if units is not None and str(units).strip().lower() not in ("hours", "hour", "h"):
        raise ValueError(f"{source}: expected a time variable in fractional hours of the day, found units {units!r}.")
    data, mask = _raw_and_mask(variable)
    invalid = _invalid_time_mask(variable, data, mask, source=source)
    reference = np.datetime64(reference, "ns")
    midnight = reference.astype("datetime64[D]").astype("datetime64[ns]")
    hours = np.where(invalid, 0.0, data).astype(np.float64)
    times = midnight + np.round(hours * 3.6e12).astype("timedelta64[ns]")
    half_day = np.timedelta64(12, "h")
    one_day = np.timedelta64(1, "D")
    times = np.where(times - reference > half_day, times - one_day, times)
    times = np.where(reference - times > half_day, times + one_day, times)
    if np.any(invalid):
        times = times.copy()
        times[invalid] = np.datetime64("NaT")
    return times


# Global attributes naming the observation span a CERES granule covers (ISO 8601 instants).
TIME_COVERAGE_START_ATTR: str = "time_coverage_start"
TIME_COVERAGE_END_ATTR: str = "time_coverage_end"


def granule_time_coverage(dataset: Any, *, source: str) -> tuple[np.datetime64, np.datetime64]:
    """Read a granule's ``time_coverage_start`` and ``time_coverage_end`` global attributes.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        An open granule.
    source : str
        Name used in error messages (reader key and file).

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        ``(start, end)`` as ``datetime64[ns]``.

    Raises
    ------
    ValueError
        If either attribute is missing or is not an ISO 8601 instant.
    """
    out = []
    for name in (TIME_COVERAGE_START_ATTR, TIME_COVERAGE_END_ATTR):
        value = getattr(dataset, name, None)
        if value is None:
            raise ValueError(
                f"{source}: no {name!r} global attribute. The reader needs it (an ISO 8601 instant) to place "
                "the granule in time."
            )
        try:
            out.append(np.datetime64(str(value).strip().rstrip("Z"), "ns"))
        except ValueError as exc:
            raise ValueError(f"{source}: {name}={value!r} is not an ISO 8601 instant.") from exc
    return out[0], out[1]


_EARTH_RADIUS_KM: float = 6371.0


def _great_circle_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Haversine distance (km) between paired points on a sphere of mean Earth radius."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    half_dlon = np.radians(lon2 - lon1) / 2.0
    a = np.sin((p2 - p1) / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(half_dlon) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def neighbor_spacing_km(lats: np.ndarray, lons: np.ndarray, *, max_rows: int = 500) -> float:
    """Median distance between neighboring samples of a swath or a track, km.

    For a 2-D ``(scanlines, pixels)`` swath this is the larger of the median across-track
    neighbor distance (pixel to next pixel on a scanline) and the median along-track
    one (scanline to next scanline at a pixel), so a swath that is subsampled on delivery
    reports the spacing it really has. For a 1-D track it is the median distance between
    consecutive samples, with jumps larger than ten times the median (scan turnarounds,
    gaps) left out. Non-finite coordinates are ignored.

    Parameters
    ----------
    lats, lons : np.ndarray
        Geodetic coordinates, degrees, 1-D or 2-D with the same shape, in sample order.
    max_rows : int, optional
        Rows sampled from a 2-D swath for the across-track estimate (evenly spaced), to
        keep the cost small on a million-pixel granule.

    Returns
    -------
    float
        Spacing in km, or NaN when fewer than two finite neighbors exist.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    if lats.ndim == 2 and min(lats.shape) >= 2:
        rows = np.linspace(0, lats.shape[0] - 1, min(max_rows, lats.shape[0])).astype(int)
        across = _great_circle_km(lats[rows, :-1], lons[rows, :-1], lats[rows, 1:], lons[rows, 1:])
        cols = np.linspace(0, lats.shape[1] - 1, min(max_rows, lats.shape[1])).astype(int)
        along = _great_circle_km(lats[:-1, cols], lons[:-1, cols], lats[1:, cols], lons[1:, cols])
        estimates = [np.nanmedian(d) for d in (across, along) if np.isfinite(d).any()]
        return float(max(estimates)) if estimates else float("nan")
    flat_lat, flat_lon = lats.ravel(), lons.ravel()
    if flat_lat.size < 2:
        return float("nan")
    d = _great_circle_km(flat_lat[:-1], flat_lon[:-1], flat_lat[1:], flat_lon[1:])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    median = float(np.median(d))
    return float(np.median(d[d <= 10.0 * median])) if median > 0 else median


def scan_to_scan_spacing_km(lats: np.ndarray, lons: np.ndarray, times: np.ndarray, sample_number: np.ndarray) -> float:
    """Median distance between a scan position and the same position on the next scan, km.

    CERES SSF lists footprints scan by scan; ``sample_number`` is the position within a
    scan. Sorting by position and time pairs each footprint with the one at the same
    position on the following scan, whose separation is the along-track spacing (about
    44 km for a 6.6 s scan at 6.7 km/s), far larger than the along-scan spacing. Pairs
    more than ten times the median apart (a gap between scans) are left out.

    Parameters
    ----------
    lats, lons : np.ndarray
        Footprint coordinates, degrees.
    times : np.ndarray
        Footprint observation times, any sortable dtype (``datetime64`` or float days).
    sample_number : np.ndarray
        Position of each footprint within its scan.

    Returns
    -------
    float
        Spacing in km, or NaN when no two footprints share a scan position.
    """
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    order = np.lexsort((np.asarray(times), np.asarray(sample_number)))
    sn = np.asarray(sample_number)[order]
    same = sn[1:] == sn[:-1]
    if not np.any(same):
        return float("nan")
    lat_s, lon_s = lats[order], lons[order]
    d = _great_circle_km(lat_s[:-1], lon_s[:-1], lat_s[1:], lon_s[1:])[same]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    median = float(np.median(d))
    return float(np.median(d[d <= 10.0 * median])) if median > 0 else median
