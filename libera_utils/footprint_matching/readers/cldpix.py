"""CERES CLDPIX imager-pixel cloud reader plugin.

Data source: CERES Cloud Pixel (CLDPIX) imager-resolution cloud retrievals
- Example product: CER_CLDPIX_NOAA20-VIIRS
- Format: NetCDF4 with flat (root-level) variables on a 2-D imager swath
  ``(Scanlines, Pixels)`` grid (e.g. 16120 × 400).
- Geolocation: per-pixel 2-D ``Latitude`` / ``Longitude`` arrays. Despite its
  name, ``Latitude`` holds **colatitude** (see below), converted here to latitude;
  longitude is stored 0..360 and normalized here to -180..180.
- Native scale: ~1 km (VIIRS imager pixel).

Geolocation conventions: colatitude and 0..360 longitude
---------------------------------------------------------
CERES products store geolocation in the colatitude convention inherited from the
CERES processing system: the root-level variable *named* ``Latitude`` carries
**colatitude in degrees**, 0° at the North Pole increasing to 180° at the South
Pole, i.e. ``latitude = 90 − colatitude``. The file itself advertises this via
the variable's ``valid_range`` attribute of ``[0, 180]`` (a true latitude would
be ``[-90, 90]``), and it is confirmed physically: binning the granule's 10.8 µm
brightness temperature (``BTemp1080``) by the raw value peaks near 90 (the
equator, ~285 K) and is coldest near 0 and 180 (the poles, ~245 K).

Reading the raw values as latitudes would place every pixel at the wrong
latitude *and* discard the entire southern hemisphere (colatitude > 90 falls
outside a −90..90 valid-range filter), so the conversion below is mandatory.
Longitude uses the CERES 0..360 degrees-east convention and is wrapped to
−180..180 by :func:`~libera_utils.footprint_matching.readers._longitude.normalize_longitude`.

Native pixels, matched by time
------------------------------
CLDPIX is a swath (the 2-D ``(Scanlines, Pixels)`` arrays are *not* a
regular lat/lon grid -- each pixel carries its own geolocation). Every pixel is
served at its own position, with its own observation time, as a
:class:`~libera_utils.footprint_matching.types.PointSamples` point, so the Libera
PSF weigher is the first place a value is weighted. The reader base class sorts
the points into 1 degree bins once, at load, and answers each footprint's box
query from that index.

Variable set
------------
The CLDPIX cloud properties: seven continuous cloud properties (optical depth, water
path, effective temperature, height and pressure, top height and particle radius) and
two mode-aggregated categories (particle phase and cloud mask). All are carried by
FMATCH-IMAGER and FMATCH-IMAGER-CAMTIME, the products CLDPIX feeds.

Surface-type variables (``IGBP_Ecosystem``, ``Snow_Map_Value``, ``Ice_Map_Value``)
are intentionally NOT extracted here. The pipeline uses the dedicated
``IGBPReader`` and ``NISEReader`` as the authoritative sources for land-cover
and ice/snow classification, avoiding duplication and ensuring consistency
across all operational modes.

References
----------
CERES cloud products: https://ceres.larc.nasa.gov/data/
File naming: CER_CLDPIX_{platform}-{imager}_{config}_{prod}.{YYYYMMDDHH}.nc
"""

from pathlib import Path
from typing import NamedTuple

import netCDF4
import numpy as np

from libera_utils.footprint_matching.readers._longitude import normalize_longitude
from libera_utils.footprint_matching.readers._points import (
    apply_fill_and_valid_range,
    decode_hours_of_day,
    field_fill_value,
    field_valid_range,
    granule_time_coverage,
    neighbor_spacing_km,
)
from libera_utils.footprint_matching.readers.base import PassReader
from libera_utils.footprint_matching.types import OperationalMode, PointSamples, TimeMatch, VariableSpec

# Root-level 2-D geolocation variable names. NOTE: ``Latitude`` holds
# COLATITUDE (0 = North Pole, 180 = South Pole) -- see the module docstring.
_LAT_VAR: str = "Latitude"
_LON_VAR: str = "Longitude"
# Per-pixel observation time, ``(Scanlines, Pixels)``, in fractional UTC hours of the
# day (units ``hours``); each hour is dated to the day nearest the granule's
# ``time_coverage_start`` global attribute. As found in the CER_CLDPIX_NOAA20-VIIRS granules.
_TIME_VAR: str = "Observation_Time"

# Valid range of the raw colatitude field, matching the file's own
# ``valid_range`` attribute on ``Latitude``. Filtering on this (rather than on
# −90..90) keeps the southern hemisphere, which colatitude encodes as > 90°.
_COLATITUDE_VALID_RANGE: tuple[float, float] = (0.0, 180.0)

# Colatitude of the North Pole; ``latitude = _NORTH_POLE_COLATITUDE − colatitude``.
_NORTH_POLE_COLATITUDE: float = 90.0


# The ``flag_meanings`` entry, compared case-insensitively, that marks a flag value as
# carrying no retrieval. CER_CLDPIX labels Cloud_Particle_Phase 5 ``bad_data`` and
# CERES_Cloud_Mask 3 ``Bad_data`` in the files' own ``flag_meanings``.
_BAD_DATA_FLAG_MEANING: str = "bad_data"


def _bad_data_flags(variable: netCDF4.Variable, *, source: str) -> tuple[int, ...]:
    """Return the flag values a categorical field's own ``flag_meanings`` label as bad data.

    A flag value whose meaning is ``bad_data`` (any case) carries no retrieval, so a
    reader sets it to ``NaN`` rather than letting it vote in a mode. The labels come from
    the file's CF ``flag_values`` / ``flag_meanings`` pair, not a hard-coded table.

    Parameters
    ----------
    variable : netCDF4.Variable
        The categorical field being read.
    source : str
        Name used in the error message (reader key and file).

    Returns
    -------
    tuple[int, ...]
        The flag values labelled bad data; empty when the field labels none.

    Raises
    ------
    ValueError
        If the field has no ``flag_values`` or ``flag_meanings`` attribute, or the two
        do not list the same number of entries.
    """
    attrs = variable.ncattrs()
    if "flag_values" not in attrs or "flag_meanings" not in attrs:
        raise ValueError(
            f"{source}: {variable.name} has no flag_values / flag_meanings attributes. The reader takes the codes "
            "to drop as bad data from them; stage the granule as distributed, which carries both on every "
            "categorical field."
        )
    values = np.ravel(variable.getncattr("flag_values"))
    meanings = str(variable.getncattr("flag_meanings")).split()
    if values.size != len(meanings):
        raise ValueError(
            f"{source}: {variable.name} lists {values.size} flag_values but {len(meanings)} flag_meanings. "
            "Stage the granule as distributed, or update the reader for this product version."
        )
    return tuple(int(v) for v, m in zip(values, meanings, strict=True) if m.lower() == _BAD_DATA_FLAG_MEANING)


class _CLDPIXField(NamedTuple):
    """Mapping from an output variable to its source variable and decoding rules.

    Each field's fill value and valid range are read from the file's own
    ``_FillValue`` and ``valid_range`` attributes. A categorical field's flag values
    that its ``flag_meanings`` label ``bad_data`` are dropped too, so they do not vote
    in the mode.

    Attributes
    ----------
    out_name : str
        Output variable name exposed via ``VARIABLES``.
    var : str
        Root-level source variable name.
    aggregation : str
        PSF aggregation strategy for the emitted spec.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous.
    """

    out_name: str
    var: str
    aggregation: str
    n_categories: int | None


_CLDPIX_FIELDS: tuple[_CLDPIXField, ...] = (
    # --- continuous cloud properties ---
    _CLDPIXField("cloud_optical_depth", "Eff_Cld_Optical_Depth", "weighted_log_mean", None),
    _CLDPIXField("cloud_water_path", "Cld_Water_Path", "weighted_mean", None),
    _CLDPIXField("cloud_effective_temperature", "Eff_Cld_Temp", "weighted_mean", None),
    _CLDPIXField("cloud_effective_height", "Eff_Cld_Height", "weighted_mean", None),
    _CLDPIXField("cloud_effective_pressure", "Eff_Cld_Pressure", "weighted_mean", None),
    _CLDPIXField("cloud_top_height", "Top_Cld_Height", "weighted_mean", None),
    # Effective cloud particle radius (μm). Cld_Radius is the combined
    # (water+ice blended) effective radius produced by the CERES cloud
    # retrieval algorithm, distinct from the phase-separated radii
    # (Cld_Radius_0124, Cld_Radius_0160) at specific wavelengths.
    _CLDPIXField("cloud_particle_radius", "Cld_Radius", "weighted_mean", None),
    # --- categorical (mode-aggregated) ---
    _CLDPIXField("cloud_particle_phase", "Cloud_Particle_Phase", "weighted_mode", 5),
    _CLDPIXField("cloud_mask", "CERES_Cloud_Mask", "weighted_mode", 4),
)


# The products that carry the CLDPIX variables.
_MODES: frozenset[OperationalMode] = frozenset({OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME})


class CLDPIXReader(PassReader):
    """Read CERES CLDPIX imager-pixel cloud data as time-stamped native points.

    Flattens the imager swath to points -- every pixel at its own position, with
    its own observation time -- and reads the CLDPIX cloud
    properties. A day of CLDPIX sees a ground location on more than one pass, so
    the reader serves the pass nearest the Libera footprint time
    (``TIME_MATCH = NEAREST_PASS``). Granules are hourly and an hour is several million
    pixels, so only the granules whose coverage lies within ``MAX_TIME_OFFSET`` of the
    current footprint time are held in memory.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"cldpix"``.
    RESOLUTION_KM : float
        ~1 km (VIIRS imager pixel scale), the fallback when a granule's own pixel
        spacing cannot be measured.
    VARIABLES : tuple[VariableSpec, ...]
        The CLDPIX cloud properties (see module docstring).

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The CERES CLDPIX NetCDF4 granule(s) covering the processing period.
    """

    READER_KEY: str = "cldpix"
    # CLDPIX imager-pixel cloud retrievals read here are the NOAA-20 stream
    # (CER_CLDPIX_NOAA20-VIIRS).
    # TODO[LIBSDC-899]: decide whether the NOAA-22 stream replaces NOAA-20 here (a cutover of this reader) or gets
    # a second reader.
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 1.0
    TIME_MATCH: TimeMatch = TimeMatch.NEAREST_PASS
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=f.out_name,
            dtype="int16" if f.aggregation == "weighted_mode" else "float32",
            aggregation=f.aggregation,
            modes=_MODES,
            n_categories=f.n_categories,
        )
        for f in _CLDPIX_FIELDS
    )

    def _granule_span(self, path: Path) -> tuple[np.datetime64, np.datetime64]:
        """The granule's ``time_coverage_start`` / ``time_coverage_end`` global attributes.

        Raises
        ------
        ValueError
            If either attribute is missing or is not an ISO 8601 instant.
        """
        with netCDF4.Dataset(str(path), "r") as ds:
            return granule_time_coverage(ds, source=f"cldpix {path.name}")

    def _load_granule(self, path: Path) -> PointSamples:
        """Parse one granule into a time-stamped pixel point cloud.

        Flattens the 2-D ``(Scanlines, Pixels)`` swath arrays to 1-D point arrays. Each
        fractional-hour ``Observation_Time`` value is dated to the day nearest the
        granule's ``time_coverage_start``, so scanlines on either side of midnight keep
        their own date.

        Returns
        -------
        PointSamples
            ``lats``/``lons`` are ``(n_pixels,)`` in ordinary WGS84 degrees (latitude
            converted from the file's colatitude, longitude normalized to -180..180),
            ``values`` is ``(n_var, n_pixels)`` float32 with fill / out-of-range
            entries, and categorical codes the file labels ``bad_data``, as NaN,
            ``times`` is each pixel's observation time,
            ``datetime64[ns]`` (NaT where the file holds fill), and ``spacing_km`` the
            pixel spacing measured on this granule.

        Raises
        ------
        ValueError
            If the granule lacks the ``time_coverage_start`` attribute that dates its
            fractional-hour times, if its time variable is not in hours, if the
            time variable is not one value per scanline (or per pixel), if
            ``Latitude`` does not declare the colatitude ``valid_range`` of [0, 180],
            if a field read (geolocation and ``Observation_Time`` included) has no
            ``_FillValue`` or ``valid_range`` attribute, or if a categorical field has
            no ``flag_values`` / ``flag_meanings`` pair.
        """
        with netCDF4.Dataset(str(path), "r") as ds:
            # Mask with apply_fill_and_valid_range rather than netCDF4's automatic
            # masking: Eff_Cld_Pressure stores its valid_range descending ([1100, 10]),
            # which netCDF4 reads as [valid_min, valid_max] and so masks every value.
            # The helper accepts either order.
            ds.set_auto_mask(False)

            # 2-D geolocation -> flatten to point lists. ``Latitude`` is colatitude
            # (0 = North Pole -> 180 = South Pole): range-check against 0..180 before
            # converting to true -90..90 latitude so fill/out-of-range pixels stay NaN
            # (NaN survives the subtraction), and convert at read time so every
            # downstream consumer sees ordinary latitudes. See the module docstring
            # for the evidence that this field really is colatitude.
            source = f"cldpix {path.name}"
            lat_var = ds.variables[_LAT_VAR]
            declared_range = field_valid_range(lat_var, source=source)
            if declared_range != _COLATITUDE_VALID_RANGE:
                raise ValueError(
                    f"{source}: {_LAT_VAR} declares valid_range {list(declared_range)}, but this reader converts "
                    f"colatitude {list(_COLATITUDE_VALID_RANGE)} to latitude. The granule's geolocation convention "
                    "has changed; update the reader's colatitude conversion before reading it."
                )
            colatitudes_2d = apply_fill_and_valid_range(
                lat_var[:], fill_value=field_fill_value(lat_var, source=source), valid_range=declared_range
            )
            shape = colatitudes_2d.shape
            lats_2d = _NORTH_POLE_COLATITUDE - colatitudes_2d
            lon_var = ds.variables[_LON_VAR]
            lons_2d = normalize_longitude(
                apply_fill_and_valid_range(
                    lon_var[:],
                    fill_value=field_fill_value(lon_var, source=source),
                    valid_range=field_valid_range(lon_var, source=source),
                )
            )
            # The granule's own pixel spacing: the delivered swath may be subsampled (the
            # CERES test granules carry 400 of the 3200 VIIRS pixels per scanline), and the
            # coverage measure must credit each pixel with the cell it really stands for.
            spacing_km = neighbor_spacing_km(lats_2d, lons_2d)
            lats = lats_2d.ravel()
            lons = lons_2d.ravel()

            start, _ = granule_time_coverage(ds, source=source)
            times = decode_hours_of_day(ds.variables[_TIME_VAR], start, source=f"{source} {_TIME_VAR}")
            if times.shape == shape[:1]:
                # One time per scanline: broadcast across the pixels of the line.
                times = np.repeat(times, shape[1])
            elif times.shape == shape:
                times = times.ravel()
            else:
                raise ValueError(
                    f"{source}: {_TIME_VAR} has shape {times.shape}, but the swath is {shape}; "
                    "expected one time per scanline or per pixel."
                )

            # Each field is written straight into one float32 array, so the float64 copy
            # of only one field is alive at a time; a granule is several million pixels.
            values = np.empty((len(_CLDPIX_FIELDS), lats.size), dtype=np.float32)
            for i, f in enumerate(_CLDPIX_FIELDS):
                var = ds.variables[f.var]
                row = apply_fill_and_valid_range(
                    var[:],
                    fill_value=field_fill_value(var, source=source),
                    valid_range=field_valid_range(var, source=source),
                ).ravel()
                if f.n_categories is not None:
                    row[np.isin(row, _bad_data_flags(var, source=source))] = np.nan
                values[i] = row

        return PointSamples(
            lats=lats,
            lons=lons,
            values=values,
            times=times,
            spacing_km=spacing_km,
        )
