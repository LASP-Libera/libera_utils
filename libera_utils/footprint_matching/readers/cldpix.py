"""CERES CLDPIX imager-pixel cloud reader plugin.

Data source: CERES Cloud Pixel (CLDPIX) imager-resolution cloud retrievals
- Example product: CER_CLDPIX_NOAA20-VIIRS
- Format: NetCDF4 with flat (root-level) variables on a 2-D imager swath
  ``(Scanlines, Pixels)`` grid (e.g. 16120 × 400).
- Geolocation: per-pixel 2-D ``Latitude`` / ``Longitude`` arrays. Despite its
  name, ``Latitude`` holds **colatitude** (see below); longitude is stored
  0..360. Both are normalized here.
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
−180..180 by :func:`~libera_utils.footprint_matching.readers._points.normalize_longitude`.

CERES SSF stores its footprint geolocation the same way, but under explicitly
named variables (``instrument_fov_latitude`` is a true −90..90 latitude), so
:mod:`~libera_utils.footprint_matching.readers.ssf` needs no such conversion.

Native pixels, matched by time
------------------------------
CLDPIX is a swath (the 2-D ``(Scanlines, Pixels)`` arrays are *not* a
regular lat/lon grid -- each pixel carries its own geolocation). Every pixel is
served at its own position, with its own observation time, as a
:class:`~libera_utils.footprint_matching.types.PointSamples` point, so the Libera
PSF weigher is the first place a value is weighted. The reader base class sorts
the points into 1 degree bins once, at load, and answers each footprint's box
query from that index.

Memory note
-----------
A CLDPIX granule covers one hour and holds about 6.4 million pixels; with the nine
variables read here that is a few hundred MB indexed. A day is 24 granules, so the
reader holds only the granules within ``MAX_TIME_OFFSET`` (3 h) of the current
footprint time, at most seven at once, and releases a granule once the footprints
have moved past its window.

Variable set
------------
This is a **minimal starter set** drawn from the footprint-matching
dependencies in ``data_products.md`` (CLDPIX supplies imager-pixel cloud
properties for the Imager / Imager-camera-time products). Expected to be
refined.

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

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from libera_utils.footprint_matching.readers._points import (
    apply_fill_and_valid_range,
    decode_hours_of_day,
    granule_time_coverage,
    neighbour_spacing_km,
    normalize_longitude,
)
from libera_utils.footprint_matching.readers.base import PassReader
from libera_utils.footprint_matching.types import OperationalMode, PointSamples, TimeMatch, VariableSpec

# Fill sentinels: floats use the float32 max; int8 categoricals use 127.
_FILL_FLOAT: float = 3.4028235e38
_FILL_INT8: int = 127

# Root-level 2-D geolocation variable names. NOTE: ``Latitude`` holds
# COLATITUDE (0 = North Pole, 180 = South Pole) -- see the module docstring.
_LAT_VAR: str = "Latitude"
_LON_VAR: str = "Longitude"
# Per-pixel observation time, ``(Scanlines, Pixels)``, in fractional UTC hours of the
# granule's day (units ``hours``); the date comes from the granule's
# ``time_coverage_start`` global attribute. As found in the CER_CLDPIX_NOAA20-VIIRS granules.
_TIME_VAR: str = "Observation_Time"

# Valid range of the raw colatitude field, matching the file's own
# ``valid_range`` attribute on ``Latitude``. Filtering on this (rather than on
# −90..90) keeps the southern hemisphere, which colatitude encodes as > 90°.
_COLATITUDE_VALID_RANGE: tuple[float, float] = (0.0, 180.0)

# Colatitude of the North Pole; ``latitude = _NORTH_POLE_COLATITUDE − colatitude``.
_NORTH_POLE_COLATITUDE: float = 90.0


class _CLDPIXField(NamedTuple):
    """Mapping from an output variable to its source variable and decoding rules.

    Attributes
    ----------
    out_name : str
        Output variable name exposed via ``VARIABLES``.
    var : str
        Root-level source variable name.
    aggregation : str
        PSF aggregation strategy for the emitted spec.
    fill : float or int
        Fill sentinel for the source variable.
    valid_range : tuple of (float, float) or None
        Inclusive valid range; values outside become NaN. Also used to drop the
        ``-1`` "no data" sentinel in the snow/ice maps.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous.
    """

    out_name: str
    var: str
    aggregation: str
    fill: float | int
    valid_range: tuple[float, float] | None
    n_categories: int | None


# Minimal starter field set (refine later — see module docstring).
# Note: surface-type variables (IGBP_Ecosystem, Snow_Map_Value, Ice_Map_Value)
# are deliberately omitted — see module docstring for the rationale.
_CLDPIX_FIELDS: tuple[_CLDPIXField, ...] = (
    # --- continuous cloud properties ---
    _CLDPIXField("cloud_optical_depth", "Eff_Cld_Optical_Depth", "weighted_log_mean", _FILL_FLOAT, (0.25, 150.0), None),
    _CLDPIXField("cloud_water_path", "Cld_Water_Path", "weighted_mean", _FILL_FLOAT, (0.0, 10000.0), None),
    _CLDPIXField("cloud_effective_temperature", "Eff_Cld_Temp", "weighted_mean", _FILL_FLOAT, (190.0, 350.0), None),
    _CLDPIXField("cloud_effective_height", "Eff_Cld_Height", "weighted_mean", _FILL_FLOAT, (0.0, 18.0), None),
    _CLDPIXField("cloud_effective_pressure", "Eff_Cld_Pressure", "weighted_mean", _FILL_FLOAT, (10.0, 1100.0), None),
    _CLDPIXField("cloud_top_height", "Top_Cld_Height", "weighted_mean", _FILL_FLOAT, None, None),
    # Effective cloud particle radius (μm). Cld_Radius is the combined
    # (water+ice blended) effective radius produced by the CERES cloud
    # retrieval algorithm, distinct from the phase-separated radii
    # (Cld_Radius_0124, Cld_Radius_0160) at specific wavelengths.
    _CLDPIXField("cloud_particle_radius", "Cld_Radius", "weighted_mean", _FILL_FLOAT, (2.0, 60.0), None),
    # --- categorical (mode-aggregated) ---
    _CLDPIXField("cloud_particle_phase", "Cloud_Particle_Phase", "weighted_mode", _FILL_INT8, (1.0, 5.0), 5),
    _CLDPIXField("cloud_mask", "CERES_Cloud_Mask", "weighted_mode", _FILL_INT8, (0.0, 3.0), 4),
)


class CLDPIXReader(PassReader):
    """Read CERES CLDPIX imager-pixel cloud data as time-stamped native points.

    Flattens the imager swath to points -- every pixel at its own position, with
    its own observation time -- and reads a minimal set of cloud
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
        Minimal starter set (see module docstring).

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The CERES CLDPIX NetCDF4 granule(s) covering the processing period.
    """

    READER_KEY: str = "cldpix"
    # CLDPIX imager-pixel cloud retrievals read here are the NOAA-20 stream
    # (CER_CLDPIX_NOAA20-VIIRS).
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 1.0
    TIME_MATCH: TimeMatch = TimeMatch.NEAREST_PASS
    # See SSFReader: 15 min keeps one imager pass together, 3 h is the largest offset
    # accepted before the footprint is scored as having no CLDPIX match.
    PASS_WINDOW: np.timedelta64 | None = np.timedelta64(15, "m")
    MAX_TIME_OFFSET: np.timedelta64 | None = np.timedelta64(3, "h")
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=f.out_name,
            dtype="float32" if f.n_categories is None else "int16",
            aggregation=f.aggregation,
            required_mode=OperationalMode.IMAGER,
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
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(path), "r") as ds:
            return granule_time_coverage(ds, source=f"cldpix {path.name}")

    def _load_granule(self, path: Path) -> PointSamples:
        """Parse one granule into a time-stamped pixel point cloud.

        Flattens the 2-D ``(Scanlines, Pixels)`` swath arrays to 1-D point arrays. The
        fractional-hour ``Observation_Time`` values are dated by the granule's
        ``time_coverage_start``; a granule covers one hour, so it never crosses midnight.

        Returns
        -------
        PointSamples
            ``lats``/``lons`` are ``(n_pixels,)`` in ordinary WGS84 degrees (latitude
            converted from the file's colatitude, longitude normalized to -180..180),
            ``values`` is ``(n_var, n_pixels)`` float32 with fill / out-of-range
            entries as NaN, ``times`` is each pixel's observation time,
            ``datetime64[ns]`` (NaT where the file holds fill), and ``spacing_km`` the
            pixel spacing measured on this granule.

        Raises
        ------
        ValueError
            If the granule lacks the ``time_coverage_start`` attribute that dates its
            fractional-hour times, if its time variable is not in hours, or if the
            time variable is not one value per scanline (or per pixel).
        """
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(path), "r") as ds:
            # Disable netCDF4 automatic masking and rely on our own
            # apply_fill_and_valid_range. This is required because some CLDPIX
            # variables (e.g. Eff_Cld_Pressure) store ``valid_range`` in
            # descending order ([1100, 10]); netCDF4 interprets that as
            # [valid_min, valid_max] and masks *every* value. Our helper
            # normalizes the range order, so it handles such fields correctly.
            # Scale/offset auto-application is left enabled (set_auto_scale).
            ds.set_auto_mask(False)

            # 2-D geolocation -> flatten to point lists. ``Latitude`` is colatitude
            # (0 = North Pole -> 180 = South Pole): range-check against 0..180 before
            # converting to true -90..90 latitude so fill/out-of-range pixels stay NaN
            # (NaN survives the subtraction), and convert at read time so every
            # downstream consumer sees ordinary latitudes. See the module docstring
            # for the evidence that this field really is colatitude.
            colatitudes_2d = apply_fill_and_valid_range(
                ds.variables[_LAT_VAR][:], fill_value=_FILL_FLOAT, valid_range=_COLATITUDE_VALID_RANGE
            )
            shape = colatitudes_2d.shape
            lats_2d = _NORTH_POLE_COLATITUDE - colatitudes_2d
            lons_2d = normalize_longitude(
                apply_fill_and_valid_range(ds.variables[_LON_VAR][:], fill_value=_FILL_FLOAT, valid_range=(0.0, 360.0))
            )
            # The granule's own pixel spacing: the delivered swath may be subsampled (the
            # CERES test granules carry 400 of the 3200 VIIRS pixels per scanline), and the
            # coverage measure must credit each pixel with the cell it really stands for.
            spacing_km = neighbour_spacing_km(lats_2d, lons_2d)
            lats = lats_2d.ravel()
            lons = lons_2d.ravel()

            day, _ = granule_time_coverage(ds, source=f"cldpix {path.name}")
            times = decode_hours_of_day(
                ds.variables[_TIME_VAR], day.astype("datetime64[D]"), source=f"cldpix {path.name} {_TIME_VAR}"
            )
            if times.shape == shape[:1]:
                # One time per scanline: broadcast across the pixels of the line.
                times = np.repeat(times, shape[1])
            elif times.shape == shape:
                times = times.ravel()
            else:
                raise ValueError(
                    f"cldpix {path.name}: {_TIME_VAR} has shape {times.shape}, but the swath is {shape}; "
                    "expected one time per scanline or per pixel."
                )

            value_rows: list[np.ndarray] = []
            for f in _CLDPIX_FIELDS:
                raw = ds.variables[f.var][:]
                value_rows.append(apply_fill_and_valid_range(raw, fill_value=f.fill, valid_range=f.valid_range).ravel())

        return PointSamples(
            lats=lats,
            lons=lons,
            values=np.stack(value_rows, axis=0).astype(np.float32),
            times=times,
            spacing_km=spacing_km,
        )
