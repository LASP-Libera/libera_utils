"""NOAA-20 VIIRS Deep Blue aerosol reader plugin (AOD + aerosol type).

Data source: Deep Blue single-sensor daily Level-3 aerosol product
- Product: AERDB_D3_VIIRS_NOAA20 (Collection 002)
- Format: NetCDF4, **all variables at the root** (no groups)
- Spatial resolution: 1° × 1° (~111 km at equator)
- Grid: 360 × 180 regular lat/lon (global), latitude ASCENDING (−89.5 → 89.5),
  longitude −179.5 → 179.5. Variables are stored in (Latitude, Longitude)
  dimension order, so — unlike the VIIRS CLDPROP_D3 cloud product — **no
  transpose is required**.
- Temporal resolution: Daily composites

What this reader exposes
------------------------
FMATCH-IMAGER depends on both **AOD** and **aerosol type** (DPI / data_products.md:
"AOD (NOAA-21 initially; NOAA-22 VIIRS when available)" plus aerosol type). The
single-sensor Deep Blue granule carries both, so this one reader serves both:

1. ``aod_550`` — the daily-mean NOAA-20 VIIRS Deep Blue AOD at 550 nm over land
   and ocean (continuous, ``weighted_log_mean``).
2. ``aerosol_type`` — the modal aerosol type over land and ocean (categorical,
   8 classes, ``weighted_mode``), plus three ranked companions
   (``aerosol_type_primary/secondary/tertiary``) declared in
   ``ADDITIONAL_PRODUCT_VARIABLES``, mirroring the IGBP reader's ranked scenes.

Aerosol type categories (from the file's ``long_name``)
-------------------------------------------------------
``0`` = dust (land+ocean), ``1`` = smoke, ``2`` = high-altitude smoke,
``3`` = pyrocumulonimbus clouds, ``4`` = non-smoke fine mode,
``5`` = mixed (land+ocean), ``6`` = background (land+ocean maritime),
``7`` = fine dominated.

Why the single-sensor granule (not the GEO-LEO merged one)?
-----------------------------------------------------------
FMATCH-IMAGER requires a single, well-characterized VIIRS aerosol source. The
dedicated single-sensor ``AERDB_D3_VIIRS_NOAA20`` Deep Blue granule provides the
NOAA-20 VIIRS Deep Blue 550 nm AOD **and** the aerosol-type field, so both are
sourced from this one authoritative granule. The cross-sensor
``AERDB_D3_GEOLEO_Merged`` granule only re-bundles the per-sensor L3s and its AOD
lacks the aerosol-type field. To switch sensors later (e.g. NOAA-21/NOAA-22),
point the reader at the corresponding ``AERDB_D3_VIIRS_*`` granule.
TODO[LIBSDC-785]

NetCDF4 layout (AERDB_D3_VIIRS_NOAA20)
-------------------------------------
Root (no groups):
  Latitude_1D  (180,)  — 1° bin centers, −89.5 → 89.5 (ascending)
  Longitude_1D (360,)  — 1° bin centers, −179.5 → 179.5
  Aerosol_Optical_Thickness_550_Land_Ocean_Mean (180, 360)  — daily-mean AOD @ 550 nm
  Aerosol_Type_Land_Ocean_Mode                  (180, 360)  — modal aerosol type (0..7)
  Aerosol_Type_Land_Ocean_Histogram    (8, 180, 360)  — per-type counts (future rank
                                                        source; not read here)

Fill value: −999 for every field (replaced with NaN). AOD ``valid_range`` in the
file is ``[0, 10]``; this reader retains the tighter ``[0, 5]`` clamp of the
``aod_550`` product contract. Aerosol-type ``valid_range`` is ``[0, 7]``.

References
----------
Deep Blue aerosol products:
    https://earthdata.nasa.gov/sensors/viirs  (Deep Blue / SOAR algorithms)
NASA Deep Blue:
    https://deepblue.gsfc.nasa.gov/
File naming:
    AERDB_D3_VIIRS_NOAA20.A{YYYYDDD}.{collection}.{YYYYDDDHHMMSS}.nc
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# Root-level coordinate and data variable names in the single-sensor Deep Blue
# L3 granule. (No group navigation: unlike AERDB_D3_GEOLEO_Merged, this product
# stores everything at the root.)
_LAT_VAR: str = "Latitude_1D"
_LON_VAR: str = "Longitude_1D"
# We read the daily *mean* AOD field. The granule also carries Min/Max/Count/
# Standard_Deviation companions; the within-footprint standard deviation is
# produced later by PSF aggregation, so the file's own std field is not read.
_AOD_VARIABLE: str = "Aerosol_Optical_Thickness_550_Land_Ocean_Mean"
# Modal (most-common) aerosol type per 1° cell over land and ocean combined.
_AEROSOL_TYPE_VARIABLE: str = "Aerosol_Type_Land_Ocean_Mode"

# Shared fill / missing sentinel for both fields in this product.
_FILL_VALUE: float = -999.0
# Physically valid AOD max. The file declares valid_range [0, 10], but we keep
# the tighter [0, 5] clamp that the aod_550 product contract already uses.
_AOD_VALID_MAX: float = 5.0
# Aerosol type has 8 categories, coded 0..7 (see module docstring). Stored in the
# VariableSpec so the PSF aggregation engine knows how many category bins to
# allocate for the modal / ranked-type histograms.
_N_AEROSOL_TYPES: int = 8


class VIIRSAODReader(GriddedDataReader):
    """Read NOAA-20 VIIRS Deep Blue AOD + aerosol type from an AERDB_D3_VIIRS_NOAA20 file.

    Loads two variables — ``aod_550`` (continuous) and ``aerosol_type``
    (categorical) — from a single-sensor Deep Blue Level-3 daily file and returns
    a 3-D data array of shape ``(2, n_lat, n_lon)`` stacked in ``VARIABLES``
    order, matching the multi-variable contract used by ``VIIRSCloudReader``.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"viirs_aod"``.
    RESOLUTION_KM : float
        111 km (1° × 1° daily L3 grid resolution at equator).
    VARIABLES : tuple[VariableSpec, ...]
        Two variables: ``aod_550`` (``weighted_log_mean``) and ``aerosol_type``
        (``weighted_mode``, 8 categories).
    ADDITIONAL_PRODUCT_VARIABLES : tuple[VariableSpec, ...]
        Three ranked aerosol-type outputs (primary/secondary/tertiary), derived
        during PSF aggregation like IGBP's ranked scenes.

    Parameters
    ----------
    file_path : Path
        Path to an AERDB_D3_VIIRS_NOAA20 NetCDF4 file.
    """

    READER_KEY: str = "viirs_aod"
    # Deep Blue single-sensor product for VIIRS aboard NOAA-20 (JPSS-1).
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 111.0
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="aod_550",
            dtype="float32",
            # AOD is approximately log-normally distributed, so a geometric
            # (log) mean is the appropriate spatial aggregation — matching how
            # cloud optical thickness is treated in VIIRSCloudReader.
            aggregation="weighted_log_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
        VariableSpec(
            name="aerosol_type",
            dtype="int16",
            # Categorical: the single aggregated (most-common) aerosol type in
            # the footprint. A std-dev companion is (correctly) not generated for
            # weighted_mode variables.
            aggregation="weighted_mode",
            required_mode=OperationalMode.IMAGER,
            n_categories=_N_AEROSOL_TYPES,
        ),
    )
    # A footprint typically spans several aerosol types. ``aerosol_type`` above is
    # the single aggregated result; these three derived outputs report the ranked
    # aerosol-type mix within the footprint — the first, second, and third most
    # common types by PSF-weighted area. They are computed during PSF aggregation
    # (from the same modal field as ``aerosol_type``), not read from a separate
    # source field, so they live here rather than in VARIABLES. This mirrors the
    # IGBP reader's ranked surface-type outputs. Distinct aggregation labels record
    # the rank; the PSF aggregation engine does not implement them yet
    # (declarations only, like every other FMATCH variable). The file's
    # ``Aerosol_Type_Land_Ocean_Histogram`` is the intended future ranking source.
    ADDITIONAL_PRODUCT_VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="aerosol_type_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            required_mode=OperationalMode.IMAGER,
            n_categories=_N_AEROSOL_TYPES,
        ),
        VariableSpec(
            name="aerosol_type_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            required_mode=OperationalMode.IMAGER,
            n_categories=_N_AEROSOL_TYPES,
        ),
        VariableSpec(
            name="aerosol_type_tertiary",
            dtype="int16",
            aggregation="weighted_mode_tertiary",
            required_mode=OperationalMode.IMAGER,
            n_categories=_N_AEROSOL_TYPES,
        ),
    )

    @cached_property
    def _native_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the full NOAA-20 VIIRS aerosol grid once and cache it on the instance.

        Opens the NetCDF4 file, reads the root coordinate arrays (ascending
        latitude, no transpose needed) and both the AOD and aerosol-type fields,
        replaces fill / out-of-range values with NaN, and stacks the two fields in
        ``VARIABLES`` order. Cached per reader instance so the file is opened and
        read once, then every tile slices these in-memory arrays (see
        :meth:`_load_spatial_region`) instead of re-reading the file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(2, n_lat, n_lon)`` with axis 0 = [aod_550, aerosol_type] (fill /
            out-of-range as NaN) and ``lats`` / ``lons`` are float64 1-D
            coordinate arrays. The categorical aerosol type is carried as float32
            with NaN fill here; its ``int16`` product dtype is applied at
            product-write time (the same convention IGBP uses).
        """
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(self._file_path), "r") as ds:
            # Root-level coordinate arrays: (180,) and (360,) respectively.
            lats_full = np.array(ds.variables[_LAT_VAR][:], dtype=np.float64)
            lons_full = np.array(ds.variables[_LON_VAR][:], dtype=np.float64)

            # Both fields are stored (Latitude, Longitude) — no transpose needed.
            aod = np.array(ds.variables[_AOD_VARIABLE][:], dtype=np.float32)
            aerosol_type = np.array(ds.variables[_AEROSOL_TYPE_VARIABLE][:], dtype=np.float32)

        # AOD: replace fill (−999.0) and out-of-range (>5 or <0) values with NaN.
        aod[aod <= _FILL_VALUE] = np.nan
        aod[(aod < 0.0) | (aod > _AOD_VALID_MAX)] = np.nan

        # Aerosol type: replace fill (−999) and anything outside the valid
        # category range [0, 7] with NaN so the weighted_mode aggregation never
        # selects fill as the modal class.
        aerosol_type[aerosol_type <= _FILL_VALUE] = np.nan
        aerosol_type[(aerosol_type < 0.0) | (aerosol_type > _N_AEROSOL_TYPES - 1)] = np.nan

        # Stack in VARIABLES order: axis 0 = [aod_550, aerosol_type].
        data = np.stack([aod, aerosol_type], axis=0)  # (2, n_lat_full, n_lon_full)
        return data, lats_full, lons_full

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slice the cached NOAA-20 VIIRS aerosol grid to ``bbox``.

        Subsets the full grid from :attr:`_native_grid` to the requested bounding
        box (ascending latitude, no transpose needed).

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(2, n_lat, n_lon)`` with axis 0 = [aod_550, aerosol_type]. Fill
            pixels (originally −999, or AOD outside [0, 5] / type outside [0, 7])
            are returned as NaN.
        """
        data_full, lats_full, lons_full = self._native_grid

        # Compute bbox index masks on the full coordinate arrays.
        lat_mask = (lats_full >= bbox.lat_min) & (lats_full <= bbox.lat_max)
        lon_mask = (lons_full >= bbox.lon_min) & (lons_full <= bbox.lon_max)

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        if lat_indices.size == 0 or lon_indices.size == 0:
            n = len(self.VARIABLES)
            return (
                np.empty((n, 0, 0), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        data = data_full[:, lat_indices, :][:, :, lon_indices]
        return data, lats_full[lat_indices], lons_full[lon_indices]
