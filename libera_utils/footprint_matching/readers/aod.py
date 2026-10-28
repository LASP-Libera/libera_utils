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
FMATCH-IMAGER carries both **AOD** and **aerosol type**. The single-sensor Deep
Blue granule carries both, so this one reader serves both:

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
The AOD is written to the FMATCH-IMAGER products, and used by L2 Surface Irradiance algorithms.
The L2 algorithms that read it there need a single, well-characterized VIIRS aerosol source. The
dedicated single-sensor ``AERDB_D3_VIIRS_NOAA20`` Deep Blue granule provides the
NOAA-20 VIIRS Deep Blue 550 nm AOD **and** the aerosol-type field, so both are
sourced from this one authoritative granule. The cross-sensor
``AERDB_D3_GEOLEO_Merged`` granule only re-bundles the per-sensor L3s and its AOD
lacks the aerosol-type field.

NetCDF4 layout (AERDB_D3_VIIRS_NOAA20)
--------------------------------------
Root (no groups)::

  Latitude_1D  (180,)  — 1° bin centers, −89.5 → 89.5 (ascending)
  Longitude_1D (360,)  — 1° bin centers, −179.5 → 179.5
  Aerosol_Optical_Thickness_550_Land_Ocean_Mean (180, 360)  — daily-mean AOD @ 550 nm
  Aerosol_Type_Land_Ocean_Mode                  (180, 360)  — modal aerosol type (0..7)
  Aerosol_Type_Land_Ocean_Histogram    (8, 180, 360)  — per-type counts (not read here)

Each field's own ``_FillValue`` (−999) and ``valid_range`` (AOD ``[0, 10]``, aerosol
type ``[0, 7]``) are read from the file and applied once; values outside them are
NaN. A field without either attribute is rejected.

References
----------
Deep Blue aerosol products:
    https://earthdata.nasa.gov/sensors/viirs  (Deep Blue / SOAR algorithms)
NASA Deep Blue:
    https://deepblue.gsfc.nasa.gov/
File naming:
    AERDB_D3_VIIRS_NOAA20.A{YYYYDDD}.{collection}.{YYYYDDDHHMMSS}.nc
"""

import netCDF4
import numpy as np

from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import GridSamples, OperationalMode, TimeMatch, VariableSpec

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

# Aerosol type has 8 categories, coded 0..7 (see module docstring). Stored in the
# VariableSpec so the PSF aggregation engine knows how many category bins to
# allocate for the modal / ranked-type histograms.
_N_AEROSOL_TYPES: int = 8


class VIIRSAODReader(StaticReader):
    """Read NOAA-20 VIIRS Deep Blue AOD + aerosol type from an AERDB_D3_VIIRS_NOAA20 file.

    Loads two variables -- ``aod_550`` (continuous) and ``aerosol_type``
    (categorical) -- from a single-sensor Deep Blue Level-3 daily file as a
    ``(2, n_lat, n_lon)`` grid stacked in ``VARIABLES`` order. The file is a daily
    mean with no time axis, so the staged day is the time match
    (``TIME_MATCH = DAILY``).

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
    file_paths : Path or Sequence[Path]
        The one AERDB_D3_VIIRS_NOAA20 NetCDF4 file for the processing day.
    """

    READER_KEY: str = "viirs_aod"
    # Deep Blue single-sensor product for VIIRS aboard NOAA-20 (JPSS-1).
    # TODO[LIBSDC-899]: decide whether NOAA-22 VIIRS replaces NOAA-20 here (a cutover of this reader) or gets a
    # second reader.
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 111.0
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="aod_550",
            dtype="float32",
            # AOD is approximately log-normally distributed, so a geometric
            # (log) mean is the appropriate spatial aggregation.
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
    # A footprint spans several aerosol types. ``aerosol_type`` above is the single
    # aggregated result; these three report the ranked mix (1st/2nd/3rd most common by
    # PSF-weighted area), computed during PSF aggregation from the same modal field rather
    # than read from a source field -- hence here rather than in VARIABLES, mirroring the
    # IGBP reader's ranked surface types.
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

    def _load_native(self) -> GridSamples:
        """Read the full NOAA-20 VIIRS aerosol grid once.

        Opens the NetCDF4 file, reads the root coordinate arrays (ascending
        latitude, no transpose needed) and both the AOD and aerosol-type fields,
        replaces each field's fill and out-of-``valid_range`` values with NaN, and
        stacks the two fields in ``VARIABLES`` order.

        Returns
        -------
        GridSamples
            ``data`` is float32 shape ``(2, n_lat, n_lon)`` with axis 0 =
            [aod_550, aerosol_type] (fill / out-of-range as NaN). The categorical
            aerosol type is carried as float32 with NaN fill here; its ``int16``
            product dtype is applied at product-write time (the same convention
            IGBP uses).

        Raises
        ------
        ValueError
            If either field lacks a ``_FillValue`` or ``valid_range`` attribute, or the
            aerosol-type ``valid_range`` admits a code outside the 8 categories (0..7).
        """
        with netCDF4.Dataset(str(self.single_file_path), "r") as ds:
            # Root-level coordinate arrays: (180,) and (360,) respectively.
            lats_full = np.array(ds.variables[_LAT_VAR][:], dtype=np.float64)
            lons_full = np.array(ds.variables[_LON_VAR][:], dtype=np.float64)

            # Both fields are stored (Latitude, Longitude) -- no transpose needed.
            aod = self._valid_values(ds.variables[_AOD_VARIABLE])
            type_variable = ds.variables[_AEROSOL_TYPE_VARIABLE]
            aerosol_type = self._valid_values(type_variable)
            type_range = np.asarray(type_variable.valid_range)

        # The modal / ranked-type histograms have one bin per category, so a code the
        # file calls valid must still be one of the 8 types.
        if type_range.min() < 0 or type_range.max() > _N_AEROSOL_TYPES - 1:
            raise ValueError(
                f"{self.single_file_path}: {_AEROSOL_TYPE_VARIABLE} valid_range is {type_range.tolist()}, but this "
                f"reader maps aerosol types 0..{_N_AEROSOL_TYPES - 1}. The product's category set has changed; "
                "update the reader's aerosol-type categories before reading it."
            )

        # Stack in VARIABLES order: axis 0 = [aod_550, aerosol_type].
        data = np.stack([aod, aerosol_type], axis=0)  # (2, n_lat_full, n_lon_full)
        return GridSamples(lat_axis=lats_full, lon_axis=lons_full, data=data)

    def _valid_values(self, variable: netCDF4.Variable) -> np.ndarray:
        """Return one field as float32 with its own fill and out-of-range values as NaN.

        Raises
        ------
        ValueError
            If the field lacks a ``_FillValue`` or ``valid_range`` attribute.
        """
        missing = [attr for attr in ("_FillValue", "valid_range") if not hasattr(variable, attr)]
        if missing:
            raise ValueError(
                f"{self.single_file_path}: {variable.name} has no {' or '.join(missing)} attribute. The Deep Blue "
                "AERDB_D3_VIIRS_NOAA20 granule carries both on every field; stage the granule as distributed."
            )
        # netCDF4's automatic masking applies the field's own _FillValue and valid_range,
        # compared in the stored dtype before the cast.
        return np.ma.asarray(variable[:]).astype(np.float32).filled(np.nan)
