"""ERA5 pressure-level reader plugin for the footprint matching pipeline.

Data source: ECMWF ERA5 Reanalysis (pressure-level fields)
- Format: NetCDF4 (.nc), distributed via the Copernicus Climate Data Store
- Spatial resolution: ~0.25° (~28 km); stored as 25 km in RESOLUTION_KM
- Grid: Regular lat/lon, global coverage, latitudes in DESCENDING order (90 → -90),
  plus a ``pressure_level`` dimension (up to 37 levels, 1–1000 hPa)
- Temporal resolution: Hourly
- Temporal coverage: 1940-present (~5 day latency for final data; ~1 day for ERA5T)

This is a *separate CDS dataset* from the single-level fields, downloaded as a
separate file, hence a separate reader from
:class:`~libera_utils.footprint_matching.readers.era5.ERA5Reader` (each
``GriddedDataReader`` reads exactly one file).

Why per-level variables?
------------------------
FMATCH products are strictly 1-D per footprint (every product variable lives on
the single time dimension) and ``GridTile.data`` is at most
``(n_var, n_lat, n_lon)``. So instead of adding a vertical dimension, each
(variable, pressure level) pair is flattened into its own ``VariableSpec`` —
e.g. ``temperature_500hPa`` — keeping the tile/aggregation contract and the
product schema unchanged. The retained levels are the ``_ERA5_PRESSURE_LEVELS``
module constant; changing the science-selected subset is a one-line edit there
(the specs and product-definition cross-check derive from it).

Variables read (at each retained pressure level)
------------------------------------------------
- t:  temperature (K)
- z:  geopotential (m² s⁻²)
- o3: ozone mass mixing ratio (kg kg⁻¹)
- q:  specific humidity (kg kg⁻¹)
- r:  relative humidity (%)

All of these began as year-one FMATCH-IMAGER substitutes for the then-unavailable
RBSP inputs. They are now retained in the post-year-one product alongside the RBSP
fields, so the reader is variant-neutral (``REQUIRED_VARIANT`` inherits the base
default of ``None``); it is active in every FMATCH-IMAGER variant, gated only by
``REQUIRED_MODE = IMAGER``.

References
----------
CDS dataset:  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
CDS API docs: https://cds.climate.copernicus.eu/how-to-api
Variable list:https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
Param DB (id): t=130, z=129, o3=203, q=133, r=157
              e.g. https://apps.ecmwf.int/codes/grib/param-db?id=130
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.readers.era5 import ERA5ReaderBase
from libera_utils.footprint_matching.types import OperationalMode, VariableSpec

# Pressure levels (hPa) retained for FMATCH, ascending. ERA5 offers 37 levels
# (1–1000 hPa); carrying all of them would add 5 × 37 × 2 = 370 product
# variables, so the science team selected this 13-level subset. The CDS
# download AND this constant must agree: the reader raises if a configured
# level is missing from the file.
_ERA5_PRESSURE_LEVELS: tuple[int, ...] = (10, 30, 50, 70, 100, 200, 250, 300, 500, 700, 850, 925, 1000)

# Ordered mapping of FMATCH base spec name -> ERA5 variable name as stored in
# CDS NetCDF4 files. Combined with _ERA5_PRESSURE_LEVELS (variable-major, then
# ascending level) to produce one spec per (variable, level) pair; that combined
# order defines axis 0 of the reader's data array.
_ERA5_PRESSURE_LEVEL_VARIABLES: tuple[tuple[str, str], ...] = (
    ("temperature", "t"),
    ("geopotential", "z"),
    ("ozone_mass_mixing_ratio", "o3"),
    ("specific_humidity", "q"),
    ("relative_humidity", "r"),
)

# Name of the vertical coordinate in CDS pressure-level files (hPa values).
_PRESSURE_LEVEL_COORD: str = "pressure_level"


def _build_pressure_level_specs() -> tuple[VariableSpec, ...]:
    """Generate one VariableSpec per (variable, pressure level) pair.

    Variable-major, levels ascending — e.g. ``temperature_10hPa`` ...
    ``temperature_1000hPa``, then ``geopotential_10hPa`` ... — matching the
    stacking order in ``ERA5PressureLevelReader._read_native_grid``. Every field
    is continuous (weighted-mean aggregated), so each also gains an automatic
    ``_standard_deviation`` companion in ``product_variable_specs()``.
    """
    return tuple(
        VariableSpec(
            name=f"{base_name}_{level}hPa",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        )
        for base_name, _ in _ERA5_PRESSURE_LEVEL_VARIABLES
        for level in _ERA5_PRESSURE_LEVELS
    )


class ERA5PressureLevelReader(ERA5ReaderBase):
    """Read ERA5 pressure-level fields from a CDS NetCDF4 file.

    Returns a 3-D data array of shape ``(n_specs, n_lat, n_lon)`` where
    ``n_specs = 5 variables × len(_ERA5_PRESSURE_LEVELS)`` and axis 0 matches
    the ``VARIABLES`` order (variable-major, levels ascending).

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"era5_pressure"``.
    RESOLUTION_KM : float
        25 km (ERA5 native ~28 km, rounded to 25 km for PSF calculations).
    REQUIRED_MODE : OperationalMode
        ``IMAGER`` — these fields exist only for the FMATCH-IMAGER product.
    REQUIRED_VARIANT : FmatchVariant or None
        Inherited base default ``None`` (variant-neutral): the reader runs in both
        the year-one and post-year-one FMATCH-IMAGER variants. It originally
        substituted for RBSP inputs during year one and is now retained
        post-year-one alongside CLDPIX/SSF (see ``fmatch_imager_post_year_one.yml``).
    VARIABLES : tuple[VariableSpec, ...]
        One continuous float32 spec per (variable, level) pair.

    Parameters
    ----------
    file_path : Path
        Path to an ERA5 pressure-levels NetCDF4 file containing ``t``, ``z``,
        ``o3``, ``q``, and ``r`` at (at least) every level in
        ``_ERA5_PRESSURE_LEVELS``.
    """

    READER_KEY: str = "era5_pressure"
    # Same producing-center token as the single-level reader so product variable
    # names stay uniform: era5_pressure_ECMWF_temperature_500hPa etc.
    INSTRUMENT: str = "ECMWF"
    RESOLUTION_KM: float = 25.0
    REQUIRED_MODE: OperationalMode = OperationalMode.IMAGER
    # REQUIRED_VARIANT is intentionally NOT overridden: it inherits the base class
    # default of None (variant-neutral). These fields began as year-one substitutes
    # for the unavailable RBSP inputs but are now retained in the post-year-one
    # FMATCH-IMAGER product alongside the RBSP fields, so the reader runs in both
    # variants (subject only to the REQUIRED_MODE=IMAGER rank gate).
    VARIABLES: tuple[VariableSpec, ...] = _build_pressure_level_specs()

    def _read_native_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read every (variable, level) layer from the file into one stacked grid.

        Opens the NetCDF4 file, normalizes longitude to −180..180, drops any
        time-like dimension (first step only), selects each configured pressure
        level, and returns the whole global grid with latitudes ASCENDING.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` with ``data`` float32 shape
            ``(n_specs, n_lat, n_lon)`` — axis 0 in ``VARIABLES`` order.

        Raises
        ------
        KeyError
            If the file is missing any of the expected ERA5 variables.
        ValueError
            If the file does not contain every configured pressure level. A
            partial download must be fixed at acquisition time — silently
            substituting a neighboring level would corrupt the product.
        """
        ds = xr.open_dataset(self._file_path, engine="netcdf4")
        try:
            ds = self._normalize_longitudes(ds)

            missing_vars = [nc_name for _, nc_name in _ERA5_PRESSURE_LEVEL_VARIABLES if nc_name not in ds]
            if missing_vars:
                raise KeyError(
                    f"ERA5 pressure-levels file {self._file_path} is missing variable(s) {missing_vars}. "
                    f"The CDS download must include every variable in "
                    f"{[nc for _, nc in _ERA5_PRESSURE_LEVEL_VARIABLES]}."
                )

            # Validate the level set up front for one clear, complete error message
            # rather than failing on the first .sel() miss.
            available_levels = ds[_PRESSURE_LEVEL_COORD].values.astype(np.float64)
            missing_levels = [level for level in _ERA5_PRESSURE_LEVELS if not np.any(available_levels == level)]
            if missing_levels:
                raise ValueError(
                    f"ERA5 pressure-levels file {self._file_path} is missing configured pressure "
                    f"level(s) {missing_levels} hPa. File contains {sorted(available_levels.tolist())}; "
                    f"FMATCH requires {list(_ERA5_PRESSURE_LEVELS)} (see _ERA5_PRESSURE_LEVELS)."
                )

            layers: list[np.ndarray] = []
            lats: np.ndarray | None = None
            lons: np.ndarray | None = None
            # Variable-major, levels ascending — must mirror _build_pressure_level_specs.
            for _, nc_name in _ERA5_PRESSURE_LEVEL_VARIABLES:
                da = self._first_time_step(ds[nc_name])
                if lats is None:
                    # All variables share the same grid; read the coordinates once.
                    lats = da["latitude"].values.astype(np.float64)
                    lons = da["longitude"].values.astype(np.float64)
                for level in _ERA5_PRESSURE_LEVELS:
                    layers.append(da.sel({_PRESSURE_LEVEL_COORD: level}).values.astype(np.float32))

            lats, layers = self._flip_lats_ascending(lats, layers)

            # Stack into (n_specs, n_lat, n_lon) — axis 0 matches VARIABLES ordering.
            data = np.stack(layers, axis=0)
        finally:
            ds.close()

        return data, lats, lons
