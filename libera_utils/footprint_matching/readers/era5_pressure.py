"""ERA5 pressure-level reader plugin for the footprint matching pipeline.

Data source: ECMWF ERA5 Reanalysis (pressure-level fields)
- Format: NetCDF4 (.nc), distributed via the Copernicus Climate Data Store
- Spatial resolution: ~0.25° (~28 km), stored as 28 km in RESOLUTION_KM
- Grid: Regular lat/lon, global coverage, latitudes in DESCENDING order (90 → -90),
  plus a ``pressure_level`` dimension (up to 37 levels, 1–1000 hPa)
- Temporal resolution: Hourly
- Temporal coverage: 1940-present (~5 day latency for final data; ~1 day for ERA5T)

This is a *separate CDS dataset* from the single-level fields, downloaded as a
separate file, hence a separate reader from
:class:`~libera_utils.footprint_matching.readers.era5_surface.ERA5SurfaceReader` (each reader
reads one source).

Why per-level variables?
------------------------
FMATCH products are strictly 1-D per footprint (every product variable lives on
the single time dimension) and a reader's samples carry one value row per
variable. So instead of adding a vertical dimension, each (variable, pressure
level) pair is flattened into its own ``VariableSpec`` -- e.g.
``temperature_500hPa`` -- keeping the sample/aggregation contract and the
product schema unchanged. The retained levels are the ``_ERA5_PRESSURE_LEVELS``
module constant — the full ERA5 set of 37 levels; changing the level list is a
one-line edit there (the specs and product-definition cross-check derive from it).

Variables read (at each retained pressure level)
------------------------------------------------
- t:  temperature (K)
- z:  geopotential (m² s⁻²)
- o3: ozone mass mixing ratio (kg kg⁻¹)
- q:  specific humidity (kg kg⁻¹)
- r:  relative humidity (%)

These pressure-level fields sit in the radiometer-timescale FMATCH-IMAGER product
alongside the RBSP CLDPIX/SSF cloud fields. The reader is deliberately absent from
the camera-timescale FMATCH-IMAGER-CAMTIME product: the pressure-level fields are a
radiometer-timescale quantity. Product membership is declared outside the reader.

References
----------
CDS dataset:  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
CDS API docs: https://cds.climate.copernicus.eu/how-to-api
Variable list:https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
Param DB (id): t=130, z=129, o3=203, q=133, r=157
              e.g. https://apps.ecmwf.int/codes/grib/param-db?id=130
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.readers.era5_surface import ERA5ReaderBase
from libera_utils.footprint_matching.types import OperationalMode, VariableSpec

# Pressure levels (hPa) retained for FMATCH, ascending — the full ERA5 set of 37
# levels (1–1000 hPa). Carrying all of them yields 5 × 37 × 2 = 370 product
# variables. The CDS download AND this constant must agree: the reader raises if a
# configured level is missing from the file.
_ERA5_PRESSURE_LEVELS: tuple[int, ...] = (
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
)

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


def _pressure_level_layers() -> Iterator[tuple[str, str, int]]:
    """Yield ``(spec name, ERA5 variable name, level hPa)`` for every reader output layer.

    Variable-major, levels ascending (``temperature_1hPa`` ... ``temperature_1000hPa``,
    then ``geopotential_1hPa`` ...). This is VARIABLES order and the plane order
    ``ERA5PressureLevelReader._read_step_layers`` writes.
    """
    for base_name, nc_name in _ERA5_PRESSURE_LEVEL_VARIABLES:
        for level in _ERA5_PRESSURE_LEVELS:
            yield f"{base_name}_{level}hPa", nc_name, level


def _build_pressure_level_specs() -> tuple[VariableSpec, ...]:
    """Generate one VariableSpec per (variable, pressure level) pair, in :func:`_pressure_level_layers` order.

    Every field is continuous (weighted-mean aggregated), so each also gains an
    automatic ``_standard_deviation`` companion in ``product_variable_specs()``.
    """
    return tuple(
        VariableSpec(
            name=name,
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        )
        for name, _, _ in _pressure_level_layers()
    )


class ERA5PressureLevelReader(ERA5ReaderBase):
    """Read ERA5 pressure-level fields from a CDS NetCDF4 file.

    Serves one hourly step at a time as a ``(n_specs, n_lat, n_lon)`` grid where
    ``n_specs = 5 variables x len(_ERA5_PRESSURE_LEVELS)`` and axis 0 matches
    the ``VARIABLES`` order (variable-major, levels ascending). One step of all 185
    layers is about 0.8 GB, which is why only the step nearest the footprint time
    is held.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"era5_pressure"``.
    RESOLUTION_KM : float
        28 km (the 0.25 degree ERA5 grid spacing at the equator).
    VARIABLES : tuple[VariableSpec, ...]
        One continuous float32 spec per (variable, level) pair.

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The ERA5 pressure-levels NetCDF4 granule(s) covering the processing period,
        each containing ``t``, ``z``, ``o3``, ``q``, and ``r`` at (at least) every
        level in ``_ERA5_PRESSURE_LEVELS`` and an hourly time axis.
    """

    READER_KEY: str = "era5_pressure"
    # Same producing-center token as the single-level reader so the long_name
    # provenance tag stays uniform: "Air temperature at 500 hPa (ECMWF)" etc.
    # Variable names are era5_pressure_temperature_500hPa (no instrument token).
    INSTRUMENT: str = "ECMWF"
    # The 0.25 degree grid spacing at the equator. A sample covers a cell of this side in
    # the coverage measure, so the cells abut and a footprint inside the grid is complete.
    RESOLUTION_KM: float = 28.0
    VARIABLES: tuple[VariableSpec, ...] = _build_pressure_level_specs()

    def _read_step_layers(self, step: xr.Dataset, path: Path, out: np.ndarray) -> None:
        """Read every (variable, level) layer from one time step into ``out``.

        Raises
        ------
        KeyError
            If the file is missing any of the expected ERA5 variables.
        ValueError
            If the file does not contain every configured pressure level. A
            partial download must be fixed at acquisition time -- silently
            substituting a neighboring level would corrupt the product.
        """
        missing_vars = [nc_name for _, nc_name in _ERA5_PRESSURE_LEVEL_VARIABLES if nc_name not in step]
        if missing_vars:
            raise KeyError(
                f"ERA5 pressure-levels file {path} is missing variable(s) {missing_vars}. "
                f"The CDS download must include every variable in "
                f"{[nc for _, nc in _ERA5_PRESSURE_LEVEL_VARIABLES]}."
            )

        # Validate the level set up front for one clear, complete error message
        # rather than failing on the first .sel() miss.
        available_levels = step[_PRESSURE_LEVEL_COORD].values.astype(np.float64)
        missing_levels = [level for level in _ERA5_PRESSURE_LEVELS if not np.any(available_levels == level)]
        if missing_levels:
            raise ValueError(
                f"ERA5 pressure-levels file {path} is missing configured pressure "
                f"level(s) {missing_levels} hPa. File contains {sorted(available_levels.tolist())}; "
                f"FMATCH requires {list(_ERA5_PRESSURE_LEVELS)} (see _ERA5_PRESSURE_LEVELS)."
            )

        for plane, (_, nc_name, level) in enumerate(_pressure_level_layers()):
            out[plane] = step[nc_name].sel({_PRESSURE_LEVEL_COORD: level}).values
