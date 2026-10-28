"""ERA5 single-level reader plugin for the footprint matching pipeline.

Data source: ECMWF ERA5 Reanalysis (single-level surface fields)

- Format: NetCDF4 (.nc), distributed via the Copernicus Climate Data Store
- Spatial resolution: ~0.25° (~28 km), stored as 28 km in RESOLUTION_KM
- Grid: Regular lat/lon, global coverage, latitudes in DESCENDING order (90 → -90)
- Temporal resolution: Hourly
- Temporal coverage: 1940-present. ERA5 final data lags real time by ~5 days
  (the preliminary ERA5T stream lags ~1 day).

Variables read
--------------
Wind components (every product, from mission start):

- u10 / v10: 10 m U/V wind components

Additional single-level fields, carried by FMATCH-IMAGER and FMATCH-IMAGER-CAMTIME
alongside the RBSP CLDPIX/SSF cloud fields:

- t2m: 2 m temperature
- d2m: 2 m dewpoint temperature
- sp:  surface pressure
- z:   geopotential (surface; orography × g)
- fal: forecast albedo

References
----------
CDS dataset:  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
CDS API docs: https://cds.climate.copernicus.eu/how-to-api
Variable list:https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
Param DB (id): u10=165, v10=166, t2m=167, d2m=168, sp=134, z=129, fal=243,
e.g. https://apps.ecmwf.int/codes/grib/param-db?id=165
"""

import abc
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.readers._longitude import normalize_longitude
from libera_utils.footprint_matching.readers.base import SteppedGridReader
from libera_utils.footprint_matching.types import GridSamples, OperationalMode, VariableSpec

# The products that carry the winds: every FMATCH product, from mission start.
_WIND_MODES: frozenset[OperationalMode] = frozenset(OperationalMode)
# The products that carry the other single-level fields, alongside the RBSP CLDPIX/SSF
# cloud fields.
_IMAGER_MODES: frozenset[OperationalMode] = frozenset({OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME})

# Each output variable, the ERA5 variable it reads as stored in CDS NetCDF4 files, and
# the products that carry it. This order is VARIABLES order and axis 0 of the reader's
# data array.
_ERA5_SURFACE_VARIABLES: tuple[tuple[str, str, frozenset[OperationalMode]], ...] = (
    ("wind_u10", "u10", _WIND_MODES),
    ("wind_v10", "v10", _WIND_MODES),
    ("temperature_2m", "t2m", _IMAGER_MODES),
    ("dew_point_temperature_2m", "d2m", _IMAGER_MODES),
    ("surface_pressure", "sp", _IMAGER_MODES),
    ("surface_geopotential", "z", _IMAGER_MODES),
    ("forecast_albedo", "fal", _IMAGER_MODES),
)


class ERA5ReaderBase(SteppedGridReader, abc.ABC):
    """Shared machinery for the ERA5 reader family (single-level and pressure-level).

    Both ERA5 CDS datasets share the same file conventions: a regular global
    lat/lon grid with descending latitudes, longitudes in either -180..180 or
    0..360 convention, and a leading time dimension (``time`` or the newer
    ``valid_time``) holding hourly steps. This intermediate class owns:

    * the step index across the staged granules (:meth:`_step_times`), and
    * loading one step as a :class:`~libera_utils.footprint_matching.types.GridSamples` (:meth:`_load_step`), with the
      longitude normalization and latitude flip both datasets need.

    A footprint is served the hourly step nearest its time
    (``TIME_MATCH = NEAREST_STEP``), with no interpolation between steps, and only
    that step is held in memory. Subclasses implement :meth:`_read_step_layers` to
    pull their variables out of one time step of an open dataset.

    This class is abstract, so ``AncillaryReader.__init_subclass__`` skips it
    during reader registration; only the concrete subclasses register.
    """

    # The hourly step nearest the footprint time is served; a footprint more than one
    # step interval from every staged step gets no ERA5 samples (coverage 0, flagged).
    MAX_TIME_OFFSET: np.timedelta64 | None = np.timedelta64(1, "h")

    def __init__(self, file_paths: Sequence[Path]) -> None:
        super().__init__(file_paths)
        # (granule path, index along that granule's time dimension) per global step,
        # filled by _step_times in ascending time order.
        self._step_locations: list[tuple[Path, int]] = []

    @staticmethod
    def _time_dimension(ds: xr.Dataset, path: Path) -> str:
        """Return the name of the dataset's time dimension (``time`` or ``valid_time``).

        Raises
        ------
        ValueError
            If the file has no time-like dimension: an ERA5 CDS download always carries
            one, and without it a step cannot be matched to a footprint time.
        """
        time_dims = [d for d in ds.dims if "time" in d]
        if len(time_dims) != 1:
            raise ValueError(
                f"ERA5 file {path} must have exactly one time dimension ('time' or 'valid_time'); "
                f"found {time_dims or 'none'}. Re-download the period from the CDS with its time axis."
            )
        return time_dims[0]

    def _step_times(self) -> np.ndarray:
        """Collect every hourly step across the staged granules, ascending.

        Also records where each step lives (granule, local index) for
        :meth:`_load_step`.
        """
        located: list[tuple[np.datetime64, Path, int]] = []
        for path in self.file_paths:
            with xr.open_dataset(path, engine="netcdf4") as ds:
                time_dim = self._time_dimension(ds, path)
                times = np.asarray(ds[time_dim].values).astype("datetime64[ns]")
            located.extend((t, path, j) for j, t in enumerate(times))
        located.sort(key=lambda item: item[0])
        self._step_locations = [(path, j) for _, path, j in located]
        return np.array([t for t, _, _ in located], dtype="datetime64[ns]")

    def _load_step(self, index: int) -> GridSamples:
        """Load one hourly step as a global grid, longitudes -180..180, latitudes ascending."""
        # _steps (the cached step list) populates _step_locations; touch it first so a
        # direct call on a fresh instance still resolves.
        _ = self._steps
        path, local_index = self._step_locations[index]
        # cache=False: xarray would otherwise keep every plane read below alive until close,
        # a second copy of the step beside ``data``.
        ds = xr.open_dataset(path, engine="netcdf4", cache=False)
        try:
            ds = self._normalize_longitudes(ds)
            time_dim = self._time_dimension(ds, path)
            step = ds.isel({time_dim: local_index})
            lats = step["latitude"].values.astype(np.float64)
            lons = step["longitude"].values.astype(np.float64)
            # One step-sized array, filled plane by plane (a 185-plane pressure step is
            # ~0.8 GB, so it is never built twice).
            data = np.empty((self.n_variables, lats.size, lons.size), dtype=np.float32)
            self._read_step_layers(step, path, data)
        finally:
            ds.close()
        # ERA5 stores latitude descending (90 -> -90); serve it ascending, as a view.
        if lats.size > 1 and lats[0] > lats[-1]:
            lats = lats[::-1]
            data = data[:, ::-1, :]
        return GridSamples(lat_axis=lats, lon_axis=lons, data=data)

    @abc.abstractmethod
    def _read_step_layers(self, step: xr.Dataset, path: Path, out: np.ndarray) -> None:
        """Read this dataset's variables from one time step into ``out``.

        Parameters
        ----------
        step : xr.Dataset
            The open dataset already selected to one time step and with longitudes
            normalized to -180..180.
        path : Path
            The granule being read, for error messages.
        out : np.ndarray
            float32 array of shape ``(n_variables, n_lat, n_lon)``, latitude as stored
            in the file (descending). Plane ``i`` receives ``VARIABLES[i]``.
        """

    @staticmethod
    def _normalize_longitudes(ds: xr.Dataset) -> xr.Dataset:
        """Normalize a CDS dataset's longitude coordinate to -180..180.

        ERA5 files from the CDS can use either -180..180 or 0..360 longitude
        convention depending on how the download was configured. Normalizing to
        -180..180 means the box query (which always uses -180..180) works
        correctly for both.
        """
        if float(ds["longitude"].min()) >= 0:
            ds = ds.assign_coords(longitude=normalize_longitude(ds["longitude"].values)).sortby("longitude")
        return ds


class ERA5SurfaceReader(ERA5ReaderBase):
    """Read ERA5 single-level fields from a CDS NetCDF4 file.

    Serves one hourly step at a time as a ``(7, n_lat, n_lon)`` grid where axis 0
    corresponds to ``_ERA5_SURFACE_VARIABLES`` / ``VARIABLES`` order.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"era5_surface"``.
    RESOLUTION_KM : float
        28 km (the 0.25 degree ERA5 grid spacing at the equator).
    VARIABLES : tuple[VariableSpec, ...]
        Seven continuous float32 variables. The winds (``wind_u10``/``wind_v10``)
        are carried by every product; the five additional single-level fields only
        by FMATCH-IMAGER and FMATCH-IMAGER-CAMTIME. So ``era5_surface`` feeds every
        product but contributes a different variable subset to each.

    Parameters
    ----------
    file_paths : Sequence[Path]
        The ERA5 single-levels NetCDF4 granule(s) covering the processing period, each
        containing all the variables in ``_ERA5_SURFACE_VARIABLES`` and an hourly
        time axis.
    """

    READER_KEY: str = "era5_surface"
    # ERA5 is a reanalysis (no single instrument); use the producing center so the
    # long_name provenance tag "... (ECMWF)" stays uniform across every source.
    INSTRUMENT: str = "ECMWF"
    # The 0.25 degree grid spacing at the equator. A sample covers a cell of this side in
    # the coverage measure, so the cells abut and a footprint inside the grid is complete.
    RESOLUTION_KM: float = 28.0
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=name,
            dtype="float32",
            aggregation="weighted_mean",
            modes=modes,
            n_categories=None,
        )
        for name, _, modes in _ERA5_SURFACE_VARIABLES
    )

    def _read_step_layers(self, step: xr.Dataset, path: Path, out: np.ndarray) -> None:
        """Read every single-level variable from one time step into ``out``.

        Raises
        ------
        KeyError
            If the file is missing any of the expected ERA5 variables (e.g. a
            download configured with only a subset of the required fields).
        """
        missing = [nc_name for _, nc_name, _ in _ERA5_SURFACE_VARIABLES if nc_name not in step]
        if missing:
            raise KeyError(
                f"ERA5 single-levels file {path} is missing variable(s) {missing}. "
                f"The CDS download must include every variable in "
                f"{[nc for _, nc, _ in _ERA5_SURFACE_VARIABLES]}."
            )
        for i, (_, nc_name, _) in enumerate(_ERA5_SURFACE_VARIABLES):
            out[i] = step[nc_name].values
