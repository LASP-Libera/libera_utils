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

Additional FMATCH-IMAGER single-level fields (``required_mode=IMAGER``), which
sit in the FMATCH-IMAGER product alongside the RBSP CLDPIX/SSF cloud fields:
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
Param DB (id): u10=165, v10=166, t2m=167, d2m=168, sp=134, z=129, fal=243
              e.g. https://apps.ecmwf.int/codes/grib/param-db?id=165
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.readers.base import SteppedGridReader
from libera_utils.footprint_matching.types import GridSamples, OperationalMode, VariableSpec

# Ordered mapping of FMATCH spec name -> ERA5 variable name as stored in CDS
# NetCDF4 files. The order here defines axis 0 of the reader's data array and
# MUST match the order of the VARIABLES tuple below (guarded by a unit test).
_ERA5_SINGLE_LEVEL_VARIABLES: tuple[tuple[str, str], ...] = (
    ("wind_u10", "u10"),
    ("wind_v10", "v10"),
    ("temperature_2m", "t2m"),
    ("dew_point_temperature_2m", "d2m"),
    ("surface_pressure", "sp"),
    ("surface_geopotential", "z"),
    ("forecast_albedo", "fal"),
)

# The ERA5 latitude dimension is stored in DESCENDING order (90° → -90°) in
# files downloaded from the CDS. xarray.sel() works correctly regardless of
# direction when using slice(max_lat, min_lat) ordering, but we must be careful
# to reverse the output lats array so that it is ASCENDING for downstream callers.
# This constant documents the known direction of the ERA5 lat coordinate.
_ERA5_LAT_DESCENDING: bool = True


class ERA5ReaderBase(SteppedGridReader, abc.ABC):
    """Shared machinery for the ERA5 reader family (single-level and pressure-level).

    Both ERA5 CDS datasets share the same file conventions: a regular global
    lat/lon grid with descending latitudes, longitudes in either -180..180 or
    0..360 convention, and a leading time dimension (``time`` or the newer
    ``valid_time``) holding hourly steps. This intermediate class owns:

    * the step index across the staged granules (:meth:`_step_times`), and
    * loading one step as a :class:`GridSamples` (:meth:`_load_step`), with the
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

    def __init__(self, file_paths: Path | Sequence[Path]) -> None:
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
        ds = xr.open_dataset(path, engine="netcdf4")
        try:
            ds = self._normalize_longitudes(ds)
            time_dim = self._time_dimension(ds, path)
            lats, lons, layers = self._read_step_layers(ds.isel({time_dim: local_index}), path)
        finally:
            ds.close()
        lats, layers = self._flip_lats_ascending(lats, layers)
        data = np.stack(layers, axis=0).astype(np.float32, copy=False)
        return GridSamples(lat_axis=lats, lon_axis=lons, data=data)

    @abc.abstractmethod
    def _read_step_layers(self, step: xr.Dataset, path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Read this dataset's variables from one time step.

        Parameters
        ----------
        step : xr.Dataset
            The open dataset already selected to one time step and with longitudes
            normalized to -180..180.
        path : Path
            The granule being read, for error messages.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, list[np.ndarray]]
            ``(lats, lons, layers)``: the file's latitude axis (as stored, descending),
            its longitude axis, and one ``(n_lat, n_lon)`` float32 layer per
            ``VARIABLES`` entry, in order.
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
            ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
        return ds

    @staticmethod
    def _flip_lats_ascending(lats: np.ndarray, layers: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return lats (and each 2-D layer) flipped to ASCENDING latitude order.

        ERA5 stores latitudes descending (see ``_ERA5_LAT_DESCENDING``); every
        other reader emits ascending latitudes, so we flip here for consistency.
        No-op if the coordinate is already ascending.
        """
        if _ERA5_LAT_DESCENDING and lats.size > 1 and lats[0] > lats[-1]:
            lats = lats[::-1]
            layers = [layer[::-1, :] for layer in layers]
        return lats, layers


class ERA5Reader(ERA5ReaderBase):
    """Read ERA5 single-level fields from a CDS NetCDF4 file.

    Serves one hourly step at a time as a ``(7, n_lat, n_lon)`` grid where axis 0
    corresponds to ``_ERA5_SINGLE_LEVEL_VARIABLES`` / ``VARIABLES`` order.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"era5"``.
    RESOLUTION_KM : float
        28 km (the 0.25 degree ERA5 grid spacing at the equator).
    VARIABLES : tuple[VariableSpec, ...]
        Seven continuous float32 variables. The winds (``wind_u10``/``wind_v10``)
        are unrestricted (``required_mode=CAM``); the five additional single-level
        fields carry per-spec ``required_mode=IMAGER`` gating, so they appear only in
        the FMATCH-IMAGER-family products. This per-spec latency split is why ``era5``
        belongs to every product's reader set yet contributes a different variable
        subset to the CAM vs IMAGER products.

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The ERA5 single-levels NetCDF4 granule(s) covering the processing period, each
        containing all the variables in ``_ERA5_SINGLE_LEVEL_VARIABLES`` and an hourly
        time axis.
    """

    READER_KEY: str = "era5"
    # ERA5 is a reanalysis (no single instrument); use the producing center so the
    # long_name provenance tag "... (ECMWF)" stays uniform across every source.
    INSTRUMENT: str = "ECMWF"
    # The 0.25 degree grid spacing at the equator. A sample covers a cell of this side in
    # the coverage measure, so the cells abut and a footprint inside the grid is complete.
    RESOLUTION_KM: float = 28.0
    VARIABLES: tuple[VariableSpec, ...] = (
        # --- Wind components: every product, from mission start ---
        VariableSpec(
            name="wind_u10",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="wind_v10",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        # --- Additional FMATCH-IMAGER single-level fields (beyond the winds).
        # Gated per-spec with required_mode=IMAGER, so they flow to the FMATCH-IMAGER
        # product while staying out of the lower-latency CAM products. They sit in the
        # product alongside the RBSP CLDPIX/SSF cloud fields.
        VariableSpec(
            name="temperature_2m",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
        VariableSpec(
            name="dew_point_temperature_2m",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
        VariableSpec(
            name="surface_pressure",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
        VariableSpec(
            name="surface_geopotential",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
        VariableSpec(
            name="forecast_albedo",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
    )

    def _read_step_layers(self, step: xr.Dataset, path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Read every single-level variable from one time step.

        Raises
        ------
        KeyError
            If the file is missing any of the expected ERA5 variables (e.g. a
            download configured with only a subset of the required fields).
        """
        missing = [nc_name for _, nc_name in _ERA5_SINGLE_LEVEL_VARIABLES if nc_name not in step]
        if missing:
            raise KeyError(
                f"ERA5 single-levels file {path} is missing variable(s) {missing}. "
                f"The CDS download must include every variable in "
                f"{[nc for _, nc in _ERA5_SINGLE_LEVEL_VARIABLES]}."
            )
        lats = step["latitude"].values.astype(np.float64)
        lons = step["longitude"].values.astype(np.float64)
        layers = [step[nc_name].values.astype(np.float32) for _, nc_name in _ERA5_SINGLE_LEVEL_VARIABLES]
        return lats, lons, layers
