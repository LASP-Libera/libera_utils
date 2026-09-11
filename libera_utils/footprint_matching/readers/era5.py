"""ERA5 single-level reader plugin for the footprint matching pipeline.

Data source: ECMWF ERA5 Reanalysis (single-level surface fields)
- Format: NetCDF4 (.nc), distributed via the Copernicus Climate Data Store
- Spatial resolution: ~0.25° (~28 km); stored as 25 km in RESOLUTION_KM
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
from functools import cached_property

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

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


class ERA5ReaderBase(GriddedDataReader, abc.ABC):
    """Shared machinery for the ERA5 reader family (single-level and pressure-level).

    Both ERA5 CDS datasets share the same file conventions — a regular global
    lat/lon grid with descending latitudes, longitudes in either −180..180 or
    0..360 convention, and a time-like leading dimension (``time`` or the newer
    ``valid_time``). This intermediate class owns:

    * the per-instance cache of the full normalized global grid
      (:attr:`_native_grid`, populated once by the subclass hook
      :meth:`_read_native_grid`), and
    * the inclusive bounding-box slicing of that cached grid
      (:meth:`_load_spatial_region`).

    Subclasses implement :meth:`_read_native_grid` to open their specific CDS
    file layout and return the whole normalized global grid.

    This class is abstract, so ``GriddedDataReader.__init_subclass__`` skips it
    during reader registration; only the concrete subclasses register.
    """

    @cached_property
    def _native_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the full ERA5 grid once and cache it on the instance.

        Cached per reader instance so the file is opened and read once, then
        every tile slices these in-memory arrays (see :meth:`_load_spatial_region`)
        instead of re-opening the file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(n_specs, n_lat, n_lon)`` — axis 0 in ``VARIABLES`` order — and
            ``lats`` (ASCENDING) / ``lons`` are float64 1-D coordinate arrays,
            both sorted ascending.
        """
        return self._read_native_grid()

    @abc.abstractmethod
    def _read_native_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Open the file and return the whole normalized global grid.

        Implementations must return ``(data, lats, lons)`` with data float32
        shape ``(n_specs, n_lat, n_lon)`` (axis 0 matching ``VARIABLES`` order),
        latitudes ASCENDING, and longitudes normalized to −180..180 ascending.
        """

    @staticmethod
    def _normalize_longitudes(ds: xr.Dataset) -> xr.Dataset:
        """Normalize a CDS dataset's longitude coordinate to −180..180.

        ERA5 files from the CDS can use either −180→180 or 0→360 longitude
        convention depending on how the download was configured. Normalizing to
        −180→180 means the bbox slice (which always uses −180→180) works
        correctly for both.
        """
        if float(ds["longitude"].min()) >= 0:
            ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")
        return ds

    @staticmethod
    def _first_time_step(data_array: xr.DataArray) -> xr.DataArray:
        """Drop any time-like dimension from a DataArray, keeping the first step.

        Covers both ``time`` and ``valid_time`` (which the CDS API uses in newer
        downloads). Callers that need a specific time slice should pre-filter the
        file (e.g., via CDO or xarray ``isel``) before passing it to the reader.
        """
        time_dims = [d for d in data_array.dims if "time" in d]
        if time_dims:
            data_array = data_array.isel({d: 0 for d in time_dims})
        return data_array

    @staticmethod
    def _flip_lats_ascending(lats: np.ndarray, layers: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return lats (and each 2-D layer) flipped to ASCENDING latitude order.

        ERA5 stores latitudes descending (see ``_ERA5_LAT_DESCENDING``); every
        other reader in the pipeline emits ascending latitudes, so we flip here
        for consistency. No-op if the coordinate is already ascending.
        """
        if _ERA5_LAT_DESCENDING and lats.size > 1 and lats[0] > lats[-1]:
            lats = lats[::-1]
            layers = [layer[::-1, :] for layer in layers]
        return lats, layers

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slice the cached ERA5 grid to ``bbox``.

        Slices the full ascending-latitude grid from :attr:`_native_grid` with an
        inclusive bounding-box mask (endpoint-inclusive, matching xarray's ``.sel``
        slicing semantics).

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where:

            - ``data`` is float32 shape ``(n_specs, n_lat, n_lon)``.
            - ``lats`` is float64 shape ``(n_lat,)``, ASCENDING order.
            - ``lons`` is float64 shape ``(n_lon,)``.
        """
        data, lats, lons = self._native_grid

        # Inclusive bbox masks on the ascending coordinate arrays select a
        # contiguous block, reproducing the previous xarray slice() behavior.
        lat_mask = (lats >= bbox.lat_min) & (lats <= bbox.lat_max)
        lon_mask = (lons >= bbox.lon_min) & (lons <= bbox.lon_max)

        sub = data[:, lat_mask, :][:, :, lon_mask]
        return sub, lats[lat_mask], lons[lon_mask]


class ERA5Reader(ERA5ReaderBase):
    """Read ERA5 single-level fields from a CDS NetCDF4 file.

    Returns a 3-D data array of shape ``(7, n_lat, n_lon)`` where axis 0
    corresponds to ``_ERA5_SINGLE_LEVEL_VARIABLES`` / ``VARIABLES`` order.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"era5"``.
    RESOLUTION_KM : float
        25 km (ERA5 native ~28 km, rounded to 25 km for PSF calculations).
    VARIABLES : tuple[VariableSpec, ...]
        Seven continuous float32 variables. The winds (``wind_u10``/``wind_v10``)
        are unrestricted (``required_mode=CAM``); the five additional single-level
        fields carry per-spec ``required_mode=IMAGER`` gating, so they appear only in
        the FMATCH-IMAGER-family products. This per-spec latency split is why ``era5``
        belongs to every product's reader set yet contributes a different variable
        subset to the CAM vs IMAGER products.

    Parameters
    ----------
    file_path : Path
        Path to an ERA5 single-levels NetCDF4 file containing all the variables
        in ``_ERA5_SINGLE_LEVEL_VARIABLES``.
    """

    READER_KEY: str = "era5"
    # ERA5 is a reanalysis (no single instrument); use the producing center so the
    # long_name provenance tag "... (ECMWF)" stays uniform across every source.
    INSTRUMENT: str = "ECMWF"
    RESOLUTION_KM: float = 25.0
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

    def _read_native_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read every single-level variable from the file into one stacked grid.

        Opens the NetCDF4 file, normalizes longitude to −180..180, drops any
        time-like dimension (first step only), and returns the whole global grid
        with latitudes in ASCENDING order.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` with ``data`` float32 shape ``(7, n_lat, n_lon)``
            — axis 0 in ``_ERA5_SINGLE_LEVEL_VARIABLES`` order.

        Raises
        ------
        KeyError
            If the file is missing any of the expected ERA5 variables (e.g. a
            download configured with only a subset of the required fields).
        """
        ds = xr.open_dataset(self._file_path, engine="netcdf4")
        try:
            ds = self._normalize_longitudes(ds)

            missing = [nc_name for _, nc_name in _ERA5_SINGLE_LEVEL_VARIABLES if nc_name not in ds]
            if missing:
                raise KeyError(
                    f"ERA5 single-levels file {self._file_path} is missing variable(s) {missing}. "
                    f"The CDS download must include every variable in "
                    f"{[nc for _, nc in _ERA5_SINGLE_LEVEL_VARIABLES]}."
                )

            layers: list[np.ndarray] = []
            lats: np.ndarray | None = None
            lons: np.ndarray | None = None
            for _, nc_name in _ERA5_SINGLE_LEVEL_VARIABLES:
                da = self._first_time_step(ds[nc_name])
                if lats is None:
                    # All variables share the same grid; read the coordinates once.
                    lats = da["latitude"].values.astype(np.float64)
                    lons = da["longitude"].values.astype(np.float64)
                layers.append(da.values.astype(np.float32))

            lats, layers = self._flip_lats_ascending(lats, layers)

            # Stack into (n_specs, n_lat, n_lon) — axis 0 matches VARIABLES ordering.
            data = np.stack(layers, axis=0)
        finally:
            ds.close()

        return data, lats, lons
