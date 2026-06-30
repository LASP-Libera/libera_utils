"""ERA5 wind speed reader plugin for the footprint matching pipeline.

Data source: ECMWF ERA5 Reanalysis (single-level surface wind components)
- Variables: u10 (10 m U wind component) and v10 (10 m V wind component)
- Format: NetCDF4 (.nc), distributed via the Copernicus Climate Data Store
- Spatial resolution: ~0.25° (~28 km); stored as 25 km in RESOLUTION_KM
- Grid: Regular lat/lon, global coverage, latitudes in DESCENDING order (90 → -90)
- Temporal resolution: Hourly
- Temporal coverage: 1940-present

References
----------
CDS dataset:  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
CDS API docs: https://cds.climate.copernicus.eu/how-to-api
Variable list:https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation
u10 variable: https://apps.ecmwf.int/codes/grib/param-db?id=165
v10 variable: https://apps.ecmwf.int/codes/grib/param-db?id=166
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# ERA5 variable names as stored in CDS NetCDF4 files.
# These must match the dimension/variable names in the downloaded .nc files.
_ERA5_U10_VAR: str = "u10"
_ERA5_V10_VAR: str = "v10"

# The ERA5 latitude dimension is stored in DESCENDING order (90° → -90°) in
# files downloaded from the CDS. xarray.sel() works correctly regardless of
# direction when using slice(max_lat, min_lat) ordering, but we must be careful
# to reverse the output lats array so that it is ASCENDING for downstream callers.
# This constant documents the known direction of the ERA5 lat coordinate.
_ERA5_LAT_DESCENDING: bool = True


class ERA5Reader(GriddedDataReader):
    """Read ERA5 10-m wind components (u10, v10) from a NetCDF4 file.

    Returns a 3-D data array of shape ``(2, n_lat, n_lon)`` where axis 0
    corresponds to [u10, v10] in ``VARIABLES`` order.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"era5"``.
    RESOLUTION_KM : float
        25 km (ERA5 native ~28 km, rounded to 25 km for PSF calculations).
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM.
    VARIABLES : tuple[VariableSpec, ...]
        Two variables: ``"wind_u10"`` and ``"wind_v10"`` (continuous, float32).

    Parameters
    ----------
    file_path : Path
        Path to an ERA5 NetCDF4 file containing ``u10`` and ``v10`` variables.
    """

    READER_KEY: str = "era5"
    # ERA5 is a reanalysis (no single instrument); use the producing center so the
    # `<source>_<instrument>_<var>` naming stays uniform across every source.
    INSTRUMENT: str = "ECMWF"
    RESOLUTION_KM: float = 25.0
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    VARIABLES: tuple[VariableSpec, ...] = (
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
    )

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load ERA5 u10/v10 data within ``bbox`` from a NetCDF4 file.

        Uses xarray for coordinate-aware slicing. The ERA5 latitude axis is
        stored in descending order (90 → -90), so the slice is specified as
        ``slice(lat_max, lat_min)`` to match xarray's direction-aware indexing.
        The output arrays are reordered so that lats are ASCENDING.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where:

            - ``data`` is float32 shape ``(2, n_lat, n_lon)`` — [u10, v10].
            - ``lats`` is float64 shape ``(n_lat,)``, ASCENDING order.
            - ``lons`` is float64 shape ``(n_lon,)``.

        Notes
        -----
        If the ERA5 file has a ``time`` dimension, only the first time step is
        loaded. Callers that need a specific time slice should pre-filter the
        file (e.g., via CDO or xarray ``isel``) before passing it to this reader.
        """
        ds = xr.open_dataset(self._file_path, engine="netcdf4")
        try:
            # ERA5 files from CDS can use either –180→180 or 0→360 longitude convention
            # depending on how the download was configured. Normalize to –180→180 so that
            # the bbox slice (which always uses –180→180) works correctly.
            if float(ds["longitude"].min()) >= 0:
                ds = ds.assign_coords(longitude=((ds["longitude"] + 180) % 360) - 180).sortby("longitude")

            # ERA5 lats are descending, so slice(max, min) selects the correct range.
            lat_slice = slice(bbox.lat_max, bbox.lat_min)
            lon_slice = slice(bbox.lon_min, bbox.lon_max)

            u10_sub = ds[_ERA5_U10_VAR].sel(latitude=lat_slice, longitude=lon_slice)
            v10_sub = ds[_ERA5_V10_VAR].sel(latitude=lat_slice, longitude=lon_slice)

            # Drop any time-like dimension (covers both "time" and "valid_time", which
            # the CDS API uses in newer downloads). Take the first time step only.
            time_dims = [d for d in u10_sub.dims if "time" in d]
            if time_dims:
                u10_sub = u10_sub.isel({d: 0 for d in time_dims})
                v10_sub = v10_sub.isel({d: 0 for d in time_dims})

            # Extract coordinate arrays and ensure ascending lat order.
            lats = u10_sub["latitude"].values.astype(np.float64)
            lons = u10_sub["longitude"].values.astype(np.float64)

            u10_arr = u10_sub.values.astype(np.float32)
            v10_arr = v10_sub.values.astype(np.float32)

            if _ERA5_LAT_DESCENDING and lats.size > 1 and lats[0] > lats[-1]:
                # Flip to ascending order for consistency with other readers.
                lats = lats[::-1]
                u10_arr = u10_arr[::-1, :]
                v10_arr = v10_arr[::-1, :]

            # Stack into (2, n_lat, n_lon) — axis 0 matches VARIABLES ordering.
            data = np.stack([u10_arr, v10_arr], axis=0)
        finally:
            ds.close()

        return data, lats, lons
