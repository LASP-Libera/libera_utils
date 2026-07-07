"""VIIRS cloud property reader plugin for the footprint matching pipeline.

Data source: NOAA-20 (JPSS-1) VIIRS Daily Level-3 Cloud Properties
- Product: CLDPROP_D3_VIIRS_NOAA20 (Collection 011)
- Format: NetCDF4 with nested groups
- Spatial resolution: 1° × 1° (~111 km at equator)
- Grid: 360 × 180 regular lat/lon (global)
- Data layout: variables stored with dimension order (longitude, latitude),
  requiring a transpose to the expected (latitude, longitude) output order
- Temporal resolution: Daily composites
- Variables: cloud_optical_thickness (Mean), cloud_top_pressure (Mean)

NetCDF4 group layout (CLDPROP_D3)
----------------------------------
Root:
  latitude  (180,)   — 1° bin centers, −89.5 to 89.5
  longitude (360,)   — 1° bin centers, −179.5 to 179.5
  Cloud_Optical_Thickness_Combined/
    Mean      (360, 180) — daily mean cloud optical thickness
  Cloud_Top_Pressure/
    Mean      (360, 180) — daily mean cloud top pressure [hPa]

Fill value: −9999.0 (replaced with NaN before output)

References
----------
Product page:
    https://www.ncei.noaa.gov/products/climate-data-records/cloud-properties-viirs
ATBD:
    https://www.ncei.noaa.gov/pub/data/sds/cdr/CDRs/Cloud_Properties_VIIRS/
    AlgorithmDescriptionVIIRS_01B-20a.pdf
Data access:
    https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
File naming:
    CLDPROP_D3_VIIRS_NOAA20.A{YYYYDDD}.{collection}.{YYYYDDDHHMMSS}.nc
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey, VariableSpec

# Map each VariableSpec name to its (group_name, variable_name_within_group)
# path inside the CLDPROP_D3 NetCDF4 file.
_D3_GROUP_MAP: dict[str, tuple[str, str]] = {
    "cloud_optical_thickness": ("Cloud_Optical_Thickness_Combined", "Mean"),
    "cloud_top_pressure": ("Cloud_Top_Pressure", "Mean"),
}

# Fill / missing value used in CLDPROP_D3 files.
_D3_FILL_VALUE: float = -9999.0


class VIIRSCloudReader(GriddedDataReader):
    """Read VIIRS daily cloud properties from a CLDPROP_D3 NetCDF4 file.

    Loads two cloud variables (cloud_optical_thickness, cloud_top_pressure)
    from a CLDPROP_D3 VIIRS Level-3 daily file and returns a 3-D data array of
    shape ``(2, n_lat, n_lon)`` stacked in ``VARIABLES`` order.

    The CLDPROP_D3 product stores data with a non-standard ``(longitude,
    latitude)`` dimension order. This reader transposes each variable to
    ``(latitude, longitude)`` before returning.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"viirs_cloud"``.
    RESOLUTION_KM : float
        111 km (1° × 1° daily L3 grid resolution at equator).
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM.
    VARIABLES : tuple[VariableSpec, ...]
        Two variables: cloud_optical_thickness, cloud_top_pressure.

    Parameters
    ----------
    file_path : Path
        Path to a CLDPROP_D3 NetCDF4 file.
    """

    READER_KEY: str = "viirs_cloud"
    # CLDPROP_D3_VIIRS_NOAA20 cloud properties are from VIIRS aboard NOAA-20 (JPSS-1).
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 111.0
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="cloud_optical_thickness",
            dtype="float32",
            aggregation="weighted_log_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="cloud_top_pressure",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
    )

    def load_tile(self, key: TileKey) -> GridTile:
        """Load a VIIRS cloud tile and set ``timestamp_source='radiometer'``.

        Overrides the base ``load_tile`` to set ``timestamp_source`` on the
        returned GridTile, since VIIRS cloud properties are collocated in time
        with the radiometer observation.

        Parameters
        ----------
        key : TileKey
            Tile cache key.

        Returns
        -------
        GridTile
            Tile with ``timestamp_source='radiometer'``.
        """
        tile = super().load_tile(key)
        return GridTile(
            data=tile.data,
            lats=tile.lats,
            lons=tile.lons,
            bounds=tile.bounds,
            source=tile.source,
            timestamp_source="radiometer",
        )

    @cached_property
    def _native_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the full VIIRS cloud-property grid once and cache it on the instance.

        Opens the NetCDF4 file, navigates to each variable's group, transposes
        from (longitude, latitude) to (latitude, longitude) storage order, and
        replaces fill values with NaN. Cached per reader instance so the file is
        opened and read once, then every tile slices these in-memory arrays
        (see :meth:`_load_spatial_region`) instead of re-reading the file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(2, n_lat, n_lon)`` with axis 0 = [cloud_optical_thickness,
            cloud_top_pressure] (fill as NaN), and ``lats`` / ``lons`` are float64
            1-D coordinate arrays.
        """
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(self._file_path), "r") as ds:
            # Root-level coordinate arrays: (180,) and (360,) respectively
            lats_full = np.array(ds.variables["latitude"][:], dtype=np.float64)
            lons_full = np.array(ds.variables["longitude"][:], dtype=np.float64)

            variable_arrays: list[np.ndarray] = []
            for var_spec in self.VARIABLES:
                group_name, var_name = _D3_GROUP_MAP[var_spec.name]
                group = ds.groups[group_name]
                # Data shape is (n_lon, n_lat) in the file — transpose to (n_lat, n_lon).
                raw = np.array(group.variables[var_name][:], dtype=np.float32).T
                # Replace fill values with NaN.
                raw[raw <= _D3_FILL_VALUE] = np.nan
                variable_arrays.append(raw)

        data = np.stack(variable_arrays, axis=0)  # (2, n_lat_full, n_lon_full)
        return data, lats_full, lons_full

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slice the cached VIIRS cloud-property grid to ``bbox``.

        Subsets the full grid from :attr:`_native_grid` to the requested bounding box.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(2, n_lat, n_lon)`` with axis 0 = [cloud_optical_thickness,
            cloud_top_pressure]. Fill pixels (originally ≤ −9999.0) are
            returned as NaN.
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
