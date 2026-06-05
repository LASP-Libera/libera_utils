"""VIIRS cloud property reader plugin for the footprint matching pipeline.

Data source: NOAA VIIRS Level-2 / Level-3 Cloud Properties (CLDPX)
- Format: HDF4 (.hdf)
- Spatial resolution: 0.75 km (750 m nadir pixel)
- Coverage: Global, swath-based L2 granules
- Variables: cloud_fraction, cloud_optical_thickness, cloud_top_pressure

References
----------
NOAA CLASS archive: https://www.avl.class.noaa.gov/saa/products/welcome
VIIRS cloud docs:   https://www.star.nesdis.noaa.gov/jpss/clouds.php
VIIRS L2 ATBD:      https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/ATBD_EPS_Cloud_VIIRS_v1.3.pdf
Product guide:      https://www.ncei.noaa.gov/products/climate-data-records/cloud-properties-viirs
SDS naming varies by granule version; see reader code for assumed names.
"""
from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers._hdf4_io import read_hdf4_lat_lon_grid
from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey, VariableSpec

# HDF4 SDS names for each VIIRS cloud variable.
# These must match the SDS names in the CLDPX HDF4 files exactly.
# If a future product version changes these names, update the mapping below.
_VIIRS_SDS_NAMES: dict[str, str] = {
    "cloud_fraction": "cloud_fraction",
    "cloud_optical_thickness": "cloud_optical_thickness",
    "cloud_top_pressure": "cloud_top_pressure",
}

# HDF4 fill value used in VIIRS cloud property files.
_VIIRS_FILL_VALUE: float = -999.0


class VIIRSL2L3Reader(GriddedDataReader):
    """Read VIIRS cloud properties from a CLDPX HDF4 file.

    Loads three cloud variables (cloud_fraction, cloud_optical_thickness,
    cloud_top_pressure) from a VIIRS Level-2/3 HDF4 granule and returns a
    3-D data array of shape ``(3, n_lat, n_lon)`` stacked in ``VARIABLES`` order.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"viirs_l2l3"``.
    RESOLUTION_KM : float
        0.75 km (VIIRS 750-m nadir resolution).
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM (cloud fraction required for
        every operational mode).
    VARIABLES : tuple[VariableSpec, ...]
        Three variables: cloud_fraction, cloud_optical_thickness,
        cloud_top_pressure.

    Parameters
    ----------
    file_path : Path
        Path to a VIIRS CLDPX HDF4 file.
    """

    READER_KEY: str = "viirs_l2l3"
    RESOLUTION_KM: float = 0.75
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="cloud_fraction",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
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
        """Load a VIIRS tile and set ``timestamp_source='radiometer'``.

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
        # Replace the frozen dataclass — create a new instance with the field updated.
        return GridTile(
            data=tile.data,
            lats=tile.lats,
            lons=tile.lons,
            bounds=tile.bounds,
            source=tile.source,
            timestamp_source="radiometer",
        )

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load VIIRS cloud pixels within ``bbox``.

        Reads three SDS variables sequentially from the HDF4 file, subsets
        each to the requested bbox, and stacks them into a 3-D array.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(3, n_lat, n_lon)`` with axis 0 = [cloud_fraction,
            cloud_optical_thickness, cloud_top_pressure].
        """
        variable_arrays: list[np.ndarray] = []
        lats_out: np.ndarray | None = None
        lons_out: np.ndarray | None = None

        for var_spec in self.VARIABLES:
            sds_name = _VIIRS_SDS_NAMES[var_spec.name]
            data_full, lats_full, lons_full = read_hdf4_lat_lon_grid(
                file_path=str(self._file_path),
                data_sds_name=sds_name,
                lat_sds_name="latitude",
                lon_sds_name="longitude",
                fill_value=_VIIRS_FILL_VALUE,
            )

            # Subset to bbox using the lat/lon grids from the first variable.
            # All three variables share the same coordinate arrays so we only
            # compute the mask once (for cloud_fraction) and reuse row/col slices.
            if lats_out is None:
                lats_out, lons_out, r_sl, c_sl = self._compute_bbox_slices(
                    lats_full, lons_full, bbox
                )
                if r_sl is None:
                    # No pixels in bbox — return empty arrays.
                    n = len(self.VARIABLES)
                    return (
                        np.empty((n, 0, 0), dtype=np.float32),
                        np.empty(0, dtype=np.float64),
                        np.empty(0, dtype=np.float64),
                    )

            variable_arrays.append(data_full[r_sl, c_sl].astype(np.float32))

        data_stacked = np.stack(variable_arrays, axis=0)  # (3, n_lat, n_lon)
        return data_stacked, lats_out, lons_out  # type: ignore[return-value]

    @staticmethod
    def _compute_bbox_slices(
        lats_full: np.ndarray,
        lons_full: np.ndarray,
        bbox: BoundingBox,
    ) -> tuple[np.ndarray | None, np.ndarray | None, slice | None, slice | None]:
        """Compute the row/column slices that fall within ``bbox``.

        Handles both 1-D and 2-D lat/lon coordinate arrays (VIIRS L2 granules
        may provide per-pixel 2-D grids).

        Parameters
        ----------
        lats_full, lons_full : np.ndarray
            Full coordinate arrays from the HDF4 file.
        bbox : BoundingBox
            Bounding box to subset to.

        Returns
        -------
        tuple
            ``(lats_out, lons_out, row_slice, col_slice)`` or all ``None``
            components if no pixels fall within the bbox.
        """
        if lats_full.ndim == 2:
            combined = (
                (lats_full >= bbox.lat_min) & (lats_full <= bbox.lat_max) &
                (lons_full >= bbox.lon_min) & (lons_full <= bbox.lon_max)
            )
            rows = np.where(combined.any(axis=1))[0]
            cols = np.where(combined.any(axis=0))[0]
            if rows.size == 0 or cols.size == 0:
                return None, None, None, None
            r_sl = slice(rows[0], rows[-1] + 1)
            c_sl = slice(cols[0], cols[-1] + 1)
            lats_out = lats_full[r_sl, c_sl].mean(axis=1)
            lons_out = lons_full[r_sl, c_sl].mean(axis=0)
        else:
            lat_mask = (lats_full >= bbox.lat_min) & (lats_full <= bbox.lat_max)
            lon_mask = (lons_full >= bbox.lon_min) & (lons_full <= bbox.lon_max)
            row_indices = np.where(lat_mask)[0]
            col_indices = np.where(lon_mask)[0]
            if row_indices.size == 0 or col_indices.size == 0:
                return None, None, None, None
            r_sl = slice(row_indices[0], row_indices[-1] + 1)
            c_sl = slice(col_indices[0], col_indices[-1] + 1)
            lats_out = lats_full[r_sl]
            lons_out = lons_full[c_sl]

        return lats_out, lons_out, r_sl, c_sl
