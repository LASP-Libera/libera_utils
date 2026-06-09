"""IGBP land cover reader plugin for the footprint matching pipeline.

Data source: MODIS MCD12Q1 (Terra+Aqua Combined Land Cover Type Yearly L3 Global 500m)
- Product: MCD12Q1
- Version: 061 (current)
- Format: HDF4 (.hdf)
- Spatial resolution: ~500m (0.5 km), stored in this reader as 1.0 km nominal
- Temporal coverage: Annual, available from 2001 to near-present
- Spatial coverage: Global, sinusoidal MODIS tile grid (e.g., h09v05)
- Classification scheme: IGBP Land Cover (17 classes + water + fill)

References
----------
Product page: https://lpdaac.usgs.gov/products/mcd12q1v061/
User guide:   https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf
LP DAAC Access: https://appeears.earthdatacloud.nasa.gov/ (EarthData login required)
SDS naming:   "LC_Type1" for IGBP classification scheme
"""
from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers._hdf4_io import read_modis_sinusoidal_hdf4
from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# IGBP fill / no-data value in the MCD12Q1 HDF4 file.
# Value 255 is the standard fill for uint8 SDS fields in this product.
# Values 0–20 are valid land cover categories.
_IGBP_FILL_VALUE: float = 255.0

# IGBP classification has 20 classes (0–19 valid, where 0 = water).
# This is stored in the VARIABLES spec so the aggregation engine knows
# how many category bins to allocate in the PSF histogram.
_N_IGBP_CATEGORIES: int = 20


class IGBPReader(GriddedDataReader):
    """Read IGBP land cover classification from a MODIS MCD12Q1 HDF4 tile file.

    The MCD12Q1 product stores IGBP land cover as ``LC_Type1`` (uint8) in
    sinusoidal-projected HDF4 tiles. Geographic coordinates are not stored as SDS
    arrays; they are computed from the HDF-EOS ``StructMetadata.0`` tile-corner
    metadata via :func:`read_modis_sinusoidal_hdf4`. This reader loads the full
    tile and returns the subset that falls within the requested bounding box.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"igbp"``.
    RESOLUTION_KM : float
        Nominal resolution 1.0 km (actual MODIS 500 m native, rounded for PSF
        weighting calculations).
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM.
    VARIABLES : tuple[VariableSpec, ...]
        Single variable: ``"surface_type"`` (categorical, 20 IGBP classes).

    Parameters
    ----------
    file_path : Path
        Path to a MCD12Q1 HDF4 tile file (e.g., ``MCD12Q1.A2023001.h09v05.061.hdf``).
    """

    READER_KEY: str = "igbp"
    RESOLUTION_KM: float = 1.0
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="surface_type",
            dtype="int16",
            aggregation="weighted_mode",
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
    )

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load IGBP land cover pixels within ``bbox``.

        Opens the MCD12Q1 HDF4 file, reads the full ``LC_Type1``,
        ``latitude``, and ``longitude`` SDS arrays, then returns the subset of
        pixels whose coordinates fall within the bounding box.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where:

            - ``data`` is float32 shape ``(n_lat, n_lon)``. Fill pixels (original
              value 255) are preserved as-is; the PSF engine is responsible for
              masking.
            - ``lats`` is float64 shape ``(n_lat, n_lon)`` — 2-D pixel-centre latitudes.
            - ``lons`` is float64 shape ``(n_lat, n_lon)`` — 2-D pixel-centre longitudes.

        Notes
        -----
        The function reads the entire HDF4 tile and subsets by coordinate, not by
        HDF4 hyperslab. For the 500 m global MODIS tile (~5000 × 5000 pixels per
        HDF4 tile), this is acceptable because the TileManager caches the GridTile
        and does not re-read the file for subsequent overlapping footprints.
        """
        data_full, lats_full, lons_full = read_modis_sinusoidal_hdf4(
            file_path=str(self._file_path),
            data_sds_name="LC_Type1",
            fill_value=_IGBP_FILL_VALUE,
        )

        # lats_full and lons_full are 2-D pixel-centre coordinate grids derived
        # from the sinusoidal projection metadata (MCD12Q1 has no lat/lon SDS).
        # Compute bounding-box membership on the per-pixel level and extract the
        # rectangular subregion that covers all matching pixels.
        lat_mask = (lats_full >= bbox.lat_min) & (lats_full <= bbox.lat_max)
        lon_mask = (lons_full >= bbox.lon_min) & (lons_full <= bbox.lon_max)

        if lat_mask.ndim == 2 and lon_mask.ndim == 2:
            # 2-D lat/lon coordinate grids — find the bounding row/col range.
            combined = lat_mask & lon_mask
            rows = np.where(combined.any(axis=1))[0]
            cols = np.where(combined.any(axis=0))[0]
            if rows.size == 0 or cols.size == 0:
                # No pixels fall within bbox; return empty arrays.
                return (
                    np.empty((0, 0), dtype=np.float32),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
            row_sl = slice(rows[0], rows[-1] + 1)
            col_sl = slice(cols[0], cols[-1] + 1)
            data_sub = data_full[row_sl, col_sl]
            # Build 1-D coordinate arrays from the mean lat/lon per row/column.
            lats_out = lats_full[row_sl, col_sl].mean(axis=1)
            lons_out = lons_full[row_sl, col_sl].mean(axis=0)
        else:
            # 1-D lat/lon coordinate arrays (simpler case used in tests).
            row_sl = np.where(lat_mask)[0]
            col_sl = np.where(lon_mask)[0]
            if row_sl.size == 0 or col_sl.size == 0:
                return (
                    np.empty((0, 0), dtype=np.float32),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )
            r_sl = slice(row_sl[0], row_sl[-1] + 1)
            c_sl = slice(col_sl[0], col_sl[-1] + 1)
            data_sub = data_full[r_sl, c_sl]
            lats_out = lats_full[r_sl]
            lons_out = lons_full[c_sl]

        return data_sub, lats_out, lons_out
