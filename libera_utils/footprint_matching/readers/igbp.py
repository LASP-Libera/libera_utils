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
from libera_utils.footprint_matching.readers._swath import rasterize_points_to_grid
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
    metadata via :func:`read_modis_sinusoidal_hdf4`, which returns **2-D
    per-pixel** latitude/longitude grids.

    Geolocation: rasterization, not mean-collapse
    ----------------------------------------------
    The sinusoidal projection is *not* axis-aligned in lat/lon — longitude varies
    along each row (``lon = X / (R·cos(lat))``), so a single longitude per column
    (a column mean) places no real pixel and mis-geolocates the data. To carry
    each pixel's true file-derived position through to footprint matching, this
    reader flattens the 2-D ``(lat, lon, value)`` pixels to points and bins them
    onto a regular sub-grid over the requested tile via
    :func:`~libera_utils.footprint_matching.readers._swath.rasterize_points_to_grid`
    — exactly like the ``ssf`` and ``cldpix`` swath readers. Every output cell
    therefore has an exact center lat/lon.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"igbp"``.
    RESOLUTION_KM : float
        Nominal resolution 1.0 km (actual MODIS 500 m native, rounded for PSF
        weighting calculations).
    OUTPUT_CELL_DEG : float
        Edge length of the rasterized output cells (degrees). 0.05° matches the
        ``cldpix`` 1-km reader and gives 40×40 cells per 2° tile.
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
    # MCD12Q1 land cover is derived from Terra+Aqua MODIS.
    INSTRUMENT: str = "MODIS"
    RESOLUTION_KM: float = 1.0
    OUTPUT_CELL_DEG: float = 0.05
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

    def _load_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the full tile and flatten it to geolocated land-cover points.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, values)`` where ``lats``/``lons`` are 1-D
            ``(n_pixels,)`` float64 pixel-centre coordinates derived from the
            sinusoidal projection metadata, and ``values`` is float64 shape
            ``(1, n_pixels)`` holding the IGBP class code per pixel. Fill pixels
            (original value 255) are set to ``NaN`` so the ``weighted_mode``
            rasterization never selects fill as the modal land-cover class.
        """
        data_2d, lats_2d, lons_2d = read_modis_sinusoidal_hdf4(
            file_path=str(self._file_path),
            data_sds_name="LC_Type1",
            fill_value=_IGBP_FILL_VALUE,
        )

        # read_modis_sinusoidal_hdf4 returns the data and coordinates as 2-D
        # per-pixel grids. Mask the 255 fill to NaN *before* flattening: the
        # rasterizer's mode aggregation counts every finite value as a category,
        # so leaving 255 in would let "fill" win a cell.
        values_2d = np.where(data_2d == _IGBP_FILL_VALUE, np.nan, data_2d)

        lats = np.asarray(lats_2d, dtype=np.float64).ravel()
        lons = np.asarray(lons_2d, dtype=np.float64).ravel()
        values = values_2d.astype(np.float64).ravel()[np.newaxis, :]  # (1, n_pixels)
        return lats, lons, values

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize IGBP land-cover pixels within ``bbox`` onto a regular sub-grid.

        Flattens the sinusoidal tile to geolocated points (see
        :meth:`_load_points`) and bins them onto a regular ``OUTPUT_CELL_DEG``
        sub-grid covering the tile, taking the modal land-cover class per cell.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(n_lat, n_lon)`` (single-variable contract), ``lats``/``lons`` are
            1-D float64 cell-centre coordinate arrays, and cells with no pixels
            are ``NaN``.

        Notes
        -----
        The function reads the entire HDF4 tile and bins by coordinate. For the
        500 m MODIS tile this is acceptable because the TileManager caches the
        GridTile and does not re-read the file for overlapping footprints.
        """
        lats, lons, values = self._load_points()

        # rasterize_points_to_grid always returns (n_var, n_lat, n_lon). IGBP is
        # single-variable, so squeeze axis 0 to honour the 2-D output contract
        # that single-variable grid readers use.
        data, lats_out, lons_out = rasterize_points_to_grid(
            point_lats=lats,
            point_lons=lons,
            values=values,
            bbox=(bbox.lat_min, bbox.lat_max, bbox.lon_min, bbox.lon_max),
            cell_size_deg=self.OUTPUT_CELL_DEG,
            aggregations=[self.VARIABLES[0].aggregation],
        )
        return data[0], lats_out, lons_out
