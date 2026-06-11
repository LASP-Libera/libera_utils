"""NISE sea ice reader plugin for the footprint matching pipeline.

Data source: NSIDC Near-real-time Ice and Snow Extent (NISE) Product
- Products: NISE (SSM/I-SSMIS, v5) and NISE_A2 (AMSR-2, v1)
- Format: HDF-EOS4 (.HDFEOS), opened via pyhdf
- Spatial resolution: 25 km nominal (NSIDC EASE-Grid North)
- Grid: 721 × 721 polar azimuthal equal-area (NSIDC EASE-Grid North, EPSG:3408)
- Temporal coverage: 1978-present (NISE SSM/I-SSMIS), 2012-present (NISE_A2 AMSR-2)
- Spatial coverage: Northern Hemisphere
- Data variable: Extent SDS (uint8 category codes 0–255)

NISE Extent encoding
--------------------
  0        Outside domain (land, Antarctica, etc.)
  1–100    Sea ice concentration in percent (1 = 1%, 100 = 100%)
  101      Permanent ice (Greenland Ice Sheet, Antarctic ice shelves)
  102      Not used
  103–110  Dry snow on land (mapped to 0.0)
  255      Missing/fill

This reader converts the Extent SDS to a continuous sea ice concentration in
the range [0.0, 1.0]:
  - Codes 1–100  →  value / 100.0   (fraction of ocean covered by ice)
  - Code 101     →  1.0             (permanent ice treated as 100% ice-covered)
  - All others   →  0.0             (land, open ocean, fill, etc.)

References
----------
Product page:    https://nsidc.org/data/nise
User guide:      https://nsidc.org/sites/default/files/nise5-v001-userguide.pdf
Data access:     https://n5eil01u.ecs.nsidc.org/NISE/  (Earthdata login required)
File naming:     NISE_SSMISF{ss}_{YYYYMMDD}.HDFEOS  (NISE v5)
                 NISE_A2_{YYYYMMDD}.HDFEOS          (NISE_A2 v1)
EPSG:3408 desc:  https://epsg.io/3408
EASE-Grid ref:   https://nsidc.org/ease
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj

from libera_utils.footprint_matching.readers._hdf4_io import _require_pyhdf
from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# NISE 25-km EASE-Grid North parameters (EPSG:3408).
# The 721 × 721 grid covers ±9,036,842.7625 m in both x and y.
# Source: NSIDC EASE-Grid North 25-km grid specification
_DEFAULT_GRID_ROWS: int = 721
_DEFAULT_GRID_COLS: int = 721
_DEFAULT_RESOLUTION_M: float = 25_067.525  # EASE-Grid North 25-km cell size in meters
# Upper-left corner of the grid in EPSG:3408 meters (x = easting, y = northing).
_DEFAULT_X_ORIGIN: float = -9_036_842.7625  # meters (upper-left x)
_DEFAULT_Y_ORIGIN: float = 9_036_842.7625   # meters (upper-left y)


class NISEReader(GriddedDataReader):
    """Read NISE sea ice concentration from an HDF-EOS4 file.

    The NISE product (Near-real-time Ice and Snow Extent) distributes sea ice
    concentration in percent as a uint8 Extent SDS within an HDF-EOS4 file.
    This reader converts those category codes to a continuous float32
    concentration in [0.0, 1.0], reprojects the EASE-Grid North (EPSG:3408)
    pixel centers to WGS84 lat/lon, and returns the subset within the
    requested bounding box.

    The grid parameters (rows, cols, resolution, origin) are exposed as
    constructor keyword arguments with real defaults so that tests can inject
    a small synthetic grid without building a full 721 × 721 fixture.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"nise"``.
    RESOLUTION_KM : float
        25 km (NISE EASE-Grid North 25-km product).
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM.
    VARIABLES : tuple[VariableSpec, ...]
        Single variable: ``"sea_ice_concentration"`` (continuous float32,
        ``weighted_mean`` aggregation, range 0.0–1.0).

    Parameters
    ----------
    file_path : Path
        Path to a NISE HDF-EOS4 file (``*.HDFEOS``).
    grid_rows : int, optional
        Number of grid rows. Default 721 (real NISE 25-km product).
    grid_cols : int, optional
        Number of grid columns. Default 721.
    resolution_m : float, optional
        Grid cell size in meters. Default 25067.525 (EASE-Grid North 25 km).
    x_origin : float, optional
        Upper-left x coordinate (meters, EPSG:3408). Default -9,036,842.7625.
    y_origin : float, optional
        Upper-left y coordinate (meters, EPSG:3408). Default 9,036,842.7625.
    """

    READER_KEY: str = "nise"
    RESOLUTION_KM: float = 25.0
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="sea_ice_concentration",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
    )

    def __init__(
        self,
        file_path: Path,
        *,
        grid_rows: int = _DEFAULT_GRID_ROWS,
        grid_cols: int = _DEFAULT_GRID_COLS,
        resolution_m: float = _DEFAULT_RESOLUTION_M,
        x_origin: float = _DEFAULT_X_ORIGIN,
        y_origin: float = _DEFAULT_Y_ORIGIN,
    ) -> None:
        super().__init__(file_path)
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._resolution_m = resolution_m
        self._x_origin = x_origin
        self._y_origin = y_origin

        # Build the EPSG:3408 → WGS84 transformer once at construction time.
        # Always use CRS objects (not bare EPSG strings) to suppress pyproj
        # FutureWarning about authority-based CRS construction.
        self._transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(3408),   # NSIDC EASE-Grid North (azimuthal equal-area)
            pyproj.CRS.from_epsg(4326),
            always_xy=True,  # Input: (x=easting, y=northing); Output: (lon, lat)
        )

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the NISE Extent SDS and return concentration values within ``bbox``.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(n_rows, n_cols)`` with values in [0.0, 1.0].
        """
        raw = self._read_extent_sds()
        lats_2d, lons_2d = self._compute_latlon_grid()

        # Convert NISE Extent encoding to sea ice concentration (float32 0.0–1.0).
        #   1–100  → value / 100.0  (concentration percentage as fraction)
        #   101    → 1.0            (permanent ice)
        #   all other codes → 0.0  (open ocean, land, outside domain, fill)
        concentration = np.where(
            (raw >= 1) & (raw <= 100),
            raw.astype(np.float32) / 100.0,
            np.where(raw == 101, np.float32(1.0), np.float32(0.0)),
        )

        # Find pixels within the bounding box.
        in_bbox = (
            (lats_2d >= bbox.lat_min) & (lats_2d <= bbox.lat_max)
            & (lons_2d >= bbox.lon_min) & (lons_2d <= bbox.lon_max)
        )
        rows = np.where(in_bbox.any(axis=1))[0]
        cols = np.where(in_bbox.any(axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            return (
                np.empty((0, 0), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        r_sl = slice(rows[0], rows[-1] + 1)
        c_sl = slice(cols[0], cols[-1] + 1)

        data_sub = concentration[r_sl, c_sl]
        # Return representative lat/lon for each row/column (mean across the slab).
        lats_out = lats_2d[r_sl, c_sl].mean(axis=1)
        lons_out = lons_2d[r_sl, c_sl].mean(axis=0)

        return data_sub, lats_out, lons_out

    def _read_extent_sds(self) -> np.ndarray:
        """Open the NISE HDF-EOS4 file and return the Extent SDS as a uint8 array.

        Returns
        -------
        np.ndarray
            Shape ``(grid_rows, grid_cols)``, dtype uint8. Each element holds
            a NISE Extent code (see module docstring for encoding).

        Raises
        ------
        ImportError
            If pyhdf is not installed (see :func:`_require_pyhdf`).
        KeyError
            If the file does not contain an ``Extent`` SDS.
        OSError
            If the file cannot be opened (not found, corrupt header, etc.).
        """
        sd = _require_pyhdf()
        hdf = sd.SD(str(self._file_path), sd.SDC.READ)
        try:
            sds = hdf.select("Extent")
            raw = np.array(sds.get(), dtype=np.uint8)
        finally:
            hdf.end()
        return raw

    def _compute_latlon_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute geographic lat/lon coordinates for every pixel in the NISE grid.

        Uses pyproj to transform the EASE-Grid North pixel centers (EPSG:3408)
        to WGS84 geographic coordinates (EPSG:4326).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lats_2d, lons_2d)`` each of shape ``(grid_rows, grid_cols)``
            in degrees.

        Notes
        -----
        Pixel centers are at half-cell offsets from the grid origin:
            x_center[col] = x_origin + (col + 0.5) * resolution_m
            y_center[row] = y_origin - (row + 0.5) * resolution_m  (y decreases southward)

        Pixels whose projected coordinates fall outside the geographic domain of
        EPSG:3408 (i.e., at very high southern latitudes) will have NaN in the
        output arrays; these are excluded by the bbox mask in
        :meth:`_load_spatial_region`.
        """
        col_indices = np.arange(self._grid_cols, dtype=np.float64)
        row_indices = np.arange(self._grid_rows, dtype=np.float64)

        x_centers = self._x_origin + (col_indices + 0.5) * self._resolution_m
        y_centers = self._y_origin - (row_indices + 0.5) * self._resolution_m

        # Build 2-D coordinate grids for the transformer.
        x_2d, y_2d = np.meshgrid(x_centers, y_centers)

        # always_xy=True means input is (x, y) and output is (lon, lat).
        lons_2d, lats_2d = self._transformer.transform(x_2d, y_2d)

        return lats_2d, lons_2d
