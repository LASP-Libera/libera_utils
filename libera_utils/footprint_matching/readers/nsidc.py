"""NSIDC sea ice reader plugin for the footprint matching pipeline.

Data source: NSIDC IMS (Interactive Multisensor Snow and Ice Mapping System)
- Product: IMS Daily Northern Hemisphere Snow and Ice Analysis at 24 km resolution
- Format: ASCII raster (plain-text) with a short header, then rows of integer category codes
- Spatial resolution: 24 km nominal (25 km used for RESOLUTION_KM in PSF weighting)
- Grid: 1024 × 1024 polar stereographic (NSIDC EASE-Grid North, EPSG:3411)
- Temporal coverage: 1997-present, daily
- Spatial coverage: Northern Hemisphere (polar stereographic)
- Category codes: 0=Outside domain, 1=Ocean, 2=Sea ice, 3=Snow, 4=Ice-free land

References
----------
Product page:    https://nsidc.org/data/g02156
User guide:      https://nsidc.org/sites/default/files/g02156-v001-userguide_1_1.pdf
Data access:     https://noaadata.apps.nsidc.org/NOAA/G02156/24km/ (no login required)
File naming:     ims{YYYY}{DDD}_24km_v1.3.asc.gz  (Julian day of year)
Projection info: https://nsidc.org/data/user-resources/help-center/guide-nsidcs-polar-stereographic-projection
EPSG:3411 desc:  https://epsg.io/3411
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyproj

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# IMS 24-km grid parameters (EPSG:3411 polar stereographic, Northern Hemisphere).
# These define the origin corner and spacing of the 1024 × 1024 Cartesian grid.
# Source: NSIDC IMS User Guide, Table 1 (https://nsidc.org/data/g02156)
_DEFAULT_GRID_ROWS: int = 1024
_DEFAULT_GRID_COLS: int = 1024
_DEFAULT_RESOLUTION_M: float = 24_000.0  # 24 km in meters
# Upper-left corner of the grid in EPSG:3411 meters (x = easting, y = northing).
# Computed as: center = (0, 0), grid extent = 1024 × 24000 m = 24576 km
_DEFAULT_X_ORIGIN: float = -12_288_000.0  # meters (upper-left x)
_DEFAULT_Y_ORIGIN: float = 12_288_000.0   # meters (upper-left y)

# IMS sea ice category codes.
# 0=outside domain, 1=ocean (open water), 2=sea ice, 3=snow-covered land, 4=ice-free land
_N_NSIDC_CATEGORIES: int = 5


class NSIDCReader(GriddedDataReader):
    """Read IMS sea ice / snow classification from an ASCII raster file.

    The IMS 24-km product is distributed as a gzip-compressed ASCII text file.
    After decompression, the file contains a short metadata header (variable
    number of lines) followed by rows of space-separated or concatenated integer
    category codes. This reader parses the ASCII format, converts the polar
    stereographic pixel coordinates to geographic lat/lon using pyproj, and
    returns the subset within the requested bounding box.

    The grid parameters (rows, cols, resolution, origin) are exposed as
    constructor keyword arguments with real defaults so that tests can pass
    smaller synthetic grids without generating a full 1024 × 1024 fixture.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"nsidc"``.
    RESOLUTION_KM : float
        24-km nominal, stored as 25 km for PSF weight calculations (matching
        CERES heritage convention for this product).
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM.
    VARIABLES : tuple[VariableSpec, ...]
        Single variable: ``"sea_ice_type"`` (categorical, 5 IMS classes).

    Parameters
    ----------
    file_path : Path
        Path to an IMS ASCII raster file (decompressed or gzip, the reader
        handles both via Python's built-in gzip module).
    grid_rows : int, optional
        Number of grid rows. Default 1024 (real IMS 24-km product).
    grid_cols : int, optional
        Number of grid columns. Default 1024.
    resolution_m : float, optional
        Grid cell size in meters. Default 24000.
    x_origin : float, optional
        Upper-left x coordinate (meters, EPSG:3411). Default -12_288_000.
    y_origin : float, optional
        Upper-left y coordinate (meters, EPSG:3411). Default 12_288_000.
    """

    READER_KEY: str = "nsidc"
    RESOLUTION_KM: float = 25.0
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="sea_ice_type",
            dtype="int16",
            aggregation="weighted_mode",
            required_mode=OperationalMode.CAM,
            n_categories=_N_NSIDC_CATEGORIES,
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

        # Build the EPSG:3411 → WGS84 transformer once at construction time.
        # Always use CRS objects (not bare EPSG strings) to suppress pyproj
        # FutureWarning about authority-based CRS construction.
        self._transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(3411),
            pyproj.CRS.from_epsg(4326),
            always_xy=True,  # Input: (x=easting, y=northing); Output: (lon, lat)
        )

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse the IMS ASCII file and return sea ice codes within ``bbox``.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is int16 shape ``(n_lat, n_lon)``.
        """
        grid = self._parse_ascii_grid()
        lats_2d, lons_2d = self._compute_latlon_grid()

        # Find pixels within the bounding box.
        in_bbox = (
            (lats_2d >= bbox.lat_min) & (lats_2d <= bbox.lat_max) &
            (lons_2d >= bbox.lon_min) & (lons_2d <= bbox.lon_max)
        )
        rows = np.where(in_bbox.any(axis=1))[0]
        cols = np.where(in_bbox.any(axis=0))[0]

        if rows.size == 0 or cols.size == 0:
            return (
                np.empty((0, 0), dtype=np.int16),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )

        r_sl = slice(rows[0], rows[-1] + 1)
        c_sl = slice(cols[0], cols[-1] + 1)

        data_sub = grid[r_sl, c_sl].astype(np.int16)
        # Return representative lat/lon for each row/column (mean across the slab).
        lats_out = lats_2d[r_sl, c_sl].mean(axis=1)
        lons_out = lons_2d[r_sl, c_sl].mean(axis=0)

        return data_sub, lats_out, lons_out

    def _parse_ascii_grid(self) -> np.ndarray:
        """Read the ASCII raster file and return a 2-D integer array.

        The IMS ASCII format begins with header lines (the exact count varies
        by version). The data section starts with the first line whose character
        count matches ``grid_cols`` (one digit per column, no separators).
        Header lines are detected by checking that the stripped line length
        equals ``grid_cols`` and that all characters are decimal digits.

        Returns
        -------
        np.ndarray
            Shape ``(grid_rows, grid_cols)``, dtype int8. Each element is the
            IMS category code (0–4) for that pixel.

        Notes
        -----
        The pattern ``r'^\\d{grid_cols}$'`` is pre-compiled to exactly match
        a data row of the correct length. Lines that are shorter or contain
        non-digit characters are treated as header lines and skipped.
        """
        import gzip  # noqa: PLC0415 - stdlib, no need to import at module level

        data_row_pattern = re.compile(rf"^\d{{{self._grid_cols}}}$")

        rows_list: list[list[int]] = []

        def _process_lines(lines: list[str]) -> None:
            for line in lines:
                stripped = line.rstrip("\n\r")
                if data_row_pattern.match(stripped):
                    # Each character is a single-digit category code.
                    rows_list.append([int(c) for c in stripped])

        try:
            with gzip.open(self._file_path, "rt", encoding="ascii") as fh:
                _process_lines(fh.readlines())
        except (OSError, gzip.BadGzipFile):
            # Not a gzip file — try plain text.
            with open(self._file_path, encoding="ascii") as fh:
                _process_lines(fh.readlines())

        if len(rows_list) != self._grid_rows:
            raise ValueError(
                f"Expected {self._grid_rows} data rows in {self._file_path}, "
                f"got {len(rows_list)}. The file may be corrupted or the wrong version."
            )

        return np.array(rows_list, dtype=np.int8)

    def _compute_latlon_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute geographic lat/lon coordinates for every pixel in the IMS grid.

        Uses pyproj to transform the 1024 × 1024 polar stereographic pixel
        centers (EPSG:3411) to WGS84 geographic coordinates (EPSG:4326).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(lats_2d, lons_2d)`` each of shape ``(grid_rows, grid_cols)``
            in degrees.

        Notes
        -----
        Pixel centers are at half-cell offsets from the grid origin. The IMS
        grid origin is the upper-left corner, so:
            x_center[col] = x_origin + (col + 0.5) * resolution_m
            y_center[row] = y_origin - (row + 0.5) * resolution_m  (y decreases southward)
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
