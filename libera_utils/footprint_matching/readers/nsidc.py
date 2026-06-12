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
  0        No ice and no snow — open ocean, snow-free land, or outside the grid
  1–100    Sea ice concentration in percent (1 = 1%, 100 = 100%)
  101      Permanent ice (Greenland Ice Sheet, Antarctic ice shelves)
  102      Not used (belongs to no class — maps to 0.0 in every layer)
  103–110  Dry snow on land
  255      Missing/fill (no retrieval)

Per-code output layers
----------------------
Rather than collapse the Extent SDS into a single sea ice field, this reader
splits it into five independent ``float32`` coverage layers, one per meaningful
code group. Each layer carries a per-pixel value that the PSF aggregation engine
turns into a footprint-level *fraction* via ``weighted_mean``; ``sea_ice`` also
preserves sub-pixel concentration magnitude:

  ===================== =========== ============================================
  Variable              Codes       Per-pixel value
  ===================== =========== ============================================
  sea_ice_concentration 1–100       code / 100.0  (0.01–1.0 concentration)
  no_ice_or_snow        0           1.0 else 0.0
  permanent_ice         101         1.0 else 0.0
  dry_snow_on_land      103–110     1.0 else 0.0
  missing               255         1.0 else 0.0
  ===================== =========== ============================================

Design notes
~~~~~~~~~~~~
- Code 102 ("not used") deliberately maps to 0.0 in *all five* layers — it is a
  reserved value that belongs to no surface class.
- The five layers do **not** sum to exactly 1.0: ``sea_ice_concentration`` holds
  a fractional concentration (intended), while the other four are 0/1 indicators.
  This is by design — four surface-class fractions plus a mean ice concentration.
- ``no_ice_or_snow`` (code 0) is *not* strictly open ocean: NISE collapses open
  water, snow-free land, and out-of-grid pixels into code 0, so this layer should
  be read as "neither ice nor snow", not as an ocean mask.

The first axis of the returned data array follows ``VARIABLES`` order exactly
(the multi-variable reader contract; see ``GridTile`` and ``ERA5Reader``).

Geolocation: rasterization, not mean-collapse
----------------------------------------------
The EASE-Grid North (EPSG:3408) projection is azimuthal — both latitude and
longitude vary in two dimensions across the pixel grid, so a 1-D row/column mean
of the pixel coordinates places no real pixel and mis-geolocates the data. To
carry each pixel's true file-derived position through to footprint matching,
this reader flattens the per-pixel ``(lat, lon, value)`` layers to points and
bins them onto a regular sub-grid over the requested tile via
``rasterize_points_to_grid`` — exactly like the ``ssf`` and ``cldpix`` swath
readers. Every output cell therefore has an exact center lat/lon. Pixels whose
EPSG:3408 → WGS84 transform falls outside the geographic domain carry NaN
coordinates and are dropped by the rasterizer's finite-coordinate filter.

References
----------
Product page:    https://nsidc.org/data/nise
User guide:      https://nsidc.org/sites/default/files/nise5-v001-userguide.pdf
Data access:     https://n5eil01u.ecs.nsidc.org/NISE/  (Earthdata login required)
File naming:     NISE_SSMISF{ss}_{YYYYMMDD}.HDFEOS  (NISE v5; ss = DMSP flight, e.g. F18)
                 NISE_AMSR2_{YYYYMMDD}.HDFEOS        (NISE_A2 v1)
EPSG:3408 desc:  https://epsg.io/3408
EASE-Grid ref:   https://nsidc.org/ease
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj

from libera_utils.footprint_matching.readers._hdf4_io import _require_pyhdf
from libera_utils.footprint_matching.readers._swath import rasterize_points_to_grid
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

# --- NISE Extent code groups (see module docstring for the full encoding) ------
# These name the raw uint8 category codes so the mask construction in
# ``_extent_to_category_masks`` reads as plain English rather than magic numbers.
_CODE_NO_ICE_OR_SNOW: int = 0          # open ocean, snow-free land, or outside grid
_SEA_ICE_CODE_MIN: int = 1             # 1 % sea ice concentration
_SEA_ICE_CODE_MAX: int = 100           # 100 % sea ice concentration
_CODE_PERMANENT_ICE: int = 101         # Greenland / Antarctic ice sheets
_SNOW_CODE_MIN: int = 103              # dry snow on land (lower bound)
_SNOW_CODE_MAX: int = 110              # dry snow on land (upper bound)
_CODE_MISSING: int = 255               # fill / no retrieval

# Percent → fraction divisor for the 1–100 sea ice concentration codes.
_SEA_ICE_PERCENT_DIVISOR: float = 100.0

# Canonical output-variable order. The first axis of the array returned by
# ``_load_spatial_region`` / ``_extent_to_category_masks`` MUST match this order,
# which in turn MUST match the ``VARIABLES`` tuple below (the multi-variable
# reader contract documented on ``GridTile``).
_VARIABLE_ORDER: tuple[str, ...] = (
    "sea_ice_concentration",
    "no_ice_or_snow",
    "permanent_ice",
    "dry_snow_on_land",
    "missing",
)


class NISEReader(GriddedDataReader):
    """Read NISE surface-class coverage layers from an HDF-EOS4 file.

    The NISE product (Near-real-time Ice and Snow Extent) distributes a single
    uint8 ``Extent`` SDS whose category codes encode sea ice concentration, snow
    on land, permanent ice, and ice/snow-free pixels. This reader splits those
    codes into five independent float32 coverage layers (see module docstring),
    reprojects the EASE-Grid North (EPSG:3408) pixel centers to WGS84 lat/lon,
    and rasterizes them onto a regular sub-grid covering the requested bounding
    box (see the module docstring's geolocation note), returning a 3-D array of
    shape ``(5, n_lat, n_lon)`` in ``VARIABLES`` order.

    The grid parameters (rows, cols, resolution, origin) are exposed as
    constructor keyword arguments with real defaults so that tests can inject
    a small synthetic grid without building a full 721 × 721 fixture.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"nise"``.
    RESOLUTION_KM : float
        25 km (NISE EASE-Grid North 25-km product).
    OUTPUT_CELL_DEG : float
        Edge length of the rasterized output cells (degrees). 0.25° ≈ 25 km,
        mirroring the ``ssf`` swath reader's cell size.
    REQUIRED_MODE : OperationalMode
        Active in all modes starting from CAM.
    VARIABLES : tuple[VariableSpec, ...]
        Five fractional-coverage variables, all continuous float32 with
        ``weighted_mean`` aggregation and range 0.0–1.0:
        ``"sea_ice_concentration"``, ``"no_ice_or_snow"``, ``"permanent_ice"``,
        ``"dry_snow_on_land"``, and ``"missing"``.

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
    OUTPUT_CELL_DEG: float = 0.25
    REQUIRED_MODE: OperationalMode = OperationalMode.CAM
    # One VariableSpec per Extent code group, in canonical ``_VARIABLE_ORDER``.
    # All five are continuous fractional layers (float32, weighted_mean): the PSF
    # engine averages them to get the fraction of each footprint occupied by the
    # class (and, for sea ice, the mean concentration). ``n_categories=None``
    # because these are fractions, not discrete class codes.
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="sea_ice_concentration",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="no_ice_or_snow",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="permanent_ice",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="dry_snow_on_land",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="missing",
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

    def _load_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read the Extent SDS and flatten it to geolocated coverage points.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, values)`` where ``lats``/``lons`` are 1-D
            ``(n_pixels,)`` float64 pixel-centre coordinates (WGS84, reprojected
            from EPSG:3408), and ``values`` is float64 shape ``(5, n_pixels)``
            holding the five coverage layers in ``_VARIABLE_ORDER``. Pixels
            outside the EPSG:3408 domain carry NaN coordinates and are dropped by
            the rasterizer's finite-coordinate filter.
        """
        raw = self._read_extent_sds()
        lats_2d, lons_2d = self._compute_latlon_grid()

        # Split the single Extent SDS into the five per-code coverage layers.
        # Shape: (5, n_rows, n_cols), axis 0 in canonical _VARIABLE_ORDER.
        masks = self._extent_to_category_masks(raw)

        lats = np.asarray(lats_2d, dtype=np.float64).ravel()
        lons = np.asarray(lons_2d, dtype=np.float64).ravel()
        # Flatten the lat/lon axes while keeping the variable axis: (5, n_pixels).
        values = masks.reshape(masks.shape[0], -1).astype(np.float64)
        return lats, lons, values

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize the five NISE coverage layers within ``bbox`` onto a sub-grid.

        Flattens the EASE-Grid pixels to geolocated points (see
        :meth:`_load_points`) and bins them onto a regular ``OUTPUT_CELL_DEG``
        sub-grid covering the tile, taking the per-cell mean of each layer.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(5, n_lat, n_lon)``. Axis 0 follows ``VARIABLES`` order
            (``_VARIABLE_ORDER``); covered cells hold values in [0.0, 1.0] and
            cells with no pixels are ``NaN``. ``lats``/``lons`` are 1-D
            cell-centre coordinate arrays.
        """
        lats, lons, values = self._load_points()
        return rasterize_points_to_grid(
            point_lats=lats,
            point_lons=lons,
            values=values,
            bbox=(bbox.lat_min, bbox.lat_max, bbox.lon_min, bbox.lon_max),
            cell_size_deg=self.OUTPUT_CELL_DEG,
            aggregations=[v.aggregation for v in self.VARIABLES],
        )

    def _extent_to_category_masks(self, raw: np.ndarray) -> np.ndarray:
        """Split the raw NISE Extent codes into five float32 coverage layers.

        Each layer is a per-pixel value in [0.0, 1.0] (see the module docstring
        for the full encoding). ``sea_ice_concentration`` keeps the fractional
        concentration from codes 1–100; the other four layers are 0/1 indicators.
        Code 102 ("not used") is intentionally absent from every layer.

        Parameters
        ----------
        raw : np.ndarray
            uint8 array of NISE Extent codes, shape ``(n_rows, n_cols)``.

        Returns
        -------
        np.ndarray
            float32 array of shape ``(5, n_rows, n_cols)``; axis 0 follows
            ``_VARIABLE_ORDER`` (and therefore ``VARIABLES``).
        """
        # Codes 1–100 are sea ice concentration in percent → fraction in (0, 1].
        sea_ice_concentration = np.where(
            (raw >= _SEA_ICE_CODE_MIN) & (raw <= _SEA_ICE_CODE_MAX),
            raw.astype(np.float32) / _SEA_ICE_PERCENT_DIVISOR,
            np.float32(0.0),
        )
        # The remaining layers are binary presence indicators (1.0 where the code
        # matches, else 0.0). ``weighted_mean`` later turns these into the
        # fraction of a footprint occupied by each class.
        no_ice_or_snow = (raw == _CODE_NO_ICE_OR_SNOW).astype(np.float32)
        permanent_ice = (raw == _CODE_PERMANENT_ICE).astype(np.float32)
        dry_snow_on_land = (
            (raw >= _SNOW_CODE_MIN) & (raw <= _SNOW_CODE_MAX)
        ).astype(np.float32)
        missing = (raw == _CODE_MISSING).astype(np.float32)

        # Stack in canonical order. Keep this list aligned with _VARIABLE_ORDER /
        # VARIABLES — the aggregation engine indexes layers by that position.
        return np.stack(
            [
                sea_ice_concentration,
                no_ice_or_snow,
                permanent_ice,
                dry_snow_on_land,
                missing,
            ],
            axis=0,
        ).astype(np.float32)

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
