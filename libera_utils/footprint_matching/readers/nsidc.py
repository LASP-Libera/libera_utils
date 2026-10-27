"""NISE sea ice reader plugin for the footprint matching pipeline.

Data source: NSIDC Near-real-time Ice and Snow Extent (NISE) Product

- Product: NISE_A2 (AMSR-2, v1)
- Format: HDF-EOS4 (.HDFEOS), opened via pyhdf
- Spatial resolution: 25 km nominal (NSIDC EASE-Grid North & South)
- Grid: two 721 × 721 polar azimuthal equal-area grids — Northern Hemisphere
  (NSIDC EASE-Grid North, EPSG:3408) and Southern Hemisphere (EASE-Grid South,
  EPSG:3409), both carried in the single granule
- Temporal coverage: 2012-present (NISE_A2)
- Spatial coverage: Northern and Southern Hemispheres (both grids are read and
  merged into one geolocated point cloud)
- Data variable: Extent SDS, one per hemisphere (uint8 category codes 0–255)

NISE Extent encoding
--------------------
The authoritative per-code meaning is the ``data_grid_key`` attribute embedded in
every granule:

  ======== ======================================================================
  Code     Meaning
  ======== ======================================================================
  0        Snow-free land
  1–100    Sea ice concentration in percent (1 = 1%, 100 = 100%)
  101      Permanent ice (Greenland Ice Sheet, Antarctic ice shelves)
  102      Not used (belongs to no class — missing, NaN, in every layer)
  103      Dry snow on land
  104      Wet snow on land
  105–251  Not used
  252      Mixed pixels at coastlines (microwave algorithm unreliable)
  253      Suspect ice value
  254      Off-Earth corner fill — the square grid's corners fall outside the
           projection disk; these pixels are dropped (see "Two hemispheres" below)
  255      Ocean (open water, no ice)
  ======== ======================================================================

Per-code output layers
----------------------
Rather than collapse the Extent SDS into a single sea ice field, this reader
splits it into five independent ``float32`` coverage layers, one per surface
class. Each layer carries a per-pixel value that the PSF aggregation engine
turns into a footprint-level *fraction* via ``weighted_mean``;
``sea_ice_concentration`` also preserves sub-pixel concentration magnitude:

  ===================== =========== ============================================
  Variable              Codes       Per-pixel value
  ===================== =========== ============================================
  sea_ice_concentration 1–100       code / 100.0  (0.01–1.0 concentration)
  snow_free_land        0           1.0 else 0.0
  permanent_ice         101         1.0 else 0.0
  snow_on_land          103–104     1.0 else 0.0   (dry + wet snow)
  open_ocean            255         1.0 else 0.0
  ===================== =========== ============================================

Design notes
~~~~~~~~~~~~
- Codes that belong to no surface class are missing (NaN) in *all five* layers: 102
  and 105–251 ("not used"), plus the quality flags 252 (coastline mixed pixels) and
  253 (suspect ice value). The weigher counts such a pixel as missing, so it lowers a
  footprint's coverage rather than pulling every class fraction toward zero. Code 254
  (off-Earth corners) is dropped from the point cloud entirely (see below).
- The five layers do **not** sum to exactly 1.0: ``sea_ice_concentration`` holds
  a fractional concentration (intended), while the other four are 0/1 indicators.
  This is by design — four surface-class fractions plus a mean ice concentration.
- ``snow_free_land`` (code 0) and ``open_ocean`` (code 255) are distinct classes:
  the granule separates snow-free land from open water, so neither subsumes the
  other. Code 255 is ocean, the single largest class in each polar grid.

The first axis of the returned ``values`` array follows ``VARIABLES`` order exactly
(the multi-variable reader contract; see ``PointSamples`` and ``ERA5SurfaceReader``).

Two hemispheres in one granule
------------------------------
A NISE granule holds two grids — Northern (EPSG:3408) and Southern (EPSG:3409) —
each with its own ``Extent`` SDS. Both fields are named ``Extent``, so they are
addressed by their HDF-EOS dimension-name hemisphere label, not by SDS name (see
``_read_extent_sds``). This reader reads both, reprojects each with its own CRS,
and concatenates the results into a single point cloud; a footprint's box query
downstream keeps only the points inside its box, so each hemisphere's footprints
draw from the correct grid automatically.

Each square 721 × 721 grid's corners fall outside the projection's Earth disk and
reproject into the *opposite* hemisphere; every such corner pixel carries Extent
code 254. These are dropped before merging (``_load_native``) so they cannot
contaminate the other hemisphere's footprints.

Geolocation: native pixels, not a regular grid
----------------------------------------------
The EASE-Grid projections are azimuthal -- both latitude and longitude vary in two
dimensions across the pixel grid, so a 1-D row/column mean of the pixel coordinates
places no real pixel and mis-geolocates the data. Every pixel is therefore served at
its own reprojected center as a ``PointSamples`` point, and the PSF weigher is the
first place a pixel is weighted. Pixels whose EASE-Grid -> WGS84 transform falls
outside the projection domain reproject to non-finite coordinates (pyproj yields
``inf``) and are dropped by the reader base class's point index.

References
----------
Product page:    https://nsidc.org/data/nise
User guide:      https://nsidc.org/sites/default/files/nise5-v001-userguide.pdf
Data access:     https://n5eil01u.ecs.nsidc.org/NISE/  (Earthdata login required)
File naming:     NISE_AMSR2_{YYYYMMDD}.HDFEOS  (NISE_A2 v1)
EPSG:3408 desc:  https://epsg.io/3408  (North)  /  https://epsg.io/3409  (South)
EASE-Grid ref:   https://nsidc.org/ease
"""

from dataclasses import dataclass

import numpy as np
import pyhdf.SD as sd
import pyproj

from libera_utils.footprint_matching.readers._eos_metadata import parse_eos_grid, struct_metadata_error
from libera_utils.footprint_matching.readers._hdf4_io import read_hdf4_struct_metadata
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import OperationalMode, PointSamples, TimeMatch, VariableSpec

# NISE 25-km EASE-Grid North parameters (EPSG:3408).
# The 721 × 721 grid covers ±9,036,842.7625 m in both x and y.
# Source: NSIDC EASE-Grid North 25-km grid specification
_DEFAULT_GRID_ROWS: int = 721
_DEFAULT_GRID_COLS: int = 721
_DEFAULT_RESOLUTION_M: float = 25_067.525  # EASE-Grid North 25-km cell size in meters
# Upper-left corner of the grid in EPSG:3408/3409 meters (x = easting, y = northing).
# Both grids share this extent; only the CRS differs. Each grid's StructMetadata.0 entry
# is checked against it at load (see ``NISEReader._check_grid``).
_DEFAULT_X_ORIGIN: float = -9_036_842.7625  # meters (upper-left x)
_DEFAULT_Y_ORIGIN: float = 9_036_842.7625  # meters (upper-left y)
# The GCTP projection StructMetadata.0 names for the EASE-Grid (Lambert azimuthal equal-area).
_EASE_GRID_PROJECTION: str = "GCTP_LAMAZ"
# How far a StructMetadata.0 corner may sit from the expected one, meters: the attribute
# writes corners to the micrometer, so this only absorbs decimal rounding.
_CORNER_TOLERANCE_M: float = 0.01
_STRUCT_METADATA_ADVICE: str = (
    "A NISE or NISE_A2 granule carries its EASE-Grid geometry there; stage the 25 km granule as distributed."
)

# The two NSIDC EASE-Grid projections carried in one NISE granule.
_DEFAULT_EPSG_NORTH: int = 3408  # NSIDC EASE-Grid North (azimuthal equal-area, +90 deg center)
_DEFAULT_EPSG_SOUTH: int = 3409  # NSIDC EASE-Grid South (azimuthal equal-area, -90 deg center)

# Substrings of the HDF-EOS grid dimension names ("YDim:Northern Hemisphere", ...)
# used to pick the correct ``Extent`` SDS per hemisphere. Both grids name their data
# field ``Extent``, so name-based selection (``hdf.select("Extent")``) always returns
# the first (Northern) SDS -- selection must go by the dimension-name hemisphere label.
_HEMISPHERE_LABEL_NORTH: str = "Northern Hemisphere"
_HEMISPHERE_LABEL_SOUTH: str = "Southern Hemisphere"

# --- NISE Extent code groups (see module docstring for the full encoding) ------
# These name the raw uint8 category codes so the mask construction in
# ``_extent_to_category_masks`` reads as plain English rather than magic numbers.
# Values follow the granule-embedded ``data_grid_key`` (authoritative).
_CODE_SNOW_FREE_LAND: int = 0  # snow-free land (distinct from ocean, code 255)
_SEA_ICE_CODE_MIN: int = 1  # 1 % sea ice concentration
_SEA_ICE_CODE_MAX: int = 100  # 100 % sea ice concentration
_CODE_PERMANENT_ICE: int = 101  # Greenland / Antarctic ice sheets
_SNOW_CODE_MIN: int = 103  # snow on land: dry snow (lower bound)
_SNOW_CODE_MAX: int = 104  # snow on land: wet snow (upper bound)
_CODE_NOT_USED: int = 102  # not used; belongs to no class
_NO_CLASS_CODE_MIN: int = 105  # 105-251 not used, 252 coastline mixed pixels, 253 suspect ice
_NO_CLASS_CODE_MAX: int = 253
_CODE_OFF_EARTH: int = 254  # off-Earth corner fill (square-grid corners outside the projection disk)
_CODE_OCEAN: int = 255  # ocean (open water, no ice)

# Percent → fraction divisor for the 1–100 sea ice concentration codes.
_SEA_ICE_PERCENT_DIVISOR: float = 100.0

# Canonical output-variable order. The first axis of the array returned by
# ``_load_native`` / ``_extent_to_category_masks`` MUST match this order,
# which in turn MUST match the ``VARIABLES`` tuple below (the multi-variable
# reader contract documented on ``PointSamples``).
_VARIABLE_ORDER: tuple[str, ...] = (
    "sea_ice_concentration",
    "snow_free_land",
    "permanent_ice",
    "snow_on_land",
    "open_ocean",
)


@dataclass(frozen=True)
class _HemisphereGrid:
    """One EASE-Grid hemisphere carried in a NISE granule.

    A NISE HDF-EOS4 file holds two grids -- Northern (EPSG:3408) and Southern
    (EPSG:3409) -- sharing the identical 721 x 721 25-km geometry and differing only
    in projection center. This bundles the per-hemisphere parameters so the reader can
    loop over both uniformly.

    Attributes
    ----------
    epsg : int
        Source CRS for the pixel-center reprojection (3408 North, 3409 South).
    hemisphere_label : str
        The HDF-EOS ``GridName`` of this hemisphere, also matched against the SDS
        dimension names to select its ``Extent`` SDS (see ``_HEMISPHERE_LABEL_NORTH``).
    grid_rows, grid_cols : int
        Grid dimensions (default 721 x 721).
    resolution_m : float
        Cell size in meters (default 25 067.525).
    x_origin, y_origin : float
        Upper-left corner in projection meters (default -/+ 9 036 842.7625).
    """

    epsg: int
    hemisphere_label: str
    grid_rows: int = _DEFAULT_GRID_ROWS
    grid_cols: int = _DEFAULT_GRID_COLS
    resolution_m: float = _DEFAULT_RESOLUTION_M
    x_origin: float = _DEFAULT_X_ORIGIN
    y_origin: float = _DEFAULT_Y_ORIGIN


class NISEReader(StaticReader):
    """Read NISE surface-class coverage layers from an HDF-EOS4 file.

    The NISE product (Near-real-time Ice and Snow Extent) distributes a uint8
    ``Extent`` SDS per hemisphere whose category codes encode sea ice concentration,
    snow on land, permanent ice, and ice/snow-free pixels. This reader splits those
    codes into five independent float32 coverage layers (see module docstring),
    reprojects each hemisphere's EASE-Grid pixel centers to WGS84 lat/lon with its own
    CRS (North EPSG:3408, South EPSG:3409), and merges both hemispheres into one
    point cloud of ``(5, n_pixels)`` values in ``VARIABLES`` order (see the module
    docstring's geolocation note). The daily granule has no time axis, so the staged
    day is the time match (``TIME_MATCH = DAILY``).

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"nise"``.
    RESOLUTION_KM : float
        25 km (NISE EASE-Grid 25-km product).
    VARIABLES : tuple[VariableSpec, ...]
        Five fractional-coverage variables, all continuous float32 with
        ``weighted_mean`` aggregation and range 0.0–1.0:
        ``"sea_ice_concentration"``, ``"snow_free_land"``, ``"permanent_ice"``,
        ``"snow_on_land"``, and ``"open_ocean"``.
    _HEMISPHERES : tuple[_HemisphereGrid, ...]
        The EASE-Grids read from every granule, in order: the Northern (EPSG:3408) and
        Southern (EPSG:3409) 721 x 721 25 km grids.

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The one NISE HDF-EOS4 file (``*.HDFEOS``) for the processing day.
    """

    READER_KEY: str = "nise"
    # The pipeline ingests NISE_A2, built from AMSR-2 passive microwave radiometry,
    # so provenance is tagged AMSR2.
    INSTRUMENT: str = "AMSR2"
    RESOLUTION_KM: float = 25.0
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
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
            name="snow_free_land",
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
            name="snow_on_land",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="open_ocean",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
    )
    # Both hemispheres of the single granule, Northern first.
    _HEMISPHERES: tuple[_HemisphereGrid, ...] = (
        _HemisphereGrid(epsg=_DEFAULT_EPSG_NORTH, hemisphere_label=_HEMISPHERE_LABEL_NORTH),
        _HemisphereGrid(epsg=_DEFAULT_EPSG_SOUTH, hemisphere_label=_HEMISPHERE_LABEL_SOUTH),
    )

    def _build_transformer(self, epsg: int) -> object:
        """Build an ``epsg`` → WGS84 transformer for one hemisphere's pixel centers.

        Called once per hemisphere from :meth:`_load_native`, which the base class
        runs once per reader, so no additional caching is needed. CRS objects (not
        bare EPSG strings) avoid the pyproj FutureWarning about authority-based CRS
        construction.

        Parameters
        ----------
        epsg : int
            Source CRS EPSG code (3408 for EASE-Grid North, 3409 for South).
        """
        return pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(epsg),  # NSIDC EASE-Grid (azimuthal equal-area)
            pyproj.CRS.from_epsg(4326),
            always_xy=True,  # Input: (x=easting, y=northing); Output: (lon, lat)
        )

    def _load_native(self) -> PointSamples:
        """Read every hemisphere's Extent SDS and flatten to one geolocated point cloud.

        Reads each configured hemisphere (Northern EPSG:3408, Southern EPSG:3409),
        reprojects its pixel centers to WGS84 with that hemisphere's CRS, drops the
        off-Earth corner pixels (see below), and concatenates the per-hemisphere
        points into a single cloud. A box query downstream keeps only the points
        inside its box, so northern and southern footprints each draw from the
        correct hemisphere automatically. Every pixel is served at its own
        reprojected center, so the PSF weigher is the first place a pixel is
        weighted.

        Returns
        -------
        PointSamples
            ``lats``/``lons`` are ``(n_pixels,)`` float64 pixel-center coordinates
            (WGS84), and ``values`` is float32 shape ``(5, n_pixels)`` holding the five
            coverage layers in ``_VARIABLE_ORDER``. Pixels outside the EASE-Grid domain
            reproject to non-finite coordinates (pyproj yields ``inf``) and are dropped
            by the reader base class's point index.

        Raises
        ------
        ValueError
            If a hemisphere's ``Extent`` SDS is not the shape its grid declares, or its
            ``StructMetadata.0`` entry is missing or gives other dimensions, corners or
            projection than the EASE-Grid it is geolocated on (see :meth:`_check_grid`).
        """
        struct_meta = read_hdf4_struct_metadata(self.single_file_path)
        hemi_lats: list[np.ndarray] = []
        hemi_lons: list[np.ndarray] = []
        hemi_values: list[np.ndarray] = []
        for hemi in self._HEMISPHERES:
            raw = self._read_extent_sds(hemi.hemisphere_label)
            if raw.shape != (hemi.grid_rows, hemi.grid_cols):
                raise ValueError(
                    f"{self.single_file_path}: the {hemi.hemisphere_label} Extent SDS has shape {raw.shape}, but the "
                    f"EASE-Grid it is geolocated on is {hemi.grid_rows} x {hemi.grid_cols}. Stage a NISE or NISE_A2 "
                    "25 km granule."
                )
            self._check_grid(struct_meta, hemi)
            lats_2d, lons_2d = self._compute_latlon_grid(hemi)
            # Split the Extent SDS into the five per-code coverage layers.
            # Shape: (5, n_rows, n_cols), axis 0 in canonical _VARIABLE_ORDER.
            masks = self._extent_to_category_masks(raw)
            lats = np.asarray(lats_2d, dtype=np.float64).ravel()
            lons = np.asarray(lons_2d, dtype=np.float64).ravel()
            # Flatten the lat/lon axes while keeping the variable axis: (5, n_pixels).
            values = masks.reshape(masks.shape[0], -1).astype(np.float32)
            # Drop off-Earth corner pixels (Extent code 254). Each square 721x721 grid
            # has ~114k corner pixels that fall outside the projection's Earth disk and
            # reproject into the OPPOSITE hemisphere; every one of them carries code 254.
            # If kept, they would land in the other hemisphere's footprints and drag the
            # weighted_mean toward zero (254 maps to 0.0 in all five layers) -- a silent
            # corruption. Distinct from code 255 (ocean), which is retained as the
            # ``open_ocean`` layer.
            keep = raw.ravel() != _CODE_OFF_EARTH
            hemi_lats.append(lats[keep])
            hemi_lons.append(lons[keep])
            hemi_values.append(values[:, keep])
        n_vars = len(self.VARIABLES)
        return PointSamples(
            lats=np.concatenate(hemi_lats) if hemi_lats else np.empty(0, dtype=np.float64),
            lons=np.concatenate(hemi_lons) if hemi_lons else np.empty(0, dtype=np.float64),
            values=np.concatenate(hemi_values, axis=1) if hemi_values else np.empty((n_vars, 0), dtype=np.float32),
        )

    def _check_grid(self, struct_meta: str, hemi: _HemisphereGrid) -> None:
        """Check one hemisphere's ``StructMetadata.0`` grid against the EASE-Grid it is geolocated on.

        Pixel centers are computed from ``hemi``'s fixed geometry, not from the file, so a
        granule whose grid differs in size, corners or projection would be misplaced; this
        stops the load instead.

        Parameters
        ----------
        struct_meta : str
            The granule's ``StructMetadata.0`` text.
        hemi : _HemisphereGrid
            The grid the hemisphere's pixels are geolocated on.

        Raises
        ------
        ValueError
            If ``StructMetadata.0`` has no grid named ``hemi.hemisphere_label`` or lacks
            one of its geometry fields, names a projection other than ``GCTP_LAMAZ``, or
            gives dimensions or corners other than ``hemi``'s.
        """
        source = self.single_file_path
        grid = parse_eos_grid(struct_meta, source, _STRUCT_METADATA_ADVICE, grid_name=hemi.hemisphere_label)
        if grid.projection != _EASE_GRID_PROJECTION:
            raise struct_metadata_error(
                source,
                f"names Projection={grid.projection} for grid {hemi.hemisphere_label!r}, not the EASE-Grid "
                f"{_EASE_GRID_PROJECTION} projection this reader uses to geolocate the pixels",
                _STRUCT_METADATA_ADVICE,
            )
        upper_left = (hemi.x_origin, hemi.y_origin)
        lower_right = (
            hemi.x_origin + hemi.grid_cols * hemi.resolution_m,
            hemi.y_origin - hemi.grid_rows * hemi.resolution_m,
        )
        same_corners = np.allclose(grid.upper_left, upper_left, rtol=0.0, atol=_CORNER_TOLERANCE_M) and np.allclose(
            grid.lower_right, lower_right, rtol=0.0, atol=_CORNER_TOLERANCE_M
        )
        if (grid.ydim, grid.xdim) != (hemi.grid_rows, hemi.grid_cols) or not same_corners:
            raise struct_metadata_error(
                source,
                f"gives grid {hemi.hemisphere_label!r} YDim={grid.ydim}, XDim={grid.xdim}, corners "
                f"{grid.upper_left} to {grid.lower_right} m, but this reader uses the {hemi.grid_rows} x "
                f"{hemi.grid_cols} EASE-Grid with corners {upper_left} to {lower_right} m to geolocate the pixels",
                _STRUCT_METADATA_ADVICE,
            )

    def _extent_to_category_masks(self, raw: np.ndarray) -> np.ndarray:
        """Split the raw NISE Extent codes into five float32 coverage layers.

        Each layer is a per-pixel value in [0.0, 1.0] (see the module docstring
        for the full encoding). ``sea_ice_concentration`` keeps the fractional
        concentration from codes 1–100; the other four layers are 0/1 indicators.
        Codes that belong to no surface class (102, 105–251, and the 252/253
        quality flags) are NaN in every layer, so the weigher counts them as missing.

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
        snow_free_land = (raw == _CODE_SNOW_FREE_LAND).astype(np.float32)
        permanent_ice = (raw == _CODE_PERMANENT_ICE).astype(np.float32)
        snow_on_land = ((raw >= _SNOW_CODE_MIN) & (raw <= _SNOW_CODE_MAX)).astype(np.float32)
        open_ocean = (raw == _CODE_OCEAN).astype(np.float32)

        # Stack in canonical order. Keep this list aligned with _VARIABLE_ORDER /
        # VARIABLES — the aggregation engine indexes layers by that position.
        layers = np.stack(
            [
                sea_ice_concentration,
                snow_free_land,
                permanent_ice,
                snow_on_land,
                open_ocean,
            ],
            axis=0,
        ).astype(np.float32)
        no_class = (raw == _CODE_NOT_USED) | ((raw >= _NO_CLASS_CODE_MIN) & (raw <= _NO_CLASS_CODE_MAX))
        layers[:, no_class] = np.nan
        return layers

    def _read_extent_sds(self, hemisphere_label: str) -> np.ndarray:
        """Open the NISE HDF-EOS4 file and return one hemisphere's Extent SDS.

        A NISE granule holds four SDS -- an ``Extent`` and an ``Age`` per hemisphere
        -- and both hemispheres name their fields identically. ``hdf.select("Extent")``
        resolves the name to the *first* index (always the Northern grid), and
        ``hdf.datasets()`` is name-keyed and hides the duplicates, so neither can
        address the Southern grid. This iterates the real dataset indices and matches
        the requested hemisphere by the SDS's dimension name
        (e.g. ``"YDim:Northern Hemisphere"``).

        Parameters
        ----------
        hemisphere_label : str
            Substring of the grid dimension name identifying the hemisphere
            (see ``_HEMISPHERE_LABEL_NORTH`` / ``_HEMISPHERE_LABEL_SOUTH``).

        Returns
        -------
        np.ndarray
            Shape ``(grid_rows, grid_cols)``, dtype uint8. Each element holds
            a NISE Extent code (see module docstring for encoding).

        Raises
        ------
        KeyError
            If the file has no ``Extent`` SDS for the requested hemisphere.
        pyhdf.error.HDF4Error
            If the file cannot be opened (not found, not HDF4, corrupt header).
        """
        hdf = sd.SD(str(self.single_file_path), sd.SDC.READ)
        try:
            # hdf.info()[0] is the true dataset count; iterate indices (not
            # hdf.datasets(), whose name keys collapse the duplicate Extent/Age SDS).
            n_datasets = hdf.info()[0]
            for index in range(n_datasets):
                sds = hdf.select(index)
                name, rank = sds.info()[0], sds.info()[1]
                if name != "Extent" or rank < 1:
                    continue
                dim_name = sds.dim(0).info()[0]
                if hemisphere_label in dim_name:
                    return np.array(sds.get(), dtype=np.uint8)
            raise KeyError(
                f"No 'Extent' SDS for hemisphere {hemisphere_label!r} in {self.single_file_path}. Check that the "
                "staged file is a NISE or NISE_A2 HDF-EOS granule carrying both the Northern and Southern "
                "Hemisphere grids."
            )
        finally:
            hdf.end()

    def _compute_latlon_grid(self, hemi: _HemisphereGrid) -> tuple[np.ndarray, np.ndarray]:
        """Compute geographic lat/lon coordinates for every pixel in one hemisphere grid.

        Uses pyproj to transform the EASE-Grid pixel centers (``hemi.epsg``: 3408 for
        the North grid, 3409 for the South) to WGS84 geographic coordinates (EPSG:4326).

        Parameters
        ----------
        hemi : _HemisphereGrid
            The hemisphere grid to reproject.

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

        Pixels whose projected coordinates fall outside the EASE-Grid domain reproject
        to non-finite coordinates (pyproj yields ``inf``); these are excluded by the
        reader base class's point index.
        """
        col_indices = np.arange(hemi.grid_cols, dtype=np.float64)
        row_indices = np.arange(hemi.grid_rows, dtype=np.float64)

        x_centers = hemi.x_origin + (col_indices + 0.5) * hemi.resolution_m
        y_centers = hemi.y_origin - (row_indices + 0.5) * hemi.resolution_m

        # Build 2-D coordinate grids for the transformer.
        x_2d, y_2d = np.meshgrid(x_centers, y_centers)

        # always_xy=True means input is (x, y) and output is (lon, lat).
        transformer = self._build_transformer(hemi.epsg)
        lons_2d, lats_2d = transformer.transform(x_2d, y_2d)

        return lats_2d, lons_2d
