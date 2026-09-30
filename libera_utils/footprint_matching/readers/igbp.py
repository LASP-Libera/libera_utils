"""IGBP land cover reader plugin for the footprint matching pipeline.

Data source: MODIS MCD12Q1 (Terra+Aqua Combined Land Cover Type Yearly L3 Global 500m)
- Product: MCD12Q1
- Version: 061 (current)
- Format: HDF4 (.hdf)
- Spatial resolution: 463.3 m sinusoidal cell (the nominal 500 m grid), stored as 0.463 km
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

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from libera_utils.footprint_matching.readers._hdf4_io import _SinusoidalGeometry, read_modis_sinusoidal_hdf4
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, TimeMatch, VariableSpec

# MCD12Q1 ``LC_Type1`` valid IGBP land-cover codes are 1..17, where 17 = Water Bodies
# (255 is fill). These codes are 1-based and align 1:1 with the low 17 members of
# ``scene_identification.scene_id.IGBPSurfaceType`` (1 = Evergreen Needleleaf .. 17 =
# Water Bodies), so a modal class read here maps straight through
# ``calculate_trmm_surface_type`` without any renumbering. The enum's 18..20 (tundra /
# fresh-snow / sea-ice) belong to the 20-class TRMM/CERES IGBP variant and are never
# emitted by MODIS ``LC_Type1``; a value of 0 is likewise not a valid class (it only
# ever arises downstream as the no-data sentinel when a footprint has no usable IGBP
# pixels). Out-of-domain codes (the 255 fill included) are masked to NaN on every served
# block, so this reader emits only enum-consistent classes.
_IGBP_VALID_MIN: int = 1
_IGBP_VALID_MAX: int = 17

# Histogram size for the categorical PSF aggregation. Sized to the full 20-class
# ``IGBPSurfaceType`` domain (an upper bound on ``LC_Type1``'s 1..17) so the bins line
# up with the enum; unused high bins simply receive zero weight.
_N_IGBP_CATEGORIES: int = 20


class IGBPReader(StaticReader):
    """Read IGBP land cover classification from MODIS MCD12Q1 HDF4 tile files.

    The MCD12Q1 product stores IGBP land cover as ``LC_Type1`` (uint8) in
    sinusoidal-projected HDF4 tiles. Geographic coordinates are not stored as SDS
    arrays; each tile's pixel geometry comes from its HDF-EOS ``StructMetadata.0``
    via :func:`read_modis_sinusoidal_hdf4`.

    Geolocation: native pixels, geolocated per query
    ------------------------------------------------
    The sinusoidal projection is *not* axis-aligned in lat/lon -- longitude varies
    along each row (``lon = X / (R cos(lat))``), so a single longitude per column
    places no real pixel. Every pixel is served at its own position, and the PSF
    weigher is the first place a pixel is weighted. The staged tiles (one file each)
    are held as their uint8 codes plus their geometry (:class:`_SinusoidalTiles`, 1
    byte per pixel); a query computes positions only for the block of each tile its
    box selects.

    The yearly product has no time axis, so the staged granules are the time match
    (``TIME_MATCH = DAILY``).

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"igbp"``.
    RESOLUTION_KM : float
        0.463 km, the MCD12Q1 sinusoidal grid cell (463.3 m; MCD12Q1 v061 User Guide).
        The coverage measure credits each pixel with a cell of this side.
    VARIABLES : tuple[VariableSpec, ...]
        Single variable: ``"surface_type"`` (categorical, 20 IGBP classes).

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The MCD12Q1 HDF4 tile file(s) (e.g., ``MCD12Q1.A2023001.h09v05.061.hdf``).
    """

    READER_KEY: str = "igbp"
    # MCD12Q1 land cover is derived from Terra+Aqua MODIS.
    INSTRUMENT: str = "MODIS"
    # The MCD12Q1 v061 User Guide's sinusoidal grid cell, 463.3 m.
    RESOLUTION_KM: float = 0.463
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="surface_type",
            dtype="int16",
            aggregation="weighted_mode", # Confirm
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
    )
    # A footprint spans several IGBP land-cover classes. ``surface_type`` above is the single
    # aggregated result; these three report the ranked mix (1st/2nd/3rd most-common class by
    # PSF-weighted area), computed during PSF aggregation from the same pixels rather than read
    # from a source field -- hence here rather than in VARIABLES.
    ADDITIONAL_PRODUCT_VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="surface_type_primary",
            dtype="int16",
            aggregation="weighted_mode_primary", # Confirm
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
        VariableSpec(
            name="surface_type_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary", # Confirm
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
        VariableSpec(
            name="surface_type_tertiary",
            dtype="int16",
            aggregation="weighted_mode_tertiary", # Confirm
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
    )

    def _load_native(self) -> "_SinusoidalTiles":
        """Read every staged tile's ``LC_Type1`` codes and pixel geometry.

        Returns
        -------
        _SinusoidalTiles
            The tiles' uint8 codes, ``(n_tiles, n_rows, n_cols)``, and their geometry.

        Raises
        ------
        ValueError
            If a tile's ``LC_Type1`` is not uint8, or the staged tiles differ in shape
            or in sphere radius.
        """
        n_tiles = len(self.file_paths)
        first_codes, first_geometry = self._read_tile(self.file_paths[0])
        # One preallocated array for every tile, filled in place.
        data = np.empty((n_tiles, *first_codes.shape), dtype=np.uint8)
        x_ul, y_ul, dx, dy = (np.empty(n_tiles) for _ in range(4))
        radius_m = first_geometry.radius_m
        for i, path in enumerate(self.file_paths):
            codes, geometry = (first_codes, first_geometry) if i == 0 else self._read_tile(path)
            if codes.shape != data.shape[1:] or not np.isclose(geometry.radius_m, radius_m):
                raise ValueError(
                    f"{path}: tile shape {codes.shape} / sphere radius {geometry.radius_m} m differs from the first "
                    f"staged tile ({data.shape[1:]}, {radius_m} m). Stage tiles of one MCD12Q1 product only."
                )
            data[i] = codes
            x_ul[i], y_ul[i], dx[i], dy[i] = geometry.x_ul, geometry.y_ul, geometry.dx, geometry.dy
        return _SinusoidalTiles(data=data, x_ul=x_ul, y_ul=y_ul, dx=dx, dy=dy, radius_m=radius_m)

    @staticmethod
    def _read_tile(path: Path) -> tuple[np.ndarray, _SinusoidalGeometry]:
        """Read one tile's ``LC_Type1`` codes and geometry, rejecting codes that are not uint8."""
        codes, geometry = read_modis_sinusoidal_hdf4(file_path=str(path), data_sds_name="LC_Type1")
        if codes.dtype != np.uint8:
            raise ValueError(
                f"{path}: LC_Type1 is {codes.dtype}, but MCD12Q1 stores IGBP classes as uint8. "
                "Stage the MCD12Q1 v061 tile as distributed."
            )
        return codes, geometry


@dataclass(frozen=True, eq=False)
class _SinusoidalTiles:
    """MODIS sinusoidal tiles held as their codes, geolocated per box query.

    Holds 1 byte per pixel; a query geolocates only the block of each tile its box can
    reach. Pixel ``(r, c)`` of tile ``t`` has its center at
    ``x = x_ul[t] + (c + 0.5) * dx[t]``, ``y = y_ul[t] + (r + 0.5) * dy[t]`` sinusoidal
    meters, i.e. latitude ``y / R`` and longitude ``x / (R cos(latitude))`` (radians).

    Attributes
    ----------
    data : np.ndarray
        uint8 ``LC_Type1`` codes, ``(n_tiles, n_rows, n_cols)``.
    x_ul, y_ul, dx, dy : np.ndarray
        ``(n_tiles,)`` upper-left corners and pixel sizes, sinusoidal meters
        (``dy < 0``).
    radius_m : float
        The projection's sphere radius, meters, shared by every tile.
    n_variables : int
        1: the IGBP class code.
    """

    data: np.ndarray
    x_ul: np.ndarray
    y_ul: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    radius_m: float

    n_variables: ClassVar[int] = 1

    def query_box(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray, None]:
        """Return ``(lats, lons, values, None)`` of the pixels inside one plain box.

        Parameters
        ----------
        bbox : BoundingBox
            A non-wrapping box inside [-180, 180].

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, None]
            ``(n,)`` float64 pixel-center latitudes and longitudes inside the box
            (inclusive edges), ``(1, n)`` float32 IGBP codes with codes outside 1..17
            as NaN, and no times.
        """
        n_tiles, n_rows, n_cols = self.data.shape
        radius = self.radius_m
        # Pixel-center latitude (degrees) of each tile's first and last row.
        lat_first = np.degrees((self.y_ul + 0.5 * self.dy) / radius)
        lat_last = np.degrees((self.y_ul + (n_rows - 0.5) * self.dy) / radius)
        band_hi = np.minimum(bbox.lat_max, np.maximum(lat_first, lat_last))
        band_lo = np.maximum(bbox.lat_min, np.minimum(lat_first, lat_last))
        x_lo, x_hi = _box_x_range(bbox, band_lo, band_hi, radius)
        x_first = self.x_ul + 0.5 * self.dx
        x_last = self.x_ul + (n_cols - 0.5) * self.dx
        candidates = np.flatnonzero(
            (band_lo <= band_hi) & (x_lo <= np.maximum(x_first, x_last)) & (x_hi >= np.minimum(x_first, x_last))
        )

        lats_out: list[np.ndarray] = []
        lons_out: list[np.ndarray] = []
        codes_out: list[np.ndarray] = []
        for t in candidates:
            # Rows: latitude is linear in the row index; dy < 0, so the northern edge of
            # the box gives the smaller row.
            r_float = (np.radians([bbox.lat_max, bbox.lat_min]) * radius - self.y_ul[t]) / self.dy[t] - 0.5
            # The prefilter kept only tiles whose rows meet the box's latitudes and whose
            # columns meet its x-range, so both index ranges below are non-empty.
            r0 = max(int(np.floor(r_float.min())), 0)
            r1 = min(int(np.ceil(r_float.max())), n_rows - 1)
            row_lat = (self.y_ul[t] + (np.arange(r0, r1 + 1) + 0.5) * self.dy[t]) / radius  # radians
            row_lat_deg = np.degrees(row_lat)
            row_x_lo, row_x_hi = _box_x_range(bbox, row_lat_deg.min(), row_lat_deg.max(), radius)
            c_float = (np.array([row_x_lo, row_x_hi]) - self.x_ul[t]) / self.dx[t] - 0.5
            c0 = max(int(np.floor(c_float.min())), 0)
            c1 = min(int(np.ceil(c_float.max())), n_cols - 1)
            x = self.x_ul[t] + (np.arange(c0, c1 + 1) + 0.5) * self.dx[t]
            with np.errstate(divide="ignore", invalid="ignore"):
                lon = np.degrees(x[np.newaxis, :] / (radius * np.cos(row_lat)[:, np.newaxis]))
            lat = np.broadcast_to(row_lat_deg[:, np.newaxis], lon.shape)
            # The inclusive box mask; every box half lies inside [-180, 180], so it also
            # excludes the pixels off the sinusoidal globe (|lon| > 180).
            inside = (lat >= bbox.lat_min) & (lat <= bbox.lat_max) & (lon >= bbox.lon_min) & (lon <= bbox.lon_max)
            codes = self.data[t, r0 : r1 + 1, c0 : c1 + 1][inside]
            lats_out.append(lat[inside])
            lons_out.append(lon[inside])
            codes_out.append(codes)

        if not codes_out:
            return np.empty(0), np.empty(0), np.empty((1, 0), dtype=np.float32), None
        codes = np.concatenate(codes_out)
        in_domain = (codes >= _IGBP_VALID_MIN) & (codes <= _IGBP_VALID_MAX)
        values = np.where(in_domain, codes, np.nan).astype(np.float32)[np.newaxis, :]
        return np.concatenate(lats_out), np.concatenate(lons_out), values, None


def _box_x_range(
    bbox: BoundingBox, band_lo: np.ndarray | float, band_hi: np.ndarray | float, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sinusoidal x-range (meters) of the box's longitudes over a latitude band.

    At latitude ``phi`` a pixel is in the box when ``x`` lies in
    ``[lon_min, lon_max] * cos(phi) * R`` (longitudes in radians). Over the band, the
    lower bound is smallest at the pole-most latitude when ``lon_min > 0`` and at the
    equator-most one otherwise; the upper bound is largest at the equator-most latitude
    when ``lon_max > 0`` and at the pole-most one otherwise.
    """
    band_lo = np.asarray(band_lo, dtype=np.float64)
    band_hi = np.asarray(band_hi, dtype=np.float64)
    crosses_equator = (band_lo <= 0.0) & (band_hi >= 0.0)
    abs_lo, abs_hi = np.abs(np.radians(band_lo)), np.abs(np.radians(band_hi))
    c_max = np.where(crosses_equator, 1.0, np.cos(np.minimum(abs_lo, abs_hi)))
    c_min = np.cos(np.maximum(abs_lo, abs_hi))
    lon_min, lon_max = np.radians(bbox.lon_min), np.radians(bbox.lon_max)
    x_lo = lon_min * (c_min if lon_min > 0 else c_max) * radius
    x_hi = lon_max * (c_max if lon_max > 0 else c_min) * radius
    return x_lo, x_hi
