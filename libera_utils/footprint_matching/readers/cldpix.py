"""CERES CLDPIX imager-pixel cloud reader plugin.

Data source: CERES Cloud Pixel (CLDPIX) imager-resolution cloud retrievals
- Example product: CER_CLDPIX_NOAA20-VIIRS
- Format: NetCDF4 with flat (root-level) variables on a 2-D imager swath
  ``(Scanlines, Pixels)`` grid (e.g. 16120 × 400).
- Geolocation: per-pixel 2-D ``Latitude`` / ``Longitude`` arrays (longitude
  stored 0..360 — normalized here).
- Native scale: ~1 km (VIIRS imager pixel).

Why rasterize?
--------------
CLDPIX is a swath (the 2-D ``(Scanlines, Pixels)`` arrays are *not* a
regular lat/lon grid — each pixel carries its own geolocation). To stay within
the ``GriddedDataReader`` / ``GridTile`` contract, this reader flattens the
swath to points and bins them onto a regular sub-grid covering each requested
2° tile (see :mod:`libera_utils.footprint_matching.readers._swath`). The file
is parsed once and cached on the instance.

Memory note
-----------
A full CLDPIX granule holds millions of pixels; reading every selected variable
into memory at once is sizable (hundreds of MB). The single parse is cached per
reader instance and reused across tiles. A future TileManager may prefer a
spatial pre-index for large-scale processing (TODO[LIBSDC-785]).

Variable set
------------
This is a **minimal starter set** drawn from the footprint-matching
dependencies in ``data_products.md`` (CLDPIX supplies imager-pixel cloud
properties for the Imager / Imager-camera-time products). Expected to be
refined.

Surface-type variables (``IGBP_Ecosystem``, ``Snow_Map_Value``, ``Ice_Map_Value``)
are intentionally NOT extracted here. The pipeline uses the dedicated
``IGBPReader`` and ``NISEReader`` as the authoritative sources for land-cover
and ice/snow classification, avoiding duplication and ensuring consistency
across all operational modes.

References
----------
CERES cloud products: https://ceres.larc.nasa.gov/data/
File naming: CER_CLDPIX_{platform}-{imager}_{config}_{prod}.{YYYYMMDDHH}.nc
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from libera_utils.footprint_matching.readers._swath import (
    apply_fill_and_valid_range,
    normalize_longitude,
    rasterize_points_to_grid,
)
from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# Fill sentinels: floats use the float32 max; int8 categoricals use 127.
_FILL_FLOAT: float = 3.4028235e38
_FILL_INT8: int = 127

# Root-level 2-D geolocation variable names.
_LAT_VAR: str = "Latitude"
_LON_VAR: str = "Longitude"


class _CLDPIXField(NamedTuple):
    """Mapping from an output variable to its source variable and decoding rules.

    Attributes
    ----------
    out_name : str
        Output variable name exposed via ``VARIABLES``.
    var : str
        Root-level source variable name.
    aggregation : str
        Spatial aggregation strategy (passed to the rasterizer).
    fill : float or int
        Fill sentinel for the source variable.
    valid_range : tuple of (float, float) or None
        Inclusive valid range; values outside become NaN. Also used to drop the
        ``-1`` "no data" sentinel in the snow/ice maps.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous.
    """

    out_name: str
    var: str
    aggregation: str
    fill: float | int
    valid_range: tuple[float, float] | None
    n_categories: int | None


# Minimal starter field set (refine later — see module docstring).
# Note: surface-type variables (IGBP_Ecosystem, Snow_Map_Value, Ice_Map_Value)
# are deliberately omitted — see module docstring for the rationale.
_CLDPIX_FIELDS: tuple[_CLDPIXField, ...] = (
    # --- continuous cloud properties ---
    _CLDPIXField("cloud_optical_depth", "Eff_Cld_Optical_Depth", "weighted_log_mean", _FILL_FLOAT, (0.25, 150.0), None),
    _CLDPIXField("cloud_water_path", "Cld_Water_Path", "weighted_mean", _FILL_FLOAT, (0.0, 10000.0), None),
    _CLDPIXField("cloud_effective_temperature", "Eff_Cld_Temp", "weighted_mean", _FILL_FLOAT, (190.0, 350.0), None),
    _CLDPIXField("cloud_effective_height", "Eff_Cld_Height", "weighted_mean", _FILL_FLOAT, (0.0, 18.0), None),
    _CLDPIXField("cloud_effective_pressure", "Eff_Cld_Pressure", "weighted_mean", _FILL_FLOAT, (10.0, 1100.0), None),
    _CLDPIXField("cloud_top_height", "Top_Cld_Height", "weighted_mean", _FILL_FLOAT, None, None),
    # Effective cloud particle radius (μm). Cld_Radius is the combined
    # (water+ice blended) effective radius produced by the CERES cloud
    # retrieval algorithm, distinct from the phase-separated radii
    # (Cld_Radius_0124, Cld_Radius_0160) at specific wavelengths.
    _CLDPIXField("cloud_particle_radius", "Cld_Radius", "weighted_mean", _FILL_FLOAT, (2.0, 60.0), None),
    # --- categorical (mode-aggregated) ---
    _CLDPIXField("cloud_particle_phase", "Cloud_Particle_Phase", "weighted_mode", _FILL_INT8, (1.0, 5.0), 5),
    _CLDPIXField("cloud_mask", "CERES_Cloud_Mask", "weighted_mode", _FILL_INT8, (0.0, 3.0), 4),
)


class CLDPIXReader(GriddedDataReader):
    """Read CERES CLDPIX imager-pixel cloud data and rasterize onto the tile grid.

    Flattens the imager swath to points and bins a minimal set of cloud
    properties onto a regular sub-grid covering each requested 2° tile.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"cldpix"``.
    RESOLUTION_KM : float
        ~1 km (VIIRS imager pixel scale).
    OUTPUT_CELL_DEG : float
        Edge length of the rasterized output cells (degrees).
    REQUIRED_MODE : OperationalMode
        ``IMAGER`` — active for the Imager and Imager-camera-time products.
    VARIABLES : tuple[VariableSpec, ...]
        Minimal starter set (see module docstring).

    Parameters
    ----------
    file_path : Path
        Path to a CERES CLDPIX NetCDF4 file.
    """

    READER_KEY: str = "cldpix"
    RESOLUTION_KM: float = 1.0
    OUTPUT_CELL_DEG: float = 0.05
    REQUIRED_MODE: OperationalMode = OperationalMode.IMAGER
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=f.out_name,
            dtype="float32" if f.n_categories is None else "int16",
            aggregation=f.aggregation,
            required_mode=OperationalMode.IMAGER,
            n_categories=f.n_categories,
        )
        for f in _CLDPIX_FIELDS
    )

    def __init__(self, file_path: Path) -> None:
        super().__init__(file_path)
        # Lazily populated point cache: (lats, lons, values (n_var, n_pts)).
        self._points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _load_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse and cache the per-pixel coordinates and variable values.

        Flattens the 2-D ``(Scanlines, Pixels)`` swath arrays to 1-D point
        arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, values)`` where ``lats``/``lons`` are 1-D
            ``(n_pixels,)`` (longitude normalized to −180..180) and ``values``
            is ``(n_var, n_pixels)`` float64 with fill / out-of-range entries
            as NaN.
        """
        if self._points is not None:
            return self._points

        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(self._file_path), "r") as ds:
            # Disable netCDF4 automatic masking and rely on our own
            # apply_fill_and_valid_range. This is required because some CLDPIX
            # variables (e.g. Eff_Cld_Pressure) store ``valid_range`` in
            # descending order ([1100, 10]); netCDF4 interprets that as
            # [valid_min, valid_max] and masks *every* value. Our helper
            # normalizes the range order, so it handles such fields correctly.
            # Scale/offset auto-application is left enabled (set_auto_scale).
            ds.set_auto_mask(False)

            # 2-D geolocation → flatten to point lists.
            lats = apply_fill_and_valid_range(
                ds.variables[_LAT_VAR][:], fill_value=_FILL_FLOAT, valid_range=(-90.0, 90.0)
            ).ravel()
            lons_raw = apply_fill_and_valid_range(
                ds.variables[_LON_VAR][:], fill_value=_FILL_FLOAT, valid_range=(0.0, 360.0)
            ).ravel()
            lons = normalize_longitude(lons_raw)

            value_rows: list[np.ndarray] = []
            for f in _CLDPIX_FIELDS:
                raw = ds.variables[f.var][:]
                value_rows.append(apply_fill_and_valid_range(raw, fill_value=f.fill, valid_range=f.valid_range).ravel())

            values = np.stack(value_rows, axis=0)

        self._points = (lats, lons, values)
        return self._points

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize CLDPIX pixels within ``bbox`` onto a regular sub-grid.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(n_var, n_lat, n_lon)`` in ``VARIABLES`` order. Cells with no
            pixels are NaN.
        """
        lats, lons, values = self._load_points()
        aggregations = [f.aggregation for f in _CLDPIX_FIELDS]
        return rasterize_points_to_grid(
            point_lats=lats,
            point_lons=lons,
            values=values,
            bbox=(bbox.lat_min, bbox.lat_max, bbox.lon_min, bbox.lon_max),
            cell_size_deg=self.OUTPUT_CELL_DEG,
            aggregations=aggregations,
        )
