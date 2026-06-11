"""CERES SSF (and FLASHFlux) footprint reader plugin.

Data source: CERES Single Scanner Footprint TOA/Surface Fluxes and Clouds (SSF)
- Example product: CER_SSF_NOAA20-FM6-VIIRS
- Format: NetCDF4, organized by **CERES footprint** (a 1-D ``Footprints``
  dimension), grouped into thematic groups (Time_and_Position, Scene_Type,
  Auxillary_Properties, …).
- Geolocation: per-footprint, in ``Time_and_Position/instrument_fov_latitude``
  and ``instrument_fov_longitude`` (longitude stored 0..360 — normalized here).
- Native footprint scale: ~20 km at nadir.

FLASHFlux
---------
The RBSP-produced FLASHFlux product shares the SSF file format, so this reader
serves both. The two differ only in latency / processing stream
(FLASHFlux → FMATCH-IMAGER-FLASH; SSF → FMATCH-IMAGER); the caller supplies the
appropriate file for the active mode. ``REQUIRED_MODE`` is therefore set to the
lower-rank ``IMAGER_FLASH`` so this reader is active for the Flash, Imager, and
Imager-camera-time products.

Why rasterize?
--------------
SSF is footprint (point) data, not a regular grid. To stay within the
``GriddedDataReader`` / ``GridTile`` contract used by every other reader, this
class bins its footprints onto a regular sub-grid covering each requested 2°
tile (see :mod:`libera_utils.footprint_matching.readers._swath`). The whole
file is parsed once and cached on the instance, then re-sliced per tile.

Variable set
------------
This is a **minimal starter set** drawn from the footprint-matching
dependencies in ``data_products.md`` (SSF supplies cloud properties and ADM /
scene types for radiometer footprints). It is expected to be refined.

Note on encoded scene/ADM codes
--------------------------------
``cloud_classification``, ``shortwave_adm_type`` and ``longwave_adm_type`` are
*encoded* CERES identifiers spanning hundreds of values (not a small category
set). They are mode-aggregated here so the dominant raw code in each cell is
preserved, but downstream consumers will likely need to decode them; an exact
decode is deferred (TODO[LIBSDC-785]).

References
----------
CERES SSF: https://ceres.larc.nasa.gov/data/#ssf-level-2
FLASHFlux: https://ceres.larc.nasa.gov/data/#fast-longwave-and-shortwave-flux-flashflux
File naming: CER_SSF_{platform}-{instrument}-{imager}_{config}_{prod}.{YYYYMMDDHH}.nc
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

# Float and integer fill sentinels used across SSF groups.
_FILL_FLOAT: float = 3.4028235e38  # float32 max, the CERES float fill
_FILL_INT16: int = 32767

# Group paths to the per-footprint geolocation variables.
_TIME_POS_GROUP: str = "Time_and_Position"
_FOV_LAT_VAR: str = "instrument_fov_latitude"
_FOV_LON_VAR: str = "instrument_fov_longitude"

# Cloudy layer to read for the layered (Footprints, LowerUpper) variables.
# 0 = lower layer. TODO[LIBSDC-785]: confirm the correct layer-selection rule
# (lower vs upper vs combined) with the science team.
_CLOUD_LAYER_INDEX: int = 0


class _SSFField(NamedTuple):
    """Mapping from an output variable to its source location and decoding rules.

    Attributes
    ----------
    out_name : str
        Output variable name exposed via ``VARIABLES``.
    group : str
        netCDF group containing the source variable.
    var : str
        Source variable name within ``group``.
    aggregation : str
        Spatial aggregation strategy (passed to the rasterizer).
    fill : float or int
        Fill sentinel for the source variable.
    valid_range : tuple of (float, float) or None
        Inclusive valid range; values outside become NaN.
    layer_index : int or None
        For 2-D ``(Footprints, LowerUpper)`` variables, the layer to select.
        ``None`` for 1-D ``(Footprints,)`` variables.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous.
    """

    out_name: str
    group: str
    var: str
    aggregation: str
    fill: float | int
    valid_range: tuple[float, float] | None
    layer_index: int | None
    n_categories: int | None


# Minimal starter field set (refine later — see module docstring).
_SSF_FIELDS: tuple[_SSFField, ...] = (
    _SSFField("aerosol_optical_depth", "Auxillary_Properties", "aerosol_optical_depth",
              "weighted_log_mean", _FILL_FLOAT, (0.0, 8.0), None, None),
    _SSFField("clear_coverage", "Clear_Footprint_Area", "clear_coverage",
              "weighted_mean", _FILL_FLOAT, (0.0, 100.0), None, None),
    _SSFField("cloud_optical_depth", "Cloudy_Imager_Footprint_Layer", "cloud_optical_depth_mean",
              "weighted_log_mean", _FILL_FLOAT, (0.0, 512.0), _CLOUD_LAYER_INDEX, None),
    _SSFField("cloud_classification", "Scene_Type", "cloud_classification",
              "weighted_mode", _FILL_INT16, (0.0, 32766.0), None, None),
    _SSFField("shortwave_adm_type", "Scene_Type", "shortwave_adm_type",
              "weighted_mode", _FILL_INT16, (0.0, 5000.0), None, None),
    _SSFField("longwave_adm_type", "Scene_Type", "longwave_adm_type",
              "weighted_mode", _FILL_INT16, (0.0, 5000.0), None, None),
)


class SSFReader(GriddedDataReader):
    """Read CERES SSF / FLASHFlux footprints and rasterize them onto the tile grid.

    Parses the per-footprint geolocation and a minimal set of cloud / aerosol /
    scene variables, then bins them onto a regular sub-grid covering each
    requested 2° tile.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"ssf"``.
    RESOLUTION_KM : float
        ~20 km (CERES footprint scale at nadir).
    OUTPUT_CELL_DEG : float
        Edge length of the rasterized output cells (degrees).
    REQUIRED_MODE : OperationalMode
        ``IMAGER_FLASH`` — active for the Flash, Imager, and Imager-camera-time
        products.
    VARIABLES : tuple[VariableSpec, ...]
        Minimal starter set (see module docstring).

    Parameters
    ----------
    file_path : Path
        Path to a CERES SSF or FLASHFlux NetCDF4 file.
    """

    READER_KEY: str = "ssf"
    RESOLUTION_KM: float = 20.0
    OUTPUT_CELL_DEG: float = 0.2
    REQUIRED_MODE: OperationalMode = OperationalMode.IMAGER_FLASH
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=f.out_name,
            dtype="float32" if f.n_categories is None else "int16",
            aggregation=f.aggregation,
            required_mode=OperationalMode.IMAGER_FLASH,
            n_categories=f.n_categories,
        )
        for f in _SSF_FIELDS
    )

    def __init__(self, file_path: Path) -> None:
        super().__init__(file_path)
        # Lazily populated point cache: (lats, lons, values (n_var, n_pts)).
        # The SSF file is parsed once and reused across all tile requests.
        self._points: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _load_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parse and cache the per-footprint coordinates and variable values.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(lats, lons, values)`` where ``lats``/``lons`` are 1-D
            ``(n_footprints,)`` (longitude normalized to −180..180) and
            ``values`` is ``(n_var, n_footprints)`` float64 with fill /
            out-of-range entries as NaN.
        """
        if self._points is not None:
            return self._points

        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(self._file_path), "r") as ds:
            # Disable netCDF4 automatic masking and rely on our own
            # apply_fill_and_valid_range (which normalizes descending
            # ``valid_range`` attributes that would otherwise make netCDF4 mask
            # every value). Scale/offset auto-application is left enabled.
            ds.set_auto_mask(False)

            tp = ds.groups[_TIME_POS_GROUP]
            lats = apply_fill_and_valid_range(
                tp.variables[_FOV_LAT_VAR][:], fill_value=_FILL_FLOAT, valid_range=(-90.0, 90.0)
            )
            lons_raw = apply_fill_and_valid_range(
                tp.variables[_FOV_LON_VAR][:], fill_value=_FILL_FLOAT, valid_range=(0.0, 360.0)
            )
            lons = normalize_longitude(lons_raw)

            value_rows: list[np.ndarray] = []
            for f in _SSF_FIELDS:
                raw = ds.groups[f.group].variables[f.var][:]
                if f.layer_index is not None:
                    # 2-D (Footprints, LowerUpper) — pick one layer.
                    raw = raw[:, f.layer_index]
                value_rows.append(
                    apply_fill_and_valid_range(raw, fill_value=f.fill, valid_range=f.valid_range)
                )

            values = np.stack(value_rows, axis=0)

        self._points = (lats, lons, values)
        return self._points

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize SSF footprints within ``bbox`` onto a regular sub-grid.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(n_var, n_lat, n_lon)`` in ``VARIABLES`` order. Cells with no
            footprints are NaN.
        """
        lats, lons, values = self._load_points()
        aggregations = [f.aggregation for f in _SSF_FIELDS]
        return rasterize_points_to_grid(
            point_lats=lats,
            point_lons=lons,
            values=values,
            bbox=(bbox.lat_min, bbox.lat_max, bbox.lon_min, bbox.lon_max),
            cell_size_deg=self.OUTPUT_CELL_DEG,
            aggregations=aggregations,
        )
