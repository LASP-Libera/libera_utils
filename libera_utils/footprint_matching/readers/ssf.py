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
appropriate file for the active mode. This reader belongs to the Flash, Imager, and
Imager-camera-time product sets, declared in ``readers/registry.py``.

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

Surface-type variables in the ``Surface_Map`` group (``surface_igbp_type``,
``surface_igbp_type_coverage``, ``snow_ice_coverage``) are intentionally NOT
extracted. The pipeline uses the dedicated ``IGBPReader`` and ``NISEReader`` as
authoritative sources for land-cover and ice/snow classification. SSF surface
type values are derived from IGBP and NISE anyway; extracting them here would
duplicate data that the dedicated readers already supply at higher resolution.

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
# 0 = lower layer, 1 = upper layer. TODO[LIBSDC-785]: confirm the correct
# layer-selection rule (lower vs upper vs combined) with the science team.
_CLOUD_LAYER_INDEX: int = 0
_CLOUD_LAYER_LOWER_INDEX: int = 0
_CLOUD_LAYER_UPPER_INDEX: int = 1


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
        For 2-D ``(Footprints, <second axis>)`` variables, the index to select
        along the second axis. Despite the name it is not limited to the
        ``LowerUpper`` layer axis -- it also selects an ``AeroTypePct`` type or an
        ``AlbedoDim`` band, since all are read as ``raw[:, layer_index]``.
        ``None`` for 1-D ``(Footprints,)`` variables.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous.
    required_mode : OperationalMode
        Minimum operational mode for the emitted spec (see
        :class:`~libera_utils.footprint_matching.types.VariableSpec`). Defaults to
        ``IMAGER_FLASH`` -- the latency at which the SSF/FLASHFlux reader first
        contributes.
    only_modes : tuple[OperationalMode, ...] or None
        When set, pins the emitted spec to exactly these products (bypassing the
        ``required_mode`` rank rule). Used to keep the extended cloud/aerosol/albedo
        fields in FMATCH-IMAGER only. ``None`` (default) uses the rank rule.
    log_floor : float or None
        Detection-limit floor for ``weighted_log_mean`` fields, passed to the
        rasterizer so valid zeros are retained rather than dropped. ``None``
        (default) uses the rasterizer's default floor; ignored for other aggregations.
    """

    out_name: str
    group: str
    var: str
    aggregation: str
    fill: float | int
    valid_range: tuple[float, float] | None
    layer_index: int | None
    n_categories: int | None
    required_mode: OperationalMode = OperationalMode.IMAGER_FLASH
    only_modes: tuple[OperationalMode, ...] | None = None
    log_floor: float | None = None


# Minimal starter field set (refine later — see module docstring).
# Note: surface-type variables in Surface_Map are deliberately omitted —
# see module docstring for the rationale.
_SSF_FIELDS: tuple[_SSFField, ...] = (
    _SSFField(
        "aerosol_optical_depth",
        "Auxillary_Properties",
        "aerosol_optical_depth",
        "weighted_log_mean",
        _FILL_FLOAT,
        (0.0, 8.0),
        None,
        None,
    ),
    _SSFField(
        "clear_coverage",
        "Clear_Footprint_Area",
        "clear_coverage",
        "weighted_mean",
        _FILL_FLOAT,
        (0.0, 100.0),
        None,
        None,
    ),
    _SSFField(
        "cloud_optical_depth",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_optical_depth_mean",
        "weighted_log_mean",
        _FILL_FLOAT,
        (0.0, 512.0),
        _CLOUD_LAYER_INDEX,
        None,
    ),
    # Effective cloud particle radius separated by thermodynamic phase (μm).
    # SSF stores water and ice radii in separate 2-D (Footprints, LowerUpper)
    # variables; both use _CLOUD_LAYER_INDEX to select the lower cloud layer,
    # consistent with cloud_optical_depth. The combined (blended) radius is
    # not available in SSF — use cldpix.cloud_particle_radius for that.
    _SSFField(
        "cloud_water_particle_radius",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_water_particle_radius_37um_mean",
        "weighted_mean",
        _FILL_FLOAT,
        (2.0, 60.0),
        _CLOUD_LAYER_INDEX,
        None,
    ),
    _SSFField(
        "cloud_ice_particle_radius",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_ice_particle_radius_37um_mean",
        "weighted_mean",
        _FILL_FLOAT,
        (5.0, 90.0),
        _CLOUD_LAYER_INDEX,
        None,
    ),
    _SSFField(
        "cloud_classification",
        "Scene_Type",
        "cloud_classification",
        "weighted_mode",
        _FILL_INT16,
        (0.0, 32766.0),
        None,
        None,
    ),
    _SSFField(
        "shortwave_adm_type",
        "Scene_Type",
        "shortwave_adm_type",
        "weighted_mode",
        _FILL_INT16,
        (0.0, 5000.0),
        None,
        None,
    ),
    _SSFField(
        "longwave_adm_type", "Scene_Type", "longwave_adm_type", "weighted_mode", _FILL_INT16, (0.0, 5000.0), None, None
    ),
)


# ---------------------------------------------------------------------------
# Extended SSF field set carried by FMATCH-IMAGER only.
# ---------------------------------------------------------------------------
# These CERES SSF cloud-layer, aerosol, and surface-albedo fields are added to the
# FMATCH-IMAGER product (the RBSP Climate-Quality radiometer product) *only* -- not
# FMATCH-IMAGER-FLASH or FMATCH-IMAGER-CAMTIME, even though those also activate this
# reader. That single-product scope cannot be expressed by the required_mode rank rule
# (IMAGER-CAMTIME outranks IMAGER), so every generated field is pinned with
# only_modes=(IMAGER,); see VariableSpec.only_modes / spec_active_in_mode.
#
# The source fields carry a second axis of varying size -- LowerUpper (2) for the cloud
# layers, AeroTypePct (7) for aerosol_type_percentage, AlbedoDim (10) for
# broadband_surface_albedo. FMATCH products are strictly 1-D per footprint, so (like
# era5_pressure's per-level flattening) each second-axis index becomes its own 1-D spec
# via a name suffix (_lower/_upper, _typeN, _bandN). Every field is continuous float32
# and therefore also gains a _standard_deviation companion in product_variable_specs().
_IMAGER_ONLY: tuple[OperationalMode, ...] = (OperationalMode.IMAGER,)

# Zero-axis members for a 1-D (Footprints,) source field: one output, no index.
_SCALAR_MEMBER: tuple[tuple[str, int | None], ...] = (("", None),)
# Both cloud layers, and the upper layer alone (for fields whose lower layer is already
# carried by a base spec above).
_BOTH_LAYERS: tuple[tuple[str, int | None], ...] = (
    ("lower", _CLOUD_LAYER_LOWER_INDEX),
    ("upper", _CLOUD_LAYER_UPPER_INDEX),
)
_UPPER_LAYER_ONLY: tuple[tuple[str, int | None], ...] = (("upper", _CLOUD_LAYER_UPPER_INDEX),)

# (out_base, group, var, aggregation, valid_range, members). ``members`` is a tuple of
# (name_suffix, second_axis_index) pairs; the flattener emits one _SSFField per member.
_IMAGER_SSF_SOURCES: tuple[
    tuple[str, str, str, str, tuple[float, float] | None, tuple[tuple[str, int | None], ...]], ...
] = (
    # --- Cloud layer fields (Cloudy_Imager_Footprint_Layer), (Footprints, LowerUpper) ---
    # Five genuinely-new layered fields: both lower and upper layers.
    (
        "layer_coverage",
        "Cloudy_Imager_Footprint_Layer",
        "layers_coverages",
        "weighted_mean",
        (0.0, 100.0),
        _BOTH_LAYERS,
    ),
    (
        "cloud_coverage_multilayer",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_coverage_multilayer",
        "weighted_mean",
        (0.0, 100.0),
        _BOTH_LAYERS,
    ),
    (
        "cloud_top_pressure",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_top_pressure_mean",
        "weighted_mean",
        (0.0, 1100.0),
        _BOTH_LAYERS,
    ),
    (
        "cloud_base_pressure",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_base_pressure_mean",
        "weighted_mean",
        (0.0, 1100.0),
        _BOTH_LAYERS,
    ),
    (
        "cloud_particle_phase_37um",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_particle_phase_37um_mean",
        "weighted_mean",
        (1.0, 2.0),
        _BOTH_LAYERS,
    ),
    # Three fields whose LOWER layer is already carried by the base specs above
    # (cloud_optical_depth / cloud_water_particle_radius / cloud_ice_particle_radius);
    # add only the UPPER layer here to avoid a redundant lower duplicate. Valid ranges
    # match the base specs so lower/upper stay consistent.
    (
        "cloud_optical_depth",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_optical_depth_mean",
        "weighted_log_mean",
        (0.0, 512.0),
        _UPPER_LAYER_ONLY,
    ),
    (
        "cloud_water_particle_radius",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_water_particle_radius_37um_mean",
        "weighted_mean",
        (2.0, 60.0),
        _UPPER_LAYER_ONLY,
    ),
    (
        "cloud_ice_particle_radius",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_ice_particle_radius_37um_mean",
        "weighted_mean",
        (5.0, 90.0),
        _UPPER_LAYER_ONLY,
    ),
    # --- Aerosol fields ---
    ("match_aot", "Assimilated_Aerosol_Properties", "match_aot", "weighted_log_mean", (0.0, 8.0), _SCALAR_MEMBER),
    # aerosol_type_percentage is (Footprints, AeroTypePct=7). The seven type labels are
    # not recorded in the file; flattened to placeholder _type0.._type6 suffixes.
    # TODO[LIBSDC-794]: replace with the real CERES aerosol-type names once confirmed.
    (
        "aerosol_type_percentage",
        "Assimilated_Aerosol_Properties",
        "aerosol_type_percentage",
        "weighted_mean",
        (0.0, 100.0),
        tuple((f"type{i}", i) for i in range(7)),
    ),
    (
        "imager_dark_target_land_055um_corrected_aerosol_optical_depth",
        "Imager_Land_Aerosols",
        "imager_dark_target_land_055um_corrected_aerosol_optical_depth",
        "weighted_log_mean",
        (0.0, 5.0),
        _SCALAR_MEMBER,
    ),
    (
        "imager_deep_blue_ocean_055um_aerosol_optical_depth",
        "Imager_Ocean_Aerosols",
        "imager_deep_blue_ocean_055um_aerosol_optical_depth",
        "weighted_log_mean",
        (0.0, 5.0),
        _SCALAR_MEMBER,
    ),
    # --- Surface field ---
    # broadband_surface_albedo is (Footprints, AlbedoDim=10) (file long_name "Imager
    # Spectral Albedo"; valid_range 0..100). The band wavelengths are not recorded in
    # the file; flattened to placeholder _band0.._band9 suffixes.
    # TODO[LIBSDC-794]: replace with the real albedo-band names once confirmed, and
    # confirm whether one broadband value (not 10 spectral bands) is actually wanted.
    (
        "broadband_surface_albedo",
        "Surface_Map",
        "broadband_surface_albedo",
        "weighted_mean",
        (0.0, 100.0),
        tuple((f"band{i}", i) for i in range(10)),
    ),
)


def _build_imager_only_ssf_fields() -> tuple[_SSFField, ...]:
    """Flatten the FMATCH-IMAGER-only SSF sources into one _SSFField per output variable.

    Each source field is flattened along its second axis (or emitted as a single 1-D
    value), named with a suffix, and pinned to FMATCH-IMAGER via ``only_modes``. All
    fields are continuous float32.

    Returns
    -------
    tuple[_SSFField, ...]
        One field per emitted output variable, in source-then-member order.
    """
    fields: list[_SSFField] = []
    for out_base, group, var, aggregation, valid_range, members in _IMAGER_SSF_SOURCES:
        for suffix, index in members:
            out_name = f"{out_base}_{suffix}" if suffix else out_base
            fields.append(
                _SSFField(
                    out_name,
                    group,
                    var,
                    aggregation,
                    _FILL_FLOAT,
                    valid_range,
                    index,
                    None,
                    OperationalMode.IMAGER,
                    _IMAGER_ONLY,
                )
            )
    return tuple(fields)


# The full SSF field set: the base FLASH+ fields followed by the FMATCH-IMAGER-only
# extension. _load_points stacks value rows in this order, matching VARIABLES.
_SSF_FIELDS = _SSF_FIELDS + _build_imager_only_ssf_fields()


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
    VARIABLES : tuple[VariableSpec, ...]
        Minimal starter set (see module docstring).

    Parameters
    ----------
    file_path : Path
        Path to a CERES SSF or FLASHFlux NetCDF4 file.
    """

    READER_KEY: str = "ssf"
    # The SSF/FLASHFlux products read here are the NOAA-20 stream (CER_SSF_NOAA20-FM6-VIIRS).
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 20.0
    OUTPUT_CELL_DEG: float = 0.2
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=f.out_name,
            dtype="float32" if f.n_categories is None else "int16",
            aggregation=f.aggregation,
            required_mode=f.required_mode,
            n_categories=f.n_categories,
            only_modes=f.only_modes,
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
                value_rows.append(apply_fill_and_valid_range(raw, fill_value=f.fill, valid_range=f.valid_range))

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
        log_floors = [f.log_floor for f in _SSF_FIELDS]
        return rasterize_points_to_grid(
            point_lats=lats,
            point_lons=lons,
            values=values,
            bbox=(bbox.lat_min, bbox.lat_max, bbox.lon_min, bbox.lon_max),
            cell_size_deg=self.OUTPUT_CELL_DEG,
            aggregations=aggregations,
            log_floors=log_floors,
        )
