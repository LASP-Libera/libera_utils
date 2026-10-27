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

Native footprints, matched by time
----------------------------------
SSF is footprint (point) data, not a regular grid. Each CERES footprint is served
at its own position and observation time as a
:class:`~libera_utils.footprint_matching.types.PointSamples` point, so the Libera
PSF weigher is the first place a value is weighted. A day of SSF sees a ground
location on more than one pass, so the reader serves the pass nearest the Libera
footprint time (``TIME_MATCH = NEAREST_PASS``): every footprint within
``PASS_WINDOW`` of the smallest time offset, and none beyond ``MAX_TIME_OFFSET``.
Granules are hourly; only those whose ``time_coverage_start``/``time_coverage_end``
lie within ``MAX_TIME_OFFSET`` of the current footprint time are held in memory.

Variable set
------------
Two sets, read in this order:

- **The base SSF fields** (``_SSF_BASE_FIELDS``): aerosol optical depth, clear
  coverage, the lower-layer cloud optical depth and water / ice particle radii, and
  the scene and ADM type codes. Carried by every product SSF feeds: FMATCH-IMAGER-FLASH,
  FMATCH-IMAGER and FMATCH-IMAGER-CAMTIME.
- **The FMATCH-IMAGER-only SSF fields** (``_SSF_IMAGER_ONLY_FIELDS``): the extended
  cloud-layer, aerosol, surface-albedo and TOA-insolation fields, carried by
  FMATCH-IMAGER alone.

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

from pathlib import Path
from typing import NamedTuple

import netCDF4
import numpy as np

from libera_utils.footprint_matching.readers._longitude import normalize_longitude
from libera_utils.footprint_matching.readers._points import (
    apply_fill_and_valid_range,
    decode_cf_times,
    field_fill_value,
    field_valid_range,
    granule_time_coverage,
    neighbor_spacing_km,
    scan_to_scan_spacing_km,
)
from libera_utils.footprint_matching.readers.base import PassReader
from libera_utils.footprint_matching.types import OperationalMode, PointSamples, TimeMatch, VariableSpec

# Group paths to the per-footprint geolocation and time variables.
_TIME_POS_GROUP: str = "Time_and_Position"
_FOV_LAT_VAR: str = "instrument_fov_latitude"
_FOV_LON_VAR: str = "instrument_fov_longitude"
# Units the footprint latitude must declare: a true -90..90 latitude, not CLDPIX's colatitude.
_LATITUDE_UNITS: str = "degrees_north"
# Per-footprint observation time in the same group, decoded through its CF ``units``
# (``days since 1970-01-01 00:00:00``, gregorian), as found in the CER_SSF_NOAA20-FM6-VIIRS
# granules. A granule covers one hour.
_TIME_VAR: str = "time"
# Position of a footprint within its scan (1..660), used to pair consecutive scans and
# measure the along-track footprint spacing.
_SAMPLE_NUMBER_VAR: str = "sample_number"

# Indices of the two cloud layers on the (Footprints, LowerUpper) variables. The base
# fields read the lower layer. TODO[LIBSDC-785]: confirm the correct layer-selection
# rule (lower vs upper vs combined) with the science team.
_CLOUD_LAYER_LOWER_INDEX: int = 0
_CLOUD_LAYER_UPPER_INDEX: int = 1

# Dimensions of the source variables, as found in the CER_SSF_NOAA20-FM6-VIIRS alpha4
# granules. The reader checks each variable against these before indexing its second axis.
_FOOTPRINT_DIMS: tuple[str, ...] = ("Footprints",)
_LAYER_DIMS: tuple[str, ...] = ("Footprints", "LowerUpper")
_AERO_TYPE_DIMS: tuple[str, ...] = ("Footprints", "AeroTypePct")


class _SSFField(NamedTuple):
    """Mapping from an output variable to its source location and decoding rules.

    Each field's fill value and valid range are read from the file's own
    ``_FillValue`` and ``valid_range`` attributes.

    Attributes
    ----------
    out_name : str
        Output variable name exposed via ``VARIABLES``.
    group : str
        netCDF group containing the source variable.
    var : str
        Source variable name within ``group``.
    aggregation : str
        PSF aggregation strategy for the emitted spec.
    dimensions : tuple[str, ...]
        The source variable's dimensions; a granule whose variable has other
        dimensions is rejected.
    layer_index : int or None
        For 2-D variables, the index to select along the second of ``dimensions``.
        Despite the name it is not limited to the ``LowerUpper`` layer axis -- it
        also selects an ``AeroTypePct`` type. ``None`` for 1-D ``(Footprints,)``
        variables.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous variables
        and for the mode-aggregated scene and ADM type codes, which have no fixed count.
    required_mode : OperationalMode
        Minimum operational mode for the emitted spec (see
        :class:`~libera_utils.footprint_matching.types.VariableSpec`). Defaults to
        ``IMAGER_FLASH`` -- the latency at which the SSF/FLASHFlux reader first
        contributes.
    only_modes : tuple[OperationalMode, ...] or None
        When set, pins the emitted spec to exactly these products (bypassing the
        ``required_mode`` rank rule). Used to keep the extended cloud/aerosol/albedo
        fields in FMATCH-IMAGER only. ``None`` (default) uses the rank rule.
    """

    out_name: str
    group: str
    var: str
    aggregation: str
    dimensions: tuple[str, ...]
    layer_index: int | None
    n_categories: int | None
    required_mode: OperationalMode = OperationalMode.IMAGER_FLASH
    only_modes: tuple[OperationalMode, ...] | None = None


# The base SSF fields, carried by every product SSF feeds (see module docstring).
_SSF_BASE_FIELDS: tuple[_SSFField, ...] = (
    _SSFField(
        "aerosol_optical_depth",
        "Auxillary_Properties",
        "aerosol_optical_depth",
        "weighted_log_mean",
        _FOOTPRINT_DIMS,
        None,
        None,
    ),
    _SSFField(
        "clear_coverage",
        "Clear_Footprint_Area",
        "clear_coverage",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        None,
        None,
    ),
    _SSFField(
        "cloud_optical_depth_lower",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_optical_depth_mean",
        "weighted_log_mean",
        _LAYER_DIMS,
        _CLOUD_LAYER_LOWER_INDEX,
        None,
    ),
    # Effective cloud particle radius separated by thermodynamic phase (μm).
    # SSF stores water and ice radii in separate 2-D (Footprints, LowerUpper)
    # variables; both read the lower cloud layer (_CLOUD_LAYER_LOWER_INDEX),
    # consistent with cloud_optical_depth_lower. The combined (blended) radius is
    # not available in SSF — use cldpix.cloud_particle_radius for that.
    _SSFField(
        "cloud_water_particle_radius_lower",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_water_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _CLOUD_LAYER_LOWER_INDEX,
        None,
    ),
    _SSFField(
        "cloud_ice_particle_radius_lower",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_ice_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _CLOUD_LAYER_LOWER_INDEX,
        None,
    ),
    _SSFField(
        "cloud_classification",
        "Scene_Type",
        "cloud_classification",
        "weighted_mode",
        _FOOTPRINT_DIMS,
        None,
        None,
    ),
    _SSFField(
        "shortwave_adm_type",
        "Scene_Type",
        "shortwave_adm_type",
        "weighted_mode",
        _FOOTPRINT_DIMS,
        None,
        None,
    ),
    _SSFField("longwave_adm_type", "Scene_Type", "longwave_adm_type", "weighted_mode", _FOOTPRINT_DIMS, None, None),
)


# ---------------------------------------------------------------------------
# Extended SSF field set carried by FMATCH-IMAGER only.
# ---------------------------------------------------------------------------
# These CERES SSF cloud-layer, aerosol, and surface-albedo fields go on FMATCH-IMAGER
# *only* -- not IMAGER-FLASH or IMAGER-CAMTIME, which also activate this reader. The
# required_mode rank rule can't express that (IMAGER-CAMTIME outranks IMAGER), so each
# field is pinned with only_modes=(IMAGER,); see VariableSpec.only_modes / spec_active_in_mode.
#
# FMATCH products are strictly 1-D per footprint, so multi-valued source fields are flattened
# (like era5_pressure's per-level split): LowerUpper (2) -> _lower/_upper, AeroTypePct (7) ->
# _typeN. Genuinely 1-D fields (surface_albedo, toa_incoming_solar_radiation) stay a single
# spec via _SCALAR_MEMBER. All are continuous float32 and so also gain a _standard_deviation
# companion in product_variable_specs().
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

# (out_base, group, var, aggregation, dimensions, members). ``members`` is a tuple of
# (name_suffix, second_axis_index) pairs; the flattener emits one _SSFField per member.
_IMAGER_SSF_SOURCES: tuple[tuple[str, str, str, str, tuple[str, ...], tuple[tuple[str, int | None], ...]], ...] = (
    # --- Cloud layer fields (Cloudy_Imager_Footprint_Layer), (Footprints, LowerUpper) ---
    # Five layered fields no base spec carries: both lower and upper layers.
    (
        "layer_coverage",
        "Cloudy_Imager_Footprint_Layer",
        "layers_coverages",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    (
        "cloud_coverage_multilayer",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_coverage_multilayer",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    (
        "cloud_top_pressure",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_top_pressure_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    (
        "cloud_base_pressure",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_base_pressure_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    (
        "cloud_particle_phase_37um",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_particle_phase_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    # Three fields whose LOWER layer is already carried by the base specs above
    # (cloud_optical_depth_lower / cloud_water_particle_radius_lower /
    # cloud_ice_particle_radius_lower); add only the UPPER layer here to avoid a
    # redundant lower duplicate.
    (
        "cloud_optical_depth",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_optical_depth_mean",
        "weighted_log_mean",
        _LAYER_DIMS,
        _UPPER_LAYER_ONLY,
    ),
    (
        "cloud_water_particle_radius",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_water_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _UPPER_LAYER_ONLY,
    ),
    (
        "cloud_ice_particle_radius",
        "Cloudy_Imager_Footprint_Layer",
        "cloud_ice_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _UPPER_LAYER_ONLY,
    ),
    # --- Aerosol fields ---
    ("match_aot", "Assimilated_Aerosol_Properties", "match_aot", "weighted_log_mean", _FOOTPRINT_DIMS, _SCALAR_MEMBER),
    # aerosol_type_percentage is (Footprints, AeroTypePct=7). The seven type labels are
    # not recorded in the file; flattened to placeholder _type0.._type6 suffixes.
    (
        "aerosol_type_percentage",
        "Assimilated_Aerosol_Properties",
        "aerosol_type_percentage",
        "weighted_mean",
        _AERO_TYPE_DIMS,
        tuple((f"type{i}", i) for i in range(7)),
    ),
    (
        "imager_dark_target_land_055um_corrected_aerosol_optical_depth",
        "Imager_Land_Aerosols",
        "imager_dark_target_land_055um_corrected_aerosol_optical_depth",
        "weighted_log_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
    ),
    (
        "imager_deep_blue_ocean_055um_aerosol_optical_depth",
        "Imager_Ocean_Aerosols",
        "imager_deep_blue_ocean_055um_aerosol_optical_depth",
        "weighted_log_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
    ),
    # --- Surface field (1-D, footprint only) ---
    # Auxillary_Properties/surface_albedo is a single 1-D (Footprints,) broadband value
    # (file long_name "Surface Albedo"; a fraction — no units attr).
    ("surface_albedo", "Auxillary_Properties", "surface_albedo", "weighted_mean", _FOOTPRINT_DIMS, _SCALAR_MEMBER),
    # --- TOA flux field (1-D, footprint only) ---
    # Observed_TOA_Fluxes/toa_incoming_solar_radiation (file units "W/m^2").
    (
        "toa_incoming_solar_radiation",
        "Observed_TOA_Fluxes",
        "toa_incoming_solar_radiation",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
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
    for out_base, group, var, aggregation, dimensions, members in _IMAGER_SSF_SOURCES:
        for suffix, index in members:
            out_name = f"{out_base}_{suffix}" if suffix else out_base
            fields.append(
                _SSFField(
                    out_name,
                    group,
                    var,
                    aggregation,
                    dimensions,
                    index,
                    None,
                    OperationalMode.IMAGER,
                    _IMAGER_ONLY,
                )
            )
    return tuple(fields)


# The FMATCH-IMAGER-only SSF fields (see module docstring).
_SSF_IMAGER_ONLY_FIELDS: tuple[_SSFField, ...] = _build_imager_only_ssf_fields()

# Every SSF field read: the base fields, then the FMATCH-IMAGER-only fields.
# _load_granule stacks value rows in this order, matching VARIABLES.
_SSF_FIELDS: tuple[_SSFField, ...] = _SSF_BASE_FIELDS + _SSF_IMAGER_ONLY_FIELDS


def _ssf_variable(ds: netCDF4.Dataset, group: str, name: str, *, source: str) -> netCDF4.Variable:
    """Return ``group/name`` from an open SSF granule, rejecting a granule laid out otherwise.

    Parameters
    ----------
    ds : netCDF4.Dataset
        The open granule.
    group : str
        Group holding the variable.
    name : str
        Variable name within ``group``.
    source : str
        Name used in the error message (reader key and file).

    Returns
    -------
    netCDF4.Variable
        The variable.

    Raises
    ------
    ValueError
        If the granule has no such group, or the group has no such variable.
    """
    missing = None
    if group not in ds.groups:
        missing = f"group {group!r}"
    elif name not in ds.groups[group].variables:
        missing = f"variable {group}/{name}"
    if missing is not None:
        raise ValueError(
            f"{source}: no {missing} in this granule (product_version "
            f"{getattr(ds, 'product_version', None)!r}). Stage a CER_SSF granule in the layout this reader reads, "
            "or update the reader's SSF field table for this product version."
        )
    return ds.groups[group].variables[name]


class SSFReader(PassReader):
    """Read CERES SSF / FLASHFlux footprints as time-stamped native points.

    Parses the per-footprint geolocation, observation time and the base and
    FMATCH-IMAGER-only SSF fields from every staged granule.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"ssf"``.
    RESOLUTION_KM : float
        ~20 km (CERES footprint scale at nadir), the fallback when a granule's own
        footprint spacing cannot be measured.
    VARIABLES : tuple[VariableSpec, ...]
        The base and FMATCH-IMAGER-only SSF fields (see module docstring).

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The CERES SSF or FLASHFlux NetCDF4 granule(s) covering the processing period.
    """

    READER_KEY: str = "ssf"
    # The SSF/FLASHFlux products read here are the NOAA-20 stream (CER_SSF_NOAA20-FM6-VIIRS).
    # TODO[LIBSDC-899]: decide whether the NOAA-22 stream replaces NOAA-20 here (a cutover of this reader) or gets
    # a second reader.
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 20.0
    TIME_MATCH: TimeMatch = TimeMatch.NEAREST_PASS
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=f.out_name,
            dtype="int16" if f.aggregation == "weighted_mode" else "float32",
            aggregation=f.aggregation,
            required_mode=f.required_mode,
            n_categories=f.n_categories,
            only_modes=f.only_modes,
        )
        for f in _SSF_FIELDS
    )

    def _granule_span(self, path: Path) -> tuple[np.datetime64, np.datetime64]:
        """The granule's ``time_coverage_start`` / ``time_coverage_end`` global attributes.

        Raises
        ------
        ValueError
            If either attribute is missing or is not an ISO 8601 instant.
        """
        with netCDF4.Dataset(str(path), "r") as ds:
            return granule_time_coverage(ds, source=f"ssf {path.name}")

    def _load_granule(self, path: Path) -> PointSamples:
        """Parse one granule into a time-stamped point cloud.

        Returns
        -------
        PointSamples
            ``lats``/``lons`` are ``(n_footprints,)`` (longitude normalized to
            -180..180), ``values`` is ``(n_var, n_footprints)`` float32 with fill /
            out-of-range entries as NaN, ``times`` is ``datetime64[ns]`` per
            footprint (NaT where the file holds fill), and ``spacing_km`` the footprint
            spacing measured on this granule (the larger of along-scan and along-track).

        Raises
        ------
        ValueError
            If the granule's time variable cannot be decoded (see
            :func:`~libera_utils.footprint_matching.readers._points.decode_cf_times`),
            if its footprint latitude is not in ``degrees_north``, if a group or
            variable read is missing, if a field's dimensions differ from the ones the
            reader expects, or if a field read (geolocation and time included) has no
            ``_FillValue`` or ``valid_range`` attribute.
        """
        with netCDF4.Dataset(str(path), "r") as ds:
            # Mask with apply_fill_and_valid_range rather than netCDF4's automatic
            # masking, as the CLDPIX reader does: netCDF4 masks every value of a field
            # whose valid_range is descending, which CERES products declare.
            ds.set_auto_mask(False)

            source = f"ssf {path.name}"
            lat_var = _ssf_variable(ds, _TIME_POS_GROUP, _FOV_LAT_VAR, source=source)
            lat_units = getattr(lat_var, "units", None)
            if lat_units != _LATITUDE_UNITS:
                raise ValueError(
                    f"{source}: {_TIME_POS_GROUP}/{_FOV_LAT_VAR} has units {lat_units!r}, but this reader reads it "
                    f"as a true -90..90 latitude in {_LATITUDE_UNITS!r}. The granule's geolocation convention has "
                    "changed; update the reader before reading it."
                )
            lats = apply_fill_and_valid_range(
                lat_var[:],
                fill_value=field_fill_value(lat_var, source=source),
                valid_range=field_valid_range(lat_var, source=source),
            )
            lon_var = _ssf_variable(ds, _TIME_POS_GROUP, _FOV_LON_VAR, source=source)
            lons_raw = apply_fill_and_valid_range(
                lon_var[:],
                fill_value=field_fill_value(lon_var, source=source),
                valid_range=field_valid_range(lon_var, source=source),
            )
            lons = normalize_longitude(lons_raw)
            times = decode_cf_times(
                _ssf_variable(ds, _TIME_POS_GROUP, _TIME_VAR, source=source),
                source=f"{source} {_TIME_POS_GROUP}/{_TIME_VAR}",
            )
            sample_number = np.asarray(_ssf_variable(ds, _TIME_POS_GROUP, _SAMPLE_NUMBER_VAR, source=source)[:])

            # Each field is written straight into one float32 array, so the float64 copy
            # of only one field is alive at a time.
            values = np.empty((len(_SSF_FIELDS), lats.size), dtype=np.float32)
            for i, f in enumerate(_SSF_FIELDS):
                var = _ssf_variable(ds, f.group, f.var, source=source)
                if tuple(var.dimensions) != f.dimensions:
                    raise ValueError(
                        f"{source}: {f.group}/{f.var} (read as {f.out_name}) has dimensions {tuple(var.dimensions)}, "
                        f"but this reader expects {f.dimensions} (product_version "
                        f"{getattr(ds, 'product_version', None)!r}). Stage a CER_SSF granule in the layout this "
                        "reader reads, or update the reader's SSF field table for this product version."
                    )
                # A 2-D (Footprints, <second axis>) field reads only its one layer or type.
                raw = var[:] if f.layer_index is None else var[:, f.layer_index]
                values[i] = apply_fill_and_valid_range(
                    raw,
                    fill_value=field_fill_value(var, source=source),
                    valid_range=field_valid_range(var, source=source),
                )

        return PointSamples(
            lats=lats,
            lons=lons,
            values=values,
            times=times,
            # The granule's own footprint spacing: the larger of the along-scan spacing
            # (consecutive footprints) and the along-track one (the same scan position on
            # consecutive scans, about 44 km), so a complete swath covers a Libera footprint
            # completely in both directions.
            spacing_km=np.nanmax(
                [neighbor_spacing_km(lats, lons), scan_to_scan_spacing_km(lats, lons, times, sample_number)]
            ),
        )
