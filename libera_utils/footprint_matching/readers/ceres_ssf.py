"""CERES SSF footprint reader plugin.

Data source: the CERES Single Scanner Footprint (SSF) Level 2 TOA/surface fluxes and
clouds for NOAA-20 (FM6, VIIRS imager), distributed by NASA Langley ASDC as
``CER_SSF_NOAA20-FM6-VIIRS_...`` (``anc_ceres_noaa20_ssf``).

- Format: NetCDF4, organized by **CERES footprint** (a 1-D ``Footprints``
  dimension), grouped into thematic groups (Time_and_Position, Scene_Type,
  Cloudy_Footprint_Area, …). The groups, variables and dimensions this reader checks
  are those of CER_SSF_NOAA20-FM6-VIIRS Edition1C granules.
- Geolocation: per-footprint, in ``Time_and_Position/instrument_fov_latitude``
  and ``instrument_fov_longitude`` (longitude stored 0..360 — normalized here).
- Native footprint scale: ~20 km at nadir.

This reader feeds the products its fields name in ``modes``: FMATCH-IMAGER and
FMATCH-IMAGER-CAMTIME (see Variable set). Its registry key, ``"CERES_SSF"``, prefixes
the names of the variables it writes to those products (``CERES_SSF_<field>``).

Native footprints, matched by time
----------------------------------
SSF is footprint (point) data, not a regular grid. Each CERES footprint is served
at its own position and observation time as a
:class:`~libera_utils.footprint_matching.types.PointSamples` point, so the Libera
PSF weigher is the first place a value is weighted. A day of SSF sees a ground
location on more than one pass, so the reader serves the pass nearest the Libera
footprint time (``TIME_MATCH = NEAREST_PASS``): every footprint within
``PASS_WINDOW`` of the observation time of the footprint nearest in time, and none
beyond ``MAX_TIME_OFFSET``.
Granules are hourly; only those whose ``time_coverage_start``/``time_coverage_end``
lie within ``MAX_TIME_OFFSET`` of the current footprint time are held in memory.

Variable set
------------
Two sets, read in this order:

- **The base SSF fields** (``_SSF_BASE_FIELDS``): clear coverage, the lower-layer
  cloud optical depth and water / ice particle radii, and the scene and ADM type
  codes. Carried by every product this reader feeds: FMATCH-IMAGER and
  FMATCH-IMAGER-CAMTIME.
- **The FMATCH-IMAGER-only SSF fields** (``_SSF_IMAGER_ONLY_FIELDS``): the
  extended cloud-layer, imager-aerosol, surface-albedo and TOA-insolation fields, carried by
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
set). Each footprint's raw code is served as read, and the codes are declared
``weighted_mode``, the PSF aggregation that keeps the most common code over a
Libera footprint. Downstream consumers will likely need to decode them; an exact
decode is deferred (TODO[LIBSDC-785]).

References
----------
CERES SSF: https://ceres.larc.nasa.gov/data/#ssf-level-2
"""

from pathlib import Path
from typing import NamedTuple

import netCDF4
import numpy as np

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.readers._longitude import normalize_longitude
from libera_utils.footprint_matching.readers._points import (
    decode_cf_times,
    neighbor_spacing_km,
    read_granule_time_coverage,
    scan_to_scan_spacing_km,
)
from libera_utils.footprint_matching.readers.base import PassReader
from libera_utils.footprint_matching.types import OperationalMode, PointSamples, TimeMatch, VariableSpec
from libera_utils.io.netcdf import apply_fill_and_valid_range, field_fill_value, field_valid_range

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

# Indices on the (Footprints, Conditions) sky-condition axis, whose coordinate variable
# documents it as "1=Clear 2=Lower clouds 3=Upper clouds 4=Upper over lower (overlap) clouds".
_CONDITION_LOWER_INDEX: int = 1
_CONDITION_UPPER_INDEX: int = 2

# Indices on the (Footprints, MultiCld) multilayer-condition axis, whose coordinate variable
# documents it as "1=Single lower 2=Single upper 3=Multi lower 4=Multi upper layer(s)".
_MULTILAYER_LOWER_INDEX: int = 2
_MULTILAYER_UPPER_INDEX: int = 3

# Dimensions of the source variables, as found in the CER_SSF_NOAA20-FM6-VIIRS Edition1C
# granules. The reader checks each variable against these before indexing its second axis.
_FOOTPRINT_DIMS: tuple[str, ...] = ("Footprints",)
_LAYER_DIMS: tuple[str, ...] = ("Footprints", "LowerUpper")
_CONDITION_DIMS: tuple[str, ...] = ("Footprints", "Conditions")
_MULTILAYER_DIMS: tuple[str, ...] = ("Footprints", "MultiCld")


class _SsfField(NamedTuple):
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
        also selects a ``Conditions`` or ``MultiCld`` entry. ``None`` for 1-D
        ``(Footprints,)`` variables.
    n_categories : int or None
        Category count for categorical variables; ``None`` for continuous variables
        and for the mode-aggregated scene and ADM type codes, which have no fixed count.
    modes : frozenset[OperationalMode]
        The products that carry the emitted spec (see
        :class:`~libera_utils.footprint_matching.types.VariableSpec`).
    """

    out_name: str
    group: str
    var: str
    aggregation: str
    dimensions: tuple[str, ...]
    layer_index: int | None
    n_categories: int | None
    modes: frozenset[OperationalMode]


# The products that carry the base SSF fields: every product this reader feeds.
_SSF_BASE_MODES: frozenset[OperationalMode] = frozenset({OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME})

# Each SSF optical depth is already a linear mean over its CERES footprint, so FMATCH combines
# footprints with the arithmetic mean: cloud_optical_depth_mean ("Mean visible optical depth for
# cloud layer") is SSF-83, the PSF-weighted mean of the pixels' optical depths. CERES SSF
# Collection Guide, Release 2, p. 75,
# https://ceres.larc.nasa.gov/documents/collect_guide/pdf/SSF_CG_R2V1.pdf.

# The base SSF fields (see module docstring).
_SSF_BASE_FIELDS: tuple[_SsfField, ...] = (
    _SsfField(
        "Clear_Coverage",
        "Clear_Footprint_Area",
        "clear_coverage",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        None,
        None,
        _SSF_BASE_MODES,
    ),
    _SsfField(
        "Cloud_Optical_Depth_Lower",
        "Cloudy_Footprint_Area",
        "cloud_optical_depth_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _CLOUD_LAYER_LOWER_INDEX,
        None,
        _SSF_BASE_MODES,
    ),
    # Effective cloud particle radius separated by thermodynamic phase (μm).
    # SSF stores water and ice radii in separate 2-D (Footprints, LowerUpper)
    # variables; both read the lower cloud layer (_CLOUD_LAYER_LOWER_INDEX),
    # consistent with Cloud_Optical_Depth_Lower. The combined (blended) radius is
    # not available in SSF — use CLDPIX_Cloud_Particle_Radius for that.
    _SsfField(
        "Cloud_Water_Particle_Radius_Lower",
        "Cloudy_Footprint_Area",
        "cloud_water_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _CLOUD_LAYER_LOWER_INDEX,
        None,
        _SSF_BASE_MODES,
    ),
    _SsfField(
        "Cloud_Ice_Particle_Radius_Lower",
        "Cloudy_Footprint_Area",
        "cloud_ice_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _CLOUD_LAYER_LOWER_INDEX,
        None,
        _SSF_BASE_MODES,
    ),
    _SsfField(
        "Cloud_Classification",
        "Scene_Type",
        "cloud_classification",
        "weighted_mode",
        _FOOTPRINT_DIMS,
        None,
        None,
        _SSF_BASE_MODES,
    ),
    _SsfField(
        "Shortwave_ADM_Type",
        "Scene_Type",
        "shortwave_adm_type",
        "weighted_mode",
        _FOOTPRINT_DIMS,
        None,
        None,
        _SSF_BASE_MODES,
    ),
    _SsfField(
        "Longwave_ADM_Type",
        "Scene_Type",
        "longwave_adm_type",
        "weighted_mode",
        _FOOTPRINT_DIMS,
        None,
        None,
        _SSF_BASE_MODES,
    ),
)


# ---------------------------------------------------------------------------
# Extended SSF field set carried by FMATCH-IMAGER only.
# ---------------------------------------------------------------------------
# These SSF cloud-layer, imager-aerosol, surface-albedo and TOA-insolation fields go on FMATCH-IMAGER
# *only* -- not IMAGER-CAMTIME, which carries the base SSF fields.
#
# Multi-valued source fields are flattened into one 1-D product variable per member:
# the lower and upper entries of LowerUpper, Conditions and MultiCld -> _Lower/_Upper.
# Genuinely 1-D fields (the imager aerosol optical depths, Surface_Albedo, TOA_Incoming_Solar_Radiation)
# stay a single spec via _SCALAR_MEMBER. All are continuous float32 and so also gain a
# _Standard_Deviation companion in product_variable_specs().
_SSF_IMAGER_ONLY_MODES: frozenset[OperationalMode] = frozenset({OperationalMode.IMAGER})

# Zero-axis members for a 1-D (Footprints,) source field: one output, no index.
_SCALAR_MEMBER: tuple[tuple[str, int | None], ...] = (("", None),)
# Both cloud layers, and the upper layer alone (for fields whose lower layer is already
# carried by a base spec above).
_BOTH_LAYERS: tuple[tuple[str, int | None], ...] = (
    ("Lower", _CLOUD_LAYER_LOWER_INDEX),
    ("Upper", _CLOUD_LAYER_UPPER_INDEX),
)
_UPPER_LAYER_ONLY: tuple[tuple[str, int | None], ...] = (("Upper", _CLOUD_LAYER_UPPER_INDEX),)
# The lower-cloud and upper-cloud sky conditions; the clear and overlap conditions are not read.
_LOWER_UPPER_CONDITIONS: tuple[tuple[str, int | None], ...] = (
    ("Lower", _CONDITION_LOWER_INDEX),
    ("Upper", _CONDITION_UPPER_INDEX),
)
# The multilayer lower and upper conditions; the single-layer conditions are not read.
_LOWER_UPPER_MULTILAYER: tuple[tuple[str, int | None], ...] = (
    ("Lower", _MULTILAYER_LOWER_INDEX),
    ("Upper", _MULTILAYER_UPPER_INDEX),
)

# (out_base, group, var, aggregation, dimensions, members). ``members`` is a tuple of
# (name_suffix, second_axis_index) pairs; the flattener emits one _SsfField per member.
_IMAGER_SSF_SOURCES: tuple[tuple[str, str, str, str, tuple[str, ...], tuple[tuple[str, int | None], ...]], ...] = (
    # --- Coverage fields: the lower and upper entries of their four-entry condition axes ---
    (
        "Layer_Coverage",
        "Cloudy_Footprint_Area",
        "layers_coverages",
        "weighted_mean",
        _CONDITION_DIMS,
        _LOWER_UPPER_CONDITIONS,
    ),
    (
        "Cloud_Coverage_Multilayer",
        "Multilayer_Footprint_Area",
        "cloud_coverage_multilayer",
        "weighted_mean",
        _MULTILAYER_DIMS,
        _LOWER_UPPER_MULTILAYER,
    ),
    # --- Cloud layer fields (Cloudy_Footprint_Area), (Footprints, LowerUpper) ---
    # Three layered fields no base spec carries: both lower and upper layers.
    (
        "Cloud_Top_Pressure",
        "Cloudy_Footprint_Area",
        "cloud_top_pressure_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    (
        "Cloud_Base_Pressure",
        "Cloudy_Footprint_Area",
        "cloud_base_pressure_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    (
        "Cloud_Particle_Phase_37um",
        "Cloudy_Footprint_Area",
        "cloud_particle_phase_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _BOTH_LAYERS,
    ),
    # Three fields whose LOWER layer is already carried by the base specs above
    # (Cloud_Optical_Depth_Lower / Cloud_Water_Particle_Radius_Lower /
    # Cloud_Ice_Particle_Radius_Lower); add only the UPPER layer here to avoid a
    # redundant lower duplicate.
    (
        "Cloud_Optical_Depth",
        "Cloudy_Footprint_Area",
        "cloud_optical_depth_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _UPPER_LAYER_ONLY,
    ),
    (
        "Cloud_Water_Particle_Radius",
        "Cloudy_Footprint_Area",
        "cloud_water_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _UPPER_LAYER_ONLY,
    ),
    (
        "Cloud_Ice_Particle_Radius",
        "Cloudy_Footprint_Area",
        "cloud_ice_particle_radius_37um_mean",
        "weighted_mean",
        _LAYER_DIMS,
        _UPPER_LAYER_ONLY,
    ),
    # --- Imager aerosol fields (1-D, footprint only) ---
    (
        "Imager_Dark_Target_Land_055um_Corrected_Aerosol_Optical_Depth",
        "Imager_Land_Aerosols",
        "imager_dark_target_land_055um_corrected_aerosol_optical_depth",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
    ),
    (
        "Imager_Deep_Blue_Ocean_055um_Aerosol_Optical_Depth",
        "Imager_Ocean_Aerosols",
        "imager_deep_blue_ocean_055um_aerosol_optical_depth",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
    ),
    # --- Surface field (1-D, footprint only) ---
    # TOA_and_Surface_Fluxes/broadband_surface_albedo (a fraction, no units attribute).
    (
        "Surface_Albedo",
        "TOA_and_Surface_Fluxes",
        "broadband_surface_albedo",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
    ),
    # --- TOA flux field (1-D, footprint only) ---
    # TOA_and_Surface_Fluxes/toa_incoming_solar_radiation (file units "W/m^2").
    (
        "TOA_Incoming_Solar_Radiation",
        "TOA_and_Surface_Fluxes",
        "toa_incoming_solar_radiation",
        "weighted_mean",
        _FOOTPRINT_DIMS,
        _SCALAR_MEMBER,
    ),
)


def _build_imager_only_ssf_fields() -> tuple[_SsfField, ...]:
    """Flatten the FMATCH-IMAGER-only SSF sources into one _SsfField per output variable.

    Each source field is flattened along its second axis (or emitted as a single 1-D
    value), named with a suffix, and given ``modes`` {FMATCH-IMAGER}. All fields are
    continuous float32.

    Returns
    -------
    tuple[_SsfField, ...]
        One field per emitted output variable, in source-then-member order.
    """
    fields: list[_SsfField] = []
    for out_base, group, var, aggregation, dimensions, members in _IMAGER_SSF_SOURCES:
        for suffix, index in members:
            out_name = f"{out_base}_{suffix}" if suffix else out_base
            fields.append(
                _SsfField(
                    out_name,
                    group,
                    var,
                    aggregation,
                    dimensions,
                    index,
                    None,
                    _SSF_IMAGER_ONLY_MODES,
                )
            )
    return tuple(fields)


# The FMATCH-IMAGER-only SSF fields (see module docstring).
_SSF_IMAGER_ONLY_FIELDS: tuple[_SsfField, ...] = _build_imager_only_ssf_fields()

# Every SSF field read: the base fields, then the FMATCH-IMAGER-only fields.
# _load_granule stacks value rows in this order, matching VARIABLES.
_SSF_FIELDS: tuple[_SsfField, ...] = _SSF_BASE_FIELDS + _SSF_IMAGER_ONLY_FIELDS


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
            f"{getattr(ds, 'product_version', None)!r}). Supply a CERES SSF granule (CER_SSF_NOAA20-FM6-VIIRS_...) "
            "in the layout this reader reads, or update the reader's SSF field table for this product version."
        )
    return ds.groups[group].variables[name]


class CeresSsfReader(PassReader):
    """Read CERES SSF footprints as time-stamped native points.

    Parses the per-footprint geolocation, observation time and the base and
    FMATCH-IMAGER-only SSF fields from every staged granule.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"CERES_SSF"``, which prefixes the FMATCH variable names this reader
        writes (``CERES_SSF_<field>``).
    RESOLUTION_KM : float
        ~20 km (CERES footprint scale at nadir), the fallback when a granule's own
        footprint spacing cannot be measured.
    VARIABLES : tuple[VariableSpec, ...]
        The base and FMATCH-IMAGER-only SSF fields (see module docstring).

    Parameters
    ----------
    file_paths : Sequence[Path]
        The CERES SSF NetCDF4 granule(s) covering the processing period.
    """

    READER_KEY: str = "CERES_SSF"
    PRODUCT_ID: DataProductIdentifier = DataProductIdentifier.anc_ceres_noaa20_ssf
    # The SSF product read here is the NOAA-20 stream (FM6, VIIRS imager).
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
            modes=f.modes,
            n_categories=f.n_categories,
        )
        for f in _SSF_FIELDS
    )

    @classmethod
    def granule_span(cls, path: Path) -> tuple[np.datetime64, np.datetime64]:
        """Return the granule's ``time_coverage_start`` / ``time_coverage_end`` global attributes.

        Parameters
        ----------
        path : Path
            One staged granule.

        Returns
        -------
        tuple[np.datetime64, np.datetime64]
            ``(start, end)`` as UTC ``datetime64[ns]``.

        Raises
        ------
        ValueError
            If either attribute is missing or is not an ISO 8601 instant.
        """
        return read_granule_time_coverage(path, source="ceres_ssf")

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
            ``decode_cf_times``),
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

            source = f"ceres_ssf {path.name}"
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
                        f"{getattr(ds, 'product_version', None)!r}). Supply a CERES SSF granule "
                        "(CER_SSF_NOAA20-FM6-VIIRS_...) in the layout this reader reads, or update the reader's SSF "
                        "field table for this product version."
                    )
                # A 2-D (Footprints, <second axis>) field reads only its one entry on that axis.
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
