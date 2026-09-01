"""FMATCH data product assembly and writing.

This module is the seam between the footprint-matching *engine* (readers, PSF
aggregation, geometry) and the Libera *data-product* machinery
(``LiberaDataProductDefinition`` / ``write_libera_data_product``). It owns the
FMATCH product definitions and the flow that turns matched footprints into a
conformant NetCDF file, for all five operational modes.

What the product carries
------------------------
Assembly and writing are wired for **every** operational mode, and each declared
variable is filled by one of three routes:

* **Input pass-through** - the geolocation and viewing angles come straight from
  the L1B Daily inputs (:data:`_RADIOMETER_L1B_VARIABLES` for the radiometer
  modes, :data:`_CAMTIME_SEGMENTATION_VARIABLES` for the camera-segmented ones).
* **Computed** - the derived viewing geometry (``sunglint_angle``, via
  :func:`compute_derived_viewing_geometry`) is always calculated, and the external
  aggregated variables (:func:`aggregate_external_variables`) plus the coverage/QA
  columns (``psf_coverage_fraction`` / ``q_flags``) are calculated whenever staged
  ancillary inputs are supplied - the PSF aggregation over
  tiling/weighting/geometry described in design doc section 2.8.1.
* **Placeholders** - only what remains (e.g. the external columns when no ancillary
  data was staged, or reference data such as FlashFlux that is still future work).

A placeholder is *structurally* correct (declared dtype, shape and attributes)
but numerically meaningless. Floating-point placeholders are filled with ``NaN``
and integer placeholders with ``0``; see :func:`_fill_placeholder_variables`.

Why a thin seam here
--------------------
The product definitions (``libera_utils/data/product_definitions/fmatch_*.yml``)
are the contract every downstream consumer (Scene ID, Camera Cloud Fraction, ADM
binning) reads against. Keeping the loaders next to the writers means there is a
single place that knows how a FMATCH file is produced, while the reader plugins
stay decoupled from product I/O.

See Also
--------
libera_utils.footprint_matching._runner : Manifest-driven runners that call into this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.config import config
from libera_utils.footprint_matching.aggregation import (
    ACCEPT_COVERAGE_THRESHOLD,
    PARTIAL_COVERAGE_THRESHOLD,
    aggregate_tile_variables,
)
from libera_utils.footprint_matching.geometry import (
    L1B_FILL_VALUE,
    NOMINAL_ALTITUDE_KM,
    OffLimbError,
    bounding_box_from_boresight,
    compute_footprint_bounding_box,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.tiling import (
    DEFAULT_MAX_CACHE_BYTES,
    TileManager,
    build_tile_manager,
)
from libera_utils.footprint_matching.types import (
    FmatchCoverageFlag,
    OperationalMode,
    RadiometerFootprint,
    spec_active_in_mode,
)
from libera_utils.footprint_matching.weighting import (
    AngularPSFWeigher,
    PixelWeigher,
    RadialWeigher,
    WeightField,
)
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    from collections.abc import Sequence

    from xarray import Dataset

    from libera_utils.footprint_matching.camera_segmentation import PseudoFootprint
    from libera_utils.footprint_matching.types import GridTile
    from libera_utils.io.filenaming import LiberaDataProductFilename

logger = logging.getLogger(__name__)

# Product-definition variable names that the camera-segmentation tool fills with
# *real* per-footprint values (centre-pixel geolocation/geometry, the corner-derived
# PSF bounding box, and the QA flags). Every other declared variable is filled by the
# derived-geometry engine (:func:`compute_derived_viewing_geometry`, always) or the
# external aggregation engine (:func:`aggregate_external_variables`, when ancillary
# inputs are staged); anything still unstaged or genuinely future (e.g. FlashFlux)
# falls back to a conformant placeholder. Kept as one set so the assembly and its
# tests agree on exactly which variables come straight off the segmentation.
_CAMTIME_SEGMENTATION_VARIABLES: frozenset[str] = frozenset(
    {
        "latitude",
        "longitude",
        "altitude",
        "solar_zenith_angle",
        "viewing_zenith_angle",
        "relative_azimuth_angle",
        "psf_bbox_lat_min",
        "psf_bbox_lat_max",
        "psf_bbox_lon_min",
        "psf_bbox_lon_max",
        "q_flags",
        # Boresight (centre) pixel provenance: which L1B camera pixel this pseudo-footprint's
        # boresight stand-in falls on. Real per-footprint integers straight off the
        # PseudoFootprint. The block's inclusive (min, max) pixel extent is emitted separately
        # as the 2-D camera_pixel_x/y range COORDINATES, which this set (real data *variables*)
        # deliberately does not list.
        "center_pixel_x",
        "center_pixel_y",
    }
)

# Mapping of FMATCH product variable name -> L1B RAD-4CH variable name, for the
# per-footprint quantities FMATCH copies straight out of L1B rather than computing.
#
# Viewing angles: the FMATCH solar/viewing zenith and relative azimuth map to the
# L1B "_Surface" angles (geodetic angles at the Earth point), whose units (degrees)
# and ranges line up with the FMATCH definition - in particular L1B
# ``Relative_Azimuth_Surface`` spans [0, 360], matching relative_azimuth_angle's
# valid_range. Note we do NOT pass through the derived ``sunglint_angle``; that is a
# computed product variable (see compute_derived_viewing_geometry), not an L1B input.
#
# Every variable listed here is declared float32 in the product definition, which is
# why the reader (``_runner.load_l1b_radiometer_inputs``) can cast them all to float32
# generically. The L1B time coordinate is handled separately because it maps to the
# product's time *coordinate* (RADIOMETER_TIME), not a data variable.
#
# This lives here, rather than in the runner that reads it, so the assembly path and
# the runner share one source of truth without the runner (which imports this module)
# creating a circular import.
L1B_PASSTHROUGH_VARIABLES: dict[str, str] = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "solar_zenith_angle": "Solar_Zenith_Surface",
    "viewing_zenith_angle": "Viewing_Zenith_Surface",
    "relative_azimuth_angle": "Relative_Azimuth_Surface",
}

# L1B fields (beyond the pass-through product columns) that FMATCH reads only to build
# each footprint's viewing geometry - the true ray-traced bounding box
# (:func:`~libera_utils.footprint_matching.geometry.compute_footprint_bounding_box`)
# and the CERES-faithful angular PSF weigher. These are NOT written to the product;
# they feed :func:`build_radiometer_footprints`.
#
# The subsatellite point is required to orient the scan plane and locate the satellite,
# so it is AND-ed into the finite-footprint mask alongside the pass-through variables
# (a footprint with no valid subsatellite geolocation cannot get a real box). It is read
# at float64 for geometry precision rather than the float32 the product columns use.
#
# Like L1B_PASSTHROUGH_VARIABLES this lives here (not in the runner that reads it) so the
# assembly path and the runner share one source of truth without a circular import.
L1B_SCAN_REFERENCE_VARIABLES: dict[str, str] = {
    "subsatellite_latitude": "Subsatellite_Latitude",
    "subsatellite_longitude": "Subsatellite_Longitude",
}

# The instrument cone-angle rate (degrees/second) sets the along-scan PSF orientation
# and flags the stationary-scanner case. Unlike the subsatellite point it is optional:
# a fill/NaN value simply means "unknown scan rate" and is normalized to None per
# footprint downstream, so it is read but deliberately kept out of the finite mask (it
# must not drop an otherwise-good Earth-viewing footprint).
L1B_CONE_ANGLE_RATE_VARIABLE: str = "Cone_Angle_Rate"
FMATCH_CONE_ANGLE_RATE_KEY: str = "cone_angle_rate"

# The radiometer-timescale counterpart of _CAMTIME_SEGMENTATION_VARIABLES: the
# product-definition variables filled with *real* values passed straight through
# from the L1B Daily radiometer product, rather than computed by footprint
# matching. These are the "(a) Geolocation inputs (from L1B Daily)" block of each
# radiometer-timed fmatch_*.yml, and they are exactly the keys (other than the
# RADIOMETER_TIME coordinate) that
# ``_runner.load_l1b_radiometer_inputs`` returns. Kept as one set so the
# assembly and its tests agree on which variables are "real" this milestone.
_RADIOMETER_L1B_VARIABLES: frozenset[str] = frozenset(L1B_PASSTHROUGH_VARIABLES)

# Product definition YAML filename for each FMATCH operational mode. Every mode
# has its own SSF-style product definition (the mode *is* the product), and the
# active reader set / variables differ by mode. Kept as one source of truth so
# callers and tests never hard-code filenames.
FMATCH_DEFINITION_FILENAMES: dict[OperationalMode, str] = {
    OperationalMode.CAM: "fmatch_cam.yml",
    OperationalMode.CAM_CAMTIME: "fmatch_cam_camtime.yml",
    OperationalMode.IMAGER_FLASH: "fmatch_imager_flash.yml",
    OperationalMode.IMAGER: "fmatch_imager.yml",
    OperationalMode.IMAGER_CAMTIME: "fmatch_imager_camtime.yml",
}

# Camera-timescale modes index footprints by camera image time; all other modes
# index by radiometer observation time. This is the dimension/coordinate name and
# the ``time_variable`` handed to ``write_libera_data_product`` for filename
# start/end-time generation.
_CAMERA_TIMESCALE_MODES = frozenset({OperationalMode.CAM_CAMTIME, OperationalMode.IMAGER_CAMTIME})

# Back-compat aliases for the CAM product (the first one delivered).
FMATCH_CAM_DEFINITION_FILENAME = FMATCH_DEFINITION_FILENAMES[OperationalMode.CAM]
FMATCH_CAM_TIME_VARIABLE = "RADIOMETER_TIME"


def fmatch_time_variable(mode: OperationalMode) -> str:
    """Return the per-footprint time coordinate name for an operational mode.

    Camera-timescale modes (``CAM_CAMTIME``, ``IMAGER_CAMTIME``) use
    ``CAMERA_TIME``; all radiometer-timescale modes use ``RADIOMETER_TIME``.
    """
    return "CAMERA_TIME" if is_camera_timescale_mode(mode) else "RADIOMETER_TIME"


def is_camera_timescale_mode(mode: OperationalMode) -> bool:
    """Return whether a mode indexes its footprints by camera image time.

    The timescale determines what a mode is built *from*: camera-timescale modes are
    assembled from camera pseudo-footprints (segmented from the L1B camera grid),
    while radiometer-timescale modes are assembled from L1B radiometer pass-through
    inputs. Runners and assembly both branch on this.

    Parameters
    ----------
    mode : OperationalMode
        The operational mode to test.

    Returns
    -------
    bool
        True for ``CAM_CAMTIME`` and ``IMAGER_CAMTIME``; False otherwise.
    """
    return mode in _CAMERA_TIMESCALE_MODES


def load_fmatch_definition(mode: OperationalMode) -> LiberaDataProductDefinition:
    """Load and validate the FMATCH product definition for an operational mode.

    Resolves the mode's YAML under the configured product-definitions directory
    and parses it into a validated :class:`LiberaDataProductDefinition`.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode whose product definition to load.

    Returns
    -------
    LiberaDataProductDefinition
        The validated product definition, ready for use with
        ``create_product_dataset`` / ``enforce_dataset_conformance`` /
        ``check_dataset_conformance``.

    Notes
    -----
    The directory is read from ``config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")``
    so packaging/test overrides are honored, matching how L1A product
    definitions are resolved elsewhere in the codebase.
    """
    filename = FMATCH_DEFINITION_FILENAMES[mode]
    definitions_dir = Path(str(config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")))
    return LiberaDataProductDefinition.from_yaml(definitions_dir / filename)


def load_fmatch_cam_definition() -> LiberaDataProductDefinition:
    """Load and validate the FMATCH-CAM product definition.

    Thin convenience wrapper around :func:`load_fmatch_definition` for the
    lowest-latency CAM product (the first one delivered).

    Returns
    -------
    LiberaDataProductDefinition
        The validated FMATCH-CAM product definition.
    """
    return load_fmatch_definition(OperationalMode.CAM)


@dataclass
class _FootprintGeometry:
    """Viewing geometry a PSF weigher needs, extracted from a footprint object."""

    boresight_lat_deg: float
    boresight_lon_deg: float
    altitude_km: float
    viewing_zenith_deg: float
    subsatellite_lat_deg: float | None
    subsatellite_lon_deg: float | None
    cone_angle_rate: float | None


def _footprint_geometry(footprint: Any) -> _FootprintGeometry:
    """Extract the viewing geometry a PSF weigher needs from a footprint object.

    Different footprint objects expose geometry differently (a camera
    ``PseudoFootprint`` has ``latitude``/``longitude``/``viewing_zenith_angle``; a bare
    footprint may only carry a ``bbox``), so we read named attributes when present and
    fall back sensibly otherwise. The
    ``subsatellite_*`` and ``cone_angle_rate`` fields are only used by the
    angular-frame weigher; they are ``None`` when the footprint does not carry them
    (the radial stand-in ignores them, and the angular weigher degrades gracefully).

    The L1B ``Cone_Angle_Rate`` fill value (``-999``) is normalized to ``None`` so a
    fill never masquerades as a real (near-zero) stationary-scanner rate.
    """
    bbox = footprint.bbox
    lat = getattr(footprint, "latitude", None)
    lon = getattr(footprint, "longitude", None)
    if lat is None or lon is None:
        # Box centre. (A dateline-wrapping box reports lon_max > 180; the mean still
        # lands inside it, which is good enough for the weight kernel.)
        lat = 0.5 * (bbox.lat_min + bbox.lat_max)
        lon = 0.5 * (bbox.lon_min + bbox.lon_max)
    # Spacecraft altitude in km for the PSF ground-radius scaling. Read only the
    # dedicated ``spacecraft_altitude_km`` field so we never mistake another quantity
    # for it -- in particular the camera ``PseudoFootprint`` carries a metres-valued
    # ``altitude`` that is the center-pixel *surface height* (an output column), not a
    # spacecraft altitude; using it here would inflate the PSF radius ~1000x and make
    # near-zero terrain heights collapse to the fallback. Camera footprints expose no
    # spacecraft altitude, so they correctly fall back to the nominal orbit altitude.
    altitude = getattr(footprint, "spacecraft_altitude_km", None)
    if altitude is None or not (altitude > 0.0):
        altitude = NOMINAL_ALTITUDE_KM
    viewing_zenith = getattr(footprint, "viewing_zenith_angle", 0.0) or 0.0

    sub_lat = getattr(footprint, "subsatellite_latitude", None)
    sub_lon = getattr(footprint, "subsatellite_longitude", None)

    # Scan direction: prefer the L1B Cone_Angle_Rate; accept an ``alpha_dot`` alias.
    cone_rate = getattr(footprint, "cone_angle_rate", None)
    if cone_rate is None:
        cone_rate = getattr(footprint, "alpha_dot", None)
    if cone_rate is not None and (cone_rate == L1B_FILL_VALUE or not np.isfinite(cone_rate)):
        cone_rate = None

    return _FootprintGeometry(
        boresight_lat_deg=float(lat),
        boresight_lon_deg=float(lon),
        altitude_km=float(altitude),
        viewing_zenith_deg=float(viewing_zenith),
        subsatellite_lat_deg=None if sub_lat is None else float(sub_lat),
        subsatellite_lon_deg=None if sub_lon is None else float(sub_lon),
        cone_angle_rate=None if cone_rate is None else float(cone_rate),
    )


def aggregate_external_variables(
    mode: OperationalMode,
    footprints: Sequence[Any],
    tile_manager: TileManager | None = None,
    *,
    source_file_paths: dict[str, Path] | None = None,
    max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
    weigher: PixelWeigher | None = None,
    return_coverage: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate every active reader's gridded data to one value per footprint.

    For the given operational mode this selects the active readers via
    ``ReaderRegistry.get_readers_for_mode(mode)``, and for each footprint loads the
    merged tile overlapping it **through the caching**
    :class:`~libera_utils.footprint_matching.tiling.TileManager`, weights each grid
    cell with the PSF ``weigher``, and applies each variable's aggregation strategy
    (weighted mean / mode / log-mean / std / ranked scene) to collapse the
    fine-resolution pixels to a single value per footprint. The active reader set -
    and therefore the keys of the returned dict - grows with the mode's latency
    (e.g. CAM has era5, igbp, nise, viirs_brdf, viirs_cloud; IMAGER additionally has
    era5_pressure, viirs_aod, and the RBSP ssf and cldpix fields).
    Per-spec gating also applies (see below).

    Every output variable is named ``<source_key>_<spec_name>`` (e.g.
    ``era5_wind_u10``, ``igbp_surface_type``, ``cldpix_cloud_mask``), matching the
    product definition variable names. The reader's ``INSTRUMENT`` is recorded in
    each variable's ``long_name`` rather than in the name. Per-spec gating uses
    :func:`~libera_utils.footprint_matching.types.spec_active_in_mode`: a spec's
    ``required_mode`` rank rule, or an exact ``only_modes`` pin (e.g. the extended
    SSF cloud/aerosol/albedo fields are aggregated for FMATCH-IMAGER only).

    Processing is done **one footprint at a time** (rather than gathering every
    footprint's tiles up front) so only the current footprint's merged tiles are
    held in memory -- the LRU cache still serves the shared tiles adjacent
    footprints reuse, keeping the working set bounded per the design doc's memory
    model.

    The PSF weigher is swappable via ``weigher``. This function keeps the boresight
    **radial stand-in** (:class:`~libera_utils.footprint_matching.weighting.RadialWeigher`)
    as its own generic default, but the radiometer-timescale assembly path defaults to
    the scientifically faithful angular-frame PSF
    (:class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher`) now that the
    L1B scan reference (subsatellite point + cone-angle rate) is read (see
    :func:`_assemble_radiometer_dataset`). The camera path keeps the radial stand-in
    until camera segmentation carries a scan reference.

    Coverage-based *acceptance* (the CERES 75%/95% discard/flag rule) is not applied
    here -- this function computes each footprint's coverage and, when
    ``return_coverage`` is set, returns it alongside the values, but turning coverage
    into QA flags (or discarding a footprint) is the caller's job (see
    :func:`_merge_coverage_qa`). Per-footprint coverage is the fraction of the PSF's
    total in-contour weight backed by usable data, taken as the **minimum** across the
    active sources so a footprint is scored no better than its worst-covered input.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode selecting the active readers and product variables.
    footprints : Sequence[Any]
        Footprints to aggregate over, each exposing a ``bbox`` BoundingBox (and,
        when available, ``latitude``/``longitude``/``altitude``/
        ``viewing_zenith_angle`` for the weigher). For cache locality the caller
        should pass them sorted along-track (the orchestrator's responsibility).
    tile_manager : TileManager, optional
        A pre-built manager to fetch tiles from. If ``None``, one is constructed from
        ``source_file_paths`` via :func:`build_tile_manager`.
    source_file_paths : dict[str, Path], optional
        Reader-key → file-path map used to build a TileManager when one is not
        supplied. Required in that case.
    max_cache_bytes : int, optional
        Byte budget for the constructed cache. Defaults to
        :data:`~libera_utils.footprint_matching.tiling.DEFAULT_MAX_CACHE_BYTES`.
    weigher : PixelWeigher, optional
        The per-cell PSF weight provider. Defaults to a fresh
        :class:`~libera_utils.footprint_matching.weighting.RadialWeigher`.
    return_coverage : bool, optional
        When ``True``, also return a per-footprint coverage array. Defaults to
        ``False`` (return only the values dict).

    Returns
    -------
    dict[str, np.ndarray] or tuple[dict[str, np.ndarray], np.ndarray]
        With ``return_coverage=False`` (default): the mapping of aggregated-variable
        name to a 1-D array indexed by footprint. Float variables carry ``NaN`` where
        a footprint had no usable data; categorical (integer) variables carry ``0``
        there. With ``return_coverage=True``: a ``(values, coverage)`` tuple, where
        ``coverage`` is a 1-D float array in ``[0, 1]`` indexed by footprint (the
        minimum source coverage; ``0`` where the footprint had no covered weight).

    Raises
    ------
    ValueError
        If neither ``tile_manager`` nor ``source_file_paths`` is supplied.
    """
    if tile_manager is None:
        if source_file_paths is None:
            raise ValueError("aggregate_external_variables requires either a tile_manager or source_file_paths.")
        tile_manager = build_tile_manager(mode, source_file_paths, max_cache_bytes=max_cache_bytes)

    if weigher is None:
        weigher = RadialWeigher()

    footprints = list(footprints)
    n_footprints = len(footprints)

    # Drive the loop off the TileManager's *active sources* rather than re-deriving
    # the registry set for the mode: a manager is constructed with only the readers
    # active for its mode, and this also lets a caller pass a manager holding a
    # subset of sources. Each source's reader *class* (needed for
    # product_variable_specs() and the INSTRUMENT token) is resolved from the
    # registry by key.
    reader_classes = {key: ReaderRegistry.get(key) for key in tile_manager.sources}

    # Pre-allocate one output array per product variable, filled with the "no data"
    # sentinel (NaN for floats, 0 for integers, matching the placeholder convention
    # used elsewhere in this module). Cache the per-reader spec dtype/name plan so we
    # do not recompute it for every footprint.
    outputs: dict[str, np.ndarray] = {}
    plan: dict[str, list[tuple[str, str, np.dtype]]] = {}
    for key, reader_cls in reader_classes.items():
        entries: list[tuple[str, str, np.dtype]] = []
        for spec in reader_cls.product_variable_specs():
            # Per-spec mode gating: a reader may declare variables that only belong to
            # higher-latency modes (e.g. the extended SSF cloud/aerosol/albedo fields
            # are FMATCH-IMAGER only), so skip specs not active in this mode.
            if not spec_active_in_mode(spec, mode):
                continue
            # Product variable name is ``<source_key>_<spec_name>`` (no instrument
            # token); the reader's INSTRUMENT is carried in the variable's long_name
            # by the product definition, per the FMATCH naming contract.
            product_name = f"{key}_{spec.name}"
            dtype = np.dtype(spec.dtype)
            fill = np.nan if np.issubdtype(dtype, np.floating) else 0
            outputs[product_name] = np.full(n_footprints, fill, dtype=dtype)
            entries.append((product_name, spec.name, dtype))
        plan[key] = entries

    # Per-footprint coverage accumulator: track the minimum across sources. Start at
    # +inf so the first source sets it; a footprint with no covered weight ends at 0.
    coverage_min = np.full(n_footprints, np.inf, dtype=float)

    for i, footprint in enumerate(footprints):
        if getattr(footprint, "off_limb", False):
            # Space/calibration view with no Earth footprint: leave every external
            # variable at its fill sentinel and force zero coverage (coverage_min[i]
            # stays +inf, mapped to 0.0 below). Never aggregate ancillary data against
            # the boresight placeholder box -- that would hand a space record valid-
            # looking external values and nonzero coverage.
            continue
        geom = _footprint_geometry(footprint)
        for key, reader_cls in reader_classes.items():
            # Merged, cached tile covering this footprint's PSF bounding box.
            tile = tile_manager.get_data(key, footprint.bbox)
            weight_field = weigher.weight_field(
                tile,
                geom.boresight_lat_deg,
                geom.boresight_lon_deg,
                altitude_km=geom.altitude_km,
                viewing_zenith_deg=geom.viewing_zenith_deg,
                subsatellite_lat_deg=geom.subsatellite_lat_deg,
                subsatellite_lon_deg=geom.subsatellite_lon_deg,
                cone_angle_rate=geom.cone_angle_rate,
            )
            if return_coverage:
                coverage_min[i] = min(coverage_min[i], _tile_coverage(tile, weight_field))
            # One scalar per product variable, keyed by the bare spec name.
            scalars = aggregate_tile_variables(reader_cls, tile, weight_field)
            for product_name, spec_name, dtype in plan[key]:
                value = scalars[spec_name]
                if np.issubdtype(dtype, np.integer):
                    # Integer (categorical) variables cannot hold NaN; leave the 0
                    # fill in place when the footprint had no usable categorical data.
                    if np.isfinite(value):
                        outputs[product_name][i] = value
                else:
                    outputs[product_name][i] = value

    if not return_coverage:
        return outputs
    # +inf survives only for footprints that saw no source at all (an empty manager);
    # score those as zero coverage.
    coverage = np.where(np.isinf(coverage_min), 0.0, coverage_min)
    return outputs, coverage


def _tile_coverage(tile: GridTile, weight_field: WeightField) -> float:
    """Fraction of a footprint's PSF weight backed by usable data in one source's tile.

    The coverage numerator is the PSF weight summed over cells whose data is finite
    (a cell is "covered" when the reader supplied a value there rather than a NaN /
    uncovered gap); the denominator is the weigher's total in-contour energy. An empty
    tile (failed / missing region) has zero total energy and returns ``0.0``, exactly
    the CERES partial-coverage signal. For a multi-variable tile a cell counts as
    covered when *any* of its variable planes is finite.

    Parameters
    ----------
    tile : GridTile
        The merged tile for this footprint/source.
    weight_field : WeightField
        Per-cell PSF weights aligned to ``tile.lats`` x ``tile.lons``.

    Returns
    -------
    float
        Coverage fraction in ``[0, 1]``.
    """
    total = weight_field.total_energy
    if total <= 0.0:
        return 0.0
    data = np.asarray(tile.data)
    valid = np.any(np.isfinite(data), axis=0) if data.ndim == 3 else np.isfinite(data)
    weights = np.asarray(weight_field.weights, dtype=float)
    if valid.shape != weights.shape:
        # Misaligned / empty region: no data placed against the weight grid.
        return 0.0
    return float(np.sum(weights[valid])) / total


def compute_derived_viewing_geometry(
    solar_zenith_angle: np.ndarray,
    viewing_zenith_angle: np.ndarray,
    relative_azimuth_angle: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute derived viewing-geometry variables from the geolocation angles.

    Produces the ``sunglint_angle`` variable present in every FMATCH product
    definition. The intended (CERES/SSF-heritage) formula, with all angles in
    degrees:

    - Sun glint angle: the angle between the sensor view direction and the
      specular reflection of the solar beam; small values indicate potential
      sun glint contamination.

    Parameters
    ----------
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : np.ndarray
        Per-footprint geolocation angles in degrees.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"sunglint_angle": ...}``, a 1-D array of glint angles in degrees (``[0, 180]``)
        indexed by footprint. ``NaN`` propagates wherever any input angle is ``NaN``.

    Notes
    -----
    The sun-glint angle is the angle between the sensor view direction and the
    specular (mirror) reflection of the solar beam off a horizontal surface. Writing
    the view and specular directions as unit vectors in the local horizontal frame and
    taking their dot product gives, with all angles in degrees,

    ``cos(glint) = cos(SZA)*cos(VZA) - sin(SZA)*sin(VZA)*cos(RAA)``.

    This is exact for the convention in which the relative azimuth ``RAA`` is the
    direct difference between the view and solar azimuths, so glint falls to zero at
    the specular geometry ``VZA == SZA``, ``RAA == 180`` (the sensor looking into the
    anti-solar azimuth, where the reflected beam goes). ``TODO[LIBSDC-785]``: confirm
    the L1B ``Relative_Azimuth_Surface`` azimuth convention matches this sign; if it is
    measured from the anti-solar direction instead, flip the sign of the third term.
    """
    sza = np.radians(np.asarray(solar_zenith_angle, dtype=float))
    vza = np.radians(np.asarray(viewing_zenith_angle, dtype=float))
    raa = np.radians(np.asarray(relative_azimuth_angle, dtype=float))
    cos_glint = np.cos(sza) * np.cos(vza) - np.sin(sza) * np.sin(vza) * np.cos(raa)
    # Clip absorbs floating-point excursions just outside [-1, 1]; NaN inputs stay NaN.
    cos_glint = np.clip(cos_glint, -1.0, 1.0)
    return {"sunglint_angle": np.degrees(np.arccos(cos_glint))}


def build_radiometer_footprints(l1b_inputs: dict[str, np.ndarray]) -> list[RadiometerFootprint]:
    """Build per-footprint :class:`RadiometerFootprint` objects from L1B pass-through arrays.

    Gives each radiometer-timescale footprint the geographic bounding box the external
    aggregation path needs.

    When the L1B scan reference is present (``subsatellite_latitude`` /
    ``subsatellite_longitude`` from
    :func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`), the
    box is the **true ray-traced footprint** from
    :func:`~libera_utils.footprint_matching.geometry.compute_footprint_bounding_box`
    (boresight + subsatellite point + viewing zenith projected onto the WGS84
    ellipsoid, limb-truncated via ``on_limb="flag"``), and the subsatellite point and
    ``cone_angle_rate`` are carried onto the footprint so the CERES-faithful
    ``AngularPSFWeigher`` can orient the PSF along the real scan plane.

    When the scan reference is absent (a minimal caller-built dict), the box degrades
    to the boresight-centred approximation from
    :func:`~libera_utils.footprint_matching.geometry.bounding_box_from_boresight`,
    which needs only lat/lon + viewing zenith, and the scan-frame fields stay ``None``.

    Either way exactly one footprint is produced per input record, in input order, so
    the list stays index-aligned with the product columns: a record whose ray-trace
    fails the limb check (a space/calibration view that slipped past the reader's finite
    filter) falls back to the boresight box rather than being dropped. The altitude is
    the nominal orbit altitude because the L1B pass-through does not carry a
    per-footprint satellite altitude; the ray-traced box recovers the slant range from
    the boresight/subsatellite geometry itself.

    Parameters
    ----------
    l1b_inputs : dict[str, np.ndarray]
        The arrays from
        :func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`
        (already filtered to finite geolocation). ``latitude``, ``longitude`` and
        ``viewing_zenith_angle`` are always read; ``subsatellite_latitude`` /
        ``subsatellite_longitude`` / ``cone_angle_rate`` are read when present.

    Returns
    -------
    list[RadiometerFootprint]
        One footprint per L1B record, in input order.
    """
    latitudes = np.asarray(l1b_inputs["latitude"], dtype=float)
    longitudes = np.asarray(l1b_inputs["longitude"], dtype=float)
    viewing_zeniths = np.asarray(l1b_inputs["viewing_zenith_angle"], dtype=float)

    # The scan reference is optional. When it is missing we keep the boresight-box
    # behaviour so minimal caller-built dicts (and the existing tests) still work.
    have_scan_ref = "subsatellite_latitude" in l1b_inputs and "subsatellite_longitude" in l1b_inputs
    if have_scan_ref:
        subsatellite_lats = np.asarray(l1b_inputs["subsatellite_latitude"], dtype=float)
        subsatellite_lons = np.asarray(l1b_inputs["subsatellite_longitude"], dtype=float)
        cone_rates = np.asarray(l1b_inputs.get("cone_angle_rate", np.full(latitudes.shape, np.nan)), dtype=float)

    footprints: list[RadiometerFootprint] = []
    for i, (lat, lon, vza) in enumerate(zip(latitudes, longitudes, viewing_zeniths, strict=True)):
        if not have_scan_ref:
            footprints.append(
                RadiometerFootprint(
                    bbox=bounding_box_from_boresight(float(lat), float(lon), float(vza)),
                    latitude=float(lat),
                    longitude=float(lon),
                    spacecraft_altitude_km=NOMINAL_ALTITUDE_KM,
                    viewing_zenith_angle=float(vza),
                )
            )
            continue

        subsatellite_lat = float(subsatellite_lats[i])
        subsatellite_lon = float(subsatellite_lons[i])
        cone_rate = float(cone_rates[i])
        off_limb = False
        try:
            bbox = compute_footprint_bounding_box(
                float(lat),
                float(lon),
                subsatellite_lat,
                subsatellite_lon,
                float(vza),
                on_limb="flag",
            )
        except OffLimbError:
            # A space/calibration view that slipped past the reader's finite filter has
            # no Earth footprint. Keep index alignment with a boresight placeholder box,
            # but mark the record off-limb so the aggregation path leaves its external
            # variables at fill and scores it zero coverage -- rather than tiling and
            # weighting a fabricated geographic box as if it were a real observation.
            bbox = bounding_box_from_boresight(float(lat), float(lon), float(vza))
            off_limb = True
        footprints.append(
            RadiometerFootprint(
                bbox=bbox,
                latitude=float(lat),
                longitude=float(lon),
                spacecraft_altitude_km=NOMINAL_ALTITUDE_KM,
                viewing_zenith_angle=float(vza),
                subsatellite_latitude=subsatellite_lat,
                subsatellite_longitude=subsatellite_lon,
                # A fill/NaN cone-angle rate means "unknown scan rate" -> None.
                cone_angle_rate=None if not np.isfinite(cone_rate) else cone_rate,
                off_limb=off_limb,
            )
        )
    return footprints


def _merge_computed_variables(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    mode: OperationalMode,
    footprints: Sequence[Any],
    *,
    solar_zenith_angle: np.ndarray,
    viewing_zenith_angle: np.ndarray,
    relative_azimuth_angle: np.ndarray,
    source_file_paths: dict[str, Path] | None,
    tile_manager: TileManager | None,
    weigher: PixelWeigher | None,
) -> None:
    """Merge the *computed* (non-input) product columns into ``data`` in place.

    Fills the two families of variable the footprint-matching engine owns, so they
    carry real values instead of placeholders:

    * **Derived viewing geometry** (``sunglint_angle``) -- always computed, from the
      real per-footprint geolocation angles (:func:`compute_derived_viewing_geometry`).
    * **External aggregated variables** and the **coverage/QA** columns
      (``psf_coverage_fraction``, ``q_flags``) -- computed only when ancillary inputs
      are available (a ``tile_manager`` or ``source_file_paths``). Without them the
      external columns are left for :func:`_fill_placeholder_variables` (the
      pre-staging / stubbed-input behavior).

    Only variables the ``definition`` actually declares are written, so a mode whose
    product omits a column is unaffected. Runs *before*
    :func:`_fill_placeholder_variables`, which then fills only whatever remains.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        The variable arrays assembled so far; mutated in place.
    definition : LiberaDataProductDefinition
        The product definition (its declared variables + dtypes gate the merge).
    mode : OperationalMode
        Operational mode selecting the active readers/specs.
    footprints : Sequence[Any]
        Footprint objects (camera ``PseudoFootprint`` or ``RadiometerFootprint``) each
        exposing ``bbox`` and the geometry the weigher reads.
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : np.ndarray
        Per-footprint geolocation angles (degrees) for the sun-glint calculation.
    source_file_paths : dict[str, Path] or None
        Reader-key -> file map used to build a TileManager when ``tile_manager`` is not
        given. When both are ``None`` the external aggregation is skipped.
    tile_manager : TileManager or None
        A pre-built manager to aggregate from. Takes precedence over
        ``source_file_paths``.
    weigher : PixelWeigher or None
        PSF weigher for the aggregation (defaults to the radial stand-in downstream).
    """
    # Derived geometry is always computable from the real angle columns.
    if "sunglint_angle" in definition.variables and "sunglint_angle" not in data:
        sunglint = compute_derived_viewing_geometry(solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle)[
            "sunglint_angle"
        ]
        data["sunglint_angle"] = np.asarray(sunglint, dtype=np.dtype(definition.variables["sunglint_angle"].dtype))

    # External aggregation needs staged ancillary data; skip (leave placeholders) when
    # none was supplied.
    if tile_manager is None and source_file_paths is None:
        return

    aggregated, coverage = aggregate_external_variables(
        mode,
        footprints,
        tile_manager,
        source_file_paths=source_file_paths,
        weigher=weigher,
        return_coverage=True,
    )
    for name, values in aggregated.items():
        if name in definition.variables:
            data[name] = np.asarray(values, dtype=np.dtype(definition.variables[name].dtype))
    _merge_coverage_qa(data, definition, coverage, footprints)


def _merge_coverage_qa(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    coverage: np.ndarray,
    footprints: Sequence[Any],
) -> None:
    """Fill the coverage fraction and OR the CERES coverage rule into ``q_flags``.

    Implements design-doc section 2.4.2.7 as *flags*, not discards (Decision 4 of the
    wiring plan): coverage in ``[0.75, 0.95)`` sets ``PARTIAL_COVERAGE``; coverage
    below ``0.75`` sets ``INSUFFICIENT_COVERAGE``; a limb-truncated bounding box sets
    ``LIMB_TRUNCATED``; an off-limb (space/calibration) view with no Earth footprint sets
    ``OFF_LIMB`` (and, via its forced zero coverage, ``INSUFFICIENT_COVERAGE``). These
    bits are OR-ed into any ``q_flags`` already present (the
    camera path sets its segmentation flags first), so a footprint carries both its
    segmentation and its coverage provenance.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Variable arrays; ``psf_coverage_fraction`` and ``q_flags`` are set/updated here.
    definition : LiberaDataProductDefinition
        Gates which of the two columns are written (both are declared by every FMATCH
        definition, but the guard keeps this robust).
    coverage : np.ndarray
        Per-footprint coverage fraction in ``[0, 1]``.
    footprints : Sequence[Any]
        Footprints, read only for each ``bbox.truncated`` flag.
    """
    if "psf_coverage_fraction" in definition.variables:
        data["psf_coverage_fraction"] = np.asarray(
            coverage, dtype=np.dtype(definition.variables["psf_coverage_fraction"].dtype)
        )

    if "q_flags" not in definition.variables:
        return
    coverage = np.asarray(coverage, dtype=float)
    qbits = np.zeros(coverage.shape, dtype=np.int64)
    partial = (coverage >= ACCEPT_COVERAGE_THRESHOLD) & (coverage < PARTIAL_COVERAGE_THRESHOLD)
    insufficient = coverage < ACCEPT_COVERAGE_THRESHOLD
    truncated = np.array([bool(getattr(f.bbox, "truncated", False)) for f in footprints], dtype=bool)
    off_limb = np.array([bool(getattr(f, "off_limb", False)) for f in footprints], dtype=bool)
    qbits[partial] |= int(FmatchCoverageFlag.PARTIAL_COVERAGE)
    qbits[insufficient] |= int(FmatchCoverageFlag.INSUFFICIENT_COVERAGE)
    qbits[truncated] |= int(FmatchCoverageFlag.LIMB_TRUNCATED)
    qbits[off_limb] |= int(FmatchCoverageFlag.OFF_LIMB)
    # OR into any q_flags already assembled (e.g. camera segmentation flags).
    existing = data.get("q_flags")
    if existing is not None:
        qbits |= np.asarray(existing, dtype=np.int64)
    data["q_flags"] = qbits.astype(np.dtype(definition.variables["q_flags"].dtype))


def assemble_fmatch_dataset(
    mode: OperationalMode,
    *args: Any,
    cloud_fraction_camera: np.ndarray | None = None,
    **kwargs: Any,
) -> Dataset:
    """Assemble a conformant FMATCH :class:`xarray.Dataset` for an operational mode.

    Combines the per-footprint geolocation inputs, the derived viewing geometry from
    :func:`compute_derived_viewing_geometry`, and the aggregated external
    variables from :func:`aggregate_external_variables` into the variable dict
    expected by the mode's product definition (from
    :func:`load_fmatch_definition`), then builds a Dataset via
    ``LiberaDataProductDefinition.create_product_dataset`` and brings it into
    conformance with ``enforce_dataset_conformance``.

    Dispatch is by timescale, because that determines what the mode is built
    *from*:

    * **Camera-timescale** (``CAM_CAMTIME``, ``IMAGER_CAMTIME``) - assembled from
      the camera pseudo-footprints produced by
      :func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
      The first positional argument is that sequence of
      :class:`PseudoFootprint` objects. See :func:`_assemble_camtime_dataset`.
    * **Radiometer-timescale** (``CAM``, ``IMAGER_FLASH``, ``IMAGER``) - assembled
      from the L1B pass-through arrays produced by
      :func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`.
      The first positional argument is that dict. See
      :func:`_assemble_radiometer_dataset`.

    In both cases the derived viewing geometry (``sunglint_angle``) is always computed
    from the real geolocation angles. The external aggregated variables and the
    coverage/QA columns are computed too **when ancillary inputs are supplied** (a
    ``tile_manager`` or ``source_file_paths`` keyword, forwarded to
    :func:`aggregate_external_variables`); without them those columns fall back to
    conformant placeholders (the pre-staging / stubbed-input behavior).

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being assembled.
    *args, **kwargs
        Mode-specific inputs, forwarded to the timescale's assembler (see above
        for the leading positional argument of each). This is also how the optional
        ``source_file_paths`` / ``tile_manager`` / ``weigher`` aggregation keywords
        reach the assemblers.
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint cloud fraction from the Camera Cloud Fraction (CF-CAM)
        algorithm (Libera WFOV camera), as a 1-D array indexed by footprint in
        the same order as the time coordinate. This is an *internal* algorithm
        output - it does not come from a reader and is already aggregated to one
        value per footprint - so it is merged directly into the ``cloud_fraction_camera``
        variable rather than going through :func:`aggregate_external_variables`.
        Only the CAM modes (``CAM``, ``CAM_CAMTIME``) declare this variable; it is
        ``None`` for the IMAGER modes.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's product definition.
    """
    if mode in _CAMERA_TIMESCALE_MODES:
        return _assemble_camtime_dataset(*args, mode=mode, cloud_fraction_camera=cloud_fraction_camera, **kwargs)
    return _assemble_radiometer_dataset(*args, mode=mode, cloud_fraction_camera=cloud_fraction_camera, **kwargs)


def _placeholder_variable_array(variable_definition: Any, n_footprints: int) -> np.ndarray:
    """Build a conformant placeholder array for a not-yet-computed product variable.

    Variables owned by the aggregation / derived-geometry engines (not built yet)
    still have to appear in the output file with the right dtype and shape so the
    product conforms to its definition. We fill them with the variable's declared
    ``_FillValue`` when it has one, and otherwise with ``NaN`` for floating-point
    variables or ``0`` for integer variables. The magnitudes are meaningless; only
    the dtype/shape/attributes form the product contract (the same stance the
    example-product generator takes in ``notebooks/generate_example_products.ipynb``).

    Parameters
    ----------
    variable_definition : LiberaVariableDefinition
        The product-definition entry for the variable.
    n_footprints : int
        Length of the footprint (time) axis.

    Returns
    -------
    np.ndarray
        A 1-D array of length ``n_footprints`` of the variable's declared dtype.
    """
    dtype = np.dtype(variable_definition.dtype)
    fill_value = variable_definition.attributes.get("_FillValue")
    if fill_value is None:
        # No declared fill: NaN reads as "missing" for floats; 0 is the neutral
        # integer stand-in (integers cannot represent NaN).
        fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
    return np.full(n_footprints, fill_value, dtype=dtype)


def _fill_placeholder_variables(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    n_footprints: int,
) -> None:
    """Fill every declared variable missing from ``data`` with a conformant placeholder.

    Mutates ``data`` in place, adding one array per not-yet-computed variable. The
    product definition requires every declared variable to be present, so this is
    what lets a product conform while the aggregation / derived-geometry engines are
    still ``TODO[LIBSDC-785]`` stubs. The placeholder values are structurally valid
    (declared dtype/shape/attributes) but numerically meaningless.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        The variable arrays assembled so far (the real, input-derived columns).
    definition : LiberaDataProductDefinition
        The product definition naming every variable the file must contain.
    n_footprints : int
        Length of the footprint (time) axis.
    """
    for name, variable_definition in definition.variables.items():
        if name not in data:
            data[name] = _placeholder_variable_array(variable_definition, n_footprints)


def _finalize_product_dataset(
    definition: LiberaDataProductDefinition,
    data: dict[str, np.ndarray],
    *,
    algorithm_version: str | None,
    input_files: str | None,
) -> Dataset:
    """Build a conformant Dataset from assembled arrays and set the dynamic global attributes.

    Shared tail of both assembly paths. Dynamic (per-run) global attributes are set
    directly on the Dataset; they are declared (as null) in the definition, so
    ``enforce_dataset_conformance`` keeps them rather than stripping them as extras.

    Parameters
    ----------
    definition : LiberaDataProductDefinition
        The product definition to build against.
    data : dict[str, np.ndarray]
        Every coordinate and variable array the product declares.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute.

    Returns
    -------
    xarray.Dataset
        The conformance-enforced dataset, ready to write.
    """
    dataset = definition.create_product_dataset(data)
    dataset = definition.enforce_dataset_conformance(dataset)
    if input_files is not None:
        dataset.attrs["input_files"] = input_files
    if algorithm_version is not None:
        dataset.attrs["algorithm_version"] = algorithm_version
    return dataset


def _normalize_longitude(longitude_deg: float) -> float:
    """Wrap a longitude into [-180, 180).

    Corner-derived bounding boxes that straddle the antimeridian can report a
    ``lon_max`` greater than 180 (the :class:`BoundingBox` dateline convention).
    The product definition's ``psf_bbox_lon_*`` variables declare a [-180, 180]
    valid range, so we wrap the stored bounds back into that convention. Downstream
    consumers can still detect a dateline-crossing box because it then has
    ``lon_min > lon_max``.
    """
    return (longitude_deg + 180.0) % 360.0 - 180.0


def _assemble_camtime_dataset(
    footprints: Sequence[PseudoFootprint],
    *,
    mode: OperationalMode = OperationalMode.CAM_CAMTIME,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Path] | None = None,
    tile_manager: TileManager | None = None,
    weigher: PixelWeigher | None = None,
) -> Dataset:
    """Assemble a camera-timescale FMATCH Dataset from camera pseudo-footprints.

    Serves both camera-timescale modes (``CAM_CAMTIME`` and ``IMAGER_CAMTIME``).
    They are built the same way - from the same L1B camera segmentation - and
    differ only in the set of aggregated variables their definitions declare,
    which the computed-and-placeholder fill handles generically.

    Builds the per-footprint variable arrays declared by the mode's product
    definition. The centre-pixel geolocation/geometry, the corner-derived PSF
    bounding box, and the QA flags come straight from the pseudo-footprints
    (:data:`_CAMTIME_SEGMENTATION_VARIABLES`). The derived ``sunglint_angle`` is
    computed from the geolocation angles, and (when ``source_file_paths`` or
    ``tile_manager`` is supplied) the external aggregated variables and coverage/QA
    columns are computed via :func:`_merge_computed_variables`; any remaining declared
    variable is a conformant placeholder.

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in write order, as returned by
        :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
    mode : OperationalMode, optional
        Which camera-timescale mode to assemble. Defaults to ``CAM_CAMTIME``.
    definition : LiberaDataProductDefinition, optional
        The product definition. Loaded via :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute
        (typically the source L1B camera filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). Only the
        CAM modes declare this variable; for ``IMAGER_CAMTIME`` it is ignored. When
        omitted (or undeclared) the variable is written as a placeholder.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's definition.

    Raises
    ------
    ValueError
        If ``mode`` is not a camera-timescale mode, or ``footprints`` is empty
        (there would be no time axis to write).
    """
    if mode not in _CAMERA_TIMESCALE_MODES:
        raise ValueError(
            f"{mode.value} is not a camera-timescale mode; camera pseudo-footprints only assemble "
            f"{', '.join(sorted(m.value for m in _CAMERA_TIMESCALE_MODES))}."
        )
    if definition is None:
        definition = load_fmatch_definition(mode)

    footprints = list(footprints)
    if not footprints:
        raise ValueError(f"Cannot assemble a {mode.value} product from zero pseudo-footprints.")
    n_footprints = len(footprints)

    time_variable = fmatch_time_variable(mode)  # "CAMERA_TIME"

    # The real, segmentation-derived 1-D columns. Longitudes of the PSF box are wrapped
    # into [-180, 180) to satisfy the product definition's valid range. center_pixel_x/y
    # are the boresight stand-in pixel; they are FMATCH-only provenance (the block's
    # inclusive extent is emitted as the camera_pixel_x/y range coordinates below).
    real_columns: dict[str, list[float]] = {
        "latitude": [f.latitude for f in footprints],
        "longitude": [f.longitude for f in footprints],
        "altitude": [f.altitude for f in footprints],
        "solar_zenith_angle": [f.solar_zenith_angle for f in footprints],
        "viewing_zenith_angle": [f.viewing_zenith_angle for f in footprints],
        "relative_azimuth_angle": [f.relative_azimuth_angle for f in footprints],
        "psf_bbox_lat_min": [f.bbox.lat_min for f in footprints],
        "psf_bbox_lat_max": [f.bbox.lat_max for f in footprints],
        "psf_bbox_lon_min": [_normalize_longitude(f.bbox.lon_min) for f in footprints],
        "psf_bbox_lon_max": [_normalize_longitude(f.bbox.lon_max) for f in footprints],
        "q_flags": [int(f.q_flags) for f in footprints],
        "center_pixel_x": [f.center_ix for f in footprints],
        "center_pixel_y": [f.center_iy for f in footprints],
    }

    # Start the data dict with the time coordinate (nanosecond datetimes; note the values
    # repeat within an image -- see segment_l1b_camera's docstring). CAMERA_TIME is a
    # coordinate on the FOOTPRINT record axis: create_product_dataset routes it to .coords
    # because the definition declares it under coordinates:.
    data: dict[str, np.ndarray] = {
        time_variable: np.array([f.time for f in footprints], dtype="datetime64[ns]"),
    }

    # Camera pixel-index ranges as inclusive (min, max) pairs on the CAMERA_PIXEL_BOUNDS
    # axis. slice_x/slice_y are half-open [start, stop), so the inclusive maximum is
    # stop - 1. These are 2-D coordinates (FOOTPRINT x CAMERA_PIXEL_BOUNDS) that both
    # camtime products declare and that pass straight through to SCENE-ID-CAM-CAMTIME.
    pixel_ranges: dict[str, list[tuple[int, int]]] = {
        "camera_pixel_x": [(f.slice_x.start, f.slice_x.stop - 1) for f in footprints],
        "camera_pixel_y": [(f.slice_y.start, f.slice_y.stop - 1) for f in footprints],
    }
    for coordinate_name, ranges in pixel_ranges.items():
        coordinate_definition = definition.coordinates[coordinate_name]
        data[coordinate_name] = np.asarray(ranges, dtype=np.dtype(coordinate_definition.dtype))

    # Cast each real 1-D column to the exact dtype the definition declares. Only columns
    # the definition actually declares are written; both camtime products declare the same
    # segmentation variables, but the guard keeps the assembly robust to definition drift.
    for name, values in real_columns.items():
        variable_definition = definition.variables.get(name)
        if variable_definition is None:
            continue
        data[name] = np.asarray(values, dtype=np.dtype(variable_definition.dtype))

    # Optional internal (non-reader) Camera Cloud Fraction values. Guarded on the
    # definition declaring the variable, because IMAGER-CAMTIME does not.
    _merge_cloud_fraction_camera(data, definition, cloud_fraction_camera)

    # Computed columns: derived geometry (always) and, when ancillary inputs are
    # supplied, the external aggregated variables + coverage/QA (OR-ed into q_flags).
    _merge_computed_variables(
        data,
        definition,
        mode,
        footprints,
        solar_zenith_angle=np.asarray(real_columns["solar_zenith_angle"], dtype=float),
        viewing_zenith_angle=np.asarray(real_columns["viewing_zenith_angle"], dtype=float),
        relative_azimuth_angle=np.asarray(real_columns["relative_azimuth_angle"], dtype=float),
        source_file_paths=source_file_paths,
        tile_manager=tile_manager,
        weigher=weigher,
    )

    # Every remaining declared variable is filled with a placeholder until its engine exists.
    _fill_placeholder_variables(data, definition, n_footprints)

    return _finalize_product_dataset(
        definition,
        data,
        algorithm_version=algorithm_version,
        input_files=input_files,
    )


def _merge_cloud_fraction_camera(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    cloud_fraction_camera: np.ndarray | None,
) -> None:
    """Merge the CF-CAM cloud fraction into ``data`` when it is supplied and declared.

    ``cloud_fraction_camera`` is an internal Libera algorithm output (from the WFOV
    Camera Cloud Fraction algorithm), already one value per footprint, so it bypasses
    the reader/aggregation path and is merged straight in. Only the CAM-family
    definitions declare it; passing values for an IMAGER mode is ignored rather than
    raising, so a caller can hand the same inputs to any mode.
    """
    if cloud_fraction_camera is None:
        return
    variable_definition = definition.variables.get("cloud_fraction_camera")
    if variable_definition is None:
        logger.warning(
            "cloud_fraction_camera values were supplied but %s does not declare that variable; ignoring them.",
            definition.attributes.get("ProductID", "this product"),
        )
        return
    data["cloud_fraction_camera"] = np.asarray(cloud_fraction_camera, dtype=np.dtype(variable_definition.dtype))


def _assemble_radiometer_dataset(
    l1b_inputs: dict[str, np.ndarray],
    *,
    mode: OperationalMode,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Path] | None = None,
    tile_manager: TileManager | None = None,
    weigher: PixelWeigher | None = None,
) -> Dataset:
    """Assemble a radiometer-timescale FMATCH Dataset from L1B pass-through inputs.

    Serves the three radiometer-timescale modes (``CAM``, ``IMAGER_FLASH``,
    ``IMAGER``). Their footprints are the L1B radiometer footprints themselves, so
    the time coordinate and the geolocation/viewing-angle columns
    (:data:`_RADIOMETER_L1B_VARIABLES`) are carried through verbatim from L1B. The
    derived ``sunglint_angle`` is computed from those angles, and (when
    ``source_file_paths`` or ``tile_manager`` is supplied) the external aggregated
    variables and coverage/QA columns are computed from bounding boxes built for each
    footprint via :func:`build_radiometer_footprints`; any remaining declared variable
    is a conformant placeholder.

    Parameters
    ----------
    l1b_inputs : dict[str, np.ndarray]
        The pass-through arrays from
        :func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`:
        the ``RADIOMETER_TIME`` coordinate plus each of
        :data:`_RADIOMETER_L1B_VARIABLES`, all the same length.
    mode : OperationalMode
        Which radiometer-timescale mode to assemble.
    definition : LiberaDataProductDefinition, optional
        The product definition. Loaded via :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute
        (typically the source L1B radiometer filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV), in the same
        footprint order as the time coordinate. Only ``CAM`` declares this variable.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's definition.

    Raises
    ------
    ValueError
        If ``mode`` is a camera-timescale mode, if a required pass-through input is
        missing, if the inputs have inconsistent lengths, or if they are empty.
    """
    if mode in _CAMERA_TIMESCALE_MODES:
        raise ValueError(
            f"{mode.value} is a camera-timescale mode and cannot be assembled from L1B radiometer pass-through "
            f"inputs; use the camera pseudo-footprint path instead."
        )
    if definition is None:
        definition = load_fmatch_definition(mode)

    time_variable = fmatch_time_variable(mode)  # "RADIOMETER_TIME"

    # Fail with the specific missing names rather than a bare KeyError deep in the
    # loop below, because a partial pass-through dict is the most likely caller error.
    required = {time_variable, *_RADIOMETER_L1B_VARIABLES}
    missing = sorted(required - set(l1b_inputs))
    if missing:
        raise ValueError(f"L1B pass-through inputs for {mode.value} are missing required key(s): {', '.join(missing)}")

    n_footprints = len(l1b_inputs[time_variable])
    if n_footprints == 0:
        raise ValueError(f"Cannot assemble a {mode.value} product from zero footprints.")
    inconsistent = sorted(name for name in required if len(l1b_inputs[name]) != n_footprints)
    if inconsistent:
        raise ValueError(
            f"L1B pass-through inputs for {mode.value} have inconsistent lengths; expected {n_footprints} footprints "
            f"(from {time_variable}) but got a different length for: {', '.join(inconsistent)}"
        )

    # Start with the time coordinate, then the real L1B columns cast to the exact
    # dtype the definition declares.
    data: dict[str, np.ndarray] = {
        time_variable: np.asarray(l1b_inputs[time_variable], dtype="datetime64[ns]"),
    }
    for name in sorted(_RADIOMETER_L1B_VARIABLES):
        data[name] = np.asarray(l1b_inputs[name], dtype=np.dtype(definition.variables[name].dtype))

    # Optional internal (non-reader) Camera Cloud Fraction values (CAM only).
    _merge_cloud_fraction_camera(data, definition, cloud_fraction_camera)

    # Computed columns: derived geometry (always) and, when ancillary inputs are
    # supplied, the external aggregated variables + coverage/QA. The radiometer
    # footprints (with their PSF bounding boxes) are built only when aggregation will
    # actually run, so the pass-through-only path stays allocation-light.
    footprints: Sequence[Any] = (
        build_radiometer_footprints(l1b_inputs) if (tile_manager is not None or source_file_paths is not None) else []
    )
    # The radiometer footprints carry the L1B scan reference (subsatellite point +
    # cone-angle rate), so default this path to the CERES-faithful AngularPSFWeigher
    # (design-doc section 2.8.1.2), which orients the PSF along the real scan plane. It
    # degrades to a nadir frame for any footprint missing the scan reference, so it is
    # safe even for the boresight-box fallback. Callers can still override via `weigher`.
    if weigher is None:
        weigher = AngularPSFWeigher()
    _merge_computed_variables(
        data,
        definition,
        mode,
        footprints,
        solar_zenith_angle=np.asarray(l1b_inputs["solar_zenith_angle"], dtype=float),
        viewing_zenith_angle=np.asarray(l1b_inputs["viewing_zenith_angle"], dtype=float),
        relative_azimuth_angle=np.asarray(l1b_inputs["relative_azimuth_angle"], dtype=float),
        source_file_paths=source_file_paths,
        tile_manager=tile_manager,
        weigher=weigher,
    )

    # Every remaining declared variable is filled with a placeholder until its engine exists.
    _fill_placeholder_variables(data, definition, n_footprints)

    return _finalize_product_dataset(
        definition,
        data,
        algorithm_version=algorithm_version,
        input_files=input_files,
    )


def write_fmatch_product(mode: OperationalMode, *args: Any, **kwargs: Any) -> Any:
    """Write a FMATCH NetCDF data product to disk for an operational mode.

    Delegates to ``libera_utils.io.netcdf.write_libera_data_product`` using the
    definition from :func:`load_fmatch_definition`, the assembled Dataset from
    :func:`assemble_fmatch_dataset`, and ``time_variable=fmatch_time_variable(mode)``
    (``RADIOMETER_TIME`` or ``CAMERA_TIME``) so the output filename encodes the
    footprint time span.

    Every operational mode is supported. The mode's timescale selects what the
    leading positional argument must be - camera pseudo-footprints for the
    camera-timescale modes, L1B pass-through arrays for the radiometer-timescale
    ones - exactly as in :func:`assemble_fmatch_dataset`.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    *args, **kwargs
        Mode-specific inputs forwarded to :func:`assemble_fmatch_dataset`, followed
        by ``output_path``. See :func:`_write_fmatch_product` for the accepted
        keyword arguments.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.
    """
    return _write_fmatch_product(mode, *args, **kwargs)


def _write_fmatch_product(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    output_path: str | Path,
    *,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Path] | None = None,
    weigher: PixelWeigher | None = None,
    strict: bool = True,
) -> LiberaDataProductFilename:
    """Assemble and write one FMATCH NetCDF product.

    Loads the product definition once (so assembly and writing cannot disagree about
    it), assembles a conformant Dataset via :func:`assemble_fmatch_dataset`, and
    writes it with ``write_libera_data_product``, which generates the standardized
    Libera filename from the product's time span.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    inputs : Sequence[PseudoFootprint] | dict[str, np.ndarray]
        The mode's assembly inputs: camera pseudo-footprints for the camera-timescale
        modes, or the L1B pass-through dict for the radiometer-timescale modes.
    output_path : str or pathlib.Path
        Directory (or S3 prefix) to write the product file into.
    algorithm_version : str, optional
        Value for the ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the ``input_files`` global attribute.
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). Only the
        CAM modes declare this variable.
    source_file_paths : dict[str, Path], optional
        Reader-key -> staged ancillary file map. When supplied, the external
        aggregated variables and coverage/QA columns are computed (via a TileManager
        built from these files); otherwise those columns are conformant placeholders.
    weigher : PixelWeigher, optional
        PSF weigher used by the aggregation path (defaults to the radial stand-in).
    strict : bool, optional
        When True (default), fail if the assembled Dataset does not conform.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.
    """
    definition = load_fmatch_definition(mode)
    dataset = assemble_fmatch_dataset(
        mode,
        inputs,
        definition=definition,
        algorithm_version=algorithm_version,
        input_files=input_files,
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=source_file_paths,
        weigher=weigher,
    )
    return write_libera_data_product(
        data_product_definition=definition,
        data=dataset,
        output_path=output_path,
        time_variable=fmatch_time_variable(mode),
        strict=strict,
    )
