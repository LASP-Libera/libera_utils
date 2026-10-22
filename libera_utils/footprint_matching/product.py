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
  ancillary inputs are supplied - the PSF aggregation of each source's native samples.
* **Placeholders** - a declared variable no input produced: the external columns
  when the caller opts in with ``allow_placeholders=True`` rather than staging
  ancillary data, or a source such as FLASHFlux that has no reader.

A placeholder is *structurally* correct (declared dtype, shape and attributes)
but numerically meaningless. Floating-point placeholders are filled with ``NaN``
and integer placeholders with :data:`FMATCH_INTEGER_FILL_VALUE` (``-1``), or ``0`` for a
bitmask; see
:func:`_fill_placeholder_variables`.

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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.config import config
from libera_utils.footprint_matching.aggregation import (
    ACCEPT_COVERAGE_THRESHOLD,
    PARTIAL_COVERAGE_THRESHOLD,
    aggregate_sample_variables,
)
from libera_utils.footprint_matching.geometry import (
    L1B_FILL_VALUE,
    WGS84_SEMI_MAJOR_AXIS_KM,
    compute_footprint_bounding_boxes,
    compute_viewing_frames,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.sources import AncillarySources, build_sources
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
)
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    from collections.abc import Callable, Sequence

    from xarray import Dataset

    from libera_utils.footprint_matching.geometry import ViewingFrame
    from libera_utils.footprint_matching.types import PseudoFootprint
    from libera_utils.io.filenaming import LiberaDataProductFilename

logger = logging.getLogger(__name__)

# Product-definition variable names that the camera-segmentation tool fills with
# *real* per-footprint values (center-pixel geolocation/geometry, the corner-derived
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
        # Boresight (center) pixel provenance: which L1B camera pixel this pseudo-footprint's
        # boresight stand-in falls on. Real per-footprint integers straight off the
        # PseudoFootprint. The block's inclusive pixel extent is emitted separately as the four
        # camera_pixel_{x,y}_{min,max} COORDINATES, which this set (real data *variables*)
        # deliberately does not list.
        "center_pixel_x",
        "center_pixel_y",
    }
)

# Mapping of FMATCH product variable name -> L1B RAD-4CH variable name, for the
# per-footprint quantities FMATCH copies straight out of L1B rather than computing.

L1B_PASSTHROUGH_VARIABLES: dict[str, str] = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "altitude": "Altitude",
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
# The subsatellite point and the satellite's distance from Earth's center are required to
# orient the scan plane and locate the satellite, so they are AND-ed into the
# finite-footprint mask alongside the pass-through variables (a footprint without valid
# ephemeris cannot get a real box). They are read at float64 for geometry precision rather
# than the float32 the product columns use.
#
# Like L1B_PASSTHROUGH_VARIABLES this lives here (not in the runner that reads it) so the
# assembly path and the runner share one source of truth without a circular import.
L1B_SCAN_REFERENCE_VARIABLES: dict[str, str] = {
    "subsatellite_latitude": "Subsatellite_Latitude",
    "subsatellite_longitude": "Subsatellite_Longitude",
    "satellite_radius_km": "Radius_of_Satellite_from_Center_of_Earth",
}

# Nominal satellite radius for camera pseudo-footprints, which carry no ephemeris. Only the
# radial stand-in weigher reads it there, to size its PSF ground radius for an ~835 km orbit.
# TODO[LIBSDC-884]: take the camera record's ephemeris once camera segmentation carries it.
_CAMERA_NOMINAL_SATELLITE_RADIUS_KM: float = WGS84_SEMI_MAJOR_AXIS_KM + 835.0

# The instrument cone-angle rate (degrees/second) sets the along-scan PSF orientation
# and flags the stationary-scanner case. Unlike the subsatellite point it is optional:
# a fill/NaN value simply means "unknown scan rate" and is normalized to None per
# footprint downstream, so it is read but deliberately kept out of the finite mask (it
# must not drop an otherwise-good Earth-viewing footprint).
L1B_CONE_ANGLE_RATE_VARIABLE: str = "Cone_Angle_Rate"
FMATCH_CONE_ANGLE_RATE_KEY: str = "cone_angle_rate"
# Key of the radiometer observation time in the L1B input map (and the product's time
# coordinate for the radiometer-timescale modes). Each footprint carries it so the readers
# can match ancillary samples to it in time.
RADIOMETER_TIME_KEY: str = "RADIOMETER_TIME"
# Configuration key for the number of worker processes the external aggregation runs on.
# ``1`` (the default in ``config.json``) keeps everything in the calling process.
FMATCH_WORKERS_CONFIG_KEY: str = "FMATCH_WORKERS"
# Chunks per worker: footprints are split into this many contiguous ranges per worker so
# a slow chunk (a stepped source reloading, a dense swath) does not idle the others.
_CHUNKS_PER_WORKER: int = 4
# Below this many footprints the pool's start-up costs more than it saves.
_MIN_FOOTPRINTS_PER_WORKER: int = 50

# The sources a forked worker inherits from its parent (copy-on-write): set by
# _aggregate_in_parallel right before the pool starts, cleared after. A worker started by
# spawn sees None here and builds its own sources from the staged paths (see
# _worker_sources).
_FORK_SOURCES: AncillarySources | None = None
# Sources a spawned worker built, kept for the next chunk it is handed. Keyed by the mode
# and the staged paths so a worker reused for another run never serves stale readers.
_WORKER_SOURCES_CACHE: dict[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], AncillarySources] = {}

# The radiometer-timescale counterpart of _CAMTIME_SEGMENTATION_VARIABLES: the product columns copied straight from L1B.
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

# Fill for a missing or padded cell of an integer category or pixel-index column. 0 is a real
# value in each of them (IGBP surface types 1-17, VIIRS Deep Blue aerosol types 0-7, CLDPIX cloud
# mask 0-3 and phase 1-5, as the igbp, aod and cldpix readers define them, and camera pixel
# indices from 0), so the fill sits below every domain. Integer bitmasks are filled with 0 instead,
# meaning no flag set (see :func:`_fill_value_for`). Public because readers of the FMATCH products
# (e.g. SCENE-ID) must recognize it: the product definitions declare no _FillValue for these columns.
FMATCH_INTEGER_FILL_VALUE: int = -1


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


@dataclass
class _FootprintGeometry:
    """Viewing geometry a PSF weigher needs, extracted from a footprint object."""

    boresight_lat_deg: float
    boresight_lon_deg: float
    viewing_zenith_deg: float
    subsatellite_lat_deg: float
    subsatellite_lon_deg: float
    satellite_radius_km: float
    cone_angle_rate: float | None
    #: False for a footprint with no L1B ephemeris (a camera pseudo-footprint), whose
    #: subsatellite point and radius are stand-ins only the radial weigher can use.
    has_ephemeris: bool


def _footprint_geometry(footprint: Any) -> _FootprintGeometry:
    """Extract the viewing geometry a PSF weigher needs from a footprint object.

    Every on-Earth footprint carries a ``bbox`` and a ``viewing_zenith_angle``; the boresight is
    read from ``latitude``/``longitude`` when present and otherwise taken as the box
    center. A :class:`RadiometerFootprint` carries the L1B
    ephemeris (subsatellite point and satellite radius) the angular-frame weigher needs.
    A camera pseudo-footprint carries none: it gets the boresight as its subsatellite
    point and :data:`_CAMERA_NOMINAL_SATELLITE_RADIUS_KM`, which only the radial stand-in
    can use, and ``has_ephemeris`` is ``False``.

    The L1B ``Cone_Angle_Rate`` fill value (``-999``) is normalized to ``None`` so a
    fill never masquerades as a real (near-zero) stationary-scanner rate.

    Raises
    ------
    ValueError
        If the footprint's ``viewing_zenith_angle`` is ``None`` or not finite, since the
        weighers size the PSF from it.
    """
    bbox = footprint.bbox
    lat = getattr(footprint, "latitude", None)
    lon = getattr(footprint, "longitude", None)
    if lat is None or lon is None:
        # Box center. (A dateline-wrapping box reports lon_max > 180; the mean still
        # lands inside it, which is good enough for the weight kernel.)
        lat = 0.5 * (bbox.lat_min + bbox.lat_max)
        lon = 0.5 * (bbox.lon_min + bbox.lon_max)
    viewing_zenith = footprint.viewing_zenith_angle
    if viewing_zenith is None or not np.isfinite(viewing_zenith):
        raise ValueError(
            f"The footprint at {bbox} has no finite viewing zenith angle (got {viewing_zenith}); the PSF weigher "
            "sizes the footprint from it, so drop the record or supply its L1B Viewing_Zenith_Surface value."
        )

    # Ephemeris. Read only the dedicated fields so we never mistake another quantity for
    # them -- in particular the camera ``PseudoFootprint`` carries a meters-valued
    # ``altitude`` that is the center-pixel *surface height* (an output column), not a
    # spacecraft position.
    sub_lat = getattr(footprint, "subsatellite_latitude", None)
    sub_lon = getattr(footprint, "subsatellite_longitude", None)
    radius = getattr(footprint, "satellite_radius_km", None)
    has_ephemeris = sub_lat is not None and sub_lon is not None and radius is not None
    if not has_ephemeris:
        sub_lat, sub_lon, radius = lat, lon, _CAMERA_NOMINAL_SATELLITE_RADIUS_KM

    # Scan direction: prefer the L1B Cone_Angle_Rate; accept an ``alpha_dot`` alias.
    cone_rate = getattr(footprint, "cone_angle_rate", None)
    if cone_rate is None:
        cone_rate = getattr(footprint, "alpha_dot", None)
    if cone_rate is not None and (cone_rate == L1B_FILL_VALUE or not np.isfinite(cone_rate)):
        cone_rate = None

    return _FootprintGeometry(
        boresight_lat_deg=float(lat),
        boresight_lon_deg=float(lon),
        viewing_zenith_deg=float(viewing_zenith),
        subsatellite_lat_deg=float(sub_lat),
        subsatellite_lon_deg=float(sub_lon),
        satellite_radius_km=float(radius),
        cone_angle_rate=None if cone_rate is None else float(cone_rate),
        has_ephemeris=has_ephemeris,
    )


def _batch_viewing_frames(geoms: Sequence[_FootprintGeometry]) -> ViewingFrame:
    """Build every footprint's angular-PSF viewing frame in one batched call.

    Takes the same ephemeris the per-footprint frame construction inside
    :meth:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher.weight_field`
    would, so the batched frame yields the same ``(delta, beta)`` projection as the
    inline path -- it just builds every frame in one vectorized call. Off-Earth views
    have no frame, so the caller passes only the on-Earth footprints.

    Parameters
    ----------
    geoms : Sequence[_FootprintGeometry]
        Per-footprint viewing geometry, in footprint order.

    Returns
    -------
    ViewingFrame
        The batched frames, indexable per footprint via ``ViewingFrame.for_index``.
    """
    boresight_lat = np.array([g.boresight_lat_deg for g in geoms], dtype=float)
    boresight_lon = np.array([g.boresight_lon_deg for g in geoms], dtype=float)
    sub_lat = np.array([g.subsatellite_lat_deg for g in geoms], dtype=float)
    sub_lon = np.array([g.subsatellite_lon_deg for g in geoms], dtype=float)
    radius = np.array([g.satellite_radius_km for g in geoms], dtype=float)
    return compute_viewing_frames(boresight_lat, boresight_lon, sub_lat, sub_lon, satellite_radius_km=radius)


def aggregate_external_variables(
    mode: OperationalMode,
    footprints: Sequence[Any],
    sources: AncillarySources | None = None,
    *,
    source_file_paths: dict[str, Sequence[Path] | Path] | None = None,
    weigher: PixelWeigher | None = None,
    return_coverage: bool = False,
    n_workers: int | None = None,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate every active reader's native samples to one value per footprint.

    For the given operational mode this selects the active readers via
    ``ReaderRegistry.get_readers_for_mode(mode)``, and for each footprint asks each
    source (:class:`~libera_utils.footprint_matching.sources.AncillarySources`) for
    the native samples inside the footprint's bounding box matched to its time,
    weights each sample with the PSF ``weigher`` at the sample's own location, and
    applies each variable's aggregation strategy (weighted mean / mode / log-mean /
    std / ranked scene) to collapse the samples to a single value per footprint. No
    sample is averaged, binned or interpolated before the weigher sees it. The active
    reader set - and therefore the keys of the returned dict - grows with the mode's
    latency (e.g. CAM has era5_single_level, igbp, nise, viirs_brdf, viirs_cldprop_d3; IMAGER
    additionally has era5_pressure, viirs_aod, and the RBSP ssf and cldpix fields).
    Per-spec gating also applies (see below).

    Every output variable is named ``<source_key>_<spec_name>`` (e.g.
    ``era5_single_level_wind_u10``, ``igbp_surface_type``, ``cldpix_cloud_mask``), matching the
    product definition variable names. The reader's ``INSTRUMENT`` is recorded in
    each variable's ``long_name`` rather than in the name. Per-spec gating uses
    :func:`~libera_utils.footprint_matching.types.spec_active_in_mode`: a spec's
    ``required_mode`` rank rule, or an exact ``only_modes`` pin (e.g. the extended
    SSF cloud/aerosol/albedo fields are aggregated for FMATCH-IMAGER only).

    Processing is done one footprint at a time, so only the current footprint's
    samples are held beyond what each reader holds of its source. With ``n_workers``
    above one the footprints are split into contiguous along-track chunks and the chunks
    run in a process pool; the result is identical to the single-process run (every
    footprint is still processed in time order within its chunk, and a stepped source
    reloads its step at most once per chunk boundary). On Linux the pool is forked after
    the sources are loaded, so the workers share one copy of every source; where
    forking is unavailable each worker builds its own sources from ``source_file_paths``.

    The PSF weigher is swappable via ``weigher``. This function keeps the boresight
    **radial stand-in** (:class:`~libera_utils.footprint_matching.weighting.RadialWeigher`)
    as its own generic default, but the radiometer-timescale assembly path defaults to
    the scientifically faithful angular-frame PSF
    (:class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher`) now that the
    L1B ephemeris (subsatellite point + satellite radius) and cone-angle rate are read (see
    :func:`_assemble_radiometer_dataset`). The camera path keeps the radial stand-in
    until camera segmentation carries ephemeris; a weigher that needs a viewing frame
    raises ``ValueError`` on footprints without it.

    Coverage-based *acceptance* (the CERES 75%/95% discard/flag rule) is not applied
    here -- this function computes each footprint's coverage and, when
    ``return_coverage`` is set, returns it alongside the values, but turning coverage
    into QA flags (or discarding a footprint) is the caller's job (see
    :func:`_merge_coverage_qa`). Per-footprint coverage is the fraction of the PSF's
    in-contour energy backed by a usable sample, measured by the weigher on its
    reference grid (see :class:`~libera_utils.footprint_matching.weighting.WeightField`),
    taken as the **minimum** across the active sources so a footprint is scored no
    better than its worst-covered input.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode selecting the active readers and product variables.
    footprints : Sequence[Any]
        Footprints to aggregate over, each exposing a ``bbox`` (``None`` for an off-limb view), a ``time``
        (``np.datetime64``) and the geometry the weigher reads (``latitude``,
        ``longitude``, ``viewing_zenith_angle`` and, for the angular weigher, the
        ephemeris). Pass them in time order: a stepped source reloads its grid
        whenever the nearest step changes.
    sources : AncillarySources, optional
        Pre-built sources to query. If ``None``, they are constructed from
        ``source_file_paths`` via :func:`~libera_utils.footprint_matching.sources.build_sources`.
    source_file_paths : dict[str, Sequence[Path] | Path], optional
        Reader-key -> staged granule(s) map used to build the sources when none are
        supplied. Required in that case.
    weigher : PixelWeigher, optional
        The per-sample PSF weight provider. Defaults to a fresh
        :class:`~libera_utils.footprint_matching.weighting.RadialWeigher`.
    return_coverage : bool, optional
        When ``True``, also return a per-footprint coverage array. Defaults to
        ``False`` (return only the values dict).
    n_workers : int, optional
        Worker processes to run the footprints on. ``None`` (the default) reads the
        ``FMATCH_WORKERS`` configuration key (``config.json``, overridable through the
        environment variable of the same name); ``1`` runs in the calling process. Fewer
        than :data:`_MIN_FOOTPRINTS_PER_WORKER` footprints per worker also runs in the
        calling process, since the pool would cost more than it saves.

    Returns
    -------
    dict[str, np.ndarray] or tuple[dict[str, np.ndarray], np.ndarray]
        With ``return_coverage=False`` (default): the mapping of aggregated-variable
        name to a 1-D array indexed by footprint, carrying ``NaN`` where a footprint had
        no usable data. Float variables keep their reader dtype; categorical (integer)
        variables are returned as ``float64`` so they can carry ``NaN`` too, and the
        product's integer fill (see :func:`_fill_value_for`) replaces it when the column is
        written. With
        ``return_coverage=True``: a ``(values, coverage)`` tuple, where
        ``coverage`` is a 1-D float array in ``[0, 1]`` indexed by footprint (the
        minimum source coverage; ``0`` where the footprint had no covered energy).

    Raises
    ------
    ValueError
        If neither ``sources`` nor ``source_file_paths`` is supplied; if ``weigher``
        needs a viewing frame and a footprint carries no L1B ephemeris; if a footprint
        has no finite viewing zenith angle; if a footprint's time is NaT and a
        source selects by time; if ``n_workers`` is below one; or if the pool must be
        spawned (no fork on this platform) and ``source_file_paths`` was not given, so a
        worker cannot build its sources.
    """
    if sources is None:
        if source_file_paths is None:
            raise ValueError("aggregate_external_variables requires either sources or source_file_paths.")
        sources = build_sources(mode, source_file_paths)

    if weigher is None:
        weigher = RadialWeigher()

    footprints = list(footprints)
    n_footprints = len(footprints)

    if n_workers is None:
        n_workers = int(config.get(FMATCH_WORKERS_CONFIG_KEY))
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}.")
    if n_workers > 1 and n_footprints >= _MIN_FOOTPRINTS_PER_WORKER * n_workers:
        return _aggregate_in_parallel(mode, footprints, sources, source_file_paths, weigher, return_coverage, n_workers)
    return _aggregate_footprints(mode, footprints, sources, weigher, return_coverage)


def _aggregate_footprints(
    mode: OperationalMode,
    footprints: list[Any],
    sources: AncillarySources,
    weigher: PixelWeigher,
    return_coverage: bool,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Run the per-footprint loop of :func:`aggregate_external_variables` in this process."""
    n_footprints = len(footprints)

    # Extract each on-Earth footprint's viewing geometry once (attribute reads, cheap)
    # rather than inside the per-source loop. Off-limb views are skipped below and carry
    # no box and, in L1B, no finite viewing zenith, so they get no geometry. Build every
    # on-Earth footprint's viewing frame in one batched call; the frame is the dominant
    # weighting cost.
    on_earth = [i for i, footprint in enumerate(footprints) if not getattr(footprint, "off_limb", False)]
    geoms = {i: _footprint_geometry(footprints[i]) for i in on_earth}
    frames: ViewingFrame | None = None
    frame_rows: dict[int, int] = {}
    if weigher.uses_viewing_frame:
        if not all(geom.has_ephemeris for geom in geoms.values()):
            raise ValueError(
                f"{type(weigher).__name__} needs each footprint's L1B ephemeris (subsatellite point and "
                "satellite radius); camera pseudo-footprints carry none, so weight them with RadialWeigher."
            )
        if on_earth:
            frames = _batch_viewing_frames([geoms[i] for i in on_earth])
            frame_rows = {i: row for row, i in enumerate(on_earth)}

    # Drive the loop off the sources' *active* readers rather than re-deriving the
    # registry set for the mode: the sources are constructed with only the readers
    # active for their mode, and this also lets a caller pass a subset of sources. Each
    # source's reader *class* (needed for product_variable_specs() and the INSTRUMENT
    # token) is resolved from the registry by key.
    reader_classes = {key: ReaderRegistry.get(key) for key in sources.sources}

    # Pre-allocate one NaN-filled output array per product variable. Integer
    # (categorical) variables are held as float64 so a footprint with no usable data
    # stays NaN; the integer fill from _fill_value_for replaces it when the column is placed.
    # Cache the per-reader spec name plan so we do not recompute it for every footprint.
    outputs: dict[str, np.ndarray] = {}
    plan: dict[str, list[tuple[str, str]]] = {}
    for key, reader_cls in reader_classes.items():
        entries: list[tuple[str, str]] = []
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
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype(np.float64)
            outputs[product_name] = np.full(n_footprints, np.nan, dtype=dtype)
            entries.append((product_name, spec.name))
        plan[key] = entries

    # Per-footprint coverage accumulator: track the minimum across sources. Start at
    # +inf so the first source sets it; a footprint with no covered energy ends at 0.
    coverage_min = np.full(n_footprints, np.inf, dtype=float)

    for i, footprint in enumerate(footprints):
        if getattr(footprint, "off_limb", False):
            # Space/calibration view with no Earth footprint: leave every external
            # variable at its fill sentinel and force zero coverage (coverage_min[i]
            # stays +inf, mapped to 0.0 below). An off-limb record has no box to query,
            # and aggregating anything for it would hand a space record valid-looking
            # external values and nonzero coverage.
            continue

        geom = geoms[i]
        frame = frames.for_index(frame_rows[i]) if frames is not None else None
        for key, reader_cls in reader_classes.items():
            # The source's native samples inside this footprint's PSF bounding box,
            # matched to the footprint time by the reader's TimeMatch policy.
            samples = sources.query(key, footprint.bbox, footprint.time)
            weight_field = weigher.weight_field(
                samples,
                geom.boresight_lat_deg,
                geom.boresight_lon_deg,
                subsatellite_lat_deg=geom.subsatellite_lat_deg,
                subsatellite_lon_deg=geom.subsatellite_lon_deg,
                satellite_radius_km=geom.satellite_radius_km,
                sample_resolution_km=samples.resolution_km,
                viewing_zenith_deg=geom.viewing_zenith_deg,
                cone_angle_rate=geom.cone_angle_rate,
                frame=frame,
            )
            if return_coverage:
                coverage_min[i] = min(coverage_min[i], weight_field.coverage)
            # One scalar per product variable, keyed by the bare spec name.
            scalars = aggregate_sample_variables(reader_cls, samples, weight_field)
            for product_name, spec_name in plan[key]:
                outputs[product_name][i] = scalars[spec_name]

    if not return_coverage:
        return outputs
    # +inf survives only for footprints that saw no source at all (empty sources) or
    # were off-limb; score those as zero coverage.
    coverage = np.where(np.isinf(coverage_min), 0.0, coverage_min)
    return outputs, coverage


def _source_paths_key(
    mode: OperationalMode, source_file_paths: dict[str, Sequence[Path] | Path]
) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """A hashable identity of a run's staged inputs, for the spawned workers' source cache."""
    entries = []
    for key in sorted(source_file_paths):
        paths = source_file_paths[key]
        if isinstance(paths, (str, Path)):
            paths = [paths]
        entries.append((key, tuple(str(p) for p in paths)))
    return mode.value, tuple(entries)


def _worker_sources(
    mode: OperationalMode, source_file_paths: dict[str, Sequence[Path] | Path] | None
) -> AncillarySources:
    """The sources a worker process aggregates with.

    A forked worker inherits the parent's loaded sources through :data:`_FORK_SOURCES`
    (copy-on-write, no reload). A spawned worker builds its own from the staged paths
    once and keeps them in :data:`_WORKER_SOURCES_CACHE` for the chunks that follow.
    """
    if _FORK_SOURCES is not None:
        return _FORK_SOURCES
    if source_file_paths is None:
        raise ValueError(
            "A spawned worker has no sources to aggregate with: pass source_file_paths to "
            "aggregate_external_variables when n_workers > 1 on a platform without fork."
        )
    key = _source_paths_key(mode, source_file_paths)
    sources = _WORKER_SOURCES_CACHE.get(key)
    if sources is None:
        _WORKER_SOURCES_CACHE.clear()
        sources = build_sources(mode, source_file_paths)
        _WORKER_SOURCES_CACHE[key] = sources
    return sources


def _aggregate_chunk(
    mode: OperationalMode,
    footprints: list[Any],
    source_file_paths: dict[str, Sequence[Path] | Path] | None,
    weigher: PixelWeigher,
    return_coverage: bool,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Worker entry point: aggregate one contiguous chunk of footprints."""
    sources = _worker_sources(mode, source_file_paths)
    return _aggregate_footprints(mode, footprints, sources, weigher, return_coverage)


def _aggregate_in_parallel(
    mode: OperationalMode,
    footprints: list[Any],
    sources: AncillarySources,
    source_file_paths: dict[str, Sequence[Path] | Path] | None,
    weigher: PixelWeigher,
    return_coverage: bool,
    n_workers: int,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Run :func:`_aggregate_footprints` over contiguous chunks on a process pool.

    The footprints are cut into ``_CHUNKS_PER_WORKER * n_workers`` contiguous ranges, in
    order, so each chunk is a time-ordered along-track segment. Where the platform forks,
    the pool is forked after every source is loaded and the workers read the parent's
    sources through copy-on-write memory; otherwise the workers are spawned and each
    builds its own sources from ``source_file_paths``. Chunk results come back in order
    and are concatenated, so the output is exactly the single-process output.
    """
    global _FORK_SOURCES  # noqa: PLW0603 - the pool's workers inherit the loaded sources through this name.

    n_chunks = min(len(footprints), _CHUNKS_PER_WORKER * n_workers)
    bounds = np.linspace(0, len(footprints), n_chunks + 1).astype(int)
    chunks = [footprints[start:stop] for start, stop in zip(bounds[:-1], bounds[1:], strict=True) if stop > start]

    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    if start_method == "spawn" and source_file_paths is None:
        raise ValueError(
            "n_workers > 1 needs source_file_paths on a platform without fork, so each worker can build its "
            "own sources."
        )
    context = multiprocessing.get_context(start_method)
    if start_method == "fork":
        # Load every static source now, so the workers inherit the indexed data rather than
        # each loading it; a stepped source loads its first step on its first query.
        for key in sources.sources:
            reader = sources.reader(key)
            if hasattr(reader, "_native"):
                _ = reader._native  # noqa: SLF001 - warming the reader's own cache before the fork.
        _FORK_SOURCES = sources
    try:
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=context) as pool:
            results = list(
                pool.map(
                    _aggregate_chunk,
                    *zip(
                        *[(mode, chunk, source_file_paths, weigher, return_coverage) for chunk in chunks], strict=True
                    ),
                )
            )
    finally:
        _FORK_SOURCES = None

    if return_coverage:
        values = [r[0] for r in results]
        coverage = np.concatenate([r[1] for r in results])
    else:
        values = results
    merged = {name: np.concatenate([chunk[name] for chunk in values]) for name in values[0]}
    return (merged, coverage) if return_coverage else merged


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
    aggregation path needs: the **true ray-traced footprint** from
    :func:`~libera_utils.footprint_matching.geometry.compute_footprint_bounding_boxes`
    (boresight placed from the satellite at the subsatellite point and
    ``Radius_of_Satellite_from_Center_of_Earth``, projected onto the WGS84 ellipsoid and
    limb-truncated). The ephemeris and ``cone_angle_rate`` are carried onto the footprint
    so the CERES-faithful ``AngularPSFWeigher`` can orient the PSF along the real scan
    plane.

    Exactly one footprint is produced per input record, in input order, so the list stays
    index-aligned with the product columns: an off-Earth view (a space/calibration view
    that slipped past the reader's finite filter) keeps its index with ``bbox=None`` and
    ``off_limb=True``, so aggregation leaves it at fill rather than querying and
    weighting a region the instrument did not see.

    Parameters
    ----------
    l1b_inputs : dict[str, np.ndarray]
        The arrays from
        :func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`
        (already filtered to finite geolocation). ``RADIOMETER_TIME``, ``latitude``,
        ``longitude``, ``viewing_zenith_angle`` and every
        :data:`L1B_SCAN_REFERENCE_VARIABLES` key are required;
        :data:`FMATCH_CONE_ANGLE_RATE_KEY` is read when present.

    Returns
    -------
    list[RadiometerFootprint]
        One footprint per L1B record, in input order.

    Raises
    ------
    ValueError
        If a required key is missing.
    MissingGeolocationError
        If any record's ephemeris is unusable or its footprint is not visible from the
        satellite (see :func:`~libera_utils.footprint_matching.geometry.compute_footprint_bounding_boxes`).
    """
    required = (RADIOMETER_TIME_KEY, "latitude", "longitude", "viewing_zenith_angle", *L1B_SCAN_REFERENCE_VARIABLES)
    missing = [name for name in required if name not in l1b_inputs]
    if missing:
        raise ValueError(f"L1B inputs are missing the key(s) radiometer footprints need: {', '.join(missing)}")

    times = np.asarray(l1b_inputs[RADIOMETER_TIME_KEY]).astype("datetime64[ns]")
    latitudes = np.asarray(l1b_inputs["latitude"], dtype=float)
    longitudes = np.asarray(l1b_inputs["longitude"], dtype=float)
    viewing_zeniths = np.asarray(l1b_inputs["viewing_zenith_angle"], dtype=float)
    subsatellite_lats = np.asarray(l1b_inputs["subsatellite_latitude"], dtype=float)
    subsatellite_lons = np.asarray(l1b_inputs["subsatellite_longitude"], dtype=float)
    satellite_radii = np.asarray(l1b_inputs["satellite_radius_km"], dtype=float)
    cone_rates = np.asarray(l1b_inputs.get(FMATCH_CONE_ANGLE_RATE_KEY, np.full(latitudes.shape, np.nan)), dtype=float)

    # Every footprint's ray-traced box in one batched call. A None result marks an
    # off-Earth view.
    boxes = compute_footprint_bounding_boxes(
        latitudes,
        longitudes,
        subsatellite_lats,
        subsatellite_lons,
        satellite_radius_km=satellite_radii,
    )
    footprints: list[RadiometerFootprint] = []
    for i, bbox in enumerate(boxes):
        lat = float(latitudes[i])
        lon = float(longitudes[i])
        off_limb = bbox is None
        cone_rate = float(cone_rates[i])
        footprints.append(
            RadiometerFootprint(
                bbox=bbox,
                time=times[i],
                latitude=lat,
                longitude=lon,
                viewing_zenith_angle=float(viewing_zeniths[i]),
                subsatellite_latitude=float(subsatellite_lats[i]),
                subsatellite_longitude=float(subsatellite_lons[i]),
                satellite_radius_km=float(satellite_radii[i]),
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
    source_file_paths: dict[str, Sequence[Path] | Path] | None,
    sources: AncillarySources | None,
    weigher: PixelWeigher | None,
    allow_placeholders: bool = False,
    to_grid: Callable[[Any, np.dtype, Any], np.ndarray] | None = None,
) -> None:
    """Merge the *computed* (non-input) product columns into ``data`` in place.

    Fills the two families of variable the footprint-matching engine owns, so they
    carry real values instead of placeholders:

    * **Derived viewing geometry** (``sunglint_angle``) -- always computed, from the
      real per-footprint geolocation angles (:func:`compute_derived_viewing_geometry`).
    * **External aggregated variables** and the **coverage/QA** columns
      (``psf_coverage_fraction``, ``q_flags``) -- computed from the ancillary inputs
      (``sources`` or ``source_file_paths``). Without them this raises, unless
      the caller passes ``allow_placeholders=True``, in which case those columns are
      left for :func:`_fill_placeholder_variables`.

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
    source_file_paths : dict[str, Sequence[Path] | Path] or None
        Reader-key -> staged granule(s) map used to build the sources when ``sources`` is not
        given.
    sources : AncillarySources or None
        A pre-built manager to aggregate from. Takes precedence over
        ``source_file_paths``.
    weigher : PixelWeigher or None
        PSF weigher for the aggregation (defaults to the radial stand-in downstream).
    allow_placeholders : bool, optional
        When ``True`` and neither ``sources`` nor ``source_file_paths`` is given,
        skip the external aggregation and leave those columns as placeholders. Defaults
        to ``False``.
    to_grid : callable, optional
        Scatter ``(values, dtype, fill_value) -> ndarray`` mapping a flat per-footprint column onto the 2-D
        ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid, supplied by the camera-timescale assembly. When ``None`` (the
        radiometer-timescale products) each computed column is stored 1-D as-is.

    Raises
    ------
    ValueError
        If neither ``sources`` nor ``source_file_paths`` is given and
        ``allow_placeholders`` is ``False``.
    """

    # The engine computes one value per footprint (flat). For the camera-timescale products the caller passes a
    # ``to_grid`` scatter so each computed column lands on the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid (padded cells taking
    # the declared fill); the radiometer products keep the 1-D column as-is. Integer columns arrive as float with NaN
    # where a footprint had no usable data, and take the integer fill there.
    def _place(name: str, values: np.ndarray) -> np.ndarray:
        dtype, fill_value = _fill_value_for(definition.variables[name])
        if np.issubdtype(dtype, np.integer):
            values = np.where(np.isnan(values), fill_value, values)
        return np.asarray(values, dtype=dtype) if to_grid is None else to_grid(values, dtype, fill_value)

    # Derived geometry is always computable from the real angle columns.
    if "sunglint_angle" in definition.variables and "sunglint_angle" not in data:
        sunglint = compute_derived_viewing_geometry(solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle)[
            "sunglint_angle"
        ]
        data["sunglint_angle"] = _place("sunglint_angle", sunglint)

    # External aggregation needs staged ancillary data. Without it the product would carry
    # placeholder external columns and an unflagged q_flags, so only an explicit opt-in
    # leaves them as placeholders.
    if sources is None and source_file_paths is None:
        if allow_placeholders:
            return
        raise ValueError(
            f"No ancillary inputs were supplied for {mode.value}, so its external variables, psf_coverage_fraction "
            "and q_flags cannot be computed. Pass sources or source_file_paths, or pass "
            "allow_placeholders=True to write those columns as fill values."
        )

    aggregated, coverage = aggregate_external_variables(
        mode,
        footprints,
        sources,
        source_file_paths=source_file_paths,
        weigher=weigher,
        return_coverage=True,
    )
    for name, values in aggregated.items():
        if name in definition.variables:
            data[name] = _place(name, values)
    _merge_coverage_qa(data, definition, coverage, footprints, to_grid=to_grid)


def _merge_coverage_qa(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    coverage: np.ndarray,
    footprints: Sequence[Any],
    to_grid: Callable[[Any, np.dtype, Any], np.ndarray] | None = None,
) -> None:
    """Fill the coverage fraction and OR the CERES coverage rule into ``q_flags``.

    Coverage sets QA flags rather than discarding the footprint, following the CERES
    75%/95% rule: coverage in ``[0.75, 0.95)`` sets ``PARTIAL_COVERAGE``; coverage
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
        Footprints, read only for each ``off_limb`` flag and, where ``bbox`` is not
        ``None``, its ``truncated`` flag.
    """
    if "psf_coverage_fraction" in definition.variables:
        dtype, fill_value = _fill_value_for(definition.variables["psf_coverage_fraction"])
        data["psf_coverage_fraction"] = (
            np.asarray(coverage, dtype=dtype) if to_grid is None else to_grid(coverage, dtype, fill_value)
        )

    if "q_flags" not in definition.variables:
        return
    coverage = np.asarray(coverage, dtype=float)
    qbits = np.zeros(coverage.shape, dtype=np.int64)
    partial = (coverage >= ACCEPT_COVERAGE_THRESHOLD) & (coverage < PARTIAL_COVERAGE_THRESHOLD)
    insufficient = coverage < ACCEPT_COVERAGE_THRESHOLD
    truncated = np.array([f.bbox is not None and f.bbox.truncated for f in footprints], dtype=bool)
    off_limb = np.array([bool(getattr(f, "off_limb", False)) for f in footprints], dtype=bool)
    qbits[partial] |= int(FmatchCoverageFlag.PARTIAL_COVERAGE)
    qbits[insufficient] |= int(FmatchCoverageFlag.INSUFFICIENT_COVERAGE)
    qbits[truncated] |= int(FmatchCoverageFlag.LIMB_TRUNCATED)
    qbits[off_limb] |= int(FmatchCoverageFlag.OFF_LIMB)
    # Scatter the per-footprint coverage bits onto the record grid (padded cells carry no flags) before OR-ing them
    # into any q_flags already assembled (e.g. the camera segmentation flags, which the camera path sets first).
    if to_grid is not None:
        qbits = to_grid(qbits, np.dtype(np.int64), 0)
    existing = data.get("q_flags")
    if existing is not None:
        qbits |= np.asarray(existing, dtype=np.int64)
    data["q_flags"] = qbits.astype(np.dtype(definition.variables["q_flags"].dtype))


def assemble_fmatch_dataset(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    *,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path] | Path] | None = None,
    sources: AncillarySources | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
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
      the camera pseudo-footprints segmented by the Camera Cloud Fraction algorithm
      and read from its CF-CAM-CAMTIME product.
      ``inputs`` is that sequence of :class:`PseudoFootprint` objects. See
      :func:`_assemble_camtime_dataset`.
    * **Radiometer-timescale** (``CAM``, ``IMAGER_FLASH``, ``IMAGER``) - assembled
      from the L1B pass-through arrays produced by
      :func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`.
      ``inputs`` is that dict. See :func:`_assemble_radiometer_dataset`.

    In both cases the derived viewing geometry (``sunglint_angle``) is always computed
    from the real geolocation angles. The external aggregated variables and the
    coverage/QA columns are computed from the ancillary inputs (``sources`` or
    ``source_file_paths``, forwarded to :func:`aggregate_external_variables`). Without
    them assembly raises, unless ``allow_placeholders=True`` asks for those columns to
    be written as conformant placeholders.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being assembled.
    inputs : Sequence[PseudoFootprint] or dict[str, np.ndarray]
        Camera pseudo-footprints for the camera-timescale modes, or the L1B
        pass-through dict for the radiometer-timescale modes.
    definition : LiberaDataProductDefinition, optional
        The product definition. Loaded via :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the dynamic ``algorithm_version`` global attribute. Writing the
        product requires it; assembly alone does not.
    input_files : str, optional
        Provenance string for the dynamic ``InputGranules`` global attribute. Writing
        the product requires it; assembly alone does not.
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint cloud fraction from the Camera Cloud Fraction (CF-CAM)
        algorithm (Libera WFOV camera), as a 1-D array indexed by footprint in
        the same order as the time coordinate. This is an *internal* algorithm
        output - it does not come from a reader and is already aggregated to one
        value per footprint - so it is merged directly into the ``cloud_fraction_camera``
        variable rather than going through :func:`aggregate_external_variables`.
        Only the CAM modes (``CAM``, ``CAM_CAMTIME``) declare this variable; it is
        ``None`` for the IMAGER modes.
    source_file_paths : dict[str, Sequence[Path] | Path], optional
        Reader-key -> staged ancillary granule(s) map, used to build the sources when
        ``sources`` is not given.
    sources : AncillarySources, optional
        A pre-built manager to aggregate from. Takes precedence over
        ``source_file_paths``.
    weigher : PixelWeigher, optional
        PSF weigher for the aggregation. The radiometer-timescale modes default to
        :class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher`.
    allow_placeholders : bool, optional
        When ``True``, a call with neither ``sources`` nor ``source_file_paths``
        writes the external, ``psf_coverage_fraction`` and ``q_flags`` columns as
        placeholders instead of raising. Defaults to ``False``.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's product definition.

    Raises
    ------
    ValueError
        If ``inputs`` is empty; if the radiometer-timescale ``inputs`` dict is missing
        a required pass-through key or its arrays have inconsistent lengths; if neither
        ``sources`` nor ``source_file_paths`` is given and ``allow_placeholders``
        is ``False``; if a footprint has no finite viewing zenith angle; if ``weigher``
        needs a viewing frame and a footprint carries no L1B ephemeris; if a weight
        time is NaT and a source selects by time.
    MissingGeolocationError
        If a radiometer record's ephemeris is unusable or its footprint is not visible
        from the satellite (see :func:`build_radiometer_footprints`).
    """
    assembler = _assemble_camtime_dataset if mode in _CAMERA_TIMESCALE_MODES else _assemble_radiometer_dataset
    return assembler(
        inputs,
        mode=mode,
        definition=definition,
        algorithm_version=algorithm_version,
        input_files=input_files,
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=source_file_paths,
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )


def _fill_value_for(variable_definition: Any) -> tuple[np.dtype, Any]:
    """Return the ``(dtype, fill_value)`` a variable or coordinate uses for absent cells.

    The fill is the definition's declared ``_FillValue`` when present, and otherwise ``NaN``
    for floating-point definitions. An integer bitmask (a definition declaring
    ``flag_masks``, such as ``q_flags``) takes ``0``, meaning no flag set. Any other integer
    takes :data:`FMATCH_INTEGER_FILL_VALUE`, because 0 is a real category or pixel index there.
    Used both for not-yet-computed placeholder variables and for the padded cells of the
    camera-timescale grid (images segmented into fewer subsections than the widest image).

    Parameters
    ----------
    variable_definition : LiberaVariableDefinition
        The product-definition entry for the variable or coordinate.

    Returns
    -------
    tuple[numpy.dtype, Any]
        The declared dtype and the fill value to use for missing/padded cells.
    """
    dtype = np.dtype(variable_definition.dtype)
    fill_value = variable_definition.attributes.get("_FillValue")
    if fill_value is not None:
        return dtype, fill_value
    if np.issubdtype(dtype, np.floating):
        return dtype, np.nan
    if "flag_masks" in variable_definition.attributes:
        return dtype, 0
    return dtype, FMATCH_INTEGER_FILL_VALUE


def _placeholder_variable_array(variable_definition: Any, shape: int | tuple[int, ...]) -> np.ndarray:
    """Build a conformant placeholder array for a declared variable no input produced.

    A declared variable still has to appear in the output file with the right dtype
    and shape so the product conforms to its definition, even when no input produced
    it (no ancillary data staged under ``allow_placeholders=True``, or a source such as
    FLASHFlux that has no reader). We fill it with the fill :func:`_fill_value_for`
    returns. The magnitudes are meaningless; only the dtype/shape/attributes form the
    product contract (the same stance the example-product generator takes in
    ``notebooks/generate_example_products.ipynb``).

    Parameters
    ----------
    variable_definition : LiberaVariableDefinition
        The product-definition entry for the variable.
    shape : int or tuple of int
        Shape of the placeholder array: the length of the 1-D record axis for the
        radiometer-timescale products, or the ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid shape
        for the camera-timescale products.

    Returns
    -------
    np.ndarray
        An array of the given ``shape`` and the variable's declared dtype.
    """
    dtype, fill_value = _fill_value_for(variable_definition)
    return np.full(shape, fill_value, dtype=dtype)


def _fill_placeholder_variables(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    shape: int | tuple[int, ...],
) -> None:
    """Fill every declared variable missing from ``data`` with a conformant placeholder.

    Mutates ``data`` in place, adding one array per declared variable no input
    produced: the external columns when no ancillary data was staged and the caller
    passed ``allow_placeholders=True``, or a source such as FLASHFlux that has no
    reader. The product definition requires every declared variable to be present, so
    this is what lets such a product conform. The placeholder values are structurally
    valid (declared dtype/shape/attributes) but numerically meaningless.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        The variable arrays assembled so far (the real, input-derived columns).
    definition : LiberaDataProductDefinition
        The product definition naming every variable the file must contain.
    shape : int or tuple of int
        Shape each placeholder array is built at: the 1-D record-axis length for the
        radiometer-timescale products, or the ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid shape for
        the camera-timescale products.
    """
    for name, variable_definition in definition.variables.items():
        if name not in data:
            data[name] = _placeholder_variable_array(variable_definition, shape)


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
        Provenance string for the required dynamic ``InputGranules`` global attribute.

    Returns
    -------
    xarray.Dataset
        The conformance-enforced dataset, ready to write.
    """
    dataset = definition.create_product_dataset(data)
    dataset = definition.enforce_dataset_conformance(dataset)
    if input_files is not None:
        dataset.attrs["InputGranules"] = input_files
    if algorithm_version is not None:
        dataset.attrs["algorithm_version"] = algorithm_version
    return dataset


def _normalize_longitude(longitude_deg: float) -> float:
    """Wrap a bounding-box longitude past the antimeridian back into [-180, 180].

    Corner-derived bounding boxes that straddle the antimeridian can report a
    ``lon_max`` greater than 180 (the
    :class:`~libera_utils.footprint_matching.types.BoundingBox` dateline convention).
    The product definition's ``psf_bbox_lon_*`` variables declare a [-180, 180]
    valid range, so only values above 180 are wrapped; 180 itself is kept, so a
    full-span ``(-180, 180)`` box stays full-span. Downstream consumers can still
    detect a dateline-crossing box because it then has ``lon_min > lon_max``.
    """
    return longitude_deg - 360.0 if longitude_deg > 180.0 else longitude_deg


def _assemble_camtime_dataset(
    footprints: Sequence[PseudoFootprint],
    *,
    mode: OperationalMode = OperationalMode.CAM_CAMTIME,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path] | Path] | None = None,
    sources: AncillarySources | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
) -> Dataset:
    """Assemble a camera-timescale FMATCH Dataset from camera pseudo-footprints.

    Serves both camera-timescale modes (``CAM_CAMTIME`` and ``IMAGER_CAMTIME``).
    They are built the same way - from the same L1B camera segmentation - and
    differ only in the set of aggregated variables their definitions declare,
    which the computed-and-placeholder fill handles generically.

    Builds the per-footprint variable arrays declared by the mode's product
    definition. The center-pixel geolocation/geometry, the corner-derived PSF
    bounding box, and the QA flags come straight from the pseudo-footprints
    (:data:`_CAMTIME_SEGMENTATION_VARIABLES`). The derived ``sunglint_angle`` is
    computed from the geolocation angles, and (when ``source_file_paths`` or
    ``sources`` is supplied) the external aggregated variables and coverage/QA
    columns are computed via :func:`_merge_computed_variables`; any remaining declared
    variable is a conformant placeholder.

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in write order, as read from CF-CAM-CAMTIME.
    mode : OperationalMode, optional
        Which camera-timescale mode to assemble. Defaults to ``CAM_CAMTIME``.
    definition : LiberaDataProductDefinition, optional
        The product definition. Loaded via :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``InputGranules`` global attribute
        (typically the source L1B camera filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). Only the
        CAM modes declare this variable; for ``IMAGER_CAMTIME`` it is ignored. When
        omitted (or undeclared) the variable is written as a placeholder.
    source_file_paths, sources, weigher, allow_placeholders
        Aggregation inputs, as in :func:`assemble_fmatch_dataset`.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's definition.

    Raises
    ------
    ValueError
        If ``mode`` is not a camera-timescale mode, or ``footprints`` is empty
        (there would be no time axis to write), or as :func:`_merge_computed_variables`
        raises.
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

    # Recover the rectangular (CAMERA_TIME, PSEUDOFOOTPRINT) grid from the flat footprint list.
    # Segmentation is ragged (each image is tiled into a variable number of subsections), but
    # the product grid is rectangular: group footprints by image (unique sorted CAMERA_TIME on
    # axis 0; subsections in segmentation order on axis 1). Images narrower than the widest are
    # padded along PSEUDOFOOTPRINT with each variable's fill value (NaN, or the integer fill from _fill_value_for), so
    # those cells carry no real data and NaN-skipping classification leaves them unmatched.
    unique_times = sorted({f.time for f in footprints})
    n_camera_times = len(unique_times)
    row_of_time = {time: row for row, time in enumerate(unique_times)}
    rows = np.array([row_of_time[f.time] for f in footprints])
    # Column = running position of each footprint within its own image (footprints arrive in
    # segmentation order), so subsections fill their image row left-to-right.
    columns = np.empty(n_footprints, dtype=int)
    footprints_in_row = np.zeros(n_camera_times, dtype=int)
    for flat_index, row in enumerate(rows):
        columns[flat_index] = footprints_in_row[row]
        footprints_in_row[row] += 1
    n_footprints_per_image = int(footprints_in_row.max())
    grid_shape = (n_camera_times, n_footprints_per_image)

    def to_grid(values: Sequence[Any], dtype: np.dtype, fill_value: Any) -> np.ndarray:
        """Scatter one flat per-footprint column into the rectangular (CAMERA_TIME, PSEUDOFOOTPRINT) grid."""
        grid = np.full(grid_shape, fill_value, dtype=dtype)
        grid[rows, columns] = np.asarray(values, dtype=dtype)
        return grid

    # The real, segmentation-derived columns. Longitudes of the PSF box are wrapped into
    # [-180, 180] to satisfy the product definition's valid range. center_pixel_x/y are the
    # boresight stand-in pixel; they are FMATCH-only provenance (the block's inclusive extent
    # is emitted as the camera_pixel_{x,y}_{min,max} coordinates below).
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

    # Grid coordinates: CAMERA_TIME is the 1-D image-acquisition axis (one unique, sorted entry per image);
    # PSEUDOFOOTPRINT is the 0-based subsection index within each image, generated here. create_product_dataset routes
    # both to .coords because the definition declares them under coordinates:.
    footprint_dtype = np.dtype(definition.coordinates["PSEUDOFOOTPRINT"].dtype)
    data: dict[str, np.ndarray] = {
        time_variable: np.array(unique_times, dtype="datetime64[ns]"),
        "PSEUDOFOOTPRINT": np.arange(n_footprints_per_image, dtype=footprint_dtype),
    }

    # Camera pixel-block provenance as four separate inclusive-bound coordinates on the 2-D
    # grid. slice_x/slice_y are half-open [start, stop), so the inclusive maximum is stop - 1.
    # Both camtime products declare these and pass them straight through to SCENE-ID-CAM-CAMTIME.
    pixel_columns: dict[str, list[int]] = {
        "camera_pixel_x_min": [f.slice_x.start for f in footprints],
        "camera_pixel_x_max": [f.slice_x.stop - 1 for f in footprints],
        "camera_pixel_y_min": [f.slice_y.start for f in footprints],
        "camera_pixel_y_max": [f.slice_y.stop - 1 for f in footprints],
    }
    for coordinate_name, values in pixel_columns.items():
        dtype, fill_value = _fill_value_for(definition.coordinates[coordinate_name])
        data[coordinate_name] = to_grid(values, dtype, fill_value)

    # Cast each real column to the exact dtype the definition declares and scatter it into the
    # grid. Only columns the definition actually declares are written; both camtime products
    # declare the same segmentation variables, but the guard keeps the assembly robust to
    # definition drift.
    for name, values in real_columns.items():
        variable_definition = definition.variables.get(name)
        if variable_definition is None:
            continue
        dtype, fill_value = _fill_value_for(variable_definition)
        data[name] = to_grid(values, dtype, fill_value)

    # Optional internal (non-reader) Camera Cloud Fraction values, scattered into the same
    # grid. Guarded on the definition declaring the variable, because IMAGER-CAMTIME does not.
    _merge_cloud_fraction_camera(data, definition, cloud_fraction_camera, to_grid=to_grid)

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
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
        to_grid=to_grid,
    )

    # Every remaining declared variable no input produced is filled with a placeholder.
    _fill_placeholder_variables(data, definition, grid_shape)

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
    to_grid: Callable[[Any, np.dtype, Any], np.ndarray] | None = None,
) -> None:
    """Merge the CF-CAM cloud fraction into ``data`` when it is supplied and declared.

    ``cloud_fraction_camera`` is an internal Libera algorithm output (from the WFOV
    Camera Cloud Fraction algorithm), already one value per footprint, so it bypasses
    the reader/aggregation path and is merged straight in. Only the CAM-family
    definitions declare it; passing values for an IMAGER mode is ignored rather than
    raising, so a caller can hand the same inputs to any mode.

    For the radiometer-timescale products the values are stored on the 1-D record axis as
    given. For the camera-timescale products the caller passes ``to_grid`` so the flat
    per-footprint values are scattered into the rectangular ``(CAMERA_TIME, PSEUDOFOOTPRINT)``
    grid (padded cells taking the variable's fill).
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
    dtype, fill_value = _fill_value_for(variable_definition)
    if to_grid is None:
        data["cloud_fraction_camera"] = np.asarray(cloud_fraction_camera, dtype=dtype)
    else:
        data["cloud_fraction_camera"] = to_grid(cloud_fraction_camera, dtype, fill_value)


def _assemble_radiometer_dataset(
    l1b_inputs: dict[str, np.ndarray],
    *,
    mode: OperationalMode,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path] | Path] | None = None,
    sources: AncillarySources | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
) -> Dataset:
    """Assemble a radiometer-timescale FMATCH Dataset from L1B pass-through inputs.

    Serves the three radiometer-timescale modes (``CAM``, ``IMAGER_FLASH``,
    ``IMAGER``). Their footprints are the L1B radiometer footprints themselves, so
    the time coordinate and the geolocation/viewing-angle columns
    (:data:`_RADIOMETER_L1B_VARIABLES`) are carried through verbatim from L1B. The
    derived ``sunglint_angle`` is computed from those angles, and (when
    ``source_file_paths`` or ``sources`` is supplied) the external aggregated
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
        Provenance string for the required dynamic ``InputGranules`` global attribute
        (typically the source L1B radiometer filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV), in the same
        footprint order as the time coordinate. Only ``CAM`` declares this variable.
    source_file_paths, sources, weigher, allow_placeholders
        Aggregation inputs, as in :func:`assemble_fmatch_dataset`.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's definition.

    Raises
    ------
    ValueError
        If ``mode`` is a camera-timescale mode, if a required pass-through input is
        missing, if the inputs have inconsistent lengths, or if they are empty, or as
        :func:`_merge_computed_variables` raises.
    MissingGeolocationError
        If a record's ephemeris is unusable (see :func:`build_radiometer_footprints`).
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
        build_radiometer_footprints(l1b_inputs) if (sources is not None or source_file_paths is not None) else []
    )
    # The radiometer footprints carry the L1B ephemeris (subsatellite point + satellite
    # radius) and cone-angle rate, so default this path to the CERES-faithful
    # AngularPSFWeigher, which orients the PSF along the real scan plane. Callers can
    # still override via `weigher`.
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
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )

    # Every remaining declared variable no input produced is filled with a placeholder.
    _fill_placeholder_variables(data, definition, n_footprints)

    return _finalize_product_dataset(
        definition,
        data,
        algorithm_version=algorithm_version,
        input_files=input_files,
    )


def write_fmatch_product(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    output_path: str | Path,
    *,
    algorithm_version: str,
    input_files: str,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path] | Path] | None = None,
    sources: AncillarySources | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
    strict: bool = True,
) -> LiberaDataProductFilename:
    """Assemble and write one FMATCH NetCDF product for an operational mode.

    Loads the product definition once (so assembly and writing cannot disagree about
    it), assembles a conformant Dataset via :func:`assemble_fmatch_dataset`, and
    writes it with ``libera_utils.io.netcdf.write_libera_data_product`` using
    ``time_variable=fmatch_time_variable(mode)`` (``RADIOMETER_TIME`` or
    ``CAMERA_TIME``), so the standardized Libera filename encodes the footprint time
    span.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    inputs : Sequence[PseudoFootprint] or dict[str, np.ndarray]
        The mode's assembly inputs: camera pseudo-footprints for the camera-timescale
        modes, or the L1B pass-through dict for the radiometer-timescale modes.
    output_path : str or pathlib.Path
        Directory (or S3 prefix) to write the product file into.
    algorithm_version : str
        Value for the ``algorithm_version`` global attribute.
    input_files : str
        Provenance string for the ``InputGranules`` global attribute.
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). Only the
        CAM modes declare this variable.
    source_file_paths : dict[str, Sequence[Path] | Path], optional
        Reader-key -> staged ancillary granule(s) map, used to build the sources when
        ``sources`` is not given.
    sources : AncillarySources, optional
        A pre-built manager to aggregate from. Takes precedence over
        ``source_file_paths``.
    weigher : PixelWeigher, optional
        PSF weigher used by the aggregation path (defaults to the radial stand-in).
    allow_placeholders : bool, optional
        When ``True``, a call with neither ``sources`` nor ``source_file_paths``
        writes the external, ``psf_coverage_fraction`` and ``q_flags`` columns as
        placeholders instead of raising. Defaults to ``False``.
    strict : bool, optional
        When True (default), fail if the assembled Dataset does not conform.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.

    Raises
    ------
    ValueError
        As :func:`assemble_fmatch_dataset` raises (empty or inconsistent inputs, no
        ancillary inputs without ``allow_placeholders``, a footprint with no viewing
        zenith angle, a viewing-frame weigher without ephemeris, a weight field that does
        time is NaT and a source selects by time), or, with ``strict``, if the assembled Dataset does not conform to
        the product definition.
    MissingGeolocationError
        If a radiometer record's ephemeris is unusable (see
        :func:`build_radiometer_footprints`).
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
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )
    return write_libera_data_product(
        data_product_definition=definition,
        data=dataset,
        output_path=output_path,
        time_variable=fmatch_time_variable(mode),
        strict=strict,
    )
