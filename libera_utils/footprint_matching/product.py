"""FMATCH data product assembly and writing.

This module is the seam between the footprint-matching *engine* (readers, PSF
aggregation, geometry) and the Libera *data-product* machinery
(``LiberaDataProductDefinition`` / ``write_libera_data_product``). It owns the
FMATCH product definitions and the flow that turns matched footprints into a
conformant NetCDF file, for all four operational modes.

What the product carries
------------------------
Assembly and writing are wired for **every** operational mode, and each declared
variable is filled by one of three routes:

* **Input pass-through** - the geolocation and viewing angles come straight from
  the L1B Daily inputs (``L1B_PASSTHROUGH_VARIABLES`` for the radiometer modes) or
  from the camera pseudo-footprints (``real_columns`` in ``_assemble_camtime_dataset``).
  Each footprint's PSF bounding box (``PSF_Bbox_*``) comes from its ray-traced box.
* **Computed** - the derived viewing geometry (``Sunglint_Angle``, via
  :func:`compute_derived_viewing_geometry`) is always calculated, and the external
  aggregated variables (:func:`aggregate_external_variables`) plus the coverage/QA
  columns (``PSF_Coverage_Fraction`` / ``Quality_Flag``) are calculated whenever staged
  ancillary inputs are supplied - the PSF aggregation of each source's native samples.
* **Placeholders** - a declared variable no input produced, when the caller opts in
  with ``allow_placeholders=True``: the external columns when no ancillary data is
  staged, a reader's columns when its source is not supplied, and
  ``Cloud_Fraction_Camera`` when it is omitted for a CAM mode. Without the opt-in,
  assembly raises.

Variable names
--------------
Assembly fills each column under the name the product definition declares
(``Sunglint_Angle``, ``Quality_Flag``, ``<READER_KEY>_<spec_name>`` such as
``ERA5_Surface_Wind_U10``), so nothing is renamed on the way to the file. The L1B input
dict is keyed by the L1B RAD-4CH variable names (``Latitude``, ``Subsatellite_Latitude``,
``Cone_Angle_Rate``) plus the ``RADIOMETER_TIME`` coordinate; the pass-through names are
also the product names, since the L1B names follow CERES BDS.

A placeholder is *structurally* correct (declared dtype, shape and attributes)
but numerically meaningless. Floating-point placeholders are filled with ``NaN``
and integer placeholders with their declared ``_FillValue`` (``FMATCH_INTEGER_FILL_VALUE``,
``-1``), or ``0`` for a bitmask; see ``_fill_placeholder_variables``.

Why a thin seam here
--------------------
The product definitions (``libera_utils/data/product_definitions/fmatch_*.yml``)
are the contract every downstream consumer (Scene ID, Camera Cloud Fraction, ADM
binning) reads against. Keeping the loaders next to the writers means there is a
single place that knows how a FMATCH file is produced, while the reader plugins
stay decoupled from product I/O.

See Also
--------
libera_utils.footprint_matching.footprint_match_algorithm : Manifest-driven runners that call into this module.
"""

from __future__ import annotations

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
    stack_product_values,
)
from libera_utils.footprint_matching.geometry import (
    L1B_FILL_VALUE,
    compute_footprint_bounding_boxes,
    compute_viewing_frames,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.sources import build_sources, require_mode_reader_keys
from libera_utils.footprint_matching.types import (
    BoundingBox,
    CameraFootprintQualityFlag,
    FmatchCoverageFlag,
    OperationalMode,
    PseudoFootprint,
    RadiometerFootprint,
)
from libera_utils.footprint_matching.weighting import (
    AngularPSFWeigher,
    PixelWeigher,
)
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    from collections.abc import Callable, Sequence

    from xarray import Dataset

    from libera_utils.footprint_matching.geometry import ViewingFrame
    from libera_utils.footprint_matching.readers.base import AncillaryReader
    from libera_utils.footprint_matching.types import VariableSpec
    from libera_utils.io.filenaming import LiberaDataProductFilename

# L1B RAD-4CH variables FMATCH copies straight out of L1B rather than computing. They keep
# their L1B name in the product, which follows CERES BDS, so each is also the FMATCH product
# variable name and the key the L1B input dict carries it under.
L1B_PASSTHROUGH_VARIABLES: tuple[str, ...] = (
    "Latitude",
    "Longitude",
    "Altitude",
    "Solar_Zenith_Surface",
    "Viewing_Zenith_Surface",
    "Relative_Azimuth_Surface",
)

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
L1B_SCAN_REFERENCE_VARIABLES: tuple[str, ...] = (
    "Subsatellite_Latitude",
    "Subsatellite_Longitude",
    "Radius_of_Satellite_from_Center_of_Earth",
)

# The instrument cone-angle rate (degrees/second) sets the along-scan PSF orientation
# and flags the stationary-scanner case. Unlike the subsatellite point it is optional:
# a fill/NaN value simply means "unknown scan rate" and is normalized to None per
# footprint downstream, so it is read but deliberately kept out of the finite mask (it
# must not drop an otherwise-good Earth-viewing footprint).
L1B_CONE_ANGLE_RATE_VARIABLE: str = "Cone_Angle_Rate"
# Key of the radiometer observation time in the L1B input map (and the product's time
# coordinate for the radiometer-timescale modes). Each footprint carries it so the readers
# can match ancillary samples to it in time.
RADIOMETER_TIME_KEY: str = "RADIOMETER_TIME"
# Configuration key for the number of worker processes the external aggregation runs on.
# ``1`` (the default in ``config.json``) keeps everything in the calling process.
FMATCH_WORKERS_CONFIG_KEY: str = "FMATCH_WORKERS"
# Chunks per worker: footprints are split into this many contiguous ranges per worker so
# a slow chunk (a stepped source reloading, a dense swath) does not idle the others.
_CHUNKS_PER_WORKER: int = 4  # TODO[LIBSDC-928]: Define the best value of this variable
# Below this many footprints the pool's start-up costs more than it saves.
_MIN_FOOTPRINTS_PER_WORKER: int = 50

# The sources a forked worker inherits from its parent (copy-on-write): set by
# _aggregate_in_parallel right before the pool starts, cleared after. A worker started by
# spawn sees None here and builds its own sources from the staged paths (see
# _worker_sources).
_FORK_SOURCES: dict[str, AncillaryReader] | None = None
# Sources a spawned worker built, kept for the next chunk it is handed. Keyed by the mode
# and the staged paths so a worker reused for another run never serves stale readers.
_WORKER_SOURCES_CACHE: dict[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], dict[str, AncillaryReader]] = {}

# Product definition YAML filename for each FMATCH operational mode. Every mode
# has its own SSF-style product definition (the mode *is* the product), and the
# active reader set / variables differ by mode. Kept as one source of truth so
# callers and tests never hard-code filenames.
FMATCH_DEFINITION_FILENAMES: dict[OperationalMode, str] = {
    OperationalMode.CAM: "fmatch_cam.yml",
    OperationalMode.CAM_CAMTIME: "fmatch_cam_camtime.yml",
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
# meaning no flag set (see :func:`_fill_value_for`). Each such column declares this value as its
# _FillValue in the FMATCH product definitions. Public because readers of the FMATCH products
# (e.g. SCENE-ID) must recognize it.
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
    and parses it into a validated
    :class:`~libera_utils.io.product_definition.LiberaDataProductDefinition`.

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


def _footprint_geometry(footprint: RadiometerFootprint | PseudoFootprint) -> _FootprintGeometry:
    """Extract the viewing geometry a PSF weigher needs from a footprint.

    Every PSF weigher sizes the footprint from the satellite's position, so the
    footprint must carry the L1B ephemeris: the subsatellite point and the satellite
    radius. A :class:`RadiometerFootprint` carries it, read from L1B by
    :func:`build_radiometer_footprints`. A camera :class:`PseudoFootprint` does not yet,
    so it raises rather than being weighted from an assumed orbit.

    The L1B ``Cone_Angle_Rate`` fill value (``-999``) is normalized to ``None`` so a
    fill never masquerades as a real (near-zero) stationary-scanner rate.
    :func:`build_radiometer_footprints` already does this; the check here covers a
    :class:`RadiometerFootprint` constructed directly.

    Parameters
    ----------
    footprint : RadiometerFootprint or PseudoFootprint
        The footprint to weight. It must be on Earth: an off-limb footprint has no box.

    Returns
    -------
    _FootprintGeometry
        The boresight, viewing zenith angle, ephemeris and cone-angle rate.

    Raises
    ------
    ValueError
        If ``footprint`` is a :class:`PseudoFootprint`, which carries no L1B ephemeris to
        size its PSF from, or its ``viewing_zenith_angle`` input is not finite.
    TypeError
        If ``footprint`` is neither a :class:`RadiometerFootprint` nor a
        :class:`PseudoFootprint`.
    """
    if isinstance(footprint, PseudoFootprint):
        # TODO[LIBSDC-884]: read the camera record's ephemeris once camera segmentation carries it.
        raise ValueError(
            "Camera pseudo-footprints carry no L1B ephemeris (subsatellite point and satellite radius), so "
            "their PSF cannot be sized: aggregate the camera-timescale modes once camera segmentation carries "
            "the ephemeris, or assemble them with allow_placeholders=True and no ancillary inputs."
        )
    if not isinstance(footprint, RadiometerFootprint):
        raise TypeError(
            f"Expected a RadiometerFootprint or PseudoFootprint, got {type(footprint).__name__}: build the "
            "footprints with build_radiometer_footprints or camera segmentation."
        )
    viewing_zenith = footprint.viewing_zenith_angle
    if not np.isfinite(viewing_zenith):
        raise ValueError(
            f"The footprint at {footprint.bbox} has no finite viewing zenith angle (got {viewing_zenith}); the PSF "
            "weigher sizes the footprint from it, so drop the record or supply its L1B Viewing_Zenith_Surface value."
        )

    cone_rate = footprint.cone_angle_rate
    if cone_rate is not None and (cone_rate == L1B_FILL_VALUE or not np.isfinite(cone_rate)):
        cone_rate = None

    return _FootprintGeometry(
        boresight_lat_deg=float(footprint.latitude),
        boresight_lon_deg=float(footprint.longitude),
        viewing_zenith_deg=float(viewing_zenith),
        subsatellite_lat_deg=float(footprint.subsatellite_latitude),
        subsatellite_lon_deg=float(footprint.subsatellite_longitude),
        satellite_radius_km=float(footprint.satellite_radius_km),
        cone_angle_rate=None if cone_rate is None else float(cone_rate),
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
    footprints: Sequence[RadiometerFootprint | PseudoFootprint],
    sources: dict[str, AncillaryReader] | None = None,
    *,
    source_file_paths: dict[str, Sequence[Path]] | None = None,
    weigher: PixelWeigher | None = None,
    return_coverage: bool = False,
    n_workers: int | None = None,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Aggregate every active reader's native samples to one value per footprint.

    For the given operational mode this selects the active readers via
    ``ReaderRegistry.get_readers_for_mode(mode)``, and for each footprint asks each
    active reader (:meth:`~libera_utils.footprint_matching.readers.base.AncillaryReader.samples_in`)
    for the native samples inside the footprint's bounding box matched to its time,
    weights each sample with the PSF ``weigher`` at the sample's own location, and
    applies each variable's aggregation strategy (weighted mean / mode / log-mean /
    std / ranked scene) to collapse the samples to a single value per footprint. No
    sample is averaged, binned or interpolated before the weigher sees it. The active
    reader set - and therefore the keys of the returned dict - grows with the mode's
    latency (e.g. CAM has ERA5_Surface, IGBP, NISE, VIIRS_BRDF, VIIRS_CLDPROP_D3; IMAGER
    additionally has ERA5_Pressure, VIIRS_AOD, and the RBSP FLASHFlux and CLDPIX fields).
    Per-spec gating also applies (see below).

    Every output variable is named ``<READER_KEY>_<spec_name>`` (e.g.
    ``ERA5_Surface_Wind_U10``, ``IGBP_Surface_Type``, ``CLDPIX_Cloud_Mask``), the name the
    product definition declares. The reader's ``INSTRUMENT`` is recorded in each
    variable's ``long_name`` rather than in the name. A spec is aggregated only
    when ``mode`` is in its ``modes`` (e.g. the extended FLASHFlux
    cloud/aerosol/albedo fields are aggregated for FMATCH-IMAGER only).

    Processing is done one footprint at a time, so only the current footprint's
    samples are held beyond what each reader holds of its source. With ``n_workers``
    above one the footprints are split into contiguous along-track chunks and the chunks
    run in a process pool; the result is identical to the single-process run (every
    footprint is still processed in time order within its chunk, and a stepped source
    reloads its step at most once per chunk boundary). Where the platform offers fork
    (Linux, macOS) the pool is forked after the sources are loaded, so the workers share
    one copy of every source; otherwise (Windows) the workers are spawned and each builds
    its own sources from ``source_file_paths``.

    The PSF weigher is swappable via ``weigher``. It defaults to the angular-frame PSF
    (:class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher`), oriented by
    the L1B ephemeris (subsatellite point + satellite radius) and each footprint's
    cone-angle rate, for both timescales. A caller whose footprints carry no cone-angle
    rate passes :class:`~libera_utils.footprint_matching.weighting.RadialWeigher`
    explicitly. Both weighers size the PSF from the ephemeris, so camera
    pseudo-footprints, which carry none yet, raise ``ValueError``.

    Coverage-based *acceptance* (the CERES 75%/95% discard/flag rule) is not applied
    here -- this function computes each footprint's coverage and, when
    ``return_coverage`` is set, returns it alongside the values, but turning coverage
    into QA flags (or discarding a footprint) is the caller's job (see
    ``_merge_coverage_qa``). Per-footprint coverage is the fraction of the PSF's
    in-contour energy backed by a usable sample, measured by the weigher on its
    reference grid (see :class:`~libera_utils.footprint_matching.weighting.WeightField`),
    taken as the **minimum** across the active sources so a footprint is scored no
    better than its worst-covered input.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode selecting the active readers and product variables.
    footprints : Sequence[libera_utils.footprint_matching.types.RadiometerFootprint | libera_utils.footprint_matching.types.PseudoFootprint]
        Footprints to aggregate over, in time order: a stepped source reloads its grid
        whenever the nearest step changes. Each on-Earth footprint must carry the L1B
        ephemeris the weigher sizes its PSF from, so a :class:`~libera_utils.footprint_matching.types.PseudoFootprint` raises
        (see ``_footprint_geometry``).
    sources : dict[str, AncillaryReader], optional
        Reader registry key to an instantiated reader, as
        :func:`~libera_utils.footprint_matching.sources.build_sources` returns them: exactly
        the readers active in ``mode``. Pass this or ``source_file_paths``, not both.
    source_file_paths : dict[str, Sequence[Path]], optional
        Reader-key -> staged granule(s) map the sources are built from. Pass this or
        ``sources``, not both; a pool that must spawn its workers needs it.
    weigher : PixelWeigher, optional
        The per-sample PSF weight provider. Defaults to a fresh
        :class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher`, which needs
        each on-Earth footprint's cone-angle rate.
    return_coverage : bool, optional
        When ``True``, also return a per-footprint coverage array. Defaults to
        ``False`` (return only the values dict).
    n_workers : int, optional
        Worker processes to run the footprints on. ``None`` (the default) reads the
        ``FMATCH_WORKERS`` configuration key (``config.json``, overridable through the
        environment variable of the same name); ``1`` runs in the calling process. Fewer
        than ``_MIN_FOOTPRINTS_PER_WORKER`` footprints per worker also runs in the
        calling process, since the pool would cost more than it saves.

    Returns
    -------
    dict[str, np.ndarray] or tuple[dict[str, np.ndarray], np.ndarray]
        With ``return_coverage=False`` (default): the mapping of aggregated-variable
        name to a 1-D array indexed by footprint, carrying ``NaN`` where a footprint had
        no usable data. Float variables keep their reader dtype; categorical (integer)
        variables are returned as ``float64`` so they can carry ``NaN`` too, and the
        product's integer fill (see ``_fill_value_for``) replaces it when the column is
        written. With
        ``return_coverage=True``: a ``(values, coverage)`` tuple, where
        ``coverage`` is a 1-D float array in ``[0, 1]`` indexed by footprint (the
        minimum source coverage; ``0`` where the footprint had no covered energy).

    Raises
    ------
    ValueError
        If neither or both of ``sources`` and ``source_file_paths`` are supplied; if an on-Earth
        footprint is a :class:`~libera_utils.footprint_matching.types.PseudoFootprint`, which carries no L1B ephemeris; if a footprint
        has no finite viewing zenith angle; if an on-Earth footprint has no usable cone-angle
        rate and ``weigher`` is the default ``AngularPSFWeigher``; if a footprint's time is NaT and a
        source selects by time; if ``n_workers``, or the ``FMATCH_WORKERS`` configuration
        key it defaults to, is below one or not an integer; or if the pool must be
        spawned (no fork on this platform) and ``source_file_paths`` was not given, so a
        worker cannot build its sources.
    KeyError
        If the keys of ``sources`` or ``source_file_paths`` are not exactly the readers active
        in ``mode``.
    TypeError
        If a footprint is neither a :class:`~libera_utils.footprint_matching.types.RadiometerFootprint` nor a :class:`~libera_utils.footprint_matching.types.PseudoFootprint`.
    """
    if sources is None:
        if source_file_paths is None:
            raise ValueError("aggregate_external_variables requires either sources or source_file_paths.")
        sources = build_sources(mode, source_file_paths)
    elif source_file_paths is not None:
        raise ValueError(
            "aggregate_external_variables takes sources or source_file_paths, not both: pass source_file_paths "
            "when the workers may be spawned, so every worker builds the same readers."
        )
    else:
        require_mode_reader_keys(mode, sources, "sources")

    if weigher is None:
        weigher = AngularPSFWeigher()

    footprints = list(footprints)
    n_footprints = len(footprints)

    if n_workers is None:
        n_workers = _configured_worker_count()
    if n_workers < 1:
        raise ValueError(f"n_workers must be at least 1, got {n_workers}.")
    if n_workers > 1 and n_footprints >= _MIN_FOOTPRINTS_PER_WORKER * n_workers:
        return _aggregate_in_parallel(mode, footprints, sources, source_file_paths, weigher, return_coverage, n_workers)
    return _aggregate_footprints(mode, footprints, sources, weigher, return_coverage)


def _configured_worker_count() -> int:
    """Read the worker count from the ``FMATCH_WORKERS`` configuration key.

    Raises
    ------
    ValueError
        If the configured value is not a positive integer.
    """
    raw = config.get(FMATCH_WORKERS_CONFIG_KEY)
    message = (
        f"{FMATCH_WORKERS_CONFIG_KEY} must be a positive integer, got {raw!r}: set it in config.json or the "
        f"{FMATCH_WORKERS_CONFIG_KEY} environment variable."
    )
    try:
        n_workers = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if n_workers < 1:
        raise ValueError(message)
    return n_workers


def _aggregate_footprints(
    mode: OperationalMode,
    footprints: list[RadiometerFootprint | PseudoFootprint],
    sources: dict[str, AncillaryReader],
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
    if weigher.uses_viewing_frame and on_earth:
        frames = _batch_viewing_frames([geoms[i] for i in on_earth])
        frame_rows = {i: row for row, i in enumerate(on_earth)}

    # The readers passed in are exactly the mode's set (aggregate_external_variables checks
    # them). Each reader's *class* (needed for product_variable_specs() and the INSTRUMENT
    # token) is resolved from the registry by key.
    reader_classes = {key: ReaderRegistry.get(key) for key in sorted(sources)}

    # Pre-allocate one NaN-filled output array per product variable: 1-D for a spec, and
    # (footprint, coordinate) for a stack of specs (e.g. ERA5 pressure levels). Integer
    # (categorical) variables are held as float64 so a footprint with no usable data
    # stays NaN; the integer fill from _fill_value_for replaces it when the column is placed.
    # Cache the per-reader name plan so we do not recompute it for every footprint.
    outputs: dict[str, np.ndarray] = {}
    plan: dict[str, list[tuple[str, str]]] = {}
    for key, reader_cls in reader_classes.items():
        specs = {spec.name: spec for spec in reader_cls.product_variable_specs()}
        stacks = reader_cls.product_stacks()
        stacked_members = {member for stack in stacks for member in stack.members}
        columns: list[tuple[str, VariableSpec, int | None]] = [
            (name, spec, None) for name, spec in specs.items() if name not in stacked_members
        ]
        for stack in stacks:
            columns.append((stack.name, specs[stack.members[0]], len(stack.members)))
        entries: list[tuple[str, str]] = []
        for name, spec, width in columns:
            # Per-spec mode gating: a reader may declare variables that only some of its
            # products carry (e.g. the extended FLASHFlux cloud/aerosol/albedo fields are
            # FMATCH-IMAGER only), so skip specs whose modes exclude this one.
            if mode not in spec.modes:
                continue
            # Column name is ``<READER_KEY>_<name>`` (no instrument token), the name the
            # product definition declares; the definition carries the reader's
            # INSTRUMENT in the variable's long_name.
            column_name = f"{key}_{name}"
            dtype = np.dtype(spec.dtype)
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype(np.float64)
            shape = n_footprints if width is None else (n_footprints, width)
            outputs[column_name] = np.full(shape, np.nan, dtype=dtype)
            entries.append((column_name, name))
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
            samples = sources[key].samples_in(footprint.bbox, footprint.time)
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
            # One scalar per spec, then one value (or one row, for a stack) per product
            # variable, keyed by the bare spec or stack name.
            scalars = aggregate_sample_variables(reader_cls, samples, weight_field)
            values = stack_product_values(reader_cls, scalars)
            for column_name, name in plan[key]:
                outputs[column_name][i] = values[name]

    if not return_coverage:
        return outputs
    # +inf survives only for footprints that saw no source at all (empty sources) or
    # were off-limb; score those as zero coverage.
    coverage = np.where(np.isinf(coverage_min), 0.0, coverage_min)
    return outputs, coverage


def _source_paths_key(
    mode: OperationalMode, source_file_paths: dict[str, Sequence[Path]]
) -> tuple[str, tuple[tuple[str, tuple[str, ...]], ...]]:
    """A hashable identity of a run's staged inputs, for the spawned workers' source cache."""
    entries = []
    for key in sorted(source_file_paths):
        entries.append((key, tuple(str(p) for p in source_file_paths[key])))
    return mode.value, tuple(entries)


def _worker_sources(
    mode: OperationalMode, source_file_paths: dict[str, Sequence[Path]] | None
) -> dict[str, AncillaryReader]:
    """The sources a worker process aggregates with.

    A forked worker inherits the parent's loaded sources through :data:`_FORK_SOURCES`
    (copy-on-write, no reload). A spawned worker builds its own from the staged paths
    once and keeps them in :data:`_WORKER_SOURCES_CACHE` for the chunks that follow;
    ``source_file_paths`` is ``None`` only for a forked worker, since
    :func:`_aggregate_in_parallel` refuses to spawn without it.
    """
    if _FORK_SOURCES is not None:
        return _FORK_SOURCES
    key = _source_paths_key(mode, source_file_paths)
    sources = _WORKER_SOURCES_CACHE.get(key)
    if sources is None:
        _WORKER_SOURCES_CACHE.clear()
        sources = build_sources(mode, source_file_paths)
        _WORKER_SOURCES_CACHE[key] = sources
    return sources


def _aggregate_chunk(
    mode: OperationalMode,
    footprints: list[RadiometerFootprint | PseudoFootprint],
    source_file_paths: dict[str, Sequence[Path]] | None,
    weigher: PixelWeigher,
    return_coverage: bool,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], np.ndarray]:
    """Worker entry point: aggregate one contiguous chunk of footprints."""
    sources = _worker_sources(mode, source_file_paths)
    return _aggregate_footprints(mode, footprints, sources, weigher, return_coverage)


def _aggregate_in_parallel(
    mode: OperationalMode,
    footprints: list[RadiometerFootprint | PseudoFootprint],
    sources: dict[str, AncillaryReader],
    source_file_paths: dict[str, Sequence[Path]] | None,
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
            "own sources: pass source_file_paths instead of sources."
        )
    context = multiprocessing.get_context(start_method)
    if start_method == "fork":
        # Load every static source now, so the workers inherit the indexed data rather than
        # each loading it; a stepped source loads its first step on its first query.
        for reader in sources.values():
            reader.preload()
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

    Produces the ``Sunglint_Angle`` column
    every FMATCH product definition declares: the angle between the sensor view direction and the specular
    reflection of the solar beam, in degrees. Small values indicate potential sun
    glint contamination.

    Parameters
    ----------
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : np.ndarray
        Per-footprint geolocation angles in degrees.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"Sunglint_Angle": ...}``, a 1-D array of glint angles in degrees (``[0, 180]``)
        indexed by footprint. ``NaN`` propagates wherever any input angle is ``NaN``.

    Notes
    -----
    The sun-glint angle is the angle between the sensor view direction and the
    specular (mirror) reflection of the solar beam off a horizontal surface. Writing
    the view and specular directions as unit vectors in the local horizontal frame and
    taking their dot product gives, with all angles in degrees,

    ``cos(glint) = cos(SZA)*cos(VZA) + sin(SZA)*sin(VZA)*cos(RAA)``.

    The sign of the third term follows the relative azimuth convention of the L1B
    ``Relative_Azimuth_Surface`` input, the CERES BDS R3V4 origin as curryer computes it
    (``curryer.compute.spatial.relative_azimuth``): ``mod(view_az - sun_az + 180, 360)``,
    with both azimuths taken at the surface point, clockwise from North, toward the
    spacecraft and the Sun. The Sun therefore sits at ``RAA == 180``, the backscatter hot
    spot, where glint is ``SZA + VZA``; the specular direction is ``RAA == 0``, where
    glint falls to zero when ``VZA == SZA``.
    """
    sza = np.radians(np.asarray(solar_zenith_angle, dtype=float))
    vza = np.radians(np.asarray(viewing_zenith_angle, dtype=float))
    raa = np.radians(np.asarray(relative_azimuth_angle, dtype=float))
    cos_glint = np.cos(sza) * np.cos(vza) + np.sin(sza) * np.sin(vza) * np.cos(raa)
    # Clip absorbs floating-point excursions just outside [-1, 1]; NaN inputs stay NaN.
    cos_glint = np.clip(cos_glint, -1.0, 1.0)
    return {"Sunglint_Angle": np.degrees(np.arccos(cos_glint))}


def build_radiometer_footprints(l1b_inputs: dict[str, np.ndarray]) -> list[RadiometerFootprint]:
    """Build one :class:`~libera_utils.footprint_matching.types.RadiometerFootprint` per L1B pass-through record.

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
        The arrays
        :func:`libera_utils.footprint_matching.footprint_match_algorithm.load_l1b_radiometer_inputs`
        returns (already filtered to finite geolocation), keyed by their L1B names. ``RADIOMETER_TIME``,
        ``Latitude``, ``Longitude``, ``Viewing_Zenith_Surface`` and every name in
        ``L1B_SCAN_REFERENCE_VARIABLES`` are required; ``Cone_Angle_Rate``
        (``L1B_CONE_ANGLE_RATE_VARIABLE``) is read when present.

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
    required = (RADIOMETER_TIME_KEY, "Latitude", "Longitude", "Viewing_Zenith_Surface", *L1B_SCAN_REFERENCE_VARIABLES)
    missing = [name for name in required if name not in l1b_inputs]
    if missing:
        raise ValueError(f"L1B inputs are missing the key(s) radiometer footprints need: {', '.join(missing)}")

    times = np.asarray(l1b_inputs[RADIOMETER_TIME_KEY]).astype("datetime64[ns]")
    latitudes = np.asarray(l1b_inputs["Latitude"], dtype=float)
    longitudes = np.asarray(l1b_inputs["Longitude"], dtype=float)
    viewing_zeniths = np.asarray(l1b_inputs["Viewing_Zenith_Surface"], dtype=float)
    subsatellite_lats = np.asarray(l1b_inputs["Subsatellite_Latitude"], dtype=float)
    subsatellite_lons = np.asarray(l1b_inputs["Subsatellite_Longitude"], dtype=float)
    satellite_radii = np.asarray(l1b_inputs["Radius_of_Satellite_from_Center_of_Earth"], dtype=float)
    cone_rates = np.asarray(l1b_inputs.get(L1B_CONE_ANGLE_RATE_VARIABLE, np.full(latitudes.shape, np.nan)), dtype=float)

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
                # A fill (-999) or NaN cone-angle rate means an unknown scan rate, so it becomes None.
                cone_angle_rate=None if (not np.isfinite(cone_rate) or cone_rate == L1B_FILL_VALUE) else cone_rate,
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
    source_file_paths: dict[str, Sequence[Path]] | None,
    sources: dict[str, AncillaryReader] | None,
    weigher: PixelWeigher | None,
    allow_placeholders: bool = False,
    to_grid: Callable[[Any, np.dtype, Any], np.ndarray] | None = None,
) -> None:
    """Merge the *computed* (non-input) product columns into ``data`` in place.

    Fills the two families of variable the footprint-matching engine owns, so they
    carry real values instead of placeholders:

    * **Derived viewing geometry** (``Sunglint_Angle``) -- always computed, from the
      real per-footprint geolocation angles (:func:`compute_derived_viewing_geometry`).
    * **External aggregated variables** and the **coverage/QA** columns
      (``PSF_Coverage_Fraction``, ``Quality_Flag``) -- computed from the ancillary inputs
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
    source_file_paths : dict[str, Sequence[Path]] or None
        Reader-key -> staged ancillary granule(s) map the sources are built from. Pass this or
        ``sources``, not both.
    sources : dict[str, AncillaryReader] or None
        Pre-built readers to aggregate from, keyed by reader registry key: exactly the readers
        active in ``mode``. Pass this or ``source_file_paths``, not both.
    weigher : PixelWeigher or None
        PSF weigher for the aggregation, or ``None`` for the
        :func:`aggregate_external_variables` default.
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
        ``allow_placeholders`` is ``False``, or as :func:`aggregate_external_variables` raises.
    KeyError
        If the keys of ``sources`` or ``source_file_paths`` are not exactly the readers active
        in ``mode``.
    """

    # The engine computes one value per footprint (flat). For the camera-timescale products the caller passes a
    # ``to_grid`` scatter so each computed column lands on the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid (padded cells taking
    # the declared fill); the radiometer products keep the 1-D column as-is. Integer columns arrive as float with NaN
    # where a footprint had no usable data, and take the integer fill there.
    def _place(name: str, values: np.ndarray) -> np.ndarray:
        dtype, fill_value = _fill_value_for(name, definition.variables[name])
        if np.issubdtype(dtype, np.integer):
            values = np.where(np.isnan(values), fill_value, values)
        return np.asarray(values, dtype=dtype) if to_grid is None else to_grid(values, dtype, fill_value)

    # Derived geometry is always computable from the real angle columns.
    if "Sunglint_Angle" in definition.variables and "Sunglint_Angle" not in data:
        sunglint = compute_derived_viewing_geometry(solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle)[
            "Sunglint_Angle"
        ]
        data["Sunglint_Angle"] = _place("Sunglint_Angle", sunglint)

    # External aggregation needs staged ancillary data. Without it the product would carry
    # placeholder external columns and an unflagged Quality_Flag, so only an explicit opt-in
    # leaves them as placeholders.
    if sources is None and source_file_paths is None:
        if allow_placeholders:
            return
        raise ValueError(
            f"No ancillary inputs were supplied for {mode.value}, so its external variables, PSF_Coverage_Fraction "
            "and Quality_Flag cannot be computed. Pass sources or source_file_paths, or pass "
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
    """Fill the coverage fraction and OR the CERES coverage rule into ``Quality_Flag``.

    Coverage sets QA flags rather than discarding the footprint, following the CERES
    75%/95% rule: coverage in ``[0.75, 0.95)`` sets ``PARTIAL_COVERAGE``; coverage
    below ``0.75`` sets ``INSUFFICIENT_COVERAGE``; a limb-truncated bounding box sets
    ``LIMB_TRUNCATED``; an off-limb (space/calibration) view with no Earth footprint sets
    ``OFF_LIMB`` (and, via its forced zero coverage, ``INSUFFICIENT_COVERAGE``). These
    bits are OR-ed into any ``Quality_Flag`` already present (the
    camera path sets its segmentation flags first), so a footprint carries both its
    segmentation and its coverage provenance.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Variable arrays; ``PSF_Coverage_Fraction`` and ``Quality_Flag`` are set/updated here.
    definition : LiberaDataProductDefinition
        Gates which of the two columns are written (both are declared by every FMATCH
        definition, but the guard keeps this robust).
    coverage : np.ndarray
        Per-footprint coverage fraction in ``[0, 1]``.
    footprints : Sequence[Any]
        Footprints, read only for each ``off_limb`` flag and, where ``bbox`` is not
        ``None``, its ``truncated`` flag.
    """
    if "PSF_Coverage_Fraction" in definition.variables:
        dtype, fill_value = _fill_value_for("PSF_Coverage_Fraction", definition.variables["PSF_Coverage_Fraction"])
        data["PSF_Coverage_Fraction"] = (
            np.asarray(coverage, dtype=dtype) if to_grid is None else to_grid(coverage, dtype, fill_value)
        )

    if "Quality_Flag" not in definition.variables:
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
    # into any Quality_Flag already assembled (e.g. the camera segmentation flags, which the camera path sets first).
    if to_grid is not None:
        qbits = to_grid(qbits, np.dtype(np.int64), 0)
    existing = data.get("Quality_Flag")
    if existing is not None:
        qbits |= np.asarray(existing, dtype=np.int64)
    data["Quality_Flag"] = qbits.astype(np.dtype(definition.variables["Quality_Flag"].dtype))


def assemble_fmatch_dataset(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    *,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path]] | None = None,
    sources: dict[str, AncillaryReader] | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
) -> Dataset:
    """Assemble a conformant FMATCH :class:`xarray.Dataset` for an operational mode.

    Combines the per-footprint geolocation inputs, the derived viewing geometry from
    :func:`compute_derived_viewing_geometry`, and the aggregated external
    variables from :func:`aggregate_external_variables` into the variable dict
    expected by the mode's product definition (from
    :func:`load_fmatch_definition`), then builds the Dataset once via
    ``LiberaDataProductDefinition.create_product_dataset``, with ``algorithm_version``
    and ``InputGranules`` as its dynamic attributes, and brings it into conformance
    with ``enforce_dataset_conformance``.

    Dispatch is by timescale, because that determines what the mode is built
    *from*:

    * **Camera-timescale** (``CAM_CAMTIME``, ``IMAGER_CAMTIME``) - assembled from
      the camera pseudo-footprints segmented by the Camera Cloud Fraction algorithm
      and read from its CF-CAM-CAMTIME product.
      ``inputs`` is that sequence of
      :class:`~libera_utils.footprint_matching.types.PseudoFootprint` objects. See
      ``_assemble_camtime_dataset``.
    * **Radiometer-timescale** (``CAM``, ``IMAGER``) - assembled
      from the L1B pass-through arrays produced by
      :func:`libera_utils.footprint_matching.footprint_match_algorithm.load_l1b_radiometer_inputs`.
      The first positional argument is that dict. See
      ``_assemble_radiometer_dataset``.

    In both cases the derived viewing geometry (``Sunglint_Angle``) is always computed
    from the real geolocation angles. The external aggregated variables and the
    coverage/QA columns are computed from the ancillary inputs (``sources`` or
    ``source_file_paths``, forwarded to :func:`aggregate_external_variables`). A declared
    variable no input produced raises, unless ``allow_placeholders=True`` asks for it to
    be written as a conformant placeholder. Columns are named as the product definition
    declares them.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being assembled.
    inputs : Sequence[PseudoFootprint] or dict[str, np.ndarray]
        Camera pseudo-footprints for the camera-timescale modes, or the L1B dict
        (keyed by L1B names) for the radiometer-timescale modes.
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
        algorithm (Libera WFOV camera), in percent ``[0, 100]`` with ``NaN`` for a
        missing value, as a 1-D array indexed by footprint in the same order as the
        time coordinate. This is an *internal* algorithm
        output - it does not come from a reader and is already aggregated to one
        value per footprint - so it is merged directly into the ``Cloud_Fraction_Camera``
        variable rather than going through :func:`aggregate_external_variables`.
        Only the CAM modes (``CAM``, ``CAM_CAMTIME``) declare this variable: it is
        required for them unless ``allow_placeholders`` is ``True``, and must be ``None``
        for the IMAGER modes.
    source_file_paths : dict[str, Sequence[Path]], optional
        Reader-key -> staged ancillary granule(s) map the sources are built from. Pass this or
        ``sources``, not both.
    sources : dict[str, AncillaryReader], optional
        Pre-built readers to aggregate from, keyed by reader registry key: exactly the readers
        active in ``mode``. Pass this or ``source_file_paths``, not both.
    weigher : PixelWeigher, optional
        PSF weigher for the aggregation. Defaults to
        :class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher` (see
        :func:`aggregate_external_variables`).
    allow_placeholders : bool, optional
        When ``True``, a declared variable no input produced is written as fill instead
        of raising: the external, ``PSF_Coverage_Fraction`` and ``Quality_Flag`` columns
        when neither ``sources`` nor ``source_file_paths`` is given, a reader's columns when
        its source is missing, and ``Cloud_Fraction_Camera`` when ``cloud_fraction_camera``
        is omitted for a CAM mode. Defaults to ``False``.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's product definition.

    Raises
    ------
    ValueError
        If ``inputs`` is empty; if the radiometer-timescale ``inputs`` dict is missing
        a required L1B key or its arrays have inconsistent lengths; if neither
        ``sources`` nor ``source_file_paths`` is given and ``allow_placeholders``
        is ``False``, or both are given; if ``cloud_fraction_camera`` is omitted for a CAM
        mode without ``allow_placeholders``, given for an IMAGER mode, not one value per
        footprint, or has a finite value outside ``[0, 100]``; if any other declared
        variable no input produced is left without ``allow_placeholders``; if a footprint
        has no finite viewing zenith angle, or no usable cone-angle rate under the default
        weigher; if a camera-timescale mode is aggregated, since camera pseudo-footprints
        carry no L1B ephemeris; if a footprint's time is NaT and a source selects by time;
        or if assembly produces a column the definition does not declare.
    KeyError
        If the keys of ``sources`` or ``source_file_paths`` are not exactly the readers active
        in ``mode``.
    MissingGeolocationError
        If a radiometer record's ephemeris is unusable or its footprint is not visible
        from the satellite (see :func:`build_radiometer_footprints`).
    """
    if definition is None:
        definition = load_fmatch_definition(mode)
    data = _assemble_fmatch_data(
        mode,
        inputs,
        definition,
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=source_file_paths,
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )
    dataset = definition.create_product_dataset(
        data, dynamic_product_attributes={"algorithm_version": algorithm_version, "InputGranules": input_files}
    )
    return definition.enforce_dataset_conformance(dataset)


def _assemble_fmatch_data(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    *,
    cloud_fraction_camera: np.ndarray | None,
    source_file_paths: dict[str, Sequence[Path]] | None,
    sources: dict[str, AncillaryReader] | None,
    weigher: PixelWeigher | None,
    allow_placeholders: bool,
) -> dict[str, np.ndarray]:
    """Assemble every array a FMATCH product declares, dispatching on the mode's timescale.

    The step :func:`assemble_fmatch_dataset` and :func:`write_fmatch_product` share: the
    first builds a Dataset from the arrays, the second hands them to
    :func:`~libera_utils.io.netcdf.write_libera_data_product`, which builds, enforces and
    checks the Dataset once.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being assembled.
    inputs : Sequence[PseudoFootprint] or dict[str, np.ndarray]
        Camera pseudo-footprints for the camera-timescale modes, or the L1B dict for the
        radiometer-timescale modes.
    definition : LiberaDataProductDefinition
        The mode's product definition.
    cloud_fraction_camera, source_file_paths, sources, weigher, allow_placeholders
        As in :func:`assemble_fmatch_dataset`.

    Returns
    -------
    dict[str, np.ndarray]
        Every coordinate and variable array the definition declares, keyed by product name.

    Raises
    ------
    ValueError
        As ``_assemble_camtime_dataset`` or ``_assemble_radiometer_dataset`` raises.
    KeyError
        If the keys of ``sources`` or ``source_file_paths`` are not exactly the readers active
        in ``mode``.
    MissingGeolocationError
        If a radiometer record's ephemeris is unusable (see :func:`build_radiometer_footprints`).
    """
    assembler = _assemble_camtime_dataset if mode in _CAMERA_TIMESCALE_MODES else _assemble_radiometer_dataset
    return assembler(
        inputs,
        mode=mode,
        definition=definition,
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=source_file_paths,
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )


def _fill_value_for(name: str, variable_definition: Any) -> tuple[np.dtype, Any]:
    """Return the ``(dtype, fill_value)`` a variable or coordinate uses for absent cells.

    The fill is the definition's declared ``_FillValue`` when present, and otherwise ``NaN``
    for floating-point definitions. An integer bitmask (a definition declaring
    ``flag_masks``, such as ``Quality_Flag``) takes ``0``, meaning no flag set. Any other integer
    must declare its ``_FillValue``, :data:`FMATCH_INTEGER_FILL_VALUE` in the FMATCH
    definitions, because no value of an integer column is missing by default.
    Used both for not-yet-computed placeholder variables and for the padded cells of the
    camera-timescale grid (images segmented into fewer subsections than the widest image).

    Parameters
    ----------
    name : str
        The variable or coordinate name, for the error message.
    variable_definition : LiberaVariableDefinition
        The product-definition entry for the variable or coordinate.

    Returns
    -------
    tuple[numpy.dtype, Any]
        The declared dtype and the fill value to use for missing/padded cells.

    Raises
    ------
    ValueError
        If an integer definition declares neither ``_FillValue`` nor ``flag_masks``.
    """
    dtype = np.dtype(variable_definition.dtype)
    fill_value = variable_definition.attributes.get("_FillValue")
    if fill_value is not None:
        return dtype, fill_value
    if np.issubdtype(dtype, np.floating):
        return dtype, np.nan
    if "flag_masks" in variable_definition.attributes:
        return dtype, 0
    raise ValueError(
        f"Integer variable {name} declares neither _FillValue nor flag_masks, so its missing cells have no fill: "
        f"declare _FillValue: {FMATCH_INTEGER_FILL_VALUE} (or flag_masks for a bitmask) on "
        f"{name} in its product definition."
    )


def _stack_coordinates(mode: OperationalMode, definition: LiberaDataProductDefinition) -> dict[str, np.ndarray]:
    """Return the coordinate values of every stacked product dimension ``mode``'s readers declare.

    A reader's ``ProductStack`` names the product dimension it stacks its specs along and the
    coordinate value of each member (e.g. ERA5's pressure levels in hPa). Those values are the
    product's coordinate on that dimension, so they are taken from the stacks rather than
    repeated here, and they are written whether or not the reader's inputs were staged.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode whose active readers' stacks to collect.
    definition : LiberaDataProductDefinition
        The product definition, which gives each coordinate its dtype.

    Returns
    -------
    dict[str, np.ndarray]
        Dimension name to its coordinate values, for every stacked dimension the definition
        declares as a coordinate.
    """
    coordinates: dict[str, np.ndarray] = {}
    for reader_cls in ReaderRegistry.get_readers_for_mode(mode).values():
        for stack in reader_cls.product_stacks():
            if stack.dimension in definition.coordinates:
                dtype = np.dtype(definition.coordinates[stack.dimension].dtype)
                coordinates[stack.dimension] = np.asarray(stack.coordinate_values, dtype=dtype)
    return coordinates


def _fill_placeholder_variables(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    shape: int | tuple[int, ...],
    *,
    allow_placeholders: bool,
) -> None:
    """Fill every declared variable missing from ``data`` with a conformant placeholder, if allowed.

    A declared variable no input produced (the external columns when no ancillary data
    was staged, a reader's columns when its source was not supplied, or
    ``Cloud_Fraction_Camera`` when it was omitted) would otherwise reach the file as a
    column of fill indistinguishable from a computed one, so it is written only when the
    caller opts in with ``allow_placeholders=True``. Mutates ``data`` in place, adding one
    array per such variable, filled with what :func:`_fill_value_for` returns: structurally
    valid (declared dtype/shape/attributes) but numerically meaningless.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        The variable arrays assembled so far (the real, input-derived columns).
    definition : LiberaDataProductDefinition
        The product definition naming every variable the file must contain.
    shape : int or tuple of int
        Shape of the record axes each placeholder array is built over: the 1-D record-axis
        length for the radiometer-timescale products, or the ``(CAMERA_TIME, PSEUDOFOOTPRINT)``
        grid shape for the camera-timescale products. A variable declared on a further
        dimension (e.g. ``PRESSURE_LEVEL``) is as long on it as that coordinate in ``data``.
    allow_placeholders : bool
        Whether a declared variable no input produced may be written as fill.

    Raises
    ------
    ValueError
        If a declared variable is missing from ``data`` and ``allow_placeholders`` is
        ``False``, naming every such variable.
    """
    missing = [name for name in definition.variables if name not in data]
    if missing and not allow_placeholders:
        product_id = definition.attributes.get("ProductID", "this product")
        raise ValueError(
            f"{product_id} declares {', '.join(missing)}, which no input produced: supply every source the "
            "mode's readers use (sources or source_file_paths) and, for a CAM mode, cloud_fraction_camera, or pass "
            "allow_placeholders=True to write them as fill values."
        )
    record_shape = (shape,) if isinstance(shape, int) else tuple(shape)
    for name in missing:
        variable_definition = definition.variables[name]
        further = variable_definition.dimensions[len(record_shape) :]
        placeholder_shape = record_shape + tuple(len(data[dimension]) for dimension in further)
        dtype, fill_value = _fill_value_for(name, variable_definition)
        data[name] = np.full(placeholder_shape, fill_value, dtype=dtype)


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


# Bit layout of the CF-CAM-CAMTIME ``psf_bbox_flags`` variable, set by the Camera Cloud Fraction
# algorithm. These carry the BoundingBox flags that cannot be recovered from the stored bounds alone
# (``is_polar`` depends on the boresight latitude and ``truncated`` on the limb-clipping history), so
# FMATCH can rebuild the exact BoundingBox of each pseudo-footprint.
CAMTIME_BBOX_WRAPS_DATELINE_BIT: int = 0b001
CAMTIME_BBOX_IS_POLAR_BIT: int = 0b010
CAMTIME_BBOX_TRUNCATED_BIT: int = 0b100

# Every CF-CAM-CAMTIME variable and coordinate :func:`pseudofootprints_from_camtime_dataset` reads to
# rebuild the camera pseudo-footprints. The bounding-box longitudes are raw (they may exceed +/-180 for a
# dateline-crossing box), ``altitude`` is in meters as in L1B, and ``q_flags`` holds only the
# :class:`~libera_utils.footprint_matching.types.CameraFootprintQualityFlag` bits.
CAMTIME_PSEUDOFOOTPRINT_VARIABLES: tuple[str, ...] = (
    "CAMERA_TIME",
    "camera_pixel_x_min",
    "camera_pixel_x_max",
    "camera_pixel_y_min",
    "camera_pixel_y_max",
    "latitude",
    "longitude",
    "altitude",
    "solar_zenith_angle",
    "viewing_zenith_angle",
    "relative_azimuth_angle",
    "center_pixel_x",
    "center_pixel_y",
    "psf_bbox_lat_min",
    "psf_bbox_lat_max",
    "psf_bbox_lon_min",
    "psf_bbox_lon_max",
    "psf_bbox_flags",
    "q_flags",
)


class MissingCamtimeVariablesError(ValueError):
    """A camera-timescale input product lacks variables needed to rebuild its pseudo-footprints."""


def camtime_real_cell_mask(dataset: Dataset) -> np.ndarray:
    """Boolean ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` mask of real (non-padded) footprint cells.

    The camera-timescale grid is rectangular but ragged: images segmented into fewer subsections than
    the widest image leave padded trailing cells. A real footprint always has a finite center-pixel
    latitude, so a finite ``latitude`` marks a real cell. The footprint reconstruction and any per-cell
    variable read (e.g. ``cloud_fraction``) both use this mask, so they select the same cells in the
    same order.

    Parameters
    ----------
    dataset : xarray.Dataset
        A camera-timescale dataset carrying a 2-D ``latitude`` variable.

    Returns
    -------
    numpy.ndarray
        Boolean array shaped ``(CAMERA_TIME, PSEUDOFOOTPRINT)``; ``True`` at real footprint cells.
    """
    return np.isfinite(np.asarray(dataset["latitude"].values, dtype=float))


def pseudofootprints_from_camtime_dataset(dataset: Dataset) -> list[PseudoFootprint]:
    """Rebuild the flat pseudo-footprint list from a CF-CAM-CAMTIME dataset.

    The Camera Cloud Fraction algorithm segments each L1B camera image into pseudo-footprints and
    writes them onto the rectangular ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid of CF-CAM-CAMTIME. Both
    camera-timescale FMATCH modes take their footprints from that product rather than segmenting the
    camera image themselves.

    Real footprints are recovered in row-major order (``CAMERA_TIME`` ascending, then
    ``PSEUDOFOOTPRINT`` ascending), skipping the padded cells (see :func:`camtime_real_cell_mask`).
    Because the per-image column order is preserved, the FMATCH product built from the result lays out
    its grid identically to the input.

    Parameters
    ----------
    dataset : xarray.Dataset
        A CF-CAM-CAMTIME dataset carrying every name in ``CAMTIME_PSEUDOFOOTPRINT_VARIABLES``.

    Returns
    -------
    list[PseudoFootprint]
        The pseudo-footprints, in an order that regenerates the source grid.

    Raises
    ------
    MissingCamtimeVariablesError
        If ``dataset`` lacks any name in ``CAMTIME_PSEUDOFOOTPRINT_VARIABLES``.
    """
    missing = [name for name in CAMTIME_PSEUDOFOOTPRINT_VARIABLES if name not in dataset.variables]
    if missing:
        raise MissingCamtimeVariablesError(
            f"The camera-timescale input lacks {', '.join(missing)}, which FMATCH needs to rebuild the camera "
            "pseudo-footprints. Check that the input is a CF-CAM-CAMTIME product written by the Camera Cloud "
            "Fraction algorithm, and report the missing variables to that algorithm's owners if it is."
        )

    def values(name: str, dtype: type | None = None) -> np.ndarray:
        return np.asarray(dataset[name].values, dtype=dtype)

    camera_time = dataset["CAMERA_TIME"].values
    lat, lon, alt = values("latitude", float), values("longitude", float), values("altitude", float)
    sza = values("solar_zenith_angle", float)
    vza = values("viewing_zenith_angle", float)
    raa = values("relative_azimuth_angle", float)
    center_x, center_y = values("center_pixel_x"), values("center_pixel_y")
    px_min, px_max = values("camera_pixel_x_min"), values("camera_pixel_x_max")
    py_min, py_max = values("camera_pixel_y_min"), values("camera_pixel_y_max")
    bbox_lat_min, bbox_lat_max = values("psf_bbox_lat_min", float), values("psf_bbox_lat_max", float)
    bbox_lon_min, bbox_lon_max = values("psf_bbox_lon_min", float), values("psf_bbox_lon_max", float)
    bbox_flags, q_flags = values("psf_bbox_flags"), values("q_flags")

    # np.nonzero is C-order (CAMERA_TIME ascending, then PSEUDOFOOTPRINT ascending), which is the order
    # that rebuilds this grid and the order a boolean-mask read of a per-cell variable returns.
    rows, cols = np.nonzero(camtime_real_cell_mask(dataset))
    footprints: list[PseudoFootprint] = []
    for row, col in zip(rows, cols, strict=True):
        flags = int(bbox_flags[row, col])
        bbox = BoundingBox(
            float(bbox_lat_min[row, col]),
            float(bbox_lat_max[row, col]),
            float(bbox_lon_min[row, col]),
            float(bbox_lon_max[row, col]),
            bool(flags & CAMTIME_BBOX_WRAPS_DATELINE_BIT),
            bool(flags & CAMTIME_BBOX_IS_POLAR_BIT),
            bool(flags & CAMTIME_BBOX_TRUNCATED_BIT),
        )
        footprints.append(
            PseudoFootprint(
                time=camera_time[row],
                # camera_pixel_*_max are inclusive; the slices are half-open.
                slice_x=slice(int(px_min[row, col]), int(px_max[row, col]) + 1),
                slice_y=slice(int(py_min[row, col]), int(py_max[row, col]) + 1),
                center_ix=int(center_x[row, col]),
                center_iy=int(center_y[row, col]),
                latitude=float(lat[row, col]),
                longitude=float(lon[row, col]),
                altitude=float(alt[row, col]),
                solar_zenith_angle=float(sza[row, col]),
                viewing_zenith_angle=float(vza[row, col]),
                relative_azimuth_angle=float(raa[row, col]),
                bbox=bbox,
                q_flags=CameraFootprintQualityFlag(int(q_flags[row, col])),
            )
        )
    return footprints


def _assemble_camtime_dataset(
    footprints: Sequence[PseudoFootprint],
    *,
    mode: OperationalMode,
    definition: LiberaDataProductDefinition,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path]] | None = None,
    sources: dict[str, AncillaryReader] | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
) -> dict[str, np.ndarray]:
    """Assemble the arrays of a camera-timescale FMATCH product from camera pseudo-footprints.

    Serves both camera-timescale modes (``CAM_CAMTIME`` and ``IMAGER_CAMTIME``).
    They are built the same way - from the same L1B camera segmentation - and
    differ only in the set of aggregated variables their definitions declare,
    which the computed-and-placeholder fill handles generically.

    Builds the per-footprint variable arrays declared by the mode's product
    definition. The center-pixel geolocation/geometry, the corner-derived PSF
    bounding box, and the QA flags come straight from the pseudo-footprints (the
    ``real_columns`` built below). The derived ``Sunglint_Angle`` is computed from the
    geolocation angles, and (when ``source_file_paths`` or ``sources`` is supplied) the
    external aggregated variables and coverage/QA columns are computed via
    :func:`_merge_computed_variables`.

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in write order, as read from CF-CAM-CAMTIME.
    mode : OperationalMode
        Which camera-timescale mode to assemble.
    definition : LiberaDataProductDefinition
        The mode's product definition.
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint Camera Cloud Fraction values (Libera WFOV), in percent, in footprint order.
        ``CAM_CAMTIME`` declares this variable and ``IMAGER_CAMTIME`` does not; see
        :func:`assemble_fmatch_dataset`.
    source_file_paths, sources, weigher, allow_placeholders
        Aggregation inputs, as in :func:`assemble_fmatch_dataset`.

    Returns
    -------
    dict[str, np.ndarray]
        Every coordinate and variable array the definition declares, keyed by product name.

    Raises
    ------
    ValueError
        If ``mode`` is not a camera-timescale mode, or ``footprints`` is empty (there
        would be no time axis to write); as :func:`_merge_cloud_fraction_camera` or
        :func:`_merge_computed_variables` raises; or if a declared variable no input
        produced is left without ``allow_placeholders`` (see
        :func:`_fill_placeholder_variables`).
    """
    if mode not in _CAMERA_TIMESCALE_MODES:
        raise ValueError(
            f"{mode.value} is not a camera-timescale mode; camera pseudo-footprints only assemble "
            f"{', '.join(sorted(m.value for m in _CAMERA_TIMESCALE_MODES))}."
        )

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
    # [-180, 180] to satisfy the product definition's valid range. Center_Pixel_X/Y are the
    # boresight stand-in pixel; they are FMATCH-only provenance (the block's inclusive extent
    # is emitted as the Camera_Pixel_{X,Y}_{Min,Max} coordinates below).
    real_columns: dict[str, list[float]] = {
        "Latitude": [f.latitude for f in footprints],
        "Longitude": [f.longitude for f in footprints],
        "Altitude": [f.altitude for f in footprints],
        "Solar_Zenith_Surface": [f.solar_zenith_angle for f in footprints],
        "Viewing_Zenith_Surface": [f.viewing_zenith_angle for f in footprints],
        "Relative_Azimuth_Surface": [f.relative_azimuth_angle for f in footprints],
        "PSF_Bbox_Lat_Min": [f.bbox.lat_min for f in footprints],
        "PSF_Bbox_Lat_Max": [f.bbox.lat_max for f in footprints],
        "PSF_Bbox_Lon_Min": [_normalize_longitude(f.bbox.lon_min) for f in footprints],
        "PSF_Bbox_Lon_Max": [_normalize_longitude(f.bbox.lon_max) for f in footprints],
        "Quality_Flag": [int(f.q_flags) for f in footprints],
        "Center_Pixel_X": [f.center_ix for f in footprints],
        "Center_Pixel_Y": [f.center_iy for f in footprints],
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
        "Camera_Pixel_X_Min": [f.slice_x.start for f in footprints],
        "Camera_Pixel_X_Max": [f.slice_x.stop - 1 for f in footprints],
        "Camera_Pixel_Y_Min": [f.slice_y.start for f in footprints],
        "Camera_Pixel_Y_Max": [f.slice_y.stop - 1 for f in footprints],
    }
    for coordinate_name, values in pixel_columns.items():
        dtype, fill_value = _fill_value_for(coordinate_name, definition.coordinates[coordinate_name])
        data[coordinate_name] = to_grid(values, dtype, fill_value)

    # Cast each real column to the exact dtype the definition declares and scatter it into the
    # grid. Only columns the definition actually declares are written; both camtime products
    # declare the same segmentation variables, but the guard keeps the assembly robust to
    # definition drift.
    for name, values in real_columns.items():
        variable_definition = definition.variables.get(name)
        if variable_definition is None:
            continue
        dtype, fill_value = _fill_value_for(name, variable_definition)
        data[name] = to_grid(values, dtype, fill_value)

    # Internal (non-reader) Camera Cloud Fraction values, scattered into the same grid, for
    # the products that declare them (CAM-CAMTIME, not IMAGER-CAMTIME).
    _merge_cloud_fraction_camera(
        data,
        definition,
        cloud_fraction_camera,
        n_footprints=n_footprints,
        allow_placeholders=allow_placeholders,
        to_grid=to_grid,
    )

    # Computed columns: derived geometry (always) and, when ancillary inputs are
    # supplied, the external aggregated variables + coverage/QA (OR-ed into Quality_Flag).
    _merge_computed_variables(
        data,
        definition,
        mode,
        footprints,
        solar_zenith_angle=np.asarray(real_columns["Solar_Zenith_Surface"], dtype=float),
        viewing_zenith_angle=np.asarray(real_columns["Viewing_Zenith_Surface"], dtype=float),
        relative_azimuth_angle=np.asarray(real_columns["Relative_Azimuth_Surface"], dtype=float),
        source_file_paths=source_file_paths,
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
        to_grid=to_grid,
    )

    _fill_placeholder_variables(data, definition, grid_shape, allow_placeholders=allow_placeholders)
    return data


def _merge_cloud_fraction_camera(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    cloud_fraction_camera: np.ndarray | None,
    *,
    n_footprints: int,
    allow_placeholders: bool,
    to_grid: Callable[[Any, np.dtype, Any], np.ndarray] | None = None,
) -> None:
    """Merge the CF-CAM cloud fraction into ``data`` for a product that declares it.

    ``cloud_fraction_camera`` is an internal Libera algorithm output (from the WFOV
    Camera Cloud Fraction algorithm), already one value per footprint, so it bypasses
    the reader/aggregation path and is merged straight in. Only the CAM-family
    definitions declare it.

    For the radiometer-timescale products the values are stored on the 1-D record axis as
    given. For the camera-timescale products the caller passes ``to_grid`` so the flat
    per-footprint values are scattered into the rectangular ``(CAMERA_TIME, PSEUDOFOOTPRINT)``
    grid (padded cells taking the variable's fill).

    Raises
    ------
    ValueError
        If the definition declares ``Cloud_Fraction_Camera`` and no values are given while
        ``allow_placeholders`` is ``False``; if values are given for a product that does not
        declare the variable; if their length is not ``n_footprints``; or if a finite value
        is outside ``[0, 100]`` (see :func:`_check_cloud_fraction_percent`).
    """
    variable_definition = definition.variables.get("Cloud_Fraction_Camera")
    product_id = definition.attributes.get("ProductID", "this product")
    if variable_definition is None:
        if cloud_fraction_camera is not None:
            raise ValueError(
                f"cloud_fraction_camera values were given, but {product_id} does not declare that variable: pass "
                "cloud_fraction_camera=None for the IMAGER modes."
            )
        return
    if cloud_fraction_camera is None:
        if not allow_placeholders:
            raise ValueError(
                f"{product_id} declares Cloud_Fraction_Camera but no values were given: pass the CF-CAM values as "
                "cloud_fraction_camera, or "
                "allow_placeholders=True to write the column as fill."
            )
        return
    if len(cloud_fraction_camera) != n_footprints:
        raise ValueError(
            f"cloud_fraction_camera has {len(cloud_fraction_camera)} values for {n_footprints} footprints: pass one "
            "value per footprint, in footprint order."
        )
    _check_cloud_fraction_percent(cloud_fraction_camera)
    dtype, fill_value = _fill_value_for("Cloud_Fraction_Camera", variable_definition)
    if to_grid is None:
        data["Cloud_Fraction_Camera"] = np.asarray(cloud_fraction_camera, dtype=dtype)
    else:
        data["Cloud_Fraction_Camera"] = to_grid(cloud_fraction_camera, dtype, fill_value)


def _check_cloud_fraction_percent(cloud_fraction_camera: np.ndarray) -> None:
    """Check that the CF-CAM cloud fraction is in percent, ``[0, 100]``.

    Cloud fraction is carried in percent in every product that carries it, as the Data
    Product Definition Working Group agreed on 2026-08-19 and the ``Cloud_Fraction_Camera``
    definitions record. ``NaN`` marks a missing value and passes. A 0-1 fraction lies
    inside ``[0, 100]`` too, so this catches an unmasked fill or an out-of-range value,
    not a fraction passed in place of a percent.

    Parameters
    ----------
    cloud_fraction_camera : np.ndarray
        Per-footprint cloud fraction values.

    Raises
    ------
    ValueError
        If any finite value is below 0 or above 100.
    """
    values = np.asarray(cloud_fraction_camera, dtype=float)
    out_of_range = np.isfinite(values) & ((values < 0.0) | (values > 100.0))
    if out_of_range.any():
        bad = values[out_of_range]
        raise ValueError(
            f"cloud_fraction_camera must be in percent, [0, 100], as Cloud_Fraction_Camera declares; "
            f"{bad.size} value(s) are outside it (min {bad.min()}, max {bad.max()}). Pass the CF-CAM cloud fraction "
            "in percent, with NaN for a missing value."
        )


def _assemble_radiometer_dataset(
    l1b_inputs: dict[str, np.ndarray],
    *,
    mode: OperationalMode,
    definition: LiberaDataProductDefinition,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path]] | None = None,
    sources: dict[str, AncillaryReader] | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
) -> dict[str, np.ndarray]:
    """Assemble the arrays of a radiometer-timescale FMATCH product from L1B inputs.

    Serves the two radiometer-timescale modes (``CAM`` and ``IMAGER``). Their footprints
    are the L1B radiometer footprints themselves, so the time coordinate and the
    geolocation/viewing-angle columns (:data:`L1B_PASSTHROUGH_VARIABLES`) are carried
    through verbatim from L1B. Each record's footprint is built by
    :func:`build_radiometer_footprints`, and its ray-traced bounding box is written as
    ``PSF_Bbox_*`` (``NaN`` for an off-limb record). The derived ``Sunglint_Angle`` is
    computed from the angles, and (when ``source_file_paths`` or ``sources`` is supplied)
    the external aggregated variables and coverage/QA columns are computed over those
    footprints.

    Parameters
    ----------
    l1b_inputs : dict[str, np.ndarray]
        The arrays
        :func:`libera_utils.footprint_matching.footprint_match_algorithm.load_l1b_radiometer_inputs`
        returns, keyed by their L1B names: the
        ``RADIOMETER_TIME`` coordinate, each name in :data:`L1B_PASSTHROUGH_VARIABLES`
        and :data:`L1B_SCAN_REFERENCE_VARIABLES`, all the same length, and optionally
        ``Cone_Angle_Rate``.
    mode : OperationalMode
        Which radiometer-timescale mode to assemble.
    definition : LiberaDataProductDefinition
        The mode's product definition.
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint Camera Cloud Fraction values (Libera WFOV), in percent, in the same
        footprint order as the time coordinate. ``CAM`` declares this variable and
        ``IMAGER`` does not; see :func:`assemble_fmatch_dataset`.
    source_file_paths, sources, weigher, allow_placeholders
        Aggregation inputs, as in :func:`assemble_fmatch_dataset`.

    Returns
    -------
    dict[str, np.ndarray]
        Every coordinate and variable array the definition declares, keyed by product name.

    Raises
    ------
    ValueError
        If ``mode`` is a camera-timescale mode, if a required L1B input is missing, if the
        inputs have inconsistent lengths, or if they are empty; as
        :func:`_merge_cloud_fraction_camera` or :func:`_merge_computed_variables` raises; or
        if a declared variable no input produced is left without ``allow_placeholders``
        (see :func:`_fill_placeholder_variables`).
    MissingGeolocationError
        If a record's ephemeris is unusable (see :func:`build_radiometer_footprints`).
    """
    if mode in _CAMERA_TIMESCALE_MODES:
        raise ValueError(
            f"{mode.value} is a camera-timescale mode and cannot be assembled from L1B radiometer pass-through "
            f"inputs; use the camera pseudo-footprint path instead."
        )

    time_variable = fmatch_time_variable(mode)  # "RADIOMETER_TIME"

    # Fail with the specific missing names rather than a bare KeyError deep in the
    # loop below, because a partial L1B dict is the most likely caller error.
    required = {time_variable, *L1B_PASSTHROUGH_VARIABLES, *L1B_SCAN_REFERENCE_VARIABLES}
    missing = sorted(required - set(l1b_inputs))
    if missing:
        raise ValueError(f"L1B inputs for {mode.value} are missing required key(s): {', '.join(missing)}")

    n_footprints = len(l1b_inputs[time_variable])
    if n_footprints == 0:
        raise ValueError(f"Cannot assemble a {mode.value} product from zero footprints.")
    inconsistent = sorted(name for name in required if len(l1b_inputs[name]) != n_footprints)
    if inconsistent:
        raise ValueError(
            f"L1B inputs for {mode.value} have inconsistent lengths; expected {n_footprints} footprints "
            f"(from {time_variable}) but got a different length for: {', '.join(inconsistent)}"
        )

    # Start with the time coordinate, then the real L1B columns cast to the exact
    # dtype the definition declares.
    data: dict[str, np.ndarray] = {
        time_variable: np.asarray(l1b_inputs[time_variable], dtype="datetime64[ns]"),
        **_stack_coordinates(mode, definition),
    }
    for name in L1B_PASSTHROUGH_VARIABLES:
        data[name] = np.asarray(l1b_inputs[name], dtype=np.dtype(definition.variables[name].dtype))

    # Each record's ray-traced PSF bounding box needs only L1B, so it is written on every run;
    # an off-limb record has no box and is written as NaN.
    footprints = build_radiometer_footprints(l1b_inputs)
    boxes = np.full((n_footprints, 4), np.nan)
    for i, footprint in enumerate(footprints):
        if footprint.bbox is not None:
            box = footprint.bbox
            boxes[i] = (box.lat_min, box.lat_max, _normalize_longitude(box.lon_min), _normalize_longitude(box.lon_max))
    for column, name in enumerate(("PSF_Bbox_Lat_Min", "PSF_Bbox_Lat_Max", "PSF_Bbox_Lon_Min", "PSF_Bbox_Lon_Max")):
        data[name] = boxes[:, column].astype(np.dtype(definition.variables[name].dtype))

    # Internal (non-reader) Camera Cloud Fraction values (CAM only).
    _merge_cloud_fraction_camera(
        data, definition, cloud_fraction_camera, n_footprints=n_footprints, allow_placeholders=allow_placeholders
    )

    # Computed columns: derived geometry (always) and, when ancillary inputs are
    # supplied, the external aggregated variables + coverage/QA.
    _merge_computed_variables(
        data,
        definition,
        mode,
        footprints,
        solar_zenith_angle=np.asarray(l1b_inputs["Solar_Zenith_Surface"], dtype=float),
        viewing_zenith_angle=np.asarray(l1b_inputs["Viewing_Zenith_Surface"], dtype=float),
        relative_azimuth_angle=np.asarray(l1b_inputs["Relative_Azimuth_Surface"], dtype=float),
        source_file_paths=source_file_paths,
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )

    _fill_placeholder_variables(data, definition, n_footprints, allow_placeholders=allow_placeholders)
    return data


def write_fmatch_product(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    output_path: str | Path,
    *,
    algorithm_version: str,
    input_files: str,
    cloud_fraction_camera: np.ndarray | None = None,
    source_file_paths: dict[str, Sequence[Path]] | None = None,
    sources: dict[str, AncillaryReader] | None = None,
    weigher: PixelWeigher | None = None,
    allow_placeholders: bool = False,
) -> LiberaDataProductFilename:
    """Assemble and write one FMATCH NetCDF product for an operational mode.

    Loads the product definition once (so assembly and writing cannot disagree about
    it), assembles the product's arrays as :func:`assemble_fmatch_dataset` does, and
    hands them, with ``algorithm_version`` and ``InputGranules`` as dynamic attributes,
    to ``libera_utils.io.netcdf.write_libera_data_product``, which builds the Dataset,
    enforces and checks its conformance once, and writes it using
    ``time_variable=fmatch_time_variable(mode)`` (``RADIOMETER_TIME`` or
    ``CAMERA_TIME``), so the standardized Libera filename encodes the footprint time
    span.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    inputs : Sequence[PseudoFootprint] or dict[str, np.ndarray]
        The mode's assembly inputs: camera pseudo-footprints for the camera-timescale
        modes, or the L1B dict for the radiometer-timescale modes.
    output_path : str or pathlib.Path
        Directory (or S3 prefix) to write the product file into.
    algorithm_version : str
        Value for the ``algorithm_version`` global attribute.
    input_files : str
        Provenance string for the ``InputGranules`` global attribute.
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint Camera Cloud Fraction values (Libera WFOV), as in
        :func:`assemble_fmatch_dataset`: required for the CAM modes unless
        ``allow_placeholders`` is ``True``, and ``None`` for the IMAGER modes.
    source_file_paths : dict[str, Sequence[Path]], optional
        Reader-key -> staged ancillary granule(s) map the sources are built from. Pass this or
        ``sources``, not both.
    sources : dict[str, AncillaryReader], optional
        Pre-built readers to aggregate from, keyed by reader registry key: exactly the readers
        active in ``mode``. Pass this or ``source_file_paths``, not both.
    weigher : PixelWeigher, optional
        PSF weigher used by the aggregation path. Defaults to
        :class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher` (see
        :func:`aggregate_external_variables`).
    allow_placeholders : bool, optional
        When ``True``, a declared variable no input produced is written as fill instead
        of raising: the external, ``PSF_Coverage_Fraction`` and ``Quality_Flag`` columns
        when neither ``sources`` nor ``source_file_paths`` is given, a reader's columns when
        its source is missing, and ``Cloud_Fraction_Camera`` when ``cloud_fraction_camera``
        is omitted for a CAM mode. Defaults to ``False``.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.

    Raises
    ------
    ValueError
        As :func:`assemble_fmatch_dataset` raises (empty or inconsistent inputs, no
        ancillary inputs without ``allow_placeholders`` or both kinds given, a missing,
        unexpected, wrong-length or out-of-range ``cloud_fraction_camera``, any other
        declared variable no input produced without ``allow_placeholders``, a footprint with
        no viewing zenith angle or no usable cone-angle rate under the default weigher,
        camera pseudo-footprints aggregated without ephemeris, a footprint whose time is NaT
        when a source selects by time, an assembled column the definition does not
        declare), or if the assembled Dataset does not conform to the product definition.
    KeyError
        If the keys of ``sources`` or ``source_file_paths`` are not exactly the readers active
        in ``mode``.
    MissingGeolocationError
        If a radiometer record's ephemeris is unusable (see
        :func:`build_radiometer_footprints`).
    """
    definition = load_fmatch_definition(mode)
    data = _assemble_fmatch_data(
        mode,
        inputs,
        definition,
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=source_file_paths,
        sources=sources,
        weigher=weigher,
        allow_placeholders=allow_placeholders,
    )
    return write_libera_data_product(
        data_product_definition=definition,
        data=data,
        output_path=output_path,
        time_variable=fmatch_time_variable(mode),
        dynamic_product_attributes={"algorithm_version": algorithm_version, "InputGranules": input_files},
        strict=True,
    )
