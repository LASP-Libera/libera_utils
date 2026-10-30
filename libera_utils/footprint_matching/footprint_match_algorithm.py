"""FMATCH algorithm runners for the Libera radiometer and WFOV camera.

This module is the single home for every FMATCH runner. The five runners (CAM, CAM-CAMTIME,
IMAGER, IMAGER-CAMTIME, IMAGER-FLASH) are structurally identical: read an input manifest, locate the
staged ancillary granules, keep the L1B input files of a particular product, run footprint matching
on each, write the resulting FMATCH product, and emit an output manifest. They differ only by a
handful of parameters:

* which :class:`~libera_utils.footprint_matching.types.OperationalMode` they produce
  (which in turn drives the product definition, the time coordinate, and the active
  reader set),
* which product counts as their input (L1B ``RAD-4CH`` for the radiometer-timescale modes,
  CF-CAM-CAMTIME for the camera-timescale modes),
* whether they consume the optional Camera Cloud Fraction product, and
* for camera-timescale modes, whether they carry the cloud fraction read with the pseudo-footprints.

Rather than duplicate the manifest/dropbox plumbing per runner, the shared body lives here in
:func:`run_algorithm` and is parameterized by a small :class:`FmatchRunnerConfig`. The concrete
runners are just the five :data:`FmatchRunnerConfig` values collected in :data:`RUNNER_CONFIGS`,
keyed by variant name (``"cam"``, ``"cam-camtime"``, ``"imager"``, ``"imager-camtime"``,
``"imager-flash"``). Every variant runs from the one ``libera-utils fmatch <manifest>`` command, and so
from one container image: :func:`fmatch_cli_handler` selects the config named by the
``LIBERA_FMATCH_VARIANT`` environment variable, which each processing step sets. A new FMATCH variant
is one config plus one registry entry, no new module. This mirrors
``libera_utils/scene_identification/scene_id_algorithm.py`` deliberately, so the algorithm families
read the same way.

The camera pseudo-footprints are segmented by the Camera Cloud Fraction algorithm, which is maintained
outside this package; both camera-timescale runners rebuild them from its CF-CAM-CAMTIME product
(:func:`~libera_utils.footprint_matching.product.pseudofootprints_from_camtime_dataset`).

Environment
-----------
Three environment variables are required at run time:

``LIBERA_FMATCH_VARIANT``
    The :data:`RUNNER_CONFIGS` entry the CLI runs (e.g. ``cam``, ``imager-camtime``).
``PROCESSING_PATH``
    The dropbox directory (or S3 prefix) products and the output manifest are written
    to. This is the standard Libera runner convention.
``FMATCH_ANCILLARY_PATH``
    Root of the staged ancillary granule tree; see
    :func:`resolve_ancillary_inputs` for its layout and "Ancillary inputs" below.

One optional variable tunes the run: ``FMATCH_WORKERS`` (also a ``config.json`` key,
default 1) is the number of worker processes the external aggregation runs the
footprints on; see :func:`~libera_utils.footprint_matching.product.aggregate_external_variables`.

Ancillary inputs
----------------
Every reader active for the mode needs at least one local granule staged under
``FMATCH_ANCILLARY_PATH`` (see :func:`resolve_ancillary_inputs`). The runner hands those
files to the product assembly, where the PSF aggregation engine computes the external
variables and the coverage/QA columns from them. A run with ``FMATCH_ANCILLARY_PATH``
unset, a missing or empty reader directory, or a granule staged in S3 raises rather
than writing those columns as fill values. ``TODO[LIBSDC-785]``: materialize S3-staged
ancillary granules locally (readers need real files).
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path

# Importing the readers subpackage is what populates ReaderRegistry: every concrete
# reader self-registers at class-definition time via AncillaryReader.__init_subclass__,
# which only happens once its module has been imported. Without this import the
# registry would be empty and every mode would resolve to zero readers.
import libera_utils.footprint_matching.readers  # noqa: F401  (imported for its registration side effect)
from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.product import (
    FMATCH_CONE_ANGLE_RATE_KEY,
    L1B_CONE_ANGLE_RATE_VARIABLE,
    L1B_PASSTHROUGH_VARIABLES,
    L1B_SCAN_REFERENCE_VARIABLES,
    camtime_real_cell_mask,
    fmatch_time_variable,
    is_camera_timescale_mode,
    pseudofootprints_from_camtime_dataset,
    write_fmatch_product,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import is_s3, smart_copy_file
from libera_utils.logutil import configure_task_logging

if TYPE_CHECKING:
    from collections.abc import Callable

    from libera_utils.footprint_matching.types import PseudoFootprint

    # A camera-timescale footprint loader turns one local CF-CAM-CAMTIME file into this mode's
    # pseudo-footprints and, for modes that carry it, the per-footprint Camera Cloud Fraction.
    CameraFootprintLoader = Callable[[Path], tuple[list[PseudoFootprint], np.ndarray | None]]

logger = logging.getLogger(__name__)

# Environment variable naming the root of the staged ancillary tree. Mirrors the
# PROCESSING_PATH convention used for the output dropbox.
ANCILLARY_PATH_ENV: str = "FMATCH_ANCILLARY_PATH"

# Name of the time coordinate variable inside the L1B radiometer file. xarray decodes
# its CF "nanoseconds since 1958-01-01" units into datetime64[ns], which is exactly
# the dtype the FMATCH RADIOMETER_TIME coordinate declares.
L1B_TIME_VARIABLE: str = "radiometer_time"

# Name of the FMATCH product's radiometer time coordinate (the key the L1B reader
# returns the decoded L1B times under).
FMATCH_RADIOMETER_TIME_COORDINATE: str = "RADIOMETER_TIME"


@dataclass(frozen=True)
class FmatchRunnerConfig:
    """Everything that distinguishes one FMATCH runner from another.

    Attributes
    ----------
    mode : OperationalMode
        The FMATCH operational mode this runner produces. Drives the product
        definition, the time coordinate, the active reader set, and which assembly
        path (camera- or radiometer-timescale) is taken.
    input_product_id : DataProductIdentifier
        The product that counts as this runner's primary input: ``l1b_rad`` (``RAD-4CH``) for
        radiometer-timescale modes, and the CF-CAM-CAMTIME product (``l2_cf_cam_camtime``) for
        camera-timescale modes, which read their pseudo-footprints (and, for CAM-CAMTIME, the
        cloud fraction) from that product. Manifest files with any other product id (or
        unparsable names) are skipped.
    cloud_fraction_product_id : DataProductIdentifier or None
        The optional SEPARATE Camera Cloud Fraction product merged for radiometer-timescale
        ``cloud_fraction_camera`` (CF-CAM, ``l2_cf_cam``). ``None`` for the camera-timescale modes
        (FMATCH-CAM-CAMTIME's cloud fraction is bundled in ``input_product_id``) and the IMAGER modes.
    log_prefix : str
        Short label used in task-log filenames (e.g. ``fmatch_cam``).
    camera_footprint_loader : CameraFootprintLoader or None
        For camera-timescale modes, the callable that turns one local input file into
        ``(pseudo-footprints, cloud_fraction_camera_or_None)``. Both read CF-CAM-CAMTIME;
        FMATCH-CAM-CAMTIME's also returns the cloud fraction. ``None`` (and unused) for
        radiometer-timescale modes.
    """

    mode: OperationalMode
    input_product_id: DataProductIdentifier
    cloud_fraction_product_id: DataProductIdentifier | None
    log_prefix: str
    camera_footprint_loader: CameraFootprintLoader | None = None

    @property
    def is_camera_timescale(self) -> bool:
        """True when this runner's mode is indexed on ``CAMERA_TIME`` rather than ``RADIOMETER_TIME``."""
        return is_camera_timescale_mode(self.mode)

    @property
    def time_variable(self) -> str:
        """Name of the per-footprint time coordinate in the written product."""
        return fmatch_time_variable(self.mode)


# --- Camera-timescale footprint loaders ----------------------------------------------------------------------------
#
# Each camera-timescale config points its camera_footprint_loader at one of these. Both rebuild the pseudo-footprints
# from a CF-CAM-CAMTIME file; they differ only in whether the mode carries the Camera Cloud Fraction.


def _load_cam_camtime_footprints(local_input_path: Path) -> tuple[list[PseudoFootprint], np.ndarray]:
    """Rebuild pseudo-footprints and cloud fraction from a local CF-CAM-CAMTIME file.

    The camera-timescale footprint loader for FMATCH-CAM-CAMTIME. The cloud fraction is flattened with
    the same real-cell mask as the footprints, so it aligns 1:1 with them.

    Parameters
    ----------
    local_input_path : pathlib.Path
        Local path to a CF-CAM-CAMTIME NetCDF file.

    Returns
    -------
    tuple[list[PseudoFootprint], numpy.ndarray]
        The pseudo-footprints and their per-footprint ``cloud_fraction_camera`` values.

    Raises
    ------
    MissingCamtimeVariablesError
        If the file lacks a variable needed to rebuild the pseudo-footprints.
    ValueError
        If the file has no ``cloud_fraction`` variable.
    """
    with xr.open_dataset(local_input_path) as cf_dataset:
        cf_dataset = cf_dataset.load()
    footprints = pseudofootprints_from_camtime_dataset(cf_dataset)
    if "cloud_fraction" not in cf_dataset:
        raise ValueError(
            f"CF-CAM-CAMTIME input {local_input_path} has no cloud_fraction variable, which FMATCH-CAM-CAMTIME "
            "writes as cloud_fraction_camera. Regenerate the input with the Camera Cloud Fraction algorithm so it "
            "carries cloud_fraction."
        )
    # Boolean-mask indexing returns row-major order, matching the reconstruction order.
    mask = camtime_real_cell_mask(cf_dataset)
    cloud_fraction_camera = np.asarray(cf_dataset["cloud_fraction"].values, dtype=np.float32)[mask]
    return footprints, cloud_fraction_camera


def _load_imager_camtime_footprints(local_input_path: Path) -> tuple[list[PseudoFootprint], None]:
    """Rebuild pseudo-footprints from a local CF-CAM-CAMTIME file, without its cloud fraction.

    The camera-timescale footprint loader for FMATCH-IMAGER-CAMTIME, which takes its cloud information
    from the imager readers and declares no ``cloud_fraction_camera``.

    Parameters
    ----------
    local_input_path : pathlib.Path
        Local path to a CF-CAM-CAMTIME NetCDF file.

    Returns
    -------
    tuple[list[PseudoFootprint], None]
        The pseudo-footprints and ``None`` (no cloud fraction).

    Raises
    ------
    MissingCamtimeVariablesError
        If the file lacks a variable needed to rebuild the pseudo-footprints.
    """
    with xr.open_dataset(local_input_path) as cf_dataset:
        footprints = pseudofootprints_from_camtime_dataset(cf_dataset.load())
    return footprints, None


# --- Per-variant runner configs ------------------------------------------------------------------------------------
#
# Each config below is a complete FMATCH runner. The comment on each records the domain rationale for its parameter
# choices (latency rank, active reader set, cloud-fraction source); the shared engine (run_algorithm and friends) is
# otherwise identical across all five.

# FMATCH-CAM (radiometer timescale): the lowest-latency (camera / near-real-time) product; runs continuously from
# mission start. Input is the L1B RAD-4CH product; the optional cloud fraction comes from CF-CAM (also radiometer
# timescale). No RBSP-sourced readers are active at its latency rank.
CAM_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.CAM,
    input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=DataProductIdentifier.l2_cf_cam,
    log_prefix="fmatch_cam",
)

# FMATCH-CAM-CAMTIME (camera timescale): the camera-timescale near-real-time product. Its input is CF-CAM-CAMTIME,
# which carries the pseudo-footprints and the cloud fraction; _load_cam_camtime_footprints reads both.
CAM_CAMTIME_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.CAM_CAMTIME,
    input_product_id=DataProductIdentifier.l2_cf_cam_camtime,
    cloud_fraction_product_id=None,  # bundled in input_product_id (CF-CAM-CAMTIME)
    log_prefix="fmatch_cam_camtime",
    camera_footprint_loader=_load_cam_camtime_footprints,
)

# FMATCH-IMAGER (radiometer timescale): the RBSP Climate Quality product -- the highest latency rank of the
# radiometer-timed modes and thus the largest active reader set (RBSP CLDPIX/FLASHFlux + ERA5 single/pressure + VIIRS).
# It declares no cloud_fraction_camera (cloud info comes from the imager readers), so it takes no cloud-fraction input.
IMAGER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER,
    input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=None,
    log_prefix="fmatch_imager",
)

# FMATCH-IMAGER-CAMTIME (camera timescale): the RBSP Climate Quality camera-timescale product; requires RBSP inputs so
# it does not run during the first operational year. Like CAM-CAMTIME it reads its pseudo-footprints from
# CF-CAM-CAMTIME, but it declares no cloud_fraction_camera (cloud info comes from the imager readers), so
# _load_imager_camtime_footprints leaves the cloud fraction unread.
IMAGER_CAMTIME_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER_CAMTIME,
    input_product_id=DataProductIdentifier.l2_cf_cam_camtime,
    cloud_fraction_product_id=None,
    log_prefix="fmatch_imager_camtime",
    camera_footprint_loader=_load_imager_camtime_footprints,
)

# FMATCH-IMAGER-FLASH (radiometer timescale): the RBSP Flash-latency product, between near-real-time CAM and
# climate-quality IMAGER (more ancillary readers than CAM, but RBSP-dependent so no first-year running). Declares no
# cloud_fraction_camera (cloud info from the imager readers), so it takes no cloud-fraction input.
IMAGER_FLASH_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER_FLASH,
    input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=None,
    log_prefix="fmatch_imager_flash",
)

# Registry of every FMATCH runner, keyed by the variant name the LIBERA_FMATCH_VARIANT environment variable selects.
# Adding a variant is one config plus one entry.
RUNNER_CONFIGS: dict[str, FmatchRunnerConfig] = {
    "cam": CAM_CONFIG,
    "cam-camtime": CAM_CAMTIME_CONFIG,
    "imager": IMAGER_CONFIG,
    "imager-camtime": IMAGER_CAMTIME_CONFIG,
    "imager-flash": IMAGER_FLASH_CONFIG,
}

# Environment variable naming the RUNNER_CONFIGS entry to run; set per processing step.
FMATCH_VARIANT_ENV = "LIBERA_FMATCH_VARIANT"


def resolve_runner_config_from_env() -> FmatchRunnerConfig:
    """Select the FMATCH runner config named by the ``LIBERA_FMATCH_VARIANT`` environment variable.

    Returns
    -------
    FmatchRunnerConfig
        The :data:`RUNNER_CONFIGS` entry for the named variant.

    Raises
    ------
    ValueError
        If the variable is not set or names no entry in :data:`RUNNER_CONFIGS`.
    """
    variant = os.getenv(FMATCH_VARIANT_ENV, "").strip()
    if not variant:
        raise ValueError(f"{FMATCH_VARIANT_ENV} environment variable is not set")
    if variant not in RUNNER_CONFIGS:
        raise ValueError(
            f"{FMATCH_VARIANT_ENV}={variant!r} is not a FMATCH variant; expected one of {sorted(RUNNER_CONFIGS)}"
        )
    return RUNNER_CONFIGS[variant]


def fmatch_cli_handler(parsed_args: argparse.Namespace) -> Path | S3Path:
    """Run the FMATCH variant selected by ``LIBERA_FMATCH_VARIANT`` from an input manifest.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Parsed CLI arguments. Uses ``parsed_args.manifest`` (the input manifest path).

    Returns
    -------
    pathlib.Path | cloudpathlib.S3Path
        Path to the written output manifest file.

    Raises
    ------
    ValueError
        If ``LIBERA_FMATCH_VARIANT`` is not set or names no FMATCH variant.
    """
    runner_config = resolve_runner_config_from_env()
    return run_algorithm(AnyPath(parsed_args.manifest), runner_config)


def run_algorithm(manifest_path: Path | S3Path, config: FmatchRunnerConfig) -> Path | S3Path:
    """Run a FMATCH processing workflow from an input manifest.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the L1B input file(s).
    config : FmatchRunnerConfig
        The per-runner parameters (mode, input products, log label).

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.

    Raises
    ------
    ValueError
        If the ``PROCESSING_PATH`` or ``FMATCH_ANCILLARY_PATH`` environment variable is not
        set, if the manifest references no input of ``config.input_product_id``, if an
        ancillary granule is staged in S3, if a FMATCH-CAM-CAMTIME input has no
        ``cloud_fraction``, or if an L1B input has no usable footprints.
    FileNotFoundError
        If the ancillary root or an active reader's directory under it is missing or empty.
    MissingCamtimeVariablesError
        If a camera-timescale input lacks a variable needed to rebuild the pseudo-footprints.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"{config.log_prefix}_{now}")

    # Step 1: Read the input manifest.
    logger.info("Step 1: Reading the input manifest file")
    input_manifest = Manifest.from_file(AnyPath(manifest_path))
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")
    logger.info("Running %s", config.mode.value)

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Locate the staged ancillary granules for this mode.
    #
    # These are resolved, inventoried in the log, and threaded into product assembly,
    # where the PSF aggregation engine computes the external variables and coverage/QA
    # columns from them. Logging the inventory first lets an operator confirm staging is
    # correct from the task log. A missing root or reader directory raises here.
    logger.info("Step 2: Locating staged ancillary inputs")
    ancillary_inputs = resolve_ancillary_inputs(config.mode)
    log_ancillary_inventory(ancillary_inputs)

    # Step 3: Collect the L1B input file(s) from the manifest.
    input_label = config.input_product_id.value
    logger.info("Step 3: Collecting %s input files from the manifest", input_label)
    input_file_paths = select_manifest_files_by_product_id(input_manifest, config.input_product_id)
    if not input_file_paths:
        raise ValueError(f"No {input_label} input files found in the input manifest")

    cloud_fraction_file_paths = _collect_cloud_fraction_files(input_manifest, config)

    # Step 4: Run footprint matching and write data products.
    logger.info("Step 4: Running footprint matching and writing data products")
    output_data_file_paths: list[LiberaDataProductFilename] = []
    for input_file_path in input_file_paths:
        output_file = create_and_write_data_product(
            input_file_path=input_file_path,
            output_path=dropbox_path,
            config=config,
            cloud_fraction_file_paths=cloud_fraction_file_paths,
            ancillary_inputs=ancillary_inputs,
        )
        output_data_file_paths.append(output_file)

    # Step 5: Create the output manifest from the input manifest.
    logger.info("Step 5: Creating the output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 6: Register the written data product file(s) on the output manifest.
    logger.info(f"Step 6: Adding {len(output_data_file_paths)} data file(s) to the output manifest")
    output_manifest.add_files(*[output_file.path for output_file in output_data_file_paths])

    # Step 7: Write the output manifest to the dropbox.
    logger.info("Step 7: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def _collect_cloud_fraction_files(input_manifest: Manifest, config: FmatchRunnerConfig) -> list[str]:
    """Select the optional Camera Cloud Fraction input files from the manifest.

    Only the CAM-family modes declare ``cloud_fraction_camera``; for the IMAGER modes
    this returns an empty list without inspecting the manifest. A CAM run with no
    cloud-fraction file staged is legal - the variable is then written as a
    placeholder - so a missing input is logged, not raised.
    """
    if config.cloud_fraction_product_id is None:
        return []
    paths = select_manifest_files_by_product_id(input_manifest, config.cloud_fraction_product_id)
    if not paths:
        logger.warning(
            "No %s input found in the manifest; cloud_fraction_camera will be written as a placeholder.",
            config.cloud_fraction_product_id.value,
        )
    return paths


def resolve_ancillary_inputs(
    mode: OperationalMode,
    root: str | Path | S3Path | None = None,
    *,
    strict: bool = True,
) -> dict[str, list[Path | S3Path]]:
    """Map each reader active for ``mode`` to its staged granule files.

    Footprint matching aggregates external, non-Libera datasets (ERA5 reanalysis, IGBP
    land cover, NISE snow/ice, VIIRS products, and - post year one - CERES CLDPIX/FLASHFlux)
    onto each radiometer footprint. Those granules are *not* Libera data products: they
    have third-party filenames, so they cannot be selected out of a manifest the way
    :func:`select_manifest_files_by_product_id` selects L1B inputs.

    The pipeline stages ancillary granules into a directory tree with **one
    subdirectory per reader, named by the reader's registry key**::

        $FMATCH_ANCILLARY_PATH/
            era5_surface/       ERA5 single-level granules
            era5_pressure/      ERA5 pressure-level granules
            igbp/               MCD12Q1 land-cover granules
            nise/               NISE snow/ice granules
            viirs_brdf/         VIIRS BRDF granules
            viirs_cldprop_d3/   VIIRS daily L3 cloud-property granules
            viirs_aod/          VIIRS Deep Blue aerosol granules (AOD + aerosol type)
            cldpix/             CERES CLDPIX granules    (IMAGER-family modes)
            flashflux/          CERES FLASHFlux granules (IMAGER-family modes)

    The active reader set comes from
    :meth:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry.get_readers_for_mode`,
    so this automatically tracks the mode's latency rank: FMATCH-CAM looks for five
    readers, while FMATCH-IMAGER additionally looks for ``era5_pressure``,
    ``viirs_aod``, and the RBSP ``cldpix``/``flashflux`` inputs. Using the registry keys as
    directory names means the set of directories required for a run is *derived* from
    the registry rather than hard-coded here.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being run.
    root : str | pathlib.Path | cloudpathlib.S3Path, optional
        Root of the staged ancillary tree. Defaults to the ``FMATCH_ANCILLARY_PATH``
        environment variable.
    strict : bool, optional
        How to treat missing inputs. When True (the default) a missing root, a missing
        reader subdirectory, or an empty one raises. When False each is logged as a
        warning and yields an empty list.

    Returns
    -------
    dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        Mapping of reader registry key to the granule files staged for it, sorted by
        name so repeated runs process files in a deterministic order. Readers with no
        staged files map to an empty list (only when ``strict`` is False).

    Raises
    ------
    ValueError
        If ``root`` is not given and ``FMATCH_ANCILLARY_PATH`` is unset, when ``strict``.
    FileNotFoundError
        If the root or a required reader subdirectory does not exist, or a required
        subdirectory is empty, when ``strict``.
    """
    active_readers = ReaderRegistry.get_readers_for_mode(mode)

    ancillary_root = _resolve_ancillary_root(root, mode=mode, strict=strict)
    if ancillary_root is None:
        return {key: [] for key in sorted(active_readers)}

    resolved: dict[str, list[Path | S3Path]] = {}
    for reader_key in sorted(active_readers):
        resolved[reader_key] = _resolve_reader_directory(ancillary_root, reader_key, strict=strict)
    return resolved


def _resolve_ancillary_root(
    root: str | Path | S3Path | None,
    *,
    mode: OperationalMode,
    strict: bool,
) -> Path | S3Path | None:
    """Resolve and validate the ancillary tree root, or return None when absent and not strict."""
    if root is None:
        root = os.getenv(ANCILLARY_PATH_ENV)
    if not root:
        message = (
            f"{ANCILLARY_PATH_ENV} environment variable is not set, so no ancillary inputs can be located for "
            f"{mode.value}."
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s Continuing with no ancillary inputs.", message)
        return None

    # AnyPath gives a local Path or an S3Path depending on the string, so a staged
    # tree works equally well on local disk and in S3.
    ancillary_root = AnyPath(root)
    if not ancillary_root.exists():
        message = f"Ancillary root directory does not exist: {ancillary_root}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s Continuing with no ancillary inputs.", message)
        return None
    return ancillary_root


def _resolve_reader_directory(
    ancillary_root: Path | S3Path,
    reader_key: str,
    *,
    strict: bool,
) -> list[Path | S3Path]:
    """List the granule files staged under one reader's subdirectory."""
    reader_directory = ancillary_root / reader_key
    if not reader_directory.exists():
        message = f"No staged ancillary directory for reader '{reader_key}': {reader_directory}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s (expected %s/%s/)", message, ANCILLARY_PATH_ENV, reader_key)
        return []

    # Sorted for determinism; skip nested directories so a reader subdirectory can
    # hold per-day subfolders in the future without those being mistaken for granules.
    files: list[Path | S3Path] = sorted((entry for entry in reader_directory.iterdir() if entry.is_file()), key=str)
    if not files:
        message = f"Ancillary directory for reader '{reader_key}' is empty: {reader_directory}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s", message)
    return files


def log_ancillary_inventory(ancillary_inputs: dict[str, list[Path | S3Path]]) -> None:
    """Log a one-line-per-reader inventory of the staged ancillary inputs.

    Written so an operator can diagnose a staging problem from the task log alone,
    without shelling into the container: every active reader appears, and the ones
    with nothing staged are called out explicitly rather than being silently absent.

    Parameters
    ----------
    ancillary_inputs : dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        The mapping returned by :func:`resolve_ancillary_inputs`.
    """
    if not ancillary_inputs:
        logger.warning("No ancillary readers are active for this run.")
        return
    for reader_key, files in sorted(ancillary_inputs.items()):
        if files:
            logger.info("Ancillary input '%s': %d file(s)", reader_key, len(files))
        else:
            logger.warning("Ancillary input '%s': 0 file(s) -- MISSING", reader_key)


def run_footprint_matching(
    input_file_path: str | Path | S3Path,
    config: FmatchRunnerConfig,
) -> tuple[list[PseudoFootprint] | dict[str, np.ndarray], np.ndarray | None]:
    """Read one input file and produce this mode's footprint assembly inputs.

    The two timescales read different inputs and produce different things, which is why the
    first element of the returned tuple is a union:

    * **camera-timescale** modes read the CF-CAM-CAMTIME product (their sole camera input) and
      rebuild the list of
      :class:`~libera_utils.footprint_matching.types.PseudoFootprint` objects from it via
      :func:`~libera_utils.footprint_matching.product.pseudofootprints_from_camtime_dataset`.
      The per-footprint Camera Cloud Fraction is read from the same product and returned as the
      second element (``None`` for modes whose output does not declare ``cloud_fraction_camera``,
      e.g. IMAGER-CAMTIME);
    * **radiometer-timescale** modes return the dict of L1B pass-through arrays (time, geolocation,
      viewing angles), whose footprints are the L1B radiometer footprints themselves, and ``None``
      cloud fraction (the radiometer CF-CAM merge is handled separately from the manifest).

    Parameters
    ----------
    input_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to this mode's input file: the CF-CAM-CAMTIME product for camera-timescale
        modes, the L1B Daily radiometer file for radiometer-timescale modes.
    config : FmatchRunnerConfig
        Runner parameters supplying the operational mode.

    Returns
    -------
    tuple[list[PseudoFootprint] | dict[str, np.ndarray], np.ndarray | None]
        The mode's assembly inputs (footprints or L1B pass-through dict) and the per-footprint
        ``cloud_fraction_camera`` values (or ``None``), ready to hand to
        :func:`~libera_utils.footprint_matching.product.write_fmatch_product`.

    Raises
    ------
    ValueError
        If a camera-timescale mode has no ``camera_footprint_loader`` configured, if a
        FMATCH-CAM-CAMTIME input has no ``cloud_fraction``, or if a radiometer-timescale L1B
        file has no usable footprints (raised by :func:`load_l1b_radiometer_inputs`).
    MissingCamtimeVariablesError
        If a camera-timescale input lacks a variable needed to rebuild the pseudo-footprints.

    Notes
    -----
    Both paths use :func:`xarray.open_dataset`, which needs a real local file, so an S3 input is first
    materialized to a temporary file by :class:`_as_local_path`; local inputs are read in place.
    """
    with _as_local_path(input_file_path) as local_input_path:
        if config.is_camera_timescale:
            if config.camera_footprint_loader is None:
                raise ValueError(
                    f"Camera-timescale mode {config.mode.value} has no camera_footprint_loader configured."
                )
            footprints, cloud_fraction_camera = config.camera_footprint_loader(local_input_path)
            logger.info("Loaded %d camera pseudo-footprints", len(footprints))
            return footprints, cloud_fraction_camera

        logger.info("Reading L1B radiometer pass-through inputs from %s", local_input_path)
        l1b_inputs = load_l1b_radiometer_inputs(local_input_path)
        logger.info("Read %d radiometer footprints", len(l1b_inputs[config.time_variable]))
        return l1b_inputs, None


def _ancillary_source_file_paths(
    ancillary_inputs: dict[str, list[Path | S3Path]],
) -> dict[str, list[Path]]:
    """Reduce the resolved ancillary inventory to the per-reader local granule lists assembly needs.

    :func:`~libera_utils.footprint_matching.footprint_match_algorithm.resolve_ancillary_inputs` returns
    a *list* of staged granules per reader, and each reader takes the granule(s) of its
    source for the period (a reader that reads one granule per period raises when handed
    more). This keeps every staged granule as a local path.

    Parameters
    ----------
    ancillary_inputs : dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        The per-reader inventory from ``resolve_ancillary_inputs``.

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Reader-key -> local granule(s).

    Raises
    ------
    ValueError
        If the inventory is empty, if a reader has no staged granule, or if a granule is in S3.
    """
    if not ancillary_inputs:
        raise ValueError(
            "No ancillary inputs were given, so the external variables cannot be computed. Pass the inventory "
            f"from resolve_ancillary_inputs, with granules staged under ${ANCILLARY_PATH_ENV}/<reader key>/."
        )
    source_file_paths: dict[str, list[Path]] = {}
    for reader_key, files in ancillary_inputs.items():
        if not files:
            raise ValueError(
                f"Ancillary source {reader_key!r} has no staged granule. Stage at least one granule under "
                f"${ANCILLARY_PATH_ENV}/{reader_key}/."
            )
        remote = [f for f in files if is_s3(f)]
        if remote:
            # TODO[LIBSDC-785]: materialize S3-staged ancillary granules locally instead of raising.
            raise ValueError(
                f"Ancillary source {reader_key!r} is staged in S3 ({remote[0]}), but the readers need local files. "
                f"Stage the ancillary tree on local disk and point ${ANCILLARY_PATH_ENV} at it."
            )
        source_file_paths[reader_key] = [Path(str(f)) for f in files]
    return source_file_paths


def create_and_write_data_product(
    input_file_path: str | Path | S3Path,
    output_path: str | Path | S3Path,
    config: FmatchRunnerConfig,
    ancillary_inputs: dict[str, list[Path | S3Path]],
    cloud_fraction_file_paths: list[str] | None = None,
) -> LiberaDataProductFilename:
    """Run footprint matching on one input file and write the FMATCH data product.

    Parameters
    ----------
    input_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to this mode's input file: the CF-CAM-CAMTIME product for camera-timescale
        modes, the L1B Daily radiometer file for radiometer-timescale modes.
    output_path : str | pathlib.Path | cloudpathlib.S3Path
        Directory / prefix in the processing dropbox where the product file is written.
    config : FmatchRunnerConfig
        Runner parameters supplying the mode and input products.
    ancillary_inputs : dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        The staged ancillary inventory from
        :func:`~libera_utils.footprint_matching.footprint_match_algorithm.resolve_ancillary_inputs`:
        at least one local granule for every reader active for the mode.
    cloud_fraction_file_paths : list[str], optional
        SEPARATE radiometer-timescale Camera Cloud Fraction files (CF-CAM) selected from the manifest,
        if any. Reading them is not implemented yet (``TODO[LIBSDC-785]``); they are logged as
        provenance and the ``cloud_fraction_camera`` variable is written as a placeholder. (Camera-
        timescale modes read real cloud fraction from ``input_file_path`` instead.)

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.

    Raises
    ------
    ValueError
        If ``ancillary_inputs`` is empty, a reader in it has no granule, or a granule is in S3; or
        for any condition :func:`run_footprint_matching` raises it for.
    MissingCamtimeVariablesError
        If a camera-timescale input lacks a variable needed to rebuild the pseudo-footprints.

    Notes
    -----
    The external-reader aggregates (``igbp_surface_type``, ``era5_*``, ``flashflux_*``,
    ``cldpix_*``, ...) and the ``psf_coverage_fraction`` and ``q_flags`` columns are computed by
    the PSF aggregation engine from ``ancillary_inputs``. The one placeholder is the
    radiometer-timescale ``cloud_fraction_camera``, written as fill until the CF-CAM read
    lands (``TODO[LIBSDC-785]``); camera-timescale modes carry the real cloud fraction from
    their CF-CAM-CAMTIME input.
    """
    inputs, cloud_fraction_camera = run_footprint_matching(input_file_path, config)

    input_file_name = AnyPath(input_file_path).name
    # Provenance records every file that fed this product, so a written file names its own inputs.
    provenance = [input_file_name]
    # Radiometer-timescale CAM merges its cloud fraction from a SEPARATE CF-CAM product listed in the
    # manifest; reading it is still future work (camera-timescale modes already carry real cloud
    # fraction from their CF-CAM-CAMTIME input, read in run_footprint_matching).
    if cloud_fraction_file_paths:
        provenance.extend(AnyPath(path).name for path in cloud_fraction_file_paths)
        # TODO[LIBSDC-785]: read the radiometer-timescale CF-CAM values and pass them as cloud_fraction_camera.
        logger.info(
            "Radiometer-timescale Camera Cloud Fraction input(s) present but not yet ingested; "
            "cloud_fraction_camera will be written as a placeholder."
        )

    source_file_paths = _ancillary_source_file_paths(ancillary_inputs)
    provenance.extend(path.name for paths in source_file_paths.values() for path in paths)
    logger.info("Aggregating external variables from %d staged ancillary source(s)", len(source_file_paths))

    logger.info("Writing %s data product for input %s", config.mode.value, input_file_name)
    output_file_path = write_fmatch_product(
        config.mode,
        inputs,
        output_path,
        algorithm_version=algorithm_version(),
        input_files=",".join(provenance),
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=source_file_paths,
        strict=True,
    )
    logger.info("Wrote %s data product to %s", output_file_path.data_product_id.value, output_file_path.path)
    return output_file_path


def select_manifest_files_by_product_id(manifest: Manifest, *product_ids: DataProductIdentifier) -> list[str]:
    """Select the files referenced by a manifest that belong to the given Libera data product(s).

    A runner receives one manifest listing every file staged for the processing step, which in
    general contains more than the runner consumes (an L1B input, an optional cloud-fraction product,
    ancillary granules, sibling products). This helper keeps exactly the records whose parsed Libera
    product id is one of ``product_ids``, in manifest order.

    Files whose names do not parse as a :class:`~libera_utils.io.filenaming.LiberaDataProductFilename`
    are skipped rather than raising: manifests legitimately carry non-Libera files (for example the
    externally-sourced ancillary granules the readers consume), and a runner should ignore them, not
    fail on them.

    This lives in the runner, alongside its callers, so that ``libera_utils.io`` stays a generic I/O
    package with no dependency on Libera product identifiers. The SCENE-ID runner keeps its own
    single-product equivalent (``collect_input_files``) for the same reason.

    Parameters
    ----------
    manifest : Manifest
        The manifest to inspect.
    *product_ids : DataProductIdentifier
        One or more Libera product ids to keep. Passing several is useful when a runner takes more
        than one Libera input from the same manifest (e.g. FMATCH-CAM reads both an L1B product and an
        optional cloud-fraction one).

    Returns
    -------
    list[str]
        The matching manifest filenames, in manifest order. Empty when the manifest references none of
        the products.

    Raises
    ------
    ValueError
        If no ``product_ids`` are given, which would silently match nothing and is always a caller bug.
    """
    if not product_ids:
        raise ValueError("At least one DataProductIdentifier must be given to select manifest files.")

    wanted = set(product_ids)
    wanted_labels = ", ".join(sorted(product_id.value for product_id in wanted))
    selected: list[str] = []
    for filename in manifest.files:
        try:
            libera_filename = LiberaDataProductFilename.from_file_path(filename)
        except Exception:
            # Not a Libera product name, so it cannot be one of the requested products.
            logger.info("Skipping non-Libera-product file (not a %s input): %s", wanted_labels, filename)
            continue
        # Parsed as a Libera product; keep it only if it is one of the requested products.
        if libera_filename.data_product_id in wanted:
            logger.info("Recording %s input file: %s", libera_filename.data_product_id.value, filename)
            selected.append(filename)
        else:
            logger.info(
                "Skipping Libera product '%s' (not %s): %s",
                libera_filename.data_product_id.value,
                wanted_labels,
                filename,
            )
    return selected


def load_l1b_radiometer_inputs(l1b_file: Path) -> dict[str, np.ndarray]:
    """Read the per-footprint L1B inputs that FMATCH passes through verbatim.

    Pulls every quantity the FMATCH product contract takes straight from L1B Daily:
    the ``RADIOMETER_TIME`` coordinate plus all variables in
    :data:`~libera_utils.footprint_matching.product.L1B_PASSTHROUGH_VARIABLES`
    (footprint ``latitude``/``longitude`` and the solar/viewing zenith and
    relative-azimuth angles).

    Footprints with non-finite values are dropped. The L1B geolocation and angles are
    NaN wherever the boresight has no valid Earth intersection (e.g. the first samples
    of a file, and any gaps), and such rows carry no usable values. We keep only
    footprints where *every* pass-through variable is finite - a logical AND of the
    per-variable finite masks. In practice the geolocation and the "_Surface" angles
    share the same gaps, but AND-ing all of them is robust if they ever diverge.

    Parameters
    ----------
    l1b_file : pathlib.Path
        Path to a local L1B RAD-4CH NetCDF file. Remote inputs must be materialized
        locally first (see :class:`_as_local_path`), because
        :func:`xarray.open_dataset` seeks within the file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping keyed by FMATCH variable name: ``"RADIOMETER_TIME"`` (datetime64[ns])
        plus each key of :data:`~libera_utils.footprint_matching.product.L1B_PASSTHROUGH_VARIABLES`
        (float32). It additionally carries the scan-reference geometry used to build
        footprints (not product columns): each key of
        :data:`~libera_utils.footprint_matching.product.L1B_SCAN_REFERENCE_VARIABLES` and
        :data:`~libera_utils.footprint_matching.product.FMATCH_CONE_ANGLE_RATE_KEY`
        (float64). All arrays are 1-D and the same length.

    Raises
    ------
    ValueError
        If no footprint has finite values for every pass-through / subsatellite
        variable, which would leave nothing to build a product from.
    """
    # Open with default decoding so the CF-encoded time coordinate ("nanoseconds since
    # 1958-01-01") is decoded into datetime64[ns] for us, and _FillValue-tagged fields
    # (the L1B fills are -999 / -9999) are masked to NaN so the finite check below works.
    with xr.open_dataset(l1b_file) as l1b:
        radiometer_time = l1b[L1B_TIME_VARIABLE].values
        # Read every pass-through variable, keyed by its FMATCH (output) name.
        passthrough = {fmatch_name: l1b[l1b_name].values for fmatch_name, l1b_name in L1B_PASSTHROUGH_VARIABLES.items()}
        # Scan-reference geometry (subsatellite point + cone-angle rate), read at
        # float64 for the ray-trace. Not written to the product; feeds footprint build.
        scan_reference = {
            fmatch_name: np.asarray(l1b[l1b_name].values, dtype=np.float64)
            for fmatch_name, l1b_name in L1B_SCAN_REFERENCE_VARIABLES.items()
        }
        cone_angle_rate = np.asarray(l1b[L1B_CONE_ANGLE_RATE_VARIABLE].values, dtype=np.float64)

    # Keep only footprints where every pass-through variable AND the subsatellite point
    # are finite. Start from an all-True mask and AND in each variable's finite mask, so
    # a NaN in ANY of them drops that footprint. The cone-angle rate is intentionally
    # excluded: an unknown scan rate must not discard an otherwise-good footprint.
    finite = np.ones(radiometer_time.shape, dtype=bool)
    for values in (*passthrough.values(), *scan_reference.values()):
        finite &= np.isfinite(values)

    n_finite = int(finite.sum())
    if n_finite == 0:
        raise ValueError(
            f"No usable footprints in L1B file {l1b_file}: every record has a non-finite value in at least one of "
            f"the pass-through or subsatellite variables "
            f"({', '.join(sorted({*L1B_PASSTHROUGH_VARIABLES, *L1B_SCAN_REFERENCE_VARIABLES}))})."
        )
    if n_finite < finite.size:
        logger.info(
            "Dropped %d of %d L1B footprints with non-finite geolocation/viewing angles",
            finite.size - n_finite,
            finite.size,
        )

    radiometer_time = radiometer_time[finite]
    passthrough = {name: values[finite] for name, values in passthrough.items()}
    scan_reference = {name: values[finite] for name, values in scan_reference.items()}
    cone_angle_rate = cone_angle_rate[finite]

    # This loader is the input boundary, so it is where the FMATCH dtypes are allocated:
    # cast every pass-through variable to the float32 the definition declares (L1B may store
    # them at wider precision) and the time coordinate to datetime64[ns]. Typing here means
    # everything downstream inherits the right dtypes and conformance passes without an auto-cast.
    result: dict[str, np.ndarray] = {
        FMATCH_RADIOMETER_TIME_COORDINATE: radiometer_time.astype("datetime64[ns]"),
    }
    result.update({name: values.astype(np.float32) for name, values in passthrough.items()})
    result.update(scan_reference)
    result[FMATCH_CONE_ANGLE_RATE_KEY] = cone_angle_rate
    return result


def algorithm_version() -> str:
    """Return the version recorded in a product's ``algorithm_version`` attribute.

    Sourced from the installed ``libera_utils`` package metadata, so a written product
    always names the exact release that produced it. Falls back to ``"0.0.0"`` only
    when the package is not installed (e.g. running from a source tree without an
    editable install), which should not happen inside the algorithm container.

    Returns
    -------
    str
        The installed ``libera_utils`` version.
    """
    try:
        return version("libera_utils")
    except PackageNotFoundError:  # pragma: no cover - not reachable in an installed environment
        logger.warning("libera_utils package metadata not found; recording algorithm_version as 0.0.0")
        return "0.0.0"


class _as_local_path:
    """Context manager yielding a real local filesystem path for a possibly-remote input file.

    Reading an L1B product with :func:`xarray.open_dataset` (and the HDF4/HDF5 libraries underneath it)
    requires a real file on a local filesystem, because those libraries seek within the file rather
    than streaming it. A runner's manifest, however, may reference either local paths or S3 object
    urls. For S3 inputs this downloads the object to a temporary directory that is removed on exit;
    local inputs are yielded unchanged with no copy.

    Kept in the runner so that ``libera_utils.io`` stays free of runner-specific plumbing; the SCENE-ID runner keeps its own identical helper for the same reason.

    Parameters
    ----------
    source_path : str | pathlib.Path | cloudpathlib.S3Path
        Path to the input file, local or in S3.
    """

    def __init__(self, source_path: str | Path | S3Path):
        self._source_path = AnyPath(source_path)
        self._tempdir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if is_s3(self._source_path):
            # Materialize the S3 object locally so file-based readers (netCDF4, HDF4/HDF5) can open it.
            self._tempdir = tempfile.TemporaryDirectory()
            local_path = Path(self._tempdir.name) / self._source_path.name
            smart_copy_file(self._source_path, local_path)
            return local_path
        # Already local; hand back a plain pathlib.Path.
        return Path(str(self._source_path))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
