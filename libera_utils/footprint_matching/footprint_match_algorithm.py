"""FMATCH algorithm runners for the Libera radiometer and WFOV camera.

This module is the single home for every FMATCH runner. The four runners (CAM, CAM-CAMTIME,
IMAGER, IMAGER-CAMTIME) are structurally identical: read an input manifest, locate the
ancillary granules it lists, keep the L1B input files of a particular product, run footprint matching
on each, write the resulting FMATCH product, and emit an output manifest. They differ only by a
handful of parameters:

* which :class:`~libera_utils.footprint_matching.types.OperationalMode` they produce
  (which in turn drives the product definition, the time coordinate, and the active
  reader set),
* which product counts as their input (L1B ``RAD-4CH`` for the radiometer-timescale modes,
  CF-CAM-CAMTIME for the camera-timescale modes),
* whether they require the separate Camera Cloud Fraction product (CF-CAM, FMATCH-CAM only), and
* for camera-timescale modes, whether they carry the cloud fraction read with the pseudo-footprints.

Rather than duplicate the manifest/dropbox plumbing per runner, the shared body lives here in
:func:`run_algorithm` and is parameterized by a small :class:`FmatchRunnerConfig`. The concrete
runners are just the four :data:`FmatchRunnerConfig` values collected in ``RUNNER_CONFIGS``,
keyed by variant name (``"cam"``, ``"cam-camtime"``, ``"imager"``, ``"imager-camtime"``). Every variant runs from the one ``libera-utils fmatch <manifest>`` command, and so
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
Two environment variables are required at run time:

``LIBERA_FMATCH_VARIANT``
    The ``RUNNER_CONFIGS`` entry the CLI runs (e.g. ``cam``, ``imager-camtime``).
``PROCESSING_PATH``
    The dropbox directory (or S3 prefix) products and the output manifest are written
    to. This is the standard Libera runner convention.

One optional variable tunes the run: ``FMATCH_WORKERS`` (also a ``config.json`` key,
default 1) is the number of worker processes the external aggregation runs the
footprints on; see :func:`~libera_utils.footprint_matching.product.aggregate_external_variables`.

Ancillary inputs
----------------
The input manifest lists the external ancillary granules alongside the Libera inputs, each
under the name its upstream provider gives it. Every reader active for the mode declares
the ``DataLevel.ANC`` product it reads (``PRODUCT_ID``), and
:func:`select_ancillary_inputs` hands it the manifest's files of that product; a run whose
manifest lacks an active reader's product raises :class:`MissingAncillaryInputError`
before any input is read. Selection is by product, not by time, so the manifest lists
exactly one granule of each daily product (VIIRS aerosol, BRDF and cloud properties, and
NISE) and the hourly granules around the processing period (ERA5, CLDPIX, and the CERES
SSF the CERES SSF reader reads). An ancillary granule in S3 is copied to a local temporary
file once per run, because the readers open real files, and the copy is removed when the
run ends. The PSF aggregation engine computes the external variables and the coverage/QA
columns from these files.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from contextlib import ExitStack
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
from libera_utils.io.filenaming import AbstractDataProductFilename, AbstractValidFilename, LiberaDataProductFilename
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

# Name of the time coordinate variable inside the L1B radiometer file. xarray decodes
# its CF "nanoseconds since 1958-01-01" units into datetime64[ns], which is exactly
# the dtype the FMATCH RADIOMETER_TIME coordinate declares.
L1B_TIME_VARIABLE: str = "radiometer_time"

# Name of the FMATCH product's radiometer time coordinate (the key the L1B reader
# returns the decoded L1B times under).
FMATCH_RADIOMETER_TIME_COORDINATE: str = "RADIOMETER_TIME"

# Name of the per-footprint cloud fraction, in percent, in both Camera Cloud Fraction products: CF-CAM (one value
# per radiometer footprint) and CF-CAM-CAMTIME (one value per camera pseudo-footprint).
CF_CAM_CLOUD_FRACTION_VARIABLE: str = "Cloud_Fraction"

# CF-CAM carries the L1B RAD-4CH time coordinate under its L1B name, and its rows are matched to the L1B footprints
# by exact equality on it.
CF_CAM_TIME_VARIABLE: str = L1B_TIME_VARIABLE


class MissingCloudFractionInputError(ValueError):
    """A FMATCH run that carries the Camera Cloud Fraction has no input carrying it.

    Raised when a FMATCH-CAM run is given no CF-CAM product, or when a CF-CAM or CF-CAM-CAMTIME input lacks a
    variable the cloud fraction is read from.
    """


class CloudFractionTimeMismatchError(ValueError):
    """The CF-CAM rows do not line up with the L1B RAD-4CH footprints they are matched to.

    Raised when two CF-CAM rows share a radiometer time, when a CF-CAM cloud fraction falls inside an L1B granule's
    time span without matching one of its footprints, or when no footprint of the granule matches any CF-CAM row.
    """


class MissingAncillaryInputError(ValueError):
    """The input manifest lists no file of an ancillary product a FMATCH run reads.

    Raised before any input is read, naming each reader active for the run's mode whose ``PRODUCT_ID`` matches no
    file in the manifest.
    """


@dataclass(frozen=True)
class RadiometerCloudFraction:
    """The CF-CAM cloud fraction of every CF-CAM input of one run, keyed by radiometer time.

    Read once per run by :func:`load_cf_cam_cloud_fraction` and matched to each L1B input's footprints by
    :meth:`values_for`.

    Attributes
    ----------
    radiometer_time : numpy.ndarray
        Sorted, unique ``datetime64[ns]`` radiometer times (UTC, decoded from nanoseconds since the CCSDS epoch,
        1958-01-01) of the CF-CAM rows.
    cloud_fraction : numpy.ndarray
        ``float32`` cloud fraction in percent, aligned with ``radiometer_time``; ``NaN`` where CF-CAM wrote fill.
    file_names : tuple[str, ...]
        Names of the CF-CAM files the values were read from, recorded in each product's provenance.
    """

    radiometer_time: np.ndarray
    cloud_fraction: np.ndarray
    file_names: tuple[str, ...]

    def values_for(self, footprint_time: np.ndarray) -> np.ndarray:
        """Return the cloud fraction of each L1B footprint, matched by exact radiometer time.

        CF-CAM covers only the footprints the camera sees, so a footprint with no CF-CAM row takes ``NaN``. A CF-CAM
        row outside the granule's time span belongs to another granule and is ignored.

        Parameters
        ----------
        footprint_time : numpy.ndarray
            ``datetime64[ns]`` radiometer times of one L1B input's usable footprints.

        Returns
        -------
        numpy.ndarray
            ``float32`` cloud fraction in percent, one per footprint; ``NaN`` where CF-CAM has no value.

        Raises
        ------
        CloudFractionTimeMismatchError
            If a CF-CAM row with a cloud fraction lies inside the span of ``footprint_time`` but matches none of its
            footprints, which means the two products disagree on the radiometer times; or if no footprint matches
            a CF-CAM row, which means the CF-CAM inputs do not cover this granule.
        """
        footprint_time = np.asarray(footprint_time, dtype="datetime64[ns]")
        n_rows = self.radiometer_time.size
        index = np.minimum(np.searchsorted(self.radiometer_time, footprint_time), max(n_rows - 1, 0))
        matched = (self.radiometer_time[index] == footprint_time) if n_rows else np.zeros(footprint_time.shape, bool)

        in_span = (self.radiometer_time >= footprint_time.min()) & (self.radiometer_time <= footprint_time.max())
        unmatched_rows = in_span & np.isfinite(self.cloud_fraction)
        unmatched_rows[index[matched]] = False
        if unmatched_rows.any():
            raise CloudFractionTimeMismatchError(
                f"{int(unmatched_rows.sum())} CF-CAM cloud fraction value(s) fall inside the L1B granule "
                f"{footprint_time.min()} to {footprint_time.max()} but match no usable L1B footprint time (first "
                f"{self.radiometer_time[unmatched_rows][0]}). CF-CAM must carry the L1B RAD-4CH {CF_CAM_TIME_VARIABLE} "
                f"values unchanged; regenerate CF-CAM ({', '.join(self.file_names)}) from this L1B RAD-4CH granule."
            )
        if not matched.any():
            raise CloudFractionTimeMismatchError(
                f"No CF-CAM row matches a footprint of the L1B granule {footprint_time.min()} to "
                f"{footprint_time.max()}; the CF-CAM inputs ({', '.join(self.file_names)}) do not cover it. Stage the "
                "CF-CAM product generated from this L1B RAD-4CH granule in the input manifest."
            )

        values = np.full(footprint_time.shape, np.nan, dtype=np.float32)
        values[matched] = self.cloud_fraction[index[matched]]
        logger.info("Matched CF-CAM cloud fraction to %d of %d footprints", int(matched.sum()), matched.size)
        return values


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
        The separate Camera Cloud Fraction product a radiometer-timescale run requires and merges as
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
    MissingCloudFractionInputError
        If the file has no ``Cloud_Fraction`` variable.
    """
    with xr.open_dataset(local_input_path) as cf_dataset:
        cf_dataset = cf_dataset.load()
    footprints = pseudofootprints_from_camtime_dataset(cf_dataset)
    if CF_CAM_CLOUD_FRACTION_VARIABLE not in cf_dataset:
        raise MissingCloudFractionInputError(
            f"CF-CAM-CAMTIME input {local_input_path} has no {CF_CAM_CLOUD_FRACTION_VARIABLE} variable, which "
            "FMATCH-CAM-CAMTIME writes as cloud_fraction_camera. Regenerate the input with the Camera Cloud Fraction "
            f"algorithm so it carries {CF_CAM_CLOUD_FRACTION_VARIABLE}."
        )
    # Boolean-mask indexing returns row-major order, matching the reconstruction order.
    mask = camtime_real_cell_mask(cf_dataset)
    cloud_fraction_camera = np.asarray(cf_dataset[CF_CAM_CLOUD_FRACTION_VARIABLE].values, dtype=np.float32)[mask]
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
# otherwise identical across all four.

# FMATCH-CAM (radiometer timescale): the lowest-latency (camera / near-real-time) product; runs continuously from
# mission start. Input is the L1B RAD-4CH product; the required cloud fraction comes from CF-CAM (also radiometer
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

# Registry of every FMATCH runner, keyed by the variant name the LIBERA_FMATCH_VARIANT environment variable selects.
# Adding a variant is one config plus one entry.
RUNNER_CONFIGS: dict[str, FmatchRunnerConfig] = {
    "cam": CAM_CONFIG,
    "cam-camtime": CAM_CAMTIME_CONFIG,
    "imager": IMAGER_CONFIG,
    "imager-camtime": IMAGER_CAMTIME_CONFIG,
}

# Environment variable naming the RUNNER_CONFIGS entry to run; set per processing step.
FMATCH_VARIANT_ENV = "LIBERA_FMATCH_VARIANT"


def resolve_runner_config_from_env() -> FmatchRunnerConfig:
    """Select the FMATCH runner config named by the ``LIBERA_FMATCH_VARIANT`` environment variable.

    Returns
    -------
    FmatchRunnerConfig
        The ``RUNNER_CONFIGS`` entry for the named variant.

    Raises
    ------
    ValueError
        If the variable is not set or names no entry in ``RUNNER_CONFIGS``.
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
        If the ``PROCESSING_PATH`` environment variable is not set, if the manifest
        references no input of ``config.input_product_id``, or if an L1B input has no
        usable footprints.
    MissingAncillaryInputError
        If the manifest lists no file of an ancillary product a reader active for the mode reads.
    MissingCloudFractionInputError
        If a FMATCH-CAM manifest lists no CF-CAM input, or a CF-CAM or CF-CAM-CAMTIME input
        lacks a variable its cloud fraction is read from.
    CloudFractionTimeMismatchError
        If the CF-CAM rows do not line up with an L1B input's footprints.
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

    # Step 2: Select each active reader's ancillary granules from the manifest. A missing product raises here, before
    # any input is read or downloaded.
    logger.info("Step 2: Selecting ancillary inputs from the manifest")
    ancillary_file_paths = select_ancillary_inputs(input_manifest, config.mode)

    # Step 3: Collect the L1B input file(s) from the manifest.
    input_label = config.input_product_id.value
    logger.info("Step 3: Collecting %s input files from the manifest", input_label)
    input_file_paths = select_manifest_files_by_product_id(input_manifest, config.input_product_id)
    if not input_file_paths:
        raise ValueError(f"No {input_label} input files found in the input manifest")

    # The CF-CAM inputs are read once here and matched to each L1B input below.
    cloud_fraction = None
    if config.cloud_fraction_product_id is not None:
        cloud_fraction = load_cf_cam_cloud_fraction(
            select_manifest_files_by_product_id(input_manifest, config.cloud_fraction_product_id)
        )

    # Step 4: Run footprint matching and write data products.
    logger.info("Step 4: Running footprint matching and writing data products")
    output_data_file_paths: list[LiberaDataProductFilename] = []
    with ExitStack() as local_copies:
        # The readers open real files, so each S3 granule is copied locally once for the whole run, and every copy is
        # removed when the stack closes. The aggregation's worker processes finish inside each product write.
        ancillary_inputs = {
            reader_key: [local_copies.enter_context(_as_local_path(path)) for path in paths]
            for reader_key, paths in ancillary_file_paths.items()
        }
        for input_file_path in input_file_paths:
            output_file = create_and_write_data_product(
                input_file_path=input_file_path,
                output_path=dropbox_path,
                config=config,
                cloud_fraction=cloud_fraction,
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


def load_cf_cam_cloud_fraction(cf_cam_file_paths: list[str] | list[Path]) -> RadiometerCloudFraction:
    """Read the cloud fraction of every CF-CAM file of a run, keyed by radiometer time.

    The files are read once and their rows combined, so a run with several L1B inputs matches each against the same
    values (see :meth:`RadiometerCloudFraction.values_for`). The ``_FillValue`` of ``Cloud_Fraction`` reads as
    ``NaN``.

    Parameters
    ----------
    cf_cam_file_paths : list[str] | list[pathlib.Path]
        Paths (local or S3) to the CF-CAM files selected from the input manifest.

    Returns
    -------
    RadiometerCloudFraction
        The combined cloud fraction in percent, sorted by radiometer time.

    Raises
    ------
    MissingCloudFractionInputError
        If ``cf_cam_file_paths`` is empty, since FMATCH-CAM writes the CF-CAM cloud fraction and cannot run without
        it, or if a file lacks ``Cloud_Fraction`` or ``radiometer_time``.
    CloudFractionTimeMismatchError
        If two CF-CAM rows share a radiometer time, which leaves the footprint's cloud fraction ambiguous.
    """
    if not cf_cam_file_paths:
        raise MissingCloudFractionInputError(
            "The input manifest lists no CF-CAM product, which FMATCH-CAM requires for Cloud_Fraction_Camera. Add "
            "the CF-CAM file(s) generated from the run's L1B RAD-4CH input to the input manifest."
        )
    times: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for path in cf_cam_file_paths:
        with _as_local_path(path) as local_path, xr.open_dataset(local_path) as cf_dataset:
            missing = [
                name for name in (CF_CAM_TIME_VARIABLE, CF_CAM_CLOUD_FRACTION_VARIABLE) if name not in cf_dataset
            ]
            if missing:
                raise MissingCloudFractionInputError(
                    f"CF-CAM input {AnyPath(path).name} has no {', '.join(missing)} variable, which FMATCH-CAM reads "
                    "Cloud_Fraction_Camera from. Regenerate the input with the Camera Cloud Fraction algorithm so it "
                    f"carries {CF_CAM_TIME_VARIABLE} and {CF_CAM_CLOUD_FRACTION_VARIABLE}."
                )
            times.append(np.asarray(cf_dataset[CF_CAM_TIME_VARIABLE].values, dtype="datetime64[ns]"))
            values.append(np.asarray(cf_dataset[CF_CAM_CLOUD_FRACTION_VARIABLE].values, dtype=np.float32))

    all_times = np.concatenate(times)
    order = np.argsort(all_times, kind="stable")
    all_times = all_times[order]
    duplicated = all_times[1:] == all_times[:-1]
    file_names = tuple(AnyPath(path).name for path in cf_cam_file_paths)
    if duplicated.any():
        raise CloudFractionTimeMismatchError(
            f"{int(duplicated.sum())} CF-CAM radiometer time(s) appear more than once across {', '.join(file_names)} "
            f"(first {all_times[1:][duplicated][0]}), so their footprints' cloud fraction is ambiguous. List each "
            "CF-CAM granule in the input manifest once, and only granules whose time ranges do not overlap."
        )
    logger.info("Read %d CF-CAM cloud fraction row(s) from %d file(s)", all_times.size, len(file_names))
    return RadiometerCloudFraction(
        radiometer_time=all_times,
        cloud_fraction=np.concatenate(values)[order],
        file_names=file_names,
    )


def select_ancillary_inputs(manifest: Manifest, mode: OperationalMode) -> dict[str, list[str]]:
    """Select, for each reader active in ``mode``, the manifest's files of the ancillary product it reads.

    Each reader names the ``DataLevel.ANC`` product it reads in its ``PRODUCT_ID``, and a manifest file belongs to
    that product when its name parses to it (the ancillary filename classes in :mod:`libera_utils.io.filenaming`
    accept the names the upstream providers give their files). Every file of the product is taken: selection is by
    product, not by time. The readers that read one granule per day (VIIRS aerosol, BRDF and cloud properties, and
    NISE) raise when given more than one, so a manifest lists exactly one of each of those; the readers of hourly
    granules (ERA5, CLDPIX, and the CERES SSF that ``CeresSsfReader`` reads) take every granule they are given and
    use the ones near each footprint, so a manifest lists the hours around its period.

    Parameters
    ----------
    manifest : Manifest
        The run's input manifest.
    mode : OperationalMode
        The FMATCH operational mode being run; its active readers come from
        :meth:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry.get_readers_for_mode`.

    Returns
    -------
    dict[str, list[str]]
        Reader registry key to the manifest filenames (local paths or S3 urls) of its product, in manifest order.
        Every active reader has at least one.

    Raises
    ------
    MissingAncillaryInputError
        If the manifest lists no file of the product of one or more active readers; the message names each one.
    TypeError
        If an active reader declares no ``PRODUCT_ID``, which only a reader pinned to a mode outside the built-in
        set can do.
    """
    selected: dict[str, list[str]] = {}
    missing: list[str] = []
    for reader_key, reader in sorted(ReaderRegistry.get_readers_for_mode(mode).items()):
        product_id = reader.PRODUCT_ID
        if product_id is None:
            raise TypeError(
                f"Reader {reader_key!r} is active for {mode.value} but declares no PRODUCT_ID, so its files cannot be "
                "selected from the input manifest. Set its PRODUCT_ID to the DataLevel.ANC DataProductIdentifier of "
                "the source it reads."
            )
        files = select_manifest_files_by_product_id(manifest, product_id)
        if files:
            logger.info("Ancillary input %r (%s): %d file(s)", reader_key, product_id.value, len(files))
        else:
            missing.append(f"{reader_key} ({product_id.value})")
        selected[reader_key] = files
    if missing:
        raise MissingAncillaryInputError(
            f"The input manifest lists no file for the ancillary reader(s) {', '.join(missing)} that {mode.value} "
            "reads. Add each product's file(s) for the processing period to the input manifest, under the name its "
            "upstream provider gives it."
        )
    return selected


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
      cloud fraction (FMATCH-CAM's CF-CAM cloud fraction is matched to these footprints by
      :func:`create_and_write_data_product`).

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
        If a camera-timescale mode has no ``camera_footprint_loader`` configured, or if a
        radiometer-timescale L1B file has no usable footprints (raised by
        :func:`load_l1b_radiometer_inputs`).
    MissingCloudFractionInputError
        If a FMATCH-CAM-CAMTIME input has no ``Cloud_Fraction``.
    MissingCamtimeVariablesError
        If a camera-timescale input lacks a variable needed to rebuild the pseudo-footprints.

    Notes
    -----
    Both paths use :func:`xarray.open_dataset`, which needs a real local file, so an S3 input is first
    materialized to a temporary file by ``_as_local_path``; local inputs are read in place.
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


def create_and_write_data_product(
    input_file_path: str | Path | S3Path,
    output_path: str | Path | S3Path,
    config: FmatchRunnerConfig,
    ancillary_inputs: dict[str, list[Path]],
    cloud_fraction: RadiometerCloudFraction | None = None,
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
    ancillary_inputs : dict[str, list[pathlib.Path]]
        Reader registry key to the local granule files of its product, as selected by
        :func:`select_ancillary_inputs` and copied locally: at least one existing file for every reader
        active for the mode.
    cloud_fraction : RadiometerCloudFraction, optional
        The CF-CAM cloud fraction from :func:`load_cf_cam_cloud_fraction`. Required when
        ``config.cloud_fraction_product_id`` is set (FMATCH-CAM), and must be ``None`` otherwise; the
        camera-timescale modes read their cloud fraction from ``input_file_path`` instead.

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.

    Raises
    ------
    MissingCloudFractionInputError
        If ``config`` requires CF-CAM and ``cloud_fraction`` is ``None``, or a CF-CAM-CAMTIME input has no
        ``Cloud_Fraction``.
    CloudFractionTimeMismatchError
        If the CF-CAM rows do not line up with the L1B footprints (see
        :meth:`RadiometerCloudFraction.values_for`).
    ValueError
        If ``cloud_fraction`` is given for a mode that takes no CF-CAM input; if a reader in
        ``ancillary_inputs`` has no granule; for a camera-timescale mode,
        whose pseudo-footprints carry no ephemeris to size the PSF from (``TODO[LIBSDC-884]``); or for
        any condition :func:`run_footprint_matching` raises it for.
    MissingCamtimeVariablesError
        If a camera-timescale input lacks a variable needed to rebuild the pseudo-footprints.
    KeyError
        If ``ancillary_inputs`` does not hold exactly the readers active for the mode.
    FileNotFoundError
        If a granule in ``ancillary_inputs`` is not an existing file.

    Notes
    -----
    The external-reader aggregates (``igbp_surface_type``, ``era5_*``, ``flashflux_*``,
    ``cldpix_*``, ...) and the ``psf_coverage_fraction`` and ``q_flags`` columns are computed by
    the PSF aggregation engine from ``ancillary_inputs``. FMATCH-CAM's ``Cloud_Fraction_Camera``
    is the CF-CAM value of each footprint, fill where CF-CAM has none; the camera-timescale modes
    carry the cloud fraction from their CF-CAM-CAMTIME input.
    """
    if config.cloud_fraction_product_id is None and cloud_fraction is not None:
        raise ValueError(
            f"A CF-CAM cloud fraction was given, but {config.mode.value} takes no CF-CAM input: pass "
            "cloud_fraction=None."
        )
    if config.cloud_fraction_product_id is not None and cloud_fraction is None:
        raise MissingCloudFractionInputError(
            f"{config.mode.value} requires the CF-CAM cloud fraction for Cloud_Fraction_Camera: read the run's CF-CAM "
            "files with load_cf_cam_cloud_fraction and pass the result as cloud_fraction."
        )

    inputs, cloud_fraction_camera = run_footprint_matching(input_file_path, config)

    input_file_name = AnyPath(input_file_path).name
    # Provenance records every file that fed this product, so a written file names its own inputs.
    provenance = [input_file_name]
    if cloud_fraction is not None:
        cloud_fraction_camera = cloud_fraction.values_for(inputs[FMATCH_RADIOMETER_TIME_COORDINATE])
        provenance.extend(cloud_fraction.file_names)

    provenance.extend(path.name for paths in ancillary_inputs.values() for path in paths)
    logger.info("Aggregating external variables from %d ancillary source(s)", len(ancillary_inputs))

    logger.info("Writing %s data product for input %s", config.mode.value, input_file_name)
    output_file_path = write_fmatch_product(
        config.mode,
        inputs,
        output_path,
        algorithm_version=algorithm_version(),
        input_files=",".join(provenance),
        cloud_fraction_camera=cloud_fraction_camera,
        source_file_paths=ancillary_inputs,
    )
    logger.info("Wrote %s data product to %s", output_file_path.data_product_id.value, output_file_path.path)
    return output_file_path


def select_manifest_files_by_product_id(manifest: Manifest, *product_ids: DataProductIdentifier) -> list[str]:
    """Select the files referenced by a manifest that belong to the given Libera or ancillary data product(s).

    A runner receives one manifest listing every file staged for the processing step, which in
    general contains more than the runner consumes (an L1B input, an optional cloud-fraction product,
    ancillary granules, sibling products). This helper keeps exactly the records whose name parses,
    through :meth:`~libera_utils.io.filenaming.AbstractValidFilename.from_file_path`, to a data
    product whose id is one of ``product_ids``, in manifest order. That covers both the Libera
    products and the external ancillary products, whose files keep their upstream providers' names.

    Files whose names parse to no data product (an unrecognized name, or a manifest) are skipped
    rather than raising: a manifest may legitimately carry files a runner does not read, and a runner
    should ignore them, not fail on them.

    This lives in the runner, alongside its callers, so that ``libera_utils.io`` stays a generic I/O
    package with no dependency on Libera product identifiers. The SCENE-ID runner keeps its own
    single-product equivalent (``collect_input_files``) for the same reason.

    Parameters
    ----------
    manifest : Manifest
        The manifest to inspect.
    *product_ids : DataProductIdentifier
        One or more product ids to keep, Libera or ancillary. Passing several is useful when a runner takes more
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
            parsed = AbstractValidFilename.from_file_path(filename)
        except ValueError:
            parsed = None
        if not isinstance(parsed, AbstractDataProductFilename):
            # The run calls this once per input product, so a file's skip is logged at DEBUG, not per call at INFO.
            logger.debug("Skipping file that names no data product (not a %s input): %s", wanted_labels, filename)
            continue
        if parsed.data_product_id in wanted:
            logger.info("Recording %s input file: %s", parsed.data_product_id.value, filename)
            selected.append(filename)
        else:
            logger.debug("Skipping product '%s' (not %s): %s", parsed.data_product_id.value, wanted_labels, filename)
    return selected


def load_l1b_radiometer_inputs(l1b_file: Path) -> dict[str, np.ndarray]:
    """Read the per-footprint L1B inputs that FMATCH passes through verbatim.

    Pulls every quantity the FMATCH product contract takes straight from L1B Daily:
    the ``RADIOMETER_TIME`` coordinate plus all variables in
    ``L1B_PASSTHROUGH_VARIABLES``
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
        locally first (see ``_as_local_path``), because
        :func:`xarray.open_dataset` seeks within the file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping keyed by L1B variable name: ``"RADIOMETER_TIME"`` (datetime64[ns]) plus
        each name in ``L1B_PASSTHROUGH_VARIABLES`` (float32), which are also the FMATCH
        product names. It additionally carries the scan-reference geometry used to build
        footprints (not product columns): each name in ``L1B_SCAN_REFERENCE_VARIABLES``
        and ``Cone_Angle_Rate`` (float64). All arrays are 1-D and the same length.

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
        # Read every pass-through variable under its L1B name, which is also its product name.
        passthrough = {name: l1b[name].values for name in L1B_PASSTHROUGH_VARIABLES}
        # Scan-reference geometry (subsatellite point + cone-angle rate), read at
        # float64 for the ray-trace. Not written to the product; feeds footprint build.
        scan_reference = {name: np.asarray(l1b[name].values, dtype=np.float64) for name in L1B_SCAN_REFERENCE_VARIABLES}
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
    result[L1B_CONE_ANGLE_RATE_VARIABLE] = cone_angle_rate
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

    Reading an input file (an L1B or CF-CAM product with :func:`xarray.open_dataset`, an ancillary
    granule with netCDF4 or the HDF4/HDF5 libraries) requires a real file on a local filesystem,
    because those libraries seek within the file rather than streaming it. A runner's manifest,
    however, may reference either local paths or S3 object urls. For S3 inputs this downloads the
    object, under its own name, to a temporary directory that is removed on exit; local inputs are
    yielded unchanged with no copy.

    Kept in the runner so that ``libera_utils.io`` stays free of runner-specific plumbing.

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
