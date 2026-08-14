"""Shared manifest-driven runner logic for the FMATCH product family.

The five FMATCH runners (``cam/``, ``cam_camtime/``, ``imager_flash/``, ``imager/``,
``imager_camtime/``) are structurally identical: read an input manifest, locate the
staged ancillary granules, keep the L1B input files of a particular product, run
footprint matching on each, write the resulting FMATCH product, and emit an output
manifest. They differ only by a handful of parameters:

* which :class:`~libera_utils.footprint_matching.types.OperationalMode` they produce
  (which in turn drives the product definition, the time coordinate, and the active
  reader set),
* which L1B product counts as their input (``RAD-4CH`` vs ``CAM``), and
* whether they consume the optional Camera Cloud Fraction product.

Rather than duplicate the manifest/dropbox plumbing five times, that shared body lives
here and is parameterized by a small :class:`FmatchRunnerConfig`. Each concrete runner
is then a thin module that builds a config and forwards its ``main``/``algorithm`` to
:func:`run_algorithm`. This mirrors
``libera_utils/scene_identification/_runner.py`` deliberately, so the two algorithm
families read the same way.

Environment
-----------
Two environment variables are required at run time:

``PROCESSING_PATH``
    The dropbox directory (or S3 prefix) products and the output manifest are written
    to. This is the standard Libera runner convention.
``FMATCH_ANCILLARY_PATH``
    Root of the staged ancillary granule tree; see
    :func:`resolve_ancillary_inputs` for its layout. Resolved
    non-strictly: an absent or incomplete tree degrades the external columns to
    conformant placeholders rather than failing the product (see "Ancillary inputs"
    below). ``TODO[LIBSDC-785]``: enforce strict availability once production
    staging is guaranteed.

Ancillary inputs
----------------
When a full ancillary tree is staged under ``FMATCH_ANCILLARY_PATH`` (one local
granule per active reader; see :func:`resolve_ancillary_inputs`), the
runner hands those files to the product assembly so the external variables and the
coverage/QA columns are computed by the PSF aggregation engine. When the tree is
absent or incomplete, those columns fall back to conformant placeholders and the
derived-geometry (``sunglint_angle``) column is still computed from the L1B angles.
``TODO[LIBSDC-785]``: materialize S3-staged ancillary granules locally (readers need
real files) and enforce strict availability once production staging is guaranteed.
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
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path

# Importing the readers subpackage is what populates ReaderRegistry: every concrete
# reader self-registers at class-definition time via GriddedDataReader.__init_subclass__,
# which only happens once its module has been imported. Without this import the
# registry would be empty and every mode would resolve to zero readers.
import libera_utils.footprint_matching.readers  # noqa: F401  (imported for its registration side effect)
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.footprint_matching.product import (
    FMATCH_CONE_ANGLE_RATE_KEY,
    L1B_CONE_ANGLE_RATE_VARIABLE,
    L1B_PASSTHROUGH_VARIABLES,
    L1B_SCAN_REFERENCE_VARIABLES,
    fmatch_time_variable,
    is_camera_timescale_mode,
    write_fmatch_product,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import is_s3, smart_copy_file
from libera_utils.logutil import configure_task_logging

if TYPE_CHECKING:
    from libera_utils.constants import DataProductIdentifier
    from libera_utils.footprint_matching.camera_segmentation import PseudoFootprint

logger = logging.getLogger(__name__)

# Name of the time coordinate variable inside the L1B radiometer file. xarray decodes
# its CF "nanoseconds since 1958-01-01" units into datetime64[ns], which is exactly
# the dtype the FMATCH RADIOMETER_TIME coordinate declares.
L1B_TIME_VARIABLE: str = "radiometer_time"

# Name of the FMATCH product's radiometer time coordinate (the key the L1B reader
# returns the decoded L1B times under).
FMATCH_RADIOMETER_TIME_COORDINATE: str = "RADIOMETER_TIME"

# Environment variable naming the root of the staged ancillary tree. Mirrors the
# PROCESSING_PATH convention used for the output dropbox.
ANCILLARY_PATH_ENV: str = "FMATCH_ANCILLARY_PATH"


def select_manifest_files_by_product_id(manifest: Manifest, *product_ids: DataProductIdentifier) -> list[str]:
    """Select the files referenced by a manifest that belong to the given Libera data product(s).

    A FMATCH runner receives one manifest listing every file staged for the processing step, which in
    general contains more than the runner consumes (an L1B input, an optional cloud-fraction product,
    ancillary granules, sibling products). This helper keeps exactly the records whose parsed Libera
    product id is one of ``product_ids``, in manifest order.

    Files whose names do not parse as a :class:`~libera_utils.io.filenaming.LiberaDataProductFilename`
    are skipped rather than raising: manifests legitimately carry non-Libera files (for example the
    externally-sourced ancillary granules the readers consume), and a runner should ignore them, not
    fail on them.

    This lives in the runner layer, alongside its only callers, so that ``libera_utils.io`` stays a
    generic I/O package with no dependency on Libera product identifiers. The SCENE-ID runner keeps
    its own single-product equivalent (``collect_input_files``) for the same reason.

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
    for file_record in manifest.files:
        filename = file_record.filename
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


@dataclass(frozen=True)
class FmatchRunnerConfig:
    """Everything that distinguishes one FMATCH runner from another.

    Attributes
    ----------
    mode : OperationalMode
        The FMATCH operational mode this runner produces. Drives the product
        definition, the time coordinate, the active reader set, and which assembly
        path (camera- or radiometer-timescale) is taken.
    output_product_id : DataProductIdentifier
        The FMATCH product this runner emits. Used for documentation/logging; the
        written filename's product id comes from the product definition's
        ``ProductID`` attribute.
    l1b_input_product_id : DataProductIdentifier
        The L1B Daily product that counts as this runner's input: ``l1b_rad``
        (``RAD-4CH``) for radiometer-timescale modes, ``l1b_cam`` (``CAM``) for
        camera-timescale modes. Manifest files with any other product id (or
        unparsable names) are skipped.
    cloud_fraction_product_id : DataProductIdentifier or None
        The optional Camera Cloud Fraction product supplying the
        ``cloud_fraction_camera`` variable. Only the CAM-family modes declare that
        variable, so this is ``None`` for the IMAGER modes.
    log_prefix : str
        Short label used in task-log filenames (e.g. ``fmatch_cam``).
    """

    mode: OperationalMode
    output_product_id: DataProductIdentifier
    l1b_input_product_id: DataProductIdentifier
    cloud_fraction_product_id: DataProductIdentifier | None
    log_prefix: str

    @property
    def is_camera_timescale(self) -> bool:
        """True when this runner's mode is indexed on ``CAMERA_TIME`` rather than ``RADIOMETER_TIME``."""
        return is_camera_timescale_mode(self.mode)

    @property
    def time_variable(self) -> str:
        """Name of the per-footprint time coordinate in the written product."""
        return fmatch_time_variable(self.mode)


def run_algorithm(
    manifest_path: Path | S3Path | argparse.Namespace,
    config: FmatchRunnerConfig,
) -> Path | S3Path:
    """Run a FMATCH processing workflow from an input manifest.

    Parameters
    ----------
    manifest_path : Path | S3Path | argparse.Namespace
        Path to the input manifest file listing the L1B input file(s). An
        ``argparse.Namespace`` (as produced by a runner's ``main``) is also accepted
        for convenience when invoked as a CLI; its ``manifest`` attribute is used.
    config : FmatchRunnerConfig
        The per-runner parameters (mode, input products, log label).

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.

    Raises
    ------
    ValueError
        If the ``PROCESSING_PATH`` environment variable is not set, or if the manifest
        references no usable L1B inputs.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"{config.log_prefix}_{now}")

    # Step 1: Read the input manifest.
    logger.info("Step 1: Reading the input manifest file")
    manifest = _resolve_manifest(manifest_path)
    input_manifest = Manifest.from_file(manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")
    logger.info("Running %s", config.mode.value)

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Locate the staged ancillary granules for this mode.
    #
    # These are resolved, inventoried in the log, and (when the tree collapses to one
    # local granule per active reader; see _ancillary_source_file_paths) threaded into
    # product assembly, where the PSF aggregation engine computes the external variables
    # and coverage/QA columns from them. Logging the inventory first lets an operator
    # confirm staging is correct from the task log. Resolution is non-strict: an absent
    # or incomplete tree degrades those columns to placeholders rather than failing the
    # product (TODO[LIBSDC-785]: enforce strict availability once staging is guaranteed).
    logger.info("Step 2: Locating staged ancillary inputs")
    ancillary_inputs = resolve_ancillary_inputs(config.mode)
    log_ancillary_inventory(ancillary_inputs)

    # Step 3: Collect the L1B input file(s) from the manifest.
    input_label = config.l1b_input_product_id.value
    logger.info("Step 3: Collecting %s input files from the manifest", input_label)
    input_file_paths = select_manifest_files_by_product_id(input_manifest, config.l1b_input_product_id)
    if not input_file_paths:
        raise ValueError(f"No {input_label} input files found in the input manifest")

    cloud_fraction_file_paths = _collect_cloud_fraction_files(input_manifest, config)

    # Step 4: Run footprint matching and write data products.
    logger.info("Step 4: Running footprint matching and writing data products")
    output_data_file_paths: list[LiberaDataProductFilename] = []
    for input_file_path in input_file_paths:
        output_file = create_and_write_data_product(
            l1b_file_path=input_file_path,
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


def _resolve_manifest(manifest_path: Path | S3Path | argparse.Namespace) -> Path | S3Path:
    """Normalize the manifest argument to a path.

    Accepting an ``argparse.Namespace`` lets a runner's ``main`` forward its parsed
    args straight through, which is how the SCENE-ID runners are written too.
    """
    if isinstance(manifest_path, argparse.Namespace):
        return AnyPath(manifest_path.manifest)
    return AnyPath(manifest_path)


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

    # Cast to the exact dtypes the FMATCH definition declares so conformance checking
    # passes without an auto-cast. Every pass-through variable is float32 in the
    # definition and the decoded time is datetime64[ns]; the casts are belt-and-braces
    # over already-correct dtypes. The scan-reference geometry stays float64.
    result: dict[str, np.ndarray] = {
        FMATCH_RADIOMETER_TIME_COORDINATE: radiometer_time.astype("datetime64[ns]"),
    }
    result.update({name: values.astype(np.float32) for name, values in passthrough.items()})
    result.update(scan_reference)
    result[FMATCH_CONE_ANGLE_RATE_KEY] = cone_angle_rate
    return result


def load_l1b_camera_dataset(l1b_file: Path) -> xr.Dataset:
    """Open an L1B Daily Camera file for segmentation into pseudo-footprints.

    The camera-timescale FMATCH modes do not pass L1B columns through; they segment the
    camera pixel grid with
    :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`,
    which needs the whole dataset (geolocation grids, altitude, and viewing angles on
    the ``CAMERA_TIME`` x ``CAMERA_PIXEL_COUNT_X`` x ``CAMERA_PIXEL_COUNT_Y`` grid).

    The dataset is loaded eagerly into memory (``.load()``) and detached from the file
    handle, so callers may use it after the source file goes away - which matters when
    the input was materialized into a temporary directory by :class:`_as_local_path`.

    Parameters
    ----------
    l1b_file : pathlib.Path
        Path to a local L1B CAM NetCDF file.

    Returns
    -------
    xarray.Dataset
        The opened, fully-loaded L1B camera dataset.
    """
    with xr.open_dataset(l1b_file) as l1b:
        return l1b.load()


def resolve_ancillary_inputs(
    mode: OperationalMode,
    root: str | Path | S3Path | None = None,
    *,
    strict: bool = False,
) -> dict[str, list[Path | S3Path]]:
    """Map each reader active for ``mode`` to its staged granule files.

    Footprint matching aggregates external, non-Libera datasets (ERA5 reanalysis, IGBP
    land cover, NISE snow/ice, VIIRS products, and - post year one - CERES CLDPIX/SSF)
    onto each radiometer footprint. Those granules are *not* Libera data products: they
    have third-party filenames, so they cannot be selected out of a manifest the way
    :func:`select_manifest_files_by_product_id` selects L1B inputs.

    The pipeline stages ancillary granules into a directory tree with **one
    subdirectory per reader, named by the reader's registry key**::

        $FMATCH_ANCILLARY_PATH/
            era5/           ERA5 single-level granules
            era5_pressure/  ERA5 pressure-level granules
            igbp/           MCD12Q1 land-cover granules
            nise/           NISE snow/ice granules
            viirs_brdf/     VIIRS BRDF granules
            viirs_cloud/    VIIRS cloud granules
            viirs_aod/      VIIRS Deep Blue aerosol granules (AOD + aerosol type)
            cldpix/         CERES CLDPIX granules   (IMAGER-family modes)
            ssf/            CERES SSF granules      (IMAGER-family modes)

    The active reader set comes from
    :meth:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry.get_readers_for_mode`,
    so this automatically tracks the mode's latency rank: FMATCH-CAM looks for five
    readers, while FMATCH-IMAGER additionally looks for ``era5_pressure``,
    ``viirs_aod``, and the RBSP ``cldpix``/``ssf`` inputs. Using the registry keys as
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
        How to treat missing inputs. When False (the default) a missing root, a
        missing reader subdirectory, or an empty one is logged as a warning and
        yields an empty list. When True each of those raises.

        The default is False **in this milestone** because the PSF aggregation engine
        that consumes these granules is not implemented yet
        (``product.aggregate_external_variables`` is still a ``TODO[LIBSDC-785]``
        stub): a runner must not fail on inputs that nothing reads. Flip this default
        to True in the same change that implements the aggregation engine, at which
        point a missing granule genuinely invalidates the product.

    Returns
    -------
    dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        Mapping of reader registry key to the granule files staged for it, sorted by
        name so repeated runs process files in a deterministic order. Readers with no
        staged files map to an empty list (when ``strict`` is False).

    Raises
    ------
    ValueError
        If ``root`` is not given and ``FMATCH_ANCILLARY_PATH`` is unset, and ``strict``.
    FileNotFoundError
        If the root or a required reader subdirectory does not exist, or a required
        subdirectory is empty, and ``strict``.
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
    l1b_file_path: str | Path | S3Path,
    config: FmatchRunnerConfig,
) -> list[PseudoFootprint] | dict[str, np.ndarray]:
    """Read one L1B input file and produce this mode's footprint assembly inputs.

    The two timescales produce different things, which is why the return type is a
    union:

    * **camera-timescale** modes return the list of
      :class:`~libera_utils.footprint_matching.camera_segmentation.PseudoFootprint`
      objects obtained by segmenting the L1B camera image grid;
    * **radiometer-timescale** modes return the dict of L1B pass-through arrays
      (time, geolocation, viewing angles), whose footprints are the L1B radiometer
      footprints themselves.

    Parameters
    ----------
    l1b_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to the L1B Daily input file.
    config : FmatchRunnerConfig
        Runner parameters supplying the operational mode.

    Returns
    -------
    list[PseudoFootprint] | dict[str, np.ndarray]
        The assembly inputs for this mode, ready to hand to
        :func:`~libera_utils.footprint_matching.product.write_fmatch_product`.

    Notes
    -----
    Both readers use :func:`xarray.open_dataset`, which needs a real local file, so an
    S3 input is first materialized to a temporary file by :class:`_as_local_path`;
    local inputs are read in place with no copy.
    """
    with _as_local_path(l1b_file_path) as local_l1b_path:
        if config.is_camera_timescale:
            logger.info("Segmenting L1B camera images from %s", local_l1b_path)
            dataset = load_l1b_camera_dataset(local_l1b_path)
            footprints = segment_l1b_camera(dataset, log=logger)
            logger.info("Segmented %d camera pseudo-footprints", len(footprints))
            return footprints

        logger.info("Reading L1B radiometer pass-through inputs from %s", local_l1b_path)
        l1b_inputs = load_l1b_radiometer_inputs(local_l1b_path)
        logger.info("Read %d radiometer footprints", len(l1b_inputs[config.time_variable]))
        return l1b_inputs


def _ancillary_source_file_paths(
    ancillary_inputs: dict[str, list[Path | S3Path]] | None,
) -> dict[str, Path] | None:
    """Reduce the resolved ancillary inventory to the one-file-per-reader map assembly needs.

    :func:`~libera_utils.footprint_matching._runner.resolve_ancillary_inputs` returns
    a *list* of staged granules per reader, but the readers (and
    :func:`~libera_utils.footprint_matching.tiling.build_tile_manager`) each take a
    single ``file_path``. This collapses the inventory to one local file per reader,
    returning ``None`` (so assembly falls back to placeholders) whenever the tree is
    not usable as-is:

    * no inventory at all (``FMATCH_ANCILLARY_PATH`` unset / nothing staged),
    * any active reader with zero or more than one staged granule (multi-granule
      readers are a follow-up; see the module ``TODO[LIBSDC-785]``), or
    * any granule in S3 (readers need a real local file; S3 materialization for
      ancillary inputs is not implemented yet -- ``TODO[LIBSDC-785]``).

    Parameters
    ----------
    ancillary_inputs : dict[str, list[pathlib.Path | cloudpathlib.S3Path]] or None
        The per-reader inventory from ``resolve_ancillary_inputs``.

    Returns
    -------
    dict[str, pathlib.Path] or None
        Reader-key -> single local file, or ``None`` when the tree is unusable.
    """
    if not ancillary_inputs:
        return None
    source_file_paths: dict[str, Path] = {}
    for reader_key, files in ancillary_inputs.items():
        if len(files) != 1:
            logger.warning(
                "Ancillary source %r has %d staged granule(s); expected exactly 1. Writing external variables as "
                "placeholders for this product.",
                reader_key,
                len(files),
            )
            return None
        only_file = files[0]
        if is_s3(only_file):
            logger.warning(
                "Ancillary source %r is staged in S3 (%s); local materialization of ancillary granules is not "
                "implemented yet. Writing external variables as placeholders.",
                reader_key,
                only_file,
            )
            return None
        source_file_paths[reader_key] = Path(str(only_file))
    return source_file_paths


def create_and_write_data_product(
    l1b_file_path: str | Path | S3Path,
    output_path: str | Path | S3Path,
    config: FmatchRunnerConfig,
    cloud_fraction_file_paths: list[str] | None = None,
    ancillary_inputs: dict[str, list[Path | S3Path]] | None = None,
) -> LiberaDataProductFilename:
    """Run footprint matching on one L1B input and write the FMATCH data product.

    Parameters
    ----------
    l1b_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to the L1B Daily input file.
    output_path : str | pathlib.Path | cloudpathlib.S3Path
        Directory / prefix in the processing dropbox where the product file is written.
    config : FmatchRunnerConfig
        Runner parameters supplying the mode and input products.
    cloud_fraction_file_paths : list[str], optional
        Camera Cloud Fraction files selected from the manifest, if any. Reading them
        is not implemented yet (``TODO[LIBSDC-785]``); they are logged as provenance
        and the ``cloud_fraction_camera`` variable is written as a placeholder.
    ancillary_inputs : dict[str, list[pathlib.Path | cloudpathlib.S3Path]], optional
        The staged ancillary inventory from
        :func:`~libera_utils.footprint_matching._runner.resolve_ancillary_inputs`.
        When it resolves to one local granule per active reader, the external variables
        and coverage/QA columns are computed; otherwise those columns are placeholders
        (see :func:`_ancillary_source_file_paths`).

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.

    Notes
    -----
    Only the L1B-derived columns are written with real values. The external-reader
    aggregates (``igbp_surface_type``, ``era5_*``, ``ssf_*``, ``cldpix_*``, ...) and the
    camera cloud-fraction values are **not** supplied here - the aggregation engine that
    computes them is still ``TODO[LIBSDC-785]`` - so ``write_fmatch_product`` writes them as
    conformant placeholders (integers as ``0``, floats as ``NaN``). A product written by this
    function is therefore **not yet consumable by SCENE-ID**: its placeholder
    ``igbp_surface_type=0`` makes
    :func:`~libera_utils.scene_identification.scene_id.calculate_trmm_surface_type` raise.
    These runners are intentionally non-operational end-to-end until that engine lands; see
    the module ``Milestone note``.
    """
    inputs = run_footprint_matching(l1b_file_path, config)

    input_file_name = AnyPath(l1b_file_path).name
    # Provenance records every file that fed this product, so a written file names its
    # own inputs even while the cloud-fraction merge itself is still future work.
    provenance = [input_file_name]
    if cloud_fraction_file_paths:
        provenance.extend(AnyPath(path).name for path in cloud_fraction_file_paths)
        # TODO[LIBSDC-785]: read the CF-CAM values and pass them as cloud_fraction_camera.
        logger.info(
            "Camera Cloud Fraction input(s) present but not yet ingested; cloud_fraction_camera "
            "will be written as a placeholder."
        )

    # Collapse the staged ancillary inventory to the one-file-per-reader map the
    # aggregation path needs; None means "aggregate nothing, placeholder those columns".
    source_file_paths = _ancillary_source_file_paths(ancillary_inputs)
    if source_file_paths is not None:
        provenance.extend(path.name for path in source_file_paths.values())
        logger.info("Aggregating external variables from %d staged ancillary source(s)", len(source_file_paths))

    logger.info("Writing %s data product for input %s", config.output_product_id.value, input_file_name)
    output_file_path = write_fmatch_product(
        config.mode,
        inputs,
        output_path,
        algorithm_version=algorithm_version(),
        input_files=",".join(provenance),
        source_file_paths=source_file_paths,
        strict=True,
    )
    logger.info(f"Wrote data product to {output_file_path.path}")
    return output_file_path


def algorithm_version() -> str:
    """Return the version recorded in the product's ``algorithm_version`` attribute.

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


def build_argument_parser(config: FmatchRunnerConfig, description: str) -> argparse.ArgumentParser:
    """Build the CLI argument parser for a FMATCH runner.

    Every runner takes the input manifest positionally.

    Parameters
    ----------
    config : FmatchRunnerConfig
        Runner parameters supplying the L1B input product label for the help text.
    description : str
        Help text describing this runner.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "manifest",
        type=str,
        help=f"Path to the input manifest file listing {config.l1b_input_product_id.value} input(s).",
    )
    return parser


def main(config: FmatchRunnerConfig, description: str, cli_args: list[str] | None = None) -> Any:
    """Shared CLI entrypoint body for a FMATCH runner.

    Parameters
    ----------
    config : FmatchRunnerConfig
        The runner's configuration.
    description : str
        Help text describing this runner.
    cli_args : list[str], optional
        Command-line arguments (primarily for testing). Defaults to ``sys.argv``.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    parser = build_argument_parser(config, description)
    args = parser.parse_args(cli_args)
    return run_algorithm(args, config)


class _as_local_path:
    """Context manager yielding a real local filesystem path for a possibly-remote input file.

    Reading an L1B product with :func:`xarray.open_dataset` (and the HDF4/HDF5 libraries underneath it)
    requires a real file on a local filesystem, because those libraries seek within the file rather
    than streaming it. A runner's manifest, however, may reference either local paths or S3 object
    urls. For S3 inputs this downloads the object to a temporary directory that is removed on exit;
    local inputs are yielded unchanged with no copy.

    Kept private to the runner layer so that ``libera_utils.io`` stays free of runner-specific
    plumbing; the SCENE-ID runner keeps its own identical helper for the same reason.

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
