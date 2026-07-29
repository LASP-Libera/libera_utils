"""Shared manifest-driven runner logic for the FMATCH product family.

The five FMATCH runners (``cam/``, ``cam_camtime/``, ``imager_flash/``, ``imager/``,
``imager_camtime/``) are structurally identical: read an input manifest, locate the
staged ancillary granules, keep the L1B input files of a particular product, run
footprint matching on each, write the resulting FMATCH product, and emit an output
manifest. They differ only by a handful of parameters:

* which :class:`~libera_utils.footprint_matching.types.OperationalMode` they produce
  (which in turn drives the product definition, the time coordinate, and the active
  reader set),
* which L1B product counts as their input (``RAD-4CH`` vs ``CAM``),
* whether they consume the optional Camera Cloud Fraction product, and
* whether they expose the input-availability variant as a CLI option.

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
    :mod:`libera_utils.footprint_matching.ancillary` for its layout. Currently
    resolved non-strictly, because the engine that consumes the granules is not
    implemented yet.

Milestone note
--------------
Footprint matching's PSF aggregation and derived-geometry engines are still
``TODO[LIBSDC-785]`` stubs, so the written products carry real values only for the
L1B-derived columns; every other variable is a conformant placeholder (structurally
valid but numerically meaningless; see
:mod:`libera_utils.footprint_matching.product`). The runners themselves do not change
when those engines land.
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

from cloudpathlib import AnyPath, S3Path

from libera_utils.footprint_matching.ancillary import log_ancillary_inventory, resolve_ancillary_inputs
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.footprint_matching.l1b_inputs import load_l1b_camera_dataset, load_l1b_radiometer_inputs
from libera_utils.footprint_matching.product import (
    fmatch_time_variable,
    is_camera_timescale_mode,
    write_fmatch_product,
)
from libera_utils.footprint_matching.types import FmatchVariant, OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import is_s3, smart_copy_file
from libera_utils.logutil import configure_task_logging

if TYPE_CHECKING:
    import numpy as np

    from libera_utils.constants import DataProductIdentifier
    from libera_utils.footprint_matching.camera_segmentation import PseudoFootprint

logger = logging.getLogger(__name__)


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
    supports_variant_override : bool
        Whether this runner's CLI exposes ``--post-year-one``. True only for
        FMATCH-IMAGER, the sole mode with a distinct post-year-one definition; see
        :class:`~libera_utils.footprint_matching.types.FmatchVariant`.
    log_prefix : str
        Short label used in task-log filenames (e.g. ``fmatch_cam``).
    """

    mode: OperationalMode
    output_product_id: DataProductIdentifier
    l1b_input_product_id: DataProductIdentifier
    cloud_fraction_product_id: DataProductIdentifier | None
    supports_variant_override: bool
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
    variant: FmatchVariant = FmatchVariant.YEAR_ONE,
) -> Path | S3Path:
    """Run a FMATCH processing workflow from an input manifest.

    Parameters
    ----------
    manifest_path : Path | S3Path | argparse.Namespace
        Path to the input manifest file listing the L1B input file(s). An
        ``argparse.Namespace`` (as produced by a runner's ``main``) is also accepted
        for convenience when invoked as a CLI; its ``manifest`` attribute is used, and
        its ``post_year_one`` flag, when present, selects the variant.
    config : FmatchRunnerConfig
        The per-runner parameters (mode, input products, variant support, log label).
    variant : FmatchVariant, optional
        Input-availability variant. Defaults to ``YEAR_ONE`` (production). Overridden
        by ``manifest_path.post_year_one`` when a Namespace carrying that flag is
        passed.

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
    manifest, variant = _resolve_manifest_and_variant(manifest_path, config, variant)
    input_manifest = Manifest.from_file(manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")
    logger.info("Running %s (%s variant)", config.mode.value, variant.value)

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Locate the staged ancillary granules for this mode/variant.
    #
    # These are resolved (and inventoried in the log) even though nothing consumes
    # them yet: the PSF aggregation engine is still a TODO[LIBSDC-785] stub. Doing it
    # now means an operator can confirm staging is correct from the task log before
    # the engine exists, and the runner will not need to change when it lands.
    logger.info("Step 2: Locating staged ancillary inputs")
    ancillary_inputs = resolve_ancillary_inputs(config.mode, variant)
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
            variant=variant,
            cloud_fraction_file_paths=cloud_fraction_file_paths,
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


def _resolve_manifest_and_variant(
    manifest_path: Path | S3Path | argparse.Namespace,
    config: FmatchRunnerConfig,
    variant: FmatchVariant,
) -> tuple[Path | S3Path, FmatchVariant]:
    """Normalize the manifest argument and pick the effective variant.

    Accepting an ``argparse.Namespace`` lets a runner's ``main`` forward its parsed
    args straight through, which is how the SCENE-ID runners are written too. When the
    Namespace carries the ``--post-year-one`` flag it takes precedence over the
    ``variant`` argument, since it is the operator's explicit choice.
    """
    if not isinstance(manifest_path, argparse.Namespace):
        return AnyPath(manifest_path), variant

    manifest = AnyPath(manifest_path.manifest)
    if getattr(manifest_path, "post_year_one", False):
        if not config.supports_variant_override:
            # Defensive: only the IMAGER runner defines the flag, so this indicates the
            # flag was wired onto a runner whose product has no post-year-one definition.
            raise ValueError(f"{config.mode.value} does not support the post-year-one variant.")
        return manifest, FmatchVariant.POST_YEAR_ONE
    return manifest, variant


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


def create_and_write_data_product(
    l1b_file_path: str | Path | S3Path,
    output_path: str | Path | S3Path,
    config: FmatchRunnerConfig,
    variant: FmatchVariant = FmatchVariant.YEAR_ONE,
    cloud_fraction_file_paths: list[str] | None = None,
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
    variant : FmatchVariant, optional
        Input-availability variant used to resolve the product definition.
    cloud_fraction_file_paths : list[str], optional
        Camera Cloud Fraction files selected from the manifest, if any. Reading them
        is not implemented yet (``TODO[LIBSDC-785]``); they are logged as provenance
        and the ``cloud_fraction_camera`` variable is written as a placeholder.

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.
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

    logger.info("Writing %s data product for input %s", config.output_product_id.value, input_file_name)
    output_file_path = write_fmatch_product(
        config.mode,
        inputs,
        output_path,
        variant=variant,
        algorithm_version=algorithm_version(),
        input_files=",".join(provenance),
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

    Every runner takes the input manifest positionally. The FMATCH-IMAGER runner
    additionally gets ``--post-year-one``, because it is the only mode with a distinct
    post-year-one product definition; adding the flag elsewhere would offer a choice
    that does not exist.

    Parameters
    ----------
    config : FmatchRunnerConfig
        Runner parameters; ``supports_variant_override`` decides whether the variant
        flag is added.
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
    if config.supports_variant_override:
        parser.add_argument(
            "--post-year-one",
            action="store_true",
            help=(
                "Use the post-year-one (RBSP CLDPIX/SSF) product definition and reader set instead of the "
                "year-one (ERA5-substitute) production default. Only valid once RBSP products are flowing."
            ),
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
