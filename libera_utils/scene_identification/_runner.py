"""Shared manifest-driven runner logic for the SCENE-ID CAM product family.

The radiometer-timescale (``cam/scene_id_cam.py``) and camera-timescale (``cam_camtime/scene_id_cam_camtime.py``)
runners are structurally identical: read an input manifest, keep the FMATCH input files of a particular product, run
scene identification on each, write the resulting SCENE-ID product, and emit an output manifest. They differ only by a
handful of parameters:

* which FMATCH product id counts as an input (``FMATCH-CAM`` vs ``FMATCH-CAM-CAMTIME``),
* which :class:`~libera_utils.scene_identification.FootprintData` factory reads it,
* which product-definition YAML / time variable the output is written against, and
* logging labels.

Rather than duplicate the ~120 lines of manifest/dropbox plumbing in both runners, that shared body lives here and is
parameterized by a small :class:`SceneIdRunnerConfig`. Each concrete runner is then a thin module that builds a config
and forwards its ``main``/``algorithm`` to :func:`run_algorithm`.
"""

import argparse
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cloudpathlib import AnyPath, S3Path

from libera_utils import Manifest, smart_copy_file
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.smart_open import is_s3
from libera_utils.logutil import configure_task_logging
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification.scene_id import standard_scene_definitions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneIdRunnerConfig:
    """Everything that distinguishes one SCENE-ID CAM-family runner from another.

    Attributes
    ----------
    input_product_id : DataProductIdentifier or None
        The Libera product id that counts as an input for this runner (e.g. ``aux_fmatch_cam`` or
        ``aux_fmatch_cam_camtime``). Files with any other product id (or unparseable names) are skipped. Pass
        ``None`` for the placeholder mode used by SCENE-ID-CAM today, where the input is a raw CERES SSF file that
        is *not* a Libera product: in that mode the runner keeps exactly the manifest files that do **not** parse
        as a Libera product filename. See :func:`collect_input_files`.
    output_product_id : DataProductIdentifier
        The SCENE-ID product this runner emits. Used only for documentation/logging; the written filename's product id
        is driven by the product definition's ``ProductID`` attribute.
    reader : Callable[[Path], FootprintData]
        The :class:`FootprintData` factory that reads one input file (e.g. ``FootprintData.from_fmatch_cam`` or
        ``FootprintData.from_fmatch_cam_camtime``).
    product_definition_path : Path
        Path to the SCENE-ID product-definition YAML the output is validated/written against.
    time_variable : str
        Name of the datetime64 coordinate variable in the written product (``radiometer_time`` or ``camera_time``).
    scene_types : list[str]
        Scene classification types to run (CAM runs ``["erbe", "unfiltering"]``).
    log_prefix : str
        Short label used in task-log filenames (e.g. ``scene_id_cam`` / ``scene_id_cam_camtime``).
    """

    input_product_id: DataProductIdentifier | None
    output_product_id: DataProductIdentifier
    reader: Callable[[Path], FootprintData]
    product_definition_path: Path
    time_variable: str
    scene_types: list[str]
    log_prefix: str


def run_algorithm(manifest_path: Path | S3Path, config: SceneIdRunnerConfig) -> Path | S3Path:
    """Run a SCENE-ID CAM-family processing workflow from an input manifest.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the FMATCH input file(s). An ``argparse.Namespace`` (as produced by a
        runner's ``main``) is also accepted for convenience when invoked as a CLI.
    config : SceneIdRunnerConfig
        The per-runner parameters (input/output product, reader, definition, time variable, scene types, log label).

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.

    Raises
    ------
    ValueError
        If the ``PROCESSING_PATH`` environment variable is not set, or if the manifest references no usable inputs.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"{config.log_prefix}_{now}")

    # Step 1: Read the input manifest.
    logger.info("Step 1: Reading the input manifest file")
    if isinstance(manifest_path, argparse.Namespace):
        manifest = AnyPath(manifest_path.manifest)
    else:
        manifest = AnyPath(manifest_path)
    input_manifest = Manifest.from_file(manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Collect the input file(s) from the manifest. In placeholder mode (input_product_id is None) these are
    # non-Libera CERES SSF files; otherwise they are the configured Libera FMATCH product.
    input_label = config.input_product_id.value if config.input_product_id is not None else "CERES SSF (placeholder)"
    logger.info("Step 2: Collecting %s input files from the manifest", input_label)
    input_file_paths = collect_input_files(input_manifest, config.input_product_id)
    if not input_file_paths:
        raise ValueError(f"No {input_label} input files found in the input manifest")

    # Step 3: Run scene identification and write data product.
    logger.info("Step 3: Running scene identification and writing data products")
    output_data_file_paths: list[LiberaDataProductFilename] = []
    for input_file_path in input_file_paths:
        footprint_data = run_scene_identification(input_file_path, config)
        output_file = create_and_write_data_product(
            footprint_data=footprint_data,
            input_file_name=AnyPath(input_file_path).name,
            output_path=dropbox_path,
            config=config,
        )
        output_data_file_paths.append(output_file)

    # Step 4: Create the output manifest from the input manifest.
    logger.info("Step 4: Creating the output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 5: Register the written data product file(s) on the output manifest.
    logger.info(f"Step 5: Adding {len(output_data_file_paths)} data file(s) to the output manifest")
    output_manifest.add_files(*[output_file.path for output_file in output_data_file_paths])

    # Step 6: Write the output manifest to the dropbox.
    logger.info("Step 6: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def collect_input_files(input_manifest: Manifest, input_product_id: DataProductIdentifier | None) -> list[str]:
    """Select the input files referenced by a manifest for this runner.

    This supports two modes, distinguished by ``input_product_id``:

    * **Libera-product mode** (``input_product_id`` is a :class:`~libera_utils.constants.DataProductIdentifier`):
      the operational case. The FMATCH inputs *are* Libera data products, so they parse as
      :class:`~libera_utils.io.filenaming.LiberaDataProductFilename` and carry a product id. We keep exactly the
      files whose product id matches ``input_product_id`` and skip everything else (unparseable names or other
      Libera products that might share the manifest).
    * **Placeholder mode** (``input_product_id`` is ``None``): the case SCENE-ID-CAM uses today, where the input is
      a raw CERES SSF file. CERES SSF files are *not* Libera products, so they do not parse as a
      ``LiberaDataProductFilename``. We use that fact to keep exactly the files that do **not** parse, and skip any
      Libera-named ancillary files that might also appear in the manifest.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.
    input_product_id : DataProductIdentifier or None
        The Libera product id to keep (e.g. ``aux_fmatch_cam`` or ``aux_fmatch_cam_camtime``), or ``None`` for the
        CERES SSF placeholder mode described above.

    Returns
    -------
    list[str]
        The manifest filenames identified as inputs, in manifest order.
    """
    input_label = input_product_id.value if input_product_id is not None else "CERES SSF (placeholder)"
    input_file_paths: list[str] = []
    for file_record in input_manifest.files:
        filename = file_record.filename
        try:
            libera_filename = LiberaDataProductFilename.from_file_path(filename)
        except Exception:
            # Not a Libera product name. In placeholder mode that is exactly the CERES SSF input we want; in
            # Libera-product mode it cannot be an FMATCH input, so skip it.
            if input_product_id is None:
                logger.info("Recording %s input file: %s", input_label, filename)
                input_file_paths.append(filename)
            else:
                logger.info("Skipping non-Libera-product file (not a %s input): %s", input_label, filename)
            continue
        # Parsed as a Libera product.
        if input_product_id is None:
            # Placeholder mode wants only non-Libera files, so a Libera-named file is not an input here.
            logger.info("Skipping Libera-named file (not a %s input): %s", input_label, filename)
        elif libera_filename.data_product_id is input_product_id:
            logger.info("Recording %s input file: %s", input_label, filename)
            input_file_paths.append(filename)
        else:
            logger.info(
                "Skipping Libera product '%s' (not %s): %s",
                libera_filename.data_product_id.value,
                input_label,
                filename,
            )
    return input_file_paths


def run_scene_identification(fmatch_file_path: str | Path | S3Path, config: SceneIdRunnerConfig) -> FootprintData:
    """Classify all footprints in a single FMATCH file into scene IDs.

    Parameters
    ----------
    fmatch_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to a Libera FMATCH NetCDF product file.
    config : SceneIdRunnerConfig
        Runner parameters supplying the reader and scene types.

    Returns
    -------
    FootprintData
        The processed footprint data, with derived variables and scene IDs added, plus the observation-time variable
        used as the product's time axis.

    Notes
    -----
    The reader (:meth:`FootprintData.from_fmatch_cam` / :meth:`FootprintData.from_fmatch_cam_camtime`) reads the file
    with :func:`xarray.open_dataset`, which we point at a real local file. When the input lives in S3 we first
    materialize it to a local temporary file; local inputs are read in place with no copy.
    """
    with _as_local_path(fmatch_file_path) as local_fmatch_path:
        logger.info("Running scene identification on %s", local_fmatch_path)
        footprint_data = config.reader(local_fmatch_path)
        # CAM runs the ERBE and unfiltering classifications (not the default full set, which also includes TRMM). With
        # report_bin_bounds=True (the default), the property-bin bounds of each matched scene are also recorded. Both
        # scene IDs and their bin bounds are part of the SCENE-ID product definition.
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(config.scene_types))
    return footprint_data


def create_and_write_data_product(
    footprint_data: FootprintData,
    input_file_name: str,
    output_path: str | Path | S3Path,
    config: SceneIdRunnerConfig,
) -> LiberaDataProductFilename:
    """Write a footprint dataset as a SCENE-ID Libera NetCDF data product.

    Parameters
    ----------
    footprint_data : FootprintData
        Processed footprint data containing scene IDs.
    input_file_name : str
        Name of the FMATCH input file, recorded on the product as provenance (``input_files`` attribute).
    output_path : str | pathlib.Path | cloudpathlib.S3Path
        Directory / prefix in the processing dropbox where the product file is written.
    config : SceneIdRunnerConfig
        Runner parameters supplying the product definition path and time variable.

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.

    Raises
    ------
    FileNotFoundError
        If the SCENE-ID product definition cannot be found in the installed libera_utils package.
    """
    if not config.product_definition_path.exists():
        raise FileNotFoundError(f"SCENE-ID product definition not found: {config.product_definition_path}")

    # Finalize onto the product's time axis (promote the time variable to a coordinate; the data already lives on the
    # correct dimension) so the product aligns 1:1 with its upstream product.
    product_dataset = footprint_data.to_time_product(config.time_variable)

    product_dataset.attrs["date_created"] = datetime.now(UTC).isoformat()
    product_dataset.attrs["input_files"] = input_file_name
    # TODO[LIBSDC-673]: source the algorithm version from package metadata once SCENE-ID is versioned/released.
    product_dataset.attrs["algorithm_version"] = "0.1.0"

    logger.info("Writing %s data product for input %s", config.output_product_id.value, input_file_name)
    output_file_path = write_libera_data_product(
        data_product_definition=config.product_definition_path,
        data=product_dataset,
        output_path=output_path,
        time_variable=config.time_variable,
        strict=True,
    )
    logger.info(f"Wrote data product to {output_file_path.path}")
    return output_file_path


class _as_local_path:
    """Context manager yielding a local filesystem path for a possibly-remote input file.

    Reading an FMATCH product with :func:`xarray.open_dataset` requires a real local file. For S3 inputs we download to
    a temporary directory that is cleaned up on exit; local inputs are yielded unchanged (no copy).
    """

    def __init__(self, source_path: str | Path | S3Path):
        self._source_path = AnyPath(source_path)
        self._tempdir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if is_s3(self._source_path):
            # Materialize the S3 object locally so netCDF4 can open it.
            self._tempdir = tempfile.TemporaryDirectory()
            local_path = Path(self._tempdir.name) / self._source_path.name
            smart_copy_file(self._source_path, local_path)
            return local_path
        # Already local; hand back a plain pathlib.Path.
        return Path(str(self._source_path))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
