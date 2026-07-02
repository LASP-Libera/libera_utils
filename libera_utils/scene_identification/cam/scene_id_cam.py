"""Scene ID CAM processing code for the Libera radiometer.

This module is the processing entrypoint for the SCENE-ID-CAM data product: the radiometer-timescale,
lowest-latency (camera / near-real-time) scene-identification product that is available from year 1 of the mission
(see the "Footprint Matching and Scene ID" design document, section 1.3, "The Five-Mode Data Product Structure").

It is written in the same style as the L1B example runner (``l1b_example/l1b.py``): the algorithm is driven by a
Libera *manifest*. The input manifest lists the CERES SSF (Single Scanner Footprint) NetCDF file(s) to process;
the runner classifies each footprint into scene IDs and writes a Libera NetCDF data product plus an output
manifest into the processing dropbox.

The heavy lifting lives in :mod:`libera_utils.scene_identification.scene_id` (the :class:`FootprintData` class).
This runner is intentionally thin: its job is to translate between the Libera pipeline's manifest-based I/O
contract and that algorithm code.
"""

import argparse
import logging
import os
import tempfile
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

# Scene classifications produced by the SCENE-ID-CAM product. CAM runs the ERBE and unfiltering classifications
# (both keyed off surface_type and cloud_fraction) but deliberately not the heavier TRMM classification. Keep this
# list in sync with the scene_id_* / scene_bin_* variables declared in scene_id_cam.yml.
SCENE_ID_CAM_SCENE_TYPES = ["erbe", "unfiltering"]

# Path to the SCENE-ID-CAM product definition that ships with libera_utils. Using importlib-free resolution via the
# installed package keeps this runner independent of the current working directory (important inside a container).
# The product definition declares the RADIOMETER_TIME axis, variable dtypes, and metadata; see the YAML for detail.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "scene_id_cam.yml"
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the SCENE-ID-CAM processing workflow from an input manifest.

    This mirrors the manifest-driven workflow used by the L1B example runner: read the input manifest, process
    every CERES SSF file it references into scene IDs, write the resulting Libera data product(s), and emit an
    output manifest describing what was produced.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the CERES SSF input file(s). An ``argparse.Namespace`` (as produced
        by :func:`main`) is also accepted for convenience when invoked as a CLI.

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
    configure_task_logging(f"scene_id_cam_{now}")

    # Step 1: Read the input manifest.
    logger.info("Step 1: Reading the input manifest file")
    # When called from the CLI, argparse hands us a Namespace; unwrap the manifest path from it. Otherwise we were
    # called directly (e.g. from a test) with a path-like object.
    if isinstance(manifest_path, argparse.Namespace):
        manifest = AnyPath(manifest_path.manifest)
    else:
        manifest = AnyPath(manifest_path)
    input_manifest = Manifest.from_file(manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")

    # The SDC provides the output location via an environment variable so the same image can write to different
    # dropboxes without code changes. This matches the L1B example's contract.
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Collect the CERES SSF input file(s) from the manifest.
    logger.info("Step 2: Collecting CERES SSF input files from the manifest")
    ssf_file_paths = collect_ssf_input_files(input_manifest)
    if not ssf_file_paths:
        raise ValueError("No CERES SSF input files found in the input manifest")

    # Step 3: Run scene identification and write one Libera data product per SSF input.
    logger.info("Step 3: Running scene identification and writing data products")
    output_data_file_paths: list[LiberaDataProductFilename] = []
    for ssf_file_path in ssf_file_paths:
        footprint_data = run_scene_identification(ssf_file_path)
        output_file = create_and_write_data_product(
            footprint_data=footprint_data,
            input_file_name=AnyPath(ssf_file_path).name,
            output_path=dropbox_path,
        )
        output_data_file_paths.append(output_file)

    # Step 4: Create the output manifest from the input manifest (carries provenance/config forward).
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


def collect_ssf_input_files(input_manifest: Manifest) -> list[str]:
    """Select the CERES SSF input files referenced by a manifest.

    A SCENE-ID input manifest references one or more CERES SSF NetCDF files. Those are *not* Libera data products,
    so they do not parse as :class:`~libera_utils.io.filenaming.LiberaDataProductFilename`. We use that fact to
    distinguish genuine SSF inputs from any Libera-named ancillary files that might also appear in the manifest.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.

    Returns
    -------
    list[str]
        The manifest filenames identified as CERES SSF inputs, in manifest order.
    """
    ssf_file_paths: list[str] = []
    for file_record in input_manifest.files:
        filename = file_record.filename
        try:
            # If this parses as a Libera data product filename, it is not a raw CERES SSF input; skip it.
            LiberaDataProductFilename.from_file_path(filename)
            logger.info("Skipping Libera-named file (not a CERES SSF input): %s", filename)
        except Exception:
            # Not a Libera product name -> treat it as a CERES SSF input file.
            logger.info("Recording CERES SSF input file: %s", filename)
            ssf_file_paths.append(filename)
    return ssf_file_paths


def run_scene_identification(ssf_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all footprints in a single CERES SSF file into scene IDs.

    Parameters
    ----------
    ssf_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to a CERES SSF NetCDF file in ``CeresSSFNOAA20FM6Ed1C`` format.

    Returns
    -------
    FootprintData
        The processed footprint data, with derived variables and scene IDs (ERBE + unfiltering) added, plus the
        ``radiometer_time`` observation-time variable used as the product's time axis.

    Notes
    -----
    :meth:`FootprintData.from_ceres_ssf` reads the file with :mod:`netCDF4`, which cannot open an S3 object
    directly. When the input lives in S3 we first materialize it to a local temporary file. Local inputs are read
    in place with no copy.
    """
    with _as_local_path(ssf_file_path) as local_ssf_path:
        logger.info("Running scene identification on %s", local_ssf_path)
        footprint_data = FootprintData.from_ceres_ssf(local_ssf_path)
        # SCENE-ID-CAM runs the ERBE and unfiltering classifications (not the default full set, which also includes
        # TRMM). With report_bin_bounds=True (the default), the property-bin bounds of each matched scene are also
        # recorded. Both scene IDs and their bin bounds are part of the SCENE-ID-CAM product definition.
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(SCENE_ID_CAM_SCENE_TYPES))
    return footprint_data


def create_and_write_data_product(
    footprint_data: FootprintData, input_file_name: str, output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
    """Write a footprint dataset as a SCENE-ID-CAM Libera NetCDF data product.

    Parameters
    ----------
    footprint_data : FootprintData
        Processed footprint data containing scene IDs.
    input_file_name : str
        Name of the CERES SSF input file, recorded on the product as provenance (``input_files`` attribute).
    output_path : str | pathlib.Path | cloudpathlib.S3Path
        Directory / prefix in the processing dropbox where the product file is written.

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.

    Raises
    ------
    FileNotFoundError
        If the SCENE-ID-CAM product definition cannot be found in the installed libera_utils package.
    """
    if not PRODUCT_DEFINITION_PATH.exists():
        raise FileNotFoundError(f"SCENE-ID-CAM product definition not found: {PRODUCT_DEFINITION_PATH}")

    # Reshape onto the Libera RADIOMETER_TIME axis (rename the internal "footprint" dimension to RADIOMETER_TIME and
    # promote radiometer_time to a coordinate) so the product aligns 1:1 with its upstream L1B radiometer product.
    product_dataset = footprint_data.to_radiometer_time_product()

    # Dynamic (per-file) global attributes required by the product definition. Static attributes (ProjectShortName,
    # Conventions, ProductID, ...) are filled in automatically from the product definition during conformance.
    # NOTE: when passing a Dataset (rather than a dict of arrays) to write_libera_data_product, dynamic attributes
    # must be set on the Dataset directly; the function rejects the dynamic_product_attributes keyword in that case.
    product_dataset.attrs["date_created"] = datetime.now(UTC).isoformat()
    product_dataset.attrs["input_files"] = input_file_name
    # TODO[LIBSDC-673]: source the algorithm version from package metadata once SCENE-ID is versioned/released.
    product_dataset.attrs["algorithm_version"] = "0.1.0"

    logger.info("Writing SCENE-ID-CAM data product for input %s", input_file_name)
    output_file_path = write_libera_data_product(
        data_product_definition=PRODUCT_DEFINITION_PATH,
        data=product_dataset,
        output_path=output_path,
        # radiometer_time carries the per-footprint observation time; the writer uses it to derive the file's
        # start/end time for the Libera filename.
        time_variable="radiometer_time",
        strict=True,
    )
    logger.info(f"Wrote data product to {output_file_path.path}")
    _ = DataProductIdentifier.scene_id_cam  # documents which product this runner emits
    return output_file_path


class _as_local_path:
    """Context manager yielding a local filesystem path for a possibly-remote input file.

    ``netCDF4.Dataset`` requires a real local file. For S3 inputs we download to a temporary directory that is
    cleaned up on exit; local inputs are yielded unchanged (no copy).
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


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the SCENE-ID-CAM runner.

    Parameters
    ----------
    cli_args : list | None
        Optional list of command-line arguments (primarily for testing). Defaults to ``sys.argv`` when None.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    parser = argparse.ArgumentParser(description="Run the Libera SCENE-ID-CAM algorithm from an input manifest.")
    parser.add_argument("manifest", type=str, help="Path to the input manifest file listing CERES SSF input(s).")
    args = parser.parse_args(cli_args)
    return algorithm(args)


if __name__ == "__main__":
    main()
