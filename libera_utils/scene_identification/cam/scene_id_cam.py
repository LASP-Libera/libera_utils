"""Scene ID CAM processing code for the Libera radiometer.

This is the *radiometer*-timescale runner for the lowest-latency (camera / near-real-time) scene-identification product.

The operational input to this product is FMATCH-CAM (see :meth:`FootprintData.from_fmatch_cam`). That reader is not
implemented yet (the FMATCH step is a separate milestone), so this runner currently reads placeholder **CERES SSF**
files via :meth:`FootprintData.from_ceres_ssf`. CERES SSF files are not Libera products, so the runner config sets
``input_product_id=None`` to select the "keep non-Libera-product files" input-collection mode in
:func:`libera_utils.scene_identification._runner.collect_input_files`.

TODO[LIBSDC-794]: switch ``reader`` to ``FootprintData.from_fmatch_cam`` and ``input_product_id`` to
``DataProductIdentifier.aux_fmatch_cam`` once the FMATCH-CAM product format is available.
"""

import argparse
from pathlib import Path

from cloudpathlib import S3Path

from libera_utils import Manifest
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification._runner import (
    SceneIdRunnerConfig,
    collect_input_files,
    create_and_write_data_product,
    run_algorithm,
    run_scene_identification,
)

# Scene classifications produced by the SCENE-ID-CAM product. CAM runs the ERBE and unfiltering classifications (both
# keyed off surface_type and cloud_fraction) but not TRMM since the classification variables are unavailable.
SCENE_ID_CAM_SCENE_TYPES = ["erbe", "unfiltering"]

# Path to the SCENE-ID-CAM product definition.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "scene_id_cam.yml"
)

# All the parameters that make this the radiometer-timescale CAM runner (see SceneIdRunnerConfig). Note input_product_id
# is None: the current input is a placeholder CERES SSF file (not a Libera product), and the reader is from_ceres_ssf.
RUNNER_CONFIG = SceneIdRunnerConfig(
    input_product_id=None,
    output_product_id=DataProductIdentifier.aux_scene_id_cam,
    reader=FootprintData.from_ceres_ssf,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="radiometer_time",
    scene_types=SCENE_ID_CAM_SCENE_TYPES,
    log_prefix="scene_id_cam",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the SCENE-ID-CAM processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.scene_identification._runner.run_algorithm` with the CAM config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the FMATCH-CAM input file(s). An ``argparse.Namespace`` (as produced
        by :func:`main`) is also accepted for convenience when invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def collect_fmatch_cam_input_files(input_manifest: Manifest) -> list[str]:
    """Select the FMATCH-CAM input files referenced by a manifest.

    Wrapper around :func:`libera_utils.scene_identification._runner.collect_input_files` in placeholder mode
    (``input_product_id=None``), which keeps exactly the manifest files that do not parse as a Libera product
    filename (i.e. the raw CERES SSF inputs).
    In operational daily processing the SCENE-ID-CAM input manifest references one or more FMATCH-CAM files. Those
    *are* Libera data products, so they parse as :class:`~libera_utils.io.filenaming.LiberaDataProductFilename` and
    carry a :class:`~libera_utils.constants.DataProductIdentifier`. We keep exactly the files whose product ID is
    ``FMATCH-CAM`` and skip everything else (unparsable names or other Libera products that might share the
    manifest). This is the inverse of the CERES SSF selection heritage, where the raw SSF inputs were the files that
    did *not* parse as Libera products.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.

    Returns
    -------
    list[str]
        The manifest filenames identified as FMATCH-CAM inputs, in manifest order.
    """
    fmatch_file_paths: list[str] = []
    for file_record in input_manifest.files:
        filename = file_record.filename
        try:
            libera_filename = LiberaDataProductFilename.from_file_path(filename)
        except Exception:
            # Not a Libera product name at all -> cannot be a FMATCH-CAM input; skip it.
            logger.info("Skipping non-Libera-product file (not a FMATCH-CAM input): %s", filename)
            continue
        # Parsed as a Libera product: keep it only if it is the FMATCH-CAM product this runner consumes.
        if libera_filename.data_product_id is DataProductIdentifier.anc_fmatch_cam:
            logger.info("Recording FMATCH-CAM input file: %s", filename)
            fmatch_file_paths.append(filename)
        else:
            logger.info(
                "Skipping Libera product '%s' (not FMATCH-CAM): %s", libera_filename.data_product_id.value, filename
            )
    return fmatch_file_paths

    return collect_input_files(input_manifest, RUNNER_CONFIG.input_product_id)

def run_scene_identification(fmatch_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all footprints in a single FMATCH-CAM file into scene IDs.

    Parameters
    ----------
    fmatch_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to a Libera FMATCH-CAM NetCDF product file.
def run_scene_identification_cam(ssf_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all footprints in a single CERES SSF file into scene IDs (CAM configuration)."""
    return run_scene_identification(ssf_file_path, RUNNER_CONFIG)


    Notes
    -----
    :meth:`FootprintData.from_fmatch_cam` reads the file with :func:`xarray.open_dataset`, which we point at a real
    local file. When the input lives in S3 we first materialize it to a local temporary file; local inputs are read
    in place with no copy.
    """
    with _as_local_path(fmatch_file_path) as local_fmatch_path:
        logger.info("Running scene identification on %s", local_fmatch_path)
        footprint_data = FootprintData.from_fmatch_cam(local_fmatch_path)
        # SCENE-ID-CAM runs the ERBE and unfiltering classifications (not the default full set, which also includes
        # TRMM). With report_bin_bounds=True (the default), the property-bin bounds of each matched scene are also
        # recorded. Both scene IDs and their bin bounds are part of the SCENE-ID-CAM product definition.
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(SCENE_ID_CAM_SCENE_TYPES))
    return footprint_data


def create_and_write_data_product(
def create_and_write_data_product_cam(
    footprint_data: FootprintData, input_file_name: str, output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
    """Write a footprint dataset as a SCENE-ID-CAM Libera NetCDF data product.

    Parameters
    ----------
    footprint_data : FootprintData
        Processed footprint data containing scene IDs.
    input_file_name : str
        Name of the FMATCH-CAM input file, recorded on the product as provenance (``input_files`` attribute).
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

    # Finalize onto the Libera RADIOMETER_TIME axis (promote radiometer_time to a coordinate; the data already
    # lives on the RADIOMETER_TIME dimension) so the product aligns 1:1 with its upstream product.
    product_dataset = footprint_data.to_radiometer_time_product()

    product_dataset.attrs["date_created"] = datetime.now(UTC).isoformat()
    product_dataset.attrs["input_files"] = input_file_name
    # TODO[LIBSDC-673]: source the algorithm version from package metadata once SCENE-ID is versioned/released.
    product_dataset.attrs["algorithm_version"] = "0.1.0"

    logger.info("Writing SCENE-ID-CAM data product for input %s", input_file_name)
    output_file_path = write_libera_data_product(
        data_product_definition=PRODUCT_DEFINITION_PATH,
        data=product_dataset,
        output_path=output_path,
        time_variable="radiometer_time",
        strict=True,
    )
    logger.info(f"Wrote data product to {output_file_path.path}")
    _ = DataProductIdentifier.scene_id_cam  # documents which product this runner emits
    return output_file_path


class _as_local_path:
    """Context manager yielding a local filesystem path for a possibly-remote input file.

    Reading an FMATCH-CAM product with :func:`xarray.open_dataset` requires a real local file. For S3 inputs we
    download to a temporary directory that is cleaned up on exit; local inputs are yielded unchanged (no copy).
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
    """Write a footprint dataset as a SCENE-ID-CAM Libera NetCDF data product (CAM configuration)."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIG)


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
    parser.add_argument("manifest", type=str, help="Path to the input manifest file listing FMATCH-CAM input(s).")
    args = parser.parse_args(cli_args)
    return algorithm(args)


if __name__ == "__main__":
    main()
