"""Scene ID IMAGER processing code for the Libera radiometer.

This is the radiometer-timescale imager scene-identification runner. Unlike SCENE-ID-CAM, it runs the full TRMM
classification in addition to ERBE and unfiltering.

The operational input is the FMATCH-IMAGER product (see
:meth:`FootprintData.from_fmatch_imager`): one footprint per ``RADIOMETER_TIME`` carrying the CERES
SSF clear coverage (for cloud fraction), the RBSP CLDPIX cloud optical depth and particle phase (for TRMM), the
ERA5 winds (for surface wind), and the IGBP surface type. The runner config sets
``input_product_id=DataProductIdentifier.aux_fmatch_imager`` so
:func:`libera_utils.scene_identification._runner.collect_input_files` keeps the FMATCH-IMAGER files from the input
manifest. :meth:`FootprintData.from_fmatch_imager` raises a clear error if handed a file that lacks the RBSP
ssf/cldpix variables (e.g. a FMATCH-IMAGER-FLASH product).
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

# Scene classifications produced by the SCENE-ID-IMAGER product. The FMATCH-IMAGER carries the extra TRMM inputs
# (surface wind, cloud phase, optical depth), so this product runs TRMM in addition to ERBE and unfiltering.
SCENE_ID_IMAGER_SCENE_TYPES = ["erbe", "unfiltering", "trmm"]

# Path to the SCENE-ID-IMAGER product definition.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "scene_id_imager.yml"
)

# All the parameters that make this the radiometer-timescale IMAGER runner (see SceneIdRunnerConfig). The input is
# the FMATCH-IMAGER Libera product, read by from_fmatch_imager onto the RADIOMETER_TIME axis.
RUNNER_CONFIG = SceneIdRunnerConfig(
    input_product_id=DataProductIdentifier.aux_fmatch_imager,
    output_product_id=DataProductIdentifier.aux_scene_id_imager,
    reader=FootprintData.from_fmatch_imager,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="RADIOMETER_TIME",
    scene_types=SCENE_ID_IMAGER_SCENE_TYPES,
    log_prefix="scene_id_imager",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the SCENE-ID-IMAGER processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.scene_identification._runner.run_algorithm` with the IMAGER config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the FMATCH-IMAGER input file(s). An ``argparse.Namespace`` (as
        produced by :func:`main`) is also accepted for convenience when invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def collect_fmatch_imager_input_files(input_manifest: Manifest) -> list[str]:
    """Select the FMATCH-IMAGER input files referenced by a manifest.

    Wrapper around :func:`libera_utils.scene_identification._runner.collect_input_files` for the configured
    FMATCH-IMAGER product id, which keeps exactly the manifest files whose Libera product id is FMATCH-IMAGER.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.

    Returns
    -------
    list[str]
        The manifest filenames identified as FMATCH-IMAGER inputs, in manifest order.
    """
    return collect_input_files(input_manifest, RUNNER_CONFIG.input_product_id)


def run_scene_identification_imager(fmatch_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all footprints in a single FMATCH-IMAGER file into scene IDs (IMAGER configuration)."""
    return run_scene_identification(fmatch_file_path, RUNNER_CONFIG)


def create_and_write_data_product_imager(
    footprint_data: FootprintData, input_file_name: str, output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
    """Write a footprint dataset as a SCENE-ID-IMAGER Libera NetCDF data product (IMAGER configuration)."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIG)


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the SCENE-ID-IMAGER runner.

    Parameters
    ----------
    cli_args : list | None
        Optional list of command-line arguments (primarily for testing). Defaults to ``sys.argv`` when None.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    parser = argparse.ArgumentParser(description="Run the Libera SCENE-ID-IMAGER algorithm from an input manifest.")
    parser.add_argument("manifest", type=str, help="Path to the input manifest file listing FMATCH-IMAGER input(s).")
    args = parser.parse_args(cli_args)
    return algorithm(args)


if __name__ == "__main__":
    main()
