"""Scene ID IMAGER-FLASH processing code for the Libera radiometer.

This is the radiometer-timescale imager (flash-latency) scene-identification runner. It runs ERBE, unfiltering, and
TRMM.

The operational input is the FMATCH-IMAGER-FLASH product (see :meth:`FootprintData.from_fmatch_imager_flash`): one
footprint per ``RADIOMETER_TIME`` carrying the CERES SSF clear coverage (for cloud fraction), the CERES SSF cloud
optical depth, the ERA5 winds (for surface wind), and the IGBP surface type. FMATCH-IMAGER-FLASH has NO cloud-phase
source, so the reader injects an all-NaN ``cloud_phase``: the TRMM classification therefore matches only the
clear/surface scenes that leave ``cloud_phase`` unbounded and leaves every phase-gated cloudy TRMM scene unmatched.

The runner config sets ``input_product_id=DataProductIdentifier.aux_fmatch_imager_flash`` so
:func:`libera_utils.scene_identification._runner.collect_input_files` keeps the FMATCH-IMAGER-FLASH files from the
input manifest.
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

# Scene classifications produced by the SCENE-ID-IMAGER-FLASH product. TRMM is run for parity with SCENE-ID-IMAGER,
# but is phase-limited because FMATCH-IMAGER-FLASH has no cloud-phase source (see the module docstring).
SCENE_ID_IMAGER_FLASH_SCENE_TYPES = ["erbe", "unfiltering", "trmm"]

# Path to the SCENE-ID-IMAGER-FLASH product definition.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "scene_id_imager_flash.yml"
)

# All the parameters that make this the radiometer-timescale IMAGER-FLASH runner (see SceneIdRunnerConfig). The
# input is the FMATCH-IMAGER-FLASH Libera product, read by from_fmatch_imager_flash onto the RADIOMETER_TIME axis.
RUNNER_CONFIG = SceneIdRunnerConfig(
    input_product_id=DataProductIdentifier.aux_fmatch_imager_flash,
    output_product_id=DataProductIdentifier.aux_scene_id_imager_flash,
    reader=FootprintData.from_fmatch_imager_flash,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="radiometer_time",
    scene_types=SCENE_ID_IMAGER_FLASH_SCENE_TYPES,
    log_prefix="scene_id_imager_flash",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the SCENE-ID-IMAGER-FLASH processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.scene_identification._runner.run_algorithm` with the IMAGER-FLASH config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the FMATCH-IMAGER-FLASH input file(s). An ``argparse.Namespace`` (as
        produced by :func:`main`) is also accepted for convenience when invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def collect_fmatch_imager_flash_input_files(input_manifest: Manifest) -> list[str]:
    """Select the FMATCH-IMAGER-FLASH input files referenced by a manifest.

    Wrapper around :func:`libera_utils.scene_identification._runner.collect_input_files` pinned to the
    ``FMATCH-IMAGER-FLASH`` product id.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.

    Returns
    -------
    list[str]
        The manifest filenames identified as FMATCH-IMAGER-FLASH inputs, in manifest order.
    """
    return collect_input_files(input_manifest, RUNNER_CONFIG.input_product_id)


def run_scene_identification_imager_flash(fmatch_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all footprints in a single FMATCH-IMAGER-FLASH file into scene IDs (IMAGER-FLASH configuration)."""
    return run_scene_identification(fmatch_file_path, RUNNER_CONFIG)


def create_and_write_data_product_imager_flash(
    footprint_data: FootprintData, input_file_name: str, output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
    """Write a footprint dataset as a SCENE-ID-IMAGER-FLASH Libera NetCDF data product (IMAGER-FLASH config)."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIG)


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the SCENE-ID-IMAGER-FLASH runner.

    Parameters
    ----------
    cli_args : list | None
        Optional list of command-line arguments (primarily for testing). Defaults to ``sys.argv`` when None.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    parser = argparse.ArgumentParser(
        description="Run the Libera SCENE-ID-IMAGER-FLASH algorithm from an input manifest."
    )
    parser.add_argument(
        "manifest", type=str, help="Path to the input manifest file listing FMATCH-IMAGER-FLASH input(s)."
    )
    args = parser.parse_args(cli_args)
    return algorithm(args)


if __name__ == "__main__":
    main()
