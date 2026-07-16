"""Scene ID CAM-CAMTIME processing code for the Libera radiometer.

This is the *camera*-timescale runner: it reads ``FMATCH-CAM-CAMTIME`` (one pseudo-footprint per ``CAMERA_TIME``) and
writes ``SCENE-ID-CAM-CAMTIME``. It mirrors the radiometer-timescale runner in ``../cam/scene_id_cam.py`` and shares the
same manifest/dropbox plumbing from :mod:`libera_utils.scene_identification._runner`. The distinctive behavior of this
product is that it additionally carries the FMATCH footprint *identifier* variables (camera time, camera pixel-block
indices, PSF bounding box, and boresight geolocation) straight through from the input, so a classified scene can be
traced back to the exact camera pixels and ground footprint it came from. That passthrough is performed by
:meth:`FootprintData.from_fmatch_cam_camtime` and declared in ``scene_id_cam_camtime.yml``.
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

# Scene classifications produced by the SCENE-ID-CAM-CAMTIME product. Identical to the radiometer-timescale CAM product:
# ERBE and unfiltering (keyed off surface_type and cloud_fraction), but not TRMM. Keep this list in sync with the
# scene_id_* / scene_bin_* variables declared in scene_id_cam_camtime.yml.
SCENE_ID_CAM_CAMTIME_SCENE_TYPES = ["erbe", "unfiltering"]

# Path to the SCENE-ID-CAM-CAMTIME product definition that ships with libera_utils.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "scene_id_cam_camtime.yml"
)

# All the parameters that make this the camera-timescale CAM runner (see SceneIdRunnerConfig). Note the CAMERA_TIME
# axis: the reader is from_fmatch_cam_camtime and the written time coordinate is "camera_time".
RUNNER_CONFIG = SceneIdRunnerConfig(
    input_product_id=DataProductIdentifier.aux_fmatch_cam_camtime,
    output_product_id=DataProductIdentifier.aux_scene_id_cam_camtime,
    reader=FootprintData.from_fmatch_cam_camtime,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="camera_time",
    scene_types=SCENE_ID_CAM_CAMTIME_SCENE_TYPES,
    log_prefix="scene_id_cam_camtime",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the SCENE-ID-CAM-CAMTIME processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.scene_identification._runner.run_algorithm` with the CAM-CAMTIME config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the FMATCH-CAM-CAMTIME input file(s). An ``argparse.Namespace`` (as
        produced by :func:`main`) is also accepted for convenience when invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def collect_fmatch_cam_camtime_input_files(input_manifest: Manifest) -> list[str]:
    """Select the FMATCH-CAM-CAMTIME input files referenced by a manifest.

    Wrapper around :func:`libera_utils.scene_identification._runner.collect_input_files` pinned to the
    ``FMATCH-CAM-CAMTIME`` product id.

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.

    Returns
    -------
    list[str]
        The manifest filenames identified as FMATCH-CAM-CAMTIME inputs, in manifest order.
    """
    return collect_input_files(input_manifest, DataProductIdentifier.aux_fmatch_cam_camtime)


def run_scene_identification_cam_camtime(fmatch_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all pseudo-footprints in a single FMATCH-CAM-CAMTIME file into scene IDs (CAM-CAMTIME configuration)."""
    return run_scene_identification(fmatch_file_path, RUNNER_CONFIG)


def create_and_write_data_product_cam_camtime(
    footprint_data: FootprintData, input_file_name: str, output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
    """Write a footprint dataset as a SCENE-ID-CAM-CAMTIME Libera NetCDF data product (CAM-CAMTIME configuration)."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIG)


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the SCENE-ID-CAM-CAMTIME runner.

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
        description="Run the Libera SCENE-ID-CAM-CAMTIME algorithm from an input manifest."
    )
    parser.add_argument(
        "manifest", type=str, help="Path to the input manifest file listing FMATCH-CAM-CAMTIME input(s)."
    )
    args = parser.parse_args(cli_args)
    return algorithm(args)


if __name__ == "__main__":
    main()
