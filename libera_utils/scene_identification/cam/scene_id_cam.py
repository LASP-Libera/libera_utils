"""Scene ID CAM processing code for the Libera radiometer.

This is the *radiometer*-timescale runner: it identifies scenes for one footprint per ``RADIOMETER_TIME`` and writes
``SCENE-ID-CAM``. It is the sibling of the camera-timescale runner in ``../cam_camtime/scene_id_cam_camtime.py`` and
shares the same manifest/dropbox plumbing from :mod:`libera_utils.scene_identification._runner`; the two differ only by
their input product, the :class:`~libera_utils.scene_identification.FootprintData` factory used to read it, and the
output product definition / time axis.

The operational input to this product is FMATCH-CAM (see :meth:`FootprintData.from_fmatch_cam`). That reader is not
implemented yet (the FMATCH step is a separate milestone), so this runner currently reads placeholder **CERES SSF**
files via :meth:`FootprintData.from_ceres_ssf`. CERES SSF files are not Libera products, so the runner config sets
``input_product_id=None`` to select the "keep non-Libera-product files" input-collection mode in
:func:`libera_utils.scene_identification._runner.collect_input_files`.

TODO[LIBSDC-673]: switch ``reader`` to ``FootprintData.from_fmatch_cam`` and ``input_product_id`` to
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
# keyed off surface_type and cloud_fraction) but deliberately not TRMM. Keep this list in sync with the scene_id_* /
# scene_bin_* variables declared in scene_id_cam.yml.
SCENE_ID_CAM_SCENE_TYPES = ["erbe", "unfiltering"]

# Path to the SCENE-ID-CAM product definition that ships with libera_utils.
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
        Path to the input manifest file listing the CERES SSF input file(s). An ``argparse.Namespace`` (as produced
        by :func:`main`) is also accepted for convenience when invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def collect_ssf_input_files(input_manifest: Manifest) -> list[str]:
    """Select the CERES SSF input files referenced by a manifest.

    Wrapper around :func:`libera_utils.scene_identification._runner.collect_input_files` in placeholder mode
    (``input_product_id=None``), which keeps exactly the manifest files that do not parse as a Libera product
    filename (i.e. the raw CERES SSF inputs).

    Parameters
    ----------
    input_manifest : Manifest
        The input manifest to inspect.

    Returns
    -------
    list[str]
        The manifest filenames identified as CERES SSF inputs, in manifest order.
    """
    return collect_input_files(input_manifest, RUNNER_CONFIG.input_product_id)


def run_scene_identification_cam(ssf_file_path: str | Path | S3Path) -> FootprintData:
    """Classify all footprints in a single CERES SSF file into scene IDs (CAM configuration)."""
    return run_scene_identification(ssf_file_path, RUNNER_CONFIG)


def create_and_write_data_product_cam(
    footprint_data: FootprintData, input_file_name: str, output_path: str | Path | S3Path
) -> LiberaDataProductFilename:
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
    parser.add_argument("manifest", type=str, help="Path to the input manifest file listing CERES SSF input(s).")
    args = parser.parse_args(cli_args)
    return algorithm(args)


if __name__ == "__main__":
    main()
