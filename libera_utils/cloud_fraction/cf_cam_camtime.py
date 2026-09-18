"""CF-CAM-CAMTIME processing code -- the camera-timescale placeholder cloud-fraction runner.

Reads an ``L1B CAM`` (Daily Camera) file, segments it into camera pseudo-footprints with the same
segmentation FMATCH uses, and writes the ``CF-CAM-CAMTIME`` product on the ``(CAMERA_TIME,
PSEUDOFOOTPRINT)`` grid. Each record gets a placeholder ``cloud_fraction`` /
``cloud_fraction_standard_deviation`` value plus the ``camera_pixel_{x,y}_{min,max}`` pixel-block
provenance, so the product aligns 1:1 with FMATCH-CAM-CAMTIME and feeds it as its
``cloud_fraction_camera`` input.

"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.cloud_fraction._runner import CfInputKind, CfRunnerConfig, run_algorithm
from libera_utils.constants import DataProductIdentifier

# Path to the CF-CAM-CAMTIME product definition shipped as package data.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "cf_cam_camtime.yml"
)

# All the parameters that make this the camera-timescale CF-CAM runner (see CfRunnerConfig).
RUNNER_CONFIG = CfRunnerConfig(
    output_product_id=DataProductIdentifier.l2_cf_cam_camtime,
    l1b_input_product_id=DataProductIdentifier.l1b_cam,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="CAMERA_TIME",
    input_kind=CfInputKind.CAMERA,
    log_prefix="cf_cam_camtime",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the CF-CAM-CAMTIME processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.cloud_fraction._runner.run_algorithm` with the CAM-CAMTIME
    config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the L1B CAM input file(s). An ``argparse.Namespace``
        (as produced by the CLI) is also accepted for convenience.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)
