"""CF-CAM processing code -- the radiometer-timescale placeholder cloud-fraction runner.

Reads an ``L1B RAD-4CH`` (Daily radiometer) file and writes the ``CF-CAM`` product on the
``RADIOMETER_TIME`` axis: one placeholder ``cloud_fraction`` / ``cloud_fraction_standard_deviation``
value per radiometer footprint. This is the radiometer-timescale mirror of FMATCH-CAM and feeds it
as its ``cloud_fraction_camera`` input, aligned 1:1 by ``RADIOMETER_TIME``. Unlike CF-CAM-CAMTIME it
performs no camera segmentation -- the footprints are the radiometer's own.

"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.cloud_fraction._runner import CfInputKind, CfRunnerConfig, run_algorithm
from libera_utils.constants import DataProductIdentifier

# Path to the CF-CAM product definition shipped as package data.
PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent / "data" / "product_definitions" / "cf_cam.yml"
)

# All the parameters that make this the radiometer-timescale CF-CAM runner (see CfRunnerConfig).
RUNNER_CONFIG = CfRunnerConfig(
    output_product_id=DataProductIdentifier.l2_cf_cam,
    l1b_input_product_id=DataProductIdentifier.l1b_rad,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="RADIOMETER_TIME",
    input_kind=CfInputKind.RADIOMETER,
    log_prefix="cf_cam",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the CF-CAM processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.cloud_fraction._runner.run_algorithm` with the CAM config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the L1B RAD-4CH input file(s). An
        ``argparse.Namespace`` (as produced by the CLI) is also accepted for convenience.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)
