"""FMATCH-IMAGER-CAMTIME processing code for the Libera WFOV camera.

This is the *camera*-timescale runner at RBSP Climate Quality latency. It inherently requires RBSP
inputs and therefore does not run during the first year of operation.

The operational input is the L1B Daily camera product (``CAM``). Like FMATCH-CAM-CAMTIME, this runner
segments each camera image's pixel grid into radiometer-sized pseudo-footprints with
:func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`, and those
pseudo-footprints become the product's records; ``CAMERA_TIME`` values therefore repeat within the
product (the time identifies the source image, not a unique footprint). The two camera-timescale
modes differ only in their declared variable set, which follows from IMAGER-CAMTIME's higher latency
rank and correspondingly larger active reader set.

Unlike the CAM products, FMATCH-IMAGER-CAMTIME does not declare ``cloud_fraction_camera`` - its cloud
information comes from the imager readers rather than the Libera WFOV camera - so this runner takes
no cloud-fraction input. It also exposes no ``--post-year-one`` option: because the mode already
requires RBSP inputs, its single product definition *is* the post-year-one one (see
:class:`libera_utils.footprint_matching.types.FmatchVariant`).
"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.types import OperationalMode

DESCRIPTION = "Run the Libera FMATCH-IMAGER-CAMTIME algorithm from an input manifest."

# All the parameters that make this the climate-quality camera-timescale runner (see
# FmatchRunnerConfig).
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER_CAMTIME,
    output_product_id=DataProductIdentifier.aux_fmatch_imager_camtime,
    l1b_input_product_id=DataProductIdentifier.l1b_cam,
    cloud_fraction_product_id=None,
    supports_variant_override=False,
    log_prefix="fmatch_imager_camtime",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the FMATCH-IMAGER-CAMTIME processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.footprint_matching._runner.run_algorithm` with the
    IMAGER-CAMTIME config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the L1B CAM input file(s). An
        ``argparse.Namespace`` (as produced by :func:`main`) is also accepted for convenience when
        invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the FMATCH-IMAGER-CAMTIME runner.

    Parameters
    ----------
    cli_args : list | None
        Optional list of command-line arguments (primarily for testing). Defaults to ``sys.argv``
        when None.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return _main(RUNNER_CONFIG, DESCRIPTION, cli_args)


if __name__ == "__main__":
    main()
