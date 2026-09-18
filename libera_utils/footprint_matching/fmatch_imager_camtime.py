"""FMATCH-IMAGER-CAMTIME processing code for the Libera WFOV camera.

This is the *camera*-timescale runner at RBSP Climate Quality latency. It inherently requires RBSP
inputs and therefore does not run during the first year of operation.

The operational input is the L1B Daily camera product (``CAM``). Unlike FMATCH-CAM-CAMTIME -- which
consumes the pseudo-footprints the cloud-fraction algorithm already wrote to CF-CAM-CAMTIME -- this
runner produces its own footprints: it does not carry cloud fraction, so there is no CF-CAM-CAMTIME
input to read from. :func:`_load_imager_camtime_footprints` segments each camera image's pixel grid
into radiometer-sized pseudo-footprints with
:func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera` -- the *same* segmentation
tooling the cloud-fraction algorithm uses, so the footprints are comparable -- run here, in this runner's
own module, rather than in the shared runner. Those pseudo-footprints become the product's records;
``CAMERA_TIME`` values therefore repeat within the product (the time identifies the source image, not a
unique footprint).

Unlike the CAM products, FMATCH-IMAGER-CAMTIME does not declare ``cloud_fraction_camera`` - its cloud
information comes from the imager readers rather than the Libera WFOV camera. The mode requires RBSP
inputs (CERES CLDPIX/SSF), which its single product definition carries alongside the ERA5 and VIIRS
fields, and so does not run during the first year of operation.
"""

import logging
from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching._runner_common import load_l1b_camera_dataset
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.footprint_matching.types import OperationalMode, PseudoFootprint

logger = logging.getLogger(__name__)

DESCRIPTION = "Run the Libera FMATCH-IMAGER-CAMTIME algorithm from an input manifest."


def _load_imager_camtime_footprints(local_input_path: Path) -> tuple[list[PseudoFootprint], None]:
    """Segment a local L1B Daily Camera file into pseudo-footprints.

    The camera-timescale footprint loader for FMATCH-IMAGER-CAMTIME (see
    :attr:`~libera_utils.footprint_matching._runner.FmatchRunnerConfig.camera_footprint_loader`).
    IMAGER-CAMTIME carries no cloud fraction, so it does not read CF-CAM-CAMTIME; it runs the same
    segmentation the cloud-fraction algorithm uses (:func:`segment_l1b_camera`) here in its own module,
    keeping the shared runner free of the segmentation dependency. The second tuple element is always
    ``None`` because this mode emits no ``cloud_fraction_camera``.

    Parameters
    ----------
    local_input_path : pathlib.Path
        Local path to an L1B Daily Camera (``CAM``) NetCDF file.

    Returns
    -------
    tuple[list[PseudoFootprint], None]
        The segmented pseudo-footprints and ``None`` (no cloud fraction).
    """
    dataset = load_l1b_camera_dataset(local_input_path)
    footprints = segment_l1b_camera(dataset, log=logger)
    return footprints, None


# All the parameters that make this the climate-quality camera-timescale runner (see FmatchRunnerConfig).
# The input is the L1B CAM product, segmented in this module by _load_imager_camtime_footprints.
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER_CAMTIME,
    output_product_id=DataProductIdentifier.aux_fmatch_imager_camtime,
    input_product_id=DataProductIdentifier.l1b_cam,
    cloud_fraction_product_id=None,
    log_prefix="fmatch_imager_camtime",
    camera_footprint_loader=_load_imager_camtime_footprints,
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
