"""FMATCH-CAM-CAMTIME processing code for the Libera WFOV camera.

This is the *camera*-timescale runner for the lowest-latency (camera / near-real-time)
footprint-matching product. It runs continuously from mission start.

Its operational input is the CF-CAM-CAMTIME product. Camera segmentation happens once, upstream, in
the cloud-fraction algorithm (which writes the full pseudo-footprint record -- geolocation, viewing
geometry, PSF bounding box, quality flags -- alongside the cloud fraction into CF-CAM-CAMTIME). This
runner therefore does not segment anything itself: :func:`_load_cam_camtime_footprints` reconstructs
the pseudo-footprints from that product with
:func:`libera_utils.footprint_matching.product.pseudofootprints_from_camtime_dataset`, and reads the
per-footprint cloud fraction from the same file. The runner config sets
``input_product_id=DataProductIdentifier.l2_cf_cam_camtime`` so
:func:`libera_utils.footprint_matching._runner.select_manifest_files_by_product_id` keeps the
CF-CAM-CAMTIME files from the input manifest.

Note that ``CAMERA_TIME`` values repeat within the product: every pseudo-footprint from one image
shares that image's time, because the time identifies the source image rather than a unique footprint.
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.product import (
    camtime_real_cell_mask,
    pseudofootprints_from_camtime_dataset,
)
from libera_utils.footprint_matching.types import OperationalMode, PseudoFootprint

logger = logging.getLogger(__name__)

DESCRIPTION = "Run the Libera FMATCH-CAM-CAMTIME algorithm from an input manifest."


def _load_cam_camtime_footprints(local_input_path: Path) -> tuple[list[PseudoFootprint], np.ndarray | None]:
    """Reconstruct pseudo-footprints and cloud fraction from a local CF-CAM-CAMTIME file.

    The camera-timescale footprint loader for FMATCH-CAM-CAMTIME (see
    :attr:`~libera_utils.footprint_matching._runner.FmatchRunnerConfig.camera_footprint_loader`). No
    segmentation happens here -- the footprints were produced once by the cloud-fraction algorithm and
    are simply rebuilt from the product, and the cloud fraction is flattened with the identical
    real-cell mask so it aligns 1:1 with them.

    Parameters
    ----------
    local_input_path : pathlib.Path
        Local path to a CF-CAM-CAMTIME NetCDF file.

    Returns
    -------
    tuple[list[PseudoFootprint], numpy.ndarray | None]
        The reconstructed pseudo-footprints and their per-footprint ``cloud_fraction_camera`` values.
    """
    with xr.open_dataset(local_input_path) as cf_dataset:
        cf_dataset = cf_dataset.load()
    footprints = pseudofootprints_from_camtime_dataset(cf_dataset)
    cloud_fraction_camera: np.ndarray | None = None
    if "cloud_fraction" in cf_dataset:
        # Boolean-mask indexing returns row-major order, matching the reconstruction order.
        mask = camtime_real_cell_mask(cf_dataset)
        cloud_fraction_camera = np.asarray(cf_dataset["cloud_fraction"].values, dtype=np.float32)[mask]
    return footprints, cloud_fraction_camera


# All the parameters that make this the camera-timescale CAM runner (see FmatchRunnerConfig). The input
# is the CF-CAM-CAMTIME product, which carries both the segmentation-derived pseudo-footprints and the
# cloud fraction. No RBSP-sourced readers are active at its latency rank.
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.CAM_CAMTIME,
    output_product_id=DataProductIdentifier.aux_fmatch_cam_camtime,
    input_product_id=DataProductIdentifier.l2_cf_cam_camtime,
    cloud_fraction_product_id=None,  # bundled in input_product_id (CF-CAM-CAMTIME)
    log_prefix="fmatch_cam_camtime",
    camera_footprint_loader=_load_cam_camtime_footprints,
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the FMATCH-CAM-CAMTIME processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.footprint_matching._runner.run_algorithm` with the
    CAM-CAMTIME config.

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
    """CLI entrypoint for the FMATCH-CAM-CAMTIME runner.

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
