"""FMATCH-CAM-CAMTIME processing code for the Libera WFOV camera.

This is the *camera*-timescale runner for the lowest-latency (camera / near-real-time)
footprint-matching product. It runs continuously from mission start.

The operational input is the L1B Daily camera product (``CAM``). Unlike the radiometer-timescale
runners, this one does not pass L1B columns through: it segments each camera image's pixel grid into
radiometer-sized pseudo-footprints with
:func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`, and those
pseudo-footprints become the product's records. The runner config sets
``l1b_input_product_id=DataProductIdentifier.l1b_cam`` so
:func:`libera_utils.footprint_matching._runner.select_manifest_files_by_product_id` keeps the CAM files from the
input manifest.

Note that ``CAMERA_TIME`` values repeat within the product: every pseudo-footprint segmented from one
image shares that image's time, because the time identifies the source image rather than a unique
footprint.

FMATCH-CAM-CAMTIME additionally declares ``cloud_fraction_camera``, supplied by the Libera WFOV
Camera Cloud Fraction algorithm on the camera timescale (``CF-CAM-CAMTIME``). That file is selected
from the manifest when present and recorded as provenance; ingesting its values is future work
(``TODO[LIBSDC-785]``).
"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.types import OperationalMode

DESCRIPTION = "Run the Libera FMATCH-CAM-CAMTIME algorithm from an input manifest."

# All the parameters that make this the camera-timescale CAM runner (see FmatchRunnerConfig). The
# input is the L1B CAM product; the optional cloud fraction comes from CF-CAM-CAMTIME (also on the
# camera timescale). CAM-CAMTIME is variant-insensitive: no RBSP-sourced readers are active at its
# latency rank, so there is no --post-year-one option.
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.CAM_CAMTIME,
    output_product_id=DataProductIdentifier.aux_fmatch_cam_camtime,
    l1b_input_product_id=DataProductIdentifier.l1b_cam,
    cloud_fraction_product_id=DataProductIdentifier.l2_cf_cam_time,
    supports_variant_override=False,
    log_prefix="fmatch_cam_camtime",
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
