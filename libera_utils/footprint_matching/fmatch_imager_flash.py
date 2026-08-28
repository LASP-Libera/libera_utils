"""FMATCH-IMAGER-FLASH processing code for the Libera radiometer.

This is the *radiometer*-timescale runner at RBSP Flash latency. It sits between the
near-real-time CAM products and the climate-quality FMATCH-IMAGER product: a higher latency rank
than CAM, so more ancillary readers are active, but it inherently requires RBSP inputs and therefore
does not run during the first year of operation.

The operational input is the L1B Daily radiometer product (``RAD-4CH``): one footprint per
``RADIOMETER_TIME``, whose geolocation and Sun-surface-sensor viewing angles are carried through to
the FMATCH product verbatim (see
:func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`).

Unlike the CAM products, FMATCH-IMAGER-FLASH does not declare ``cloud_fraction_camera`` - its cloud
information comes from the imager readers rather than the Libera WFOV camera - so this runner takes
no cloud-fraction input. The mode requires RBSP inputs (CERES SSF), which its single product
definition carries alongside the ERA5 and VIIRS fields.
"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.types import OperationalMode

DESCRIPTION = "Run the Libera FMATCH-IMAGER-FLASH algorithm from an input manifest."

# All the parameters that make this the RBSP-Flash-latency radiometer runner (see
# FmatchRunnerConfig).
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER_FLASH,
    output_product_id=DataProductIdentifier.aux_fmatch_imager_flash,
    l1b_input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=None,
    log_prefix="fmatch_imager_flash",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the FMATCH-IMAGER-FLASH processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.footprint_matching._runner.run_algorithm` with the
    IMAGER-FLASH config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the L1B RAD-4CH input file(s). An
        ``argparse.Namespace`` (as produced by :func:`main`) is also accepted for convenience when
        invoked as a CLI.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG)


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the FMATCH-IMAGER-FLASH runner.

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
