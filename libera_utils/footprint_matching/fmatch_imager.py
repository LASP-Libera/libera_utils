"""FMATCH-IMAGER processing code for the Libera radiometer.

This is the *radiometer*-timescale runner at RBSP Climate Quality latency - the highest latency rank
of the radiometer-timed modes, and therefore the one with the largest active reader set.

The operational input is the L1B Daily radiometer product (``RAD-4CH``): one footprint per
``RADIOMETER_TIME``, whose geolocation and Sun-surface-sensor viewing angles are carried through to
the FMATCH product verbatim (see
:func:`libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs`).

The product (``fmatch_imager.yml``) carries the RBSP CLDPIX/SSF cloud fields alongside the ERA5
single-level and pressure-level reanalysis fields and the VIIRS imager fields.

Unlike the CAM products, FMATCH-IMAGER does not declare ``cloud_fraction_camera`` - its cloud
information comes from the imager readers rather than the Libera WFOV camera - so this runner takes
no cloud-fraction input.
"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.types import OperationalMode

DESCRIPTION = "Run the Libera FMATCH-IMAGER algorithm from an input manifest."

# All the parameters that make this the climate-quality radiometer runner (see FmatchRunnerConfig).
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER,
    output_product_id=DataProductIdentifier.aux_fmatch_imager,
    l1b_input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=None,
    log_prefix="fmatch_imager",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the FMATCH-IMAGER processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.footprint_matching._runner.run_algorithm` with the IMAGER
    config.

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
    """CLI entrypoint for the FMATCH-IMAGER runner.

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
