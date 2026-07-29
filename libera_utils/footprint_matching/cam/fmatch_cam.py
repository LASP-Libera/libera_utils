"""FMATCH-CAM processing code for the Libera radiometer.

This is the *radiometer*-timescale runner for the lowest-latency (camera / near-real-time)
footprint-matching product. It runs continuously from mission start.

The operational input is the L1B Daily radiometer product (``RAD-4CH``): one footprint per
``RADIOMETER_TIME``, whose geolocation and Sun-surface-sensor viewing angles are carried through to
the FMATCH product verbatim (see
:func:`libera_utils.footprint_matching.l1b_inputs.load_l1b_radiometer_inputs`). The runner config
sets ``l1b_input_product_id=DataProductIdentifier.l1b_rad`` so
:func:`libera_utils.footprint_matching._runner.select_manifest_files_by_product_id` keeps the RAD-4CH files from the
input manifest.

FMATCH-CAM additionally declares ``cloud_fraction_camera``, supplied by the Libera WFOV Camera Cloud
Fraction algorithm (``CF-CAM``). That file is selected from the manifest when present and recorded as
provenance; ingesting its values is future work (``TODO[LIBSDC-785]``).
"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.types import OperationalMode

DESCRIPTION = "Run the Libera FMATCH-CAM algorithm from an input manifest."

# All the parameters that make this the radiometer-timescale CAM runner (see FmatchRunnerConfig).
# The input is the L1B RAD-4CH product; the optional cloud fraction comes from CF-CAM (also on the
# radiometer timescale). CAM is variant-insensitive: no RBSP-sourced readers are active at its
# latency rank, so there is no --post-year-one option.
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.CAM,
    output_product_id=DataProductIdentifier.aux_fmatch_cam,
    l1b_input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=DataProductIdentifier.l2_cf_rad_time,
    supports_variant_override=False,
    log_prefix="fmatch_cam",
)


def algorithm(manifest_path: Path | S3Path) -> Path | S3Path:
    """Run the FMATCH-CAM processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.footprint_matching._runner.run_algorithm` with the CAM
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
    """CLI entrypoint for the FMATCH-CAM runner.

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
