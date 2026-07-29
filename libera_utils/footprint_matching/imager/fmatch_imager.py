"""FMATCH-IMAGER processing code for the Libera radiometer.

This is the *radiometer*-timescale runner at RBSP Climate Quality latency - the highest latency rank
of the radiometer-timed modes, and therefore the one with the largest active reader set.

The operational input is the L1B Daily radiometer product (``RAD-4CH``): one footprint per
``RADIOMETER_TIME``, whose geolocation and Sun-surface-sensor viewing angles are carried through to
the FMATCH product verbatim (see
:func:`libera_utils.footprint_matching.l1b_inputs.load_l1b_radiometer_inputs`).

Year-one vs post-year-one
-------------------------
FMATCH-IMAGER is the only mode with two product definitions, because the RBSP CLDPIX/SSF products it
would otherwise consume are unavailable during the first year of operation:

* **Default (year one)** - ``fmatch_imager.yml``, in which ERA5 single-level and pressure-level
  reanalysis fields substitute for the missing RBSP cloud inputs.
* **``--post-year-one``** - ``fmatch_imager_post_year_one.yml``, the RBSP-based definition, selected
  manually once RBSP data flows.

Both encode the same ``FMATCH-IMAGER`` ProductID; the variant changes the active reader set and the
declared variables, not the product identity. See
:class:`libera_utils.footprint_matching.types.FmatchVariant`.

Unlike the CAM products, FMATCH-IMAGER does not declare ``cloud_fraction_camera`` - its cloud
information comes from the imager readers rather than the Libera WFOV camera - so this runner takes
no cloud-fraction input.
"""

from pathlib import Path

from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import FmatchRunnerConfig, run_algorithm
from libera_utils.footprint_matching._runner import main as _main
from libera_utils.footprint_matching.types import FmatchVariant, OperationalMode

DESCRIPTION = "Run the Libera FMATCH-IMAGER algorithm from an input manifest."

# All the parameters that make this the climate-quality radiometer runner (see FmatchRunnerConfig).
# This is the only runner with supports_variant_override=True, which is what adds --post-year-one to
# its CLI.
RUNNER_CONFIG = FmatchRunnerConfig(
    mode=OperationalMode.IMAGER,
    output_product_id=DataProductIdentifier.aux_fmatch_imager,
    l1b_input_product_id=DataProductIdentifier.l1b_rad,
    cloud_fraction_product_id=None,
    supports_variant_override=True,
    log_prefix="fmatch_imager",
)


def algorithm(manifest_path: Path | S3Path, variant: FmatchVariant = FmatchVariant.YEAR_ONE) -> Path | S3Path:
    """Run the FMATCH-IMAGER processing workflow from an input manifest.

    Thin wrapper over :func:`libera_utils.footprint_matching._runner.run_algorithm` with the IMAGER
    config.

    Parameters
    ----------
    manifest_path : Path | S3Path
        Path to the input manifest file listing the L1B RAD-4CH input file(s). An
        ``argparse.Namespace`` (as produced by :func:`main`) is also accepted for convenience when
        invoked as a CLI; a Namespace carrying ``post_year_one=True`` overrides ``variant``.
    variant : FmatchVariant, optional
        Input-availability variant. Defaults to ``YEAR_ONE``, the production default for the first
        year of operation.

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.
    """
    return run_algorithm(manifest_path, RUNNER_CONFIG, variant)


def main(cli_args: list | None = None) -> Path | S3Path:
    """CLI entrypoint for the FMATCH-IMAGER runner.

    Accepts ``--post-year-one`` in addition to the input manifest; see the module docstring.

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
