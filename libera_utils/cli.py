"""Module for the Libera SDC utilities CLI
"""
# Standard
import argparse
# Local
from libera_utils import kernel_maker
from libera_utils.aws import ecr_upload, constants
from libera_utils.aws import processing_step_function_trigger as psfn
from libera_utils.version import version as libera_utils_version


def main(cli_args: list = None):
    """Main CLI entrypoint that runs the function inferred from the specified subcommand"""
    args = parse_cli_args(cli_args)
    args.func(args)


def print_version_info(*args):
    """Print CLI version information"""
    print(f"Libera SDC utilities CLI\n\tVersion {libera_utils_version()}"
          f"\n\tCopyright 2023 University of Colorado\n\tReleased under BSD3 license")


def parse_cli_args(cli_args: list):
    """Parse CLI arguments

    Parameters
    ----------
    cli_args : list
        List of CLI arguments to parse

    Returns
    -------
    : argparse.Namespace
        Parsed arguments in a Namespace object
    """
    parser = argparse.ArgumentParser(prog="libera-utils", description="Libera SDC utilities CLI")
    parser.add_argument("--version",
                        action='store_const', dest='func', const=print_version_info,
                        help="print current version of the CLI")

    subparsers = parser.add_subparsers(description="sub-commands for libera-utils CLI")

    # make-kernel
    make_kernel_parser = subparsers.add_parser('make-kernel',
                                               help='generate SPICE kernel from telemetry data')

    make_kernel_subparsers = make_kernel_parser.add_subparsers(description="sub-commands for make-kernel sub-command")

    # TODO: the interfaces to these spice kernel makers need to be changed to accept a manifest file path, which
    #   points to the PDS files from which to generate the kernels.
    # TODO: The JPSS CK and JPSS SPK makers should be combined to a single entrypoint that produces both an SPK and CK
    # ==============
    # JPSS SPK MAKER
    # ==============
    jpss_spk_parser = make_kernel_subparsers.add_parser('jpss-spk', help="generate JPSS SPK kernel from telemetry")
    jpss_spk_parser.set_defaults(func=kernel_maker.make_jpss_spk)
    jpss_spk_parser.add_argument('packet_data_filepaths', nargs='+', type=str,
                                 help="paths to L0 packet files")
    jpss_spk_parser.add_argument('--outdir', '-o', type=str,
                                 required=True,
                                 help="output directory for generated SPK")
    jpss_spk_parser.add_argument('--overwrite', action='store_true',
                                 help="force overwriting an existing kernel if it exists")
    jpss_spk_parser.add_argument('-v', '--verbose', action='store_true',
                                 help="set DEBUG level logging output")

    # =============
    # JPSS CK MAKER
    # =============
    jpss_ck_parser = make_kernel_subparsers.add_parser('jpss-ck', help="generate JPSS CK kernel from telemetry")
    jpss_ck_parser.set_defaults(func=kernel_maker.make_jpss_ck)
    jpss_ck_parser.add_argument('packet_data_filepaths', nargs='+', type=str,
                                help="paths to L0 packet files")
    jpss_ck_parser.add_argument('--outdir', '-o', type=str,
                                required=True,
                                help="output directory for generated CK")
    jpss_ck_parser.add_argument('--overwrite', action='store_true',
                                help="force overwriting an existing kernel if it exists")
    jpss_ck_parser.add_argument('-v', '--verbose', action='store_true',
                                help="set DEBUG level logging output")

    # ==============
    # AZ EL CK MAKER
    # ==============
    azel_ck_parser = make_kernel_subparsers.add_parser('azel-ck',
                                                       help="generate Libera Az-El CK kernels from telemetry")
    azel_ck_parser.set_defaults(func=kernel_maker.make_azel_ck)
    # TODO: Modify this CLI to take a manifest file as input rather than direct input of filepaths
    azel_ck_parser.add_argument('packet_data_filepaths', nargs='+', type=str,
                                help="paths to L0 packet files")
    azel_ck_parser.add_argument('--azimuth', action='store_true',
                                help="generate ck for Azimuth")
    azel_ck_parser.add_argument('--elevation', action='store_true',
                                help="generate ck for Elevation")
    azel_ck_parser.add_argument('--outdir', '-o', type=str, required=True,
                                help="output directory for generated CK")
    azel_ck_parser.add_argument('--overwrite', action='store_true',
                                help="force overwriting an existing kernel if it exists")
    azel_ck_parser.add_argument('--csv', action='store_true',
                                help="the provided Az and El packet_data_filepaths are ASCII csv files instead of "
                                     "binary CCSDS")
    azel_ck_parser.add_argument('-v', '--verbose', action='store_true',
                                help="set DEBUG level logging output (otherwise set by LIBSDP_STREAM_LOG_LEVEL)")

    # ==========
    # ECR UPLOAD
    # ==========
    algorithm_names = [name.value for name in constants.ProcessingStepIdentifier if name.ecr_name]
    ecr_upload_parser = subparsers.add_parser('ecr-upload',
                                              help="Upload docker image to ECR repository for a specific algorithm")
    ecr_upload_parser.set_defaults(func=ecr_upload.ecr_upload_cli_func)
    ecr_upload_parser.add_argument('image_name', type=str,
                                   help="Image name of image to upload (image-name:image-tag)")
    ecr_upload_parser.add_argument('image_tag', type=str, default="latest",
                                   help="Image tag of image to upload (image-name:image-tag)")
    ecr_upload_parser.add_argument('algorithm_name', type=str,
                                   help=f"Algorithm name that matches an ECR repo name, "
                                        f"inputs to names:\n {algorithm_names}")
    ecr_upload_parser.add_argument('--ecr-image-tags', type=str, nargs="+",
                                   help="List of tags to apply to the uploaded image in the ECR "
                                        "(e.g. `--ecr-image-tags latest 1.3.4`) "
                                        "Note, latest is applied if this option is not set. If it is set, you must "
                                        "specify latest if you want it tagged as such in the ECR.")
    ecr_upload_parser.add_argument('--ignore-docker-config', action='store_true',
                                   help="Ignore the standard docker config.json to bypass the credential store")

    # ============================
    # STEP FUNCTION MANUAL TRIGGER
    # ============================
    sfn_trigger_parser = subparsers.add_parser('step-function-trigger',
                                               help="Manually trigger a specific step function")
    sfn_trigger_parser.set_defaults(func=psfn.step_function_trigger)
    sfn_trigger_parser.add_argument('algorithm_name', type=str,
                                    help="Algorithm name you want to run")
    sfn_trigger_parser.add_argument('applicable_day', type=str,
                                    help="Day of data you want to rerun. Format of date: YYYY-MM-DD")
    sfn_trigger_parser.add_argument('-w', '--wait_for_finish', action='store_true',
                                    help="Block command line until step function completes (may be a long time)")
    sfn_trigger_parser.add_argument('-v', '--verbose', action='store_true',
                                    help="Prints out the result of the step_function_trigger run")

    parsed_args = parser.parse_args(cli_args)
    return parsed_args
