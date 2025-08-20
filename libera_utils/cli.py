"""Module for the Libera SDC utilities CLI"""

import argparse

from libera_utils import kernel_maker
from libera_utils.aws import constants, ecr_upload, s3_utilities
from libera_utils.aws import processing_step_function_trigger as psfn
from libera_utils.constants import ProcessingStepIdentifier
from libera_utils.version import version as libera_utils_version


def main(cli_args: list = None):
    """Main CLI entrypoint that runs the function inferred from the specified subcommand"""
    args = parse_cli_args(cli_args)
    args.func(args)


def print_version_info(*args):
    """Print CLI version information"""
    print(
        f"Libera SDC utilities CLI\n\tVersion {libera_utils_version()}"
        f"\n\tCopyright 2025 University of Colorado\n\tReleased under BSD3 license"  # TODO[LIBSDC-607]: Automate date?
    )


# pylint: disable=too-many-statements
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
    parser.add_argument(
        "--version",
        action="store_const",
        dest="func",
        const=print_version_info,
        help="print current version of the CLI",
    )

    subparsers = parser.add_subparsers(description="sub-commands for libera-utils CLI")

    # ============
    # Kernel Maker
    # ============
    kernel_parser = subparsers.add_parser("make-kernel", help="generate SPICE kernels from a manifest file")
    kernel_subparsers = kernel_parser.add_subparsers(description="sub-commands for make-kernel sub-command")

    jpss_kernel_parser = kernel_subparsers.add_parser(
        "jpss", help="generate SPICE JPSS SPK/CK kernels from a manifest file"
    )
    jpss_kernel_parser.set_defaults(func=kernel_maker.jpss_kernel_cli_handler)
    jpss_kernel_parser.add_argument("input_manifest", type=str, help="path to input manifest file")
    jpss_kernel_parser.add_argument("-v", "--verbose", action="store_true", help="set DEBUG level logging output")

    azel_kernel_parser = kernel_subparsers.add_parser(
        "azel", help="generate SPICE Az/El CK kernels from a manifest file"
    )
    azel_kernel_parser.set_defaults(func=kernel_maker.azel_kernel_cli_handler)
    azel_kernel_parser.add_argument("input_manifest", type=str, help="path to input manifest file")
    azel_kernel_parser.add_argument("-v", "--verbose", action="store_true", help="set DEBUG level logging output")

    # ==============
    # AWS CLI TOOLS
    # ==============
    steps_with_ecrs = [f"{name}" for name in ProcessingStepIdentifier if name.ecr_name]
    processing_steps = [f"{name}" for name in ProcessingStepIdentifier]
    account_suffixes = [f"{name}" for name in constants.LiberaAccountSuffix]

    # ==========
    # ECR UPLOAD
    # ==========
    ecr_upload_parser = subparsers.add_parser(
        "ecr-upload", help="Upload docker image to ECR repository for a specific algorithm"
    )
    ecr_upload_parser.set_defaults(func=ecr_upload.ecr_upload_cli_handler)
    ecr_upload_parser.add_argument(
        "algorithm_name",
        type=str,
        choices=processing_steps,
        help=f"Algorithm name used to determine the ECR repo name, Options are:\n {steps_with_ecrs}",
    )
    ecr_upload_parser.add_argument("image_name", type=str, help="Image name of image to upload (image-name:image-tag)")
    ecr_upload_parser.add_argument(
        "--image-tag",
        type=str,
        default="latest",
        help="The current image tag of the local image that will be uploaded "
        "(image-name:image-tag). Default value is 'latest'",
    )
    ecr_upload_parser.add_argument(
        "--ecr-tags",
        type=str,
        nargs="+",
        help="List of tags to be used in the ECR for the uploaded image in the ECR "
        "(e.g. `--ecr_tags latest 1.3.4`) "
        "Note, latest is applied if this option is not set. If it is set, you must "
        "specify latest if you want it tagged as such in the ECR.",
    )
    ecr_upload_parser.add_argument(
        "--ignore-docker-config",
        action="store_true",
        help="Ignore the standard docker config.json to bypass the credential store",
    )

    # ============================
    # STEP FUNCTION MANUAL TRIGGER
    # ============================
    sfn_trigger_parser = subparsers.add_parser(
        "step-function-trigger", help="Manually trigger a specific step function"
    )
    sfn_trigger_parser.set_defaults(func=psfn.step_function_trigger_cli_handler)
    sfn_trigger_parser.add_argument(
        "algorithm_name",
        type=str,
        choices=processing_steps,
        help=f"Algorithm name you want to run. Options are: {processing_steps}",
    )
    sfn_trigger_parser.add_argument(
        "applicable_day", type=str, help="Day of data you want to rerun. Format of date: YYYY-MM-DD"
    )
    sfn_trigger_parser.add_argument(
        "--wait-time", type=float, default=5, help="Time in seconds to wait for step function completes "
    )

    # ============================
    # S3 UTILITIES
    # ============================
    s3_utilities_parser = subparsers.add_parser(
        "s3-utils", help="Utilities for working with S3 archives for processing steps"
    )
    s3_utilities_subparser = s3_utilities_parser.add_subparsers(description="sub-commands for s3-utils sub-command")

    # ============================
    # S3 PUT
    # ============================
    s3_put_parser = s3_utilities_subparser.add_parser(
        "put", help="Upload tool for putting files into S3 archives for designated processing steps"
    )
    s3_put_parser.set_defaults(func=s3_utilities.s3_put_cli_handler)
    s3_put_parser.add_argument(
        "algorithm_name",
        type=str,
        choices=processing_steps,
        help=f"Algorithm name string. Used to determine the S3 archive bucket name. Options are: {processing_steps}",
    )
    s3_put_parser.add_argument("file_path", type=str, help="Path to the file to upload")
    s3_put_parser.add_argument(
        "--account-suffix",
        type=str,
        default="-stage",
        help=f"Account suffix for the bucket name. Default is -stage. Common options are: {account_suffixes}",
    )

    # ============================
    # S3 LIST
    # ============================
    s3_list_parser = s3_utilities_subparser.add_parser(
        "ls", help="List files in an S3 archive for a designated processing step"
    )
    s3_list_parser.set_defaults(func=s3_utilities.s3_list_cli_handler)
    s3_list_parser.add_argument(
        "algorithm_name",
        type=str,
        choices=processing_steps,
        help=f"Algorithm name string. Used to determine the S3 archive bucket name.Options are: {processing_steps}",
    )
    s3_list_parser.add_argument(
        "--account-suffix",
        type=str,
        default="-stage",
        help=f"Account suffix for the bucket name. Default is -stage. Common options are: {account_suffixes}",
    )

    # ============================
    # S3 COPY
    # ============================
    s3_get_object_parser = s3_utilities_subparser.add_parser("cp", help="Copy an object from one location to another.")
    s3_get_object_parser.set_defaults(func=s3_utilities.s3_copy_cli_handler)
    s3_get_object_parser.add_argument("source_path", type=str, help="The current path path to the object to retrieve")
    s3_get_object_parser.add_argument("dest_path", type=str, help="Destination path to save the object to.")
    s3_get_object_parser.add_argument("--delete", action="store_true", help="If set, deletes files copied from source")

    parsed_args = parser.parse_args(cli_args)
    return parsed_args


if __name__ == "__main__":
    main()
