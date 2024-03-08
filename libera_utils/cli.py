"""Module for the Libera SDC utilities CLI

libera-utils
    make-kernel
        jpss-spk
        jpss-ck
        azel-ck
"""
# Standard
import argparse
# Local
from libera_utils import kernel_maker
from libera_utils.io import pds_ingest
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

    # packet-ingest
    pds_ingest_parser = subparsers.add_parser('pds-ingest',
                                              help='write construction record data to database')
    pds_ingest_parser.set_defaults(func=pds_ingest.ingest)
    pds_ingest_parser.add_argument('manifest_filepath', type=str,
                                   help="path to L0 manifest file")
    pds_ingest_parser.add_argument('-d', '--delete', action='store_true',
                                   help='delete data files from s3 bucket once they are moved')
    pds_ingest_parser.add_argument('-v', '--verbose', action='store_true',
                                   help="set DEBUG level logging output (otherwise set by LIBERA_CONSOLE_LOG_LEVEL)")

    # make-kernel
    make_kernel_parser = subparsers.add_parser('make-kernel',
                                               help='generate SPICE kernel from telemetry data')

    make_kernel_subparsers = make_kernel_parser.add_subparsers(description="sub-commands for make-kernel sub-command")

    # TODO: the interfaces to these spice kernel makers need to be changed to accept a manifest file path, which
    #   points to the PDS files from which to generate the kernels.
    # make-kernel jpss-spk
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
                                 help="set DEBUG level logging output (otherwise set by LIBERA_CONSOLE_LOG_LEVEL)")

    # make-kernel jpss-ck
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
                                help="set DEBUG level logging output (otherwise set by LIBSDP_STREAM_LOG_LEVEL)")

    # make-kernel azel-ck
    azel_ck_parser = make_kernel_subparsers.add_parser('azel-ck',
                                                       help="generate Libera Az-El CK kernel from telemetry")
    azel_ck_parser.set_defaults(func=kernel_maker.make_azel_ck)
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

    parsed_args = parser.parse_args(cli_args)
    return parsed_args
