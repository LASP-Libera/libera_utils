"""Module containing CLI tool for creating SPICE kernels from packets"""

import argparse
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as nprf
import pandas as pd
from cloudpathlib import AnyPath
from curryer import kernels, meta, spicetime
from space_packet_parser import parser, xtcedef

from libera_utils import packets as libera_packets
from libera_utils import time
from libera_utils.aws.constants import DataProductIdentifier
from libera_utils.config import config
from libera_utils.io import filenaming
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_copy_file, smart_open
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


def make_jpss_kernels_from_manifest(manifest_file_path: str or AnyPath, output_directory: str or AnyPath):
    """Alpha function triggering kernel generation from manifest file.

    If the manifest configuration field contains "start_time" and "end_time"
    fields then this function will select only packet data that falls in that
    range. If these are not given, then all packet data will be used.

    Parameters
    ----------
    manifest_file_path : str or cloudpathlib.anypath.AnyPath
        Path to the manifest file that includes end_time and start_time
        in the configuration section
    output_directory :  str or cloudpathlib.anypath.AnyPath
        Path to save the completed kernels
    Returns
    -------
    output_directory : str or cloudpathlib.anypath.AnyPath
        Path to the directory containing the completed kernels
    """
    # TODO: Consider cases to return/error if the entire range is not covered

    m = Manifest.from_file(manifest_file_path)
    m.validate_checksums()
    files_in_range = []

    if "start_time" not in m.configuration:
        # No time range information is provided. Process all files in the manifest
        for file_entry in m.files:
            files_in_range.append(str(file_entry.filename))
    else:
        # Load desired time range from the manifest configuration
        start_time_text = m.configuration["start_time"]
        desired_start_time = datetime.strptime(start_time_text, "%Y-%m-%d:%H:%M:%S")
        end_time_text = m.configuration["end_time"]
        desired_end_time = datetime.strptime(end_time_text, "%Y-%m-%d:%H:%M:%S")

        # Load the packet files and check the time ranges against the manifest configuration
        # TODO update this if possible to use the metadata files when those are more defined

        for file_entry in m.files:
            file_path_from_list = file_entry.filename
            packet_data = get_spice_packet_data_from_filepaths([file_path_from_list])
            packet_dt64 = time.multipart_to_dt64(packet_data, "ADAET1DAY", "ADAET1MS", "ADAET1US")

            # Check if any of the packet data are in increasing order by comparing an array element to
            # its right neighbor and ensuring that is always greater or equal. If this is not true
            # throw an error
            if not np.all(packet_dt64[:-1] <= packet_dt64[1:]):
                raise ValueError(f"The data in {file_path_from_list} are not monotonic in time")
            packet_start_time = packet_dt64[0].to_pydatetime()
            packet_end_time = packet_dt64[-1].to_pydatetime()

            # Packet range starts before desired range - first packet or full data
            if packet_start_time < desired_start_time < packet_end_time:
                files_in_range.append(str(file_path_from_list))
            # Desired range starts before packet range - middle or end packet
            if desired_start_time < packet_start_time < desired_end_time:
                files_in_range.append(str(file_path_from_list))

        if not files_in_range:
            raise ValueError(f"No files contained packets in timerange ({desired_start_time}, {desired_end_time})")

    # Create the arguments to pass to the kernel generation
    parsed_args = argparse.Namespace(
        packet_data_filepaths=files_in_range, outdir=str(output_directory), overwrite=False, verbose=False
    )
    make_jpss_spk(parsed_args)
    make_jpss_ck(parsed_args)

    return output_directory


def get_spice_packet_data_from_filepaths(packet_data_filepaths: list[str or AnyPath]):
    """Utility function to return an array of packet data from a list of file paths of raw JPSS APID 11
    geolocation packet data files.

     Parameters
    ----------
    packet_data_filepaths : list of str or cloudpathlib.anypath.AnyPath
        The list of file paths to the raw packet data

    Returns
    -------
    packet_data : numpy.ndarray
        The configured packet data. See packets.py for more details on structure
    """
    packet_definition_uri = AnyPath(config.get("JPSS_GEOLOCATION_PACKET_DEFINITION"))
    logger.info("Using packet definition %s", packet_definition_uri)

    with smart_open(packet_definition_uri) as packet_definition_filepath:
        packet_definition = xtcedef.XtcePacketDefinition(packet_definition_filepath)

    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    packet_data = libera_packets.parse_packets(packet_parser, packet_data_filepaths)

    return packet_data


def make_kernel(
    config_file: str or Path,
    output_kernel: str or AnyPath,
    input_data: str or Path = None,
    overwrite: bool = False,
    append: bool = False,
):
    """Create a SPICE kernel from a configuration file and input data.

    Parameters
    ----------
    config_file : str or pathlib.Path
        JSON configuration file defining how to create the kernel.
    output_kernel : str or cloudpathlib.anypath.AnyPath
        Output directory or file to create the kernel. If a directory, the
        file name will be based on the config_file, but with the SPICE file
        extension.
    input_data : str or pathlib.Path or pd.DataFrame, optional
        Input data file or object. Not required if defined within the config.
    overwrite : bool, optional
        Option to overwrite an existing file.
    append : bool, optional
        Option to append to an existing file.

    Returns
    -------
    str or cloudpathlib.anypath.AnyPath
        Output kernel file path

    """
    output_kernel = AnyPath(output_kernel)
    config_file = Path(config_file)

    # Load meta kernel details. Required to auto-map frame IDs.
    meta_kernel_file = Path(config.get("LIBERA_KERNEL_META"))
    _ = meta.MetaKernel.from_json(meta_kernel_file, relative=True)

    # Create the kernels from the JSONs definitions.
    creator = kernels.create.KernelCreator(overwrite=overwrite, append=append)

    with tempfile.TemporaryDirectory(prefix="/tmp/") as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        if output_kernel.is_file():
            tmp_path = tmp_dir / output_kernel.name

        out_fn = creator.write_from_json(config_file, output_kernel=tmp_path, input_data=input_data)

        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        if output_kernel.is_dir():
            output_kernel = output_kernel / out_fn.name
        smart_copy_file(out_fn, output_kernel)
        logger.info("Kernel copied to %s", output_kernel)
    return output_kernel


def make_jpss_spk(parsed_args: argparse.Namespace):
    # TODO Make low level functions that are more python usable
    # TODO: If we're going to keep using this same structure moving forward, we should consider refactoring this into
    # TODO: two separate functions. One is a cli_handler that is called when the cli tool is used to make a
    # TODO: kernel and has only the argparse.Namespace input parameter. This method should explicitly pull out the
    # TODO: the arguments from the Namespace and call the second function which has the explicit arguments and does the
    # TODO: work. This will allow for easier unit testing of the core functionality vs the cli interface.
    #
    """Create a JPSS SPK from APID 11 CCSDS packets.
    The SPK system is the component of SPICE concerned with ephemeris data (position/velocity).

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """

    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    configure_task_logging(
        f"spk_generator_{now}",
        limit_debug_loggers="libera_utils",
        console_log_level=logging.DEBUG if parsed_args.verbose else None,
    )

    logger.info("Starting SPK maker. This CLI tool creates an SPK from a list of geolocation packet files.")

    output_dir = AnyPath(parsed_args.outdir)
    logger.info("Writing resulting SPK to %s", output_dir)

    logger.info("Parsing packets...")
    packet_data = get_spice_packet_data_from_filepaths(parsed_args.packet_data_filepaths)
    logger.info("Done.")

    # Compute the ephemeris time from the multipart ephemeris time.
    packet_dt64 = time.multipart_to_dt64(packet_data, "ADAET1DAY", "ADAET1MS", "ADAET1US")
    packet_data = pd.DataFrame(packet_data)
    packet_data["SPK_ET"] = spicetime.adapt(packet_dt64.values, "dt64", "et")

    spk_filename = filenaming.EphemerisKernelFilename.from_filename_parts(
        spk_object=DataProductIdentifier.spice_jpss_spk,
        utc_start=packet_dt64[0].to_pydatetime(),
        utc_end=packet_dt64[-1].to_pydatetime(),
        version=filenaming.get_current_version_str("libera_utils"),
        revision=datetime.now(UTC),
    )
    output_full_path = output_dir / spk_filename.path.name  # pylint: disable=no-member

    config_file = config.get("LIBERA_KERNEL_SC_SPK_CONFIG")
    make_kernel(
        config_file=config_file, output_kernel=output_full_path, input_data=packet_data, overwrite=parsed_args.overwrite
    )


def make_jpss_ck(parsed_args: argparse.Namespace):
    # TODO: If we're going to keep using this same structure moving forward, we should consider refactoring this into
    # TODO: two separate functions. One is a cli_handler that is called when the cli tool is used to make a
    # TODO: kernel and has only the argparse.Namespace input parameter. This method should explicitly pull out the
    # TODO: the arguments from the Namespace and call the second function which has the explicit arguments and does the
    # TODO: work. This will allow for easier unit testing of the core functionality vs the cli interface.
    """Create a JPSS CK from APID 11 CCSDS packets.
    The C-kernel (CK) is the component of SPICE concerned with attitude of spacecraft structures or instruments.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    now = datetime.now(UTC).strftime("%Y%m%dt%H%M%S")
    configure_task_logging(
        f"ck_generator_{now}",
        limit_debug_loggers="libera_utils",
        console_log_level=logging.DEBUG if parsed_args.verbose else None,
    )

    logger.info("Starting CK maker. This CLI tool creates a CK from a list of JPSS attitue/quaternion packet files.")

    output_dir = AnyPath(parsed_args.outdir)
    logger.info("Writing resulting CK to %s", output_dir)
    packet_data = get_spice_packet_data_from_filepaths(parsed_args.packet_data_filepaths)
    logger.info("Done.")

    # Compute the ephemeris time from the multipart attitude time.
    packet_dt64 = time.multipart_to_dt64(packet_data, "ADAET2DAY", "ADAET2MS", "ADAET2US")
    packet_data = pd.DataFrame(packet_data)
    packet_data["CK_ET"] = spicetime.adapt(packet_dt64.values, "dt64", "et")

    ck_filename = filenaming.AttitudeKernelFilename.from_filename_parts(
        ck_object=DataProductIdentifier.spice_jpss_ck,
        utc_start=packet_dt64[0].to_pydatetime(),
        utc_end=packet_dt64[-1].to_pydatetime(),
        version=filenaming.get_current_version_str("libera_utils"),
        revision=datetime.now(UTC),
    )
    output_full_path = output_dir / ck_filename.path.name  # pylint: disable=no-member

    config_file = config.get("LIBERA_KERNEL_SC_CK_CONFIG")
    make_kernel(
        config_file=config_file, output_kernel=output_full_path, input_data=packet_data, overwrite=parsed_args.overwrite
    )


def make_azel_ck(parsed_args: argparse.Namespace):  # pylint: disable=too-many-statements
    # TODO: If we're going to keep using this same structure moving forward, we should consider refactoring this into
    # TODO: two separate functions. One is a cli_handler that is called when the cli tool is used to make a
    # TODO: kernel and has only the argparse.Namespace input parameter. This method should explicitly pull out the
    # TODO: the arguments from the Namespace and call the second function which has the explicit arguments and does the
    # TODO: work. This will allow for easier unit testing of the core functionality vs the cli interface.
    """Create a Libera Az-El CK from CCSDS packets or ASCII input files
    The C-kernel (CK) is the component of SPICE concerned with attitude of spacecraft structures or instruments.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    print(parsed_args)

    now = datetime.now(UTC).strftime("%Y%m%dt%H%M%S")
    configure_task_logging(
        f"ck_generator_{now}",
        limit_debug_loggers="libera_utils",
        console_log_level=logging.DEBUG if parsed_args.verbose else None,
    )

    logger.info("Starting CK maker. This CLI tool creates a CK from a list of Azimuth or Elevation files.")

    output_dir = AnyPath(parsed_args.outdir)
    logger.info("Writing resulting CK to %s", output_dir)

    if not parsed_args.csv:
        logger.info("Parsing packets...")
        packet_data = get_spice_packet_data_from_filepaths(parsed_args.packet_data_filepaths)
        # Add a column that is the SCLK string, formatted with delimiters, to the input data recarray
        # TODO: the timing for the Az and El will most likely be labelled differently in the XTCE xml file for Libera
        # TODO: get the config depending on AZ or El
        # TODO: the MSOPCK expects ET time stamps: for packets this will need to be convert to ET

        # TODO: identify which APID we're reading AZ or EL
        # TODO: assign this_config and ck_object below based on the APID of the packet decoded

        azel_sclk_string = [f"{row['ADAET2DAY']}:{row['ADAET2MS']}:{row['ADAET2US']}" for row in packet_data]
        packet_data = nprf.append_fields(packet_data, "ATTSCLKSTR", azel_sclk_string)
        utc_start = time.et_2_datetime(time.scs2e_wrapper(azel_sclk_string[0]))
        utc_end = time.et_2_datetime(time.scs2e_wrapper(azel_sclk_string[-1]))
    else:
        logger.info("Parsing CSV file...")
        # get the data from the ASCII file
        packet_data = np.genfromtxt(parsed_args.packet_data_filepaths[0], delimiter=",", dtype="double")
        # make sure we have all 3 axis defined: X is RAM, Y is Elev when Az is at 0.0, Z is nadir
        if (parsed_args.azimuth is True) and (parsed_args.elevation is True):
            try:
                raise ValueError("Expecting only one: --azimuth or --elevation. Got both\n")
            except ValueError as error:
                logger.exception(error)

        if parsed_args.azimuth:
            packet_data = packet_data.view([("ET_TIME", "double"), ("AZIMUTH", "double")])
            packet_data = nprf.append_fields(packet_data, "ELEVATION", np.zeros(packet_data.size, dtype="double"))
        elif parsed_args.elevation:
            packet_data = packet_data.view([("ET_TIME", "double"), ("ELEVATION", "double")])
            packet_data = nprf.append_fields(packet_data, "AZIMUTH", np.zeros(packet_data.size, dtype="double"))
        else:
            try:
                raise ValueError("Expecting at least one: --azimuth or --elevation. None provided.\n")
            except ValueError as error:
                logger.exception(error)

        packet_data = nprf.append_fields(packet_data, "AZEL_Z", np.zeros(packet_data.size, dtype="double"))
        azel_sclk_string = [f"{d}" for d in packet_data["ET_TIME"]]
        packet_data = nprf.append_fields(packet_data, "AZELSCLKSTR", azel_sclk_string)
        utc_start = time.et_2_datetime(packet_data["ET_TIME"][0])
        utc_end = time.et_2_datetime(packet_data["ET_TIME"][-1])

    logger.info("Done.")
    revision = datetime.now(UTC)

    if parsed_args.azimuth:
        ck_filename = filenaming.AttitudeKernelFilename.from_filename_parts(
            ck_object=DataProductIdentifier.spice_az_ck,
            utc_start=utc_start,
            utc_end=utc_end,
            version=filenaming.get_current_version_str("libera_utils"),
            revision=revision,
        )
        output_full_path = output_dir / ck_filename.path.name  # pylint: disable=no-member

        config_file = config.get("LIBERA_KERNEL_AZ_CK_CONFIG")
        make_kernel(
            config_file=config_file,
            output_kernel=output_full_path,
            input_data=packet_data,
            overwrite=parsed_args.overwrite,
        )

    if parsed_args.elevation:
        ck_filename = filenaming.AttitudeKernelFilename.from_filename_parts(
            ck_object=DataProductIdentifier.spice_el_ck,
            utc_start=utc_start,
            utc_end=utc_end,
            version=filenaming.get_current_version_str("libera_utils"),
            revision=revision,
        )
        output_full_path = output_dir / ck_filename.path.name  # pylint: disable=no-member

        config_file = config.get("LIBERA_KERNEL_EL_CK_CONFIG")
        make_kernel(
            config_file=config_file,
            output_kernel=output_full_path,
            input_data=packet_data,
            overwrite=parsed_args.overwrite,
        )
