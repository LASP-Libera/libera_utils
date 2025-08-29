"""Module containing CLI tool for creating SPICE kernels from packets"""

import argparse
import logging
import os
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

KERNEL_DPI = (
    DataProductIdentifier.spice_jpss_spk,
    DataProductIdentifier.spice_jpss_ck,
    DataProductIdentifier.spice_az_ck,
    DataProductIdentifier.spice_el_ck,
)


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
            tmp_path = tmp_path / output_kernel.name

        out_fn = creator.write_from_json(config_file, output_kernel=tmp_path, input_data=input_data)

        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        if output_kernel.is_dir():
            output_kernel = output_kernel / out_fn.name
        smart_copy_file(out_fn, output_kernel)
        logger.info("Kernel copied to %s", output_kernel)
    return output_kernel


def preprocess_data(input_data_file, nominal_time_field: str = None, pkt_time_fields: list[str] = None):
    """Preprocess kernel data to perform conversions and determine time range.

    Parameters
    ----------
    input_data_file : str or pathlib.Path
        Input data file.
    nominal_time_field : str
        Name of the field to store the converted time field as.
    pkt_time_fields : list of str
        Name of the telemetry packet time fields used to convert the time.

    Returns
    -------
    pd.DataFrame
        Loaded SPICE kernel data.
    datetime.datetime, datetime.datetime
        The date time range of the data.

    """
    # Load the input data.
    input_data_file = AnyPath(input_data_file)
    if input_data_file.suffix == ".csv":
        # TODO[LIBSDC-279]: Implement or remove (xfail test case uses csv)
        input_dataset, utc_range = None, (None, None)
        raise NotImplementedError

    # Assume a binary file of raw packets.
    else:
        input_dataset = get_spice_packet_data_from_filepaths([input_data_file])

        # Compute the ephemeris time from the multipart ephemeris time.
        packet_dt64 = time.multipart_to_dt64(input_dataset, *pkt_time_fields)
        input_dataset = pd.DataFrame(input_dataset)
        input_dataset[nominal_time_field] = spicetime.adapt(packet_dt64.values, "dt64", "et")

        utc_range = (packet_dt64[0].to_pydatetime(), packet_dt64[-1].to_pydatetime())

    return input_dataset, utc_range


def from_args(
    input_data_files: list[str or AnyPath],
    kernel_identifier: str or DataProductIdentifier,
    output_dir: str or AnyPath,
    overwrite=False,
    append=False,
    verbose=False,
):
    """Create a SPICE kernel from an input file and kernel data product type.

    Parameters
    ----------
    input_data_files : list of str or pathlib.Path
        Input data files.
    kernel_identifier : str or DataProductIdentifier
        Data product identifier that is associated with a kernel.
    output_dir : str or AnyPath
        Output location for the SPICE kernels and output manifest.
    overwrite : bool, optional
        Option to overwrite any existing similar-named SPICE kernels.
    append : bool, optional
        Option to append to any existing similar-named SPICE kernels.
    verbose : bool, optional
        Option to log with extra verbosity.

    Returns
    -------
    str or AnyPath
        Output kernel file path.

    """
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    configure_task_logging(
        f"kernel_generator_{now}",
        limit_debug_loggers=["libera_utils", "curryer"],
        console_log_level=logging.DEBUG if verbose else None,
    )

    # Validate and parse the input arguments.
    output_dir = AnyPath(output_dir)

    kernel_identifier = DataProductIdentifier(kernel_identifier)
    if kernel_identifier not in KERNEL_DPI:
        raise ValueError(
            f"The `kernel_identifier` [{kernel_identifier}] is not a Data Product Identifier associated"
            f" with a SPICE kernel, expected one of: [{KERNEL_DPI}]"
        )

    if kernel_identifier == DataProductIdentifier.spice_jpss_spk:
        config_file = config.get("LIBERA_KERNEL_SC_SPK_CONFIG")
        pkt_time_fields = ("ADAET1DAY", "ADAET1MS", "ADAET1US")
        nominal_time_field = "SPK_ET"

    elif kernel_identifier == DataProductIdentifier.spice_jpss_ck:
        config_file = config.get("LIBERA_KERNEL_SC_CK_CONFIG")
        pkt_time_fields = ("ADAET2DAY", "ADAET2MS", "ADAET2US")
        nominal_time_field = "CK_ET"

    elif kernel_identifier == DataProductIdentifier.spice_az_ck:
        config_file = config.get("LIBERA_KERNEL_AZ_CK_CONFIG")
        # TODO[LIBSDC-279]: Check if fields are correct for Az/El science packets!
        pkt_time_fields = ("ADAET2DAY", "ADAET2MS", "ADAET2US")
        nominal_time_field = "CK_ET"

    elif kernel_identifier == DataProductIdentifier.spice_el_ck:
        config_file = config.get("LIBERA_KERNEL_EL_CK_CONFIG")
        pkt_time_fields = ("ADAET2DAY", "ADAET2MS", "ADAET2US")
        nominal_time_field = "CK_ET"

    else:
        raise ValueError(kernel_identifier)

    # Prepare the input data and determine the min/max time span.
    input_datasets = []
    input_time_range = None
    for file_name in input_data_files:
        in_dataset, in_range = preprocess_data(
            file_name, nominal_time_field=nominal_time_field, pkt_time_fields=pkt_time_fields
        )
        input_datasets.append(in_dataset)
        if input_time_range is None:
            input_time_range = in_range
        else:
            input_time_range = (min(input_time_range[0], in_range[0]), max(input_time_range[1], in_range[1]))

    # Generate the output file name.
    fn_kwargs = dict(
        utc_start=input_time_range[0],
        utc_end=input_time_range[1],
        version=filenaming.get_current_version_str("libera_utils"),
        revision=datetime.now(UTC),
    )
    if kernel_identifier.value.endswith("SPK"):
        krn_filename = filenaming.EphemerisKernelFilename.from_filename_parts(spk_object=kernel_identifier, **fn_kwargs)
    elif kernel_identifier.value.endswith("CK"):
        krn_filename = filenaming.AttitudeKernelFilename.from_filename_parts(ck_object=kernel_identifier, **fn_kwargs)
    else:
        raise ValueError(f"Incorrectly named SPICE kernel Data Product Identifier: {kernel_identifier}")

    output_full_path = output_dir / krn_filename.path.name

    # Create the kernel(s).
    output_kernel = None
    for ith, an_input_dataset in enumerate(input_datasets):
        output_kernel = make_kernel(
            config_file=config_file,
            output_kernel=output_full_path,
            input_data=an_input_dataset,
            overwrite=overwrite,
            append=append or ith,
        )
    return output_kernel


def from_manifest(
    input_manifest: str or AnyPath,
    data_product_identifiers: list[str],
    output_dir: str or AnyPath,
    overwrite=False,
    append=False,
    verbose=False,
):
    """Generate SPICE kernels from a manifest file.

    Parameters
    ----------
    input_manifest : str or AnyPath
        Input manifest file containing one or more input data files.
    data_product_identifiers : list[str]
        One or more SPICE kernel data product identifiers.
    output_dir : str or AnyPath
        Output location for the SPICE kernels and output manifest.
    overwrite : bool, optional
        Option to overwrite any existing similar-named SPICE kernels.
    append : bool, optional
        Option to append to any existing similar-named SPICE kernels.
    verbose : bool, optional
        Option to log with extra verbosity.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    # Process input manifest
    mani = Manifest.from_file(input_manifest)
    mani.validate_checksums()

    input_data_files = mani.files
    if isinstance(data_product_identifiers, str):
        data_product_identifiers = [data_product_identifiers]

    # Perform processing.
    input_file_names = [file_entry.filename for file_entry in input_data_files]
    outputs = []
    for kernel_identifier in data_product_identifiers:
        try:
            outputs.append(
                from_args(
                    input_data_files=input_file_names,
                    kernel_identifier=kernel_identifier,
                    output_dir=output_dir,
                    overwrite=overwrite,
                    append=append,
                    verbose=verbose,
                )
            )
        except Exception as _:
            # Dev note: At a future time, additional information might need to
            # be captured to indicate a "partial" failure through the output
            # manifest file.
            logger.exception(
                "Kernel generation failed for DPI [%s] and inputs [%s]. Suppressing and continuing with"
                "other kernels (if any)",
                kernel_identifier,
                input_file_names,
            )

    # Duplicates are possible depending on file naming and append flag.
    outputs = sorted(set(outputs))

    # Generate output manifest.
    pedi = Manifest.output_manifest_from_input_manifest(mani)
    pedi.add_files(*outputs)

    # Automatically generates a proper output manifest filename and writes it to the path specified,
    # usually this path is retrieved from the environment.
    pedi.write(output_dir)

    return pedi


def jpss_kernel_cli_handler(parsed_args: argparse.Namespace):
    """Generate SPICE JPSS kernels from command line arguments.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    return from_manifest(
        input_manifest=parsed_args.input_manifest,
        data_product_identifiers=[DataProductIdentifier.spice_jpss_spk, DataProductIdentifier.spice_jpss_ck],
        output_dir=AnyPath(os.environ["PROCESSING_PATH"]),
        overwrite=False,
        append=False,
        verbose=parsed_args.verbose,
    )


def azel_kernel_cli_handler(parsed_args: argparse.Namespace):
    """Generate SPICE Az/El kernels from command line arguments.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    return from_manifest(
        input_manifest=parsed_args.input_manifest,
        data_product_identifiers=[DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck],
        output_dir=AnyPath(os.environ["PROCESSING_PATH"]),
        overwrite=False,
        append=False,
        verbose=parsed_args.verbose,
    )


# TODO[LIBSDC-279]: Delete after the defunct unit test is replaced by something functional!
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
