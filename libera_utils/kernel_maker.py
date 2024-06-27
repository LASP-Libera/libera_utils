"""Module containing CLI tool for creating SPICE kernels from packets"""
# Standard
import argparse
import warnings
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import subprocess  # nosec B404
import tempfile
# Installed
import numpy as np
import numpy.lib.recfunctions as nprf
from cloudpathlib import S3Path, AnyPath
from space_packet_parser import parser, xtcedef
# Local
from libera_utils import spice_utils
from libera_utils.logutil import configure_task_logging
from libera_utils.config import config
from libera_utils.io import filenaming
from libera_utils import packets as libera_packets
from libera_utils import time
from libera_utils.io.smart_open import smart_open, smart_copy_file
from libera_utils.io.manifest import Manifest

logger = logging.getLogger(__name__)


def make_jpss_kernels_from_manifest(manifest_file_path: str or AnyPath,
                                    output_directory: str or AnyPath):
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
            files_in_range.append(str(file_entry["filename"]))
    else:
        # Load desired time range from the manifest configuration
        start_time_text = m.configuration["start_time"]
        desired_start_time = datetime.strptime(start_time_text, '%Y-%m-%d:%H:%M:%S')
        end_time_text = m.configuration["end_time"]
        desired_end_time = datetime.strptime(end_time_text, '%Y-%m-%d:%H:%M:%S')

        # Load the packet files and check the time ranges against the manifest configuration
        # TODO update this if possible to use the metadata files when those are more defined

        for file_entry in m.files:
            file_path_from_list = file_entry["filename"]
            packet_data = get_spice_packet_data_from_filepaths([file_path_from_list])
            ephemeris_time = time.scs2e_wrapper(
                [f"{d}:{ms}:{us}" for d, ms, us in
                 zip(packet_data['ADAET1DAY'], packet_data['ADAET1MS'], packet_data['ADAET1US'])]
            )
            # Check if any of the packet data are in increasing order by comparing an array element to
            # its right neighbor and ensuring that is always greater or equal. If this is not true
            # throw an error
            if not np.all(ephemeris_time[:-1] <= ephemeris_time[1:]):
                raise ValueError(f"The data in {file_path_from_list} are not monotonic in time")
            packet_start_time = time.et_2_datetime(ephemeris_time[0])
            packet_end_time = time.et_2_datetime(ephemeris_time[-1])

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
        packet_data_filepaths=files_in_range,
        outdir=str(output_directory),
        overwrite=False,
        verbose=False
    )
    make_jpss_spk(parsed_args)
    make_jpss_ck(parsed_args)

    return output_directory


def get_spice_packet_data_from_filepaths(packet_data_filepaths):
    """Utility function to return an array of packet data from a list of file paths of raw JPSS APID 11
    geolocation packet data files.

     Parameters
    ----------
    packet_data_filepaths : list
        The list of file paths to the raw packet data

    Returns
    -------
    packet_data : numpy.ndarray
        The configured packet data. See packets.py for more details on structure
     """
    packet_definition_uri = AnyPath(config.get('JPSS_GEOLOCATION_PACKET_DEFINITION'))
    logger.info("Using packet definition %s", packet_definition_uri)

    with smart_open(packet_definition_uri) as packet_definition_filepath:
        packet_definition = xtcedef.XtcePacketDefinition(packet_definition_filepath)

    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    packet_data = libera_packets.parse_packets(packet_parser, packet_data_filepaths)

    return packet_data


def make_jpss_spk(parsed_args: argparse.Namespace):
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

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    configure_task_logging(f'spk_generator_{now}',
                           limit_debug_loggers='libera_utils',
                           console_log_level=logging.DEBUG if parsed_args.verbose else None)

    logger.info("Starting SPK maker. This CLI tool creates an SPK from a list of geolocation packet files.")

    output_dir = AnyPath(parsed_args.outdir)
    logger.info("Writing resulting SPK to %s", output_dir)

    logger.info("Parsing packets...")
    packet_data = get_spice_packet_data_from_filepaths(parsed_args.packet_data_filepaths)
    logger.info("Done.")

    # Calculate and append a ET representation of the epochs. MKSPK is picky about time formats.
    ephemeris_time = time.scs2e_wrapper(
        [f"{d}:{ms}:{us}" for d, ms, us in
         zip(packet_data['ADAET1DAY'], packet_data['ADAET1MS'], packet_data['ADAET1US'])]
    )
    packet_data = nprf.append_fields(packet_data, 'ET', ephemeris_time, dtypes=(np.float64,))

    with tempfile.TemporaryDirectory(prefix='/tmp/') as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        spk_data_filepath = write_kernel_input_file(
            packet_data,
            filepath=tmp_path / 'mkspk_data.txt',
            fields=['ET', 'ADGPSPOSX', 'ADGPSPOSY', 'ADGPSPOSZ', 'ADGPSVELX', 'ADGPSVELY', 'ADGPSVELZ'])
        logger.info("MKSPK input data written to %s", spk_data_filepath)

        spk_setup_filepath = write_kernel_setup_file(
            config.get("MKSPK_SETUPFILE_CONTENTS"),
            filepath=tmp_path / 'mkspk_setup.txt')
        logger.info("MKSPK setup file written to %s", spk_setup_filepath)

        utc_start = time.et_2_datetime(ephemeris_time[0])
        utc_end = time.et_2_datetime(ephemeris_time[-1])
        revision_time = datetime.now(timezone.utc)
        spk_filename = filenaming.EphemerisKernelFilename.from_filename_parts(
            spk_object='jpss',
            utc_start=utc_start,
            utc_end=utc_end,
            version=filenaming.get_current_version_str('libera_utils'),
            revision=revision_time)
        output_filepath = tmp_path / spk_filename.path.name  # pylint: disable=no-member

        if parsed_args.overwrite is True:
            output_filepath.unlink(missing_ok=True)

        logger.info("Running MKSPK...")
        try:
            result = subprocess.run(['mkspk',  # nosec B603 B607
                                     '-setup', str(spk_setup_filepath),
                                     '-input', str(spk_data_filepath),
                                     '-output', str(output_filepath)],
                                    capture_output=True, check=True)
        except subprocess.CalledProcessError as cpe:
            logger.info("Captured stdout: \n%s", cpe.stdout.decode())
            if cpe.stderr:
                logger.error("Captured stderr: \n%s", cpe.stderr.decode())
            raise

        logger.info("Captured stdout:\n%s", result.stdout.decode())
        if result.stderr:
            logger.error(result.stderr.decode())
        logger.info("Finished! SPK written to %s", output_filepath)

        output_full_path = output_dir / spk_filename.path.name  # pylint: disable=no-member
        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        smart_copy_file(output_filepath, output_full_path)
        logger.info("SPK copied to %s", output_full_path)


def make_jpss_ck(parsed_args: argparse.Namespace):
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
    now = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")
    configure_task_logging(f'ck_generator_{now}',
                           limit_debug_loggers='libera_utils',
                           console_log_level=logging.DEBUG if parsed_args.verbose else None)

    logger.info("Starting CK maker. This CLI tool creates a CK from a list of JPSS attitue/quaternion packet files.")

    output_dir = AnyPath(parsed_args.outdir)
    logger.info("Writing resulting CK to %s", output_dir)
    packet_data = get_spice_packet_data_from_filepaths(parsed_args.packet_data_filepaths)
    logger.info("Done.")

    # Add a column that is the SCLK string, formatted with delimiters, to the input data recarray
    attitude_sclk_string = [f"{row['ADAET2DAY']}:{row['ADAET2MS']}:{row['ADAET2US']}" for row in packet_data]
    packet_data = nprf.append_fields(packet_data, 'ATTSCLKSTR', attitude_sclk_string)

    with tempfile.TemporaryDirectory(prefix='/tmp/') as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        ck_data_filepath = write_kernel_input_file(
            packet_data,
            filepath=tmp_path / 'msopck_data.txt',
            fields=['ATTSCLKSTR', 'ADCFAQ4', 'ADCFAQ1', 'ADCFAQ2', 'ADCFAQ3'],
            fmt=['%s', '%.16f', '%.16f', '%.16f', '%.16f']
        )  # produces w + i + j + k in SPICE_QUATERNION style
        logger.info("MSOPCK input data written to %s", ck_data_filepath)

        ck_setup_filepath = write_kernel_setup_file(
            config.get("MSOPCK_SETUPFILE_CONTENTS"),
            filepath=tmp_path / 'msopck_setup.txt')
        logger.info("MSOPCK setup file written to %s", ck_setup_filepath)

        utc_start = time.et_2_datetime(time.scs2e_wrapper(attitude_sclk_string[0]))
        utc_end = time.et_2_datetime(time.scs2e_wrapper(attitude_sclk_string[-1]))
        revision_time = datetime.now(timezone.utc)
        ck_filename = filenaming.AttitudeKernelFilename.from_filename_parts(
            ck_object='jpss',
            utc_start=utc_start,
            utc_end=utc_end,
            version=filenaming.get_current_version_str('libera_utils'),
            revision=revision_time)
        output_filepath = tmp_path / ck_filename.path.name  # pylint: disable=no-member

        if parsed_args.overwrite is True:
            output_filepath.unlink(missing_ok=True)

        logger.info("Running MSOPCK...")
        try:
            result = subprocess.run(['msopck',  # nosec B603 B607
                                     str(ck_setup_filepath), str(ck_data_filepath), str(output_filepath)],
                                    capture_output=True, check=True)
        except subprocess.CalledProcessError as cpe:
            logger.info("Captured stdout: \n%s", cpe.stdout.decode())
            if cpe.stderr:
                logger.error("Captured stderr: \n%s", cpe.stderr.decode())
            raise

        logger.info("Captured stdout:\n%s", result.stdout.decode())
        if result.stderr:
            logger.error(result.stderr.decode())
        logger.info("Finished! CK written to %s", output_filepath)

        output_full_path = output_dir / ck_filename.path.name  # pylint: disable=no-member
        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        smart_copy_file(output_filepath, output_full_path)
        logger.info("CK copied to %s", output_full_path)


def make_azel_ck(parsed_args: argparse.Namespace):  # pylint: disable=too-many-statements
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

    now = datetime.utcnow().strftime("%Y%m%dt%H%M%S")
    configure_task_logging(f'ck_generator_{now}',
                           limit_debug_loggers='libera_utils',
                           console_log_level=logging.DEBUG if parsed_args.verbose else None)

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
        this_config = config.get("MSOPCK_AZ_SETUPFILE_CONTENTS")
        ck_object = 'azrot'

        azel_sclk_string = [f"{row['ADAET2DAY']}:{row['ADAET2MS']}:{row['ADAET2US']}" for row in packet_data]
        packet_data = nprf.append_fields(packet_data, 'ATTSCLKSTR', azel_sclk_string)
        utc_start = time.et_2_datetime(time.scs2e_wrapper(azel_sclk_string[0]))
        utc_end = time.et_2_datetime(time.scs2e_wrapper(azel_sclk_string[-1]))
    else:
        logger.info("Parsing CSV file...")
        # get the data from the ASCII file
        packet_data=np.genfromtxt(parsed_args.packet_data_filepaths[0], delimiter=',', dtype='double')
        # make sure we have all 3 axis defined: X is RAM, Y is Elev when Az is at 0.0, Z is nadir
        if (parsed_args.azimuth is True) and (parsed_args.elevation is True):
            try:
                raise ValueError("Expecting only one: --azimuth or --elevation. Got both\n")
            except ValueError as error:
                logger.exception(error)

        if parsed_args.azimuth:
            packet_data=packet_data.view([('ET_TIME', 'double'), ('AZIMUTH', 'double')])
            packet_data = nprf.append_fields(packet_data, 'ELEVATION', np.zeros(packet_data.size,dtype='double'))
            this_config = config.get("MSOPCK_AZ_SETUPFILE_CONTENTS")
            ck_object = 'azrot'
        elif parsed_args.elevation:
            packet_data=packet_data.view([('ET_TIME', 'double'), ('ELEVATION', 'double')])
            packet_data = nprf.append_fields(packet_data, 'AZIMUTH', np.zeros(packet_data.size, dtype='double'))
            this_config = config.get("MSOPCK_EL_SETUPFILE_CONTENTS")
            ck_object = 'elscan'
        else:
            try:
                raise ValueError("Expecting at least one: --azimuth or --elevation. None provided.\n")
            except ValueError as error:
                logger.exception(error)

        packet_data = nprf.append_fields(packet_data, 'AZEL_Z', np.zeros(packet_data.size, dtype='double'))
        azel_sclk_string = [f"{d}" for d in packet_data['ET_TIME']]
        packet_data = nprf.append_fields(packet_data, 'AZELSCLKSTR', azel_sclk_string)
        utc_start = time.et_2_datetime(packet_data['ET_TIME'][0])
        utc_end = time.et_2_datetime(packet_data['ET_TIME'][-1])

    logger.info("Done.")

    with tempfile.TemporaryDirectory(prefix='/tmp/') as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        ck_data_filepath = write_kernel_input_file(
            packet_data,
            filepath=tmp_path / 'msopck_data.txt',
            fields=['AZELSCLKSTR', 'AZIMUTH', 'ELEVATION', 'AZEL_Z'],
            fmt=['%s', '%.16f', '%.16f', '%.16f']
        )  # produces  X, Y, Z angles for the specific mechanism
        logger.info("MSOPCK input data written to %s", ck_data_filepath)

        ck_setup_filepath = write_kernel_setup_file(
            this_config,
            filepath=tmp_path / 'msopck_setup.txt')
        logger.info("MSOPCK setup file written to %s", ck_setup_filepath)

        revision_time = datetime.utcnow()
        ck_filename = filenaming.AttitudeKernelFilename.from_filename_parts(
            ck_object=ck_object,
            utc_start=utc_start,
            utc_end=utc_end,
            version=filenaming.get_current_version_str('libera_utils'),
            revision=revision_time)
        output_filepath = tmp_path / ck_filename.path.name  # pylint: disable=no-member

        if parsed_args.overwrite is True:
            output_filepath.unlink(missing_ok=True)

        logger.info("Running MSOPCK...")
        try:
            result = subprocess.run(['msopck',  # nosec B603 B607
                                     str(ck_setup_filepath), str(ck_data_filepath), str(output_filepath)],
                                    capture_output=True, check=True)
        except subprocess.CalledProcessError as cpe:
            logger.info("Captured stdout: \n%s", cpe.stdout.decode())
            if cpe.stderr:
                logger.error("Captured stderr: \n%s", cpe.stderr.decode())
            raise

        logger.info("Captured stdout:\n%s", result.stdout.decode())
        if result.stderr:
            logger.error(result.stderr.decode())
        logger.info("Finished! CK written to %s", output_filepath)

        output_full_path = output_dir / ck_filename.path.name  # pylint: disable=no-member
        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        smart_copy_file(output_filepath, output_full_path)
        logger.info("CK copied to %s", output_full_path)


def write_kernel_input_file(data: np.ndarray, filepath: str or Path or S3Path,
                            fields: list = None, fmt: str or list = "%.16f"):
    """Write ephemeris and attitude data to MKSPK and MSOPCK input data files, respectively.

    See MSOPCK documentation here:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/msopck.html
    See MKSPK documentation here:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/mkspk.html

    Parameters
    ----------
    data : numpy.ndarray
        Structured array (named, with data types) of attitude or ephemeris data.
    filepath : str or pathlib.Path
        Filepath to write to.
    fields : list
        Optional. List of field names to write out to the data file. If not specified, assume fields are already
        in the proper order.
    fmt : str or list
        Format specifier(s) to pass to np.savetxt. Default is to assume everything should be floats with 16 decimal
        places of precision (%.16f). If a list is passed, it must contain a format specifier for each column in data.

    Returns
    -------
    : pathlib.Path
        Absolute path to written file.
    """
    if fields:
        np.savetxt(filepath, data[fields], delimiter=" ", fmt=fmt)
    else:
        np.savetxt(filepath, data[:], delimiter=" ", fmt=fmt)
    return filepath.absolute()


def write_kernel_setup_file(data: dict, filepath: Path):
    """Write an MSOPCK or MKSPK compatible setup file of key-value pairs.
    See documentation here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/msopck.html#Input%20Data%20Format

    Parameters
    ----------
    data : dict
        Dictionary of key-value pairs to write to the setup file.
    filepath : pathlib.Path
        Filepath to write to.

    Returns
    -------
    : pathlib.Path
        Absolute path to written file.
    """
    with open(filepath, 'x+', encoding='utf_8') as fh:
        fh.write("\\begindata\n")
        for key, value in data.items():
            if key in ('PATH_VALUES', 'PATH_SYMBOLS', 'KERNELS_TO_LOAD'):
                inside = ", ".join([f"\n\t'{item}'" for item in value])
                value_str = f"({inside}\n)"
            elif key in ('LSK_FILE_NAME', 'LEAPSECONDS_FILE'):
                lsk_url = spice_utils.find_most_recent_naif_kernel(spice_utils.NAIF_LSK_INDEX_URL,
                                                                   spice_utils.NAIF_LSK_REGEX)
                lsk = spice_utils.KernelFileCache(lsk_url)
                if len(str(lsk.kernel_path)) > 78:
                    # MSOPCK is limited to 80 character file names (including single quotes)
                    # so we copy the kernel to the same directory where we are writing this setup file
                    copied_kernel = shutil.copy(str(lsk.kernel_path), filepath.parent)
                    value_str = f"'{copied_kernel}'"
                else:
                    value_str = f"'{str(lsk.kernel_path)}'"
            elif key == 'SCLK_FILE_NAME' and len(value) > 78:
                # MSOPCK is limited to 80 character file names (including single quotes) so we copy the SCLK to the
                # same directory where we are writing this setup file
                copied_sclk = shutil.copy(value, filepath.parent)
                value_str = f"'{copied_sclk}'"
            elif key == 'EULER_ROTATIONS_ORDER':
                value_str = f"{value}"
            elif isinstance(value, str):
                value_str = f"'{value}'"
            elif isinstance(value, list):
                list_str = " ".join(value)
                value_str = f"'{list_str}'"
            elif isinstance(value, dict):
                dict_str = " ".join([f"\n\t'{k}={v}'" for k, v in value.items()])
                value_str = f"({dict_str}\n)"
            else:
                value_str = f"{value}"
            if len(value_str) > 80:
                warnings.warn("Detected a SPICE setup file value that is over 80 characters. "
                              f"This will likely cause an error. {value_str}")
            fh.write(f"{key}={value_str}\n")
        fh.write("\\begintext\n")
        fh.seek(0)
        logger.info("Setup file contents:\n%s", ''.join(fh.readlines()))
    return filepath.absolute()
