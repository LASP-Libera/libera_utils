"""Module containing CLI tool for creating SPICE kernels from packets"""
# Standard
import argparse
import warnings
from datetime import datetime
import logging
from pathlib import Path
import shutil
import subprocess  # nosec B404
import tempfile
# Installed
import numpy as np
import numpy.lib.recfunctions as nprf
from lasp_packets import parser, xtcedef
# Local
from libera_utils import spice_utils
from libera_utils.logutil import setup_task_logger
from libera_utils.config import config
from libera_utils.io import filenaming
from libera_utils import packets as libera_packets
from libera_utils import time

logger = logging.getLogger(__name__)


def make_jpss_spk(parsed_args: argparse.Namespace):
    """Create a JPSS SPK from APID 11 CCSDS packets.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    if parsed_args.verbose:
        stream_log_level = logging.DEBUG
    else:
        stream_log_level = logging.NOTSET

    now = datetime.utcnow().strftime("%Y%m%dt%H%M%S")
    log_filepath = setup_task_logger(f'spk_generator_{now}', stream_log_level=stream_log_level)

    logger.info("Starting SPK maker. This CLI tool creates an SPK from a list of geolocation packet files.")
    logger.info("Logging to %s", log_filepath)

    output_dir = Path(parsed_args.outdir).expanduser().absolute()
    logger.info("Writing resulting SPK to %s", output_dir)

    packet_definition_filepath = Path(config.get('JPSS_GEOLOCATION_PACKET_DEFINITION'))
    logger.info("Using packet definition %s", packet_definition_filepath)

    packet_definition = xtcedef.XtcePacketDefinition(packet_definition_filepath)
    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    logger.info("Parsing packets...")
    packet_data = libera_packets.parse_packets(packet_parser, parsed_args.packet_data_filepaths)
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

        utc_start_str = time.et_2_datetime(ephemeris_time[0])
        utc_end_str = time.et_2_datetime(ephemeris_time[-1])
        spk_filename = filenaming.EphemerisKernelFilename(utc_start=utc_start_str, utc_end=utc_end_str)
        output_filepath = Path(output_dir) / spk_filename.name

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


def make_jpss_ck(parsed_args: argparse.Namespace):
    """Create a JPSS CK from APID 11 CCSDS packets.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    if parsed_args.verbose:
        stream_log_level = logging.DEBUG
    else:
        stream_log_level = logging.NOTSET

    now = datetime.utcnow().strftime("%Y%m%dt%H%M%S")
    log_filepath = setup_task_logger(f'ck_generator_{now}', stream_log_level=stream_log_level)

    logger.info("Starting CK maker. This CLI tool creates a CK from a list of geolocation packet files.")
    logger.info("Logging to %s", log_filepath)

    output_dir = Path(parsed_args.outdir).expanduser().absolute()
    logger.info("Writing resulting CK to %s", output_dir)

    packet_definition_filepath = Path(config.get('JPSS_GEOLOCATION_PACKET_DEFINITION'))
    logger.info("Using packet definition %s", packet_definition_filepath)

    packet_definition = xtcedef.XtcePacketDefinition(packet_definition_filepath)
    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    logger.info("Parsing packets...")
    packet_data = libera_packets.parse_packets(packet_parser, parsed_args.packet_data_filepaths)
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

        utc_start_str = time.et_2_datetime(time.scs2e_wrapper(attitude_sclk_string[0]))
        utc_end_str = time.et_2_datetime(time.scs2e_wrapper(attitude_sclk_string[-1]))
        ck_filename = filenaming.AttitudeKernelFilename(ck_object='jpss', utc_start=utc_start_str, utc_end=utc_end_str)
        output_filepath = Path(output_dir) / ck_filename.name

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


def make_azel_ck(parsed_args: argparse.Namespace):
    """Create a Libera Az-El CK from CCSDS packets

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    print(parsed_args)
    raise NotImplementedError("CK generation for the Az-El mechanism isn't implemented yet.")


def write_kernel_input_file(data: np.ndarray, filepath: str or Path, fields: list = None, fmt: str or list = "%.16f"):
    """Write ephemeris and attitude data to MKSPK and MSOPCK input data files, respectively.

    See MSOPCK documentation here:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/msopck.html
    See MKSPK documentation here:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/mkspk.html

    Parameters
    ----------
    data : np.ndarray
        Structured array (named, with data types) of attitude or ephemeris data.
    filepath : str or Path
        Filepath to write to.
    fields : list
        Optional. List of field names to write out to the data file. If not specified, assume fields are already
        in the proper order.
    fmt : str or list
        Format specifier(s) to pass to np.savetxt. Default is to assume everything should be floats with 16 decimal
        places of precision (%.16f). If a list is passed, it must contain a format specifier for each column in data.

    Returns
    -------
    : Path
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
    filepath : Path
        Filepath to write to.

    Returns
    -------
    : Path
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
