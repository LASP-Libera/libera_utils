"""Module containing CLI tool for creating SPICE kernels from packets"""
# Standard
import argparse
import logging
from pathlib import Path
import subprocess
import tempfile
# Installed
import numpy as np
import numpy.lib.recfunctions as nprf
from lasp_packets import parser, xtce
# Local
from libera_sdp import LOG_MESSAGE_FORMAT
from libera_sdp import kernels as libera_kernels
from libera_sdp.config import config
from libera_sdp.io import filenaming
from libera_sdp import packets as libera_packets
from libera_sdp import time


def make_jpss_spk(cli_args: list = None):
    """CLI tool that creates a JPSS SPK from APID 11 CCSDS packets.

    Parameters
    ----------
    cli_args : list
        Allows programmatic testing of this tool, including argparsing.

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_MESSAGE_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info("Starting SPK maker. This CLI tool creates an SPK from a list of geolocation packet files.")

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('packet_data_filepaths',
                           nargs='+',
                           type=str,
                           help="Path to an L0 packet file.")
    argparser.add_argument('--outdir',
                           type=str,
                           default='/tmp',
                           help="Output directory for SPK")
    argparser.add_argument('--overwrite',
                           action='store_true',
                           help="Force overwriting an existing kernel if it exists.")
    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help="Verbose output (DEBUG level instead of INFO).")
    parsed_args = argparser.parse_args(cli_args)

    if parsed_args.verbose:
        stream_handler.setLevel(logging.DEBUG)

    output_dir = Path(parsed_args.outdir)
    logger.info(f"Writing resulting SPK to {output_dir}")

    packet_definition_filepath = Path(config.get('JPSS_GEOLOCATION_PACKET_DEFINITION'))
    logger.info(f"Using packet definition {packet_definition_filepath}")

    packet_definition = xtce.XtcePacketDefinition(xtce_document=str(packet_definition_filepath))
    # TODO: eventually XtcePacketDefinition will accept a Path object. When we update that, remove the str() above
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

    with tempfile.TemporaryDirectory(prefix='/tmp/') as tmp_dir:
        tmp_path = Path(tmp_dir)
        spk_data_filepath = libera_kernels.write_kernel_input_file(
            packet_data,
            filepath=tmp_path / 'mkspk_data.txt',
            fields=['ET', 'ADGPSPOSX', 'ADGPSPOSY', 'ADGPSPOSZ', 'ADGPSVELX', 'ADGPSVELY', 'ADGPSVELZ'])
        logger.info(f"MKSPK input data written to {spk_data_filepath}")

        spk_setup_filepath = libera_kernels.write_kernel_setup_file(
            config.get("MKSPK_SETUPFILE_CONTENTS"),
            filepath=tmp_path / 'mkspk_setup.txt')
        logger.info(f"MKSPK setup file written to {spk_setup_filepath}")

        utc_start_str = time.et_2_datetime(ephemeris_time[0])
        utc_end_str = time.et_2_datetime(ephemeris_time[-1])
        spk_filename = filenaming.EphemerisKernelFilename(utc_start=utc_start_str, utc_end=utc_end_str)
        output_filepath = Path(output_dir) / spk_filename.name

        if parsed_args.overwrite is True:
            output_filepath.unlink(missing_ok=True)

        logger.info("Running MKSPK...")
        subprocess.run(['mkspk',
                        '-setup', str(spk_setup_filepath),
                        '-input', str(spk_data_filepath),
                        '-output', str(output_filepath)],
                       check=True)
        logger.info(f"Finished! SPK written to {output_filepath}")


def make_jpss_ck(cli_args: list = None):
    """CLI tool that creates a JPSS CK from APID 11 CCSDS packets.

    Parameters
    ----------
    cli_args : list
        Allows programmatic testing of this tool, including argparsing.

    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_MESSAGE_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info("Starting CK maker. This CLI tool creates a CK from a list of geolocation packet files.")

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('packet_data_filepaths',
                           nargs='+',
                           type=str,
                           help="Path to an L0 packet file.")
    argparser.add_argument('--outdir',
                           type=str,
                           default='/tmp',
                           help="Output directory for CK")
    argparser.add_argument('--overwrite',
                           action='store_true',
                           help="Force overwriting an existing kernel if it exists.")
    argparser.add_argument('-v', '--verbose',
                           action='store_true',
                           help="Verbose output (DEBUG level instead of INFO).")
    parsed_args = argparser.parse_args(cli_args)

    if parsed_args.verbose:
        stream_handler.setLevel(logging.DEBUG)

    output_dir = Path(parsed_args.outdir)
    logger.info(f"Writing resulting CK to {output_dir}")

    packet_definition_filepath = Path(config.get('JPSS_GEOLOCATION_PACKET_DEFINITION'))
    logger.info(f"Using packet definition {packet_definition_filepath}")

    packet_definition = xtce.XtcePacketDefinition(xtce_document=str(packet_definition_filepath))
    # TODO: eventually XtcePacketDefinition will accept a Path object. When we update that, remove the str() above
    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    logger.info("Parsing packets...")
    packet_data = libera_packets.parse_packets(packet_parser, parsed_args.packet_data_filepaths)
    logger.info("Done.")

    # Add a column that is the SCLK string, formatted with delimiters, to the input data recarray
    attitude_sclk_string = [f"{row['ADAET2DAY']}:{row['ADAET2MS']}:{row['ADAET2US']}" for row in packet_data]
    packet_data = nprf.append_fields(packet_data, 'ATTSCLKSTR', attitude_sclk_string)

    with tempfile.TemporaryDirectory(prefix='/tmp/') as tmp_dir:
        tmp_path = Path(tmp_dir)
        ck_data_filepath = libera_kernels.write_kernel_input_file(
            packet_data,
            filepath=tmp_path / 'msopck_data.txt',
            fields=['ATTSCLKSTR', 'ADCFAQ4', 'ADCFAQ1', 'ADCFAQ2', 'ADCFAQ3'],
            fmt=['%s', '%.16f', '%.16f', '%.16f', '%.16f']
        )  # produces w + i + j + k in SPICE_QUATERNION style
        logger.info(f"MSOPCK input data written to {ck_data_filepath}")

        ck_setup_filepath = libera_kernels.write_kernel_setup_file(
            config.get("MSOPCK_SETUPFILE_CONTENTS"),
            filepath=tmp_path / 'msopck_setup.txt')
        logger.info(f"MSOPCK setup file written to {ck_setup_filepath}")

        utc_start_str = time.et_2_datetime(time.scs2e_wrapper(attitude_sclk_string[0]))
        utc_end_str = time.et_2_datetime(time.scs2e_wrapper(attitude_sclk_string[-1]))
        ck_filename = filenaming.AttitudeKernelFilename(ck_object='jpss', utc_start=utc_start_str, utc_end=utc_end_str)
        output_filepath = Path(output_dir) / ck_filename.name

        if parsed_args.overwrite is True:
            output_filepath.unlink(missing_ok=True)

        logger.info("Running MSOPCK...")
        subprocess.run(['msopck', str(ck_setup_filepath), str(ck_data_filepath), str(output_filepath)],
                       check=True)
        logger.info(f"Finished! SPK written to {output_filepath}")
