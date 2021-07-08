"""Module containing CLI tool for creating SPICE kernels from packets"""
# Standard library
import argparse
import logging
from pathlib import Path
import subprocess
# Installed modules
import bitstring
import numpy as np
import numpy.lib.recfunctions as nprf
from lasp_packets import parser, xtce
# Local packages
from libera_sdp import kernels as libera_kernels
from libera_sdp.config import config
from libera_sdp.io import filenaming
from libera_sdp import packets as libera_packets
from libera_sdp import time

logger = logging.getLogger(__name__)


_tmp_mkspk_data_location = Path('/tmp/mkspk_data.txt')
_tmp_mkspk_setup_location = Path('/tmp/mkspk_setup.txt')
_tmp_msopck_data_location = Path('/tmp/msopck_data.txt')
_tmp_msopck_setup_location = Path('/tmp/msopck_setup.txt')


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
    parsed_args = argparser.parse_args(cli_args)

    output_dir = Path(parsed_args.outdir)

    packet_definition_filepath = Path(config.get('JPSS1_APID11_PACKET_DEFINITION'))

    packet_definition = xtce.XtcePacketDefinition(xtce_document=str(packet_definition_filepath))
    # TODO: eventually XtcePacketDefinition will accept a Path object. When we update that, remove the str() above
    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    packet_data = libera_packets.parse_packets(packet_parser, parsed_args.packet_data_filepaths)

    # Calculate and append a Julian Day representation of the epochs. MKSPK is picky about time formats.
    ephemeris_jd = time.ccsdsjd_2_jd(
        time.days_ms_us_2_decimal_days(
            packet_data['ADAET1DAY'], packet_data['ADAET1MS'], packet_data['ADAET1US']
        )
    )
    packet_data = nprf.append_fields(packet_data, 'EPHEMJD', ephemeris_jd, dtypes=(np.float64,))

    data_filepath = libera_kernels.write_kernel_input_file(
        packet_data,
        filepath=_tmp_mkspk_data_location,
        fields=['EPHEMJD', 'ADGPSPOSX', 'ADGPSPOSY', 'ADGPSPOSZ', 'ADGPSVELX', 'ADGPSVELY', 'ADGPSVELZ'])
    setup_filepath = libera_kernels.write_kernel_setup_file(
        libera_kernels.default_spk_setup,
        filepath=_tmp_mkspk_setup_location)

    utc_start_str = filenaming.isot_printable(time.jd_2_utc(ephemeris_jd[0]))
    utc_end_str = filenaming.isot_printable(time.jd_2_utc(ephemeris_jd[-1]))
    spk_filename = filenaming.EphemerisKernelFilename(utc_start=utc_start_str, utc_end=utc_end_str)
    output_filepath = Path(output_dir) / spk_filename.name

    if parsed_args.overwrite is True:
        output_filepath.unlink(missing_ok=True)

    subprocess.run(['mkspk',
                    '-setup', str(setup_filepath),
                    '-input', str(data_filepath),
                    '-output', str(output_filepath)],
                   check=True)


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
    parsed_args = argparser.parse_args(cli_args)

    output_dir = Path(parsed_args.outdir)

    packet_definition_filepath = Path(config.get('JPSS1_APID11_PACKET_DEFINITION'))
    packet_definition = xtce.XtcePacketDefinition(xtce_document=str(packet_definition_filepath))
    # TODO: eventually XtcePacketDefinition will accept a Path object. When we update that, remove the str() above
    packet_parser = parser.PacketParser(packet_definition=packet_definition)

    packet_data = libera_packets.parse_packets(packet_parser, parsed_args.packet_data_filepaths)

    attitude_jd = time.ccsdsjd_2_jd(
        time.days_ms_us_2_decimal_days(
            packet_data['ADAET2DAY'], packet_data['ADAET2MS'], packet_data['ADAET2US']))
    attitude_isot = time.jd_2_utc(attitude_jd)
    packet_data = nprf.append_fields(packet_data, 'ATTISO', attitude_isot)

    ck_data_path = libera_kernels.write_kernel_input_file(
        packet_data,
        filepath=_tmp_msopck_data_location,
        fields=['ATTISO', 'ADCFAQ4', 'ADCFAQ1', 'ADCFAQ2', 'ADCFAQ3'],
        fmt=['%s', '%.16f', '%.16f', '%.16f', '%.16f']
    )  # produces w + i + j + k in SPICE_QUATERNION style
    ck_setup_path = libera_kernels.write_kernel_setup_file(
        libera_kernels.default_ck_setup,
        filepath=_tmp_msopck_setup_location)

    utc_start_str = filenaming.isot_printable(attitude_isot[0])
    utc_end_str = filenaming.isot_printable(attitude_isot[-1])
    ck_filename = filenaming.AttitudeKernelFilename(ck_object='jpss', utc_start=utc_start_str, utc_end=utc_end_str)
    output_filepath = Path(output_dir) / ck_filename.name

    if parsed_args.overwrite is True:
        output_filepath.unlink(missing_ok=True)

    subprocess.run(['msopck', str(ck_setup_path), str(ck_data_path), str(output_filepath)],
                   check=True)
