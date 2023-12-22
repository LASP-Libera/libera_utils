"""Module for reading packet data"""
# Standard
import logging
# Installed
from space_packet_parser import parser
import numpy as np
import numpy.lib.recfunctions as nprf
# Local
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def array_from_packets(packets: list, apid: int = None):
    """Create an array from a list of packets. This function assumes
    that the fields and format for every packet is identical for a given APID.

    Parameters
    ----------
    packets : list
        List of lasp_packets.parser.Packet objects.
    apid : int
        Application Packet ID to create an array from. We can only create an array for a single APID because we need
        to assume the same fields in every packet. If not specified, every packet must be of the same APID.

    Returns
    -------
    numpy.recarray
        Record array with one column per field name in the packet type. Values are derived if a derived value exists,
        otherwise, the values are the raw values.
    """
    apids_present = {packet.header['PKT_APID'].raw_value for packet in packets}
    if apid is not None and apid not in apids_present:
        raise ValueError("Requested APID not found in parsed packets.")
    if apid is None and len(apids_present) > 1:
        raise ValueError("Multiple APIDs present. To create an array you must specify which APID you want.")

    apid = apid or apids_present.pop()

    field_values = [
        tuple(pdi.derived_value or pdi.raw_value for pdi in packet.data.values())
        for packet in packets
        if packet.header['PKT_APID'].raw_value == apid
    ]
    names = tuple(pdi.name for pdi in packets[0].data.values())  # Get data field names from the first Packet
    formats = tuple(type(val) if not isinstance(val, str) else object for val in field_values[0])
    return np.array(field_values, dtype={'names': names, 'formats': formats})


def parse_packets(packet_parser: parser.PacketParser, packet_data_filepaths: list, apid: int = None):
    """Parse a recarray from a list of packet filepaths, assuming the same parser for all

    Parameters
    ----------
    packet_parser : space_packet_parser.parser.PacketParser
        Parser, already initialized with the anticipated definition.
    packet_data_filepaths : list
        List of filepaths to packets files.
    apid : int
        Filter on APID so we don't get mismatches in case the parser finds multiple parsable packet definitions
        in the files. This can happen if the XTCE document contains definitions for multiple packet types and >1 of
        those packet types is present in the packet data files.

    Returns
    -------
    numpy.recarray
        Concatenated arrays of packet data.
    """
    data_arrays = []
    for packet_data_filepath in packet_data_filepaths:
        logger.info("Packet data filepath %s", packet_data_filepath)
        with smart_open(packet_data_filepath) as packet_bytes:
            packet_generator = packet_parser.generator(packet_bytes, parse_bad_pkts=False)
            data_arrays.append(array_from_packets(list(packet_generator), apid=apid))

    packet_data = nprf.stack_arrays(data_arrays, asrecarray=True, usemask=False)  # Stack recarrays
    # Remove any duplicates in case we got the same packet twice in different files
    return np.unique(packet_data, axis=0)
