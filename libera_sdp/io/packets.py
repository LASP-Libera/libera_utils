"""Module for reading packet data"""
import logging

import numpy as np


logger = logging.getLogger(__name__)
_attitude_fields = [
    'ADAET2DAY', 'ADAET2MS', 'ADAET2US', 'ADCFAQ1', 'ADCFAQ2', 'ADCFAQ3', 'ADCFAQ4'
]  # Name origin is the JPSS packet definition
_ephemeris_fields = [
    'ADAET1DAY', 'ADAET1MS', 'ADAET1US', 'ADGPSPOSX', 'ADGPSPOSY', 'ADGPSPOSZ', 'ADGPSVELX', 'ADGPSVELY', 'ADGPSVELZ'
]  # Name origin is the JPSS packet definition


def array_from_packets(packets: list, apid: int = None):
    """Create an array from a packet generator, as returned from a lasp_packets PacketParser. This function assumes
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
    : np.recarray
        Record array with one column per field name in the packet type. Values are derived if a derived value exists,
        otherwise, the values are the raw values.
    """
    apids_present = {packet.header['PKT_APID'].raw_value for packet in packets}
    if apid is not None and apid not in apids_present:
        raise ValueError(f"Requested APID not found in parsed packets.")
    elif apid is None and len(apids_present) > 1:
        raise ValueError(f"Multiple APIDs present. To create an array you must specify which APID you want.")
    else:
        apid = apid or apids_present.pop()

    field_values = [
        tuple(pdi.derived_value or pdi.raw_value for pdi in packet.data.values())
        for packet in packets
        if packet.header['PKT_APID'].raw_value == apid
    ]
    names = tuple(pdi.name for pdi in packets[0].data.values())  # Get data field names from the first Packet
    formats = tuple(type(val) if not isinstance(val, str) else object for val in field_values[0])
    return np.array(field_values, dtype={'names': names, 'formats': formats})
