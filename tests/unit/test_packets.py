"""Tests for libera_utils.io.packets module"""

import numpy as np
from space_packet_parser import parser

from libera_utils import packets as libera_packets


def test_array_from_packets():
    """Test creating a numpy array from a list of Packet objects"""
    packets = [
        parser.Packet(
            header={
                "VERSION": parser.ParsedDataItem("VERSION", 0),
                "TYPE": parser.ParsedDataItem("TYPE", 0),
                "SEC_HDR_FLG": parser.ParsedDataItem("SEC_HDR_FLG", 1),
                "PKT_APID": parser.ParsedDataItem("PKT_APID", 11),
                "SEQ_FLGS": parser.ParsedDataItem("SEQ_FLGS", 3),
                "SRC_SEQ_CTR": parser.ParsedDataItem("SRC_SEQ_CTR", 2605),
                "PKT_LEN": parser.ParsedDataItem("PKT_LEN", 64),
            },
            data={
                "PARAM_1": parser.ParsedDataItem("PARAM_1", "foostring"),
                "PARAM_2": parser.ParsedDataItem("PARAM_2", 42, derived_value=42.42),
                "PARAM_3": parser.ParsedDataItem("PARAM_3", 3.14),
            },
        ),
        parser.Packet(
            header={
                "VERSION": parser.ParsedDataItem("VERSION", 0),
                "TYPE": parser.ParsedDataItem("TYPE", 0),
                "SEC_HDR_FLG": parser.ParsedDataItem("SEC_HDR_FLG", 1),
                "PKT_APID": parser.ParsedDataItem("PKT_APID", 11),
                "SEQ_FLGS": parser.ParsedDataItem("SEQ_FLGS", 3),
                "SRC_SEQ_CTR": parser.ParsedDataItem("SRC_SEQ_CTR", 2606),
                "PKT_LEN": parser.ParsedDataItem("PKT_LEN", 64),
            },
            data={
                "PARAM_1": parser.ParsedDataItem("PARAM_1", "barstring"),
                "PARAM_2": parser.ParsedDataItem("PARAM_2", 43, derived_value=52.43),
                "PARAM_3": parser.ParsedDataItem("PARAM_3", 4.15),
            },
        ),
        parser.Packet(
            header={
                "VERSION": parser.ParsedDataItem("VERSION", 0),
                "TYPE": parser.ParsedDataItem("TYPE", 0),
                "SEC_HDR_FLG": parser.ParsedDataItem("SEC_HDR_FLG", 1),
                "PKT_APID": parser.ParsedDataItem("PKT_APID", 12),
                "SEQ_FLGS": parser.ParsedDataItem("SEQ_FLGS", 3),
                "SRC_SEQ_CTR": parser.ParsedDataItem("SRC_SEQ_CTR", 2607),
                "PKT_LEN": parser.ParsedDataItem("PKT_LEN", 64),
            },
            data={
                "PARAM_1": parser.ParsedDataItem("PARAM_1", "foostring"),
                "PARAM_2": parser.ParsedDataItem("PARAM_2", 1, derived_value=99.99),
                "PARAM_3": parser.ParsedDataItem("PARAM_3", 6.18),
            },
        ),
    ]
    result = libera_packets.array_from_packets(packets, apid=11)
    expected = np.array(
        [
            ("foostring", 42.42, 3.14),
            ("barstring", 52.43, 4.15),
        ],
        dtype=[("PARAM_1", object), ("PARAM_2", float), ("PARAM_3", float)],
    )
    assert np.array_equal(result, expected)
