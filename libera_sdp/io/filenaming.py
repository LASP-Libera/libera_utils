"""Module for file naming utilities"""
import re

from libera_sdp.time import ISOT_REGEX


SPK_REGEX = re.compile(r"^libera_jpss"
                       r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                       r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                       r"\.bsp$")

CK_REGEX = re.compile(r"^libera_(?P<object>jpss|rad)"
                      r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                      r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                      r"\.bc$")

LIBERA_PRODUCT_REGEX = re.compile(r"^libera"
                                  r"_(?P<instrument>cam|rad)"
                                  r"_(?P<level>l0|l1a|l1b|l2)"
                                  r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                                  r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                                  r"\.(?P<extension>pkts|h5)$")


def isot_printable(isot: str):
    """Make an ISO-T timestamp printable in filenames by removing hyphens, semicolons, and the trailing
    fractional seconds.

    Parameters
    ----------
    isot : str
        ISO T timestamp or an array of them

    Returns
    -------
    : str
        Reformatted string or strings for printing
    """
    m = ISOT_REGEX.match(isot)
    return f"{m['year']}{m['month']}{m['day']}t{m['hour']}{m['minute']}{m['second']}"


