"""Module for file naming utilities"""
from pathlib import Path
import re

from libera_sdp.time import ISOT_REGEX, PRINTABLE_TS_REGEX


SPK_REGEX = re.compile(r"^libera_jpss"
                       r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                       r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                       r"\.bsp$")

CK_REGEX = re.compile(r"^libera_(?P<ck_object>jpss|rad)"
                      r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                      r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                      r"\.bc$")

LIBERA_PRODUCT_REGEX = re.compile(r"^libera"
                                  r"_(?P<instrument>cam|rad)"
                                  r"_(?P<level>l0|l1a|l1b|l2)"
                                  r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                                  r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                                  r"\.(?P<extension>pkts|h5)$")


class EphemerisKernelFilename:
    """Class to construct, store, and manipulate an SPK filename"""
    _regex = SPK_REGEX
    _fmt_str = "libera_jpss_{utc_start}_{utc_end}.bsp"

    def __init__(self, utc_start: str, utc_end: str):
        """Constructor

        Parameters
        ----------
        utc_start : str
            First timestamp in the SPK
        utc_end : str
            Last timestamp in the SPK
        """
        # TODO: Store these start and end times in a more useful format for comparisons
        assert PRINTABLE_TS_REGEX.match(utc_start)
        assert PRINTABLE_TS_REGEX.match(utc_end)
        self.utc_start = utc_start
        self.utc_end = utc_end

    @classmethod
    def from_path(cls, path: str or Path):
        """Create an instance from a given path

        Parameters
        ----------
        path : str or Path
            Path from which to construct the filename

        Returns
        -------
        : cls
        """
        if isinstance(path, str):
            path = Path(path)

        m = cls._regex.match(path.name)
        return cls(utc_start=m['utc_start'], utc_end=m['utc_end'])

    @property
    def name(self):
        """String filename

        Returns
        -------
        : str
            String filename
        """
        return self._fmt_str.format(**{'utc_start': self.utc_start, 'utc_end': self.utc_end})

    @property
    def path(self):
        """Path filename

        Returns
        -------
        : Path
            Path representation of filename
        """
        return Path(self.name)


class AttitudeKernelFilename:
    """Class to construct, store, and manipulate an SPK filename"""
    _regex = CK_REGEX
    _fmt_str = "libera_{object}_{utc_start}_{utc_end}.bc"

    def __init__(self, ck_object: str, utc_start: str, utc_end: str):
        """Constructor

        Parameters
        ----------
        ck_object : str
            Object for which the CK is valid. e.g. a particular spacecraft name or an instrument component.
        utc_start : str
            First timestamp in the CK
        utc_end : str
            Last timestamp in the CK
        """
        # TODO: Store these start and end times in a more useful format for comparisons
        assert PRINTABLE_TS_REGEX.match(utc_start)
        assert PRINTABLE_TS_REGEX.match(utc_end)
        self.ck_object = ck_object
        self.utc_start = utc_start
        self.utc_end = utc_end

    @classmethod
    def from_path(cls, path: str or Path):
        """Create an instance from a given path

        Parameters
        ----------
        path : str or Path
            Path from which to construct the filename

        Returns
        -------
        : cls
        """
        if isinstance(path, str):
            path = Path(path)

        m = cls._regex.match(path.name)
        return cls(ck_object=m['ck_object'], utc_start=m['utc_start'], utc_end=m['utc_end'])

    @property
    def name(self):
        """String filename

        Returns
        -------
        : str
            String filename
        """
        return self._fmt_str.format(
            **{'object': self.ck_object, 'utc_start': self.utc_start, 'utc_end': self.utc_end})

    @property
    def path(self):
        """Path filename

        Returns
        -------
        : Path
            Path representation of filename
        """
        return Path(self.name)


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
