"""Module for file naming utilities"""
# Standard
import re
from datetime import datetime
from pathlib import Path
# Local
from libera_utils.time import PRINTABLE_TS_FORMAT


SPK_REGEX = re.compile(r"^libera_jpss"
                       r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                       r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                       r"\.bsp$")

CK_REGEX = re.compile(r"^libera_(?P<ck_object>jpss|azrot|elscan)"
                      r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                      r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                      r"\.bc$")

LIBERA_PRODUCT_REGEX = re.compile(r"^libera"
                                  r"_(?P<instrument>cam|rad)"
                                  r"_(?P<level>l0|l1a|l1b|l2)"
                                  r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                                  r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                                  r"\.(?P<extension>pkts|h5)$")

# TODO: Make an ABC for KernelFilename for EphemerisKernelFilename and AttitudeKernelFilename
#  to inherit from to clarify the interface.

# TODO: Need a LiberaProductFilename that should probably also inherit from the ABC KernelFilename


class EphemerisKernelFilename:
    """Class to construct, store, and manipulate an SPK filename"""
    _regex = SPK_REGEX
    _fmt_str = "libera_jpss_{utc_start}_{utc_end}.bsp"

    def __init__(self, utc_start: datetime, utc_end: datetime):
        """Constructor

        Parameters
        ----------
        utc_start : datetime
            First timestamp in the SPK
        utc_end : datetime
            Last timestamp in the SPK
        """
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
        utc_start = datetime.strptime(m['utc_start'], PRINTABLE_TS_FORMAT)
        utc_end = datetime.strptime(m['utc_end'], PRINTABLE_TS_FORMAT)
        return cls(utc_start=utc_start, utc_end=utc_end)

    @property
    def name(self):
        """String filename

        Returns
        -------
        : str
            String filename
        """
        return self._fmt_str.format(**{
            'utc_start': self.utc_start.strftime(PRINTABLE_TS_FORMAT),
            'utc_end': self.utc_end.strftime(PRINTABLE_TS_FORMAT)})

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

    def __init__(self, ck_object: str, utc_start: datetime, utc_end: datetime):
        """Constructor

        Parameters
        ----------
        ck_object : str
            Object for which the CK is valid. e.g. a particular spacecraft name or an instrument component.
        utc_start : datetime
            First timestamp in the CK
        utc_end : datetime
            Last timestamp in the CK
        """
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
        utc_start = datetime.strptime(m['utc_start'], PRINTABLE_TS_FORMAT)
        utc_end = datetime.strptime(m['utc_end'], PRINTABLE_TS_FORMAT)
        return cls(ck_object=m['ck_object'], utc_start=utc_start, utc_end=utc_end)

    @property
    def name(self):
        """String filename

        Returns
        -------
        : str
            String filename
        """
        return self._fmt_str.format(
            **{'object': self.ck_object,
               'utc_start': self.utc_start.strftime(PRINTABLE_TS_FORMAT),
               'utc_end': self.utc_end.strftime(PRINTABLE_TS_FORMAT)})

    @property
    def path(self):
        """Path filename

        Returns
        -------
        : Path
            Path representation of filename
        """
        return Path(self.name)
