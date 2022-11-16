"""Module for file naming utilities"""
# Standard
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import re
from types import SimpleNamespace
from pathlib import Path
# Installed
from cloudpathlib import AnyPath, CloudPath
# Local
from libera_utils.time import PRINTABLE_TS_FORMAT


SPK_REGEX = re.compile(r"^libera_(?P<spk_object>jpss)"
                       r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                       r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                       r"\.bsp$")

CK_REGEX = re.compile(r"^libera_(?P<ck_object>jpss|azrot|elscan)"
                      r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                      r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                      r"\.bc$")

LIBERA_PRODUCT_REGEX = re.compile(r"^libera"
                                  r"_(?P<instrument>cam|rad)"
                                  r"_(?P<level>l0|l1b|l2)"
                                  r"_(?P<utc_start>[0-9]{8}t[0-9]{6})"
                                  r"_(?P<utc_end>[0-9]{8}t[0-9]{6})"
                                  r"_(?P<version>vM[0-9]*m[0-9]*p[0-9]*)"
                                  r"_(?P<revision>r[0-9]{11})"
                                  r"\.(?P<extension>pkts|h5)$")

MANIFEST_FILE_REGEX = re.compile(r"^libera"
                                 r"_(?P<manifest_type>input|output)"
                                 r"_manifest"
                                 r"_(?P<created_time>[0-9]{8}(?:t[0-9]{6})?)"
                                 r"\.json")


class DataLevel(Enum):
    """Data product level"""
    L0 = "l0"
    L1B = 'l1b'
    L2 = 'l2'


class ManifestType(Enum):
    """Enumerated legal manifest type values"""
    INPUT = 'INPUT'
    input = INPUT
    OUTPUT = 'OUTPUT'
    output = OUTPUT


class AbstractValidFilename(ABC):
    """Composition of a CloudPath/Path instance with some methods to perform
    regex validation on filenames
    """
    _regex: re.Pattern
    _fmt: str

    def __init__(self, *args, **kwargs):
        self.path = AnyPath(*args, **kwargs)

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self.path)})"

    def __eq__(self, other):
        if self.path == other.path and self.filename_parts == other.filename_parts:
            return True
        return False

    @property
    def path(self):
        """Property containing the file path"""
        return self._path

    @path.setter
    def path(self, new_path: str or Path or CloudPath):
        if isinstance(new_path, str):
            new_path = AnyPath(new_path)
        self.regex_match(new_path)  # validates against regex pattern
        self._path: CloudPath or Path = AnyPath(new_path)

    @property
    def filename_parts(self):
        """Property that contains a namespace of filename parts"""
        return self._parse_filename_parts()

    @classmethod
    def from_filename_parts(cls,
                            basepath: str or Path = None,
                            **parts):
        """Create instance from filename parts.

        The part arg names are named according to the regex for the file type.

        Parameters
        ----------
        basepath : str or Path, Optional
            Allows prepending a basepath for local filepaths. Does not work with
            cloud paths because there is no notion of a basepath in cloud storage.
        parts :
            Passed directly to _format_filename_parts

        Returns
        -------
        : cls
        """
        filename = cls._format_filename_parts(**parts)
        if basepath is not None:
            return cls(basepath, filename)
        return cls(filename)

    @classmethod
    @abstractmethod
    def _format_filename_parts(cls, **parts):
        """Format parts into a filename

        Note: When this is implemented by concrete classes, **parts becomes a set of explicitly named arguments"""
        raise NotImplementedError()

    @abstractmethod
    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        # Do stuff to parse the elements of d into a SimpleNamespace
        raise NotImplementedError()

    def regex_match(self, path: str or Path or CloudPath):
        """Parse and validate a given path against class-attribute defined regex

        Returns
        -------
        : dict
        """
        # AnyPath is polymorphic but self.path will always be a CloudPath or Path object with a name attribute.
        match = self._regex.match(path.name)  # pylint: disable=no-member
        if not match:
            raise ValueError(f"Proposed path {path} failed validation against regex pattern {self._regex}")
        return match.groupdict()


class ProductFilename(AbstractValidFilename):
    """Abstract base for filenaming and validation."""

    _regex = LIBERA_PRODUCT_REGEX
    _fmt = "libera_{instrument}_{level}_{utc_start}_{utc_end}_{version}_{revision}.{extension}"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               instrument: str,
                               level: DataLevel,
                               utc_start: datetime,
                               utc_end: datetime,
                               version: str,
                               revision: str,
                               extension: str):
        """Construct a path from filename parts

        Parameters
        ----------
        instrument : str
        level : DataLevel
        utc_start : datetime
            First timestamp in the SPK
        utc_end : datetime
            Last timestamp in the SPK
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        revision : str
            %y%j%H%M%S formatted time when the file was created.
        extension : str
        """
        return cls._fmt.format(instrument=instrument,
                               level=level.value,
                               utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
                               utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT),
                               version=version,
                               revision=revision,
                               extension=extension)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['level'] = DataLevel(d['level'])
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        return SimpleNamespace(**d)


class ManifestFilename(AbstractValidFilename):
    """Class for naming manifest files"""
    _regex = MANIFEST_FILE_REGEX
    _fmt = "libera_{manifest_type}_manifest_{created_time}.json"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               manifest_type: ManifestType,
                               created_time: datetime):
        """Construct a path from filename parts

        Parameters
        ----------
        manifest_type : ManifestType
            Input or output
        created_time : datetime
            Time of manifest creation (writing).
        """
        return cls._fmt.format(manifest_type=manifest_type.value.lower(),
                               created_time=created_time.strftime(PRINTABLE_TS_FORMAT))

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['manifest_type'] = ManifestType(d['manifest_type'].upper())
        d['created_time'] = datetime.strptime(d['created_time'], PRINTABLE_TS_FORMAT)
        return SimpleNamespace(**d)


class EphemerisKernelFilename(AbstractValidFilename):
    """Class to construct, store, and manipulate an SPK filename"""
    _regex = SPK_REGEX
    _fmt = "libera_{spk_object}_{utc_start}_{utc_end}.bsp"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               spk_object: str,
                               utc_start: datetime,
                               utc_end: datetime):
        """Create an instance from a given path

        Parameters
        ----------
        spk_object : str
            Name of object whose ephemeris is represented in this SPK.
        utc_start : datetime
            Start time of data.
        utc_end : datetime
            End time of data.

        Returns
        -------
        : cls
        """
        return cls._fmt.format(spk_object=spk_object,
                               utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
                               utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT))

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        return SimpleNamespace(**d)


class AttitudeKernelFilename(AbstractValidFilename):
    """Class to construct, store, and manipulate an SPK filename"""
    _regex = CK_REGEX
    _fmt = "libera_{ck_object}_{utc_start}_{utc_end}.bc"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               ck_object: str,
                               utc_start: datetime,
                               utc_end: datetime):
        """Create an instance from a given path

        Parameters
        ----------
        ck_object : str
            Name of object whose attitude is represented in this CK.
        utc_start : datetime
            Start time of data.
        utc_end : datetime
            End time of data.

        Returns
        -------
        : cls
        """
        return cls._fmt.format(ck_object=ck_object,
                               utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
                               utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT))

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        return SimpleNamespace(**d)


def get_current_revision():
    """Get the current `r%y%j%H%M%S` string for naming file revisions.

    Returns
    -------
    : str
    """
    return f"r{datetime.utcnow().strftime('%y%j%H%M%S')}"


def format_version(semantic_version: str):
    """Formats a semantic version string X.Y.Z into a filename-compatible string like vMXmYpZ, for Major, minor, patch.

    Parameters
    ----------
    semantic_version : str
        String matching X.Y.Z where X, Y and Z are integers of any length

    Returns
    -------
    : str
    """
    major, minor, patch = semantic_version.split('.')
    return f"vM{major}m{minor}p{patch}"
