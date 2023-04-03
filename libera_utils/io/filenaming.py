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
from libera_utils.time import PRINTABLE_TS_FORMAT, EDOS_TS_FORMAT


SPK_REGEX = re.compile(r"^libera_(?P<spk_object>jpss)"
                       r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                       r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                       r"\.bsp$")

CK_REGEX = re.compile(r"^libera_(?P<ck_object>jpss|azrot|elscan)"
                      r"_(?P<utc_start>[0-9]{8}(?:t[0-9]{6})?)"
                      r"_(?P<utc_end>[0-9]{8}(?:t[0-9]{6})?)"
                      r"\.bc$")

# L0 filename format determined by EDOS Production Data Set and Construction Record filenaming conventions
LIBERA_L0_REGEX = re.compile(r"^(?P<id_char>P|X)"
                             r"(?P<scid>[0-9]{3})"
                             r"(?P<first_apid>[0-9]{4})"
                             # In some cases at least, the last character of the fill field specifies a time (T)
                             # or session (S) based product. e.g. VIIRSSCIENCEAT
                             r"(?P<fill>.{14})"
                             r"(?P<created_time>[0-9]{11})"
                             r"(?P<numeric_id>[0-9])"
                             r"(?P<file_number>[0-9]{2})"
                             r".(?P<extension>PDR|PDS)"
                             r"(?P<signal>.XFR)?$")

LIBERA_L1B_REGEX = re.compile(r"^libera_l1b"
                              r"_(?P<instrument>cam|rad)"
                              r"_(?P<utc_start>[0-9]{8}t[0-9]{6})"
                              r"_(?P<utc_end>[0-9]{8}t[0-9]{6})"
                              r"_(?P<version>vM[0-9]*m[0-9]*p[0-9]*)"
                              r"_(?P<revision>r[0-9]{11})"
                              r"\.(?P<extension>nc|h5)$")

LIBERA_L2_REGEX = re.compile(r"^libera_l2"
                             r"_(?P<product_name>[^_]*)"
                             r"_(?P<utc_start>[0-9]{8}t[0-9]{6})"
                             r"_(?P<utc_end>[0-9]{8}t[0-9]{6})"
                             r"_(?P<version>vM[0-9]*m[0-9]*p[0-9]*)"
                             r"_(?P<revision>r[0-9]{11})"
                             r"\.(?P<extension>nc|h5)$")

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


class L0Filename(AbstractValidFilename):
    """Filename validation class for L0 files from EDOS."""

    _regex = LIBERA_L0_REGEX
    _fmt = "{id_char}{scid:03}{first_apid:04}{fill:A<14}{created_time}{numeric_id}{file_number:02}.{extension}{signal}"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               id_char: str,
                               scid: int,
                               first_apid: int,
                               fill: str,
                               created_time: datetime,
                               numeric_id: int,
                               file_number: int,
                               extension: str,
                               signal: str = None):
        """Construct a path from filename parts

        Parameters
        ----------
        id_char : str
            Either P (for PDS files, Construction Records) or X (for Delivery Records)
        scid : int
            Spacecraft ID
        first_apid : int
            First APID in the file
        fill : str
            Custom string up to 14 characters long
        created_time : datetime
            Creation time of the file
        numeric_id : int
            Data set ID, 0-9, one digit
        file_number : str
            File number within the data set. Construction records are always file number zero.
        extension : str
            File name extension. Either PDR or PDS
        signal : str or None
            Optional signal suffix. Always '.XFR'

        Returns
        -------
        : str
            Formatted filename
        """
        signal = signal if signal else ""

        return cls._fmt.format(id_char=id_char,
                               scid=scid,
                               first_apid=first_apid,
                               fill=fill,
                               created_time=created_time.strftime(EDOS_TS_FORMAT),
                               numeric_id=numeric_id,
                               file_number=file_number,
                               extension=extension,
                               signal=signal)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['scid'] = int(d['scid'])
        d['first_apid'] = int(d['first_apid'])
        d['numeric_id'] = int(d['numeric_id'])
        d['file_number'] = int(d['file_number'])
        d['created_time'] = datetime.strptime(d['created_time'], EDOS_TS_FORMAT)
        return SimpleNamespace(**d)


class L1bFilename(AbstractValidFilename):
    """Filename validation class for L1b products"""

    _regex = LIBERA_L1B_REGEX
    _fmt = "libera_l1b_{instrument}_{utc_start}_{utc_end}_{version}_{revision}.{extension}"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               instrument: str,
                               utc_start: datetime,
                               utc_end: datetime,
                               version: str,
                               revision: str,
                               extension: str):
        """Construct a path from filename parts

        Parameters
        ----------
        instrument : str
            Libera instrument, cam or rad
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
            File extension (.nc or .h5)

        Returns
        -------
        : str
            Formatted filename
        """
        return cls._fmt.format(instrument=instrument,
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
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        return SimpleNamespace(**d)


class L2Filename(AbstractValidFilename):
    """Filename validation class for L2 data products."""

    _regex = LIBERA_L2_REGEX
    _fmt = "libera_l2_{product_name}_{utc_start}_{utc_end}_{version}_{revision}.{extension}"

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               product_name: str,
                               utc_start: datetime,
                               utc_end: datetime,
                               version: str,
                               revision: str,
                               extension: str):
        """Construct a path from filename parts

        Parameters
        ----------
        product_name : str
            L2 product type. e.g. cloud-fraction. May contain anything except for underscores.
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
            File extension (.nc or .h5)

        Returns
        -------
        : str
            Formatted filename
        """
        return cls._fmt.format(product_name=product_name,
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

        Returns
        -------
        : str
            Formatted filename
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
