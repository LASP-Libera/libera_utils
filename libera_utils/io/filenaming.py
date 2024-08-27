"""Module for file naming utilities"""
# Standard
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
import re
from types import SimpleNamespace
from typing import Union, Optional, Any
from pathlib import Path
# Installed
from cloudpathlib import AnyPath, S3Path
import ulid
# Local
from libera_utils.time import PRINTABLE_TS_FORMAT, NUMERIC_DOY_TS_FORMAT

REVISION_TS_FORMAT = f"R{NUMERIC_DOY_TS_FORMAT}"  # Just adds an r in front

SPK_REGEX = re.compile(r"^LIBERA_(?P<spk_object>JPSS)"
                       r"_(?P<version>V[0-9]*-[0-9]*-[0-9]*(RC[0-9])?)"
                       r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
                       r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
                       r"_(?P<revision>R[0-9]{11})"
                       r"\.bsp$")

CK_REGEX = re.compile(r"^LIBERA_(?P<ck_object>JPSS|AZROT|ELSCAN)"
                      r"_(?P<version>V[0-9]*-[0-9]*-[0-9]*(RC[0-9])?)"
                      r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
                      r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
                      r"_(?P<revision>R[0-9]{11})"
                      r"\.bc$")

# L0 filename format determined by EDOS Production Data Set and Construction Record filenaming conventions
LIBERA_L0_REGEX = re.compile(r"^(?P<id_char>[PX])"
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

LIBERA_DATA_PRODUCT_REGEX = re.compile(r"^LIBERA_(?P<data_level>L1B|L2)"
                                       r"_(?P<product_name>[^_]*)"
                                       r"_(?P<version>V[0-9]*-[0-9]*-[0-9]*(RC[0-9])?)"
                                       r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
                                       r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
                                       r"_(?P<revision>R[0-9]{11})"
                                       r"\.(?P<extension>nc|h5)$")

MANIFEST_FILE_REGEX = re.compile(r"^LIBERA"
                                 r"_(?P<manifest_type>INPUT|OUTPUT)"
                                 r"_MANIFEST"
                                 r"_(?P<ulid_code>[0-9A-HJ-NP-TV-Z]{26})"
                                 r"\.json")


class DataLevel(Enum):
    """Data product level"""
    L0 = "L0"
    SPICE = "SPICE"
    CAL = "CAL"
    L1B = 'L1B'
    L2 = 'L2'


class ManifestType(Enum):
    """Enumerated legal manifest type values"""
    INPUT = 'INPUT'
    input = INPUT
    OUTPUT = 'OUTPUT'
    output = OUTPUT


class AnyFilename:
    """Polymorphic class for creating a Filename object"""

    def __new__(cls, *args, **kwargs) -> 'AbstractValidFilename':
        for CandidateClass in (L0Filename, AttitudeKernelFilename,
                               EphemerisKernelFilename, LiberaDataProductFilename,
                               ManifestFilename):
            try:
                filename = CandidateClass(*args, **kwargs)
                return filename
            except ValueError:
                continue

        raise ValueError(f"Unable to create a valid filename from {args}. "
                         "Are you sure this is a valid Libera file name?")


class AbstractValidFilename(ABC):
    """Composition of a CloudPath/Path instance with some methods to perform
    regex validation on filenames
    """
    _regex: re.Pattern
    _fmt: str
    _required_parts: tuple

    def __init__(self, *args, **kwargs):
        self.path = AnyPath(*args, **kwargs)

    def __str__(self):
        return str(self.path)

    def __eq__(self, other):
        if self.path == other.path and self.filename_parts == other.filename_parts:
            return True
        return False

    @property
    def path(self) -> Union[Path, S3Path]:
        """Property containing the file path"""
        return self._path

    @path.setter
    def path(self, new_path: Union[str, Path, S3Path]):
        if isinstance(new_path, str):
            new_path = AnyPath(new_path)
        self.regex_match(new_path)  # validates against regex pattern
        self._path: Union[Path, S3Path] = AnyPath(new_path)

    @property
    def filename_parts(self):
        """Property that contains a namespace of filename parts"""
        return self._parse_filename_parts()

    @property
    @abstractmethod
    def archive_prefix(self):
        """Property that contains the generated prefix used for archiving, when applicable"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_filename_parts(cls,
                            *args: Any,  # Required filename parts as defined by child class
                            basepath: Union[str, Path, S3Path] = None,
                            **kwargs: Any  # Optional filename parts or options as defined by child class
                            ):
        """Abstract method that must be implemented to provide hinting for required parts"""
        raise NotImplementedError()

    @classmethod
    def _from_filename_parts(cls,
                             *,  # No positional arguments
                             basepath: Union[str, Path, S3Path] = None,
                             **parts: Any):
        """Create instance from filename parts.

        The part kwarg names are named according to the regex for the file type.

        Parameters
        ----------
        basepath : Union[str, Path, S3Path], Optional
            Allows prepending a basepath or prefix.
        parts : Any
            Passed directly to _format_filename_parts. This is a dict of variable kwargs that will differ in each
            filename class based on the required parts for that particular filename type.

        Returns
        -------
        : AbstractValidFilename
        """
        filename = cls._format_filename_parts(**parts)
        if basepath is not None:
            return cls(AnyPath(basepath) / filename)
        return cls(filename)

    @classmethod
    @abstractmethod
    def _format_filename_parts(cls, **parts):
        """Format parts into a filename

        Note: When this is implemented by concrete classes, **parts becomes a set of explicitly named arguments
        """
        raise NotImplementedError()

    @abstractmethod
    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        # Do stuff to parse the elements of d into a SimpleNamespace
        raise NotImplementedError()

    @staticmethod
    def _calculate_applicable_time(start: datetime, end: datetime):
        """Based on the start time and end time of a file, returns the applicable time (date)

        Parameters
        ----------
        start : datetime.datetime
            Start of the applicable time range
        end : datetime.datetime
            End of the applicable time range

        Returns
        -------
        : datetime.date
            The date of the mean time between start and end
        """
        # In all production processing cases, utc_start and utc_end should be midnight on consecutive days
        # The applicable date is considered to be the mean between the two, ignoring hours, minutes, seconds
        t_0 = start
        t_1 = end
        t_mean = t_0 + 0.5 * (t_1 - t_0)
        applicable_date = datetime.date(t_mean)
        return applicable_date

    def regex_match(self, path: Union[str, Path, S3Path]):
        """Parse and validate a given path against class-attribute defined regex

        Returns
        -------
        : dict
            Match group dict of filename parts
        """
        # AnyPath is polymorphic but self.path will always be a CloudPath or Path object with a name attribute.
        match = self._regex.match(path.name)  # pylint: disable=no-member
        if not match:
            raise ValueError(f"Proposed path {path} failed validation against regex pattern {self._regex}")
        return match.groupdict()

    def generate_prefixed_path(self, parent_path: Union[str, Path, S3Path]) -> Union[Path, S3Path]:
        """Generates an absolute path of the form {parent_path}/{prefix_structure}/{file_basename}
        The parent_path can be an S3 bucket or an absolute local filepath (must start with /)

        Parameters
        ----------
        parent_path : Union[str, Path, S3Path]
            Absolute path to the parent directory or S3 bucket prefix. The generated path prefix is appended to the
            parent path and followed by the file basename.

        Returns
        -------
        : pathlib.Path or cloudpathlib.s3.s3path.S3Path
        """
        if isinstance(parent_path, str):
            parent_path = AnyPath(parent_path)

        if not parent_path.is_absolute():
            raise ValueError(f"Detected relative parent_path {parent_path} passed to generate_prefixed_path. "
                             "The parent_path must be an absolute path. e.g. s3://my-bucket or /starts/with/root.")

        return parent_path / self.archive_prefix / self.path.name


class L0Filename(AbstractValidFilename):
    """Filename validation class for L0 files from EDOS."""

    _regex = LIBERA_L0_REGEX
    _fmt = "{id_char}{scid:03}{first_apid:04}{fill:A<14}{created_time}{numeric_id}{file_number:02}.{extension}{signal}"

    @property
    def archive_prefix(self):
        """Property that contains the generated prefix for L0 archiving"""
        # Generate prefix structure
        l0_file_type = "CR" if self.filename_parts.file_number == 0 else "PDS"  # CR is always PDS file_number 0
        apid = self.filename_parts.first_apid

        # 2023-07-14: This prefix might become too large over the course of the Libera mission
        return f"{l0_file_type}/{apid:0>4}"

    @classmethod
    def from_filename_parts(cls,  # noqa pylint: disable=arguments-differ
                            id_char: str,
                            scid: int,
                            first_apid: int,
                            fill: str,
                            created_time: datetime,
                            numeric_id: int,
                            file_number: int,
                            extension: str,
                            signal: Optional[str] = None,
                            basepath: Optional[Union[str, Path, S3Path]] = None):
        """Create instance from filename parts

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part names are named according to the regex for the file type.

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
        created_time : datetime.datetime
            Creation time of the file
        numeric_id : int
            Data set ID, 0-9, one digit
        file_number : str
            File number within the data set. Construction records are always file number zero.
        extension : str
            File name extension. Either PDR or PDS
        signal : Optional[str]
            Optional signal suffix. Always '.XFR'
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : L0Filename
        """
        return cls._from_filename_parts(basepath=basepath,
                                        id_char=id_char,
                                        scid=scid,
                                        first_apid=first_apid,
                                        fill=fill,
                                        created_time=created_time,
                                        numeric_id=numeric_id,
                                        file_number=file_number,
                                        extension=extension,
                                        signal=signal)

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
                               signal: Optional[str] = None):
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
        created_time : datetime.datetime
            Creation time of the file
        numeric_id : int
            Data set ID, 0-9, one digit
        file_number : str
            File number within the data set. Construction records are always file number zero.
        extension : str
            File name extension. Either PDR or PDS
        signal : Optional[str], Optional
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
                               created_time=created_time.strftime(NUMERIC_DOY_TS_FORMAT),
                               numeric_id=numeric_id,
                               file_number=file_number,
                               extension=extension,
                               signal=signal)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['scid'] = int(d['scid'])
        d['first_apid'] = int(d['first_apid'])
        d['numeric_id'] = int(d['numeric_id'])
        d['file_number'] = int(d['file_number'])
        d['created_time'] = datetime.strptime(d['created_time'], NUMERIC_DOY_TS_FORMAT)
        return SimpleNamespace(**d)


class LiberaDataProductFilename(AbstractValidFilename):
    """Filename validation class for L1B and L2 science products"""

    _regex = LIBERA_DATA_PRODUCT_REGEX
    _fmt = "LIBERA_{data_level}_{product_name}_{version}_{utc_start}_{utc_end}_{revision}.{extension}"

    @property
    def archive_prefix(self):
        """Property that contains the generated prefix for L1B and L2 archiving"""
        # Generate prefix structure
        # <product_type>/<year>/<month>/<day>
        product_name = self.filename_parts.product_name

        applicable_date = self._calculate_applicable_time(self.filename_parts.utc_start, self.filename_parts.utc_end)

        return f"{product_name}/{applicable_date.year:0>4}/{applicable_date.month:0>2}/{applicable_date.day:0>2}"

    @classmethod
    def from_filename_parts(cls,  # noqa pylint: disable=arguments-differ
                            data_level: str,
                            product_name: str,
                            version: str,
                            utc_start: datetime,
                            utc_end: datetime,
                            revision: datetime,
                            extension: str = 'nc',
                            basepath: Optional[Union[str, Path, S3Path]] = None):
        """Create instance from filename parts. All keyword arguments other than basepath are required!

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part names are named according to the regex for the file type.

        Parameters
        ----------
        data_level : str
            L1B or L2 identifying the level of the data product
        product_name : str
            Product type. e.g. cloud-fraction for L2 or cam for L1B. May contain anything except for underscores.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            First timestamp in the SPK
        utc_end : datetime.datetime
            Last timestamp in the SPK
        revision: datetime.datetime
            Time when the file was created.
        extension : str
            File extension (.nc or .h5)
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : LiberaDataProductFilename
        """
        return cls._from_filename_parts(basepath=basepath,
                                        data_level=data_level,
                                        product_name=product_name,
                                        version=version,
                                        utc_start=utc_start,
                                        utc_end=utc_end,
                                        revision=revision,
                                        extension=extension)

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               data_level: str,
                               product_name: str,
                               version: str,
                               utc_start: datetime,
                               utc_end: datetime,
                               revision: datetime,
                               extension: str):
        """Construct a path from filename parts

        Parameters
        ----------
        data_level : str
            L1B or L2
        product_name : str
            Libera instrument, cam or rad for L1B and cloud-fraction etc. for L2. May contain anything except
            for underscores.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            First timestamp in the SPK
        utc_end : datetime.datetime
            Last timestamp in the SPK
        revision: datetime.datetime
            Time when the file was created.
        extension : str
            File extension (.nc or .h5)

        Returns
        -------
        : str
            Formatted filename
        """
        return cls._fmt.format(data_level=data_level.upper(),
                               product_name=product_name.upper(),
                               version=version.upper(),
                               utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
                               utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT),
                               revision=revision.strftime(REVISION_TS_FORMAT),
                               extension=extension)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        d['revision'] = datetime.strptime(d['revision'], REVISION_TS_FORMAT)
        return SimpleNamespace(**d)


class ManifestFilename(AbstractValidFilename):
    """Class for naming manifest files"""

    _regex = MANIFEST_FILE_REGEX
    _fmt = "LIBERA_{manifest_type}_MANIFEST_{ulid_code}.json"

    @property
    def archive_prefix(self):
        """Manifests are not archived like data products, but for convenience and ease of debugging they will be kept
        in the dropbox bucket by input/output and day they were made. This is used by the step function clean up
        function in the CDK.
        # Generate prefix structure
        # <manifest_type>/<year>/<month>/<day>
        """
        manifest_type = self.filename_parts.manifest_type.value

        applicable_date = self.filename_parts.ulid_code.datetime

        return f"{manifest_type}/{applicable_date.year:0>4}/{applicable_date.month:0>2}/{applicable_date.day:0>2}"

    @classmethod
    def from_filename_parts(cls,  # noqa pylint: disable=arguments-differ
                            manifest_type: ManifestType,
                            ulid_code: ulid.ULID,
                            basepath: Union[str, Path, S3Path] = None):
        """Create instance from filename parts.

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part names are named according to the regex for the file type.

        Parameters
        ----------
        manifest_type : ManifestType
            Input or output
        ulid_code : ulid.ULID
            ULID code for use in filename parts
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : ManifestFilename
        """
        return cls._from_filename_parts(basepath=basepath,
                                        manifest_type=manifest_type,
                                        ulid_code=ulid_code)

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               manifest_type: ManifestType,
                               ulid_code: ulid.ULID):
        """Construct a path from filename parts

        Parameters
        ----------
        manifest_type : ManifestType
            Input or output
        ulid_code : ulid.ULID
            ULID code for use in filename parts

        Returns
        -------
        : str
            Formatted filename
        """
        return cls._fmt.format(manifest_type=manifest_type.value.upper(),
                               ulid_code=ulid_code)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['manifest_type'] = ManifestType(d['manifest_type'].upper())
        d['ulid_code'] = ulid.ULID.from_str(d['ulid_code'])
        return SimpleNamespace(**d)


class EphemerisKernelFilename(AbstractValidFilename):
    """Class to construct, store, and manipulate an SPK filename"""

    _regex = SPK_REGEX
    _fmt = "LIBERA_{spk_object}_{version}_{utc_start}_{utc_end}_{revision}.bsp"

    @property
    def archive_prefix(self):
        """Property that contains the generated prefix for SPICE archiving"""
        # Generate prefix structure
        # <type>/<year>/<month>/<day>
        spk_object = self.filename_parts.spk_object

        applicable_date = self._calculate_applicable_time(self.filename_parts.utc_start, self.filename_parts.utc_end)

        return f"{spk_object}/{applicable_date.year:0>4}/{applicable_date.month:0>2}/{applicable_date.day:0>2}"

    @classmethod
    def from_filename_parts(cls,  # noqa pylint: disable=arguments-differ
                            spk_object: str,
                            version: str,
                            utc_start: datetime,
                            utc_end: datetime,
                            revision: datetime,
                            basepath: Optional[Union[str, Path, S3Path]] = None):
        """Create instance from filename parts.

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part arg names are named according to the regex for the file type.

        Parameters
        ----------
        spk_object : str
            Name of object whose attitude is represented in this SPK.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            Start time of data.
        utc_end : datetime.datetime
            End time of data.
        revision: datetime.datetime
            When the file was last revised.
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : EphemerisKernelFilename
        """
        return cls._from_filename_parts(basepath=basepath,
                                        spk_object=spk_object,
                                        version=version,
                                        utc_start=utc_start,
                                        utc_end=utc_end,
                                        revision=revision)

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               spk_object: str,
                               version: str,
                               utc_start: datetime,
                               utc_end: datetime,
                               revision: datetime):
        """Format filename parts as a string

        Parameters
        ----------
        spk_object : str
            Name of object whose ephemeris is represented in this SPK.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            Start time of data.
        utc_end : datetime.datetime
            End time of data.
        revision: datetime.datetime
            Time when the file was last revised

        Returns
        -------
        : str
        """
        return cls._fmt.format(spk_object=spk_object.upper(),
                               version=version.upper(),
                               utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
                               utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT),
                               revision=revision.strftime(REVISION_TS_FORMAT))

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        d['revision'] = datetime.strptime(d['revision'], REVISION_TS_FORMAT)
        return SimpleNamespace(**d)


class AttitudeKernelFilename(AbstractValidFilename):
    """Class to construct, store, and manipulate an SPK filename"""

    _regex = CK_REGEX
    _fmt = "LIBERA_{ck_object}_{version}_{utc_start}_{utc_end}_{revision}.bc"

    @property
    def archive_prefix(self):
        """Property that contains the generated prefix for SPICE archiving"""
        # Generate prefix structure
        # <type>/<year>/<month>/<day>
        ck_object = self.filename_parts.ck_object

        applicable_date = self._calculate_applicable_time(self.filename_parts.utc_start, self.filename_parts.utc_end)

        return f"{ck_object}/{applicable_date.year:0>4}/{applicable_date.month:0>2}/{applicable_date.day:0>2}"

    @classmethod
    def from_filename_parts(cls,  # noqa pylint: disable=arguments-differ
                            ck_object: str,
                            version: str,
                            utc_start: datetime,
                            utc_end: datetime,
                            revision: datetime,
                            basepath: Optional[Union[str, Path, S3Path]] = None):
        """Create instance from filename parts.

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part arg names are named according to the regex for the file type.

        Parameters
        ----------
        ck_object : str
            Name of object whose attitude is represented in this CK.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            Start time of data.
        utc_end : datetime.datetime
            End time of data.
        revision: datetime.datetime
            When the file was last revised.
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : AttitudeKernelFilename
        """
        return cls._from_filename_parts(basepath=basepath,
                                        ck_object=ck_object,
                                        version=version,
                                        utc_start=utc_start,
                                        utc_end=utc_end,
                                        revision=revision)

    @classmethod
    def _format_filename_parts(cls,  # pylint: disable=arguments-differ
                               ck_object: str,
                               version: str,
                               utc_start: datetime,
                               utc_end: datetime,
                               revision: datetime):
        """Format filename parts as a string

        Parameters
        ----------
        ck_object : str
            Name of object whose attitude is represented in this CK.
        utc_start : datetime.datetime
            Start time of data.
        utc_end : datetime.datetime
            End time of data.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        revision: datetime.datetime
            When the file was last revised.

        Returns
        -------
        : str
        """
        return cls._fmt.format(ck_object=ck_object.upper(),
                               version=version.upper(),
                               utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
                               utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT),
                               revision=revision.strftime(REVISION_TS_FORMAT))

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d['utc_start'] = datetime.strptime(d['utc_start'], PRINTABLE_TS_FORMAT)
        d['utc_end'] = datetime.strptime(d['utc_end'], PRINTABLE_TS_FORMAT)
        d['revision'] = datetime.strptime(d['revision'], REVISION_TS_FORMAT)
        return SimpleNamespace(**d)


def get_current_revision_str():
    """Get the current `r%y%j%H%M%S` string for filename revisions.

    Returns
    -------
    : str
        Current (now) revision string.
    """
    return datetime.now(timezone.utc).strftime(REVISION_TS_FORMAT)


def format_semantic_version(semantic_version: str):
    """Formats a semantic version string X.Y.Z into a filename-compatible string like VX-Y-Z, for X = major version,
    Y = minor version, Z = patch.

    Result is uppercase.
    Release candidate suffixes are allowed as no strict checking is done on the contents of X, Y, or Z.
    e.g. 1.2.3rc1 becomes V1-2-3RC1

    Parameters
    ----------
    semantic_version : str
        String matching X.Y.Z where X, Y and Z are integers of any length

    Returns
    -------
    : str
    """
    major, minor, patch = semantic_version.split('.')
    return f"V{major}-{minor}-{patch}".upper()


def get_current_version_str(package_name: str):
    """Retrieve the current version of a (algorithm) package and format it for inclusion in a filename

    Parameters
    ----------
    package_name : str
        Package for which to retrieve a version string. This should be your algorithm package and it must use a
        semantic versioning scheme, configured in project metadata.

    Returns
    -------
    : str
        Version string in format vM1m2p3
    """
    semver = metadata.version(package_name)
    return format_semantic_version(semver)
