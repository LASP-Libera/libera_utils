"""Module for file naming utilities"""

import re
import warnings
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ulid
from cloudpathlib import AnyPath, CloudPath, S3Path

from libera_utils.constants import (
    DataLevel,
    DataProductIdentifier,
    LiberaApid,
    ManifestType,
    ProcessingStepIdentifier,
)
from libera_utils.time import NUMERIC_DOY_TS_FORMAT, PRINTABLE_TS_FORMAT

REVISION_TS_FORMAT = f"R{NUMERIC_DOY_TS_FORMAT}"  # Just adds an r in front


def _ensure_utc_timezone(dt_obj: datetime) -> datetime:
    """Ensure datetime object has UTC timezone info.

    If the datetime is timezone-naive, assume it is in UTC and add timezone info.
    If the datetime is timezone-aware, convert it to UTC.

    Parameters
    ----------
    dt_obj : datetime
        Input datetime object

    Returns
    -------
    : datetime
        Timezone-aware datetime in UTC
    """
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=UTC)
    return dt_obj.astimezone(UTC)


# Type alias for paths returned by AnyPath() constructor
PathType = CloudPath | Path

# L0 filename format determined by EDOS Production Data Set and Construction Record filenaming conventions
LIBERA_L0_REGEX = re.compile(
    r"^(?P<id_char>[PX])"
    r"(?P<scid>[0-9]{3})"
    r"(?P<first_apid>[0-9]{4})"
    # In some cases at least, the last character of the fill field specifies a time (T)
    # or session (S) based product. e.g. VIIRSSCIENCEAT
    r"(?P<fill>.{14})"
    r"(?P<created_time>[0-9]{11})"
    r"(?P<numeric_id>[0-9])"
    r"(?P<file_number>[0-9]{2})"
    r".(?P<extension>PDR|PDS)"
    r"(?P<signal>.XFR)?$"
)

# Get all data levels for the regex
DATA_LEVELS = "|".join([level.value for level in DataLevel])

# Get all data product names
DATA_PRODUCT_NAMES = "|".join([str(dpi) for dpi in DataProductIdentifier])

LIBERA_DATA_PRODUCT_REGEX = re.compile(
    rf"^LIBERA_(?P<data_level>{DATA_LEVELS})"
    rf"_(?P<product_name>{DATA_PRODUCT_NAMES})"
    r"_(?P<version>V[0-9]*-[0-9]*-[0-9]*(RC[0-9])?)"
    r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
    r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
    r"_(?P<revision>R[0-9]{11})"
    r"\.(?P<extension>nc|h5|bsp|bc)$"
)

MANIFEST_FILE_REGEX = re.compile(
    r"^LIBERA"
    r"_(?P<manifest_type>INPUT|OUTPUT)"
    r"_MANIFEST"
    r"_(?P<ulid_code>[0-9A-HJ-NP-TV-Z]{26})"
    r"\.json"
)


class AbstractValidFilename(ABC):
    """Filename class that ensures validity of a filename based on regex pattern.

    Notes
    -----
    - This is an abstract base class that must be inherited by concrete filename classes.
    - This class internally stores a CloudPath or Path object in the `path` property (composition).
    """

    _regex: re.Pattern
    _fmt: str
    _required_parts: tuple
    _path: PathType

    def __init__(self, *args, **kwargs):
        self.path = AnyPath(*args, **kwargs)

    def __str__(self):
        return str(self.path)

    def __eq__(self, other):
        if self.path == other.path and self.filename_parts == other.filename_parts:
            return True
        return False

    @classmethod
    def from_file_path(cls, *args, **kwargs):
        """Factory method to produce an AbstractValidFilename from a valid Libera file path (str or Path)"""
        for CandidateClass in (
            L0Filename,
            LiberaDataProductFilename,
            ManifestFilename,
        ):
            try:
                filename = CandidateClass(*args, **kwargs)
                return filename
            except ValueError:
                continue

        raise ValueError(
            f"Unable to create a valid filename from {args}. Are you sure this is a valid Libera file name?"
        )

    @property
    def path(self) -> PathType:
        """Property containing the file path"""
        return self._path

    @path.setter
    def path(self, new_path: str | PathType):
        if isinstance(new_path, str):
            _new_path: PathType = cast(PathType, AnyPath(new_path))
        else:
            _new_path = new_path
        self.regex_match(_new_path)  # validates against regex pattern
        self._path = _new_path

    @property
    def filename_parts(self):
        """Property that contains a namespace of filename parts"""
        return self._parse_filename_parts()

    @property
    @abstractmethod
    def archive_prefix(self) -> str:
        """Property that contains the generated prefix used for archiving, when applicable"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_filename_parts(cls, *args: Any, **kwargs: Any):
        """Abstract method that must be implemented to provide hinting for required parts"""
        raise NotImplementedError()

    @classmethod
    def _from_filename_parts(
        cls,
        *,  # No positional arguments
        basepath: str | Path | S3Path | None = None,
        **parts: Any,
    ):
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
    def _format_filename_parts(cls, *args: Any, **kwargs: Any):
        """Format parts into a filename

        Note: When this is implemented by concrete classes, *args and **kwargs become specific parameters
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
        _ = self.regex_match(self.path)
        # Do stuff to parse the elements of d into a SimpleNamespace
        raise NotImplementedError()

    def regex_match(self, path: PathType):
        """Parse and validate a given path against class-attribute defined regex

        Parameters
        ----------
        path : Union[Path, CloudPath]
            Path to validate

        Returns
        -------
        : dict
            Match group dict of filename parts
        """
        match = self._regex.match(path.name)
        if not match:
            raise ValueError(f"Proposed path {path} failed validation against regex pattern {self._regex}")
        return match.groupdict()

    def generate_prefixed_path(self, parent_path: str | PathType) -> PathType:
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
            _parent_path = cast(PathType, AnyPath(parent_path))
        else:
            _parent_path = parent_path

        if not _parent_path.is_absolute():
            raise ValueError(
                f"Detected relative parent_path {parent_path} passed to generate_prefixed_path. "
                "The parent_path must be an absolute path. e.g. s3://my-bucket or /starts/with/root."
            )

        return _parent_path / self.archive_prefix / self.path.name


class AbstractDataProductFilename(AbstractValidFilename):
    """Abstract base class for data product filenames.

    This class adds the data product specific requirements that all data products
    must have: a processing step ID and a data product ID.
    For example, an L0Filename or a LiberaDataProductFilename are both AbstractDataProductFilenames.
    """

    @property
    @abstractmethod
    def data_product_id(self) -> DataProductIdentifier:
        """Property that contains the DataProductIdentifier for this file type"""
        raise NotImplementedError()


class L0Filename(AbstractDataProductFilename):
    """Filename validation class for L0 Production Data Set (PDS) files from EDOS."""

    _regex = LIBERA_L0_REGEX
    _fmt = "{id_char}{scid:03}{first_apid:04}{fill:A<14}{created_time}{numeric_id}{file_number:02}.{extension}{signal}"

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """Property that contains the DataProductIdentifier for this file type"""
        if self.filename_parts.file_number == 0:
            return DataProductIdentifier.l0_pds_cr
        apid_enum = LiberaApid(self.filename_parts.first_apid)
        return apid_enum.data_product_id

    @property
    def archive_prefix(self) -> str:
        """Property that contains the generated prefix for L0 archiving"""
        # Generate prefix structure
        l0_file_type = "CR" if self.filename_parts.file_number == 0 else "PDS"  # CR is always PDS file_number 0
        apid = self.filename_parts.first_apid

        # 2023-07-14: This prefix might become too large over the course of the Libera mission
        return f"{l0_file_type}/{apid:0>4}"

    @classmethod
    def from_filename_parts(
        cls,  # noqa pylint: disable=arguments-differ
        *,  # No positional arguments
        id_char: str,
        scid: int,
        first_apid: int,
        fill: str,
        created_time: datetime,
        numeric_id: int,
        file_number: int,
        extension: str,
        signal: str | None = None,
        basepath: str | Path | S3Path | None = None,
    ):
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
        return cls._from_filename_parts(
            basepath=basepath,
            id_char=id_char,
            scid=scid,
            first_apid=first_apid,
            fill=fill,
            created_time=created_time,
            numeric_id=numeric_id,
            file_number=file_number,
            extension=extension,
            signal=signal,
        )

    @classmethod
    def _format_filename_parts(
        cls,  # pylint: disable=arguments-differ
        *,  # No positional arguments
        id_char: str,
        scid: int,
        first_apid: int,
        fill: str,
        created_time: datetime,
        numeric_id: int,
        file_number: int,
        extension: str,
        signal: str | None = None,
    ):
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

        return cls._fmt.format(
            id_char=id_char,
            scid=scid,
            first_apid=first_apid,
            fill=fill,
            created_time=created_time.strftime(NUMERIC_DOY_TS_FORMAT),
            numeric_id=numeric_id,
            file_number=file_number,
            extension=extension,
            signal=signal,
        )

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d["scid"] = int(d["scid"])
        d["first_apid"] = int(d["first_apid"])
        d["numeric_id"] = int(d["numeric_id"])
        d["file_number"] = int(d["file_number"])
        d["created_time"] = datetime.strptime(d["created_time"], NUMERIC_DOY_TS_FORMAT)
        return SimpleNamespace(**d)


class LiberaDataProductFilename(AbstractDataProductFilename):
    """Filename validation class for Libera SDC data products."""

    _regex = LIBERA_DATA_PRODUCT_REGEX
    _fmt = "LIBERA_{data_level}_{product_name}_{version}_{utc_start}_{utc_end}_{revision}.{extension}"

    @property
    def processing_step_id(self) -> ProcessingStepIdentifier | None:
        """Property that contains the ProcessingStepIdentifier that generates this file"""
        return ProcessingStepIdentifier.from_data_product(self.data_product_id)

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """Property that contains the DataProductIdentifier for this file type"""
        return DataProductIdentifier(self.filename_parts.product_name)

    @property
    def archive_prefix(self) -> str:
        """Property that contains the generated prefix for L1B and L2 archiving"""
        # Generate prefix structure
        # <product_type>/<year>/<month>/<day>
        product_name = self.filename_parts.product_name

        applicable_date = self.applicable_date

        return f"{product_name}/{applicable_date.year:0>4}/{applicable_date.month:0>2}/{applicable_date.day:0>2}"

    @property
    def applicable_date(self) -> date:
        """Property that returns the applicable date based on the midpoint of start and end times.

        Issues a warning if the time range covers more than 24 hours.

        Returns
        -------
        : datetime.date
            The date of the midpoint between utc_start and utc_end
        """
        utc_start = self.filename_parts.utc_start
        utc_end = self.filename_parts.utc_end

        # Check if time range covers more than 24 hours and issue warning
        if utc_end - utc_start > timedelta(hours=24):
            warnings.warn("Time range for filename spans more than 24 hours", UserWarning, stacklevel=2)

        # Calculate the midpoint date
        # In all production processing cases, utc_start and utc_end should be roughly midnight on consecutive days
        # The applicable date is considered to be the mean between the two, ignoring hours, minutes, seconds
        t_mean = utc_start + 0.5 * (utc_end - utc_start)
        return t_mean.date()

    @classmethod
    def from_filename_parts(
        cls,
        *,  # No positional arguments
        product_name: str | DataProductIdentifier,
        version: str,
        utc_start: datetime,
        utc_end: datetime,
        data_level: str | DataLevel | None = None,
        revision: datetime = datetime.now(tz=UTC),
        extension: str | None = None,
        basepath: str | Path | S3Path | None = None,
    ):
        """Create instance from filename parts. All keyword arguments other than basepath are required!

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part names are named according to the regex for the file type.

        Parameters
        ----------
        data_level : str | DataLevel | None
            L1B or L2 identifying the level of the data product. Default None will infer the data level from the product name (DataProductIdentifier)
        product_name : str | DataProductIdentifier
            Product type. e.g. CF-RAD for L2 or RAD-4CH for L1B. May contain anything except for underscores.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            First timestamp in the SPK
        utc_end : datetime.datetime
            Last timestamp in the SPK
        revision: datetime.datetime
            Time when the file was created. Default is now in UTC time.
        extension : str | None
            File extension. Default None will infer extension based on product_name.
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : LiberaDataProductFilename
        """
        dpi = DataProductIdentifier(product_name)
        if not extension:
            match dpi:
                case DataProductIdentifier.spice_jpss_spk:
                    # Special case for our only SPK
                    extension = "bsp"
                case _ if dpi.data_level == DataLevel.SPICE:
                    # All other SPICE products are CKs
                    extension = "bc"
                case _:
                    # Everything else is NetCDF4
                    extension = "nc"

        if not data_level:
            data_level = dpi.data_level
        elif dpi.data_level != data_level:
            warnings.warn(
                f"Provided data level {data_level} does not match data level of data product identifier {dpi}:{dpi.data_level}"
            )

        return cls._from_filename_parts(
            basepath=basepath,
            data_level=data_level,
            product_name=product_name,
            version=version,
            utc_start=_ensure_utc_timezone(utc_start),
            utc_end=_ensure_utc_timezone(utc_end),
            revision=_ensure_utc_timezone(revision),
            extension=extension,
        )

    @classmethod
    def _format_filename_parts(
        cls,
        *,  # No positional arguments
        data_level: str,
        product_name: str,
        version: str,
        utc_start: datetime,
        utc_end: datetime,
        revision: datetime,
        extension: str,
    ):
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
        return cls._fmt.format(
            data_level=data_level.upper(),
            product_name=product_name.upper(),
            version=version.upper(),
            utc_start=_ensure_utc_timezone(utc_start).strftime(PRINTABLE_TS_FORMAT),
            utc_end=_ensure_utc_timezone(utc_end).strftime(PRINTABLE_TS_FORMAT),
            revision=_ensure_utc_timezone(revision).strftime(REVISION_TS_FORMAT),
            extension=extension,
        )

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d["utc_start"] = datetime.strptime(d["utc_start"], PRINTABLE_TS_FORMAT).replace(tzinfo=UTC)
        d["utc_end"] = datetime.strptime(d["utc_end"], PRINTABLE_TS_FORMAT).replace(tzinfo=UTC)
        d["revision"] = datetime.strptime(d["revision"], REVISION_TS_FORMAT).replace(tzinfo=UTC)
        return SimpleNamespace(**d)


class ManifestFilename(AbstractValidFilename):
    """Class for naming manifest files"""

    _regex = MANIFEST_FILE_REGEX
    _fmt = "LIBERA_{manifest_type}_MANIFEST_{ulid_code}.json"

    @property
    def archive_prefix(self) -> str:
        """Manifests are not archived like data products, but for convenience and ease of debugging they will be kept
        in the dropbox bucket by input/output and day they were made. This is used by the step function clean up
        function in the CDK.
        # Generate prefix structure
        # <manifest_type>/<year>/<month>/<day>
        """
        manifest_type = self.filename_parts.manifest_type

        applicable_date = self.filename_parts.ulid_code.datetime

        return f"{manifest_type}/{applicable_date.year:0>4}/{applicable_date.month:0>2}/{applicable_date.day:0>2}"

    @classmethod
    def from_filename_parts(
        cls,  # noqa pylint: disable=arguments-differ
        manifest_type: ManifestType,
        ulid_code: ulid.ULID,
        basepath: str | Path | S3Path | None = None,
    ):
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
        return cls._from_filename_parts(basepath=basepath, manifest_type=manifest_type, ulid_code=ulid_code)

    @classmethod
    def _format_filename_parts(
        cls,  # pylint: disable=arguments-differ
        manifest_type: ManifestType,
        ulid_code: ulid.ULID,
    ):
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
        return cls._fmt.format(manifest_type=manifest_type.upper(), ulid_code=ulid_code)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        d = self.regex_match(self.path)
        d["manifest_type"] = ManifestType(d["manifest_type"].upper())
        d["ulid_code"] = ulid.ULID.from_str(d["ulid_code"])
        return SimpleNamespace(**d)


def format_semantic_version(semantic_version: str) -> str:
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
    major, minor, patch = semantic_version.split(".")
    return f"V{major}-{minor}-{patch}".upper()


def get_current_version_str(package_name: str) -> str:
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
