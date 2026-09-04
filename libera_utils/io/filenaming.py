"""Module for file naming utilities

Libera data product filenames follow the convention::

    LIBERA_{data_level}_{product_name}_{version}_{applicable_date}_{utc_start}_{utc_end}_{revision}.{extension}

e.g. ``LIBERA_L1B_RAD-4CH_V1-2-3_2027-01-02_20270102T112233_20270102T122233_01J8ZQ3K9X7M2N4P6Q8R0S1T2V.nc``

- ``data_level`` and ``product_name`` are the string values of :class:`~libera_utils.constants.DataLevel` and
  :class:`~libera_utils.constants.DataProductIdentifier`.
- ``version`` is the algorithm semantic version in ``VM-m-p[RCn]`` form.
- ``applicable_date`` is the ``YYYY-MM-DD`` date the product applies to (by default the midpoint of the time range).
- ``utc_start`` / ``utc_end`` bound the data in the file, formatted ``YYYYMMDDTHHMMSS``.
- ``revision`` is a ULID that uniquely identifies this production of the file; its embedded timestamp is the
  creation time.

All filename classes are hashable and orderable. ``LiberaDataProductFilename`` sorts by data level, product name,
numeric algorithm version, applicable date, and then revision, so ``max(filenames)`` yields the newest file.
"""

import functools
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
from packaging.version import InvalidVersion, Version

from libera_utils.constants import (
    DataLevel,
    DataProductIdentifier,
    LiberaApid,
    ManifestType,
    ProcessingStepIdentifier,
)
from libera_utils.time import NUMERIC_DOY_TS_FORMAT, PRINTABLE_TS_FORMAT

APPLICABLE_DATE_FORMAT = "%Y-%m-%d"


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

# A ULID rendered in Crockford base32 (26 characters, excludes I, L, O, U)
ULID_REGEX_FRAGMENT = r"[0-9A-HJ-NP-TV-Z]{26}"

# Libera filename version format VM-m-p with an optional release candidate suffix RCn
LIBERA_SEM_VER_REGEX_FRAGMENT = r"V[0-9]+-[0-9]+-[0-9]+(?:RC[0-9]+)?"
LIBERA_SEM_VER_REGEX = re.compile(LIBERA_SEM_VER_REGEX_FRAGMENT)

# Everything in a data product filename except the extension. Shared by the data file and its UMM-G metadata file.
_LIBERA_DATA_PRODUCT_BODY_REGEX = (
    rf"^LIBERA_(?P<data_level>{DATA_LEVELS})"
    rf"_(?P<product_name>{DATA_PRODUCT_NAMES})"
    rf"_(?P<version>{LIBERA_SEM_VER_REGEX_FRAGMENT})"
    r"_(?P<applicable_date>[0-9]{4}-[0-9]{2}-[0-9]{2})"
    r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
    r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
    rf"_(?P<revision>{ULID_REGEX_FRAGMENT})"
)

LIBERA_DATA_PRODUCT_REGEX = re.compile(_LIBERA_DATA_PRODUCT_BODY_REGEX + r"\.(?P<extension>nc|h5|bsp|bc)$")

LIBERA_METADATA_PRODUCT_REGEX = re.compile(_LIBERA_DATA_PRODUCT_BODY_REGEX + r"\.(?P<extension>cmr\.json)$")

MANIFEST_FILE_REGEX = re.compile(
    r"^LIBERA"
    r"_(?P<manifest_type>INPUT|OUTPUT)"
    r"_MANIFEST"
    rf"_(?P<ulid_code>{ULID_REGEX_FRAGMENT})"
    r"\.json"
)


@functools.total_ordering
class AbstractValidFilename(ABC):
    """Filename class that ensures validity of a filename based on regex pattern.

    Notes
    -----
    - This is an abstract base class that must be inherited by concrete filename classes.
    - This class internally stores a CloudPath or Path object in the `path` property (composition).
    - Instances are hashable and compare equal when their paths are equal (as given; a relative and an absolute
      spelling of the same file are distinct). Reassigning `path` changes the hash, so do not mutate an instance
      that is already a member of a set or a dict key.
    - Instances of the same concrete class are totally ordered by `_sort_key`. Comparing instances of different
      concrete classes raises TypeError.
    - `filename_parts` is parsed once per path assignment and cached; the returned namespace is shared, so callers
      must not mutate it.
    """

    _regex: re.Pattern
    _fmt: str
    _path: PathType
    _filename_parts: SimpleNamespace

    def __init__(self, *args, **kwargs):
        self.path = AnyPath(*args, **kwargs)

    def __str__(self):
        return str(self.path)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AbstractValidFilename):
            return NotImplemented
        return self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)

    def __lt__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    @abstractmethod
    def _sort_key(self) -> tuple:
        """Key tuple that defines the ordering of instances of this class.

        Implementations must end the tuple with ``str(self.path)`` so that the ordering is total and consistent
        with equality (two instances compare equal only when their paths are equal).

        Returns
        -------
        : tuple
            Comparable key
        """
        raise NotImplementedError()

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
        self._filename_parts = self._parse_filename_parts()

    @property
    def filename_parts(self) -> SimpleNamespace:
        """Property that contains a namespace of filename parts, parsed when the path was last set"""
        return self._filename_parts

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

    def _sort_key(self) -> tuple:
        """Order L0 files by APID, then creation time, then file number (construction record first)"""
        parts = self.filename_parts
        return (parts.first_apid, parts.created_time, parts.file_number, str(self.path))

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
        cls,
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
    _fmt = "LIBERA_{data_level}_{product_name}_{version}_{applicable_date}_{utc_start}_{utc_end}_{revision}.{extension}"

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
    def ummg_metadata_filename(self) -> Path | S3Path:
        """Property that returns the corresponding UMM-G metadata filename for this data product file.

        Returns
        -------
        : Path | S3Path
           Same base filename with a Common Metadata Repository(CMR) JSON extension.
        """
        # TODO[LIBSDC-731]: Decide where to write metadata files (currently this sets it at the same place as the data file)
        ummg_filename = self.path.with_suffix(".cmr.json")

        if not LIBERA_METADATA_PRODUCT_REGEX.match(ummg_filename.name):
            raise ValueError(
                f"Proposed path {ummg_filename} failed validation against regex pattern {LIBERA_METADATA_PRODUCT_REGEX}"
            )

        if ummg_filename.name.rsplit(".")[0] != self.path.name.rsplit(".")[0]:
            raise ValueError(
                f"Proposed path {ummg_filename} does not match its data file path {self.path}. They must have the same name with different extensions"
            )

        return ummg_filename

    @property
    def applicable_date(self) -> date:
        """Property that returns the applicable date carried in the filename.

        Returns
        -------
        : datetime.date
            The YYYY-MM-DD applicable date part of the filename
        """
        return self.filename_parts.applicable_date

    def _sort_key(self) -> tuple:
        """Order data products by level, product, numeric version, applicable date, then revision.

        The ULID revision is time-ordered to the millisecond; two files produced within the same millisecond sort
        arbitrarily relative to each other. The time range and path are trailing tie-breakers only.
        """
        parts = self.filename_parts
        return (
            parts.data_level,
            parts.product_name,
            semantic_version_from_format(parts.version),
            parts.applicable_date,
            parts.revision,
            parts.utc_start,
            parts.utc_end,
            str(self.path),
        )

    @classmethod
    def from_filename_parts(
        cls,
        *,  # No positional arguments
        product_name: str | DataProductIdentifier,
        version: str,
        utc_start: datetime,
        utc_end: datetime,
        data_level: str | DataLevel | None = None,
        applicable_date: date | None = None,
        revision: ulid.ULID | None = None,
        extension: str | None = None,
        basepath: str | Path | S3Path | None = None,
    ):
        """Create instance from filename parts.

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part names are named according to the regex for the file type.

        Parameters
        ----------
        product_name : str | DataProductIdentifier
            Product type. e.g. CF-CAM for L2 or RAD-4CH for L1B. May contain anything except for underscores.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            First timestamp of the data in the file
        utc_end : datetime.datetime
            Last timestamp of the data in the file
        data_level : str | DataLevel | None
            Level of the data product. Default None will infer the data level from the product name
            (DataProductIdentifier). If provided, it must agree with the product name.
        applicable_date : datetime.date | None
            Date the product applies to. Default None uses the date of the midpoint of the time range (see
            `midpoint_applicable_date`). An explicit date outside the days covered by the time range is allowed but
            issues a UserWarning because it is likely a mistake.
        revision : ulid.ULID | None
            ULID that uniquely identifies this production of the file. Default None generates a new ULID, whose
            embedded timestamp is the creation time.
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

        if data_level and dpi.data_level != data_level:
            raise ValueError(
                f"Provided data level {data_level} does not match data level of data product identifier {dpi}:{dpi.data_level}"
            )

        data_level = dpi.data_level

        utc_start = _ensure_utc_timezone(utc_start)
        utc_end = _ensure_utc_timezone(utc_end)

        if applicable_date is None:
            applicable_date = midpoint_applicable_date(utc_start, utc_end)
        else:
            if isinstance(applicable_date, datetime):
                # datetime is a subclass of date; reduce it so it formats as a date only
                applicable_date = _ensure_utc_timezone(applicable_date).date()
            if not utc_start.date() <= applicable_date <= utc_end.date():
                warnings.warn(
                    f"Applicable date {applicable_date.isoformat()} lies outside the filename time range "
                    f"[{utc_start.isoformat()}, {utc_end.isoformat()}]; this is likely a mistake",
                    UserWarning,
                    stacklevel=2,
                )

        if revision is None:
            revision = ulid.ULID()

        return cls._from_filename_parts(
            basepath=basepath,
            data_level=data_level,
            product_name=product_name,
            version=version,
            applicable_date=applicable_date,
            utc_start=utc_start,
            utc_end=utc_end,
            revision=revision,
            extension=extension,
        )

    @classmethod
    def _format_filename_parts(
        cls,
        *,  # No positional arguments
        data_level: str,
        product_name: str,
        version: str,
        applicable_date: date,
        utc_start: datetime,
        utc_end: datetime,
        revision: ulid.ULID,
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
        applicable_date : datetime.date
            Date the product applies to
        utc_start : datetime.datetime
            First timestamp of the data in the file
        utc_end : datetime.datetime
            Last timestamp of the data in the file
        revision : ulid.ULID
            ULID identifying this production of the file
        extension : str
            File extension (.nc or .h5)

        Returns
        -------
        : str
            Formatted filename
        """
        if not check_version_number_format(version):
            version = format_from_semantic_version(version)

        return cls._fmt.format(
            data_level=data_level.upper(),
            product_name=product_name.upper(),
            version=version,
            applicable_date=applicable_date.strftime(APPLICABLE_DATE_FORMAT),
            utc_start=_ensure_utc_timezone(utc_start).strftime(PRINTABLE_TS_FORMAT),
            utc_end=_ensure_utc_timezone(utc_end).strftime(PRINTABLE_TS_FORMAT),
            revision=str(revision),
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
        d["applicable_date"] = date.fromisoformat(d["applicable_date"])
        d["utc_start"] = datetime.strptime(d["utc_start"], PRINTABLE_TS_FORMAT).replace(tzinfo=UTC)
        d["utc_end"] = datetime.strptime(d["utc_end"], PRINTABLE_TS_FORMAT).replace(tzinfo=UTC)
        d["revision"] = ulid.ULID.from_str(d["revision"])
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

    def _sort_key(self) -> tuple:
        """Order manifests by type, then by ULID (i.e. creation time)"""
        parts = self.filename_parts
        return (parts.manifest_type, parts.ulid_code, str(self.path))

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
        cls,
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


def midpoint_applicable_date(utc_start: datetime, utc_end: datetime) -> date:
    """Compute the default applicable date for a data product as the date of the midpoint of its time range.

    In all production processing cases, utc_start and utc_end should be roughly midnight on consecutive days, so
    the midpoint falls on the day the product applies to. Issues a UserWarning if the range spans more than 24 hours.

    Parameters
    ----------
    utc_start : datetime.datetime
        First timestamp of the data in the file. Naive datetimes are assumed to be UTC.
    utc_end : datetime.datetime
        Last timestamp of the data in the file. Naive datetimes are assumed to be UTC.

    Returns
    -------
    : datetime.date
        The date of the midpoint between utc_start and utc_end
    """
    utc_start = _ensure_utc_timezone(utc_start)
    utc_end = _ensure_utc_timezone(utc_end)

    if utc_end - utc_start > timedelta(hours=24):
        warnings.warn("Time range for filename spans more than 24 hours", UserWarning, stacklevel=2)

    t_mean = utc_start + 0.5 * (utc_end - utc_start)
    return t_mean.date()


def check_version_number_format(version: str) -> bool:
    """Ensures that a version string is in VM-m-p format for Libera filenaming. M, m, and p are integers
    representing Major, minor, and patch respectively. An optional release candidate suffix RCn is allowed.

    Parameters
    ----------
    version : str
        Version string to validate

    Returns
    -------
    : bool
        True if version string is in VM-m-p format, False otherwise
    """
    if LIBERA_SEM_VER_REGEX.fullmatch(version) is None:
        return False
    return True


def semantic_version_from_format(version: str) -> Version:
    """Parse a Libera filename version string like V3-14-159RC1 into a comparable semantic version object.

    This is the inverse of `format_from_semantic_version` and is used to order filenames by version numerically
    (V1-10-0 sorts after V1-9-0, and V1-2-3RC1 sorts before V1-2-3).

    Parameters
    ----------
    version : str
        Version string in VM-m-p[RCn] form

    Returns
    -------
    : packaging.version.Version
    """
    if not check_version_number_format(version):
        raise ValueError(f"Version string {version} is not in Libera filename format VM-m-p[RCn]")
    return Version(version[1:].replace("-", "."))


def format_from_semantic_version(semantic_version: str) -> str:
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
    # Use packaging's version class to handle more complex versions
    try:
        ver_object = Version(semantic_version)
    except InvalidVersion as e:
        raise ValueError(f"Invalid semantic version string: {semantic_version}") from e
    # We only want a major, minor, and patch style version string
    major = ver_object.major
    minor = ver_object.minor
    patch = ver_object.micro

    # Allow an option for pre-release notations like rc1
    if ver_object.is_prerelease:
        patch = str(patch) + ver_object.pre[0] + str(ver_object.pre[1])
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
        Version string in format V1-2-3
    """
    semver = metadata.version(package_name)
    return format_from_semantic_version(semver)
