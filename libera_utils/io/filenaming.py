"""Classes and utilities for naming and parsing files in the Libera SDC production system"""

import re
import warnings
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime, timedelta
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

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

REVISION_TS_FORMAT = f"R{PRINTABLE_TS_FORMAT}%f"


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

# Demuxed ground-test CCSDS names (no extension), one APID per file:
# LIBERA_SDC_<apid>_ccsds_<yyyy>_<doy>_<hh>_<mm>_<ss>
LIBERA_GROUND_CCSDS_REGEX = re.compile(
    r"^LIBERA_SDC_(?P<apid>[0-9]{1,4})"
    r"_ccsds_(?P<year>[0-9]{4})"
    r"_(?P<doy>00[1-9]|0[1-9][0-9]|[12][0-9]{2}|3[0-5][0-9]|36[0-6])"
    r"_(?P<hour>[01][0-9]|2[0-3])"
    r"_(?P<minute>[0-5][0-9])"
    r"_(?P<second>[0-5][0-9])$"
)

# CCSDS primary header APID field is 11 bits.
MAX_CCSDS_APID = 2047

# Get all data levels for the regex
DATA_LEVELS = "|".join([level.value for level in DataLevel])

# Get all data product names
DATA_PRODUCT_NAMES = "|".join([str(dpi) for dpi in DataProductIdentifier])

# Everything in a data product filename except the extension. Shared by the data file and its UMM-G metadata file.
# The revision is REVISION_TS_FORMAT: R + 8-digit date, T, 6-digit time, 6-digit microseconds.
_LIBERA_DATA_PRODUCT_BODY_REGEX = (
    rf"^LIBERA_(?P<data_level>{DATA_LEVELS})"
    rf"_(?P<product_name>{DATA_PRODUCT_NAMES})"
    r"_(?P<version>V[0-9]*-[0-9]*-[0-9]*(RC[0-9])?)"
    r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
    r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
    r"_(?P<revision>R[0-9]{8}T[0-9]{12})"
)

LIBERA_DATA_PRODUCT_REGEX = re.compile(_LIBERA_DATA_PRODUCT_BODY_REGEX + r"\.(?P<extension>nc|bsp|bc)$")

LIBERA_METADATA_PRODUCT_REGEX = re.compile(_LIBERA_DATA_PRODUCT_BODY_REGEX + r"\.(?P<extension>cmr\.json)$")

MANIFEST_FILE_REGEX = re.compile(
    r"^LIBERA"
    r"_(?P<manifest_type>INPUT|OUTPUT)"
    r"_MANIFEST"
    r"_(?P<ulid_code>[0-9A-HJ-NP-TV-Z]{26})"
    r"\.json$"
)

LIBERA_SEM_VER_REGEX = re.compile(r"V[0-9]*-[0-9]*-[0-9]*(RC[0-9])?")

# External ancillary products are ingested under the names their upstream providers give them. Each pattern below
# is the provider's distribution name, as on the files in tests/test_data/footprint_matching, except ERA5, whose
# downloads the SDC names (see Era5Filename). The CERES and VIIRS patterns fix the NOAA-20 platform their
# identifiers name.

# NOAA-20 CERES Level 2 products from NASA Langley ASDC: the product and instrument part of the name, mapped to its
# identifier. FLASHFlux files carry the SSF name with an ``alpha`` production strategy.
_CERES_PRODUCTS = {
    "SSF_NOAA20-FM6-VIIRS": DataProductIdentifier.anc_ceres_noaa20_ssf,
    "CLDPIX_NOAA20-VIIRS": DataProductIdentifier.anc_ceres_noaa20_cldpix,
}

CERES_REGEX = re.compile(
    rf"^CER_(?P<product>{'|'.join(map(re.escape, _CERES_PRODUCTS))})"
    r"_(?P<version>[A-Za-z0-9]+)"
    r"_(?P<configuration_code>[0-9]{6})"
    r"\.(?P<utc_start>[0-9]{10})"
    r"\.nc$"
)

# NOAA-20 VIIRS Level 3 daily products from NASA LAADS DAAC (Deep Blue aerosol, cloud properties) and LP DAAC
# (BRDF/albedo): the short name, mapped to its identifier and the extension the provider gives it.
_VIIRS_L3_PRODUCTS = {
    "AERDB_D3_VIIRS_NOAA20": (DataProductIdentifier.anc_viirs_noaa20_aod, "nc"),
    "CLDPROP_D3_VIIRS_NOAA20": (DataProductIdentifier.anc_viirs_noaa20_cloud_properties, "nc"),
    "VJ143C1": (DataProductIdentifier.anc_viirs_noaa20_brdf, "h5"),
}

VIIRS_L3_REGEX = re.compile(
    rf"^(?P<short_name>{'|'.join(map(re.escape, _VIIRS_L3_PRODUCTS))})"
    r"\.A(?P<year>[0-9]{4})(?P<doy>[0-9]{3})"
    r"\.(?P<version>[0-9]{3})"
    r"\.(?P<production_time>[0-9]{13})"
    r"\.(?P<extension>nc|h5)$"
)

# NOAA-20 (j01) unaggregated VIIRS M-band SDR files from NOAA CLASS, in JPSS IDPS HDF5 naming (as described in the
# JPSS VIIRS SDR Data Dictionary): radiances (IVCDB) and geolocation (ICDBG), operational (ops) domain only.
VIIRS_SDR_REGEX = re.compile(
    r"^(?P<kind>IVCDB|ICDBG)_j01"
    r"_d(?P<start_date>[0-9]{8})"
    r"_t(?P<start_time>[0-9]{7})"
    r"_e(?P<end_time>[0-9]{7})"
    r"_b(?P<orbit>[0-9]{5})"
    r"_c(?P<creation_time>[0-9]{20})"
    r"_(?P<origin>[a-z]{4})_ops"
    r"\.h5$"
)

# NSIDC Near-real-time Ice and Snow Extent from AMSR2 (NISE_A2). The name carries no version; NISE_VERSION is the
# collection's VERSIONID from the files' core metadata.
NSIDC_NISE_REGEX = re.compile(r"^NISE_AMSR2_(?P<date>[0-9]{8})\.HDFEOS$")
NISE_VERSION = "1"

# NSIDC EASE-Grid 2.0 Land-Ocean-Coastline-Ice Masks (NSIDC-0609), global (M) grids only, at the resolutions the
# dataset provides. The name carries no version; NSIDC_LAND_OCEAN_MASK_VERSION is the dataset version.
NSIDC_LAND_OCEAN_MASK_REGEX = re.compile(
    r"^EASE2_M(?P<resolution_km>03|6\.25|09|12\.5|25|36)km"
    r"\.LOCImask_land50_coast0km"
    r"\.(?P<columns>[0-9]+)x(?P<rows>[0-9]+)"
    r"\.bin$"
)
NSIDC_LAND_OCEAN_MASK_VERSION = "1"

# ERA5 files from the Copernicus Climate Data Store, named by the SDC in the Libera data product form; the version
# is the stream, ERA5 (final) or ERA5T (preliminary).
_ERA5_PRODUCTS = (DataProductIdentifier.anc_era5_pressure_level, DataProductIdentifier.anc_era5_single_level)

ERA5_REGEX = re.compile(
    rf"^ECMWF_(?P<product_name>{'|'.join(str(product) for product in _ERA5_PRODUCTS)})"
    r"_(?P<version>ERA5T?)"
    r"_(?P<utc_start>[0-9]{8}T[0-9]{6})"
    r"_(?P<utc_end>[0-9]{8}T[0-9]{6})"
    r"_(?P<revision>R[0-9]{8}T[0-9]{12})"
    r"\.nc$"
)


class AbstractValidFilename(ABC):
    """Filename class that ensures validity of a filename based on regex pattern.

    Notes
    -----
    - This is an abstract base class that must be inherited by concrete filename classes.
    - This class internally stores a CloudPath or Path object in the `path` property (composition).
    - Instances are hashable and compare equal when their paths are equal (as given; a relative and an absolute
      spelling of the same file are distinct).
    """

    _regex: re.Pattern
    _fmt: str
    _path: PathType

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

    @classmethod
    def from_file_path(cls, *args, **kwargs):
        """Factory method to produce an AbstractValidFilename from a file path (str or Path) of any known form.

        The forms are L0, ground CCSDS, Libera data product, manifest, and the external ancillary products' names.

        Parameters
        ----------
        *args
            The file path (str, Path or S3Path), and any further positional arguments, passed to each candidate
            class's constructor and on to ``AnyPath``.
        **kwargs
            Keyword arguments passed to each candidate class's constructor and on to ``AnyPath``.

        Returns
        -------
        : AbstractValidFilename
            An instance of the first filename class that accepts the path.

        Raises
        ------
        ValueError
            If no filename class accepts the path.
        """
        for CandidateClass in (
            L0Filename,
            LiberaGroundCcsdsFilename,
            LiberaDataProductFilename,
            ManifestFilename,
            CeresFilename,
            ViirsL3Filename,
            ViirsSdrFilename,
            NsidcNiseFilename,
            NsidcLandOceanMaskFilename,
            Era5Filename,
        ):
            try:
                filename = CandidateClass(*args, **kwargs)
                return filename
            except ValueError:
                continue

        raise ValueError(
            f"Unable to create a valid filename from {args}. The name is not an L0, ground CCSDS, Libera data "
            "product, manifest or external ancillary product filename; see the File Naming Conventions "
            "documentation for the accepted forms."
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

        Note: When this is implemented by concrete classes, ``*args`` and ``**kwargs`` become specific parameters
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


def _validate_ccsds_apid(apid: int) -> int:
    """Return ``apid`` if it fits the 11-bit CCSDS APID field, else raise ``ValueError``."""
    if not 0 <= apid <= MAX_CCSDS_APID:
        raise ValueError(f"APID {apid} is outside the CCSDS range 0-{MAX_CCSDS_APID}")
    return apid


def _parse_doy_datetime(year: int, doy: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """Build a UTC datetime from year, day-of-year and time-of-day filename fields.

    Raises
    ------
    ValueError
        If the fields do not name a real instant. This includes DOY 366 in a common year,
        which ``strptime`` rolls into 1 January of the following year rather than rejecting.
    """
    instant = datetime.strptime(f"{year:04d}{doy:03d}{hour:02d}{minute:02d}{second:02d}", "%Y%j%H%M%S")
    if instant.year != year:
        raise ValueError(f"Day of year {doy} does not exist in {year}")
    return _ensure_utc_timezone(instant)


class LiberaGroundCcsdsFilename(AbstractDataProductFilename):
    """Filename validation class for demuxed ground-test CCSDS files.

    Canonical form: ``LIBERA_SDC_<apid>_ccsds_<yyyy>_<doy>_<hh>_<mm>_<ss>`` (no extension),
    e.g. ``LIBERA_SDC_1057_ccsds_2026_191_14_00_00``. Each file holds packets for the single
    APID named in the basename, with no record header before the CCSDS primary header.

    The time fields are the start of the time bin the file was cut from, not the span of the
    packets inside it: packet times can fall outside the bin. The archive prefix
    ``GroundCCSDS/<apid>/<yyyy>/<mm>/<dd>/`` is an expansion of these fields and carries no
    guarantee about packet or data times. Searchable times come from File Metadata at ingest.
    """

    _regex = LIBERA_GROUND_CCSDS_REGEX
    _fmt = "LIBERA_SDC_{apid:d}_ccsds_{year:04d}_{doy:03d}_{hour:02d}_{minute:02d}_{second:02d}"

    @AbstractValidFilename.path.setter
    def path(self, new_path: str | PathType):
        """Set the path, rejecting fields that do not form a real instant or a legal APID.

        The base setter validates against the regex only, which cannot express leap years
        and admits APIDs above the 11-bit CCSDS maximum.
        """
        AbstractValidFilename.path.fset(self, new_path)
        self._parse_filename_parts()

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """Property that contains the DataProductIdentifier for this file type."""
        return DataProductIdentifier.l0_ground_ccsds

    @property
    def apid(self) -> int:
        """CCSDS APID encoded in the filename.

        Not necessarily a defined ``LiberaApid`` member; ingest accepts any legal APID.
        """
        return self.filename_parts.apid

    @property
    def bin_start(self) -> datetime:
        """UTC start of the time bin encoded in the filename."""
        return self.filename_parts.bin_start

    @property
    def archive_prefix(self) -> str:
        """L0 archive prefix from the filename's APID and bin start."""
        start = self.bin_start
        return f"GroundCCSDS/{self.apid:04d}/{start.year:04d}/{start.month:02d}/{start.day:02d}"

    @classmethod
    def from_filename_parts(
        cls,  # noqa pylint: disable=arguments-differ
        *,
        apid: int,
        year: int,
        doy: int,
        hour: int,
        minute: int,
        second: int,
        basepath: str | Path | S3Path | None = None,
    ):
        """Create instance from filename parts.

        Parameters
        ----------
        apid : int
            CCSDS APID of the packets in the file (0-2047).
        year : int
            Four-digit UTC year of the bin start.
        doy : int
            Day of year (1-366).
        hour, minute, second : int
            Bin start time-of-day in UTC.
        basepath : Optional[Union[str, Path, S3Path]]
            Optional directory or S3 prefix prepended to the basename.

        Returns
        -------
        LiberaGroundCcsdsFilename
        """
        return cls._from_filename_parts(
            basepath=basepath,
            apid=apid,
            year=year,
            doy=doy,
            hour=hour,
            minute=minute,
            second=second,
        )

    @classmethod
    def _format_filename_parts(
        cls,
        *,
        apid: int,
        year: int,
        doy: int,
        hour: int,
        minute: int,
        second: int,
    ):
        """Construct a basename from filename parts."""
        _validate_ccsds_apid(apid)
        _parse_doy_datetime(year, doy, hour, minute, second)
        return cls._fmt.format(apid=apid, year=year, doy=doy, hour=hour, minute=minute, second=second)

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings."""
        d = self.regex_match(self.path)
        apid = _validate_ccsds_apid(int(d["apid"]))
        year = int(d["year"])
        doy = int(d["doy"])
        hour = int(d["hour"])
        minute = int(d["minute"])
        second = int(d["second"])
        return SimpleNamespace(
            apid=apid,
            year=year,
            doy=doy,
            hour=hour,
            minute=minute,
            second=second,
            bin_start=_parse_doy_datetime(year, doy, hour, minute, second),
        )


class LiberaDataProductFilename(AbstractDataProductFilename):
    """Filename validation class for Libera SDC data products.

    Libera data product filenames follow the convention::

        LIBERA_{data_level}_{product_name}_{version}_{utc_start}_{utc_end}_{revision}.{extension}

    e.g. ``LIBERA_L1B_RAD-4CH_V1-2-3_20270102T112233_20270102T122233_R20270103T080910123456.nc``

    - ``data_level`` and ``product_name`` are the string values of :class:`~libera_utils.constants.DataLevel` and
      :class:`~libera_utils.constants.DataProductIdentifier`.
    - ``version`` is the algorithm semantic version in ``VM-m-p[RCn]`` form.
    - ``utc_start`` / ``utc_end`` bound the data in the file, formatted ``YYYYMMDDTHHMMSS``.
    - ``revision`` is the UTC creation time of the file, ``R`` followed by ``YYYYMMDDTHHMMSSffffff`` (microseconds).
    - ``extension`` is ``nc`` for NetCDF-4 products, or ``bsp`` / ``bc`` for SPICE kernels.
    """

    _regex = LIBERA_DATA_PRODUCT_REGEX
    _fmt = "LIBERA_{data_level}_{product_name}_{version}_{utc_start}_{utc_end}_{revision}.{extension}"

    @AbstractValidFilename.path.setter
    def path(self, new_path: str | PathType):
        """Set the path, rejecting a utc_start, utc_end or revision that is not a real UTC time.

        The regex admits any digits in those fields, so the name is parsed before it is assigned.

        Parameters
        ----------
        new_path : str | PathType
            Path to the data product file.

        Raises
        ------
        ValueError
            If the path does not match the filename regex, or a time field is not a real UTC time.
        """
        if isinstance(new_path, str):
            _new_path: PathType = cast(PathType, AnyPath(new_path))
        else:
            _new_path = new_path
        parts = self.regex_match(_new_path)
        try:
            self._parse_match(parts)
        except ValueError as err:
            raise ValueError(
                f"Proposed path {_new_path} has a utc_start, utc_end or revision that is not a real UTC time: "
                f"{err}. Correct that field."
            ) from err
        self._path = _new_path

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
        revision: datetime | None = None,
        extension: str | None = None,
        basepath: str | Path | S3Path | None = None,
    ):
        """Create instance from filename parts.

        This method exists primarily to expose typehinting to the user for use with the generic _from_filename_parts.
        The part names are named according to the regex for the file type.

        Parameters
        ----------
        data_level : str | DataLevel | None
            L1B or L2 identifying the level of the data product. Default None will infer the data level from the product name (DataProductIdentifier)
        product_name : str | DataProductIdentifier
            Product type. e.g. CF-CAM for L2 or RAD-4CH for L1B. May contain anything except for underscores.
        version : str
            Software version that the file was created with. Corresponds to the algorithm version as determined
            by the algorithm software.
        utc_start : datetime.datetime
            First timestamp in the SPK
        utc_end : datetime.datetime
            Last timestamp in the SPK
        revision : datetime.datetime | None
            Time when the file was created. Default None uses the current UTC time at the time of the call.
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

        if revision is None:
            revision = datetime.now(tz=UTC)

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
        revision : datetime.datetime
            Time when the file was created.
        extension : str
            File extension (nc, bsp or bc)

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
        return self._parse_match(self.regex_match(self.path))

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects

        Raises
        ------
        ValueError
            If utc_start, utc_end or revision is not a real UTC time.
        """
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


class AncillaryDataProductFilename(AbstractDataProductFilename):
    """Abstract base for the filenames of external ancillary products, named by their upstream provider.

    The SDC ingests these files under the names their providers give them, so these classes parse names but do
    not build them. Setting the path parses every field, so a name whose fields match the pattern but do not form
    a real date or time is rejected when the object is created.

    Each concrete class's parsed parts include ``data_product_id`` and ``version``, the upstream version string.
    """

    @AbstractValidFilename.path.setter
    def path(self, new_path: str | PathType):
        """Set the path, rejecting a name whose date or time fields do not form a real UTC instant.

        The name is parsed before it is assigned, so a rejected path leaves the previous path in place.

        Parameters
        ----------
        new_path : str | PathType
            Path to the ancillary file.

        Raises
        ------
        ValueError
            If the name does not match the class's pattern, or a date or time field is not a real UTC instant.
        """
        if isinstance(new_path, str):
            _new_path: PathType = cast(PathType, AnyPath(new_path))
        else:
            _new_path = new_path
        parts = self.regex_match(_new_path)
        try:
            self._parse_match(parts)
        except ValueError as err:
            raise ValueError(
                f"Proposed path {_new_path} is not a valid {type(self).__name__}: {err}. Correct the field in the name."
            ) from err
        self._path = _new_path

    def _parse_filename_parts(self):
        """Parse the filename parts into objects from regex matched strings

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        return self._parse_match(self.regex_match(self.path))

    @staticmethod
    @abstractmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects, including ``data_product_id`` and
            ``version``

        Raises
        ------
        ValueError
            If a field does not parse to a valid value.
        """
        raise NotImplementedError()

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """Property that contains the DataProductIdentifier the filename names"""
        return self.filename_parts.data_product_id

    @classmethod
    def from_filename_parts(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Not supported: upstream names are parsed, not built.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            f"{cls.__name__} parses names given by an upstream provider and cannot build one. "
            "Construct it from an existing file path instead."
        )

    @classmethod
    def _format_filename_parts(cls, *args: Any, **kwargs: Any) -> NoReturn:
        """Not supported: upstream names are parsed, not built.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(f"{cls.__name__} parses names given by an upstream provider and cannot build one.")


class DailyAncillaryDataProductFilename(AncillaryDataProductFilename):
    """Abstract base for ancillary products whose filename names the UTC day the data apply to.

    The file is archived under ``<ProductID>/YYYY/MM/DD`` for that day, keeping its upstream name.
    """

    @property
    def applicable_date(self) -> date:
        """The UTC date the file's data apply to, as named in the filename.

        A granule shorter than a day (hourly CERES, an ~85 s VIIRS SDR granule) or longer (the 16-day VIIRS BRDF
        composite) takes the date its name gives: the start date, or the BRDF composite's nominal date.
        """
        return self.filename_parts.applicable_date

    @property
    def archive_prefix(self) -> str:
        """Archive prefix ``<ProductID>/YYYY/MM/DD`` from the applicable date"""
        applicable_date = self.applicable_date
        return (
            f"{self.data_product_id}/{applicable_date.year:04d}/{applicable_date.month:02d}/{applicable_date.day:02d}"
        )


class CeresFilename(DailyAncillaryDataProductFilename):
    """Filename class for NOAA-20 CERES Level 2 products from NASA Langley ASDC.

    Form: ``CER_<product>_<platform and instruments>_<production strategy>_<configuration code>.<YYYYMMDDHH>.nc``,
    e.g. ``CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc``. One file holds one hour of footprints.

    - ``version`` is the production strategy, e.g. ``Edition1C``, or ``alpha4`` for the FLASHFlux stream.
    - ``configuration_code`` is the six-digit CERES configuration code.
    - ``utc_start`` is the start of the hour; ``applicable_date`` is its UTC date.
    """

    _regex = CERES_REGEX

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects

        Raises
        ------
        ValueError
            If the hour field is not a real UTC hour.
        """
        utc_start = datetime.strptime(d["utc_start"], "%Y%m%d%H").replace(tzinfo=UTC)
        return SimpleNamespace(
            data_product_id=_CERES_PRODUCTS[d["product"]],
            version=d["version"],
            configuration_code=d["configuration_code"],
            utc_start=utc_start,
            applicable_date=utc_start.date(),
        )


class ViirsL3Filename(DailyAncillaryDataProductFilename):
    """Filename class for NOAA-20 VIIRS Level 3 daily products from NASA LAADS DAAC and LP DAAC.

    Form: ``<short name>.A<YYYYDDD>.<collection>.<production YYYYDDDHHMMSS>.<extension>``,
    e.g. ``AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc``.

    - ``applicable_date`` is the ``A`` date. For the 16-day BRDF composite (``VJ143C1``) it is the composite's
      nominal date, which the provider issues daily.
    - ``version`` is the three-digit collection.
    - ``production_time`` is the UTC time the provider produced the file.
    """

    _regex = VIIRS_L3_REGEX

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects

        Raises
        ------
        ValueError
            If the extension is not its product's, or a date field is not a real UTC instant.
        """
        data_product_id, extension = _VIIRS_L3_PRODUCTS[d["short_name"]]
        if d["extension"] != extension:
            raise ValueError(
                f"the extension is .{d['extension']}, but {d['short_name']} files are .{extension}. "
                "Check that the file is the product its name says"
            )
        production = d["production_time"]
        return SimpleNamespace(
            short_name=d["short_name"],
            data_product_id=data_product_id,
            version=d["version"],
            applicable_date=_parse_doy_datetime(int(d["year"]), int(d["doy"])).date(),
            production_time=_parse_doy_datetime(
                int(production[:4]),
                int(production[4:7]),
                int(production[7:9]),
                int(production[9:11]),
                int(production[11:13]),
            ),
        )


class ViirsSdrFilename(DailyAncillaryDataProductFilename):
    """Filename class for NOAA-20 unaggregated VIIRS M-band SDR files from NOAA CLASS.

    Form (JPSS IDPS HDF5 naming):
    ``<kind>_j01_d<YYYYMMDD>_t<HHMMSSs>_e<HHMMSSs>_b<orbit>_c<YYYYMMDDHHMMSSffffff>_<origin>_ops.h5``,
    e.g. ``IVCDB_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5``.

    - ``kind`` is ``IVCDB`` (radiances, VIIRS-DualGain-Cal-IP) or ``ICDBG`` (geolocation, VIIRS-MOD-UNAGG-GEO).
      Both are the one product ``VIIRS-NOAA20-M4-SDR``, as single granules or aggregates.
    - ``utc_start`` and ``utc_end`` carry tenths of a second; an end earlier than the start is on the next day.
    - ``applicable_date`` is the ``d`` (start) date.
    - ``version`` is the ``c`` creation time, as its 20 digits.
    """

    _regex = VIIRS_SDR_REGEX

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects

        Raises
        ------
        ValueError
            If a date or time field is not a real UTC instant.
        """
        start_date = datetime.strptime(d["start_date"], "%Y%m%d").replace(tzinfo=UTC)
        utc_start = start_date + _parse_sdr_time_of_day(d["start_time"])
        utc_end = start_date + _parse_sdr_time_of_day(d["end_time"])
        if utc_end < utc_start:
            utc_end += timedelta(days=1)
        return SimpleNamespace(
            kind=d["kind"],
            data_product_id=DataProductIdentifier.anc_viirs_noaa20_m4_sdr,
            version=d["creation_time"],
            utc_start=utc_start,
            utc_end=utc_end,
            applicable_date=start_date.date(),
            orbit=int(d["orbit"]),
            creation_time=datetime.strptime(d["creation_time"], "%Y%m%d%H%M%S%f").replace(tzinfo=UTC),
            origin=d["origin"],
        )


class NsidcNiseFilename(DailyAncillaryDataProductFilename):
    """Filename class for the NSIDC Near-real-time Ice and Snow Extent from AMSR2 (NISE_A2).

    Form: ``NISE_AMSR2_<YYYYMMDD>.HDFEOS``, e.g. ``NISE_AMSR2_20260610.HDFEOS``. ``applicable_date`` is the named
    date. The name carries no version, so ``version`` is the NISE_A2 collection version, ``"1"``, the
    ``VERSIONID`` in the files' core metadata.
    """

    _regex = NSIDC_NISE_REGEX

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects

        Raises
        ------
        ValueError
            If the date is not a real date.
        """
        return SimpleNamespace(
            data_product_id=DataProductIdentifier.anc_nsidc_ice_snow,
            # TODO[LIBSDC-859]: Confirm the version Data Availability keys NISE files on
            version=NISE_VERSION,
            applicable_date=datetime.strptime(d["date"], "%Y%m%d").date(),
        )


class NsidcLandOceanMaskFilename(AncillaryDataProductFilename):
    """Filename class for the NSIDC EASE-Grid 2.0 Land-Ocean-Coastline-Ice Masks (NSIDC-0609).

    Form: ``EASE2_M<resolution>km.LOCImask_land50_coast0km.<columns>x<rows>.bin``, a global (``M``) grid at
    3, 6.25, 9, 12.5, 25 or 36 km, e.g. ``EASE2_M36km.LOCImask_land50_coast0km.964x406.bin``. The browse images
    (``.png``) and the temperate (``T``) grids are not accepted.

    The mask is static: it has no applicable date and is archived under ``<ProductID>/<version>``. The name
    carries no version, so ``version`` is the NSIDC-0609 dataset version, ``"1"``.
    """

    _regex = NSIDC_LAND_OCEAN_MASK_REGEX

    @property
    def archive_prefix(self) -> str:
        """Archive prefix ``<ProductID>/<version>``"""
        return f"{self.data_product_id}/{self.filename_parts.version}"

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects
        """
        return SimpleNamespace(
            data_product_id=DataProductIdentifier.anc_nsidc_land_ocean_mask,
            # TODO[LIBSDC-859]: Confirm the version Data Availability keys the mask on
            version=NSIDC_LAND_OCEAN_MASK_VERSION,
            resolution_km=float(d["resolution_km"]),
            columns=int(d["columns"]),
            rows=int(d["rows"]),
        )


class Era5Filename(DailyAncillaryDataProductFilename):
    """Filename class for ERA5 files the SDC requests from the Copernicus Climate Data Store (CDS).

    The CDS gives downloads no consistent name, so the SDC names them, following the Libera data product form:
    ``ECMWF_<product_name>_<version>_<utc_start>_<utc_end>_<revision>.nc``, e.g.
    ``ECMWF_ERA5-SINGLE-LEVEL_ERA5T_20250101T000000_20250101T230000_R20251028T120000000000.nc``.

    - ``product_name`` is ``ERA5-PRESSURE-LEVEL`` or ``ERA5-SINGLE-LEVEL``.
    - ``version`` is the stream: ``ERA5`` (final) or ``ERA5T`` (preliminary).
    - ``utc_start`` / ``utc_end`` bound the hourly steps in the file, formatted ``YYYYMMDDTHHMMSS``.
    - ``revision`` is the UTC time the SDC named the file, ``R`` followed by ``YYYYMMDDTHHMMSSffffff``.
    - ``applicable_date`` is the UTC date of ``utc_start``, also for a file whose steps run into the next day.
    """

    _regex = ERA5_REGEX
    _fmt = "ECMWF_{product_name}_{version}_{utc_start}_{utc_end}_{revision}.nc"

    @classmethod
    def from_filename_parts(  # type: ignore[override]
        cls,  # noqa pylint: disable=arguments-differ
        *,
        product_name: str | DataProductIdentifier,
        version: str,
        utc_start: datetime,
        utc_end: datetime,
        revision: datetime | None = None,
        basepath: str | Path | S3Path | None = None,
    ) -> "Era5Filename":
        """Create instance from filename parts.

        Parameters
        ----------
        product_name : str | DataProductIdentifier
            ``ERA5-PRESSURE-LEVEL`` or ``ERA5-SINGLE-LEVEL``.
        version : str
            Stream: ``ERA5`` (final) or ``ERA5T`` (preliminary).
        utc_start : datetime.datetime
            Time of the first hourly step in the file. A naive datetime is taken as UTC.
        utc_end : datetime.datetime
            Time of the last hourly step in the file. A naive datetime is taken as UTC.
        revision : datetime.datetime | None
            Time the file was named. Default None uses the current UTC time.
        basepath : Optional[Union[str, Path, S3Path]]
            Allows prepending a basepath or prefix.

        Returns
        -------
        : Era5Filename

        Raises
        ------
        ValueError
            If ``product_name`` is not an ERA5 product, ``version`` is not ``ERA5`` or ``ERA5T``, or ``utc_end`` is
            before ``utc_start``.
        """
        if revision is None:
            revision = datetime.now(tz=UTC)
        return cls._from_filename_parts(
            basepath=basepath,
            product_name=str(DataProductIdentifier(product_name)),
            version=version,
            utc_start=_ensure_utc_timezone(utc_start),
            utc_end=_ensure_utc_timezone(utc_end),
            revision=_ensure_utc_timezone(revision),
        )

    @classmethod
    def _format_filename_parts(  # type: ignore[override]
        cls,
        *,
        product_name: str,
        version: str,
        utc_start: datetime,
        utc_end: datetime,
        revision: datetime,
    ) -> str:
        """Construct a basename from filename parts

        Returns
        -------
        : str
            Formatted filename
        """
        return cls._fmt.format(
            product_name=product_name,
            version=version,
            utc_start=utc_start.strftime(PRINTABLE_TS_FORMAT),
            utc_end=utc_end.strftime(PRINTABLE_TS_FORMAT),
            revision=revision.strftime(REVISION_TS_FORMAT),
        )

    @staticmethod
    def _parse_match(d: dict[str, Any]) -> SimpleNamespace:
        """Parse regex matched filename parts into objects

        Parameters
        ----------
        d : dict
            Named groups from matching the filename regex

        Returns
        -------
        : types.SimpleNamespace
            namespace object containing filename parts as parsed objects

        Raises
        ------
        ValueError
            If a time field is not a real UTC time, or ``utc_end`` is before ``utc_start``.
        """
        parts = LiberaDataProductFilename._parse_match(d)
        if parts.utc_end < parts.utc_start:
            raise ValueError(
                f"utc_end {parts.utc_end:%Y-%m-%dT%H:%M:%S} is before utc_start {parts.utc_start:%Y-%m-%dT%H:%M:%S}"
            )
        parts.data_product_id = DataProductIdentifier(parts.product_name)
        parts.applicable_date = parts.utc_start.date()
        return parts


def _parse_sdr_time_of_day(hhmmss_tenths: str) -> timedelta:
    """Convert a JPSS IDPS ``HHMMSSs`` time field (tenths of a second) to an offset from midnight.

    Raises
    ------
    ValueError
        If the hour, minute or second is out of range.
    """
    clock = datetime.strptime(hhmmss_tenths[:6], "%H%M%S")
    return timedelta(
        hours=clock.hour, minutes=clock.minute, seconds=clock.second, milliseconds=100 * int(hhmmss_tenths[6])
    )


def check_version_number_format(version: str) -> bool:
    """Ensures that a version string is in VM-m-p format for Libera filenaming. M, m, and p are integers
    representing Major, minor, and patch respectively.

    Parameters
    ----------
    version : str
        Version string to validate

    Returns
    -------
    : bool
        True if version string is in VM-m-p format, False otherwise
    """
    if LIBERA_SEM_VER_REGEX.match(version) is None:
        return False
    return True


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
