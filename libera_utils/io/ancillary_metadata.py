"""Read the version and revision of an external ancillary file from the metadata inside it.

A filename class parses a name and never opens its file, so a value that only the file holds is read here. A CERES
SSF or CLDPIX file's name carries no production time, and a NISE file's name carries neither a version nor a
production time; each file's own metadata holds them.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

import netCDF4
from cloudpathlib import CloudPath
from pyhdf import SD

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import CeresFilename, NsidcNiseFilename

# CERES global attributes holding the time the granule was produced. A CLDPIX granule gives it as a ctime-style
# string with no zone, e.g. "Thu Jul 10 19:02:50 2025", which is read as UTC.
_CERES_SSF_REVISION_ATTR = "date_created"
_CERES_CLDPIX_REVISION_ATTR = "processTime"
_CERES_CLDPIX_REVISION_FORMAT = "%a %b %d %H:%M:%S %Y"

# NISE's HDF-EOS ECS core metadata, an ODL text attribute, and the objects in it read here.
_NISE_CORE_METADATA_ATTR = "coremetadata.0"
_NISE_REVISION_OBJECT = "PRODUCTIONDATETIME"
_NISE_VERSION_OBJECT = "VERSIONID"


@dataclass(frozen=True)
class AncillaryFileMetadata:
    """The version and revision of an external ancillary file, as the file records them.

    Attributes
    ----------
    version : str
        The product version: the production strategy the name carries for CERES (``Edition1C``), and the core
        metadata ``VERSIONID`` for NISE (``1``).
    revision : datetime.datetime
        The UTC time the provider produced the file, timezone-aware.
    """

    version: str
    revision: datetime


def read_ancillary_file_metadata(filename: CeresFilename | NsidcNiseFilename) -> AncillaryFileMetadata:
    """Read a CERES or NISE file's version and revision from the metadata inside the file.

    - CERES SSF: the revision is the ``date_created`` global attribute, an ISO 8601 instant.
    - CERES CLDPIX: the revision is the ``processTime`` global attribute, e.g. ``Thu Jul 10 19:02:50 2025``, which
      gives no zone and is read as UTC. CLDPIX granules carry no ``date_created``.
    - CERES version: the production strategy the name carries, as ``filename.version`` gives it.
    - NISE: the revision is ``PRODUCTIONDATETIME`` and the version ``VERSIONID``, both in the file's HDF-EOS core
      metadata (``coremetadata.0``).

    The file is opened at ``filename.path``. A cloud path is downloaded to cloudpathlib's local cache first, since
    neither netCDF4 nor pyhdf reads from a stream.

    Parameters
    ----------
    filename : CeresFilename | NsidcNiseFilename
        The parsed name of the file to read.

    Returns
    -------
    AncillaryFileMetadata
        The file's version and its revision as a timezone-aware UTC datetime.

    Raises
    ------
    ValueError
        If the attribute holding the revision or version is missing, or does not parse.
    OSError
        If netCDF4 cannot open a CERES file.
    pyhdf.error.HDF4Error
        If pyhdf cannot open a NISE file.
    """
    local_path = str(filename.path.fspath) if isinstance(filename.path, CloudPath) else str(filename.path)
    if isinstance(filename, CeresFilename):
        return AncillaryFileMetadata(version=filename.version, revision=_read_ceres_revision(filename, local_path))
    return _read_nise_metadata(filename, local_path)


def _read_ceres_revision(filename: CeresFilename, local_path: str) -> datetime:
    """Read a CERES file's production time from its global attributes.

    Parameters
    ----------
    filename : CeresFilename
        The parsed name of the file, for its product and for messages.
    local_path : str
        A local path to the file.

    Returns
    -------
    datetime.datetime
        The production time, timezone-aware UTC.

    Raises
    ------
    ValueError
        If the attribute is missing or does not parse.
    """
    is_cldpix = filename.data_product_id is DataProductIdentifier.anc_ceres_noaa20_cldpix
    attr = _CERES_CLDPIX_REVISION_ATTR if is_cldpix else _CERES_SSF_REVISION_ATTR
    with netCDF4.Dataset(local_path, "r") as dataset:
        value = getattr(dataset, attr, None)
    if value is None:
        raise ValueError(
            f"{filename.path.name}: global attribute {attr!r} is missing; the SDC takes a "
            f"{filename.data_product_id} file's revision from it. Check that the file is the granule as the ASDC "
            "distributes it."
        )
    try:
        if is_cldpix:
            return datetime.strptime(str(value).strip(), _CERES_CLDPIX_REVISION_FORMAT).replace(tzinfo=UTC)
        return _utc_instant(str(value))
    except ValueError as err:
        expected = "a time such as 'Thu Jul 10 19:02:50 2025'" if is_cldpix else "an ISO 8601 instant with a zone"
        raise ValueError(
            f"{filename.path.name}: global attribute {attr}={value!r} is not {expected}, so the file's revision "
            "cannot be read. Check that the file is the granule as the ASDC distributes it."
        ) from err


def _read_nise_metadata(filename: NsidcNiseFilename, local_path: str) -> AncillaryFileMetadata:
    """Read a NISE file's version and production time from its HDF-EOS core metadata.

    Parameters
    ----------
    filename : NsidcNiseFilename
        The parsed name of the file, for messages.
    local_path : str
        A local path to the file.

    Returns
    -------
    AncillaryFileMetadata
        ``VERSIONID`` as the version and ``PRODUCTIONDATETIME`` as the revision.

    Raises
    ------
    ValueError
        If the core metadata or either object is missing, or ``PRODUCTIONDATETIME`` does not parse.
    pyhdf.error.HDF4Error
        If the file cannot be opened.
    """
    hdf = SD.SD(local_path, SD.SDC.READ)
    try:
        core_metadata = hdf.attributes().get(_NISE_CORE_METADATA_ATTR)
    finally:
        hdf.end()
    if core_metadata is None:
        raise ValueError(
            f"{filename.path.name}: attribute {_NISE_CORE_METADATA_ATTR!r} is missing; the SDC reads a NISE file's "
            f"{_NISE_VERSION_OBJECT} and {_NISE_REVISION_OBJECT} from it. Check that the file is the granule as "
            "NSIDC distributes it."
        )
    version = _odl_value(filename.path.name, core_metadata, _NISE_VERSION_OBJECT, "version")
    production_time = _odl_value(filename.path.name, core_metadata, _NISE_REVISION_OBJECT, "revision")
    try:
        revision = _utc_instant(production_time)
    except ValueError as err:
        raise ValueError(
            f"{filename.path.name}: {_NISE_REVISION_OBJECT}={production_time!r} in {_NISE_CORE_METADATA_ATTR} is "
            "not an ISO 8601 instant with a zone, so the file's revision cannot be read. Check that the file is the "
            "granule as NSIDC distributes it."
        ) from err
    return AncillaryFileMetadata(version=version, revision=revision)


def _odl_value(file_name: str, odl_text: str, name: str, field: str) -> str:
    """Return the ``VALUE`` of one ``OBJECT`` in ECS core metadata, an ODL text, with its quotes removed.

    Parameters
    ----------
    file_name : str
        The file's name, for messages.
    odl_text : str
        The core metadata text.
    name : str
        The object's name, e.g. ``PRODUCTIONDATETIME``.
    field : str
        What the SDC takes from the object, ``version`` or ``revision``, for messages.

    Returns
    -------
    str
        The object's value.

    Raises
    ------
    ValueError
        If the text has no such object, or the object has no value.
    """
    obj = re.search(rf"OBJECT\s*=\s*{name}\s*$(.*?)END_OBJECT\s*=\s*{name}\s*$", odl_text, re.DOTALL | re.MULTILINE)
    value = re.search(r"^\s*VALUE\s*=\s*(.+?)\s*$", obj.group(1), re.MULTILINE) if obj else None
    if value is None:
        raise ValueError(
            f"{file_name}: {_NISE_CORE_METADATA_ATTR} has no {name} value; the SDC takes a NISE file's {field} from "
            "it. Check that the file is the granule as NSIDC distributes it."
        )
    return value.group(1).strip('"')


def _utc_instant(text: str) -> datetime:
    """Parse an ISO 8601 instant that carries its zone, as a UTC datetime.

    Parameters
    ----------
    text : str
        The instant, e.g. ``2024-08-27T19:46:15Z``.

    Returns
    -------
    datetime.datetime
        The instant, timezone-aware UTC.

    Raises
    ------
    ValueError
        If the text is not an ISO 8601 instant, or gives no zone.
    """
    instant = datetime.fromisoformat(text.strip())
    if instant.tzinfo is None:
        raise ValueError(f"{text!r} gives no time zone")
    return instant.astimezone(UTC)
