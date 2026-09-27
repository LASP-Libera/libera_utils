"""Filename parsing for the external auxiliary data products consumed by footprint matching.

Footprint matching (FMATCH) aggregates a set of third-party granules -- ERA5 reanalysis, MODIS
IGBP land cover, NSIDC NISE snow/ice, VIIRS BRDF/cloud/aerosol products, and CERES SSF/CLDPIX --
onto each Libera radiometer footprint. These are *inputs* Libera consumes, not products Libera
produces; each carries its vendor's own filename convention rather than the Libera convention that
:mod:`libera_utils.io.filenaming` parses. This module recognizes those vendor names and maps each
to its :class:`~libera_utils.constants.DataProductIdentifier` ``auxiliary_*`` member, extracting
the observation date and whatever version / production-time / source-variant tokens the name
carries.

Reader dispatch stays out of this module by design. An ``AuxiliaryFilename`` carries the
``product_id`` only; the mapping from a product to the FMATCH reader that consumes it lives in
:mod:`libera_utils.footprint_matching.readers.registry`
(``reader_key_for_auxiliary_product``). Keeping that mapping in the footprint-matching layer means
``libera_utils.io`` never has to import (and pull in the heavyweight numpy/xarray/h5py dependencies
of) the readers package.

ERA5 is a special case. Raw Copernicus CDS granules have no stable filename (hash names, or
ambiguous descriptive names) and single- vs pressure-level cannot be told apart by name, so the
Libera ingest layer renames each ERA5 granule to a Libera-assigned canonical name at staging --
``ERA5-SINGLE-LEVEL_<YYYYMMDD>.nc`` / ``ERA5-PRESSURE-LEVEL_<YYYYMMDD>.nc``. This module parses
those canonical names like any other convention; construct them with
:func:`build_era5_canonical_filename` (the single source of truth for the format, so ingest and
the parser cannot drift). Raw CDS names are intentionally *not* matched.

Examples
--------
>>> from libera_utils.io.auxiliary_filenaming import parse_auxiliary_filename
>>> parsed = parse_auxiliary_filename("MCD12Q1.A2024001.h21v07.061.2025205222320.hdf")
>>> parsed.product_id.name
'auxiliary_igbp_mcd12q1'
>>> parsed.observation_date.isoformat()
'2024-01-01'
>>> parsed.version
'061'
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from cloudpathlib import AnyPath

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import PathType

# Vendor timestamp formats.
# The MODIS/VIIRS "A-date" family stamps the acquisition day as an ordinal (YYYYDDD) and the
# production time as YYYYDDDHHMMSS; NISE uses a plain calendar day; CERES uses an hourly stamp.
_A_DATE_FORMAT = "%Y%j"
_A_PRODUCTION_FORMAT = "%Y%j%H%M%S"
_NISE_DATE_FORMAT = "%Y%m%d"
_CERES_DATE_FORMAT = "%Y%m%d%H"
_ERA5_DATE_FORMAT = "%Y%m%d"

# The two ERA5 members carry Libera-assigned canonical ids (no stable vendor filename). The ingest
# layer renames CDS granules to ``<value>_<YYYYMMDD>.nc`` using these; the parser reads them back.
_ERA5_PRODUCTS: frozenset[DataProductIdentifier] = frozenset(
    {
        DataProductIdentifier.auxiliary_era5_single_level,
        DataProductIdentifier.auxiliary_era5_pressure_level,
    }
)


@dataclass(frozen=True)
class AuxiliaryFilename:
    """Parsed identity and metadata for one external auxiliary granule.

    Attributes
    ----------
    product_id : DataProductIdentifier
        The ``auxiliary_*`` member this granule belongs to.
    observation_date : datetime.date
        The day the granule's data describes (the MODIS/VIIRS A-date, the NISE calendar day, or
        the CERES granule day; any sub-day hour is dropped).
    version : str | None
        The collection / version token when the name carries one (MODIS/VIIRS collection such as
        ``"061"``/``"002"``/``"011"``, or the CERES production version such as ``"alpha4"``);
        ``None`` otherwise (e.g. NISE).
    production_time : datetime.datetime | None
        The vendor's processing timestamp when the name carries one (the MODIS/VIIRS
        ``YYYYDDDHHMMSS`` field); ``None`` otherwise.
    source_variant : str | None
        The product-specific discriminator token when present: the MODIS sinusoidal tile
        (``"h21v07"``), the VIIRS/CERES platform (``"NOAA20"``, ``"NOAA20-FM6-VIIRS"``), the AERDB
        source (``"VIIRS_NOAA20"`` vs ``"GEOLEO_Merged"``), or the NISE sensor (``"AMSR2"``);
        ``None`` when the family has no such token.
    path : pathlib.Path | cloudpathlib.CloudPath
        The path the name was parsed from.

    Notes
    -----
    The FMATCH reader that consumes this product is *not* stored here; resolve it in the
    footprint-matching layer with
    :func:`~libera_utils.footprint_matching.readers.registry.reader_key_for_auxiliary_product`
    (``reader_key_for_auxiliary_product(parsed.product_id)``).
    """

    product_id: DataProductIdentifier
    observation_date: date
    version: str | None
    production_time: datetime | None
    source_variant: str | None
    path: PathType


# ---------------------------------------------------------------------------
# Per-convention regexes and field builders
# ---------------------------------------------------------------------------
# Each entry is (compiled regex, builder). A builder receives the regex match and returns the
# metadata kwargs (everything except ``path``) for an AuxiliaryFilename. Prefixes are mutually
# exclusive, so a name matches at most one pattern; the dispatcher returns the first match.

_MCD12Q1_REGEX = re.compile(
    r"^MCD12Q1\.A(?P<adate>\d{7})\.(?P<tile>h\d{2}v\d{2})\.(?P<coll>\d{3})\.(?P<prod>\d{13})\.hdf$"
)
_VJ143_REGEX = re.compile(r"^VJ143C(?P<band>[13])\.A(?P<adate>\d{7})\.(?P<coll>\d{3})\.(?P<prod>\d{13})\.h5$")
_CLDPROP_REGEX = re.compile(
    r"^CLDPROP_D3_VIIRS_(?P<platform>[A-Z0-9]+)\.A(?P<adate>\d{7})\.(?P<coll>\d{3})\.(?P<prod>\d{13})\.nc$"
)
_AERDB_REGEX = re.compile(
    r"^AERDB_D3_(?P<source>[A-Za-z0-9_]+)\.A(?P<adate>\d{7})\.(?P<coll>\d{3})\.(?P<prod>\d{13})\.nc$"
)
_NISE_REGEX = re.compile(r"^NISE_(?P<sensor>SSMISF\d{2}|AMSR2)_(?P<date>\d{8})\.HDFEOS$")
_CERES_REGEX = re.compile(
    r"^CER_(?P<kind>SSF|CLDPIX)_(?P<platform>[A-Za-z0-9-]+)_(?P<prodver>[^_]+)_(?P<cfg>\d+)\.(?P<date>\d{10})\.nc$"
)
# Libera-assigned canonical names applied by the ingest layer (see build_era5_canonical_filename).
_ERA5_REGEX = re.compile(r"^(?P<product>ERA5-SINGLE-LEVEL|ERA5-PRESSURE-LEVEL)_(?P<date>\d{8})\.nc$")


def _build_mcd12q1(m: re.Match[str]) -> dict:
    return {
        "product_id": DataProductIdentifier.auxiliary_igbp_mcd12q1,
        "observation_date": datetime.strptime(m["adate"], _A_DATE_FORMAT).date(),
        "version": m["coll"],
        "production_time": datetime.strptime(m["prod"], _A_PRODUCTION_FORMAT),
        "source_variant": m["tile"],
    }


def _build_vj143(m: re.Match[str]) -> dict:
    # Band 1 = BRDF/Albedo model parameters (VJ143C1); band 3 = derived albedo (VJ143C3).
    product_id = (
        DataProductIdentifier.auxiliary_viirs_brdf
        if m["band"] == "1"
        else DataProductIdentifier.auxiliary_viirs_brdf_albedo
    )
    return {
        "product_id": product_id,
        "observation_date": datetime.strptime(m["adate"], _A_DATE_FORMAT).date(),
        "version": m["coll"],
        "production_time": datetime.strptime(m["prod"], _A_PRODUCTION_FORMAT),
        "source_variant": None,
    }


def _build_cldprop(m: re.Match[str]) -> dict:
    return {
        "product_id": DataProductIdentifier.auxiliary_viirs_cloud,
        "observation_date": datetime.strptime(m["adate"], _A_DATE_FORMAT).date(),
        "version": m["coll"],
        "production_time": datetime.strptime(m["prod"], _A_PRODUCTION_FORMAT),
        "source_variant": m["platform"],
    }


def _build_aerdb(m: re.Match[str]) -> dict:
    # source_variant distinguishes the wanted single-sensor granule ("VIIRS_NOAA20") from the
    # cross-sensor merged one ("GEOLEO_Merged"); both are the same auxiliary product family.
    return {
        "product_id": DataProductIdentifier.auxiliary_viirs_aod,
        "observation_date": datetime.strptime(m["adate"], _A_DATE_FORMAT).date(),
        "version": m["coll"],
        "production_time": datetime.strptime(m["prod"], _A_PRODUCTION_FORMAT),
        "source_variant": m["source"],
    }


def _build_nise(m: re.Match[str]) -> dict:
    return {
        "product_id": DataProductIdentifier.auxiliary_nise,
        "observation_date": datetime.strptime(m["date"], _NISE_DATE_FORMAT).date(),
        "version": None,
        "production_time": None,
        "source_variant": m["sensor"],
    }


def _build_ceres(m: re.Match[str]) -> dict:
    product_id = (
        DataProductIdentifier.auxiliary_ceres_ssf
        if m["kind"] == "SSF"
        else DataProductIdentifier.auxiliary_ceres_cldpix
    )
    return {
        "product_id": product_id,
        # The CERES granule is hourly; keep the day for granule-by-date selection.
        "observation_date": datetime.strptime(m["date"], _CERES_DATE_FORMAT).date(),
        "version": m["prodver"],
        "production_time": None,
        "source_variant": m["platform"],
    }


def _build_era5(m: re.Match[str]) -> dict:
    # The matched token is exactly the DataProductIdentifier value, so it resolves straight through.
    return {
        "product_id": DataProductIdentifier(m["product"]),
        "observation_date": datetime.strptime(m["date"], _ERA5_DATE_FORMAT).date(),
        "version": None,
        "production_time": None,
        "source_variant": None,
    }


_PARSERS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], dict]], ...] = (
    (_MCD12Q1_REGEX, _build_mcd12q1),
    (_VJ143_REGEX, _build_vj143),
    (_CLDPROP_REGEX, _build_cldprop),
    (_AERDB_REGEX, _build_aerdb),
    (_NISE_REGEX, _build_nise),
    (_CERES_REGEX, _build_ceres),
    (_ERA5_REGEX, _build_era5),
)


def build_era5_canonical_filename(product_id: DataProductIdentifier, observation_date: date) -> str:
    """Build the Libera-assigned canonical filename for an ERA5 granule.

    This is the single source of truth for the ERA5 canonical naming contract: the ingest layer
    renames raw Copernicus CDS granules to the string returned here, and
    :func:`parse_auxiliary_filename` reads it back, so the two cannot drift.

    Parameters
    ----------
    product_id : DataProductIdentifier
        Either :attr:`~libera_utils.constants.DataProductIdentifier.auxiliary_era5_single_level`
        or :attr:`~libera_utils.constants.DataProductIdentifier.auxiliary_era5_pressure_level`.
    observation_date : datetime.date
        The day the granule describes.

    Returns
    -------
    str
        The canonical filename, e.g. ``"ERA5-SINGLE-LEVEL_20260327.nc"``.

    Raises
    ------
    ValueError
        If ``product_id`` is not one of the two ERA5 auxiliary products.
    """
    if product_id not in _ERA5_PRODUCTS:
        raise ValueError(
            f"{product_id!r} is not an ERA5 auxiliary product; "
            f"expected one of {sorted(p.name for p in _ERA5_PRODUCTS)}."
        )
    return f"{product_id.value}_{observation_date.strftime(_ERA5_DATE_FORMAT)}.nc"


def parse_auxiliary_filename(path: str | PathType) -> AuxiliaryFilename:
    """Parse an external auxiliary granule filename into an :class:`AuxiliaryFilename`.

    Only the basename is inspected, so either a bare filename or a full local/cloud path works.

    Parameters
    ----------
    path : str | pathlib.Path | cloudpathlib.CloudPath
        The granule path or filename to parse.

    Returns
    -------
    AuxiliaryFilename
        The parsed identity and metadata.

    Raises
    ------
    ValueError
        If the filename matches none of the known auxiliary product conventions. Libera product
        files, raw ERA5 CDS granules, and unrelated files all raise here; callers that want to
        skip unrecognized files should use :func:`try_parse_auxiliary_filename`.
    """
    resolved = AnyPath(path)
    name = resolved.name
    for regex, build in _PARSERS:
        match = regex.match(name)
        if match:
            return AuxiliaryFilename(path=resolved, **build(match))
    raise ValueError(
        f"Filename {name!r} does not match any known auxiliary product convention "
        f"(MCD12Q1, VJ143C1/C3, CLDPROP_D3_VIIRS, AERDB_D3, NISE, CER_SSF/CER_CLDPIX, "
        f"ERA5-SINGLE-LEVEL/ERA5-PRESSURE-LEVEL). Raw ERA5 CDS names are renamed at staging."
    )


def try_parse_auxiliary_filename(path: str | PathType) -> AuxiliaryFilename | None:
    """Parse an auxiliary granule filename, returning ``None`` instead of raising on no match.

    The non-raising companion to :func:`parse_auxiliary_filename`, for scanning a manifest that
    legitimately mixes auxiliary granules with Libera products and other files.

    Parameters
    ----------
    path : str | pathlib.Path | cloudpathlib.CloudPath
        The granule path or filename to parse.

    Returns
    -------
    AuxiliaryFilename | None
        The parsed result, or ``None`` if the filename is not a recognized auxiliary product.
    """
    try:
        return parse_auxiliary_filename(path)
    except ValueError:
        return None


__all__ = [
    "AuxiliaryFilename",
    "build_era5_canonical_filename",
    "parse_auxiliary_filename",
    "try_parse_auxiliary_filename",
]
