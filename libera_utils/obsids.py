"""ICIE software ObsID catalog for NOM-HK trimming and calibration pipelines.

Radiometer and camera ObsID numeric values are not globally unique: the same
integer can mean different events depending on whether it appears in
``ICIE__SW_OBSID_RAD`` or ``ICIE__SW_OBSID_WFOV``. Registry keys are therefore
``(NomHkObsidSource, obsid)``.

Several ObsIDs share one TRIMMED product. A TRIMMED product names a *calibration
dependency family* — ``NOM-HK-SWC-FAMILY-TRIMMED`` covers all six shortwave LED
ObsIDs — and one processing step is deployed per family, not per ObsID. The trimmed
file carries the source NOM-HK ObsID field, so a consumer recovers the exact ObsID
from the data rather than from the ProductID. Each ObsID keeps its own CAL product,
which is what a family step dispatches on.

The catalog itself lives in :data:`OBSID_REGISTRY_CSV` (``libera_utils/data``)
rather than in this module. Product columns hold
:class:`~libera_utils.constants.DataProductIdentifier` *member names* (e.g.
``cal_gain``), which are resolved and validated when this module is imported.
An unknown member name, a product named at the wrong data level, a ``kind`` that
disagrees with ``source``, a duplicate ``(source, obsid)`` key, a CAL product claimed
by more than one ObsID, or a TRIMMED family spanning more than one NOM-HK ObsID field
all raise :class:`ValueError` at import time.

A companion file, :data:`TRIM_FAMILY_INPUTS_CSV`, records the L1A products each family's
cal step consumes *besides* its own TRIMMED product, exposed as :data:`FAMILY_INPUTS` /
:func:`get_family_inputs`. The two files must cover exactly the same set of families, which
is checked at import; an empty ``required_inputs`` cell means the dependency set is still
undecided. A cal step is expected to take its family's NOM-HK already trimmed as the family
product, so ``required_inputs`` normally names only the other APIDs. Trimming is not applied
to those inputs here: a cal container reads its run's time range off the TRIMMED NOM-HK
filename it was handed and subsets the full daily L1A products itself.

The list of ObsIDs in this repo is meant for practical purposes of science data
processing and is a subset of the instrument level source of truth of all ObsIDs, which
is owned by the engineering team and available in internal team documentation. That
documentation is the authority: when calibration ObsIDs are added, removed, or renamed
upstream, this registry has to be reconciled against it.
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files

from libera_utils.constants import DataLevel, DataProductIdentifier

#: Package resource holding the ObsID catalog.
OBSID_REGISTRY_CSV = files("libera_utils.data") / "obsid_registry.csv"

#: Package resource holding the per-family calibration input catalog.
TRIM_FAMILY_INPUTS_CSV = files("libera_utils.data") / "trim_family_inputs.csv"


class NomHkObsidSource(StrEnum):
    """Which NOM-HK ObsID variable to use for run detection / trimming."""

    RAD = "ICIE__SW_OBSID_RAD"
    WFOV = "ICIE__SW_OBSID_WFOV"


class ObsIdKind(StrEnum):
    """Category of a known ObsID entry."""

    RAD_CAL = "rad_cal"
    CAM_CAL = "cam_cal"
    SCIENCE = "science"


#: NOM-HK ObsID field each calibration kind must be registered against. The camera cal ObsIDs
#: (CT video 129-131, RAPS video 133-135, darks 256-258) are asserted only on
#: ``ICIE__SW_OBSID_WFOV``, so ``get_obsid_spec(RAD, 129)`` raises KeyError.
_SOURCE_BY_CAL_KIND = {
    ObsIdKind.RAD_CAL: NomHkObsidSource.RAD,
    ObsIdKind.CAM_CAL: NomHkObsidSource.WFOV,
}

#: Kinds that must carry both a TRIMMED and a CAL ProductID.
_CAL_KINDS = tuple(_SOURCE_BY_CAL_KIND)

#: Columns every ObsID catalog row must provide.
_COLUMNS = ("source", "obsid", "kind", "trimmed_product", "cal_product", "description")

#: Columns every family-inputs catalog row must provide.
_FAMILY_INPUT_COLUMNS = ("trimmed_product", "required_inputs")

#: Separator between product names inside the ``required_inputs`` cell.
_INPUT_SEPARATOR = ";"

#: Key that :class:`csv.DictReader` parks surplus cells under (a row with too many columns).
_EXTRA_COLUMNS_KEY = "_extra_columns"


@dataclass(frozen=True)
class ObsIdSpec:
    """One known ICIE software ObsID and its product / telemetry binding."""

    obsid: int
    source: NomHkObsidSource
    kind: ObsIdKind
    description: str
    trimmed_product: DataProductIdentifier | None
    cal_product: DataProductIdentifier | None


def _resolve_product(
    name: str,
    column: str,
    level: DataLevel,
    line: int,
    catalog: str = OBSID_REGISTRY_CSV.name,
) -> DataProductIdentifier | None:
    """Resolve a DataProductIdentifier member name from a catalog cell.

    Parameters
    ----------
    name : str
        Member name of :class:`~libera_utils.constants.DataProductIdentifier`,
        or an empty cell for entries with no such product.
    column : str
        Column name, used in error messages.
    level : DataLevel
        Data level the resolved product must be at, so a value in the wrong
        column (e.g. a CAL ProductID under ``trimmed_product``) is rejected.
    line : int
        Line number in the catalog, used in error messages.
    catalog : str
        Name of the catalog file being read, used in error messages.

    Returns
    -------
    DataProductIdentifier or None
        The resolved ProductID, or None if the cell was empty.

    Raises
    ------
    ValueError
        If the cell is non-empty but names no DataProductIdentifier member, or
        names one at the wrong data level.
    """
    name = name.strip()
    if not name:
        return None
    try:
        product = DataProductIdentifier[name]
    except KeyError as exc:
        raise ValueError(
            f"{catalog}:{line} column {column!r}: {name!r} is not a DataProductIdentifier member name"
        ) from exc
    if product.data_level is not level:
        raise ValueError(
            f"{catalog}:{line} column {column!r}: {name!r} is a {product.data_level} product, expected {level}"
        )
    return product


def _parse_row(row: dict[str, str], line: int) -> ObsIdSpec:
    """Build an ObsIdSpec from one catalog row.

    Parameters
    ----------
    row : dict
        Mapping of column name to raw cell value.
    line : int
        Line number in the catalog, used in error messages.

    Returns
    -------
    ObsIdSpec
        Validated catalog entry.

    Raises
    ------
    ValueError
        If the row has the wrong number of columns, if any cell fails to parse, or
        if the products or ``source`` are inconsistent with ``kind``.
    """
    where = f"{OBSID_REGISTRY_CSV.name}:{line}"
    missing = [column for column in _COLUMNS if row.get(column) is None]
    if missing or row.get(_EXTRA_COLUMNS_KEY):
        raise ValueError(
            f"{where}: expected exactly {len(_COLUMNS)} columns "
            f"({', '.join(_COLUMNS)}); missing {missing}, surplus {row.get(_EXTRA_COLUMNS_KEY) or []}. "
            "Quote any cell containing a comma."
        )
    try:
        source = NomHkObsidSource[row["source"].strip()]
    except KeyError as exc:
        raise ValueError(f"{where} column 'source': {row['source']!r} is not one of RAD, WFOV") from exc
    try:
        kind = ObsIdKind(row["kind"].strip())
    except ValueError as exc:
        kinds = ", ".join(k.value for k in ObsIdKind)
        raise ValueError(f"{where} column 'kind': {row['kind']!r} is not one of {kinds}") from exc
    try:
        obsid = int(row["obsid"])
    except ValueError as exc:
        raise ValueError(f"{where} column 'obsid': {row['obsid']!r} is not an integer") from exc

    spec = ObsIdSpec(
        obsid=obsid,
        source=source,
        kind=kind,
        description=row["description"].strip(),
        trimmed_product=_resolve_product(row["trimmed_product"], "trimmed_product", DataLevel.L1A, line),
        cal_product=_resolve_product(row["cal_product"], "cal_product", DataLevel.CAL, line),
    )

    if kind is ObsIdKind.SCIENCE and (spec.trimmed_product or spec.cal_product):
        raise ValueError(f"{where}: science ObsID {obsid} must not name TRIMMED or CAL products")
    if kind in _CAL_KINDS:
        if not (spec.trimmed_product and spec.cal_product):
            raise ValueError(f"{where}: {kind.value} ObsID {obsid} must name both TRIMMED and CAL products")
        expected_source = _SOURCE_BY_CAL_KIND[kind]
        if source is not expected_source:
            raise ValueError(
                f"{where}: {kind.value} ObsID {obsid} must be registered on {expected_source.name}, got {source.name}"
            )
    return spec


def _load_registry() -> tuple[
    dict[tuple[NomHkObsidSource, int], ObsIdSpec],
    dict[DataProductIdentifier, tuple[ObsIdSpec, ...]],
]:
    """Read and validate the ObsID catalog CSV.

    Returns
    -------
    registry : dict
        Mapping of ``(source, obsid)`` to :class:`ObsIdSpec`, in catalog order.
    families : dict
        Mapping of each TRIMMED family ProductID to the ObsIDs that belong to it, in
        catalog order.

    Raises
    ------
    ValueError
        If the catalog is malformed, if two rows share a ``(source, obsid)`` key, if two
        rows claim the same CAL product, or if one TRIMMED family spans both NOM-HK ObsID
        fields.
    """
    registry: dict[tuple[NomHkObsidSource, int], ObsIdSpec] = {}
    families: dict[DataProductIdentifier, list[ObsIdSpec]] = {}
    cal_owner: dict[DataProductIdentifier, ObsIdSpec] = {}
    with OBSID_REGISTRY_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, restkey=_EXTRA_COLUMNS_KEY)
        missing = set(_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{OBSID_REGISTRY_CSV.name} is missing column(s): {', '.join(sorted(missing))}")
        for row in reader:
            spec = _parse_row(row, reader.line_num)
            key = (spec.source, spec.obsid)
            if key in registry:
                raise ValueError(
                    f"{OBSID_REGISTRY_CSV.name}:{reader.line_num}: "
                    f"duplicate entry for ObsID {spec.obsid} on {spec.source.name}"
                )
            if spec.cal_product is not None:
                owner = cal_owner.get(spec.cal_product)
                if owner is not None:
                    raise ValueError(
                        f"{OBSID_REGISTRY_CSV.name}:{reader.line_num}: CAL product "
                        f"{spec.cal_product.name!r} is already claimed by ObsID {owner.obsid} on "
                        f"{owner.source.name}. Each ObsID needs its own CAL product."
                    )
                cal_owner[spec.cal_product] = spec
            if spec.trimmed_product is not None:
                members = families.setdefault(spec.trimmed_product, [])
                if members and members[0].source is not spec.source:
                    raise ValueError(
                        f"{OBSID_REGISTRY_CSV.name}:{reader.line_num}: TRIMMED family "
                        f"{spec.trimmed_product.name!r} is registered on {members[0].source.name} by ObsID "
                        f"{members[0].obsid} but on {spec.source.name} by ObsID {spec.obsid}. A family is "
                        "trimmed by scanning a single NOM-HK ObsID field; register a separate family "
                        "per source."
                    )
                members.append(spec)
            registry[key] = spec
    return registry, {product: tuple(members) for product, members in families.items()}


def _load_family_inputs(
    families: Collection[DataProductIdentifier],
) -> dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]]:
    """Read and validate the per-family calibration input catalog.

    Parameters
    ----------
    families : collection of DataProductIdentifier
        TRIMMED family ProductIDs registered in :data:`OBSID_REGISTRY_CSV`. The two catalogs
        must cover exactly the same families.

    Returns
    -------
    dict
        Mapping of each TRIMMED family ProductID to the L1A products a cal step for that family
        consumes, in catalog order. An empty tuple means the dependency set has not been settled
        yet.

    Raises
    ------
    ValueError
        If the catalog is malformed, if a family is listed twice, if a cell names something
        other than an L1A DataProductIdentifier member, or if the families listed here do not
        exactly match ``families``.
    """
    inputs: dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]] = {}
    with TRIM_FAMILY_INPUTS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, restkey=_EXTRA_COLUMNS_KEY)
        missing_columns = set(_FAMILY_INPUT_COLUMNS).difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                f"{TRIM_FAMILY_INPUTS_CSV.name} is missing column(s): {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            where = f"{TRIM_FAMILY_INPUTS_CSV.name}:{reader.line_num}"
            missing = [column for column in _FAMILY_INPUT_COLUMNS if row.get(column) is None]
            if missing or row.get(_EXTRA_COLUMNS_KEY):
                raise ValueError(
                    f"{where}: expected exactly {len(_FAMILY_INPUT_COLUMNS)} columns "
                    f"({', '.join(_FAMILY_INPUT_COLUMNS)}); missing {missing}, "
                    f"surplus {row.get(_EXTRA_COLUMNS_KEY) or []}."
                )
            family = _resolve_product(
                row["trimmed_product"],
                "trimmed_product",
                DataLevel.L1A,
                reader.line_num,
                TRIM_FAMILY_INPUTS_CSV.name,
            )
            if family is None:
                raise ValueError(f"{where} column 'trimmed_product': a family ProductID is required")
            if family in inputs:
                raise ValueError(f"{where}: duplicate entry for TRIMMED family {family.name!r}")
            required = tuple(
                product
                for cell in row["required_inputs"].split(_INPUT_SEPARATOR)
                if (
                    product := _resolve_product(
                        cell, "required_inputs", DataLevel.L1A, reader.line_num, TRIM_FAMILY_INPUTS_CSV.name
                    )
                )
                is not None
            )
            inputs[family] = required

    listed = set(inputs)
    registered = set(families)
    if listed != registered:
        undeclared = sorted(product.name for product in registered - listed)
        unregistered = sorted(product.name for product in listed - registered)
        raise ValueError(
            f"{TRIM_FAMILY_INPUTS_CSV.name} must list exactly the TRIMMED families registered in "
            f"{OBSID_REGISTRY_CSV.name}. Families with no declared inputs: {undeclared}. "
            f"Families declared here but registered against no ObsID: {unregistered}. "
            "Add the family to both files (an empty 'required_inputs' cell is allowed while the "
            "dependency set is still undecided)."
        )
    return inputs


#: Sole source of truth for ObsID → CAL / TRIMMED ProductIDs and catalog metadata.
#: Keyed by (source, obsid) because RAD and WFOV namespaces overlap.
#: Loaded from :data:`OBSID_REGISTRY_CSV`; edit that file to register a new ObsID.
OBSID_REGISTRY: dict[tuple[NomHkObsidSource, int], ObsIdSpec]

#: Inverse view of the TRIMMED column: each calibration dependency family ProductID mapped to the
#: ObsIDs it covers. This is the membership a downstream cal step dispatches over.
TRIM_FAMILIES: dict[DataProductIdentifier, tuple[ObsIdSpec, ...]]

#: L1A products each calibration dependency family's cal step consumes *besides* its own TRIMMED
#: product, loaded from :data:`TRIM_FAMILY_INPUTS_CSV`. Together with the family product these are
#: the dependency set a family step is deployed against; an empty tuple means it has not been
#: settled yet. Cal containers subset these daily products themselves using the time range on the
#: TRIMMED NOM-HK filename they are handed.
FAMILY_INPUTS: dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]]

OBSID_REGISTRY, TRIM_FAMILIES = _load_registry()
FAMILY_INPUTS = _load_family_inputs(TRIM_FAMILIES)


def get_obsid_spec(source: NomHkObsidSource, obsid: int) -> ObsIdSpec:
    """Return the registry entry for ``(source, obsid)``.

    Parameters
    ----------
    source : NomHkObsidSource
        NOM-HK ObsID field that owns this ObsID namespace.
    obsid : int
        Software ObsID value.

    Returns
    -------
    ObsIdSpec
        Matching catalog entry.

    Raises
    ------
    KeyError
        If the pair is not in :data:`OBSID_REGISTRY`.
    """
    try:
        return OBSID_REGISTRY[(source, obsid)]
    except KeyError as exc:
        raise KeyError(f"Unknown ObsID {obsid} for source {source.name} ({source.value})") from exc


def get_family_specs(trimmed_product: DataProductIdentifier) -> tuple[ObsIdSpec, ...]:
    """Return every ObsID belonging to a TRIMMED calibration dependency family.

    A TRIMMED ProductID covers a family of ObsIDs that downstream algorithms process the same
    way (``NOM-HK-SWC-FAMILY-TRIMMED`` covers all six shortwave LED ObsIDs), so this is how a
    cal step enumerates the ObsIDs — and the CAL products — it is responsible for.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A TRIMMED family ProductID from the registry.

    Returns
    -------
    tuple of ObsIdSpec
        Family members in catalog order. Always at least one entry.

    Raises
    ------
    KeyError
        If ``trimmed_product`` is not a registered TRIMMED family ProductID.
    """
    try:
        return TRIM_FAMILIES[trimmed_product]
    except KeyError as exc:
        raise KeyError(
            f"{trimmed_product.name!r} is not a TRIMMED calibration family ProductID in {OBSID_REGISTRY_CSV.name}"
        ) from exc


def get_family_inputs(trimmed_product: DataProductIdentifier) -> tuple[DataProductIdentifier, ...]:
    """Return the L1A products a calibration dependency family's cal step consumes.

    These are the *other* L1A products a family's algorithm needs: a step's full
    ``input-products`` set is ``trimmed_product`` itself plus this tuple. The family's NOM-HK is
    expected to arrive already trimmed as ``trimmed_product``, so the full-day ``NOM-HK-DECODED``
    granule is normally not listed here. The cal container subsets these to the calibration run
    itself, using the time range carried by the TRIMMED NOM-HK filename it was handed; no other
    APID is trimmed by the L1A preprocessor.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A TRIMMED family ProductID from the registry.

    Returns
    -------
    tuple of DataProductIdentifier
        L1A input products in catalog order. Empty for families whose dependency set is still
        undecided.

    Raises
    ------
    KeyError
        If ``trimmed_product`` is not a registered TRIMMED family ProductID.
    """
    try:
        return FAMILY_INPUTS[trimmed_product]
    except KeyError as exc:
        raise KeyError(
            f"{trimmed_product.name!r} is not a TRIMMED calibration family ProductID in {OBSID_REGISTRY_CSV.name}"
        ) from exc


def iter_trim_eligible(source: NomHkObsidSource | None = None) -> Iterator[ObsIdSpec]:
    """Yield registry entries that produce TRIMMED NOM-HK products.

    Parameters
    ----------
    source : NomHkObsidSource or None
        If set, only yield entries for that NOM-HK ObsID field.

    Yields
    ------
    ObsIdSpec
        Entries with a non-null ``trimmed_product``.
    """
    for spec in OBSID_REGISTRY.values():
        if spec.trimmed_product is None:
            continue
        if source is not None and spec.source is not source:
            continue
        yield spec
