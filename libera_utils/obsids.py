"""ICIE software ObsID catalog for NOM-HK trimming and calibration pipelines.

Radiometer and camera ObsID numeric values are not globally unique: the same
integer can mean different events depending on whether it appears in
``ICIE__SW_OBSID_RAD`` or ``ICIE__SW_OBSID_WFOV``. Registry keys are therefore
``(NomHkObsidSource, obsid)``.

Each calibration ObsID has its own TRIMMED NOM-HK product, its own CAL product and one
combiner processing step that turns the first into the second, so the three are 1:1:1.
A TRIMMED or CAL file holds one calibration event: an ObsID can occur more than once a
day, and each occurrence is its own file, told apart by the time range in its filename.

The catalog itself lives in ``OBSID_REGISTRY_CSV`` (``libera_utils/data``)
rather than in this module. Product columns hold
:class:`~libera_utils.constants.DataProductIdentifier` *member names* (e.g.
``cal_gain``), which are resolved and validated when this module is imported.
An unknown member name, a product named at the wrong data level, a ``kind`` that
disagrees with ``source``, a duplicate ``(source, obsid)`` key, or a TRIMMED or CAL
product claimed by more than one ObsID all raise :class:`ValueError` at import time, as
does a CAL product that no :class:`~libera_utils.constants.ProcessingStepIdentifier`
produces.

A companion file, ``CAL_COMBINER_INPUTS_CSV``, records the L1A products each combiner
consumes *besides* its own TRIMMED product, keyed by that TRIMMED product and exposed as
:data:`COMBINER_INPUTS` / :func:`get_combiner_inputs`. It must list exactly the TRIMMED
products registered here, which is checked at import; an empty ``required_inputs`` cell
means the dependency set is still undecided. ``required_inputs`` normally omits the
full-day NOM-HK, since the combiner gets its NOM-HK already trimmed. Trimming is not
applied to those inputs here: a combiner reads its event's time range off the TRIMMED
NOM-HK filename it was handed and subsets the full daily L1A products itself.

ObsIDs whose combiners take the same inputs, such as the six shortwave LED ObsIDs, are
often called a calibration family. A family is a way to talk about calibration events;
it is neither a product nor a processing step, and no file here records it.

The list of ObsIDs in this repo is meant for practical purposes of science data
processing and is a subset of the instrument level source of truth of all ObsIDs, which
is owned by the engineering team and available in internal team documentation. That
documentation is the authority: when calibration ObsIDs are added, removed, or renamed
upstream, this registry has to be reconciled against it.
"""

from __future__ import annotations

import csv
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files
from typing import TypeVar

from libera_utils.constants import DataLevel, DataProductIdentifier, ProcessingStepIdentifier

_T = TypeVar("_T")

#: Package resource holding the ObsID catalog.
OBSID_REGISTRY_CSV = files("libera_utils.data") / "obsid_registry.csv"

#: Package resource holding the calibration combiner input catalog.
CAL_COMBINER_INPUTS_CSV = files("libera_utils.data") / "cal_combiner_inputs.csv"


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

#: Columns every combiner-inputs catalog row must provide.
_COMBINER_INPUT_COLUMNS = ("trimmed_product", "required_inputs")

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
    dict[DataProductIdentifier, ObsIdSpec],
]:
    """Read and validate the ObsID catalog CSV.

    Returns
    -------
    registry : dict
        Mapping of ``(source, obsid)`` to :class:`ObsIdSpec`, in catalog order.
    by_trimmed : dict
        Mapping of each TRIMMED ProductID to the one ObsID that owns it, in catalog order.

    Raises
    ------
    ValueError
        If the catalog is malformed, if two rows share a ``(source, obsid)`` key, or if two
        rows claim the same TRIMMED or CAL product.
    """
    registry: dict[tuple[NomHkObsidSource, int], ObsIdSpec] = {}
    by_trimmed: dict[DataProductIdentifier, ObsIdSpec] = {}
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
            for product, owners, column in (
                (spec.cal_product, cal_owner, "CAL"),
                (spec.trimmed_product, by_trimmed, "TRIMMED"),
            ):
                if product is None:
                    continue
                owner = owners.get(product)
                if owner is not None:
                    raise ValueError(
                        f"{OBSID_REGISTRY_CSV.name}:{reader.line_num}: {column} product "
                        f"{product.name!r} is already claimed by ObsID {owner.obsid} on "
                        f"{owner.source.name}. Each ObsID needs its own {column} product."
                    )
                owners[product] = spec
            registry[key] = spec
    return registry, by_trimmed


def _combiner_steps(
    by_trimmed: dict[DataProductIdentifier, ObsIdSpec],
) -> dict[DataProductIdentifier, ProcessingStepIdentifier]:
    """Map each TRIMMED ProductID to the processing step that produces its ObsID's CAL product.

    Parameters
    ----------
    by_trimmed : dict
        Mapping of each TRIMMED ProductID to the ObsID that owns it.

    Returns
    -------
    dict
        Mapping of each TRIMMED ProductID to its combiner step, in the order of ``by_trimmed``.

    Raises
    ------
    ValueError
        If no :class:`~libera_utils.constants.ProcessingStepIdentifier` produces an ObsID's CAL
        product.
    """
    steps: dict[DataProductIdentifier, ProcessingStepIdentifier] = {}
    for trimmed, spec in by_trimmed.items():
        step = ProcessingStepIdentifier.from_data_product(spec.cal_product)
        if step is None:
            raise ValueError(
                f"No ProcessingStepIdentifier produces {spec.cal_product.name!r}, the CAL product of ObsID "
                f"{spec.obsid} on {spec.source.name}. Add a cal_combine_* step for it in libera_utils/constants.py."
            )
        steps[trimmed] = step
    return steps


def _load_combiner_inputs(
    trimmed_products: Collection[DataProductIdentifier],
) -> dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]]:
    """Read and validate the calibration combiner input catalog.

    Parameters
    ----------
    trimmed_products : collections.abc.Collection[DataProductIdentifier]
        TRIMMED ProductIDs registered in ``OBSID_REGISTRY_CSV``. The two catalogs must cover
        exactly the same TRIMMED products.

    Returns
    -------
    dict
        Mapping of each TRIMMED ProductID to the L1A products its combiner consumes besides it, in
        catalog order. An empty tuple means the dependency set has not been settled yet.

    Raises
    ------
    ValueError
        If the catalog is malformed, if a TRIMMED product is listed twice, if a cell names something
        other than an L1A DataProductIdentifier member, or if the TRIMMED products listed here do not
        exactly match ``trimmed_products``.
    """
    catalog = CAL_COMBINER_INPUTS_CSV.name
    inputs: dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]] = {}
    with CAL_COMBINER_INPUTS_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, restkey=_EXTRA_COLUMNS_KEY)
        missing_columns = set(_COMBINER_INPUT_COLUMNS).difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{catalog} is missing column(s): {', '.join(sorted(missing_columns))}")
        for row in reader:
            where = f"{catalog}:{reader.line_num}"
            missing = [column for column in _COMBINER_INPUT_COLUMNS if row.get(column) is None]
            if missing or row.get(_EXTRA_COLUMNS_KEY):
                raise ValueError(
                    f"{where}: expected exactly {len(_COMBINER_INPUT_COLUMNS)} columns "
                    f"({', '.join(_COMBINER_INPUT_COLUMNS)}); missing {missing}, "
                    f"surplus {row.get(_EXTRA_COLUMNS_KEY) or []}."
                )
            trimmed = _resolve_product(
                row["trimmed_product"], "trimmed_product", DataLevel.L1A, reader.line_num, catalog
            )
            if trimmed is None:
                raise ValueError(f"{where} column 'trimmed_product': a TRIMMED ProductID is required")
            if trimmed in inputs:
                raise ValueError(f"{where}: duplicate entry for TRIMMED product {trimmed.name!r}")
            inputs[trimmed] = tuple(
                product
                for cell in row["required_inputs"].split(_INPUT_SEPARATOR)
                if (product := _resolve_product(cell, "required_inputs", DataLevel.L1A, reader.line_num, catalog))
                is not None
            )

    listed = set(inputs)
    registered = set(trimmed_products)
    if listed != registered:
        undeclared = sorted(product.name for product in registered - listed)
        unregistered = sorted(product.name for product in listed - registered)
        raise ValueError(
            f"{catalog} must list exactly the TRIMMED products registered in {OBSID_REGISTRY_CSV.name}. "
            f"TRIMMED products with no row here: {undeclared}. "
            f"Rows here for TRIMMED products registered against no ObsID: {unregistered}. "
            "Add the TRIMMED product to both files (an empty 'required_inputs' cell is allowed while the "
            "dependency set is still undecided)."
        )
    return inputs


#: Sole source of truth for ObsID → CAL / TRIMMED ProductIDs and catalog metadata.
#: Keyed by (source, obsid) because RAD and WFOV namespaces overlap.
#: Loaded from ``OBSID_REGISTRY_CSV``; edit that file to register a new ObsID.
OBSID_REGISTRY: dict[tuple[NomHkObsidSource, int], ObsIdSpec]

#: Each TRIMMED ProductID mapped to the one ObsID that owns it.
_SPEC_BY_TRIMMED: dict[DataProductIdentifier, ObsIdSpec]

#: L1A products each calibration combiner consumes *besides* its own TRIMMED product, keyed by
#: that TRIMMED product and loaded from ``CAL_COMBINER_INPUTS_CSV``. Together with the TRIMMED
#: product these are the input products a combiner step is deployed against; an empty tuple
#: means they have not been settled yet. Combiners subset these daily products themselves using
#: the time range on the TRIMMED NOM-HK filename they are handed.
COMBINER_INPUTS: dict[DataProductIdentifier, tuple[DataProductIdentifier, ...]]

OBSID_REGISTRY, _SPEC_BY_TRIMMED = _load_registry()
_COMBINER_BY_TRIMMED = _combiner_steps(_SPEC_BY_TRIMMED)
COMBINER_INPUTS = _load_combiner_inputs(_SPEC_BY_TRIMMED)

#: TRIMMED and CAL ProductIDs of every calibration ObsID: the products whose files each hold one
#: calibration event.
_CALIBRATION_EVENT_PRODUCTS = frozenset(_SPEC_BY_TRIMMED) | frozenset(
    spec.cal_product for spec in _SPEC_BY_TRIMMED.values()
)


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


def _by_trimmed(table: Mapping[DataProductIdentifier, _T], trimmed_product: DataProductIdentifier) -> _T:
    """Look up a TRIMMED ProductID in one of the registry's tables keyed by TRIMMED product.

    Parameters
    ----------
    table : collections.abc.Mapping
        A mapping keyed by every TRIMMED ProductID in the registry.
    trimmed_product : DataProductIdentifier
        The TRIMMED ProductID to look up.

    Returns
    -------
    object
        The value ``table`` holds for ``trimmed_product``.

    Raises
    ------
    KeyError
        If ``trimmed_product`` is not a TRIMMED ProductID in the registry.
    """
    try:
        return table[trimmed_product]
    except KeyError as exc:
        raise KeyError(f"{trimmed_product.name!r} is not a TRIMMED ProductID in {OBSID_REGISTRY_CSV.name}") from exc


def get_trimmed_obsid_spec(trimmed_product: DataProductIdentifier) -> ObsIdSpec:
    """Return the registry entry of the ObsID that owns a TRIMMED ProductID.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A TRIMMED NOM-HK ProductID from the registry.

    Returns
    -------
    ObsIdSpec
        The one catalog entry whose ``trimmed_product`` is ``trimmed_product``.

    Raises
    ------
    KeyError
        If ``trimmed_product`` is not a TRIMMED ProductID in the registry.
    """
    return _by_trimmed(_SPEC_BY_TRIMMED, trimmed_product)


def get_combiner_step(trimmed_product: DataProductIdentifier) -> ProcessingStepIdentifier:
    """Return the combiner processing step that consumes a TRIMMED ProductID.

    The step produces the CAL product of the ObsID that owns ``trimmed_product``, and nothing else.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A TRIMMED NOM-HK ProductID from the registry.

    Returns
    -------
    ProcessingStepIdentifier
        The ``cal_combine_*`` step for that ObsID.

    Raises
    ------
    KeyError
        If ``trimmed_product`` is not a TRIMMED ProductID in the registry.
    """
    return _by_trimmed(_COMBINER_BY_TRIMMED, trimmed_product)


def get_combiner_inputs(trimmed_product: DataProductIdentifier) -> tuple[DataProductIdentifier, ...]:
    """Return the L1A products a calibration combiner consumes besides its TRIMMED product.

    A combiner step's full input product set is ``trimmed_product`` itself plus this tuple. Its
    NOM-HK arrives already trimmed as ``trimmed_product``, so the full-day ``NOM-HK-DECODED``
    granule is normally not listed here. The combiner subsets these to the calibration event
    itself, using the time range carried by the TRIMMED NOM-HK filename it was handed.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A TRIMMED NOM-HK ProductID from the registry.

    Returns
    -------
    tuple of DataProductIdentifier
        L1A input products in catalog order. Empty while the dependency set is still undecided.

    Raises
    ------
    KeyError
        If ``trimmed_product`` is not a TRIMMED ProductID in the registry.
    """
    return _by_trimmed(COMBINER_INPUTS, trimmed_product)


def is_calibration_event_product(product: DataProductIdentifier) -> bool:
    """Return whether a ProductID is a calibration ObsID's TRIMMED or CAL product.

    Each file of such a product holds one calibration event rather than one day.

    Parameters
    ----------
    product : DataProductIdentifier
        Any ProductID.

    Returns
    -------
    bool
        True if ``product`` is the TRIMMED or CAL product of an ObsID in the registry.
    """
    return product in _CALIBRATION_EVENT_PRODUCTS


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
