"""Trim decoded NOM-HK L1A Datasets to contiguous calibration ObsID runs."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import xarray as xr

from libera_utils.constants import DataProductIdentifier, LiberaApid
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path
from libera_utils.obsids import NomHkObsidSource, ObsIdSpec, iter_trim_eligible

logger = logging.getLogger(__name__)

PACKET_DIM = "PACKET"
DEFAULT_TIME_VARIABLE = "PACKET_ICIE_TIME"


def get_trimmed_nom_hk_product_definition(
    trimmed_product: DataProductIdentifier,
) -> LiberaDataProductDefinition:
    """Load the NOM-HK L1A product definition with ``ProductID`` set to a TRIMMED DPI.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A known TRIMMED NOM-HK ProductID from the ObsID registry.

    Returns
    -------
    LiberaDataProductDefinition
        Product definition identical to ``NOM-HK-DECODED`` except for ``ProductID``.

    Raises
    ------
    ValueError
        If ``trimmed_product`` is not a trim-eligible registry ProductID.
    """
    known = {spec.trimmed_product for spec in iter_trim_eligible()}
    if trimmed_product not in known:
        raise ValueError(
            f"ProductID {trimmed_product.value!r} is not a known TRIMMED NOM-HK product. "
            f"Expected one of: {sorted(p.value for p in known if p is not None)}"
        )

    base_path = get_l1a_product_definition_path(LiberaApid.icie_nom_hk.value)
    base = LiberaDataProductDefinition.from_yaml(base_path)
    updated_attrs = {**base.attributes, "ProductID": trimmed_product.value}
    return base.model_copy(update={"attributes": updated_attrs})


def _contiguous_run_slices(mask: np.ndarray) -> list[slice]:
    """Return ``slice(start, end)`` for each contiguous True run in ``mask``."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [slice(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def find_obsid_runs(
    nom_hk: xr.Dataset,
    *,
    source: NomHkObsidSource | None = None,
) -> list[tuple[ObsIdSpec, slice]]:
    """Detect contiguous trim-eligible ObsID runs in a NOM-HK Dataset.

    Parameters
    ----------
    nom_hk : xr.Dataset
        Decoded NOM-HK Dataset with a ``PACKET`` dimension.
    source : NomHkObsidSource or None
        If set, only scan that ObsID field; otherwise scan all trim-eligible sources.

    Returns
    -------
    list of (ObsIdSpec, slice)
        Ordered list of runs. Each slice indexes ``PACKET``.
    """
    if PACKET_DIM not in nom_hk.dims:
        raise ValueError(f"NOM-HK Dataset is missing required dimension {PACKET_DIM!r}")

    # Ensure time order so contiguous runs are chronologically meaningful
    time_var = DEFAULT_TIME_VARIABLE
    working = nom_hk
    if time_var in nom_hk:
        working = nom_hk.sortby(time_var)

    runs: list[tuple[ObsIdSpec, slice]] = []
    for spec in iter_trim_eligible(source):
        field = spec.source.value
        if field not in working:
            logger.debug("Skipping ObsID %s: field %s missing from Dataset", spec.obsid, field)
            continue
        values = working[field].values
        mask = values == spec.obsid
        for pkt_slice in _contiguous_run_slices(mask):
            runs.append((spec, pkt_slice))

    # Order by start packet index for stable output
    runs.sort(key=lambda item: item[1].start)
    return runs


def trim_nom_hk_run(nom_hk: xr.Dataset, packet_indexer: slice | np.ndarray) -> xr.Dataset | None:
    """Subset a NOM-HK Dataset to one ObsID run along ``PACKET``.

    Parameters
    ----------
    nom_hk : xr.Dataset
        Full or partially sorted NOM-HK Dataset.
    packet_indexer : slice or array-like
        Packet indices for the run (typically a ``slice`` from :func:`find_obsid_runs`).

    Returns
    -------
    xr.Dataset or None
        Trimmed Dataset, or ``None`` if no packets were selected.
    """
    # Match find_obsid_runs time ordering when slicing by absolute packet indices
    working = nom_hk
    if DEFAULT_TIME_VARIABLE in nom_hk:
        working = nom_hk.sortby(DEFAULT_TIME_VARIABLE)

    trimmed = working.isel({PACKET_DIM: packet_indexer})
    if trimmed.sizes.get(PACKET_DIM, 0) == 0:
        return None
    return trimmed


def _prepare_trimmed_attrs(
    trimmed: xr.Dataset,
    trimmed_product: DataProductIdentifier,
    source_attrs: dict[str, Any],
) -> xr.Dataset:
    """Stamp ProductID and refresh dynamic attributes on a trimmed Dataset."""
    out = trimmed.copy(deep=False)
    # Drop source-file encodings so conformance enforce does not warn on leftovers
    for name in list(out.variables):
        out[name].encoding = {}
    out.attrs = dict(source_attrs)
    out.attrs["ProductID"] = trimmed_product.value
    out.attrs["date_created"] = datetime.now(tz=UTC).isoformat()
    return out


def write_trimmed_nom_hk_products(
    nom_hk: xr.Dataset,
    output_path: str | PathType,
    *,
    time_variable: str = DEFAULT_TIME_VARIABLE,
    add_archive_path_prefix: bool = False,
    strict: bool = True,
    source: NomHkObsidSource | None = None,
) -> list[LiberaDataProductFilename]:
    """Detect ObsID runs in ``nom_hk`` and write one TRIMMED product per run.

    When the same ``(source, obsid)`` appears in multiple disjoint runs, each run
    is written separately and a warning is logged (unexpected in normal ops).

    Parameters
    ----------
    nom_hk : xr.Dataset
        Decoded ``NOM-HK-DECODED`` Dataset.
    output_path : str or PathType
        Directory (or S3 prefix) for output files.
    time_variable : str
        Time coordinate used for filename start/end times.
    add_archive_path_prefix : bool
        Forwarded to :func:`write_libera_data_product`.
    strict : bool
        Forwarded to :func:`write_libera_data_product`.
    source : NomHkObsidSource or None
        Optional filter to only emit TRIMMED products for one ObsID field.

    Returns
    -------
    list of LiberaDataProductFilename
        Paths of written TRIMMED products.
    """
    runs = find_obsid_runs(nom_hk, source=source)
    counts: dict[tuple[NomHkObsidSource, int], int] = defaultdict(int)
    for spec, _ in runs:
        counts[(spec.source, spec.obsid)] += 1
    for (src, obsid), n in counts.items():
        if n > 1:
            logger.warning(
                "ObsID %s on %s (%s) appears in %d disjoint runs in this NOM-HK Dataset; "
                "writing separate TRIMMED files per run (unexpected in normal operations)",
                obsid,
                src.name,
                src.value,
                n,
            )

    written: list[LiberaDataProductFilename] = []
    source_attrs = dict(nom_hk.attrs)
    for spec, pkt_slice in runs:
        if spec.trimmed_product is None:
            continue
        trimmed_product = spec.trimmed_product
        trimmed = trim_nom_hk_run(nom_hk, pkt_slice)
        if trimmed is None:
            continue
        trimmed = _prepare_trimmed_attrs(trimmed, trimmed_product, source_attrs)
        definition = get_trimmed_nom_hk_product_definition(trimmed_product)
        filename = write_libera_data_product(
            definition,
            trimmed,
            output_path,
            time_variable=time_variable,
            strict=strict,
            add_archive_path_prefix=add_archive_path_prefix,
        )
        logger.info(
            "Wrote TRIMMED NOM-HK product %s (%d packets) for ObsID %s / %s",
            filename.path.name,
            trimmed.sizes[PACKET_DIM],
            spec.obsid,
            spec.source.name,
        )
        written.append(filename)
    return written
