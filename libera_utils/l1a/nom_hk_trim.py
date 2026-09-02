"""Trim decoded NOM-HK L1A Datasets to contiguous calibration ObsID runs.

One file is written per contiguous ObsID run, stamped with the run's *calibration dependency
family* ProductID rather than its ObsID: ObsIDs that downstream algorithms process identically
share one TRIMMED ProductID (see :mod:`libera_utils.obsids`). A day therefore normally yields
several files sharing one family ProductID — six ``NOM-HK-SWC-FAMILY-TRIMMED`` granules for the
six shortwave LED ObsIDs, for example — distinguished by their filename time ranges. Each file
covers exactly one ObsID run, and the ObsID is read from the ``ICIE__SW_OBSID_*`` variable the
file carries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from libera_utils.constants import DataProductIdentifier, LiberaApid
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path
from libera_utils.l1a.packet_slicing import PACKET_DIM, select_packets
from libera_utils.obsids import TRIM_FAMILIES, NomHkObsidSource, ObsIdSpec, iter_trim_eligible
from libera_utils.version import version as libera_utils_version

logger = logging.getLogger(__name__)

DEFAULT_TIME_VARIABLE = "PACKET_ICIE_TIME"


def get_trimmed_nom_hk_product_definition(
    trimmed_product: DataProductIdentifier,
) -> LiberaDataProductDefinition:
    """Load the NOM-HK L1A product definition with ``ProductID`` set to a TRIMMED DPI.

    Parameters
    ----------
    trimmed_product : DataProductIdentifier
        A known TRIMMED NOM-HK calibration family ProductID from the ObsID registry.

    Returns
    -------
    LiberaDataProductDefinition
        Fresh product definition identical to ``NOM-HK-DECODED`` except for ``ProductID``.

    Raises
    ------
    ValueError
        If ``trimmed_product`` is not a registered TRIMMED family ProductID.
    """
    if trimmed_product not in TRIM_FAMILIES:
        raise ValueError(
            f"ProductID {trimmed_product.value!r} is not a known TRIMMED NOM-HK calibration family "
            f"product. Expected one of: {sorted(product.value for product in TRIM_FAMILIES)}"
        )

    base = LiberaDataProductDefinition.from_yaml(get_l1a_product_definition_path(LiberaApid.icie_nom_hk.value))
    updated_attrs = {**base.attributes, "ProductID": trimmed_product.value}
    return base.model_copy(update={"attributes": updated_attrs})


def _check_packet_time_sorted(nom_hk: xr.Dataset) -> None:
    """Verify the ``PACKET`` axis is in non-decreasing packet-time order.

    Decoded L1A products are written packet-time sorted, so this is a guard, not a fixup; the
    positional run indices from :func:`find_obsid_runs` are only meaningful against a
    time-ordered packet axis. If ``PACKET_ICIE_TIME`` is absent the check is skipped with a
    warning and the packet axis is trusted as-is.

    Parameters
    ----------
    nom_hk : xr.Dataset
        Decoded NOM-HK Dataset.

    Raises
    ------
    ValueError
        If ``PACKET_ICIE_TIME`` is present but not monotonically non-decreasing.
    """
    if DEFAULT_TIME_VARIABLE not in nom_hk:
        logger.warning(
            "NOM-HK Dataset has no %s variable; skipping the packet-time ordering check. ObsID run "
            "slices will be trusted as-is.",
            DEFAULT_TIME_VARIABLE,
        )
        return
    times = nom_hk[DEFAULT_TIME_VARIABLE].values
    if times.size and np.any(np.diff(times) < np.timedelta64(0, "ns")):
        raise ValueError(
            f"NOM-HK Dataset is not sorted by {DEFAULT_TIME_VARIABLE}. Decoded L1A products are "
            f"written in packet-time order; re-decode the source product."
        )


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

    Raises
    ------
    ValueError
        If ``nom_hk`` has no ``PACKET`` dimension, or if its ``PACKET_ICIE_TIME`` variable is
        present but not in non-decreasing order (see :func:`_check_packet_time_sorted`).
    """
    if PACKET_DIM not in nom_hk.dims:
        raise ValueError(f"NOM-HK Dataset is missing required dimension {PACKET_DIM!r}")

    _check_packet_time_sorted(nom_hk)

    runs: list[tuple[ObsIdSpec, slice]] = []
    for spec in iter_trim_eligible(source):
        field = spec.source.value
        if field not in nom_hk:
            logger.debug("Skipping ObsID %s: field %s missing from Dataset", spec.obsid, field)
            continue
        values = nom_hk[field].values
        mask = values == spec.obsid
        for pkt_slice in _contiguous_run_slices(mask):
            runs.append((spec, pkt_slice))

    # Order by start packet index for stable output
    runs.sort(key=lambda item: item[1].start)
    return runs


def _prepare_trimmed_attrs(
    trimmed: xr.Dataset,
    trimmed_product: DataProductIdentifier,
    source_attrs: dict[str, Any],
    source_product_filename: str | PathType,
) -> xr.Dataset:
    """Stamp ProductID and refresh dynamic attributes on a trimmed Dataset.

    ``input_files`` is set to the single NOM-HK-DECODED granule this product was trimmed from,
    replacing the parent's own L0 packet-file provenance, and ``algorithm_version`` is restamped
    with the running libera_utils version.
    """
    out = trimmed.copy(deep=False)
    # Drop source-file encodings so conformance enforce does not warn on leftovers
    for name in list(out.variables):
        out[name].encoding = {}
    out.attrs = dict(source_attrs)
    out.attrs["ProductID"] = trimmed_product.value
    out.attrs["date_created"] = datetime.now(tz=UTC).isoformat()
    out.attrs["input_files"] = [Path(str(source_product_filename)).name]
    out.attrs["algorithm_version"] = libera_utils_version()
    return out


def write_trimmed_nom_hk_products(
    nom_hk: xr.Dataset,
    output_path: str | PathType,
    *,
    source_product_filename: str | PathType,
    time_variable: str = DEFAULT_TIME_VARIABLE,
    add_archive_path_prefix: bool = False,
    strict: bool = True,
    source: NomHkObsidSource | None = None,
) -> list[LiberaDataProductFilename]:
    """Detect ObsID runs in ``nom_hk`` and write one TRIMMED file per run.

    Each file is stamped with its run's calibration dependency family ProductID, so several
    files written from one Dataset normally share a ProductID — one per ObsID in that family —
    and are told apart by their filename time ranges.

    When the same ``(source, obsid)`` appears in multiple disjoint runs, each run
    is written separately and a warning is logged (unexpected in normal ops).

    Parameters
    ----------
    nom_hk : xr.Dataset
        Decoded ``NOM-HK-DECODED`` Dataset.
    output_path : str or PathType
        Directory (or S3 prefix) for output files.
    source_product_filename : str or PathType
        Filename of the ``NOM-HK-DECODED`` granule ``nom_hk`` was read from. Recorded as the
        sole entry of each TRIMMED product's ``input_files`` attribute.
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

    # One definition per distinct family, not per run; dict.fromkeys keeps the run order
    distinct_products = dict.fromkeys(spec.trimmed_product for spec, _ in runs)
    definitions = {product: get_trimmed_nom_hk_product_definition(product) for product in distinct_products}

    written: list[LiberaDataProductFilename] = []
    source_attrs = dict(nom_hk.attrs)
    for spec, pkt_slice in runs:
        trimmed_product = spec.trimmed_product
        trimmed = select_packets(nom_hk, pkt_slice)
        trimmed = _prepare_trimmed_attrs(trimmed, trimmed_product, source_attrs, source_product_filename)
        filename = write_libera_data_product(
            definitions[trimmed_product],
            trimmed,
            output_path,
            time_variable=time_variable,
            strict=strict,
            add_archive_path_prefix=add_archive_path_prefix,
        )
        logger.info(
            "Wrote TRIMMED NOM-HK product %s (%d packets) for ObsID %s / %s into family %s",
            filename.path.name,
            trimmed.sizes[PACKET_DIM],
            spec.obsid,
            spec.source.name,
            trimmed_product.value,
        )
        written.append(filename)
    return written
