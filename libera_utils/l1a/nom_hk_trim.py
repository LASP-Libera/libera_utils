"""Trim decoded NOM-HK L1A Datasets to contiguous calibration ObsID runs.

One file is written per contiguous ObsID run that starts on the UTC day of the ``NOM-HK-DECODED``
granule the Dataset was decoded into, stamped with that ObsID's own TRIMMED ProductID (see
:mod:`libera_utils.obsids`). A run is a stretch of one ObsID: it ends where the ObsID changes, and
packets lost inside it do not end it. A run is never split at midnight: the day it starts on writes it
whole when it ends inside the day window (:func:`libera_utils.l1a.day_window.day_bounds`), where every
APID's L1A day product ends, and a run lasting past the window raises after the day's other runs are
written. Files of one ProductID are told apart by their filename time ranges. A run at a gap is written
and its gaps are logged; the file carries no record of them.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from libera_utils.constants import DataProductIdentifier, LiberaApid
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import (
    ALGORITHM_VERSION_ATTRIBUTE,
    DATE_CREATED_ATTRIBUTE,
    LiberaDataProductDefinition,
)
from libera_utils.l1a.day_coverage import DEFAULT_SEAM_TOLERANCE
from libera_utils.l1a.day_window import day_bounds
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path
from libera_utils.l1a.packet_ordering import SEQUENCE_COUNTER_MODULUS, SEQUENCE_COUNTER_VAR, out_of_order_positions
from libera_utils.l1a.packet_slicing import PACKET_DIM, select_packets
from libera_utils.obsids import NomHkObsidSource, ObsIdSpec, get_trimmed_obsid_spec, iter_trim_eligible
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
        A TRIMMED NOM-HK ProductID from the ObsID registry.

    Returns
    -------
    LiberaDataProductDefinition
        Fresh product definition identical to ``NOM-HK-DECODED`` except for ``ProductID``.

    Raises
    ------
    ValueError
        If ``trimmed_product`` is not a TRIMMED ProductID in the ObsID registry.
    """
    try:
        get_trimmed_obsid_spec(trimmed_product)
    except KeyError as exc:
        raise ValueError(
            f"ProductID {trimmed_product.value!r} is not a TRIMMED NOM-HK product in the ObsID registry. "
            "Register the calibration ObsID in libera_utils/data/obsid_registry.csv first."
        ) from exc

    base = LiberaDataProductDefinition.from_yaml(get_l1a_product_definition_path(LiberaApid.icie_nom_hk.value))
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

    A run ends where the ObsID changes. Packets lost inside a stretch of one ObsID do not end its run,
    so two events of one ObsID whose boundary fell in a gap are one run.

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
        If ``nom_hk`` has no ``PACKET`` dimension, or no ``PACKET_ICIE_TIME`` or ``SRC_SEQ_CTR``
        variable, or if ``SRC_SEQ_CTR`` repeats or steps backward between packets whose packet
        times are within ``packet_ordering.REORDER_WINDOW`` (10 s).

    Notes
    -----
    Run slices are positions on the ``PACKET`` axis, so they are contiguous acquisition only when
    the axis is in ``SRC_SEQ_CTR`` order, as ``parse_packets_to_l1a_dataset`` writes it. This raises
    when it is not.
    """
    if PACKET_DIM not in nom_hk.dims:
        raise ValueError(f"NOM-HK Dataset is missing required dimension {PACKET_DIM!r}")

    out_of_order = out_of_order_positions(nom_hk, DEFAULT_TIME_VARIABLE)
    if out_of_order.size:
        first = int(out_of_order[0])
        counters = nom_hk["SRC_SEQ_CTR"].values
        times = nom_hk[DEFAULT_TIME_VARIABLE].values
        raise ValueError(
            f"NOM-HK Dataset is not in SRC_SEQ_CTR order: the counter repeats or steps backward at "
            f"{out_of_order.size} position(s), first SRC_SEQ_CTR {counters[first]} -> {counters[first + 1]} at "
            f"{times[first]} -> {times[first + 1]}, so ObsID runs would not be contiguous acquisition. "
            f"Decode its L0 files again with parse_packets_to_l1a_dataset, which writes the PACKET axis in "
            f"SRC_SEQ_CTR order."
        )

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


def _edge_after_gap(counters: np.ndarray, before: int) -> bool:
    """Return True when packet ``before`` or ``before + 1`` is absent, or the counter does not step by exactly 1 between them."""
    if before < 0 or before + 1 >= counters.size:
        return True
    return bool((counters[before + 1] - counters[before]) % SEQUENCE_COUNTER_MODULUS != 1)


def _runs_started_on_day(
    nom_hk: xr.Dataset,
    runs: list[tuple[ObsIdSpec, slice]],
    *,
    day: date,
    time_variable: str,
) -> tuple[list[tuple[ObsIdSpec, slice, bool, bool]], list[tuple[ObsIdSpec, np.datetime64, np.datetime64]]]:
    """Sort the runs whose first time falls on ``day`` into those to write and those lasting past the day window.

    A run's first time is the minimum of ``time_variable`` over its packets, the time its filename
    starts at, so the day kept here is the day its filename dates it to. A run that starts before
    ``day`` belongs to the day before and one that starts after it to the day after; both are left
    out without a check.

    A run starting on ``day`` lasts past the day window when its last packet is after the window's
    end, or when it is still going at the last packet of ``nom_hk`` and that packet is within
    ``day_coverage.DEFAULT_SEAM_TOLERANCE`` of the window's end, so its end is not seen.

    An edge follows a gap when the packet beyond it is not in ``nom_hk``, or ``SRC_SEQ_CTR`` does not
    step by exactly 1 (modulo 16384) across it: the run's first or last packet may then not be its
    event's start or end.

    Returns
    -------
    tuple of (list of (ObsIdSpec, slice, bool, bool), list of (ObsIdSpec, numpy.datetime64, numpy.datetime64))
        The runs to write, each with whether its start follows a gap and whether its end precedes one;
        and the runs lasting past the day window, each with its first and last time.

    Raises
    ------
    ValueError
        If ``nom_hk`` has no ``time_variable``.
    """
    if time_variable not in nom_hk:
        raise ValueError(f"NOM-HK Dataset is missing time variable {time_variable!r}")
    bounds = day_bounds(day)
    # The data axes are datetime64, which carries no timezone; the bounds are UTC by construction.
    day_start = np.datetime64(bounds.day_start.replace(tzinfo=None), "us")
    day_end = np.datetime64(bounds.day_end.replace(tzinfo=None), "us")
    window_end = np.datetime64(bounds.window_end.replace(tzinfo=None), "us")
    # Data ending within the seam tolerance of the window's end counts as reaching it, so a packet period
    # short of the edge is not read as the data stopping early.
    reach = window_end - np.timedelta64(DEFAULT_SEAM_TOLERANCE)
    times = nom_hk[time_variable].values
    counters = nom_hk[SEQUENCE_COUNTER_VAR].values.astype(np.int64)

    kept: list[tuple[ObsIdSpec, slice, bool, bool]] = []
    too_long: list[tuple[ObsIdSpec, np.datetime64, np.datetime64]] = []
    for spec, pkt_slice in runs:
        first_time = times[pkt_slice].min()
        if not day_start <= first_time < day_end:
            continue
        last_time = times[pkt_slice].max()
        if last_time > window_end or (pkt_slice.stop == counters.size and last_time >= reach):
            too_long.append((spec, first_time, last_time))
            continue
        kept.append(
            (
                spec,
                pkt_slice,
                _edge_after_gap(counters, pkt_slice.start - 1),
                _edge_after_gap(counters, pkt_slice.stop - 1),
            )
        )
    return kept, too_long


def _prepare_trimmed_attrs(
    trimmed: xr.Dataset,
    trimmed_product: DataProductIdentifier,
    source_attrs: dict[str, Any],
    source_product_filename: str | PathType,
) -> xr.Dataset:
    """Stamp ProductID and refresh dynamic attributes on a trimmed Dataset.

    ``input_files`` is set to the single NOM-HK-DECODED granule this product was trimmed from,
    replacing the parent's own L0 packet-file provenance, and ``algorithm_version`` is restamped
    with the running libera_utils version. The parent's ``date_created`` is dropped, since
    ``write_libera_data_product`` sets it for the trimmed file.
    """
    out = trimmed.copy(deep=False)
    # Drop source-file encodings so conformance enforce does not warn on leftovers
    for name in list(out.variables):
        out[name].encoding = {}
    out.attrs = dict(source_attrs)
    out.attrs.pop(DATE_CREATED_ATTRIBUTE, None)
    out.attrs["ProductID"] = trimmed_product.value
    out.attrs["input_files"] = [Path(str(source_product_filename)).name]
    out.attrs[ALGORITHM_VERSION_ATTRIBUTE] = libera_utils_version()
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
    """Write one TRIMMED file for each ObsID run in ``nom_hk`` that starts on the day of ``source_product_filename``.

    Each file is stamped with its run's ObsID's TRIMMED ProductID and spans the whole run, from its
    first packet to its last. A run ends at an ObsID change, not at a ``SRC_SEQ_CTR`` gap (see
    :func:`find_obsid_runs`); a run at a gap is written and logged at WARNING. "ObsID-trimmed NOM-HK
    products" in the L1A processing user guide describes the day rule, the gaps, and what a later
    revision with the missing packets does to each file.

    Parameters
    ----------
    nom_hk : xr.Dataset
        Decoded ``NOM-HK-DECODED`` Dataset, as :func:`~libera_utils.l1a.packets.parse_packets_to_l1a_dataset`
        returns it, covering the day window of its day (:func:`~libera_utils.l1a.day_window.day_bounds`).
    output_path : str or PathType
        Directory (or S3 prefix) for output files.
    source_product_filename : str or PathType
        Filename of the ``NOM-HK-DECODED`` granule ``nom_hk`` was decoded into. Its
        :attr:`~libera_utils.io.filenaming.LiberaDataProductFilename.applicable_date` is the UTC day
        whose runs are written, and it is recorded as the sole entry of each TRIMMED product's
        ``input_files`` attribute.
    time_variable : str
        Time coordinate used for the day selection and the filename start/end times.
    add_archive_path_prefix : bool
        Forwarded to :func:`~libera_utils.io.netcdf.write_libera_data_product`.
    strict : bool
        Forwarded to :func:`~libera_utils.io.netcdf.write_libera_data_product`.
    source : NomHkObsidSource or None
        Optional filter to only emit TRIMMED products for one ObsID field.

    Returns
    -------
    list of LiberaDataProductFilename
        Paths of written TRIMMED products, in run order; empty when no run starts on the day.

    Raises
    ------
    ValueError
        Before any file is written: as :func:`find_obsid_runs` raises it; if ``source_product_filename``
        is not a Libera filename or does not name a ``NOM-HK-DECODED`` product; or if ``nom_hk`` has no
        ``time_variable``. After the day's other TRIMMED files are written: if a run starting on the day
        lasts past the end of the day window, or is still going at the last packet of ``nom_hk`` within
        ``day_coverage.DEFAULT_SEAM_TOLERANCE`` of it.
    """
    parent = LiberaDataProductFilename(source_product_filename)
    if parent.data_product_id is not DataProductIdentifier.l1a_icie_nom_hk_decoded:
        raise ValueError(
            f"source_product_filename {parent.path.name} is a {parent.data_product_id.value} product, not "
            f"{DataProductIdentifier.l1a_icie_nom_hk_decoded.value}. Pass the filename of the "
            f"{DataProductIdentifier.l1a_icie_nom_hk_decoded.value} granule the NOM-HK Dataset was decoded into."
        )
    day = parent.applicable_date
    runs, too_long = _runs_started_on_day(
        nom_hk, find_obsid_runs(nom_hk, source=source), day=day, time_variable=time_variable
    )

    # One definition per distinct ProductID, not per run; dict.fromkeys keeps the run order
    distinct_products = dict.fromkeys(spec.trimmed_product for spec, *_ in runs)
    definitions = {product: get_trimmed_nom_hk_product_definition(product) for product in distinct_products}

    written: list[LiberaDataProductFilename] = []
    source_attrs = dict(nom_hk.attrs)
    for spec, pkt_slice, start_after_gap, end_before_gap in runs:
        trimmed_product = spec.trimmed_product
        trimmed = select_packets(nom_hk, pkt_slice)
        # Modulo 16384 per step, so a lower bound: a gap of a multiple of 16384 packets steps the counter by 1.
        counter_steps = np.diff(trimmed[SEQUENCE_COUNTER_VAR].values.astype(np.int64))
        missing_packets = int(((counter_steps - 1) % SEQUENCE_COUNTER_MODULUS).sum())
        trimmed = _prepare_trimmed_attrs(trimmed, trimmed_product, source_attrs, source_product_filename)
        filename = write_libera_data_product(
            definitions[trimmed_product],
            trimmed,
            output_path,
            time_variable=time_variable,
            strict=strict,
            add_archive_path_prefix=add_archive_path_prefix,
        )
        # TODO[LIBSDC-930]: report these gaps where Dagster shows them, not only in this log line.
        logger.log(
            logging.WARNING if start_after_gap or end_before_gap or missing_packets else logging.INFO,
            "Wrote TRIMMED NOM-HK product %s (%d packets) for ObsID %s / %s as %s, start after gap %s, "
            "end before gap %s, %d packets missing inside",
            filename.path.name,
            trimmed.sizes[PACKET_DIM],
            spec.obsid,
            spec.source.name,
            trimmed_product.value,
            start_after_gap,
            end_before_gap,
            missing_packets,
        )
        written.append(filename)

    if too_long:
        described = "; ".join(
            f"the {spec.trimmed_product.value} run of ObsID {spec.obsid} on {spec.source.name} starting {first_time}, "
            f"still going at {last_time}"
            for spec, first_time, last_time in too_long
        )
        raise ValueError(
            f"Cannot write {len(too_long)} TRIMMED NOM-HK run(s) for {day.isoformat()} ({described}); the day's "
            f"other {len(written)} TRIMMED file(s) were written. Each run lasts past the end of the day window, "
            f"{day_bounds(day).window_end.isoformat()}, where every APID's L1A day product ends, so a calibration "
            f"step would get incomplete inputs. Nominal calibration events end well inside the day window. Check "
            f"this event with the calibration team: a known long event, such as an 18-minute video mode, is "
            f"processed outside the automatic trim."
        )
    return written
