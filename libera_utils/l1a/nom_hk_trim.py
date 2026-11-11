"""Trim decoded NOM-HK L1A Datasets to contiguous calibration ObsID runs.

One file is written per contiguous ObsID run that starts on the requested UTC day, stamped with that
ObsID's own TRIMMED ProductID (see :mod:`libera_utils.obsids`). A run is a stretch of one ObsID: it
ends where the ObsID changes, and packets lost inside it do not end it but are counted on its file. A
run is never split at midnight: the day it starts on writes it whole, which is why a caller decodes
NOM-HK :data:`CALIBRATION_TRIM_PADDING` past the end of the day. Files of one ProductID are told apart
by their filename time ranges.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
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
from libera_utils.l1a.day_window import DAY_BUFFER, day_bounds
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path
from libera_utils.l1a.packet_ordering import SEQUENCE_COUNTER_MODULUS, SEQUENCE_COUNTER_VAR, out_of_order_positions
from libera_utils.l1a.packet_slicing import PACKET_DIM, select_packets
from libera_utils.obsids import NomHkObsidSource, ObsIdSpec, get_trimmed_obsid_spec, iter_trim_eligible
from libera_utils.version import version as libera_utils_version

logger = logging.getLogger(__name__)

DEFAULT_TIME_VARIABLE = "PACKET_ICIE_TIME"

# TODO[LIBSDC-900]: confirm the longest calibration event with the calibration team and set this from it.
#: How far past the end of day D a caller decodes NOM-HK so that every calibration run starting on D is
#: seen to end. It must exceed the longest calibration event. The value is a placeholder: the "Design Doc -
#: Calibration Processing Orchestration" (Confluence, LIBERA space) assumes a longest event without
#: confirming it, and 2 hours is one L0 PDS file. A run longer than this fails its day rather than being split.
CALIBRATION_TRIM_PADDING = timedelta(hours=2)

#: TRIMMED global attribute: 1 when packets just before the run are missing, so the run's first packet may
#: not be its event's start; 0 when the packet before it is present and its ObsID differs.
RUN_START_AFTER_GAP_ATTRIBUTE = "run_start_after_gap"
#: TRIMMED global attribute: 1 when packets just after the run are missing, so the run's last packet may
#: not be its event's end; 0 when the packet after it is present and its ObsID differs.
RUN_END_BEFORE_GAP_ATTRIBUTE = "run_end_before_gap"
#: TRIMMED global attribute: the number of packets missing between the run's first and last packet, from
#: ``SRC_SEQ_CTR`` steps other than 1, counted modulo 16384 so a lower bound; 0 when none are missing. Two
#: events of one ObsID whose boundary fell in the gap are counted as one run.
RUN_MISSING_PACKETS_ATTRIBUTE = "run_missing_packets"


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
        Fresh product definition identical to ``NOM-HK-DECODED`` except for ``ProductID`` and the
        dynamic :data:`RUN_START_AFTER_GAP_ATTRIBUTE`, :data:`RUN_END_BEFORE_GAP_ATTRIBUTE` and
        :data:`RUN_MISSING_PACKETS_ATTRIBUTE`.

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
    updated_attrs = {
        **base.attributes,
        "ProductID": trimmed_product.value,
        RUN_START_AFTER_GAP_ATTRIBUTE: None,
        RUN_END_BEFORE_GAP_ATTRIBUTE: None,
        RUN_MISSING_PACKETS_ATTRIBUTE: None,
    }
    return base.model_copy(update={"attributes": updated_attrs})


def _contiguous_run_slices(mask: np.ndarray) -> list[slice]:
    """Return ``slice(start, end)`` for each contiguous True run in ``mask``."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [slice(int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def _missing_packets(counters: np.ndarray) -> int:
    """Return how many packets are missing between the first and last of ``counters``, modulo 16384 per step."""
    return int(((np.diff(counters.astype(np.int64)) - 1) % SEQUENCE_COUNTER_MODULUS).sum())


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
) -> list[tuple[ObsIdSpec, slice, bool, bool]]:
    """Keep the runs whose first time falls on ``day``, and say which of each kept run's edges follows a gap.

    A run's first time is the minimum of ``time_variable`` over its packets, the time its filename
    starts at, so the day kept here is the day its filename dates it to. A run that starts before
    ``day`` belongs to the day before and one that starts after it to the day after; both are left
    out without a check.

    An edge follows a gap when the packet beyond it is not in ``nom_hk``, or ``SRC_SEQ_CTR`` does not
    step by exactly 1 (modulo 16384) across it: the run's first or last packet may then not be its
    event's start or end.

    Returns
    -------
    list of (ObsIdSpec, slice, bool, bool)
        Each kept run with whether its start follows a gap and whether its end precedes one.

    Raises
    ------
    ValueError
        If ``nom_hk`` has no ``time_variable``, or if a run starting on ``day`` is still going at the
        last packet of ``nom_hk`` and that packet is within ``DAY_BUFFER`` of
        ``CALIBRATION_TRIM_PADDING`` past the end of ``day``: the run outlasts the padding.
    """
    if time_variable not in nom_hk:
        raise ValueError(f"NOM-HK Dataset is missing time variable {time_variable!r}")
    bounds = day_bounds(day)
    # The data axes are datetime64, which carries no timezone; the bounds are UTC by construction.
    day_start = np.datetime64(bounds.day_start.replace(tzinfo=None), "us")
    day_end = np.datetime64(bounds.day_end.replace(tzinfo=None), "us")
    # Data ending within DAY_BUFFER of the decode's reach counts as reaching it, so an L0 file seam or a
    # packet period short of the reach is not read as the data stopping early.
    padded_reach = np.datetime64((bounds.day_end + CALIBRATION_TRIM_PADDING - DAY_BUFFER).replace(tzinfo=None), "us")
    times = nom_hk[time_variable].values
    counters = nom_hk[SEQUENCE_COUNTER_VAR].values.astype(np.int64)

    kept: list[tuple[ObsIdSpec, slice, bool, bool]] = []
    for spec, pkt_slice in runs:
        first_time = times[pkt_slice].min()
        if not day_start <= first_time < day_end:
            continue
        last_time = times[pkt_slice].max()
        if pkt_slice.stop == counters.size and last_time >= padded_reach:
            raise ValueError(
                f"Cannot write the {spec.trimmed_product.value} run of ObsID {spec.obsid} on {spec.source.name} "
                f"starting {first_time} for {day.isoformat()}: it is still going at {last_time}, the end of NOM-HK "
                f"data that reaches CALIBRATION_TRIM_PADDING ({CALIBRATION_TRIM_PADDING}) past the end of the day, "
                f"so the run lasts longer than CALIBRATION_TRIM_PADDING. Raise "
                f"libera_utils.l1a.nom_hk_trim.CALIBRATION_TRIM_PADDING above the longest calibration event, "
                f"release libera_utils and assemble the day again."
            )
        kept.append(
            (
                spec,
                pkt_slice,
                _edge_after_gap(counters, pkt_slice.start - 1),
                _edge_after_gap(counters, pkt_slice.stop - 1),
            )
        )
    return kept


def _prepare_trimmed_attrs(
    trimmed: xr.Dataset,
    trimmed_product: DataProductIdentifier,
    source_attrs: dict[str, Any],
    source_product_filename: str | PathType,
    *,
    start_after_gap: bool,
    end_before_gap: bool,
) -> xr.Dataset:
    """Stamp ProductID and refresh dynamic attributes on a trimmed Dataset.

    ``input_files`` is set to the single NOM-HK-DECODED granule this product was trimmed from,
    replacing the parent's own L0 packet-file provenance, and ``algorithm_version`` is restamped
    with the running libera_utils version. The parent's ``date_created`` is dropped, since
    ``write_libera_data_product`` sets it for the trimmed file. The edge gap flags are 1 or 0, and
    the missing-packet count is taken from ``trimmed``'s ``SRC_SEQ_CTR``.
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
    out.attrs[RUN_START_AFTER_GAP_ATTRIBUTE] = int(start_after_gap)
    out.attrs[RUN_END_BEFORE_GAP_ATTRIBUTE] = int(end_before_gap)
    out.attrs[RUN_MISSING_PACKETS_ATTRIBUTE] = _missing_packets(trimmed[SEQUENCE_COUNTER_VAR].values)
    return out


def write_trimmed_nom_hk_products(
    nom_hk: xr.Dataset,
    output_path: str | PathType,
    *,
    day: date,
    source_product_filename: str | PathType,
    time_variable: str = DEFAULT_TIME_VARIABLE,
    add_archive_path_prefix: bool = False,
    strict: bool = True,
    source: NomHkObsidSource | None = None,
) -> list[LiberaDataProductFilename]:
    """Write one TRIMMED file for each ObsID run in ``nom_hk`` that starts on ``day``.

    Each file is stamped with its run's ObsID's TRIMMED ProductID and spans the run, from its
    first packet to its last. An ObsID may recur within a day; each run is its own file. A run ends
    at an ObsID change, not at a ``SRC_SEQ_CTR`` gap (see :func:`find_obsid_runs`). Each file says in
    :data:`RUN_START_AFTER_GAP_ATTRIBUTE` and :data:`RUN_END_BEFORE_GAP_ATTRIBUTE` whether its start
    or end is at a gap rather than an observed ObsID change, and in
    :data:`RUN_MISSING_PACKETS_ATTRIBUTE` how many packets are missing inside it. Missing packets that
    later arrive can change a file's range: an end before a gap moves later in the next revision,
    under the same start time; a start after a gap moves earlier, so the next revision files the run
    under a new start time and the earlier file is not replaced; and a gap that hid the boundary
    between two events of one ObsID makes two files where there was one. Runs that start before or after ``day`` are left out. Every run is checked before
    any file is written, so a rejected day writes nothing.

    Parameters
    ----------
    nom_hk : xr.Dataset
        Decoded ``NOM-HK-DECODED`` Dataset, as :func:`~libera_utils.l1a.packets.parse_packets_to_l1a_dataset`
        returns it. To see each run that starts on ``day`` end, it reaches at least
        :data:`CALIBRATION_TRIM_PADDING` past the end of ``day``; a run still going there raises.
        Data that stops earlier ends a run at its last packet, flagged as a gap.
    output_path : str or PathType
        Directory (or S3 prefix) for output files.
    day : datetime.date
        UTC day whose runs are written: those whose first time is in ``[day, day + 1)``.
    source_product_filename : str or PathType
        Filename of the ``NOM-HK-DECODED`` granule for ``day``. Recorded as the sole entry of each
        TRIMMED product's ``input_files`` attribute.
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
        Paths of written TRIMMED products, in run order; empty when no run starts on ``day``.

    Raises
    ------
    ValueError
        As :func:`find_obsid_runs` raises it; if ``nom_hk`` has no ``time_variable``; or, before any
        file is written, if a run starting on ``day`` is still going at the last packet of ``nom_hk``
        and that packet is within ``day_window.DAY_BUFFER`` of :data:`CALIBRATION_TRIM_PADDING` past
        the end of ``day``, meaning the run outlasts the padding.
    """
    runs = _runs_started_on_day(nom_hk, find_obsid_runs(nom_hk, source=source), day=day, time_variable=time_variable)

    # One definition per distinct ProductID, not per run; dict.fromkeys keeps the run order
    distinct_products = dict.fromkeys(spec.trimmed_product for spec, *_ in runs)
    definitions = {product: get_trimmed_nom_hk_product_definition(product) for product in distinct_products}

    written: list[LiberaDataProductFilename] = []
    source_attrs = dict(nom_hk.attrs)
    for spec, pkt_slice, start_after_gap, end_before_gap in runs:
        trimmed_product = spec.trimmed_product
        trimmed = select_packets(nom_hk, pkt_slice)
        trimmed = _prepare_trimmed_attrs(
            trimmed,
            trimmed_product,
            source_attrs,
            source_product_filename,
            start_after_gap=start_after_gap,
            end_before_gap=end_before_gap,
        )
        filename = write_libera_data_product(
            definitions[trimmed_product],
            trimmed,
            output_path,
            time_variable=time_variable,
            strict=strict,
            add_archive_path_prefix=add_archive_path_prefix,
        )
        logger.info(
            "Wrote TRIMMED NOM-HK product %s (%d packets) for ObsID %s / %s as %s, start after gap %s, "
            "end before gap %s, %d packets missing inside",
            filename.path.name,
            trimmed.sizes[PACKET_DIM],
            spec.obsid,
            spec.source.name,
            trimmed_product.value,
            start_after_gap,
            end_before_gap,
            trimmed.attrs[RUN_MISSING_PACKETS_ATTRIBUTE],
        )
        written.append(filename)
    return written
