"""Module for reading packet data using Space Packet Parser"""

import logging
import warnings
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from os import PathLike
from typing import cast

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath
from space_packet_parser import load_xtce
from space_packet_parser.generators.ccsds import CCSDSPacketBytes
from space_packet_parser.xarr import create_dataset

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io import filenaming
from libera_utils.l1a.l1a_packet_configs import (
    AggregationGroup,
    ArrayGroup,
    SampleGroup,
    TimeFieldMapping,
    get_packet_config,
)
from libera_utils.l1a.packet_ordering import (
    check_packet_acquisition_order,
    order_packet_files,
)
from libera_utils.l1a.quality import DuplicateEvidence, GranuleQualityRecord
from libera_utils.l1a.wfov_image_metadata import enhance_wfov_l1a_dataset
from libera_utils.time import multipart_to_dt64
from libera_utils.version import version

logger = logging.getLogger(__name__)

# SPP always creates Datasets with a non-coordinate "packet" dimension
# This is a constant and is not expected to change in SPP
SDC_PACKET_DIMENSION = "PACKET"
SPP_PACKET_DIMENSION = "packet"  # The original dimension name from SPP before we swap it to uppercase

DATETIME_USEC_DTYPE = np.dtype("datetime64[us]")


def drop_implausible_telemetry_times(times_us: np.ndarray, *, context: str) -> np.ndarray:
    """Filter out packet/sample times outside the plausible telemetry window.

    The window is ``MIN_VALID_TELEMETRY_TIME`` to ``MAX_VALID_TELEMETRY_TIME`` from config. Both
    are fixed dates, not offsets from now, so a span written to File Metadata does not depend on
    when extraction ran; the ceiling admits simulated-clock captures running at a mission-era
    epoch. A clock that has not yet received its first time-sync command reads out a near-zero
    day/second counter, decoding to just after ``CCSDS_EPOCH`` (1958-01-01), and one such packet
    stretches a ``min()``/``max()`` span across ~68 years.

    A time exactly at the floor is dropped. ``NaT`` is dropped explicitly, since it compares false
    against both bounds.

    Parameters
    ----------
    times_us : np.ndarray
        Array of ``datetime64[us]`` packet or sample times.
    context : str
        Description of what is being processed (APID/file), used in the warning log and in the
        error raised if nothing remains.

    Returns
    -------
    np.ndarray
        ``times_us`` with any implausible entries removed.

    Raises
    ------
    ValueError
        If no times remain after filtering.
    """
    floor = np.datetime64(config.get("MIN_VALID_TELEMETRY_TIME"), "us")
    ceiling = np.datetime64(config.get("MAX_VALID_TELEMETRY_TIME"), "us")

    invalid = np.isnat(times_us)
    below = times_us <= floor
    above = times_us > ceiling
    n_invalid = int(invalid.sum())
    n_below = int(below.sum())
    n_above = int(above.sum())
    if n_invalid:
        logger.warning("Excluded %d NaT time(s) for %s", n_invalid, context)
    if n_below:
        logger.warning(
            "Excluded %d time(s) at or before sanity floor %s (likely clock not yet time-synced) for %s",
            n_below,
            floor,
            context,
        )
    if n_above:
        logger.warning(
            "Excluded %d time(s) after sanity ceiling %s (likely corrupted time counter) for %s",
            n_above,
            ceiling,
            context,
        )

    filtered = times_us[~(invalid | below | above)]
    if filtered.size == 0:
        raise ValueError(
            f"No times for {context} fall between the sanity floor {floor} and ceiling {ceiling}; none remain."
        )
    return filtered


def parse_packets_to_dataset(
    packet_files: list[PathLike | str], packet_definition: str | PathLike, apid: int, **generator_kwargs
) -> xr.Dataset:
    """Parse packets from files into an xarray Dataset using specified packet definition.

    This function does not make any changes to the packet data other than filtering by a single APID.

    Parameters
    ----------
    packet_files : list[PathLike | str]
        List of filepaths to packet files.
    packet_definition : str | PathLike
        Path to the XTCE packet definition file.
    apid : int
        Application Process Identifier to filter for.
    **generator_kwargs
        Additional keyword arguments passed to the packet generator.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing parsed packet data.
    """
    logger.info("Parsing packets (APID %d) from %d file(s)", apid, len(packet_files))

    def _packet_filter(packet_bytes: CCSDSPacketBytes) -> bool:
        return packet_bytes.apid == apid

    # Parse packets using space_packet_parser
    dataset_dict = create_dataset(
        packet_files=[AnyPath(f) for f in packet_files],
        xtce_packet_definition=packet_definition,
        generator_kwargs=generator_kwargs,
        packet_filter=_packet_filter,
    )

    if set(dataset_dict.keys()) != {apid}:
        raise ValueError(
            f"Expected only APID {apid} in parsed dataset, but found APIDs: {list(dataset_dict.keys())}. "
            f"This probably means the packet filter function is not working."
        )

    # Swap standard SPP "packet" dimension to uppercase "PACKET" to align with config naming
    if SPP_PACKET_DIMENSION in dataset_dict[apid].dims:
        dataset_dict[apid] = dataset_dict[apid].swap_dims({SPP_PACKET_DIMENSION: SDC_PACKET_DIMENSION})

    return dataset_dict[apid]


def parse_packets_to_l1a_dataset(
    packet_files: list[PathLike | str],
    apid: int,
    *,
    ground_data: bool = False,
    verbose: bool = False,
    skip_header_bytes: int | None = None,
    quality_record: GranuleQualityRecord | None = None,
) -> xr.Dataset:
    """Parse packets to L1A dataset with configurable sample expansion.

    This function parses binary packet files and expands multi-sample fields
    according to the a configuration identified by APID. It creates proper xarray Datasets
    with time coordinates as dimensions.

    Parameters
    ----------
    packet_files : list[PathLike | str]
        List of filepaths to packet files.
    apid : int
        The APID (Application Process Identifier) value for the packet type. Used to select the appropriate
        configuration for generating the L1A Dataset structure.
    ground_data : bool, optional
        If True, non-identical duplicate timestamps will produce a warning instead of a ValueError. This is useful for ground
        test data where duplicate timestamps with differing data may be expected. Default is False.
    verbose : bool, optional
        If True and ground_data is True, a warning will be issued for each duplicate coordinate value. Default is False.
    skip_header_bytes : int | None, optional
        Bytes to skip before each CCSDS primary header. When ``None``, uses ``SKIP_PACKET_HEADER_BYTES`` from
        config (default ``0``, correct for flight PDS and demuxed ground CCSDS; raw ground captures that still
        carry a per-packet record header need ``8``).
    quality_record : GranuleQualityRecord | None, optional
        When given, it is filled in with this granule's quality counters and the per-event
        evidence behind them. The counters also land on the returned Dataset as global
        attributes; the evidence is too large for that and reaches a caller only this way.

    Returns
    -------
    xr.Dataset
        xarray Dataset with:
        - Main packet data array with packet timestamp dimension
        - Separate arrays for each sample group with optional multi-field expansion
        - All time coordinates properly set as dimensions

    Notes
    -----
    For APID 1040 (ICIE WFOV SCI), the packet dataset is post-processed by
    ``enhance_wfov_l1a_dataset``: complete SOP→EOP images are stitched onto a ``CAMERA_TIME``
    dimension, compressed payloads and decoded header metadata are attached, ``ICIE__WFOV_DATA``
    is dropped, and ``PACKET_IMAGE_ID`` traces packets back to stitched images.
    """
    _packet_files = [cast(filenaming.PathType, AnyPath(f)) for f in packet_files]
    packet_config = get_packet_config(LiberaApid(apid))
    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    if skip_header_bytes is None:
        skip_header_bytes = config.get("SKIP_PACKET_HEADER_BYTES")
    if len(_packet_files) > 1:
        # Byte order within a file is acquisition order, but file order is whatever the caller
        # supplied, so the files have to be placed in time before their packets are concatenated.
        _packet_files = cast(
            list[filenaming.PathType],
            order_packet_files(
                _packet_files,
                load_xtce(packet_definition_path),
                apid,
                skip_header_bytes=skip_header_bytes,
                multipart_kwargs=packet_config.packet_time_fields.multipart_kwargs,
            ),
        )
    packet_ds = parse_packets_to_dataset(
        _packet_files, packet_definition_path, apid, skip_header_bytes=skip_header_bytes
    )
    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype(DATETIME_USEC_DTYPE)

    # Set packet time as a non-dimension coordinate with "PACKET" dimension
    # The packet dimension remains "PACKET" to enable sample-to-packet tracing
    packet_time_coordinate = packet_config.packet_time_coordinate
    packet_ds = packet_ds.assign_coords({packet_time_coordinate: (SDC_PACKET_DIMENSION, packet_times_us)})

    quality = quality_record if quality_record is not None else GranuleQualityRecord()
    quality.apid = int(apid)

    # Drop duplicates from the packet dataset before we process samples
    # This drops full duplicate packets based on identical packet timestamps
    packet_ds, packet_duplicates = _drop_duplicates(
        packet_ds, packet_time_coordinate, ground_data=ground_data, verbose=verbose
    )
    quality.duplicate_packet_time_count = packet_duplicates.n_duplicates
    quality.duplicate_value_mismatch_count += packet_duplicates.n_value_mismatches
    quality.mismatched_variables.extend(packet_duplicates.mismatched_variables)
    quality.duplicate_evidence.extend(packet_duplicates.evidence)

    # The packet axis is already in acquisition order: files were placed in time above and byte
    # order within a file is acquisition order. Do not sort it on packet time. The secondary
    # header time steps backward on ~0.2-1.4% of packets while SRC_SEQ_CTR marches on
    # (LIBSDC-830), so a time sort permutes packets away from the order they were taken in,
    # which breaks the contiguous-run assumption WFOV image stitching depends on. The
    # "{sample_group}_packet_index" variables built below enumerate this axis positionally.
    packet_ds, order_diagnostics = check_packet_acquisition_order(
        packet_ds,
        packet_time_coordinate,
        packet_dimension=SDC_PACKET_DIMENSION,
        ground_data=ground_data,
        verbose=verbose,
    )
    quality.n_packets = order_diagnostics.n_packets
    quality.packet_time_inversion_count = order_diagnostics.n_time_inversions
    quality.packets_out_of_time_order_count = order_diagnostics.n_packets_displaced
    quality.max_packet_time_inversion_microseconds = order_diagnostics.max_time_inversion_us
    quality.missing_packet_count = order_diagnostics.n_missing_packets
    quality.sequence_order_violation_count = order_diagnostics.n_order_violations
    quality.sequence_reset_count = max(0, order_diagnostics.n_segments - 1)
    packet_times_us = packet_ds[packet_time_coordinate].values

    # Start building the dataset containing expanded sample fields
    sample_ds = xr.Dataset()

    # Process each sample group
    expanded_fields = set()  # Track fields that are expanded to remove from main array

    for sample_group in packet_config.sample_groups:
        # Process sample group (unified handling for single and multi-sample cases)
        field_arrays, sample_times = _expand_sample_group(packet_ds, sample_group)

        # Create dimension name
        sample_time_dimension = sample_group.sample_time_dimension

        # Create separate DataArray for each field
        for field_name, field_data in field_arrays.items():
            sample_ds[field_name] = xr.DataArray(
                data=field_data,
                dims=[sample_time_dimension],
                coords={sample_time_dimension: (sample_time_dimension, sample_times)},
            )

        # Create packet_index variable to map samples back to their originating packets
        n_packets = packet_ds.sizes[SDC_PACKET_DIMENSION]
        n_samples = sample_group.sample_count
        # Create an array that repeats each packet index n_samples times
        # e.g., for 3 packets with 2 samples each: [0, 0, 1, 1, 2, 2]
        packet_indices = np.repeat(np.arange(n_packets), n_samples)
        packet_index_var_name = f"{sample_group.name}_packet_index"
        sample_ds[packet_index_var_name] = xr.DataArray(
            data=packet_indices,
            dims=[sample_time_dimension],
            coords={sample_time_dimension: (sample_time_dimension, sample_times)},
        )

        # Track expanded sample fields (including time fields) to remove from main array
        expanded_fields.update(_get_expanded_field_names(packet_ds, sample_group))

        # Drop and warn about duplicate samples
        # NOTE: This should never find duplicates in flight but in ground testing, FSW was generating
        # packets that had repeated sample timestamps due to an issue with Hydra simulating SC time pulses
        # incorrectly, causing a microsecond counter to roll over at 1E6 without incrementing the second counter.
        sample_ds, sample_duplicates = _drop_duplicates(
            sample_ds, sample_time_dimension, ground_data=ground_data, verbose=verbose
        )
        quality.duplicate_sample_time_count += sample_duplicates.n_duplicates
        quality.duplicate_value_mismatch_count += sample_duplicates.n_value_mismatches
        quality.mismatched_variables.extend(sample_duplicates.mismatched_variables)
        quality.duplicate_evidence.extend(sample_duplicates.evidence)

        # Sort the data by the newly added dimension for the sample group
        sample_ds = sample_ds.sortby(sample_time_dimension)
        sample_times_sorted = sample_ds[sample_time_dimension].values
        quality.n_samples = max(quality.n_samples, int(sample_times_sorted.size))
        if sample_times_sorted.size > 1:
            gaps = np.diff(sample_times_sorted.astype(DATETIME_USEC_DTYPE).astype(np.int64))
            quality.max_sample_gap_microseconds = max(quality.max_sample_gap_microseconds, int(gaps.max(initial=0)))

    # Drop expanded sample fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(expanded_fields)

    # Process aggregation groups
    aggregated_fields = set()  # Track fields that are aggregated to remove from main array

    for agg_group in packet_config.aggregation_groups:
        # Aggregate the fields
        aggregated_data = _aggregate_fields(packet_ds, agg_group)

        # Add aggregated variable to packet dataset with packet dimension
        packet_ds[agg_group.name] = xr.DataArray(
            data=aggregated_data,
            dims=[SDC_PACKET_DIMENSION],
            coords={packet_time_coordinate: (SDC_PACKET_DIMENSION, packet_times_us)},
        )

        # Track aggregated fields to remove from main array
        aggregated_fields.update(_get_aggregated_field_names(packet_ds, agg_group))

    # Drop aggregated fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(aggregated_fields)

    # Process array groups
    array_group_fields = set()  # Track fields that are stacked to remove from main array

    for array_group in packet_config.array_groups:
        # Stack the fields
        stacked_data = _stack_fields(packet_ds, array_group)
        index_coord = np.arange(array_group.field_count, dtype=np.int64)

        # Add stacked variable to packet dataset with packet and array dimensions
        packet_ds[array_group.name] = xr.DataArray(
            data=stacked_data,
            dims=[SDC_PACKET_DIMENSION, array_group.dimension],
            coords={
                packet_time_coordinate: (SDC_PACKET_DIMENSION, packet_times_us),
                array_group.dimension: (array_group.dimension, index_coord),
            },
        )

        # Track stacked fields to remove from main array
        array_group_fields.update(_get_array_group_field_names(packet_ds, array_group))

    # Drop stacked fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(array_group_fields)

    # Merge sample variables into packet_ds
    # This works because the coordinates and dimensions in sample_ds are different than the
    # coordinates and dimensions in packet_ds
    packet_ds = packet_ds.merge(sample_ds)

    # The "PACKET" dimension is retained to enable sample-to-packet traceability
    # packet_time_dimension remains as a non-dimension coordinate

    # Force any Unicode (U-type) string variables to byte (S-type) strings. We know all the FSW strings
    # are ASCII and sticking to bytes preserves better compatibility in NetCDF (i.e. doesn't rely on the NC_STRING
    # type).
    for var_name, var in packet_ds.data_vars.items():
        if var.dtype.kind == "U":
            packet_ds[var_name] = var.astype("S")

    # Add global dynamic attributes that are required per the data product configs but do not have static values
    global_attrs: dict[str, str | list | set] = {
        "algorithm_version": version(),
        "date_created": datetime.now(tz=UTC).isoformat(),
    }
    # Recorded in acquisition order, which is the order the packets were concatenated in and so
    # may differ from the order the caller passed.
    global_attrs["input_files"] = [f.name for f in _packet_files]
    packet_ds.attrs.update(global_attrs)
    # Written unconditionally, zeros included: every product definition declares these, and a
    # trending query should not have to distinguish "clean" from "not reported".
    quality.product_id = str(packet_ds.attrs.get("ProductID", ""))
    packet_ds.attrs.update(quality.global_attributes())

    if packet_config.packet_apid == LiberaApid.icie_wfov_sci:
        packet_ds = enhance_wfov_l1a_dataset(packet_ds)

    return packet_ds


DEFAULT_DUPLICATE_EVIDENCE_LIMIT = 200


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """What deduplicating one coordinate found and dropped.

    Attributes
    ----------
    n_duplicates : int
        Rows dropped, i.e. coordinate values beyond the first occurrence of each.
    n_value_mismatches : int
        Duplicated coordinate values whose rows did not agree across every data variable.
        Dropping one of those rows discarded a distinct measurement, not redundancy, so this is
        the number that decides whether a granule lost science.
    mismatched_variables : tuple of str
        Data variables that disagreed, in the order first seen.
    evidence : tuple of DuplicateEvidence
        Per-value detail for the mismatches, capped at the caller's evidence limit.
    n_evidence_omitted : int
        Mismatches beyond the cap, which have no evidence recorded.
    """

    n_duplicates: int = 0
    n_value_mismatches: int = 0
    mismatched_variables: tuple[str, ...] = ()
    evidence: tuple[DuplicateEvidence, ...] = ()
    n_evidence_omitted: int = 0


def _validate_duplicate_values(
    dataset: xr.Dataset,
    coordinate_name: str,
    duplicates: np.ndarray,
    ground_data: bool = False,
    verbose: bool = False,
    *,
    strict: bool = False,
    evidence_limit: int = DEFAULT_DUPLICATE_EVIDENCE_LIMIT,
) -> DuplicateReport:
    """Report whether duplicate coordinate entries are identical across all data variables.

    Only applies when the coordinate being deduplicated is itself a dimension coordinate
    (i.e. coordinate_name == dim_name). For non-dimension coordinates, the underlying dimension
    indices are inherently distinct so value-identity checks are not meaningful.

    Identity, not the ground/flight distinction, is what decides whether a duplicate is safe to
    drop: rows that agree are redundancy in either mode, and rows that disagree are a lost
    measurement in either mode. So this counts and reports unconditionally, and ``strict``
    alone decides whether a mismatch also stops the granule. It is off by default because
    stopping does not recover the measurement, while a recorded count lets the granule through
    and leaves the loss visible and trendable. Turn it on where a mismatch must block.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset containing the duplicates to validate.
    coordinate_name : str
        The name of the coordinate being deduplicated (used in messages).
    duplicates : np.ndarray
        The coordinate values that appear more than once.
    ground_data : bool, optional
        Recorded on the log record; does not change what is checked or reported.
    verbose : bool, optional
        Emit a ``UserWarning`` for each non-identical duplicate. Default False, since a real
        granule can carry ~10,000 of them.
    strict : bool, optional
        Raise ``ValueError`` on the first non-identical duplicate instead of reporting it.
        Default False.
    evidence_limit : int, optional
        Maximum number of mismatches to record values for.

    Returns
    -------
    DuplicateReport
        Mismatch counts and per-value evidence. ``n_duplicates`` is left at 0 here; the caller
        fills it in.

    Raises
    ------
    ValueError
        If any duplicate has differing values in any variable and ``strict`` is True.
    """
    # Non-dimension coordinates share a dimension with other variables but
    # don't index them directly — rows are inherently distinct, so
    # value-identity validation is not applicable.
    dim_name = dataset[coordinate_name].dims[0]
    coord_values = dataset[coordinate_name].values
    if coordinate_name != dim_name:
        return DuplicateReport()

    # Build a boolean mask marking every row whose coord value appears in `duplicates`.
    dup_mask = np.isin(coord_values, duplicates)
    dup_indices_all = np.where(dup_mask)[0]
    dup_coord_vals = coord_values[dup_indices_all]

    sort_order = np.argsort(dup_coord_vals, kind="stable")
    dup_indices_sorted = dup_indices_all[sort_order]
    dup_coord_vals_sorted = dup_coord_vals[sort_order]

    dup_slice = dataset.isel({dim_name: dup_indices_sorted})

    # Computed once, outside the loop, using the sorted array:
    unique_dup_vals, group_starts = np.unique(dup_coord_vals_sorted, return_index=True)
    group_ends = np.append(group_starts[1:], len(dup_coord_vals_sorted))

    mismatched = np.zeros(unique_dup_vals.size, dtype=bool)
    mismatched_variables: list[str] = []
    evidence_by_group: dict[int, DuplicateEvidence] = {}

    for var_name, var in dup_slice.data_vars.items():
        data = var.values
        variable_mismatched = False

        for group, (dup_val, start, end) in enumerate(zip(unique_dup_vals, group_starts, group_ends)):
            group_data = data[start:end]
            # Vectorized identity check: broadcast first row against all rows in the group.
            if np.all(group_data == group_data[0]):
                continue
            variable_mismatched = True
            mismatched[group] = True
            if strict:
                pairs = list(zip(group_data, dup_coord_vals_sorted[start:end]))
                raise ValueError(
                    f"Duplicate coordinate value '{dup_val}' in '{coordinate_name}' "
                    f"has differing values in variable '{var_name}' at {pairs}. "
                    f"Dropping this duplicate would result in data loss."
                )
            if verbose:
                pairs = list(zip(group_data, dup_coord_vals_sorted[start:end]))
                warnings.warn(
                    f"Duplicate coordinate value '{dup_val}' in '{coordinate_name}' "
                    f"has differing values in variable '{var_name}' at {pairs}."
                )
            if len(evidence_by_group) < evidence_limit or group in evidence_by_group:
                row_indices = [int(i) for i in dup_indices_sorted[start:end]]
                record = evidence_by_group.setdefault(
                    group,
                    DuplicateEvidence(
                        coordinate=coordinate_name,
                        timestamp=str(dup_val),
                        multiplicity=int(end - start),
                        row_indices=row_indices,
                    ),
                )
                record.variables[var_name] = [_jsonable(v) for v in group_data]

        if variable_mismatched:
            mismatched_variables.append(str(var_name))

    n_mismatches = int(mismatched.sum())
    return DuplicateReport(
        n_value_mismatches=n_mismatches,
        mismatched_variables=tuple(mismatched_variables),
        evidence=tuple(evidence_by_group[group] for group in sorted(evidence_by_group)),
        n_evidence_omitted=max(0, n_mismatches - len(evidence_by_group)),
    )


def _jsonable(value):
    """Convert a numpy scalar to something ``json`` can write."""
    return value.item() if isinstance(value, np.generic) else value


def _drop_duplicates(
    dataset: xr.Dataset,
    coordinate_name: str,
    ground_data: bool = False,
    verbose: bool = False,
    *,
    strict: bool = False,
    evidence_limit: int = DEFAULT_DUPLICATE_EVIDENCE_LIMIT,
) -> tuple[xr.Dataset, DuplicateReport]:
    """Drop rows beyond the first occurrence of each coordinate value, and report what was lost.

    Rows are kept in their original order; deduplicating does not reorder the axis.

    Duplicates whose rows agree are redundancy and cost nothing to drop. Duplicates whose rows
    disagree are a real measurement discarded to make the timestamp unique, and the granule is
    let through with that recorded rather than stopped, because stopping does not recover the
    measurement. Pass ``strict=True`` where a mismatch must block instead.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to deduplicate.
    coordinate_name : str
        The name of the coordinate over which to search for duplicates. Can be either a
        dimension coordinate or a non-dimension coordinate; value identity is only checkable
        for the former.
    ground_data : bool, optional
        Recorded on the log record; does not change what is dropped.
    verbose : bool, optional
        Emit a ``UserWarning`` for each non-identical duplicate. Default False.
    strict : bool, optional
        Raise ``ValueError`` on a non-identical duplicate rather than reporting it.
    evidence_limit : int, optional
        Maximum number of mismatches to record values for.

    Returns
    -------
    dataset : xr.Dataset
        Deduplicated dataset.
    report : DuplicateReport
        What was dropped, and how much of it disagreed.

    Raises
    ------
    KeyError
        If the coordinate is not found in the dataset.
    ValueError
        If the coordinate is not 1-dimensional, or a duplicate's rows differ while ``strict``.
    """
    # Validate coordinate exists
    if coordinate_name not in dataset.coords:
        raise KeyError(f"Coordinate '{coordinate_name}' not found in dataset")

    coord = dataset[coordinate_name]

    # Ensure coordinate is 1-dimensional
    if len(coord.dims) != 1:
        raise ValueError(
            f"Coordinate '{coordinate_name}' must be 1-dimensional to deduplicate, but has dimensions: {coord.dims}"
        )

    dim_name = coord.dims[0]

    # Optimize for the common case (no duplicates) by checking size first
    # Use np.unique to find first occurrence of each unique value
    coord_values = coord.values
    # Obtain unique values, their first-occurrence indices, and per-value counts in one pass.
    unique_values, unique_indices, counts = np.unique(coord_values, return_index=True, return_counts=True)

    # Sort indices to maintain original order in the dataset
    # np.unique sorts the VALUES (not indices), so indices may be out of order
    # We want to preserve the original row order when selecting
    unique_indices_sorted = np.sort(unique_indices)

    original_size = len(coord_values)
    n_duplicates = original_size - len(unique_indices_sorted)

    if n_duplicates == 0:
        return dataset, DuplicateReport()

    duplicates = unique_values[counts > 1]
    report = _validate_duplicate_values(
        dataset,
        coordinate_name,
        duplicates,
        ground_data,
        verbose,
        strict=strict,
        evidence_limit=evidence_limit,
    )
    report = replace(report, n_duplicates=n_duplicates)

    # Select only the first occurrence of each unique coordinate value
    dataset_deduped = dataset.isel({dim_name: unique_indices_sorted})

    warnings.warn(
        f"Detected {n_duplicates} duplicate {coordinate_name} in dataset, "
        f"{report.n_value_mismatches} of which have differing data values. "
        f"Use verbose=True to see warnings for the duplicates."
    )
    logger.warning(
        {
            "msg": "duplicate_coordinate_values_dropped",
            "coordinate": coordinate_name,
            "ground_data": ground_data,
            "n_duplicates": n_duplicates,
            "n_value_mismatches": report.n_value_mismatches,
            "mismatched_variables": list(report.mismatched_variables),
        }
    )
    if report.n_value_mismatches:
        logger.error(
            {
                "msg": "duplicate_coordinate_values_differed",
                "coordinate": coordinate_name,
                "ground_data": ground_data,
                "n_value_mismatches": report.n_value_mismatches,
                "mismatched_variables": list(report.mismatched_variables),
                "detail": (
                    "Rows sharing a timestamp carried different data. Dropping one of each pair "
                    "discarded a distinct measurement, not redundancy."
                ),
            }
        )

    return dataset_deduped, report


def _expand_sample_group(dataset: xr.Dataset, group: SampleGroup) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Expand a sample group (timestamps and measured values) into separate field arrays.

    For samples within a packet (1 or many), expand those samples into separate arrays,
    with coordinates of sample time rather than packet time.

    Notes
    -----
    For periodic samples based on an epoch, we use the epoch and the period to calculate sample times assuming
    that the epoch is the first sample time.
    For samples that each have their own timestamp, we convert each sample time to microseconds.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the packet data.
    group : SampleGroup
        Configuration for the sample group.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray]
        Dictionary of field name to field array, and time array.
    """
    n_samples = group.sample_count

    # Calculate sample times
    if group.time_field_patterns:
        # Explicit per-sample timestamps
        sample_times = expand_sample_times(dataset, group.time_field_patterns, n_samples)
    elif group.epoch_time_fields and group.sample_period:
        # Use epoch + period to calculate sample timestamps
        epoch_times_dt64 = multipart_to_dt64(dataset, **group.epoch_time_fields.multipart_kwargs)
        epoch_times_us = epoch_times_dt64.values.astype(DATETIME_USEC_DTYPE)
        period_us = np.timedelta64(int(group.sample_period.total_seconds() * 1e6), "us")
        # Epoch times are 1 per packet so create an array that is (n_samples, n_packets), transpose, and flatten it
        sample_times = np.array([epoch_times_us + i * period_us for i in range(n_samples)]).T.flatten()
    else:
        raise ValueError(f"Sample group {group.name} must have either time_fields or epoch_time_fields+sample_period")

    # Expand data fields into individual arrays
    field_arrays = {}
    for field_pattern, clean_field_name in zip(group.data_field_patterns, group.sample_data_fields):
        if group.sample_count > 1:
            # Multi-sample field - collect all samples for this field pattern
            field_data = []
            for i in range(n_samples):
                if (field_name_i := field_pattern % i) in dataset:
                    field_data.append(dataset[field_name_i].values)
            # field_data is a list of length n_samples containing arrays with length n_packets
            # Stack samples (n_packets, n_samples) and flatten: (n_packets, n_samples) -> (n_packets * n_samples,)
            stacked_data = np.stack(field_data, axis=1)
            field_arrays[clean_field_name] = stacked_data.flatten()
        else:
            # Single sample per packet
            field_arrays[field_pattern] = dataset[field_pattern].values

    return field_arrays, sample_times


def expand_sample_times(dataset: xr.Dataset, time_fields: TimeFieldMapping, n_samples: int) -> np.ndarray:
    """Expand sample time fields into a flat array.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the time fields.
    time_fields : TimeFieldMapping
        Time field mapping with patterns that may include %i placeholders.
    n_samples : int
        Number of samples per packet.

    Returns
    -------
    np.ndarray
        Flattened array of sample times as datetime64[us].
    """
    if n_samples > 1:
        # Multiple samples per packet - need to expand %i patterns
        sample_times_list = []
        for i in range(n_samples):
            # Create TimeFieldMapping for this specific sample index
            sample_kwargs = {}
            for field_type, field_pattern in time_fields.multipart_kwargs.items():
                if field_pattern is not None:
                    sample_field_name = field_pattern % i
                    if sample_field_name in dataset:
                        sample_kwargs[field_type] = sample_field_name

            if sample_kwargs:
                sample_time_dt64 = multipart_to_dt64(dataset, **sample_kwargs)
                sample_times_list.append(sample_time_dt64.values.astype(DATETIME_USEC_DTYPE))

        # Stack samples (n_packets, n_samples) and flatten
        if sample_times_list:
            stacked_times = np.stack(sample_times_list, axis=1)
            return stacked_times.flatten()
        else:
            # No valid time fields found
            return np.array([], dtype=DATETIME_USEC_DTYPE)
    else:
        # Single sample per packet - use time_fields directly
        sample_time_dt64 = multipart_to_dt64(dataset, **time_fields.multipart_kwargs)
        return sample_time_dt64.values.astype(DATETIME_USEC_DTYPE)


def _get_expanded_field_names(dataset: xr.Dataset, group: SampleGroup) -> set[str]:
    """Get all field names that are expanded for a sample group.

    This extracts all the field names for a sample group that we use to expand the samples
    (time fields and data fields) so that we can remove these fields from the primary array to save space.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the fields.
    group : SampleGroup
        Sample group configuration.

    Returns
    -------
    set[str]
        Set of field names that are expanded.
    """
    expanded = set()

    # Add data fields
    for field_pattern in group.data_field_patterns:
        if (n_samples := group.sample_count) > 1:
            # Multi-sample pattern
            for i in range(n_samples):
                if (field_name := field_pattern % i) in dataset:
                    expanded.add(field_name)
        else:
            # Single field
            if field_pattern in dataset:
                expanded.add(field_pattern)

    # Add time fields
    if group.time_field_patterns:
        # Times provided per sample
        for field_pattern in group.time_field_patterns.multipart_kwargs.values():
            if field_pattern is not None:
                if (n_samples := group.sample_count) > 1:
                    for i in range(n_samples):
                        if (field_name := field_pattern % i) in dataset:
                            expanded.add(field_name)
                else:
                    if field_pattern in dataset:
                        expanded.add(field_pattern)
    elif group.epoch_time_fields:
        # Times calculated from epoch and periodic sampling
        for field_name in group.epoch_time_fields.multipart_kwargs.values():
            if field_name is not None and field_name in dataset:
                expanded.add(field_name)

    return expanded


def _aggregate_fields(dataset: xr.Dataset, group: AggregationGroup) -> np.ndarray:
    """Aggregate multiple sequential fields into a single binary blob per packet.

    Optimized using vectorized numpy operations with zero-copy view conversion.
    Assumes all fields exist (validated by Space Packet Parser during parsing).
    Assumes all fields are interpretable as bytes objects regardless of original dtype.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the individual fields to aggregate.
    group : AggregationGroup
        Configuration for the aggregation group.

    Returns
    -------
    np.ndarray
        Array of aggregated binary data with dtype matching group.dtype.
    """
    n_packets = dataset.sizes[SDC_PACKET_DIMENSION]

    if group.dtype.kind != "S" or group.dtype.itemsize <= 0:
        raise ValueError(f"Aggregation group {group.name} requires fixed-width byte-string dtype; got {group.dtype}.")
    if group.dtype.itemsize % group.field_count != 0:
        raise ValueError(
            f"Aggregation group {group.name} has invalid dtype size {group.dtype.itemsize} for "
            f"field_count {group.field_count}; size must be divisible by field_count."
        )

    # Extract all field arrays at once (fail fast if any missing)
    field_arrays = []
    aggregate_size = 0  # Track total size of aggregated fields (per packet)
    bytes_per_field = group.dtype.itemsize // group.field_count

    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name not in dataset:
            raise KeyError(f"Required field {field_name} not found for aggregation group {group.name}")

        # The field_array is the data for one field across all packets
        field_array = dataset[field_name].values
        field_size = field_array.dtype.itemsize

        if field_array.dtype.kind in {"U", "S"}:
            # Packet parser commonly emits fixed-width ASCII-like telemetry as Unicode.
            # Cast explicitly to bytes so itemsize checks and final view() are deterministic.
            field_array = field_array.astype(f"S{bytes_per_field}")
            field_size = field_array.dtype.itemsize

        # Adds up total size of the aggregated fields (per packet)
        # This allows for fields that are not all the same size/dtype to be aggregated together as bytes
        aggregate_size += field_size

        field_arrays.append(field_array)

    if aggregate_size != group.dtype.itemsize:
        raise ValueError(
            f"Aggregation group {group.name} size mismatch: "
            f"expected total size {group.dtype.itemsize} bytes, got {aggregate_size} bytes."
        )

    # Stack all fields: shape (n_fields, n_packets)
    stacked_and_transposed = np.stack(field_arrays, axis=-1)

    # Use view() to reinterpret each row as a single bytes string - zero copy!
    # This is a key optimization - no iteration, just memory reinterpretation
    return stacked_and_transposed.view(dtype=group.dtype).reshape(n_packets)


def _get_aggregated_field_names(dataset: xr.Dataset, group: AggregationGroup) -> set[str]:
    """Get all field names that are aggregated for an aggregation group.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the fields.
    group : AggregationGroup
        Aggregation group configuration.

    Returns
    -------
    set[str]
        Set of field names that are aggregated.
    """
    aggregated = set()
    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name in dataset:
            aggregated.add(field_name)
    return aggregated


def _normalize_field_dtype(
    field_array: np.ndarray,
    target_dtype: np.dtype,
    *,
    field_name: str,
    group_name: str,
) -> np.ndarray:
    """Normalize a field array to the expected group dtype with explicit coercion rules.

    Only Unicode-to-bytes normalization is allowed for string fields. All other dtype
    mismatches raise ValueError to avoid silent cross-kind coercion (e.g. int -> S).
    """
    if field_array.dtype == target_dtype:
        return field_array

    source_kind = field_array.dtype.kind
    target_kind = target_dtype.kind

    if source_kind == "U" and target_kind == "S":
        return field_array.astype(target_dtype)

    if source_kind == "S" and target_kind == "S":
        if field_array.dtype.itemsize != target_dtype.itemsize:
            raise ValueError(
                f"Array group {group_name}: field {field_name} has width {field_array.dtype}, expected {target_dtype}."
            )
        return field_array.astype(target_dtype)

    raise ValueError(
        f"Array group {group_name}: field {field_name} has dtype {field_array.dtype}, expected {target_dtype}."
    )


def _stack_fields(dataset: xr.Dataset, group: ArrayGroup) -> np.ndarray:
    """Stack multiple sequential fields into a fixed-size array per packet.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the individual fields to stack.
    group : ArrayGroup
        Configuration for the array group.

    Returns
    -------
    np.ndarray
        Array with shape (n_packets, field_count) and dtype matching group.dtype.
    """
    field_arrays = []

    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name not in dataset:
            raise KeyError(f"Required field {field_name} not found for array group {group.name}")

        field_array = _normalize_field_dtype(
            dataset[field_name].values,
            group.dtype,
            field_name=field_name,
            group_name=group.name,
        )
        field_arrays.append(field_array)

    return np.stack(field_arrays, axis=1)


def _get_array_group_field_names(dataset: xr.Dataset, group: ArrayGroup) -> set[str]:
    """Get all field names that are stacked for an array group.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the fields.
    group : ArrayGroup
        Array group configuration.

    Returns
    -------
    set[str]
        Set of field names that are stacked.
    """
    array_fields = set()
    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name in dataset:
            array_fields.add(field_name)
    return array_fields
