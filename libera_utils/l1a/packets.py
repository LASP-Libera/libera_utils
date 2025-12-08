"""Module for reading packet data using Space Packet Parser"""

import logging
import warnings
from datetime import UTC, datetime
from os import PathLike
from typing import cast

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath
from space_packet_parser.xarr import create_dataset

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io import filenaming
from libera_utils.l1a.l1a_packet_configs import (
    AggregationGroup,
    SampleGroup,
    TimeFieldMapping,
    get_packet_config,
)
from libera_utils.time import multipart_to_dt64
from libera_utils.version import version

logger = logging.getLogger(__name__)

# SPP always creates Datasets with a non-coordinate "packet" dimension
# This is a constant and is not expected to change in SPP
SPP_PACKET_DIMENSION = "packet"

DATETIME_USEC_DTYPE = np.dtype("datetime64[us]")


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

    # Parse packets using space_packet_parser
    dataset_dict = create_dataset(
        packet_files=[AnyPath(f) for f in packet_files],
        xtce_packet_definition=packet_definition,
        generator_kwargs=generator_kwargs,
    )

    # Filter by APID
    try:
        ds = dataset_dict[apid]
    except KeyError as ke:
        raise KeyError(
            f"Requested APID {apid} not found in parsed packets. Available APIDs: {list(dataset_dict.keys())}"
        ) from ke

    return ds


def parse_packets_to_l1a_dataset(packet_files: list[PathLike | str], apid: int) -> xr.Dataset:
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

    Returns
    -------
    xr.Dataset
        xarray Dataset with:
        - Main packet data array with packet timestamp dimension
        - Separate arrays for each sample group with optional multi-field expansion
        - All time coordinates properly set as dimensions
    """
    _packet_files = [cast(filenaming.PathType, AnyPath(f)) for f in packet_files]
    packet_config = get_packet_config(LiberaApid(apid))
    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    # Ground test data packets have extra 8 byte headers that need to be skipped
    # When running ground test data, set SKIP_PACKET_HEADER_BYTES environment variable to 8
    skip_header_bytes = config.get("SKIP_PACKET_HEADER_BYTES")
    packet_ds = parse_packets_to_dataset(
        _packet_files, packet_definition_path, apid, skip_header_bytes=skip_header_bytes
    )
    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype(DATETIME_USEC_DTYPE)

    # Set packet time as a non-dimension coordinate with "packet" dimension
    # The packet dimension remains as-is from SPP to enable sample-to-packet tracing
    packet_time_coordinate = packet_config.packet_time_coordinate
    packet_ds = packet_ds.assign_coords({packet_time_coordinate: (SPP_PACKET_DIMENSION, packet_times_us)})

    # Drop duplicates from the packet dataset before we process samples
    # This drops full duplicate packets based on identical packet timestamps
    packet_ds, _ = _drop_duplicates(packet_ds, packet_time_coordinate)

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
        n_packets = packet_ds.sizes[SPP_PACKET_DIMENSION]
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
        sample_ds, _ = _drop_duplicates(sample_ds, sample_time_dimension)

        # Sort the data by the newly added dimension for the sample group
        sample_ds = sample_ds.sortby(sample_time_dimension)

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
            dims=[SPP_PACKET_DIMENSION],
            coords={packet_time_coordinate: (SPP_PACKET_DIMENSION, packet_times_us)},
        )

        # Track aggregated fields to remove from main array
        aggregated_fields.update(_get_aggregated_field_names(packet_ds, agg_group))

    # Drop aggregated fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(aggregated_fields)

    # Merge sample variables into packet_ds
    # This works because the coordinates and dimensions in sample_ds are different than the
    # coordinates and dimensions in packet_ds
    packet_ds = packet_ds.merge(sample_ds)

    # The "packet" dimension is retained to enable sample-to-packet traceability
    # packet_time_dimension remains as a non-dimension coordinate

    # Sort the resulting Dataset by the packet time coordinate to ensure data is properly ordered
    packet_ds = packet_ds.sortby(packet_time_coordinate)

    # Add global dynamic attributes that are required per the data product configs but do not have static values
    global_attrs: dict[str, str | list | set] = {
        "algorithm_version": version(),
        "date_created": datetime.now(tz=UTC).isoformat(),
    }
    global_attrs["input_files"] = [f.name for f in _packet_files]
    packet_ds.attrs.update(global_attrs)

    return packet_ds


def _drop_duplicates(dataset: xr.Dataset, coordinate_name: str):
    """Detect and drop duplicate values based on a coordinate

    Issues warnings when duplicates are detected

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to deduplicate
    coordinate_name : str
        The name of the coordinate over which to search for duplicates.
        Can be either a dimension coordinate or a non-dimension coordinate.

    Returns
    -------
    dataset : xr.Dataset
        Deduplicated dataset
    n_duplicates : int
        Number of duplicates detected and dropped
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
    unique_values, unique_indices = np.unique(coord_values, return_index=True)

    # Sort indices to maintain original order in the dataset
    # np.unique sorts the VALUES (not indices), so indices may be out of order
    # We want to preserve the original row order when selecting
    unique_indices_sorted = np.sort(unique_indices)

    original_size = len(coord_values)
    n_duplicates = original_size - len(unique_indices_sorted)

    if n_duplicates > 0:
        # Select only the first occurrence of each unique coordinate value
        dataset_deduped = dataset.isel({dim_name: unique_indices_sorted})

        # Log the duplicate values (not indices)
        _, counts = np.unique(coord_values, return_counts=True)
        duplicates = unique_values[counts > 1]

        warnings.warn(f"Detected {n_duplicates} duplicate packet timestamps in dataset")
        logger.warning(f"Duplicate coordinates detected ({n_duplicates}) in {coordinate_name}: {duplicates}")
    else:
        # No duplicates, return original dataset
        dataset_deduped = dataset

    return dataset_deduped, n_duplicates


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
        sample_times = _expand_sample_times(dataset, group.time_field_patterns, n_samples)
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


def _expand_sample_times(dataset: xr.Dataset, time_fields: TimeFieldMapping, n_samples: int) -> np.ndarray:
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
    n_packets = dataset.sizes[SPP_PACKET_DIMENSION]

    # Extract all field arrays at once (fail fast if any missing)
    field_arrays = []
    aggregate_size = 0  # Track total size of aggregated fields (per packet)

    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name not in dataset:
            raise KeyError(f"Required field {field_name} not found for aggregation group {group.name}")

        # The field_array is the data for one field across all packets
        field_array = dataset[field_name].values

        # Adds up total size of the aggregated fields (per packet)
        # This allows for fields that are not all the same size/dtype to be aggregated together as bytes
        aggregate_size += field_array.dtype.itemsize

        field_arrays.append(field_array)

    if not group.dtype.itemsize:
        warnings.warn(
            f"Aggregation group {group.name} has a dtype with unspecified size ({group.dtype}). This may lead to unexpected results."
        )
    elif aggregate_size != group.dtype.itemsize:
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
