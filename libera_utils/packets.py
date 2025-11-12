"""Module for reading packet data using Space Packet Parser"""

import logging
import warnings
from datetime import UTC, datetime
from os import PathLike
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import AnyPath
from space_packet_parser.xarr import create_dataset
from space_packet_parser.xtce.definitions import XtcePacketDefinition

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.filenaming import PathType
from libera_utils.packet_configs import (
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


def parse_packets_to_dataframe(
    packet_definition: str | PathType | XtcePacketDefinition,
    packet_data_filepaths: list[str | PathType],
    apid: int | None = None,
    skip_header_bytes: int = 0,
) -> pd.DataFrame:
    """Parse packets from files into a pandas DataFrame using Space Packet Parser v6.0.0rc3.

    Parameters
    ----------
    packet_definition : str | PathType | XtcePacketDefinition
        XTCE packet definition file path or pre-loaded XtcePacketDefinition object.
    packet_data_filepaths : list[str | PathType]
        List of filepaths to packet files.
    apid : Optional[int]
        Filter on APID so we don't get mismatches in case the parser finds multiple parsable packet definitions
        in the files. This can happen if the XTCE document contains definitions for multiple packet types and >1 of
        those packet types is present in the packet data files.
    skip_header_bytes : int
        Number of header bytes to skip when reading packet files. Default is 0.

    Returns
    -------
    pd.DataFrame
        pandas DataFrame containing parsed packet data.
    """
    logger.info("Parsing packets from %d file(s) into pandas DataFrame", len(packet_data_filepaths))

    dataset_dict = create_dataset(
        packet_files=[AnyPath(f) for f in packet_data_filepaths],
        xtce_packet_definition=AnyPath(packet_definition),
        generator_kwargs=dict(skip_header_bytes=skip_header_bytes),
    )

    # Handle APID filtering
    if apid is not None:
        if apid in dataset_dict:
            dataset = dataset_dict[apid]
        else:
            raise ValueError(
                f"Requested APID {apid} not found in parsed packets. Available APIDs: {list(dataset_dict.keys())}"
            )
    else:
        # No APID specified - check if we have multiple APIDs
        if len(dataset_dict) > 1:
            raise ValueError(
                f"Multiple APIDs present ({list(dataset_dict.keys())}). You must specify which APID you want."
            )
        elif len(dataset_dict) == 1:
            # Single APID present, use it
            dataset = next(iter(dataset_dict.values()))
        else:
            raise ValueError("No packets found in files")

    # Remove duplicates by converting to DataFrame, dropping duplicates, and converting back
    # This maintains compatibility with the original behavior
    df = dataset.to_dataframe().reset_index()
    # Drop duplicates based on packet data, not including the original index
    packet_columns = [col for col in df.columns if col not in ["index", "packet"]]
    df_unique = df.drop_duplicates(subset=packet_columns)

    if len(df_unique) < len(df):
        logger.info("Removed %d duplicate packets", len(df) - len(df_unique))

    # Return the unique DataFrame
    return df_unique


def read_sc_packet_data(packet_data_filepaths: list[str | PathType], apid: int = 11) -> pd.DataFrame:
    """Read spacecraft packet data from a list of file paths.

    Parameters
    ----------
    packet_data_filepaths : list[str | PathType]
        The list of file paths to the raw packet data
    apid : int
        Application Packet ID to filter for. Default is 11 for JPSS geolocation packets.

    Returns
    -------
    packet_data : pd.DataFrame
        The configured packet data as a pandas DataFrame.
    """
    packet_definition_uri = cast(PathType, AnyPath(config.get("JPSS_GEOLOCATION_PACKET_DEFINITION")))
    logger.info("Using packet definition %s", packet_definition_uri)

    # Parse packets to DataFrame
    df = parse_packets_to_dataframe(
        packet_definition=packet_definition_uri, packet_data_filepaths=packet_data_filepaths, apid=apid
    )

    return df


def read_azel_packet_data(packet_data_filepaths: list[str | PathType], apid: int = 1048) -> pd.DataFrame:
    """Read Az/El packet data from a list of file paths.

     Parameters
    ----------
    packet_data_filepaths : list[str | Path | CloudPath]]
        The list of file paths to the raw packet data
    apid : int
        Application Packet ID to filter for. Default is 1048 for Az/El sample packets.

    Returns
    -------
    packet_data : pd.DataFrame
        The configured packet data as a pandas DataFrame with restructured samples.
    """
    packet_definition_uri = cast(PathType, AnyPath(config.get("LIBERA_PACKET_DEFINITION")))
    logger.info("Using packet definition %s", packet_definition_uri)

    # Parse packets to DataFrame
    df = parse_packets_to_dataframe(
        packet_definition=packet_definition_uri,
        packet_data_filepaths=packet_data_filepaths,
        apid=apid,
        skip_header_bytes=8,
    )

    # Restructure the DataFrame to have one row per sample (50 samples per packet)
    # Each packet contains 50 samples with fields like:
    # ICIE__AXIS_SAMPLE_TM_SEC0, ICIE__AXIS_SAMPLE_TM_SUB0, ICIE__AXIS_EL_FILT0, ICIE__AXIS_AZ_FILT0
    # ...
    # ICIE__AXIS_SAMPLE_TM_SEC49, ICIE__AXIS_SAMPLE_TM_SUB49, ICIE__AXIS_EL_FILT49, ICIE__AXIS_AZ_FILT49

    samples_list = []
    for _, packet_row in df.iterrows():
        # Get packet metadata for debugging
        src_seq_ctr = packet_row["SRC_SEQ_CTR"]
        pkt_day = packet_row["ICIE__TM_DAY_AXIS_SAMPLE"]
        pkt_ms = packet_row["ICIE__TM_MS_AXIS_SAMPLE"]
        pkt_us = packet_row["ICIE__TM_US_AXIS_SAMPLE"]

        for i in range(50):
            sample = {
                "SAMPLE_SEC": packet_row[f"ICIE__AXIS_SAMPLE_TM_SEC{i}"],
                "SAMPLE_USEC": packet_row[f"ICIE__AXIS_SAMPLE_TM_SUB{i}"],
                "AZ_ANGLE_RAD": packet_row[f"ICIE__AXIS_AZ_FILT{i}"],
                "EL_ANGLE_RAD": packet_row[f"ICIE__AXIS_EL_FILT{i}"],
                # Keep metadata for debugging
                "SRC_SEQ_CTR": src_seq_ctr,
                "PKT_DAY": pkt_day,
                "PKT_MS": pkt_ms,
                "PKT_US": pkt_us,
                "SAMPLE_INDEX": i,
            }
            samples_list.append(sample)

    restructured_df = pd.DataFrame(samples_list)

    # Check for duplicate timestamps and log detailed information
    # FIXME: [LIBSDC-608] This is only here to help with debugging for FSW purposes. This logs a verbose listing of duplicate sample timestamps.
    duplicates_mask = restructured_df.duplicated(subset=["SAMPLE_SEC", "SAMPLE_USEC"], keep=False)
    if duplicates_mask.any():
        duplicate_samples = restructured_df[duplicates_mask].sort_values(["SAMPLE_SEC", "SAMPLE_USEC"])

        logger.warning("Found %d samples with duplicate timestamps", duplicates_mask.sum())
        logger.warning("Duplicate timestamp details:")

        # Group duplicates by timestamp and show details
        for (sec, usec), group in duplicate_samples.groupby(["SAMPLE_SEC", "SAMPLE_USEC"]):
            logger.warning("  Timestamp: SEC=%d, USEC=%d", sec, usec)
            for _, row in group.iterrows():
                logger.warning(
                    "    - Packet SRC_SEQ_CTR=%s, PKT_TIME=(DAY=%s, MS=%s, US=%s), "
                    "Sample #%d, AZ=%.6f rad, EL=%.6f rad",
                    row["SRC_SEQ_CTR"],
                    row["PKT_DAY"],
                    row["PKT_MS"],
                    row["PKT_US"],
                    row["SAMPLE_INDEX"],
                    row["AZ_ANGLE_RAD"],
                    row["EL_ANGLE_RAD"],
                )
        # Remove duplicates, keeping the first occurrence
        restructured_df = restructured_df.drop_duplicates(subset=["SAMPLE_SEC", "SAMPLE_USEC"], keep="first")
        num_removed = duplicates_mask.sum() - len(
            restructured_df[restructured_df.duplicated(subset=["SAMPLE_SEC", "SAMPLE_USEC"], keep=False)]
        )
        logger.info("Removed %d duplicate timestamps from Az/El data (kept first occurrence)", num_removed)

    # Drop the debugging columns before returning
    restructured_df = restructured_df[["SAMPLE_SEC", "SAMPLE_USEC", "AZ_ANGLE_RAD", "EL_ANGLE_RAD"]]

    return restructured_df


# L1A Processing Additions
# ========================
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
    if apid in dataset_dict:
        ds = dataset_dict[apid]
    else:
        raise ValueError(
            f"Requested APID {apid} not found in parsed packets. Available APIDs: {list(dataset_dict.keys())}"
        )

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
    packet_config = get_packet_config(LiberaApid(apid))
    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    packet_ds = parse_packets_to_dataset(
        packet_files, packet_definition_path, apid, **packet_config.packet_generator_kwargs
    )
    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype("datetime64[us]")

    # Set packet time as a non-dimension coordinate with "packet" dimension
    # The packet dimension remains as-is from SPP to enable sample-to-packet tracing
    packet_time_coordinate = packet_config.packet_time_coordinate
    packet_ds = packet_ds.assign_coords({packet_time_coordinate: (SPP_PACKET_DIMENSION, packet_times_us)})

    # Drop duplicates from the packet dataset before we process samples
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

    # Add global attributes
    # TODO[LIBSDC-622]: Once the netcdf writer is working, this step should be outsourced to the writer to add correct metadata
    # Any existing dataset metadata should also be included in the product though
    global_attrs = {
        "Format": "NetCDF-4",
        "Conventions": "CF-1.8",
        "ProjectLongName": "Libera",
        "ProjectShortName": "Libera",
        "PlatformLongName": "TBD",  # Probably only needed if we are going with JPSS-4 as the platform identifier instead of NOAA-22
        "PlatformShortName": "NOAA-22",
        "AlgorithmVersion": version(),
        "Created": datetime.now(tz=UTC).isoformat(),
    }
    for i, f in enumerate(packet_files):
        global_attrs[f"INPUT_FILE{i}"] = str(f)
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
        The name of the coordinate over which to search for duplicates

    Returns
    -------
    dataset : xr.Dataset
        Deduplicated dataset
    n_duplicates : int
        Number of duplicates detected and dropped
    """
    coords = dataset[coordinate_name]
    unique, counts = np.unique(coords, return_counts=True)
    duplicates = unique[counts > 1]

    if n_duplicates := len(duplicates) > 0:
        warnings.warn(f"Detected {n_duplicates} duplicate packet timestamps in dataset")
        logger.warning(f"Duplicate coordinates detected ({n_duplicates}) in {coordinate_name}: {duplicates}")
        # Create a mask for all duplicate records
        dataset = dataset.drop_duplicates(coordinate_name)

    return dataset, len(duplicates)


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
        epoch_times_us = epoch_times_dt64.values.astype("datetime64[us]")
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
                sample_times_list.append(sample_time_dt64.values.astype("datetime64[us]"))

        # Stack samples (n_packets, n_samples) and flatten
        if sample_times_list:
            stacked_times = np.stack(sample_times_list, axis=1)
            return stacked_times.flatten()
        else:
            # No valid time fields found
            return np.array([], dtype="datetime64[us]")
    else:
        # Single sample per packet - use time_fields directly
        sample_time_dt64 = multipart_to_dt64(dataset, **time_fields.multipart_kwargs)
        return sample_time_dt64.values.astype("datetime64[us]")


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

    Optimized version using vectorized numpy operations with zero-copy view conversion.
    Assumes all fields exist (validated by Space Packet Parser during parsing).

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
    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name not in dataset:
            raise KeyError(f"Required field {field_name} not found for aggregation group {group.name}")
        field_arrays.append(dataset[field_name].values)

    # Stack all fields: shape (n_fields, n_packets)
    stacked = np.stack(field_arrays, axis=0)

    # Transpose to (n_packets, n_fields) with contiguous memory layout
    transposed = np.ascontiguousarray(stacked.T)

    # Use view() to reinterpret each row as a single bytes string - zero copy!
    # This is the key optimization - no iteration, just memory reinterpretation
    return transposed.view(dtype=group.dtype).reshape(n_packets)


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
