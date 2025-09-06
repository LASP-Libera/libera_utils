"""Module for reading packet data using Space Packet Parser"""

import logging
from typing import cast

import pandas as pd
from cloudpathlib import AnyPath
from space_packet_parser.xarr import create_dataset
from space_packet_parser.xtce.definitions import XtcePacketDefinition

from libera_utils.config import config
from libera_utils.io.filenaming import PathType

logger = logging.getLogger(__name__)


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
    packet_definition_uri = cast(PathType, AnyPath(config.get("AZEL_PACKET_DEFINITION")))
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
