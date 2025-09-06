"""Tests for libera_utils.io.packets module"""

from unittest import mock

import pandas as pd
import pytest

from libera_utils import packets as libera_packets


@mock.patch("libera_utils.packets.create_dataset")
def test_parse_packets_to_dataframe_with_apid_filtering(mock_create_dataset):
    """Test parse_packets_to_dataframe properly filters by APID and removes duplicates"""
    # Mock the create_dataset return value - simulate multiple APIDs
    # Create actual duplicates by using identical data in all columns
    mock_df_1048 = pd.DataFrame(
        {
            "VERSION": [0, 0, 0],
            "TYPE": [0, 0, 0],
            "PKT_APID": [1048, 1048, 1048],
            "SRC_SEQ_CTR": [2428, 2429, 2428],  # Third row same as first
            "PARAM_1": ["foo", "bar", "foo"],  # Third row same as first
            "PARAM_2": [1.1, 2.2, 1.1],  # Third row same as first
        }
    )
    mock_df_1048.index.name = "packet"
    mock_dataset_1048 = mock_df_1048.to_xarray()

    mock_df_1059 = pd.DataFrame(
        {
            "VERSION": [0, 0],
            "TYPE": [0, 0],
            "PKT_APID": [1059, 1059],
            "SRC_SEQ_CTR": [85, 86],
            "PARAM_1": ["baz", "qux"],
            "PARAM_2": [3.3, 4.4],
        }
    )
    mock_df_1059.index.name = "packet"
    mock_dataset_1059 = mock_df_1059.to_xarray()

    mock_create_dataset.return_value = {1048: mock_dataset_1048, 1059: mock_dataset_1059}

    # Test filtering for APID 1048
    result_df = libera_packets.parse_packets_to_dataframe(
        packet_definition="fake_path.xml", packet_data_filepaths=["fake_file.bin"], apid=1048
    )

    # Should be a DataFrame
    assert isinstance(result_df, pd.DataFrame)

    # Should only contain APID 1048 packets
    assert all(result_df["PKT_APID"] == 1048)

    # Should have removed duplicates (2 unique packets from 3 original)
    assert len(result_df) == 2

    # Verify duplicate removal worked correctly - should have foo and bar
    assert set(result_df["PARAM_1"]) == {"foo", "bar"}


@mock.patch("libera_utils.packets.create_dataset")
def test_parse_packets_to_dataframe_no_apid_specified(mock_create_dataset):
    """Test parse_packets_to_dataframe with no APID specified"""
    # Mock single APID dataset
    mock_df = pd.DataFrame(
        {
            "VERSION": [0, 0],
            "TYPE": [0, 0],
            "PKT_APID": [1048, 1048],
            "SRC_SEQ_CTR": [2428, 2429],
            "PARAM_1": ["foo", "bar"],
            "PARAM_2": [1.1, 2.2],
        }
    )
    mock_df.index.name = "packet"
    mock_dataset = mock_df.to_xarray()

    mock_create_dataset.return_value = {1048: mock_dataset}

    result_df = libera_packets.parse_packets_to_dataframe(
        packet_definition="fake_path.xml", packet_data_filepaths=["fake_file.bin"]
    )

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == 2


@mock.patch("libera_utils.packets.create_dataset")
def test_parse_packets_to_dataframe_multiple_apids_no_filter(mock_create_dataset):
    """Test parse_packets_to_dataframe raises error when multiple APIDs present and none specified"""
    mock_df_1048 = pd.DataFrame({"PKT_APID": [1048]})
    mock_df_1048.index.name = "packet"

    mock_df_1059 = pd.DataFrame({"PKT_APID": [1059]})
    mock_df_1059.index.name = "packet"
    mock_create_dataset.return_value = {1048: mock_df_1048.to_xarray(), 1059: mock_df_1059.to_xarray()}

    with pytest.raises(ValueError, match="Multiple APIDs present.*You must specify which APID you want"):
        libera_packets.parse_packets_to_dataframe(
            packet_definition="fake_path.xml", packet_data_filepaths=["fake_file.bin"]
        )


@mock.patch("libera_utils.packets.create_dataset")
def test_parse_packets_to_dataframe_apid_not_found(mock_create_dataset):
    """Test parse_packets_to_dataframe when requested APID is not found"""
    # Create a proper xarray Dataset with the expected dimensions
    mock_df = pd.DataFrame({"PKT_APID": [1048, 1048], "PARAM_1": ["foo", "bar"]})
    # Set index to 'packet' to match what Space Packet Parser would create
    mock_df.index.name = "packet"
    mock_dataset = mock_df.to_xarray()

    mock_create_dataset.return_value = {1048: mock_dataset}

    # Request APID that doesn't exist - should raise ValueError
    with pytest.raises(ValueError, match="Requested APID 9999 not found in parsed packets"):
        libera_packets.parse_packets_to_dataframe(
            packet_definition="fake_path.xml", packet_data_filepaths=["fake_file.bin"], apid=9999
        )
