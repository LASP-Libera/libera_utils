"""Tests for libera_utils.l1a.packets module for L1A processing"""

from datetime import timedelta
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.l1a import packets as libera_packets
from libera_utils.l1a.l1a_packet_configs import (
    AggregationGroup,
    PacketConfiguration,
    SampleGroup,
    SampleTimeSource,
    TimeFieldMapping,
)


@mock.patch("libera_utils.l1a.packets.create_dataset")
def test_parse_packets_to_dataset_with_single_apid(mock_create_dataset):
    """Test parse_packets_to_dataset returns correct dataset for specified APID"""
    # Create mock dataset for APID 1048
    mock_df = pd.DataFrame(
        {
            "PKT_APID": [1048, 1048],
            "SRC_SEQ_CTR": [100, 101],
            "DATA_FIELD": [1.1, 2.2],
        }
    )
    mock_df.index.name = "packet"
    mock_dataset = mock_df.to_xarray()

    mock_create_dataset.return_value = {1048: mock_dataset}

    result = libera_packets.parse_packets_to_dataset(packet_files=["fake.bin"], packet_definition="fake.xml", apid=1048)

    assert isinstance(result, type(mock_dataset))
    assert len(result["PKT_APID"]) == 2


@mock.patch("libera_utils.l1a.packets.create_dataset")
def test_parse_packets_to_dataset_apid_not_found(mock_create_dataset):
    """Test parse_packets_to_dataset raises ValueError when APID not in dataset"""
    mock_df = pd.DataFrame({"PKT_APID": [1048]})
    mock_df.index.name = "packet"
    mock_create_dataset.return_value = {1048: mock_df.to_xarray()}

    with pytest.raises(KeyError, match="Requested APID 9999 not found"):
        libera_packets.parse_packets_to_dataset(packet_files=["fake.bin"], packet_definition="fake.xml", apid=9999)


def test_drop_duplicates_no_duplicates():
    """Test _drop_duplicates with no duplicate coordinates"""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3]),
            "time": (["time"], [100, 200, 300]),
        }
    )

    result_ds, n_duplicates = libera_packets._drop_duplicates(ds, "time")

    assert n_duplicates == 0
    assert len(result_ds["time"]) == 3


def test_drop_duplicates_with_duplicates():
    """Test _drop_duplicates removes duplicate coordinates and warns"""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "time": (["time"], [100, 200, 200, 300]),  # 200 appears twice
        }
    )

    with pytest.warns(UserWarning, match="duplicate packet timestamps"):
        result_ds, n_duplicates = libera_packets._drop_duplicates(ds, "time")

    assert n_duplicates > 0
    # After deduplication, should have 3 unique time values
    assert len(result_ds["time"]) == 3
    assert 100 in result_ds["time"].values
    assert 200 in result_ds["time"].values
    assert 300 in result_ds["time"].values


def test_drop_duplicates_non_dimension_coordinate():
    """Test _drop_duplicates with a non-dimension coordinate bound to a dimension"""
    ds = xr.Dataset({"data": (["packet"], [10, 20, 30, 40, 50, 60])})
    # Add non-dimension coordinate bound to "packet" dimension with duplicates
    ds = ds.assign_coords({"packet_time": (["packet"], [1, 2, 5, 8, 8, 8])})

    result_ds, n_duplicates = libera_packets._drop_duplicates(ds, "packet_time")

    # Should drop 2 duplicates (keep first occurrence of value 8)
    assert n_duplicates == 2
    # Should have 4 unique values: [1, 2, 5, 8]
    assert len(result_ds["packet_time"]) == 4
    assert list(result_ds["packet_time"].values) == [1, 2, 5, 8]
    # Data should also be reduced accordingly (keep first 4 items)
    assert len(result_ds["data"]) == 4
    assert list(result_ds["data"].values) == [10, 20, 30, 40]


@mock.patch("libera_utils.l1a.packets.multipart_to_dt64")
def test_expand_sample_times_single_sample(mock_multipart_to_dt64):
    """Test _expand_sample_times with single sample per packet"""
    # Create mock dataset
    ds = xr.Dataset({"SEC_FIELD": (["packet"], [10, 20]), "USEC_FIELD": (["packet"], [100, 200])})

    # Mock the multipart_to_dt64 function
    mock_times = pd.Series([np.datetime64("2025-01-01T00:00:10.000100"), np.datetime64("2025-01-01T00:00:20.000200")])
    mock_multipart_to_dt64.return_value = mock_times

    time_fields = TimeFieldMapping(s_field="SEC_FIELD", us_field="USEC_FIELD")

    result = libera_packets._expand_sample_times(ds, time_fields, n_samples=1)

    assert len(result) == 2
    assert result.dtype == np.dtype("datetime64[us]")


@mock.patch("libera_utils.l1a.packets.multipart_to_dt64")
def test_expand_sample_times_multi_sample(mock_multipart_to_dt64):
    """Test _expand_sample_times with multiple samples per packet"""
    # Create mock dataset with 2 packets, 3 samples each
    ds = xr.Dataset(
        {
            "SEC_FIELD0": (["packet"], [10, 20]),
            "SEC_FIELD1": (["packet"], [11, 21]),
            "SEC_FIELD2": (["packet"], [12, 22]),
            "USEC_FIELD0": (["packet"], [100, 200]),
            "USEC_FIELD1": (["packet"], [110, 210]),
            "USEC_FIELD2": (["packet"], [120, 220]),
        }
    )

    # Mock returns for each sample index
    def multipart_side_effect(dataset, **kwargs):
        # Return different times based on which field is being accessed
        if "SEC_FIELD0" in kwargs.values():
            return pd.Series([np.datetime64("2025-01-01T00:00:10.000100"), np.datetime64("2025-01-01T00:00:20.000200")])
        elif "SEC_FIELD1" in kwargs.values():
            return pd.Series([np.datetime64("2025-01-01T00:00:11.000110"), np.datetime64("2025-01-01T00:00:21.000210")])
        else:  # SEC_FIELD2
            return pd.Series([np.datetime64("2025-01-01T00:00:12.000120"), np.datetime64("2025-01-01T00:00:22.000220")])

    mock_multipart_to_dt64.side_effect = multipart_side_effect

    time_fields = TimeFieldMapping(s_field="SEC_FIELD%i", us_field="USEC_FIELD%i")

    result = libera_packets._expand_sample_times(ds, time_fields, n_samples=3)

    # Should have 2 packets * 3 samples = 6 total times
    assert len(result) == 6
    assert result.dtype == np.dtype("datetime64[us]")


def test_get_expanded_field_names_single_sample():
    """Test _get_expanded_field_names for single sample per packet"""

    ds = xr.Dataset(
        {
            "DATA_FIELD": (["packet"], [1.0, 2.0]),
            "SEC_FIELD": (["packet"], [10, 20]),
            "USEC_FIELD": (["packet"], [100, 200]),
        }
    )

    sample_group = SampleGroup(
        name="TEST",
        sample_count=1,
        data_field_patterns=["DATA_FIELD"],
        time_field_patterns=TimeFieldMapping(s_field="SEC_FIELD", us_field="USEC_FIELD"),
        time_source=SampleTimeSource.ICIE,
    )

    result = libera_packets._get_expanded_field_names(ds, sample_group)

    assert result == {"DATA_FIELD", "SEC_FIELD", "USEC_FIELD"}


def test_get_expanded_field_names_multi_sample():
    """Test _get_expanded_field_names for multiple samples per packet"""
    ds = xr.Dataset(
        {
            "DATA_FIELD0": (["packet"], [1.0, 2.0]),
            "DATA_FIELD1": (["packet"], [1.1, 2.1]),
            "SEC_FIELD0": (["packet"], [10, 20]),
            "SEC_FIELD1": (["packet"], [11, 21]),
            "USEC_FIELD0": (["packet"], [100, 200]),
            "USEC_FIELD1": (["packet"], [110, 210]),
        }
    )

    sample_group = SampleGroup(
        name="TEST",
        sample_count=2,
        data_field_patterns=["DATA_FIELD%i"],
        time_field_patterns=TimeFieldMapping(s_field="SEC_FIELD%i", us_field="USEC_FIELD%i"),
        time_source=SampleTimeSource.ICIE,
    )

    result = libera_packets._get_expanded_field_names(ds, sample_group)

    expected = {"DATA_FIELD0", "DATA_FIELD1", "SEC_FIELD0", "SEC_FIELD1", "USEC_FIELD0", "USEC_FIELD1"}
    assert result == expected


def test_get_expanded_field_names_with_epoch():
    """Test _get_expanded_field_names with epoch time fields"""
    ds = xr.Dataset(
        {
            "DATA_FIELD0": (["packet"], [1.0, 2.0]),
            "DATA_FIELD1": (["packet"], [1.1, 2.1]),
            "EPOCH_SEC": (["packet"], [10, 20]),
            "EPOCH_USEC": (["packet"], [100, 200]),
        }
    )

    sample_group = SampleGroup(
        name="TEST",
        sample_count=2,
        data_field_patterns=["DATA_FIELD%i"],
        epoch_time_fields=TimeFieldMapping(s_field="EPOCH_SEC", us_field="EPOCH_USEC"),
        sample_period=timedelta(milliseconds=5),
        time_source=SampleTimeSource.FPE,
    )

    result = libera_packets._get_expanded_field_names(ds, sample_group)

    expected = {"DATA_FIELD0", "DATA_FIELD1", "EPOCH_SEC", "EPOCH_USEC"}
    assert result == expected


@mock.patch("libera_utils.l1a.packets.multipart_to_dt64")
def test_expand_sample_group_multi_sample(mock_multipart_to_dt64):
    """Test _expand_sample_group with multiple samples per packet"""
    # 2 packets, 3 samples each
    ds = xr.Dataset(
        {
            "DATA0": (["packet"], [1.0, 4.0]),
            "DATA1": (["packet"], [2.0, 5.0]),
            "DATA2": (["packet"], [3.0, 6.0]),
            "SEC0": (["packet"], [10, 20]),
            "SEC1": (["packet"], [11, 21]),
            "SEC2": (["packet"], [12, 22]),
        }
    )

    # Mock multipart_to_dt64 to return different times for each sample
    def multipart_side_effect(dataset, **kwargs):
        if "SEC0" in kwargs.values():
            return pd.Series([np.datetime64("2025-01-01T00:00:10"), np.datetime64("2025-01-01T00:00:20")])
        elif "SEC1" in kwargs.values():
            return pd.Series([np.datetime64("2025-01-01T00:00:11"), np.datetime64("2025-01-01T00:00:21")])
        else:  # SEC2
            return pd.Series([np.datetime64("2025-01-01T00:00:12"), np.datetime64("2025-01-01T00:00:22")])

    mock_multipart_to_dt64.side_effect = multipart_side_effect

    sample_group = SampleGroup(
        name="TEST",
        sample_count=3,
        data_field_patterns=["DATA%i"],
        time_field_patterns=TimeFieldMapping(s_field="SEC%i"),
        time_source=SampleTimeSource.ICIE,
    )

    field_arrays, sample_times = libera_packets._expand_sample_group(ds, sample_group)

    # Check field arrays
    assert "DATA" in field_arrays
    # Should have 2 packets * 3 samples = 6 values, interleaved by packet
    # [pkt0_sample0, pkt0_sample1, pkt0_sample2, pkt1_sample0, pkt1_sample1, pkt1_sample2]
    assert len(field_arrays["DATA"]) == 6
    np.testing.assert_array_equal(field_arrays["DATA"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Check sample times
    assert len(sample_times) == 6
    assert sample_times.dtype == np.dtype("datetime64[us]")


@mock.patch("libera_utils.l1a.packets.multipart_to_dt64")
def test_expand_sample_group_with_epoch_and_period(mock_multipart_to_dt64):
    """Test _expand_sample_group using epoch time + period"""
    # 2 packets, 3 samples each
    ds = xr.Dataset(
        {
            "DATA0": (["packet"], [1.0, 4.0]),
            "DATA1": (["packet"], [2.0, 5.0]),
            "DATA2": (["packet"], [3.0, 6.0]),
            "EPOCH_SEC": (["packet"], [10, 20]),
        }
    )

    # Mock epoch time conversion
    mock_multipart_to_dt64.return_value = pd.Series(
        [np.datetime64("2025-01-01T00:00:10.000000"), np.datetime64("2025-01-01T00:00:20.000000")]
    )

    sample_group = SampleGroup(
        name="TEST",
        sample_count=3,
        data_field_patterns=["DATA%i"],
        epoch_time_fields=TimeFieldMapping(s_field="EPOCH_SEC"),
        sample_period=timedelta(milliseconds=5),
        time_source=SampleTimeSource.FPE,
    )

    field_arrays, sample_times = libera_packets._expand_sample_group(ds, sample_group)

    # Check field arrays
    assert "DATA" in field_arrays
    assert len(field_arrays["DATA"]) == 6
    np.testing.assert_array_equal(field_arrays["DATA"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Check sample times - should be epoch + i*period for each packet
    assert len(sample_times) == 6
    # First packet samples: 10s, 10.005s, 10.010s
    # Second packet samples: 20s, 20.005s, 20.010s
    expected_times = np.array(
        [
            np.datetime64("2025-01-01T00:00:10.000000"),
            np.datetime64("2025-01-01T00:00:10.005000"),
            np.datetime64("2025-01-01T00:00:10.010000"),
            np.datetime64("2025-01-01T00:00:20.000000"),
            np.datetime64("2025-01-01T00:00:20.005000"),
            np.datetime64("2025-01-01T00:00:20.010000"),
        ]
    )
    np.testing.assert_array_equal(sample_times, expected_times)


def test_aggregate_fields_uint8s():
    """Test _aggregate_fields aggregates sequential uint8 fields into binary blobs"""
    # Create dataset with 2 packets, each with 3 sequential byte fields
    ds = xr.Dataset(
        {
            "FIELD_0": (["packet"], np.array([65, 97], dtype=np.uint8)),  # 'A', 'a'
            "FIELD_1": (["packet"], np.array([66, 98], dtype=np.uint8)),  # 'B', 'b'
            "FIELD_2": (["packet"], np.array([67, 99], dtype=np.uint8)),  # 'C', 'c'
        }
    )

    agg_group = AggregationGroup(name="FIELD", field_pattern="FIELD_%i", field_count=3, dtype=np.dtype("|S3"))

    result = libera_packets._aggregate_fields(ds, agg_group)

    assert len(result) == 2
    assert result.dtype == np.dtype("|S3")
    assert result[0] == b"ABC"
    assert result[1] == b"abc"


def test_aggregate_fields_bytes():
    """Test _aggregate_fields aggregates sequential byte string fields into binary blobs"""
    # Create dataset with 2 packets, each with 3 sequential bytestring fields
    ds = xr.Dataset(
        {
            "FIELD_0": (["packet"], np.array([b"ABC", b"abc"])),
            "FIELD_1": (["packet"], np.array([b"DEF", b"def"])),
            "FIELD_2": (["packet"], np.array([b"GHI", b"ghi"])),
        }
    )

    agg_group = AggregationGroup(name="FIELD", field_pattern="FIELD_%i", field_count=3, dtype=np.dtype("|S9"))

    result = libera_packets._aggregate_fields(ds, agg_group)

    assert len(result) == 2
    assert result.dtype == np.dtype("|S9")
    assert result[0] == b"ABCDEFGHI"
    assert result[1] == b"abcdefghi"


def test_aggregate_fields_size_mismatch():
    """Test _aggregate_fields raises ValueError on size mismatch"""
    ds = xr.Dataset(
        {
            "FIELD_0": (["packet"], np.array([1, 2], dtype=np.uint16)),
            "FIELD_1": (["packet"], np.array([3, 4], dtype=np.uint16)),
            "FIELD_2": (["packet"], np.array([5, 6], dtype=np.uint16)),
        }
    )

    # Agg group is expecting one byte per field (uint8 or S1) but dtype is 2 bytes per field (uint16)
    agg_group = AggregationGroup(name="FIELD", field_pattern="FIELD_%i", field_count=3, dtype=np.dtype("|S3"))

    with pytest.raises(
        ValueError,
        match="Aggregation group FIELD size mismatch: expected total size 3 bytes, got 6 bytes.",
    ):
        libera_packets._aggregate_fields(ds, agg_group)


def test_aggregate_fields_missing_field():
    """Test _aggregate_fields raises KeyError when field is missing"""
    ds = xr.Dataset(
        {
            "FIELD_0": (["packet"], np.array([1, 2], dtype=np.dtype("S1"))),
            "FIELD_1": (["packet"], np.array([3, 4], dtype=np.dtype("S1"))),
            # FIELD_2 is missing
        }
    )

    agg_group = AggregationGroup(name="FIELD", field_pattern="FIELD_%i", field_count=3)

    with pytest.raises(KeyError, match="Required field FIELD_2 not found"):
        libera_packets._aggregate_fields(ds, agg_group)


def test_get_aggregated_field_names():
    """Test _get_aggregated_field_names returns all aggregated field names"""
    ds = xr.Dataset(
        {
            "DATA_0": (["packet"], [1, 2]),
            "DATA_1": (["packet"], [3, 4]),
            "DATA_2": (["packet"], [5, 6]),
            "OTHER_FIELD": (["packet"], [7, 8]),
        }
    )

    agg_group = AggregationGroup(name="DATA", field_pattern="DATA_%i", field_count=3)

    result = libera_packets._get_aggregated_field_names(ds, agg_group)

    assert result == {"DATA_0", "DATA_1", "DATA_2"}
    assert "OTHER_FIELD" not in result


@mock.patch("libera_utils.l1a.packets.parse_packets_to_dataset")
@mock.patch("libera_utils.l1a.packets.multipart_to_dt64")
@mock.patch("libera_utils.l1a.packets.get_packet_config")
@mock.patch("libera_utils.config.config.get")
def test_parse_packets_to_l1a_dataset_basic(
    mock_config_get, mock_get_packet_config, mock_multipart, mock_parse_packets
):
    """Test parse_packets_to_l1a_dataset with basic single-sample configuration"""

    # Create a minimal packet configuration using Pydantic
    config = PacketConfiguration(
        packet_apid=LiberaApid.icie_nom_hk,
        packet_time_fields=TimeFieldMapping(day_field="PKT_DAY", ms_field="PKT_MS"),
        sample_groups=[
            SampleGroup(
                name="TEST_SAMPLE",
                sample_count=1,
                data_field_patterns=["SAMPLE_DATA"],
                time_field_patterns=TimeFieldMapping(day_field="SAMPLE_DAY", ms_field="SAMPLE_MS"),
                time_source=SampleTimeSource.ICIE,
            )
        ],
    )
    mock_config_get.return_value = "fake_definition.xml"
    mock_get_packet_config.return_value = config

    # Create mock packet dataset
    packet_ds = xr.Dataset(
        {
            "PKT_DAY": (["packet"], [1000, 1001]),
            "PKT_MS": (["packet"], [0, 1000]),
            "SAMPLE_DAY": (["packet"], [1000, 1001]),
            "SAMPLE_MS": (["packet"], [500, 1500]),
            "SAMPLE_DATA": (["packet"], [1.5, 2.5]),
            "OTHER_FIELD": (["packet"], [100, 200]),
        }
    )

    mock_parse_packets.return_value = packet_ds

    # Mock multipart_to_dt64 calls
    def multipart_side_effect(ds, **kwargs):
        if "PKT_DAY" in kwargs.values():
            # Packet times
            return pd.Series([np.datetime64("2025-01-01T00:00:00"), np.datetime64("2025-01-01T00:00:01")])
        else:
            # Sample times
            return pd.Series([np.datetime64("2025-01-01T00:00:00.500"), np.datetime64("2025-01-01T00:00:01.500")])

    mock_multipart.side_effect = multipart_side_effect

    # Run the function
    result = libera_packets.parse_packets_to_l1a_dataset(packet_files=["fake.bin"], apid=LiberaApid.icie_nom_hk.value)

    # Verify result structure
    assert isinstance(result, xr.Dataset)
    assert "packet" in result.dims
    assert "PACKET_ICIE_TIME" in result.coords
    assert "TEST_SAMPLE_ICIE_TIME" in result.coords
    assert "TEST_SAMPLE_ICIE_TIME" in result.dims

    # Verify sample data was expanded
    assert "SAMPLE_DATA" in result.data_vars
    assert result["SAMPLE_DATA"].dims == ("TEST_SAMPLE_ICIE_TIME",)

    # Verify packet_index variable was created
    assert "TEST_SAMPLE_packet_index" in result.data_vars

    # Verify expanded fields were removed from packet dataset
    assert "SAMPLE_DAY" not in result.data_vars
    assert "SAMPLE_MS" not in result.data_vars

    # Verify non-expanded fields remain
    assert "OTHER_FIELD" in result.data_vars

    # Verify global attributes were added
    assert "algorithm_version" in result.attrs
    assert "date_created" in result.attrs
