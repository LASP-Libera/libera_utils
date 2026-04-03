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
    mock_df.index.name = "PACKET"
    mock_dataset = mock_df.to_xarray()

    mock_create_dataset.return_value = {1048: mock_dataset}

    result = libera_packets.parse_packets_to_dataset(packet_files=["fake.bin"], packet_definition="fake.xml", apid=1048)

    assert isinstance(result, type(mock_dataset))
    assert len(result["PKT_APID"]) == 2


@mock.patch("libera_utils.l1a.packets.create_dataset")
def test_parse_packets_to_dataset_apid_not_found(mock_create_dataset):
    """Test parse_packets_to_dataset raises ValueError when APID not in dataset"""
    mock_df = pd.DataFrame({"PKT_APID": [1048]})
    mock_df.index.name = "PACKET"
    mock_create_dataset.return_value = {1048: mock_df.to_xarray()}

    with pytest.raises(ValueError, match="Expected only APID 9999 in parsed dataset"):
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
            "data": (["time"], [1, 2, 2, 4]),
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


def test_drop_duplicates_with_non_identical_duplicates():
    """Test _drop_duplicates removes duplicate coordinates and warns"""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "time": (["time"], [100, 200, 200, 300]),  # 200 appears twice
        }
    )

    with pytest.raises(ValueError, match="Dropping this duplicate would result in data loss"):
        result_ds, n_duplicates = libera_packets._drop_duplicates(ds, "time")


def test_drop_duplicates_non_dimension_coordinate():
    """Test _drop_duplicates with a non-dimension coordinate bound to a dimension"""
    ds = xr.Dataset({"data": (["PACKET"], [10, 20, 30, 40, 50, 60])})
    # Add non-dimension coordinate bound to "PACKET" dimension with duplicates
    ds = ds.assign_coords({"packet_time": (["PACKET"], [1, 2, 5, 8, 8, 8])})

    result_ds, n_duplicates = libera_packets._drop_duplicates(ds, "packet_time")

    # Should drop 2 duplicates (keep first occurrence of value 8)
    assert n_duplicates == 2
    # Should have 4 unique values: [1, 2, 5, 8]
    assert len(result_ds["packet_time"]) == 4
    assert list(result_ds["packet_time"].values) == [1, 2, 5, 8]
    # Data should also be reduced accordingly (keep first 4 items)
    assert len(result_ds["data"]) == 4
    assert list(result_ds["data"].values) == [10, 20, 30, 40]


def test_validate_duplicate_values_non_dimension_coordinate_skips_validation():
    """Non-dimension coordinates should skip value-identity checks entirely,
    even when duplicate rows have differing data values."""
    ds = xr.Dataset({"data": (["PACKET"], [10, 20, 30])})
    ds = ds.assign_coords({"packet_time": (["PACKET"], [1, 8, 8])})

    coord_values = ds["packet_time"].values
    unique_values, unique_indices = np.unique(coord_values, return_index=True)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    # Should not raise even though duplicate rows [20, 30] differ
    libera_packets._validate_duplicate_values(ds, "packet_time", "PACKET", coord_values, duplicates)


def test_validate_duplicate_values_dimension_coordinate_identical_rows():
    """Dimension coordinate duplicates with identical data values should pass validation."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 2, 3]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )

    coord_values = ds["time"].values
    unique_values, unique_indices = np.unique(coord_values, return_index=True)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    # Should not raise — both rows at time=200 have data value 2
    libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates)


def test_validate_duplicate_values_dimension_coordinate_differing_rows():
    """Dimension coordinate duplicates with differing data values should raise ValueError."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 99, 3]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )

    coord_values = ds["time"].values
    unique_values, unique_indices = np.unique(coord_values, return_index=True)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    with pytest.raises(ValueError, match="200.*time.*data"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates)


def test_validate_duplicate_values_multiple_duplicates_all_identical():
    """Multiple duplicate coordinate values that are all internally identical should pass."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 2, 3, 3, 4]),
            "time": (["time"], [100, 200, 200, 300, 300, 400]),
        }
    )

    coord_values = ds["time"].values
    unique_values, unique_indices = np.unique(coord_values, return_index=True)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    # Both time=200 (data=2,2) and time=300 (data=3,3) are identical — should pass
    libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates)


def test_validate_duplicate_values_multiple_duplicates_one_differs():
    """If any one duplicate group has differing values, a ValueError should be raised."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 2, 3, 99, 4]),
            "time": (["time"], [100, 200, 200, 300, 300, 400]),
        }
    )

    coord_values = ds["time"].values
    unique_values, unique_indices = np.unique(coord_values, return_index=True)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    # time=200 is fine (2,2), but time=300 differs (3 vs 99)
    with pytest.raises(ValueError, match="300"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates)


def test_validate_duplicate_values_multiple_variables_one_differs():
    """Validation should catch differing values in any variable, not just the first."""
    ds = xr.Dataset(
        {
            "data_a": (["time"], [1, 2, 2]),
            "data_b": (["time"], [10, 20, 99]),  # differs at time=200
            "time": (["time"], [100, 200, 200]),
        }
    )

    coord_values = ds["time"].values
    unique_values, unique_indices = np.unique(coord_values, return_index=True)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    with pytest.raises(ValueError, match="data_b"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates)


def test_validate_duplicate_values_empty_duplicates():
    """Passing an empty duplicates array should be a no-op with no errors raised."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3]),
            "time": (["time"], [100, 200, 300]),
        }
    )

    coord_values = ds["time"].values
    duplicates = np.array([])

    libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates)


@mock.patch("libera_utils.l1a.packets.multipart_to_dt64")
def test_expand_sample_times_single_sample(mock_multipart_to_dt64):
    """Test _expand_sample_times with single sample per packet"""
    # Create mock dataset
    ds = xr.Dataset({"SEC_FIELD": (["PACKET"], [10, 20]), "USEC_FIELD": (["PACKET"], [100, 200])})

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
            "SEC_FIELD0": (["PACKET"], [10, 20]),
            "SEC_FIELD1": (["PACKET"], [11, 21]),
            "SEC_FIELD2": (["PACKET"], [12, 22]),
            "USEC_FIELD0": (["PACKET"], [100, 200]),
            "USEC_FIELD1": (["PACKET"], [110, 210]),
            "USEC_FIELD2": (["PACKET"], [120, 220]),
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
            "DATA_FIELD": (["PACKET"], [1.0, 2.0]),
            "SEC_FIELD": (["PACKET"], [10, 20]),
            "USEC_FIELD": (["PACKET"], [100, 200]),
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
            "DATA_FIELD0": (["PACKET"], [1.0, 2.0]),
            "DATA_FIELD1": (["PACKET"], [1.1, 2.1]),
            "SEC_FIELD0": (["PACKET"], [10, 20]),
            "SEC_FIELD1": (["PACKET"], [11, 21]),
            "USEC_FIELD0": (["PACKET"], [100, 200]),
            "USEC_FIELD1": (["PACKET"], [110, 210]),
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
            "DATA_FIELD0": (["PACKET"], [1.0, 2.0]),
            "DATA_FIELD1": (["PACKET"], [1.1, 2.1]),
            "EPOCH_SEC": (["PACKET"], [10, 20]),
            "EPOCH_USEC": (["PACKET"], [100, 200]),
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
            "DATA0": (["PACKET"], [1.0, 4.0]),
            "DATA1": (["PACKET"], [2.0, 5.0]),
            "DATA2": (["PACKET"], [3.0, 6.0]),
            "SEC0": (["PACKET"], [10, 20]),
            "SEC1": (["PACKET"], [11, 21]),
            "SEC2": (["PACKET"], [12, 22]),
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
            "DATA0": (["PACKET"], [1.0, 4.0]),
            "DATA1": (["PACKET"], [2.0, 5.0]),
            "DATA2": (["PACKET"], [3.0, 6.0]),
            "EPOCH_SEC": (["PACKET"], [10, 20]),
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
            "FIELD_0": (["PACKET"], np.array([65, 97], dtype=np.uint8)),  # 'A', 'a'
            "FIELD_1": (["PACKET"], np.array([66, 98], dtype=np.uint8)),  # 'B', 'b'
            "FIELD_2": (["PACKET"], np.array([67, 99], dtype=np.uint8)),  # 'C', 'c'
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
            "FIELD_0": (["PACKET"], np.array([b"ABC", b"abc"])),
            "FIELD_1": (["PACKET"], np.array([b"DEF", b"def"])),
            "FIELD_2": (["PACKET"], np.array([b"GHI", b"ghi"])),
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
            "FIELD_0": (["PACKET"], np.array([1, 2], dtype=np.uint16)),
            "FIELD_1": (["PACKET"], np.array([3, 4], dtype=np.uint16)),
            "FIELD_2": (["PACKET"], np.array([5, 6], dtype=np.uint16)),
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
            "FIELD_0": (["PACKET"], np.array([1, 2], dtype=np.dtype("S1"))),
            "FIELD_1": (["PACKET"], np.array([3, 4], dtype=np.dtype("S1"))),
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
            "DATA_0": (["PACKET"], [1, 2]),
            "DATA_1": (["PACKET"], [3, 4]),
            "DATA_2": (["PACKET"], [5, 6]),
            "OTHER_FIELD": (["PACKET"], [7, 8]),
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
            "PKT_DAY": (["PACKET"], [1000, 1001]),
            "PKT_MS": (["PACKET"], [0, 1000]),
            "SAMPLE_DAY": (["PACKET"], [1000, 1001]),
            "SAMPLE_MS": (["PACKET"], [500, 1500]),
            "SAMPLE_DATA": (["PACKET"], [1.5, 2.5]),
            "OTHER_FIELD": (["PACKET"], [100, 200]),
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
    assert "PACKET" in result.dims
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


def test_monotonic_sequence_returns_true():
    """Monotonically increasing sequence counters should return True."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102, 103]),
            "time": (["time"], [10, 20, 20, 30]),
        }
    )
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is True


def test_non_monotonic_sequence_returns_false():
    """Non-monotonic sequence counters (gap) should return False."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 105, 103]),
            "time": (["time"], [10, 20, 20, 30]),
        }
    )
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is False


def test_rollover_at_zero_current_returns_true():
    """Sequence counter rolling over to 0 at the current index should be treated as monotonic."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [16382, 16383, 0, 1]),
            "time": (["time"], [10, 20, 20, 30]),
        }
    )
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is True


def test_rollover_at_zero_previous_returns_true():
    """Sequence counter at 0 for the previous neighbor should be treated as monotonic."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [0, 1, 2, 3]),
            "time": (["time"], [10, 20, 20, 30]),
        }
    )
    # Duplicate at index 1 and 2; previous of index 1 is index 0 which is 0
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is True


def test_rollover_at_zero_next_returns_true():
    """Sequence counter at 0 for the next neighbor should be treated as monotonic."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [16382, 16383, 16383, 0]),
            "time": (["time"], [10, 20, 20, 30]),
        }
    )
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is True


def test_missing_src_seq_ctr_returns_false():
    """If SRC_SEQ_CTR is not in the dataset, should return False."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "time": (["time"], [10, 20, 20, 30]),
        }
    )
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is False


def test_first_index_duplicate_no_previous():
    """Duplicate at the very first index should only check the next neighbor."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102]),
            "time": (["time"], [10, 10, 20]),
        }
    )
    dup_indices = np.array([0, 1])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is True


def test_last_index_duplicate_no_next():
    """Duplicate at the very last index should only check the previous neighbor."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102]),
            "time": (["time"], [10, 20, 20]),
        }
    )
    dup_indices = np.array([1, 2])
    assert libera_packets._is_sequence_counter_monotonic(ds, "time", dup_indices) is True


def test_ground_data_false_raises_valueerror():
    """With ground_data=False (default), non-identical duplicates should raise ValueError."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102, 103]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )
    coord_values = ds["time"].values
    _, counts = np.unique(coord_values, return_counts=True)
    unique_values = np.unique(coord_values)
    duplicates = unique_values[counts > 1]

    with pytest.raises(ValueError, match="Dropping this duplicate would result in data loss"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates, ground_data=False)


def test_ground_data_true_monotonic_seq_warns():
    """With ground_data=True and monotonic sequence counters, should warn instead of raise."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102, 103]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )
    coord_values = ds["time"].values
    unique_values = np.unique(coord_values)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    with pytest.warns(UserWarning, match="sequence counter is monotonic"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates, ground_data=True)


def test_ground_data_true_non_monotonic_seq_raises():
    """With ground_data=True but non-monotonic sequence counters, should still raise ValueError."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 999, 103]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )
    coord_values = ds["time"].values
    unique_values = np.unique(coord_values)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    with pytest.raises(ValueError, match="Dropping this duplicate would result in data loss"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates, ground_data=True)


def test_ground_data_true_no_seq_ctr_raises():
    """With ground_data=True but no SRC_SEQ_CTR field, should still raise ValueError."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )
    coord_values = ds["time"].values
    unique_values = np.unique(coord_values)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    with pytest.raises(ValueError, match="Dropping this duplicate would result in data loss"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates, ground_data=True)


def test_ground_data_true_with_rollover_warns():
    """With ground_data=True and sequence counter rolling over at 0, should warn."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [16382, 16383, 0, 1]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )
    coord_values = ds["time"].values
    unique_values = np.unique(coord_values)
    _, counts = np.unique(coord_values, return_counts=True)
    duplicates = unique_values[counts > 1]

    with pytest.warns(UserWarning, match="sequence counter is monotonic"):
        libera_packets._validate_duplicate_values(ds, "time", "time", coord_values, duplicates, ground_data=True)


def test_drop_duplicates_ground_data_false_raises_on_non_identical():
    """_drop_duplicates with ground_data=False should raise on non-identical duplicates."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102, 103]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )

    with pytest.raises(ValueError, match="Dropping this duplicate would result in data loss"):
        libera_packets._drop_duplicates(ds, "time", ground_data=False)


def test_drop_duplicates_ground_data_true_monotonic_warns_and_deduplicates():
    """_drop_duplicates with ground_data=True and monotonic seq should warn and deduplicate."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102, 103]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )

    with pytest.warns(UserWarning, match="sequence counter is monotonic"):
        result_ds, n_duplicates = libera_packets._drop_duplicates(ds, "time", ground_data=True)

    assert n_duplicates == 1
    assert len(result_ds["time"]) == 3
    assert list(result_ds["time"].values) == [100, 200, 300]


def test_drop_duplicates_default_ground_data_is_false():
    """The default value of ground_data should be False, raising on non-identical duplicates."""
    ds = xr.Dataset(
        {
            "data": (["time"], [1, 2, 3, 4]),
            "SRC_SEQ_CTR": (["time"], [100, 101, 102, 103]),
            "time": (["time"], [100, 200, 200, 300]),
        }
    )

    with pytest.raises(ValueError, match="Dropping this duplicate would result in data loss"):
        libera_packets._drop_duplicates(ds, "time")
