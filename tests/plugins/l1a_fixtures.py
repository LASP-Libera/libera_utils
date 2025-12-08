"""Fixtures for L1A dataset testing."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a.l1a_packet_configs import (
    PacketConfiguration,
    SampleGroup,
    SampleTimeSource,
    TimeFieldMapping,
)


@pytest.fixture(scope="session")
def test_packet_configuration() -> PacketConfiguration:
    """Create a test PacketConfiguration.

    This configuration mimics the structure of real packet configurations
    but with simplified test data. It includes a single sample group with
    explicit timestamps, similar to AXIS_SAMPLE.

    Note that this packet configuration is NOT registered with the packet configuration registry!
    It is intended to mock out returns from calls to get_packet_configuration() in unit tests.

    Returns
    -------
    PacketConfiguration
        Test configuration with AXIS_SAMPLE-like structure
    """
    return PacketConfiguration(
        packet_apid=LiberaApid.icie_nom_hk,
        packet_time_fields=TimeFieldMapping(day_field="PACKET_DAY", ms_field="PACKET_MS"),
        sample_groups=[
            SampleGroup(
                name="AXIS_SAMPLE",
                sample_count=50,
                data_field_patterns=["ICIE__AXIS_AZ_FILT%i", "ICIE__AXIS_EL_FILT%i"],
                time_source=SampleTimeSource.ICIE,
                time_field_patterns=TimeFieldMapping(
                    day_field="ICIE__AXIS_SAMPLE_DAY%i", ms_field="ICIE__AXIS_SAMPLE_MS%i"
                ),
            )
        ],
    )


@pytest.fixture(scope="session")
def l1a_test_product(
    test_packet_configuration: PacketConfiguration, test_l1a_product_definition_file: Path
) -> xr.Dataset:
    """Create a test L1A Dataset following PacketConfiguration structure.

    This fixture creates an xarray Dataset that mimics the structure of
    L1A datasets created by parse_packets_to_l1a_dataset(). It includes:
    - Packet dimension and time coordinate
    - Sample dimension and time coordinate
    - Sample data variables
    - Packet index variable
    - Global attributes

    The Dataset is validated against the test_l1a_product_definition.yml to ensure
    it conforms to the expected structure.

    Returns
    -------
    xr.Dataset
        Test L1A dataset with 10 packets, 50 samples each (500 total samples)
    """
    # Load the product definition for validation
    product_definition = LiberaDataProductDefinition.from_yaml(test_l1a_product_definition_file)

    # Configuration
    num_packets = 10
    samples_per_packet = 50
    total_samples = num_packets * samples_per_packet

    # Create packet timestamps (1 per second)
    # Use naive datetimes and convert to numpy datetime64
    packet_times = np.array([datetime(2025, 8, 6, 12, 0, i) for i in range(num_packets)], dtype="datetime64[ns]")

    # Create sample timestamps (20 Hz = 50 samples/second)
    # Spread across the packet intervals
    sample_times = np.array(
        [
            datetime(
                2025,
                8,
                6,
                12,
                0,
                i // samples_per_packet,
                int((i % samples_per_packet) * 20000),  # microseconds
            )
            for i in range(total_samples)
        ],
        dtype="datetime64[ns]",
    )

    # Create sample data (synthetic sine waves)
    az_data = 45.0 + 10.0 * np.sin(np.linspace(0, 4 * np.pi, total_samples))
    el_data = 30.0 + 5.0 * np.cos(np.linspace(0, 4 * np.pi, total_samples))

    # Create packet index (maps each sample to its originating packet)
    packet_indices = np.repeat(np.arange(num_packets), samples_per_packet)

    # Get sample group for naming
    sample_group = test_packet_configuration.sample_groups[0]
    time_dim = sample_group.sample_time_dimension  # "AXIS_SAMPLE_ICIE_TIME"

    # Build dataset with proper attributes and encoding
    # Note: We need to set encoding on the DataArrays, not in the Dataset constructor
    packet_time_coord = xr.DataArray(
        packet_times,
        dims=["packet"],
        attrs={"long_name": "Packet timestamp from ICIE main processor"},
    )
    packet_time_coord.encoding = {
        "units": "nanoseconds since 1958-01-01",
        "calendar": "standard",
        "dtype": "int64",
        "zlib": True,
        "complevel": 4,
    }

    sample_time_coord = xr.DataArray(
        sample_times,
        dims=[time_dim],
        attrs={"long_name": "Azimuth and elevation encoder sample timestamp"},
    )
    sample_time_coord.encoding = {
        "units": "nanoseconds since 1958-01-01",
        "calendar": "standard",
        "dtype": "int64",
        "zlib": True,
        "complevel": 4,
    }

    az_data_var = xr.DataArray(
        az_data,
        dims=[time_dim],
        attrs={"long_name": "ICIE azimuth axis filtered encoder reading", "units": "degrees"},
    )
    az_data_var.encoding = {"zlib": True, "complevel": 4}

    el_data_var = xr.DataArray(
        el_data,
        dims=[time_dim],
        attrs={"long_name": "ICIE elevation axis filtered encoder reading", "units": "degrees"},
    )
    el_data_var.encoding = {"zlib": True, "complevel": 4}

    packet_index_var = xr.DataArray(
        packet_indices,
        dims=[time_dim],
        attrs={
            "long_name": "Packet index for axis sample data",
            "comment": "Maps each axis sample to its originating packet index",
        },
    )
    packet_index_var.encoding = {"zlib": True, "complevel": 4}

    ds = xr.Dataset(
        coords={
            "packet": np.arange(num_packets),
            "PACKET_ICIE_TIME": packet_time_coord,
            time_dim: sample_time_coord,
        },
        data_vars={
            "ICIE__AXIS_AZ_FILT": az_data_var,
            "ICIE__AXIS_EL_FILT": el_data_var,
            "AXIS_SAMPLE_packet_index": packet_index_var,
        },
        attrs={
            "Conventions": "CF-1.8",
            "Format": "NetCDF-4",
            "ProductID": "TEST-AXIS-SAMPLE-L1A",
            "algorithm_version": "1.0.0",
            "ProjectLongName": "Libera",
            "ProjectShortName": "Libera",
            "PlatformLongName": "TBD",
            "PlatformShortName": "NOAA-22",
        },
    )

    # Validate the Dataset structure matches the product definition
    # This ensures the fixture-generated Dataset conforms to test_l1a_product_definition.yml
    product_definition.check_dataset_conformance(ds, strict=True)

    return ds
