"""Pytest fixtures for data product testing"""

import numpy as np
import pytest
from xarray import DataArray, Dataset


@pytest.fixture
def test_dataset():
    """Test Dataset for testing the validation functions of the LiberaDataProductDefinition class

    This is a fully manually created Dataset object that perfectly matches the data product definition in
    unit_test_product_definition.yml
    """
    n_times = 20
    start_us = np.datetime64("2024-01-01 00:00:00.000000").astype("datetime64[us]").astype(np.int64)
    end_us = np.datetime64("2024-01-01 23:59:59.999999").astype("datetime64[us]").astype(np.int64)
    random_us = np.random.randint(start_us, end_us, size=n_times)
    random_us.sort()
    times = random_us.astype("datetime64[ns]")
    time = DataArray(times, dims=["time"], attrs={"long_name": "Time of sample collection"})
    time.encoding = {
        "units": "nanoseconds since 1958-01-01",
        "calendar": "standard",
        "dtype": "int64",
        "zlib": True,
        "complevel": 4,
    }
    lat = DataArray(
        np.linspace(-90, 90, num=n_times),
        dims=["time"],
        attrs={"long_name": "Geolocation latitude", "units": "degrees", "valid_range": [-90, 90]},
    )
    lat.encoding = {"zlib": True, "complevel": 4}

    lon = DataArray(
        np.linspace(-180, 180, num=n_times),
        dims=["time"],
        attrs={"long_name": "Geolocation longitude", "units": "degrees", "valid_range": [-180, 180]},
    )
    lon.encoding = {"zlib": True, "complevel": 4}

    fil_rad = DataArray(
        np.random.rand(n_times),
        dims=["time"],
        attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
    )
    fil_rad.encoding = {"zlib": True, "complevel": 4}

    q_flag = DataArray(
        np.random.randint(2147483647, size=n_times, dtype=np.int32),
        dims=["time"],
        attrs={"long_name": "Quality Flags", "valid_range": [0, 2147483647]},
    )
    q_flag.encoding = {"zlib": True, "complevel": 4}
    ds = Dataset(
        data_vars={"fil_rad": fil_rad, "q_flag": q_flag},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={
            "ProductID": "RAD-4CH",
            "algorithm_version": "0.0.1",
            "Format": "NetCDF-4",
            "Conventions": "CF-1.8",
            "ProjectLongName": "Libera",
            "ProjectShortName": "Libera",
            "PlatformLongName": "TBD",
            "PlatformShortName": "NOAA-22",
        },
    )
    return ds


@pytest.fixture
def test_data_dict():
    """Dictionary of numpy arrays for creating test datasets

    Returns data that matches the unit_test_product_definition.yml structure
    """
    n_times = 20
    start_us = np.datetime64("2024-01-01 00:00:00.000000").astype("datetime64[us]").astype(np.int64)
    end_us = np.datetime64("2024-01-01 23:59:59.999999").astype("datetime64[us]").astype(np.int64)
    random_us = np.random.randint(start_us, end_us, size=n_times)
    random_us.sort()
    times = random_us.astype("datetime64[ns]")

    return {
        "time": times,
        "lat": np.linspace(-90, 90, num=n_times),
        "lon": np.linspace(-180, 180, num=n_times),
        "fil_rad": np.random.rand(n_times),
        "q_flag": np.random.randint(2147483647, size=n_times, dtype=np.int32),
    }


@pytest.fixture
def test_data_dict_missing_variable():
    """Dictionary missing required variables for testing strict mode"""
    n_times = 20
    start_us = np.datetime64("2024-01-01 00:00:00.000000").astype("datetime64[us]").astype(np.int64)
    end_us = np.datetime64("2024-01-01 23:59:59.999999").astype("datetime64[us]").astype(np.int64)
    random_us = np.random.randint(start_us, end_us, size=n_times)
    random_us.sort()
    times = random_us.astype("datetime64[ns]")

    return {
        "time": times,
        "lat": np.linspace(-90, 90, num=n_times),
        "lon": np.linspace(-180, 180, num=n_times),
        "fil_rad": np.random.rand(n_times),
        # Missing q_flag variable
    }
