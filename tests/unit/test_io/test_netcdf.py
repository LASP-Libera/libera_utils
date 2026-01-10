"""Unit tests for netcdf.py module"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cloudpathlib import AnyPath, S3Path

from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.io.smart_open import smart_open


class TestWriteLiberaDataProduct:
    """Tests for the write_libera_data_product function"""

    def test_write_libera_data_product_from_arrays(self, test_product_definition, test_data_dict, tmp_path):
        """Test successful writing of a data product with valid data as data arrays"""
        # Write the data product
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
        )

        # Verify return type
        assert isinstance(result, LiberaDataProductFilename)

        # Verify file was created
        assert result.path.exists()
        assert result.path.parent == tmp_path

        # Verify filename format
        assert result.path.name.startswith("LIBERA_L1B_RAD-4CH_V0-0-1_")
        assert result.path.name.endswith(".nc")

        # Read back the file and verify basic structure
        ds = xr.open_dataset(result.path)
        assert "time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars
        ds.close()

    def test_write_libera_data_product_from_dataset(self, test_product_definition, test_dataset, tmp_path):
        """Test successful writing of a data product with valid data as a Dataset"""
        # Write the data product
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_dataset,
            output_path=tmp_path,
            time_variable="time",
        )

        # Verify return type
        assert isinstance(result, LiberaDataProductFilename)

        # Verify file was created
        assert result.path.exists()
        assert result.path.parent == tmp_path

        # Verify filename format
        assert result.path.name.startswith("LIBERA_L1B_RAD-4CH_V0-0-1_")
        assert result.path.name.endswith(".nc")

        # Read back the file and verify basic structure
        ds = xr.open_dataset(result.path)
        assert "time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars
        ds.close()

    def test_write_libera_data_product_with_s3_path(self, test_product_definition, test_data_dict, create_mock_bucket):
        """Test writing to an S3 path (mocked)"""
        mock_bucket = create_mock_bucket()
        output_path = S3Path(f"s3://{mock_bucket.name}/test-prefix")

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=output_path,
            time_variable="time",
        )

        # Verify result has correct path
        assert result.path.name.startswith("LIBERA_L1B_RAD-4CH_V0-0-1_")
        assert result.path.name.endswith(".nc")

    def test_write_libera_data_product_creates_correct_filename(
        self, test_product_definition, test_data_dict, tmp_path
    ):
        """Verify the generated filename matches expected format"""
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
        )

        # Verify filename contains correct product ID and version
        assert "RAD-4CH" in result.path.name
        assert "V0-0-1" in result.path.name

        # Verify filename contains time range (formatted as expected)
        # The filename should contain the start and end times in ISO format
        filename = result.path.name
        assert filename.startswith("LIBERA_L1B_RAD-4CH_V0-0-1_")

    def test_write_libera_data_product_add_archive_path_prefix(self, test_product_definition, test_data_dict, tmp_path):
        """Test that archive path prefix is added when specified"""
        # Mock config to return an archive path prefix
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
            add_archive_path_prefix=True,
        )
        print(result.path)

        # Verify that the output path includes the archive prefix
        expected_parent = tmp_path / "RAD-4CH/2024/01/01"
        assert result.path.parent == expected_parent

    def test_write_libera_data_product_strict_mode_valid(self, test_product_definition, test_data_dict, tmp_path):
        """Test that strict mode passes with valid data"""
        # Should not raise any exception
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
            strict=True,
        )

        assert result.path.exists()

    def test_write_libera_data_product_strict_mode_invalid(
        self, test_product_definition, test_data_dict_missing_variable, tmp_path
    ):
        """Test that strict mode raises exception with invalid data"""
        with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
            write_libera_data_product(
                data_product_definition=test_product_definition,
                data=test_data_dict_missing_variable,
                output_path=tmp_path,
                time_variable="time",
                strict=True,
            )

    def test_write_libera_data_product_non_strict_mode(
        self, test_product_definition, test_data_dict_missing_variable, tmp_path
    ):
        """Test non-strict mode allows writing with warnings"""
        # Should not raise exception, but will log warnings
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict_missing_variable,
            output_path=tmp_path,
            time_variable="time",
            strict=False,
        )

        # File should still be created
        assert result.path.exists()

        # Verify the file has the variables that were provided
        ds = xr.open_dataset(result.path)
        assert "time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" not in ds.data_vars  # This was missing
        ds.close()

    def test_write_libera_data_product_time_range_extraction(self, test_product_definition, tmp_path):
        """Verify correct extraction of start/end times from data"""
        # Create data with known time range
        n_times = 10
        start_time = np.datetime64("2025-01-01T00:00:00", "ns")
        end_time = np.datetime64("2025-01-01T23:59:59", "ns")
        time_data = np.array(
            [start_time, end_time]
            + list(np.linspace(start_time.astype("i8"), end_time.astype("i8"), n_times - 2).astype("datetime64[ns]"))
        )
        time_data.sort()

        data = {
            "time": time_data,
            "lat": np.linspace(-90, 90, num=n_times),
            "lon": np.linspace(-180, 180, num=n_times),
            "fil_rad": np.random.rand(n_times),
            "q_flag": np.random.randint(100, size=n_times, dtype=np.int32),
        }

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=data,
            output_path=tmp_path,
            time_variable="time",
        )

        # Verify filename contains the correct date (2025-01-01)
        filename = result.path.name
        # The filename format includes the time range
        assert "20250101" in filename  # Date should be present

    def test_write_libera_data_product_netcdf_content(self, test_product_definition, test_data_dict, tmp_path):
        """Verify the written NetCDF file contains correct data and metadata"""
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
        )

        # Read back the file
        ds = xr.open_dataset(result.path)

        # Check global attributes
        assert ds.attrs["ProductID"] == "RAD-4CH"
        assert ds.attrs["algorithm_version"] == "0.0.1"
        assert ds.attrs["Format"] == "NetCDF-4"
        assert ds.attrs["Conventions"] == "CF-1.8"

        # Check coordinates
        assert "time" in ds.coords
        assert "lat" in ds.coords
        assert "lon" in ds.coords

        # Check data variables
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars

        # Check variable attributes
        assert ds["fil_rad"].attrs["long_name"] == "Filtered Radiance"
        assert ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"
        assert ds["q_flag"].attrs["long_name"] == "Quality Flags"

        # Check data values match input
        np.testing.assert_array_almost_equal(ds["fil_rad"].values, test_data_dict["fil_rad"])
        np.testing.assert_array_equal(ds["q_flag"].values, test_data_dict["q_flag"])
        np.testing.assert_array_almost_equal(ds["lat"].values, test_data_dict["lat"])
        np.testing.assert_array_almost_equal(ds["lon"].values, test_data_dict["lon"])

        ds.close()

    def test_write_libera_data_product_conformance_after_write(self, test_product_definition, test_data_dict, tmp_path):
        """Verify written file passes conformance check when read back"""
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
        )

        # Read back the file
        ds = xr.open_dataset(result.path)

        # Load the product definition and check conformance
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        errors = definition.check_dataset_conformance(ds, strict=False)

        ds.close()

        # The dataset should be valid (no errors)
        assert not errors
        assert len(errors) == 0

    def test_write_libera_data_product_overwrite_existing(self, test_product_definition, test_data_dict, tmp_path):
        """Test that write_libera_data_product overwrites existing files"""
        # Write first time
        result1 = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="time",
        )

        # Modify data slightly
        modified_data = test_data_dict.copy()
        modified_data["fil_rad"] = np.ones_like(test_data_dict["fil_rad"]) * 0.5

        # Write to same location (should overwrite)
        # Need to manually set the path to be the same
        with patch.object(LiberaDataProductDefinition, "generate_data_product_filename") as mock_gen:
            mock_filename = LiberaDataProductFilename.from_filename_parts(
                product_name="RAD-4CH",
                version="V0-0-1",
                utc_start=pd.Timestamp(test_data_dict["time"][0]).to_pydatetime(),
                utc_end=pd.Timestamp(test_data_dict["time"][-1]).to_pydatetime(),
            )
            mock_filename.path = result1.path
            mock_gen.return_value = mock_filename

            result2 = write_libera_data_product(
                data_product_definition=test_product_definition,
                data=modified_data,
                output_path=tmp_path,
                time_variable="time",
            )

            # Should be the same file path
            assert result2.path == result1.path

            # Verify new data was written
            ds = xr.open_dataset(result2.path)
            np.testing.assert_array_almost_equal(ds["fil_rad"].values, modified_data["fil_rad"])
            ds.close()

    def test_write_libera_data_product_different_time_variable(self, test_product_definition, tmp_path):
        """Test using a different time variable name if it exists in coordinates"""
        # Create data with an alternate time coordinate
        n_times = 10
        time_data = np.arange(n_times).astype("datetime64[ns]")

        # Use a different name for the primary time variable
        data = {
            "time": time_data,  # This will be the coordinate
            "lat": np.linspace(-90, 90, num=n_times),
            "lon": np.linspace(-180, 180, num=n_times),
            "fil_rad": np.random.rand(n_times),
            "q_flag": np.random.randint(100, size=n_times, dtype=np.int32),
        }

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=data,
            output_path=tmp_path,
            time_variable="time",  # Specify which variable to use for time range
        )

        assert result.path.exists()

    def test_write_libera_data_product_path_types(self, test_product_definition, test_data_dict, tmp_path):
        """Test that the function accepts various path types"""
        # Test with string path
        result1 = write_libera_data_product(
            data_product_definition=str(test_product_definition),
            data=test_data_dict,
            output_path=str(tmp_path),
            time_variable="time",
        )
        assert result1.path.exists()

        # Test with Path object
        result2 = write_libera_data_product(
            data_product_definition=Path(test_product_definition),
            data=test_data_dict,
            output_path=Path(tmp_path),
            time_variable="time",
        )
        assert result2.path.exists()


class TestNetCDFSupport:
    """Tests for NetCDF reading/writing with various path types and engines"""

    # Test smart_open context manager for reading

    def test_smart_open_read_local_path(self, test_dataset, tmp_path):
        """Test reading NetCDF file from local path using smart_open"""
        # Write test dataset to local path
        nc_file = tmp_path / "test_data.nc"
        test_dataset.to_netcdf(nc_file, engine="h5netcdf")

        # Read using smart_open (must use h5netcdf for file-like objects)
        with smart_open(nc_file, mode="rb") as f:
            ds_read = xr.open_dataset(f, engine="h5netcdf")
            # Verify data matches
            assert "fil_rad" in ds_read.data_vars
            assert "q_flag" in ds_read.data_vars
            assert "time" in ds_read.coords
            np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
            ds_read.close()

    def test_smart_open_read_s3_path(self, test_dataset, tmp_path, create_mock_bucket, write_file_to_s3):
        """Test reading NetCDF file from S3 path using smart_open"""
        # First write dataset to local file
        local_nc_file = tmp_path / "test_data_s3.nc"
        test_dataset.to_netcdf(local_nc_file, engine="h5netcdf")

        # Upload to S3
        bucket = create_mock_bucket()
        s3_uri = f"s3://{bucket.name}/data/test_data.nc"
        write_file_to_s3(local_nc_file, s3_uri)

        # Read using smart_open (must use h5netcdf for file-like objects)
        with smart_open(s3_uri, mode="rb") as f:
            ds_read = xr.open_dataset(f, engine="h5netcdf")
            # Verify data matches
            assert "fil_rad" in ds_read.data_vars
            assert "q_flag" in ds_read.data_vars
            np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
            ds_read.close()

    # Test smart_open context manager for writing

    def test_smart_open_write_local_path(self, test_dataset, tmp_path):
        """Test writing NetCDF file to local path using smart_open"""
        nc_file = tmp_path / "test_write.nc"

        # Write using smart_open (must use h5netcdf for file-like objects)
        with smart_open(nc_file, mode="wb") as f:
            test_dataset.to_netcdf(f, engine="h5netcdf")

        # Read back and verify
        ds_read = xr.open_dataset(nc_file)
        assert "fil_rad" in ds_read.data_vars
        assert "q_flag" in ds_read.data_vars
        np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
        ds_read.close()

    def test_smart_open_write_s3_path(self, test_dataset, create_mock_bucket):
        """Test writing NetCDF file to S3 path using smart_open"""
        bucket = create_mock_bucket()
        s3_uri = f"s3://{bucket.name}/output/test_write.nc"

        # Write using smart_open (must use h5netcdf for file-like objects)
        with smart_open(s3_uri, mode="wb") as f:
            test_dataset.to_netcdf(f, engine="h5netcdf")

        # Read back and verify
        s3_path = S3Path(s3_uri)
        assert s3_path.exists()
        with smart_open(s3_uri, mode="rb") as f:
            ds_read = xr.open_dataset(f, engine="h5netcdf")
            assert "fil_rad" in ds_read.data_vars
            np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
            ds_read.close()

    # Test AnyPath.open() context manager for reading

    def test_anypath_open_read_local_path(self, test_dataset, tmp_path):
        """Test reading NetCDF file from local path using AnyPath.open()"""
        # Write test dataset to local path
        nc_file = tmp_path / "test_anypath.nc"
        test_dataset.to_netcdf(nc_file, engine="h5netcdf")

        # Read using AnyPath.open() (must use h5netcdf for file-like objects)
        any_path = AnyPath(nc_file)
        with any_path.open(mode="rb") as f:
            ds_read = xr.open_dataset(f, engine="h5netcdf")
            assert "fil_rad" in ds_read.data_vars
            assert "q_flag" in ds_read.data_vars
            np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
            ds_read.close()

    def test_anypath_open_read_s3_path(self, test_dataset, tmp_path, create_mock_bucket, write_file_to_s3):
        """Test reading NetCDF file from S3 path using AnyPath.open()"""
        # First write dataset to local file
        local_nc_file = tmp_path / "test_anypath_s3.nc"
        test_dataset.to_netcdf(local_nc_file, engine="h5netcdf")

        # Upload to S3
        bucket = create_mock_bucket()
        s3_uri = f"s3://{bucket.name}/data/test_anypath.nc"
        write_file_to_s3(local_nc_file, s3_uri)

        # Read using AnyPath.open() (must use h5netcdf for file-like objects)
        any_path = AnyPath(s3_uri)
        with any_path.open(mode="rb") as f:
            ds_read = xr.open_dataset(f, engine="h5netcdf")
            assert "fil_rad" in ds_read.data_vars
            np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
            ds_read.close()

    # Test AnyPath.open() context manager for writing

    def test_anypath_open_write_local_path(self, test_dataset, tmp_path):
        """Test writing NetCDF file to local path using AnyPath.open()"""
        nc_file = tmp_path / "test_anypath_write.nc"
        any_path = AnyPath(nc_file)

        # Write using AnyPath.open() (must use h5netcdf for file-like objects)
        with any_path.open(mode="wb") as f:
            test_dataset.to_netcdf(f, engine="h5netcdf")

        # Read back and verify
        ds_read = xr.open_dataset(nc_file)
        assert "fil_rad" in ds_read.data_vars
        np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
        ds_read.close()

    def test_anypath_open_write_s3_path(self, test_dataset, create_mock_bucket):
        """Test writing NetCDF file to S3 path using AnyPath.open()"""
        bucket = create_mock_bucket()
        s3_uri = f"s3://{bucket.name}/output/test_anypath_write.nc"
        any_path = AnyPath(s3_uri)

        # Write using AnyPath.open() (must use h5netcdf for file-like objects)
        with any_path.open(mode="wb") as f:
            test_dataset.to_netcdf(f, engine="h5netcdf")

        # Read back and verify
        assert any_path.exists()
        with any_path.open(mode="rb") as f:
            ds_read = xr.open_dataset(f, engine="h5netcdf")
            assert "fil_rad" in ds_read.data_vars
            np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
            ds_read.close()

    # Test engine compatibility

    def test_netcdf4_engine_roundtrip(self, test_dataset, tmp_path):
        """Test reading and writing with engine='netcdf4'"""
        nc_file = tmp_path / "test_netcdf4.nc"

        # Write with netcdf4
        test_dataset.to_netcdf(nc_file, engine="netcdf4")

        # Read with netcdf4
        ds_read = xr.open_dataset(nc_file, engine="netcdf4")
        assert "fil_rad" in ds_read.data_vars
        np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
        ds_read.close()

    def test_h5netcdf_engine_roundtrip(self, test_dataset, tmp_path):
        """Test reading and writing with engine='h5netcdf'"""
        nc_file = tmp_path / "test_h5netcdf.nc"

        # Write with h5netcdf
        test_dataset.to_netcdf(nc_file, engine="h5netcdf")

        # Read with h5netcdf
        ds_read = xr.open_dataset(nc_file, engine="h5netcdf")
        assert "fil_rad" in ds_read.data_vars
        np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
        ds_read.close()

    def test_write_netcdf4_read_h5netcdf(self, test_dataset, tmp_path):
        """Test writing with netcdf4 engine and reading with h5netcdf engine"""
        nc_file = tmp_path / "test_cross_engine1.nc"

        # Write with netcdf4
        test_dataset.to_netcdf(nc_file, engine="netcdf4")

        # Read with h5netcdf
        ds_read = xr.open_dataset(nc_file, engine="h5netcdf")
        assert "fil_rad" in ds_read.data_vars
        assert "q_flag" in ds_read.data_vars
        np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
        ds_read.close()

    def test_write_h5netcdf_read_netcdf4(self, test_dataset, tmp_path):
        """Test writing with h5netcdf engine and reading with netcdf4 engine"""
        nc_file = tmp_path / "test_cross_engine2.nc"

        # Write with h5netcdf
        test_dataset.to_netcdf(nc_file, engine="h5netcdf")

        # Read with netcdf4
        ds_read = xr.open_dataset(nc_file, engine="netcdf4")
        assert "fil_rad" in ds_read.data_vars
        assert "q_flag" in ds_read.data_vars
        np.testing.assert_array_almost_equal(ds_read["fil_rad"].values, test_dataset["fil_rad"].values)
        ds_read.close()
