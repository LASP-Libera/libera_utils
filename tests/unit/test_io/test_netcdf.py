"""Unit tests for netcdf.py module"""

import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import dask
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from cloudpathlib import AnyPath, S3Path
from distributed import Client

from libera_utils.io import netcdf, product_definition
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.netcdf import (
    NetcdfEngine,
    _load_netcdf_into_memory,
    _write_worker_compressed,
    write_libera_data_product,
)
from libera_utils.io.product_definition import DATE_CREATED_ATTRIBUTE, LiberaDataProductDefinition
from libera_utils.io.smart_open import smart_open
from tests.marks import strict_warnings


@strict_warnings
class TestWriteLiberaDataProduct:
    """Tests for the write_libera_data_product function"""

    def test_write_libera_data_product_from_arrays(self, test_product_definition, test_data_dict, tmp_path):
        """Test successful writing of a data product with valid data as data arrays"""
        # Write the data product
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="radiometer_time",
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
        ds = xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config())
        assert "radiometer_time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars
        ds.close()

    def test_write_libera_data_product_from_dataset(self, test_product_definition, test_dataset, tmp_path):
        """Test successful writing of a data product with valid data as a Dataset"""
        # The writer sets date_created, so an input Dataset does not carry it
        del test_dataset.attrs["date_created"]

        # Write the data product
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_dataset,
            output_path=tmp_path,
            time_variable="radiometer_time",
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
        ds = xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config())
        assert "radiometer_time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars
        ds.close()

    def test_write_libera_data_product_from_arrays_with_dynamic_product_attribute(
        self, test_product_definition, test_data_dict, tmp_path
    ):
        """Test writing a data product from Dataset with dynamically set product level attributes"""
        # Modify data product definition to set algorithm_version as dynamic (required but set to null in product definition yaml)
        modified_product_definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        # Indicate this attribute is required but dynamic (set it to None in the definition object)
        modified_product_definition.attributes["algorithm_version"] = None

        # Write the data product
        result = write_libera_data_product(
            data_product_definition=modified_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="radiometer_time",
            dynamic_product_attributes={"algorithm_version": "1.2.3"},  # Dynamically set algorithm version
        )

        # Verify return type
        assert isinstance(result, LiberaDataProductFilename)

        # Verify file was created
        assert result.path.exists()
        assert result.path.parent == tmp_path

        # Verify filename format
        assert result.path.name.startswith("LIBERA_L1B_RAD-4CH_V1-2-3_")
        assert result.path.name.endswith(".nc")

        # Read back the file and verify basic structure
        ds = xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config())
        assert "radiometer_time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars
        assert ds.attrs["algorithm_version"] == "1.2.3"
        ds.close()

    def test_write_libera_data_product_from_dataset_with_dynamic_product_attribute(
        self, test_product_definition, test_dataset, tmp_path
    ):
        """Test writing a data product from Dataset with dynamically set product level attributes"""
        # Modify data product definition to set algorithm_version as dynamic (required but set to null in product definition yaml)
        modified_product_definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        # Indicate this attribute is required but dynamic (set it to None in the definition object)
        modified_product_definition.attributes["algorithm_version"] = None
        # The test dataset fixture already contains the algorithm_version attribute
        # The writer sets date_created, so an input Dataset does not carry it
        del test_dataset.attrs["date_created"]

        # Write the data product
        with pytest.raises(ValueError, match="dynamic_product_attributes is invalid when passing in a Dataset"):
            _ = write_libera_data_product(
                data_product_definition=modified_product_definition,
                data=test_dataset,
                output_path=tmp_path,
                time_variable="radiometer_time",
                dynamic_product_attributes={
                    "algorithm_version": "1.2.3"
                },  # Invalid usage because attribute is already dynamically set on the Dataset
            )

        result = write_libera_data_product(
            data_product_definition=modified_product_definition,
            data=test_dataset,
            output_path=tmp_path,
            time_variable="radiometer_time",
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
        ds = xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config())
        assert "radiometer_time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars
        assert ds.attrs["algorithm_version"] == "0.0.1"
        ds.close()

    def test_write_libera_data_product_with_s3_path(self, test_product_definition, test_data_dict, create_mock_bucket):
        """Test writing to an S3 path (mocked)"""
        mock_bucket = create_mock_bucket()
        output_path = S3Path(f"s3://{mock_bucket.name}/test-prefix")

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=output_path,
            time_variable="radiometer_time",
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
            time_variable="radiometer_time",
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
            time_variable="radiometer_time",
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
            time_variable="radiometer_time",
            strict=True,
        )

        assert result.path.exists()

    def test_write_libera_data_product_invalid_dimension(self, test_product_definition, test_data_dict, tmp_path):
        """Test that write_libera_data_product raises when product definition has invalid dimension"""
        with test_product_definition.open("r") as f:
            definition_contents = yaml.safe_load(f)
        definition_contents["coordinates"]["radiometer_time"]["dimensions"] = ["dne_dimension"]
        # Write the invalid definition to a temp file so it is loaded inside write_libera_data_product
        invalid_def_path = tmp_path / "invalid_def.yml"
        invalid_def_path.write_text(yaml.dump(definition_contents))

        with pytest.raises(ValueError, match="Undefined dimension name 'dne_dimension'"):
            write_libera_data_product(
                data_product_definition=invalid_def_path,
                data=test_data_dict,
                output_path=tmp_path,
                time_variable="radiometer_time",
            )

    def test_write_libera_data_product_strict_mode_invalid_exception(
        self, test_product_definition, test_data_dict, tmp_path
    ):
        """Test that strict mode raises exception in strict mode for problems (e.g. missing variable)"""

        del test_data_dict["q_flag"]  # Remove a required variable to trigger an error

        with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
            with pytest.warns(UserWarning, match="q_flag: missing variable"):
                write_libera_data_product(
                    data_product_definition=test_product_definition,
                    data=test_data_dict,
                    output_path=tmp_path,
                    time_variable="radiometer_time",
                    strict=True,
                )

    def test_write_libera_data_product_non_strict_mode(self, test_product_definition, test_data_dict, tmp_path, caplog):
        """Test non-strict mode allows writing with warnings

        This test covers:
        - missing variable
        - extra attribute
        - accidental override of standard attribute via dynamic attributes
        - safe dtype casting
        """
        with test_product_definition.open("r") as f:
            definition_contents = yaml.safe_load(f)
        modified_definition = LiberaDataProductDefinition(**definition_contents)

        del test_data_dict["q_flag"]  # Remove a variable
        test_data_dict["fil_rad"] = test_data_dict["fil_rad"].astype(
            np.float32
        )  # Change dtype to something that can be safely cast to the expected dtype (float32 -> float64)

        # Should not raise exception, but will issue and log warnings
        with caplog.at_level("INFO"):
            with pytest.warns(UserWarning, match=r".*") as warning_list:
                result = write_libera_data_product(
                    data_product_definition=modified_definition,
                    data=test_data_dict,
                    output_path=tmp_path,
                    time_variable="radiometer_time",
                    dynamic_product_attributes={
                        "ProjectShortName": "NotLibera",  # This overrides a standard attr incorrectly
                        "ExtraAttribute": "extra_value",  # This adds an extra attribute not present in definition
                    },
                    strict=False,
                )
        warning_messages = [str(w.message) for w in warning_list]

        # Missing variable warning and log message. This is the only piece here that would actually prevent
        # writing a data product in strict mode.
        assert any(["q_flag: missing variable" in msg for msg in warning_messages])
        assert any(["q_flag: missing variable" in msg for msg in caplog.messages])

        # Static attribute gets forced to correct value during enforcement step
        assert any(
            [
                "Dataset attribute value mismatch for 'ProjectShortName': Expected 'Libera' but got 'NotLibera'" in msg
                for msg in warning_messages
            ]
        )
        assert any(
            [
                "Overwrote global static attribute 'ProjectShortName' from 'NotLibera' to 'Libera'" in msg
                for msg in caplog.messages
            ]
        )

        # Extra attribute gets removed during enforcement step
        assert any(["Dataset has unexpected attribute 'ExtraAttribute'" in msg for msg in warning_messages])
        assert any(
            [
                "Removed unexpected global attribute 'ExtraAttribute' with value 'extra_value'" in msg
                for msg in caplog.messages
            ]
        )

        # File should still be created
        assert result.path.exists()

        # Verify the file has the variables that were provided
        ds = xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config())
        assert "radiometer_time" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" not in ds.data_vars  # This was missing
        assert ds.attrs["ProjectShortName"] == "Libera"  # This should have been enforced to correct value
        assert "ExtraAttribute" not in ds.attrs  # This should be removed since it's not in the definition
        assert ds["fil_rad"].dtype == np.float64  # This should have been safely upcast from float32
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
            "radiometer_time": time_data,
            "lat": np.linspace(-90, 90, num=n_times),
            "lon": np.linspace(-180, 180, num=n_times),
            "fil_rad": np.random.rand(n_times),
            "q_flag": np.random.randint(100, size=n_times, dtype=np.int32),
            "cartesian_position": np.zeros((n_times, 3), dtype=np.float32),
        }

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=data,
            output_path=tmp_path,
            time_variable="radiometer_time",
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
            time_variable="radiometer_time",
        )

        # Read back the file
        ds = xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config())

        # Check global attributes
        assert ds.attrs["ProductID"] == "RAD-4CH"
        assert ds.attrs["algorithm_version"] == "0.0.1"
        assert ds.attrs["Format"] == "NetCDF-4"
        assert ds.attrs["Conventions"] == "CF-1.8"

        # Check coordinates
        assert "radiometer_time" in ds.coords
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
            time_variable="radiometer_time",
        )
        # Load the product definition
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Read back the file and check conformance in strict mode
        with xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config()) as ds:
            definition.check_dataset_conformance(ds, strict=True)

    def test_write_libera_data_product_same_second_writes_get_distinct_names(
        self, test_product_definition, test_data_dict, tmp_path
    ):
        """Two writes within the same second get different filenames, each with a microsecond revision"""
        first_now = datetime(2027, 1, 3, 8, 9, 10, 1, tzinfo=UTC)
        second_now = datetime(2027, 1, 3, 8, 9, 10, 2, tzinfo=UTC)
        with patch.object(product_definition, "datetime", mock.Mock(wraps=datetime)) as mock_datetime:
            mock_datetime.now.side_effect = [first_now, second_now]
            first = write_libera_data_product(
                data_product_definition=test_product_definition,
                data=test_data_dict,
                output_path=tmp_path,
                time_variable="radiometer_time",
            )
            second = write_libera_data_product(
                data_product_definition=test_product_definition,
                data=test_data_dict,
                output_path=tmp_path,
                time_variable="radiometer_time",
            )

        assert first.path.name != second.path.name
        assert first.path.name.endswith("_R20270103T080910000001.nc")
        assert second.path.name.endswith("_R20270103T080910000002.nc")
        assert first.path.exists()
        assert second.path.exists()

    def test_write_libera_data_product_sets_date_created(self, test_product_definition, test_data_dict, tmp_path):
        """The written product carries a date_created equal to its filename revision"""
        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="radiometer_time",
        )

        with xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config()) as ds:
            date_created = datetime.fromisoformat(ds.attrs[DATE_CREATED_ATTRIBUTE])
        assert date_created == result.filename_parts.revision

    @pytest.mark.parametrize("from_dataset", [False, True])
    def test_write_libera_data_product_sets_date_created_once(
        self, test_product_definition, test_data_dict, test_dataset, tmp_path, from_dataset
    ):
        """date_created is set exactly once per write, for array input and for Dataset input"""
        if from_dataset:
            del test_dataset.attrs["date_created"]
        with (
            patch.object(netcdf, "_set_date_created", wraps=product_definition._set_date_created) as mock_writer_stamp,
            patch.object(
                product_definition, "_set_date_created", wraps=product_definition._set_date_created
            ) as mock_dataset_stamp,
        ):
            write_libera_data_product(
                data_product_definition=test_product_definition,
                data=test_dataset if from_dataset else test_data_dict,
                output_path=tmp_path,
                time_variable="radiometer_time",
            )
        assert mock_writer_stamp.call_count + mock_dataset_stamp.call_count == 1

    def test_write_libera_data_product_overwrites_date_created(
        self, test_product_definition, test_dataset, tmp_path, caplog
    ):
        """An input date_created is replaced by the write time, with a warning"""
        test_dataset.attrs["date_created"] = "2000-01-01T00:00:00+00:00"

        with caplog.at_level(logging.WARNING, logger="libera_utils.io.product_definition"):
            with pytest.warns(UserWarning, match="'date_created' was set to '2000-01-01T00:00:00\\+00:00'"):
                result = write_libera_data_product(
                    data_product_definition=test_product_definition,
                    data=test_dataset,
                    output_path=tmp_path,
                    time_variable="radiometer_time",
                )

        assert "Overwrote global attribute 'date_created'" in caplog.text
        with xr.open_dataset(result.path, engine=NetcdfEngine.get_from_config()) as ds:
            date_created = datetime.fromisoformat(ds.attrs[DATE_CREATED_ATTRIBUTE])
        assert date_created == result.filename_parts.revision
        assert date_created.year != 2000

    def test_write_libera_data_product_overwrite_existing(self, test_product_definition, test_data_dict, tmp_path):
        """Test that write_libera_data_product overwrites existing files"""
        # Write first time
        result1 = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="radiometer_time",
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
                utc_start=pd.Timestamp(test_data_dict["radiometer_time"][0]).to_pydatetime(),
                utc_end=pd.Timestamp(test_data_dict["radiometer_time"][-1]).to_pydatetime(),
            )
            mock_filename.path = result1.path
            mock_gen.return_value = mock_filename

            result2 = write_libera_data_product(
                data_product_definition=test_product_definition,
                data=modified_data,
                output_path=tmp_path,
                time_variable="radiometer_time",
            )

            # Should be the same file path
            assert result2.path == result1.path

            # Verify new data was written
            ds = xr.open_dataset(result2.path, engine=NetcdfEngine.get_from_config())
            np.testing.assert_array_almost_equal(ds["fil_rad"].values, modified_data["fil_rad"])
            ds.close()

    def test_write_libera_data_product_path_types(self, test_product_definition, test_data_dict, tmp_path):
        """Test that the function accepts various path types"""
        # Test with string path
        result1 = write_libera_data_product(
            data_product_definition=str(test_product_definition),
            data=test_data_dict,
            output_path=str(tmp_path),
            time_variable="radiometer_time",
        )
        assert result1.path.exists()

        # Test with Path object
        result2 = write_libera_data_product(
            data_product_definition=Path(test_product_definition),
            data=test_data_dict,
            output_path=Path(tmp_path),
            time_variable="radiometer_time",
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
            assert "radiometer_time" in ds_read.coords
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
        ds_read = xr.open_dataset(nc_file, engine="h5netcdf")
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
        ds_read = xr.open_dataset(nc_file, engine="h5netcdf")
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


class TestNetcdfEngineConfig:
    """Tests for NetcdfEngine configuration handling in write_libera_data_product"""

    def test_write_libera_data_netcdf4_local(self, monkeypatch, test_product_definition, test_data_dict, tmp_path):
        """
        Test that when config is set to 'netcdf4', the code writes using the file path directly.
        We use monkeypatch to set the environment variable, which overrides config.json.
        """
        # Force the configuration to be 'netcdf4'
        monkeypatch.setenv("XARRAY_NETCDF_ENGINE", "netcdf4")

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=tmp_path,
            time_variable="radiometer_time",
        )

        assert result.path.exists()

        # Verify we can read it back using the engine we insisted on
        with xr.open_dataset(result.path, engine="netcdf4") as ds:
            assert "fil_rad" in ds

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    def test_write_libera_data_product_to_s3_with_either_engine(
        self, engine, monkeypatch, test_product_definition, test_data_dict, create_mock_bucket, tmp_path
    ):
        """Both engines write to S3, because the product is written to a local file and uploaded"""
        monkeypatch.setenv("XARRAY_NETCDF_ENGINE", engine)

        mock_bucket = create_mock_bucket()
        output_path = S3Path(f"s3://{mock_bucket.name}/test-prefix")

        result = write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=output_path,
            time_variable="radiometer_time",
        )

        uploaded = [obj.key for obj in mock_bucket.objects.all()]
        assert uploaded == [f"test-prefix/{result.path.name}"]

        # The key alone passes for an empty or truncated staged file, so read the product back.
        downloaded = tmp_path / result.path.name
        mock_bucket.download_file(uploaded[0], str(downloaded))
        with xr.open_dataset(downloaded) as ds:
            assert "fil_rad" in ds
            assert ds["fil_rad"].shape == test_data_dict["fil_rad"].shape

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize("to_s3", [False, True], ids=["local", "s3"])
    def test_write_libera_data_product_never_hands_the_engine_a_file_object(
        self, engine, to_s3, monkeypatch, test_product_definition, test_data_dict, tmp_path, create_mock_bucket
    ):
        """The engine must receive a path, never an open file object

        xarray wraps the argument in a CachingFileManager that pickles as its opener and
        arguments so a Dask worker can reopen the file. An open file object has no path to
        reopen from, so the distributed scheduler fails on it.
        """
        monkeypatch.setenv("XARRAY_NETCDF_ENGINE", engine)
        output_path = S3Path(f"s3://{create_mock_bucket().name}/test-prefix") if to_s3 else tmp_path

        seen = []
        original_to_netcdf = xr.Dataset.to_netcdf

        def record_target(self, path=None, *args, **kwargs):
            seen.append(path)
            return original_to_netcdf(self, path, *args, **kwargs)

        monkeypatch.setattr(xr.Dataset, "to_netcdf", record_target)
        write_libera_data_product(
            data_product_definition=test_product_definition,
            data=test_data_dict,
            output_path=output_path,
            time_variable="radiometer_time",
        )

        assert seen, "to_netcdf was never called"
        for target in seen:
            assert isinstance(target, str | Path), f"{engine} was handed {type(target).__name__}"

    def test_s3_product_is_uploaded_and_leaves_no_local_copy(
        self, test_product_definition, test_data_dict, create_mock_bucket, tmp_path, monkeypatch
    ):
        """The product is written to a temporary file, uploaded, and the temporary file removed"""
        local_tmp = tmp_path / "tmp"
        local_tmp.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(local_tmp))
        output_path = S3Path(f"s3://{create_mock_bucket().name}/test-prefix")

        result = write_libera_data_product(test_product_definition, test_data_dict, output_path, "radiometer_time")

        assert result.path.exists()
        assert list(local_tmp.iterdir()) == []


def test_load_netcdf_into_memory_reads_the_file_and_closes_it(tmp_path):
    """The Dataset is fully loaded, so it reads back unchanged after the file is closed and removed"""
    expected = xr.Dataset({"values": (("t",), np.arange(5.0))}, coords={"t": np.arange(5)})
    path = tmp_path / "small.nc"
    expected.to_netcdf(path, engine="h5netcdf")
    loaded = _load_netcdf_into_memory(path)
    path.unlink()
    xr.testing.assert_identical(loaded, expected)


def test_load_netcdf_into_memory_unreadable_file_raises(tmp_path):
    """A file that is not NetCDF raises ValueError naming the file"""
    not_netcdf = tmp_path / "not_netcdf.nc"
    not_netcdf.write_text("not a NetCDF file")
    with pytest.raises(ValueError, match="Failed to open or read NetCDF file"):
        _load_netcdf_into_memory(not_netcdf)


@pytest.fixture
def distributed_client():
    """An in-process dask.distributed Client, closed after the test so later tests find none"""
    with Client(processes=False, n_workers=2, threads_per_worker=1, dashboard_address=None) as client:
        yield client


def _small_dataset(counter: Path, chunks: tuple[int, int, int] = (1, 16, 16), **float_encoding) -> xr.Dataset:
    """A 6-frame 64x64 Dataset whose float32 `floats` and uint16 `ints` share one task per 2-frame block

    Each run of that task appends a line to `counter`.
    """

    def produce(frame_start):
        with counter.open("a") as log:
            log.write("call\n")
        values = np.random.default_rng(frame_start).normal(size=(2, 64, 64)).astype(np.float32)
        values[:, :16] = -999
        return values, (np.abs(values) % 100).astype(np.uint16)

    task = dask.delayed(produce, nout=2, pure=False)
    float_blocks, int_blocks = [], []
    for start in range(0, 6, 2):
        values, counts = task(start)
        float_blocks.append(da.from_delayed(values, (2, 64, 64), np.float32))
        int_blocks.append(da.from_delayed(counts, (2, 64, 64), np.uint16))
    floats = da.concatenate(float_blocks)
    dataset = xr.Dataset(
        {
            "floats": (("t", "y", "x"), floats, {"_FillValue": np.float32(-999)}),
            "ints": (("t", "y", "x"), da.concatenate(int_blocks)),
            "frame_flag": (("t",), (floats.max(axis=(1, 2)) > 0).astype(np.uint32)),
        },
        coords={"t": np.arange(6)},
    )
    dataset["floats"].encoding = {"zlib": True, "complevel": 4, "chunksizes": chunks, "shuffle": True, **float_encoding}
    dataset["ints"].encoding = {"zlib": True, "complevel": 4, "chunksizes": chunks}
    return dataset


def _calls(counter: Path) -> int:
    """Number of times the shared block task has run, from the lines in `counter`; 0 before the first run"""
    return len(counter.read_text().split()) if counter.exists() else 0


def _stored_bits(dset: h5py.Dataset) -> np.ndarray:
    """Values read through HDF5's filters, without CF decoding, as unsigned integers of the same width"""
    values = dset[()]
    return values.view(f"u{values.dtype.itemsize}")


class TestWorkerCompression:
    """`worker_compressed_variables` stores worker-compressed chunks that read back as a to_netcdf write does"""

    @staticmethod
    def _camera_dataset() -> xr.Dataset:
        n_frames, size = 2, 2048
        rng = np.random.default_rng(0)
        values = rng.normal(size=(n_frames, size, size)).astype(np.float32)
        values[0, :512, :512] = -999
        counts = (np.abs(values) * 100).astype(np.uint16)
        counts[1, 512:1024, 512:1024] = 0
        times = pd.date_range("2026-07-12", periods=n_frames, freq="10s").to_numpy("datetime64[ns]")
        dims = ("CAMERA_TIME", "CAMERA_PIXEL_COUNT_X", "CAMERA_PIXEL_COUNT_Y")
        # Built (time, y, x) and transposed, so every block reaches the workers non-contiguous.
        shuffled = da.from_array(np.ascontiguousarray(values.transpose(0, 2, 1)), chunks=(1, 2048, 2048))
        return xr.Dataset(
            {
                "float_fill": (dims, da.from_array(values, chunks=(1, 1024, 2048))),
                "float_fill_shuffled": (dims, shuffled.transpose(0, 2, 1)),
                "counts": (dims, da.from_array(counts, chunks=(2, 512, 1024))),
                "mask": (dims, da.from_array((counts % 2).astype(np.uint8), chunks=(1, 2048, 2048))),
                "frame_flag": (("CAMERA_TIME",), da.from_array(np.array([1, 2], dtype=np.uint32), chunks=1)),
            },
            coords={"CAMERA_TIME": times},
            attrs={"ProductID": "CAM", "algorithm_version": "0.0.1"},
        )

    @pytest.mark.parametrize("to_s3", [False, True], ids=["local", "s3"])
    def test_reads_back_identical_to_a_to_netcdf_write(
        self, to_s3, distributed_client, test_worker_compression_product_definition, tmp_path, create_mock_bucket
    ):
        """Every variable's stored values, chunks, settings and metadata match a to_netcdf write"""
        names = ("float_fill", "float_fill_shuffled", "counts", "mask")
        (tmp_path / "reference").mkdir()
        (tmp_path / "direct").mkdir()
        reference = write_libera_data_product(
            test_worker_compression_product_definition, self._camera_dataset(), tmp_path / "reference", "CAMERA_TIME"
        )
        if to_s3:
            bucket = create_mock_bucket()
            written = write_libera_data_product(
                test_worker_compression_product_definition,
                self._camera_dataset(),
                S3Path(f"s3://{bucket.name}/prefix"),
                "CAMERA_TIME",
                worker_compressed_variables=names,
            )
            direct_path = tmp_path / written.path.name
            bucket.download_file(f"prefix/{written.path.name}", str(direct_path))
        else:
            direct_path = write_libera_data_product(
                test_worker_compression_product_definition,
                self._camera_dataset(),
                tmp_path / "direct",
                "CAMERA_TIME",
                worker_compressed_variables=names,
            ).path

        with h5py.File(reference.path) as expected, h5py.File(direct_path) as actual:
            for name in (*names, "frame_flag", "CAMERA_TIME"):
                for setting in ("dtype", "chunks", "compression", "compression_opts", "shuffle", "fillvalue"):
                    assert getattr(actual[name], setting) == getattr(expected[name], setting), (name, setting)
                np.testing.assert_array_equal(_stored_bits(actual[name]), _stored_bits(expected[name]), err_msg=name)
            # The all-fill and all-zero chunks in _camera_dataset are stored, as to_netcdf stores them.
            for name in names:
                assert actual[name].id.get_num_chunks() == expected[name].id.get_num_chunks(), name
        with xr.open_dataset(reference.path) as expected, xr.open_dataset(direct_path) as actual:
            # Each write stamps its own creation time.
            del actual.attrs[DATE_CREATED_ATTRIBUTE], expected.attrs[DATE_CREATED_ATTRIBUTE]
            assert actual.identical(expected)

    def test_a_task_shared_by_variables_runs_once_per_block(self, tmp_path, distributed_client):
        """One compute holds every variable's blocks, so a task feeding two variables runs once per block"""
        counter = tmp_path / "calls.log"
        dataset = _small_dataset(counter)
        dataset.to_netcdf(tmp_path / "reference.nc", engine="h5netcdf")
        counter.unlink()
        _write_worker_compressed(dataset, tmp_path / "out.nc", ("floats", "ints"))
        assert _calls(counter) == 3
        with h5py.File(tmp_path / "reference.nc") as expected, h5py.File(tmp_path / "out.nc") as actual:
            for name in ("floats", "ints", "frame_flag", "t"):
                np.testing.assert_array_equal(_stored_bits(actual[name]), _stored_bits(expected[name]), err_msg=name)

    def test_identically_built_variables_are_each_written(self, tmp_path, distributed_client):
        """Variables whose blocks share a task key each get the result, as a to_netcdf write does"""
        dataset = _small_dataset(tmp_path / "calls.log")
        for name in ("zeros_a", "zeros_b"):
            dataset[name] = (("t",), da.zeros(6, chunks=2, dtype=np.uint8) + 1)
        assert dataset["zeros_a"].data.name == dataset["zeros_b"].data.name
        dataset.to_netcdf(tmp_path / "reference.nc", engine="h5netcdf")
        _write_worker_compressed(dataset, tmp_path / "out.nc", ("floats",))
        with h5py.File(tmp_path / "reference.nc") as expected, h5py.File(tmp_path / "out.nc") as actual:
            for name in ("floats", "zeros_a", "zeros_b"):
                np.testing.assert_array_equal(_stored_bits(actual[name]), _stored_bits(expected[name]), err_msg=name)

    def test_unlimited_dims_named_by_a_string_are_unlimited(self, tmp_path, distributed_client):
        """A string ``unlimited_dims`` encoding marks that one dimension unlimited, as to_netcdf does"""
        dataset = _small_dataset(tmp_path / "calls.log")
        dataset.encoding["unlimited_dims"] = "t"
        _write_worker_compressed(dataset, tmp_path / "out.nc", ("floats",))
        with h5py.File(tmp_path / "out.nc") as written:
            assert written["floats"].maxshape == (None, 64, 64)
            assert written["ints"].maxshape == (None, 64, 64)

    def test_configured_engine_other_than_h5netcdf_raises(
        self, monkeypatch, test_worker_compression_product_definition, tmp_path
    ):
        monkeypatch.setenv("XARRAY_NETCDF_ENGINE", "netcdf4")
        with pytest.raises(ValueError, match="requires the h5netcdf engine"):
            write_libera_data_product(
                test_worker_compression_product_definition,
                self._camera_dataset(),
                tmp_path,
                "CAMERA_TIME",
                worker_compressed_variables=("counts",),
            )
        assert not list(tmp_path.iterdir())

    @pytest.mark.parametrize(
        ("names", "match"),
        [
            (("floats", "missing"), "not Dask-backed data variables"),
            (("t",), "not Dask-backed data variables"),
            (("floats", "floats"), "repeats a name"),
            (("numpy_values",), "not Dask-backed data variables"),
        ],
        ids=["unknown", "coordinate", "repeated", "numpy-backed"],
    )
    def test_names_that_are_not_distinct_dask_data_variables_raise(self, names, match, tmp_path, distributed_client):
        counter = tmp_path / "calls.log"
        dataset = _small_dataset(counter)
        dataset["numpy_values"] = (("t",), np.arange(6))
        with pytest.raises(ValueError, match=match):
            _write_worker_compressed(dataset, tmp_path / "out.nc", names)
        assert _calls(counter) == 0

    def test_variable_not_chunked_on_disk_raises(self, tmp_path, distributed_client):
        counter = tmp_path / "calls.log"
        dataset = _small_dataset(counter)
        dataset["floats"].encoding = {}
        with pytest.raises(ValueError, match="is not chunked on disk"):
            _write_worker_compressed(dataset, tmp_path / "out.nc", ("floats",))
        assert not (tmp_path / "out.nc").exists()
        assert _calls(counter) == 0

    def test_named_variable_that_is_not_numeric_raises(self, tmp_path, distributed_client):
        counter = tmp_path / "calls.log"
        dataset = _small_dataset(counter)
        dataset["labels"] = (("t",), da.from_array(np.array([b"ab"] * 6), chunks=3))
        # xarray stores the byte strings as a (t, string2) character array, hence two chunk lengths.
        dataset["labels"].encoding = {"zlib": True, "complevel": 4, "chunksizes": (3, 2)}
        with pytest.raises(ValueError, match="compress only integer and floating-point data"):
            _write_worker_compressed(dataset, tmp_path / "out.nc", ("labels",))
        assert not (tmp_path / "out.nc").exists()
        assert _calls(counter) == 0

    def test_no_distributed_client_raises(self, tmp_path):
        counter = tmp_path / "calls.log"
        with pytest.raises(ValueError, match="requires an active dask.distributed Client"):
            _write_worker_compressed(_small_dataset(counter), tmp_path / "out.nc", ("floats",))
        assert _calls(counter) == 0

    def test_filter_the_workers_do_not_apply_raises(self, tmp_path, distributed_client):
        counter = tmp_path / "calls.log"
        with pytest.raises(ValueError, match="the workers apply only deflate"):
            _write_worker_compressed(_small_dataset(counter, fletcher32=True), tmp_path / "out.nc", ("floats",))
        assert not (tmp_path / "out.nc").exists()
        assert _calls(counter) == 0

    def test_dask_chunks_not_a_multiple_of_the_disk_chunks_raise(self, tmp_path, distributed_client):
        counter = tmp_path / "calls.log"
        with pytest.raises(ValueError, match="each must be a whole multiple"):
            _write_worker_compressed(_small_dataset(counter, chunks=(1, 24, 64)), tmp_path / "out.nc", ("floats",))
        assert not (tmp_path / "out.nc").exists()
        assert _calls(counter) == 0

    def test_block_that_differs_from_its_declared_dtype_raises(self, tmp_path, distributed_client):
        """Dask does not check a computed block against from_delayed's declaration, so the worker does"""
        mislabeled = da.from_delayed(dask.delayed(np.zeros)((2, 32, 32), np.float64), (2, 32, 32), np.float32)
        dataset = xr.Dataset({"values": (("t", "y", "x"), mislabeled)})
        dataset["values"].encoding = {"zlib": True, "complevel": 4, "chunksizes": (1, 16, 16)}
        with pytest.raises(ValueError, match="A computed block of values has dtype float64"):
            _write_worker_compressed(dataset, tmp_path / "out.nc", ("values",))
        assert not (tmp_path / "out.nc").exists()

    def test_other_dask_variable_that_is_not_numeric_raises(self, tmp_path, distributed_client):
        counter = tmp_path / "calls.log"
        dataset = _small_dataset(counter)
        dataset["labels"] = (("t",), da.from_array(np.array([b"ab"] * 6), chunks=3))
        with pytest.raises(ValueError, match="not numeric or boolean"):
            _write_worker_compressed(dataset, tmp_path / "out.nc", ("floats",))
        assert not (tmp_path / "out.nc").exists()
        assert _calls(counter) == 0
