"""Tests for data product definition YAML parsing and validation."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.product_definition import LiberaDataProductDefinition, LiberaVariableDefinition


class TestLiberaDataProductDefinition:
    """Tests for the LiberaDataProductDefinition class."""

    def test_valid_definition(self, test_product_definition):
        """Test loading a valid product definition."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        assert definition is not None
        assert "time" in definition.coordinates
        assert "fil_rad" in definition.variables
        assert isinstance(definition.variables["fil_rad"], LiberaVariableDefinition)

        # Verify that the class attributes are frozen
        with pytest.raises(ValidationError, match="Instance is frozen"):
            definition.attributes = {"should": "fail"}

    def test_check_dataset_conformance(self, tmp_path, test_product_definition, test_dataset):
        """Test validating a dataset against a product definition object"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        errors = definition.check_dataset_conformance(test_dataset)
        assert not errors
        assert len(errors) == 0
        outpath = tmp_path / "test_dataset.nc"
        test_dataset.to_netcdf(outpath)
        read_dataset = xr.open_dataset(outpath)
        print(read_dataset)
        errors = definition.check_dataset_conformance(read_dataset)
        assert not errors
        assert len(errors) == 0

    def test_check_dataset_conformance_invalid_version(self, test_product_definition, test_dataset):
        """Test that check_dataset_conformance catches invalid algorithm_version format"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Set an invalid algorithm_version
        test_dataset.attrs["algorithm_version"] = "version_one"

        errors = definition.check_dataset_conformance(test_dataset, strict=False)
        assert errors
        assert any("algorithm_version: invalid format - Expected semantic versioning" in e for e in errors)

        with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
            definition.check_dataset_conformance(test_dataset)

    def test_generate_filename(self, test_product_definition):
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        start = datetime.fromisoformat("2025-01-01T00:00:00.000000")
        end = datetime.fromisoformat("2025-01-01T23:59:59.999999")
        fn = definition.generate_data_product_filename(utc_start=start, utc_end=end)
        assert isinstance(fn, LiberaDataProductFilename)

    def test_enforce_dataset_conformance_missing_attributes(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance adds missing attributes."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Remove some global attributes
        del test_dataset.attrs["Format"]
        del test_dataset.attrs["Conventions"]

        # Remove some variable attributes
        del test_dataset["fil_rad"].attrs["units"]
        del test_dataset["time"].attrs["long_name"]

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(test_dataset)

        # Check global attributes were added back
        assert fixed_ds.attrs["Format"] == "NetCDF-4"
        assert fixed_ds.attrs["Conventions"] == "CF-1.8"

        # Check variable attributes were added back
        assert fixed_ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"
        assert fixed_ds["time"].attrs["long_name"] == "Time of sample collection"

        # Should be valid after fixing (no errors)
        assert not errors

    def test_enforce_dataset_conformance_extra_attributes(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance removes extra attributes."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Add extra global attributes
        test_dataset.attrs["extra_global"] = "should be removed"
        test_dataset.attrs["another_extra"] = 123

        # Add extra variable attributes
        test_dataset["fil_rad"].attrs["extra_var_attr"] = "remove me"
        test_dataset["time"].attrs["unexpected"] = 42

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(test_dataset)

        # Check extra global attributes were removed
        assert "extra_global" not in fixed_ds.attrs
        assert "another_extra" not in fixed_ds.attrs

        # Check extra variable attributes were removed
        assert "extra_var_attr" not in fixed_ds["fil_rad"].attrs
        assert "unexpected" not in fixed_ds["time"].attrs

        # Should be valid after fixing (no errors)
        assert not errors

    def test_enforce_dataset_conformance_wrong_attribute_values(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance corrects wrong attribute values."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Set wrong values for some attributes
        test_dataset.attrs["ProductID"] = "WRONG-ID"
        test_dataset.attrs["algorithm_version"] = "999.0.0"
        test_dataset["fil_rad"].attrs["units"] = "wrong_units"

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(test_dataset)

        # Check values were corrected
        assert fixed_ds.attrs["ProductID"] == "RAD-4CH"
        assert fixed_ds.attrs["algorithm_version"] == "0.0.1"
        assert fixed_ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"

        # Should be valid after fixing (no errors)
        assert not errors

    def test_enforce_dataset_conformance_dtype_conversion(self, test_product_definition):
        """Test that enforce_dataset_conformance converts dtypes correctly."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create a dataset with wrong dtypes
        n_times = 10
        times = np.arange(n_times).astype("datetime64[s]")  # Wrong precision

        ds = xr.Dataset(
            data_vars={
                "fil_rad": xr.DataArray(
                    np.random.rand(n_times).astype("float32"),  # Should be float64
                    dims=["time"],
                    attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
                ),
                "q_flag": xr.DataArray(
                    np.random.randint(100, size=n_times, dtype=np.int64),  # Should be int32
                    dims=["time"],
                    attrs={"long_name": "Quality Flags", "valid_range": [0, 2147483647]},
                ),
            },
            coords={
                "time": xr.DataArray(times, dims=["time"], attrs={"long_name": "Time of sample collection"}),
                "lat": xr.DataArray(
                    np.linspace(-90, 90, n_times).astype("float32"),  # Should be float64
                    dims=["time"],
                    attrs={"long_name": "Geolocation latitude", "units": "degrees", "valid_range": [-90, 90]},
                ),
                "lon": xr.DataArray(
                    np.linspace(-180, 180, n_times),  # Already float64
                    dims=["time"],
                    attrs={"long_name": "Geolocation longitude", "units": "degrees", "valid_range": [-180, 180]},
                ),
            },
            attrs={
                "ProductID": "RAD-4CH",
                "version": "0.0.1",
                "Format": "NetCDF-4",
                "Conventions": "CF-1.8",
                "ProjectLongName": "Libera",
                "ProjectShortName": "Libera",
                "PlatformLongName": "TBD",
                "PlatformShortName": "NOAA-22",
            },
        )

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(ds)

        # Check dtypes were converted
        assert str(fixed_ds["time"].dtype) == "datetime64[ns]"
        assert str(fixed_ds["fil_rad"].dtype) == "float64"
        assert str(fixed_ds["q_flag"].dtype) == "int32"
        assert str(fixed_ds["lat"].dtype) == "float64"
        assert str(fixed_ds["lon"].dtype) == "float64"

        # Should be valid after fixing (no errors)
        assert not errors

    def test_enforce_dataset_conformance_missing_variables(self, test_product_definition):
        """Test that enforce_dataset_conformance handles missing variables correctly."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create dataset missing some variables
        n_times = 10
        times = np.arange(n_times).astype("datetime64[ns]")

        ds = xr.Dataset(
            data_vars={
                # Missing "q_flag" variable
                "fil_rad": xr.DataArray(
                    np.random.rand(n_times),
                    dims=["time"],
                    attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
                ),
            },
            coords={
                "time": xr.DataArray(times, dims=["time"], attrs={"long_name": "Time of sample collection"}),
                # Missing "lat" coordinate
                "lon": xr.DataArray(
                    np.linspace(-180, 180, n_times),
                    dims=["time"],
                    attrs={"long_name": "Geolocation longitude", "units": "degrees", "valid_range": [-180, 180]},
                ),
            },
            attrs={
                "ProductID": "RAD-4CH",
                "version": "0.0.1",
                "Format": "NetCDF-4",
                "Conventions": "CF-1.8",
                "ProjectLongName": "Libera",
                "ProjectShortName": "Libera",
                "PlatformLongName": "TBD",
                "PlatformShortName": "NOAA-22",
            },
        )

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(ds)

        # Dataset should be modified but have errors due to missing variables
        assert errors  # Should have errors

        # Existing variables should still be fixed
        assert fixed_ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"

    def test_enforce_dataset_conformance_encoding_updates(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance updates encoding correctly."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Clear or modify some encoding settings
        test_dataset["time"].encoding = {}  # Clear all encoding
        test_dataset["fil_rad"].encoding = {"zlib": False}  # Wrong value

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(test_dataset)

        # Check encoding was updated for time coordinate
        assert fixed_ds["time"].encoding["units"] == "nanoseconds since 1958-01-01"
        assert fixed_ds["time"].encoding["calendar"] == "standard"
        assert fixed_ds["time"].encoding["dtype"] == "int64"
        assert fixed_ds["time"].encoding["zlib"] is True
        assert fixed_ds["time"].encoding["complevel"] == 4

        # Check encoding was updated for fil_rad variable
        # Note: Variables in the product definition get default encoding (zlib=True, complevel=4)
        assert fixed_ds["fil_rad"].encoding["zlib"] is True
        assert fixed_ds["fil_rad"].encoding["complevel"] == 4

        # Check other variables also get default encoding
        assert fixed_ds["q_flag"].encoding["zlib"] is True
        assert fixed_ds["q_flag"].encoding["complevel"] == 4
        assert fixed_ds["lat"].encoding["zlib"] is True
        assert fixed_ds["lat"].encoding["complevel"] == 4
        assert fixed_ds["lon"].encoding["zlib"] is True
        assert fixed_ds["lon"].encoding["complevel"] == 4

        # Should be valid after fixing (no errors)
        assert not errors

    def test_enforce_dataset_conformance_dimension_mismatch(self, test_product_definition):
        """Test that enforce_dataset_conformance warns about dimension mismatches."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create dataset with wrong dimensions
        n_times = 10
        n_extra = 5
        times = np.arange(n_times).astype("datetime64[ns]")

        ds = xr.Dataset(
            data_vars={
                "fil_rad": xr.DataArray(
                    np.random.rand(n_times, n_extra),  # Wrong: should be 1D
                    dims=["time", "extra_dim"],
                    attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
                ),
                "q_flag": xr.DataArray(
                    np.random.randint(100, size=n_times, dtype=np.int32),
                    dims=["time"],
                    attrs={"long_name": "Quality Flags", "valid_range": [0, 2147483647]},
                ),
            },
            coords={
                "time": xr.DataArray(times, dims=["time"], attrs={"long_name": "Time of sample collection"}),
                "lat": xr.DataArray(
                    np.linspace(-90, 90, n_times),
                    dims=["time"],
                    attrs={"long_name": "Geolocation latitude", "units": "degrees", "valid_range": [-90, 90]},
                ),
                "lon": xr.DataArray(
                    np.linspace(-180, 180, n_times),
                    dims=["time"],
                    attrs={"long_name": "Geolocation longitude", "units": "degrees", "valid_range": [-180, 180]},
                ),
            },
            attrs={
                "ProductID": "RAD-4CH",
                "version": "0.0.1",
                "Format": "NetCDF-4",
                "Conventions": "CF-1.8",
                "ProjectLongName": "Libera",
                "ProjectShortName": "Libera",
                "PlatformLongName": "TBD",
                "PlatformShortName": "NOAA-22",
            },
        )

        # Fix the dataset
        fixed_ds, errors = definition.enforce_dataset_conformance(ds)

        # Should have errors due to dimension mismatch
        assert errors

        # Other aspects should still be fixed
        assert fixed_ds.attrs["ProductID"] == "RAD-4CH"

    def test_check_dataset_conformance_error_reporting(self, test_product_definition):
        """Test that check_dataset_conformance returns detailed error messages"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create a dataset with various problems
        n_times = 10
        times = np.arange(n_times).astype("datetime64[ns]")

        ds = xr.Dataset(
            data_vars={
                "fil_rad": xr.DataArray(
                    np.random.rand(n_times).astype("float32"),  # Wrong dtype
                    dims=["time"],
                    attrs={
                        "long_name": "Wrong Name",  # Wrong value
                        # Missing "units" attribute
                        "valid_range": [0, 1000],
                        "extra_attr": "should not be here",  # Extra attribute
                    },
                ),
                # Missing "q_flag" variable
            },
            coords={
                "time": xr.DataArray(times, dims=["time"], attrs={"long_name": "Time of sample collection"}),
                # Missing "lat" and "lon" coordinates
            },
            attrs={
                "ProductID": "WRONG-ID",  # Wrong value
                "algorithm_version": "9.9.9",  # Wrong value
                # Missing required attributes
                "ExtraGlobalAttr": "unexpected",  # Extra attribute
            },
        )

        with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
            definition.check_dataset_conformance(ds)  # strict=True default

        errors = definition.check_dataset_conformance(ds, strict=False)

        # Should have errors
        assert errors
        assert len(errors) > 0

        # Check for specific error messages
        # Global attribute errors
        assert any(
            "PRODUCT: attribute value mismatch - Expected ProductID=RAD-4CH but got WRONG-ID" in e for e in errors
        )
        assert any(
            "PRODUCT: attribute value mismatch - Expected algorithm_version=0.0.1 but got 9.9.9" in e for e in errors
        )
        assert any("PRODUCT: extra attribute - Unexpected attribute 'ExtraGlobalAttr' found" in e for e in errors)
        assert any("PRODUCT: missing attribute" in e for e in errors)

        # Variable errors
        assert any(
            "fil_rad: attribute value mismatch - Expected long_name=Filtered Radiance but got Wrong Name" in e
            for e in errors
        )
        assert any("fil_rad: missing attribute - Expected attribute 'units' not found" in e for e in errors)
        assert any("fil_rad: extra attribute - Unexpected attribute 'extra_attr' found" in e for e in errors)
        assert any("fil_rad: dtype mismatch - Expected float64 but got float32" in e for e in errors)

        # Missing variables/coordinates
        assert any("q_flag: missing variable" in e for e in errors)
        assert any("lat: missing coordinate" in e for e in errors)
        assert any("lon: missing coordinate" in e for e in errors)


class TestVariableCreateMethods:
    """Tests for Variable class methods"""

    def test_create_conforming_data_array(self, test_product_definition):
        """Test creating a valid DataArray from numpy array"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        var_def = definition.variables["fil_rad"]

        # Create test data
        n_times = 20
        data = np.random.rand(n_times)

        # Create the DataArray with no additional attributes
        da = var_def.create_conforming_data_array(data, "fil_rad", {})

        # Check it's valid
        errors = var_def.check_data_array_conformance(da, "fil_rad")
        assert not errors
        assert len(errors) == 0

        # Check dimensions and attributes
        assert list(da.dims) == ["time"]
        assert da.shape == (n_times,)
        assert da.attrs["long_name"] == "Filtered Radiance"
        assert da.attrs["units"] == "W/(m^2*sr*nm)"
        assert da.attrs["valid_range"] == [0, 1000]

    def test_create_variable_data_array_with_user_attributes(self, test_product_definition):
        """Test creating a DataArray with dynamic user attributes"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        var_def = definition.variables["fil_rad"]

        # Create test data
        n_times = 10
        data = np.random.rand(n_times)

        # Override an existing attribute (this is allowed)
        user_attrs = {"long_name": "Custom Filtered Radiance"}

        # Create the DataArray
        da = var_def.create_conforming_data_array(data, "fil_rad", user_attrs)

        # Check the user attribute overrode the default
        assert da.attrs["long_name"] == "Custom Filtered Radiance"
        # Other attributes should still be present
        assert da.attrs["units"] == "W/(m^2*sr*nm)"

    def test_create_variable_data_array_wrong_shape(self, test_product_definition):
        """Test that create_variable_data_array fails with wrong shaped data"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        var_def = definition.variables["fil_rad"]

        # Create test data with wrong shape (2D instead of 1D)
        data = np.random.rand(10, 5)

        # This should raise a ValueError because xarray validates dimensions vs shape
        with pytest.raises(ValueError, match="different number of dimensions"):
            var_def.create_conforming_data_array(data, "fil_rad", {})


class TestLiberaDataProductDefinitionCreateMethods:
    """Tests for LiberaDataProductDefinition.create_conforming_dataset method"""

    def test_create_conforming_dataset(self, test_product_definition):
        """Test creating a valid Dataset from numpy arrays"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create test data
        n_times = 15
        time_data = np.arange(n_times).astype("datetime64[ns]")
        lat_data = np.linspace(-90, 90, n_times)
        lon_data = np.linspace(-180, 180, n_times)
        fil_rad_data = np.random.rand(n_times)
        q_flag_data = np.random.randint(0, 100, n_times, dtype=np.int32)

        data = {
            "time": time_data,
            "lat": lat_data,
            "lon": lon_data,
            "fil_rad": fil_rad_data,
            "q_flag": q_flag_data,
        }

        # No additional global or variable attributes needed (all are static in the definition)
        user_global_attrs = {}
        user_var_attrs = {
            "time": {},
            "lat": {},
            "lon": {},
            "fil_rad": {},
            "q_flag": {},
        }

        # Create the dataset
        ds, _ds_errors = definition.create_conforming_dataset(data, user_global_attrs, user_var_attrs)
        assert not _ds_errors

        # Check it's valid
        errors = definition.check_dataset_conformance(ds)
        assert not errors
        assert len(errors) == 0

        # Check structure
        assert "time" in ds.coords
        assert "lat" in ds.coords
        assert "lon" in ds.coords
        assert "fil_rad" in ds.data_vars
        assert "q_flag" in ds.data_vars

        # Check global attributes
        assert ds.attrs["ProductID"] == "RAD-4CH"
        assert ds.attrs["algorithm_version"] == "0.0.1"
        assert ds.attrs["Format"] == "NetCDF-4"
        assert ds.attrs["Conventions"] == "CF-1.8"

        # Check variable attributes
        assert ds["fil_rad"].attrs["long_name"] == "Filtered Radiance"
        assert ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"
        assert ds["q_flag"].attrs["long_name"] == "Quality Flags"

        # Check data types
        assert str(ds["time"].dtype) == "datetime64[ns]"
        assert str(ds["lat"].dtype) == "float64"
        assert str(ds["lon"].dtype) == "float64"
        assert str(ds["fil_rad"].dtype) == "float64"
        assert str(ds["q_flag"].dtype) == "int32"

    def test_create_conforming_dataset_missing_data(self, test_product_definition):
        """Test that create_conforming_dataset fails gracefully with missing data"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create test data missing some required variables
        n_times = 10
        time_data = np.arange(n_times).astype("datetime64[ns]")
        lat_data = np.linspace(-90, 90, n_times)
        # Missing lon, fil_rad, and q_flag

        data = {
            "time": time_data,
            "lat": lat_data,
        }

        user_global_attrs = {}
        # Provide attributes for only the variables we have data for
        incomplete_var_attrs = {
            "time": {},
            "lat": {},
        }

        with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
            definition.create_conforming_dataset(data, user_global_attrs, incomplete_var_attrs)

        # This should raise an error for missing required data
        ds, ds_errors = definition.create_conforming_dataset(
            data, user_global_attrs, incomplete_var_attrs, strict=False
        )
        assert "time" in ds
        assert "lat" in ds
        assert ds_errors  # Should have errors for missing variables
        print(ds_errors)

    def test_create_conforming_dataset_unknown_variable(self, test_product_definition):
        """Test that create_conforming_dataset fails for unknown variables"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create test data with an unknown variable
        n_times = 10
        data = {
            "time": np.arange(n_times).astype("datetime64[ns]"),
            "unknown_var": np.random.rand(n_times),  # Not in definition
        }

        user_global_attrs = {}
        user_var_attrs = {
            "time": {},
            "unknown_var": {},
        }

        # This should raise an error for unknown variable
        with pytest.raises(ValueError, match="Unknown variable/coordinate"):
            definition.create_conforming_dataset(data, user_global_attrs, user_var_attrs)

    def test_create_conforming_dataset_with_user_attributes(self, test_product_definition):
        """Test creating a Dataset with user-provided dynamic attributes"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create test data
        n_times = 8
        time_data = np.arange(n_times).astype("datetime64[ns]")
        lat_data = np.linspace(-90, 90, n_times)
        lon_data = np.linspace(-180, 180, n_times)
        fil_rad_data = np.random.rand(n_times)
        q_flag_data = np.random.randint(0, 50, n_times, dtype=np.int32)

        data = {
            "time": time_data,
            "lat": lat_data,
            "lon": lon_data,
            "fil_rad": fil_rad_data,
            "q_flag": q_flag_data,
        }

        # Override some attributes (overriding is allowed, adding new ones will fail validation)
        user_global_attrs = {}  # Can't add new global attrs that aren't in definition
        user_var_attrs = {
            "time": {"long_name": "Custom Time"},  # Override existing
            "lat": {},
            "lon": {},
            "fil_rad": {"long_name": "Custom Radiance"},  # Override existing
            "q_flag": {},
        }

        # Create the dataset
        ds, ds_errors = definition.create_conforming_dataset(data, user_global_attrs, user_var_attrs, strict=False)
        assert ds_errors  # Should have errors because we put incorrect values into static attribute metadata
        assert any(["attribute value mismatch" for err in ds_errors])

        # Check user attributes overrode the defaults
        assert ds["time"].attrs["long_name"] == "Custom Time"
        assert ds["fil_rad"].attrs["long_name"] == "Custom Radiance"
        # Other attributes should still be present
        assert ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"
