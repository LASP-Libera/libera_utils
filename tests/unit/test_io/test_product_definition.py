"""Tests for data product definition YAML parsing and validation."""

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.product_definition import LiberaDataProductDefinition, LiberaVariableDefinition


@pytest.mark.filterwarnings("error")
class TestLiberaDataProductDefinition:
    """Basic tests for the LiberaDataProductDefinition class."""

    def test_valid_definition(self, test_product_definition):
        """Test loading a valid product definition."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        assert definition is not None
        assert "radiometer_time" in definition.coordinates
        assert "fil_rad" in definition.variables
        assert isinstance(definition.variables["fil_rad"], LiberaVariableDefinition)

        # Verify that the class attributes are frozen
        with pytest.raises(ValidationError, match="Instance is frozen"):
            definition.attributes = {"should": "fail"}

    def test_undefined_dimension_exception(self, test_product_definition):
        """Test that an exception is raised for undefined dimension in the product definition."""
        # Modify the test product definition to include an undefined dimension
        with test_product_definition.open("r") as f:
            definition_contents = yaml.safe_load(f)
        definition_contents["coordinates"]["radiometer_time"]["dimensions"] = ["dne_dimension"]

        with pytest.raises(ValueError, match="Undefined dimension name 'dne_dimension' used in product definition"):
            _definition = LiberaDataProductDefinition(**definition_contents)

    def test_generate_filename(self, test_product_definition, test_dataset):
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        fn = definition.generate_data_product_filename(test_dataset, time_variable="radiometer_time")
        assert isinstance(fn, LiberaDataProductFilename)
        print(fn.path.name)
        assert fn.filename_parts.product_name == "RAD-4CH"
        assert fn.filename_parts.version == "V0-0-1"
        assert fn.filename_parts.utc_start == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        assert fn.filename_parts.utc_end == datetime(2024, 1, 1, 23, 59, 59, tzinfo=UTC)


@pytest.mark.filterwarnings("error")
class TestLiberaDataProductDefinitionConformanceChecking:
    """Tests for the check_dataset_conformance method of LiberaDataProductDefinition"""

    def test_check_dataset_conformance_valid(self, tmp_path, test_product_definition, test_dataset):
        """Test validating a dataset against a product definition object"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        errors = definition.check_dataset_conformance(test_dataset)
        assert not errors
        assert len(errors) == 0
        outpath = tmp_path / "test_dataset.nc"
        test_dataset.to_netcdf(outpath, engine="h5netcdf")
        read_dataset = xr.open_dataset(outpath, engine="h5netcdf")
        print(read_dataset)
        errors = definition.check_dataset_conformance(read_dataset, strict=True)
        assert not errors
        assert len(errors) == 0

    def test_check_dataset_conformance_undefined_dimension(self, test_product_definition, test_dataset):
        """Test validating a dataset with an undefined dimension against a product definition object"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Reference an unknown dimension
        test_dataset["radiometer_time"] = test_dataset["radiometer_time"].swap_dims(
            {"radiometer_time": "dne_dimension"}
        )

        # strict=false: Warning is issued in realtime
        with pytest.warns(
            UserWarning,
            match=r"radiometer_time: dimension mismatch - Expected dimensions \['radiometer_time'\] but got \['dne_dimension'\].",
        ):
            errors = definition.check_dataset_conformance(test_dataset, strict=False)

        # Errors are returned in the list of errors
        assert errors
        print(errors)
        assert any(
            "radiometer_time: dimension mismatch - Expected dimensions ['radiometer_time'] but got ['dne_dimension']"
            in e
            for e in errors
        )

        # strict=True: Exception raised for undefined dimension
        with pytest.warns(
            UserWarning,
            match=r"radiometer_time: dimension mismatch - Expected dimensions \['radiometer_time'\] but got \['dne_dimension'\].",
        ):
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(test_dataset, strict=True)

    def test_check_dataset_conformance_incorrect_size_dimension(self, test_product_definition, test_dataset):
        """Test validating a dataset with an incorrect dimension size against a product definition object"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)
        # Remove one of the columns of the cartesian_position variable to create a dimension size mismatch
        test_dataset = test_dataset.drop_vars("cartesian_position")
        n_times = test_dataset.sizes["radiometer_time"]
        test_dataset["cartesian_position"] = xr.DataArray(
            np.zeros((n_times, 2), dtype=np.float32),
            dims=["radiometer_time", "euclidean_dim"],
            attrs={"long_name": "Cartesian SC position", "units": "km", "valid_range": [-100000, 100000]},
        )
        test_dataset["cartesian_position"].encoding = {"zlib": True, "complevel": 4}

        with pytest.warns(
            UserWarning,
            match="cartesian_position: dimension size mismatch for dimension 'euclidean_dim' - Expected size 3 but got 2",
        ):
            errors = definition.check_dataset_conformance(test_dataset, strict=False)
        assert errors
        assert any(
            "cartesian_position: dimension size mismatch for dimension 'euclidean_dim' - Expected size 3 but got 2" in e
            for e in errors
        )

        with pytest.warns(
            UserWarning,
            match="cartesian_position: dimension size mismatch for dimension 'euclidean_dim' - Expected size 3 but got 2",
        ):
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(test_dataset, strict=True)

    def test_check_dataset_conformance_invalid_version(self, test_product_definition, test_dataset):
        """Test that check_dataset_conformance catches invalid algorithm_version format"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Set an invalid algorithm_version
        test_dataset.attrs["algorithm_version"] = "version_one"

        with pytest.warns(UserWarning, match="algorithm_version"):
            errors = definition.check_dataset_conformance(test_dataset, strict=False)

        assert errors
        assert any("algorithm_version: invalid format - Expected semantic versioning" in e for e in errors)
        assert any(
            "attribute value mismatch - Expected algorithm_version=0.0.1 but got version_one" in e for e in errors
        )

        with pytest.warns(UserWarning, match="algorithm_version"):
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(test_dataset)

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
                    dims=["radiometer_time"],
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
                "radiometer_time": xr.DataArray(
                    times, dims=["radiometer_time"], attrs={"long_name": "Time of sample collection"}
                ),
                # Missing "lat" and "lon" coordinates
            },
            attrs={
                "ProductID": "WRONG-ID",  # Wrong value
                "algorithm_version": "9.9.9",  # Wrong value
                # Missing required attributes
                "ExtraGlobalAttr": "unexpected",  # Extra attribute
            },
        )
        ds["radiometer_time"].encoding = {
            "units": "nanoseconds since 1958-01-01",
            "calendar": "standard",
            "dtype": "int64",
            "zlib": True,
            "complevel": 4,
        }
        standard_encoding = {"zlib": True, "complevel": 4}
        ds["fil_rad"].encoding = standard_encoding

        with pytest.warns(UserWarning, match=r"(missing|extra|mismatch)"):
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(ds)  # strict=True default

        with pytest.warns(UserWarning, match=r"(missing|extra|mismatch)"):
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

    def test_check_dataset_conformance_dimension_shape_mismatch(self, test_product_definition, test_dataset):
        """Test that check_dataset_conformance catches the wrong number of dimensions."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create dataset with wrong dimensions
        n_times = len(test_dataset["radiometer_time"])
        n_lat = len(test_dataset["lat"])
        test_dataset["fil_rad"] = xr.DataArray(
            np.random.rand(n_times, n_lat),  # Wrong: should be 1D
            dims=["radiometer_time", "lat"],  # These are both defined dimensions but this is the wrong shape
            attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
        )
        test_dataset["fil_rad"].encoding = {"zlib": True, "complevel": 4}

        # Fix the dataset to the extent possible
        with pytest.warns(
            UserWarning,
            match=r"fil_rad: dimension mismatch",
        ):
            errors = definition.check_dataset_conformance(test_dataset, strict=False)

        assert errors
        assert any(["fil_rad: dimension mismatch" in e for e in errors])

        with pytest.warns(
            UserWarning,
            match=r"fil_rad: dimension mismatch",
        ):
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(test_dataset, strict=True)


@pytest.mark.filterwarnings("error")
class TestLiberaDataProductDefinitionConformanceEnforcement:
    """Tests for the enforce_dataset_conformance method of LiberaDataProductDefinition"""

    def test_enforce_dataset_conformance_missing_attributes(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance adds missing attributes."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Remove some global attributes
        del test_dataset.attrs["Format"]
        del test_dataset.attrs["Conventions"]

        # Remove some variable attributes
        del test_dataset["fil_rad"].attrs["units"]
        del test_dataset["radiometer_time"].attrs["long_name"]

        # Fix the dataset
        fixed_ds = definition.enforce_dataset_conformance(test_dataset)

        # Check global attributes were added back
        assert fixed_ds.attrs["Format"] == "NetCDF-4"
        assert fixed_ds.attrs["Conventions"] == "CF-1.8"

        # Check variable attributes were added back
        assert fixed_ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"
        assert fixed_ds["radiometer_time"].attrs["long_name"] == "Time of sample collection"

        # Should be valid after fixing (no errors)
        errors = definition.check_dataset_conformance(fixed_ds, strict=True)
        assert not errors

    def test_enforce_dataset_conformance_extra_attributes(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance removes extra attributes."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Add extra global attributes
        test_dataset.attrs["extra_global"] = "should be removed"
        test_dataset.attrs["another_extra"] = 123

        # Add extra variable attributes
        test_dataset["fil_rad"].attrs["extra_var_attr"] = "remove me"
        test_dataset["radiometer_time"].attrs["unexpected"] = 42

        # Fix the dataset
        with pytest.warns(UserWarning, match="not defined in the product definition") as warning_list:
            fixed_ds = definition.enforce_dataset_conformance(test_dataset)

        warning_messages = [str(w.message) for w in warning_list]
        assert any(
            [
                "Dataset has unexpected attribute 'extra_global' with value 'should be removed'" in m
                for m in warning_messages
            ]
        )
        assert any(["Dataset has unexpected attribute 'another_extra' with value '123'" in m for m in warning_messages])
        assert any(
            [
                "Variable radiometer_time has unexpected extra attribute 'unexpected' with value '42'" in m
                for m in warning_messages
            ]
        )
        assert any(
            [
                "Variable fil_rad has unexpected extra attribute 'extra_var_attr' with value 'remove me'" in m
                for m in warning_messages
            ]
        )

        # Check extra global attributes were removed
        assert "extra_global" not in fixed_ds.attrs
        assert "another_extra" not in fixed_ds.attrs

        # Check extra variable attributes were removed
        assert "extra_var_attr" not in fixed_ds["fil_rad"].attrs
        assert "unexpected" not in fixed_ds["radiometer_time"].attrs

        # Should be valid after fixing (no errors)
        errors = definition.check_dataset_conformance(fixed_ds, strict=True)
        assert not errors

    def test_enforce_dataset_conformance_wrong_attribute_values(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance corrects wrong attribute values."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Set wrong values for some attributes
        test_dataset.attrs["ProductID"] = "WRONG-ID"
        test_dataset.attrs["algorithm_version"] = "999.0.0"
        test_dataset["fil_rad"].attrs["units"] = "wrong_units"

        # Fix the dataset
        with pytest.warns(UserWarning, match="attribute value mismatch") as warning_list:
            fixed_ds = definition.enforce_dataset_conformance(test_dataset)

        # Check that expected warnings were issued
        warning_messages = [str(w.message) for w in warning_list]
        print(warning_messages)
        assert any(["'ProductID': Expected 'RAD-4CH' but got 'WRONG-ID'" in m for m in warning_messages])
        assert any(["'algorithm_version': Expected '0.0.1' but got '999.0.0'" in m for m in warning_messages])
        assert any(["'units': Expected 'W/(m^2*sr*nm)' but got 'wrong_units'" in m for m in warning_messages])

        # Check values were corrected
        assert fixed_ds.attrs["ProductID"] == "RAD-4CH"
        assert fixed_ds.attrs["algorithm_version"] == "0.0.1"
        assert fixed_ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"

        # Should be valid after fixing (no errors)
        errors = definition.check_dataset_conformance(fixed_ds, strict=True)
        assert not errors

    def test_enforce_dataset_conformance_dtype_conversion_exception(self, test_product_definition):
        """Test that enforce_dataset_conformance refuses to convert unsafe dtypes and raises an exception."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create a dataset with wrong dtypes
        n_times = 10

        ds = xr.Dataset(
            data_vars={
                "q_flag": xr.DataArray(
                    np.random.randint(100, size=n_times, dtype=np.int64),  # Should be int32 raises exception
                ),
            },
        )

        # Enforcement fails because cannot safely convert int64 to int32 without potential data loss.
        with pytest.raises(
            ValueError, match="q_flag has dtype 'int64' that cannot be safely converted to expected dtype 'int32'"
        ):
            definition.enforce_dataset_conformance(ds)

    def test_enforce_dataset_conformance_missing_variable_exception(self, test_product_definition):
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
                    dims=["radiometer_time"],
                    attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
                ),
            },
            coords={
                "radiometer_time": xr.DataArray(
                    times, dims=["radiometer_time"], attrs={"long_name": "Time of sample collection"}
                ),
                # Missing "lat" coordinate and cartesian_position
                "lon": xr.DataArray(
                    np.linspace(-180, 180, n_times),
                    dims=["radiometer_time"],
                    attrs={"long_name": "Geolocation longitude", "units": "degrees", "valid_range": [-180, 180]},
                ),
            },
        )

        # Fix the dataset
        fixed_ds = definition.enforce_dataset_conformance(ds)

        # Dataset should be modified but have errors due to missing variables
        with pytest.warns(UserWarning, match=r"missing (variable|coordinate)") as warning_list:
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(fixed_ds, strict=True)

        # Check that warnings were issued for missing variables
        warning_messages = [str(w.message) for w in warning_list]
        assert any(["Expected coordinate 'lat' not found" in m for m in warning_messages])
        assert any(["Expected variable 'q_flag' not found" in m for m in warning_messages])
        assert any(["Expected variable 'cartesian_position' not found" in m for m in warning_messages])

        # Existing variables should still be fixed
        assert fixed_ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"

    def test_enforce_dataset_conformance_encoding_updates_success(self, test_product_definition, test_dataset):
        """Test that enforce_dataset_conformance updates encoding correctly."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Clear and modify some encoding settings
        test_dataset["radiometer_time"].encoding = {}  # Clear all encoding, should be silently updated
        test_dataset["fil_rad"].encoding = {"zlib": False}  # Wrong value

        # Fix the dataset
        with pytest.warns(UserWarning, match=r"fil_rad has encoding setting 'zlib' with value 'False' that conflicts"):
            fixed_ds = definition.enforce_dataset_conformance(test_dataset)

        # Check encoding was updated for time coordinate
        assert fixed_ds["radiometer_time"].encoding["units"] == "nanoseconds since 1958-01-01"
        assert fixed_ds["radiometer_time"].encoding["calendar"] == "standard"
        assert fixed_ds["radiometer_time"].encoding["dtype"] == "int64"
        assert fixed_ds["radiometer_time"].encoding["zlib"] is True
        assert fixed_ds["radiometer_time"].encoding["complevel"] == 4

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
        errors = definition.check_dataset_conformance(fixed_ds, strict=True)
        assert not errors

    def test_enforce_dataset_conformance_undefined_dimension_exception(self, test_product_definition):
        """Test that enforce_dataset_conformance raises an exception for undefined dimension."""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create dataset with wrong dimensions
        n_times = 10
        n_extra = 5

        ds = xr.Dataset(
            data_vars={
                "fil_rad": xr.DataArray(
                    np.random.rand(n_times, n_extra),  # Wrong: should be 1D
                    dims=["radiometer_time", "extra_dim"],
                    attrs={"long_name": "Filtered Radiance", "units": "W/(m^2*sr*nm)", "valid_range": [0, 1000]},
                )
            }
        )

        # Fix the dataset to the extent possible
        with pytest.raises(ValueError, match=r"Variable fil_rad has undefined dimensions \['extra_dim'\]"):
            definition.enforce_dataset_conformance(ds)


@pytest.mark.filterwarnings("error")
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
        da = var_def.create_variable_data_array(data, "fil_rad", {})

        # Check it's valid
        errors = var_def.check_data_array_conformance(da, "fil_rad")
        assert not errors
        assert len(errors) == 0

        # Check dimensions and attributes
        assert list(da.dims) == ["radiometer_time"]
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
        da = var_def.create_variable_data_array(data, "fil_rad", user_attrs)

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
            var_def.create_variable_data_array(data, "fil_rad", {})


@pytest.mark.filterwarnings("error")
class TestLiberaDataProductDefinitionCreateMethods:
    """Tests for LiberaDataProductDefinition.create_conforming_dataset method"""

    def test_create_conforming_dataset(self, test_product_definition, test_data_dict):
        """Test creating a valid Dataset from numpy arrays"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # No additional global or variable attributes needed (all are static in the definition)
        user_global_attrs = {}
        user_var_attrs = {
            "radiometer_time": {},
            "lat": {},
            "lon": {},
            "fil_rad": {},
            "q_flag": {},
            "cartesian_position": {},
        }

        # Create the dataset
        ds = definition.create_product_dataset(test_data_dict, user_global_attrs, user_var_attrs)

        # Check it's valid
        errors = definition.check_dataset_conformance(ds)
        assert not errors
        assert len(errors) == 0

        # Check structure
        assert "radiometer_time" in ds.coords
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
        assert str(ds["radiometer_time"].dtype) == "datetime64[ns]"
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
            "radiometer_time": time_data,
            "lat": lat_data,
        }

        user_global_attrs = {}
        # Provide attributes for only the variables we have data for
        incomplete_var_attrs = {
            "radiometer_time": {},
            "lat": {},
        }

        ds = definition.create_product_dataset(data, user_global_attrs, incomplete_var_attrs)

        assert "radiometer_time" in ds
        assert "lat" in ds

        with pytest.warns(UserWarning, match=r"missing (variable|coordinate)") as warning_list:
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(ds, strict=True)

        warning_messages = [str(w.message) for w in warning_list]
        assert any(["Expected coordinate 'lon' not found" in m for m in warning_messages])
        assert any(["Expected variable 'fil_rad' not found" in m for m in warning_messages])
        assert any(["Expected variable 'q_flag' not found" in m for m in warning_messages])
        assert any(["Expected variable 'cartesian_position' not found" in m for m in warning_messages])

    def test_create_conforming_dataset_unknown_variable(self, test_product_definition):
        """Test that create_conforming_dataset fails for unknown variables"""
        definition = LiberaDataProductDefinition.from_yaml(test_product_definition)

        # Create test data with an unknown variable
        n_times = 10
        data = {
            "radiometer_time": np.arange(n_times).astype("datetime64[ns]"),
            "unknown_var": np.random.rand(n_times),  # Not in definition
        }

        user_global_attrs = {}
        user_var_attrs = {
            "radiometer_time": {},
            "unknown_var": {},
        }

        # This should raise an error for unknown variable
        with pytest.raises(ValueError, match="Unknown variable/coordinate"):
            definition.create_product_dataset(data, user_global_attrs, user_var_attrs)

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
            "radiometer_time": time_data,
            "lat": lat_data,
            "lon": lon_data,
            "fil_rad": fil_rad_data,
            "q_flag": q_flag_data,
        }

        # Override some attributes (setting dynamic attributes is allowed, adding new ones or overriding static
        # attributes will fail validation)
        user_global_attrs = {}  # Can't add new global attrs that aren't in definition
        user_var_attrs = {
            "radiometer_time": {"long_name": "Custom Time"},  # Override existing
            "lat": {},
            "lon": {},
            "fil_rad": {"long_name": "Custom Radiance"},  # Override existing
            "q_flag": {},
        }

        # Create the dataset. This dataset is invalid because attributes don't match the product definition!
        ds = definition.create_product_dataset(data, user_global_attrs, user_var_attrs)

        with pytest.warns(UserWarning, match=r"(attribute value mismatch|missing variable)") as warning_list:
            with pytest.raises(ValueError, match="Errors detected during dataset conformance check"):
                definition.check_dataset_conformance(ds, strict=True)

        warning_messages = [str(w.message) for w in warning_list]
        assert any(["radiometer_time: attribute value mismatch" in m for m in warning_messages])
        assert any(["fil_rad: attribute value mismatch" in m for m in warning_messages])
        assert any(["cartesian_position: missing variable" in m for m in warning_messages])

        # Check user attributes overrode the defaults
        assert ds["radiometer_time"].attrs["long_name"] == "Custom Time"
        assert ds["fil_rad"].attrs["long_name"] == "Custom Radiance"
        # Other attributes should still be present
        assert ds["fil_rad"].attrs["units"] == "W/(m^2*sr*nm)"
