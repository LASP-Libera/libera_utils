# Standard
import datetime as dt
import os
from pathlib import Path

import numpy as np

# Installed
import pytest
import xarray as xr
import yaml
from cloudpathlib import AnyPath
from pydantic import ValidationError
from xarray import DataArray

from libera_utils.constants import DataProductIdentifier

# Local
from libera_utils.io.netcdf import DataProductConfig, LiberaDimension, LiberaVariable, VariableMetadata


# ---- LiberaDimension Tests ----#
def test_get_available_dimensions():
    """Test that the available dimensions are LiberaDimensions"""
    dimension_dict = LiberaDimension.get_available_dimensions_dict()
    assert len(dimension_dict) == 3
    assert all(isinstance(v, LiberaDimension) for v in dimension_dict.values())
    assert dimension_dict["n_samples"].size == -1
    assert dimension_dict["n_samples"].is_dynamic_size is True
    assert dimension_dict["camera_pixel_count_x"].size == 2048
    assert dimension_dict["camera_pixel_count_x"].is_dynamic_size is False
    assert dimension_dict["camera_pixel_count_x"].is_set is True


@pytest.mark.parametrize(
    ("dimension_name", "input_size", "output_size", "output_is_dynamic", "output_is_set"),
    [
        ("dynamic_test", -1, -1, True, False),
        ("static_test", 2048, 2048, False, True),
    ],
)
def test_model_validation(dimension_name, input_size, output_size, output_is_dynamic, output_is_set):
    """Test that a LiberaDimension can be made from a definition"""
    dimension = LiberaDimension(name=dimension_name, long_name="A test description", size=input_size)

    assert dimension.name == dimension_name
    assert dimension.size == output_size
    assert dimension.is_dynamic_size == output_is_dynamic
    assert dimension.is_set == output_is_set


# ---- VariableMetadata Tests ----#
def test_available_dimensions():
    """Test that the class variable holding the templated dimensions is as expected"""
    assert VariableMetadata.available_dimensions is not None
    assert len(VariableMetadata.available_dimensions) == 3


def test_set_dimensions_validation_successes(test_variable_definitions):
    """Test that dimensions are set correctly from validation"""
    with test_variable_definitions.open("r", encoding="utf-8") as f:
        variable_definitions = yaml.safe_load(f)
        test_variable_definition = variable_definitions["pix_rad"]
        assert test_variable_definition["dimensions"] == ["n_samples", "camera_pixel_count_x", "camera_pixel_count_y"]
        metadata = VariableMetadata(**test_variable_definition)
        assert isinstance(metadata.dimensions, dict)
        assert len(metadata.dimensions) == 3
        assert isinstance(metadata.dimensions["n_samples"], LiberaDimension)

        # Make a list of LiberaDimensions and pass that in
        str_list = test_variable_definition["dimensions"]
        object_list = []
        for dimension in str_list:
            object_list.append(VariableMetadata.available_dimensions[dimension])
        test_variable_definition["dimensions"] = object_list
        good_metadata = VariableMetadata(**test_variable_definition)
        assert len(good_metadata.dimensions) == 3


@pytest.mark.parametrize(
    ("bad_dimensions", "error_type", "error_message"),
    [
        (["bad_entry", "camera_pixel_count_x", "camera_pixel_count_y"], ValueError, "The specified dimension*"),
        ("not a list", TypeError, "Dimensions field of a variable metadata must be a list"),
        ([1], TypeError, "Items in the dimension list must be of type str or LiberaDimension"),
    ],
)
def test_set_dimensions_validation_failures(test_variable_definitions, bad_dimensions, error_type, error_message):
    """Test that dimensions are set correctly from validation"""
    with test_variable_definitions.open("r", encoding="utf-8") as f:
        variable_definitions = yaml.safe_load(f)
        test_variable_definition = variable_definitions["pix_rad"]

        # Test for dimension not in list
        test_variable_definition["dimensions"] = bad_dimensions
        with pytest.raises(error_type, match=error_message):
            VariableMetadata(**test_variable_definition)


# ---- LiberaDimension Tests ----#
@pytest.fixture
def variable_metadata_for_testing(test_variable_definitions):
    """Making a LiberaVariable from a yaml file definition for testing"""
    with test_variable_definitions.open("r", encoding="utf-8") as f:
        variable_definitions = yaml.safe_load(f)
        test_variable_definition = variable_definitions["pix_rad"]
        variable_metadata = VariableMetadata(**test_variable_definition)
        # Change the y-axis pixel count of the image dimensions for testing
        test_metadata = variable_metadata.model_copy(deep=True)
        test_metadata.dimensions["camera_pixel_count_y"].size = 1024
        return test_metadata


def test_check_data_dimensions(variable_metadata_for_testing):
    """Test that the dimensions check is working"""
    test_variable = LiberaVariable(name="test_variable", metadata=variable_metadata_for_testing)

    # Wrong number of dimensions
    wrong_dims_data = np.full((10, 2048), 11)
    with pytest.raises(ValueError, match=r"The provided data has 2 dimensions*"):
        test_variable._check_for_bad_dimensions(wrong_dims_data)

    # Wrong order of dimensions
    bad_order_dims_data = np.full((10, 1024, 2048), 11)
    with pytest.raises(ValueError, match=r"There is a mismatch in data shape*"):
        test_variable._check_for_bad_dimensions(bad_order_dims_data)

    # Good data
    good_data = np.full((10, 2048, 1024), 11)
    test_variable._check_for_bad_dimensions(good_data)


def test_set_dynamic_dimension(caplog, variable_metadata_for_testing):
    """Test that a dynamic dimension can be set correctly"""
    first_dimension = variable_metadata_for_testing.dimensions["n_samples"]

    # Set the dimension in the metadata
    with caplog.at_level("WARNING"):
        variable_metadata_for_testing.set_dynamic_dimension(first_dimension, 0)

    with caplog.at_level("WARNING"):
        variable_metadata_for_testing.set_dynamic_dimension(first_dimension, 10)

    variable_metadata_for_testing.dimensions["n_samples"].size = -1
    variable_metadata_for_testing.dimensions["n_samples"].is_set = False
    first_dimension = variable_metadata_for_testing.dimensions["n_samples"]
    # Now set the dimension to a dynamic value
    with pytest.raises(ValueError, match=r"Cannot set the"):
        variable_metadata_for_testing.set_dynamic_dimension(first_dimension, -5)


def test_set_dynamic_data_dimension(variable_metadata_for_testing):
    """Test that dimensions of a supplied data file are checked against the variables metadata"""
    test_data = np.full((10, 2048, 1024), 21)
    test_variable = LiberaVariable(name="test_variable", metadata=variable_metadata_for_testing)

    assert test_variable.metadata.dimensions["n_samples"].size == -1
    test_variable._set_all_dynamic_dimension_lengths(test_data)
    assert test_variable.metadata.dimensions["n_samples"].size == test_data.shape[0]

    # Use the same data in a different class against expected dimension
    test_data_2 = DataArray(test_data)
    assert test_variable.metadata.dimensions["n_samples"].size == test_data.shape[0]
    test_variable._check_for_bad_dimensions(test_data_2)

    # Test bad dimension length in the newly set dynamic
    bad_data = np.full((21, 2048, 1024), 21)
    with pytest.raises(ValueError, match=r"There is a mismatch in data shape*"):
        test_variable._check_for_bad_dimensions(bad_data)


def test_set_data_dimension_match_from_ndarray(variable_metadata_for_testing):
    """Testing that dimensions get set correctly in the DataArray object with a ndarray input"""
    test_data = np.full((10, 2048, 1024), 21)
    test_variable = LiberaVariable(name="test_variable", metadata=variable_metadata_for_testing)

    # Before calling the dimension match function
    assert isinstance(test_data, np.ndarray)

    test_variable._set_data_with_dimensions_to_match_metadata(test_data)
    assert isinstance(test_variable.data, xr.DataArray)
    metadata_dimensions = test_variable.metadata.dimensions_name_list
    data_dimensions = test_variable.data.dims
    assert np.all([data_dimension in metadata_dimensions for data_dimension in data_dimensions])


@pytest.mark.parametrize(
    ("data_shape", "data_value", "input_dims"),
    [
        ((10, 2048, 1024), 12, ("test_dim1", "test_dim2", "test_dim2")),
        ((10, 2048, 1024), 34, ("n_samples", "test_dim1", "test_dim2")),
        ((10, 2048, 1024), 56, ("n_samples", "camera_pixel_count_x", "camera_pixel_count_y")),
    ],
)
def test_set_data_with_dimension_match_from_dataarray(
    variable_metadata_for_testing, data_shape, data_value, input_dims
):
    """Testing that dimensions get set correctly in the DataArray object with a DataArray input"""
    test_data = xr.DataArray(np.full(data_shape, data_value))
    test_variable = LiberaVariable(name="test_variable", metadata=variable_metadata_for_testing)

    # Before calling the dimension match function ensure data is not set
    assert test_variable.data is None

    test_variable._set_data_with_dimensions_to_match_metadata(test_data)
    assert isinstance(test_variable.data, xr.DataArray)
    metadata_dimensions = test_variable.metadata.dimensions_name_list
    data_dimensions = test_variable.data.dims
    assert np.all([data_dimension in metadata_dimensions for data_dimension in data_dimensions])


def test_libera_variable_set_data(variable_metadata_for_testing):
    """Test that data can be added to a LiberaVariable that is defined in a yaml file"""
    test_variable = LiberaVariable(name="test_variable", metadata=variable_metadata_for_testing)

    with pytest.raises(TypeError):
        test_variable.set_data("not an array")

    correct_array = np.full((10, 2048, 1024), 5)
    test_variable.set_data(correct_array)
    assert test_variable.data is not None
    assert isinstance(test_variable.data, xr.DataArray)
    # Check for dimension matching
    metadata_dimensions = test_variable.metadata.dimensions_name_list
    data_dimensions = test_variable.data.dims
    assert np.all([data_dimension in metadata_dimensions for data_dimension in data_dimensions])


def test_create_libera_variable_with_data(variable_metadata_for_testing):
    """Test the instantiation of LiberaVariable object when a data object is given"""
    test_data = np.full((10, 2048, 1024), 123)
    test_variable = LiberaVariable(name="test_variable", metadata=variable_metadata_for_testing, data=test_data)
    assert isinstance(test_variable.data, xr.DataArray)
    assert np.all(test_variable.data == 123)
    # Check for dimension matching
    metadata_dimensions = test_variable.metadata.dimensions_name_list
    data_dimensions = test_variable.data.dims
    assert np.all([data_dimension in metadata_dimensions for data_dimension in data_dimensions])


# ---- DataProductConfig Tests ----#
def test_get_static_project_metadata():
    """Test that the Libera Project metadata is correct"""
    project_metadata = DataProductConfig.get_static_project_metadata()

    assert project_metadata.ProjectLongName == "Libera"
    assert project_metadata.ProjectShortName == "Libera"
    # TODO[LIBSDC-613]: Add this when known
    # assert empty_config.static_project_metadata.PlatformLongName == "TBD"

    assert project_metadata.PlatformShortName == "NOAA-22"
    assert project_metadata.Format == "NetCDF-4"
    assert project_metadata.Conventions == "CF-1.8"


@pytest.mark.parametrize("data_product_id", [DataProductIdentifier.l1b_rad, "RAD-4CH"])
def test_config_data_product_id_validation(data_product_id):
    """Test that a configuration object can be made with a data product identifier string or object"""
    DataProductConfig(data_product_id=data_product_id, version="1.0.0")


def test_config_version_validation():
    """Test that a configuration object can be only be made with a properly formatted version string"""
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0")
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0.0")
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="alpha")

    # The good case
    DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0")


@pytest.mark.parametrize("input_type", [str, Path])
def test_config_variable_configuration_validation(test_variable_definitions, input_type):
    """Test the type validation for the variable_configuration_path is working correctly"""
    DataProductConfig(
        data_product_id=DataProductIdentifier.l1b_rad,
        version="1.0.0",
        variable_configuration_path=input_type(test_variable_definitions),
    )


def test_load_data_product_variables_with_metadata(test_variable_definitions):
    """Test that the variables objects are made correctly from a yaml file"""
    variables_dict = DataProductConfig.load_data_product_variables_with_metadata(test_variable_definitions)
    assert len(variables_dict) == 6
    assert isinstance(variables_dict["time"], LiberaVariable)
    assert isinstance(variables_dict["pix_rad"], LiberaVariable)
    assert isinstance(variables_dict["time"].metadata, VariableMetadata)
    assert isinstance(variables_dict["pix_rad"].metadata, VariableMetadata)


def test_variable_encodings(test_variable_definitions):
    """Test that the variable encodings work as expected"""
    new_config = DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0")
    assert new_config.variable_encoding_dict is None

    new_config = DataProductConfig(
        data_product_id=DataProductIdentifier.l1b_rad,
        version="1.0.0",
        variable_configuration_path=test_variable_definitions,
    )
    assert len(new_config.variable_encoding_dict) == 0

    new_config.add_data_to_variable("time", np.full((100,), dt.datetime(2023, 1, 1)))
    assert len(new_config.variable_encoding_dict) == 1


def test_version_format():
    """Test that the version format is correct for a filename"""
    new_config = DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0")
    formatted_version = new_config._format_version_for_filename()
    assert formatted_version == "V1-0-0"


def test_check_for_complete_variables(test_variable_definitions):
    product_config = DataProductConfig(
        data_product_id=DataProductIdentifier.l1b_cam,
        version="1.0.0",
        variable_configuration_path=test_variable_definitions,
    )
    test_array = np.full((100,), 2)
    test_images = np.full((100, 2048, 2048), 3)

    product_config.add_data_to_variable("time", test_array)
    product_config.add_data_to_variable("pix_rad", test_images)
    product_config.add_data_to_variable("pix_lat", test_images)
    product_config.add_data_to_variable("pix_lon", test_images)
    product_config.add_data_to_variable("mask", test_images)
    assert not product_config._check_for_complete_variables()

    product_config.add_data_to_variable("q_flags", test_array)
    assert product_config._check_for_complete_variables()


def test_add_variable_metadata(test_variable_definitions):
    """Test that metadata for variables can be added to a configuration object"""
    new_config = DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0")

    new_config.add_variable_metadata_from_file(test_variable_definitions)

    assert len(new_config.variables) == 6
    assert isinstance(new_config.variables["time"], LiberaVariable)

    assert new_config.variables["time"].metadata.long_name == "Time of sample collection"
    assert len(new_config.variables["time"].metadata.dimensions) == 1
    assert new_config.variables["time"].metadata.units == "datetime64[ns]"
    assert new_config.variables["time"].metadata.valid_range == [0, 1]
    assert new_config.variables["time"].metadata.missing_value == -9999.0
    assert new_config.variables["time"].metadata.dtype == "datetime64[ns]"

    # Only returns variable encoding for variables that have data
    assert len(new_config.variable_encoding_dict) == 0


@pytest.mark.parametrize("data_type", [np.array, xr.DataArray])
@pytest.mark.parametrize(
    ("variable_name", "data_size", "data_value"),
    [
        ("pix_rad", (10, 2048, 2048), 123),
        ("time", (10,), 456),
    ],
)
def test_add_data_to_variable(test_variable_definitions, data_type, variable_name, data_size, data_value):
    """Test that data can be added to a variable that is defined in a yaml file"""
    new_config = DataProductConfig(
        data_product_id=DataProductIdentifier.l1b_cam,
        version="1.0.0",
        variable_configuration_path=test_variable_definitions,
    )

    # Set up the starting data
    data = np.full(data_size, data_value)
    # Take that array and give it a higher level type (ndarray or DataArray)
    typed_data = data_type(data)

    new_config.add_data_to_variable(variable_name=variable_name, variable_data=typed_data)
    # Resulting data in the object should be a properly made DataArray
    assert isinstance(new_config.variables[variable_name].data, xr.DataArray)
    assert np.all(new_config.variables[variable_name].data == data_value)

    # Check for matching dimensions in the DataArray and metadata (order independent)
    expected_dimensions = new_config.variables[variable_name].metadata.dimensions_name_list
    found_dimensions = new_config.variables[variable_name].data.dims
    assert np.all([found_dimension in expected_dimensions for found_dimension in found_dimensions])


def test_create_libera_data_config_no_variable_info():
    # Cannot make a data product config without a data product id and version
    with pytest.raises(ValidationError):
        DataProductConfig()
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad)

    new_config = DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0")

    assert new_config.variables is None
    assert new_config.data_product_id == DataProductIdentifier.l1b_rad
    assert new_config.version == "1.0.0"
    assert new_config.static_project_metadata is not None
    assert new_config.variables is None
    assert new_config.variable_configuration_path is None


def test_create_libera_data_config_with_variable_metadata(test_variable_definitions):
    config_with_variables = DataProductConfig(
        data_product_id=DataProductIdentifier.l1b_rad,
        version="1.0.0",
        variable_configuration_path=test_variable_definitions,
    )
    assert len(config_with_variables.variables) == 6

    example_variable = config_with_variables.variables["time"]
    assert isinstance(example_variable, LiberaVariable)
    assert example_variable.name == "time"

    assert isinstance(example_variable.metadata.dimensions["n_samples"], LiberaDimension)


def test_create_libera_data_product_from_config_file(test_product_definition):
    """Test that a DataProductConfig can be made from a single proper yml file"""
    test_config = DataProductConfig.from_data_config_file(test_product_definition)
    assert len(test_config.variables) == 5

    example_variable = test_config.variables["time_stamp"]
    assert isinstance(example_variable, LiberaVariable)

    assert isinstance(example_variable.metadata.dimensions["n_samples"], LiberaDimension)


@pytest.mark.parametrize(
    ("filename", "product_id", "version", "parts"),
    [
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            "CAM",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
                revision=dt.datetime(2027, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_L1B_RAD-4CH_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc",
            "RAD-4CH",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2025, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2025, 1, 2, 12, 22, 33),
                revision=dt.datetime(2027, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            "CF-RAD",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
                revision=dt.datetime(2027, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_L2_SSW-TOA-FLUXES-ERBE_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc",
            "SSW-TOA-FLUXES-ERBE",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2025, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2025, 1, 2, 12, 22, 33),
                revision=dt.datetime(2027, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
            "JPSS-SPK",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
                revision=dt.datetime(2028, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
            "JPSS-CK",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
                revision=dt.datetime(2028, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_SPICE_ELSCAN-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
            "ELSCAN-CK",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
                revision=dt.datetime(2028, 1, 2, 11, 22, 33),
            ),
        ),
        (
            "LIBERA_SPICE_AZROT-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
            "AZROT-CK",
            "3.14.159",
            dict(
                utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
                utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
                revision=dt.datetime(2028, 1, 2, 11, 22, 33),
            ),
        ),
    ],
)
def test_generate_data_product_filename(filename, product_id, version, parts):
    """Test that a filename is generated correctly"""
    new_config = DataProductConfig(data_product_id=product_id, version=version)
    fn = new_config._generate_data_product_filename(**parts)
    assert str(fn) == filename


def test_create_libera_data_config_with_data(test_product_definition):
    """Test a complete libera data config object is created"""
    sample_length = 32
    time_data = np.full((sample_length,), dt.datetime(2023, 1, 1))
    rad_data = np.full((sample_length,), 123.456)
    lat_data = np.full((sample_length,), 45.678)
    lon_data = np.full((sample_length,), 123.456)
    qflags_data = np.full((sample_length,), 0)

    incomplete_data = [time_data, rad_data, lat_data, lon_data]
    with pytest.raises(ValueError, match="The number of data entries*"):
        DataProductConfig.from_data_config_file(test_product_definition, data=incomplete_data)

    full_data = [time_data, rad_data, lat_data, lon_data, qflags_data]
    new_config = DataProductConfig.from_data_config_file(test_product_definition, data=full_data)
    assert new_config.data_product_id == DataProductIdentifier.l1b_rad
    assert new_config.version == "0.0.1"
    assert len(new_config.variables) == 5
    assert new_config._check_for_complete_variables()


def test_write_netcdf_output_file(test_product_definition, tmp_path):
    new_config = DataProductConfig.from_data_config_file(test_product_definition)

    sample_length = 100
    time_data = np.full((sample_length,), dt.datetime(2023, 1, 1))
    rad_data = np.full((sample_length,), 123.456)
    new_config.add_data_to_variable("time_stamp", time_data)
    new_config.add_data_to_variable("fil_rad", rad_data)

    # Write the file to a temporary location
    new_config.write(folder_location=tmp_path, allow_incomplete=True)

    expected_filename = tmp_path / "LIBERA_L1B_RAD-4CH_V0-0-1_19900102T112233_19900102T122233_R90002122233.nc"
    assert expected_filename.exists()
    os.remove(expected_filename)  # Clean up the file after the test

    with pytest.raises(ValueError, match="Not all variables have*"):
        new_config.write(folder_location=tmp_path, allow_incomplete=False)

    new_config.add_data_to_variable("lat", np.full((sample_length,), 45.678))
    new_config.add_data_to_variable("lon", np.full((sample_length,), 123.456))
    new_config.add_data_to_variable("q_flags", np.full((sample_length,), 0))

    # Now write the file again with all variables
    assert not expected_filename.exists()
    new_config.write(folder_location=tmp_path, allow_incomplete=False)
    assert expected_filename.exists()


def test_write_netcdf_output_file_s3(test_product_definition, create_mock_bucket):
    """Test writing a NetCDF file to an S3 bucket"""
    bucket = create_mock_bucket()
    s3_path = f"s3://{bucket.name}/test_path"

    new_config = DataProductConfig.from_data_config_file(test_product_definition)

    sample_length = 100
    time_data = np.full((sample_length,), dt.datetime(2023, 1, 1))
    rad_data = np.full((sample_length,), 123.456)
    new_config.add_data_to_variable("time_stamp", time_data)
    new_config.add_data_to_variable("fil_rad", rad_data)
    new_config.add_data_to_variable("lat", np.full((sample_length,), 45.678))
    new_config.add_data_to_variable("lon", np.full((sample_length,), 123.456))
    new_config.add_data_to_variable("q_flags", np.full((sample_length,), 0))

    # Write the file to the S3 bucket
    new_config.write(folder_location=s3_path)

    filename = "LIBERA_L1B_RAD-4CH_V0-0-1_19900102T112233_19900102T122233_R90002122233.nc"
    expected_filepath = AnyPath(s3_path) / filename
    assert expected_filepath.exists()
    assert not (AnyPath(os.getcwd()) / filename).exists()
