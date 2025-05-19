# Standard
import datetime as dt

# Installed
import pytest
from pydantic import ValidationError

from libera_utils.aws.constants import DataProductIdentifier

# Local
from libera_utils.io.netcdf import DataProductConfig, LiberaVariable, VariableMetadata


def test_create_libera_data_config():
    # Cannot make a data product config without a data product id and version
    with pytest.raises(ValidationError):
        DataProductConfig()
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad)

    new_config = DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad,
                                   version="1.0.0")

    assert new_config.variables is None
    assert new_config.data_product_id == DataProductIdentifier.l1b_rad
    assert new_config.version == "1.0.0"
    assert new_config.static_project_metadata is not None
    assert new_config.variables is None
    assert new_config.variable_configuration_path is None


def test_create_libera_data_config_with_variable_data(test_variable_definitions):
    config_with_variables = DataProductConfig(
        data_product_id=DataProductIdentifier.l1b_rad,
        version="1.0.0",
        variable_configuration_path=test_variable_definitions
    )
    assert len(config_with_variables.variables) == 2
    assert isinstance(config_with_variables.variables["variable1"], LiberaVariable)


@pytest.mark.parametrize(
    "data_product_id",
    [DataProductIdentifier.l1b_rad, "L1B_RAD-4CH"]
)
def test_config_data_product_id_validation(data_product_id):
    """Test that a configuration object can be made with a data product identifier string or object"""
    DataProductConfig(data_product_id=data_product_id, version="1.0.0")


def test_config_version_validation():
    """Test that a configuration object can be only be made with a properly formatted version string"""
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad,
                          version="1.0")
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad,
                              version="1.0.0.0")
    with pytest.raises(ValidationError):
        DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad,
                              version="alpha")

    # The good case
    DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad,
                      version="1.0.0")


def test_get_static_project_metadata():
    """Test that the Libera Project metadata is correct"""
    project_metadata = DataProductConfig.get_static_project_metadata()

    assert project_metadata.ProjectLongName == "Libera"
    assert project_metadata.ProjectShortName == "Libera"
    # TODO Add this when known
    # assert empty_config.static_project_metadata.PlatformLongName == "TBD"

    assert project_metadata.PlatformShortName == "NOAA-22"
    assert project_metadata.Format == "NetCDF-4"
    assert project_metadata.Conventions == "CF-1.8"


def test_load_data_product_variables_with_metadata(test_variable_definitions):
    variables_dict = DataProductConfig.load_data_product_variables_with_metadata(test_variable_definitions)
    assert len(variables_dict) == 2
    assert isinstance(variables_dict["variable1"], LiberaVariable)
    assert isinstance(variables_dict["variable2"], LiberaVariable)
    assert isinstance(variables_dict["variable1"].metadata, VariableMetadata)
    assert isinstance(variables_dict["variable2"].metadata, VariableMetadata)


def test_add_variables_with_metadata(test_variable_definitions):
    new_config = DataProductConfig(data_product_id=DataProductIdentifier.l1b_rad, version="1.0.0")

    new_config.add_variables_with_metadata(test_variable_definitions)

    assert len(new_config.variables) == 2
    assert isinstance(new_config.variables["variable1"], LiberaVariable)

    assert new_config.variables["variable1"].metadata.long_name == "Test variable long name"
    assert len(new_config.variables["variable1"].metadata.dimensions) == 2
    assert new_config.variables["variable1"].metadata.units == "W/(m^2*sr*nm)"
    assert new_config.variables["variable1"].metadata.valid_range == [0,1]
    assert new_config.variables["variable1"].metadata.missing_value == -9999.
    assert new_config.variables["variable1"].metadata.dtype == "TBD"

@pytest.mark.parametrize(
    ("filename", "product_id", "version", "parts"),
    [
        ('LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
         'L1B_CAM',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
         )),
        ('LIBERA_L1B_RAD-4CH_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc',
         'L1B_RAD-4CH',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2025, 1, 2, 12, 22, 33),
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
         )),
        ('LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
         'L2_CF-RAD',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
         )),
        ('LIBERA_L2_SSW-TOA-FLUXES-ERBE_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc',
         'L2_SSW-TOA-FLUXES-ERBE',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2025, 1, 2, 12, 22, 33),
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
         )),
        ('LIBERA_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp',
         'JPSS-SPK',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
        ('LIBERA_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc',
         'JPSS-CK',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
        ('LIBERA_ELSCAN-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc',
         'ELSCAN-CK',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
        ('LIBERA_AZROT-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc',
         'AZROT-CK',
         '3.14.159',
         dict(
             utc_start_time=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end_time=dt.datetime(2027, 1, 2, 12, 22, 33),
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
    ]
)

def test_generate_data_product_filename(filename, product_id, version, parts):
    new_config = DataProductConfig(data_product_id=product_id,version=version)
    fn = new_config.generate_data_product_filename(**parts)
    assert str(fn) == filename
