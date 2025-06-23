# Standard
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import numpy as np
import yaml
from cloudpathlib import AnyPath

# Installed
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator
from xarray import DataArray

# Local
from libera_utils.aws.constants import DataProductIdentifier
from libera_utils.config import config
from libera_utils.io.filenaming import AttitudeKernelFilename, EphemerisKernelFilename, LiberaDataProductFilename
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)

DYNAMIC_SIZE = -1


class StaticProjectMetadata(BaseModel):
    """Pydantic model for unchanging NetCDF-4 file metadata.

    Notes
    -----
    See more details at https://wiki.earthdata.nasa.gov/display/CMR/UMM-C+Schema+Representation
    """

    Format: str = Field(description="Required to be NetCDF-4 by PLRA")
    Conventions: str = Field(
        description="Specifies that we are using Climate and Forecast (CF) conventions"
        "along with which version. (Ex: CF-1.12). See https://cfconventions.org/conventions.html and"
        "https://cfconventions.org/faq.html#my-file-was-written-using-an-earlier-version-of-cf-is-it-still-compliant"
    )
    ProjectLongName: str = Field(description="Libera")
    ProjectShortName: str = Field(description="Libera")
    PlatformLongName: str = Field(
        description="This will only be needed if JPSS-4 is the platform identifier instead of NOAA-22"
        "and then it will be Joint Polar Satellite System 4"
    )
    PlatformShortName: str = Field(description="Likely to be NOAA-22. Need to confirm")


class LiberaDimension(BaseModel):
    """Pydantic model for any dimension used for Libera data

    Attributes
    ----------
    name: str
        The name of the dimension, which should match the names defined in the available dimensions.
    size: str | int | None
        The size of the dimension. If it is dynamic, it should be set to "dynamic". If it is a fixed size,
        it should be an integer representing the size. If it is not set, it can be None.
    long_name: str
        A long name for the dimension, providing a human-readable description of what the dimension represents.

    Notes
    -----
    This class defines the needed aspects of a Libera Dimension and also contains the list of known and defined
    dimensions that are available to be used.
    """

    name: str
    size: int
    long_name: str
    is_set: bool = True

    # Private dynamic indicator
    _is_dynamic: bool = False

    @property
    def is_dynamic_size(self):
        return self._is_dynamic

    @model_validator(mode="after")
    def validate_dynamic_size_and_is_set(self):
        """Ensure that the dynamic size and set status are properly set after initialization.

        Returns
        -------
        LiberaDimension
            The instance of LiberaDimension with the dynamic size and set status updated.

        Notes
        -----
        This method is called after the model is validated. It sets the _is_dynamic and is_set attributes correctly
        based on the size of the dimension. If the size is -1, called DYNAMIC_SIZE, it indicates a dynamic size, and
        is_set is set to False.
        """
        if self.size == DYNAMIC_SIZE:
            self._is_dynamic = True
            self.is_set = False
            self.size = DYNAMIC_SIZE
        if self.size < DYNAMIC_SIZE:
            raise ValueError("The size of a dimension cannot be negative")
        return self

    @staticmethod
    def get_available_dimensions_dict(file_path: AnyPath | None = None):
        """Loads the available dimensions for variables on the Libera project.

        Parameters
        ----------
        file_path: Path
            The path to the corresponding yml file.

        Notes
        -----
        These are the only available dimensions to be used with official Libera variables.
        """
        if file_path is None:
            file_path = AnyPath(config.get("LIBERA_UTILS_DATA_DIR")) / "libera_dimensions.yml"
        with file_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        dimensions = {}
        for k, v in yaml_data.items():
            if v["size"] == "dynamic":
                v["size"] = DYNAMIC_SIZE
            dimensions[k] = LiberaDimension(name=k, **v)
        return dimensions


class DynamicProductMetadata(BaseModel):
    """Pydantic model for file specific metadata."""

    GranuleID: str  # output filename
    input_files: list[str]


class GPolygon(BaseModel):
    """Pydantic model for file specifics geolocation metadata."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class DynamicSpatioTemporalMetadata(BaseModel):
    """Pydantic model for file specific spatial and temporal metadata"""

    ProductionDateTime: datetime  # May end up needing to be a string
    RangeBeginningDate: datetime  # May end up needing to be a string
    RangeBeginningTime: datetime  # May end up needing to be a string
    RangeEndingDate: datetime  # May end up needing to be a string
    RangeEndingTime: datetime  # May end up needing to be a string
    GPolygon: list[GPolygon]


class ProductMetadata(BaseModel):
    """Pydantic model for file-level metadata.

     Notes
     -----
    This data will change between files and is obtained from the science Dataset.
    The create_file_metadata method makes this object.
    """

    # Dynamic File Metadata
    dynamic_metadata: DynamicProductMetadata | None
    # Dynamic Spatio-Temporal Metadata
    dynamic_spatio_temporal_metadata: DynamicSpatioTemporalMetadata | None


class VariableMetadata(BaseModel):
    """Pydantic model for variable-level metadata for NetCDF-4 files.

    Attributes
    ----------
    available_dimensions: ClassVar[list[LiberaDimension]]
        A class variable listing all available dimensions that can be used in the variable metadata.
    long_name: str
        A long name for the variable, providing a human-readable description of what the variable represents.
    dimensions: list[LiberaDimension]
        A list of dimensions that the variable's data array will have. These should be instances of LiberaDimension.
    valid_range: list
        A list specifying the valid range of values for the variable's data, excluding missing values.
    missing_value: int | float
        The value used to represent missing data in the variable's data array. This should be the same type as the data.
    units: str | None
        The units of the variable's data, if applicable. This can be None if the variable does not have units.
    dtype: str | None
        The data type of the variable's data, if applicable. This can be None if the variable does not have a specific
        data type.

    Notes
    -----
    These derive from the CF conventions. More information at:
    https://cfconventions.org/cf-conventions/cf-conventions.html
    """

    # Class variable listing all available dimensions
    available_dimensions: ClassVar[list[LiberaDimension]] = LiberaDimension.get_available_dimensions_dict()

    # Instance variables through pydantic
    long_name: str = Field(description="Description of the variable")
    dimensions: dict[str, LiberaDimension] = Field(description="Named dimensions of the variable's data array")
    valid_range: list = Field(description="The range of the possible valid data values, not including missing values")
    missing_value: int | float = Field(
        description="The value used to represent missing data. Should be the same type as the data"
    )
    units: str | None = Field(description="The units of the data")
    dtype: str | None = Field(description="The data type of the data")

    @field_validator("dimensions", mode="before")
    @classmethod
    def set_dimensions(cls, input_dimensions: list[str] | list[LiberaDimension]):
        """Class method validating the list of input dimensions when they are provided.

        Parameters
        ----------
        input_dimensions: list[str] | list[LiberaDimension]
            A list of dimension names as strings or LiberaDimension instances.

        Returns
        -------
        list[LiberaDimension]
            A list of LiberaDimension instances corresponding to the input dimensions.

        Raises
        ------
        TypeError
            If the input dimensions are not a list or if the items in the list are not of type str or LiberaDimension.

        Notes
        -----
        This method ensures that the dimensions are either strings that match the available dimensions or
        instances of the LiberaDimension class. It converts string dimensions to LiberaDimension instances.
        """
        if not isinstance(input_dimensions, list):
            raise TypeError("Dimensions field of a variable metadata must be a list")

        output_dimensions = {}
        for dimension in input_dimensions:
            if isinstance(dimension, str):
                if dimension in VariableMetadata.available_dimensions.keys():
                    output_dimensions[dimension] = VariableMetadata.available_dimensions[dimension].model_copy(
                        deep=True
                    )
                else:
                    raise ValueError(f"The specified dimension {dimension} is not a listed Libera dimension.")
            elif isinstance(dimension, LiberaDimension):
                output_dimensions[dimension.name] = dimension
            else:
                raise TypeError("Items in the dimension list must be of type str or LiberaDimension")
        return output_dimensions

    def set_dynamic_dimension(self, selected_dimension: LiberaDimension, data_length: int):
        """Internal method to set a dynamic dimension length

        Parameters
        ----------
        selected_dimension: LiberaDimension
            The dimension that has a dynamic dimension to be set.
        data_length: int
            The length of the dimension that will replace the dynamic size.

        Raises
        ------
        ValueError
            If the selected dimension is not dynamic or if the size is already set.

        Notes
        -----
        This method is used to set the size of a dimension that has been defined as dynamic in the metadata.
        """
        if self.dimensions[selected_dimension.name].is_dynamic_size:
            if not self.dimensions[selected_dimension.name].is_set:
                if data_length == 0:
                    warnings.warn(
                        f"Setting the {selected_dimension.name} dimension to 0. This may cause issues "
                        f"with data processing."
                    )
                if data_length < 0:
                    raise ValueError(f"Cannot set the {selected_dimension.name} dimension to a negative size")
                self.dimensions[selected_dimension.name].size = data_length
                self.dimensions[selected_dimension.name].is_set = True
            else:
                raise ValueError(f"The {selected_dimension.name} dimension has already been set")
        else:
            raise ValueError(f"The {selected_dimension.name} dimension is not listed as dynamic")

    @computed_field
    def dimensions_name_list(self) -> tuple:
        """A computed property tuple of the string names of the dimensions

        Returns
        -------
        tuple
            A tuple of dimension names as strings.

        Notes
        -----
        This property returns a tuple of the names of the dimensions in the order they are defined in the metadata.

        """
        name_list = tuple(dimension.name for dimension in self.dimensions.values())
        return name_list

    @computed_field
    def dimensions_shape(self) -> tuple:
        """A tuple of the shape of the dimensions for this variable

        Returns
        -------
        tuple
            A tuple of dimension sizes, where dynamic dimensions are represented as "dynamic".

        Notes
        -----
        This property returns a tuple of the sizes of the dimensions in the order they are defined in the metadata.
        """
        shapes = tuple(dimension.size for dimension in self.dimensions.values())
        return shapes


class LiberaVariable(BaseModel):
    """Pydantic model for a Libera variable.

    Attributes
    ----------
    metadata: VariableMetadata
        The metadata associated with the variable, including its dimensions, valid range, missing value, units, and
        data type.
    variable_encoding: dict | None
        A dictionary specifying how the variable's data should be encoded when written to a NetCDF file.
    data: DataArray | None
        The data associated with the variable, stored as an xarray DataArray
    """

    metadata: VariableMetadata
    variable_encoding: dict | None = {"_FillValue": None, "zlib": True, "complevel": 4}

    # To allow pydantic use of DataArray, ndarray, and Dataframes
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: DataArray | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_incoming_metadata(cls, input_metadata: VariableMetadata):
        """Ensure that the metadata is copied in as a VariableMetadata object

        Parameters
        ----------
        input_metadata: VariableMetadata
            The metadata to be copied into the variable.

        Returns
        -------
        VariableMetadata
            A deep copy of the input metadata.

        Raises
        ------
        TypeError
            If the input metadata is not of type VariableMetadata.

        Notes
        -----
        This method validates that the input metadata is of type VariableMetadata and returns a deep copy of it.
        """
        if not isinstance(input_metadata, VariableMetadata):
            raise TypeError("Metadata must be of type VariableMetadata")
        return input_metadata.model_copy(deep=True)

    @field_validator("data", mode="before")
    @classmethod
    def set_data_as_dataarray(cls, input_data: DataArray | np.ndarray | None):
        """Ensure that the data is internally stored as a DataArray when passed in to the model.

        Parameters
        ----------
        input_data: DataArray | np.ndarray | None
            The data to be set for the variable. It can be a DataArray, a numpy ndarray, or None.

        Returns
        -------
        DataArray | None
            A DataArray containing the input data, or None if the input data is None.

        Raises
        ------
        TypeError
            If the input data is not of type DataArray or np.ndarray.

        Notes
        -----
        This method checks if the input data is None, a numpy ndarray, or an xarray DataArray. If it is None,
        it returns None. If it is a numpy ndarray, it converts it to a DataArray. If it is already a DataArray,
        it returns it as is. If the input data is of any other type, it raises a TypeError.
        """
        if isinstance(input_data, np.ndarray):
            return DataArray(input_data)
        return input_data

    @model_validator(mode="after")
    def confirm_and_set_data_with_dimensions(self):
        """Ensure the dimensions in provided data are correct on instantiation.

        Returns
        -------
        LiberaVariable
            The instance of LiberaVariable with the data set if it was provided.

        Notes
        -----
        This method is called after the model is validated. It checks if the data is not None and sets the data
        using the set_data method. If the data is None, it does nothing.

        """
        if self.data is not None:
            self.set_data(self.data)
        return self

    def _set_all_dynamic_dimension_lengths(self, data_array: DataArray | np.ndarray) -> None:
        """Internal method to set any dynamic dimension lengths to match a given array

        Parameters
        ----------
        data_array: DataArray | np.ndarray
            The data array whose shape will be used to set the dynamic dimensions in the metadata.

        Notes
        -----
        This should be run after the _check_for_bad_dimensions below. This method iterates through the dimensions of
        the metadata and sets the size of any dynamic dimensions to the corresponding size in the provided data array.
        """
        for dimension in self.metadata.dimensions.values():
            if dimension.is_dynamic_size:
                data_size = data_array.shape[self.metadata.dimensions_name_list.index(dimension.name)]
                self.metadata.set_dynamic_dimension(dimension, data_size)

    def _check_for_bad_dimensions(self, data_array: DataArray | np.ndarray):
        """Internal method to check provided data set has dimension that match the shape expected in the metadata.

        Parameters
        ----------
        data_array: DataArray | np.ndarray
            The data array whose shape will be checked against the metadata dimensions.

        Raises
        ------
        ValueError
            If the provided data array does not match the expected dimensions in the metadata.

        Notes
        -----
        This will ignore any "dynamic" data dimensions. If the provided data does not match the expected dimensions
        in the metadata, it raises a ValueError.
        """
        if len(self.metadata.dimensions_shape) != len(data_array.shape):
            raise ValueError(
                f"The provided data has {len(data_array.shape)} dimensions but was expected to have "
                f"{len(self.metadata.dimensions_shape)}"
            )

        for i in range(len(self.metadata.dimensions_shape)):
            if self.metadata.dimensions_shape[i] != DYNAMIC_SIZE:  # DYNAMIC_SIZE = -1 is used for dynamic dimensions
                if self.metadata.dimensions_shape[i] != data_array.shape[i]:
                    raise ValueError(
                        f"There is a mismatch in data shape provided. Provided data has shape "
                        f"{data_array.shape} but the metadata expects a shape: "
                        f"{self.metadata.dimensions_shape}"
                    )
        return

    def _set_data_with_dimensions_to_match_metadata(self, data_array: DataArray | np.ndarray):
        """Internal method to ensure the dimensions from the metadata are used in the data itself

        Parameters
        ----------
        data_array: DataArray | np.ndarray
            The data array to be set for the variable. It can be a DataArray or a numpy ndarray.

        Raises
        ------
        TypeError
            If the provided data is not of type DataArray or np.ndarray.
        ValueError
            If the provided data does not match the expected dimensions in the metadata.

        Notes
        -----
        This method checks the provided data array against the metadata dimensions and sets the data to a new DataArray
        with the correct dimensions. It also checks for bad dimensions and sets any dynamic dimension lengths.
        """
        # Start by checking that the provided dimensions are good and setting any dynamic
        self._check_for_bad_dimensions(data_array)
        self._set_all_dynamic_dimension_lengths(data_array)

        if isinstance(data_array, np.ndarray):
            # make a new DataArray with correctly named dimensions
            self.data = DataArray(data=data_array, dims=self.metadata.dimensions_name_list)
        elif isinstance(data_array, DataArray):
            self.data = DataArray(data=data_array.data, dims=self.metadata.dimensions_name_list)
        else:
            raise TypeError("Data to be added must be an Xarray DataArray or numpy ndarray")
        self._check_for_bad_dimensions(self.data)

    def set_data(self, data: DataArray | np.ndarray):
        """Takes the provided data into an internal DataArray with Libera defined dimensions

        Parameters
        ----------
        data: DataArray | np.ndarray
            The data to be set for the variable. It can be a DataArray or a numpy ndarray.

        Raises
        ------
        TypeError
            If the provided data is not of type DataArray or np.ndarray.
        ValueError
            If the provided data does not match the expected dimensions in the metadata.

        Notes
        -----
        This method is used to set the data for the variable. It checks if the data is a DataArray or a numpy ndarray,
        and if so, it sets the data while ensuring that the dimensions match those defined in the metadata.

        """
        if isinstance(data, (DataArray | np.ndarray)):
            self._set_data_with_dimensions_to_match_metadata(data)
        else:
            raise TypeError("Data to be added must be an Xarray DataArray or numpy ndarray")


class DataProductConfig(BaseModel):
    """
    Pydantic model for a Libera data product configuration.

    Attributes
    ----------
    data_product_id: DataProductIdentifier
        The identifier for the data product, which is used to generate the filename.
    static_project_metadata: StaticProjectMetadata
        The static metadata associated with the Libera project, loaded automatically.
    version: str
        The version number of the data product in X.Y.Z format, where X is the major version, Y is the minor version,
        and Z is the patch version.
    variable_configuration_path: Path | None
        The path to the variable configuration file, which can be used to load variable metadata.
    variables: dict[str, LiberaVariable] | None
        A dictionary of variable names and their corresponding LiberaVariable objects, which contain metadata and data.
    product_metadata: ProductMetadata | None
        The metadata associated with the data product, including dynamic metadata and spatio-temporal metadata.

    Notes
    -----
    This is the primary object used to configure and write properly formatted NetCDF4 files that can be archived
    with the Libera SDC. It includes methods for loading variable metadata, validating the configuration,
    and writing the data product to a file.
    """

    # Required fields to be filled at instantiation
    data_product_id: DataProductIdentifier = Field(
        description="The libera_utils defined data product identifier used to generate a specified filename"
    )
    static_project_metadata: StaticProjectMetadata = Field(
        description="The metadata associated with the Libera Project. Loaded automatically.",
        default_factory=lambda: DataProductConfig.get_static_project_metadata(),
    )
    version: str = Field(
        description="The version number in X.Y.Z format with X = Major version, Y = Minor version, Z = Patch version"
    )

    # Optional fields to be filled after instantiation
    variable_configuration_path: Path | None = None
    variables: dict[str, LiberaVariable] | None = None
    product_metadata: ProductMetadata | None = None

    @classmethod
    def get_static_project_metadata(
        cls, file_path=Path(config.get("LIBERA_UTILS_DATA_DIR")) / "static_project_metadata.yml"
    ):
        """Loads the static project metadata field of the object from a file

        Parameters
        ----------
        file_path: Path
            The path to the corresponding yml file.

        Returns
        -------
        StaticProjectMetadata
            An instance of StaticProjectMetadata containing the static metadata for the Libera project.
        """
        with file_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            return StaticProjectMetadata(**yaml_data)

    @classmethod
    def from_data_config_file(cls, product_config_filepath: str | AnyPath):
        """Primary means of making a data product config all at once

        Parameters
        ----------
        product_config_filepath: str | AnyPath
            The path to the configuration file containing the product metadata and variable definitions.

        Returns
        -------
        DataProductConfig
            An instance of DataProductConfig with the loaded data product ID, version, and variables.

        Notes
        -----
        This method reads a configuration file (in YAML format) that contains the static product metadata,
        version, and variable metadata. It then creates an instance of DataProductConfig with the loaded data.

        """
        with smart_open(product_config_filepath, "r") as f:
            yaml_data = yaml.safe_load(f)
            product_id = yaml_data["static_product_metadata"]["ProductID"]
            version = yaml_data["static_product_metadata"]["version"]
            variables = yaml_data["variables"]
            # Convert variable metadata as text into metadata objects
            for k, v in variables.items():
                metadata = VariableMetadata(**v)
                variable_object = LiberaVariable(metadata=metadata)
                variables[k] = variable_object
            return cls(data_product_id=product_id, version=version, variables=variables)

    @field_validator("data_product_id", mode="before")
    @classmethod
    def ensure_data_product_id(cls, raw_data_product_id: str | DataProductIdentifier) -> DataProductIdentifier:
        """Converts raw data product id string to DataProductIdentifier class if necessary.

        Parameters
        ----------
        raw_data_product_id: str | DataProductIdentifier
            The raw data product ID, which can be a string or an instance of DataProductIdentifier.

        Returns
        -------
        DataProductIdentifier
            An instance of DataProductIdentifier representing the data product ID.

        Notes
        -----
        This method checks if the provided data product ID is already an instance of DataProductIdentifier.
        If it is, it returns it as is. If it is a string, it converts it to a DataProductIdentifier instance.

        """
        if isinstance(raw_data_product_id, DataProductIdentifier):
            return raw_data_product_id
        return DataProductIdentifier(raw_data_product_id)

    @field_validator("version", mode="before")
    @classmethod
    def enforce_version_format(cls, version_string: str):
        """Enforces the proper formatting of the version string as M.m.p.

        Parameters
        ----------
        version_string: str
            The version string to be validated, expected to be in the format M.m.p.

        Returns
        -------
        str
            The validated version string in the format M.m.p.

        Raises
        ------
        ValueError

        Notes
        -----
        This method checks if the version string is formatted as M.m.p, where M, m, and p are integers.
        If the version string does not match this format, it raises a ValueError.
        """
        if len(version_string.split(".")) != 3:
            raise ValueError("Version string must be formatted as M.m.p")
        for part in version_string.split("."):
            if not part.isdigit():
                raise ValueError("Version string must be formatted as M.m.p")
        return version_string

    @field_validator("variable_configuration_path", mode="before")
    @classmethod
    def use_variable_configuration(cls, variable_configuration_path: str | Path):
        """Optional validator method that allows the user to specify a path to the variable configuration file.

        Parameters
        ----------
        variable_configuration_path: str | Path | None
            The path to the variable configuration file. It can be a string, a Path object, or None.

        Returns
        -------
        Path | None
            A Path object representing the variable configuration path, or None if no path is provided.

        Notes
        -----
        This method checks if the provided variable configuration path is None or a string. If it is a string,
        it converts it to a Path object. If the path is None, it returns None.
        """
        if variable_configuration_path is None:
            return None
        if isinstance(variable_configuration_path, str):
            variable_configuration_path = Path(variable_configuration_path)
        return variable_configuration_path

    @model_validator(mode="after")
    def load_variables_from_config(self):
        """If a model is instantiated with a configuration path listed then populate the variables from that file

        Returns
        -------
        DataProductConfig
            The instance of DataProductConfig with the variables loaded from the configuration file, if applicable.

        Notes
        -----
        This method is called after the model is validated. It checks if the variable_configuration_path is not None
        and if the variables are None. If so, it calls the add_variable_metadata_from_file method to load the variable
        metadata from the specified file. This allows the model to be validated after the variables have been added.
        """
        if self.variable_configuration_path is not None and self.variables is None:
            self.add_variable_metadata_from_file(self.variable_configuration_path)
        return self

    @classmethod
    def load_data_product_variables_with_metadata(cls, file_path: str | Path):
        """Method to create a properly made LiberaVariables from a config file.

        Parameters
        ----------
        file_path: str | Path
            The path to the configuration file containing variable metadata.

        Returns
        -------
        dict
            A dictionary where the keys are variable names and the values are LiberaVariable objects

        Notes
        -----
        This method is used as part of  validator if a filepath is passed in to construct the Data
        ProductConfig object. It reads a JSON or YAML file containing variable metadata,
        and returns a dictionary of LiberaVariable objects with their metadata.
        """
        if isinstance(file_path, str):
            file_path = AnyPath(file_path)

        if file_path.suffix == ".json":
            with smart_open(file_path, "r") as f:
                config_data = json.load(f)
        elif file_path.suffix in (".yaml", ".yml"):
            with smart_open(file_path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file type. Must be JSON or YAML.")
        for k, v in config_data.items():
            metadata = VariableMetadata(**v)
            variable_object = LiberaVariable(metadata=metadata)
            config_data[k] = variable_object
        return config_data

    def _format_version_for_filename(self):
        """Internal method for ensuring version string is proper for file output

        Returns
        -------
        str
            A formatted version string suitable for use in filenames, with dots replaced by dashes and prefixed with
            "V".

        Notes
        -----
        This method replaces the dots in the version string with dashes and prepends a "V" to it.
        """
        swap_dots_for_dashes = self.version.replace(".", "-")
        return "V" + swap_dots_for_dashes

    def add_variable_metadata_from_file(self, variable_config_file_path):
        """A wrapper around the load_data_product_variables_with_metadata method.

        Parameters
        ----------
        variable_config_file_path: str | Path
            The path to the configuration file containing variable metadata.

        Raises
        ------
        ValueError
            If the provided file path does not point to a valid JSON or YAML file.

        Notes
        -----
        This allows the model to be validated after the variables have been added.
        """
        self.variables = DataProductConfig.load_data_product_variables_with_metadata(variable_config_file_path)
        DataProductConfig.model_validate(self)

    def _generate_data_product_filename(
        self,
        utc_start_time: datetime,
        utc_end_time: datetime,
        revision: datetime | None = None,
    ) -> LiberaDataProductFilename:
        """Generate a valid data product filename using the Filenaming methods

        Parameters
        ----------
        utc_start_time: datetime
            The start time of the data product, used to generate the filename.
        utc_end_time: datetime
            The end time of the data product, used to generate the filename.
        revision: datetime | None
            The revision date of the data product, used to generate the filename. If None, the current UTC time is used.

        Returns
        -------
        LiberaDataProductFilename
            An instance of LiberaDataProductFilename representing the generated filename for the data product.

        Notes
        -----
        This method generates a filename for the data product based on its ID, version, start and end times,
        and revision date. It uses the LiberaDataProductFilename class to create the filename.
        """
        filename_version = self._format_version_for_filename()
        match self.data_product_id.split("_", maxsplit=1)[0]:
            case "L2" | "L1B" | "L0":
                data_level = self.data_product_id.split("_", maxsplit=1)[0]
                product_name = self.data_product_id.split("_", maxsplit=1)[1]
                filename = LiberaDataProductFilename.from_filename_parts(
                    data_level=data_level,
                    product_name=product_name,
                    version=filename_version,
                    utc_start=utc_start_time,
                    utc_end=utc_end_time,
                    revision=revision or datetime.now(UTC),
                )
            case self.data_product_id:
                product_name = self.data_product_id
                # Second half of a SPICE product id will be -CK or -SPK
                match product_name.split("-")[1]:
                    case "SPK":
                        filename = EphemerisKernelFilename.from_filename_parts(
                            spk_object=product_name,
                            version=filename_version,
                            utc_start=utc_start_time,
                            utc_end=utc_end_time,
                            revision=revision or datetime.now(UTC),
                        )
                    case "CK":
                        filename = AttitudeKernelFilename.from_filename_parts(
                            ck_object=product_name,
                            version=filename_version,
                            utc_start=utc_start_time,
                            utc_end=utc_end_time,
                            revision=revision or datetime.now(UTC),
                        )
                    case _:
                        raise ValueError(f"Got unexpected product name {product_name}")
        return filename

    def add_data_to_variable(self, variable_name: str, variable_data: np.ndarray | DataArray):
        """Adds the actual data to an existing LiberaVariable

        Parameters
        ----------
        variable_name: str
            The name of the variable to which the data will be added.
        variable_data: np.ndarray | DataArray
            The data to be added to the variable. It can be a numpy ndarray or an xarray DataArray.

        Raises
        ------
        KeyError
            If the variable name does not exist in the configuration.
        TypeError
            If the variable data is not of type np.ndarray or DataArray.
        ValueError
            If the variable data does not match the expected dimensions defined in the variable's metadata.

        Notes
        -----
        This method takes the name of a variable and the data to be added to that variable.
        It checks if the variable exists in the configuration and then sets the data for that variable.
        """
        self.variables[variable_name].set_data(variable_data)
