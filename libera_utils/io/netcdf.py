# Standard
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from pandas import DataFrame

# Installed
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from xarray import DataArray

# Local
from libera_utils.aws.constants import DataProductIdentifier
from libera_utils.config import config
from libera_utils.io.filenaming import AttitudeKernelFilename, EphemerisKernelFilename, LiberaDataProductFilename

logger = logging.getLogger(__name__)


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
    """Pydantic model for variable-level metadata for NetCDF-4 files."""

    long_name: str
    dimensions: list
    valid_range: list
    missing_value: int | float  # this has to end up whatever type as the data, it may make more sense as an enum
    units: str | None = None
    dtype: str | None = None


class LiberaVariable(BaseModel):
    """Pydantic model for a Libera variable."""

    metadata: VariableMetadata
    variable_encoding: dict | None = {"_FillValue": None, "zlib": True, "complevel": 4}

    # To allow pydantic use of DataArray, ndarray, and Dataframes
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: DataArray | np.ndarray | DataFrame | None = None


class DataProductConfig(BaseModel):
    """
    Pydantic model for a Libera data product configuration.

    Notes
    -----
    This is the primary object used to configure and write properly formatted NetCDF4 files that can be archived
    with the Libera SDC.
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

        """
        with file_path.open("r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
            return StaticProjectMetadata(**yaml_data)

    @field_validator("data_product_id", mode="before")
    @classmethod
    def ensure_data_product_id(cls, raw_data_product_id: str | DataProductIdentifier) -> DataProductIdentifier:
        """Converts raw data product id string to DataProductIdentifier class if necessary."""
        if isinstance(raw_data_product_id, DataProductIdentifier):
            return raw_data_product_id
        return DataProductIdentifier(raw_data_product_id)

    @field_validator("version", mode="before")
    @classmethod
    def enforce_version_format(cls, version_string: str):
        """Enforces the proper formatting of the version string as M.m.p."""
        if len(version_string.split(".")) != 3:
            raise ValueError("Version string must be formatted as M.m.p")
        for part in version_string.split("."):
            if not part.isdigit():
                raise ValueError("Version string must be formatted as M.m.p")
        return version_string

    @field_validator("variable_configuration_path", mode="before")
    @classmethod
    def use_variable_configuration(cls, variable_configuration_path: str | Path):
        """Optional validator method that allows the user to specify a path to the variable configuration file."""
        if variable_configuration_path is None:
            return None
        if isinstance(variable_configuration_path, str):
            variable_configuration_path = Path(variable_configuration_path)
        return variable_configuration_path

    @model_validator(mode="after")
    def load_variables_from_config(self):
        """If a model is instantiated with a configuration path listed then populate the variables from that file"""
        if self.variable_configuration_path is not None and self.variables is None:
            self.add_variables_with_metadata(self.variable_configuration_path)
        return self

    @classmethod
    def load_data_product_variables_with_metadata(cls, file_path: str | Path):
        """Method to create a properly made LiberaVariables from a config file.

        Notes
        -----
        This method is used as part of  validator if a filepath is passed in to construct the Data
        ProductConfig object.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_path.suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
        elif file_path.suffix in (".yaml", ".yml"):
            with file_path.open("r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError("Unsupported file type. Must be JSON or YAML.")
        for k, v in config_data.items():
            metadata = VariableMetadata(**v)
            variable_object = LiberaVariable(metadata=metadata)
            # Remake the dictionary entry with the LiberaVariable object with the metadata included
            config_data[k] = variable_object
        return config_data

    def add_variables_with_metadata(self, variable_config_file_path):
        """A wrapper around the load_data_product_variables_with_metadata method.

        Notes
        -----
        This allows the model to be validated after the variables have been added.
        """
        self.variables = DataProductConfig.load_data_product_variables_with_metadata(variable_config_file_path)
        DataProductConfig.model_validate(self)

    def format_version(self):
        swap_dots_for_dashes = self.version.replace(".", "-")
        return "V" + swap_dots_for_dashes

    def generate_data_product_filename(
        self,
        utc_start_time: datetime,
        utc_end_time: datetime,
        revision: datetime | None = None,
    ) -> LiberaDataProductFilename:
        """Generate a valid data product filename using the Filenaming methods"""
        filename_version = self.format_version()
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

    def add_data_to_variable(self, variable_name, variable_data):
        """Adds the actual data to an existing LiberaVariable"""
        self.variables[variable_name].data = variable_data
