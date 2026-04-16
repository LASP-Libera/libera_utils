"""Data Product configuration and writing for Libera NetCDF4 data product files"""

import logging
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
import yaml
from cloudpathlib import AnyPath
from pydantic import BaseModel, ConfigDict, Field, field_validator
from xarray import DataArray, Dataset

from libera_utils.config import config
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.version import ALGORITHM_VERSION_REGEX

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = {"zlib": True, "complevel": 4}


class LiberaDimensionDefinition(BaseModel):
    """Pydantic model describing rules for a Libera dimension definition.

    Attributes
    ----------
    size: int | None
        The size of the dimension. If None, the dimension is dynamic.
    long_name: str
        A descriptive human-readable name for the dimension.
    """

    model_config = ConfigDict(frozen=True)

    size: int | None = Field(default=None, description="The size of the dimension. If None, the dimension is dynamic.")
    long_name: str = Field(..., description="A descriptive human-readable name for the dimension.")


class LiberaVariableDefinition(BaseModel):
    """Pydantic model for a Libera variable definition.

    This model is the same for both data variables and coordinate variables

    Attributes
    ----------
    dtype: str
        The data type of the variable's data array, specified as a string
    attributes: VariableAttributes
        The attribute metadata for the variable, containing specific key value pairs for CF metadata compliance
    dimensions: list[str]
        A list of dimension names that the variable's data array references.
    encoding: dict
        A dictionary specifying how the variable's data should be encoded when written to a NetCDF file.
    """

    model_config = ConfigDict(frozen=True)

    _standard_allowed_dimensions: ClassVar[dict[str, LiberaDimensionDefinition]] = dict()

    dtype: str = Field(description="The data type of the variable's data array, specified as a string")
    attributes: dict[str, Any] = Field(default=dict(), description="Attribute metadata for the variable")
    dimensions: list[str] = Field(default=list(), description="Dimensions of the variable's data array")
    encoding: dict = Field(
        default_factory=lambda: DEFAULT_ENCODING.copy(),
        description="Encoding settings for the variable, determining how it is stored on disk",
    )

    @staticmethod
    def _get_standard_dimensions(
        file_path: PathLike | None = None,
    ) -> dict[str, LiberaDimensionDefinition]:
        """Loads standard dimension metadata from a YAML file.

        These standard dimensions are expected to be used in every Libera data product so we store them in a global config.

        Parameters
        ----------
        file_path: PathLike | None
            The path to the standard dimension metadata YAML file.

        Returns
        -------
        dict[str, LiberaDimensionDefinition]
            Dictionary of dimension name to LiberaDimensionDefinition instances.
        """
        if file_path is None:
            file_path = Path(str(config.get("LIBERA_DIMENSIONS_DEFINITION_PATH")))
        with AnyPath(file_path).open("r", encoding="utf-8") as f:
            dim_dict = yaml.safe_load(f)
            return {k: LiberaDimensionDefinition.model_validate(v) for k, v in dim_dict.items()}

    @field_validator("dimensions", mode="before")
    @classmethod
    def _check_allowed_dimensions(cls, raw_dimensions: list[str]) -> list[str]:
        """Validates that all dimensions used in coordinates and variables are defined in the global standard dimensions.

        This is just an early preliminary check and does not check anything related to data.
        Verification of dimension size is done when checking conformance once data is provided.

        Parameters
        ----------
        raw_dimensions : list[str]
            The raw dimensions specification in the product definition.

        Returns
        -------
        list[str]
            The validated dimensions list, unchanged.
        """
        if not cls._standard_allowed_dimensions:
            cls._standard_allowed_dimensions = cls._get_standard_dimensions()

        for dimension in raw_dimensions:
            if dimension.upper() not in cls._standard_allowed_dimensions.keys():
                raise ValueError(
                    f"Undefined dimension name '{dimension}' used in product definition. "
                    "All dimensions must be defined in the global standard dimensions. "
                    "If you need a new dimension name added to the global standard list, you must work with the SDC "
                    "team to add it.",
                )

        return raw_dimensions

    @field_validator("dtype", mode="before")
    @classmethod
    def _validate_dtype(cls, dtype: str) -> str:
        """Validates that the dtype specified in the product definition is a valid numpy dtype string.

        Parameters
        ----------
        dtype : str
            The raw dtype specification in the product definition.

        Returns
        -------
        str
            The validated dtype string, unchanged.
        """
        try:
            np.dtype(dtype)
        except TypeError as e:
            raise ValueError(
                f"Invalid dtype '{dtype}' specified in product definition. The dtype must be a valid numpy dtype string."
            ) from e
        return dtype

    @field_validator("encoding", mode="before")
    @classmethod
    def _set_encoding(cls, encoding: dict | None):
        """Merge configured encoding with required defaults, issuing warnings on conflicts."""
        if encoding is None:
            return DEFAULT_ENCODING.copy()
        for k, v in DEFAULT_ENCODING.items():
            if k in encoding and encoding[k] != v:
                # This is only a warning because the SDC reserves the right to change the required encoding settings
                # in order to improve data compression without requiring an update to every single product definition.
                # This automatic override feature was specifically requested by Heather
                warnings.warn(
                    f"Overwriting encoding '{k}': replacing '{encoding[k]}' with '{v}' from defaults. To suppress "
                    f"this warning, set the encoding value to '{v}' in your product definition.",
                    UserWarning,
                )
        return {**encoding, **DEFAULT_ENCODING}

    @property
    def static_attributes(self) -> dict:
        """Variable level attributes defined with non-null values in the product definition YAML

        Returns
        -------
        dict
            Dictionary of static variable level attributes with their defined values
        """
        return {k: v for k, v in self.attributes.items() if v is not None}

    @property
    def dynamic_attributes(self) -> dict:
        """Variable level attributes defined with null values in the product definition YAML.

        These attributes are _required_ but are expected to be passed explicitly during data product creation.

        Returns
        -------
        dict
            Dictionary of dynamic variable level attributes with null values that must be set during product creation.
        """
        return {k: v for k, v in self.attributes.items() if v is None}

    def _check_data_array_attributes(self, data_array_attrs: dict[str, Any], variable_name: str) -> list[str]:
        """Validate the variable level attributes of a DataArray against the product definition

        All attributes must have values. Static attributes defined in the product definition must match exactly. Dynamic
        attributes defined in the product definition may have any value but must be present.

        Parameters
        ----------
        data_array_attrs : dict[str, Any]
            DataArray attributes to validate
        variable_name : str
            Name of the variable being checked (for error messages)

        Returns
        -------
        list[str]
            List of error messages describing problems found. Empty list if no problems.
        """
        error_messages = []

        # Check for presence of expected attributes
        missing_variable_attributes = [k for k in self.attributes if k not in data_array_attrs]
        extra_variable_attributes = [k for k in data_array_attrs if k not in self.attributes]
        null_variable_attributes = [k for k, v in data_array_attrs.items() if v is None]

        if missing_variable_attributes:
            for attr in missing_variable_attributes:
                _err_msg = f"{variable_name}: missing attribute - Expected attribute '{attr}' not found"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        if extra_variable_attributes:
            for attr in extra_variable_attributes:
                _err_msg = f"{variable_name}: extra attribute - Unexpected attribute '{attr}' found"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        if null_variable_attributes:
            for attr in null_variable_attributes:
                _err_msg = (
                    f"{variable_name}: null attribute - Attribute '{attr}' has null value. This probably means "
                    "you forgot to pass a required dynamic attribute during product creation."
                )
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        # Check for value mismatches (only check static attributes from definition, allow dynamic attributes from user)
        for k, v in self.attributes.items():
            if (
                v is not None
                and k in data_array_attrs
                and type(data_array_attrs[k]) is type(v)
                and data_array_attrs[k] != v
            ):
                _err_msg = f"{variable_name}: attribute value mismatch - Expected {k}={v} but got {data_array_attrs[k]}"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        return error_messages

    def check_data_array_conformance(self, data_array: DataArray, variable_name: str) -> list[str]:
        """Check the conformance of a DataArray object against a data variable definition.

        This method is responsible only for finding errors, not fixing them. It warns on every violation and returns a
        list of error messages.

        Notes
        -----
        This does not verify that all required coordinate data exists on the DataArray. Dimensions lacking coordinates
        are treated as index dimensions. If coordinate data is later added to a Dataset under a dimension of the same
        name, the dimension will reference that coordinate data.

        Parameters
        ----------
        data_array: DataArray
            The data array to validate with this variable's metadata configuration.
        variable_name: str
            Name of the variable being checked (for error messages)

        Returns
        -------
        list[str]
            List of error messages describing problems found. Empty list if no problems.
        """
        error_messages = []

        # Check variable level attributes match product definition
        attrs_errors = self._check_data_array_attributes(data_array.attrs, variable_name)
        error_messages.extend(attrs_errors)

        # Check dimension names and ordering match product definition
        if list(self.dimensions) == list(data_array.sizes.keys()):
            for dim in self.dimensions:
                if dim.upper() not in LiberaVariableDefinition._standard_allowed_dimensions:
                    _err_msg = f"{variable_name}: undefined dimension '{dim}' - Dimension '{dim}' is not defined in the global standard dimensions"
                    warnings.warn(_err_msg)
                    error_messages.append(_err_msg)
                    continue
                expected_size = LiberaVariableDefinition._standard_allowed_dimensions[dim.upper()].size
                actual_size = data_array.sizes[dim]
                if expected_size is not None and expected_size != actual_size:
                    _err_msg = f"{variable_name}: dimension size mismatch for dimension '{dim}' - Expected size {expected_size} but got {actual_size}"
                    warnings.warn(_err_msg)
                    error_messages.append(_err_msg)
        else:
            _err_msg = f"{variable_name}: dimension mismatch - Expected dimensions {self.dimensions} but got {list(data_array.dims)}. Order matters too!"
            warnings.warn(_err_msg)
            error_messages.append(_err_msg)

        # Check encoding specification contains the specification from the product definition Because encodings are very
        # complex, we don't require every encoding setting to be specified in the product definition but any that are
        # specified must match exactly. We allow extra encoding settings on the DataArray that may not be present in the
        # product definition.
        encoding_mismatches = [
            k for k, v in self.encoding.items() if k not in data_array.encoding or data_array.encoding[k] != v
        ]
        for field in encoding_mismatches:
            expected_val = self.encoding.get(field)
            found_val = data_array.encoding.get(field, "not set")
            _err_msg = (
                f"{variable_name}: encoding mismatch - Expected encoding['{field}']={expected_val} but got {found_val}"
            )
            warnings.warn(_err_msg)
            error_messages.append(_err_msg)

        # Check data dtype matches product definition
        if str(data_array.dtype) != str(self.dtype):
            _err_msg = f"{variable_name}: dtype mismatch - Expected {self.dtype} but got {data_array.dtype}. Data type matters for proper NetCDF storage!"
            warnings.warn(_err_msg)
            error_messages.append(_err_msg)

        return error_messages

    def enforce_data_array_conformance(self, data_array: DataArray, variable_name: str) -> DataArray:
        """Update a variable or coordinate DataArray to conform to specifications in data product definition.

        This method attempts to bring a DataArray into conformance with a variable definition. When making changes, the
        data variable definition takes precedence over any existing metadata or settings on the DataArray. Logs are
        emitted for all changes made. When the DataArray configuration contradicts the data product definition, warnings
        are also issued. This method is not responsible for validating the final result and does not guarantee that the
        resulting DataArray will pass the validation checks because some problems simply can't be fixed.

        Parameters
        ----------
        data_array : DataArray
            The variable data array to analyze and update
        variable_name : str
            Name of the variable being enforced (for logging)

        Returns
        -------
        DataArray
            The updated DataArray. This DataArray is not guaranteed to be fully conformant and should be checked with
            `check_data_array_conformance` after enforcement to verify.

        Warns
        -----
        UserWarning
            If any conflicts are found between the DataArray and the product definition attributes or encoding settings.

        Raises
        ------
        ValueError
            Raise for problems that can't be fixed.
        """
        logger.info(f"Enforcing DataArray conformance to variable definition for variable '{variable_name}'")
        # Ensure all static variable attributes match product definition
        for key, value in self.static_attributes.items():
            if key not in data_array.attrs:
                logger.debug(f"Added missing static attribute to '{variable_name}' as '{key}:{value}'")
                data_array.attrs[key] = value
            elif data_array.attrs[key] != value:
                # This is fixable but should be fixed by the user because because they have attempted to set a static
                # attribute with the incorrect value. Issue a warning and a warning level log message.
                old_value = data_array.attrs[key]
                data_array.attrs[key] = value
                warnings.warn(
                    f"Variable {variable_name} attribute value mismatch for '{key}': Expected '{value}' but got '{old_value}'. "
                    "Fix this attribute value in the input DataArray or update your product definition."
                )
                logger.warning(
                    f"Updated static variable attribute '{key}' of '{variable_name}' from '{old_value}' to '{value}'"
                )

        # Remove extra attributes
        extra_attrs = [k for k in data_array.attrs.keys() if k not in self.attributes]
        for key in extra_attrs:
            # This is fixable but should be fixed by the user because they have attempted to set an attribute that is
            # not defined in the product definition. Issue a warning and a warning level log message.
            old_value = data_array.attrs[key]
            del data_array.attrs[key]
            warnings.warn(
                f"Variable {variable_name} has unexpected extra attribute '{key}' with value '{old_value}' that is not "
                "defined in the product definition. Remove this attribute from the input DataArray or update your product definition."
            )
            logger.warning(f"Removed unexpected attribute '{key}' from '{variable_name}' with value '{old_value}'")

        # Allow conservative automatic type casting
        # We allow automatic conversion of dtypes when the conversion is between dtypes of the same 'kind' and it is
        # safe to cast from the array dtype to the product definition dtype. This is either allowed quietly or raises
        # an exception
        current_dtype = np.dtype(data_array.dtype)
        expected_dtype = np.dtype(self.dtype)
        if current_dtype != expected_dtype:
            # Only permit automatic dtype conversions for safe castings within the same kind type.
            # e.g. float to int is forbidden but datetime64[us] to datetime64[us] is allowed.
            safe_casting = bool(
                np.can_cast(current_dtype, expected_dtype, casting="safe")
                and np.can_cast(current_dtype, expected_dtype, casting="same_kind")
            )
            if not safe_casting:
                raise ValueError(
                    f"Variable array for {variable_name} has dtype '{current_dtype}' that cannot be safely "
                    f"converted to expected dtype '{expected_dtype}' defined in the product definition. "
                    "Allowed conversions are limited to safe castings within the same kind type. e.g. float32->float64 "
                    "is allowed but int8->float64 is not. "
                    "Fix the dtype of this variable in the input DataArray (or numpy array) or update your product definition."
                )

            logger.info(f"Safely casting dtype of '{variable_name}' from {current_dtype} to {expected_dtype}.")
            try:
                previous_encoding = data_array.encoding.copy()
                data_array = data_array.astype(expected_dtype)
                # After conversion, re-apply encoding from product definition because dtype conversion may have changed it
                data_array.encoding = previous_encoding
            except Exception as e:
                raise ValueError(
                    f"Could not convert dtype of '{variable_name}' from {current_dtype} to {expected_dtype}"
                ) from e

        # Update encoding configuration to match product definition
        # Settings that are set on the DataArray but not in the product definition are removed and cause warnings
        extra_encoding_settings = [k for k in data_array.encoding if k not in self.encoding]
        for key in extra_encoding_settings:
            old_value = data_array.encoding[key]
            del data_array.encoding[key]
            warnings.warn(
                f"Variable {variable_name} has unexpected extra encoding setting '{key}' with value '{old_value}' that is not "
                "defined in the product definition. Remove this encoding setting from the input DataArray or update your product definition."
            )
            logger.warning(
                f"Removed unexpected encoding setting '{key}' from '{variable_name}' with value '{old_value}'"
            )

        # Settings that are different in the DataArray and product definition are overwritten by product definition
        # and warnings are issued
        conflicting_encoding_settings = [
            k for k, v in self.encoding.items() if k in data_array.encoding and data_array.encoding[k] != v
        ]
        for key in conflicting_encoding_settings:
            expected_value = self.encoding[key]
            old_value = data_array.encoding[key]
            data_array.encoding[key] = expected_value
            warnings.warn(
                f"Variable {variable_name} has encoding setting '{key}' with value '{old_value}' that conflicts with "
                f"the expected value '{expected_value}' defined in the product definition. "
                "Fix this encoding setting in the input DataArray or update your product definition."
            )
            logger.warning(
                f"Updated encoding setting '{key}' of '{variable_name}' from '{old_value}' to '{expected_value}'"
            )

        # Settings that are in the product definition but not the DataArray are simply added without warning
        # because encodings are complicated and adding them explicitly on every variable is extremely onerous
        data_array.encoding.update(self.encoding)

        # Check dimensions. We can't fix this so we just raise.
        if undefined_dims := [d for d in data_array.sizes.keys() if d.upper() not in self._standard_allowed_dimensions]:
            raise ValueError(
                f"Variable {variable_name} has undefined dimensions {undefined_dims} that are not in the standard "
                f"allowed dimensions {list(self._standard_allowed_dimensions)}. "
                "Fix the dimensions of this variable in the input DataArray."
            )

        if list(self.dimensions) != list(data_array.sizes.keys()):
            raise ValueError(
                f"Variable dimensions do not match product definition and cannot be automatically fixed. "
                f"Expected dimensions {list(self.dimensions)}, got {list(data_array.sizes.keys())}. "
                "Fix the dimensions of this variable in the input DataArray or update your product definition."
            )

        return data_array

    def create_variable_data_array(
        self, data: np.ndarray, variable_name: str, dynamic_variable_attributes: dict[str, Any] | None = None
    ) -> DataArray:
        """Create a DataArray for a single variable from a numpy array.

        Sets encoding and attributes from product definition, adding dynamic attributes if provided.

        Coordinate data is not required. Dimensions that reference coordinate dimensions are created as index
        dimensions. If coordinate data is added later (e.g. to a Dataset), these dimensions will reference the
        coordinates based on dimension name matching coordinate name.

        Parameters
        ----------
        data : np.ndarray
            Data for the variable DataArray.
        variable_name : str
            Name of the variable. Used for log messages and warnings.
        dynamic_variable_attributes : dict[str, Any] | None
            *Algorithm developers should not need to use this kwarg.* Variable level attributes defined by the user.
            This allows a user to specify dynamic attributes that may be required by the definition but not statically
            defined in yaml.

        Returns
        -------
        DataArray
            A minimal DataArray for the specified variable. This DataArray may not be fully conformant to the product
            definition. To bring it into conformance, use `enforce_dataset_conformance` on a Dataset containing this
            DataArray.
        """
        # Use product definition to set variable attributes
        if dynamic_variable_attributes is not None:
            variable_attrs = {**self.attributes, **dynamic_variable_attributes}
        else:
            variable_attrs = self.attributes

        da = DataArray(data=data, dims=self.dimensions, attrs=variable_attrs)

        # Use product definition to set encoding on the DataArray
        da.encoding = self.encoding.copy()
        logger.debug(f"Created DataArray for variable {variable_name} from numpy array.")

        return da


class LiberaDataProductDefinition(BaseModel):
    """
    Pydantic model for a Libera data product definition.

    Used for validating existing data product Datasets with helper methods for creating valid Datasets and DataArrays.

    Attributes
    ----------
    data_variables: dict[str, LiberaVariable]
        A dictionary of variable names and their corresponding LiberaVariable objects, which contain metadata and data.
    product_metadata: ProductMetadata | None
        The metadata associated with the data product, including dynamic metadata and spatio-temporal metadata.
    """

    model_config = ConfigDict(frozen=True)

    _standard_product_attributes: ClassVar[dict[str, Any]] = dict()

    coordinates: dict[str, LiberaVariableDefinition]
    variables: dict[str, LiberaVariableDefinition]
    attributes: dict[str, Any]

    @staticmethod
    def _get_static_project_attributes(
        file_path=None,
    ) -> dict[str, Any]:
        """Loads project-wide consistent product-level attribute metadata from a YAML file.

        These global attributes are expected on every Libera data product so we store them in a global config.

        Parameters
        ----------
        file_path: Path
            The path to the global attribute metadata YAML file.

        Returns
        -------
        dict[str, Any]
            Dictionary of key-value pairs for static product attributes.
        """
        if file_path is None:
            file_path = Path(str(config.get("LIBERA_GLOBAL_PRODUCT_ATTRIBUTES_PATH")))
        with AnyPath(file_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @field_validator("attributes", mode="before")
    @classmethod
    def _set_attributes(cls, raw_attributes: dict[str, Any]) -> dict[str, Any]:
        """Validates product level attributes and adds requirements for globally consistent attributes.

        Any attributes defined with null values are treated as required dynamic attributes that must be set either
        by the user's data product definition or dynamically on the Dataset before writing.

        Parameters
        ----------
        raw_attributes : dict[str, Any]
            The attributes specification in the product definition.

        Returns
        -------
        dict[str, Any]
            The validated attributes dictionary, including standard defaults that we always require.
        """
        if not cls._standard_product_attributes:
            cls._standard_product_attributes = cls._get_static_project_attributes()

        # Check for value conflicts between standard attributes and product definition attributes. This is an error.
        conflicts = [
            # For standard global attributes *that have non-null values*, check that values match
            k
            for k, v in cls._standard_product_attributes.items()
            if v and k in raw_attributes and v != raw_attributes[k]
        ]
        if conflicts:
            # This is an exception because the user is hard-coding conflicting standard attribute values in their
            # product definition YAML.
            conflicts_expected = {k: v for k, v in cls._standard_product_attributes.items() if k in conflicts}
            conflicts_provided = {k: v for k, v in raw_attributes.items() if k in conflicts}
            raise ValueError(
                "Conflicting standard product attributes detected. "
                f"Expected {conflicts_expected} but got {conflicts_provided}. "
                "Simply remove the conflicting attributes from your product definition YAML. "
                "Standard values will be automatically populated."
            )
        # Standard attributes with null values are required but must be set by the user
        null_standard_attributes = {k: v for k, v in cls._standard_product_attributes.items() if v is None}
        # Standard attributes with non-null values are required exactly (conflicts checked above)
        non_null_standard_attributes = {k: v for k, v in cls._standard_product_attributes.items() if v is not None}
        # Null standard attributes are overridden by user-specified attributes if provided in the product definition
        # and further overridden by non-null statically defined attribute values. Any attributes that still have
        # null values at this point are dynamic attributes and must be set during product creation.
        final_attributes = {**null_standard_attributes, **raw_attributes, **non_null_standard_attributes}

        # Validate that ProductID is statically defined and is a valid DataProductIdentifier
        product_id = final_attributes.get("ProductID")
        if product_id is None:
            raise ValueError(
                "ProductID must be statically defined in the product definition YAML file and must be a valid "
                f"DataProductIdentifier. Allowed values: {[e.value for e in DataProductIdentifier]}"
            )
        try:
            DataProductIdentifier(product_id)
        except ValueError:
            raise ValueError(
                f"ProductID '{product_id}' is not a valid DataProductIdentifier. "
                "ProductID must be statically defined in the product definition YAML file and must be a valid "
                f"DataProductIdentifier. Allowed values: {[e.value for e in DataProductIdentifier]}"
            )

        dynamic_attributes = [k for k, v in final_attributes.items() if v is None]
        if dynamic_attributes:
            logger.info(
                f"Dynamic attributes: {dynamic_attributes}. These attribute value must be set explicitly "
                "during product creation, otherwise subsequent conformance checks will fail. This is just a reminder!"
            )
        return final_attributes

    @classmethod
    def from_yaml(
        cls,
        product_definition_filepath: str | PathType,
    ):
        """Create a DataProductDefinition from a Libera data product definition YAML file.

        Parameters
        ----------
        product_definition_filepath: str | PathType
            Path to YAML file with product and variable definitions

        Returns
        -------
        DataProductDefinition
            Configured instance with loaded metadata and optional data
        """
        _path = cast(PathType, AnyPath(product_definition_filepath))
        with _path.open("r") as f:
            logger.info(f"Creating product definition model from file {product_definition_filepath}")
            yaml_data = yaml.safe_load(f)
            return cls(**yaml_data)

    @property
    def static_attributes(self):
        """Return product-level attributes that are statically defined (have values) in the data product definition"""
        return {k: v for k, v in self.attributes.items() if v is not None}

    @property
    def dynamic_attributes(self):
        """Return product-level attributes that are dynamically defined (null values) in the data product definition

        These attributes are _required_ but are expected to be defined externally to the data product definition
        """
        return {k: v for k, v in self.attributes.items() if v is None}

    def generate_data_product_filename(self, dataset: Dataset, time_variable: str) -> LiberaDataProductFilename:
        """Generate a standardized Libera data product filename.

        Parameters
        ----------
        dataset : Dataset
            The Dataset for which to create a filename. Used to extract algorithm version and start and end times.
        time_variable : str
            Name of the time dimension to use for determining the start and end time.

        Returns
        -------
        LiberaDataProductFilename
            Properly formatted filename object
        """
        # Convert numpy.datetime64 to Python datetime for filename generation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=r"Discarding nonzero nanoseconds in conversion"
            )
            utc_start = pd.Timestamp(dataset[time_variable].values[0]).to_pydatetime()
            utc_end = pd.Timestamp(dataset[time_variable].values[-1]).to_pydatetime()

        return LiberaDataProductFilename.from_filename_parts(
            product_name=DataProductIdentifier(dataset.attrs["ProductID"]),
            version=dataset.attrs["algorithm_version"],
            utc_start=utc_start,
            utc_end=utc_end,
        )

    def _check_dataset_attrs(self, dataset_attrs: dict[str, Any]) -> list[str]:
        """Validate the product level attributes of a Dataset against the product definition

        Static attributes must match exactly. Some special attributes have their values checked for validity.

        Parameters
        ----------
        dataset_attrs : dict[str, Any]
            Dataset attributes to validate

        Returns
        -------
        list[str]
            List of error messages describing problems found. Empty list if no problems.
        """
        error_messages = []

        # Check for presence of expected attributes
        missing_product_level_attributes = [k for k in self.attributes if k not in dataset_attrs]
        extra_product_level_attributes = [k for k in dataset_attrs if k not in self.attributes]
        null_product_level_attributes = [k for k, v in dataset_attrs.items() if v is None]

        if missing_product_level_attributes:
            for attr in missing_product_level_attributes:
                _err_msg = f"PRODUCT: missing attribute - Expected attribute '{attr}' not found in dataset attributes"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        if extra_product_level_attributes:
            for attr in extra_product_level_attributes:
                _err_msg = f"PRODUCT: extra attribute - Unexpected attribute '{attr}' found in dataset attributes"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        if null_product_level_attributes:
            for attr in null_product_level_attributes:
                _err_msg = f"PRODUCT: null attribute - Attribute '{attr}' has null value"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        # Check for value mismatches
        for k, v in self.attributes.items():
            if v and k in dataset_attrs and type(dataset_attrs[k]) is type(v) and dataset_attrs[k] != v:
                _err_msg = f"PRODUCT: attribute value mismatch - Expected {k}={v} but got {dataset_attrs[k]}"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        # Check some attribute values for validity using custom logic
        # NOTE: If we find that we are adding code here frequently to do validation on attribute values,
        # refactor this into a more generic system.
        if "algorithm_version" in dataset_attrs and dataset_attrs["algorithm_version"] is not None:
            # Check that algorithm_version strictly follows semantic versioning
            if not ALGORITHM_VERSION_REGEX.match(dataset_attrs["algorithm_version"]):
                _err_msg = f"PRODUCT: algorithm_version: invalid format - Expected semantic versioning (e.g., 1.0.0), got {dataset_attrs['algorithm_version']}"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        if "ProductID" in dataset_attrs and dataset_attrs["ProductID"] is not None:
            try:
                DataProductIdentifier(dataset_attrs["ProductID"])
            except ValueError:
                _err_msg = (
                    f"PRODUCT: ProductID: invalid value - '{dataset_attrs['ProductID']}' is not a valid "
                    f"DataProductIdentifier. Valid values: {[e.value for e in DataProductIdentifier]}"
                )
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)

        return error_messages

    def check_dataset_conformance(self, dataset: Dataset, strict: bool = True) -> list[str]:
        """Check the conformance of a Dataset object against a data product definition.

        This method is responsible only for finding errors, not fixing them. It warns on every violation and logs all
        errors it finds at the end. If strict is True, it raises an exception if any errors are found. If strict is
        False, it just returns the list of error messages.

        Parameters
        ----------
        dataset : Dataset
            Dataset object to validate against expectations in the product configuration
        strict : bool
            Default True. Raises an exception for nonconformance.

        Returns
        -------
        list[str]
            List of error messages describing problems found. Empty list if no problems.
        """
        error_messages = []

        # Check product level attributes against definition
        attrs_errors = self._check_dataset_attrs(dataset.attrs)
        error_messages.extend(attrs_errors)

        # Check each coordinate
        logger.info("Checking Dataset coordinates against product definition.")
        for coord_name, coord_def in self.coordinates.items():
            if coord_name not in dataset.coords:
                _err_msg = f"{coord_name}: missing coordinate - Expected coordinate '{coord_name}' not found in dataset"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)
                continue
            coord_errors = coord_def.check_data_array_conformance(dataset[coord_name], coord_name)
            error_messages.extend(coord_errors)

        # Check each variable
        logger.info("Checking Dataset variables against product definition.")
        for var_name, var_def in self.variables.items():
            if var_name not in dataset.data_vars:
                _err_msg = f"{var_name}: missing variable - Expected variable '{var_name}' not found in dataset"
                warnings.warn(_err_msg)
                error_messages.append(_err_msg)
                continue
            var_errors = var_def.check_data_array_conformance(dataset[var_name], var_name)
            error_messages.extend(var_errors)

        if error_messages:
            for msg in error_messages:
                logger.error(msg)
            if strict:
                raise ValueError(
                    "Errors detected during dataset conformance check. See previous logs and warnings for violations. "
                    "For testing you can run with strict=False to prevent this exception from raising and instead return a list of error messages."
                )

        return error_messages

    def enforce_dataset_conformance(self, dataset: Dataset) -> Dataset:
        """Analyze and update a Dataset to conform to the expectations of the DataProductDefinition

        This method attempts to bring a Dataset into conformance with a product definition, including enforcing
        conformance of variable DataArrays. When making changes, the data product definition takes precedence over any
        existing metadata or settings on the Dataset. Logs are emitted for all changes made. When the Dataset
        configuration contradicts the data product definition, warnings are also issued. This method is not responsible
        for validating the final result and does not guarantee that the resulting Dataset will pass the validation
        checks because some problems simply can't be fixed.

        Parameters
        ----------
        dataset : Dataset
            Possibly non-compliant dataset

        Returns
        -------
        Dataset
            The updated Dataset. This Dataset is not guaranteed to be fully
            conformant and should be checked with check_dataset_conformance to verify.
        """
        logger.info("Enforcing dataset conformance to product definition.")
        # Enforce global static attributes
        # We can't enforce global dynamic attributes (they are simply checked in check_dataset_conformance)
        for key, value in self.static_attributes.items():
            if key not in dataset.attrs:
                dataset.attrs[key] = value
                # This is acceptable without a warning because it is reasonable to populate static attributes directly
                # from the product definition if they are missing from the input Dataset.
                logger.debug(f"Added missing global static attribute '{key}': {value}")
            elif dataset.attrs[key] != value:
                old_value = dataset.attrs[key]
                dataset.attrs[key] = value
                warnings.warn(
                    f"Dataset attribute value mismatch for '{key}': Expected '{value}' but got '{old_value}'. "
                    "Fix this attribute value in the input Dataset or update your product definition."
                )
                logger.warning(f"Overwrote global static attribute '{key}' from '{old_value}' to '{value}'")

        # Remove extra attributes
        extra_attrs = [k for k in dataset.attrs.keys() if k not in self.attributes]
        for key in extra_attrs:
            old_value = dataset.attrs[key]
            del dataset.attrs[key]
            warnings.warn(
                f"Dataset has unexpected attribute '{key}' with value '{old_value}' that is not defined in the "
                "product definition. Remove this attribute from the input Dataset or update your product definition."
            )
            logger.warning(f"Removed unexpected global attribute '{key}' with value '{old_value}'")

        # Process all coordinates and variables with same logic
        all_vars = {**self.coordinates, **self.variables}

        for name, var_def in all_vars.items():
            if name not in dataset:
                # Can't do anything about this. Guaranteed the dataset will fail validation checks
                continue

            # Use the Variable class method to enforce conformance for each variable
            dataset[name] = var_def.enforce_data_array_conformance(dataset[name], name)

        # Return the updated dataset; validation should be performed separately
        return dataset

    def create_product_dataset(
        self,
        data: dict[str, np.ndarray],
        dynamic_product_attributes: dict[str, Any] | None = None,
        dynamic_variable_attributes: dict[str, dict[str, Any]] | None = None,
    ) -> Dataset:
        """Create a product Dataset from numpy arrays.

        This method creates a Dataset from numpy arrays, setting attributes and encodings according to the product
        definition. This does not guarantee a fully conformant Dataset. To bring the Dataset into conformance, use
        `enforce_dataset_conformance` on the resulting Dataset and check the result with `check_dataset_conformance`.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Dictionary of variable/coordinate data keyed by variable/coordinate name.
        dynamic_product_attributes : dict[str, Any] | None
            *Algorithm developers should not need to use this kwarg.* Product level attributes for the data product.
            This allows the user to specify product level attributes that are required but not statically specified in
            the product definition (e.g. the algorithm version used to generate the product)
        dynamic_variable_attributes : dict[str, dict[str, Any]] | None
            *Algorithm developers should not need to use this kwarg.* Per-variable attributes for each variable's
            DataArray. Key is variable name, value is an attributes dict. This allows the user to specify variable level
            attributes that are required but not statically defined in the product definition.

        Returns
        -------
        Dataset
            The created Dataset. This Dataset is not guaranteed to be conformant and should be checked with
            `check_dataset_conformance`.

        Notes
        -----
        - We make no distinction between coordinate and data variable input data and determine which
          is which based on coordinate/variable sections in the product definition.
        - This method is not responsible for primary validation or error reporting. The caller is responsible for
          checking the result with `check_dataset_conformance` and fixing any errors that arise.
        """
        logger.info("Creating product Dataset from numpy data arrays.")
        if dynamic_product_attributes is not None:
            product_attrs = {**self.attributes, **dynamic_product_attributes}
        else:
            product_attrs = self.attributes

        # Initialize Dataset object - first create coordinates, then data variables
        coords_dict = {}
        data_vars_dict = {}

        for var_name, var_data in data.items():
            if dynamic_variable_attributes is not None and var_name in dynamic_variable_attributes:
                var_attrs = dynamic_variable_attributes[var_name]
            else:
                var_attrs = None

            if var_name in self.coordinates:
                var_def = self.coordinates[var_name]
                coords_dict[var_name] = var_def.create_variable_data_array(
                    var_data, var_name, dynamic_variable_attributes=var_attrs
                )
            elif var_name in self.variables:
                var_def = self.variables[var_name]
                data_vars_dict[var_name] = var_def.create_variable_data_array(
                    var_data, var_name, dynamic_variable_attributes=var_attrs
                )
            else:
                # This is such an obvious error that we raise immediately.
                raise ValueError(
                    f"Unknown variable/coordinate name {var_name}. Unable to create Dataset. "
                    "Check your product definition and input data variable names."
                )

        # Create Dataset with coords and data_vars properly separated
        ds = Dataset(data_vars=data_vars_dict, coords=coords_dict, attrs=product_attrs)

        return ds
