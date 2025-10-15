"""Data Product configuration and writing for Libera NetCDF4 data product files"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import yaml
from cloudpathlib import AnyPath
from pydantic import BaseModel, ConfigDict, Field, field_validator
from xarray import DataArray, Dataset

from libera_utils.config import config
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType, format_semantic_version
from libera_utils.version import ALGORITHM_VERSION_REGEX

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = {"zlib": True, "complevel": 4}

# TODO[LIBSDC-623]: Add a global Dimensions cache that gets populated on first request for a dimension by name


class LiberaVariableDefinition(BaseModel):
    """Pydantic model for a Libera variable definition.

    This model is the same for both data variables and coordinate variables

    Attributes
    ----------
    dtype: str
        The data type of the variable's data array, specified as a string
    attributes: VariableAttributes
        The attribute metadata for the variable, containing specific key value pairs for CF metadata compliance
    dimensions: list[LiberaDimension]
        A list of dimensions that the variable's data array references. These should be instances of LiberaDimension.
    encoding: dict
        A dictionary specifying how the variable's data should be encoded when written to a NetCDF file.
    """

    model_config = ConfigDict(frozen=True)

    dtype: str = Field(description="The data type of the variable's data array, specified as a string")
    attributes: dict[str, Any] = Field(default=dict(), description="Attribute metadata for the variable")
    dimensions: list[str] = Field(default=list(), description="Dimensions of the variable's data array")
    encoding: dict = Field(
        default_factory=lambda: DEFAULT_ENCODING.copy(),
        description="Encoding settings for the variable, determining how it is stored on disk",
    )

    @field_validator("encoding", mode="before")
    @classmethod
    def _set_encoding(cls, encoding: dict | None):
        """Merge configured encoding with required defaults, issuing warnings on conflicts."""
        if encoding is None:
            return DEFAULT_ENCODING.copy()
        for k, v in DEFAULT_ENCODING.items():
            if k in encoding and encoding[k] != v:
                warnings.warn(
                    f"Overwriting encoding '{k}': replacing '{encoding[k]}' with '{v}' from defaults", UserWarning
                )
        return {**encoding, **DEFAULT_ENCODING}

    @property
    def static_attributes(self):
        """Return attributes for a variable that are statically defined (have values) in the data product definition"""
        return {k: v for k, v in self.attributes.items() if v is not None}

    @property
    def dynamic_attributes(self):
        """Return attributes for a variable that are dynamically defined (null values) in the data product definition

        These attributes are _required_ but are expected to be defined externally to the data product definition
        """
        return {k: v for k, v in self.attributes.items() if v is None}

    def _check_data_array_attributes(self, data_array_attrs: dict[str, Any], variable_name: str) -> list[str]:
        """Validate the variable level attributes of a DataArray against the product definition

        Attributes must match exactly

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
                error_messages.append(f"{variable_name}: missing attribute - Expected attribute '{attr}' not found")
            logger.warning(f"Missing variable attributes: {missing_variable_attributes}")

        if extra_variable_attributes:
            for attr in extra_variable_attributes:
                error_messages.append(f"{variable_name}: extra attribute - Unexpected attribute '{attr}' found")
            logger.warning(f"Extra variable attributes: {extra_variable_attributes}")

        if null_variable_attributes:
            for attr in null_variable_attributes:
                error_messages.append(f"{variable_name}: null attribute - Attribute '{attr}' has null value")
            logger.warning(f"Some variable attributes are not set: {null_variable_attributes}")

        # Check for value mismatches (only check static attributes from definition, allow overrides from user)
        for k, v in self.attributes.items():
            if (
                v is not None
                and k in data_array_attrs
                and type(data_array_attrs[k]) is type(v)
                and data_array_attrs[k] != v
            ):
                error_messages.append(
                    f"{variable_name}: attribute value mismatch - Expected {k}={v} but got {data_array_attrs[k]}"
                )
                logger.warning(f"Variable attribute value mismatch for {k}. Expected {v} but got {data_array_attrs[k]}")

        return error_messages

    def check_data_array_conformance(self, data_array: DataArray, variable_name: str) -> list[str]:
        """Validate variable data array based on product definition.

        This does not verify that all required coordinate data exists on the DataArray.
        Dimensions lacking coordinates are treated as index dimensions. If coordinate data
        is later added to a Dataset under a dimension of the same name,
        the dimension will reference that coordinate data.

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
        if list(self.dimensions) != list(data_array.dims):
            # TODO[LIBSDC-623]: Add validation that dimension name exists in the global cache and
            #   for fixed dimensions, the size of the variable data matches the dimension size
            #   along the correct index
            error_messages.append(
                f"{variable_name}: dimension mismatch - Expected dimensions {self.dimensions} but got {list(data_array.dims)}"
            )
            warnings.warn(
                f"The provided data has dimensions {data_array.dims} but was expected to have "
                f"dimensions {self.dimensions}. Order matters too!"
            )

        # Check encoding specification matches product definition
        encoding_mismatches = [
            k for k, v in self.encoding.items() if k not in data_array.encoding or data_array.encoding[k] != v
        ]
        if encoding_mismatches:
            expected = {k: v for k, v in self.encoding.items() if k in encoding_mismatches}
            found = {k: v for k, v in data_array.encoding.items() if k in encoding_mismatches}
            for field in encoding_mismatches:
                expected_val = self.encoding.get(field)
                found_val = data_array.encoding.get(field, "not set")
                error_messages.append(
                    f"{variable_name}: encoding mismatch - Expected encoding['{field}']={expected_val} but got {found_val}"
                )
            warnings.warn(
                f"Detected encoding mismatches on data array in fields {encoding_mismatches}. "
                f"Expected {expected} but found {found}"
            )

        # Check data dtype matches product definition
        if str(data_array.dtype) != str(self.dtype):
            error_messages.append(f"{variable_name}: dtype mismatch - Expected {self.dtype} but got {data_array.dtype}")
            warnings.warn(
                f"The provided data has dtype {data_array.dtype} but was expected to have "
                f"dtype {self.dtype}. "
                f"Data type matters for proper NetCDF storage!"
            )

        return error_messages

    def enforce_data_array_conformance(self, data_array: DataArray, variable_name: str) -> tuple[DataArray, list[str]]:
        """Analyze and fix a DataArray to conform to variable specifications in data product definition

        Parameters
        ----------
        data_array : DataArray
            The variable data array to analyze and update
        variable_name : str
            Name of the variable being enforced (for logging)

        Returns
        -------
        tuple[DataArray, list[str]]
            Tuple of (updated DataArray, error_messages) where error_messages contains any problems
            that could not be fixed. Empty list if all problems were fixed.
        """
        # Fix static variable attributes (can't fix dynamic attributes but those are checked in check_data_array_conformance)
        for key, value in self.static_attributes.items():
            if key not in data_array.attrs:
                logger.debug(f"Added missing static attribute to '{variable_name}' as '{key}:{value}'")
                data_array.attrs[key] = value
            elif data_array.attrs[key] != value:
                old_value = data_array.attrs[key]
                data_array.attrs[key] = value
                logger.debug(
                    f"Updated static variable attribute '{key}' of '{variable_name}' from '{old_value}' to '{value}'"
                )

        # Remove extra attributes
        extra_attrs = [k for k in data_array.attrs.keys() if k not in self.attributes]
        for key in extra_attrs:
            old_value = data_array.attrs[key]
            del data_array.attrs[key]
            logger.debug(f"Removed unexpected attribute '{key}' from '{variable_name}' with value '{old_value}'")

        # Fix dtype if needed
        current_dtype = str(data_array.dtype)
        expected_dtype = str(self.dtype)
        if current_dtype != expected_dtype:
            try:
                logger.debug(f"Converting dtype of '{variable_name}' from {current_dtype} to {expected_dtype}")
                data_array = data_array.astype(expected_dtype)
            except Exception as e:
                logger.warning(
                    f"Could not convert dtype of '{variable_name}' from {current_dtype} to {expected_dtype}: {e}"
                )

        # Update encoding configuration
        for key, value in self.encoding.items():
            if key not in data_array.encoding or data_array.encoding[key] != value:
                old_value = data_array.encoding.get(key, "not set")
                data_array.encoding[key] = value
                logger.debug(f"Updated encoding '{key}' of '{variable_name}' from '{old_value}' to '{value}'")

        for key, value in data_array.encoding.items():
            if key not in self.encoding:
                del data_array.encoding[key]
                logger.debug(f"Removed unexpected encoding item: {key} with value {value}")

        # Run check_data_array_conformance to validate the modifications and report any unfixable errors
        validation_errors = self.check_data_array_conformance(data_array, variable_name)
        if validation_errors:
            logger.warning(
                f"Some problems could not be fixed! Variable DataArray validation errors after enforcement:\n"
                + "\n".join(validation_errors)
            )

        return data_array, validation_errors

    def create_conforming_data_array(
        self, data: np.ndarray, variable_name: str, user_variable_attributes: dict[str, Any] | None = None
    ) -> DataArray:
        """Create a DataArray for a single variable that is valid against the data product definition.

        Coordinate data is not required. Dimensions that reference coordinate dimensions are created as index
        dimensions. If coordinate data is added later (e.g. to a Dataset), these dimensions will reference the coordinates.

        Parameters
        ----------
        data : np.ndarray
            Data for the variable DataArray.
        variable_name : str
            Name of the variable. Used for log messages and warnings.
        user_variable_attributes : dict[str, Any] | None
            *Algorithm developers should not need to use this kwarg.*
            Variable level attributes defined by the user. This allows a user to specify dynamic attributes
            that may be required by the definition but not statically defined in yaml.

        Returns
        -------
        DataArray
            A valid DataArray for the specified variable
        """
        if user_variable_attributes is not None:
            variable_attrs = {**self.attributes, **user_variable_attributes}
        else:
            variable_attrs = self.attributes

        da = DataArray(data=data, dims=self.dimensions, attrs=variable_attrs)

        # Set encoding on the DataArray
        da.encoding = self.encoding.copy()

        # Only validate dimensions and data type
        if str(da.dtype) != str(self.dtype):
            try:
                old_dtype = da.dtype
                da = da.astype(self.dtype)
                warnings.warn(
                    f"Coerced variable data for {variable_name} from {old_dtype} to {self.dtype}. "
                    "If the incoming dtype was incorrect, this may not convert as you expect! "
                    "This especially affects conversions to datetime64 types. "
                    "Make sure your data dtypes match your product definition."
                )
            except Exception as e:
                raise ValueError(f"Could not convert data for {variable_name} to required dtype {self.dtype}: {e}")

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
    ):
        """Loads project-wide consistent product-level attribute metadata from a YAML file.

        These global attributes are expected on every Libera data product so we store them in a global config.

        Parameters
        ----------
        file_path: Path
            The path to the global attribute metadata YAML file.

        Returns
        -------
        dict
            Dictionary of key-value pairs for static product attributes.
        """
        if file_path is None:
            file_path = Path(str(config.get("LIBERA_UTILS_DATA_DIR"))) / "static_project_metadata.yml"
        with AnyPath(file_path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @field_validator("attributes", mode="before")
    @classmethod
    def _set_attributes(cls, raw_attributes: dict[str, Any]) -> dict[str, Any]:
        """Validates product level attributes and adds requirements for globally consistent attributes

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
        conflicts = [
            # For standard global attributes *that have non-null values*, check that values match
            k
            for k, v in cls._standard_product_attributes.items()
            if v and k in raw_attributes and v != raw_attributes[k]
        ]
        if conflicts:
            raise ValueError(
                f"Conflicting standard product metadata. These keys are reserved for standard attributes: {conflicts}"
            )
        non_null_standard_attributes = {k: v for k, v in cls._standard_product_attributes.items() if v is not None}
        return {**raw_attributes, **non_null_standard_attributes}

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

    def generate_data_product_filename(self, utc_start: datetime, utc_end: datetime) -> LiberaDataProductFilename:
        """Generate a standardized Libera data product filename.

        Parameters
        ----------
        utc_start: datetime
            Start time of data in the file
        utc_end: datetime
            End time of data in the file

        Returns
        -------
        LiberaDataProductFilename
            Properly formatted filename object
        """
        return LiberaDataProductFilename.from_filename_parts(
            product_name=DataProductIdentifier(self.attributes["ProductID"]),
            version=format_semantic_version(self.attributes["algorithm_version"]),
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
                error_messages.append(f"PRODUCT: missing attribute - Expected attribute '{attr}' not found")
            logger.warning(f"Missing product level attributes: {missing_product_level_attributes}")

        if extra_product_level_attributes:
            for attr in extra_product_level_attributes:
                error_messages.append(f"PRODUCT: extra attribute - Unexpected attribute '{attr}' found")
            logger.warning(f"Extra product level attributes: {extra_product_level_attributes}")

        if null_product_level_attributes:
            for attr in null_product_level_attributes:
                error_messages.append(f"PRODUCT: null attribute - Attribute '{attr}' has null value")
            logger.warning(f"Some product level attributes not set: {null_product_level_attributes}")

        # Check for value mismatches
        for k, v in self.attributes.items():
            if v and k in dataset_attrs and type(dataset_attrs[k]) is type(v) and dataset_attrs[k] != v:
                error_messages.append(
                    f"PRODUCT: attribute value mismatch - Expected {k}={v} but got {dataset_attrs[k]}"
                )
                logger.warning(f"Attribute value mismatch for {k}. Expected {v} but got {dataset_attrs[k]}")

        # Check some attribute values for validity using custom logic
        # NOTE: If we find that we are adding code here frequently to do validation on attribute values,
        # refactor this into a more generic system.
        if "algorithm_version" in dataset_attrs:
            # Check that algorithm_version strictly follows semantic versioning
            if not ALGORITHM_VERSION_REGEX.match(dataset_attrs["algorithm_version"]):
                error_messages.append(
                    f"PRODUCT: algorithm_version: invalid format - Expected semantic versioning (e.g., 1.0.0), got {dataset_attrs['algorithm_version']}"
                )
                logger.warning(
                    f"Invalid algorithm_version format: Expected semantic versioning (e.g., 1.0.0), got {dataset_attrs['algorithm_version']}"
                )

        return error_messages

    def check_dataset_conformance(self, dataset: Dataset, strict: bool = True) -> list[str]:
        """Check the conformance of a Dataset object against a DataProductDefinition

        This method is responsible only for finding errors, not fixing them.

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
        for coord_name, coord_def in self.coordinates.items():
            if coord_name not in dataset.coords:
                error_messages.append(
                    f"{coord_name}: missing coordinate - Expected coordinate '{coord_name}' not found in dataset"
                )
                logger.warning(f"Missing coordinate '{coord_name}' during validation")
                continue
            logger.debug(f"Validating coordinate data for '{coord_name}")
            coord_errors = coord_def.check_data_array_conformance(dataset[coord_name], coord_name)
            if coord_errors:
                logger.warning(f"Validation failed for coordinate {coord_name}")
            error_messages.extend(coord_errors)

        # Check each variable
        for var_name, var_def in self.variables.items():
            if var_name not in dataset.data_vars:
                error_messages.append(
                    f"{var_name}: missing variable - Expected variable '{var_name}' not found in dataset"
                )
                logger.warning(f"Missing variable '{var_name}' during validation")
                continue
            logger.debug(f"Validating variable data for '{var_name}'")
            var_errors = var_def.check_data_array_conformance(dataset[var_name], var_name)
            if var_errors:
                logger.warning(f"Validation failed for variable {var_name}")
            error_messages.extend(var_errors)

        if error_messages:
            logger.error(f"Errors detected during conformance check: {', '.join(error_messages)}")
        if strict and error_messages:
            raise ValueError(
                "Errors detected during dataset conformance check. See logs for failures. "
                "For testing you can run with strict=False to return error messages instead of raising."
            )

        return error_messages

    def enforce_dataset_conformance(self, dataset: Dataset) -> tuple[Dataset, list[str]]:
        """Analyze and update a Dataset to conform to the expectations of the DataProductDefinition

        This method is for modifying an existing xarray Dataset.
        If you are creating a Dataset from scratch with numpy arrays, consider using `create_conforming_dataset` instead.

        Parameters
        ----------
        dataset : Dataset
            Possibly non-compliant dataset

        Returns
        -------
        tuple[Dataset, list[str]]
            Tuple of (updated Dataset, error_messages) where error_messages contains any problems
            that could not be fixed. Empty list if all problems were fixed.

        Notes
        -----
        - This method is responsible for trying (and possibly failing) to coerce a Dataset
          into a valid form with attributes and encodings. We use check_dataset_conformance to check for validation errors.
        """
        # Enforce global static attributes
        # We can't enforce global dynamic attributes (they are simply checked in check_dataset_conformance)
        for key, value in self.static_attributes.items():
            if key not in dataset.attrs:
                dataset.attrs[key] = value
                logger.info(f"Added missing global static attribute '{key}': {value}")
            elif dataset.attrs[key] != value:
                old_value = dataset.attrs[key]
                dataset.attrs[key] = value
                logger.info(f"Overwrote global static attribute '{key}' from '{old_value}' to '{value}'")

        # Remove extra attributes
        extra_attrs = [k for k in dataset.attrs.keys() if k not in self.attributes]
        for key in extra_attrs:
            old_value = dataset.attrs[key]
            del dataset.attrs[key]
            logger.info(f"Removed unexpected global attribute '{key}' with value '{old_value}'")

        # Process all coordinates and variables with same logic
        all_vars = {**self.coordinates, **self.variables}

        for name, var_def in all_vars.items():
            if name not in dataset:
                # Can't do anything about this. Guaranteed the dataset will fail validation checks
                continue

            # Use the Variable class method to enforce conformance for each variable
            dataset[name], _ = var_def.enforce_data_array_conformance(dataset[name], name)

        # Run check_dataset_conformance to validate the modifications and report any unfixable errors
        validation_errors = self.check_dataset_conformance(dataset, strict=False)
        if validation_errors:
            logger.warning(
                f"Some problems could not be fixed! Dataset validation errors after enforcement:\n"
                + "\n".join(validation_errors)
            )

        # Return dataset and validation errors
        return dataset, validation_errors

    def create_conforming_dataset(
        self,
        data: dict[str, np.ndarray],
        user_product_attributes: dict[str, Any] | None = None,
        user_variable_attributes: dict[str, dict[str, Any]] | None = None,
        strict: bool = True,
    ) -> tuple[Dataset, list[str]]:
        """Create a Dataset from numpy arrays that is valid against the data product definition

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Dictionary of variable/coordinate data keyed by variable/coordinate name.
        user_product_attributes : dict[str, Any] | None
            *Algorithm developers should not need to use this kwarg.*
            Product level attributes for the data product. This allows the user to specify product level attributes that are required but
            not statically specified in the product definition (e.g. the algorithm version used to generate the product)
        user_variable_attributes : dict[str, dict[str, Any]] | None
            *Algorithm developers should not need to use this kwarg.*
            Per-variable attributes for each variable's DataArray. Key is variable name, value is an attributes dict.
            This allows the user to specify variable level attributes that
            are required but not statically defined in the product definition.
        strict : bool
            Default True. Raises an exception for nonconformance.

        Returns
        -------
        tuple[Dataset, list[str]]
            Tuple of (Dataset, error_messages) where error_messages contains any validation problems.
            Empty list if the dataset is fully valid.

        Notes
        -----
        - We make no distinction between coordinate and data variable input data and assume that we can
          determine which is which based on coordinate/variable names the product definition.
        - This method is not responsible for primary validation or error reporting.
          We call out to check_dataset_conformance at the end for that.
        """
        if user_product_attributes is not None:
            product_attrs = {**self.attributes, **user_product_attributes}
        else:
            product_attrs = self.attributes

        # Initialize Dataset object - first create coordinates, then data variables
        coords_dict = {}
        data_vars_dict = {}

        for var_name, var_data in data.items():
            if user_variable_attributes is not None and var_name in user_variable_attributes:
                var_attrs = user_variable_attributes[var_name]
            else:
                var_attrs = None

            if var_name in self.coordinates:
                var_def = self.coordinates[var_name]
                coords_dict[var_name] = var_def.create_conforming_data_array(
                    var_data, var_name, user_variable_attributes=var_attrs
                )
            elif var_name in self.variables:
                var_def = self.variables[var_name]
                data_vars_dict[var_name] = var_def.create_conforming_data_array(
                    var_data, var_name, user_variable_attributes=var_attrs
                )
            else:
                raise ValueError(f"Unknown variable/coordinate name {var_name}.")

        # Create Dataset with coords and data_vars properly separated
        ds = Dataset(data_vars=data_vars_dict, coords=coords_dict, attrs=product_attrs)

        error_messages = self.check_dataset_conformance(ds, strict=strict)

        return ds, error_messages
