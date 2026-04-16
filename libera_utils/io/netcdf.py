"""Module containing utilities for writing Libera-conforming NetCDF4 data products"""

import logging
from enum import StrEnum
from typing import Any, Literal, Dict

import xarray as xr
from cloudpathlib import AnyPath
from numpy.typing import NDArray

from libera_utils.config import config
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.product_definition import LiberaDataProductDefinition

T_XarrayNetcdfEngine = Literal["netcdf4", "h5netcdf"]

logger = logging.getLogger(__name__)


class NetcdfEngine(StrEnum):
    """String enum class for our allowed NetCDF engines for xarray

    The netcdf4 engine does not support writing to filelike objects (e.g. S3 objects via cloudpathlib).
    The h5netcdf engine does support writing to filelike objects.
    The netcdf4 engine is required for use with the Dask distributed client, since
    the h5netcdf engine does not manage the necessary multiprocessing lock.
    """

    netcdf4 = "netcdf4"
    h5netcdf = "h5netcdf"

    @classmethod
    def get_from_config(cls) -> T_XarrayNetcdfEngine:
        """Retrieve the current netcdf engine config from the package configuration"""
        return cls(config.get("XARRAY_NETCDF_ENGINE"))  # type: ignore[return-value]


def write_libera_data_product(
    data_product_definition: str | PathType | LiberaDataProductDefinition,
    data: dict[str, NDArray] | xr.Dataset,
    output_path: str | PathType,
    time_variable: str,
    dynamic_product_attributes: dict[str, Any] | None = None,
    strict: bool = True,
    add_archive_path_prefix: bool = False,
    netcdf_engine: NetcdfEngine | None = None,
    encoding: Dict[str, Any] | None = None,
) -> LiberaDataProductFilename:
    """Write a Libera data product NetCDF4 file that conforms to data product definition requirements

    Steps:

    1. (Numpy array input only) Create a product Dataset from the input arrays and optional dynamic attributes, using the product definition to determine the expected structure and metadata.
    2. Bring the Dataset into conformance with the product definition, fixing any issues that can be automatically fixed and issuing warnings for any issues that cannot be automatically fixed.
    3. Check the final conformance of the Dataset against the product definition, raising an exception if any issues are found in strict mode.
    4. Generate the data product filename using the product definition and the specified time variable.
    5. Write the Dataset to a NetCDF4 file at the specified output path with the generated filename, using the configured NetCDF engine.

    Parameters
    ----------
    data_product_definition : str | PathType | LiberaDataProductDefinition
        Path to the data product definition against which to verify conformance
    data : dict[str, NDarray] | xr.Dataset
        Data mapping variable names to numpy data arrays or a fully formed L1A xarray Dataset.
    output_path : str | PathType
        Base path (directory or S3 prefix) at which to write the product file
    time_variable : str
        Name of variable that indicates time. This is used to generate the start and end time for the filename.
    dynamic_product_attributes : dict[str, Any] | None
        Optional dictionary of additional global attributes to add to the data product file. Must conform to the data
        product definition.
    strict : bool
        Default True. Raises an exception if the final Dataset doesn't conform to the data product definition.
    add_archive_path_prefix : bool
        Note: do not use this to write to a processing dropbox! L2 devs do not need this kwarg. Default False. If True,
        adds the archive path prefix to the output path when generating the full output path.
    netcdf_engine : NetcdfEngine
        NetCDFEngine to use for writing the file. Defaults to the value in the package configuration.
    encoding : Dict[str, Any] | None
        Encoding passed to to_netcdf, default None.

    Returns
    -------
    : LiberaDataProductFilename
        Filename object containing the full path to the written NetCDF4 data product file.
    """
    logger.info("Writing Libera data product")

    if isinstance(data_product_definition, LiberaDataProductDefinition):
        logger.info("Using provided LiberaDataProductDefinition object")
        definition = data_product_definition
    else:
        logger.info(f"Loading data product definition from {data_product_definition}")
        definition = LiberaDataProductDefinition.from_yaml(data_product_definition)

    logger.info(f"Product definition defines product level attributes: {definition.attributes}")

    if isinstance(data, xr.Dataset):
        if dynamic_product_attributes is not None:
            raise ValueError(
                "dynamic_product_attributes is invalid when passing in a Dataset. To set dynamic attributes for a dataset, modify the Dataset attrs before passing it in."
            )
        # This is how L1A products are typically created (starting as a Dataset)
        dataset = data
    else:
        # This is the expectation for L2 product creation (from numpy arrays)
        logger.info(f"Creating Dataset from data arrays with variables: {list(data.keys())}")
        dataset = definition.create_product_dataset(data, dynamic_product_attributes=dynamic_product_attributes)

    logger.info(f"Bringing Dataset into conformance with product definition")
    dataset = definition.enforce_dataset_conformance(dataset)

    logger.info("Checking final Dataset conformance against the product definition")
    definition.check_dataset_conformance(dataset, strict=strict)

    if "datetime64" not in str(dataset[time_variable].dtype):
        raise ValueError(f"Specified time variable {time_variable} does not have dtype datetime64.")

    data_product_filename = definition.generate_data_product_filename(dataset, time_variable)

    if add_archive_path_prefix:
        prefixed_path = AnyPath(output_path) / data_product_filename.archive_prefix
        prefixed_path.mkdir(parents=True, exist_ok=True)
        data_product_filename.path = prefixed_path / data_product_filename.path.name
    else:
        data_product_filename.path = AnyPath(output_path) / data_product_filename.path.name

    if netcdf_engine is None:
        netcdf_engine = NetcdfEngine.get_from_config()
    if netcdf_engine == NetcdfEngine.netcdf4:
        logger.info("Using netcdf4 engine to write data product, this will not work for S3 paths")
        dataset.to_netcdf(data_product_filename.path, engine=netcdf_engine, encoding=encoding)
    else:
        with data_product_filename.path.open("w+b") as fh:
            dataset.to_netcdf(fh, engine=NetcdfEngine.h5netcdf, encoding=encoding)
    return data_product_filename
