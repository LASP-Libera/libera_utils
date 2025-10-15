"""Module containing utilities for writing Libera-conforming NetCDF4 data products"""

from enum import StrEnum
from typing import Literal

import pandas as pd
from cloudpathlib import AnyPath
from numpy.typing import NDArray

from libera_utils.config import config
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.product_definition import LiberaDataProductDefinition

T_XarrayNetcdfEngine = Literal["netcdf4", "h5netcdf"]


class NetcdfEngine(StrEnum):
    """String enum class for our allowed NetCDF engines for xarray"""

    netcdf4 = "netcdf4"
    h5netcdf = "h5netcdf"

    @classmethod
    def get_from_config(cls) -> T_XarrayNetcdfEngine:
        """Retrieve the current netcdf engine config from the package configuration"""
        return cls(config.get("XARRAY_NETCDF_ENGINE"))  # type: ignore[return-value]


# TODO[LIBSDC-681]: Add UMM-G metadata file generation to this function call
def write_libera_data_product(
    data_product_definition: str | PathType,
    data: dict[str, NDArray],
    output_path: str | PathType,
    time_variable: str,
    strict: bool = True,
) -> LiberaDataProductFilename:
    """Write a Libera data product NetCDF4 file that conforms to data product definition requirements

    Parameters
    ----------
    data_product_definition : str | PathType
        Path to the data product definition against which to verify conformance
    data : dict[str, NDarray]
        Data mapping variable names to numpy data arrays
    output_path : str | PathType
        Base path (directory or S3 prefix) at which to write the product file
    time_variable : str
        Name of variable that indicates time. This is used to generate the start and end time for the filename.
    strict : bool
        Default True. Raises an exception if the final Dataset doesn't conform to the data product definition.

    Returns
    -------
    : LiberaDataProductFilename
        Filename object containing the full path to the written NetCDF4 data product file.
    """
    definition = LiberaDataProductDefinition.from_yaml(data_product_definition)
    dataset, _errors = definition.create_conforming_dataset(data, strict=strict)
    if "datetime64" not in str(dataset[time_variable].dtype):
        raise ValueError(f"Variable {time_variable} does not have dtype datetime64.")

    # Convert numpy.datetime64 to Python datetime for filename generation
    start = pd.Timestamp(dataset[time_variable].values[0]).to_pydatetime()
    end = pd.Timestamp(dataset[time_variable].values[-1]).to_pydatetime()
    data_product_filename = definition.generate_data_product_filename(utc_start=start, utc_end=end)
    data_product_filename.path = AnyPath(output_path) / data_product_filename.path.name

    netcdf4_engine = NetcdfEngine.get_from_config()
    dataset.to_netcdf(data_product_filename.path, engine=netcdf4_engine)
    return data_product_filename
