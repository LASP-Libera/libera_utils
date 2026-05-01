import argparse
import logging
import os
from pathlib import Path

import xarray as xr
from cloudpathlib import S3Path

from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.smart_open import smart_open
from libera_utils.io.umm_g import UMMGranule

logger = logging.getLogger(__name__)


def parse_cli_args():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing the data file path and options
    """
    parser = argparse.ArgumentParser(
        prog="libera-metadata-creator", description="Libera metadata-from-data-file creator"
    )

    parser.add_argument(
        "data_product_file_path", type=str, help="Absolute path to the NetCDF data file create metadata for"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    return parser.parse_args()


def read_input_netcdf4_data_file(data_file_path: str | Path | S3Path) -> xr.Dataset:
    """Read the input L1 or L2 data file and return its contents as an xarray Dataset

    Parameters
    ----------
    data_file_path : str | Path | S3Path
        Path to the NetCDF Libera data product file to read

    Returns
    -------
    xr.Dataset
        Contents of the data file as an xarray Dataset
    """

    try:
        with smart_open(data_file_path) as file_handle:
            # Load the NetCDF dataset
            dataset = xr.open_dataset(file_handle, engine="h5netcdf")
            logger.info(f"Successfully loaded dataset with variables: {list(dataset.variables)}")
            return dataset
    except Exception as e:
        logger.error(f"Failed to open file {data_file_path}: {e}")
        raise


def write_data_product_metadata_ummg(
    dataset: xr.Dataset,
    data_product_filename: LiberaDataProductFilename | str,
) -> PathType:
    """Write UMM-G metadata file for a Libera data product

    Parameters
    ----------
    dataset : xr.Dataset
        The xarray Dataset for which to create UMM-G metadata
    data_product_filename : str
        Filename of the data product, used to derive the UMM-G metadata filename

    Returns
    -------
    PathType
        Path to the written UMM-G metadata file.
    """
    logger.info("Writing UMM-G metadata for Libera data product")

    data_product_filename = LiberaDataProductFilename(data_product_filename)

    granule = UMMGranule.from_dataset(dataset, data_product_filename)
    ummg_filename = data_product_filename.ummg_metadata_filename

    with smart_open(ummg_filename, "w") as fh:
        fh.write(granule.model_dump_json(exclude_none=True))

    logger.info(f"Wrote UMM-G metadata file to {ummg_filename}")

    return ummg_filename


def main() -> None:
    """Entry point for the Libera metadata writer CLI

    Reads a Libera NetCDF data product file, generates UMM-G metadata from its
    contents, and writes the metadata file to the configured processing path
    """
    args = parse_cli_args()

    # Extract the data file path from command line arguments
    if not args.data_product_file_path:
        raise ValueError("Data product file path must be provided as a command line argument")
    data_filename = args.data_product_file_path

    # TODO[LIBSDC-776]: Add ability to handle other Libera data products that are not NetCDF-4
    dataset = read_input_netcdf4_data_file(data_filename)

    # Set the output location to write to in the output dropbox
    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    output_metadata_path = write_data_product_metadata_ummg(dataset, data_filename)
    logger.info(f"Processing complete. Output UMM-G: {output_metadata_path}")


if __name__ == "__main__":
    main()
