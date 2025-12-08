"""Module containing CLI tool for creating SPICE kernels from packets"""

import argparse
import logging
import os
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pandas as pd
import xarray as xr
from cloudpathlib import AnyPath
from curryer import kernels, meta, spicetime

from libera_utils.config import config
from libera_utils.constants import DataLevel, DataProductIdentifier, get_l1a_apid
from libera_utils.io import filenaming
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.l1a import packets as libera_packets
from libera_utils.l1a.l1a_packet_configs import get_packet_config
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# This stores the mapping from SPICE kernel DPI to the L1A DPI that contains the input data for the kernel
# This is effectively a very simple representation of the processing system dependency graph for SPICE kernels
SPICE_DPI_TO_L1A_DPI_MAP = {
    DataProductIdentifier.spice_jpss_spk: DataProductIdentifier.l1a_jpss_sc_pos_decoded,
    DataProductIdentifier.spice_jpss_ck: DataProductIdentifier.l1a_jpss_sc_pos_decoded,
    DataProductIdentifier.spice_az_ck: DataProductIdentifier.l1a_icie_axis_sample_decoded,
    DataProductIdentifier.spice_el_ck: DataProductIdentifier.l1a_icie_axis_sample_decoded,
}

# This stores the mapping from SPICE kernel DPI to the sample group name within the L1A packet config that contains
# the required data for generating the specific kernel
# The Az and El samples are both recorded with the same timestamps but SC position and attitude have separate timestamps
# within the packets and therefore separate sample groups
SPICE_DPI_TO_L1A_SAMPLE_GROUP_MAP = {
    DataProductIdentifier.spice_jpss_spk: "ADGPS",  # SC position
    DataProductIdentifier.spice_jpss_ck: "ADCFA",  # SC attitude
    DataProductIdentifier.spice_az_ck: "AXIS_SAMPLE",  # Azimuth mechanism attitude
    DataProductIdentifier.spice_el_ck: "AXIS_SAMPLE",  # Elevation mechanism attitude
}

# This stores the kernel config key for each SPICE kernel DPI
SPICE_DPI_TO_KERNEL_CONFIG_KEY_MAP = {
    DataProductIdentifier.spice_jpss_spk: "LIBERA_KERNEL_SC_SPK_CONFIG",
    DataProductIdentifier.spice_jpss_ck: "LIBERA_KERNEL_SC_CK_CONFIG",
    DataProductIdentifier.spice_az_ck: "LIBERA_KERNEL_AZ_CK_CONFIG",
    DataProductIdentifier.spice_el_ck: "LIBERA_KERNEL_EL_CK_CONFIG",
}


def create_kernel_dataframe_from_l1a(
    l1a_dataset: xr.Dataset,
    apid: int,
    sample_group_name: str,
) -> tuple[pd.DataFrame, tuple[datetime, datetime]]:
    """Create a Curryer-compatible kernel DataFrame from an L1A Dataset.

    This function extracts sample group data from an L1A Dataset and converts
    it into a pandas DataFrame suitable for SPICE kernel generation with Curryer.
    The time coordinates are converted from datetime64 to SPICE Ephemeris Time (ET).

    Parameters
    ----------
    l1a_dataset : xr.Dataset
        L1A Dataset containing sample group data with time coordinates.
        Should be created by parse_packets_to_l1a_dataset() or read from
        an L1A NetCDF file.
    apid : int
        The APID (Application Process Identifier) that identifies which
        packet configuration to use for extracting sample group metadata.
        Use LiberaApid enum values.
    sample_group_name : str
        The name of the sample group to extract from the L1A dataset. This is used to infer the
        time dimension and data fields.

    Returns
    -------
    tuple[pd.DataFrame, tuple[datetime, datetime]]
        A tuple containing:
        - DataFrame with ET time column and sample data fields
        - UTC time range tuple (start, end) for the data

    Raises
    ------
    ValueError
        If the APID has multiple sample groups and sample_group_name is not provided,
        or if the specified sample_group_name is not found in the packet configuration
    KeyError
        If required fields are missing from the L1A dataset
    TypeError
        If l1a_dataset is not an xarray.Dataset
    """
    # Validate inputs
    if not isinstance(l1a_dataset, xr.Dataset):
        raise TypeError(f"l1a_dataset must be an xarray.Dataset, got {type(l1a_dataset)}")

    # Get packet configuration
    try:
        packet_config = get_packet_config(apid)
    except Exception as e:
        raise ValueError(f"Invalid APID {apid}: {e}") from e

    # Get sample group
    try:
        sample_group = packet_config.get_sample_group(sample_group_name)
    except KeyError as ke:
        available_groups = [sg.name for sg in packet_config.sample_groups]
        raise ValueError(
            f"Sample group '{sample_group_name}' not found for APID {apid}. Available sample groups: {available_groups}"
        ) from ke

    # Validate required fields exist in dataset
    time_dim = sample_group.sample_time_dimension
    if time_dim not in l1a_dataset:
        raise KeyError(
            f"Required time dimension '{time_dim}' not found in L1A dataset. "
            f"Available coordinates: {list(l1a_dataset.coords)}"
        )

    missing_fields = [field for field in sample_group.sample_data_fields if field not in l1a_dataset]
    if missing_fields:
        raise KeyError(
            f"Required data fields missing from L1A dataset: {missing_fields}. "
            f"Available variables: {list(l1a_dataset.data_vars)}"
        )

    logger.debug("Creating kernel DataFrame from L1A dataset for APID %s, sample group %s", apid, sample_group.name)
    logger.debug(
        "Time dimension: %s, data fields: %s", sample_group.sample_time_dimension, sample_group.sample_data_fields
    )

    # Extract time coordinates and data fields
    time_coords = l1a_dataset[sample_group.sample_time_dimension]
    data_arrays = [l1a_dataset[field] for field in sample_group.sample_data_fields]

    # Convert datetime64 to Ephemeris Time
    utc_times = time_coords.values
    utc_range = (pd.Timestamp(utc_times.min()).to_pydatetime(), pd.Timestamp(utc_times.max()).to_pydatetime())
    et_times = spicetime.adapt(utc_times, from_="dt64", to="et")

    # Create DataFrame
    et_column_name = sample_group.sample_time_dimension.replace("TIME", "ET")
    df = pd.DataFrame({et_column_name: et_times})
    for data_array in data_arrays:
        df[data_array.name] = data_array.values

    logger.info(
        "Created kernel DataFrame with %d samples covering %s to %s", len(time_coords), utc_range[0], utc_range[1]
    )

    return df, utc_range


def create_kernel_dataframe_from_l1a_netcdf(
    netcdf_path: filenaming.PathType,
    apid: int,
    sample_group_name: str,
) -> tuple[pd.DataFrame, tuple[datetime, datetime]]:
    """Create a Curryer-compatible kernel DataFrame from an L1A NetCDF file.

    This convenience function opens an L1A NetCDF file and extracts sample group
    data for SPICE kernel generation. It wraps create_kernel_dataframe_from_l1a()
    to provide a simpler interface when working with NetCDF files.

    Parameters
    ----------
    netcdf_path : filenaming.PathType
        Path to the L1A NetCDF file (local or S3). The file should be created
        by parse_packets_to_l1a_dataset() or equivalent processing.
    apid : int
        The APID (Application Process Identifier) that identifies which
        packet configuration to use for extracting sample group metadata.
        Use LiberaApid enum values.
    sample_group_name : str
        The name of the sample group to extract from the L1A dataset.

    Returns
    -------
    tuple[pd.DataFrame, tuple[datetime, datetime]]
        A tuple containing:
        - DataFrame with ET time column and sample data fields
        - UTC time range tuple (start, end) for the data

    Raises
    ------
    FileNotFoundError
        If the NetCDF file does not exist at the specified path
    ValueError
        If the file cannot be opened as a valid NetCDF file, or if validation
        errors occur in create_kernel_dataframe_from_l1a()

    Examples
    --------
    >>> from libera_utils.constants import LiberaApid
    >>> df, utc_range = create_kernel_dataframe_from_l1a_netcdf(
    ...     netcdf_path="path/to/l1a_file.nc",
    ...     apid=LiberaApid.jpss_sc_pos,
    ...     sample_group_name="ADGPS"
    ... )
    """
    netcdf_path = cast(filenaming.PathType, AnyPath(netcdf_path))

    # Check if file exists
    if not netcdf_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")

    logger.info("Opening L1A NetCDF file: %s", netcdf_path)

    # Open NetCDF file with h5netcdf engine
    try:
        with xr.open_dataset(netcdf_path, engine="h5netcdf") as l1a_dataset:
            # Delegate to the main function
            df, utc_range = create_kernel_dataframe_from_l1a(
                l1a_dataset=l1a_dataset,
                apid=apid,
                sample_group_name=sample_group_name,
            )
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Failed to open or read NetCDF file {netcdf_path}: {e}") from e

    logger.info("Successfully created kernel DataFrame from NetCDF file")
    return df, utc_range


def create_jpss_kernel_dataframe_from_csv(
    csv_path: filenaming.PathType,
) -> tuple[pd.DataFrame, tuple[datetime, datetime]]:
    """Create a Curryer-compatible JPSS kernel DataFrame from a CSV ephemeris file.

    This function reads a CSV file containing simulated JPSS ephemeris data with GPS time,
    position, and velocity information. It converts the GPS time to UTC and then to
    SPICE Ephemeris Time (ET) for use in kernel generation.

    Parameters
    ----------
    csv_path : filenaming.PathType
        Path to the CSV ephemeris file (local or S3).

    Returns
    -------
    tuple[pd.DataFrame, tuple[datetime, datetime]]
        A tuple containing:
        - DataFrame with ET time columns (ADGPS_JPSS_ET and ADCFA_JPSS_ET)
          and position/velocity data fields
        - UTC time range tuple (start, end) for the data

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the specified path
    ValueError
        If required columns are missing from the CSV file

    Notes
    -----
    - Requires SPICE leap second kernel (LSK) to be loaded for ET conversion
    - Creates both ADGPS_JPSS_ET and ADCFA_JPSS_ET columns with identical values
    """
    csv_path = cast(filenaming.PathType, AnyPath(csv_path))

    # Check if file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info("Reading JPSS ephemeris CSV file: %s", csv_path)

    # Read CSV file
    df = pd.read_csv(csv_path)

    expected_columns = [
        "Time (UTCG)",
        "x (km)",
        "y (km)",
        "z (km)",
        "vx (km/sec)",
        "vy (km/sec)",
        "vz (km/sec)",
        "q1",
        "q2",
        "q3",
        "q4",
    ]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV file: {missing_columns}")

    df.set_index(df.columns[0], inplace=True)  # Set 'Time (UTCG)' as index

    # Rename columns to standard column names referenced in Curryer kernel configs
    df = df.rename(
        columns={
            "x (km)": "ADGPSPOSX",
            "y (km)": "ADGPSPOSY",
            "z (km)": "ADGPSPOSZ",
            "vx (km/sec)": "ADGPSVELX",
            "vy (km/sec)": "ADGPSVELY",
            "vz (km/sec)": "ADGPSVELZ",
            "q1": "ADCFAQ1",
            "q2": "ADCFAQ2",
            "q3": "ADCFAQ3",
            "q4": "ADCFAQ4",
        }
    )

    for col in ["ADGPSPOSX", "ADGPSPOSY", "ADGPSPOSZ", "ADGPSVELX", "ADGPSVELY", "ADGPSVELZ"]:
        df[col] *= 1e3  # KM to meters.

    df["ADGPS_JPSS_ET"] = spicetime.adapt(df.index.values, "iso", "et")
    df["ADCFA_JPSS_ET"] = spicetime.adapt(df.index.values, "iso", "et")

    logger.debug("Simulated JPSS position CSV file contains %d rows", len(df))

    utc_range: list[datetime] = spicetime.adapt(
        [df["ADGPS_JPSS_ET"].iloc[0], df["ADGPS_JPSS_ET"].iloc[-1]], "et", "dt64"
    )
    utc_range_tuple: tuple[datetime, datetime] = (utc_range[0], utc_range[1])

    return df, utc_range_tuple


def make_kernel(
    config_file: str | Path,
    output_kernel: str | filenaming.PathType,
    input_data: pd.DataFrame | None = None,
    overwrite: bool = False,
    append: bool | int = False,
) -> filenaming.PathType:
    """Create a SPICE kernel from a configuration file and input data.

    Parameters
    ----------
    config_file : str | pathlib.Path
        JSON configuration file defining how to create the kernel.
    output_kernel : str | filenaming.PathType
        Output directory or file to create the kernel. If a directory, the
        file name will be based on the config_file, but with the SPICE file
        extension.
    input_data : pd.DataFrame | None
        pd.DataFrame containing kernel input data. If not supplied, the config is assumed to reference an input data file.
    overwrite : bool
        Option to overwrite an existing file.
    append : bool | int
        Option to append to an existing file. Anything truthy will be treated as True.

    Returns
    -------
    filenaming.PathType
        Output kernel file path

    """
    output_kernel = cast(filenaming.PathType, AnyPath(output_kernel))
    config_file = Path(config_file)  # This is always a local path because the configs are package data

    # Load meta kernel details. Required to auto-map frame IDs.
    meta_kernel_file = Path(str(config.get("LIBERA_KERNEL_META")))
    _ = meta.MetaKernel.from_json(meta_kernel_file, relative=True)

    # Create the kernels from the JSONs definitions.
    creator = kernels.create.KernelCreator(overwrite=overwrite, append=bool(append))

    with tempfile.TemporaryDirectory(prefix="/tmp/") as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        if output_kernel.is_file():
            tmp_path = tmp_path / output_kernel.name

        out_fn = creator.write_from_json(config_file, output_kernel=tmp_path, input_data=input_data)

        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        if output_kernel.is_dir():
            output_kernel = output_kernel / out_fn.name
        smart_copy_file(out_fn, output_kernel)
        logger.info("Kernel copied to %s", output_kernel)
    return output_kernel


def create_kernel_from_l1a(
    l1a_data: str | filenaming.PathType | xr.Dataset,
    kernel_identifier: str | DataProductIdentifier,
    output_dir: str | filenaming.PathType,
    overwrite=False,
) -> filenaming.PathType:
    """Create a SPICE kernel from a single L1A Dataset for a kernel data product type.

    This operates on a single L1A Dataset (or path to an L1A NetCDF file). The assumption is that this L1A
    file should contain all the necessary data to create the requested SPICE kernel type. Multiple inputs
    are expected to be handled at the level of creating the L1A data product from source data (packets).

    Parameters
    ----------
    l1a_data : str | filenaming.PathType | xr.Dataset
        L1A Dataset or path to L1A NetCDF file containing decoded packet data
        suitable for generating the requested SPICE kernel.
    kernel_identifier : str | DataProductIdentifier
        Data product identifier that is associated with a kernel.
    output_dir : str | filenaming.PathType
        Output location for the SPICE kernels and output manifest.
    overwrite : bool
        Option to overwrite any existing similar-named SPICE kernels.

    Returns
    -------
    filenaming.PathType
        Output kernel file path.
    """
    if isinstance(l1a_data, xr.Dataset):
        l1a_dataset = l1a_data
    else:
        l1a_path = cast(filenaming.PathType, AnyPath(l1a_data))
        logger.info("Opening L1A NetCDF file: %s", l1a_path)
        try:
            with xr.open_dataset(l1a_path, engine="h5netcdf") as l1a_dataset:
                l1a_dataset = l1a_dataset.load()  # Load into memory
        except Exception as e:
            raise ValueError(f"Failed to open or read L1A NetCDF file {l1a_path}: {e}") from e

    kernel_identifier = DataProductIdentifier(kernel_identifier)

    # Get the DPI for the L1A file that contains the data needed to create the requested kernel
    l1a_dpi = SPICE_DPI_TO_L1A_DPI_MAP[kernel_identifier]

    # Get the APID and sample group name within the L1A packet config required for finding the kernel data
    # in the L1A file
    apid = get_l1a_apid(l1a_dpi)
    sample_group_name = SPICE_DPI_TO_L1A_SAMPLE_GROUP_MAP[kernel_identifier]

    # Create Curryer-compatible kernel DataFrame from L1A dataset
    kernel_df, utc_range = create_kernel_dataframe_from_l1a(
        l1a_dataset=l1a_dataset,
        apid=apid,
        sample_group_name=sample_group_name,
    )

    # Store as a single-element list for compatibility with the existing kernel creation loop
    input_dataframe = kernel_df
    input_time_range = [utc_range[0], utc_range[1]]

    # Generate the output file name.
    fn_kwargs = dict(
        utc_start=input_time_range[0],
        utc_end=input_time_range[1],
        version=filenaming.get_current_version_str("libera_utils"),
        revision=datetime.now(UTC),
    )
    if kernel_identifier.value.endswith("SPK"):
        extension = "bsp"
    elif kernel_identifier.value.endswith("CK"):
        extension = "bc"
    else:
        raise ValueError(f"Incorrectly named SPICE kernel Data Product Identifier: {kernel_identifier}")

    krn_filename = filenaming.LiberaDataProductFilename.from_filename_parts(
        data_level=DataLevel.SPICE, product_name=kernel_identifier, extension=extension, **fn_kwargs
    )
    output_full_path = AnyPath(output_dir) / krn_filename.path.name

    # Get the kernel config file from the environment config
    kernel_config_file = Path(str(config.get(SPICE_DPI_TO_KERNEL_CONFIG_KEY_MAP[kernel_identifier])))

    # Create the kernel
    output_kernel = make_kernel(
        config_file=kernel_config_file,
        output_kernel=output_full_path,
        input_data=input_dataframe,
        overwrite=overwrite,
        append=False,
    )
    return output_kernel


def create_kernel_from_packets(
    input_data_files: list[str | filenaming.PathType],
    kernel_identifier: str | DataProductIdentifier,
    output_dir: str | filenaming.PathType,
    overwrite=False,
) -> filenaming.PathType:
    """Create a SPICE kernel from one or more input packet files and kernel data product type.

    The packet files passed as input must be convertible to L1A Dataset products. This function
    is a light wrapper around the core kernel creation logic that handles creating kernels from
    L1A Datasets.

    This function is not intended for use in the production pipeline since L1A processing is a
    a separate step from kernel generation. However, for development and analysis purposes,
    this allows us to run both steps in one function call: packets -> L1A Dataset -> SPICE kernel.

    Parameters
    ----------
    input_data_files : list[str, filenaming.PathType]
        List of packet files to process. Multiple files are combined into
        a single L1A dataset before extracting data for kernel generation.
    kernel_identifier : str | DataProductIdentifier
        Data product identifier that is associated with a kernel.
    output_dir : str | filenaming.PathType
        Output location for the SPICE kernels and output manifest.
    overwrite : bool
        Option to overwrite any existing similar-named SPICE kernels.

    Returns
    -------
    filenaming.PathType
        Output kernel file path.
    """
    # Validate and parse the input arguments.
    output_dir = cast(filenaming.PathType, AnyPath(output_dir))

    kernel_identifier = DataProductIdentifier(kernel_identifier)
    if kernel_identifier.data_level != DataLevel.SPICE:
        raise ValueError(
            f"kernel_identifier must be a SPICE level Data Product Identifier, got {kernel_identifier} with data level {kernel_identifier.data_level}"
        )

    logger.info("Generating SPICE kernel %s from packet files: %s", kernel_identifier, input_data_files)

    # Get the APID required to generate the requested kernel type
    apid = get_l1a_apid(SPICE_DPI_TO_L1A_DPI_MAP[kernel_identifier])

    logger.info("Using L1A data for APID %s to generate kernel type %s", apid, kernel_identifier)

    # Parse packet files to create L1A dataset
    # Multiple files are combined into a single L1A dataset before being
    # trimmed down to the requested time range for kernel generation
    logger.info("Parsing packet files to L1A dataset: %s", input_data_files)
    l1a_dataset = libera_packets.parse_packets_to_l1a_dataset(
        packet_files=input_data_files,
        apid=apid,
    )
    logger.info(
        "Created L1A dataset from %d packet files, combining all data before time trimming",
        len(input_data_files),
    )
    logger.debug("L1A dataset created with dimensions: %s", dict(l1a_dataset.dims))

    # Create kernel DataFrame from L1A dataset
    return create_kernel_from_l1a(
        l1a_data=l1a_dataset, kernel_identifier=kernel_identifier, output_dir=output_dir, overwrite=overwrite
    )


def create_kernels_from_manifest(
    input_manifest: str | filenaming.PathType,
    kernel_product_ids: str | DataProductIdentifier | list[str | DataProductIdentifier],
    output_dir: str | filenaming.PathType,
    overwrite=False,
):
    """Generate SPICE kernels from a manifest file of L1A NetCDF products.

    Parameters
    ----------
    input_manifest : str | filenaming.PathType
        Input manifest file containing one or more L1A NetCDF product files containing decoded packet data
        suitable for generating the requested SPICE kernels.
    kernel_product_ids : str | DataProductIdentifier | list[str | DataProductIdentifier]
        One or more SPICE kernel data product identifiers for SPICE kernel products to generate.
    output_dir : str | filenaming.PathType
        Output location for the SPICE kernels and output manifest.
    overwrite : bool, optional
        Option to overwrite any existing similar-named SPICE kernels.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    logger.info("Starting SPICE kernel generation from manifest: %s", input_manifest)
    # Process input manifest
    mani = Manifest.from_file(input_manifest)
    mani.validate_checksums()

    logger.debug("Checksum validation passed for input manifest: %s", input_manifest)

    # Files containing decoded L1A packet data
    input_l1a_files = mani.files

    if isinstance(kernel_product_ids, str):
        _kernel_list = [kernel_product_ids]
    else:
        _kernel_list = kernel_product_ids

    # Ensure we have DPI objects, not just strings
    kernel_list: list[DataProductIdentifier] = [DataProductIdentifier(dpi) for dpi in _kernel_list]

    for dpi in kernel_list:
        if dpi.data_level != DataLevel.SPICE:
            raise ValueError(
                f"The `kernel_product_ids` [{kernel_list}] contain a non-SPICE Data Product Identifier [{dpi}]."
            )

    # Verify expectations for processing L1A files to SPICE kernels
    input_file_names = [file_entry.filename for file_entry in input_l1a_files]
    if len(input_file_names) == 0:
        raise ValueError("Input manifest contains no input files.")
    elif len(input_file_names) > 1:
        raise ValueError(
            "Input manifest contains multiple input files. Only one (daily) L1A product is supported per kernel generation run."
        )

    l1a_file = input_file_names[0]

    try:
        libera_filename = filenaming.LiberaDataProductFilename(l1a_file)
        if libera_filename.data_product_id.data_level != DataLevel.L1A:
            raise ValueError(f"File is not correctly named as an L1A data product: {l1a_file}")
    except ValueError as ve:
        # This is left as a warning to allow testing with non-conforming L1A filenames
        warnings.warn(f"Input manifest file contains non-L1A product filename: {l1a_file} ({ve})")

    logger.info("Using input file for kernel generation: %s", l1a_file)

    outputs = []
    kernel_processing_failures: list[tuple[str, list]] = []
    for kernel_identifier in kernel_list:
        # Make each type of kernel requested (each kernel type has a unique DPI)
        try:
            outputs.append(
                create_kernel_from_l1a(
                    l1a_data=l1a_file,
                    kernel_identifier=kernel_identifier,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )
            )
        except Exception as _:
            kernel_processing_failures.append((kernel_identifier, input_file_names))
            logger.exception(
                "Kernel generation failed for DPI [%s] and inputs [%s]. Suppressing and continuing with"
                "other kernels (if any)",
                kernel_identifier,
                input_file_names,
            )

    # If failures occurred during kernel generation, raise before we write out a manifest
    # This allows the kernel maker to try making each kernel but if any fail, we don't want to continue.
    if kernel_processing_failures:
        raise ValueError(f"Kernel processing steps failed (kernel DPI, input_files): {kernel_processing_failures}")

    # Duplicates are possible depending on file naming and append flag.
    outputs = sorted(set(outputs))

    # Generate output manifest.
    pedi = Manifest.output_manifest_from_input_manifest(mani)
    pedi.add_files(*outputs)

    # Automatically generates a proper output manifest filename and writes it to the path specified,
    # usually this path is retrieved from the environment.
    pedi.write(output_dir)

    return pedi


def jpss_kernel_cli_handler(parsed_args: argparse.Namespace):
    """Generate SPICE JPSS kernels from command line arguments.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    configure_task_logging(
        f"jpss_kernel_generator_{now}",
        limit_debug_loggers=["libera_utils", "curryer"],
        console_log_level=logging.DEBUG if parsed_args.verbose else logging.INFO,
    )

    return create_kernels_from_manifest(
        input_manifest=parsed_args.input_manifest,
        kernel_product_ids=[DataProductIdentifier.spice_jpss_spk, DataProductIdentifier.spice_jpss_ck],
        output_dir=os.environ["PROCESSING_PATH"],
        overwrite=False,
    )


def azel_kernel_cli_handler(parsed_args: argparse.Namespace):
    """Generate SPICE Az/El kernels from command line arguments.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    configure_task_logging(
        f"azel_kernel_generator_{now}",
        limit_debug_loggers=["libera_utils", "curryer"],
        console_log_level=logging.DEBUG if parsed_args.verbose else logging.INFO,
    )

    return create_kernels_from_manifest(
        input_manifest=parsed_args.input_manifest,
        kernel_product_ids=[DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck],
        output_dir=os.environ["PROCESSING_PATH"],
        overwrite=False,
    )
