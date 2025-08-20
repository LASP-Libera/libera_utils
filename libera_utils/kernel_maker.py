"""Module containing CLI tool for creating SPICE kernels from packets"""

import argparse
import logging
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from curryer import kernels, meta, spicetime

from libera_utils import packets as libera_packets
from libera_utils import time
from libera_utils.config import config
from libera_utils.constants import DataLevel, DataProductIdentifier
from libera_utils.io import filenaming
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

KERNEL_DPI = (
    DataProductIdentifier.spice_jpss_spk,
    DataProductIdentifier.spice_jpss_ck,
    DataProductIdentifier.spice_az_ck,
    DataProductIdentifier.spice_el_ck,
)


def make_kernel(
    config_file: str | Path,
    output_kernel: str | filenaming.PathType,
    input_data: str | Path | None = None,
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
    input_data : str | filenaming.PathType or pd.DataFrame, optional
        Input data file or object. Not required if defined within the config.
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
    config_file = Path(config_file)

    # Load meta kernel details. Required to auto-map frame IDs.
    meta_kernel_file = Path(config.get("LIBERA_KERNEL_META"))
    _ = meta.MetaKernel.from_json(meta_kernel_file, relative=True)

    # Create the kernels from the JSONs definitions.
    creator = kernels.create.KernelCreator(overwrite=overwrite, append=append)

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


def preprocess_data(
    input_data_file: str | filenaming.PathType,
    nominal_time_field: str,
    pkt_time_fields: Sequence[str],
    kernel_identifier: DataProductIdentifier,
) -> tuple[pd.DataFrame, tuple[datetime, datetime]]:
    """Preprocess kernel data to perform conversions and determine time range.

    Parameters
    ----------
    input_data_file : str | filenaming.PathType
        Input data file.
    nominal_time_field : str
        Name of the field to store the converted time field as.
    pkt_time_fields : Sequence[str]
        Names of the telemetry packet time fields used to convert the time.
    kernel_identifier : DataProductIdentifier
        The kernel type being generated (needed to determine which packet reader to use).

    Returns
    -------
    pd.DataFrame
        Loaded SPICE kernel data.
    datetime.datetime, datetime.datetime
        The date time range of the data.

    """
    # Load the input data.
    input_data_file = cast(filenaming.PathType, AnyPath(input_data_file))
    if input_data_file.suffix == ".csv":
        # TODO[LIBSDC-485]: Implement or remove when adding support for other input file types. Test data exists for this but the format is not defined.
        raise NotImplementedError(
            "This function previously worked with CSV files but tests were never written so functionality was removed."
        )
    elif input_data_file.suffix == ".nc":
        # TODO[LIBSDC-485]: Implement when we have combined packet data NetCDF files.
        raise NotImplementedError("This function does not support NetCDF input files yet.")

    # Assume a binary file of raw packets.
    else:
        # Use the appropriate packet reader based on kernel type
        if kernel_identifier in [DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck]:
            input_dataset = libera_packets.read_azel_packet_data([input_data_file])
        elif kernel_identifier in [DataProductIdentifier.spice_jpss_ck, DataProductIdentifier.spice_jpss_spk]:
            input_dataset = libera_packets.read_sc_packet_data([input_data_file])
        else:
            raise ValueError(
                f"Unexpected kernel data product identifier {kernel_identifier}. Expected one of {KERNEL_DPI}."
            )

        # Compute the ephemeris time from the multipart ephemeris time.
        if kernel_identifier in [DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck]:
            # For Az/El packets, use seconds and microseconds
            packet_dt64 = time.multipart_to_dt64(input_dataset, s_field=pkt_time_fields[0], us_field=pkt_time_fields[1])
            # Prepare the data for Az or El kernel generation
            if kernel_identifier == DataProductIdentifier.spice_az_ck:
                # For azimuth kernel, select and rename columns appropriately
                input_dataset["AZ_ET"] = spicetime.adapt(packet_dt64.values, "dt64", "et")
                # Convert radians to degrees
                input_dataset["AZ_ANGLE"] = np.degrees(input_dataset["AZ_ANGLE_RAD"])
                # Keep only the columns needed for the kernel
                input_dataset = input_dataset[["AZ_ET", "AZ_ANGLE"]]
            else:  # spice_el_ck
                # For elevation kernel, select and rename columns appropriately
                input_dataset["EL_ET"] = spicetime.adapt(packet_dt64.values, "dt64", "et")
                # Convert radians to degrees
                input_dataset["EL_ANGLE"] = np.degrees(input_dataset["EL_ANGLE_RAD"])
                # Keep only the columns needed for the kernel
                input_dataset = input_dataset[["EL_ET", "EL_ANGLE"]]
        else:
            # For spacecraft packets, use days, milliseconds, and microseconds
            packet_dt64 = time.multipart_to_dt64(input_dataset, *pkt_time_fields)
            input_dataset[nominal_time_field] = spicetime.adapt(packet_dt64.values, "dt64", "et")

        utc_range = (packet_dt64.iloc[0].to_pydatetime(), packet_dt64.iloc[-1].to_pydatetime())

    return input_dataset, utc_range


def from_args(
    input_data_files: list[str | filenaming.PathType],
    kernel_identifier: str | DataProductIdentifier,
    output_dir: str | filenaming.PathType,
    overwrite=False,
    append=False,
    verbose=False,
) -> filenaming.PathType:
    """Create a SPICE kernel from an input file and kernel data product type.

    Parameters
    ----------
    input_data_files : list[str, filenaming.PathType]
        Input data files.
    kernel_identifier : str | DataProductIdentifier
        Data product identifier that is associated with a kernel.
    output_dir : str | filenaming.PathType
        Output location for the SPICE kernels and output manifest.
    overwrite : bool
        Option to overwrite any existing similar-named SPICE kernels.
    append : bool
        Option to append to any existing similar-named SPICE kernels.
        If multiple input files are provided with append=False, the first file will create a new kernel,
        and subsequent files will append to it.
    verbose : bool
        Option to log with extra verbosity.

    Returns
    -------
    filenaming.PathType
        Output kernel file path.

    """
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    configure_task_logging(
        f"kernel_generator_{now}",
        limit_debug_loggers=["libera_utils", "curryer"],
        console_log_level=logging.DEBUG if verbose else logging.INFO,
    )

    # Validate and parse the input arguments.
    output_dir = cast(filenaming.PathType, AnyPath(output_dir))

    kernel_identifier = DataProductIdentifier(kernel_identifier)
    if kernel_identifier not in KERNEL_DPI:
        raise ValueError(
            f"The `kernel_identifier` [{kernel_identifier}] is not a Data Product Identifier associated"
            f" with a SPICE kernel, expected one of: [{KERNEL_DPI}]"
        )

    if kernel_identifier == DataProductIdentifier.spice_jpss_spk:
        config_file = config.get("LIBERA_KERNEL_SC_SPK_CONFIG")
        pkt_time_fields = ["ADAET1DAY", "ADAET1MS", "ADAET1US"]
        nominal_time_field = "SPK_ET"

    elif kernel_identifier == DataProductIdentifier.spice_jpss_ck:
        config_file = config.get("LIBERA_KERNEL_SC_CK_CONFIG")
        pkt_time_fields = ["ADAET2DAY", "ADAET2MS", "ADAET2US"]
        nominal_time_field = "CK_ET"

    elif kernel_identifier == DataProductIdentifier.spice_az_ck:
        config_file = config.get("LIBERA_KERNEL_AZ_CK_CONFIG")
        # For Az/El packets, we use seconds and microseconds from SAMPLE_SEC and SAMPLE_USEC
        pkt_time_fields = ["SAMPLE_SEC", "SAMPLE_USEC"]
        nominal_time_field = "CK_ET"

    elif kernel_identifier == DataProductIdentifier.spice_el_ck:
        config_file = config.get("LIBERA_KERNEL_EL_CK_CONFIG")
        # For Az/El packets, we use seconds and microseconds from SAMPLE_SEC and SAMPLE_USEC
        pkt_time_fields = ["SAMPLE_SEC", "SAMPLE_USEC"]
        nominal_time_field = "CK_ET"

    else:
        raise ValueError(kernel_identifier)

    # Prepare the input data and determine the min/max time span.
    input_datasets = []
    input_time_range: list[datetime] = []
    for file_name in input_data_files:
        in_dataset, in_range = preprocess_data(
            file_name,
            nominal_time_field=nominal_time_field,
            pkt_time_fields=pkt_time_fields,
            kernel_identifier=kernel_identifier,
        )
        input_datasets.append(in_dataset)
        if not input_time_range:
            input_time_range = list(in_range)
        else:
            input_time_range = [min(input_time_range[0], in_range[0]), max(input_time_range[1], in_range[1])]

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
        data_level=DataLevel.SPICE,
        product_name=kernel_identifier,
        extension=extension,
        **fn_kwargs
    )
    output_full_path = output_dir / krn_filename.path.name

    # Create the kernel(s).
    for ith, an_input_dataset in enumerate(input_datasets):
        output_kernel = make_kernel(
            config_file=config_file,
            output_kernel=output_full_path,
            input_data=an_input_dataset,
            overwrite=overwrite,
            append=append or ith,
        )
    return output_kernel


def from_manifest(
    input_manifest: str | filenaming.PathType,
    data_product_identifiers: list[str],
    output_dir: str | filenaming.PathType,
    overwrite=False,
    append=False,
    verbose=False,
):
    """Generate SPICE kernels from a manifest file.

    Parameters
    ----------
    input_manifest : str | filenaming.PathType
        Input manifest file containing one or more input data files.
    data_product_identifiers : list[str]
        One or more SPICE kernel data product identifiers.
    output_dir : str | filenaming.PathType
        Output location for the SPICE kernels and output manifest.
    overwrite : bool, optional
        Option to overwrite any existing similar-named SPICE kernels.
    append : bool, optional
        Option to append to any existing similar-named SPICE kernels.
    verbose : bool, optional
        Option to log with extra verbosity.

    Returns
    -------
    libera_utils.io.manifest.Manifest
        Output manifest file containing one or more kernel files.

    """
    # Process input manifest
    mani = Manifest.from_file(input_manifest)
    mani.validate_checksums()

    input_data_files = mani.files
    if isinstance(data_product_identifiers, str):
        data_product_identifiers = [data_product_identifiers]

    # Perform processing.
    input_file_names = [file_entry.filename for file_entry in input_data_files]
    outputs = []
    kernel_processing_failures: list[tuple[str, list]] = []
    for kernel_identifier in data_product_identifiers:
        # Make each type of kernel requested (each kernel type has a unique DPI)
        try:
            outputs.append(
                from_args(
                    input_data_files=input_file_names,
                    kernel_identifier=kernel_identifier,
                    output_dir=output_dir,
                    overwrite=overwrite,
                    append=append,
                    verbose=verbose,
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
    return from_manifest(
        input_manifest=parsed_args.input_manifest,
        data_product_identifiers=[DataProductIdentifier.spice_jpss_spk, DataProductIdentifier.spice_jpss_ck],
        output_dir=os.environ["PROCESSING_PATH"],
        overwrite=False,
        append=False,
        verbose=parsed_args.verbose,
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
    return from_manifest(
        input_manifest=parsed_args.input_manifest,
        data_product_identifiers=[DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck],
        output_dir=os.environ["PROCESSING_PATH"],
        overwrite=False,
        append=False,
        verbose=parsed_args.verbose,
    )
