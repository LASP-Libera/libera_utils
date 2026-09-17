"""Shared manifest-driven runner logic for the CF-CAM (Camera Cloud Fraction) product family.

The two cloud-fraction runners (``cf_cam.py`` radiometer timescale, ``cf_cam_camtime.py`` camera
timescale) are structurally identical: read an input manifest, keep the L1B input file(s) of a
particular product, turn each into placeholder cloud-fraction values, write the resulting CF-CAM
product, and emit an output manifest. They differ only by a handful of parameters captured in
:class:`CfRunnerConfig`:

* which L1B product counts as their input (``RAD-4CH`` vs ``CAM``),
* which CF-CAM product they emit, and
* whether the input is segmented into camera pseudo-footprints (camera timescale) or read straight as
  radiometer footprints (radiometer timescale).

Environment
-----------
``PROCESSING_PATH``
    The dropbox directory (or S3 prefix) products and the output manifest are written to. This is the
    standard Libera runner convention. Unlike FMATCH, this runner needs no ancillary tree
    (``FMATCH_ANCILLARY_PATH``): the only input is the L1B file itself.
"""

from __future__ import annotations

import argparse
import enum
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from cloudpathlib import AnyPath, S3Path

from libera_utils.cloud_fraction.cloud_fraction import (
    generate_placeholder_cloud_fraction_camtime,
    generate_placeholder_cloud_fraction_radiometer,
)
from libera_utils.footprint_matching._runner import (
    FMATCH_RADIOMETER_TIME_COORDINATE,
    _as_local_path,
    algorithm_version,
    load_l1b_camera_dataset,
    load_l1b_radiometer_inputs,
    select_manifest_files_by_product_id,
)
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.io.manifest import Manifest
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.logutil import configure_task_logging

if TYPE_CHECKING:
    from libera_utils.constants import DataProductIdentifier
    from libera_utils.io.filenaming import LiberaDataProductFilename

logger = logging.getLogger(__name__)


class CfInputKind(enum.Enum):
    """How a CF-CAM runner turns its L1B input into cloud-fraction records.

    ``CAMERA`` segments an L1B Daily Camera image grid into pseudo-footprints (camera timescale);
    ``RADIOMETER`` reads the L1B RAD-4CH record axis straight through (radiometer timescale).
    """

    CAMERA = "camera"
    RADIOMETER = "radiometer"


@dataclass(frozen=True)
class CfRunnerConfig:
    """Everything that distinguishes one CF-CAM runner from another.

    Attributes
    ----------
    output_product_id : DataProductIdentifier
        The CF-CAM product this runner emits. Used for documentation/logging; the written filename's
        product id comes from the product definition's ``ProductID`` attribute.
    l1b_input_product_id : DataProductIdentifier
        The L1B Daily product that counts as this runner's input (``l1b_cam`` / ``l1b_rad``). Manifest
        files with any other product id (or unparsable names) are skipped.
    product_definition_path : Path
        Path to the CF-CAM product-definition YAML the output is validated/written against.
    time_variable : str
        Name of the record-time coordinate in the written product (``CAMERA_TIME`` / ``RADIOMETER_TIME``);
        used to derive the product filename's time bounds.
    input_kind : CfInputKind
        Whether the L1B input is segmented into camera pseudo-footprints or read as radiometer footprints.
    log_prefix : str
        Short label used in task-log filenames (e.g. ``cf_cam_camtime``).
    """

    output_product_id: DataProductIdentifier
    l1b_input_product_id: DataProductIdentifier
    product_definition_path: Path
    time_variable: str
    input_kind: CfInputKind
    log_prefix: str


def run_algorithm(manifest_path: Path | S3Path | argparse.Namespace, config: CfRunnerConfig) -> Path | S3Path:
    """Run a CF-CAM processing workflow from an input manifest.

    Parameters
    ----------
    manifest_path : Path | S3Path | argparse.Namespace
        Path to the input manifest file listing the L1B input file(s). An ``argparse.Namespace`` (as
        produced by the CLI) is also accepted for convenience; its ``manifest`` attribute is used.
    config : CfRunnerConfig
        The per-runner parameters (output product, input product, definition, time variable, input
        kind, log label).

    Returns
    -------
    Path | S3Path
        Path to the written output manifest file.

    Raises
    ------
    ValueError
        If the ``PROCESSING_PATH`` environment variable is not set, or if the manifest references no
        usable L1B inputs.
    """
    # Configure task logging with a per-run label, exactly as the FMATCH/SCENE-ID runners do.
    now = datetime.now(UTC)
    configure_task_logging(f"{config.log_prefix}_{now}")

    # Step 1: Read the input manifest.
    logger.info("Step 1: Reading the input manifest file")
    if isinstance(manifest_path, argparse.Namespace):
        manifest = AnyPath(manifest_path.manifest)
    else:
        manifest = AnyPath(manifest_path)
    input_manifest = Manifest.from_file(manifest)
    logger.info(f"Loaded manifest with {len(input_manifest.files)} files")

    dropbox_path = os.getenv("PROCESSING_PATH")
    if not dropbox_path:
        raise ValueError("PROCESSING_PATH environment variable is not set")

    # Step 2: Collect the L1B input file(s) from the manifest.
    input_label = config.l1b_input_product_id.value
    logger.info("Step 2: Collecting %s input files from the manifest", input_label)
    input_file_paths = select_manifest_files_by_product_id(input_manifest, config.l1b_input_product_id)
    if not input_file_paths:
        raise ValueError(f"No {input_label} input files found in the input manifest")

    # Step 3: Generate cloud fraction and write data products.
    logger.info("Step 3: Generating placeholder cloud fraction and writing data products")
    output_data_file_paths: list[LiberaDataProductFilename] = []
    for input_file_path in input_file_paths:
        output_file = create_and_write_data_product(
            l1b_file_path=input_file_path,
            output_path=dropbox_path,
            config=config,
        )
        output_data_file_paths.append(output_file)

    # Step 4: Create the output manifest from the input manifest.
    logger.info("Step 4: Creating the output manifest")
    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

    # Step 5: Register the written data product file(s) on the output manifest.
    logger.info(f"Step 5: Adding {len(output_data_file_paths)} data file(s) to the output manifest")
    output_manifest.add_files(*[output_file.path for output_file in output_data_file_paths])

    # Step 6: Write the output manifest to the dropbox.
    logger.info("Step 6: Writing the output manifest")
    output_manifest_filepath = output_manifest.write(dropbox_path)
    logger.info(f"Output manifest written to: {output_manifest_filepath}")

    return output_manifest_filepath


def create_and_write_data_product(
    l1b_file_path: str | Path | S3Path,
    output_path: str | Path | S3Path,
    config: CfRunnerConfig,
) -> LiberaDataProductFilename:
    """Turn one L1B input into placeholder cloud fraction and write the CF-CAM data product.

    Parameters
    ----------
    l1b_file_path : str | pathlib.Path | cloudpathlib.S3Path
        Path (local or S3) to the L1B Daily input file.
    output_path : str | pathlib.Path | cloudpathlib.S3Path
        Directory / prefix in the processing dropbox where the product file is written.
    config : CfRunnerConfig
        Runner parameters supplying the product definition path, time variable, and input kind.

    Returns
    -------
    LiberaDataProductFilename
        The written data product file, with a proper Libera filename.

    Raises
    ------
    FileNotFoundError
        If the CF-CAM product definition cannot be found in the installed libera_utils package.
    """
    if not config.product_definition_path.exists():
        raise FileNotFoundError(f"CF-CAM product definition not found: {config.product_definition_path}")

    definition = LiberaDataProductDefinition.from_yaml(config.product_definition_path)
    input_file_name = AnyPath(l1b_file_path).name

    # Materialize a local copy for the file-based L1B readers, then build the placeholder records the
    # same way FMATCH derives its record axis (segmentation for the camera timescale, RADIOMETER_TIME
    # pass-through for the radiometer timescale), so each CF product aligns 1:1 with its FMATCH sibling.
    with _as_local_path(l1b_file_path) as local_l1b_path:
        if config.input_kind is CfInputKind.CAMERA:
            logger.info("Segmenting L1B camera images from %s", local_l1b_path)
            dataset = load_l1b_camera_dataset(local_l1b_path)
            footprints = segment_l1b_camera(dataset, log=logger)
            logger.info("Segmented %d camera pseudo-footprints", len(footprints))
            data = generate_placeholder_cloud_fraction_camtime(footprints, definition)
        else:
            logger.info("Reading L1B radiometer times from %s", local_l1b_path)
            l1b_inputs = load_l1b_radiometer_inputs(local_l1b_path)
            data = generate_placeholder_cloud_fraction_radiometer(l1b_inputs[FMATCH_RADIOMETER_TIME_COORDINATE])

    logger.info("Writing %s data product for input %s", config.output_product_id.value, input_file_name)
    output_file_path = write_libera_data_product(
        data_product_definition=definition,
        data=data,
        output_path=output_path,
        time_variable=config.time_variable,
        dynamic_product_attributes={
            "algorithm_version": algorithm_version(),
            "InputGranules": input_file_name,
        },
        strict=True,
    )
    logger.info(f"Wrote data product to {output_file_path.path}")
    return output_file_path
