"""Module for S3 cli utilities"""

import argparse
import json
import logging
from datetime import UTC, datetime
from typing import cast

import boto3
from cloudpathlib import AnyPath, S3Path

from libera_utils.aws.utils import (
    find_bucket_in_account_by_partial_name,
    find_event_bus_in_account_by_partial_name,
    get_libera_utils_session,
)
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import L0Filename, LiberaDataProductFilename, PathType
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# Partial names used to uniquely identify SDC resources by regex search (see find_*_by_partial_name in aws.utils).
INGEST_DROPBOX_BUCKET_PARTIAL_NAME = "ingestdropbox"
SDC_EVENT_BUS_PARTIAL_NAME = "LiberaSDCEventBus"

# These values are part of the NewFilesAvailable event contract and must match exactly what the SDC expects.
NEW_FILES_AVAILABLE_EVENT_SOURCE = "manual-processing"
NEW_FILES_AVAILABLE_EVENT_DETAIL_TYPE = "NewFilesAvailableEventDetail"


def _validate_filename_for_ingest(path: PathType) -> L0Filename | LiberaDataProductFilename:
    """Validate a path as a Libera L0 or data product filename eligible for manual ingest.

    Manifest and any other filename types are rejected.

    Parameters
    ----------
    path : Path or S3Path
        Path to the file to validate.

    Returns
    -------
    L0Filename or LiberaDataProductFilename
        The parsed filename object.
    """
    for filename_class in (L0Filename, LiberaDataProductFilename):
        try:
            return filename_class(path)
        except ValueError:
            continue
    raise ValueError(f"File {path} is not a valid Libera L0 or data product filename and cannot be manually ingested.")


def s3_put_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``s3-utils put`` subcommand.

    Stages one or more Libera data product files into the SDC Ingest Dropbox bucket and emits a single
    ``NewFilesAvailable`` event to the SDC event bus. The SDC Data Ingester service then archives the files and
    creates the associated file metadata and data availability records. This is the manual analog of the automated
    ingest that happens for files produced by SDC processing steps.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"aws_s3_put_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    profile_name = parsed_args.profile
    local_file_paths: list[PathType] = [cast(PathType, AnyPath(file_path)) for file_path in parsed_args.file_paths]

    # The boto session originates here and is passed to every function that needs it. It assumes the LiberaUtils
    # role so the CLI has the permissions it needs. Keeping session creation in a single place lets integration
    # tests inject a custom session (e.g. with a specific test role) and call the workflow functions directly.
    boto_session = get_libera_utils_session(profile_name=profile_name)

    manual_ingest_data_products(local_file_paths, boto_session=boto_session)

    logger.info(
        "Staged %d file(s) to the Ingest Dropbox and emitted a NewFilesAvailable event. The SDC Data Ingester "
        "should now be running; it may take a few minutes for the files to appear in their archive bucket.",
        len(local_file_paths),
    )


def manual_ingest_data_products(
    paths_to_files: list[PathType],
    *,
    boto_session: boto3.Session,
) -> None:
    """Stage data product files to the Ingest Dropbox and emit a single NewFilesAvailable event.

    The SDC Data Ingester picks up the staged files and handles archiving them in the correct bucket as well as
    creating file metadata and data availability records.

    Parameters
    ----------
    paths_to_files : list of Path or S3Path
        Paths to the files to ingest. Each must be a validly named Libera L0 or data product file.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions. Created once by the CLI handler and passed in so that the
        same authenticated session is used throughout the workflow.
    """
    # Validate every filename up front so we don't stage a partial set before discovering a bad name.
    libera_filenames = [_validate_filename_for_ingest(path) for path in paths_to_files]

    dropbox_bucket_name = find_bucket_in_account_by_partial_name(boto_session, INGEST_DROPBOX_BUCKET_PARTIAL_NAME)
    s3_client = boto_session.client("s3")

    # Stage all files first. Only if every upload succeeds do we emit the single NewFilesAvailable event.
    files = []
    for path, libera_filename in zip(paths_to_files, libera_filenames, strict=True):
        file_name = libera_filename.path.name
        s3_client.upload_file(str(path), dropbox_bucket_name, file_name)
        uri = f"s3://{dropbox_bucket_name}/{file_name}"
        logger.info(f"Staged {path} to {uri}")
        files.append(
            {
                "type": "data",
                "uri": uri,
                "name": file_name,
                "size": path.stat().st_size,
            }
        )

    put_new_files_available_event(files, boto_session=boto_session)


def put_new_files_available_event(files: list[dict], *, boto_session: boto3.Session) -> None:
    """Emit a single NewFilesAvailable event to the SDC event bus.

    Parameters
    ----------
    files : list of dict
        File descriptors matching the NewFilesAvailableEventDetail ``files`` schema (``type``, ``uri``, ``name``,
        ``size``).
    boto_session : boto3.Session
        Boto3 session used to discover the event bus and put the event. The AWS region is derived from this session.
    """
    event_bus_name = find_event_bus_in_account_by_partial_name(boto_session, SDC_EVENT_BUS_PARTIAL_NAME)
    events_client = boto_session.client("events")

    response = events_client.put_events(
        Entries=[
            {
                "Source": NEW_FILES_AVAILABLE_EVENT_SOURCE,
                "DetailType": NEW_FILES_AVAILABLE_EVENT_DETAIL_TYPE,
                "Detail": json.dumps({"files": files}),
                "EventBusName": event_bus_name,
            }
        ]
    )

    if response.get("FailedEntryCount", 0) > 0:
        raise RuntimeError(
            f"Failed to put NewFilesAvailable event to event bus {event_bus_name}. Response entries: "
            f"{response['Entries']}"
        )

    logger.info(f"Put NewFilesAvailable event with {len(files)} file(s) to event bus {event_bus_name}")


def s3_list_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for s3-utils list CLI subcommand."""
    now = datetime.now(UTC)
    configure_task_logging(f"aws_upload_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    # The other two subcommands have more complex logic as functions with some shared arguments
    dpi_string = parsed_args.product_name
    processing_step = DataProductIdentifier(dpi_string)
    s3_list_archive_files(processing_step, profile_name=parsed_args.profile)


def s3_list_archive_files(data_product_id: str | DataProductIdentifier, *, profile_name: str = None) -> list:
    """List all files in an archive S3 bucket for a given processing step."""
    if isinstance(data_product_id, str):
        data_product_id = DataProductIdentifier(data_product_id)

    boto_session = boto3.Session(profile_name=profile_name)
    bucket_name = find_bucket_in_account_by_partial_name(boto_session, data_product_id.data_level.archive_bucket_name)

    prefix = data_product_id

    # 3. Create client FROM the session
    client = boto_session.client("s3")

    bucket_objects = [
        S3Path(f"s3://{bucket_name}/{obj['Key']}")
        for obj in client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents", [])
    ]

    for obj in bucket_objects:
        logger.info(obj)
    return bucket_objects


def s3_copy_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for s3-utils cp CLI subcommand."""
    now = datetime.now(UTC)
    configure_task_logging(f"aws_s3_cp_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    # The other two subcommands have more complex logic as functions with some shared arguments
    current_path = AnyPath(parsed_args.source_path)
    destination_path = AnyPath(parsed_args.dest_path)
    profile_name = parsed_args.profile
    boto3.Session(profile_name=profile_name)
    delete = parsed_args.delete
    s3_copy_file(current_path, destination_path, delete=delete)


# The copy functionality already exists, use it from the smart_open module.
s3_copy_file = smart_copy_file
