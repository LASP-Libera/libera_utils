"""Module for S3 cli utilities"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from cloudpathlib import AnyPath, S3Path

from libera_utils.aws.utils import (
    find_bucket_in_account_by_partial_name,
    find_dynamodb_table_in_account_by_partial_name,
    find_event_bus_in_account_by_partial_name,
    get_l2_team_role_session,
)
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import L0Filename, LiberaDataProductFilename, PathType
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# Partial names used to uniquely identify SDC resources by regex search (see find_*_by_partial_name in aws.utils).
INGEST_DROPBOX_BUCKET_PARTIAL_NAME = "ingestdropbox"
SDC_EVENT_BUS_PARTIAL_NAME = "LiberaSDCEventBus"
DATA_AVAILABILITY_TABLE_PARTIAL_NAME = "DataAvailabilityTable"
FILE_METADATA_TABLE_PARTIAL_NAME = "FileMetadataTable"

# These values are part of the NewFilesAvailable event contract and must match exactly what the SDC expects.
NEW_FILES_AVAILABLE_EVENT_SOURCE = "manual-processing"
NEW_FILES_AVAILABLE_EVENT_DETAIL_TYPE = "NewFilesAvailableEventDetail"

# Ingestion verification (--verify) polling parameters.
DEFAULT_VERIFY_TIMEOUT_SECONDS = 300.0  # 5 minutes
VERIFY_POLL_INTERVAL_SECONDS = 10.0


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
    # The put workflow uploads from the local filesystem, so resolve inputs to plain local Paths.
    local_file_paths: list[Path] = [Path(file_path) for file_path in parsed_args.file_paths]

    # The boto session originates here and is passed to every function that needs it. It assumes the LiberaUtils
    # role so the CLI has the permissions it needs. Keeping session creation in a single place lets integration
    # tests inject a custom session (e.g. with a specific test role) and call the workflow functions directly.
    boto_session = get_l2_team_role_session(profile_name=profile_name)

    libera_filenames = manual_ingest_data_products(local_file_paths, boto_session=boto_session)

    if parsed_args.verify:
        logger.info(
            "Verifying full ingestion of %d file(s) (timeout %.0fs)...", len(libera_filenames), parsed_args.timeout
        )
        verify_ingestion(libera_filenames, boto_session=boto_session, timeout=parsed_args.timeout)
        logger.info("Verified full ingestion of %d file(s).", len(libera_filenames))
    else:
        logger.info(
            "Staged %d file(s) to the Ingest Dropbox and emitted a NewFilesAvailable event. The SDC Data Ingester "
            "should now be running; it may take a few minutes for the files to appear in their archive bucket.",
            len(local_file_paths),
        )


def manual_ingest_data_products(
    paths_to_files: list[Path],
    *,
    boto_session: boto3.Session,
) -> list[L0Filename | LiberaDataProductFilename]:
    """Stage data product files to the Ingest Dropbox and emit a single NewFilesAvailable event.

    The SDC Data Ingester picks up the staged files and handles archiving them in the correct bucket as well as
    creating file metadata and data availability records.

    Parameters
    ----------
    paths_to_files : list of Path
        Local filesystem paths to the files to ingest. Each must be a validly named Libera L0 or data product file.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions. Created once by the CLI handler and passed in so that the
        same authenticated session is used throughout the workflow.

    Returns
    -------
    list of L0Filename or LiberaDataProductFilename
        The validated filename objects for the staged files (useful for subsequent verification).
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
    return libera_filenames


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


def _archive_object_exists(s3_client, bucket: str, key: str) -> bool:
    """Return whether an object exists at the given bucket/key (read-only ``head_object``)."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as err:
        if err.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def verify_ingestion(
    libera_filenames: list[L0Filename | LiberaDataProductFilename],
    *,
    boto_session: boto3.Session,
    timeout: float = DEFAULT_VERIFY_TIMEOUT_SECONDS,
    poll_interval: float = VERIFY_POLL_INTERVAL_SECONDS,
) -> None:
    """Block until every staged file is confirmed fully ingested, or raise on timeout.

    For each file, up to three read-only checks are polled until they pass:

    1. The file exists in its expected archive bucket (at its filename-derived archive prefix).
    2. A Data Availability record exists for the data product/version/applicable-date (skipped for L0 PDS/CR files,
       which the SDC does not write availability records for).
    3. A File Metadata record exists for the file basename.

    All required AWS resources are resolved once up front; finding zero or more than one of any resource raises
    immediately (it indicates a mismatch between Libera Utils and the deployed SDC). Checks are polled every
    ``poll_interval`` seconds and each check stops being polled as soon as it passes. A per-file summary is always
    logged; if any check is still unsatisfied at ``timeout``, a ``TimeoutError`` is raised.

    Parameters
    ----------
    libera_filenames : list of L0Filename or LiberaDataProductFilename
        The validated filenames staged for ingest (as returned by ``manual_ingest_data_products``).
    boto_session : boto3.Session
        Boto3 session used for all (read-only) AWS interactions.
    timeout : float, optional
        Maximum number of seconds to wait for full ingestion. Default 300 (5 minutes).
    poll_interval : float, optional
        Number of seconds between polling passes. Default 10.

    Raises
    ------
    TimeoutError
        If any expected record/object is still missing when the timeout elapses.
    """
    s3_client = boto_session.client("s3")
    dynamodb = boto_session.resource("dynamodb")

    # Resolve all required resources once. Any 0/>1 match raises immediately (Libera Utils / SDC mismatch).
    archive_bucket_by_level: dict = {}
    needs_availability = any(isinstance(f, LiberaDataProductFilename) for f in libera_filenames)
    file_specs = []
    for libera_filename in libera_filenames:
        data_level = libera_filename.data_product_id.data_level
        if data_level not in archive_bucket_by_level:
            archive_bucket_by_level[data_level] = find_bucket_in_account_by_partial_name(
                boto_session, data_level.archive_bucket_name
            )
        spec = {
            "name": libera_filename.path.name,
            "bucket": archive_bucket_by_level[data_level],
            "key": f"{libera_filename.archive_prefix}/{libera_filename.path.name}",
            "is_data_product": isinstance(libera_filename, LiberaDataProductFilename),
        }
        if spec["is_data_product"]:
            spec["applicable_date"] = libera_filename.applicable_date.isoformat()
            spec["data_product_id"] = str(libera_filename.data_product_id)
            spec["version"] = libera_filename.filename_parts.version
        else:
            # L0: a CR (construction record) gets only its base metadata record (SK="#"); a PDS gets both a base
            # record and a product record (SK=applicable_date). We can't derive a PDS's applicable date here, so we
            # verify via record count: one record for a CR, two for a PDS.
            is_construction_record = libera_filename.data_product_id == DataProductIdentifier.l0_pds_cr
            spec["expected_metadata_count"] = 1 if is_construction_record else 2
        file_specs.append(spec)

    metadata_table = dynamodb.Table(
        find_dynamodb_table_in_account_by_partial_name(boto_session, FILE_METADATA_TABLE_PARTIAL_NAME)
    )
    availability_table = (
        dynamodb.Table(
            find_dynamodb_table_in_account_by_partial_name(boto_session, DATA_AVAILABILITY_TABLE_PARTIAL_NAME)
        )
        if needs_availability
        else None
    )

    # Outstanding checks as (file_index, check_name) pairs; each is dropped as soon as it passes.
    pending = set()
    for i, spec in enumerate(file_specs):
        pending.add((i, "archive"))
        pending.add((i, "metadata"))
        if spec["is_data_product"]:
            pending.add((i, "availability"))

    deadline = time.monotonic() + timeout
    while True:
        # Memoize availability lookups within a pass so files sharing a record aren't fetched twice.
        availability_memo: dict = {}
        newly_passed = set()
        for i, check_name in pending:
            spec = file_specs[i]
            if check_name == "archive":
                passed = _archive_object_exists(s3_client, spec["bucket"], spec["key"])
            elif check_name == "metadata":
                if spec["is_data_product"]:
                    # Data products: fetch the product record by its full (PK=basename, SK=applicable_date) key.
                    response = metadata_table.get_item(Key={"PK": spec["name"], "SK": spec["applicable_date"]})
                    passed = "Item" in response
                else:
                    # L0: the ingester writes a base record (SK="#") for every file plus a product record
                    # (SK=applicable_date) for PDS files. We can't derive a PDS's applicable date here, so verify by
                    # counting records under the unique basename PK: 1 for a CR, 2 for a PDS.
                    response = metadata_table.query(KeyConditionExpression=Key("PK").eq(spec["name"]))
                    passed = response.get("Count", 0) == spec["expected_metadata_count"]
            else:  # availability
                # The Data Availability table is keyed by PK=applicable_date, SK="<DataProductId>#<Version>".
                memo_key = (spec["applicable_date"], spec["data_product_id"], spec["version"])
                if memo_key not in availability_memo:
                    availability_response = availability_table.get_item(
                        Key={"PK": spec["applicable_date"], "SK": f"{spec['data_product_id']}#{spec['version']}"}
                    )
                    availability_memo[memo_key] = "Item" in availability_response
                passed = availability_memo[memo_key]
            if passed:
                newly_passed.add((i, check_name))

        pending -= newly_passed
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval)

    _log_ingestion_verification_summary(file_specs, pending)
    if pending:
        raise TimeoutError(
            f"Ingestion verification timed out after {timeout:.0f}s: {len(pending)} check(s) across "
            f"{len({i for i, _ in pending})} file(s) did not pass. See the logged summary for details."
        )


def _log_ingestion_verification_summary(file_specs: list[dict], pending: set) -> None:
    """Log a per-file PASS/MISSING summary of the ingestion verification checks."""
    for i, spec in enumerate(file_specs):
        check_names = ["archive", "metadata"] + (["availability"] if spec["is_data_product"] else [])
        statuses = [f"{name}={'MISSING' if (i, name) in pending else 'OK'}" for name in check_names]
        logger.info("Ingestion verification for %s: %s", spec["name"], ", ".join(statuses))


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
