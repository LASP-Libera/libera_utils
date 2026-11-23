"""Module for S3 cli utilities"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from operator import itemgetter
from pathlib import Path

import boto3
from cloudpathlib import AnyPath, S3Path

from libera_utils.aws.utils import (
    SDC_EVENT_BUS_PARTIAL_NAME,
    find_bucket_in_account_by_partial_name,
    find_dynamodb_table_in_account_by_partial_name,
    find_event_bus_in_account_by_partial_name,
    get_l2_team_role_session,
)
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import (
    AbstractValidFilename,
    AncillaryDataProductFilename,
    L0Filename,
    LiberaDataProductFilename,
    LiberaGroundCcsdsFilename,
    LiberaVersionPartitionedDataProductFilename,
    PathType,
)
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# Partial name used to uniquely identify the SDC Ingest Dropbox bucket by regex search (see find_*_by_partial_name in
# aws.utils). The SDC event bus partial name is shared via aws.utils.SDC_EVENT_BUS_PARTIAL_NAME.
INGEST_DROPBOX_BUCKET_PARTIAL_NAME = "ingestdropbox"
FILE_METADATA_TABLE_PARTIAL_NAME = "FileMetadataTable"

# These values are part of the NewFilesAvailable event contract and must match exactly what the SDC expects.
NEW_FILES_AVAILABLE_EVENT_SOURCE = "manual-processing"
NEW_FILES_AVAILABLE_EVENT_DETAIL_TYPE = "NewFilesAvailableEventDetail"

# Ingestion verification (--verify) polling parameters.
DEFAULT_VERIFY_TIMEOUT_SECONDS = 300.0  # 5 minutes
VERIFY_POLL_INTERVAL_SECONDS = 10.0

# Maximum number of files described by a single NewFilesAvailable event. The Data Ingester archives a
# record's files serially inside one 10 minute Lambda invocation, at a measured 38-47 s per ground CCSDS
# capture, so a full day of two-hour captures in one event exceeds the timeout. Five leaves better than
# half the budget as headroom for larger files.
# TODO[LIBSDC-911]: remove once the Data Ingester fans L0 ingest out to one dataset per invocation.
MAX_FILES_PER_INGEST_EVENT = 5

ManualIngestFilename = (
    L0Filename
    | LiberaDataProductFilename
    | LiberaVersionPartitionedDataProductFilename
    | LiberaGroundCcsdsFilename
    | AncillaryDataProductFilename
)

# The File Metadata base record of every ingested file (sort key "#") carries these attributes. The Data Ingester sets
# ingested-time when it ingests the file and archived-time once it has moved the file to its archive bucket.
_BASE_RECORD_SORT_KEY = "#"
_INGESTED_TIME_ATTRIBUTE = "ingested-time"
_ARCHIVED_TIME_ATTRIBUTE = "archived-time"


def _validate_filename_for_ingest(path: PathType) -> ManualIngestFilename:
    """Validate a path as a filename eligible for manual ingest.

    Eligible filenames are L0, ground CCSDS, daily data product, version-partitioned data product and external
    ancillary product filenames.

    Manifest and any other filename types are rejected.

    Parameters
    ----------
    path : Path or S3Path
        Path to the file to validate.

    Returns
    -------
    L0Filename, LiberaGroundCcsdsFilename, LiberaDataProductFilename, LiberaVersionPartitionedDataProductFilename, or AncillaryDataProductFilename
        The parsed filename object.

    Raises
    ------
    ValueError
        If the name is not one of the ingestable forms.
    """
    try:
        filename = AbstractValidFilename.from_file_path(path)
    except ValueError as err:
        raise ValueError(f"File {path} cannot be manually ingested: {err}") from err
    if not isinstance(filename, ManualIngestFilename):
        raise ValueError(
            f"File {path} is not an L0, ground CCSDS, daily or version-partitioned Libera data product, or external "
            "ancillary product filename and cannot be manually ingested. Check the name against the File Naming "
            "Conventions documentation."
        )
    return filename


def s3_put_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``s3-utils put`` subcommand.

    Stages one or more Libera data product files into the SDC Ingest Dropbox bucket and emits one or more
    ``NewFilesAvailable`` events to the SDC event bus. The SDC Data Ingester service then archives the files and
    creates the associated file metadata and data availability records. This is the manual analog of the automated
    ingest that happens for files produced by SDC processing steps.

    With ``--verify``, every file is staged and then each one's ingestion is verified from its File Metadata base
    record (see ``verify_ingestion``).

    Raises
    ------
    TimeoutError
        If ``--verify`` is given and the files are not all ingested and archived within ``--timeout`` seconds.
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
            "Staged %d file(s) to the Ingest Dropbox and emitted the NewFilesAvailable event(s). The SDC Data "
            "Ingester should now be running; it may take a few minutes for the files to appear in their archive "
            "bucket.",
            len(local_file_paths),
        )


def manual_ingest_data_products(
    paths_to_files: list[Path],
    *,
    boto_session: boto3.Session,
) -> list[ManualIngestFilename]:
    """Stage data product files to the Ingest Dropbox and emit NewFilesAvailable events for them.

    The SDC Data Ingester picks up the staged files and handles archiving them in the correct bucket as well as
    creating file metadata and data availability records.

    Files are staged in full before any event is emitted, then described across as many events as
    ``_chunk_files_for_ingest_events`` requires: the ingester handles one event per Lambda invocation and
    archives that event's files serially against a 10 minute timeout.

    Parameters
    ----------
    paths_to_files : list of Path
        Local filesystem paths to the files to ingest. Each must be a validly named L0, ground CCSDS, daily or
        version-partitioned Libera data product, or external ancillary product file.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions. Created once by the CLI handler and passed in so that the
        same authenticated session is used throughout the workflow.

    Returns
    -------
    list of L0Filename, LiberaGroundCcsdsFilename, LiberaDataProductFilename, LiberaVersionPartitionedDataProductFilename, or AncillaryDataProductFilename
        The validated filename objects for the staged files (useful for subsequent verification).

    Raises
    ------
    ValueError
        If a filename is not an ingestable filename, which is checked before anything is staged, or if
        the dropbox bucket or the event bus cannot be identified by name.
    RuntimeError
        If EventBridge rejects an event. The events before it were already emitted; running the command again
        with the same files is safe, because the Data Ingester skips files it has already archived.
    """
    # Validate every filename up front so we don't stage a partial set before discovering a bad name.
    libera_filenames = [_validate_filename_for_ingest(path) for path in paths_to_files]

    dropbox_bucket_name = find_bucket_in_account_by_partial_name(boto_session, INGEST_DROPBOX_BUCKET_PARTIAL_NAME)
    s3_client = boto_session.client("s3")

    # Stage all files first. Only if every upload succeeds do we emit any NewFilesAvailable events.
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

    for chunk in _chunk_files_for_ingest_events(files, libera_filenames):
        put_new_files_available_event(chunk, boto_session=boto_session)
    return libera_filenames


# TODO[LIBSDC-911]: remove once the Data Ingester fans L0 ingest out to one dataset per invocation.
def _chunk_files_for_ingest_events(
    files: list[dict],
    libera_filenames: list[ManualIngestFilename],
) -> list[list[dict]]:
    """Split file descriptors into per-event groups small enough for one Data Ingester invocation.

    Chunks hold at most ``MAX_FILES_PER_INGEST_EVENT`` entries, except that a construction record
    (``file_number`` 0) and the PDS files that share its SCID, first APID, created time and numeric ID
    are kept in one chunk, CR first, regardless of count: the ingester archives a CR ahead of the PDS
    files that read it only within a single event. PDS files whose construction record is not among
    ``files`` are split at ``MAX_FILES_PER_INGEST_EVENT`` like any other files.

    Parameters
    ----------
    files : list of dict
        File descriptors in the same order as ``libera_filenames``.
    libera_filenames : list of L0Filename, LiberaGroundCcsdsFilename, LiberaDataProductFilename, or LiberaVersionPartitionedDataProductFilename
        The parsed filenames corresponding to ``files``.

    Returns
    -------
    list of list of dict
        Chunks of file descriptors, one per NewFilesAvailable event. Every input descriptor appears in
        exactly one chunk.
    """
    l0_groups: dict[tuple, list[tuple[int, dict]]] = {}
    other_files = []
    for f, name in zip(files, libera_filenames, strict=True):
        if isinstance(name, L0Filename):
            parts = name.filename_parts
            key = (parts.scid, parts.first_apid, parts.created_time, parts.numeric_id)
            l0_groups.setdefault(key, []).append((parts.file_number, f))
        else:
            other_files.append(f)

    def split_at_limit(entries: list[dict]) -> list[list[dict]]:
        return [entries[i : i + MAX_FILES_PER_INGEST_EVENT] for i in range(0, len(entries), MAX_FILES_PER_INGEST_EVENT)]

    chunks = split_at_limit(other_files)
    for group in l0_groups.values():
        ordered = [f for _, f in sorted(group, key=itemgetter(0))]
        if any(file_number == 0 for file_number, _ in group):
            chunks.append(ordered)
        else:
            chunks.extend(split_at_limit(ordered))
    return chunks


def put_new_files_available_event(files: list[dict], *, boto_session: boto3.Session) -> None:
    """Emit a single NewFilesAvailable event to the SDC event bus.

    Parameters
    ----------
    files : list of dict
        File descriptors matching the NewFilesAvailableEventDetail ``files`` schema (``type``, ``uri``, ``name``,
        ``size``).
    boto_session : boto3.Session
        Boto3 session used to discover the event bus and put the event. The AWS region is derived from this session.

    Raises
    ------
    ValueError
        If the event bus cannot be identified by name.
    RuntimeError
        If EventBridge reports the entry as failed.
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
            f"Failed to put NewFilesAvailable event for {[f['name'] for f in files]} to event bus "
            f"{event_bus_name}. Response entries: {response['Entries']}. Any earlier events from this run were "
            f"already emitted; re-run the ingest with the same files, which the Data Ingester skips if archived."
        )

    logger.info(f"Put NewFilesAvailable event with {len(files)} file(s) to event bus {event_bus_name}")


def verify_ingestion(
    libera_filenames: list[ManualIngestFilename],
    *,
    boto_session: boto3.Session,
    timeout: float = DEFAULT_VERIFY_TIMEOUT_SECONDS,
    poll_interval: float = VERIFY_POLL_INTERVAL_SECONDS,
) -> None:
    """Block until every staged file is confirmed ingested and archived, or raise on timeout.

    A file is confirmed when its File Metadata base record, ``(PK=<basename>, SK="#")``, exists and has both
    ``ingested-time``, which the Data Ingester sets when it ingests the file, and ``archived-time``, which it sets once
    it has moved the file to its archive bucket. Every kind of staged file has a base record, so every file is checked
    the same way.

    The File Metadata table is resolved once up front; finding zero tables or more than one raises immediately (it
    indicates a mismatch between Libera Utils and the deployed SDC). Each file's base record is read every
    ``poll_interval`` seconds until it passes. A per-file summary is always logged; if any file has not passed at
    ``timeout``, a ``TimeoutError`` is raised.

    Parameters
    ----------
    libera_filenames : list of AbstractDataProductFilename
        The L0, ground CCSDS, Libera data product and external ancillary filenames staged for ingest, as returned by
        ``manual_ingest_data_products``.
    boto_session : boto3.Session
        Boto3 session used for all (read-only) AWS interactions.
    timeout : float, optional
        Maximum number of seconds to wait for full ingestion. Default 300 (5 minutes).
    poll_interval : float, optional
        Number of seconds between polling passes. Default 10.

    Raises
    ------
    ValueError
        If zero or more than one File Metadata table is found in the account.
    TimeoutError
        If any file's base record is still missing, or lacks ``ingested-time`` or ``archived-time``, when the timeout
        elapses.
    """
    metadata_table = boto_session.resource("dynamodb").Table(
        find_dynamodb_table_in_account_by_partial_name(boto_session, FILE_METADATA_TABLE_PARTIAL_NAME)
    )
    names = [libera_filename.path.name for libera_filename in libera_filenames]
    # The last state seen for each file not yet verified: what its base record is missing.
    missing = dict.fromkeys(names, "no base record")

    deadline = time.monotonic() + timeout
    while True:
        for name in list(missing):
            item = metadata_table.get_item(Key={"PK": name, "SK": _BASE_RECORD_SORT_KEY}).get("Item")
            if item is None:
                missing[name] = "no base record"
                continue
            unset = [attr for attr in (_INGESTED_TIME_ATTRIBUTE, _ARCHIVED_TIME_ATTRIBUTE) if not item.get(attr)]
            if unset:
                missing[name] = f"{' and '.join(unset)} not set"
            else:
                del missing[name]
        if not missing or time.monotonic() >= deadline:
            break
        time.sleep(poll_interval)

    _log_ingestion_verification_summary(names, missing)
    if missing:
        raise TimeoutError(
            f"Ingestion verification timed out after {timeout:.0f}s: {len(missing)} of {len(names)} file(s) are not "
            "yet ingested and archived. See the logged summary for what each one's File Metadata base record lacks."
        )


def _log_ingestion_verification_summary(names: list[str], missing: dict[str, str]) -> None:
    """Log a per-file summary of ingestion verification: PASS, or what the file's base record lacks."""
    for name in names:
        logger.info(
            "Ingestion verification for %s: %s", name, f"MISSING ({missing[name]})" if name in missing else "PASS"
        )


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
