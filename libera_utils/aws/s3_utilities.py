"""Module for S3 cli utilities"""

import argparse
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import boto3
from cloudpathlib import AnyPath, S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


def find_bucket_in_account_by_partial_name(boto_session, partial_name: str):
    """Finds a Bucket by substring match to the bucket name"""
    s3 = boto_session.client("s3")
    response = s3.list_buckets()

    name_pattern = re.compile(partial_name)
    matches = [b["Name"] for b in response["Buckets"] if name_pattern.search(b["Name"])]
    if len(matches) != 1:
        raise ValueError(
            f"Error finding a single bucket matching name {partial_name}. Found {len(matches)} buckets: {matches}"
        )
    return matches.pop()


def s3_put_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for s3-utils put CLI subcommand."""
    now = datetime.now(UTC)
    configure_task_logging(f"aws_s3_put_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    # The other two subcommands have more complex logic as functions with some shared arguments
    profile_name = parsed_args.profile
    local_file_path = AnyPath(parsed_args.file_path)
    s3_put_in_archive_for_processing_step(local_file_path, profile_name=profile_name)


def s3_put_in_archive_for_processing_step(
    path_to_file: Path | S3Path,
    *,
    profile_name: str = None,
):
    """Upload a file to the archive S3 bucket associated with a given processing step.

    Parameters
    ----------
    path_to_file : Path
        Local path to the file to upload
    profile_name : str, optional
        Boto3 profile name to use for authentication, by default None
    """
    libera_filename = LiberaDataProductFilename.from_file_path(path_to_file)

    boto_session = boto3.Session(profile_name=profile_name)
    archive_bucket_name = find_bucket_in_account_by_partial_name(
        boto_session, libera_filename.data_product_id.data_level.archive_bucket_name
    )
    bucket_path = S3Path(f"s3://{archive_bucket_name}")

    upload_path = libera_filename.generate_prefixed_path(bucket_path)
    smart_copy_file(path_to_file, upload_path)


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
