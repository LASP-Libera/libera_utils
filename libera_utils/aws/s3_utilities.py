"""Module for S3 cli utilities"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import boto3
from cloudpathlib import AnyPath, S3Path

from libera_utils.aws import constants
from libera_utils.aws.constants import LiberaAccountSuffix as AccountSuffix
from libera_utils.io.filenaming import AbstractValidFilename
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


def s3_put_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for s3-utils put CLI subcommand."""
    now = datetime.now(UTC)
    configure_task_logging(f"aws_s3_put_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    # The other two subcommands have more complex logic as functions with some shared arguments
    algorithm_name_string = parsed_args.algorithm_name
    processing_step = constants.ProcessingStepIdentifier(algorithm_name_string)
    account_suffix = parsed_args.account_suffix
    local_file_path = AnyPath(parsed_args.file_path)
    s3_put_in_archive_for_processing_step(local_file_path, processing_step, account_suffix=account_suffix)


def s3_put_in_archive_for_processing_step(
    path_to_file: Path | S3Path,
    processing_step: str | constants.ProcessingStepIdentifier,
    *,
    account_suffix: str | AccountSuffix | None = AccountSuffix.STAGE,
):
    """Upload a file to the archive S3 bucket associated with a given processing step.

    Parameters
    ----------
    path_to_file : Path
        Local path to the file to upload
    processing_step : str
        processing_step : Union[str, constants.ProcessingStepIdentifier]
        Processing step ID string or object. Used to infer the S3 archive bucket name.
    account_suffix : Union[str, constants.LiberaAccountSuffix], optional
        Account suffix for the bucket name, by default constants.LiberaAccountSuffix.STAGE
    """
    if isinstance(processing_step, str):
        processing_step = constants.ProcessingStepIdentifier(processing_step)
    if isinstance(account_suffix, str):
        account_suffix = AccountSuffix(account_suffix)

    bucket_name = constants.ProcessingStepIdentifier.get_archive_bucket_name(
        processing_step, account_suffix=account_suffix
    )
    bucket_path = S3Path(f"s3://{bucket_name}")
    filename_object = AbstractValidFilename.from_file_path(path_to_file)

    upload_path = filename_object.generate_prefixed_path(bucket_path)
    smart_copy_file(path_to_file, upload_path)


def s3_list_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for s3-utils list CLI subcommand."""
    now = datetime.now(UTC)
    configure_task_logging(f"aws_upload_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    # The other two subcommands have more complex logic as functions with some shared arguments
    algorithm_name_string = parsed_args.algorithm_name
    processing_step = constants.ProcessingStepIdentifier(algorithm_name_string)
    account_suffix = parsed_args.account_suffix
    s3_list_archive_files(processing_step, account_suffix=account_suffix)


def s3_list_archive_files(
    processing_step: str | constants.ProcessingStepIdentifier,
    *,
    account_suffix: str | AccountSuffix | None = AccountSuffix.STAGE,
    print_out: bool | None = False,
) -> list:
    """List all files in an archive S3 bucket for a given processing step.

    Parameters
    ----------
    processing_step : str
        Processing step ID string. Used to infer the S3 archive bucket name.
    account_suffix : Union[str, constants.LiberaAccountSuffix], optional
        Account suffix for the bucket name, by default constants.LiberaAccountSuffix.STAGE
    print_out : bool, optional
        Print the list of files to the console, by default False

    Returns
    -------
    bucket_objects : list
        S3Path objects for each file in the bucket
    """
    if isinstance(processing_step, str):
        processing_step = constants.ProcessingStepIdentifier(processing_step)
    if isinstance(account_suffix, str):
        account_suffix = AccountSuffix(account_suffix)

    bucket_name = constants.ProcessingStepIdentifier.get_archive_bucket_name(
        processing_step, account_suffix=account_suffix
    )
    client = boto3.client("s3")

    bucket_objects = [
        S3Path(f"s3://{bucket_name}/{obj['Key']}")
        for obj in client.list_objects_v2(Bucket=bucket_name).get("Contents", [])
    ]

    if print_out:
        for obj in bucket_objects:
            print(obj)
    return bucket_objects


def s3_copy_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for s3-utils cp CLI subcommand."""
    now = datetime.now(UTC)
    configure_task_logging(f"aws_s3_cp_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    # The other two subcommands have more complex logic as functions with some shared arguments
    current_path = AnyPath(parsed_args.source_path)
    destination_path = AnyPath(parsed_args.dest_path)
    delete = parsed_args.delete
    s3_copy_file(current_path, destination_path, delete=delete)


# The copy functionality already exists, use it from the smart_open module.
s3_copy_file = smart_copy_file
