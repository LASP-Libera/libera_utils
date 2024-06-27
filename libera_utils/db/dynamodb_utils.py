"""Module for database utilities"""
# Standard
import logging
# Installed
from datetime import datetime, timezone
from boto3 import resource as boto3_resource

logger = logging.getLogger(__name__)


def get_dynamodb_table(dynamo_table_name: str):
    """Get the DynamoDB table"""
    logger.debug(f"Getting DynamoDB table: {dynamo_table_name}")
    dynamo_resource = boto3_resource('dynamodb', region_name="us-west-2")
    dynamo_table = dynamo_resource.Table(dynamo_table_name)
    return dynamo_table


def create_ddb_metadata_file_item(filename: str,
                                  algorithm_version: str,
                                  include_archive_time: bool = False,
                                  additional_metadata: dict = None):
    """Write metadata record to DynamoDB for a single file """
    ddb_metadata_item = {
        'PK': filename,
        'SK': "#",
        'algorithm-version': algorithm_version
    }
    if include_archive_time:
        add_archive_time_to_ddb_item(ddb_metadata_item)
    if additional_metadata:
        ddb_metadata_item.update(additional_metadata)
    logger.debug(f"Metadata record written to DynamoDB for {filename}")
    return ddb_metadata_item


def add_archive_time_to_ddb_item(ddb_item: dict):
    """Add archive time to DynamoDB item"""
    ddb_item.update({'archive-time': datetime.now(timezone.utc).isoformat()})
    return ddb_item


def create_ddb_metadata_applicable_date_item(filename: str,
                                             data_level: str,
                                             data_type: str,
                                             applicable_date: str,
                                             data_subtype: str = None,
                                             additional_metadata: dict = None):
    """Write metadata record to DynamoDB for a single file """
    sort_key = f"#{data_level}#{data_type}"
    if data_subtype:
        sort_key += f"#{data_subtype}"
    ddb_metadata_item = {
        'PK': filename,
        'SK': sort_key,
        'applicable-date': applicable_date
    }
    if additional_metadata:
        ddb_metadata_item.update(additional_metadata)
    logger.debug("Metadata applicable-date record written to DynamoDB")
    return ddb_metadata_item
