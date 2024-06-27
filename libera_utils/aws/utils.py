"""Helper functions for AWS access"""
# Standard
import logging
# Installed
import boto3

logger = logging.getLogger(__name__)


def get_aws_account_number(region_name="us-west-2"):
    """Get a users AWS account ID number
    Parameters
    ----------
    region_name : string
        Region that the users AWS account is on

    Returns
    -------
    account_id : int
        users account_id number
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='sts',
        region_name=region_name)
    account_id = client.get_caller_identity()["Account"]
    logger.info(f"AWS Account ID: {account_id}")
    return account_id
