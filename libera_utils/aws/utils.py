"""Helper functions for AWS access"""

import logging

import boto3

logger = logging.getLogger(__name__)


def get_aws_account_number(region_name="us-west-2", profile_name=None):
    """Get a users AWS account ID number

    Parameters
    ----------
    region_name : string
        Region that the users AWS account is on
    profile_name : str, optional
        The name of the AWS profile to use for credentials.

    Returns
    -------
    account_id : str
        users account_id number
    """
    # If profile_name is None, this uses standard resolution (Env vars, default, IAM role)
    session = boto3.Session(profile_name=profile_name)

    # Create the client from the session to ensure it uses the profile's creds
    client = session.client(service_name="sts", region_name=region_name)

    account_id = client.get_caller_identity()["Account"]
    logger.info(f"AWS Account ID: {account_id}")
    return account_id
