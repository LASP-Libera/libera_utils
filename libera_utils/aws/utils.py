"""Helper functions for AWS access"""

import logging
import re

import boto3

logger = logging.getLogger(__name__)


def _single_match_by_partial_name(partial_name: str, names: list[str], *, resource_description: str) -> str:
    """Return the single name matching partial_name, raising if zero or more than one match is found.

    Parameters
    ----------
    partial_name : str
        Regex/substring pattern to search for within each candidate name.
    names : list of str
        Candidate names to search.
    resource_description : str
        Human-readable singular description of the resource (e.g. "bucket"), used in the error message.

    Returns
    -------
    str
        The single matching name.
    """
    name_pattern = re.compile(partial_name)
    matches = [name for name in names if name_pattern.search(name)]
    if len(matches) != 1:
        raise ValueError(
            f"Error finding a single {resource_description} matching name {partial_name}. "
            f"Found {len(matches)} matches: {matches}"
        )
    return matches.pop()


def find_bucket_in_account_by_partial_name(boto_session: boto3.Session, partial_name: str) -> str:
    """Finds a bucket by substring match to the bucket name. Raises if zero or more than one bucket matches."""
    s3 = boto_session.client("s3")
    response = s3.list_buckets()
    return _single_match_by_partial_name(
        partial_name, [b["Name"] for b in response["Buckets"]], resource_description="bucket"
    )


def find_event_bus_in_account_by_partial_name(boto_session: boto3.Session, partial_name: str) -> str:
    """Finds an EventBridge event bus by substring match to its name. Raises if zero or more than one bus matches."""
    events = boto_session.client("events")
    response = events.list_event_buses()
    return _single_match_by_partial_name(
        partial_name, [bus["Name"] for bus in response["EventBuses"]], resource_description="event bus"
    )


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
