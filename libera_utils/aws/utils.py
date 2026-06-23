"""Helper functions for AWS access"""

import logging
import re

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Canonical IAM role that Libera Utils CLI handlers assume to obtain the permissions they need. Users authenticate
# to a "base" role (e.g. via AWS SSO) that grants no permissions directly but is allowed to assume this role.
LIBERA_UTILS_ROLE_NAME = "L2Developer/LiberaUtils"


def get_libera_utils_session(
    profile_name: str | None = None, *, role_name: str = LIBERA_UTILS_ROLE_NAME
) -> boto3.Session:
    """Create a boto3 session that has assumed the LiberaUtils IAM role.

    Libera SDC users authenticate (via their AWS config/SSO or an explicit profile) to a "base" role that grants no
    permissions directly but is permitted to assume the canonical ``LiberaUtils`` role, which holds the permissions
    needed by the CLI. This function resolves the base credentials, assumes the role, and returns a new session
    backed by the assumed-role credentials.

    Parameters
    ----------
    profile_name : str, optional
        AWS profile name used to create the base session. If None, standard boto resolution is used (e.g. the
        ``AWS_PROFILE`` environment variable, the default profile, or an instance role).
    role_name : str, optional
        Name (or path-qualified name) of the IAM role to assume. Defaults to ``"L2Developer/LiberaUtils"``.

    Returns
    -------
    boto3.Session
        A session whose credentials are those of the assumed role. The region is inherited from the base session.

    Raises
    ------
    ValueError
        If the base profile is not permitted to assume the role.
    """
    # If profile_name is None, this uses standard resolution (env vars, AWS_PROFILE, default profile, instance role).
    base_session = boto3.Session(profile_name=profile_name)
    sts_client = base_session.client("sts")

    # get_caller_identity requires no permissions, so it works even from a base role with no direct permissions.
    account_id = sts_client.get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

    try:
        response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName="libera-utils-cli")
    except ClientError as err:
        raise ValueError(
            f"Could not assume role {role_name} ({role_arn}). This may be because your base profile/role does not "
            f"have the correct permissions."
        ) from err

    credentials = response["Credentials"]
    logger.info(f"Assumed role {role_arn} for Libera Utils CLI session.")
    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=base_session.region_name,
    )


def _single_match_by_partial_name(partial_name: str, names: list[str], *, resource_description: str) -> str:
    """Return the single name matching partial_name, raising if zero or more than one match is found.

    Parameters
    ----------
    partial_name : str
        Literal substring to search for within each candidate name. Regex metacharacters are escaped, so the match
        is a plain substring match (not a regex).
    names : list of str
        Candidate names to search.
    resource_description : str
        Human-readable singular description of the resource (e.g. "bucket"), used in the error message.

    Returns
    -------
    str
        The single matching name.
    """
    name_pattern = re.compile(re.escape(partial_name))
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


def find_dynamodb_table_in_account_by_partial_name(boto_session: boto3.Session, partial_name: str) -> str:
    """Finds a DynamoDB table by substring match to its name. Raises if zero or more than one table matches."""
    dynamodb = boto_session.client("dynamodb")
    table_names: list[str] = []
    paginator = dynamodb.get_paginator("list_tables")
    for page in paginator.paginate():
        table_names.extend(page["TableNames"])
    return _single_match_by_partial_name(partial_name, table_names, resource_description="DynamoDB table")


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
