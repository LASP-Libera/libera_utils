"""Helper functions for AWS access"""

import logging
import re
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from libera_utils.constants import ProcessingStepIdentifier

logger = logging.getLogger(__name__)

# IAM path prefix shared by every role the L2 developers assume (the generic LiberaUtils role and the per-team L2
# roles). Roles are referenced as "<L2_DEVELOPER_ROLE_PATH>/<role>".
L2_DEVELOPER_ROLE_PATH = "L2Developer"

# Canonical generic IAM role that Libera Utils CLI handlers assume to obtain the permissions they need. Users
# authenticate to a "base" role (e.g. via AWS SSO) that grants no permissions directly but is allowed to assume it.
LIBERA_UTILS_ROLE_NAME = f"{L2_DEVELOPER_ROLE_PATH}/LiberaUtils"

# Partial name used to uniquely identify the SDC central EventBridge bus by regex search (see find_*_by_partial_name).
# Both the manual ingest (s3-utils put) and manual processing flows emit events to this single bus.
SDC_EVENT_BUS_PARTIAL_NAME = "LiberaSDCEventBus"


def get_l2_team_role_session(
    profile_name: str | None = None, *, role_name: str = LIBERA_UTILS_ROLE_NAME
) -> boto3.Session:
    """Create a boto3 session that has assumed an L2 team IAM role.

    Libera SDC users authenticate (via their AWS config/SSO or an explicit profile) to a "base" role that grants no
    permissions directly but is permitted to assume one or more L2 team roles. This includes the generic
    ``LiberaUtils`` role (used by ``s3-utils put`` and ``manual-processing``) as well as per-team L2 roles (used by
    ``ecr-upload`` to push to a specific algorithm's ECR repo). This function resolves the base credentials, assumes
    the requested role, and returns a new session backed by the assumed-role credentials.

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
        If the base profile is not permitted to assume the role. The message names both the base role and the
        target role.
    """
    # If profile_name is None, this uses standard resolution (env vars, AWS_PROFILE, default profile, instance role).
    base_session = boto3.Session(profile_name=profile_name)
    sts_client = base_session.client("sts")

    # get_caller_identity requires no permissions, so it works even from a base role with no direct permissions.
    # Its Arn identifies the base role the user is currently authenticated as.
    base_identity = sts_client.get_caller_identity()
    account_id = base_identity["Account"]
    base_role_arn = base_identity["Arn"]
    # The base role name is the resource-name segment of the ARN, e.g.
    # arn:aws:sts::<acct>:assumed-role/<BaseRoleName>/<session> -> <BaseRoleName>.
    base_role_name = base_role_arn.split("/")[1] if "/" in base_role_arn else base_role_arn
    role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"

    try:
        response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName="libera-utils-cli")
    except ClientError as err:
        raise ValueError(
            f"Could not assume role {role_name} ({role_arn}) from base role {base_role_name} ({base_role_arn}). "
            f"Check that you are using the profile that logs in as the L2 Developer base role. If this error "
            f"persists, contact the SDC team."
        ) from err

    credentials = response["Credentials"]
    logger.info(f"Assumed role {role_arn} for Libera Utils CLI session.")
    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name=base_session.region_name,
    )


def _resolve_algorithm_specific_session(
    algorithm: "ProcessingStepIdentifier", profile_name: str | None = None
) -> boto3.Session:
    """Resolve the boto3 session to use for operations scoped to a specific algorithm.

    L2 algorithms (those with a ``ProcessingStepIdentifier.l2_team_iam_role``) are owned by an L2 team whose
    per-team L2 Team Role holds the permissions for that algorithm's resources, so this assumes that role. All
    other algorithms (SPICE, L1B, scene-id, and other SDC-owned steps) use the ambient/``--profile`` session
    directly. This lets SDC developers, whose admin credentials cannot assume the LiberaUtils/L2 roles, operate
    on SDC-owned algorithms with their own ambient credentials instead of failing on a role-assumption chain
    they are not part of.

    Parameters
    ----------
    algorithm : ProcessingStepIdentifier
        The processing step whose algorithm is being operated on.
    profile_name : str, optional
        AWS profile name from the CLI (``--profile``), or None for default resolution.

    Returns
    -------
    boto3.Session
        The session to use: the per-team L2 Team Role for L2 algorithms, or the ambient session otherwise.

    Raises
    ------
    ValueError
        If the algorithm requires an L2 Team Role that the base profile cannot assume.
    """
    team_role = algorithm.l2_team_iam_role
    if team_role is None:
        logger.info(f"{algorithm} is not an L2 algorithm; using the ambient/--profile session.")
        return boto3.Session(profile_name=profile_name)

    role_name = f"{L2_DEVELOPER_ROLE_PATH}/{team_role}"
    logger.info(f"{algorithm} is an L2 algorithm; assuming the {role_name} role.")
    return get_l2_team_role_session(profile_name=profile_name, role_name=role_name)


def _session_region(boto_session: boto3.Session) -> str:
    """Return the AWS region configured for the session, raising a clear error if none is set.

    Region is intentionally taken from the user's AWS configuration (profile, ``AWS_REGION``/``AWS_DEFAULT_REGION``,
    or instance metadata) rather than hard-coded, so it follows the caller's environment. A session with no region
    resolved cannot address regional resources (ECR, EventBridge, Batch), so this raises rather than silently
    guessing one.

    Parameters
    ----------
    boto_session : boto3.Session
        The session whose region should be used.

    Returns
    -------
    str
        The resolved AWS region name (e.g. ``"us-west-2"``).

    Raises
    ------
    ValueError
        If the session has no region configured.
    """
    region = boto_session.region_name
    if not region:
        raise ValueError(
            "No AWS region is configured. Set a region in your AWS profile, or via the AWS_REGION / "
            "AWS_DEFAULT_REGION environment variable, and re-run."
        )
    return region


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
