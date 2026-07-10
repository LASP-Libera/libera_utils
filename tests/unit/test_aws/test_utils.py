"""Tests for AWS utils functions"""

from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from libera_utils.aws import utils


@mock_aws
def test_get_l2_team_role_session_assumes_role(mock_s3_context_with_profile):
    """A session backed by assumed-role credentials is returned, inheriting the base session's region."""
    session = utils.get_l2_team_role_session(profile_name="test-profile")

    assert isinstance(session, boto3.Session)
    # The returned session carries (assumed-role) credentials and the base session's region.
    assert session.get_credentials() is not None
    assert session.region_name == boto3.Session(profile_name="test-profile").region_name


def test_get_l2_team_role_session_assumes_custom_role():
    """A custom (L2 team) role_name is assumed at the correct path-qualified ARN."""
    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:sts::123456789012:assumed-role/L2DeveloperBaseRole/session-name",
    }
    mock_sts.assume_role.return_value = {
        "Credentials": {"AccessKeyId": "a", "SecretAccessKey": "b", "SessionToken": "c"}
    }
    mock_base_session = MagicMock()
    mock_base_session.client.return_value = mock_sts

    with patch("libera_utils.aws.utils.boto3.Session", return_value=mock_base_session):
        utils.get_l2_team_role_session(profile_name="test-profile", role_name="L2Developer/L2-CloudFraction")

    assert (
        mock_sts.assume_role.call_args.kwargs["RoleArn"]
        == "arn:aws:iam::123456789012:role/L2Developer/L2-CloudFraction"
    )


def test_get_l2_team_role_session_raises_when_assume_role_denied():
    """If the base profile cannot assume the role, a helpful ValueError naming both roles is raised."""
    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:sts::123456789012:assumed-role/L2DeveloperBaseRole/session-name",
    }
    mock_sts.assume_role.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "not authorized to perform sts:AssumeRole"}},
        "AssumeRole",
    )
    mock_base_session = MagicMock()
    mock_base_session.client.return_value = mock_sts

    with patch("libera_utils.aws.utils.boto3.Session", return_value=mock_base_session):
        with pytest.raises(
            ValueError,
            match=(
                "Could not assume role L2Developer/LiberaUtils .* from base role L2DeveloperBaseRole .*"
                "contact the SDC team"
            ),
        ):
            utils.get_l2_team_role_session(profile_name="test-profile")

    mock_sts.assume_role.assert_called_once()
    assert mock_sts.assume_role.call_args.kwargs["RoleArn"] == "arn:aws:iam::123456789012:role/L2Developer/LiberaUtils"


def test_single_match_treats_partial_name_as_literal():
    """Regex metacharacters in partial_name are matched literally, not interpreted as a regex."""
    names = ["my.bucket", "myxbucket"]
    # As a literal substring, "my.bucket" matches only "my.bucket"; a regex "." would also match "myxbucket"
    # and make this ambiguous.
    assert utils._single_match_by_partial_name("my.bucket", names, resource_description="bucket") == "my.bucket"


@mock_aws
def test_find_bucket_ambiguity():
    """
    Test that find_bucket... raises ValueError if it finds 0 or >1 buckets.
    """
    s3 = boto3.client("s3", region_name="us-east-1")

    # Setup: Create two buckets that share a substring
    s3.create_bucket(Bucket="libera-l1b-cam-test-v1")
    s3.create_bucket(Bucket="libera-l1b-cam-test-v2")
    s3.create_bucket(Bucket="libera-l2-rad-test")

    session = boto3.Session()

    # Case 1: Ambiguity (finding 'l1b' matches v1 and v2)
    with pytest.raises(ValueError, match="Error finding a single bucket"):
        utils.find_bucket_in_account_by_partial_name(session, "l1b")

    # Case 2: No match
    with pytest.raises(ValueError, match="Error finding a single bucket"):
        utils.find_bucket_in_account_by_partial_name(session, "l0-packet")

    # Case 3: Success (Specific enough to match only one)
    # Matching "l2" should only find the one l2 bucket
    result = utils.find_bucket_in_account_by_partial_name(session, "l2")
    assert result == "libera-l2-rad-test"


@mock_aws
def test_find_event_bus_ambiguity():
    """Test that find_event_bus... raises ValueError if it finds 0 or >1 event buses."""
    events = boto3.client("events", region_name="us-east-1")
    events.create_event_bus(Name="SomeLiberaSDCEventBusOne")
    events.create_event_bus(Name="SomeLiberaSDCEventBusTwo")

    session = boto3.Session(region_name="us-east-1")

    # Ambiguity: two buses match
    with pytest.raises(ValueError, match="Error finding a single event bus"):
        utils.find_event_bus_in_account_by_partial_name(session, "LiberaSDCEventBus")

    # No match
    with pytest.raises(ValueError, match="Error finding a single event bus"):
        utils.find_event_bus_in_account_by_partial_name(session, "NonexistentBus")


@mock_aws
def test_find_dynamodb_table_ambiguity():
    """Test that find_dynamodb_table... raises ValueError if it finds 0 or >1 tables."""
    client = boto3.client("dynamodb", region_name="us-east-1")
    for name in ("SomeFileMetadataTableOne", "SomeFileMetadataTableTwo", "SomeDataAvailabilityTable"):
        client.create_table(
            TableName=name,
            KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

    session = boto3.Session(region_name="us-east-1")

    # Ambiguity: two tables match
    with pytest.raises(ValueError, match="Error finding a single DynamoDB table"):
        utils.find_dynamodb_table_in_account_by_partial_name(session, "FileMetadataTable")

    # No match
    with pytest.raises(ValueError, match="Error finding a single DynamoDB table"):
        utils.find_dynamodb_table_in_account_by_partial_name(session, "NonexistentTable")

    # Success: specific enough to match only one
    result = utils.find_dynamodb_table_in_account_by_partial_name(session, "DataAvailabilityTable")
    assert result == "SomeDataAvailabilityTable"
