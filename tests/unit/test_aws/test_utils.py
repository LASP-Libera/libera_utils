"""Tests for AWS utils functions"""

from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from libera_utils.aws import utils


@mock_aws
def test_get_aws_account_number(mock_aws_credentials):
    """Test standard retrieval"""
    fake_id = utils.get_aws_account_number()
    assert fake_id == "123456789012"


@mock_aws
def test_get_aws_account_number_with_profile(mock_s3_context_with_profile):
    """Test retrieval using a specific profile name"""
    # This works because mock_s3_context_with_profile created the config file
    fake_id = utils.get_aws_account_number(profile_name="test-profile")
    assert fake_id == "123456789012"


@mock_aws
def test_get_libera_utils_session_assumes_role(mock_s3_context_with_profile):
    """A session backed by assumed-role credentials is returned, inheriting the base session's region."""
    session = utils.get_libera_utils_session(profile_name="test-profile")

    assert isinstance(session, boto3.Session)
    # The returned session carries (assumed-role) credentials and the base session's region.
    assert session.get_credentials() is not None
    assert session.region_name == boto3.Session(profile_name="test-profile").region_name


def test_get_libera_utils_session_raises_when_assume_role_denied():
    """If the base profile cannot assume the LiberaUtils role, a helpful ValueError is raised."""
    mock_sts = MagicMock()
    mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
    mock_sts.assume_role.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "not authorized to perform sts:AssumeRole"}},
        "AssumeRole",
    )
    mock_base_session = MagicMock()
    mock_base_session.client.return_value = mock_sts

    with patch("libera_utils.aws.utils.boto3.Session", return_value=mock_base_session):
        with pytest.raises(ValueError, match="not permitted to assume the L2Developer/LiberaUtils role"):
            utils.get_libera_utils_session(profile_name="test-profile")

    mock_sts.assume_role.assert_called_once()
    assert mock_sts.assume_role.call_args.kwargs["RoleArn"] == "arn:aws:iam::123456789012:role/L2Developer/LiberaUtils"


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
