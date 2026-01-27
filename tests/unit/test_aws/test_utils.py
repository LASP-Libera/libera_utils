"""Tests for AWS utils functions"""

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
