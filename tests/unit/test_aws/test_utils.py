"""Tests for AWS utils functions"""
# Standard
# Installed
from moto import mock_aws
# Local
from libera_utils.aws import utils


@mock_aws
def test_get_aws_account_number():
    """Test that we can successfully get the AWS account number"""
    fake_id = utils.get_aws_account_number()
    assert fake_id == '123456789012'
