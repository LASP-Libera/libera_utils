"""E2E tests read real S3, so the unit tests' session-wide AWS mocks are replaced here with no-ops."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_aws_credentials():
    """Leave the AWS credentials as the environment provides them."""


@pytest.fixture(scope="session", autouse=True)
def set_up_cloudpathlib_s3client():
    """Leave cloudpathlib on its default client, which uses the environment's credentials."""
