"""Plugin module for mocking S3 buckets"""
# Standard
from pathlib import Path
# Installed
import boto3
from cloudpathlib import S3Path, S3Client
from moto import mock_s3
import pytest


@pytest.fixture(scope='session', autouse=True)
def mock_aws_credentials(monkeypatch_session):
    """Mocked AWS Credentials for moto."""
    monkeypatch_session.setenv('AWS_ACCESS_KEY_ID', 'testing')
    monkeypatch_session.setenv('AWS_SECRET_ACCESS_KEY', 'testing')
    monkeypatch_session.setenv('AWS_SECURITY_TOKEN', 'testing')
    monkeypatch_session.setenv('AWS_SESSION_TOKEN', 'testing')
    monkeypatch_session.delenv('AWS_PROFILE', raising=False)
    monkeypatch_session.delenv('AWS_REGION', raising=False)
    monkeypatch_session.delenv('AWS_DEFAULT_REGION', raising=False)


@pytest.fixture(scope='session', autouse=True)
def set_up_cloudpathlib_s3client(mock_aws_credentials):
    """This sets the default client used for S3Path objects. Make sure this code runs before any code that tries to
    instantiate an S3Path object without an explicit client."""
    with mock_s3():
        client = S3Client()
        client.set_as_default_client()


@pytest.fixture
def mock_s3_context():
    """Everything under/inherited by this runs in the mock_s3 context manager"""
    with mock_s3():
        # Yield the (mocked) s3 Resource object
        # (see boto3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/resources.html)
        yield boto3.resource('s3')


@pytest.fixture
def create_mock_bucket(mock_s3_context):
    """Returns a function that allows dynamic creation of s3 buckets
    Note: if the bucket already exists, this doesn't overwrite it. Previous contents will remain."""
    s3 = mock_s3_context

    def _create_bucket(bucket_name: str):
        bucket = s3.Bucket(bucket_name)
        if not bucket.creation_date:
            bucket.create()
            print(f"Created mock S3 bucket {bucket}.")
        return bucket
    return _create_bucket


@pytest.fixture
def write_file_to_s3(mock_s3_context, create_mock_bucket):
    """Write file contents to mocked s3 bucket. If the bucket doesn't exist, it is created."""
    def _write(filepath: Path, uri: str):
        """Write the contents of the file at filepath to the (mocked) S3 URI.

        Parameters
        ----------
        filepath : Path
            Path object pointing to the file to be put into the S3 bucket.
        uri : str
            URI of the mock bucket.

        Returns
        -------
        : S3Path
            S3Path object
        """
        content = filepath.read_bytes()
        s3_path = S3Path(uri)
        create_mock_bucket(s3_path.bucket)  # Ensure bucket exists
        s3_path.mkdir(parents=True)  # Make additional directories (key paths) if necessary
        s3_path.write_bytes(content)
        print(f"Wrote {filepath} contents to (mocked) S3 object {s3_path.as_uri()}")
        return s3_path
    return _write
