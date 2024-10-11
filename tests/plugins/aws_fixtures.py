"""Plugin module for mocking AWS resources"""
# Standard
import string
import json
from pathlib import Path
import random
# Installed
import boto3
from cloudpathlib import S3Path, S3Client
from moto import mock_aws
import pytest
# Local
from libera_utils.config import config


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


@pytest.fixture
# TODO: Change this fixture scope to the entire test session
def mock_cloudwatch_context(monkeypatch):
    """Everything under/inherited by this runs in the mock_logs context manager"""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")  # CW requires region be set
    with mock_aws():
        yield boto3.resource('cloudwatch')


@pytest.fixture(scope='session', autouse=True)
def set_up_cloudpathlib_s3client(mock_aws_credentials, monkeypatch_session):
    """This sets the default client used for S3Path objects as a mocked S3 context.
    Make sure this code runs before any code that tries to instantiate an S3Path object without an explicit client.

    This fixture is session scoped so that we don't have to call it every time we use cloudpathlib
    """
    # Tell cloudpathlib to clear its local file cache whenever a file operation is completed.
    # https://cloudpathlib.drivendata.org/stable/caching/#file-cache-mode-close_file
    monkeypatch_session.setenv("CLOUDPATHLIB_FILE_CACHE_MODE", "close_file")
    with mock_aws():
        client = S3Client()
        client.set_as_default_client()


@pytest.fixture
def mock_s3_context():
    """Everything under/inherited by this runs in the mock_s3 context manager

    This fixture is function scoped so that S3 buckets get cleared between tests.
    """
    with mock_aws():
        # Yield the (mocked) s3 Resource object
        # (see boto3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/resources.html)
        yield boto3.resource('s3')


@pytest.fixture
def create_mock_bucket(mock_s3_context):
    """Returns a function that allows dynamic creation of s3 buckets with option to specify the name.

    Note: if the bucket already exists, this doesn't overwrite it. Previous contents will remain.
    Caution: If you create multiple objects at the same location, you may get conflicts"""
    s3 = mock_s3_context

    # The following call to Random() creates a locally seeded random generator. This prevents the pytest-randomly
    # seeded global PRN generator from creating the same "random" bucket names for every test.
    local_random = random.Random()

    def _create_bucket(bucket_name: str = None) -> s3.Bucket:
        """Creates a mock bucket, optionally with a custom name.

        Returns
        -------
        : s3.Bucket
        """
        if not bucket_name:
            bucket_name = ''.join(local_random.choice(string.ascii_letters) for _ in range(16))
        bucket = s3.Bucket(bucket_name)
        if not bucket.creation_date:  # If bucket doesn't already exist
            bucket.create()
            print(f"Created mock S3 bucket {bucket}.")
        else:
            print(f"Using existing mock S3 bucket {bucket}. You may see FileExistsErrors if you are writing the same"
                  f" file as a previous test due to the behavior of cloudpathlib S3Path objects.")
        return bucket

    yield _create_bucket


@pytest.fixture
def write_file_to_s3(mock_s3_context, create_mock_bucket):
    """Write file contents to mocked s3 bucket. If the bucket doesn't exist, it is created."""

    def _write(filepath: Path, uri: str or S3Path, exists_ok: bool = False) -> S3Path:
        """Write the contents of the file at filepath to the (mocked) S3 URI.

        Parameters
        ----------
        filepath : Path
            Path object pointing to the file to be put into the S3 bucket.
        uri : str
            Fully specified desired s3 object path (<bucket>/<key>)
        exists_ok : bool, Optional
            Whether it's ok to overwrite an existing object. Default is False.

        Returns
        -------
        : S3Path
            S3Path object
        """
        content = filepath.read_bytes()
        s3_path = S3Path(uri)
        create_mock_bucket(s3_path.bucket)  # Ensure bucket exists
        if not exists_ok and s3_path.exists():
            raise ValueError(f"Object {uri} already exists in mock bucket.")
        s3_path.mkdir(parents=True)  # Make additional directories (key paths) if necessary
        s3_path.write_bytes(content)
        print(f"Wrote {filepath} contents to (mocked) S3 object {s3_path.as_uri()}")
        return s3_path

    return _write


@pytest.fixture
def mock_secret_manager():
    """Everything under/inherited by this runs in the mock_secretmanager context manager"""
    with mock_aws():
        # Yield the (mocked) secretmanager client object
        # (see boto3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html)
        yield boto3.client("secretsmanager", region_name="us-west-2")


@pytest.fixture
def create_mock_secret_manager(mock_secret_manager):
    """Returns a function that allows dynamic creation of secret manager client with option to specify the name."""
    client = mock_secret_manager

    def _create_mock_secret_manager(secret_name: str, username="libera_unit_tester"):
        """Creates a mocked secret manager that works with the unit testing database"""
        host_name = config.get("LIBERA_DB_HOST")
        if not host_name:
            host_name = "localhost"
        secret_json_string = f'{{\n "host":"{host_name}",\n  "password":"testerpass",\n ' \
                             f'"dbname":"libera",\n "username":"{username}" }}\n'
        try:
            client.create_secret(Name=secret_name, SecretString=secret_json_string)
        except client.exceptions.ResourceExistsException:
            print(f"Mock secret {secret_name} already exists. Using existing secret.")

        return

    return _create_mock_secret_manager


@pytest.fixture
def mock_step_function():
    """Everything under/inherited by this runs in the mock_step_function context manager"""
    with mock_aws(config={'stepfunctions': {"execute_state_machine": True}}):
        # Yield the (mocked) stepfunction client object
        # (see boto3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html)
        yield boto3.client("stepfunctions", 'us-west-2')


@pytest.fixture
def make_step_function(mock_step_function, monkeypatch_session):
    """Creates a fake AWS step function"""
    client = mock_step_function

    def _make_step_function(sfn_name: str, status: str = "FAILED"):
        """Creates a fake AWS step function with the given name"""
        monkeypatch_session.setenv("SF_EXECUTION_HISTORY_TYPE", status)
        state_machine = client.create_state_machine(
            name=sfn_name,
            definition=json.dumps({
                "Comment": "A description of my state machine",
                "StartAt": "Pass",
                "States": {
                    "Pass": {
                        "Type": "Pass",
                        "End": True
                    }
                }
            }),
            roleArn='arn:aws:iam::123456789012:role/role-name',
        )
        return state_machine

    yield _make_step_function
