"""Plugin module for mocking AWS resources"""

import json
import random
import string
from pathlib import Path

import boto3
import pytest
from cloudpathlib import S3Client, S3Path
from moto import mock_aws

from libera_utils.config import config


@pytest.fixture(scope="session", autouse=True)
def mock_aws_credentials(monkeypatch_session):
    """Mocked AWS Credentials for moto."""
    monkeypatch_session.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch_session.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch_session.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch_session.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch_session.delenv("AWS_REGION", raising=False)
    monkeypatch_session.delenv("AWS_DEFAULT_REGION", raising=False)


@pytest.fixture
# TODO[LIBSDC-616]: Change this fixture scope to the entire test session
def mock_cloudwatch_context(monkeypatch):
    """Everything under/inherited by this runs in the mock_logs context manager"""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")  # CW requires region be set
    with mock_aws():
        yield boto3.resource("cloudwatch")


@pytest.fixture(scope="session", autouse=True)
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
def mock_s3_context(mock_aws_credentials):
    """Simple S3 context using default environment creds"""
    with mock_aws():
        session = boto3.Session()
        yield session.resource("s3", region_name="us-east-1")


@pytest.fixture
def mock_s3_context_with_profile(mock_aws_credentials, monkeypatch, tmp_path):
    """
    S3 context that sets up a specific 'test-profile' in a config file.
    Use this when testing code that specifically requests profile_name='test-profile'.
    """
    config_file = tmp_path / "fake_config"
    config_file.write_text("[profile test-profile]\nregion=us-east-1")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))

    creds_file = tmp_path / "fake_credentials"
    creds_file.write_text(
        "[test-profile]\naws_access_key_id=testing\naws_secret_access_key=testing\naws_session_token=testing\n"
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds_file))

    with mock_aws():
        # Yield the (mocked) s3 Resource object
        # (see boto3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/resources.html)
        # We specify the region because S3 requires us-east-1 to be used for bucket
        # creation requests. If the machine running the tests provides default regions,
        # it can cause tests to fail.
        session = boto3.Session(profile_name="test-profile")
        yield session.resource("s3", region_name="us-east-1")


@pytest.fixture
def create_mock_bucket(mock_s3_context_with_profile):
    """Returns a function that allows dynamic creation of s3 buckets with option to specify the name.

    Note: if the bucket already exists, this doesn't overwrite it. Previous contents will remain.
    Caution: If you create multiple objects at the same location, you may get conflicts"""
    s3 = mock_s3_context_with_profile

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
            bucket_name = "".join(local_random.choice(string.ascii_letters) for _ in range(16))
        bucket = s3.Bucket(bucket_name)
        if not bucket.creation_date:  # If bucket doesn't already exist
            bucket.create()
            print(f"Created mock S3 bucket {bucket}.")
        else:
            print(
                f"Using existing mock S3 bucket {bucket}. You may see FileExistsErrors if you are writing the same"
                f" file as a previous test due to the behavior of cloudpathlib S3Path objects."
            )
        return bucket

    return _create_bucket


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
        secret_json_string = (
            f'{{\n "host":"{host_name}",\n  '
            f'"password":"testerpass",\n '
            f'"dbname":"libera",\n '
            f'"username":"{username}" }}\n'
        )
        try:
            client.create_secret(Name=secret_name, SecretString=secret_json_string)
        except client.exceptions.ResourceExistsException:
            print(f"Mock secret {secret_name} already exists. Using existing secret.")

        return

    return _create_mock_secret_manager


@pytest.fixture
def mock_step_function():
    """Everything under/inherited by this runs in the mock_step_function context manager"""
    with mock_aws(config={"stepfunctions": {"execute_state_machine": True}}):
        # Yield the (mocked) stepfunction client object
        # (see boto3 docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/clients.html)
        yield boto3.client("stepfunctions", "us-west-2")


@pytest.fixture
def make_step_function(mock_step_function, monkeypatch_session):
    """Creates a fake AWS step function"""
    client = mock_step_function

    def _make_step_function(sfn_name: str, status: str = None):
        """Creates a fake AWS step function with the given name"""
        monkeypatch_session.delenv("SF_EXECUTION_HISTORY_TYPE", raising=False)
        if status is not None:
            monkeypatch_session.setenv("SF_EXECUTION_HISTORY_TYPE", status)
        state_machine = client.create_state_machine(
            name=sfn_name,
            definition=json.dumps(
                {
                    "Comment": "A simple test state machine",
                    "StartAt": "Pass",
                    "States": {"Pass": {"Type": "Pass", "End": True}},
                }
            ),
            roleArn="arn:aws:iam::123456789012:role/role-name",
        )
        return state_machine

    return _make_step_function


@pytest.fixture
def make_test_archive_buckets(create_mock_bucket):
    """Creates the test archive buckets for the libera unit tests"""
    create_mock_bucket("libera-l0-data-test")
    create_mock_bucket("libera-spice-kernels-test")
    create_mock_bucket("libera-l1b-data-test")
    create_mock_bucket("libera-l2-data-test")


@pytest.fixture
def make_ingest_dropbox_bucket(create_mock_bucket):
    """Creates a mocked SDC Ingest Dropbox bucket whose name matches the real resource naming pattern.

    Returns the bucket name so tests can assert against it.
    """
    bucket_name = "sdc-dataingesteringestdropbox-test"
    create_mock_bucket(bucket_name)
    return bucket_name


@pytest.fixture
def make_sdc_event_bus(mock_s3_context_with_profile):
    """Creates a mocked SDC EventBridge event bus whose name matches the real resource naming pattern.

    Uses the 'test-profile' session so it shares the same mock_aws context as the S3 fixtures. The event bus is
    created in whatever region the session resolves to, so the fixture stays region-agnostic and matches the region
    the code under test (which derives its region from the session) will query.
    Returns the event bus name so tests can discover and assert against it.
    """
    bus_name = "SDCOrchestrationLiberaSDCEventBusTest123"
    session = boto3.Session(profile_name="test-profile")
    events_client = session.client("events", region_name=session.region_name)
    events_client.create_event_bus(Name=bus_name)
    return bus_name


@pytest.fixture
def make_coordination_table(mock_s3_context_with_profile):
    """Creates a mocked SDC Coordination Table (DynamoDB) with the PK/SK key schema.

    Uses the 'test-profile' session so it shares the same mock_aws context as the other AWS fixtures. Returns a
    ``(table_name, seed)`` tuple where ``seed(job_id)`` writes a ``#JOBMETADATA`` item for the given job id so that
    verification polling can find it.
    """
    table_name = "SDCOrchestrationCoordinationTableTest123"
    session = boto3.Session(profile_name="test-profile")
    dynamodb_client = session.client("dynamodb", region_name=session.region_name)
    dynamodb_client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    table = session.resource("dynamodb", region_name=session.region_name).Table(table_name)

    def _seed(job_id) -> None:
        table.put_item(Item={"PK": str(job_id), "SK": "#JOBMETADATA", "created-time": "2026-06-24T00:00:00Z"})

    return table_name, _seed


@pytest.fixture
def make_event_capturing_session(mock_s3_context_with_profile):
    """Returns a factory that builds a boto3 session whose ``events`` client records ``put_events`` calls.

    This is useful for asserting on EventBridge events without relying on moto delivering them to a target. The
    returned factory hands back a ``(session, captured)`` tuple; after the code under test calls ``put_events``, the
    ``captured`` dict will contain the ``entries`` that were passed.
    """

    def _make_event_capturing_session(profile_name: str = "test-profile"):
        session = boto3.Session(profile_name=profile_name)
        captured: dict = {}
        real_client = session.client

        def capturing_client(service_name, *args, **kwargs):
            client = real_client(service_name, *args, **kwargs)
            if service_name == "events":
                original_put_events = client.put_events

                def put_events_spy(**put_kwargs):
                    captured["entries"] = put_kwargs["Entries"]
                    return original_put_events(**put_kwargs)

                client.put_events = put_events_spy
            return client

        session.client = capturing_client
        return session, captured

    return _make_event_capturing_session


@pytest.fixture
def make_data_availability_table(mock_s3_context_with_profile):
    """Creates a mocked SDC Data Availability DynamoDB table (PK=applicable_date, SK=DataProductId#Version).

    Uses the 'test-profile' session so it stays in the same mock_aws context and region as the other fixtures.
    Returns the table name so tests can discover, seed, and assert against it.
    """
    table_name = "SDCDataIngesterDataAvailabilityTableTest123"
    session = boto3.Session(profile_name="test-profile")
    client = session.client("dynamodb", region_name=session.region_name)
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return table_name


@pytest.fixture
def make_file_metadata_table(mock_s3_context_with_profile):
    """Creates a mocked SDC File Metadata DynamoDB table (PK=file basename, SK=applicable_date).

    Uses the 'test-profile' session so it stays in the same mock_aws context and region as the other fixtures.
    Returns the table name so tests can discover, seed, and assert against it.
    """
    table_name = "SDCDataIngesterFileMetadataTableTest789"
    session = boto3.Session(profile_name="test-profile")
    client = session.client("dynamodb", region_name=session.region_name)
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "PK", "KeyType": "HASH"},
            {"AttributeName": "SK", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "PK", "AttributeType": "S"},
            {"AttributeName": "SK", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return table_name
