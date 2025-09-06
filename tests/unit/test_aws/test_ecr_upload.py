"""File for testing ECR upload module"""

import argparse
import base64
import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import docker
import pytest
from moto import mock_aws

import libera_utils.aws.ecr_upload as ecr_upload
from libera_utils.aws.constants import ProcessingStepIdentifier
from libera_utils.aws.ecr_upload import (
    DockerConfigManager,
    _get_fresh_ecr_auth,
    _push_single_tag,
    build_docker_image,
    push_image_to_ecr,
)


def test_docker_config_manager():
    with DockerConfigManager() as dcfg:
        # When no override, this should just be None
        assert dcfg.dockercfg_path is None

    with DockerConfigManager(override_default_config=True) as dcfg:
        # When overriding, we make a minimal temp config file to pass to the DockerClient.login method
        assert dcfg.dockercfg_path is not None
        assert Path(dcfg.dockercfg_path).exists()
        config_file = Path(dcfg.dockercfg_path) / "config.json"
        assert config_file.is_file()
        with config_file.open("r") as f:
            s = f.read()
            d = json.loads(s)
            assert d == {"auths": {}, "HttpHeaders": {}}

    assert not config_file.exists()  # Ensure temp config gets cleaned up correctly


def test_build_docker_image(test_data_path):
    """Test building a docker image programmatically.
    This mocks the Docker client to avoid needing a real Docker daemon."""
    # Create a mock Docker client
    mock_docker_client = mock.MagicMock(spec=docker.DockerClient)

    # Mock the build logs
    mock_logs = [
        {"stream": "Step 1/3 : FROM python:3.9\n"},
        {"stream": "Step 2/3 : COPY . /app\n"},
        {"stream": 'Step 3/3 : CMD ["python", "app.py"]\n'},
        {"stream": "Successfully built abc123\n"},
    ]

    # Configure the mock client's images.build() method
    mock_docker_client.images.build.return_value = (mock.MagicMock(), mock_logs)

    # Mock docker.from_env to return our mock client
    with mock.patch("docker.from_env", return_value=mock_docker_client):
        build_docker_image(context_dir=test_data_path / "docker_test", image_name="test-image", target="test-target")

        # Verify build was called with expected arguments
        assert mock_docker_client.images.build.called
        call_args = mock_docker_client.images.build.call_args
        assert call_args[1]["target"] == "test-target"
        assert call_args[1]["tag"] == "test-image:latest"
        assert call_args[1]["platform"] == "linux/amd64"


@pytest.mark.parametrize(
    ("algorithm_name", "image_name", "image_tag", "ecr_tags", "ignore_docker_config"),
    [("l1b-cam", "test-image", "latest", None, True), ("l1b-rad", "test-image", "latest", ["latest", "v1.0"], False)],
)
@mock.patch("libera_utils.aws.ecr_upload.push_image_to_ecr")
def test_ecr_upload_cli_handler(
    mock_push_image_to_ecr, image_name, algorithm_name, image_tag, ecr_tags, ignore_docker_config
):
    """Test the ECR upload CLI handler for file upload."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=ecr_upload.ecr_upload_cli_handler,
        algorithm_name=algorithm_name,
        image_name=image_name,
        image_tag=image_tag,
        ecr_tags=ecr_tags,
        ignore_docker_config=ignore_docker_config,
    )

    ecr_upload.ecr_upload_cli_handler(args)

    expected_algorithm = ProcessingStepIdentifier(algorithm_name)
    mock_push_image_to_ecr.assert_called_once_with(
        image_name, image_tag, expected_algorithm, ecr_image_tags=ecr_tags, ignore_docker_config=ignore_docker_config
    )


class TestGetFreshEcrAuth:
    """Test the _get_fresh_ecr_auth function."""

    @mock_aws
    @mock.patch("boto3.client")
    def test_get_fresh_ecr_auth_success(self, mock_boto_client):
        """Test successful ECR authentication token retrieval."""
        # Mock the ECR client and response
        mock_ecr_client = MagicMock()
        mock_boto_client.return_value = mock_ecr_client

        # Mock the authorization token response
        username = "AWS"
        password = "test-password-123"
        token = base64.b64encode(f"{username}:{password}".encode()).decode()

        mock_ecr_client.get_authorization_token.return_value = {
            "authorizationData": [{"authorizationToken": token, "expiresAt": "2024-01-01T00:00:00Z"}]
        }

        result = _get_fresh_ecr_auth("us-west-2")

        assert result == {"username": username, "password": password}
        mock_boto_client.assert_called_once_with("ecr", region_name="us-west-2")
        mock_ecr_client.get_authorization_token.assert_called_once()

    @mock.patch("boto3.client")
    def test_get_fresh_ecr_auth_failure(self, mock_boto_client):
        """Test ECR authentication failure handling."""
        mock_ecr_client = MagicMock()
        mock_boto_client.return_value = mock_ecr_client
        mock_ecr_client.get_authorization_token.side_effect = Exception("ECR error")

        with pytest.raises(Exception, match="ECR error"):
            _get_fresh_ecr_auth("us-west-2")


class TestPushSingleTag:
    """Test the _push_single_tag function."""

    @mock.patch("libera_utils.aws.ecr_upload._get_fresh_ecr_auth")
    def test_push_single_tag_success(self, mock_get_auth):
        """Test successful single tag push."""
        # Mock authentication
        mock_get_auth.return_value = {"username": "AWS", "password": "test-password"}

        # Mock Docker client and image
        mock_docker_client = MagicMock()
        mock_local_image = MagicMock()

        # Mock successful push logs
        mock_docker_client.api.push.return_value = [{"status": "Pushing", "progress": "1/2"}, {"status": "Pushed"}]

        _push_single_tag(
            docker_client=mock_docker_client,
            local_image=mock_local_image,
            full_ecr_tag="123456789.dkr.ecr.us-west-2.amazonaws.com/test-repo:latest",
            region_name="us-west-2",
            max_retries=3,
        )

        # Verify tag and push were called
        mock_local_image.tag.assert_called_once_with("123456789.dkr.ecr.us-west-2.amazonaws.com/test-repo:latest")
        mock_docker_client.api.push.assert_called_once()
        mock_get_auth.assert_called_once_with("us-west-2")

    @mock.patch("libera_utils.aws.ecr_upload._get_fresh_ecr_auth")
    def test_push_single_tag_with_errors_in_logs(self, mock_get_auth):
        """Test push with error messages in logs."""
        mock_get_auth.return_value = {"username": "AWS", "password": "test-password"}

        mock_docker_client = MagicMock()
        mock_local_image = MagicMock()

        # Mock push logs with errors
        mock_docker_client.api.push.return_value = [{"status": "Pushing"}, {"error": "Authentication failed"}]

        with pytest.raises(ValueError, match="Push errors"):
            _push_single_tag(
                docker_client=mock_docker_client,
                local_image=mock_local_image,
                full_ecr_tag="123456789.dkr.ecr.us-west-2.amazonaws.com/test-repo:latest",
                region_name="us-west-2",
                max_retries=1,
            )

    @mock.patch("libera_utils.aws.ecr_upload._get_fresh_ecr_auth")
    def test_push_single_tag_with_retries(self, mock_get_auth):
        """Test push with retry logic."""
        mock_get_auth.return_value = {"username": "AWS", "password": "test-password"}

        mock_docker_client = MagicMock()
        mock_local_image = MagicMock()

        # First call fails, second succeeds
        mock_docker_client.api.push.side_effect = [docker.errors.APIError("Network error"), [{"status": "Pushed"}]]

        _push_single_tag(
            docker_client=mock_docker_client,
            local_image=mock_local_image,
            full_ecr_tag="123456789.dkr.ecr.us-west-2.amazonaws.com/test-repo:latest",
            region_name="us-west-2",
            max_retries=2,
        )

        # Should be called twice (one failure, one success)
        assert mock_docker_client.api.push.call_count == 2
        assert mock_get_auth.call_count == 2  # Fresh auth for each attempt


@pytest.mark.parametrize("ecr_tags", [None, ["latest"], ["latest", "v1.0"]])
@mock_aws
@mock.patch("libera_utils.aws.utils.get_aws_account_number", return_value="123456789012")
@mock.patch("docker.from_env")
@mock.patch("libera_utils.aws.ecr_upload._push_single_tag")
def test_push_image_to_ecr(mock_push_single_tag, mock_docker_from_env, mock_get_account, ecr_tags):
    """Test the push_image_to_ecr function."""
    # Mock Docker client and image
    mock_docker_client = MagicMock()
    mock_local_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_local_image
    mock_docker_from_env.return_value = mock_docker_client

    # Test the function
    push_image_to_ecr(
        "test-image", "latest", ProcessingStepIdentifier.l1b_rad, ecr_image_tags=ecr_tags, ignore_docker_config=True
    )

    # Verify Docker client setup
    mock_docker_from_env.assert_called_once()
    mock_docker_client.images.get.assert_called_once_with("test-image:latest")

    # Verify push calls
    expected_calls = 1 if ecr_tags is None else len(ecr_tags)
    assert mock_push_single_tag.call_count == expected_calls

    # Verify the tags being pushed
    if ecr_tags is None:
        expected_tags = ["latest"]
    else:
        expected_tags = ecr_tags

    for i, expected_tag in enumerate(expected_tags):
        call_args = mock_push_single_tag.call_args_list[i]
        assert call_args[1]["full_ecr_tag"].endswith(f":{expected_tag}")


@mock_aws
@mock.patch("libera_utils.aws.utils.get_aws_account_number", return_value="123456789012")
@mock.patch("docker.from_env")
def test_push_image_to_ecr_image_not_found(mock_docker_from_env, mock_get_account):
    """Test push_image_to_ecr when local image is not found."""
    mock_docker_client = MagicMock()
    mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("Image not found")
    mock_docker_from_env.return_value = mock_docker_client

    with pytest.raises(ValueError, match="Local image not found: test-image:latest"):
        push_image_to_ecr("test-image", "latest", ProcessingStepIdentifier.l1b_rad)


@mock_aws
def test_push_image_to_ecr_invalid_processing_step():
    """Test push_image_to_ecr with an invalid processing step that has no ECR name."""
    # Create a mock processing step that returns None for ecr_name
    with mock.patch.object(ProcessingStepIdentifier, "ecr_name", None):
        processing_step = ProcessingStepIdentifier.l1b_rad

        with pytest.raises(ValueError, match="Unable to determine ECR repository name"):
            push_image_to_ecr("test-image", "latest", processing_step)


@mock_aws
@mock.patch("libera_utils.aws.utils.get_aws_account_number", return_value="123456789012")
@mock.patch("docker.from_env")
@mock.patch("libera_utils.aws.ecr_upload._push_single_tag")
def test_push_image_to_ecr_partial_failure(mock_push_single_tag, mock_docker_from_env, mock_get_account):
    """Test push_image_to_ecr when one tag succeeds and another fails."""
    mock_docker_client = MagicMock()
    mock_local_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_local_image
    mock_docker_from_env.return_value = mock_docker_client

    # First call succeeds, second fails
    mock_push_single_tag.side_effect = [None, Exception("Push failed")]

    with pytest.raises(Exception, match="Push failed"):
        push_image_to_ecr("test-image", "latest", ProcessingStepIdentifier.l1b_rad, ecr_image_tags=["v1.0", "latest"])

    # Should have attempted both pushes
    assert mock_push_single_tag.call_count == 2
