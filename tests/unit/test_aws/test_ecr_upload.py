"""File for testing ECR upload module"""

import argparse
import json
from pathlib import Path
from unittest import mock

import docker
import pytest
from moto import mock_aws

import libera_utils.aws.ecr_upload as ecr_upload
from libera_utils.aws.constants import ProcessingStepIdentifier
from libera_utils.aws.ecr_upload import (
    DockerConfigManager,
    build_docker_image,
    get_ecr_docker_client,
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


@mock_aws
@mock.patch("libera_utils.aws.ecr_upload.docker.DockerClient.login", return_value="Mock login succeeded!")
def test_ecr_login_success(mock_docker_client_login):
    """Test getting an auth token from a boto3 ECR client. We can't actually test using that token to log in
    to a registry because that registry may not actually exist and the Docker API does the logging in so we can't
    mock it.
    We just mock out the DockerClient.login method to always return success for this test.
    """
    result = get_ecr_docker_client(region_name="us-west-2")
    assert isinstance(result, docker.DockerClient)


def test_build_docker_image(test_data_path):
    """Test building a docker image programmatically.
    This actually builds the image locally so Docker must be running."""
    build_docker_image(context_dir=test_data_path / "docker_test", image_name="test-image", target="test-target")


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


@pytest.mark.parametrize("ecr_tags", [None, ["latest"], ["latest", "v1.0"]])
@mock_aws()
@mock.patch("libera_utils.aws.ecr_upload.get_ecr_docker_client", return_value=docker.from_env())
@mock.patch("docker.models.images.ImageCollection.get", return_value=docker.models.images.Image())
@mock.patch("docker.models.images.Image.tag", return_value="Successfully Tagged the Mock")
@mock.patch("docker.models.images.ImageCollection.push", return_value=["Successfully mock pushed"])
def test_push_image_to_ecr(
    mock_docker_push, mock_docker_tag_image, mock_docker_get_image, mock_get_ecr_docker_client, ecr_tags
):
    """Test the push_image_to_ecr function."""
    # Mock the docker push method to simulate a successful push

    # We don't actually push to ECR, but we can test that the function is called correctly
    push_image_to_ecr(
        "test-image", "latest", ProcessingStepIdentifier.l1b_rad, ecr_image_tags=ecr_tags, ignore_docker_config=True
    )

    assert mock_get_ecr_docker_client.call_count == 1

    expected_calls = 1 if ecr_tags is None else len(ecr_tags)
    assert mock_docker_push.call_count == expected_calls
    assert mock_docker_tag_image.call_count == expected_calls
    assert mock_docker_get_image.call_count == expected_calls
