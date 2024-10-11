"""File for testing ECR upload module"""
# Installed
import docker
from unittest import mock
from moto import mock_aws
# Local
from libera_utils.aws.ecr_upload import get_ecr_docker_client, build_docker_image


@mock_aws
@mock.patch("libera_utils.aws.ecr_upload.docker.DockerClient.login", return_value="Mock login succeeded!")
def test_ecr_login_success(mock_docker_client_login):
    """Test getting an auth token from a boto3 ECR client. We can't actually test using that token to log in
    to a registry because that registry may not actually exist and the Docker API does the logging in so we can't
    mock it.
    We just mock out the DockerClient.login method to always return success for this test.
    """
    result = get_ecr_docker_client(region_name='us-west-2')
    assert isinstance(result, docker.DockerClient)


def test_build_docker_image(test_data_path):
    """Test building a docker image programmatically.
    This actually builds the image locally so Docker must be running."""
    build_docker_image(
        context_dir=test_data_path / "docker_test",
        image_name="test-image",
        target="test-target"
    )



