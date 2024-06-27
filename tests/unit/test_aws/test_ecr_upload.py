"""File for testing ECR upload module"""
# Standard
import argparse
# Installed
from unittest import mock
from moto import mock_aws
# Local
from libera_utils.aws.ecr_upload import upload_image_to_ecr, login_to_ecr


def test_ecr_login_success(fake_process):
    """Test that running the subprocess command is successful"""
    ecr_login_command = ('aws ecr get-login-password --region us-west-2 | docker login --username '
                         'AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com')
    fake_process.register(ecr_login_command, stdout='Login Succeeded')
    result = login_to_ecr(account_id=123456789012, region_name='us-west-2')
    assert result.stdout == "Login Succeeded"


def test_ecr_login_failure(fake_process):
    """Test that running the subprocess command is successful"""
    ecr_login_command = ('aws ecr get-login-password --region us-west-2 | docker login --username AWS '
                         '--password-stdin 1111111111.dkr.ecr.us-west-2.amazonaws.com')
    fake_process.register(ecr_login_command, stderr="Login Failed")
    result = login_to_ecr(account_id=1111111111, region_name='us-west-2')
    assert result.stderr == "Login Failed"


@mock_aws
@mock.patch('docker.from_env.images.get.tag')
@mock.patch('docker.from_env.images.push')
@mock.patch('docker.from_env')
def test_docker_client_push(mock_tag_function, mock_push_function, mock_from_env_function, fake_process):
    """Test that the docker client pushes correctly to the ECR"""
    args = argparse.Namespace(
        image_name="fake_image",
        image_tag="fake_tag",
        algorithm_name="pds_ingest",
        verbose=False
    )
    ecr_login_command = ('aws ecr get-login-password --region us-west-2 | docker login --username AWS '
                         '--password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com')
    fake_process.register(ecr_login_command, stdout="Login Succeeded", returncode=0)
    resp = upload_image_to_ecr(args)
    assert resp is None
