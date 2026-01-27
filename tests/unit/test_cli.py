"""Tests for cli module"""

import argparse

import pytest

from libera_utils import cli, kernel_maker
from libera_utils.aws import ecr_upload, s3_utilities
from libera_utils.aws import processing_step_function_trigger as psfn


@pytest.mark.parametrize(("cli_args", "parsed"), [(["--version"], argparse.Namespace(func=cli.print_version_info))])
def test_parse_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["make-kernel", "jpss", "file.manifest"],
            argparse.Namespace(
                func=kernel_maker.jpss_kernel_cli_handler,
                input_manifest="file.manifest",
                verbose=False,
            ),
        ),
        (
            ["make-kernel", "azel", "file.manifest", "--verbose"],
            argparse.Namespace(
                func=kernel_maker.azel_kernel_cli_handler,
                input_manifest="file.manifest",
                verbose=True,
            ),
        ),
    ],
)
def test_make_kernel_parse_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["step-function-trigger", "l1b-rad", "2030-01-01"],
            argparse.Namespace(
                func=psfn.step_function_trigger_cli_handler,
                algorithm_name="l1b-rad",
                applicable_day="2030-01-01",
                wait_time=5,
                profile=None,
            ),
        ),
        (
            [
                "step-function-trigger",
                "l1b-cam",
                "2030-01-01",
                "--wait-time=5",
                "--profile=test-profile",
            ],
            argparse.Namespace(
                func=psfn.step_function_trigger_cli_handler,
                algorithm_name="l1b-cam",
                applicable_day="2030-01-01",
                wait_time=5,
                profile="test-profile",
            ),
        ),
    ],
)
def test_step_function_trigger_parse_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["ecr-upload", "l1b-rad", "test-image"],
            argparse.Namespace(
                func=ecr_upload.ecr_upload_cli_handler,
                algorithm_name="l1b-rad",
                image_name="test-image",
                image_tag="latest",
                ecr_tags=None,
                ignore_docker_config=False,
                profile=None,
            ),
        ),
        (
            ["ecr-upload", "l1b-rad", "test-image", "--ignore-docker-config", "--image-tag", "tag1.2"],
            argparse.Namespace(
                func=ecr_upload.ecr_upload_cli_handler,
                algorithm_name="l1b-rad",
                image_name="test-image",
                image_tag="tag1.2",
                ecr_tags=None,
                ignore_docker_config=True,
                profile=None,
            ),
        ),
        (
            ["ecr-upload", "l1b-rad", "test-image", "--image-tag=latest"],
            argparse.Namespace(
                func=ecr_upload.ecr_upload_cli_handler,
                image_name="test-image",
                algorithm_name="l1b-rad",
                image_tag="latest",
                ecr_tags=None,
                ignore_docker_config=False,
                profile=None,
            ),
        ),
        (
            ["ecr-upload", "l1b-rad", "test-image", "--ecr-tags", "latest", "tag2", "--profile", "test-profile"],
            argparse.Namespace(
                func=ecr_upload.ecr_upload_cli_handler,
                algorithm_name="l1b-rad",
                image_name="test-image",
                image_tag="latest",
                ecr_tags=["latest", "tag2"],
                ignore_docker_config=False,
                profile="test-profile",
            ),
        ),
    ],
)
def test_ecr_upload_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (
            ["s3-utils", "put", "some/file/path.nc"],
            argparse.Namespace(
                func=s3_utilities.s3_put_cli_handler,
                file_path="some/file/path.nc",
                profile=None,
            ),
        ),
        (
            ["s3-utils", "--profile=test", "put", "some/file/path.nc"],
            argparse.Namespace(
                func=s3_utilities.s3_put_cli_handler,
                file_path="some/file/path.nc",
                profile="test",
            ),
        ),
        (
            ["s3-utils", "ls", "CAM"],
            argparse.Namespace(func=s3_utilities.s3_list_cli_handler, product_name="CAM", profile=None),
        ),
        (
            ["s3-utils", "--profile=test", "ls", "CAM"],
            argparse.Namespace(func=s3_utilities.s3_list_cli_handler, product_name="CAM", profile="test"),
        ),
        (
            ["s3-utils", "cp", "s3://somebucket/with/file/path.nc", "."],
            argparse.Namespace(
                func=s3_utilities.s3_copy_cli_handler,
                source_path="s3://somebucket/with/file/path.nc",
                dest_path=".",
                delete=False,
                profile=None,
            ),
        ),
        (
            [
                "s3-utils",
                "--profile=test-profile",
                "cp",
                "s3://somebucket/with/file/path.nc",
                "some/local/path.nc",
                "--delete",
            ],
            argparse.Namespace(
                func=s3_utilities.s3_copy_cli_handler,
                source_path="s3://somebucket/with/file/path.nc",
                dest_path="some/local/path.nc",
                delete=True,
                profile="test-profile",
            ),
        ),
    ],
)
def test_s3_utils_parse_cli_args(cli_args, parsed):
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    "cli_args",
    [
        ["s3-utils", "ls", "NOT-A-PRODUCT"],
        ["ecr-upload", "not-an-alg", "test-image"],
        ["step-function-trigger", "not-an-alg", "2030-01-01"],
    ],
)
def test_wrong_libera_ids(cli_args):
    with pytest.raises(SystemExit):
        cli.parse_cli_args(cli_args)
