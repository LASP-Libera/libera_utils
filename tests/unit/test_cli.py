"""Tests for cli module"""

import argparse

import pytest

from libera_utils import cli, kernel_maker
from libera_utils.aws import ecr_upload, s3_utilities
from libera_utils.aws import manual_processing as mp


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
                func=mp.step_function_trigger_cli_handler,
                algorithm_name="l1b-rad",
                applicable_day="2030-01-01",
                verify=False,
                wait_time=60,
                profile=None,
            ),
        ),
        (
            [
                "step-function-trigger",
                "l1b-cam",
                "2030-01-01",
                "--verify",
                "--wait-time=5",
                "--profile=test-profile",
            ],
            argparse.Namespace(
                func=mp.step_function_trigger_cli_handler,
                algorithm_name="l1b-cam",
                applicable_day="2030-01-01",
                verify=True,
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
            ["manual-processing", "2026-06-01"],
            argparse.Namespace(
                func=mp.manual_processing_cli_handler,
                applicable_dates=["2026-06-01"],
                dag_config=None,
                start_steps=None,
                process_downstream=True,
                verify=False,
                wait_time=60,
                profile=None,
            ),
        ),
        (
            [
                "manual-processing",
                "2026-06-01",
                "2026-06-02",
                "--dag-config=my_dag.json",
                "--start-steps",
                "l1b-rad",
                "l1b-cam",
                "--no-process-downstream",
                "--verify",
                "--wait-time=10",
                "--profile=test-profile",
            ],
            argparse.Namespace(
                func=mp.manual_processing_cli_handler,
                applicable_dates=["2026-06-01", "2026-06-02"],
                dag_config="my_dag.json",
                start_steps=["l1b-rad", "l1b-cam"],
                process_downstream=False,
                verify=True,
                wait_time=10,
                profile="test-profile",
            ),
        ),
    ],
)
def test_manual_processing_parse_cli_args(cli_args, parsed):
    """
    Test that manual-processing cli args are parsed properly
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
                file_paths=["some/file/path.nc"],
                profile=None,
                verify=False,
                timeout=s3_utilities.DEFAULT_VERIFY_TIMEOUT_SECONDS,
            ),
        ),
        (
            ["s3-utils", "put", "some/file/path.nc", "another/file/path.nc"],
            argparse.Namespace(
                func=s3_utilities.s3_put_cli_handler,
                file_paths=["some/file/path.nc", "another/file/path.nc"],
                profile=None,
                verify=False,
                timeout=s3_utilities.DEFAULT_VERIFY_TIMEOUT_SECONDS,
            ),
        ),
        (
            ["s3-utils", "put", "some/file/path.nc", "--verify", "--timeout", "60"],
            argparse.Namespace(
                func=s3_utilities.s3_put_cli_handler,
                file_paths=["some/file/path.nc"],
                profile=None,
                verify=True,
                timeout=60.0,
            ),
        ),
        (
            ["s3-utils", "--profile=test", "put", "some/file/path.nc"],
            argparse.Namespace(
                func=s3_utilities.s3_put_cli_handler,
                file_paths=["some/file/path.nc"],
                profile="test",
                verify=False,
                timeout=s3_utilities.DEFAULT_VERIFY_TIMEOUT_SECONDS,
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
        ["manual-processing", "2026-06-01", "--start-steps", "not-a-step"],
    ],
)
def test_wrong_libera_ids(cli_args):
    with pytest.raises(SystemExit):
        cli.parse_cli_args(cli_args)
