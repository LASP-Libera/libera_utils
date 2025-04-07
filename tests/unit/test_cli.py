"""Tests for cli module"""
import argparse

import pytest

from libera_utils import cli, kernel_maker
from libera_utils.aws import ecr_upload, s3_utilities
from libera_utils.aws import processing_step_function_trigger as psfn


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (['--version'],
         argparse.Namespace(
             func=cli.print_version_info)
         )
    ]
)
def test_parse_base_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (['make-kernel', 'jpss-spk', '-ofakedir', 'file1.pkts', 'file2.pkts'],
         argparse.Namespace(
             func=kernel_maker.make_jpss_spk,
             packet_data_filepaths=['file1.pkts', 'file2.pkts'],
             outdir='fakedir',
             verbose=False,
             overwrite=False)),
        (['make-kernel', 'jpss-ck', '--outdir', 'fakedir', 'file1.pkts', 'file2.pkts'],
         argparse.Namespace(
             func=kernel_maker.make_jpss_ck,
             packet_data_filepaths=['file1.pkts', 'file2.pkts'],
             outdir='fakedir',
             verbose=False,
             overwrite=False))
    ]
)
def test_parse_make_kernel_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (['ecr-upload', 'fake_image', 'fake_tag', 'fake_algo', '--ignore-docker-config'],
         argparse.Namespace(
             func=ecr_upload.ecr_upload_cli_func,
             image_name='fake_image',
             image_tag='fake_tag',
             algorithm_name='fake_algo',
             ecr_image_tags=None,
             ignore_docker_config=True)
         ),
        (['step-function-trigger', 'fake_algorithm', '2000-01-01', '--wait_for_finish', '--verbose'],
         argparse.Namespace(
             func=psfn.step_function_trigger,
             algorithm_name='fake_algorithm',
             applicable_day='2000-01-01',
             wait_for_finish=True,
             verbose=True)
         )
    ]
)
def test_parse_aws_tools_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (['s3-utils', 'put', 'l1b-cam', 'some/file/path.nc'],
         argparse.Namespace(
             func=s3_utilities.s3_put_cli_handler,
             file_path='some/file/path.nc',
             algorithm_name='l1b-cam',
             account_suffix='-stage')),
        (['s3-utils', 'put', 'l1b-cam', 'some/file/path.nc', '--account_suffix=-test'],
         argparse.Namespace(
             func=s3_utilities.s3_put_cli_handler,
             file_path='some/file/path.nc',
             algorithm_name='l1b-cam',
             account_suffix='-test')),
        (['s3-utils', 'ls', 'l1b-cam'],
         argparse.Namespace(
             func=s3_utilities.s3_list_cli_handler,
             algorithm_name='l1b-cam',
             account_suffix='-stage')),
        (['s3-utils', 'ls', 'l1b-cam', '--account_suffix=-test'],
         argparse.Namespace(
             func=s3_utilities.s3_list_cli_handler,
             algorithm_name='l1b-cam',
             account_suffix='-test')),
        (['s3-utils', 'cp', 's3://somebucket/with/file/path.nc', '.'],
         argparse.Namespace(
             func=s3_utilities.s3_copy_cli_handler,
             source_path='s3://somebucket/with/file/path.nc',
             dest_path='.',
             delete=False)),
        (['s3-utils', 'cp', 's3://somebucket/with/file/path.nc', "some/local/path.nc", '--delete'],
         argparse.Namespace(
             func=s3_utilities.s3_copy_cli_handler,
             source_path='s3://somebucket/with/file/path.nc',
             dest_path="some/local/path.nc",
             delete=True)),
    ]
)
def test_s3_utils_parse_cli_args(cli_args, parsed):
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed
