"""Tests for cli module"""
# Installed
import argparse
import pytest
# Local
from libera_utils import cli, kernel_maker
from libera_utils.aws import ecr_upload
from libera_utils.aws import processing_step_function_trigger as psfn


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
             overwrite=False)),
        (['--version'],
         argparse.Namespace(
             func=cli.print_version_info)
         ),
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
def test_parse_cli_args(cli_args, parsed):
    """
    Test that cli args are parsed properly
    """
    print(f"CLI ARGS \n{cli_args}\n")
    print(f"Parsed args: {parsed} \n")
    assert cli.parse_cli_args(cli_args) == parsed
