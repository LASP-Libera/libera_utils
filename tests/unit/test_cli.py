"""Tests for cli module"""
# Installed
import argparse
import pytest
# Local
from libera_utils import cli, kernel_maker


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
    ]
)
def test_parse_cli_args(cli_args, parsed):
    assert dict(vars(cli.parse_cli_args(cli_args))) == dict(vars(parsed))
