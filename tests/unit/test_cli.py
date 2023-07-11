"""Tests for cli module"""
# Installed
import argparse
import pytest
# Local
from libera_utils import cli, kernel_maker
from libera_utils.io import packet_ingest


@pytest.mark.parametrize(
    ("cli_args", "parsed"),
    [
        (['packet-ingest', '--delete', 'fakedir.json'],
         argparse.Namespace(
             func=packet_ingest.ingest,
             manifest_filepath='fakedir.json',
             delete=True,
             verbose=False)),
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
    """
    Test that cli args are parsed properly
    """
    assert cli.parse_cli_args(cli_args) == parsed
