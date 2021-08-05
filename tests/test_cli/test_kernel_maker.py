"""Tests for kernel_maker CLI module"""
# Standard
import logging
# Installed

# Local
import os

from libera_sdp.cli import kernel_maker


def test_make_jpss_spk(libera_sdp_test_data_dir, tmp_path):
    """Test creating a SPK from packets"""
    packet_data_path = libera_sdp_test_data_dir / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    opts_and_args = [str(packet_data_path), '--outdir', str(tmp_path)]
    kernel_maker.make_jpss_spk(opts_and_args)
    assert (tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bsp').exists()


def test_make_jpss_ck(libera_sdp_test_data_dir, tmp_path):
    """Test creating a CK from packets"""
    packet_data_path = libera_sdp_test_data_dir / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    opts_and_args = [str(packet_data_path), '--outdir', str(tmp_path)]
    kernel_maker.make_jpss_ck(opts_and_args)
    assert (tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bc').exists()
