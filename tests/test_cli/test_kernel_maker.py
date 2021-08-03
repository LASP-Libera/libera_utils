"""Tests for kernel_maker CLI module"""
# Standard
import logging
# Installed

# Local
from libera_sdp.cli import kernel_maker


def test_make_jpss_spk(caplog, test_data_dir, tmp_path):
    """Test creating a SPK from packets"""
    logger = logging.getLogger()
    logging.basicConfig()
    caplog.set_level(logging.DEBUG)
    logger.error("Logging inside a function")
    packet_data_path = test_data_dir / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    opts_and_args = [str(packet_data_path), '--outdir', str(tmp_path)]
    kernel_maker.make_jpss_spk(opts_and_args)
    assert (tmp_path / 'libera_jpss_20210409t000000_20210409t015959.bsp').exists()


def test_make_jpss_ck(test_data_dir, tmp_path):
    """Test creating a CK from packets"""
    packet_data_path = test_data_dir / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    opts_and_args = [str(packet_data_path), '--outdir', str(tmp_path)]
    kernel_maker.make_jpss_ck(opts_and_args)
    assert (tmp_path / 'libera_jpss_20210408t235959_20210409t015958.bc').exists()
