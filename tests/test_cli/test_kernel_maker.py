"""Tests for kernel_maker CLI module"""
# Standard
from importlib.metadata import version
from unittest import mock
# Installed
# Local
from libera_sdp.cli import kernel_maker


# FIXME: These are bad unit tests in that they are not "unit".
#  They both call out to NAIF to download real kernels.
#  Those calls should be mocked and replaced with local test data.

def test_make_jpss_spk(libera_sdp_test_data_dir, short_tmp_path):
    """Test creating a SPK from packets"""
    with mock.patch('libera_sdp.spiceutil.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = libera_sdp_test_data_dir / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
        opts_and_args = [str(packet_data_path), '--outdir', str(short_tmp_path)]
        kernel_maker.make_jpss_spk(opts_and_args)
        assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bsp').exists()


def test_make_jpss_ck(libera_sdp_test_data_dir, short_tmp_path):
    """Test creating a CK from packets"""
    with mock.patch('libera_sdp.spiceutil.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = libera_sdp_test_data_dir / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
        opts_and_args = [str(packet_data_path), '--outdir', str(short_tmp_path)]
        kernel_maker.make_jpss_ck(opts_and_args)
        assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bc').exists()


def test_test():
    print(version('libera_sdp'))