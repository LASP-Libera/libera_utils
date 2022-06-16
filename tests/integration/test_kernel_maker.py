"""Tests for kernel_maker CLI module"""
# Standard
import argparse
from unittest import mock
# Installed
import pytest
# Local
from libera_utils import kernel_maker


def test_make_jpss_spk(test_data_path, short_tmp_path):
    """Test creating a SPK from packets"""
    with mock.patch('libera_utils.spiceutil.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(packet_data_path)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_jpss_spk(mock_parsed_args)
        assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bsp').exists()


def test_make_jpss_ck(test_data_path, short_tmp_path):
    """Test creating a CK from packets"""
    with mock.patch('libera_utils.spiceutil.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(packet_data_path)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_jpss_ck(mock_parsed_args)
        assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bc').exists()


@pytest.mark.xfail
def test_make_azel_ck(test_data_path, short_tmp_path):
    """Test creating a CK from packets"""
    with mock.patch('libera_utils.spiceutil.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = test_data_path / 'add-a-test-data-file.pkts'
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(packet_data_path)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_azel_ck(mock_parsed_args)
        assert (short_tmp_path / 'libera_azel_20210408t235850_20210409t015849.bc').exists()
