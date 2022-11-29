"""Tests for kernel_maker CLI module"""
# Standard
import argparse
from unittest import mock
# Installed
import pytest
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils import kernel_maker


def test_make_jpss_spk(test_data_path, short_tmp_path):
    """Test creating a SPK from packets"""
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
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


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
def test_make_jpss_spk_aws(test_data_path, short_tmp_path, create_mock_bucket, write_file_to_s3, wrapper):
    """Test creating a SPK from packets stored in AWS S3"""
    bucket = 'spk-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, kernel_uri)
    packet_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"

    mock_parsed_args = argparse.Namespace(
        packet_data_filepaths=[str(packet_s3_path)],
        outdir=str(s3_output_directory),
        overwrite=False,
        verbose=False
    )
    kernel_maker.make_jpss_spk(mock_parsed_args)

    s3_output_path = S3Path(s3_output_directory)
    assert (s3_output_path / 'libera_jpss_20210408t235850_20210409t015849.bsp').exists()


def test_make_jpss_ck(test_data_path, short_tmp_path):
    """Test creating a CK from packets"""
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
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

@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
def test_make_jpss_ck_aws(test_data_path, short_tmp_path, create_mock_bucket, write_file_to_s3, wrapper):
    """Test creating a CK from packets"""
    bucket = 'ck-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, kernel_uri)
    kernel_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"

    mock_parsed_args = argparse.Namespace(
        packet_data_filepaths=[str(kernel_s3_path)],
        outdir=str(s3_output_directory),
        overwrite=False,
        verbose=False
    )
    kernel_maker.make_jpss_ck(mock_parsed_args)

    s3_output_path = S3Path(s3_output_directory)
    assert (s3_output_path / 'libera_jpss_20210408t235850_20210409t015849.bc').exists()


@pytest.mark.xfail
def test_make_azel_ck(test_data_path, short_tmp_path):
    """Test creating a CK from packets"""
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
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


@pytest.mark.xfail
@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
def test_make_azel_ck_aws(test_data_path, short_tmp_path, create_mock_bucket, write_file_to_s3, wrapper):
    """Test creating a CK from packets"""
    bucket = 'azel-ck-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, kernel_uri)
    kernel_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"

    mock_parsed_args = argparse.Namespace(
        packet_data_filepaths=[str(kernel_s3_path)],
        outdir=str(s3_output_directory),
        overwrite=False,
        verbose=False
    )
    kernel_maker.make_azel_ck(mock_parsed_args)

    s3_output_path = S3Path(s3_output_directory)
    assert (s3_output_path / 'libera_azel_20210408t235850_20210409t015849.bc').exists()
