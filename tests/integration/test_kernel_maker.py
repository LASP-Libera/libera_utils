"""Tests for kernel_maker CLI module"""
# Standard
import argparse
from unittest import mock
# Installed
import pytest
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils import kernel_maker
from libera_utils.io.manifest import Manifest, ManifestType


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
    bucket = create_mock_bucket()
    bucket = bucket.name
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
    bucket = create_mock_bucket()
    bucket = bucket.name
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
    bucket = create_mock_bucket()
    bucket = bucket.name
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


def test_make_jpss_kernels_from_manifest_no_time_range(test_data_path, short_tmp_path):
    # Test that the kernels are generated when no desired range
    # is given.
    m = Manifest(
        manifest_type=ManifestType.INPUT
    )
    data_files = [
        test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T04-00-00Z_V01.DAT1"
    ]
    m.add_files(*data_files)
    manifest_path = short_tmp_path / "libera_input_manifest_20220101t112233.json"
    m.write(short_tmp_path, "libera_input_manifest_20220101t112233.json")
    kernel_maker.make_jpss_kernels_from_manifest(manifest_path, short_tmp_path)

    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t055849.bsp').exists()
    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t055849.bc').exists()


def test_make_jpss_kernels_from_manifest_local_one_file(test_data_path, short_tmp_path):
    # Test that the kernels are generated when the desired range
    # falls within only one local file as in the example manifest file
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        configuration={
            "start_time": "2021-04-09:00:00:00",
            "end_time": "2021-04-09:01:00:00"
                       }
    )
    data_files = [
        test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T04-00-00Z_V01.DAT1"
    ]
    m.add_files(*data_files)
    manifest_path = short_tmp_path / "libera_input_manifest_20220101t112233.json"
    m.write(short_tmp_path, "libera_input_manifest_20220101t112233.json")
    kernel_maker.make_jpss_kernels_from_manifest(manifest_path, short_tmp_path)

    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bsp').exists()
    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849.bc').exists()


def test_make_jpss_kernels_from_manifest_local_two_files(test_data_path, short_tmp_path):
    # Test that the kernels are generated when the desired range
    # falls within two local files. This includes changing the time range
    # in the example manifest file and the expected output kernel names
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        configuration={
            "start_time": "2021-04-09:00:00:00",
            "end_time": "2021-04-09:02:00:00"
                       }
    )
    data_files = [
        test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T04-00-00Z_V01.DAT1"
    ]
    m.add_files(*data_files)
    manifest_path = short_tmp_path / "libera_input_manifest_20220101t112233.json"
    m.write(short_tmp_path, "libera_input_manifest_20220101t112233.json")

    kernel_maker.make_jpss_kernels_from_manifest(manifest_path, short_tmp_path)

    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t035849.bsp').exists()
    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t035849.bc').exists()


def test_make_jpss_kernels_from_manifest_local_three_files(test_data_path, short_tmp_path):
    # Test that the kernels are generated when the desired range
    # falls within three local files. This includes changing the time range
    # in the example manifest file and the expected output kernel names
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        configuration={
            "start_time": "2021-04-09:00:00:00",
            "end_time": "2021-04-09:04:00:00"
                       }
    )
    data_files = [
        test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.DAT1",
        test_data_path / "J01_G011_LZ_2021-04-09T04-00-00Z_V01.DAT1"
    ]
    m.add_files(*data_files)
    m.write(short_tmp_path, "libera_input_manifest_20220101t112233.json")
    manifest_path = short_tmp_path / "libera_input_manifest_20220101t112233.json"
    kernel_maker.make_jpss_kernels_from_manifest(manifest_path, short_tmp_path)

    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t055849.bsp').exists()
    assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t055849.bc').exists()


def test_make_jpss_kernels_from_manifest_aws_three_files(test_data_path, short_tmp_path,
                                                         create_mock_bucket, write_file_to_s3):
    # Test that the kernels are generated when the desired range
    # falls within three remote files. This includes changing the time range
    # in the example manifest file and the expected output kernel names and
    # creating a mock bucket holding the data and the manifest file
    bucket = create_mock_bucket()
    bucket_name = bucket.name
    key = 'some_path'
    m = Manifest(manifest_type=ManifestType.INPUT)
    m.configuration["start_time"] = "2021-04-09:00:00:00"
    m.configuration["end_time"] = "2021-04-09:04:00:00"

    packet_uri = f"s3://{bucket_name}/{key}/test_kernel/J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, packet_uri)
    m.add_files(packet_uri)

    packet_uri = f"s3://{bucket_name}/{key}/test_kernel/J01_G011_LZ_2021-04-09T02-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T02-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, packet_uri)
    m.add_files(packet_uri)

    packet_uri = f"s3://{bucket_name}/{key}/test_kernel/J01_G011_LZ_2021-04-09T04-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T04-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, packet_uri)
    m.add_files(packet_uri)

    manifest_location = f"s3://{bucket_name}/{key}/manifest/"
    manifest_path = m.write(manifest_location, "libera_input_manifest_20220101t112233.json")

    s3_output_directory = S3Path(f"s3://{bucket_name}/{key}/kernel_output/")

    kernel_maker.make_jpss_kernels_from_manifest(manifest_path, s3_output_directory)
    assert (s3_output_directory / 'libera_jpss_20210408t235850_20210409t055849.bsp').exists()
    assert (s3_output_directory / 'libera_jpss_20210408t235850_20210409t055849.bc').exists()


def test_make_jpss_kernels_from_manifest_aws_one_file(test_data_path, short_tmp_path,
                                                      create_mock_bucket, write_file_to_s3):
    # Test that the kernels are generated when the desired range
    # falls within one remote file. This includes creating a mock bucket
    # holding the data and the newly made manifest file also in the bucket
    bucket = create_mock_bucket()
    bucket_name = bucket.name
    key = 'some_path'
    m = Manifest(manifest_type=ManifestType.INPUT)
    m.configuration["start_time"] = "2021-04-09:00:00:00"
    m.configuration["end_time"] = "2021-04-09:01:00:00"

    packet_uri = f"s3://{bucket_name}/{key}/test_kernel/J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1"
    packet_data_path = test_data_path / 'J01_G011_LZ_2021-04-09T00-00-00Z_V01.DAT1'
    write_file_to_s3(packet_data_path, packet_uri)
    m.add_files(packet_uri)

    manifest_location = f"s3://{bucket_name}/{key}/manifest/"
    manifest_path = m.write(manifest_location, "libera_input_manifest_20220101t112233.json")

    s3_output_directory = S3Path(f"s3://{bucket_name}/{key}/kernel_output/")

    kernel_maker.make_jpss_kernels_from_manifest(manifest_path, s3_output_directory)
    assert (s3_output_directory / 'libera_jpss_20210408t235850_20210409t015849.bsp').exists()
    assert (s3_output_directory / 'libera_jpss_20210408t235850_20210409t015849.bc').exists()
