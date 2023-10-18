"""Tests for kernel_maker CLI module"""
# Standard
import argparse
from datetime import datetime
from unittest import mock
import time
# Installed
import pytest
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils import kernel_maker
from libera_utils.io.manifest import Manifest, ManifestType


@pytest.fixture
def add_start_end_time_to_manifest():
    def _add_start_end_time_to_manifest(manifest_path, offset_hours: int):
        # Add the configuration constraint that the time of interest is just 1 hour long and will fall in only one file
        # To do this must update and re-save the manifest file
        m = Manifest.from_file(manifest_path)
        m.add_desired_time_range(datetime(2021, 4, 9, 0, 0, 0), datetime(2021, 4, 9, offset_hours, 0, 0))
        # Manifest names are unique as long as they are made at least 1 second apart so wait at least that 1 second to
        # make sure the system generates a new filename for the updated input manifest
        m.filename = None
        time.sleep(1)
        updated_manifest_path = m.write(outpath=manifest_path.parent)
        return updated_manifest_path

    return _add_start_end_time_to_manifest

@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_spk(mocked_get_current_version_str, test_pds_file_1, short_tmp_path):
    """Test creating a SPK from packets"""
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(test_pds_file_1)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_jpss_spk(mock_parsed_args)
        assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bsp').exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_spk_aws(mocked_get_current_version_str, test_pds_file_1, short_tmp_path, create_mock_bucket,
                           write_file_to_s3, wrapper):
    """Test creating a SPK from packets stored in AWS S3"""
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)
    bucket = create_mock_bucket()
    bucket = bucket.name
    key = 'some_path'
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_pds_file_1.name}"
    write_file_to_s3(test_pds_file_1, kernel_uri)
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
    assert (s3_output_path / 'libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bsp').exists()


@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_ck(mocked_get_current_version_str, test_pds_file_1, short_tmp_path):
    """Test creating a CK from packets"""
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(test_pds_file_1)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_jpss_ck(mock_parsed_args)
        assert (short_tmp_path / 'libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_ck_aws(mocked_get_current_version_str, test_pds_file_1, short_tmp_path, create_mock_bucket,
                          write_file_to_s3, wrapper):
    """Test creating a CK from packets"""
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)

    bucket = create_mock_bucket()
    bucket = bucket.name
    key = 'some_path'
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_pds_file_1.name}"
    write_file_to_s3(test_pds_file_1, kernel_uri)
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
    assert (s3_output_path / 'libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.xfail
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_azel_ck(mocked_get_current_version_str, test_data_path, short_tmp_path):
    """Test creating a CK from packets"""
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
        assert (short_tmp_path / 'libera_azel_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_azel_ck_aws(mocked_get_current_version_str, test_data_path, short_tmp_path, create_mock_bucket,
                          write_file_to_s3, wrapper):
    """Test creating a CK from packets"""
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)

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
    assert (s3_output_path / 'libera_azel_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_kernels_from_manifest_no_time_range(mocked_get_current_version_str,
                                                       setup_kernel_maker_environment_with_manifest):
    # Test that the kernels are generated when no desired range
    # is given.
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest

    kernel_maker.make_jpss_kernels_from_manifest(input_manifest_path, output_path)

    assert (output_path / 'libera_jpss_20210408t235850_20210409t055849_vM3m14p159_r25056154513.bsp').exists()
    assert (output_path / 'libera_jpss_20210408t235850_20210409t055849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_kernels_from_manifest_one_file(mocked_get_current_version_str,
                                                  setup_kernel_maker_environment_with_manifest,
                                                  add_start_end_time_to_manifest):
    # Test that the kernels are generated when the desired range
    # falls within only one local file as in the example manifest file
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest
    updated_manifest_path = add_start_end_time_to_manifest(input_manifest_path, offset_hours=1)

    kernel_maker.make_jpss_kernels_from_manifest(updated_manifest_path, output_path)

    assert (output_path / 'libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bsp').exists()
    assert (output_path / 'libera_jpss_20210408t235850_20210409t015849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_kernels_from_manifest_two_files(mocked_get_current_version_str,
                                                   setup_kernel_maker_environment_with_manifest,
                                                   add_start_end_time_to_manifest):
    # Test that the kernels are generated when the desired range
    # falls within two local files. This includes changing the time range
    # in the example manifest file and the expected output kernel names
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest
    updated_manifest_path = add_start_end_time_to_manifest(input_manifest_path, offset_hours=2)

    kernel_maker.make_jpss_kernels_from_manifest(updated_manifest_path, output_path)

    assert (output_path / 'libera_jpss_20210408t235850_20210409t035849_vM3m14p159_r25056154513.bsp').exists()
    assert (output_path / 'libera_jpss_20210408t235850_20210409t035849_vM3m14p159_r25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="vM3m14p159")
def test_make_jpss_kernels_from_manifest_three_files(mocked_get_current_version_str,
                                                   setup_kernel_maker_environment_with_manifest,
                                                   add_start_end_time_to_manifest):
    # Test that the kernels are generated when the desired range
    # falls within three local files. This includes changing the time range
    # in the example manifest file and the expected output kernel names
    kernel_maker.datetime.utcnow.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest
    updated_manifest_path = add_start_end_time_to_manifest(input_manifest_path, offset_hours=4)

    kernel_maker.make_jpss_kernels_from_manifest(updated_manifest_path, output_path)

    assert (output_path / 'libera_jpss_20210408t235850_20210409t055849_vM3m14p159_r25056154513.bsp').exists()
    assert (output_path / 'libera_jpss_20210408t235850_20210409t055849_vM3m14p159_r25056154513.bc').exists()
