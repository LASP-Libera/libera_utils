"""Tests for kernel_maker CLI module"""
import argparse
import time
from datetime import datetime
from unittest import mock

import pytest
from cloudpathlib import AnyPath, S3Path

from libera_utils import kernel_maker
from libera_utils.io.manifest import Manifest

# Mark test module as integration tests
pytestmark = pytest.mark.integration


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
        updated_manifest_path = m.write(out_path=manifest_path.parent)
        return updated_manifest_path

    return _add_start_end_time_to_manifest


@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_spk(mocked_get_current_version_str, test_pds_file_1, short_tmp_path, curryer_lsk):
    """Test creating a SPK from packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(test_pds_file_1)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_jpss_spk(mock_parsed_args)
        assert (short_tmp_path / 'LIBERA_JPSS_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp').exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_spk_aws(mocked_get_current_version_str, test_pds_file_1, short_tmp_path, create_mock_bucket,
                           write_file_to_s3, wrapper, curryer_lsk):
    """Test creating a SPK from packets stored in AWS S3"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
    assert (s3_output_path / 'LIBERA_JPSS_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp').exists()


@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_ck(mocked_get_current_version_str, test_pds_file_1, short_tmp_path, curryer_lsk):
    """Test creating a CK from packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(test_pds_file_1)],
            outdir=str(short_tmp_path),
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_jpss_ck(mock_parsed_args)
        assert (short_tmp_path / 'LIBERA_JPSS_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc').exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_ck_aws(mocked_get_current_version_str, test_pds_file_1, short_tmp_path, create_mock_bucket,
                          write_file_to_s3, wrapper, curryer_lsk):
    """Test creating a CK from packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

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
    assert (s3_output_path / 'LIBERA_JPSS_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc').exists()


@pytest.mark.xfail
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_azel_ck(mocked_get_current_version_str, test_data_path, short_tmp_path):
    """Test creating a CK from packets"""
    """
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
        assert (short_tmp_path / 'LIBERA_AZEL_V3-14-159_20210408T235850_20210409T015849_R25056154513.bc').exists()
        """

    """Test creating a n Elevation CK from csv"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = test_data_path / 'spice/Elevation_angle_20210409_10m.csv'
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(packet_data_path)],
            outdir=str(short_tmp_path),
            csv=True,
            elevation=True,
            azimuth=False,
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_azel_ck(mock_parsed_args)
        assert (short_tmp_path / 'LIBERA_ELSCAN_V3-14-159_20210409T000000_20210409T000959_R25056154513.bc').exists()

    """Test creating a n Azimuth CK from csv"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=short_tmp_path):
        packet_data_path = test_data_path / 'spice/Azimuth_angle_20210409_10m.csv'
        mock_parsed_args = argparse.Namespace(
            packet_data_filepaths=[str(packet_data_path)],
            outdir=str(short_tmp_path),
            csv=True,
            elevation=False,
            azimuth=True,
            overwrite=False,
            verbose=False
        )
        kernel_maker.make_azel_ck(mock_parsed_args)
        assert (short_tmp_path / 'LIBERA_AZROT_V3-14-159_20210409T000000_20210409T000958_R25056154513.bc').exists()


@pytest.mark.xfail
@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_azel_ck_aws(mocked_get_current_version_str, test_data_path, short_tmp_path, create_mock_bucket,
                          write_file_to_s3, wrapper, test_pds_file_1):
    """Test creating a CK from packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    bucket = create_mock_bucket()
    bucket = bucket.name
    key = 'some_path'
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_pds_file_1.name}"
    packet_data_path = test_data_path / test_pds_file_1.name
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
    assert (s3_output_path / 'LIBERA_AZEL_V3-14-159_20210408T235850_20210409T015849_R25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_kernels_from_manifest_no_time_range(mocked_get_current_version_str,
                                                       setup_kernel_maker_environment_with_manifest, curryer_lsk):
    # Test that the kernels are generated when no desired range
    # is given.
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest

    kernel_maker.make_jpss_kernels_from_manifest(input_manifest_path, output_path)

    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210409T000000_20210409T055959_R25056154513.bsp').exists()
    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210408T235959_20210409T055958_R25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_kernels_from_manifest_one_file(mocked_get_current_version_str,
                                                  setup_kernel_maker_environment_with_manifest,
                                                  add_start_end_time_to_manifest, curryer_lsk):
    # Test that the kernels are generated when the desired range
    # falls within only one local file as in the example manifest file
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest
    updated_manifest_path = add_start_end_time_to_manifest(input_manifest_path, offset_hours=1)

    kernel_maker.make_jpss_kernels_from_manifest(updated_manifest_path, output_path)

    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp').exists()
    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_kernels_from_manifest_two_files(mocked_get_current_version_str,
                                                   setup_kernel_maker_environment_with_manifest,
                                                   add_start_end_time_to_manifest, curryer_lsk):
    # Test that the kernels are generated when the desired range
    # falls within two local files. This includes changing the time range
    # in the example manifest file and the expected output kernel names
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest
    updated_manifest_path = add_start_end_time_to_manifest(input_manifest_path, offset_hours=2)

    kernel_maker.make_jpss_kernels_from_manifest(updated_manifest_path, output_path)

    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp').exists()
    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc').exists()


@pytest.mark.parametrize(
    "test_type", ["S3", "Local"], indirect=True
)
@mock.patch.object(kernel_maker, 'datetime', mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v3-14-159")
def test_make_jpss_kernels_from_manifest_three_files(mocked_get_current_version_str,
                                                     setup_kernel_maker_environment_with_manifest,
                                                     add_start_end_time_to_manifest, curryer_lsk):
    # Test that the kernels are generated when the desired range
    # falls within three local files. This includes changing the time range
    # in the example manifest file and the expected output kernel names
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_kernel_maker_environment_with_manifest
    updated_manifest_path = add_start_end_time_to_manifest(input_manifest_path, offset_hours=4)

    kernel_maker.make_jpss_kernels_from_manifest(updated_manifest_path, output_path)

    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210409T000000_20210409T035959_R25056154513.bsp').exists()
    assert (output_path / 'LIBERA_JPSS_V3-14-159_20210408T235959_20210409T035958_R25056154513.bc').exists()
