"""File for testing ECR upload module"""
# Standard
import argparse
# Installed
import pytest
from unittest.mock import patch
from pathlib import Path
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.io import filenaming
from libera_utils.aws import s3_utilities
from libera_utils.aws.constants import ProcessingStepIdentifier, LiberaAccountSuffix


@pytest.mark.parametrize(
    ("file_path", "algorithm_name", "account_suffix"),
    [
        ('some/file/path.nc', "l1b-cam", "-stage"),
        ('some/file/path.nc', "l1b-cam", "-test"),
    ]
)
@patch('libera_utils.aws.s3_utilities.s3_put_in_archive_for_processing_step')
def test_s3_utils_put_cli_handler(mock_s3_put_for_processing_step, file_path, algorithm_name, account_suffix):
    """Test the S3 utilities CLI handler for file upload."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=s3_utilities.s3_put_cli_handler,
        file_path=file_path,
        algorithm_name=algorithm_name,
        account_suffix=account_suffix)

    # Call the CLI handler
    s3_utilities.s3_put_cli_handler(args)

    # Check the mocked function was called correctly
    expected_file_path = AnyPath(file_path)
    expected_processing_step = ProcessingStepIdentifier(algorithm_name)
    expected_suffix = account_suffix
    mock_s3_put_for_processing_step.assert_called_once_with(expected_file_path, expected_processing_step,
                                                            account_suffix=expected_suffix)


@pytest.mark.parametrize(
    ("processing_step", "file_name"),
    [
        (ProcessingStepIdentifier.l0_rad_pds,
         "P1590006SOMESCIENCEAAA99030231459001.PDS"),
        (ProcessingStepIdentifier.spice_jpss,
         "LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp"),
        (ProcessingStepIdentifier.spice_azel,
         "LIBERA_AZROT_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc"),
        (ProcessingStepIdentifier.l1b_cam,
         "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        (ProcessingStepIdentifier.l1b_rad,
         "LIBERA_L1B_RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        (ProcessingStepIdentifier.l2_cam_cf,
         "LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        (ProcessingStepIdentifier.l2_rad_ssw_toa,
         "LIBERA_L2_SSW-TOA_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
    ]
)
def test_s3_put_for_processing_step(tmp_path, make_test_archive_buckets,
                                    processing_step, file_name):
    """Test uploading an object to an S3 bucket for a given processing step."""

    # Create an empty file that is correctly named file in a temp directory
    Path.touch(tmp_path / file_name)

    # Make the expected file path in S3
    filename_object = filenaming.AbstractValidFilename.from_file_path(tmp_path / file_name)
    bucket_name = processing_step.get_archive_bucket_name(account_suffix=LiberaAccountSuffix.TEST)
    s3_path = f"s3://{bucket_name}"
    expected_s3_path = filename_object.generate_prefixed_path(s3_path)

    # Run the upload function
    s3_utilities.s3_put_in_archive_for_processing_step(tmp_path / file_name, processing_step,
                                                       account_suffix=LiberaAccountSuffix.TEST)
    # Check that the expected file is in the S3 bucket
    assert S3Path(expected_s3_path).exists()


@pytest.mark.parametrize(
    ("algorithm_name", "account_suffix"),
    [
        ("l1b-cam", "-stage"),
        ("l1b-cam", "-test"),
    ]
)
@patch('libera_utils.aws.s3_utilities.s3_list_archive_files')
def test_s3_utils_list_cli_handler(mock_s3_list_files, algorithm_name, account_suffix):
    """Test the S3 utilities CLI handler for listing files in a bucket."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=s3_utilities.s3_list_cli_handler,
        algorithm_name=algorithm_name,
        account_suffix=account_suffix)

    # Call the CLI handler
    s3_utilities.s3_list_cli_handler(args)

    # Check the mocked function was called correctly
    expected_processing_step = ProcessingStepIdentifier(algorithm_name)
    expected_suffix = account_suffix
    mock_s3_list_files.assert_called_once_with(expected_processing_step,
                                               account_suffix=expected_suffix)


@pytest.mark.parametrize(
    ("processing_step", "file_names"),
    [
        (ProcessingStepIdentifier.l0_rad_pds,
         ["P1590006SOMESCIENCEAAA99030231459001.PDS",
          "P1590007SOMESCIENCEAAA99030231459001.PDS"]),
        (ProcessingStepIdentifier.spice_jpss,
         ["LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
          "LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112234.bsp"]),
        (ProcessingStepIdentifier.spice_azel,
         ["LIBERA_AZROT_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc",
          "LIBERA_AZROT_V3-14-159_20270101T010203_20270130T010203_R28002112234.bc"]),
        (ProcessingStepIdentifier.l1b_cam,
         ["LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
          "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc"]),
        (ProcessingStepIdentifier.l1b_rad,
         ["LIBERA_L1B_RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
          "LIBERA_L1B_RAD_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc"]),
        (ProcessingStepIdentifier.l2_cam_cf,
         ["LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
          "LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc"]),
        (ProcessingStepIdentifier.l2_rad_ssw_toa,
         ["LIBERA_L2_SSW-TOA_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
          "LIBERA_L2_SSW-TOA_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc"]),
    ]
)
def test_s3_list_objects(tmp_path, make_test_archive_buckets,
                         processing_step, file_names):
    """Test listing objects in an S3 bucket for a given processing step."""
    expected_s3_paths = []
    for file in file_names:
        Path.touch(tmp_path / file)

        # Make the expected file path in S3
        filename_object = filenaming.AbstractValidFilename.from_file_path(tmp_path / file)
        processing_step = ProcessingStepIdentifier(processing_step)
        archive_bucket_name = processing_step.get_archive_bucket_name(account_suffix=LiberaAccountSuffix.TEST)
        s3_path = f"s3://{archive_bucket_name}"
        expected_s3_paths.append(filename_object.generate_prefixed_path(s3_path))

        # Run the upload function
        s3_utilities.s3_put_in_archive_for_processing_step(tmp_path / file, processing_step,
                                                           account_suffix=LiberaAccountSuffix.TEST)
    # List out all files in the s3 bucket
    found_files = s3_utilities.s3_list_archive_files(processing_step, account_suffix=LiberaAccountSuffix.TEST)
    # Check that the number of files found matches the number of files uploaded
    assert len(found_files) == len(file_names)
    # Check that the expected files are in the S3 bucket
    for file in found_files:
        assert file in expected_s3_paths


@pytest.mark.parametrize(
    ("from_path", "to_path"),
    [
        ("s3://somebucket/with/file/path.nc", "."),
        ("s3://somebucket/with/file/path.nc", "some/local/path.nc"),
    ]
)
@patch('libera_utils.aws.s3_utilities.s3_copy_file')
def test_s3_utils_copy_cli_handler(mock_s3_copy_files, from_path, to_path):
    """Test the S3 utilities CLI handler for file copy."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=s3_utilities.s3_copy_cli_handler,
        source_path=from_path,
        dest_path=to_path,
        delete=False
    )

    # Call the CLI handler
    s3_utilities.s3_copy_cli_handler(args)

    # Check the mocked function was called correctly
    expected_from_path = AnyPath(from_path)
    expected_to_path = AnyPath(to_path)
    mock_s3_copy_files.assert_called_once_with(expected_from_path, expected_to_path, delete=args.delete)


# s3_copy_files is a wrapper around smart_copy and is tested by test_smart_open.py. Look to
# test_smart_copy_file_remote_to_local_directory and test_smart_copy_file_remote_to_local_file
