"""File for testing ECR upload module"""

import argparse
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from cloudpathlib import AnyPath, S3Path
from moto import mock_aws

from libera_utils.aws import s3_utilities
from libera_utils.constants import DataProductIdentifier
from libera_utils.io import filenaming


@mock_aws
def test_find_bucket_ambiguity():
    """
    Test that find_bucket... raises ValueError if it finds 0 or >1 buckets.
    """
    s3 = boto3.client("s3", region_name="us-east-1")

    # Setup: Create two buckets that share a substring
    s3.create_bucket(Bucket="libera-l1b-cam-test-v1")
    s3.create_bucket(Bucket="libera-l1b-cam-test-v2")
    s3.create_bucket(Bucket="libera-l2-rad-test")

    session = boto3.Session()

    # Case 1: Ambiguity (finding 'l1b' matches v1 and v2)
    with pytest.raises(ValueError, match="Error finding a single bucket"):
        s3_utilities.find_bucket_in_account_by_partial_name(session, "l1b")

    # Case 2: No match
    with pytest.raises(ValueError, match="Error finding a single bucket"):
        s3_utilities.find_bucket_in_account_by_partial_name(session, "l0-packet")

    # Case 3: Success (Specific enough to match only one)
    # Matching "l2" should only find the one l2 bucket
    result = s3_utilities.find_bucket_in_account_by_partial_name(session, "l2")
    assert result == "libera-l2-rad-test"


@pytest.mark.parametrize(
    ("file_path", "profile"),
    [
        ("some/file/path.nc", None),
        ("some/file/path.nc", "test"),
    ],
)
@patch("libera_utils.aws.s3_utilities.s3_put_in_archive_for_processing_step")
def test_s3_utils_put_cli_handler(mock_s3_put_for_processing_step, file_path, profile):
    """Test the S3 utilities CLI handler for file upload."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=s3_utilities.s3_put_cli_handler,
        file_path=file_path,
        profile=profile,
    )

    # Call the CLI handler
    s3_utilities.s3_put_cli_handler(args)

    # Check the mocked function was called correctly
    expected_file_path = AnyPath(file_path)
    expected_profile = profile
    mock_s3_put_for_processing_step.assert_called_once_with(expected_file_path, profile_name=expected_profile)


@pytest.mark.parametrize(
    "file_name",
    [
        "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
        "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc",
        "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "LIBERA_L2_SSW-TOA-FLUXES-OSSE_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
    ],
)
def test_s3_put_for_processing_step(tmp_path, make_test_archive_buckets, file_name):
    """Test uploading an object to an S3 bucket for a given processing step."""

    # Create an empty file that is correctly named file in a temp directory
    Path.touch(tmp_path / file_name)

    # Make the expected file path in S3
    filename_object = filenaming.LiberaDataProductFilename.from_file_path(tmp_path / file_name)
    bucket_name = filename_object.data_product_id.data_level.archive_bucket_name + "-test"
    s3_path = f"s3://{bucket_name}"
    expected_s3_path = filename_object.generate_prefixed_path(s3_path)

    # Run the upload function
    s3_utilities.s3_put_in_archive_for_processing_step(tmp_path / file_name, profile_name="test-profile")
    # Check that the expected file is in the S3 bucket
    assert S3Path(expected_s3_path).exists()


@pytest.mark.parametrize(
    ("product_name", "profile"),
    [
        ("CAM", None),
        ("CAM", "test"),
    ],
)
@patch("libera_utils.aws.s3_utilities.s3_list_archive_files")
def test_s3_utils_list_cli_handler(mock_s3_list_files, product_name, profile):
    """Test the S3 utilities CLI handler for listing files in a bucket."""
    # Make the input namespace object
    args = argparse.Namespace(func=s3_utilities.s3_list_cli_handler, product_name=product_name, profile=profile)

    # Call the CLI handler
    s3_utilities.s3_list_cli_handler(args)

    # Check the mocked function was called correctly
    expected_product_id = DataProductIdentifier(product_name)
    expected_profile = profile
    mock_s3_list_files.assert_called_once_with(expected_product_id, profile_name=expected_profile)


@mock_aws
@pytest.mark.parametrize(
    ("product_id", "file_names"),
    [
        (
            DataProductIdentifier.spice_jpss_spk,
            [
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112234.bsp",
            ],
        ),
        (
            DataProductIdentifier.spice_az_ck,
            [
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc",
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R28002112234.bc",
            ],
        ),
        (
            DataProductIdentifier.l1b_cam,
            [
                "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc",
            ],
        ),
        (
            DataProductIdentifier.l1b_rad,
            [
                "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc",
            ],
        ),
        (
            DataProductIdentifier.l2_cf_cam,
            [
                "LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc",
            ],
        ),
        (
            DataProductIdentifier.l2_ssw_toa_osse,
            [
                "LIBERA_L2_SSW-TOA-FLUXES-OSSE_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L2_SSW-TOA-FLUXES-OSSE_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc",
            ],
        ),
    ],
)
def test_s3_list_objects(tmp_path, make_test_archive_buckets, product_id, file_names):
    """Test listing objects in an S3 bucket for a given processing step."""
    expected_s3_paths = []
    for file in file_names:
        Path.touch(tmp_path / file)

        # Make the expected file path in S3
        filename_object = filenaming.AbstractValidFilename.from_file_path(tmp_path / file)
        product_id = DataProductIdentifier(product_id)
        archive_bucket_name = product_id.data_level.archive_bucket_name + "-test"
        s3_path = f"s3://{archive_bucket_name}"
        expected_s3_paths.append(filename_object.generate_prefixed_path(s3_path))

        # Run the upload function
        s3_utilities.s3_put_in_archive_for_processing_step(tmp_path / file, profile_name="test-profile")
    # List out all files in the s3 bucket
    found_files = s3_utilities.s3_list_archive_files(product_id, profile_name="test-profile")
    # Check that the number of files found matches the number of files uploaded
    assert len(found_files) == len(file_names)
    # Check that the expected files are in the S3 bucket
    for file in found_files:
        assert file in expected_s3_paths


def test_s3_list_objects_correct_prefix(
    tmp_path,
    make_test_archive_buckets,
):
    """Test that listing objects in an S3 bucket uses the correct prefix for a given processing step."""
    # Create files in two different prefixes
    file_in_prefix_1 = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
    file_in_prefix_2 = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
    Path.touch(tmp_path / file_in_prefix_1)
    Path.touch(tmp_path / file_in_prefix_2)

    # Upload both files to the CAM processing step archive bucket
    s3_utilities.s3_put_in_archive_for_processing_step(
        tmp_path / file_in_prefix_1,
    )
    s3_utilities.s3_put_in_archive_for_processing_step(
        tmp_path / file_in_prefix_2,
    )

    # List out all files in the CAM archive bucket
    found_files = s3_utilities.s3_list_archive_files(DataProductIdentifier.l1b_cam)

    # Check that only the CAM file is listed
    assert len(found_files) == 1
    assert any(file_in_prefix_1 in str(file) for file in found_files)
    assert all(file_in_prefix_2 not in str(file) for file in found_files)


@pytest.mark.parametrize(
    ("from_path", "to_path"),
    [
        ("s3://somebucket/with/file/path.nc", "."),
        ("s3://somebucket/with/file/path.nc", "some/local/path.nc"),
    ],
)
@patch("libera_utils.aws.s3_utilities.s3_copy_file")
def test_s3_utils_copy_cli_handler(mock_s3_copy_files, from_path, to_path):
    """Test the S3 utilities CLI handler for file copy."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=s3_utilities.s3_copy_cli_handler, source_path=from_path, dest_path=to_path, delete=False, profile=None
    )

    # Call the CLI handler
    s3_utilities.s3_copy_cli_handler(args)

    # Check the mocked function was called correctly
    expected_from_path = AnyPath(from_path)
    expected_to_path = AnyPath(to_path)
    mock_s3_copy_files.assert_called_once_with(expected_from_path, expected_to_path, delete=args.delete)


# s3_copy_files is a wrapper around smart_copy and is tested by test_smart_open.py. Look to
# test_smart_copy_file_remote_to_local_directory and test_smart_copy_file_remote_to_local_file
