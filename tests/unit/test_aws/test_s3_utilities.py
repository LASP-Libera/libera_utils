"""Tests for the s3_utilities module (s3-utils CLI handlers and manual data ingest workflow)"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from cloudpathlib import AnyPath, S3Path
from moto import mock_aws

from libera_utils.aws import s3_utilities
from libera_utils.constants import DataProductIdentifier
from libera_utils.io import filenaming


class TestManualIngestPut:
    """Tests for the manual data ingest ``s3-utils put`` CLI handler and its workflow functions.

    The ``put`` command stages Libera data product files into the SDC Ingest Dropbox bucket and emits a single
    NewFilesAvailable event to the SDC event bus. These tests mock all AWS interactions with moto and do not need
    real data product file contents (the Data Ingester does not open the files).
    """

    @pytest.mark.parametrize(
        ("file_paths", "profile"),
        [
            (["some/file/path.nc"], None),
            (["some/file/path.nc", "another/file/path.nc"], "test"),
        ],
    )
    @patch("libera_utils.aws.s3_utilities.manual_ingest_data_products")
    @patch("libera_utils.aws.s3_utilities.get_l2_team_role_session")
    def test_cli_handler_creates_session_and_delegates(self, mock_get_session, mock_manual_ingest, file_paths, profile):
        """The CLI handler builds a LiberaUtils session from the profile and delegates to manual_ingest_data_products."""
        args = argparse.Namespace(
            func=s3_utilities.s3_put_cli_handler,
            file_paths=file_paths,
            profile=profile,
            verify=False,
            timeout=s3_utilities.DEFAULT_VERIFY_TIMEOUT_SECONDS,
        )

        s3_utilities.s3_put_cli_handler(args)

        # A single (role-assumed) session is created from the profile and threaded into the workflow function.
        mock_get_session.assert_called_once_with(profile_name=profile)
        expected_paths = [Path(p) for p in file_paths]
        mock_manual_ingest.assert_called_once_with(expected_paths, boto_session=mock_get_session.return_value)

    @pytest.mark.parametrize(
        "file_names",
        [
            # Single L1B data product
            ["LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"],
            # SPICE kernels
            [
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc",
            ],
            # L0 PDS (file number 01) and L0 CR (file number 00)
            [
                "P1590011SOMESCIENCEAAA99030231459001.PDS",
                "P1590011SOMESCIENCEAAA99030231459000.PDS",
            ],
            # Mixed batch spanning data levels, including L0, CR, SPICE, L1B, and L2
            [
                "P1590011SOMESCIENCEAAA99030231459001.PDS",
                "P1590011SOMESCIENCEAAA99030231459000.PDS",
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
                "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            ],
        ],
    )
    def test_manual_ingest_stages_files_and_emits_event(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session, file_names
    ):
        """All file types are staged to the dropbox and described in a single NewFilesAvailable event."""
        dropbox_bucket = make_ingest_dropbox_bucket

        # Write dummy files with distinct sizes so we can verify the size field maps to the right file.
        paths = []
        expected_sizes = {}
        for i, name in enumerate(file_names):
            file_path = tmp_path / name
            file_path.write_bytes(b"x" * (i + 1))
            paths.append(file_path)
            expected_sizes[name] = i + 1

        session, captured = make_event_capturing_session()
        s3_utilities.manual_ingest_data_products(paths, boto_session=session)

        # 1. Every file was uploaded to the Ingest Dropbox at the bucket root (bare filename key).
        for name in file_names:
            assert S3Path(f"s3://{dropbox_bucket}/{name}").exists()

        # 2. A single NewFilesAvailable event was emitted with the exact source and detail-type values.
        entries = captured["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["Source"] == "manual-processing"
        assert entry["DetailType"] == "NewFilesAvailableEventDetail"
        assert entry["EventBusName"] == make_sdc_event_bus

        # 3. The event detail lists every staged file with correct type, uri, name, and size.
        detail = json.loads(entry["Detail"])
        expected_files = [
            {
                "type": "data",
                "uri": f"s3://{dropbox_bucket}/{name}",
                "name": name,
                "size": expected_sizes[name],
            }
            for name in file_names
        ]
        assert detail["files"] == expected_files

    def test_invalid_filename_in_batch_raises_before_staging(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session
    ):
        """A single invalid filename aborts the whole batch before any file is staged or any event is emitted."""
        good_name = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
        bad_name = "not_a_valid_libera_file.nc"
        Path.touch(tmp_path / good_name)
        Path.touch(tmp_path / bad_name)

        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="not a valid Libera L0 or data product filename"):
            s3_utilities.manual_ingest_data_products([tmp_path / good_name, tmp_path / bad_name], boto_session=session)

        # Nothing was staged and no event was emitted.
        assert not S3Path(f"s3://{make_ingest_dropbox_bucket}/{good_name}").exists()
        assert "entries" not in captured

    def test_manifest_filename_rejected(self, mock_s3_context_with_profile):
        """Manifest filenames are not eligible for manual ingest and are rejected during validation."""
        # A valid INPUT manifest filename (26-character Crockford base32 ULID code).
        manifest_name = "LIBERA_INPUT_MANIFEST_0123456789ABCDEFGHJKMNPQRS.json"
        # Sanity check that this really is a valid manifest filename, so the test proves manifests are rejected.
        filenaming.ManifestFilename(manifest_name)

        session = boto3.Session(profile_name="test-profile")
        with pytest.raises(ValueError, match="not a valid Libera L0 or data product filename"):
            s3_utilities.manual_ingest_data_products([AnyPath(manifest_name)], boto_session=session)

    def test_put_event_raises_on_failed_entry(self, make_sdc_event_bus):
        """A failed event entry from put_events surfaces as a RuntimeError."""
        session = boto3.Session(profile_name="test-profile")
        real_client = session.client

        def failing_put_client(service_name, *args, **kwargs):
            client = real_client(service_name, *args, **kwargs)
            if service_name == "events":
                client.put_events = lambda **kwargs: {"FailedEntryCount": 1, "Entries": [{"ErrorCode": "Boom"}]}
            return client

        session.client = failing_put_client

        files = [{"type": "data", "uri": "s3://bucket/file.nc", "name": "file.nc", "size": 1}]
        with pytest.raises(RuntimeError, match="Failed to put NewFilesAvailable event"):
            s3_utilities.put_new_files_available_event(files, boto_session=session)

    @patch("libera_utils.aws.s3_utilities.verify_ingestion")
    @patch("libera_utils.aws.s3_utilities.manual_ingest_data_products")
    @patch("libera_utils.aws.s3_utilities.get_l2_team_role_session")
    def test_cli_handler_verify_invokes_verification(self, mock_get_session, mock_manual_ingest, mock_verify):
        """When --verify is set, the handler verifies ingestion using the ingest function's returned filenames."""
        returned_filenames = ["filename-1", "filename-2"]
        mock_manual_ingest.return_value = returned_filenames
        args = argparse.Namespace(
            func=s3_utilities.s3_put_cli_handler,
            file_paths=["some/file/path.nc"],
            profile=None,
            verify=True,
            timeout=42.0,
        )

        s3_utilities.s3_put_cli_handler(args)

        mock_verify.assert_called_once_with(
            returned_filenames, boto_session=mock_get_session.return_value, timeout=42.0
        )


class TestVerifyIngestion:
    """Tests for ``verify_ingestion``, the blocking post-ingest verification used by ``s3-utils put --verify``."""

    DATA_PRODUCT_FILE = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
    L0_PDS_FILE = "P1590011SOMESCIENCEAAA99030231459001.PDS"
    L0_CR_FILE = "P1590011SOMESCIENCEAAA99030231459000.PDS"

    @staticmethod
    def _seed_archive_object(session, libera_filename):
        """Put the file at its expected archive bucket + key in the (mocked) ``-test`` archive bucket."""
        bucket = libera_filename.data_product_id.data_level.archive_bucket_name + "-test"
        key = f"{libera_filename.archive_prefix}/{libera_filename.path.name}"
        session.client("s3").put_object(Bucket=bucket, Key=key, Body=b"x")

    @staticmethod
    def _seed_metadata_record(session, table_name, basename, sk):
        """Seed a single File Metadata record (PK=basename, SK=sk). The base record uses SK="#"; a data product /
        L0 PDS product record uses SK=applicable_date."""
        session.resource("dynamodb").Table(table_name).put_item(Item={"PK": basename, "SK": sk})

    @staticmethod
    def _seed_availability_record(session, table_name, libera_filename):
        session.resource("dynamodb").Table(table_name).put_item(
            Item={
                "PK": libera_filename.applicable_date.isoformat(),
                "SK": f"{libera_filename.data_product_id}#{libera_filename.filename_parts.version}",
            }
        )

    def test_verify_succeeds_when_all_records_present(
        self, make_test_archive_buckets, make_data_availability_table, make_file_metadata_table
    ):
        """All three checks present for a data product file -> verification returns without raising."""
        session = boto3.Session(profile_name="test-profile")
        libera_filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)

        self._seed_archive_object(session, libera_filename)
        self._seed_metadata_record(
            session, make_file_metadata_table, self.DATA_PRODUCT_FILE, libera_filename.applicable_date.isoformat()
        )
        self._seed_availability_record(session, make_data_availability_table, libera_filename)

        # Should not raise. poll_interval=0 keeps the (single, already-satisfied) pass instant.
        s3_utilities.verify_ingestion([libera_filename], boto_session=session, timeout=30, poll_interval=0)

    def test_verify_times_out_when_a_record_is_missing(
        self, make_test_archive_buckets, make_data_availability_table, make_file_metadata_table
    ):
        """A missing data availability record leaves the check pending and raises TimeoutError."""
        session = boto3.Session(profile_name="test-profile")
        libera_filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)

        # Seed archive + metadata but NOT the availability record.
        self._seed_archive_object(session, libera_filename)
        self._seed_metadata_record(
            session, make_file_metadata_table, self.DATA_PRODUCT_FILE, libera_filename.applicable_date.isoformat()
        )

        with pytest.raises(TimeoutError, match="Ingestion verification timed out"):
            s3_utilities.verify_ingestion([libera_filename], boto_session=session, timeout=0, poll_interval=0)

    def test_verify_l0_pds_requires_base_and_product_records(self, make_test_archive_buckets, make_file_metadata_table):
        """An L0 PDS file is verified with archive + two metadata records (base SK="#" and product SK=date); no
        availability table is required for L0 files."""
        session = boto3.Session(profile_name="test-profile")
        libera_filename = filenaming.L0Filename(self.L0_PDS_FILE)

        self._seed_archive_object(session, libera_filename)
        self._seed_metadata_record(session, make_file_metadata_table, self.L0_PDS_FILE, "#")
        self._seed_metadata_record(session, make_file_metadata_table, self.L0_PDS_FILE, "2027-01-02")

        # No data availability table exists; verification must not look for one for L0 files.
        s3_utilities.verify_ingestion([libera_filename], boto_session=session, timeout=30, poll_interval=0)

    def test_verify_l0_cr_requires_only_base_record(self, make_test_archive_buckets, make_file_metadata_table):
        """An L0 CR (construction record) file is verified with archive + exactly one (base) metadata record."""
        session = boto3.Session(profile_name="test-profile")
        libera_filename = filenaming.L0Filename(self.L0_CR_FILE)

        self._seed_archive_object(session, libera_filename)
        self._seed_metadata_record(session, make_file_metadata_table, self.L0_CR_FILE, "#")

        s3_utilities.verify_ingestion([libera_filename], boto_session=session, timeout=30, poll_interval=0)

    def test_verify_l0_pds_times_out_with_only_base_record(self, make_test_archive_buckets, make_file_metadata_table):
        """An L0 PDS file with only its base record (missing the product record) does not verify and times out."""
        session = boto3.Session(profile_name="test-profile")
        libera_filename = filenaming.L0Filename(self.L0_PDS_FILE)

        self._seed_archive_object(session, libera_filename)
        self._seed_metadata_record(session, make_file_metadata_table, self.L0_PDS_FILE, "#")

        with pytest.raises(TimeoutError, match="Ingestion verification timed out"):
            s3_utilities.verify_ingestion([libera_filename], boto_session=session, timeout=0, poll_interval=0)

    def test_verify_raises_immediately_when_table_missing(self, make_test_archive_buckets):
        """A missing required resource (the file metadata table) raises immediately, before any polling."""
        session = boto3.Session(profile_name="test-profile")
        libera_filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)
        self._seed_archive_object(session, libera_filename)

        with pytest.raises(ValueError, match="Error finding a single DynamoDB table"):
            s3_utilities.verify_ingestion([libera_filename], boto_session=session, timeout=30, poll_interval=0)


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
            DataProductIdentifier.l2_cf_cam_camtime,
            [
                "LIBERA_L2_CF-CAM-CAMTIME_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L2_CF-CAM-CAMTIME_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc",
            ],
        ),
        (
            DataProductIdentifier.l2_toa_flux_cam,
            [
                "LIBERA_L2_TOA-FLUX-CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
                "LIBERA_L2_TOA-FLUX-CAM_V3-14-159_20270102T112233_20270102T122233_R27002112234.nc",
            ],
        ),
    ],
)
def test_s3_list_objects(tmp_path, make_test_archive_buckets, write_file_to_s3, product_id, file_names):
    """Test listing objects in an S3 bucket for a given processing step."""
    expected_s3_paths = []
    for file in file_names:
        Path.touch(tmp_path / file)

        # Build the correct archive path for the file and seed it into the (mocked) archive bucket.
        filename_object = filenaming.AbstractValidFilename.from_file_path(tmp_path / file)
        archive_bucket_name = filename_object.data_product_id.data_level.archive_bucket_name + "-test"
        archive_path = filename_object.generate_prefixed_path(f"s3://{archive_bucket_name}")
        expected_s3_paths.append(write_file_to_s3(tmp_path / file, str(archive_path)))

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
    write_file_to_s3,
):
    """Test that listing objects in an S3 bucket uses the correct prefix for a given processing step."""
    # Create files in two different prefixes
    file_in_prefix_1 = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
    file_in_prefix_2 = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
    Path.touch(tmp_path / file_in_prefix_1)
    Path.touch(tmp_path / file_in_prefix_2)

    # Seed both files into the L1B archive bucket (different product prefixes within the same bucket)
    for file in (file_in_prefix_1, file_in_prefix_2):
        filename_object = filenaming.AbstractValidFilename.from_file_path(tmp_path / file)
        archive_bucket_name = filename_object.data_product_id.data_level.archive_bucket_name + "-test"
        archive_path = filename_object.generate_prefixed_path(f"s3://{archive_bucket_name}")
        write_file_to_s3(tmp_path / file, str(archive_path))

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
