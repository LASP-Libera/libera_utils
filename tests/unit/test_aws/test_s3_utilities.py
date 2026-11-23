"""Tests for the s3_utilities module (s3-utils CLI handlers and manual data ingest workflow)"""

import argparse
import json
from operator import itemgetter
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
        ("file_names", "n_events"),
        [
            # Single L1B data product
            (["LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"], 1),
            # SPICE kernels
            (
                [
                    "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20280102T112233000000.bsp",
                    "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R20280102T112233000000.bc",
                ],
                1,
            ),
            # L0 PDS (file number 01) and L0 CR (file number 00)
            (
                [
                    "P1590011SOMESCIENCEAAA99030231459001.PDS",
                    "P1590011SOMESCIENCEAAA99030231459000.PDS",
                ],
                1,
            ),
            # Mixed batch spanning data levels, including L0, CR, SPICE, L1B, and L2: the construction
            # record and its PDS files (same SCID, first APID, created time and numeric ID) travel in
            # their own event
            (
                [
                    "P1590011SOMESCIENCEAAA99030231459001.PDS",
                    "P1590011SOMESCIENCEAAA99030231459000.PDS",
                    "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20280102T112233000000.bsp",
                    "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc",
                    "LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc",
                ],
                2,
            ),
            # Demuxed ground CCSDS basename (single APID, no extension)
            (["LIBERA_SDC_1040_ccsds_2025_318_12_00_00"], 1),
            # Version-partitioned data product
            (["LIBERA_AUX_ADM-CAM_V3-14-159_R20270102T112233000000.nc"], 1),
            # External ancillary products, under their upstream names (ERA5 under the SDC's scheme), and the
            # version archives of IGBP and the land/ocean mask
            (
                [
                    "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc",
                    "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc",
                    "AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc",
                    "CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc",
                    "VJ143C1.A2026153.002.2026161161054.h5",
                    "IVCDB_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5",
                    "ICDBG_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5",
                    "NISE_AMSR2_20260610.HDFEOS",
                    "NSIDC-LAND-OCEAN-MASK_1_20000101_R20261020T120000000000.zip",
                    "MODIS-IGBP_061_20250724_R20261020T120000000000.zip",
                    "ECMWF_ERA5-SINGLE-LEVEL_ERA5T_20250101T000000_20250101T230000_R20251028T120000000000.nc",
                    "ECMWF_ERA5-PRESSURE-LEVEL_ERA5_20250101T000000_20250101T230000_R20251028T120000000000.nc",
                ],
                3,
            ),
        ],
    )
    def test_manual_ingest_stages_files_and_emits_event(
        self,
        tmp_path,
        make_ingest_dropbox_bucket,
        make_sdc_event_bus,
        make_event_capturing_session,
        file_names,
        n_events,
    ):
        """All file types are staged to the dropbox and described in NewFilesAvailable events."""
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

        # 2. Every emitted NewFilesAvailable event carries the exact source and detail-type values.
        entries = captured["entries"]
        assert len(entries) == n_events
        for entry in entries:
            assert entry["Source"] == "manual-processing"
            assert entry["DetailType"] == "NewFilesAvailableEventDetail"
            assert entry["EventBusName"] == make_sdc_event_bus

        # 3. Across all events, every staged file is described exactly once with correct type, uri, name, and size.
        emitted_files = [file for entry in entries for file in json.loads(entry["Detail"])["files"]]
        expected_files = [
            {
                "type": "data",
                "uri": f"s3://{dropbox_bucket}/{name}",
                "name": name,
                "size": expected_sizes[name],
            }
            for name in file_names
        ]
        assert sorted(emitted_files, key=itemgetter("name")) == sorted(expected_files, key=itemgetter("name"))

    def test_large_batch_is_split_across_multiple_events(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session
    ):
        """A batch larger than one ingester invocation can handle is split across several events."""
        file_names = [
            f"LIBERA_SDC_1040_ccsds_2025_318_{2 * hour:02d}_00_00"
            for hour in range(2 * s3_utilities.MAX_FILES_PER_INGEST_EVENT + 1)
        ]
        paths = []
        for name in file_names:
            file_path = tmp_path / name
            file_path.write_bytes(b"x")
            paths.append(file_path)

        session, captured = make_event_capturing_session()
        s3_utilities.manual_ingest_data_products(paths, boto_session=session)

        entries = captured["entries"]
        assert len(entries) == 3

        chunks = [json.loads(entry["Detail"])["files"] for entry in entries]
        # No event describes more files than one Data Ingester invocation can archive in its timeout.
        assert all(len(chunk) <= s3_utilities.MAX_FILES_PER_INGEST_EVENT for chunk in chunks)
        # Every file is described exactly once across the events.
        emitted_names = [file["name"] for chunk in chunks for file in chunk]
        assert sorted(emitted_names) == sorted(file_names)

    def test_cr_and_its_pds_files_share_one_event_cr_first(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session
    ):
        """A construction record and the PDS files that share its SCID, first APID, created time and numeric ID
        stay in one event, so the ingester archives the CR ahead of the PDS files that read it."""
        # One construction record (file number 00) and more PDS files than fit in a normal chunk, listed
        # PDS first, then one group per key part that differs from it in only that part.
        large_group = [
            f"P1590011SOMESCIENCEAAA9903023145900{number}.PDS"
            for number in reversed(range(s3_utilities.MAX_FILES_PER_INGEST_EVENT + 2))
        ]
        other_scid = [f"P1600011SOMESCIENCEAAA9903023145900{number}.PDS" for number in range(2)]
        other_apid = [f"P1590012SOMESCIENCEAAA9903023145900{number}.PDS" for number in range(2)]
        other_created_time = [f"P1590011SOMESCIENCEAAA9903123145900{number}.PDS" for number in range(2)]
        other_numeric_id = [f"P1590011SOMESCIENCEAAA9903023145910{number}.PDS" for number in range(2)]
        groups = [large_group, other_scid, other_apid, other_created_time, other_numeric_id]
        l0_names = [name for group in groups for name in group]
        other_names = ["LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"]
        paths = []
        for name in l0_names + other_names:
            file_path = tmp_path / name
            file_path.write_bytes(b"x")
            paths.append(file_path)

        session, captured = make_event_capturing_session()
        s3_utilities.manual_ingest_data_products(paths, boto_session=session)

        chunks = [json.loads(entry["Detail"])["files"] for entry in captured["entries"]]
        l0_chunks = [chunk for chunk in chunks if any(file["name"].endswith(".PDS") for file in chunk)]
        assert len(l0_chunks) == len(groups), (
            "each construction record and the PDS files sharing its SCID, first APID, created time and numeric ID "
            "must be described by exactly one event"
        )
        assert sorted(sorted(file["name"] for file in chunk) for chunk in l0_chunks) == sorted(
            sorted(group) for group in groups
        )
        assert all(chunk[0]["name"].endswith("00.PDS") for chunk in l0_chunks), "the CR leads its event"
        # The CR/PDS events hold only L0 files, so the non-L0 product went out separately.
        assert len(chunks) == len(groups) + 1

    def test_pds_files_without_their_cr_are_split_at_the_limit(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session
    ):
        """PDS files sharing a key but not its construction record have no ordering dependency, so they take the
        ordinary limit."""
        limit = s3_utilities.MAX_FILES_PER_INGEST_EVENT
        names = [f"P1590011SOMESCIENCEAAA990302314590{number:02d}.PDS" for number in range(1, 2 * limit + 2)]
        paths = []
        for name in names:
            file_path = tmp_path / name
            file_path.write_bytes(b"x")
            paths.append(file_path)

        session, captured = make_event_capturing_session()
        s3_utilities.manual_ingest_data_products(paths, boto_session=session)

        chunks = [json.loads(entry["Detail"])["files"] for entry in captured["entries"]]
        assert len(chunks) == 3
        assert all(len(chunk) <= limit for chunk in chunks)
        assert sorted(file["name"] for chunk in chunks for file in chunk) == sorted(names)

    def test_invalid_filename_in_batch_raises_before_staging(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session
    ):
        """A single invalid filename aborts the whole batch before any file is staged or any event is emitted."""
        good_name = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"
        bad_name = "not_a_valid_libera_file.nc"
        Path.touch(tmp_path / good_name)
        Path.touch(tmp_path / bad_name)

        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="cannot be manually ingested"):
            s3_utilities.manual_ingest_data_products([tmp_path / good_name, tmp_path / bad_name], boto_session=session)

        # Nothing was staged and no event was emitted.
        assert not S3Path(f"s3://{make_ingest_dropbox_bucket}/{good_name}").exists()
        assert "entries" not in captured

    def test_loose_archive_member_rejected_naming_its_archive(
        self, tmp_path, make_ingest_dropbox_bucket, make_sdc_event_bus, make_event_capturing_session
    ):
        """A loose IGBP tile is rejected with a message naming its version archive, and nothing is staged."""
        tile_name = "MCD12Q1.A2024001.h21v07.061.2025205222320.hdf"
        Path.touch(tmp_path / tile_name)

        session, captured = make_event_capturing_session()
        with pytest.raises(
            ValueError,
            match="cannot be manually ingested: .* ingested only inside its version archive "
            r"MODIS-IGBP_<version>_<production YYYYMMDD>_<revision>\.zip",
        ):
            s3_utilities.manual_ingest_data_products([tmp_path / tile_name], boto_session=session)

        assert not S3Path(f"s3://{make_ingest_dropbox_bucket}/{tile_name}").exists()
        assert "entries" not in captured

    def test_manifest_filename_rejected(self, mock_s3_context_with_profile):
        """Manifest filenames are not eligible for manual ingest and are rejected during validation."""
        # A valid INPUT manifest filename (26-character Crockford base32 ULID code).
        manifest_name = "LIBERA_INPUT_MANIFEST_0123456789ABCDEFGHJKMNPQRS.json"
        # Sanity check that this really is a valid manifest filename, so the test proves manifests are rejected.
        filenaming.ManifestFilename(manifest_name)

        session = boto3.Session(profile_name="test-profile")
        with pytest.raises(ValueError, match="cannot be manually ingested"):
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
        with pytest.raises(RuntimeError, match=r"Failed to put NewFilesAvailable event for \['file.nc'\].*re-run"):
            s3_utilities.put_new_files_available_event(files, boto_session=session)

    @patch("libera_utils.aws.s3_utilities.verify_ingestion")
    @patch("libera_utils.aws.s3_utilities.manual_ingest_data_products")
    @patch("libera_utils.aws.s3_utilities.get_l2_team_role_session")
    def test_cli_handler_verify_invokes_verification(self, mock_get_session, mock_manual_ingest, mock_verify):
        """When --verify is set, the handler verifies ingestion using the ingest function's returned filenames."""
        returned_filenames = [
            filenaming.LiberaDataProductFilename(
                "some/dir/LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"
            ),
            filenaming.L0Filename("some/dir/P1590011SOMESCIENCEAAA99030231459001.PDS"),
            filenaming.ModisViirsDailyFilename("AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc"),
        ]
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

    @patch("libera_utils.aws.s3_utilities.verify_ingestion")
    @patch("libera_utils.aws.s3_utilities.manual_ingest_data_products")
    @patch("libera_utils.aws.s3_utilities.get_l2_team_role_session")
    def test_cli_handler_verify_verifies_every_staged_file(self, mock_get_session, mock_manual_ingest, mock_verify):
        """With --verify, every staged file is verified, version-partitioned ones included, and nothing raises."""
        filenames = [
            filenaming.LiberaDataProductFilename(
                "some/dir/LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"
            ),
            filenaming.LiberaVersionPartitionedDataProductFilename(
                "some/dir/LIBERA_AUX_ADM-CAM_V3-14-159_R20270102T112233000000.nc"
            ),
            filenaming.AncillaryVersionArchiveFilename("NSIDC-LAND-OCEAN-MASK_1_20000101_R20261020T120000000000.zip"),
        ]
        mock_manual_ingest.return_value = filenames
        args = argparse.Namespace(
            func=s3_utilities.s3_put_cli_handler,
            file_paths=[str(f.path) for f in filenames],
            profile=None,
            verify=True,
            timeout=42.0,
        )

        s3_utilities.s3_put_cli_handler(args)

        mock_manual_ingest.assert_called_once_with(
            [Path(f.path) for f in filenames], boto_session=mock_get_session.return_value
        )
        mock_verify.assert_called_once_with(filenames, boto_session=mock_get_session.return_value, timeout=42.0)


class TestVerifyIngestion:
    """Tests for ``verify_ingestion``, the blocking post-ingest verification used by ``s3-utils put --verify``.

    Only the File Metadata table is created: verification reads each file's base record (SK "#") and nothing else, so
    a test passing with no archive bucket or Data Availability table shows neither is read.
    """

    DATA_PRODUCT_FILE = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"

    @staticmethod
    def _seed_record(table_name, basename, sk="#", **attributes):
        """Seed one File Metadata record (PK=basename, SK=sk) with the given attributes."""
        boto3.Session(profile_name="test-profile").resource("dynamodb").Table(table_name).put_item(
            Item={"PK": basename, "SK": sk, **attributes}
        )

    @pytest.mark.parametrize(
        "filename",
        [
            pytest.param(filenaming.LiberaDataProductFilename(DATA_PRODUCT_FILE), id="libera-daily"),
            pytest.param(
                filenaming.LiberaDataProductFilename(
                    "LIBERA_L1A_NOM-HK-NOMINAL-DARKS-TRIMMED_V1-0-0_20270102T193000_20270102T194500_"
                    "R20270103T000000000000.nc"
                ),
                id="libera-event",
            ),
            pytest.param(
                filenaming.LiberaVersionPartitionedDataProductFilename(
                    "LIBERA_AUX_ADM-CAM_V3-14-159_R20270102T112233000000.nc"
                ),
                id="libera-version-partitioned",
            ),
            pytest.param(
                filenaming.CeresFilename("CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"), id="ceres"
            ),
            pytest.param(
                filenaming.ModisViirsDailyFilename("AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc"),
                id="daily-ancillary",
            ),
            pytest.param(filenaming.NsidcNiseFilename("NISE_AMSR2_20260610.HDFEOS"), id="nise"),
            pytest.param(
                filenaming.AncillaryVersionArchiveFilename(
                    "NSIDC-LAND-OCEAN-MASK_1_20000101_R20261020T120000000000.zip"
                ),
                id="version-archive",
            ),
            pytest.param(filenaming.L0Filename("P1590011SOMESCIENCEAAA99030231459001.PDS"), id="l0-pds"),
            pytest.param(filenaming.L0Filename("P1590011SOMESCIENCEAAA99030231459000.PDS"), id="l0-cr"),
            pytest.param(filenaming.LiberaGroundCcsdsFilename("LIBERA_SDC_1040_ccsds_2025_318_12_00_00"), id="ground"),
        ],
    )
    def test_verify_passes_on_a_base_record_ingested_and_archived(self, make_file_metadata_table, filename):
        """Any staged file passes once its base record has both ingested-time and archived-time."""
        self._seed_record(
            make_file_metadata_table,
            filename.path.name,
            **{"ingested-time": "2027-01-03T00:00:00Z", "archived-time": "2027-01-03T00:00:05Z"},
        )

        s3_utilities.verify_ingestion(
            [filename], boto_session=boto3.Session(profile_name="test-profile"), timeout=0, poll_interval=0
        )

    @pytest.mark.parametrize(
        ("attributes", "missing"),
        [
            pytest.param(None, "no base record", id="no-record"),
            pytest.param({"ingested-time": "2027-01-03T00:00:00Z"}, "archived-time not set", id="not-archived"),
            pytest.param({"archived-time": "2027-01-03T00:00:05Z"}, "ingested-time not set", id="not-ingested"),
            pytest.param({}, "ingested-time and archived-time not set", id="neither"),
            pytest.param(
                {"ingested-time": "", "archived-time": "2027-01-03T00:00:05Z"}, "ingested-time not set", id="empty"
            ),
        ],
    )
    def test_verify_times_out_naming_what_the_base_record_lacks(
        self, make_file_metadata_table, caplog, attributes, missing
    ):
        """A file whose base record is missing or lacks a timestamp times out, and the summary says what it lacks."""
        filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)
        if attributes is not None:
            self._seed_record(make_file_metadata_table, filename.path.name, **attributes)

        with caplog.at_level("INFO"), pytest.raises(TimeoutError, match="1 of 1 file"):
            s3_utilities.verify_ingestion(
                [filename], boto_session=boto3.Session(profile_name="test-profile"), timeout=0, poll_interval=0
            )

        assert f"{filename.path.name}: MISSING ({missing})" in caplog.text

    def test_verify_passes_once_a_later_poll_finds_the_record(self, make_file_metadata_table):
        """A base record that appears while verification polls is found on the next pass, with no timeout."""
        filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)

        def archive_during_the_wait(_seconds):
            self._seed_record(
                make_file_metadata_table,
                filename.path.name,
                **{"ingested-time": "2027-01-03T00:00:00Z", "archived-time": "2027-01-03T00:00:05Z"},
            )

        with patch("libera_utils.aws.s3_utilities.time.sleep", side_effect=archive_during_the_wait) as mock_sleep:
            s3_utilities.verify_ingestion(
                [filename], boto_session=boto3.Session(profile_name="test-profile"), timeout=60, poll_interval=10
            )

        mock_sleep.assert_called_once_with(10)

    def test_verify_reads_only_the_base_record(self, make_file_metadata_table):
        """A PARTITION# record carrying both timestamps does not stand in for the base record."""
        filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)
        self._seed_record(
            make_file_metadata_table,
            filename.path.name,
            sk=f"PARTITION#{filename.partition_key}",
            **{"ingested-time": "2027-01-03T00:00:00Z", "archived-time": "2027-01-03T00:00:05Z"},
        )

        with pytest.raises(TimeoutError, match="Ingestion verification timed out"):
            s3_utilities.verify_ingestion(
                [filename], boto_session=boto3.Session(profile_name="test-profile"), timeout=0, poll_interval=0
            )

    def test_verify_reports_only_the_files_not_yet_ingested(self, make_file_metadata_table, caplog):
        """With several files, those whose base records pass are reported PASS and the rest MISSING."""
        done = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)
        pending = filenaming.L0Filename("P1590011SOMESCIENCEAAA99030231459001.PDS")
        self._seed_record(
            make_file_metadata_table,
            done.path.name,
            **{"ingested-time": "2027-01-03T00:00:00Z", "archived-time": "2027-01-03T00:00:05Z"},
        )

        with caplog.at_level("INFO"), pytest.raises(TimeoutError, match="1 of 2 file"):
            s3_utilities.verify_ingestion(
                [done, pending], boto_session=boto3.Session(profile_name="test-profile"), timeout=0, poll_interval=0
            )

        assert f"{done.path.name}: PASS" in caplog.text
        assert f"{pending.path.name}: MISSING (no base record)" in caplog.text

    def test_verify_raises_immediately_when_table_missing(self, mock_s3_context_with_profile):
        """A missing File Metadata table raises immediately, before any polling."""
        filename = filenaming.LiberaDataProductFilename(self.DATA_PRODUCT_FILE)

        with pytest.raises(ValueError, match="Error finding a single DynamoDB table"):
            s3_utilities.verify_ingestion(
                [filename], boto_session=boto3.Session(profile_name="test-profile"), timeout=30, poll_interval=0
            )


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
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20280102T112233000000.bsp",
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20280102T112234000000.bsp",
            ],
        ),
        (
            DataProductIdentifier.spice_az_ck,
            [
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R20280102T112233000000.bc",
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R20280102T112234000000.bc",
            ],
        ),
        (
            DataProductIdentifier.l1b_cam,
            [
                "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc",
                "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112234000000.nc",
            ],
        ),
        (
            DataProductIdentifier.l1b_rad,
            [
                "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc",
                "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112234000000.nc",
            ],
        ),
        (
            DataProductIdentifier.l2_cf_cam_camtime,
            [
                "LIBERA_L2_CF-CAM-CAMTIME_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc",
                "LIBERA_L2_CF-CAM-CAMTIME_V3-14-159_20270102T112233_20270102T122233_R20270102T112234000000.nc",
            ],
        ),
        (
            DataProductIdentifier.l2_toa_flux_cam,
            [
                "LIBERA_L2_TOA-FLUX-CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc",
                "LIBERA_L2_TOA-FLUX-CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112234000000.nc",
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
    file_in_prefix_1 = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"
    file_in_prefix_2 = "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270102T112233000000.nc"
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
