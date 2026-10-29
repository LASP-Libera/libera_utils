"""Tests for filenaming module"""

import datetime as dt
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from cloudpathlib import S3Path
from ulid import ULID

from libera_utils.constants import DataLevel, DataProductIdentifier, ManifestType
from libera_utils.io import filenaming


@pytest.mark.parametrize(
    ("filename", "filename_type"),
    [
        ("/some/fake/path/P1590011SOMESCIENCEAAA99030231459001.PDS", filenaming.L0Filename),
        (Path("/fake-path/P1590011SOMESCIENCEAAA99030231459001.PDS"), filenaming.L0Filename),
        ("s3://fake-bucket/P1590011SOMESCIENCEAAA99030231459001.PDS", filenaming.L0Filename),
        ("/data/LIBERA_SDC_1040_ccsds_2025_318_13_16_34", filenaming.LiberaGroundCcsdsFilename),
        (Path("/data/LIBERA_SDC_1057_ccsds_2025_318_13_53_06"), filenaming.LiberaGroundCcsdsFilename),
        ("s3://bucket/dropbox/LIBERA_SDC_11_ccsds_2025_346_13_29_47", filenaming.LiberaGroundCcsdsFilename),
        (
            "/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            filenaming.LiberaDataProductFilename,
        ),
        (
            "/some/fake/path/LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            filenaming.LiberaDataProductFilename,
        ),
        (
            "/some/foobar/path/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bsp",
            filenaming.LiberaDataProductFilename,
        ),
        (
            "/some/foobar/path/LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bc",
            filenaming.LiberaDataProductFilename,
        ),
    ],
)
def test_from_filename(filename, filename_type):
    """Test factory method that automatically figures out what type of filename it was passed"""
    assert isinstance(filenaming.AbstractValidFilename.from_file_path(filename), filename_type)


@pytest.mark.parametrize(
    ("filename", "parent_path", "expected_path"),
    [
        (
            "/ignore/this/P1590011SOMESCIENCEAAA99030231459001.PDS",
            "s3://my-bucket",
            S3Path("s3://my-bucket/PDS/0011/P1590011SOMESCIENCEAAA99030231459001.PDS"),
        ),
        (
            "/ignore/this/LIBERA_SDC_1040_ccsds_2025_318_13_16_34",
            "s3://my-l0-bucket",
            S3Path("s3://my-l0-bucket/GroundCCSDS/1040/2025/11/14/LIBERA_SDC_1040_ccsds_2025_318_13_16_34"),
        ),
        (
            "/ignore/this/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            "/absolute/local",
            Path(
                "/absolute/local/CAM/2027/01/02/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
            ),
        ),
        (
            "LIBERA_L2_CF-CAM_V3-14-159_20270102T000000_20270103T000000_R20270103T080910123456.nc",
            "/absolute/local",
            Path(
                "/absolute/local/CF-CAM/2027/01/02/"
                "LIBERA_L2_CF-CAM_V3-14-159_20270102T000000_20270103T000000_R20270103T080910123456.nc"
            ),
        ),
        (
            "ignore/relative/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bsp",
            "/absolute/local",
            Path(
                "/absolute/local/JPSS-SPK/2027/01/02/"
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bsp"
            ),
        ),
        (
            # The archive prefix follows the derived applicable date (midpoint of the time range), not utc_start
            "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R20270103T080910654321.bc",
            "/absolute/local",
            Path(
                "/absolute/local/AZROT-CK/2027/01/15/"
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R20270103T080910654321.bc"
            ),
        ),
        (
            "/ignore/this/LIBERA_L1A_SC-POS-DECODED_V3-14-159_20270102T000000_20270103T000000_R20270103T080910123456.nc",
            "s3://my-archive-bucket",
            S3Path(
                "s3://my-archive-bucket/SC-POS-DECODED/2027/01/02/"
                "LIBERA_L1A_SC-POS-DECODED_V3-14-159_20270102T000000_20270103T000000_R20270103T080910123456.nc"
            ),
        ),
        (
            "/ignore/this/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json",
            "s3://my-dropbox-bucket",
            S3Path("s3://my-dropbox-bucket/INPUT/2027/01/02/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json"),
        ),
    ],
)
def test_generate_prefixed_path(filename, parent_path, expected_path):
    """Test generating archive prefixes for filenames"""
    assert (
        filenaming.AbstractValidFilename.from_file_path(filename).generate_prefixed_path(parent_path) == expected_path
    )


@pytest.mark.parametrize(
    "filename",
    [
        "/some/fake/path/P1590011SOMESCIENCEAAA99030231459001.PDS",
        Path("/fake-path/P1590011SOMESCIENCEAAA99030231459001.PDS"),
        "s3://fake-bucket/P1590011SOMESCIENCEAAA99030231459001.PDS",
        S3Path("s3://fake-bucket/P1590011SOMESCIENCEAAA99030231459001.PDS"),
        "~/X1590011SOMESCIENCEAAA99030231459001.PDR",
        "P1590011SOMESCIENCEAAA99030231459001.PDS.XFR",
    ],
)
def test_L0Filename(filename):
    """Test L0Filename class"""
    filenaming.L0Filename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        (
            "P1590011SOMESCIENCEAAA99030231459001.PDS",
            None,
            dict(
                id_char="P",
                scid=159,
                first_apid=11,
                fill="SOMESCIENCEAAA",
                created_time=dt.datetime(1999, 1, 30, 23, 14, 59),
                numeric_id=0,
                file_number=1,
                extension="PDS",
                signal=None,
            ),
        ),
        (
            "/tmp/foo/X1590011SOMESCIENCEAAA99030231459001.PDR",
            "/tmp/foo",
            dict(
                id_char="X",
                scid=159,
                first_apid=11,
                fill="SOMESCIENCEAAA",
                created_time=dt.datetime(1999, 1, 30, 23, 14, 59),
                numeric_id=0,
                file_number=1,
                extension="PDR",
                signal=None,
            ),
        ),
        (
            "s3://bucket/P1590011SOMESCIENCEAAA99030231459001.PDS.XFR",
            "s3://bucket",
            dict(
                id_char="P",
                scid=159,
                first_apid=11,
                fill="SOMESCIENCEAAA",
                created_time=dt.datetime(1999, 1, 30, 23, 14, 59),
                numeric_id=0,
                file_number=1,
                extension="PDS",
                signal=".XFR",
            ),
        ),
    ],
)
def test_L0Filename_parts(filename, basepath, parts):
    """Test creating an L0 filename from parts"""
    fn = filenaming.L0Filename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.L0Filename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path == fn.path
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        "/some/fake/path/LIBERA_SDC_1040_ccsds_2025_318_13_16_34",
        Path("/fake-path/LIBERA_SDC_1057_ccsds_2025_318_13_53_06"),
        "s3://fake-bucket/LIBERA_SDC_1036_ccsds_2025_346_13_29_47",
        S3Path("s3://fake-bucket/LIBERA_SDC_1040_ccsds_2025_318_13_16_34"),
        "LIBERA_SDC_11_ccsds_2025_001_00_00_00",
        "LIBERA_SDC_0_ccsds_2025_001_00_00_00",
        "LIBERA_SDC_2047_ccsds_2025_001_00_00_00",
    ],
)
def test_LiberaGroundCcsdsFilename(filename):
    """Test LiberaGroundCcsdsFilename accepts demuxed ground names."""
    fn = filenaming.LiberaGroundCcsdsFilename(filename)
    assert fn.data_product_id == DataProductIdentifier.l0_ground_ccsds


def test_LiberaGroundCcsdsFilename_parts_and_archive_prefix():
    """Test round-trip from parts and the APID-scoped GroundCCSDS archive prefix."""
    filename = "LIBERA_SDC_1040_ccsds_2025_318_13_16_34"
    fn = filenaming.LiberaGroundCcsdsFilename(filename)
    parts = fn.filename_parts
    assert parts.apid == 1040
    assert parts.year == 2025
    assert parts.doy == 318
    assert parts.hour == 13
    assert parts.minute == 16
    assert parts.second == 34
    assert parts.bin_start == dt.datetime(2025, 11, 14, 13, 16, 34, tzinfo=dt.UTC)
    assert fn.archive_prefix == "GroundCCSDS/1040/2025/11/14"
    assert fn.apid == parts.apid
    assert fn.bin_start == parts.bin_start

    fn_from_parts = filenaming.LiberaGroundCcsdsFilename.from_filename_parts(
        apid=1040, year=2025, doy=318, hour=13, minute=16, second=34
    )
    assert fn_from_parts.path.name == filename
    assert fn_from_parts.archive_prefix == fn.archive_prefix


def test_LiberaGroundCcsdsFilename_archive_prefix_zero_pads_short_apid():
    """A sub-1000 APID pads to four digits so the archive sorts consistently."""
    fn = filenaming.LiberaGroundCcsdsFilename("LIBERA_SDC_11_ccsds_2025_318_13_16_34")
    assert fn.apid == 11
    assert fn.archive_prefix == "GroundCCSDS/0011/2025/11/14"


@pytest.mark.parametrize(
    "filename",
    [
        "/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        Path("/fake-path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"),
        "s3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        S3Path("s3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"),
        "~/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        "/some/fake/path/LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        Path("/fake-path/LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"),
        "s3://fake-bucket/LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        S3Path(
            "s3://fake-bucket/LIBERA_L2_UNF-RAD-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
        ),
        "~/LIBERA_L2_TOA-FLUX-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        "LIBERA_L2_COMP-FLUX_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        # Release candidate version
        "LIBERA_L2_COMP-FLUX_V3-14-159RC1_20270102T112233_20270102T122233_R20270103T080910123456.nc",
        # SPICE filenames
        "/some/fake/path/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bsp",
        Path("/fake-path/LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bc"),
        "s3://fake-bucket/LIBERA_SPICE_ELSCAN-CK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bc",
        S3Path(
            "s3://fake-bucket/LIBERA_SPICE_AZROT-CK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bc"
        ),
    ],
)
def test_LiberaDataProductFilename(filename):
    """Test LiberaDataProductFilename class"""
    filenaming.LiberaDataProductFilename(filename)


@pytest.mark.parametrize(
    ("filename", "match"),
    [
        # HDF5 is not a data product extension
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.h5",
            "failed validation against regex pattern",
        ),
        # Second-resolution revision
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            "failed validation against regex pattern",
        ),
        # Microsecond revision with seven fractional digits
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T0809101234567.nc",
            "failed validation against regex pattern",
        ),
        # Microsecond revision without the T separator
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103080910123456.nc",
            "failed validation against regex pattern",
        ),
        # Revision without the R prefix
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_20270103T080910123456.nc",
            "failed validation against regex pattern",
        ),
        # Unknown product
        (
            "LIBERA_L1B_NOT-A-PRODUCT_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            "failed validation against regex pattern",
        ),
        # Revision in month 13
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20271303T080910123456.nc",
            "is not a real UTC time",
        ),
        # utc_end on February 30
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270230T122233_R20270103T080910123456.nc",
            "is not a real UTC time",
        ),
    ],
)
def test_LiberaDataProductFilename_invalid(filename, match):
    """Test that malformed data product filenames are rejected"""
    with pytest.raises(ValueError, match=match):
        filenaming.LiberaDataProductFilename(filename)


def test_LiberaDataProductFilename_impossible_date_leaves_path_unchanged():
    """Test that assigning a name with an impossible revision raises and keeps the original path"""
    original = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
    fn = filenaming.LiberaDataProductFilename(original)
    with pytest.raises(ValueError, match="is not a real UTC time"):
        fn.path = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20271303T080910123456.nc"
    assert fn.path.name == original


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            None,
            dict(
                data_level=DataLevel.L1B,
                product_name=DataProductIdentifier.l1b_cam,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "/tmp/foo/LIBERA_L1B_RAD-4CH_V3-14-159_20250102T112233_20250102T122233_R20270103T080910123456.nc",
            "/tmp/foo",
            dict(
                data_level=DataLevel.L1B,
                product_name=DataProductIdentifier.l1b_rad,
                utc_start=dt.datetime(2025, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2025, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "s3://bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            "s3://bucket/",
            dict(
                data_level=DataLevel.L1B,
                product_name=DataProductIdentifier.l1b_cam,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            None,
            dict(
                data_level=DataLevel.L2,
                product_name=DataProductIdentifier.l2_cf_cam,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "/tmp/foo/LIBERA_L2_TOA-FLUX-CAM_V3-14-159_20250102T112233_20250102T122233_R20270103T080910123456.nc",
            "/tmp/foo",
            dict(
                data_level=DataLevel.L2,
                product_name=DataProductIdentifier.l2_toa_flux_cam,
                utc_start=dt.datetime(2025, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2025, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="V3-14-159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "s3://bucket/LIBERA_L2_CF-CAM_V3-14-159RC1_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            "s3://bucket/",
            dict(
                data_level=DataLevel.L2,
                product_name=DataProductIdentifier.l2_cf_cam,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159RC1",  # Release candidate version
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        # SPICE filename test cases
        (
            "/some/foobar/path/LIBERA_SPICE_JPSS-SPK_V3-14-159RC1_20270102T112233_20270102T122233_R20270103T080910654321.bsp",
            "/some/foobar/path",
            dict(
                data_level=DataLevel.SPICE,
                product_name=DataProductIdentifier.spice_jpss_spk,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159RC1",  # Release candidate
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 654321, tzinfo=dt.UTC),
                extension="bsp",
            ),
        ),
        (
            "s3://bucket/LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bc",
            "s3://bucket",
            dict(
                data_level=DataLevel.SPICE,
                product_name=DataProductIdentifier.spice_jpss_ck,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="V3-14-159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 654321, tzinfo=dt.UTC),
                extension="bc",
            ),
        ),
        (
            "LIBERA_SPICE_AZROT-CK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bc",
            None,
            dict(
                data_level=DataLevel.SPICE,
                product_name=DataProductIdentifier.spice_az_ck,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 654321, tzinfo=dt.UTC),
                extension="bc",
            ),
        ),
    ],
)
def test_LiberaDataProductFilename_parts(filename, basepath, parts):
    """Test creating a LiberaDataProductFilename from parts"""
    fn = filenaming.LiberaDataProductFilename(filename)
    fn_from_parts = filenaming.LiberaDataProductFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path == fn.path
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (
            dict(
                data_level="SPICE",
                product_name="JPSS-SPK",
                version="1.2.3",
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 654321, tzinfo=dt.UTC),
            ),
            "LIBERA_SPICE_JPSS-SPK_V1-2-3_20270102T112233_20270102T122233_R20270103T080910654321.bsp",
        ),
        (
            dict(
                data_level="SPICE",
                product_name="JPSS-CK",
                version="V1-2-3",
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 654321, tzinfo=dt.UTC),
            ),
            "LIBERA_SPICE_JPSS-CK_V1-2-3_20270102T112233_20270102T122233_R20270103T080910654321.bc",
        ),
        (
            dict(
                data_level="L1B",
                product_name="RAD-4CH",
                version="V1-2-3",
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                revision=dt.datetime(2027, 1, 3, 8, 9, 10, 654321, tzinfo=dt.UTC),
            ),
            "LIBERA_L1B_RAD-4CH_V1-2-3_20270102T112233_20270102T122233_R20270103T080910654321.nc",
        ),
    ],
)
def test_LiberaDataProductFilename_from_filename_parts(parts, expected):
    """Test creating filenames from parts"""
    fn = filenaming.LiberaDataProductFilename.from_filename_parts(**parts)
    assert str(fn.path) == expected


def test_round_trip_filename_parts():
    """Test that parts passed to from_filename_parts come back out of filename_parts with the right types"""
    fn = filenaming.LiberaDataProductFilename.from_filename_parts(
        product_name=DataProductIdentifier.l1b_rad,
        version="3.14.159rc1",
        utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),  # naive, assumed UTC
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
        revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
    )
    parts = fn.filename_parts
    assert parts.data_level == "L1B"
    assert parts.product_name == "RAD-4CH"
    assert parts.version == "V3-14-159RC1"
    assert parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC)
    assert parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC)
    assert parts.revision == dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC)
    assert parts.extension == "nc"
    assert fn.data_product_id == DataProductIdentifier.l1b_rad


def test_revision_default_is_current_time_per_call():
    """Test that omitting the revision uses the current time at each call, to the microsecond"""
    kwargs = dict(
        product_name="CAM",
        version="V1-0-0",
        utc_start=dt.datetime(2027, 1, 2, tzinfo=dt.UTC),
        utc_end=dt.datetime(2027, 1, 3, tzinfo=dt.UTC),
    )
    # Two calls within the same second
    first_now = dt.datetime(2027, 1, 3, 8, 9, 10, 1, tzinfo=dt.UTC)
    second_now = dt.datetime(2027, 1, 3, 8, 9, 10, 2, tzinfo=dt.UTC)
    with mock.patch.object(filenaming, "datetime", mock.Mock(wraps=dt.datetime)) as mock_datetime:
        mock_datetime.now.side_effect = [first_now, second_now]
        first = filenaming.LiberaDataProductFilename.from_filename_parts(**kwargs)
        second = filenaming.LiberaDataProductFilename.from_filename_parts(**kwargs)

    assert first.filename_parts.revision == first_now
    assert second.filename_parts.revision == second_now
    assert first.path.name.endswith("_R20270103T080910000001.nc")
    assert first != second


@pytest.mark.parametrize(
    ("revision", "expected"),
    [
        ("R20250105T112233123456", dt.datetime(2025, 1, 5, 11, 22, 33, 123456, tzinfo=dt.UTC)),
    ],
)
def test_revision_parses_to_utc_datetime(revision, expected):
    """Test that the revision parses to the creation time as a timezone-aware UTC datetime"""
    fn = filenaming.LiberaDataProductFilename(f"LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_{revision}.nc")
    assert fn.filename_parts.revision == expected


@pytest.mark.parametrize(
    "filename",
    [
        "/some/fake/path/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json",
        Path("/fake-path/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json"),
        "s3://fake-bucket/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json",
        S3Path("s3://fake-bucket/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json"),
        "~/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json",
    ],
)
def test_ManifestFilename(filename):
    """Test ManifestFilename"""
    _ = filenaming.ManifestFilename(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json.bak",
        "LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json.gz",
        "LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.jsonx",
    ],
)
def test_ManifestFilename_rejects_trailing_characters(filename):
    """The whole basename must match; a valid manifest name followed by more characters is rejected"""
    with pytest.raises(ValueError, match="failed validation against regex pattern"):
        filenaming.ManifestFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        (
            "/some/fake/path/LIBERA_INPUT_MANIFEST_01MBBXN2589RSGT2NZKDS6QM3F.json",
            "/some/fake/path",
            dict(
                manifest_type=ManifestType.INPUT,
                ulid_code=ULID.from_str("01MBBXN2589RSGT2NZKDS6QM3F"),
            ),
        ),
        (
            "s3://some/fake/path/LIBERA_OUTPUT_MANIFEST_01MBBXN2589RSGT2NZKDS6QM3F.json",
            "s3://some/fake/path",
            dict(
                manifest_type=ManifestType.OUTPUT,
                ulid_code=ULID.from_str("01MBBXN2589RSGT2NZKDS6QM3F"),
            ),
        ),
        (
            "LIBERA_INPUT_MANIFEST_01MBBXN2589RSGT2NZKDS6QM3F.json",
            None,
            dict(
                manifest_type=ManifestType.INPUT,
                ulid_code=ULID.from_str("01MBBXN2589RSGT2NZKDS6QM3F"),
            ),
        ),
    ],
)
def test_ManifestFilename_parts(filename, basepath, parts):
    """Test ManifestFilename from parts"""
    fn = filenaming.ManifestFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.ManifestFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


def test_missing_required_parts_argument():
    """Test that we get a TypeError when passing incomplete set of parts to from_filename_parts"""
    with pytest.raises(TypeError):
        filenaming.ManifestFilename.from_filename_parts(manifest_type=ManifestType.INPUT)


def test_changing_path():
    """Test ability to mess with a filename object's path"""
    name = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
    p = filenaming.LiberaDataProductFilename(name)
    # Add an S3 prefix
    p.path = S3Path("s3://bucket") / p.path
    assert isinstance(p.path, S3Path)
    # Change prefix to local
    p.path = Path("/tmp/path") / p.path.name
    assert isinstance(p.path, Path)
    # Remove basepath altogether
    p.path = p.path.name
    assert isinstance(p.path, Path)
    # Check that providing a bad value for a basepath doesn't pollute the instance's valid path
    with pytest.raises(ValueError, match="failed validation against regex pattern"):
        p.path = "/bad/prefix" + p.path.name  # The missing / will make this fail regex validation
    assert p.path.name == name


@pytest.mark.parametrize(
    ("mock_version", "version_string"), [("3.1.4", "V3-1-4"), ("1.2.3rc0", "V1-2-3RC0"), ("foo.bar", ValueError())]
)
@mock.patch("libera_utils.io.filenaming.metadata")
def test_get_current_version_str(mock_metadata, mock_version, version_string):
    """Test getting the current version string for writing a new filename"""
    mock_metadata.version.return_value = mock_version
    if isinstance(version_string, Exception):
        with pytest.raises(version_string.__class__):
            filenaming.get_current_version_str("irrelevant_since_metadata_is_mocked")
    else:
        assert filenaming.get_current_version_str("irrelevant_since_metadata_is_mocked") == version_string


def test_check_version_number_format():
    """Test version number format checking utility function"""
    # Valid versions
    assert filenaming.check_version_number_format("V1-0-0")
    assert filenaming.check_version_number_format("V12-34-56RC1")

    assert not filenaming.check_version_number_format("2.3.4")  # Semantic versioning fails
    assert not filenaming.check_version_number_format("V1.0.0")  # Dots instead of dashes
    assert not filenaming.check_version_number_format("Version1-0-0")  # Missing 'V' prefix
    assert not filenaming.check_version_number_format("V1-0")  # Too few components


def test_format_from_semantic_version():
    """Test formatting a semantic version string into the filenaming format"""
    assert filenaming.format_from_semantic_version("1.0.0") == "V1-0-0"
    assert filenaming.format_from_semantic_version("2.3.4rc1") == "V2-3-4RC1"
    assert filenaming.format_from_semantic_version("10.20.30") == "V10-20-30"

    # Add a patch 0 if not provided
    assert filenaming.format_from_semantic_version("1.0") == "V1-0-0"

    with pytest.raises(ValueError, match="Invalid semantic version string"):
        filenaming.format_from_semantic_version("v1-0-0")  # Not semantic versioning

    with pytest.raises(ValueError, match="Invalid semantic version string"):
        filenaming.format_from_semantic_version("foo.bar.baz")  # Not numeric values


def test_working_with_mocked_s3_paths(create_mock_bucket):
    """Test using our filenaming classes with mocked S3 objects"""
    bucket = create_mock_bucket()
    basepath = S3Path(f"s3://{bucket.name}/test-path")
    fn = filenaming.L0Filename.from_filename_parts(
        basepath=basepath,
        id_char="P",
        scid=987,
        first_apid=11,
        fill="FAKE",
        created_time=dt.datetime.now(dt.UTC),
        numeric_id=1,
        file_number=1,
        extension="PDS",
    )
    assert isinstance(fn.path, S3Path)


@pytest.mark.parametrize(
    ("utc_start", "utc_end", "expected_date", "should_warn"),
    [
        # Normal case - same day, 1 hour difference, no warning
        (
            dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
            dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
            dt.date(2027, 1, 2),
            False,
        ),
        # Edge case - exactly 24 hours, no warning
        (
            dt.datetime(2027, 1, 2, 0, 0, 0),
            dt.datetime(2027, 1, 3, 0, 0, 0),
            dt.date(2027, 1, 2),
            False,
        ),
        # Crossing midnight - 12 hours, no warning
        (
            dt.datetime(2027, 1, 2, 18, 0, 0),
            dt.datetime(2027, 1, 3, 6, 0, 0),
            dt.date(2027, 1, 3),
            False,
        ),
        # More than 24 hours - should warn
        (
            dt.datetime(2027, 1, 2, 0, 0, 0),
            dt.datetime(2027, 1, 3, 0, 0, 1),
            dt.date(2027, 1, 2),
            True,
        ),
        # Much longer range - should warn, midpoint calculation
        (
            dt.datetime(2027, 1, 1, 12, 0, 0),
            dt.datetime(2027, 1, 5, 12, 0, 0),
            dt.date(2027, 1, 3),
            True,
        ),
    ],
)
def test_applicable_date(utc_start, utc_end, expected_date, should_warn):
    """Test the applicable_date property with various time ranges"""
    filename = filenaming.LiberaDataProductFilename.from_filename_parts(
        product_name="CAM",
        version="V3-14-159",
        utc_start=utc_start,
        utc_end=utc_end,
        revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
    )

    if should_warn:
        with pytest.warns(UserWarning, match="Time range for filename spans more than 24 hours"):
            result = filename.applicable_date
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors to ensure no warning is issued
            result = filename.applicable_date

    assert result == expected_date


def test_applicable_date_midpoint_calculation():
    """Test that applicable_date correctly calculates the midpoint"""
    # Test specific midpoint calculation
    utc_start = dt.datetime(2027, 6, 15, 6, 0, 0)
    utc_end = dt.datetime(2027, 6, 15, 18, 0, 0)  # 12 hours later
    expected_midpoint = dt.date(2027, 6, 15)  # Should be same day

    filename = filenaming.LiberaDataProductFilename.from_filename_parts(
        data_level="L2",
        product_name="CF-CAM",
        version="V1-0-0",
        utc_start=utc_start,
        utc_end=utc_end,
        revision=dt.datetime(2027, 6, 15, 12, 0, 0),
        extension="nc",
    )

    assert filename.applicable_date == expected_midpoint


def test_applicable_date_warning_message():
    """Test that the warning message is correctly formatted"""
    filename = filenaming.LiberaDataProductFilename.from_filename_parts(
        data_level="SPICE",
        product_name="JPSS-SPK",
        version="V1-0-0",
        utc_start=dt.datetime(2027, 1, 1, 0, 0, 0),
        utc_end=dt.datetime(2027, 1, 3, 0, 0, 0),  # 48 hours
        revision=dt.datetime(2027, 1, 3, 8, 9, 10, 123456, tzinfo=dt.UTC),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = filename.applicable_date

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Time range for filename spans more than 24 hours" in str(w[0].message)


@pytest.mark.parametrize(
    ("filename", "expected_ummg_filename"),
    [
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.cmr.json",
        ),
        (
            "/tmp/foo/LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc",
            "/tmp/foo/LIBERA_L2_CF-CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.cmr.json",
        ),
        (
            "s3://bucket/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.bsp",
            "s3://bucket/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.cmr.json",
        ),
    ],
)
def test_ummg_metadata_filename(filename, expected_ummg_filename):
    """Test that the UMM-G filename property returns the correct filename"""
    fn = filenaming.LiberaDataProductFilename(filename)
    ummg_fn = fn.ummg_metadata_filename

    assert str(ummg_fn) == expected_ummg_filename
    assert isinstance(ummg_fn, type(fn.path))
    assert ummg_fn.parent == fn.path.parent
    assert ummg_fn.suffixes == [".cmr", ".json"]

    # The two regexes share a body but accept disjoint extensions
    assert filenaming.LIBERA_METADATA_PRODUCT_REGEX.match(ummg_fn.name)
    assert not filenaming.LIBERA_METADATA_PRODUCT_REGEX.match(fn.path.name)
    assert not filenaming.LIBERA_DATA_PRODUCT_REGEX.match(ummg_fn.name)
    with pytest.raises(ValueError, match="failed validation against regex pattern"):
        filenaming.LiberaDataProductFilename(ummg_fn)


def _make_mock_path(data_name: str, ummg_name: str) -> mock.MagicMock:
    """Return a mock _path whose with_suffix() yields a controlled UMM-G path name.

    Parameters
    ----------
    data_name : str
        The filename string the mock reports as its .name (the data file side).
    ummg_name : str
        The filename string the mock's with_suffix() return value reports as its .name.
    """
    mock_ummg = mock.MagicMock()
    mock_ummg.name = ummg_name
    mock_data = mock.MagicMock()
    mock_data.name = data_name
    mock_data.with_suffix.return_value = mock_ummg
    return mock_data


# A valid L1B data product filename used as the base for ummg_metadata_filename error tests
_VALID_L1B_NC = "LIBERA_L1B_RAD-4CH_V0-5-0_20251120T175950_20251120T180950_R20270103T080910123456.nc"


@pytest.mark.parametrize(
    "bad_ummg_name",
    [
        # Extension is .json only — the required .cmr part is missing
        "LIBERA_L1B_RAD-4CH_V0-5-0_20251120T175950_20251120T180950_R20270103T080910123456.json",
        # Extension has a typo: .jsno instead of .json
        "LIBERA_L1B_RAD-4CH_V0-5-0_20251120T175950_20251120T180950_R20270103T080910123456.cmr.jsno",
    ],
)
def test_ummg_metadata_filename_invalid_extension(bad_ummg_name):
    """Test that ummg_metadata_filename raises ValueError when the derived CMR filename fails regex validation."""
    fn = filenaming.LiberaDataProductFilename(_VALID_L1B_NC)
    fn._path = _make_mock_path(_VALID_L1B_NC, bad_ummg_name)
    with pytest.raises(ValueError, match="failed validation against regex pattern"):
        fn.ummg_metadata_filename


def test_ummg_metadata_filename_stem_mismatch():
    """Test that ummg_metadata_filename raises ValueError when the UMM-G filename stem doesn't match its data file."""
    fn = filenaming.LiberaDataProductFilename(_VALID_L1B_NC)
    # Structurally valid CMR filename, but product name (CAM) differs from the data file (RAD-4CH)
    mismatched_ummg_name = "LIBERA_L1B_CAM_V0-5-0_20251120T175950_20251120T180950_R20270103T080910123456.cmr.json"
    fn._path = _make_mock_path(_VALID_L1B_NC, mismatched_ummg_name)
    with pytest.raises(ValueError, match="does not match its data file path"):
        fn.ummg_metadata_filename


@pytest.mark.parametrize(
    "basename",
    [
        "LIBERA_SDC_1040_ccsds_2025_000_00_00_00",  # DOY below range
        "LIBERA_SDC_1040_ccsds_2025_999_00_00_00",  # DOY above range
        "LIBERA_SDC_1040_ccsds_2025_367_00_00_00",  # DOY above range
        "LIBERA_SDC_1040_ccsds_2025_366_00_00_00",  # DOY 366 in a common year
        "LIBERA_SDC_1040_ccsds_2025_001_24_00_00",  # hour out of range
        "LIBERA_SDC_1040_ccsds_2025_001_00_60_00",  # minute out of range
        "LIBERA_SDC_1040_ccsds_2025_001_00_00_60",  # second out of range
        "LIBERA_SDC_1040_ccsds_2025_001_99_99_99",
        "LIBERA_SDC_2048_ccsds_2025_001_00_00_00",  # APID above the 11-bit CCSDS maximum
        "LIBERA_SDC_9999_ccsds_2025_001_00_00_00",  # APID above the 11-bit CCSDS maximum
        "ccsds_2025_001_00_00_00",  # pre-demux naming, no APID
        "LIBERA_SDC_ccsds_2025_001_00_00_00",  # APID field missing
        "LIBERA_SDC_1040_ccsds_2025_001_00_00_00.bin",  # extension not part of the convention
    ],
)
def test_LiberaGroundCcsdsFilename_rejects_invalid_names(basename):
    """Invalid field values are rejected at construction, not deferred to archive_prefix.

    A name that constructs but blows up later passes ingest validation and only fails at
    staging, when the archive prefix is computed.
    """
    # Out-of-range fields and wrong shapes fail the regex; DOY 366 in a common year fails the
    # date parse; an in-regex but illegal APID fails the CCSDS range check.
    bad_name = "failed validation against regex pattern|does not exist|outside the CCSDS range"
    with pytest.raises(ValueError, match=bad_name):
        filenaming.LiberaGroundCcsdsFilename(basename)

    with pytest.raises(ValueError, match="Unable to create a valid filename"):
        filenaming.AbstractValidFilename.from_file_path(basename)


def test_LiberaGroundCcsdsFilename_accepts_leap_day_366():
    """DOY 366 is valid in a leap year."""
    fn = filenaming.LiberaGroundCcsdsFilename("LIBERA_SDC_1040_ccsds_2024_366_00_00_00")
    assert fn.archive_prefix == "GroundCCSDS/1040/2024/12/31"


# --- Hashing and equality -------------------------------------------------------------------------------------------


def test_hash_and_equality_data_product():
    """Test that data product filenames are hashable and compare equal when their paths are equal"""
    name = "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
    a = filenaming.LiberaDataProductFilename(f"/some/dir/{name}")
    b = filenaming.LiberaDataProductFilename(Path("/some/dir") / name)

    assert a == b
    assert hash(a) == hash(b)
    assert hash(a) == hash(a.path)
    assert len({a, b}) == 1
    assert {a: "value"}[b] == "value"

    # Same basename in a different directory or bucket is a different file
    local_other_dir = filenaming.LiberaDataProductFilename(f"/other/dir/{name}")
    in_s3 = filenaming.LiberaDataProductFilename(f"s3://bucket/some/dir/{name}")
    assert a != local_other_dir
    assert a != in_s3
    assert len({a, local_other_dir, in_s3}) == 3

    # Same directory, different revision
    other_revision = filenaming.LiberaDataProductFilename(
        "/some/dir/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910654321.nc"
    )
    assert a != other_revision
    assert len({a, other_revision}) == 2


def test_hash_and_equality_other_filename_types():
    """Test that L0 and manifest filenames are hashable too, and that different classes never compare equal"""
    l0_a = filenaming.L0Filename("/dir/P1590011SOMESCIENCEAAA99030231459001.PDS")
    l0_b = filenaming.L0Filename("/dir/P1590011SOMESCIENCEAAA99030231459001.PDS")
    manifest_a = filenaming.ManifestFilename("/dir/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json")
    manifest_b = filenaming.ManifestFilename("/dir/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json")
    product = filenaming.LiberaDataProductFilename(
        "/dir/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
    )

    assert l0_a == l0_b
    assert hash(l0_a) == hash(l0_b)
    assert manifest_a == manifest_b
    assert hash(manifest_a) == hash(manifest_b)
    assert len({l0_a, l0_b, manifest_a, manifest_b, product}) == 3
    assert l0_a != manifest_a
    assert l0_a != product
    assert manifest_a != product


def test_equality_with_non_filename_objects():
    """Test that comparing a filename with something that is not a filename is False rather than an error"""
    fn = filenaming.LiberaDataProductFilename(
        "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
    )
    assert fn != str(fn.path)
    assert fn != fn.path
    assert fn is not None
    assert (fn == None) is False  # noqa: E711 - exercising __eq__ with None explicitly
    assert (fn != None) is True  # noqa: E711
    assert fn != 42


def test_hash_changes_when_path_is_reassigned():
    """Test that the hash follows the path, so an instance must not be mutated while it is a set member"""
    fn = filenaming.LiberaDataProductFilename(
        "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R20270103T080910123456.nc"
    )
    before = hash(fn)
    fn.path = Path("/tmp/elsewhere") / fn.path.name
    assert hash(fn) != before
    assert hash(fn) == hash(fn.path)


# Each external ancillary sample in tests/test_data/footprint_matching, with what its upstream name (the sample
# name without "_subset") parses to: class, product, version, applicable date and archive prefix.
ANCILLARY_SAMPLES = [
    pytest.param(
        "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015_subset.nc",
        filenaming.CeresFilename,
        DataProductIdentifier.anc_ceres_noaa20_cldpix,
        "1P9test",
        dt.date(2020, 4, 10),
        "CERES-NOAA20-CLDPIX/2020/04/10",
        id="cldpix",
    ),
    pytest.param(
        "CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115_subset.nc",
        filenaming.CeresFilename,
        DataProductIdentifier.anc_ceres_noaa20_ssf,
        "alpha4",
        dt.date(2020, 4, 1),
        "CERES-NOAA20-SSF/2020/04/01",
        id="ssf",
    ),
    pytest.param(
        "AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051_subset.nc",
        filenaming.ViirsL3Filename,
        DataProductIdentifier.anc_viirs_noaa20_aod,
        "021",
        dt.date(2026, 9, 27),
        "VIIRS-NOAA20-AOD/2026/09/27",
        id="aod",
    ),
    pytest.param(
        "CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710_subset.nc",
        filenaming.ViirsL3Filename,
        DataProductIdentifier.anc_viirs_noaa20_cloud_properties,
        "011",
        dt.date(2026, 5, 27),
        "VIIRS-NOAA20-CLOUD-PROPERTIES/2026/05/27",
        id="cloud-properties",
    ),
    pytest.param(
        "VJ143C1.A2026153.002.2026161161054_subset.h5",
        filenaming.ViirsL3Filename,
        DataProductIdentifier.anc_viirs_noaa20_brdf,
        "002",
        dt.date(2026, 6, 2),
        "VIIRS-NOAA20-BRDF/2026/06/02",
        id="brdf",
    ),
    pytest.param(
        "IVCDB_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops_subset.h5",
        filenaming.ViirsSdrFilename,
        DataProductIdentifier.anc_viirs_noaa20_m4_sdr,
        "20241004192414197049",
        dt.date(2024, 10, 2),
        "VIIRS-NOAA20-M4-SDR/2024/10/02",
        id="sdr-radiance",
    ),
    pytest.param(
        "ICDBG_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops_subset.h5",
        filenaming.ViirsSdrFilename,
        DataProductIdentifier.anc_viirs_noaa20_m4_sdr,
        "20241004192414197049",
        dt.date(2024, 10, 2),
        "VIIRS-NOAA20-M4-SDR/2024/10/02",
        id="sdr-geolocation",
    ),
    pytest.param(
        "NISE_AMSR2_20260610_subset.HDFEOS",
        filenaming.NsidcNiseFilename,
        DataProductIdentifier.anc_nsidc_ice_snow,
        "1",
        dt.date(2026, 6, 10),
        "NSIDC-ICE-SNOW/2026/06/10",
        id="nise",
    ),
    pytest.param(
        "EASE2_M36km.LOCImask_land50_coast0km.964x406_subset.bin",
        filenaming.NsidcLandOceanMaskFilename,
        DataProductIdentifier.anc_nsidc_land_ocean_mask,
        "1",
        None,
        "NSIDC-LAND-OCEAN-MASK/1",
        id="land-ocean-mask",
    ),
]

# Every concrete filename class, to check that each ancillary name is accepted by exactly one of them.
ALL_FILENAME_CLASSES = (
    filenaming.L0Filename,
    filenaming.LiberaGroundCcsdsFilename,
    filenaming.LiberaDataProductFilename,
    filenaming.ManifestFilename,
    filenaming.CeresFilename,
    filenaming.ViirsL3Filename,
    filenaming.ViirsSdrFilename,
    filenaming.NsidcNiseFilename,
    filenaming.NsidcLandOceanMaskFilename,
    filenaming.Era5Filename,
)


def _accepting_classes(name):
    """The filename classes whose constructor accepts the name."""
    accepted = []
    for filename_class in ALL_FILENAME_CLASSES:
        try:
            filename_class(name)
        except ValueError:
            continue
        accepted.append(filename_class)
    return accepted


@pytest.mark.parametrize(
    ("sample", "filename_class", "product", "version", "applicable_date", "archive_prefix"), ANCILLARY_SAMPLES
)
def test_ancillary_sample_upstream_names_parse(
    fmatch_real_data_path, sample, filename_class, product, version, applicable_date, archive_prefix
):
    """Each sample's upstream name resolves to its one class, product, version, date and archive prefix."""
    sample_path = fmatch_real_data_path / sample
    assert sample_path.exists()
    upstream_path = sample_path.with_name(sample.replace("_subset", ""))

    fn = filenaming.AbstractValidFilename.from_file_path(upstream_path)

    assert type(fn) is filename_class
    assert _accepting_classes(upstream_path.name) == [filename_class]
    assert fn.data_product_id is product
    assert fn.filename_parts.version == version
    assert getattr(fn, "applicable_date", None) == applicable_date
    assert fn.archive_prefix == archive_prefix


@pytest.mark.parametrize(
    ("sample", "product"),
    [
        ("era5_single_levels_20250101T00_subset.nc", DataProductIdentifier.anc_era5_single_level),
        ("era5_pressure_levels_20250101T00_subset.nc", DataProductIdentifier.anc_era5_pressure_level),
    ],
)
def test_era5_name_built_from_sample_round_trips(fmatch_real_data_path, sample, product):
    """An ERA5 name built by the SDC scheme from a sample's own times parses back to the same parts."""
    xr = pytest.importorskip("xarray")
    with xr.open_dataset(fmatch_real_data_path / sample) as ds:
        valid_times = ds["valid_time"].values
        assert str(ds["expver"].values) == "0001"  # expver 0001 is the final ERA5 stream
    utc_start = dt.datetime.fromisoformat(str(valid_times.min())[:19])
    utc_end = dt.datetime.fromisoformat(str(valid_times.max())[:19])
    revision = dt.datetime(2026, 8, 12, 16, 11, tzinfo=dt.UTC)

    fn = filenaming.Era5Filename.from_filename_parts(
        product_name=product, version="ERA5", utc_start=utc_start, utc_end=utc_end, revision=revision
    )

    assert fn.path.name == f"ECMWF_{product}_ERA5_20250101T000000_20250101T000000_R20260812T161100000000.nc"
    assert _accepting_classes(fn.path.name) == [filenaming.Era5Filename]
    parsed = filenaming.AbstractValidFilename.from_file_path(fn.path.name)
    assert parsed.data_product_id is product
    assert parsed.filename_parts.version == "ERA5"
    assert parsed.filename_parts.utc_start == utc_start.replace(tzinfo=dt.UTC)
    assert parsed.filename_parts.revision == revision
    assert parsed.applicable_date == dt.date(2025, 1, 1)
    assert parsed.archive_prefix == f"{product}/2025/01/01"


@pytest.mark.parametrize(
    ("name", "filename_class", "expected"),
    [
        pytest.param(
            "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc",
            filenaming.CeresFilename,
            {"version": "Edition1C", "configuration_code": "101103", "applicable_date": dt.date(2023, 1, 1)},
            id="ssf-edition",
        ),
        pytest.param(
            "ICDBG_j01_d20240201_t0107372_e0113176_b63541_c20240201023927745000_oeac_ops.h5",
            filenaming.ViirsSdrFilename,
            {
                "utc_start": dt.datetime(2024, 2, 1, 1, 7, 37, 200000, tzinfo=dt.UTC),
                "utc_end": dt.datetime(2024, 2, 1, 1, 13, 17, 600000, tzinfo=dt.UTC),
                "applicable_date": dt.date(2024, 2, 1),
            },
            id="sdr-aggregate",
        ),
        pytest.param(
            "IVCDB_j01_d20241231_t2359372_e0001014_b71000_c20250101064435679611_oebc_ops.h5",
            filenaming.ViirsSdrFilename,
            {
                "utc_end": dt.datetime(2025, 1, 1, 0, 1, 1, 400000, tzinfo=dt.UTC),
                "applicable_date": dt.date(2024, 12, 31),
            },
            id="sdr-crossing-midnight",
        ),
        pytest.param(
            "EASE2_M03km.LOCImask_land50_coast0km.11568x4872.bin",
            filenaming.NsidcLandOceanMaskFilename,
            {"resolution_km": 3.0, "columns": 11568, "rows": 4872},
            id="mask-3km",
        ),
        pytest.param(
            "ECMWF_ERA5-SINGLE-LEVEL_ERA5T_20250101T000000_20250101T230000_R20251028T120000000000.nc",
            filenaming.Era5Filename,
            {"version": "ERA5T", "applicable_date": dt.date(2025, 1, 1)},
            id="era5-day",
        ),
        pytest.param(
            "ECMWF_ERA5-PRESSURE-LEVEL_ERA5_20250101T120000_20250102T110000_R20251028T120000000000.nc",
            filenaming.Era5Filename,
            {"applicable_date": dt.date(2025, 1, 1)},
            id="era5-crossing-midnight",
        ),
    ],
)
def test_ancillary_names_beyond_the_samples(name, filename_class, expected):
    """Real name forms the samples do not show parse to the expected parts."""
    fn = filenaming.AbstractValidFilename.from_file_path(name)

    assert type(fn) is filename_class
    for part, value in expected.items():
        assert getattr(fn.filename_parts, part) == value


@pytest.mark.parametrize(
    ("name", "filename_class"),
    [
        pytest.param(
            "CER_SSF_Aqua-FM3-MODIS_Edition4A_403406.2020040115.nc", filenaming.CeresFilename, id="ceres-platform"
        ),
        pytest.param(
            "CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020041024.nc", filenaming.CeresFilename, id="ceres-hour-24"
        ),
        pytest.param(
            "AERDB_D3_VIIRS_SNPP.A2026270.021.2026274001051.nc", filenaming.ViirsL3Filename, id="viirs-l3-platform"
        ),
        pytest.param(
            "AERDB_D3_VIIRS_NOAA20.A2025366.021.2026274001051.nc", filenaming.ViirsL3Filename, id="viirs-l3-doy-366"
        ),
        pytest.param(
            "AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.h5", filenaming.ViirsL3Filename, id="viirs-l3-extension"
        ),
        pytest.param(
            "IVCDB_npp_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5",
            filenaming.ViirsSdrFilename,
            id="sdr-npp",
        ),
        pytest.param(
            "IVCDB_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_dev.h5",
            filenaming.ViirsSdrFilename,
            id="sdr-domain",
        ),
        pytest.param(
            "ICDBG_j01_d20240231_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5",
            filenaming.ViirsSdrFilename,
            id="sdr-date",
        ),
        pytest.param("NISE_SSMISF18_20260610.HDFEOS", filenaming.NsidcNiseFilename, id="nise-sensor"),
        pytest.param("NISE_AMSR2_20260231.HDFEOS", filenaming.NsidcNiseFilename, id="nise-date"),
        pytest.param(
            "EASE2_T6.25km.LOCImask_land50_coast0km.5552x2160.bin",
            filenaming.NsidcLandOceanMaskFilename,
            id="mask-temperate-grid",
        ),
        pytest.param(
            "EASE2_M36km.LOCImask_land50_coast0km.964x406.png",
            filenaming.NsidcLandOceanMaskFilename,
            id="mask-browse-image",
        ),
        pytest.param(
            "ECMWF_ERA5-SINGLE-LEVEL_ERA5X_20250101T000000_20250101T230000_R20251028T120000000000.nc",
            filenaming.Era5Filename,
            id="era5-stream",
        ),
        pytest.param(
            "ECMWF_CERES-NOAA20-SSF_ERA5_20250101T000000_20250101T230000_R20251028T120000000000.nc",
            filenaming.Era5Filename,
            id="era5-product",
        ),
        pytest.param(
            "ECMWF_ERA5-SINGLE-LEVEL_ERA5_20250102T000000_20250101T000000_R20251028T120000000000.nc",
            filenaming.Era5Filename,
            id="era5-end-before-start",
        ),
    ],
)
def test_ancillary_malformed_names_rejected(name, filename_class):
    """A malformed name is rejected by its class at construction, and no class accepts it."""
    with pytest.raises(ValueError, match="failed validation|is not a valid"):
        filename_class(name)

    with pytest.raises(ValueError, match="Unable to create a valid filename"):
        filenaming.AbstractValidFilename.from_file_path(name)


def test_ancillary_impossible_date_reassignment_leaves_path_unchanged():
    """Assigning a name with an impossible date raises and keeps the original path"""
    original = "NISE_AMSR2_20260610.HDFEOS"
    fn = filenaming.NsidcNiseFilename(original)
    with pytest.raises(ValueError, match="is not a valid NsidcNiseFilename"):
        fn.path = "NISE_AMSR2_20260231.HDFEOS"
    assert fn.path.name == original
    assert fn.applicable_date == dt.date(2026, 6, 10)


def test_ancillary_parse_only_class_cannot_build_a_name():
    """Classes for provider-named files parse names and refuse to build one."""
    with pytest.raises(NotImplementedError, match="cannot build one"):
        filenaming.CeresFilename.from_filename_parts()

    with pytest.raises(NotImplementedError, match="cannot build one"):
        filenaming.CeresFilename._format_filename_parts()


def test_era5_name_revision_defaults_to_now():
    """Without a revision, the ERA5 name records the current UTC time."""
    before = dt.datetime.now(tz=dt.UTC)
    fn = filenaming.Era5Filename.from_filename_parts(
        product_name=DataProductIdentifier.anc_era5_single_level,
        version="ERA5T",
        utc_start=dt.datetime(2025, 1, 1),
        utc_end=dt.datetime(2025, 1, 1, 23),
    )
    after = dt.datetime.now(tz=dt.UTC)

    # The revision keeps microseconds, so it lands between the two clock reads.
    assert before <= fn.filename_parts.revision <= after
