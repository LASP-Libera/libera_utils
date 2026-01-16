"""Tests for filenaming module"""

import datetime as dt
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from cloudpathlib import S3Path
from packaging.version import InvalidVersion
from ulid import ULID

from libera_utils.constants import DataLevel, DataProductIdentifier, ManifestType
from libera_utils.io import filenaming


@pytest.mark.parametrize(
    ("filename", "filename_type"),
    [
        ("/some/fake/path/P1590011SOMESCIENCEAAA99030231459001.PDS", filenaming.L0Filename),
        (Path("/fake-path/P1590011SOMESCIENCEAAA99030231459001.PDS"), filenaming.L0Filename),
        ("s3://fake-bucket/P1590011SOMESCIENCEAAA99030231459001.PDS", filenaming.L0Filename),
        (
            "/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            filenaming.LiberaDataProductFilename,
        ),
        (
            "/some/fake/path/LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            filenaming.LiberaDataProductFilename,
        ),
        (
            "/some/foobar/path/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
            filenaming.LiberaDataProductFilename,
        ),
        (
            "/some/foobar/path/LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
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
            "/ignore/this/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            "/absolute/local",
            Path(
                "/absolute/local/CAM/2027/01/02/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"
            ),
        ),
        (
            "LIBERA_L2_CF-RAD_V3-14-159_20270102T000000_20270103T000000_R27002112233.nc",
            "/absolute/local",
            Path(
                "/absolute/local/CF-RAD/2027/01/02/"
                "LIBERA_L2_CF-RAD_V3-14-159_20270102T000000_20270103T000000_R27002112233.nc"
            ),
        ),
        (
            "ignore/relative/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
            "/absolute/local",
            Path(
                "/absolute/local/JPSS-SPK/2027/01/02/"
                "LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp"
            ),
        ),
        (
            "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc",
            "/absolute/local",
            Path(
                "/absolute/local/AZROT-CK/2027/01/15/"
                "LIBERA_SPICE_AZROT-CK_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc"
            ),
        ),
        (
            "/ignore/this/LIBERA_L1A_SC-POS-DECODED_V3-14-159_20270102T000000_20270103T000000_R27002112233.nc",
            "s3://my-archive-bucket",
            S3Path(
                "s3://my-archive-bucket/SC-POS-DECODED/2027/01/02/LIBERA_L1A_SC-POS-DECODED_V3-14-159_20270102T000000_20270103T000000_R27002112233.nc"
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
        "/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        Path("/fake-path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        "s3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        S3Path("s3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        "~/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "/some/fake/path/LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        Path("/fake-path/LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        "s3://fake-bucket/LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        S3Path("s3://fake-bucket/LIBERA_L2_UNF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc"),
        "~/LIBERA_L2_SSW-TOA-FLUXES-ERBE_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "LIBERA_L2_SFC-FLUXES_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
        "LIBERA_L2_SFC-FLUXES_V3-14-159RC1_20270102T112233_20270102T122233_R27002112233.nc",  # Release candidate version
        # SPICE filenames
        "/some/fake/path/LIBERA_SPICE_JPSS-SPK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp",
        Path("/fake-path/LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc"),
        "s3://fake-bucket/LIBERA_SPICE_ELSCAN-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
        S3Path("s3://fake-bucket/LIBERA_SPICE_AZROT-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc"),
    ],
)
def test_LiberaDataProductFilename(filename):
    """Test LiberaDataProductFilename class"""
    filenaming.LiberaDataProductFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        (
            "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            None,
            dict(
                data_level=DataLevel.L1B,
                product_name=DataProductIdentifier.l1b_cam,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "/tmp/foo/LIBERA_L1B_RAD-4CH_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc",
            "/tmp/foo",
            dict(
                data_level=DataLevel.L1B,
                product_name=DataProductIdentifier.l1b_rad,
                utc_start=dt.datetime(2025, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2025, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "s3://bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            "s3://bucket/",
            dict(
                data_level=DataLevel.L1B,
                product_name=DataProductIdentifier.l1b_cam,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "LIBERA_L2_CF-RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc",
            None,
            dict(
                data_level=DataLevel.L2,
                product_name=DataProductIdentifier.l2_cf_rad,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "/tmp/foo/LIBERA_L2_SSW-TOA-FLUXES-ERBE_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc",
            "/tmp/foo",
            dict(
                data_level=DataLevel.L2,
                product_name=DataProductIdentifier.l2_ssw_toa_erbe,
                utc_start=dt.datetime(2025, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2025, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="V3-14-159",
                revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        (
            "s3://bucket/LIBERA_L2_CF-RAD_V3-14-159RC1_20270102T112233_20270102T122233_R27002112233.nc",
            "s3://bucket/",
            dict(
                data_level=DataLevel.L2,
                product_name=DataProductIdentifier.l2_cf_rad,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159RC1",  # Release candidate version
                revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="nc",
            ),
        ),
        # SPICE filename test cases
        (
            "/some/foobar/path/LIBERA_SPICE_JPSS-SPK_V3-14-159RC1_20270102T112233_20270102T122233_R28002112233.bsp",
            "/some/foobar/path",
            dict(
                data_level=DataLevel.SPICE,
                product_name=DataProductIdentifier.spice_jpss_spk,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159RC1",  # Release candidate
                revision=dt.datetime(2028, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="bsp",
            ),
        ),
        (
            "s3://bucket/LIBERA_SPICE_JPSS-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
            "s3://bucket",
            dict(
                data_level=DataLevel.SPICE,
                product_name=DataProductIdentifier.spice_jpss_ck,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="V3-14-159",
                revision=dt.datetime(2028, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                extension="bc",
            ),
        ),
        (
            "LIBERA_SPICE_AZROT-CK_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc",
            None,
            dict(
                data_level=DataLevel.SPICE,
                product_name=DataProductIdentifier.spice_az_ck,
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                version="3.14.159",
                revision=dt.datetime(2028, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
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
                revision=dt.datetime(2028, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
            ),
            "LIBERA_SPICE_JPSS-SPK_V1-2-3_20270102T112233_20270102T122233_R28002112233.bsp",
        ),
        (
            dict(
                data_level="SPICE",
                product_name="JPSS-CK",
                version="V1-2-3",
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                revision=dt.datetime(2028, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
            ),
            "LIBERA_SPICE_JPSS-CK_V1-2-3_20270102T112233_20270102T122233_R28002112233.bc",
        ),
        (
            dict(
                data_level="L1B",
                product_name="RAD-4CH",
                version="V1-2-3",
                utc_start=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
                utc_end=dt.datetime(2027, 1, 2, 12, 22, 33, tzinfo=dt.UTC),
                revision=dt.datetime(2028, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
            ),
            "LIBERA_L1B_RAD-4CH_V1-2-3_20270102T112233_20270102T122233_R28002112233.nc",
        ),
    ],
)
def test_LiberaDataProductFilename_from_filename_parts(parts, expected):
    """Test creating filenames from parts"""
    fn = filenaming.LiberaDataProductFilename.from_filename_parts(**parts)
    assert str(fn.path) == expected


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
    p = filenaming.LiberaDataProductFilename("LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.h5")
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
    assert p.path.name == "LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.h5"


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

    with pytest.raises(InvalidVersion):
        filenaming.format_from_semantic_version("v1-0-0")  # Not semantic versioning

    with pytest.raises(InvalidVersion):
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
        data_level="L1B",
        product_name="CAM",
        version="V3-14-159",
        utc_start=utc_start,
        utc_end=utc_end,
        revision=dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.UTC),
        extension="nc",
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
        product_name="CF-RAD",
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
        revision=dt.datetime(2027, 1, 2, 0, 0, 0),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = filename.applicable_date

        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "Time range for filename spans more than 24 hours" in str(w[0].message)
