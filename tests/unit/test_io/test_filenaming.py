"""Tests for filenaming module"""
# Standard
from types import SimpleNamespace
import datetime as dt
from pathlib import Path
from unittest import mock
# Installed
from cloudpathlib import S3Path
import pytest
from ulid import ULID
# Local
from libera_utils.io import filenaming


@pytest.mark.parametrize(
    ("filename", "filename_type"),
    [
        ('/some/fake/path/P1590006SOMESCIENCEAAA99030231459001.PDS', filenaming.L0Filename),
        (Path('/fake-path/P1590006SOMESCIENCEAAA99030231459001.PDS'), filenaming.L0Filename),
        ('s3://fake-bucket/P1590006SOMESCIENCEAAA99030231459001.PDS', filenaming.L0Filename),
        ('/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc', filenaming.LiberaDataProductFilename),
        ('/some/fake/path/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc', filenaming.LiberaDataProductFilename),
        ('/some/foobar/path/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp', filenaming.EphemerisKernelFilename),
        ('/some/foobar/path/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc', filenaming.AttitudeKernelFilename)
    ]
)
def test_AnyFilename(filename, filename_type):
    """Test polymorphic class that automatically figures out what type of filename it was passed"""
    assert isinstance(filenaming.AnyFilename(filename), filename_type)


@pytest.mark.parametrize(
    ("filename", "parent_path", "expected_path"),
    [
        ('/ignore/this/P1590006SOMESCIENCEAAA99030231459001.PDS', 's3://my-bucket',
         S3Path('s3://my-bucket/PDS/0006/P1590006SOMESCIENCEAAA99030231459001.PDS')),
        ('/ignore/this/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc', '/absolute/local',
         Path('/absolute/local/CAM/2027/01/02/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc')),
        ('LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T000000_20270103T000000_R27002112233.nc', '/absolute/local',
         Path('/absolute/local/CLOUD-FRACTION/2027/01/02/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T000000_20270103T000000_R27002112233.nc')),
        ('ignore/relative/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp', '/absolute/local',
         Path('/absolute/local/JPSS/2027/01/02/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp')),
        ('LIBERA_AZROT_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc', '/absolute/local',
         Path('/absolute/local/AZROT/2027/01/15/LIBERA_AZROT_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc')),
        ('/ignore/this/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json', 's3://my-dropbox-bucket',
         S3Path('s3://my-dropbox-bucket/INPUT/2027/01/02/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json')),

    ]
)
def test_generate_prefixed_path(filename, parent_path, expected_path):
    """Test generating archive prefixes for filenames"""
    assert filenaming.AnyFilename(filename).generate_prefixed_path(parent_path) == expected_path


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/P1590006SOMESCIENCEAAA99030231459001.PDS',
        Path('/fake-path/P1590006SOMESCIENCEAAA99030231459001.PDS'),
        's3://fake-bucket/P1590006SOMESCIENCEAAA99030231459001.PDS',
        S3Path('s3://fake-bucket/P1590006SOMESCIENCEAAA99030231459001.PDS'),
        '~/X1590006SOMESCIENCEAAA99030231459001.PDR',
        'P1590006SOMESCIENCEAAA99030231459001.PDS.XFR'
    ]
)
def test_L0Filename(filename):
    """Test L0Filename class"""
    filenaming.L0Filename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('P1590006SOMESCIENCEAAA99030231459001.PDS',
         None,
         dict(
             id_char='P',
             scid=159,
             first_apid=6,
             fill="SOMESCIENCEAAA",
             created_time=dt.datetime(1999, 1, 30, 23, 14, 59),
             numeric_id=0,
             file_number=1,
             extension='PDS',
             signal=None
         )),
        ('/tmp/foo/X1590006SOMESCIENCEAAA99030231459001.PDR',
         "/tmp/foo",
         dict(
             id_char='X',
             scid=159,
             first_apid=6,
             fill="SOMESCIENCEAAA",
             created_time=dt.datetime(1999, 1, 30, 23, 14, 59),
             numeric_id=0,
             file_number=1,
             extension='PDR',
             signal=None
         )),
        ('s3://bucket/P1590006SOMESCIENCEAAA99030231459001.PDS.XFR',
         "s3://bucket",
         dict(
             id_char='P',
             scid=159,
             first_apid=6,
             fill="SOMESCIENCEAAA",
             created_time=dt.datetime(1999, 1, 30, 23, 14, 59),
             numeric_id=0,
             file_number=1,
             extension='PDS',
             signal=".XFR"
         )),
    ]
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
        '/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        Path('/fake-path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        's3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        S3Path('s3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        '~/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        'LIBERA_L1B_RAD-4CH_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        '/some/fake/path/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        Path('/fake-path/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        's3://fake-bucket/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        S3Path(
            's3://fake-bucket/LIBERA_L2_UNFILTERED-RADIANCE_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        '~/LIBERA_L2_TOA-FLUX_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        'LIBERA_L2_SFC-FLUX_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        'LIBERA_L2_SFC-FLUX_V3-14-159RC1_20270102T112233_20270102T122233_R27002112233.nc'  # Release candidate version
    ]
)
def test_LiberaDataProductFilename(filename):
    """Test LiberaDataProductFilename class"""
    filenaming.LiberaDataProductFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
         None,
         dict(
             data_level='L1B',
             product_name='CAM',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('/tmp/foo/LIBERA_L1B_RAD-4CH_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc',
         "/tmp/foo",
         dict(
             data_level='L1B',
             product_name='RAD-4CH',
             utc_start=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2025, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('s3://bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
         "s3://bucket/",
         dict(
             data_level='L1B',
             product_name='CAM',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
         None,
         dict(
             data_level='L2',
             product_name='CLOUD-FRACTION',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('/tmp/foo/LIBERA_L2_TOA-FLUX_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc',
         "/tmp/foo",
         dict(
             data_level='L2',
             product_name='TOA-FLUX',
             utc_start=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2025, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('s3://bucket/LIBERA_L2_ABCD-EFGH-1234_V3-14-159RC1_20270102T112233_20270102T122233_R27002112233.nc',
         "s3://bucket/",
         dict(
             data_level='L2',
             product_name='ABCD-EFGH-1234',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='V3-14-159RC1',  # Release candidate version
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
    ]
)
def test_LiberaDataProductFilename_parts(filename, basepath, parts):
    """Test creating a LiberaDataProductFilename from parts"""
    fn = filenaming.LiberaDataProductFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.LiberaDataProductFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path == fn.path
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json',
        Path('/fake-path/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json'),
        's3://fake-bucket/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json',
        S3Path('s3://fake-bucket/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json'),
        '~/LIBERA_INPUT_MANIFEST_01MBAK5DC06HX46P3PG0M6HJR0.json',
    ]
)
def test_ManifestFilename(filename):
    """Test ManifestFilename"""
    fn = filenaming.ManifestFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('/some/fake/path/LIBERA_INPUT_MANIFEST_01MBBXN2589RSGT2NZKDS6QM3F.json',
         '/some/fake/path',
         dict(
             manifest_type=filenaming.ManifestType.INPUT,
             ulid_code=ULID.from_str("01MBBXN2589RSGT2NZKDS6QM3F")
         )),
        ('s3://some/fake/path/LIBERA_OUTPUT_MANIFEST_01MBBXN2589RSGT2NZKDS6QM3F.json',
         "s3://some/fake/path",
         dict(
             manifest_type=filenaming.ManifestType.OUTPUT,
             ulid_code=ULID.from_str("01MBBXN2589RSGT2NZKDS6QM3F")
         )),
        ('LIBERA_INPUT_MANIFEST_01MBBXN2589RSGT2NZKDS6QM3F.json',
         None,
         dict(
             manifest_type=filenaming.ManifestType.INPUT,
             ulid_code=ULID.from_str("01MBBXN2589RSGT2NZKDS6QM3F")
         )),
    ]
)
def test_ManifestFilename_parts(filename, basepath, parts):
    """Test ManifestFilename from parts"""
    fn = filenaming.ManifestFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.ManifestFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('/some/foobar/path/LIBERA_JPSS_V3-14-159RC1_20270102T112233_20270102T122233_R28002112233.bsp',
         '/some/foobar/path',
         dict(
             spk_object='JPSS',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version="V3-14-159RC1",  # Release candidate
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
        ('s3://bucket/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp',
         's3://bucket',
         dict(
             spk_object='JPSS',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version="V3-14-159",
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
    ]
)
def test_EphemerisKernelFilename(filename, basepath, parts):
    """Test EphemerisKernelFilename"""
    fn = filenaming.EphemerisKernelFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.EphemerisKernelFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('/some/foobar/path/LIBERA_JPSS_V3-14-159RC1_20270102T112233_20270102T122233_R28002112233.bc',
         '/some/foobar/path',
         dict(
             ck_object='JPSS',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version="V3-14-159RC1",
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
        ('s3://bucket/LIBERA_ELSCAN_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc',
         's3://bucket',
         dict(
             ck_object='ELSCAN',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version="V3-14-159",
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
        ('LIBERA_AZROT_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc',
         None,
         dict(
             ck_object='AZROT',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version="V3-14-159",
             revision=dt.datetime(2028, 1, 2, 11, 22, 33)
         )),
    ]
)
def test_AttitudeKernelFilename(filename, basepath, parts):
    """Test AttitudeKernelFilename"""
    fn = filenaming.AttitudeKernelFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.AttitudeKernelFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


def test_changing_path():
    """Test ability to mess with a filename object's path"""
    p = filenaming.LiberaDataProductFilename('LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.h5')
    # Add an S3 prefix
    p.path = S3Path('s3://bucket') / p.path
    assert isinstance(p.path, S3Path)
    # Change prefix to local
    p.path = Path('/tmp/path') / p.path.name
    assert isinstance(p.path, Path)
    # Remove basepath altogether
    p.path = p.path.name
    assert isinstance(p.path, Path)
    # Check that providing a bad value for a basepath doesn't pollute the instance's valid path
    with pytest.raises(ValueError):
        p.path = '/bad/prefix' + p.path.name  # The missing / will make this fail regex validation
    assert p.path.name == 'LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.h5'


@mock.patch("libera_utils.io.filenaming.datetime")
def test_get_current_revision_str(mock_datetime):
    """Test getting the current revision string for writing a new filename"""
    mock_datetime.now.return_value = dt.datetime(2027, 1, 2, 11, 22, 33, tzinfo=dt.timezone.utc)
    assert filenaming.get_current_revision_str() == 'R27002112233'


@pytest.mark.parametrize(
    ("mock_version", "version_string"),
    [
        ("3.1.4", "V3-1-4"),
        ("1.2.3rc0", "V1-2-3RC0"),
        ("1.0", ValueError())
    ]
)
@mock.patch("libera_utils.io.filenaming.metadata")
def test_get_current_version_str(mock_metadata, mock_version, version_string):
    """Test getting the current version string for writing a new filename"""
    mock_metadata.version.return_value = mock_version
    if isinstance(version_string, Exception):
        with pytest.raises(version_string.__class__):
            filenaming.get_current_version_str('irrelevant_since_metadata_is_mocked')
    else:
        assert filenaming.get_current_version_str('irrelevant_since_metadata_is_mocked') == version_string
