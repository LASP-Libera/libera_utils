"""Tests for filenaming module"""
# Standard
from types import SimpleNamespace
import datetime as dt
from pathlib import Path
from unittest import mock
# Installed
from cloudpathlib import S3Path
import pytest
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
def test_any_filename(filename, filename_type):
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
         Path('/absolute/local/AZROT/2027/01/15/LIBERA_AZROT_V3-14-159_20270101T010203_20270130T010203_R28002112233.bc'))
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
def test_l0_filename(filename):
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
             signal=".XFR"
         )),
    ]
)
def test_l0_filename_parts(filename, basepath, parts):
    """Test creating an L0 filename from parts"""
    fn = filenaming.L0Filename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.L0Filename.from_filename_parts(**parts)
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        Path('/fake-path/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        's3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        S3Path('s3://fake-bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        '~/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        'LIBERA_L1B_RAD_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'
    ]
)
def test_l1b_filename(filename):
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
        ('/tmp/foo/LIBERA_L1B_RAD_V3-14-159_20250102T112233_20250102T122233_R27002112233.nc',
         "/tmp/foo",
         dict(
             data_level='L1B',
             product_name='RAD',
             utc_start=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2025, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('s3://bucket/LIBERA_L1B_CAM_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
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
    ]
)
def test_l1b_filename_parts(filename, basepath, parts):
    """Test creating an L1b filename from parts"""
    fn = filenaming.LiberaDataProductFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.LiberaDataProductFilename.from_filename_parts(**parts)
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        Path('/fake-path/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        's3://fake-bucket/LIBERA_L2_CLOUD-FRACTION_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        S3Path('s3://fake-bucket/LIBERA_L2_UNFILTERED-RADIANCE_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'),
        '~/LIBERA_L2_TOA-FLUX_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
        'LIBERA_L2_SFC-FLUX_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc'
    ]
)
def test_l2_filename(filename):
    """Test LiberaDataProductFilename class"""
    filenaming.LiberaDataProductFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
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
        ('s3://bucket/LIBERA_L2_ABCD-EFGH-1234_V3-14-159_20270102T112233_20270102T122233_R27002112233.nc',
         None,
         dict(
             data_level='L2',
             product_name='ABCD-EFGH-1234',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='V3-14-159',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
    ]
)
def test_l2_filename_parts(filename, basepath, parts):
    fn = filenaming.LiberaDataProductFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.LiberaDataProductFilename.from_filename_parts(**parts)
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/LIBERA_INPUT_MANIFEST_20270102T122233.json',
        Path('/fake-path/LIBERA_INPUT_MANIFEST_20270102T122233.json'),
        's3://fake-bucket/LIBERA_INPUT_MANIFEST_20270102T122233.json',
        S3Path('s3://fake-bucket/LIBERA_INPUT_MANIFEST_20270102T122233.json'),
        '~/LIBERA_INPUT_MANIFEST_20270102T122233.json',
    ]
)
def test_manifest_filename(filename):
    """Test object"""
    fn = filenaming.ManifestFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('/some/fake/path/LIBERA_INPUT_MANIFEST_20270102T122233.json',
         '/some/fake/path',
         dict(
             manifest_type=filenaming.ManifestType.INPUT,
             created_time=dt.datetime(2027, 1, 2, 12, 22, 33)
         )),
        ('s3://some/fake/path/LIBERA_OUTPUT_MANIFEST_20270102T122233.json',
         None,
         dict(
             manifest_type=filenaming.ManifestType.OUTPUT,
             created_time=dt.datetime(2027, 1, 2, 12, 22, 33)
         )),
        ('LIBERA_INPUT_MANIFEST_20240102T122233.json',
         None,
         dict(
             manifest_type=filenaming.ManifestType.INPUT,
             created_time=dt.datetime(2024, 1, 2, 12, 22, 33)
         )),
    ]
)
def test_manifest_filename_parts(filename, basepath, parts):
    fn = filenaming.ManifestFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.ManifestFilename.from_filename_parts(**parts, basepath=basepath)
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


def test_ephemeris_kernel_filename():
    """Test object"""
    parts = dict(
        spk_object='JPSS',
        utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
        version="V3-14-159",
        revision=dt.datetime(2028, 1, 2, 11, 22, 33)
    )
    basepath = '/some/foobar/path'
    fn = filenaming.EphemerisKernelFilename(
        '/some/foobar/path/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp')
    assert fn.path.name == 'LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bsp'
    assert fn.filename_parts.spk_object == 'JPSS'
    assert fn.filename_parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33)
    assert fn.filename_parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33)
    assert fn.filename_parts.version == "V3-14-159"
    assert fn.filename_parts.revision == dt.datetime(2028, 1, 2, 11, 22, 33)
    fn_from_parts = filenaming.EphemerisKernelFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn


def test_attitude_kernel_filename():
    """Test object"""
    parts = dict(
        ck_object='JPSS',
        utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
        version="V3-14-159",
        revision=dt.datetime(2028, 1, 2, 11, 22, 33)
    )
    basepath = '/some/foobar/path'
    fn = filenaming.AttitudeKernelFilename(
        '/some/foobar/path/LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc')
    assert fn.path.name == 'LIBERA_JPSS_V3-14-159_20270102T112233_20270102T122233_R28002112233.bc'
    assert fn.filename_parts.ck_object == 'JPSS'
    assert fn.filename_parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33)
    assert fn.filename_parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33)
    assert fn.filename_parts.revision == dt.datetime(2028, 1, 2, 11, 22, 33)
    fn_from_parts = filenaming.AttitudeKernelFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn


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
    mock_datetime.utcnow.return_value = dt.datetime(2027, 1, 2, 11, 22, 33)
    assert filenaming.get_current_revision_str() == 'R27002112233'


@mock.patch("libera_utils.io.filenaming.metadata")
def test_get_current_version_str(mock_metadata):
    """Test getting the current version string for writing a new filename"""
    mock_metadata.version.return_value = "3.1.4"
    assert filenaming.get_current_version_str('irrelevant_since_metadata_is_mocked') == "V3-1-4"

    mock_metadata.version.return_value = "1.0"  # Not a full semantic version string
    with pytest.raises(ValueError):
        filenaming.get_current_version_str('irrelevant_since_metadata_is_mocked')
