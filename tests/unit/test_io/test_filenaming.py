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
        '/some/fake/path/libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
        Path('/fake-path/libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc'),
        's3://fake-bucket/libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
        S3Path('s3://fake-bucket/libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc'),
        '~/libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
        'libera_l1b_rad_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc'
    ]
)
def test_l1b_filename(filename):
    """Test L1bFilename class"""
    filenaming.L1bFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
         None,
         dict(
             instrument='cam',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('/tmp/foo/libera_l1b_rad_20250102t112233_20250102t122233_vM1m2p3_r27002112233.nc',
         "/tmp/foo",
         dict(
             instrument='rad',
             utc_start=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2025, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('s3://bucket/libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
         None,
         dict(
             instrument='cam',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
    ]
)
def test_l1b_filename_parts(filename, basepath, parts):
    """Test creating an L1b filename from parts"""
    fn = filenaming.L1bFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.L1bFilename.from_filename_parts(**parts)
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/libera_l2_cloud-fraction_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
        Path('/fake-path/libera_l2_cloud-fraction_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc'),
        's3://fake-bucket/libera_l2_cloud-fraction_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
        S3Path('s3://fake-bucket/libera_l2_unfiltered-radiance_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc'),
        '~/libera_l2_toa-flux_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
        'libera_l2_sfc-flux_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc'
    ]
)
def test_l2_filename(filename):
    """Test L2Filename class"""
    filenaming.L2Filename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('libera_l2_cloud-fraction_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
         None,
         dict(
             product_name='cloud-fraction',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('/tmp/foo/libera_l2_toa-flux_20250102t112233_20250102t122233_vM1m2p3_r27002112233.nc',
         "/tmp/foo",
         dict(
             product_name='toa-flux',
             utc_start=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2025, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
        ('s3://bucket/libera_l2_abcd-EFGH-1234_20270102t112233_20270102t122233_vM1m2p3_r27002112233.nc',
         None,
         dict(
             product_name='abcd-EFGH-1234',
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision=dt.datetime(2027, 1, 2, 11, 22, 33),
             extension="nc"
         )),
    ]
)
def test_l2_filename_parts(filename, basepath, parts):
    fn = filenaming.L2Filename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.L2Filename.from_filename_parts(**parts)
    assert fn_from_parts.path.name == fn.path.name
    assert fn_from_parts.filename_parts == fn.filename_parts


@pytest.mark.parametrize(
    "filename",
    [
        '/some/fake/path/libera_input_manifest_20270102t122233.json',
        Path('/fake-path/libera_input_manifest_20270102t122233.json'),
        's3://fake-bucket/libera_input_manifest_20270102t122233.json',
        S3Path('s3://fake-bucket/libera_input_manifest_20270102t122233.json'),
        '~/libera_input_manifest_20270102t122233.json',
    ]
)
def test_manifest_filename(filename):
    """Test object"""
    fn = filenaming.ManifestFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('/some/fake/path/libera_input_manifest_20270102t122233.json',
         '/some/fake/path',
         dict(
             manifest_type=filenaming.ManifestType.INPUT,
             created_time=dt.datetime(2027, 1, 2, 12, 22, 33)
         )),
        ('s3://some/fake/path/libera_output_manifest_20270102t122233.json',
         None,
         dict(
             manifest_type=filenaming.ManifestType.OUTPUT,
             created_time=dt.datetime(2027, 1, 2, 12, 22, 33)
         )),
        ('libera_input_manifest_20240102t122233.json',
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
        spk_object='jpss',
        utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
        revision=dt.datetime(2028, 1, 2, 11, 22, 33)
    )
    basepath = '/some/foobar/path'
    fn = filenaming.EphemerisKernelFilename(
        '/some/foobar/path/libera_jpss_20270102t112233_20270102t122233_r28002112233.bsp')
    assert fn.path.name == 'libera_jpss_20270102t112233_20270102t122233_r28002112233.bsp'
    assert fn.filename_parts.spk_object == 'jpss'
    assert fn.filename_parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33)
    assert fn.filename_parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33)
    assert fn.filename_parts.revision == dt.datetime(2028, 1, 2, 11, 22, 33)
    fn_from_parts = filenaming.EphemerisKernelFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn


def test_attitude_kernel_filename():
    """Test object"""
    parts = dict(
        ck_object='jpss',
        utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
        revision=dt.datetime(2028, 1, 2, 11, 22, 33)
    )
    basepath = '/some/foobar/path'
    fn = filenaming.AttitudeKernelFilename(
        '/some/foobar/path/libera_jpss_20270102t112233_20270102t122233_r28002112233.bc')
    assert fn.path.name == 'libera_jpss_20270102t112233_20270102t122233_r28002112233.bc'
    assert fn.filename_parts.ck_object == 'jpss'
    assert fn.filename_parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33)
    assert fn.filename_parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33)
    assert fn.filename_parts.revision == dt.datetime(2028, 1, 2, 11, 22, 33)
    fn_from_parts = filenaming.AttitudeKernelFilename.from_filename_parts(basepath=basepath, **parts)
    assert fn_from_parts == fn


def test_changing_path():
    """Test ability to mess with a filename object's path"""
    p = filenaming.L1bFilename('libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5')
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
    assert p.path.name == 'libera_l1b_cam_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5'


@mock.patch("libera_utils.io.filenaming.datetime")
def test_get_current_revision_str(mock_datetime):
    """Test getting the current revision string for writing a new filename"""
    mock_datetime.utcnow.return_value = dt.datetime(2027, 1, 2, 11, 22, 33)
    assert filenaming.get_current_revision_str() == 'r27002112233'


@mock.patch("libera_utils.io.filenaming.metadata")
def test_get_current_version_str(mock_metadata):
    """Test getting the current version string for writing a new filename"""
    mock_metadata.version.return_value = "3.1.4"
    assert filenaming.get_current_version_str('irrelevant_since_metadata_is_mocked') == "vM3m1p4"

    mock_metadata.version.return_value = "1.0"  # Not a full semantic version string
    with pytest.raises(ValueError):
        filenaming.get_current_version_str('irrelevant_since_metadata_is_mocked')
