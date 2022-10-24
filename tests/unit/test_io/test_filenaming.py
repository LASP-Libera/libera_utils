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
        '/some/fake/path/libera_cam_l2_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5',
        Path('/fake-path/libera_cam_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5'),
        's3://fake-bucket/libera_cam_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5',
        S3Path('s3://fake-bucket/libera_cam_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5'),
        '~/libera_cam_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5',
        'libera_rad_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5'
    ]
)
def test_product_filename(filename):
    """Test object"""
    filenaming.ProductFilename(filename)


@pytest.mark.parametrize(
    ("filename", "basepath", "parts"),
    [
        ('libera_cam_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5',
         None,
         dict(
             instrument='cam',
             level=filenaming.DataLevel.L1B,
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision='r27002112233',
             extension='h5'
         )),
        ('/tmp/foo/libera_rad_l2_20250102t112233_20250102t122233_vM1m2p3_r27002112233.h5',
         "/tmp/foo",
         dict(
             instrument='rad',
             level=filenaming.DataLevel.L2,
             utc_start=dt.datetime(2025, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2025, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision='r27002112233',
             extension='h5'
         )),
        ('s3://bucket/libera_cam_l1b_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5',
         None,
         dict(
             instrument='cam',
             level=filenaming.DataLevel.L1B,
             utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
             utc_end=dt.datetime(2027, 1, 2, 12, 22, 33),
             version='vM1m2p3',
             revision='r27002112233',
             extension='h5'
         )),
    ]
)
def test_product_filename_parts(filename, basepath, parts):
    fn = filenaming.ProductFilename(filename)
    assert fn.filename_parts == SimpleNamespace(**parts)
    fn_from_parts = filenaming.ProductFilename.from_filename_parts(**parts)
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
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33)
    )
    basepath = '/some/foobar/path'
    fn = filenaming.EphemerisKernelFilename(
        '/some/foobar/path/libera_jpss_20270102t112233_20270102t122233.bsp')
    assert fn.path.name == 'libera_jpss_20270102t112233_20270102t122233.bsp'
    assert fn.filename_parts.spk_object == 'jpss'
    assert fn.filename_parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33)
    assert fn.filename_parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33)
    fn_from_parts = filenaming.EphemerisKernelFilename.from_filename_parts(**parts, basepath=basepath)
    assert fn_from_parts == fn


def test_attitude_kernel_filename():
    """Test object"""
    parts = dict(
        ck_object='jpss',
        utc_start=dt.datetime(2027, 1, 2, 11, 22, 33),
        utc_end=dt.datetime(2027, 1, 2, 12, 22, 33)
    )
    basepath = '/some/foobar/path'
    fn = filenaming.AttitudeKernelFilename(
        '/some/foobar/path/libera_jpss_20270102t112233_20270102t122233.bc')
    assert fn.path.name == 'libera_jpss_20270102t112233_20270102t122233.bc'
    assert fn.filename_parts.ck_object == 'jpss'
    assert fn.filename_parts.utc_start == dt.datetime(2027, 1, 2, 11, 22, 33)
    assert fn.filename_parts.utc_end == dt.datetime(2027, 1, 2, 12, 22, 33)
    fn_from_parts = filenaming.AttitudeKernelFilename.from_filename_parts(**parts, basepath=basepath)
    assert fn_from_parts == fn


def test_changing_path():
    """Test ability to mess with a filename object's path"""
    p = filenaming.ProductFilename('libera_cam_l2_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5')
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
    assert p.path.name == 'libera_cam_l2_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5'


@mock.patch("libera_utils.io.filenaming.datetime")
def test_get_current_revision(mock_datetime):
    """Test getting the current revision for writing a new file"""
    mock_datetime.utcnow.return_value = dt.datetime(2027, 1, 2, 11, 22, 33)
    assert filenaming.get_current_revision() == 'r27002112233'


def test_format_semantic_version():
    """Test formatting a semantic version string in a way that is friendly to file naming."""
    assert filenaming.format_version('3.14.159') == 'vM3m14p159'
