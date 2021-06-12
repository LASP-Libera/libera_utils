"""Tests for filenaming module"""
from pathlib import Path

import pytest

from libera_sdp.io import filenaming


def test_ephemeris_kernel_filename():
    """Test object"""
    fn = filenaming.EphemerisKernelFilename.from_path(
        '/some/foobar/path/libera_jpss_20270102t112233_20270102t122233.bsp')
    assert fn.name == 'libera_jpss_20270102t112233_20270102t122233.bsp'
    assert fn.path == Path('libera_jpss_20270102t112233_20270102t122233.bsp')


def test_attitude_kernel_filename():
    """Test object"""
    fn = filenaming.AttitudeKernelFilename.from_path(
        '/some/foobar/path/libera_rad_20270102t112233_20270102t122233.bc')
    assert fn.name == 'libera_rad_20270102t112233_20270102t122233.bc'
    assert fn.path == Path('libera_rad_20270102t112233_20270102t122233.bc')


@pytest.mark.parametrize(
    ('isot', 'expected'),
    [
        ('2027-05-06T11:22:33.456789', '20270506t112233'),
        ('2027-05-06T11:22:33.4567891234567890123456789', '20270506t112233'),
        ('2027-05-06t11:22:33', '20270506t112233'),
        ('2027-05-06t11:22:33.456789', '20270506t112233'),
    ]
)
def test_isot_printable(isot, expected):
    """Test creation of filename-compatible timestamps from iso strings"""
    result = filenaming.isot_printable(isot)
    assert result == expected
