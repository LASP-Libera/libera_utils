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
