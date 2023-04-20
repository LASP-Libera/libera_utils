"""Pytest plugin module for test data paths"""
# Standard
from pathlib import Path
import sys
# Installed
import pytest


# Paths to test data directories
# ------------------------------
# pylint: disable-all
@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def spice_test_data_path(test_data_path):
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return test_data_path / 'spice'


# Paths to commonly used test data files
# --------------------------------------
@pytest.fixture
def test_txt(test_data_path):
    """Path to a simple txt file"""
    return test_data_path / 'testtextfile.txt'


@pytest.fixture
def test_hdf5(test_data_path):
    """Path to a simple hdf5 file"""
    return test_data_path / 'testhdf5file.he5'


@pytest.fixture
def test_txt_gz(test_data_path):
    """Path to a gzipped version of the simple txt file"""
    return test_data_path / 'testtextfile.txt.gz'


@pytest.fixture
def test_json_manifest(test_data_path):
    """Path to test manifest file"""
    return test_data_path / 'libera_input_manifest_20220922t123456.json'


@pytest.fixture
def test_construction_record_09t00(test_data_path):
    """Path to test construction record"""
    return test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS"


@pytest.fixture
def test_construction_record_09t02(test_data_path):
    """Path to test construction record"""
    return test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.CONS"


# SPICE test data
# ---------------
@pytest.fixture
def test_lsk(spice_test_data_path):
    """Path to the LSK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'naif0012.tls'


@pytest.fixture
def test_jpss_ck(spice_test_data_path):
    """Path to the testing JPSS CK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'libera_jpss_20210408t235850_20210409t015849_vM2m1p0_r23110123456.bc'


@pytest.fixture
def test_jpss_spk(spice_test_data_path):
    """Path to the testing JPSS SPK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'libera_jpss_20210408t235850_20210409t015849_vM2m1p0_r23110123456.bsp'


@pytest.fixture
def test_de_spk(spice_test_data_path):
    """Path to the testing default ephemeris kernel stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / 'de440.bsp'


@pytest.fixture
def test_pck(spice_test_data_path):
    """Path to the testing standard text planetary constants kernel (PCK) stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / 'pck00010.tpc'


@pytest.fixture
def test_itrf93_pck(spice_test_data_path):
    """Path to the testing high precision planetary constants kernel (PCK) stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / 'earth_000101_211220_210926.bpc'
