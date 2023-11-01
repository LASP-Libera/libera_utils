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
    return test_data_path / 'LIBERA_INPUT_MANIFEST_20220922T123456.json'


@pytest.fixture
def test_construction_record_1(test_data_path):
    """Path to test construction record"""
    return test_data_path / "P1590011AAAAAAAAAAAAAT21099051420500.PDS"


@pytest.fixture
def test_pds_file_1(test_data_path):
    """Path to the test PDS file associated with construction record 1"""
    return test_data_path / "P1590011AAAAAAAAAAAAAT21099051420501.PDS"


@pytest.fixture
def test_construction_record_2(test_data_path):
    """Path to test construction record"""
    return test_data_path / "P1590011AAAAAAAAAAAAAT21099065436900.PDS"


@pytest.fixture
def test_pds_file_2(test_data_path):
    """Path to the test PDS file associated with construction record 2"""
    return test_data_path / "P1590011AAAAAAAAAAAAAT21099065436901.PDS"


@pytest.fixture
def test_construction_record_3(test_data_path):
    """Path to test construction record"""
    return test_data_path / "P1590011AAAAAAAAAAAAAT21099091211400.PDS"


@pytest.fixture
def test_pds_file_3(test_data_path):
    """Path to the test PDS file associated with construction record 3"""
    return test_data_path / "P1590011AAAAAAAAAAAAAT21099091211401.PDS"


# SPICE test data
# ---------------
@pytest.fixture
def test_lsk(spice_test_data_path):
    """Path to the LSK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'naif0012.tls'


@pytest.fixture
def test_jpss_ck(spice_test_data_path):
    """Path to the testing JPSS CK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'LIBERA_JPSS_V2-1-0_20210408T235850_20210409T015849_R23110123456.bc'


@pytest.fixture
def test_jpss_spk(spice_test_data_path):
    """Path to the testing JPSS SPK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'LIBERA_JPSS_V2-1-0_20210408T235850_20210409T015849_R23110123456.bsp'


@pytest.fixture
def test_de_spk(spice_test_data_path):
    """Path to the testing default ephemeris kernel stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / 'de440s.bsp'


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
