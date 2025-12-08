"""Pytest plugin module for test data paths"""

import sys
from pathlib import Path

import pytest


# Paths to test data directories
# ------------------------------
@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(str(sys.modules[__name__.split(".")[0]].__file__)).parent / "test_data"


@pytest.fixture
def spice_test_data_path(test_data_path):
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_utils/data directory should be sourced from there.
    """
    return test_data_path / "spice"


@pytest.fixture
def product_definitions_test_data_path(test_data_path):
    """Returns the product_definitions subdirectory of the test_data directory

    This directory contains yml product/project definition files used for proper NetCDF4 file creation
    """
    return test_data_path / "product_definitions"


# Paths to commonly used test data files
# --------------------------------------
@pytest.fixture
def test_txt(test_data_path):
    """Path to a simple txt file"""
    return test_data_path / "testtextfile.txt"


@pytest.fixture
def test_hdf5(test_data_path):
    """Path to a simple hdf5 file"""
    return test_data_path / "testhdf5file.he5"


@pytest.fixture
def test_txt_gz(test_data_path):
    """Path to a gzipped version of the simple txt file"""
    return test_data_path / "testtextfile.txt.gz"


# JPSS-1 PDS Data
# ---------------
@pytest.fixture
def test_jpss_manifest(test_data_path):
    """Path to test JPSS manifest file"""
    return test_data_path / "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"


@pytest.fixture
def test_jpss1_cr_1(test_data_path):
    """Path to test JPSS-1 construction record"""
    return test_data_path / "packets/jpss1_sc_pos_packets/P1590011AAAAAAAAAAAAAT21099051420500.PDS"


@pytest.fixture
def test_jpss1_pds_file_1(test_data_path):
    """Path to the test JPSS-1 PDS file associated with construction record 1"""
    return test_data_path / "packets/jpss1_sc_pos_packets/P1590011AAAAAAAAAAAAAT21099051420501.PDS"


@pytest.fixture
def test_jpss1_cr_2(test_data_path):
    """Path to test JPSS-1 construction record"""
    return test_data_path / "packets/jpss1_sc_pos_packets/P1590011AAAAAAAAAAAAAT21099065436900.PDS"


@pytest.fixture
def test_jpss1_pds_file_2(test_data_path):
    """Path to the test JPSS-1 PDS file associated with construction record 2"""
    return test_data_path / "packets/jpss1_sc_pos_packets/P1590011AAAAAAAAAAAAAT21099065436901.PDS"


@pytest.fixture
def test_jpss1_cr_3(test_data_path):
    """Path to test JPSS-1 construction record"""
    return test_data_path / "packets/jpss1_sc_pos_packets/P1590011AAAAAAAAAAAAAT21099091211400.PDS"


@pytest.fixture
def test_jpss1_pds_file_3(test_data_path):
    """Path to the test JPSS-1 PDS file associated with construction record 3"""
    return test_data_path / "packets/jpss1_sc_pos_packets/P1590011AAAAAAAAAAAAAT21099091211401.PDS"


# JPSS-4 PDS Data
# ---------------
@pytest.fixture
def test_jpss4_cr_1(test_data_path):
    """Path to test JPSS4 construction record"""
    return test_data_path / "packets/jpss4_sc_pos_packets/P1790011AAAAAAAAAAAAAT25255141303500.PDS"


@pytest.fixture
def test_jpss4_pds_file_1(test_data_path):
    """Path to the test JPSS4 PDS file associated with construction record 1"""
    return test_data_path / "packets/jpss4_sc_pos_packets/P1790011AAAAAAAAAAAAAT25255141303501.PDS"


# Libera ISTR Packets
# -------------------
@pytest.fixture
def test_ccsds_2025_218_18_37_32(test_data_path):
    """See test_data/packets/libera_istr_packets/notes.md for details"""
    return test_data_path / "packets/libera_istr_packets/ccsds_2025_218_18_37_32"


@pytest.fixture
def test_ccsds_2025_218_18_41_30(test_data_path):
    """See test_data/packets/libera_istr_packets/notes.md for details"""
    return test_data_path / "packets/libera_istr_packets/ccsds_2025_218_18_41_30"


@pytest.fixture
def test_ccsds_2025_221_16_56_48(test_data_path):
    """See test_data/packets/libera_istr_packets/notes.md for details"""
    return test_data_path / "packets/libera_istr_packets/ccsds_2025_221_16_56_48"


@pytest.fixture
def test_ccsds_2025_221_17_17_58(test_data_path):
    """See test_data/packets/libera_istr_packets/notes.md for details"""
    return test_data_path / "packets/libera_istr_packets/ccsds_2025_221_17_17_58"


# SPICE test data
# ---------------
@pytest.fixture
def test_lsk(spice_test_data_path):
    """Path to the LSK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / "naif0012.tls"


@pytest.fixture
def test_jpss_ck(spice_test_data_path):
    """Path to the testing JPSS CK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / "LIBERA_JPSS_V2-1-0_20210408T235850_20210409T015849_R23110123456.bc"


@pytest.fixture
def test_jpss_spk(spice_test_data_path):
    """Path to the testing JPSS SPK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / "LIBERA_JPSS_V2-1-0_20210408T235850_20210409T015849_R23110123456.bsp"


@pytest.fixture
def test_de_spk(spice_test_data_path):
    """Path to the testing default ephemeris kernel stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / "de440s.bsp"


@pytest.fixture
def test_pck(spice_test_data_path):
    """Path to the testing standard text planetary constants kernel (PCK) stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / "pck00010.tpc"


@pytest.fixture
def test_itrf93_pck(spice_test_data_path):
    """Path to the testing high precision planetary constants kernel (PCK) stored in the test_data directory
    to provide a single configuration for all tests"""
    return spice_test_data_path / "earth_000101_211220_210926.bpc"


@pytest.fixture
def test_scene_id(test_data_path):
    """Path to folder containing scene ID integration test files"""
    return test_data_path / "scene_id"


# Test configuration files
# ------------------------
@pytest.fixture
def test_product_definition(product_definitions_test_data_path):
    """Path to an example product description yaml file for unit testing"""
    return product_definitions_test_data_path / "unit_test_product_definition.yml"


@pytest.fixture
def test_camera_product_definition(product_definitions_test_data_path):
    """Path to a full camera product definition yaml file for unit testing

    This file contains a complete LiberaDataProductDefinition for camera data products,
    including attributes, coordinates, and variables sections.
    """
    return product_definitions_test_data_path / "test_camera_product_definition.yml"
