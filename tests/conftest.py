"""Pytest fixtures"""
# Standard
import logging
import sys
import tempfile
from pathlib import Path
# Installed
import pytest
import spiceypy as spice
# Local
from libera_sdp.config import config
from libera_sdp import logutil

# TODO: Investigate creating a stub conftest.py and a conftest package so we can split up our fixtures more logically,
#  e.g. here: https://stackoverflow.com/a/27068195/2970906


@pytest.fixture
def short_tmp_path():
    """Creates a short temporary directory and returns a Path object pointing to it.
    This is specifically useful for mocking directories that need to be pointed to from SPICE
    text kernels (e.g. MSOPCK and MKSPK setup files).
    For all other use cases that don't require a short path, use the pytest tmp_path fixture.
    """
    with tempfile.TemporaryDirectory(prefix='/tmp/') as td:
        yield Path(td)


# Paths to test data directories
# ------------------------------
@pytest.fixture
def test_data_path():
    """Returns the Path to the test_data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def spice_test_data_path(test_data_path):
    """Returns the spice subdirectory of the test_data directory
    This directory contains kernel that are either generated (SPK and CK) or dynamically downloaded.
    Any kernels that are available directly in the libera_sdp/data directory should be sourced from there.
    """
    return test_data_path / 'spice'


# Paths to commonly used test data files
# --------------------------------------
@pytest.fixture
def test_lsk(spice_test_data_path):
    """Path to the LSK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'naif0012.tls'


@pytest.fixture
def test_jpss_ck(spice_test_data_path):
    """Path to the testing JPSS CK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'libera_jpss_20210408t235850_20210409t015849.bc'


@pytest.fixture
def test_jpss_spk(spice_test_data_path):
    """Path to the testing JPSS SPK stored in the test_data directory to provide a single configuration for all tests"""
    return spice_test_data_path / 'libera_jpss_20210408t235850_20210409t015849.bsp'


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


# Furnishing fixtures for testing kernels
# ---------------------------------------
@pytest.fixture
def furnish_fk():
    """Furnishes (temporarily) the Libera frame kernel (FK) stored in the package data directory"""
    spice.furnsh(config.get('LIBERA_FK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_sclk():
    """Furnishes (temporarily) the SCLK for JPSS stored in the package data directory"""
    spice.furnsh(config.get('JPSS_SCLK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_test_lsk(test_lsk):
    """Furnishes (temporarily) the testing LSK"""
    spice.furnsh(str(test_lsk))
    yield
    spice.kclear()


@pytest.fixture
def furnish_test_jpss_ck(test_jpss_ck):
    """Furnishes (temporarily) a testing JPSS CK"""
    spice.furnsh(str(test_jpss_ck))
    yield
    spice.kclear()


@pytest.fixture
def furnish_test_jpss_spk(test_jpss_spk):
    """Furnishes (temporarily) a testing JPSS SPK"""
    spice.furnsh(str(test_jpss_spk))
    yield
    spice.kclear()


@pytest.fixture
def furnish_test_de_spk(test_de_spk):
    """Furnishes (temporarily) a testing development ephemeris SPK kernel"""
    spice.furnsh(str(test_de_spk))
    yield
    spice.kclear()


@pytest.fixture
def furnish_test_pck(test_pck):
    """Furnishes (temporarily) a testing text PCK kernel"""
    spice.furnsh(str(test_pck))
    yield
    spice.kclear()


@pytest.fixture
def furnish_test_itrf93_pck(test_itrf93_pck):
    """Furnishes (temporarily) a testing ITRF93 high precision binary PCK kernel
    Also furnishes (temporarily) a NAIF-produced FK that associates the Earth body with the ITRF93 reference frame.
    """
    spice.furnsh(str(test_itrf93_pck))
    spice.furnsh(config.get('EARTH_ASSOC_ITRF93_FK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_testing_kernels(furnish_fk,
                            furnish_sclk, furnish_test_lsk,
                            furnish_test_de_spk, furnish_test_pck, furnish_test_itrf93_pck,
                            furnish_test_jpss_ck, furnish_test_jpss_spk):
    """Furnishes all the testing kernels provided above, basically as a syntactic shortcut. Fixtures are executed
    from left to right in order so the first argument to this fixture furnishes first.
    Note: Order matters here if multiple files furnish the same data. The latest file furnished is always used.
    e.g. if two files provide overlapping attitude data, be sure to put the latest/most up to date file last.
    """
    yield


@pytest.fixture
def setup_test_logging(tmp_path, monkeypatch):
    """Sets up a task logger for a test"""
    monkeypatch.setenv('LIBSDP_STREAM_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('LIBSDP_LOG_DIR', str(tmp_path))
    log_filepath = logutil.setup_task_logger('test_log')
    yield log_filepath
    # Remove all handlers from root logger. No other loggers should ever have handlers attached.
    logging.getLogger().handlers = []
