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


@pytest.fixture
def short_tmp_path():
    """Creates a short temporary directory and returns a Path object pointing to it.
    This is specifically useful for mocking directories that need to be pointed to from SPICE
    text kernels (e.g. MSOPCK and MKSPK setup files).
    For all other use cases that don't require a short path, use the pytest tmp_path fixture.
    """
    with tempfile.TemporaryDirectory(prefix='/tmp/') as td:
        yield Path(td)


@pytest.fixture
def libera_sdp_test_data_dir():
    """Returns the test data directory"""
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def furnish_fk():
    """Furnishes (temporarily) the Libera frame kernel (FK)"""
    spice.furnsh(config.get('LIBERA_FK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_sclk():
    """Furnishes (temporarily) the default SCLK for JPSS"""
    spice.furnsh(config.get('JPSS_SCLK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_lsk(libera_sdp_test_data_dir):
    """Furnishes (temporarily) the testing LSK"""
    spice.furnsh(str(libera_sdp_test_data_dir / 'naif0012.tls'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_jpss_ck(libera_sdp_test_data_dir):
    """Furnishes (temporarily) a testing JPSS CK"""
    spice.furnsh(str(libera_sdp_test_data_dir / 'libera_jpss_20210408t235850_20210409t015849.bc'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_jpss_spk(libera_sdp_test_data_dir):
    """Furnishes (temporarily) a testing JPSS SPK"""
    spice.furnsh(str(libera_sdp_test_data_dir / 'libera_jpss_20210408t235850_20210409t015849.bsp'))
    yield
    spice.kclear()


@pytest.fixture
def setup_test_logging(tmpdir, monkeypatch):
    """Sets up a task logger for a test"""
    monkeypatch.setenv('LIBSDP_STREAM_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('LIBSDP_LOG_DIR', str(tmpdir))
    log_filepath = logutil.setup_task_logger('test_log')
    yield log_filepath
    # Remove all handlers from root logger. No other loggers should ever have handlers attached.
    logging.getLogger().handlers = []
