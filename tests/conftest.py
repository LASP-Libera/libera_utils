"""Pytest fixtures"""
import logging
from pathlib import Path
import sys

import pytest
import spiceypy as spice

from libera_sdp.config import config
from libera_sdp import logutil


@pytest.fixture
def libera_sdp_test_data_dir():
    """Returns the test data directory

    Returns
    -------
    : Path
    """
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'


@pytest.fixture
def furnish_sclk():
    """Furnishes (temporarily) the default SCLK for JPSS"""
    spice.furnsh(config.get('JPSS_SCLK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_lsk():
    """Furnishes (temporarily) the fallback LSK"""
    spice.furnsh(config.get('FALLBACK_LSK'))
    yield
    spice.kclear()


@pytest.fixture
def furnish_jpss_ck(libera_sdp_test_data_dir):
    """Furnishes (temporarily) a JPSS CK"""
    spice.furnsh(str(libera_sdp_test_data_dir / 'libera_jpss_20210408t235959_20210409t015958.bc'))
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
