"""Pytest fixtures"""
from pathlib import Path
import sys

import pytest
import spiceypy as spice

from libera_sdp.config import config


@pytest.fixture
def test_data_dir():
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
def furnish_jpss_ck(test_data_dir):
    """Furnishes (temporarily) a JPSS CK"""
    spice.furnsh(str(test_data_dir / 'libera_jpss_20210408t235959_20210409t015958.bc'))
    yield
    spice.kclear()
