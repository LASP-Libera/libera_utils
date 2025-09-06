"""Pytest plugin module for SPICE-related fixtures"""

import tempfile
from pathlib import Path

import pytest
import spiceypy as spice
from curryer import spicetime

from libera_utils.config import config


@pytest.fixture
def short_tmp_path():
    """Creates a short temporary directory and returns a Path object pointing to it.
    This is specifically useful for mocking directories that need to be pointed to from SPICE
    text kernels (e.g. MSOPCK and MKSPK setup files).
    For all other use cases that don't require a short path, use the pytest tmp_path fixture.
    """
    with tempfile.TemporaryDirectory(prefix="/tmp/") as td:
        yield Path(td)


@pytest.fixture
def curryer_lsk(test_lsk, monkeypatch):
    """Loads the Libera LSK for use by Curryer and sets the environment variable temporarily"""
    # TODO[LIBSDC-600]: Reconsider after curryer LSK logic is updated.
    monkeypatch.setenv("LEAPSECOND_FILE_ENV", str(test_lsk.parent))
    spicetime.leapsecond.load(test_lsk)
    return test_lsk


# Furnishing fixtures for testing kernels
# ---------------------------------------
@pytest.fixture(autouse=True)
def autoclear_spice():
    """Automatically clears out all SPICE remnants after every single test to prevent the kernel pool from
    interfering with future tests. Option autouse ensures this is run after every test."""
    yield
    spice.kclear()


@pytest.fixture
def furnish_fk():
    """Furnishes (temporarily) the Libera frame kernel (FK) stored in the package data directory"""
    spice.furnsh(config.get("LIBERA_FK"))
    yield
    spice.kclear()


@pytest.fixture
def furnish_sclk():
    """Furnishes (temporarily) the SCLK for JPSS stored in the package data directory"""
    spice.furnsh(config.get("JPSS_SCLK"))
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
    spice.furnsh(config.get("EARTH_ASSOC_ITRF93_FK"))
    yield
    spice.kclear()


@pytest.fixture
def furnish_testing_kernels(
    furnish_fk,
    furnish_sclk,
    furnish_test_lsk,
    furnish_test_de_spk,
    furnish_test_pck,
    furnish_test_itrf93_pck,
    furnish_test_jpss_ck,
    furnish_test_jpss_spk,
):
    """Furnishes all the testing kernels provided above, basically as a syntactic shortcut. Fixtures are executed
    from left to right in order so the first argument to this fixture furnishes first.
    Note: Order matters here if multiple files furnish the same data. The latest file furnished is always used.
    e.g. if two files provide overlapping attitude data, be sure to put the latest/most up to date file last.
    """
    yield
    spice.kclear()
