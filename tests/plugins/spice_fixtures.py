"""Pytest plugin module for SPICE-related fixtures"""

import re
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
    """Loads the test LSK into the kernel pool and points curryer's default-LSK lookup at it.

    Curryer resolves a leapsecond kernel via ``spicetime.leapsecond.find_default_file()`` whenever a
    kernel config omits ``leapsecond_kernel`` (all Libera configs do). When ``LEAPSECOND_FILE_ENV``
    is set, that directory is the only one searched; otherwise curryer falls back to the LSK packaged
    with it. Production sets the variable in ``KernelManager.load_naif_kernels()`` and tests set it
    here, so kernel creation runs against the LSK furnished rather than curryer's own.
    """
    monkeypatch.setenv("LEAPSECOND_FILE_ENV", str(test_lsk.parent))
    spicetime.leapsecond.load(test_lsk)
    return test_lsk


@pytest.fixture
def noaa20_environment(monkeypatch):
    # Set environment variables to point to the NOAA 20 older kernel definitions.
    monkeypatch.setenv("LIBERA_KERNEL_DIR", "{LIBERA_UTILS_DATA_DIR}/spice/noaa20")
    monkeypatch.setenv("LIBERA_KERNEL_CLOCK", "{LIBERA_KERNEL_DIR}/noaa20.fakeclock.sclk.tsc")
    monkeypatch.setenv("LIBERA_KERNEL_SC_SPK_CONFIG", "{LIBERA_KERNEL_DIR}/noaa20_sc.ephemeris.spk.json")
    monkeypatch.setenv("LIBERA_KERNEL_SC_CK_CONFIG", "{LIBERA_KERNEL_DIR}/noaa20_sc.attitude.ck.json")


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


#: Keywords carrying the OAV3 measured misalignments (LIBSDC-806), with their ideal values.
#: The three ``*_IN_STAND`` vectors are read by ``kernel_maker`` to build the mechanism CKs;
#: ``TKFRAME_-143013011_Q`` is the radiometer boresight rotation derived from ``LIBERA_EL0_Z_IN_STAND``.
_NOMINAL_FRAME_VALUES = {
    "LIBERA_EL0_Z_IN_STAND": "( 0.0, 0.0, 1.0 )",
    "LIBERA_EL_AOR_IN_STAND": "( 1.0, 0.0, 0.0 )",
    "LIBERA_AZ_AOR_IN_STAND": "( 0.0, 0.0, 1.0 )",
    "TKFRAME_-143013011_Q": "( 1.0, 0.0, 0.0, 0.0 )",
}


@pytest.fixture
def nominal_frame_kernel(tmp_path):
    """A misalignment-free copy of the production Libera frame kernel, derived at test time.

    Zeroes the measured axes of rotation and the radiometer boresight rotation, leaving every
    other frame definition -- including the spacecraft parentage -- exactly as the production
    kernel has it. Tests that need ideal geometry (an A/B against the measured misalignment, or
    a comparison against an external reference computed without it) furnish this in place of
    ``LIBERA_KERNEL_FRAME``.

    Derived rather than checked in on purpose: a frozen copy silently stops tracking the
    production kernel in every respect except the misalignment it was made to remove.
    """
    source = Path(config.get("LIBERA_KERNEL_FRAME"))
    text = source.read_text()
    for keyword, ideal in _NOMINAL_FRAME_VALUES.items():
        # Unbounded count on purpose: capping at one would make the check below mean "at least
        # once", and a duplicated definition would be silently half-substituted.
        text, count = re.subn(
            rf"^(\s*{re.escape(keyword)}\s*=\s*)\([^)]*\)",
            lambda m, ideal=ideal: m.group(1) + ideal,
            text,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(
                f"{keyword} found {count} times in {source}, expected exactly once; the frame kernel layout changed."
            )

    nominal = tmp_path / "libera_nominal.frames.fk.tf"
    nominal.write_text(text)
    return nominal


@pytest.fixture
def isolated_kernel_cache(tmp_path, monkeypatch):
    """Redirect the on-disk kernel cache into ``tmp_path`` for the duration of a test.

    :class:`~libera_utils.libera_spice.spice_utils.KernelFileCache` keys on basename and
    returns any cached copy younger than its timeout without inspecting content, so a test
    that materializes kernels can otherwise pick up a stale copy left by an earlier run or by
    a developer's own work. Production filenames carry a unique creation-time field and never
    collide; test fixtures regenerated under a reused name can.
    """
    cache_root = tmp_path / "kernel_cache"
    cache_root.mkdir()
    monkeypatch.setattr("libera_utils.libera_spice.spice_utils.caching.get_local_cache_dir", lambda: cache_root)
    monkeypatch.setattr("libera_utils.libera_spice.kernel_manager.get_local_cache_dir", lambda: cache_root)
    return cache_root


@pytest.fixture
def generic_kernel_dir(monkeypatch, spice_test_data_path):
    """Set ``GENERIC_KERNEL_DIR`` to bundled NAIF kernels under tests/test_data/spice."""
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    return spice_test_data_path


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
