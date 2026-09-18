"""Pytest plugin module for SPICE-related fixtures"""

import re
import tempfile
from pathlib import Path

import pytest
import spiceypy as spice
from curryer import spicetime

from libera_utils.config import config
from libera_utils.libera_spice import spice_utils


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
def noaa20_environment(monkeypatch, test_data_path):
    """Point the kernel configuration at the NOAA-20 family in ``tests/test_data/noaa20_spice``.

    NOAA-20 is not a Libera spacecraft and is not a configuration production can select; it is
    retained because it is the only kernel generation we can drive from *real* decoded spacecraft
    telemetry and validate against a third-party geolocation product (CERES). It therefore lives in
    test data rather than in the installed package (LIBSDC-703).

    Only the keys the family actually needs are overridden. ``LIBERA_KERNEL_STATIC_CONFIGS`` still
    resolves under this directory, which holds no static offset configs -- those are exercised
    against the shipped jpss4 set instead -- so do not call ``KernelManager.load_static_kernels``
    under this fixture.
    """
    monkeypatch.setenv("LIBERA_KERNEL_DIR", str(test_data_path / "noaa20_spice"))
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
def furnish_time_kernels(test_lsk):
    """Furnishes (temporarily) the checked-in LSK and JPSS SCLK, the kernels time conversions need.

    :func:`~libera_utils.libera_spice.spice_utils.ensure_spice` recovers from an unfurnished kernel
    pool by furnishing ``SPICE_METAKERNEL``, or -- for a function needing only time kernels -- by
    downloading the newest LSK from NAIF. Neither belongs in a test of a time conversion: the
    download makes the expected value depend on what NAIF published, and it is a network call inside
    a unit test. Furnishing the checked-in kernels up front means ``ensure_spice`` succeeds on its
    first attempt and never reaches either fallback, which are covered directly in
    ``tests/unit/test_libera_spice/test_spice_utils.py``.
    """
    spice.furnsh(str(test_lsk))
    spice.furnsh(config.get("JPSS_SCLK"))
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


@pytest.fixture
def recorded_retry_backoff(monkeypatch):
    """Record the retry backoff in :mod:`libera_utils.libera_spice.spice_utils` instead of sleeping.

    Both NAIF retry loops wait a second between attempts. That delay is politeness toward the NAIF
    server, not behaviour a test needs to sit through, and it accounted for roughly twelve seconds
    of the unit lane. Recording the calls keeps the loop's control flow intact -- the same number of
    attempts against the same mocked responses -- and turns the backoff into something a test can
    assert on rather than merely endure.

    Returns
    -------
    list of float
        The requested sleep durations, in call order. One entry per retry, so a loop that makes
        ``n`` attempts before succeeding or giving up records ``n - 1`` entries.
    """
    slept: list[float] = []
    monkeypatch.setattr(spice_utils.time, "sleep", slept.append)
    return slept
