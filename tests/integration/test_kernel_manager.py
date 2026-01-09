import datetime
import os
import re
from pathlib import Path

import curryer.spicierpy as sp
import pytest

from libera_utils.config import config
from libera_utils.io.caching import get_local_cache_dir
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager


def test_ensure_known_kernels_are_furnished(spice_test_data_path):
    """Test _ensure_ready internal validation method."""
    km = KernelManager()

    # Should return true with no kernels furnished but with a warning
    with pytest.warns(UserWarning, match="No kernels are currently furnished by SPICE."):
        km.ensure_known_kernels_are_furnished()

    km.load_static_kernels()
    # Normal case should pass when all static kernels are furnished
    km.ensure_known_kernels_are_furnished()

    # Manually furnish an extra kernel not managed by KernelManager
    sp.furnsh(str(spice_test_data_path / "pck00011.tpc"))
    with pytest.warns(UserWarning, match="More kernels are furnished by SPICE than expected"):
        km.ensure_known_kernels_are_furnished()

    # Unload one known kernel
    known_kernels = km._loaded_kernels.loaded
    sp.unload(known_kernels[0])
    # Number of furnished kernel matches but not all known kernels are furnished so this counts as a mismatch
    # where not all expected kernels are furnished
    with pytest.raises(RuntimeError, match="Not all kernels are furnished by SPICE."):
        km.ensure_known_kernels_are_furnished()

    # Unload the extra kernel to now have explicitly fewer furnished than known
    sp.unload(str(spice_test_data_path / "pck00011.tpc"))
    with pytest.raises(RuntimeError, match="Not all kernels are furnished by SPICE."):
        km.ensure_known_kernels_are_furnished()


def test_static_kernels_loading(monkeypatch, spice_test_data_path):
    """
    Test loading static kernels using the KernelManager.
    """
    # This ensures that the KernelManager looks in a local directory for NAIF generic kernels and won't try to
    # download them
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))

    with KernelManager() as km:
        assert km._static_loaded
        assert km._naif_kernels_loaded

        # Using the load_kernels class from curryer, get the list of kernels loaded
        loaded_kernels = km._loaded_kernels.loaded
        loaded_kernel_names = [Path(k).name.split(".")[0] for k in loaded_kernels]

        # Get the list of generic kernels from the config
        generic_kernels_path = Path(config.get("GENERIC_KERNEL_DIR"))
        generic_kernel_files = [
            generic_kernels_path / "pck00011.tpc",
            generic_kernels_path / "naif0012.tls",
            generic_kernels_path / "earth_assoc_itrf93.tf",
            generic_kernels_path / "earth_000101_211220_210926.bpc",
        ]

        # Get the Libera instrument kernel
        libera_instrument_kernel = [config.get("LIBERA_KERNEL_INSTRUMENT")]

        # Get the list Libera static kernels from the config
        libera_static_kernels = config.get("LIBERA_KERNEL_STATIC_CONFIGS")

        all_libera_expected_kernels = set(generic_kernel_files + libera_static_kernels + libera_instrument_kernel)

        for file in all_libera_expected_kernels:
            filename = Path(file).name.split(".")[0]
            assert filename in loaded_kernel_names


def test_dynamic_kernels_loading(monkeypatch, spice_test_data_path, test_data_path):
    """
    Test loading dynamic kernels using the KernelManager.
    """

    # This ensures that the KernelManager looks in a local directory for NAIF generic kernels and won't try to
    # download them
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))

    with KernelManager() as km:
        # Initially, dynamic kernels should not be loaded
        assert not km._dynamic_loaded

        # Load dynamic kernels
        km.load_libera_dynamic_kernels(Path(test_data_path) / "dynamic_kernels")

        # Now, dynamic kernels should be loaded
        assert km._dynamic_loaded

        # Using the load_kernels class from curryer, get the list of kernels loaded
        loaded_kernels = km._loaded_kernels.loaded
        loaded_kernel_names = [Path(k).name.split(".")[0] for k in loaded_kernels]

        libera_dynamic_kernel_dir = Path(test_data_path / "dynamic_kernels")
        for file in libera_dynamic_kernel_dir.iterdir():
            filename = Path(file).name.split(".")[0]
            assert filename in loaded_kernel_names


def test_load_naif_kernels_with_real_caching_from_naif(test_data_path):
    """
    Test loading Earth-related kernels using the KernelManager.
    """
    km = KernelManager()
    assert not km._static_loaded

    # Ensure at least 1 kernel is downloaded by deleting a cached version if it exists
    pre_delete_time = datetime.datetime.now()
    kernel_cache_path = get_local_cache_dir()
    if kernel_cache_path.exists():
        cached_files = [f for f in kernel_cache_path.iterdir()]
        for file in cached_files:
            if re.search(spice_utils.NAIF_PCK_REGEX, str(file), re.IGNORECASE):
                os.remove(file)
                assert not file.exists()
                break

    # Load NAIF kernels
    km.load_naif_kernels()
    assert km._naif_kernels_loaded

    # Check that the expected NAIF kernels are loaded from cache or local directories
    loaded_kernels = km._loaded_kernels.loaded

    assert len(loaded_kernels) == 5

    for kernel_path in loaded_kernels:
        # PCK is downloaded from NAIF (and should be each test as we deleted it above)
        if re.search(spice_utils.NAIF_PCK_REGEX, kernel_path):
            assert re.search("cache", kernel_path, re.IGNORECASE)
            assert os.path.getctime(kernel_path) >= pre_delete_time.timestamp()
        # Default Ephemeris is downloaded from NAIF (maybe cached from previous tests)
        elif re.search(spice_utils.NAIF_DE_REGEX, kernel_path):
            assert re.search("cache", kernel_path, re.IGNORECASE)
        # High precision Earth PCK is downloaded from NAIF (maybe cached from previous tests)
        elif re.search(spice_utils.NAIF_HIGH_PREC_PCK_REGEX, kernel_path):
            assert re.search("cache", kernel_path, re.IGNORECASE)
        # Leap seconds is not downloaded from NAIF should be present locally
        elif re.search(spice_utils.NAIF_LSK_REGEX, kernel_path):
            assert not re.match("cache", kernel_path, re.IGNORECASE)
        # Earth high precision frame kernel is not downloaded from NAIF should be present locally
        elif re.search(r"earth_assoc_itrf93\.tf", kernel_path):
            assert re.search("cache", kernel_path, re.IGNORECASE)
        else:
            pytest.fail(f"Unexpected kernel loaded: {kernel_path}")

    # Check that another load does not re-download the PCK
    pre_second_load_time = datetime.datetime.now()
    km.load_naif_kernels()
    cached_files = [f for f in kernel_cache_path.iterdir()]
    for file in cached_files:
        if re.search(spice_utils.NAIF_PCK_REGEX, str(file), re.IGNORECASE):
            # Ensure the file timestamps have not changed
            assert os.path.getctime(file) <= pre_second_load_time.timestamp()
            assert os.path.getmtime(file) <= pre_second_load_time.timestamp()
