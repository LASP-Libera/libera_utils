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


def _index_of_furnished_kernel_matching(pattern: str) -> int:
    """Return the SPICE kernel-pool index of the first furnished file matching ``pattern``."""
    pool = spice_utils.ls_kernels()
    for index, kernel in enumerate(pool):
        if re.search(pattern, Path(kernel.file_name).name):
            return index
    pool_names = [kernel.file_name for kernel in pool]
    pytest.fail(f"No furnished kernel matching {pattern!r}. Kernel pool: {pool_names}")


def test_earth_orientation_kernel_furnishing_order(generic_kernel_dir):
    """
    NAIF guidance: furnish predict before ops high-precision so overlapping intervals
    prefer the newer high-precision file. SPICE uses the last-loaded kernel when coverage overlaps.
    """
    km = KernelManager()
    km.load_naif_kernels()

    fk_idx = _index_of_furnished_kernel_matching(r"earth_assoc_itrf93\.tf")
    predict_idx = _index_of_furnished_kernel_matching(spice_utils.NAIF_EARTH_EXTENDED_PCK_REGEX)
    high_prec_idx = _index_of_furnished_kernel_matching(spice_utils.NAIF_HIGH_PREC_PCK_REGEX)

    assert fk_idx < predict_idx < high_prec_idx, (
        "Earth orientation kernels must be furnished FK, then predict PCK, then ops high-precision PCK. "
        f"Pool order: {[kernel.file_name for kernel in spice_utils.ls_kernels()]}"
    )


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


def test_static_kernels_loading(
    generic_kernel_dir,
    test_naif_text_pck,
    test_lsk,
    test_de_spk,
    test_earth_assoc_itrf93_fk,
    test_earth_predict_pck,
    test_itrf93_pck,
):
    """
    Test loading static kernels using the KernelManager.
    """
    with KernelManager() as km:
        assert km._static_loaded
        assert km._naif_kernels_loaded

        # Using the load_kernels class from curryer, get the list of kernels loaded
        loaded_kernels = km._loaded_kernels.loaded
        loaded_kernel_names = [Path(k).name.split(".")[0] for k in loaded_kernels]

        generic_kernel_files = [
            test_naif_text_pck,
            test_lsk,
            test_earth_assoc_itrf93_fk,
            test_earth_predict_pck,
            test_itrf93_pck,
        ]

        # Get the Libera instrument kernel
        libera_instrument_kernel = [config.get("LIBERA_KERNEL_INSTRUMENT")]

        # Static generated kernels are cached/furnished as binary .bc/.bsp outputs from JSON configs.
        libera_static_generated_kernels = []
        for kernel_config in config.get("LIBERA_KERNEL_STATIC_CONFIGS"):
            config_path = Path(kernel_config)
            stem = config_path.stem
            if stem.endswith(".spk"):
                libera_static_generated_kernels.append(config_path.with_suffix("").name + ".bsp")
            elif stem.endswith(".ck"):
                libera_static_generated_kernels.append(config_path.with_suffix("").name + ".bc")
            else:
                pytest.fail(f"Unexpected static kernel config type: {kernel_config}")

        all_libera_expected_kernels = set(generic_kernel_files + libera_instrument_kernel)
        all_libera_expected_kernels.update(libera_static_generated_kernels)

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

        dk_dir = Path(test_data_path) / "dynamic_kernels"
        kernel_paths = sorted(f for f in dk_dir.iterdir() if f.is_file())
        assert kernel_paths
        km.load_libera_dynamic_kernels(kernel_paths)

        # Now, dynamic kernels should be loaded
        assert km._dynamic_loaded

        # Using the load_kernels class from curryer, get the list of kernels loaded
        loaded_kernels = km._loaded_kernels.loaded
        loaded_kernel_names = [Path(k).name.split(".")[0] for k in loaded_kernels]

        dk_dir = Path(test_data_path) / "dynamic_kernels"
        for file in dk_dir.iterdir():
            if not file.is_file():
                continue
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

    assert len(loaded_kernels) == 6

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
        # Extended Earth predict PCK is downloaded from NAIF (maybe cached from previous tests)
        elif re.search(spice_utils.NAIF_EARTH_EXTENDED_PCK_REGEX, kernel_path):
            assert re.search("cache", kernel_path, re.IGNORECASE)
        # Leap seconds is not downloaded from NAIF should be present locally
        elif re.search(spice_utils.NAIF_LSK_REGEX, kernel_path):
            assert not re.match("cache", kernel_path, re.IGNORECASE)
        # Earth association FK is downloaded from NAIF (maybe cached from previous tests)
        elif re.search(r"earth_assoc_itrf93\.tf", kernel_path):
            assert re.search("cache", kernel_path, re.IGNORECASE)
        else:
            pytest.fail(f"Unexpected kernel loaded: {kernel_path}")

    predict_idx = _index_of_furnished_kernel_matching(spice_utils.NAIF_EARTH_EXTENDED_PCK_REGEX)
    high_prec_idx = _index_of_furnished_kernel_matching(spice_utils.NAIF_HIGH_PREC_PCK_REGEX)
    assert predict_idx < high_prec_idx, (
        "Predict Earth PCK must be furnished before ops high-precision PCK when intervals overlap."
    )

    # Check that another load does not re-download the PCK
    pre_second_load_time = datetime.datetime.now()
    km.load_naif_kernels()
    cached_files = [f for f in kernel_cache_path.iterdir()]
    for file in cached_files:
        if re.search(spice_utils.NAIF_PCK_REGEX, str(file), re.IGNORECASE):
            # Ensure the file timestamps have not changed
            assert os.path.getctime(file) <= pre_second_load_time.timestamp()
            assert os.path.getmtime(file) <= pre_second_load_time.timestamp()
