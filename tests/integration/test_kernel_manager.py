"""KernelManager loading behaviour against checked-in kernels.

Everything here is hermetic: NAIF generic kernels come from ``tests/test_data/spice`` rather than
the NAIF server. The one test that exercises a real download and its cache lives in
``tests/e2e/test_naif_kernel_cache.py``.
"""

from pathlib import Path

import curryer.spicierpy as sp
import pytest

from libera_utils.config import config
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager
from tests.helpers import index_of_furnished_kernel_matching

pytestmark = pytest.mark.integration


def test_earth_orientation_kernel_furnishing_order(generic_kernel_dir):
    """
    NAIF guidance: furnish predict before ops high-precision so overlapping intervals
    prefer the newer high-precision file. SPICE uses the last-loaded kernel when coverage overlaps.
    """
    km = KernelManager()
    km.load_naif_kernels()

    fk_idx = index_of_furnished_kernel_matching(r"earth_assoc_itrf93\.tf")
    predict_idx = index_of_furnished_kernel_matching(spice_utils.NAIF_EARTH_EXTENDED_PCK_REGEX)
    high_prec_idx = index_of_furnished_kernel_matching(spice_utils.NAIF_HIGH_PREC_PCK_REGEX)

    assert fk_idx < predict_idx < high_prec_idx, (
        "Earth orientation kernels must be furnished FK, then predict PCK, then ops high-precision PCK. "
        f"Pool order: {[kernel.file_name for kernel in spice_utils.ls_kernels()]}"
    )


def test_ensure_known_kernels_are_furnished(generic_kernel_dir, test_jpss_spk):
    """Test _ensure_ready internal validation method."""
    km = KernelManager()

    # Should return true with no kernels furnished but with a warning
    with pytest.warns(UserWarning, match="No kernels are currently furnished by SPICE."):
        km.ensure_known_kernels_are_furnished()

    km.load_static_kernels()
    # Normal case should pass when all static kernels are furnished
    km.ensure_known_kernels_are_furnished()

    # Manually furnish an extra kernel not managed by KernelManager. It has to be one the manager
    # would never load itself -- a dynamic kernel rather than a generic NAIF one -- or it is already
    # in the pool and the count does not change.
    sp.furnsh(str(test_jpss_spk))
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
    sp.unload(str(test_jpss_spk))
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


def test_dynamic_kernels_loading(monkeypatch, spice_test_data_path, test_data_path, isolated_kernel_cache):
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
