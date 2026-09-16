"""Live checks against the NAIF server.

These are the only tests allowed to reach an external service. Everything in ``tests/unit`` and
``tests/integration`` runs under the ``block_outbound_network`` guard in ``tests/conftest.py``;
the ``e2e`` marker lifts it.

They run daily rather than per-PR because their subject is the *integration with NAIF* -- that the
scrapers still match the published index, that a download lands in the cache, and that a second
load reuses it -- and that can break without anything in this repo changing. A failure here is as
likely to be news about NAIF as a regression in libera_utils.
"""

import datetime
import os
import re

import pytest

from libera_utils.io.caching import get_local_cache_dir
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager
from tests.helpers import index_of_furnished_kernel_matching

pytestmark = pytest.mark.e2e


def test_load_naif_kernels_with_real_caching_from_naif():
    """NAIF kernels download into the local cache, furnish in order, and are reused on a second load."""
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

    predict_idx = index_of_furnished_kernel_matching(spice_utils.NAIF_EARTH_EXTENDED_PCK_REGEX)
    high_prec_idx = index_of_furnished_kernel_matching(spice_utils.NAIF_HIGH_PREC_PCK_REGEX)
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
