"""Drift guard for the frozen dynamic kernels.

``tests/test_data/dynamic_kernels`` is a *snapshot* of what the SPICE processing step produced at
the time it was committed. Every geolocation test reads it instead of building kernels, which is
what keeps those tests fast and mirrors production, but it also means the fixture can fall behind
the code that is supposed to have produced it -- silently, because nothing reads the generator.

That is exactly what happened between 2026-01 and LIBSDC-703: the committed CKs predated the
encoder correction (LIBSDC-668) and the measured frame misalignments (LIBSDC-806), so for eight
months the geolocation tests were validating a kernel/frame pairing no real instrument has.

This test regenerates the fixture from its source CSVs through the production
``create_kernel_from_l1a`` path and compares the *geometry* the two sets encode. It runs daily
rather than per-PR because rebuilding four kernels through ``mkspk``/``msopck`` is too slow to sit
in front of a pull request, and because the drift it catches accrues over months, not commits.

Comparison is on sampled geometry, never on bytes or filenames. Kernel filenames carry a creation
timestamp, so every regeneration produces a new name by design -- see the generator's module
docstring for why reusing a name would be worse than useless.

When this fails, the fix is normally to regenerate and commit the fixture:

    python tests/fixture_generation/generate_dynamic_kernels.py

and then to work out what changed and whether ``libera_rad``, which carries its own copy of the
same fixture set, needs the update too.
"""

import shutil

import numpy as np
import pandas as pd
import pytest
import spiceypy as spice
from curryer import spicetime
from curryer import spicierpy as sp

from libera_utils.libera_spice.kernel_manager import KernelManager
from tests.fixture_generation import generate_dynamic_kernels as fixture_gen

pytestmark = pytest.mark.e2e

#: Epochs to compare the two kernel sets at. The mechanism CKs turn through a full cross-track
#: sweep every 5.087 s, so a half-second grid samples every part of the scan many times over.
SAMPLE_TIMES = pd.date_range(fixture_gen.WINDOW_START, fixture_gen.WINDOW_END, freq="500ms", inclusive="left")

#: The four kernels and the builder that produces each one's input frame.
FIXTURE_BUILDERS = {
    "JPSS-SPK": fixture_gen.spacecraft_frame,
    "JPSS-CK": fixture_gen.spacecraft_frame,
    "AZROT-CK": fixture_gen.azimuth_frame,
    "ELSCAN-CK": fixture_gen.elevation_frame,
}

#: Frame pairs whose relative orientation is the geometry each kernel carries.
SAMPLED_FRAMES = [
    ("JPSS4_SC_COORD", "ITRF93"),
    ("LIBERA_AZ_COORD", "LIBERA_BASE_COORD"),
    ("LIBERA_EL_COORD", "LIBERA_AZ_COORD"),
]


def _sample_geometry(kernel_paths: list) -> dict[str, np.ndarray]:
    """Furnish ``kernel_paths`` and read back the geometry they encode at :data:`SAMPLE_TIMES`.

    Parameters
    ----------
    kernel_paths : list of pathlib.Path
        The four dynamic kernels of one set.

    Returns
    -------
    dict of str to numpy.ndarray
        Spacecraft position and velocity in ITRF93, plus one stack of rotation matrices per entry
        in :data:`SAMPLED_FRAMES`. The kernel pool is cleared before returning, so two sets can be
        sampled in turn without their identical body and frame IDs colliding.

    Notes
    -----
    The spacecraft SPK and CK are both written in ITRF93, so sampling in that frame reads them
    directly. Asking for J2000 instead would require Earth orientation for the frame change, which
    the checked-in NAIF PCKs do not cover at the fixture's simulated 2028 epochs, and which would
    make the comparison depend on data that changes on NAIF's schedule rather than on ours.
    """
    ephemeris_times = spicetime.adapt(SAMPLE_TIMES, "dt64", "et")
    ugps_times = spicetime.adapt(SAMPLE_TIMES, "dt64", "ugps")
    try:
        km = KernelManager()
        km.load_libera_dynamic_kernels(kernel_paths, needs_naif_kernels=True, needs_static_kernels=True)

        sampled = {
            "spacecraft_state": sp.ext.query_ephemeris(
                ugps_times, "JPSS4_SC", "EARTH", ref_frame="ITRF93", velocity=True
            ).values
        }
        for from_frame, to_frame in SAMPLED_FRAMES:
            sampled[f"{from_frame}->{to_frame}"] = np.array(
                [sp.pxform(from_frame, to_frame, et) for et in ephemeris_times]
            )
        return sampled
    finally:
        spice.kclear()


def test_frozen_kernels_match_a_fresh_regeneration(
    generic_kernel_dir, isolated_kernel_cache, curryer_lsk, test_data_path, short_tmp_path
):
    """The committed fixture encodes the same geometry today's code generates from the same CSVs."""
    assert shutil.which("mkspk")
    assert shutil.which("msopck")

    committed = sorted(f for f in (test_data_path / "dynamic_kernels").iterdir() if f.is_file())
    assert len(committed) == len(FIXTURE_BUILDERS), (
        f"Expected {len(FIXTURE_BUILDERS)} frozen kernels, found {committed}"
    )

    for identifier, builder in FIXTURE_BUILDERS.items():
        frame, utc_range = builder()
        fixture_gen.write_kernel(frame, utc_range, identifier, short_tmp_path)
    regenerated = sorted(f for f in short_tmp_path.iterdir() if f.is_file())
    assert len(regenerated) == len(FIXTURE_BUILDERS)

    expected = _sample_geometry(committed)
    actual = _sample_geometry(regenerated)

    for key, expected_values in expected.items():
        # rtol=0 on purpose. assert_allclose defaults to rtol=1e-7, which for a ~7000 km
        # spacecraft state is ~0.7 m -- far looser than this guard claims and enough to hide a
        # real drift. Regeneration from identical inputs is deterministic, so the only difference
        # that should ever appear is float noise, and an absolute bound is what says that.
        np.testing.assert_allclose(
            actual[key],
            expected_values,
            rtol=0,
            atol=1e-9,
            err_msg=(
                f"The committed fixture and a fresh regeneration disagree on {key}. The fixture in "
                "tests/test_data/dynamic_kernels no longer reflects what the production kernel path "
                "produces from tests/test_data/tier1_geo. See this module's docstring."
            ),
        )
