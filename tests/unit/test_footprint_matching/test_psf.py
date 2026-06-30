"""Unit tests for the partial PSF module (libera_utils.footprint_matching.psf).

These confirm the CERES PSF port behaves as the ATBD specifies:
- detector time response F(0) = 0 (ATBD Eq. 4.4-2 boundary condition)
- cross-scan symmetry P(delta, beta) = P(delta, -beta) (ATBD Eq. 4.4-1)
- zero response outside the optical FOV in the cross-scan direction
- the 95%-energy angular extent matches the ATBD Table 4.4-2 cutoffs
- the static FOV and conservative-extent helpers behave as documented
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching import psf


class TestFResponse:
    def test_response_at_zero_is_zero(self):
        # ATBD Eq. 4.4-2: F(0) = 0 (no integrated response at the leading edge).
        assert psf._f_response(np.array(0.0)) == pytest.approx(0.0, abs=1e-12)

    def test_response_grows_then_settles_near_one(self):
        # The integrated step response rises toward unity well into the tail.
        assert psf._f_response(np.array(5.0)) == pytest.approx(1.0, abs=0.05)


class TestPsfWeight:
    def test_cross_scan_symmetry(self):
        # ATBD Eq. 4.4-1: P(delta, beta) = P(delta, -beta).
        for delta in (-1.5, -1.0, -0.5, 0.0):
            for beta in (0.1, 0.5, 1.0):
                assert psf.psf_weight(delta, beta) == pytest.approx(psf.psf_weight(delta, -beta))

    def test_zero_outside_fov(self):
        # Beyond +/- 2a in cross-scan the hexagonal FOV has no response.
        outside = 2.0 * psf.CERES_FOV_HALF_WIDTH_DEG + 0.01
        assert psf.psf_weight(-1.0, outside) == 0.0

    def test_nonnegative_and_peaked(self):
        # The PSF is a non-negative response with a clear interior peak.
        peak = psf.psf_weight(-1.0, 0.0)
        assert peak > 0.0
        assert peak > psf.psf_weight(2.0, 0.0)  # tail is weaker than the core

    def test_vectorized_matches_scalar(self):
        deltas = np.array([-1.5, -1.0, 0.0, 1.0])
        betas = np.array([0.0, 0.3, 0.6, 0.9])
        vec = psf.psf_weight(deltas, betas)
        scalar = np.array([psf.psf_weight(d, b) for d, b in zip(deltas, betas, strict=True)])
        np.testing.assert_allclose(vec, scalar)


class TestPsf95EnergyExtent:
    def test_matches_atbd_table_cutoffs(self):
        # ATBD Table 4.4-2 95%-energy cutoffs: back ~1.25 deg, front ~1.35 deg,
        # cross ~1.27 deg. Our integrated extent should land within a grid step.
        ext = psf.psf_95_energy_extent()
        assert ext.delta_back_deg == pytest.approx(1.25, abs=0.05)
        assert ext.delta_front_deg == pytest.approx(1.35, abs=0.05)
        assert ext.beta_max_deg == pytest.approx(1.27, abs=0.05)

    def test_along_scan_is_asymmetric(self):
        # The detector time-response tail makes the along-scan extent asymmetric.
        ext = psf.psf_95_energy_extent()
        assert ext.delta_front_deg != ext.delta_back_deg

    def test_result_is_cached(self):
        # lru_cache: identical args return the identical object.
        assert psf.psf_95_energy_extent() is psf.psf_95_energy_extent()


class TestStaticFovExtent:
    def test_uniform_on_all_axes(self):
        ext = psf.static_fov_extent(1.0)
        assert ext.delta_back_deg == ext.delta_front_deg == ext.beta_max_deg == 1.0

    def test_defaults_to_libera_fov(self):
        ext = psf.static_fov_extent()
        assert ext.beta_max_deg == psf.LIBERA_FOV_HALFANGLE_DEG


class TestConservativeAlongScanExtent:
    def test_returns_larger_half_extent(self):
        ext = psf.PSFAngularExtent(delta_back_deg=1.2, delta_front_deg=1.5, beta_max_deg=1.0)
        assert psf.conservative_along_scan_extent(ext) == 1.5
