"""Unit tests for the swappable PSF weighting layer (weighting.py)."""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.types import BoundingBox, GridTile
from libera_utils.footprint_matching.weighting import (
    AngularPSFWeigher,
    PixelWeigher,
    RadialWeigher,
    WeightField,
)


def _tile(lats: np.ndarray, lons: np.ndarray) -> GridTile:
    """A GridTile with matching (n_lat, n_lon) data (values are irrelevant to weighting)."""
    data = np.zeros((lats.size, lons.size), dtype=np.float32)
    # Guard against empty coordinate arrays (the empty-tile case) when deriving bounds.
    if lats.size and lons.size:
        bounds = BoundingBox(float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max()))
    else:
        bounds = BoundingBox(0.0, 0.0, 0.0, 0.0)
    return GridTile(data=data, lats=lats, lons=lons, bounds=bounds, source="test")


class TestRadialWeigher:
    def test_weight_field_shape_matches_grid(self):
        lats = np.array([0.0, 0.1, 0.2])
        lons = np.array([0.0, 0.1])
        wf = RadialWeigher().weight_field(_tile(lats, lons), 0.1, 0.05)
        assert wf.weights.shape == (3, 2)
        assert isinstance(wf, WeightField)

    def test_peak_at_boresight_and_decreasing_outward(self):
        # Boresight sits on the centre cell; its weight must be the maximum.
        lats = np.array([-0.1, 0.0, 0.1])
        lons = np.array([-0.1, 0.0, 0.1])
        wf = RadialWeigher().weight_field(_tile(lats, lons), 0.0, 0.0)
        centre = wf.weights[1, 1]
        assert centre == pytest.approx(wf.weights.max())
        # A corner is farther from the boresight than an edge, so weighs no more.
        assert wf.weights[0, 0] <= wf.weights[0, 1] + 1e-12

    def test_symmetric_about_boresight(self):
        lats = np.array([-0.1, 0.0, 0.1])
        lons = np.array([-0.1, 0.0, 0.1])
        wf = RadialWeigher().weight_field(_tile(lats, lons), 0.0, 0.0)
        # Mirror symmetry in latitude and longitude about the centred boresight.
        np.testing.assert_allclose(wf.weights, wf.weights[::-1, :], rtol=1e-6)
        np.testing.assert_allclose(wf.weights, wf.weights[:, ::-1], rtol=1e-6)

    def test_zero_beyond_ground_radius(self):
        # A far-away cell (many degrees from the boresight) is outside the PSF radius
        # and must receive exactly zero weight.
        lats = np.array([0.0, 40.0])
        lons = np.array([0.0])
        wf = RadialWeigher().weight_field(_tile(lats, lons), 0.0, 0.0)
        assert wf.weights[1, 0] == 0.0
        assert wf.weights[0, 0] > 0.0

    def test_total_energy_is_weight_sum(self):
        lats = np.array([0.0, 0.05])
        lons = np.array([0.0, 0.05])
        wf = RadialWeigher().weight_field(_tile(lats, lons), 0.025, 0.025)
        assert wf.total_energy == pytest.approx(float(wf.weights.sum()))

    def test_high_vza_elongates_ground_radius(self):
        lats = np.array([-0.1, 0.1])
        lons = np.array([-0.1, 0.1])
        nadir = RadialWeigher().weight_field(_tile(lats, lons), 0.0, 0.0)
        oblique = RadialWeigher().weight_field(_tile(lats, lons), 0.0, 0.0, viewing_zenith_deg=70.0)
        assert oblique.max_radius_km > nadir.max_radius_km

    def test_empty_tile_gives_zero_energy(self):
        empty = _tile(np.array([]), np.array([]))
        wf = RadialWeigher().weight_field(empty, 0.0, 0.0)
        assert wf.total_energy == 0.0
        assert wf.weights.size == 0

    def test_conforms_to_interface(self):
        assert isinstance(RadialWeigher(), PixelWeigher)


class TestAngularPSFWeigher:
    """CERES-PSF weighting via the angular-frame projection."""

    # A footprint grid ~0.35 deg half-width around the boresight, with a subsatellite
    # point to the north (so the scan plane is well defined at oblique viewing).
    B_LAT, B_LON = 10.0, 20.0
    S_LAT, S_LON = 11.5, 20.0
    VZA = 25.0

    def _grid_tile(self, half=0.35, n=41):
        lats = np.linspace(self.B_LAT - half, self.B_LAT + half, n)
        lons = np.linspace(self.B_LON - half, self.B_LON + half, n)
        return _tile(lats, lons), lats, lons

    def _wf(self, cone_angle_rate=-1.0):
        tile, _, _ = self._grid_tile()
        return AngularPSFWeigher().weight_field(
            tile,
            self.B_LAT,
            self.B_LON,
            altitude_km=835.0,
            viewing_zenith_deg=self.VZA,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            cone_angle_rate=cone_angle_rate,
        )

    def test_conforms_to_interface(self):
        assert isinstance(AngularPSFWeigher(), PixelWeigher)

    def test_positive_weight_at_boresight_zero_far_away(self):
        tile, lats, lons = self._grid_tile()
        wf = self._wf()
        assert wf.weights.shape == (lats.size, lons.size)
        ci, cj = lats.size // 2, lons.size // 2  # boresight cell
        assert wf.weights[ci, cj] > 0.0
        assert wf.weights[0, 0] == 0.0  # far corner is outside the PSF contour
        assert np.all(wf.weights >= 0.0)

    def test_total_energy_is_weight_sum(self):
        wf = self._wf()
        assert wf.total_energy == pytest.approx(float(wf.weights.sum()))
        assert wf.total_energy > 0.0

    def test_scan_direction_changes_the_field(self):
        # The CERES PSF is asymmetric along-scan, so reversing the cone-angle-rate sign
        # (outward vs inward scan) must produce a different weight field.
        outward = self._wf(cone_angle_rate=+1.0)
        inward = self._wf(cone_angle_rate=-1.0)
        assert not np.allclose(outward.weights, inward.weights)

    def test_differs_from_radial_stand_in(self):
        # Confirms this is genuinely the CERES shape, not the isotropic Gaussian.
        tile, _, _ = self._grid_tile()
        radial = RadialWeigher().weight_field(
            tile, self.B_LAT, self.B_LON, altitude_km=835.0, viewing_zenith_deg=self.VZA
        )
        angular = self._wf()
        # Normalize both to unit sum before comparing spatial shape.
        r = radial.weights / radial.weights.sum()
        a = angular.weights / angular.weights.sum()
        assert not np.allclose(r, a, atol=1e-3)

    def test_stationary_scanner_uses_uniform_static_fov(self):
        # cone_angle_rate ~ 0 -> uniform response inside the FOV (binary weights).
        wf = self._wf(cone_angle_rate=0.0)
        unique = np.unique(wf.weights)
        assert set(unique.tolist()) <= {0.0, 1.0}
        tile, lats, lons = self._grid_tile()
        assert wf.weights[lats.size // 2, lons.size // 2] == 1.0

    def test_empty_tile_gives_zero_energy(self):
        wf = AngularPSFWeigher().weight_field(
            _tile(np.array([]), np.array([])),
            self.B_LAT,
            self.B_LON,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
        )
        assert wf.total_energy == 0.0
        assert wf.weights.size == 0
