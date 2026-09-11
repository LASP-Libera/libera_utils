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

    def test_fast_path_matches_exact_geodesic(self, monkeypatch):
        # For a normal box the spherical great-circle fast path must agree with the exact
        # ellipsoidal Geod.inv distance to well within the Gaussian stand-in's tolerance.
        lats = np.linspace(-0.1, 0.1, 7)
        lons = np.linspace(19.9, 20.1, 8)
        tile = _tile(lats, lons)
        fast = RadialWeigher().weight_field(tile, 0.0, 20.0).weights
        # Force the exact ellipsoidal path for the same tile by shrinking the fast-path
        # span window so the box is treated as "large".
        monkeypatch.setattr("libera_utils.footprint_matching.weighting._FAST_DISTANCE_MAX_SPAN_DEG", -1.0)
        exact = RadialWeigher().weight_field(tile, 0.0, 20.0).weights
        np.testing.assert_allclose(fast, exact, atol=1e-4)

    def test_polar_tile_uses_exact_path(self, monkeypatch):
        # A near-pole tile must already use the exact Geod path, so forcing it changes
        # nothing (bit-identical), unlike a normal tile.
        lats = np.linspace(86.0, 86.2, 6)
        lons = np.linspace(10.0, 10.3, 6)
        tile = _tile(lats, lons)
        default = RadialWeigher().weight_field(tile, 86.1, 10.15).weights
        monkeypatch.setattr("libera_utils.footprint_matching.weighting._FAST_DISTANCE_MAX_SPAN_DEG", -1.0)
        forced_exact = RadialWeigher().weight_field(tile, 86.1, 10.15).weights
        np.testing.assert_array_equal(default, forced_exact)


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

    def test_cell_ecef_cache_matches_uncached(self):
        # The per-tile ECEF cache is pure memoization: cache on must be bit-identical to off.
        tile, _, _ = self._grid_tile()
        kw = dict(
            altitude_km=835.0,
            viewing_zenith_deg=self.VZA,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            cone_angle_rate=-1.0,
        )
        cached = AngularPSFWeigher(cell_geometry_cache_size=8).weight_field(tile, self.B_LAT, self.B_LON, **kw)
        uncached = AngularPSFWeigher(cell_geometry_cache_size=0).weight_field(tile, self.B_LAT, self.B_LON, **kw)
        np.testing.assert_array_equal(cached.weights, uncached.weights)
        assert cached.total_energy == uncached.total_energy

    def test_repeated_tile_computes_ecef_once(self, monkeypatch):
        # A weigher reused across consecutive footprints on the SAME tile object must compute
        # the (frame-independent) cell ECEF exactly once, while each footprint's frame-dependent
        # field is still computed correctly.
        import libera_utils.footprint_matching.weighting as weighting_mod

        calls = {"n": 0}
        original = weighting_mod.surface_cell_ecef_km

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(weighting_mod, "surface_cell_ecef_km", counting)

        weigher = AngularPSFWeigher(cell_geometry_cache_size=8)
        tile, _, _ = self._grid_tile()
        kw = dict(
            altitude_km=835.0,
            viewing_zenith_deg=self.VZA,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            cone_angle_rate=-1.0,
        )
        w0 = weigher.weight_field(tile, self.B_LAT, self.B_LON, **kw)
        w1 = weigher.weight_field(tile, self.B_LAT + 0.05, self.B_LON, **kw)  # shifted frame

        assert calls["n"] == 1  # ECEF computed once despite two weightings of the same tile
        assert w0.total_energy > 0.0
        # The frame differs (boresight moved), so the weight fields must differ.
        assert not np.allclose(w0.weights, w1.weights)

    def test_distinct_tiles_are_cached_separately(self, monkeypatch):
        # Two different tile objects each get their own ECEF computed (cache keys on identity).
        import libera_utils.footprint_matching.weighting as weighting_mod

        calls = {"n": 0}
        original = weighting_mod.surface_cell_ecef_km

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(weighting_mod, "surface_cell_ecef_km", counting)

        weigher = AngularPSFWeigher(cell_geometry_cache_size=8)
        tile_a, _, _ = self._grid_tile()
        tile_b, _, _ = self._grid_tile()  # same coords, distinct object
        kw = dict(subsatellite_lat_deg=self.S_LAT, subsatellite_lon_deg=self.S_LON, viewing_zenith_deg=self.VZA)
        weigher.weight_field(tile_a, self.B_LAT, self.B_LON, **kw)
        weigher.weight_field(tile_b, self.B_LAT, self.B_LON, **kw)
        weigher.weight_field(tile_a, self.B_LAT, self.B_LON, **kw)  # tile_a again -> still cached
        assert calls["n"] == 2  # one per distinct tile object, tile_a reused from cache
