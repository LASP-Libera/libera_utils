"""Unit tests for the swappable PSF weighting layer (weighting.py)."""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.geometry import compute_viewing_frames
from libera_utils.footprint_matching.types import BoundingBox, SampleSet
from libera_utils.footprint_matching.weighting import (
    AngularPSFWeigher,
    PixelWeigher,
    RadialWeigher,
    WeightField,
)

# Spacecraft distance from Earth's center (~835 km orbit), km.
_SATELLITE_RADIUS_KM = 7213.0
# Spacecraft geolocation the radial weigher needs only for its ground radius, plus a
# 1 km sample spacing for the coverage reference grid.
_GEOM = dict(
    subsatellite_lat_deg=0.0,
    subsatellite_lon_deg=0.0,
    satellite_radius_km=_SATELLITE_RADIUS_KM,
    sample_resolution_km=1.0,
)


def _grid_samples(lats: np.ndarray, lons: np.ndarray, values: np.ndarray | None = None) -> SampleSet:
    """A SampleSet of every (lat, lon) grid point, row-major, with one zero-valued variable."""
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    n = lat_grid.size
    if values is None:
        values = np.zeros((1, n), dtype=np.float32)
    if n:
        bounds = BoundingBox(float(lats.min()), float(lats.max()), float(lons.min()), float(lons.max()))
    else:
        bounds = BoundingBox(0.0, 0.0, 0.0, 0.0)
    return SampleSet(
        lats=lat_grid.ravel().astype(float),
        lons=lon_grid.ravel().astype(float),
        values=values,
        times=None,
        source="test",
        bounds=bounds,
    )


def _centre_index(lats: np.ndarray, lons: np.ndarray) -> int:
    """Flat index of the central grid point of a row-major (lats x lons) sample set."""
    return (lats.size // 2) * lons.size + lons.size // 2


class TestRadialWeigher:
    def test_weight_field_is_one_weight_per_sample(self):
        lats = np.array([0.0, 0.1, 0.2])
        lons = np.array([0.0, 0.1])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.1, 0.05, **_GEOM)
        assert wf.weights.shape == (6,)
        assert isinstance(wf, WeightField)

    def test_peak_at_boresight_and_decreasing_outward(self):
        # Boresight sits on the centre sample; its weight must be the maximum.
        lats = np.array([-0.1, 0.0, 0.1])
        lons = np.array([-0.1, 0.0, 0.1])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        centre = wf.weights[_centre_index(lats, lons)]
        assert centre == pytest.approx(wf.weights.max())
        # A corner is farther from the boresight than an edge, so weighs no more.
        assert wf.weights[0] <= wf.weights[1] + 1e-12

    def test_symmetric_about_boresight(self):
        lats = np.array([-0.1, 0.0, 0.1])
        lons = np.array([-0.1, 0.0, 0.1])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        grid = wf.weights.reshape(3, 3)
        # Mirror symmetry in latitude and longitude about the centred boresight.
        np.testing.assert_allclose(grid, grid[::-1, :], rtol=1e-6)
        np.testing.assert_allclose(grid, grid[:, ::-1], rtol=1e-6)

    def test_zero_beyond_ground_radius(self):
        # A far-away sample (many degrees from the boresight) is outside the PSF radius
        # and must receive exactly zero weight.
        lats = np.array([0.0, 40.0])
        lons = np.array([0.0])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        assert wf.weights[1] == 0.0
        assert wf.weights[0] > 0.0

    def test_total_energy_is_weight_sum(self):
        lats = np.array([0.0, 0.05])
        lons = np.array([0.0, 0.05])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.025, 0.025, **_GEOM)
        assert wf.total_energy == pytest.approx(float(wf.weights.sum()))

    def test_high_vza_elongates_ground_radius(self):
        lats = np.array([-0.1, 0.1])
        lons = np.array([-0.1, 0.1])
        nadir = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        oblique = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, viewing_zenith_deg=70.0, **_GEOM)
        assert oblique.max_radius_km > nadir.max_radius_km

    def test_empty_samples_give_zero_energy_and_coverage(self):
        empty = _grid_samples(np.array([]), np.array([]))
        wf = RadialWeigher().weight_field(empty, 0.0, 0.0, **_GEOM)
        assert wf.total_energy == 0.0
        assert wf.weights.size == 0
        assert wf.coverage == 0.0

    def test_conforms_to_interface(self):
        assert isinstance(RadialWeigher(), PixelWeigher)

    def test_fast_path_matches_exact_geodesic(self, monkeypatch):
        # For a normal box the local-tangent-plane fast path must agree with the exact
        # ellipsoidal Geod.inv distance to well within the Gaussian stand-in's tolerance.
        lats = np.linspace(-0.1, 0.1, 7)
        lons = np.linspace(19.9, 20.1, 8)
        samples = _grid_samples(lats, lons)
        fast = RadialWeigher().weight_field(samples, 0.0, 20.0, **_GEOM)
        # Force the exact ellipsoidal path for the same samples by shrinking the fast-path
        # span window so the box is treated as "large".
        monkeypatch.setattr("libera_utils.footprint_matching.weighting._FAST_DISTANCE_MAX_SPAN_DEG", -1.0)
        exact = RadialWeigher().weight_field(samples, 0.0, 20.0, **_GEOM)
        np.testing.assert_allclose(fast.weights, exact.weights, atol=1e-4)
        assert fast.coverage == pytest.approx(exact.coverage, abs=1e-6)

    def test_polar_samples_use_exact_path(self, monkeypatch):
        # Near-pole samples must already use the exact Geod path, so forcing it changes
        # nothing (bit-identical), unlike a normal box.
        lats = np.linspace(86.0, 86.2, 6)
        lons = np.linspace(10.0, 10.3, 6)
        samples = _grid_samples(lats, lons)
        default = RadialWeigher().weight_field(samples, 86.1, 10.15, **_GEOM).weights
        monkeypatch.setattr("libera_utils.footprint_matching.weighting._FAST_DISTANCE_MAX_SPAN_DEG", -1.0)
        forced_exact = RadialWeigher().weight_field(samples, 86.1, 10.15, **_GEOM).weights
        np.testing.assert_array_equal(default, forced_exact)


class TestRadialCoverage:
    """Coverage is the share of the kernel's energy in reference bins that hold a finite sample."""

    @staticmethod
    def _dense(half_deg: float, step_deg: float) -> tuple[np.ndarray, np.ndarray]:
        axis = np.arange(-half_deg, half_deg + step_deg / 2, step_deg)
        return axis, axis

    def test_complete_fine_sampling_covers_fully(self):
        # 1 km-spaced samples (~0.009 deg) over the whole ~26 km disc: every bin is hit.
        lats, lons = self._dense(0.35, 0.009)
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        assert wf.coverage == pytest.approx(1.0, abs=0.02)

    def test_half_missing_swath_covers_about_half(self):
        # Samples only on the northern half of the disc: about half the kernel energy.
        lats, lons = self._dense(0.35, 0.009)
        north = lats[lats >= 0.0]
        wf = RadialWeigher().weight_field(_grid_samples(north, lons), 0.0, 0.0, **_GEOM)
        assert 0.4 < wf.coverage < 0.6

    def test_nan_valued_samples_do_not_count(self):
        # Present but all-NaN samples back no energy.
        lats, lons = self._dense(0.35, 0.009)
        samples = _grid_samples(lats, lons)
        nan_values = np.full_like(samples.values, np.nan)
        blank = SampleSet(samples.lats, samples.lons, nan_values, None, "test", samples.bounds)
        wf = RadialWeigher().weight_field(blank, 0.0, 0.0, **_GEOM)
        assert wf.coverage == 0.0

    def test_coarse_source_covers_its_cell(self):
        # One 28 km sample at the boresight covers the central 28 km square of the ~27 km
        # radius disc: most of the Gaussian energy, but not the rim.
        samples = _grid_samples(np.array([0.0]), np.array([0.0]))
        geom = {**_GEOM, "sample_resolution_km": 28.0}
        wf = RadialWeigher().weight_field(samples, 0.0, 0.0, **geom)
        assert 0.4 < wf.coverage < 0.9
        # A 111 km cell swallows the whole disc.
        wide = RadialWeigher().weight_field(samples, 0.0, 0.0, **{**_GEOM, "sample_resolution_km": 111.0})
        assert wide.coverage == pytest.approx(1.0)


class TestAngularPSFWeigher:
    """CERES-PSF weighting via the angular-frame projection."""

    # A footprint grid ~0.35 deg half-width around the boresight, with a subsatellite
    # point to the north (so the scan plane is well defined at oblique viewing).
    B_LAT, B_LON = 10.0, 20.0
    S_LAT, S_LON = 11.5, 20.0
    VZA = 25.0

    def _grid(self, half=0.35, n=41):
        lats = np.linspace(self.B_LAT - half, self.B_LAT + half, n)
        lons = np.linspace(self.B_LON - half, self.B_LON + half, n)
        return _grid_samples(lats, lons), lats, lons

    def _kw(self, cone_angle_rate=-1.0, resolution_km=1.0):
        return dict(
            satellite_radius_km=_SATELLITE_RADIUS_KM,
            viewing_zenith_deg=self.VZA,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            cone_angle_rate=cone_angle_rate,
            sample_resolution_km=resolution_km,
        )

    def _wf(self, cone_angle_rate=-1.0):
        samples, _, _ = self._grid()
        return AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw(cone_angle_rate))

    def test_conforms_to_interface(self):
        assert isinstance(AngularPSFWeigher(), PixelWeigher)

    def test_positive_weight_at_boresight_zero_far_away(self):
        samples, lats, lons = self._grid()
        wf = self._wf()
        assert wf.weights.shape == (lats.size * lons.size,)
        assert wf.weights[_centre_index(lats, lons)] > 0.0
        assert wf.weights[0] == 0.0  # far corner is outside the PSF contour
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
        samples, _, _ = self._grid()
        radial = RadialWeigher().weight_field(samples, self.B_LAT, self.B_LON, viewing_zenith_deg=self.VZA, **_GEOM)
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
        _, lats, lons = self._grid()
        assert wf.weights[_centre_index(lats, lons)] == 1.0

    def test_empty_samples_give_zero_energy_and_coverage(self):
        wf = AngularPSFWeigher().weight_field(
            _grid_samples(np.array([]), np.array([])),
            self.B_LAT,
            self.B_LON,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            satellite_radius_km=_SATELLITE_RADIUS_KM,
            sample_resolution_km=1.0,
        )
        assert wf.total_energy == 0.0
        assert wf.weights.size == 0
        assert wf.coverage == 0.0

    def test_precomputed_frame_matches_inline_frame(self):
        # The orchestrator batches the viewing frames and hands each footprint its own; the
        # weights must equal those from building the frame inline from the geolocation.
        samples, _, _ = self._grid()
        kw = self._kw()
        frames = compute_viewing_frames(
            [self.B_LAT], [self.B_LON], [self.S_LAT], [self.S_LON], satellite_radius_km=_SATELLITE_RADIUS_KM
        )
        inline = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **kw)
        batched = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, frame=frames.for_index(0), **kw)
        np.testing.assert_allclose(batched.weights, inline.weights)
        assert batched.coverage == pytest.approx(inline.coverage)

    def test_complete_fine_sampling_covers_fully(self):
        # A dense 1 km-ish grid (0.009 deg) over the whole contour hits every reference bin.
        samples, _, _ = self._grid(half=0.9, n=201)
        wf = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw())
        assert wf.coverage == pytest.approx(1.0, abs=0.02)

    def test_half_missing_swath_covers_about_half(self):
        # Samples only south of the boresight: roughly half of the PSF energy is backed.
        lats = np.linspace(self.B_LAT - 0.9, self.B_LAT, 101)
        lons = np.linspace(self.B_LON - 0.9, self.B_LON + 0.9, 201)
        wf = AngularPSFWeigher().weight_field(_grid_samples(lats, lons), self.B_LAT, self.B_LON, **self._kw())
        assert 0.3 < wf.coverage < 0.7

    def test_stationary_coverage_uses_the_disc(self):
        samples, _, _ = self._grid(half=0.9, n=201)
        wf = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw(cone_angle_rate=0.0))
        assert wf.coverage == pytest.approx(1.0, abs=0.02)

    def test_coarse_source_covers_from_a_few_samples(self):
        # A 111 km source (the VIIRS composites) has one reference bin wider than the
        # whole contour: any sample inside the contour is a complete match.
        samples = _grid_samples(np.array([self.B_LAT - 0.05, self.B_LAT + 0.05]), np.array([self.B_LON]))
        wf = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw(resolution_km=111.0))
        assert wf.coverage == pytest.approx(1.0)
