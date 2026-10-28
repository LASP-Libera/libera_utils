"""Unit tests for the swappable PSF weighting layer (weighting.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from libera_utils.footprint_matching.geometry import compute_viewing_frames, project_to_angular
from libera_utils.footprint_matching.psf import psf_95_energy_extent, psf_weight
from libera_utils.footprint_matching.types import BoundingBox, SampleSet
from libera_utils.footprint_matching.weighting import (
    _COVERAGE_STEP_KM,
    AngularPSFWeigher,
    PixelWeigher,
    RadialWeigher,
    WeightField,
    _radial_reference,
)

# Spacecraft distance from Earth's center (~835 km orbit), km.
_SATELLITE_RADIUS_KM = 7213.0
# Spacecraft geolocation the radial weigher needs only for its ground radius, a nadir
# view, and a 1 km sample spacing for the coverage reference grid.
_GEOM = dict(
    subsatellite_lat_deg=0.0,
    subsatellite_lon_deg=0.0,
    satellite_radius_km=_SATELLITE_RADIUS_KM,
    sample_resolution_km=1.0,
    viewing_zenith_deg=0.0,
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


def _center_index(lats: np.ndarray, lons: np.ndarray) -> int:
    """Flat index of the central grid point of a row-major (lats x lons) sample set."""
    return (lats.size // 2) * lons.size + lons.size // 2


class TestRadialWeigher:
    def test_weight_field_is_one_weight_per_sample(self):
        """``RadialWeigher.weight_field`` returns a ``WeightField`` with one weight per sample.

        The aggregation engine pairs ``weights[i]`` with sample ``i``. A weight array of another length, or one shaped
        like a grid, would misalign every value with its weight.

        Weighs a 3 x 2 grid of samples and asserts the result is a ``WeightField`` whose ``weights`` is a flat array of
        6.
        """
        lats = np.array([0.0, 0.1, 0.2])
        lons = np.array([0.0, 0.1])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.1, 0.05, **_GEOM)
        assert wf.weights.shape == (6,)
        assert isinstance(wf, WeightField)

    def test_peak_at_boresight_and_decreasing_outward(self):
        """``RadialWeigher`` weighs the boresight sample most, and a farther sample no more than a nearer one.

        The radial stand-in models a centrally peaked instrument response: the ground the instrument looks straight at
        counts most. A kernel peaking anywhere else would shift every footprint value toward an off-center scene.

        Weighs a 3 x 3 grid centered on the boresight. Asserts the center sample has the maximum weight and that a
        corner sample weighs no more than its neighboring edge sample, which is closer to the boresight.
        """
        lats = np.array([-0.1, 0.0, 0.1])
        lons = np.array([-0.1, 0.0, 0.1])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        center = wf.weights[_center_index(lats, lons)]
        assert center == pytest.approx(wf.weights.max())
        assert wf.weights[0] <= wf.weights[1] + 1e-12

    def test_symmetric_about_boresight(self):
        """``RadialWeigher`` weights are mirror-symmetric about the boresight.

        The stand-in kernel depends only on distance from the boresight, unlike the real CERES PSF. Any asymmetry would
        point to an error in the distance calculation, such as a mixed-up latitude or longitude scale.

        Weighs a 3 x 3 grid centered on the boresight, reshapes the weights to the grid, and asserts the grid equals its
        own north-south flip and its own east-west flip.
        """
        lats = np.array([-0.1, 0.0, 0.1])
        lons = np.array([-0.1, 0.0, 0.1])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        grid = wf.weights.reshape(3, 3)
        np.testing.assert_allclose(grid, grid[::-1, :], rtol=1e-6)
        np.testing.assert_allclose(grid, grid[:, ::-1], rtol=1e-6)

    def test_zero_beyond_ground_radius(self):
        """``RadialWeigher`` gives exactly zero weight to a sample beyond the PSF ground radius.

        A reader can serve samples well outside the PSF. Those samples were not seen by the instrument, and even a tiny
        Gaussian tail weight would let their values leak into the footprint and count toward its total energy.

        Weighs one sample at the boresight and one 40 degrees away. Asserts the far sample's weight is exactly 0.0 and
        the boresight sample's weight is positive.
        """
        lats = np.array([0.0, 40.0])
        lons = np.array([0.0])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        assert wf.weights[1] == 0.0
        assert wf.weights[0] > 0.0

    def test_total_energy_is_weight_sum(self):
        """``WeightField.total_energy`` from ``RadialWeigher`` is the sum of its sample weights.

        ``total_energy`` is the PSF energy the served samples carry. If it disagreed with the weights, a caller reading
        it would see a different footprint from the one that was aggregated.

        Weighs a 2 x 2 grid and asserts ``total_energy`` equals ``weights.sum()``.
        """
        lats = np.array([0.0, 0.05])
        lons = np.array([0.0, 0.05])
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.025, 0.025, **_GEOM)
        assert wf.total_energy == pytest.approx(float(wf.weights.sum()))

    def test_high_vza_elongates_ground_radius(self):
        """``RadialWeigher`` uses a larger ground radius at a 70 degree viewing zenith than at nadir.

        An oblique view strikes the ground at a slant and covers a larger area. If the radius stayed at its nadir value,
        the weigher would cut off ground the instrument actually saw at the edge of the scan.

        Weighs the same samples at viewing zenith 0 and 70 degrees and asserts the oblique ``max_radius_km`` is the
        larger.
        """
        lats = np.array([-0.1, 0.1])
        lons = np.array([-0.1, 0.1])
        nadir = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        oblique = RadialWeigher().weight_field(
            _grid_samples(lats, lons), 0.0, 0.0, **{**_GEOM, "viewing_zenith_deg": 70.0}
        )
        assert oblique.max_radius_km > nadir.max_radius_km

    def test_empty_samples_give_zero_energy_and_coverage(self):
        """``RadialWeigher`` returns empty weights, zero total energy and zero coverage for an empty sample set.

        An empty sample set is how a footprint outside a source's coverage arrives. It must yield a well-formed,
        zero-coverage result rather than fail on the minimum or maximum of an empty array.

        Weighs an empty grid and asserts ``total_energy`` is 0, ``weights`` has size 0 and ``coverage`` is 0.
        """
        empty = _grid_samples(np.array([]), np.array([]))
        wf = RadialWeigher().weight_field(empty, 0.0, 0.0, **_GEOM)
        assert wf.total_energy == 0.0
        assert wf.weights.size == 0
        assert wf.coverage == 0.0

    def test_conforms_to_interface(self):
        """``RadialWeigher`` is a ``PixelWeigher``.

        The orchestrator accepts any ``PixelWeigher``, so the radial stand-in and the angular weigher can be swapped
        without touching aggregation code. That only works if both implement the interface.

        Asserts ``isinstance(RadialWeigher(), PixelWeigher)``.
        """
        assert isinstance(RadialWeigher(), PixelWeigher)

    def test_fast_path_matches_exact_geodesic(self, monkeypatch):
        """``RadialWeigher``'s flat-plane distance fast path matches the exact WGS84 geodesic on a normal box.

        For speed, the weigher measures distance on a local tangent plane instead of calling ``pyproj``'s iterative
        geodesic for every footprint. That shortcut is valid only if it agrees with the exact distance over a
        footprint's span.

        Weighs a box under a degree across, once by default (fast path) and once with ``_FAST_DISTANCE_MAX_SPAN_DEG``
        patched to -1, which forces the exact path. Asserts the weights agree to within 1e-4 and the coverage to within
        1e-6.
        """
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
        """``RadialWeigher`` already uses the exact WGS84 geodesic for samples poleward of 85 degrees.

        Near a pole, a degree of longitude shrinks toward zero, and the flat-plane shortcut's constant east-west scale
        no longer holds. Polar footprints must therefore take the exact path whatever the size of their box.

        Weighs a small box at 86 degrees latitude by default and again with the exact path forced. Asserts the two
        weight arrays are bit-for-bit identical, which holds only if the default run was already exact.
        """
        lats = np.linspace(86.0, 86.2, 6)
        lons = np.linspace(10.0, 10.3, 6)
        samples = _grid_samples(lats, lons)
        default = RadialWeigher().weight_field(samples, 86.1, 10.15, **_GEOM).weights
        monkeypatch.setattr("libera_utils.footprint_matching.weighting._FAST_DISTANCE_MAX_SPAN_DEG", -1.0)
        forced_exact = RadialWeigher().weight_field(samples, 86.1, 10.15, **_GEOM).weights
        np.testing.assert_array_equal(default, forced_exact)

    def test_fast_path_across_the_antimeridian(self, monkeypatch):
        """``RadialWeigher``'s fast path measures the short way across the antimeridian.

        When every served sample lies on the far side of the antimeridian from the boresight, the samples' own span
        is small, so the fast path runs. An unwrapped longitude offset would put them 359 degrees away: zero weight
        and zero coverage, with no error.

        Puts the boresight at -179.99 and the samples at 179.6 to 179.99. Asserts some weight is positive and the
        weights and coverage match the exact geodesic path.
        """
        lats = np.linspace(-0.1, 0.1, 7)
        lons = np.linspace(179.6, 179.99, 8)
        samples = _grid_samples(lats, lons)
        fast = RadialWeigher().weight_field(samples, 0.0, -179.99, **_GEOM)
        monkeypatch.setattr("libera_utils.footprint_matching.weighting._FAST_DISTANCE_MAX_SPAN_DEG", -1.0)
        exact = RadialWeigher().weight_field(samples, 0.0, -179.99, **_GEOM)
        assert fast.weights.max() > 0.0
        np.testing.assert_allclose(fast.weights, exact.weights, atol=1e-4)
        assert fast.coverage == pytest.approx(exact.coverage, abs=1e-6)


class TestRadialCoverage:
    """Coverage is the share of the kernel's energy in reference bins that hold a finite sample."""

    @staticmethod
    def _dense(half_deg: float, step_deg: float) -> tuple[np.ndarray, np.ndarray]:
        axis = np.arange(-half_deg, half_deg + step_deg / 2, step_deg)
        return axis, axis

    def test_complete_fine_sampling_covers_fully(self):
        """``RadialWeigher`` coverage is about 1.0 when 1 km samples fill the whole PSF disc.

        Coverage feeds the CERES 75% / 95% rule. A complete source must score 1.0, or good footprints would be flagged
        as partial or discarded.

        Weighs a 0.009 degree (about 1 km) grid that spans the whole disc of about 27 km radius, using a 1 km sample
        resolution, and asserts coverage is 1.0 to within 0.02.
        """
        lats, lons = self._dense(0.35, 0.009)
        wf = RadialWeigher().weight_field(_grid_samples(lats, lons), 0.0, 0.0, **_GEOM)
        assert wf.coverage == pytest.approx(1.0, abs=0.02)

    def test_half_missing_swath_covers_about_half(self):
        """``RadialWeigher`` coverage is about 0.5 when samples fill only the northern half of the PSF disc.

        A swath edge or data gap shows up as missing samples, not as NaN-valued ones. Coverage must therefore come from
        where samples lie relative to the PSF, so that a half-covered footprint is reported as half covered.

        Weighs the same dense grid as the full-coverage test, cut to latitudes at or north of the boresight. Asserts
        coverage is between 0.4 and 0.6.
        """
        lats, lons = self._dense(0.35, 0.009)
        north = lats[lats >= 0.0]
        wf = RadialWeigher().weight_field(_grid_samples(north, lons), 0.0, 0.0, **_GEOM)
        assert 0.4 < wf.coverage < 0.6

    def test_nan_valued_samples_do_not_count(self):
        """``RadialWeigher`` coverage is 0.0 when every sample is present but NaN.

        Readers mark fill or invalid data as NaN. A footprint whose samples are all fill has no usable data, so it must
        not pass the coverage rule just because sample locations exist.

        Weighs the full dense grid with every value replaced by NaN and asserts coverage is exactly 0.0.
        """
        lats, lons = self._dense(0.35, 0.009)
        samples = _grid_samples(lats, lons)
        nan_values = np.full_like(samples.values, np.nan)
        blank = SampleSet(samples.lats, samples.lons, nan_values, None, "test", samples.bounds)
        wf = RadialWeigher().weight_field(blank, 0.0, 0.0, **_GEOM)
        assert wf.coverage == 0.0

    def test_coarse_source_covers_its_cell(self):
        """A single coarse ``RadialWeigher`` sample covers the reference square its native cell spans.

        Coarse sources such as the 1 degree VIIRS composites have only one or a few samples per footprint. Coverage must
        credit each sample with the ground its cell represents, not a single point, or a coarse source could never pass
        the 75% rule.

        Weighs one sample at the boresight. With ``sample_resolution_km=28``, the sample covers a square of half-width
        28 / sqrt(2), about 19.8 km, inside a disc of about 27 km radius, and the test asserts coverage is within 0.02 of
        the truncated kernel's energy inside that square; the inscribed 14 km square would give about 0.57. With 111 km,
        the square encloses the whole disc, and the test asserts coverage is 1.0.
        """
        samples = _grid_samples(np.array([0.0]), np.array([0.0]))
        geom = {**_GEOM, "sample_resolution_km": 28.0}
        wf = RadialWeigher().weight_field(samples, 0.0, 0.0, **geom)
        reference = _radial_reference(wf.max_radius_km)
        centers = -wf.max_radius_km + (np.arange(reference.shape[0]) + 0.5) * _COVERAGE_STEP_KM
        east, north = np.meshgrid(centers, centers)
        half_width_km = 28.0 * math.sqrt(2.0) / 2.0
        inside = (np.abs(east) <= half_width_km) & (np.abs(north) <= half_width_km)
        assert wf.coverage == pytest.approx(reference[inside].sum() / reference.sum(), abs=0.02)
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
        """``AngularPSFWeigher`` is a ``PixelWeigher``.

        The orchestrator accepts any ``PixelWeigher``, so the CERES-faithful weigher must implement the same interface
        as the radial stand-in it can replace.

        Asserts ``isinstance(AngularPSFWeigher(), PixelWeigher)``.
        """
        assert isinstance(AngularPSFWeigher(), PixelWeigher)

    def test_positive_weight_at_boresight_zero_far_away(self):
        """``AngularPSFWeigher`` weighs the boresight above 0, outside the 95% contour 0, and nothing below 0.

        The weigher evaluates the instrument's point spread function at each sample. A sample outside the contour lies
        outside what the instrument measured and must not contribute, and a negative weight would have no physical
        meaning.

        Weighs a 41 x 41 grid about 0.35 degrees across each side of an oblique boresight. Asserts there is one weight
        per sample, the center weight is above 0, the far corner weight is exactly 0, and all weights are at least 0.
        """
        samples, lats, lons = self._grid()
        wf = self._wf()
        assert wf.weights.shape == (lats.size * lons.size,)
        assert wf.weights[_center_index(lats, lons)] > 0.0
        assert wf.weights[0] == 0.0  # far corner is outside the PSF contour
        assert np.all(wf.weights >= 0.0)

    def test_total_energy_is_weight_sum(self):
        """``WeightField.total_energy`` from ``AngularPSFWeigher`` is the sum of its sample weights and is positive.

        ``total_energy`` is the PSF energy the served samples carry. A zero value for a footprint that clearly has
        samples inside the PSF would mean the weights were lost.

        Weighs the standard grid and asserts ``total_energy`` equals ``weights.sum()`` and is greater than 0.
        """
        wf = self._wf()
        assert wf.total_energy == pytest.approx(float(wf.weights.sum()))
        assert wf.total_energy > 0.0

    def test_scan_direction_sets_the_tail_side(self):
        """``AngularPSFWeigher`` puts the PSF's weight centroid on the side the scan direction calls for.

        The CERES PSF has a trailing tail along the scan, from the detector's lagging response, so it points one way
        when the scanner moves outward and the other way when it moves inward. A field that merely differs between the
        two directions would still pass with the tail on the wrong side.

        Projects the grid into the angular frame and takes the weight centroid in ``delta``. Asserts the inward scan
        (rate -1, ``delta`` used as is) puts it on the same side as the PSF's own centroid, and the outward scan
        (rate +1, ``delta`` reversed) on the opposite side.
        """
        samples, _, _ = self._grid()
        delta, _ = project_to_angular(
            samples.lats,
            samples.lons,
            boresight_lat_deg=self.B_LAT,
            boresight_lon_deg=self.B_LON,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            satellite_radius_km=_SATELLITE_RADIUS_KM,
        )
        extent = psf_95_energy_extent(0.95)
        grid_delta, grid_beta = np.meshgrid(
            np.linspace(-extent.delta_back_deg, extent.delta_front_deg, 201),
            np.linspace(-extent.beta_max_deg, extent.beta_max_deg, 201),
        )
        psf = psf_weight(grid_delta, grid_beta)
        psf_side = np.sign(np.sum(psf * grid_delta))
        assert psf_side != 0.0

        inward = self._wf(cone_angle_rate=-1.0).weights
        outward = self._wf(cone_angle_rate=+1.0).weights
        assert np.sign(np.sum(inward * delta)) == psf_side
        assert np.sign(np.sum(outward * delta)) == -psf_side

    def test_missing_cone_angle_rate_raises(self):
        """``AngularPSFWeigher`` raises ``ValueError`` when ``cone_angle_rate`` is missing or not finite.

        The rate's sign sets which side of the footprint the PSF's tail weights. Without it the weigher would have to
        guess a scan direction, and a wrong guess weights the wrong side silently.

        Asserts a ``ValueError`` matching "Cone_Angle_Rate" for ``None`` and for NaN.
        """
        samples, _, _ = self._grid()
        for rate in (None, float("nan")):
            with pytest.raises(ValueError, match="Cone_Angle_Rate"):
                AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw(cone_angle_rate=rate))

    def test_non_finite_viewing_zenith_raises(self):
        """``AngularPSFWeigher`` raises ``ValueError`` for a NaN viewing zenith.

        The viewing zenith sizes each sample's covering cell, so a NaN would turn the coverage into nonsense.

        Asserts a ``ValueError`` matching "viewing zenith" for ``viewing_zenith_deg=nan``.
        """
        samples, _, _ = self._grid()
        kw = {**self._kw(), "viewing_zenith_deg": float("nan")}
        with pytest.raises(ValueError, match="viewing zenith"):
            AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **kw)

    def test_differs_from_radial_stand_in(self):
        """``AngularPSFWeigher`` weights differ in shape from ``RadialWeigher`` weights for the same footprint.

        The angular weigher exists to apply the real asymmetric CERES PSF. If its output matched the isotropic Gaussian
        stand-in, the science-accurate path would add nothing.

        Weighs the same grid with both weighers at the same viewing zenith, normalizes each weight array to sum to 1,
        and asserts they differ by more than 1e-3 somewhere.
        """
        samples, _, _ = self._grid()
        radial = RadialWeigher().weight_field(
            samples, self.B_LAT, self.B_LON, **{**_GEOM, "viewing_zenith_deg": self.VZA}
        )
        angular = self._wf()
        # Normalize both to unit sum before comparing spatial shape.
        r = radial.weights / radial.weights.sum()
        a = angular.weights / angular.weights.sum()
        assert not np.allclose(r, a, atol=1e-3)

    def test_stationary_scanner_uses_uniform_static_fov(self):
        """``AngularPSFWeigher`` uses a flat, on-or-off field-of-view response when ``cone_angle_rate`` is zero.

        At a scan turnaround the scanner is momentarily still, so the moving-scan PSF and its trailing tail do not
        apply. The response is then the instrument's static circular field of view, in which every point inside counts
        equally.

        Weighs the grid with ``cone_angle_rate=0`` and asserts every weight is 0 or 1 and the boresight sample's weight
        is 1.
        """
        wf = self._wf(cone_angle_rate=0.0)
        unique = np.unique(wf.weights)
        assert set(unique.tolist()) <= {0.0, 1.0}
        _, lats, lons = self._grid()
        assert wf.weights[_center_index(lats, lons)] == 1.0

    def test_empty_samples_give_zero_energy_and_coverage(self):
        """``AngularPSFWeigher`` returns empty weights, zero total energy and zero coverage for an empty sample set.

        A footprint outside a source's coverage arrives with no samples. The weigher must return a well-formed,
        zero-coverage result without trying to project an empty array into the angular frame.

        Weighs an empty grid and asserts ``total_energy`` is 0, ``weights`` has size 0 and ``coverage`` is 0.
        """
        wf = AngularPSFWeigher().weight_field(
            _grid_samples(np.array([]), np.array([])),
            self.B_LAT,
            self.B_LON,
            subsatellite_lat_deg=self.S_LAT,
            subsatellite_lon_deg=self.S_LON,
            satellite_radius_km=_SATELLITE_RADIUS_KM,
            sample_resolution_km=1.0,
            viewing_zenith_deg=self.VZA,
            cone_angle_rate=-1.0,
        )
        assert wf.total_energy == 0.0
        assert wf.weights.size == 0
        assert wf.coverage == 0.0

    def test_precomputed_frame_matches_inline_frame(self):
        """``AngularPSFWeigher`` gives the same result with a precomputed viewing frame as with its own.

        For speed, the orchestrator builds the viewing frames for all footprints at once and passes each footprint its
        own. That optimization must not change any weight.

        Builds the frame with ``compute_viewing_frames`` and passes ``frames.for_index(0)`` as ``frame``. Asserts the
        weights and coverage match a call with no ``frame``.
        """
        samples, _, _ = self._grid()
        kw = self._kw()
        frames = compute_viewing_frames(
            [self.B_LAT], [self.B_LON], [self.S_LAT], [self.S_LON], satellite_radius_km=[_SATELLITE_RADIUS_KM]
        )
        inline = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **kw)
        batched = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, frame=frames.for_index(0), **kw)
        np.testing.assert_allclose(batched.weights, inline.weights)
        assert batched.coverage == pytest.approx(inline.coverage)

    def test_complete_fine_sampling_covers_fully(self):
        """``AngularPSFWeigher`` coverage is about 1.0 when a dense grid spans the whole PSF contour.

        A complete source must score full coverage under the CERES 75% / 95% rule, or good footprints would be flagged
        or discarded.

        Weighs a 201 x 201 grid, about 0.009 degrees (1 km) apart and spanning 0.9 degrees each side of the boresight,
        and asserts coverage is 1.0 to within 0.02.
        """
        samples, _, _ = self._grid(half=0.9, n=201)
        wf = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw())
        assert wf.coverage == pytest.approx(1.0, abs=0.02)

    def test_half_missing_swath_covers_about_half(self):
        """``AngularPSFWeigher`` coverage drops to about half when samples lie only on the south side of the boresight.

        Missing samples, such as a swath edge, must lower coverage according to how much of the PSF's energy they would
        have carried. Otherwise a half-observed footprint would pass as complete.

        Weighs a dense grid cut off at the boresight latitude and asserts coverage is between 0.3 and 0.7. The band is
        wider than the radial test's because the asymmetric PSF does not split its energy evenly across the cut.
        """
        lats = np.linspace(self.B_LAT - 0.9, self.B_LAT, 101)
        lons = np.linspace(self.B_LON - 0.9, self.B_LON + 0.9, 201)
        wf = AngularPSFWeigher().weight_field(_grid_samples(lats, lons), self.B_LAT, self.B_LON, **self._kw())
        assert 0.3 < wf.coverage < 0.7

    def test_stationary_coverage_uses_the_disc(self):
        """``AngularPSFWeigher`` coverage for a stationary scanner is about 1.0 over a dense, complete grid.

        The stationary case measures coverage against the field-of-view disc rather than the moving-scan PSF contour. A
        complete source must still score 1.0 against it.

        Weighs the dense 201 x 201 grid with ``cone_angle_rate=0`` and asserts coverage is 1.0 to within 0.02.
        """
        samples, _, _ = self._grid(half=0.9, n=201)
        wf = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw(cone_angle_rate=0.0))
        assert wf.coverage == pytest.approx(1.0, abs=0.02)

    def test_coarse_source_covers_from_a_few_samples(self):
        """``AngularPSFWeigher`` coverage is 1.0 for a 111 km source with only two samples near the boresight.

        The 1 degree VIIRS composites have cells wider than the whole PSF contour, so any sample inside the contour
        stands for all of it. Without crediting each sample with its cell, such a source would fail the coverage rule
        everywhere.

        Weighs two samples 0.05 degrees either side of the boresight with ``sample_resolution_km=111`` and asserts
        coverage is exactly 1.0.
        """
        samples = _grid_samples(np.array([self.B_LAT - 0.05, self.B_LAT + 0.05]), np.array([self.B_LON]))
        wf = AngularPSFWeigher().weight_field(samples, self.B_LAT, self.B_LON, **self._kw(resolution_km=111.0))
        assert wf.coverage == pytest.approx(1.0)

    def test_oblique_view_does_not_cover_a_missing_row(self):
        """``AngularPSFWeigher`` coverage drops below 1.0 for a one-row gap at a 60 degree viewing zenith.

        Along the scan plane an oblique view foreshortens each ground cell by ``cos(VZA)``. A covering square sized as
        if seen face-on would be twice as tall as the cell at 60 degrees and bridge a missing row of samples, so a
        footprint with a real gap would pass the coverage rule.

        Places the sub-satellite point 10 degrees north of the boresight (a 60 degree view, the scan plane north-south)
        and weighs a 5 km grid. Asserts the complete grid covers 1.0 and the same grid missing the boresight's row of
        latitude covers less than 0.995.
        """
        kw = dict(
            subsatellite_lat_deg=self.B_LAT + 10.0,
            subsatellite_lon_deg=self.B_LON,
            satellite_radius_km=_SATELLITE_RADIUS_KM,
            viewing_zenith_deg=60.0,
            cone_angle_rate=-1.0,
            sample_resolution_km=5.0,
        )
        lats = np.arange(self.B_LAT - 1.5, self.B_LAT + 1.5 + 1e-9, 0.045)
        lons = np.arange(self.B_LON - 1.0, self.B_LON + 1.0 + 1e-9, 0.0457)
        gapped_lats = np.delete(lats, np.argmin(np.abs(lats - self.B_LAT)))
        full = AngularPSFWeigher().weight_field(_grid_samples(lats, lons), self.B_LAT, self.B_LON, **kw)
        gapped = AngularPSFWeigher().weight_field(_grid_samples(gapped_lats, lons), self.B_LAT, self.B_LON, **kw)
        assert full.coverage == pytest.approx(1.0, abs=0.01)
        assert gapped.coverage < 0.995
