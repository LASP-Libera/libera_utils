"""Unit tests for the geometry module (libera_utils.footprint_matching.geometry).

Coverage:
- WGS84 internals (geodetic<->ECEF round-trip, ellipsoid normal, ray-ellipsoid
  intersection) and placing the satellite from the L1B subsatellite point and radius
- the public bounding-box entry point: nadir vs stretched oblique footprints,
  enclosure of the boresight, growth toward the limb
- edge cases: pole enclosure, dateline crossing, off-limb (raise/clamp), fill
  values, inconsistent geolocation, and invalid arguments
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from libera_utils.footprint_matching import geometry as geo
from libera_utils.footprint_matching import psf
from libera_utils.footprint_matching.geometry import (
    WGS84_SEMI_MAJOR_AXIS_KM,
    WGS84_SEMI_MINOR_AXIS_KM,
    MissingGeolocationError,
    OffLimbError,
    PartialFootprintError,
    compute_footprint_bounding_box,
    compute_footprint_bounding_boxes,
    compute_viewing_frames,
    psf_ground_radius_km,
)

# A representative spacecraft geodetic altitude for synthetic cases (JPSS orbit), km.
ALT = 835.0


def _radius(sub_lat, sub_lon, altitude_km=ALT):
    """L1B-style ``Radius_of_Satellite_from_Center_of_Earth`` for a satellite at ``altitude_km``."""
    sub_lat = np.atleast_1d(np.asarray(sub_lat, dtype=float))
    sub_lon = np.atleast_1d(np.asarray(sub_lon, dtype=float))
    satellite = geo._geodetic_to_ecef_batch(sub_lat, sub_lon, np.full_like(sub_lat, altitude_km))
    radius = np.linalg.norm(satellite, axis=1)
    return float(radius[0]) if radius.size == 1 else radius


def _box(lat, lon, sub_lat, sub_lon, **kwargs):
    """Scalar box for a satellite at :data:`ALT` over ``(sub_lat, sub_lon)``."""
    return compute_footprint_bounding_box(
        lat, lon, sub_lat, sub_lon, satellite_radius_km=_radius(sub_lat, sub_lon), **kwargs
    )


def _vza(lat, lon, sub_lat, sub_lon):
    """Viewing zenith (deg) at the footprint, from the satellite :func:`_box` places."""
    _satellite, boresight = geo._viewing_geometry(lat, lon, sub_lat, sub_lon, _radius(sub_lat, sub_lon))
    return math.degrees(math.acos(float(-boresight @ geo._ellipsoid_normal(lat, lon))))


class TestGeodeticEcefRoundTrip:
    @pytest.mark.parametrize(
        ("lat", "lon", "height_km"),
        [(0.0, 0.0, 0.0), (45.0, -75.0, 0.0), (89.9, 120.0, 0.0), (-30.0, 179.0, ALT)],
    )
    def test_round_trip(self, lat, lon, height_km):
        xyz = geo._geodetic_to_ecef(lat, lon, height_km)
        rlat, rlon, rheight = geo._ecef_to_geodetic(xyz)
        assert rlat == pytest.approx(lat, abs=1e-7)
        assert rlon == pytest.approx(lon, abs=1e-7)
        assert rheight == pytest.approx(height_km, abs=1e-4)

    def test_equator_radius_is_semi_major(self):
        xyz = geo._geodetic_to_ecef(0.0, 0.0, 0.0)
        assert np.linalg.norm(xyz) == pytest.approx(WGS84_SEMI_MAJOR_AXIS_KM, abs=1e-6)

    def test_pole_radius_is_semi_minor(self):
        xyz = geo._geodetic_to_ecef(90.0, 0.0, 0.0)
        assert np.linalg.norm(xyz) == pytest.approx(WGS84_SEMI_MINOR_AXIS_KM, abs=1e-6)


class TestEllipsoidNormal:
    def test_unit_length(self):
        assert np.linalg.norm(geo._ellipsoid_normal(37.0, -100.0)) == pytest.approx(1.0)

    def test_radial_at_equator_and_pole(self):
        # On the equator and at the poles the geodetic normal equals the geocentric
        # radial direction.
        eq = geo._geodetic_to_ecef(0.0, 0.0, 0.0)
        assert np.allclose(geo._ellipsoid_normal(0.0, 0.0), eq / np.linalg.norm(eq), atol=1e-9)
        assert np.allclose(geo._ellipsoid_normal(90.0, 0.0), [0.0, 0.0, 1.0], atol=1e-9)

    def test_geodetic_differs_from_geocentric_at_mid_latitude(self):
        # Off the equator the geodetic normal is NOT the geocentric radial direction;
        # this is exactly what the spherical model got wrong.
        lat = 45.0
        normal = geo._ellipsoid_normal(lat, 0.0)
        radial = geo._geodetic_to_ecef(lat, 0.0, 0.0)
        radial = radial / np.linalg.norm(radial)
        angle_deg = math.degrees(math.acos(np.clip(np.dot(normal, radial), -1.0, 1.0)))
        assert angle_deg > 0.1


class TestRayEllipsoidIntersection:
    def test_straight_down_hits_below_point(self):
        # A ray from above (0, 0) pointing straight down hits near (0, 0).
        origin = geo._geodetic_to_ecef(0.0, 0.0, ALT)
        direction = -geo._ellipsoid_normal(0.0, 0.0)
        hit = geo._ray_ellipsoid_intersection(origin, direction)
        assert hit is not None
        lat, lon, height = geo._ecef_to_geodetic(hit)
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert lon == pytest.approx(0.0, abs=1e-6)
        assert height == pytest.approx(0.0, abs=1e-6)

    def test_ray_into_space_misses(self):
        origin = geo._geodetic_to_ecef(0.0, 0.0, ALT)
        direction = geo._ellipsoid_normal(0.0, 0.0)  # pointing up, away from Earth
        assert geo._ray_ellipsoid_intersection(origin, direction) is None


class TestViewingGeometry:
    @pytest.mark.parametrize(("sub_lat", "sub_lon"), [(2.0, 0.0), (45.0, -75.0), (-70.0, 120.0), (89.0, 10.0)])
    def test_satellite_placed_from_subpoint_and_radius(self, sub_lat, sub_lon):
        # The satellite is the point at the given radius on the ellipsoid normal through
        # the geodetic sub-point, i.e. it has that sub-point and the altitude implied.
        satellite = geo._satellite_from_subpoint_batch(
            np.array([sub_lat]), np.array([sub_lon]), np.array([_radius(sub_lat, sub_lon)])
        )[0]
        rlat, rlon, rheight = geo._ecef_to_geodetic(satellite)
        assert rlat == pytest.approx(sub_lat, abs=1e-9)
        assert rlon == pytest.approx(sub_lon, abs=1e-9)
        assert rheight == pytest.approx(ALT, abs=1e-6)

    def test_boresight_points_at_footprint(self):
        satellite, direction = geo._viewing_geometry(10.0, 20.0, 12.0, 20.0, _radius(12.0, 20.0))
        ground = geo._geodetic_to_ecef(10.0, 20.0, 0.0)
        expected = (ground - satellite) / np.linalg.norm(ground - satellite)
        assert np.allclose(direction, expected, atol=1e-12)

    def test_nadir_needs_no_special_case(self):
        # Footprint directly below the satellite: the boresight is the downward normal.
        _satellite, direction = geo._viewing_geometry(30.0, 40.0, 30.0, 40.0, _radius(30.0, 40.0))
        assert np.allclose(direction, -geo._ellipsoid_normal(30.0, 40.0), atol=1e-12)


class TestComputeBoundingBoxBasics:
    def test_nadir_box_is_small_and_encloses_boresight(self):
        bb = _box(0.0, 0.0, 0.0, 0.0)
        assert bb.lat_min <= 0.0 <= bb.lat_max
        assert bb.lon_min <= 0.0 <= bb.lon_max
        # Nadir footprint is small (well under a degree across).
        assert (bb.lat_max - bb.lat_min) < 1.0
        assert not bb.is_polar
        assert not bb.wraps_dateline

    def test_oblique_box_is_much_larger_than_nadir(self):
        nadir = _box(0.0, 0.0, 0.5, 0.0)
        oblique = _box(0.0, 0.0, 18.0, 0.0)  # VZA ~78 deg
        nadir_span = (nadir.lat_max - nadir.lat_min) + (nadir.lon_max - nadir.lon_min)
        oblique_span = (oblique.lat_max - oblique.lat_min) + (oblique.lon_max - oblique.lon_min)
        assert oblique_span > 3.0 * nadir_span

    def test_box_grows_monotonically_toward_the_limb(self):
        # The footprint stretches as the footprint moves away from the subsatellite point.
        prev = 0.0
        for distance_deg in (1.0, 5.0, 10.0, 15.0, 19.0):
            bb = _box(0.0, 0.0, distance_deg, 0.0)
            span = (bb.lat_max - bb.lat_min) + (bb.lon_max - bb.lon_min)
            assert span > prev
            prev = span

    def test_along_scan_axis_follows_subsatellite_bearing(self):
        # Subsatellite due south -> along-scan is N-S, so the latitude span (along)
        # exceeds the longitude span (cross) at an oblique view.
        bb = _box(45.0, 10.0, 30.0, 10.0)
        lat_span_km = (bb.lat_max - bb.lat_min) * 111.0
        lon_span_km = (bb.lon_max - bb.lon_min) * 111.0 * math.cos(math.radians(45.0))
        assert lat_span_km > lon_span_km


def _psf_contour_angles(energy_fraction: float = 0.95, step_deg: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """(delta, beta) of every angular cell inside the PSF energy contour, both scan directions."""
    delta_axis = np.arange(-3.0, 3.0 + step_deg, step_deg)
    beta_axis = np.arange(-1.4, 1.4 + step_deg, step_deg)
    delta, beta = np.meshgrid(delta_axis, beta_axis)
    density = (psf.psf_weight(delta, beta) * np.cos(np.radians(beta))).ravel()
    order = np.argsort(density)[::-1]
    cumulative = np.cumsum(density[order])
    inside = order[: int(np.searchsorted(cumulative, energy_fraction * cumulative[-1])) + 1]
    d, b = delta.ravel()[inside], beta.ravel()[inside]
    # The box is scan-direction independent, so the contour must fit either way round.
    return np.concatenate([d, -d]), np.concatenate([b, b])


class TestPerimeterCoversPsfContour:
    """The perimeter traces the PSF extent rectangle, so its corners are not cut off."""

    def test_samples_include_rectangle_corners(self):
        delta, beta = geo._perimeter_angles(1.4, 1.3, geo._N_PERIMETER_SAMPLES)
        corners = {(round(d, 9), round(b, 9)) for d, b in zip(delta, beta, strict=True)}
        for corner in [(1.4, 1.3), (-1.4, 1.3), (-1.4, -1.3), (1.4, -1.3)]:
            assert corner in corners

    @pytest.mark.parametrize(
        ("sub_lat", "sub_lon"),
        [(15.0, 20.0), (13.5, 23.5), (12.5, 24.3), (10.0, 25.0)],
        ids=["az0", "az45", "az60", "az90"],
    )
    def test_box_encloses_ray_traced_psf_contour(self, sub_lat, sub_lon):
        # Ray-trace every cell of the 95% PSF contour onto the ground and check it lies in
        # the box. With an inscribed-ellipse perimeter the contour pokes out of the box at
        # scan azimuths ~30-60 deg from north.
        lat, lon = 10.0, 20.0
        box = _box(lat, lon, sub_lat, sub_lon)

        satellite, boresight = geo._viewing_geometry(lat, lon, sub_lat, sub_lon, _radius(sub_lat, sub_lon))
        cross_axis = geo._scan_frame_axes(boresight, satellite)
        delta, beta = _psf_contour_angles()
        n = delta.size
        directions = geo._offset_ray_direction_batch(
            np.broadcast_to(boresight, (n, 3)), np.broadcast_to(cross_axis, (n, 3)), delta, beta
        )
        points, hit = geo._ray_ellipsoid_intersection_batch(np.broadcast_to(satellite, (n, 3)), directions)
        assert hit.all()
        contour_lat, contour_lon, _ = geo._ecef_to_geodetic_batch(points)

        assert contour_lat.min() >= box.lat_min
        assert contour_lat.max() <= box.lat_max
        assert contour_lon.min() >= box.lon_min
        assert contour_lon.max() <= box.lon_max


class TestComputeBoundingBoxEdgeCases:
    def test_pole_enclosure(self):
        # A footprint very close to the North pole encloses it: full longitude range,
        # latitude pinned to 90, flagged polar.
        bb = _box(89.9, 0.0, 88.0, 0.0)
        assert bb.is_polar
        assert bb.lat_max == pytest.approx(90.0)
        assert bb.lon_min == pytest.approx(-180.0)
        assert bb.lon_max == pytest.approx(180.0)

    def test_dateline_crossing(self):
        bb = _box(0.0, 179.8, 0.0, 170.0)
        assert bb.wraps_dateline
        # By convention a wrapping box is represented with lon_max > 180.
        assert bb.lon_max > 180.0

    def test_boxes_away_from_dateline_do_not_wrap(self):
        # Away from 0/180 deg both longitude representations have the same span; rounding in
        # lon % 360 must not tip such a box into the wrapped [0, 360) form.
        rng = np.random.default_rng(1)
        n = 300
        lat = rng.uniform(-60.0, 60.0, n)
        lon = rng.uniform(-170.0, -10.0, n)
        sub_lat = lat + rng.uniform(-5.0, 5.0, n)
        sub_lon = lon + rng.uniform(-5.0, 5.0, n)
        radius = _radius(sub_lat, sub_lon)
        batch = compute_footprint_bounding_boxes(lat, lon, sub_lat, sub_lon, satellite_radius_km=radius)
        for i in range(n):
            scalar = compute_footprint_bounding_box(
                lat[i], lon[i], sub_lat[i], sub_lon[i], satellite_radius_km=radius[i]
            )
            assert not scalar.wraps_dateline
            assert not batch[i].wraps_dateline
            assert scalar.lon_max < 0.0

    @pytest.mark.parametrize(
        ("lat", "lon"),
        [(float("nan"), 0.0), (0.0, float("nan")), (geo.L1B_FILL_VALUE, 0.0), (9.96921e36, 0.0)],
        ids=["nan-lat", "nan-lon", "fill-lat", "netcdf-fill-lat"],
    )
    def test_missing_boresight_is_an_off_earth_view(self, lat, lon):
        # L1B RAD writes NaN latitude/longitude where the boresight misses the Earth.
        with pytest.raises(OffLimbError) as excinfo:
            _box(lat, lon, 10.0, 0.0)
        assert not isinstance(excinfo.value, PartialFootprintError)

    @pytest.mark.parametrize(
        ("sub_lat", "sub_lon"),
        [(geo.L1B_FILL_VALUE, 0.0), (10.0, geo.L1B_FILL_VALUE), (float("nan"), 0.0), (9.96921e36, 0.0)],
        ids=["fill-lat", "fill-lon", "nan-lat", "netcdf-fill-lat"],
    )
    def test_invalid_subsatellite_raises(self, sub_lat, sub_lon):
        # Missing spacecraft geolocation is not an off-Earth view, so not an OffLimbError.
        with pytest.raises(MissingGeolocationError) as excinfo:
            compute_footprint_bounding_box(0.0, 0.0, sub_lat, sub_lon, satellite_radius_km=_radius(10.0, 0.0))
        assert not isinstance(excinfo.value, OffLimbError)

    @pytest.mark.parametrize("radius", [-9999.0, float("nan"), 6000.0], ids=["fill", "nan", "below-surface"])
    def test_invalid_radius_raises(self, radius):
        with pytest.raises(MissingGeolocationError):
            compute_footprint_bounding_box(0.0, 0.0, 5.0, 0.0, satellite_radius_km=radius)

    def test_footprint_beyond_the_horizon_raises(self):
        # 30 deg from the subsatellite point is past the horizon of an 835 km orbit: the
        # satellite could not have seen it, so the inputs disagree.
        with pytest.raises(MissingGeolocationError, match="not visible"):
            _box(0.0, 0.0, 30.0, 0.0)

    def test_radius_is_required(self):
        with pytest.raises(TypeError):
            compute_footprint_bounding_box(0.0, 0.0, 5.0, 0.0)  # type: ignore[call-arg]

    def test_invalid_on_limb_argument(self):
        with pytest.raises(ValueError, match="on_limb"):
            _box(0.0, 0.0, 0.5, 0.0, on_limb="bogus")


class TestPartialOffLimb:
    # 27 deg from the subsatellite point is VZA ~89 deg: the boresight is on Earth but the
    # limb-ward box corners are not.
    SEVERE = (0.0, 0.0, 27.0, 0.0)

    def test_default_flags_partial_coverage(self):
        assert _box(*self.SEVERE).truncated is True

    def test_raise_mode_raises_partial_footprint_error(self):
        with pytest.raises(PartialFootprintError):
            _box(*self.SEVERE, on_limb="raise")

    def test_partial_is_offlimb_subclass(self):
        # Callers using `except OffLimbError` must still catch the partial case.
        with pytest.raises(OffLimbError) as excinfo:
            _box(*self.SEVERE, on_limb="raise")
        assert isinstance(excinfo.value, PartialFootprintError)

    def test_moderate_angle_not_truncated(self):
        # Regression guard against false positives well inside the limb (VZA ~72 deg).
        assert _box(0.0, 0.0, 15.0, 0.0).truncated is False

    def test_threshold_band(self):
        # The corner-based check engages just past VZA ~80 deg (well before the pure
        # along-scan edge would, which only happens near 90 deg).
        assert _vza(0.0, 0.0, 19.0, 0.0) == pytest.approx(79.2, abs=0.1)
        assert _vza(0.0, 0.0, 21.0, 0.0) == pytest.approx(82.1, abs=0.1)
        assert _box(0.0, 0.0, 19.0, 0.0).truncated is False
        assert _box(0.0, 0.0, 21.0, 0.0).truncated is True


class TestBoundingBoxFromPoints:
    """The box assembler was extracted to a public helper shared with the camera path.

    These tests pin its behavior directly (the radiometer entry point above still
    exercises it end-to-end, guarding the extraction).
    """

    def test_simple_box_from_corner_points(self):
        # Four corner points -> the enclosing lat/lon rectangle, centered anchor irrelevant.
        box = geo.bounding_box_from_points(10.0, 20.0, [9.0, 11.0, 9.0, 11.0], [19.0, 19.0, 21.0, 21.0])
        assert (box.lat_min, box.lat_max, box.lon_min, box.lon_max) == (9.0, 11.0, 19.0, 21.0)
        assert not box.wraps_dateline
        assert not box.truncated

    def test_dateline_crossing_points(self):
        # Points straddling the antimeridian choose the [0, 360) representation.
        box = geo.bounding_box_from_points(0.0, 179.5, [-1.0, 1.0], [179.0, -179.0])
        assert box.wraps_dateline
        assert box.lon_max > 180.0

    def test_truncated_flag_passes_through(self):
        box = geo.bounding_box_from_points(0.0, 0.0, [-1.0, 1.0], [-1.0, 1.0], truncated=True)
        assert box.truncated


class TestProjectToAngular:
    """project_to_angular must be the exact inverse of the forward offset ray-trace."""

    BLAT, BLON, SLAT, SLON = 10.0, 20.0, 12.0, 20.0

    @pytest.mark.parametrize("delta", [-1.2, -0.4, 0.0, 0.7, 1.5])
    @pytest.mark.parametrize("beta", [-0.6, 0.0, 0.5])
    def test_round_trip_recovers_delta_beta(self, delta, beta):
        radius = _radius(self.SLAT, self.SLON)
        sat, boresight_dir = geo._viewing_geometry(self.BLAT, self.BLON, self.SLAT, self.SLON, radius)
        cross_axis = geo._scan_frame_axes(boresight_dir, sat)

        # Forward: rotate the boresight by (delta, beta) and ray-trace to the ground.
        direction = geo._offset_ray_direction(boresight_dir, cross_axis, delta, beta)
        hit = geo._ray_ellipsoid_intersection(sat, direction)
        assert hit is not None
        lat, lon, _ = geo._ecef_to_geodetic(hit)

        # Inverse: project that ground point back to angles.
        rec_delta, rec_beta = geo.project_to_angular(
            np.array([lat]), np.array([lon]), self.BLAT, self.BLON, self.SLAT, self.SLON, satellite_radius_km=radius
        )
        assert float(rec_delta[0]) == pytest.approx(delta, abs=1e-3)
        assert float(rec_beta[0]) == pytest.approx(beta, abs=1e-3)

    def test_boresight_projects_to_origin(self):
        blat, blon, slat, slon = -5.0, 100.0, -4.0, 101.0
        rec_delta, rec_beta = geo.project_to_angular(
            np.array([blat]), np.array([blon]), blat, blon, slat, slon, satellite_radius_km=_radius(slat, slon)
        )
        assert float(rec_delta[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(rec_beta[0]) == pytest.approx(0.0, abs=1e-6)

    def test_preserves_input_shape(self):
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        d, b = geo.project_to_angular(lats, lons, 10.15, 20.15, 11.0, 20.15, satellite_radius_km=_radius(11.0, 20.15))
        assert d.shape == (2, 2)
        assert b.shape == (2, 2)

    def test_precomputed_cell_ecef_is_identical(self):
        # Passing the tile's cell ECEF (what the weigher caches) must give bit-identical
        # angles to computing it inline -- the value is frame-independent.
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        radius = _radius(11.0, 20.15)
        cell_ecef = geo.surface_cell_ecef_km(lats, lons)
        d0, b0 = geo.project_to_angular(lats, lons, 10.15, 20.15, 11.0, 20.15, satellite_radius_km=radius)
        d1, b1 = geo.project_to_angular(
            lats, lons, 10.15, 20.15, 11.0, 20.15, satellite_radius_km=radius, cell_ecef_km=cell_ecef
        )
        np.testing.assert_array_equal(d0, d1)
        np.testing.assert_array_equal(b0, b1)

    def test_batched_frame_matches_inline_frame(self):
        # compute_viewing_frames is the batched frame the orchestrator hands each footprint;
        # feeding it back must give the inline construction's angles.
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        blat, blon = np.array([10.15, -5.0]), np.array([20.15, 100.0])
        slat, slon = np.array([11.0, -4.0]), np.array([20.15, 101.0])
        radius = _radius(slat, slon)
        frames = compute_viewing_frames(blat, blon, slat, slon, satellite_radius_km=radius)
        assert len(frames) == 2
        inline = geo.project_to_angular(lats, lons, blat[0], blon[0], slat[0], slon[0], satellite_radius_km=radius[0])
        batched = geo.project_to_angular(lats, lons, 0.0, 0.0, 0.0, 0.0, frame=frames.for_index(0))
        np.testing.assert_allclose(batched[0], inline[0], atol=1e-12)
        np.testing.assert_allclose(batched[1], inline[1], atol=1e-12)

    def test_needs_radius_or_frame(self):
        with pytest.raises(ValueError, match="satellite_radius_km"):
            geo.project_to_angular(np.array([10.0]), np.array([20.0]), 10.0, 20.0, 11.0, 20.0)

    def test_surface_cell_ecef_matches_inline_transform(self):
        # surface_cell_ecef_km is the single source of truth for the cached value: it must
        # equal the point-by-point surface transform over the grid raveled in row-major order.
        lats = np.array([[1.0, 2.0], [3.0, 4.0]])
        lons = np.array([[5.0, 6.0], [7.0, 8.0]])
        expected = np.array([geo._geodetic_to_ecef(la, lo, 0.0) for la, lo in zip(lats.ravel(), lons.ravel())])
        np.testing.assert_allclose(geo.surface_cell_ecef_km(lats, lons), expected, rtol=0, atol=1e-9)


class TestPsfGroundRadius:
    """psf_ground_radius_km projects the PSF angular extent to a ground radius."""

    def test_grows_with_altitude(self):
        low = psf_ground_radius_km(400.0, 0.0)
        high = psf_ground_radius_km(800.0, 0.0)
        assert high > low > 0.0

    def test_grows_with_viewing_zenith(self):
        # The 1/cos(vza) elongation makes an oblique view reach farther on the ground.
        assert psf_ground_radius_km(ALT, 60.0) > psf_ground_radius_km(ALT, 0.0)

    @pytest.mark.parametrize("distance_deg", [0.0, 4.0, 7.0, 10.0, 13.0])
    def test_matches_ray_traced_along_view_reach(self, distance_deg):
        # A FOV floor above the PSF's diagonal extent pins the angle the radius uses, so
        # the radius can be checked against rays traced +/- that angle along the view
        # direction. Their mean ground reach is what rho * tan(theta) / cos(vza) predicts;
        # a flat-Earth h * tan(theta) / cos(vza) falls ~40% short at vza 60.
        theta = 2.0
        lat, lon = 0.0, 0.0
        satellite, boresight = geo._viewing_geometry(lat, lon, distance_deg, 0.0, _radius(distance_deg, 0.0))
        cross_axis = geo._scan_frame_axes(boresight, satellite)
        geod = geo._wgs84_geod()
        reach_km = []
        for delta in (theta, -theta):
            plat, plon, missed = geo._perimeter_point(satellite, boresight, cross_axis, delta, 0.0)
            assert not missed
            reach_km.append(geod.inv(lon, lat, plon, plat)[2] / 1000.0)

        vza = _vza(lat, lon, distance_deg, 0.0)
        radius = psf_ground_radius_km(ALT, vza, fov_halfangle_deg=theta)
        assert radius == pytest.approx(np.mean(reach_km), rel=0.05)


class TestBatchBoundingBoxParity:
    """The vectorized compute_footprint_bounding_boxes matches the scalar oracle.

    The scalar compute_footprint_bounding_box is the reference implementation; the batch
    path must reproduce it per footprint. Both share the closed-form satellite placement,
    so they agree to floating-point noise; the structural flags and the off-Earth (None)
    decision must match exactly.
    """

    BOUND_ATOL_DEG = 1e-9

    def _scalar_or_none(self, lat, lon, sub_lat, sub_lon, radius):
        try:
            return compute_footprint_bounding_box(
                lat, lon, sub_lat, sub_lon, satellite_radius_km=radius, on_limb="flag"
            )
        except OffLimbError:
            return None

    def _assert_box_close(self, scalar, batch):
        if scalar is None:
            assert batch is None
            return
        assert batch is not None
        assert batch.lat_min == pytest.approx(scalar.lat_min, abs=self.BOUND_ATOL_DEG)
        assert batch.lat_max == pytest.approx(scalar.lat_max, abs=self.BOUND_ATOL_DEG)
        assert batch.lon_min == pytest.approx(scalar.lon_min, abs=self.BOUND_ATOL_DEG)
        assert batch.lon_max == pytest.approx(scalar.lon_max, abs=self.BOUND_ATOL_DEG)
        # Structural flags must match exactly.
        assert batch.wraps_dateline == scalar.wraps_dateline
        assert batch.is_polar == scalar.is_polar
        assert batch.truncated == scalar.truncated

    def test_matches_scalar_over_random_and_edge_cases(self):
        rng = np.random.default_rng(0)
        n = 400
        lat = rng.uniform(-89.0, 89.0, n)
        lon = rng.uniform(-180.0, 180.0, n)
        sub_lat = np.clip(lat + rng.uniform(-8.0, 8.0, n), -89.9, 89.9)
        sub_lon = ((lon + rng.uniform(-8.0, 8.0, n) + 180.0) % 360.0) - 180.0

        # Hand-picked edge cases mixed into the batch: poles, dateline, severe angle,
        # nadir, and off-Earth views.
        lat[0], lon[0], sub_lat[0], sub_lon[0] = 89.9, 10.0, 89.5, 12.0
        lat[1], lon[1], sub_lat[1], sub_lon[1] = -89.9, -30.0, -89.4, -28.0
        lat[2], lon[2], sub_lat[2], sub_lon[2] = 12.0, 179.8, 12.0, -179.5
        lat[3], lon[3], sub_lat[3], sub_lon[3] = 0.0, 0.0, 27.3, 0.0  # VZA ~89.4 -> limb-truncated corners
        sub_lat[6], sub_lon[6] = lat[6], lon[6]  # nadir
        radius = _radius(sub_lat, sub_lon)
        lat[4] = geo.L1B_FILL_VALUE  # fill -> off-Earth view (None)
        lon[7] = np.nan  # L1B writes NaN for an off-Earth view -> None

        batch = compute_footprint_bounding_boxes(lat, lon, sub_lat, sub_lon, satellite_radius_km=radius)
        assert len(batch) == n
        # Sanity: the injected off-Earth footprints are None; at least one truncated/polar
        # case is exercised so the parity assertions actually cover those branches.
        assert batch[4] is None
        assert batch[7] is None
        assert any(b is not None and b.truncated for b in batch)
        assert any(b is not None and b.is_polar for b in batch)

        for i in range(n):
            scalar = self._scalar_or_none(lat[i], lon[i], sub_lat[i], sub_lon[i], radius[i])
            self._assert_box_close(scalar, batch[i])

    def test_empty_input_returns_empty_list(self):
        empty = np.array([], dtype=float)
        assert compute_footprint_bounding_boxes(empty, empty, empty, empty, satellite_radius_km=empty) == []

    def test_all_off_earth_returns_all_none(self):
        boxes = compute_footprint_bounding_boxes(
            np.array([np.nan, 5.0]),
            np.array([1.0, np.nan]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            satellite_radius_km=_radius(np.array([1.0, 1.0]), np.array([1.0, 1.0])),
        )
        assert boxes == [None, None]

    @pytest.mark.parametrize(
        "bad_sub_lat", [geo.L1B_FILL_VALUE, 9.96921e36, np.nan], ids=["fill", "netcdf-fill", "nan"]
    )
    def test_invalid_subsatellite_raises(self, bad_sub_lat):
        # Missing spacecraft geolocation is bad input, not an off-Earth view: it must not
        # hide among the None entries.
        with pytest.raises(MissingGeolocationError, match="1 footprint"):
            compute_footprint_bounding_boxes(
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([bad_sub_lat, 5.0]),
                np.array([0.0, 0.0]),
                satellite_radius_km=np.full(2, _radius(5.0, 0.0)),
            )

    @pytest.mark.parametrize("bad_radius", [-9999.0, np.nan, 6000.0], ids=["fill", "nan", "below-surface"])
    def test_invalid_radius_raises(self, bad_radius):
        with pytest.raises(MissingGeolocationError, match="1 footprint"):
            compute_footprint_bounding_boxes(
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([5.0, 5.0]),
                np.array([0.0, 0.0]),
                satellite_radius_km=np.array([bad_radius, _radius(5.0, 0.0)]),
            )

    def test_footprint_beyond_the_horizon_raises(self):
        with pytest.raises(MissingGeolocationError, match="1 footprint"):
            compute_footprint_bounding_boxes(
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([30.0, 5.0]),
                np.array([0.0, 0.0]),
                satellite_radius_km=_radius(np.array([30.0, 5.0]), np.array([0.0, 0.0])),
            )

    def test_scalar_radius_broadcasts(self):
        lat, lon = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        sub_lat, sub_lon = np.array([2.0, 3.0]), np.array([0.0, 1.0])
        radius = _radius(2.0, 0.0)
        broadcast = compute_footprint_bounding_boxes(lat, lon, sub_lat, sub_lon, satellite_radius_km=radius)
        explicit = compute_footprint_bounding_boxes(lat, lon, sub_lat, sub_lon, satellite_radius_km=np.full(2, radius))
        assert broadcast == explicit
