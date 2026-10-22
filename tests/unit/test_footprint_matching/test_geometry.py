"""Unit tests for the geometry module (libera_utils.footprint_matching.geometry).

Coverage:
- WGS84 internals (geodetic<->ECEF round-trip, ellipsoid normal, ray-ellipsoid
  intersection) and placing the satellite from the L1B subsatellite point and radius
- the public bounding-box entry points: nadir vs stretched oblique footprints,
  enclosure of the boresight and the PSF contour, growth toward the limb
- edge cases: pole enclosure, dateline crossing, off-limb (raise/clamp), fill
  values, inconsistent geolocation, input shapes, and invalid arguments
- the inverse projection to the radiometer angular frame
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
)

# A representative spacecraft geodetic altitude for synthetic cases (JPSS orbit), km.
ALT = 835.0


def _radius(sub_lat, sub_lon, altitude_km=ALT):
    """L1B-style ``Radius_of_Satellite_from_Center_of_Earth`` for a satellite at ``altitude_km``."""
    satellite = geo._geodetic_to_ecef(sub_lat, sub_lon, altitude_km)
    radius = np.linalg.norm(satellite, axis=1)
    return float(radius[0]) if radius.size == 1 else radius


def _box(lat, lon, sub_lat, sub_lon, **kwargs):
    """Scalar box for a satellite at :data:`ALT` over ``(sub_lat, sub_lon)``."""
    return compute_footprint_bounding_box(
        lat, lon, sub_lat, sub_lon, satellite_radius_km=_radius(sub_lat, sub_lon), **kwargs
    )


def _geometry(lat, lon, sub_lat, sub_lon):
    """``(satellite, boresight, cross_axis)``, each ``(1, 3)``, for a satellite at :data:`ALT`."""
    satellite, boresight = geo._viewing_geometry(
        *(np.atleast_1d(np.asarray(v, dtype=float)) for v in (lat, lon, sub_lat, sub_lon, _radius(sub_lat, sub_lon)))
    )
    return satellite, boresight, geo._cross_scan_axis(boresight, satellite)


def _vza(lat, lon, sub_lat, sub_lon):
    """Viewing zenith (deg) at the footprint, from the satellite :func:`_box` places."""
    _satellite, boresight, _ = _geometry(lat, lon, sub_lat, sub_lon)
    return math.degrees(math.acos(float(-boresight[0] @ geo._ellipsoid_normal(lat, lon)[0])))


class TestGeodeticEcefRoundTrip:
    @pytest.mark.parametrize(
        ("lat", "lon", "height_km"),
        [(0.0, 0.0, 0.0), (45.0, -75.0, 0.0), (89.9, 120.0, 0.0), (-30.0, 179.0, ALT)],
    )
    def test_round_trip(self, lat, lon, height_km):
        xyz = geo._geodetic_to_ecef(lat, lon, height_km)
        rlat, rlon, rheight = geo._ecef_to_geodetic(xyz)
        assert rlat[0] == pytest.approx(lat, abs=1e-7)
        assert rlon[0] == pytest.approx(lon, abs=1e-7)
        assert rheight[0] == pytest.approx(height_km, abs=1e-4)

    def test_equator_radius_is_semi_major(self):
        xyz = geo._geodetic_to_ecef(0.0, 0.0, 0.0)[0]
        assert np.linalg.norm(xyz) == pytest.approx(WGS84_SEMI_MAJOR_AXIS_KM, abs=1e-6)

    def test_pole_radius_is_semi_minor(self):
        xyz = geo._geodetic_to_ecef(90.0, 0.0, 0.0)[0]
        assert np.linalg.norm(xyz) == pytest.approx(WGS84_SEMI_MINOR_AXIS_KM, abs=1e-6)


class TestEllipsoidNormal:
    def test_unit_length(self):
        assert np.linalg.norm(geo._ellipsoid_normal(37.0, -100.0)[0]) == pytest.approx(1.0)

    def test_radial_at_equator_and_pole(self):
        # On the equator and at the poles the geodetic normal equals the geocentric
        # radial direction.
        eq = geo._geodetic_to_ecef(0.0, 0.0, 0.0)[0]
        assert np.allclose(geo._ellipsoid_normal(0.0, 0.0)[0], eq / np.linalg.norm(eq), atol=1e-9)
        assert np.allclose(geo._ellipsoid_normal(90.0, 0.0)[0], [0.0, 0.0, 1.0], atol=1e-9)

    def test_geodetic_differs_from_geocentric_at_mid_latitude(self):
        # Off the equator the geodetic normal is NOT the geocentric radial direction; a
        # spherical Earth model would conflate the two.
        lat = 45.0
        normal = geo._ellipsoid_normal(lat, 0.0)[0]
        radial = geo._geodetic_to_ecef(lat, 0.0, 0.0)[0]
        radial = radial / np.linalg.norm(radial)
        angle_deg = math.degrees(math.acos(np.clip(np.dot(normal, radial), -1.0, 1.0)))
        assert angle_deg > 0.1


class TestRayEllipsoidIntersection:
    def test_straight_down_hits_below_point(self):
        # A ray from above (0, 0) pointing straight down hits near (0, 0).
        origin = geo._geodetic_to_ecef(0.0, 0.0, ALT)
        direction = -geo._ellipsoid_normal(0.0, 0.0)
        points, hit = geo._ray_ellipsoid_intersection(origin, direction)
        assert hit[0]
        lat, lon, height = geo._ecef_to_geodetic(points)
        assert lat[0] == pytest.approx(0.0, abs=1e-6)
        assert lon[0] == pytest.approx(0.0, abs=1e-6)
        assert height[0] == pytest.approx(0.0, abs=1e-6)

    def test_ray_into_space_misses(self):
        origin = geo._geodetic_to_ecef(0.0, 0.0, ALT)
        direction = geo._ellipsoid_normal(0.0, 0.0)  # pointing up, away from Earth
        points, hit = geo._ray_ellipsoid_intersection(origin, direction)
        assert not hit[0]
        assert np.all(np.isnan(points))


class TestViewingGeometry:
    @pytest.mark.parametrize(("sub_lat", "sub_lon"), [(2.0, 0.0), (45.0, -75.0), (-70.0, 120.0), (89.0, 10.0)])
    def test_satellite_placed_from_subpoint_and_radius(self, sub_lat, sub_lon):
        # The satellite is the point at the given radius on the ellipsoid normal through
        # the geodetic sub-point, i.e. it has that sub-point and the altitude implied.
        satellite = geo._satellite_from_subpoint(
            np.array([sub_lat]), np.array([sub_lon]), np.array([_radius(sub_lat, sub_lon)])
        )
        rlat, rlon, rheight = geo._ecef_to_geodetic(satellite)
        assert rlat[0] == pytest.approx(sub_lat, abs=1e-9)
        assert rlon[0] == pytest.approx(sub_lon, abs=1e-9)
        assert rheight[0] == pytest.approx(ALT, abs=1e-6)

    def test_boresight_points_at_footprint(self):
        satellite, direction, _ = _geometry(10.0, 20.0, 12.0, 20.0)
        ground = geo._geodetic_to_ecef(10.0, 20.0, 0.0)
        expected = (ground - satellite) / np.linalg.norm(ground - satellite)
        assert np.allclose(direction, expected, atol=1e-12)

    def test_nadir_boresight_is_downward_normal(self):
        # Footprint directly below the satellite: the boresight is the downward normal.
        _satellite, direction, _ = _geometry(30.0, 40.0, 30.0, 40.0)
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
        ("lat", "lon", "sub_lat", "sub_lon"),
        [
            (10.0, 20.0, 15.0, 20.0),
            (10.0, 20.0, 13.5, 23.5),
            (10.0, 20.0, 12.5, 24.3),
            (10.0, 20.0, 10.0, 25.0),
            (80.0, 20.0, 77.0, 32.0),
        ],
        ids=["az0", "az45", "az60", "az90", "high-lat-rotated"],
    )
    def test_box_encloses_ray_traced_psf_contour(self, lat, lon, sub_lat, sub_lon):
        # Ray-trace every cell of the 95% PSF contour onto the ground and check it lies in
        # the box. The contour reaches into the corners of its extent rectangle, which
        # matters most at scan azimuths ~30-60 deg from north, and near (but not enclosing)
        # a pole, where meridians converge.
        box = _box(lat, lon, sub_lat, sub_lon)
        assert box.lon_max - box.lon_min < 360.0  # not a pole-enclosing box

        satellite, boresight, cross_axis = _geometry(lat, lon, sub_lat, sub_lon)
        delta, beta = _psf_contour_angles()
        directions = geo._offset_ray_direction(boresight, cross_axis, delta, beta)
        points, hit = geo._ray_ellipsoid_intersection(satellite, directions)
        assert hit.all()
        contour_lat, contour_lon, _ = geo._ecef_to_geodetic(points)

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
        batch = compute_footprint_bounding_boxes(
            lat, lon, sub_lat, sub_lon, satellite_radius_km=_radius(sub_lat, sub_lon)
        )
        for box in batch:
            assert not box.wraps_dateline
            assert box.lon_max < 0.0

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

    def test_raise_mode_returns_untruncated_box(self):
        assert _box(0.0, 0.0, 15.0, 0.0, on_limb="raise").truncated is False

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

    def test_clipped_rays_land_on_the_horizon(self):
        # A ray that misses at its full offset is pulled back to where it grazes the limb.
        satellite, boresight, cross_axis = _geometry(*self.SEVERE)
        delta, beta = geo._perimeter_angles(*geo._box_angular_extents(psf.LIBERA_FOV_HALFANGLE_DEG), 8)
        lat, lon, missed = geo._trace_perimeter(satellite, boresight, cross_axis, delta, beta)
        assert missed.any()
        # Grazing incidence: the line of sight to a clipped point is tangent to the surface.
        clipped = geo._geodetic_to_ecef(lat[missed], lon[missed], 0.0)
        look = geo._unit(clipped - satellite)
        cos_vza = -np.sum(look * geo._ellipsoid_normal(lat[missed], lon[missed]), axis=1)
        assert np.all(cos_vza < 0.02)


class TestBoundingBoxFromPoints:
    """The box assembler shared by the radiometer and camera paths."""

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

    def _geolocation(self, blat=BLAT, blon=BLON, slat=SLAT, slon=SLON):
        return {
            "boresight_lat_deg": blat,
            "boresight_lon_deg": blon,
            "subsatellite_lat_deg": slat,
            "subsatellite_lon_deg": slon,
            "satellite_radius_km": _radius(slat, slon),
        }

    @pytest.mark.parametrize("delta", [-1.2, -0.4, 0.0, 0.7, 1.5])
    @pytest.mark.parametrize("beta", [-0.6, 0.0, 0.5])
    def test_round_trip_recovers_delta_beta(self, delta, beta):
        satellite, boresight, cross_axis = _geometry(self.BLAT, self.BLON, self.SLAT, self.SLON)

        # Forward: rotate the boresight by (delta, beta) and ray-trace to the ground.
        direction = geo._offset_ray_direction(boresight, cross_axis, delta, beta)
        points, hit = geo._ray_ellipsoid_intersection(satellite, direction)
        assert hit[0]
        lat, lon, _ = geo._ecef_to_geodetic(points)

        # Inverse: project that ground point back to angles.
        rec_delta, rec_beta = geo.project_to_angular(lat, lon, **self._geolocation())
        assert float(rec_delta[0]) == pytest.approx(delta, abs=1e-3)
        assert float(rec_beta[0]) == pytest.approx(beta, abs=1e-3)

    def test_boresight_projects_to_origin(self):
        blat, blon, slat, slon = -5.0, 100.0, -4.0, 101.0
        rec_delta, rec_beta = geo.project_to_angular(
            np.array([blat]), np.array([blon]), **self._geolocation(blat, blon, slat, slon)
        )
        assert float(rec_delta[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(rec_beta[0]) == pytest.approx(0.0, abs=1e-6)

    def test_preserves_input_shape(self):
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        d, b = geo.project_to_angular(lats, lons, **self._geolocation(10.15, 20.15, 11.0, 20.15))
        assert d.shape == (2, 2)
        assert b.shape == (2, 2)

    def test_precomputed_cell_ecef_is_identical(self):
        # Passing the tile's cell ECEF (what the weigher caches) must give bit-identical
        # angles to computing it inline -- the value is frame-independent.
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        geolocation = self._geolocation(10.15, 20.15, 11.0, 20.15)
        cell_ecef = geo.surface_cell_ecef_km(lats, lons)
        d0, b0 = geo.project_to_angular(lats, lons, **geolocation)
        d1, b1 = geo.project_to_angular(lats, lons, **geolocation, cell_ecef_km=cell_ecef)
        np.testing.assert_array_equal(d0, d1)
        np.testing.assert_array_equal(b0, b1)

    def test_batched_frame_matches_inline_frame(self):
        # compute_viewing_frames is the batched frame the orchestrator hands each footprint;
        # feeding it back must give the inline construction's angles.
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        blat, blon = np.array([10.15, -5.0]), np.array([20.15, 100.0])
        slat, slon = np.array([11.0, -4.0]), np.array([20.15, 101.0])
        frames = compute_viewing_frames(blat, blon, slat, slon, satellite_radius_km=_radius(slat, slon))
        assert len(frames) == 2
        assert len(frames.for_index(1)) == 1
        inline = geo.project_to_angular(lats, lons, **self._geolocation(blat[0], blon[0], slat[0], slon[0]))
        batched = geo.project_to_angular(lats, lons, frame=frames.for_index(0))
        np.testing.assert_allclose(batched[0], inline[0], atol=1e-12)
        np.testing.assert_allclose(batched[1], inline[1], atol=1e-12)

    def test_needs_geolocation_or_frame(self):
        geolocation = self._geolocation()
        del geolocation["satellite_radius_km"]
        with pytest.raises(ValueError, match="satellite_radius_km"):
            geo.project_to_angular(np.array([10.0]), np.array([20.0]), **geolocation)

    def test_frame_must_hold_one_footprint(self):
        frames = compute_viewing_frames(
            [10.0, 11.0], [20.0, 20.0], [12.0, 12.0], [20.0, 20.0], satellite_radius_km=np.full(2, _radius(12.0, 20.0))
        )
        with pytest.raises(ValueError, match="exactly one footprint"):
            geo.project_to_angular(np.array([10.0]), np.array([20.0]), frame=frames)

    def test_missing_boresight_is_an_off_earth_view(self):
        with pytest.raises(OffLimbError):
            geo.project_to_angular(np.array([10.0]), np.array([20.0]), **self._geolocation(blat=float("nan")))

    def test_surface_cell_ecef_matches_inline_transform(self):
        # surface_cell_ecef_km is the single source of truth for the cached value: it must
        # equal the point-by-point surface transform over the grid raveled in row-major order.
        lats = np.array([[1.0, 2.0], [3.0, 4.0]])
        lons = np.array([[5.0, 6.0], [7.0, 8.0]])
        expected = np.array([geo._geodetic_to_ecef(la, lo, 0.0)[0] for la, lo in zip(lats.ravel(), lons.ravel())])
        np.testing.assert_allclose(geo.surface_cell_ecef_km(lats, lons), expected, rtol=0, atol=1e-9)


class TestComputeViewingFrames:
    def test_missing_boresight_raises_off_limb(self):
        # An off-Earth view has no frame; it is reported as such, not as bad ephemeris.
        with pytest.raises(OffLimbError, match="1 footprint"):
            compute_viewing_frames(
                np.array([10.0, np.nan]),
                np.array([20.0, 0.0]),
                np.array([10.0, 10.0]),
                np.array([20.0, 20.0]),
                satellite_radius_km=np.full(2, _radius(10.0, 20.0)),
            )

    @pytest.mark.parametrize("radius", [7200.0, np.array([7200.0])], ids=["scalar", "length-1"])
    def test_requires_one_radius_per_footprint(self, radius):
        with pytest.raises(ValueError, match="one radius per footprint"):
            compute_viewing_frames([10.0, 0.0], [20.0, 0.0], [12.0, 0.0], [20.0, 0.0], satellite_radius_km=radius)

    def test_rejects_multidimensional_input(self):
        grid = np.full((2, 2), 10.0)
        with pytest.raises(ValueError, match="1-D"):
            compute_viewing_frames(grid, grid, grid, grid, satellite_radius_km=_radius(10.0, 10.0))

    def test_axes_are_orthonormal(self):
        frames = compute_viewing_frames(
            [10.0, 0.0], [20.0, 0.0], [12.0, 0.0], [20.0, 0.0], satellite_radius_km=np.full(2, 7200.0)
        )
        for axis in (frames.b_hat, frames.c_hat, frames.n_hat):
            np.testing.assert_allclose(np.linalg.norm(axis, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(frames.b_hat * frames.c_hat, axis=1), 0.0, atol=1e-12)


class TestBatchBoundingBoxes:
    """compute_footprint_bounding_boxes over many footprints at once."""

    BOUND_ATOL_DEG = 1e-9

    def test_each_box_is_independent_of_the_batch(self):
        # A footprint's box must be the same whether computed alone or among others,
        # including the edge cases (poles, dateline, severe angle, nadir, off-Earth views).
        rng = np.random.default_rng(0)
        n = 400
        lat = rng.uniform(-89.0, 89.0, n)
        lon = rng.uniform(-180.0, 180.0, n)
        sub_lat = np.clip(lat + rng.uniform(-8.0, 8.0, n), -89.9, 89.9)
        sub_lon = ((lon + rng.uniform(-8.0, 8.0, n) + 180.0) % 360.0) - 180.0

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
        assert batch[4] is None
        assert batch[7] is None
        assert any(b is not None and b.truncated for b in batch)
        assert any(b is not None and b.is_polar for b in batch)
        assert any(b is not None and b.wraps_dateline for b in batch)

        for i in range(n):
            (alone,) = compute_footprint_bounding_boxes(
                lat[i], lon[i], sub_lat[i], sub_lon[i], satellite_radius_km=radius[i : i + 1]
            )
            if alone is None:
                assert batch[i] is None
                continue
            np.testing.assert_allclose(batch[i][:4], alone[:4], rtol=0, atol=self.BOUND_ATOL_DEG)
            assert batch[i][4:] == alone[4:]

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

    @pytest.mark.parametrize("as_array", [False, True], ids=["scalar", "length-1"])
    def test_requires_one_radius_per_footprint(self, as_array):
        # One nominal radius must not stand in for the per-record L1B ephemeris.
        radius = _radius(2.0, 0.0)
        with pytest.raises(ValueError, match="one radius per footprint"):
            compute_footprint_bounding_boxes(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                np.array([2.0, 3.0]),
                np.array([0.0, 1.0]),
                satellite_radius_km=np.array([radius]) if as_array else radius,
            )

    def test_rejects_multidimensional_input(self):
        grid = np.array([[10.0, 11.0], [12.0, 13.0]])
        with pytest.raises(ValueError, match="1-D"):
            compute_footprint_bounding_boxes(grid, grid, grid, grid, satellite_radius_km=_radius(10.0, 10.0))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="share one length"):
            compute_footprint_bounding_boxes(
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, 1.0]),
                satellite_radius_km=_radius(0.0, 0.0),
            )
