"""Unit tests for the geometry module (libera_utils.footprint_matching.geometry).

Coverage:
- WGS84 internals (geodetic<->ECEF round-trip, ellipsoid normal, ray-ellipsoid
  intersection, satellite-position recovery from the viewing geometry)
- the public bounding-box entry point: nadir vs stretched high-VZA footprints,
  enclosure of the boresight, growth with VZA
- edge cases: pole enclosure, dateline crossing, off-limb (raise/clamp), fill
  values, and invalid arguments
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from libera_utils.footprint_matching import geometry as geo
from libera_utils.footprint_matching.geometry import (
    NOMINAL_ALTITUDE_KM,
    WGS84_SEMI_MAJOR_AXIS_KM,
    WGS84_SEMI_MINOR_AXIS_KM,
    OffLimbError,
    PartialFootprintError,
    compute_footprint_bounding_box,
)

# A representative satellite altitude for synthetic cases (JPSS orbit).
ALT = NOMINAL_ALTITUDE_KM


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
    def test_satellite_recovered_from_positions(self):
        # Build a self-consistent footprint / subsatellite / VZA triple: place the
        # satellite at a known altitude over the subsatellite point, pick a footprint,
        # and compute the VZA it implies. The no-altitude path must then recover that
        # altitude and subsatellite location.
        subsat_lat = 2.0
        satellite = geo._geodetic_to_ecef(subsat_lat, 0.0, ALT)
        ground = geo._geodetic_to_ecef(0.0, 0.0, 0.0)
        normal_p = geo._ellipsoid_normal(0.0, 0.0)
        vza = math.degrees(math.acos(np.dot(normal_p, satellite - ground) / np.linalg.norm(satellite - ground)))

        recovered, _direction, _normal = geo._viewing_geometry(0.0, 0.0, subsat_lat, 0.0, vza, None)
        rlat, rlon, rheight = geo._ecef_to_geodetic(recovered)
        assert rheight == pytest.approx(ALT, rel=1e-3)
        assert rlat == pytest.approx(subsat_lat, abs=1e-3)
        assert rlon == pytest.approx(0.0, abs=1e-3)

    def test_boresight_points_at_footprint(self):
        # The boresight direction from the satellite must point at the footprint.
        satellite, direction, _normal = geo._viewing_geometry(10.0, 20.0, 12.0, 20.0, 30.0, ALT)
        ground = geo._geodetic_to_ecef(10.0, 20.0, 0.0)
        expected = (ground - satellite) / np.linalg.norm(ground - satellite)
        assert np.allclose(direction, expected, atol=1e-9)


class TestComputeBoundingBoxBasics:
    def test_nadir_box_is_small_and_encloses_boresight(self):
        bb = compute_footprint_bounding_box(0.0, 0.0, 0.5, 0.0, 2.0, altitude_km=ALT)
        # Boresight is inside the box.
        assert bb.lat_min <= 0.0 <= bb.lat_max
        assert bb.lon_min <= 0.0 <= bb.lon_max
        # Nadir footprint is small (well under a degree across).
        assert (bb.lat_max - bb.lat_min) < 1.0
        assert not bb.is_polar
        assert not bb.wraps_dateline

    def test_high_vza_box_is_much_larger_than_nadir(self):
        nadir = compute_footprint_bounding_box(0.0, 0.0, 0.5, 0.0, 2.0, altitude_km=ALT)
        high = compute_footprint_bounding_box(0.0, 0.0, 25.0, 0.0, 70.0, altitude_km=ALT)
        nadir_span = (nadir.lat_max - nadir.lat_min) + (nadir.lon_max - nadir.lon_min)
        high_span = (high.lat_max - high.lat_min) + (high.lon_max - high.lon_min)
        assert high_span > 3.0 * nadir_span

    def test_box_grows_monotonically_with_vza(self):
        # The footprint stretches as the view angle increases toward the limb.
        prev = 0.0
        for vza in (5.0, 30.0, 50.0, 70.0):
            bb = compute_footprint_bounding_box(0.0, 0.0, vza, 0.0, vza, altitude_km=ALT)
            span = (bb.lat_max - bb.lat_min) + (bb.lon_max - bb.lon_min)
            assert span > prev
            prev = span

    def test_along_scan_axis_follows_subsatellite_bearing(self):
        # Subsatellite due north -> along-scan is N-S, so the latitude span (along)
        # should exceed the longitude span (cross) at high VZA.
        bb = compute_footprint_bounding_box(45.0, 10.0, 47.0, 10.0, 70.0, altitude_km=ALT)
        lat_span_km = (bb.lat_max - bb.lat_min) * 111.0
        lon_span_km = (bb.lon_max - bb.lon_min) * 111.0 * math.cos(math.radians(45.0))
        assert lat_span_km > lon_span_km


class TestComputeBoundingBoxEdgeCases:
    def test_pole_enclosure(self):
        # A footprint very close to the North pole encloses it: full longitude range,
        # latitude pinned to 90, flagged polar.
        bb = compute_footprint_bounding_box(89.9, 0.0, 88.0, 0.0, 60.0, altitude_km=ALT)
        assert bb.is_polar
        assert bb.lat_max == pytest.approx(90.0)
        assert bb.lon_min == pytest.approx(-180.0)
        assert bb.lon_max == pytest.approx(180.0)

    def test_dateline_crossing(self):
        bb = compute_footprint_bounding_box(0.0, 179.8, 0.0, 178.0, 70.0, altitude_km=ALT)
        assert bb.wraps_dateline
        # By convention a wrapping box is represented with lon_max > 180.
        assert bb.lon_max > 180.0

    def test_partial_off_limb_flagged_not_raised_by_default(self):
        # Boresight on Earth but a box corner is off-limb: the default policy
        # truncates the box at the horizon and flags it, instead of raising.
        bb = compute_footprint_bounding_box(0.0, 0.0, 8.5, 0.0, 85.0, altitude_km=ALT)
        assert isinstance(bb, geo.BoundingBox)
        assert bb.truncated is True

    def test_fill_value_inputs_raise(self):
        with pytest.raises(OffLimbError):
            compute_footprint_bounding_box(-999.0, -999.0, 10.0, 0.0, -999.0, altitude_km=ALT)

    def test_nan_inputs_raise(self):
        with pytest.raises(OffLimbError):
            compute_footprint_bounding_box(float("nan"), 0.0, 10.0, 0.0, 30.0, altitude_km=ALT)

    def test_invalid_on_limb_argument(self):
        with pytest.raises(ValueError, match="on_limb"):
            compute_footprint_bounding_box(0.0, 0.0, 0.5, 0.0, 10.0, altitude_km=ALT, on_limb="bogus")


class TestPartialOffLimb:
    def test_default_flags_partial_coverage(self):
        # Boresight is well on Earth (VZA 85 < 90), but the limb-ward box corner is
        # off-limb. Default policy: truncate + flag.
        bb = compute_footprint_bounding_box(0.0, 0.0, 8.5, 0.0, 85.0, altitude_km=ALT)
        assert bb.truncated is True

    def test_raise_mode_raises_partial_footprint_error(self):
        with pytest.raises(PartialFootprintError):
            compute_footprint_bounding_box(0.0, 0.0, 8.5, 0.0, 85.0, altitude_km=ALT, on_limb="raise")

    def test_partial_is_offlimb_subclass(self):
        # Callers using `except OffLimbError` must still catch the partial case.
        with pytest.raises(OffLimbError) as excinfo:
            compute_footprint_bounding_box(0.0, 0.0, 8.5, 0.0, 85.0, altitude_km=ALT, on_limb="raise")
        assert isinstance(excinfo.value, PartialFootprintError)

    def test_moderate_angle_not_truncated(self):
        # Regression guard against false positives well inside the limb.
        bb = compute_footprint_bounding_box(0.0, 0.0, 7.0, 0.0, 70.0, altitude_km=ALT)
        assert bb.truncated is False

    def test_threshold_band(self):
        # The corner-based check engages just past VZA ~80 deg (well before the pure
        # along-scan edge would, which only happens near 90 deg).
        assert compute_footprint_bounding_box(0.0, 0.0, 8.0, 0.0, 80.0, altitude_km=ALT).truncated is False
        assert compute_footprint_bounding_box(0.0, 0.0, 8.2, 0.0, 82.0, altitude_km=ALT).truncated is True

    def test_centroid_off_limb_raises_regardless_of_mode(self):
        # VZA >= 90: even the centroid misses the Earth -> always OffLimbError (even in
        # the default flag mode), and NOT the partial subclass (no footprint at all).
        with pytest.raises(OffLimbError) as excinfo:
            compute_footprint_bounding_box(0.0, 0.0, 30.0, 0.0, 90.0, altitude_km=ALT)
        assert not isinstance(excinfo.value, PartialFootprintError)


class TestAltitudeRecoveryPath:
    def test_box_without_altitude_matches_box_with_altitude(self):
        # Build a self-consistent footprint/subsatellite pair for a known altitude
        # (placing the satellite over the subsatellite point and computing the VZA the
        # footprint implies), then confirm the no-altitude path (recovering the range
        # from positions) gives a box close to the altitude-supplied path.
        subsat_lat = 2.0
        satellite = geo._geodetic_to_ecef(subsat_lat, 0.0, ALT)
        ground = geo._geodetic_to_ecef(0.0, 0.0, 0.0)
        normal_p = geo._ellipsoid_normal(0.0, 0.0)
        vza = math.degrees(math.acos(np.dot(normal_p, satellite - ground) / np.linalg.norm(satellite - ground)))

        with_alt = compute_footprint_bounding_box(0.0, 0.0, subsat_lat, 0.0, vza, altitude_km=ALT)
        without_alt = compute_footprint_bounding_box(0.0, 0.0, subsat_lat, 0.0, vza)
        assert without_alt.lat_min == pytest.approx(with_alt.lat_min, abs=0.05)
        assert without_alt.lat_max == pytest.approx(with_alt.lat_max, abs=0.05)
