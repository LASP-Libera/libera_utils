"""Unit tests for the geometry module (libera_utils.footprint_matching.geometry).

Coverage:
- spherical helpers (great-circle distance, bearing, destination point)
- the viewing-triangle solver, including altitude recovery from positions
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
    EARTH_RADIUS_KM,
    NOMINAL_ALTITUDE_KM,
    OffLimbError,
    PartialFootprintError,
    compute_footprint_bounding_box,
)

# A representative satellite altitude for synthetic cases (JPSS orbit).
ALT = NOMINAL_ALTITUDE_KM


class TestGreatCircleDistance:
    def test_one_degree_along_equator(self):
        # One degree of arc along the equator is Re * radians(1).
        d = geo._great_circle_distance_km(0.0, 0.0, 0.0, 1.0, EARTH_RADIUS_KM)
        assert d == pytest.approx(EARTH_RADIUS_KM * math.radians(1.0), rel=1e-9)

    def test_zero_distance(self):
        assert geo._great_circle_distance_km(12.0, 34.0, 12.0, 34.0, EARTH_RADIUS_KM) == pytest.approx(0.0)


class TestInitialBearing:
    def test_due_north(self):
        assert geo._initial_bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=1e-9)

    def test_due_east(self):
        assert geo._initial_bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=1e-9)


class TestDestinationPoint:
    def test_north_by_one_degree(self):
        lat, lon = geo._destination_point(0.0, 0.0, 0.0, EARTH_RADIUS_KM * math.radians(1.0), EARTH_RADIUS_KM)
        assert lat == pytest.approx(1.0, abs=1e-6)
        assert lon == pytest.approx(0.0, abs=1e-6)

    def test_longitude_normalized(self):
        # Travelling east from near the dateline must wrap into [-180, 180].
        _, lon = geo._destination_point(0.0, 179.5, 90.0, EARTH_RADIUS_KM * math.radians(1.0), EARTH_RADIUS_KM)
        assert -180.0 <= lon <= 180.0
        assert lon < 0.0  # wrapped past +180


class TestSolveViewingTriangle:
    def test_nadir_has_zero_cone_angle(self):
        alpha, _rho, _reh = geo._solve_viewing_triangle(0.0, 0.0, 0.0, 0.0, 0.0, ALT, EARTH_RADIUS_KM)
        assert alpha == pytest.approx(0.0, abs=1e-6)

    def test_altitude_recovered_from_positions(self):
        # Place a footprint at a known ground distance from the subsatellite point
        # consistent with VZA, then confirm the solver recovers ~ALT without being
        # given the altitude. (Build a self-consistent case from the forward model.)
        theta = math.radians(40.0)
        reh = EARTH_RADIUS_KM + ALT
        alpha = math.asin(EARTH_RADIUS_KM * math.sin(theta) / reh)
        gamma = theta - alpha
        ground_arc_km = EARTH_RADIUS_KM * gamma
        # Subsatellite point due north of the footprint by that ground arc.
        boresight_lat = 0.0
        subsat_lat = math.degrees(ground_arc_km / EARTH_RADIUS_KM)
        _alpha, _rho, reh_recovered = geo._solve_viewing_triangle(
            boresight_lat, 0.0, subsat_lat, 0.0, 40.0, None, EARTH_RADIUS_KM
        )
        assert (reh_recovered - EARTH_RADIUS_KM) == pytest.approx(ALT, rel=1e-3)

    def test_off_limb_raises(self):
        # At VZA = 90 the line of sight is tangent to the Earth (cone angle == limb),
        # so even the centroid no longer intersects the surface.
        with pytest.raises(OffLimbError):
            geo._solve_viewing_triangle(0.0, 0.0, 30.0, 0.0, 90.0, ALT, EARTH_RADIUS_KM)


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
        # Boresight on Earth but the box corner is off-limb: the default policy
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


class TestEffectiveConeAngle:
    def test_zero_cross_reduces_to_inplane(self):
        # With no cross-scan offset, the effective cone angle is just the in-plane one.
        assert geo._effective_cone_angle_deg(50.0, 0.0) == pytest.approx(50.0)

    def test_cross_offset_increases_cone_angle(self):
        # Adding a perpendicular (cross-scan) leg can only increase the total angle.
        assert geo._effective_cone_angle_deg(50.0, 5.0) > 50.0

    def test_nadir_plus_cross_equals_cross(self):
        # From nadir, a pure cross-scan offset is the cone angle itself.
        assert geo._effective_cone_angle_deg(0.0, 3.0) == pytest.approx(3.0)


class TestPartialOffLimb:
    def test_default_flags_partial_coverage(self):
        # Boresight is well on Earth (VZA 85 -> cone angle ~61 deg < limb ~62 deg),
        # but the limb-ward box corner is off-limb. Default policy: truncate + flag.
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


class TestCheckBoxWithinLimb:
    def test_passthrough_when_fully_on_earth(self):
        reh = EARTH_RADIUS_KM + NOMINAL_ALTITUDE_KM
        result = geo._check_box_within_limb(30.0, 1.34, 1.26, reh, EARTH_RADIUS_KM, "flag")
        assert result == (1.34, 1.26, False)

    def test_flag_truncates_corner_inside_limb(self):
        reh = EARTH_RADIUS_KM + NOMINAL_ALTITUDE_KM
        limb = geo._limb_cone_angle_deg(reh, EARTH_RADIUS_KM)
        alpha0 = limb - 0.5  # close enough to the limb that the corner overshoots
        # Raise mode rejects it.
        with pytest.raises(PartialFootprintError):
            geo._check_box_within_limb(alpha0, 1.34, 1.26, reh, EARTH_RADIUS_KM, "raise")
        # Flag mode shrinks the along-scan extent so the corner sits inside the limb
        # and reports the truncation.
        new_along, new_cross, truncated = geo._check_box_within_limb(alpha0, 1.34, 1.26, reh, EARTH_RADIUS_KM, "flag")
        assert truncated is True
        assert new_along < 1.34
        assert new_cross == 1.26
        corner = geo._effective_cone_angle_deg(alpha0 + new_along, new_cross)
        assert corner <= limb + 1e-6


class TestAltitudeRecoveryPath:
    def test_box_without_altitude_matches_box_with_altitude(self):
        # Build a self-consistent footprint/subsatellite pair for a known altitude,
        # then confirm the no-altitude path (recovering Re+h from positions) gives a
        # box close to the altitude-supplied path.
        theta = math.radians(50.0)
        reh = EARTH_RADIUS_KM + ALT
        alpha = math.asin(EARTH_RADIUS_KM * math.sin(theta) / reh)
        gamma = theta - alpha
        subsat_lat = math.degrees(EARTH_RADIUS_KM * gamma / EARTH_RADIUS_KM)

        with_alt = compute_footprint_bounding_box(0.0, 0.0, subsat_lat, 0.0, 50.0, altitude_km=ALT)
        without_alt = compute_footprint_bounding_box(0.0, 0.0, subsat_lat, 0.0, 50.0)
        assert without_alt.lat_min == pytest.approx(with_alt.lat_min, abs=0.05)
        assert without_alt.lat_max == pytest.approx(with_alt.lat_max, abs=0.05)
