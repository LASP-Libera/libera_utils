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
    bounding_box_from_boresight,
    compute_footprint_bounding_box,
    psf_ground_radius_km,
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


class TestBoundingBoxFromPoints:
    """The box assembler was extracted to a public helper shared with the camera path.

    These tests pin its behaviour directly (the radiometer entry point above still
    exercises it end-to-end, guarding the extraction).
    """

    def test_simple_box_from_corner_points(self):
        # Four corner points -> the enclosing lat/lon rectangle, centred anchor irrelevant.
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

    def _frame(self, blat, blon, slat, slon, vza, alt):
        sat, boresight_dir, subsat_normal = geo._viewing_geometry(blat, blon, slat, slon, vza, alt)
        cross_axis = geo._scan_frame_axes(boresight_dir, subsat_normal)
        return sat, boresight_dir, cross_axis

    @pytest.mark.parametrize("delta", [-1.2, -0.4, 0.0, 0.7, 1.5])
    @pytest.mark.parametrize("beta", [-0.6, 0.0, 0.5])
    def test_round_trip_recovers_delta_beta(self, delta, beta):
        blat, blon, slat, slon, vza = 10.0, 20.0, 12.0, 20.0, 30.0
        sat, boresight_dir, cross_axis = self._frame(blat, blon, slat, slon, vza, ALT)

        # Forward: rotate the boresight by (delta, beta) and ray-trace to the ground.
        direction = geo._offset_ray_direction(boresight_dir, cross_axis, delta, beta)
        hit = geo._ray_ellipsoid_intersection(sat, direction)
        assert hit is not None
        lat, lon, _ = geo._ecef_to_geodetic(hit)

        # Inverse: project that ground point back to angles (duplicate the point so
        # pyproj gets a length-2 array and does not emit its length-1 scalar warning).
        rec_delta, rec_beta = geo.project_to_angular(
            np.array([lat, lat]), np.array([lon, lon]), blat, blon, slat, slon, vza, altitude_km=ALT
        )
        assert float(rec_delta[0]) == pytest.approx(delta, abs=1e-3)
        assert float(rec_beta[0]) == pytest.approx(beta, abs=1e-3)

    def test_boresight_projects_to_origin(self):
        blat, blon, slat, slon, vza = -5.0, 100.0, -4.0, 101.0, 20.0
        rec_delta, rec_beta = geo.project_to_angular(
            np.array([blat, blat]), np.array([blon, blon]), blat, blon, slat, slon, vza, altitude_km=ALT
        )
        assert float(rec_delta[0]) == pytest.approx(0.0, abs=1e-3)
        assert float(rec_beta[0]) == pytest.approx(0.0, abs=1e-3)

    def test_preserves_input_shape(self):
        lats = np.array([[10.0, 10.1], [10.2, 10.3]])
        lons = np.array([[20.0, 20.1], [20.2, 20.3]])
        d, b = geo.project_to_angular(lats, lons, 10.15, 20.15, 11.0, 20.15, 25.0, altitude_km=ALT)
        assert d.shape == (2, 2)
        assert b.shape == (2, 2)


class TestPsfGroundRadius:
    """psf_ground_radius_km projects the PSF angular extent to a ground radius."""

    def test_grows_with_altitude(self):
        low = psf_ground_radius_km(400.0, 0.0)
        high = psf_ground_radius_km(800.0, 0.0)
        assert high > low > 0.0

    def test_grows_with_viewing_zenith(self):
        # The 1/cos(vza) elongation makes an oblique view reach farther on the ground.
        assert psf_ground_radius_km(ALT, 60.0) > psf_ground_radius_km(ALT, 0.0)


class TestBoundingBoxFromBoresight:
    """The boresight-centred box is a symmetric superset needing no subsatellite point."""

    def test_box_encloses_and_is_centred_on_the_boresight(self):
        box = bounding_box_from_boresight(12.0, -45.0, 10.0)
        assert box.lat_min < 12.0 < box.lat_max
        assert box.lon_min < -45.0 < box.lon_max
        # Symmetric about the boresight in latitude.
        assert (12.0 - box.lat_min) == pytest.approx(box.lat_max - 12.0, rel=1e-3)
        assert box.truncated is False

    def test_oblique_view_yields_a_larger_box(self):
        nadir = bounding_box_from_boresight(0.0, 0.0, 0.0)
        oblique = bounding_box_from_boresight(0.0, 0.0, 60.0)
        assert (oblique.lat_max - oblique.lat_min) > (nadir.lat_max - nadir.lat_min)

    def test_high_latitude_box_is_flagged_polar(self):
        box = bounding_box_from_boresight(88.0, 0.0, 0.0)
        assert box.is_polar
