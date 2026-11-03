"""Integration tests for product.aggregate_external_variables.

These wire the whole samples -> footprint path together: fake readers in the
readers dict, the RadialWeigher, and the aggregation engine. The fake readers
serve *uniform* grids so the expected aggregated values are independent of the
(stand-in) PSF weight kernel and can be asserted exactly:

* a weighted mean of a constant field is that constant (weights cancel);
* a weighted mode of a single-class field is that class.

They also lock in the output contract: variable naming, per-footprint array shapes,
dtypes, the standard-deviation companions, the ranked-scene extras, and the NaN
"no data" fill (integer variables are returned as float64 so they can carry it).
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.geometry import MissingGeolocationError
from libera_utils.footprint_matching.product import aggregate_external_variables, build_radiometer_footprints
from libera_utils.footprint_matching.types import (
    BoundingBox,
    CameraFootprintQualityFlag,
    OperationalMode,
    PseudoFootprint,
    RadiometerFootprint,
)
from tests.plugins.footprint_matching_fixtures import FakeMeanReader, fake_radiometer_footprint


def _footprints() -> list[RadiometerFootprint]:
    return [
        fake_radiometer_footprint(BoundingBox(0.2, 0.8, 0.2, 0.8), 0.5, 0.5),
        fake_radiometer_footprint(BoundingBox(0.2, 0.8, 0.3, 0.9), 0.5, 0.6),
    ]


# Spacecraft distance from Earth's center for an ~835 km orbit, km.
_SATELLITE_RADIUS_KM = 7213.0


def _with_ephemeris(l1b: dict, sub_lat, sub_lon) -> dict:
    """``l1b`` plus the time and L1B ephemeris build_radiometer_footprints requires."""
    n = len(l1b["latitude"])
    return {
        "RADIOMETER_TIME": np.array([np.datetime64("2026-06-11T00:00:00", "ns")] * n),
        **l1b,
        "subsatellite_latitude": np.broadcast_to(np.asarray(sub_lat, dtype=np.float64), (n,)).copy(),
        "subsatellite_longitude": np.broadcast_to(np.asarray(sub_lon, dtype=np.float64), (n,)).copy(),
        "satellite_radius_km": np.full(n, _SATELLITE_RADIUS_KM),
    }


class TestAggregateExternalVariables:
    def test_output_keys_and_shapes(self, fake_sources):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), fake_sources)
        expected_keys = {
            "_fake_agg_mean_field_a",
            "_fake_agg_mean_field_a_standard_deviation",
            "_fake_agg_mean_field_b",
            "_fake_agg_mean_field_b_standard_deviation",
            "_fake_agg_cat_surface",
            "_fake_agg_cat_surface_primary",
            "_fake_agg_cat_surface_secondary",
        }
        assert set(result) == expected_keys
        for array in result.values():
            assert array.shape == (2,)

    def test_weighted_mean_of_constant_field_is_that_constant(self, fake_sources):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), fake_sources)
        np.testing.assert_allclose(result["_fake_agg_mean_field_a"], [5.0, 5.0])
        np.testing.assert_allclose(result["_fake_agg_mean_field_b"], [2.0, 2.0])
        # Constant field -> zero within-footprint standard deviation.
        np.testing.assert_allclose(result["_fake_agg_mean_field_a_standard_deviation"], [0.0, 0.0], atol=1e-6)

    def test_categorical_mode_and_ranked_scenes(self, fake_sources):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), fake_sources)
        # Single-class field -> mode and primary are that class, secondary does not exist.
        # Integer variables come back as float64 so a missing value can stay NaN until the
        # product's integer fill replaces it.
        assert result["_fake_agg_cat_surface"].dtype == np.float64
        np.testing.assert_array_equal(result["_fake_agg_cat_surface"], [3, 3])
        np.testing.assert_array_equal(result["_fake_agg_cat_surface_primary"], [3, 3])
        # No second class -> NaN.
        assert np.isnan(result["_fake_agg_cat_surface_secondary"]).all()

    def test_missing_data_yields_nan_fill(self, fake_sources):
        # A footprint whose box is far from the readers' grid gets no samples -> no usable
        # data -> NaN for float and integer variables alike.
        far = [fake_radiometer_footprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5)]
        result = aggregate_external_variables(OperationalMode.CAM, far, fake_sources)
        assert np.isnan(result["_fake_agg_mean_field_a"][0])
        # The integer categorical stays NaN; the product's -1 integer fill replaces it on write.
        assert np.isnan(result["_fake_agg_cat_surface"][0])

    def test_requires_sources_or_paths(self):
        with pytest.raises(ValueError, match="either sources or source_file_paths"):
            aggregate_external_variables(OperationalMode.CAM, _footprints())

    def test_subset_of_readers_raises(self, fake_sources):
        subset = {key: reader for key, reader in fake_sources.items() if key == FakeMeanReader.READER_KEY}
        with pytest.raises(KeyError, match="sources has no entry for active sources"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(), subset)

    def test_reader_the_mode_does_not_use_raises(self, fake_sources, tmp_path):
        extra_path = tmp_path / "extra.bin"
        extra_path.touch()
        extra = {**fake_sources, "_fake_extra": FakeMeanReader([extra_path])}
        with pytest.raises(KeyError, match=r"keys \['_fake_extra'\] that mode 'FMATCH-CAM' does not use"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(), extra)

    def test_sources_and_paths_together_raise(self, fake_sources, tmp_path):
        paths = {key: [tmp_path / f"{key}.bin"] for key in fake_sources}
        with pytest.raises(ValueError, match="not both"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(), fake_sources, source_file_paths=paths)

    def test_runs_end_to_end_with_angular_psf_weigher(self, fake_sources):
        from libera_utils.footprint_matching.weighting import AngularPSFWeigher

        # Footprints carrying the geometry the CERES-PSF weigher needs.
        footprints = [
            fake_radiometer_footprint(
                BoundingBox(0.2, 0.8, 0.2, 0.8),
                latitude=0.5,
                longitude=0.5,
                subsatellite_latitude=1.5,
                subsatellite_longitude=0.5,
                satellite_radius_km=7213.0,
                cone_angle_rate=-1.0,
                viewing_zenith_angle=20.0,
            )
        ]
        result = aggregate_external_variables(
            OperationalMode.CAM, footprints, fake_sources, weigher=AngularPSFWeigher()
        )
        # Constant input fields -> the weighted mean is that constant regardless of the
        # (PSF) weights, so the value is exact even with the real angular weighting.
        np.testing.assert_allclose(result["_fake_agg_mean_field_a"], [5.0])
        np.testing.assert_allclose(result["_fake_agg_mean_field_b"], [2.0])
        np.testing.assert_array_equal(result["_fake_agg_cat_surface"], [3])

    def test_pseudo_footprints_raise(self, fake_sources):
        # A camera pseudo-footprint carries no subsatellite point or satellite radius, so no
        # weigher can size its PSF.
        footprint = PseudoFootprint(
            time=np.datetime64("2026-01-01T00:00:00", "ns"),
            slice_x=slice(0, 4),
            slice_y=slice(0, 4),
            center_ix=2,
            center_iy=2,
            latitude=0.5,
            longitude=0.5,
            altitude=0.0,
            solar_zenith_angle=30.0,
            viewing_zenith_angle=0.0,
            relative_azimuth_angle=90.0,
            bbox=BoundingBox(0.2, 0.8, 0.2, 0.8),
            q_flags=CameraFootprintQualityFlag(0),
        )
        with pytest.raises(ValueError, match="no L1B ephemeris"):
            aggregate_external_variables(OperationalMode.CAM, [footprint], fake_sources)


class TestBuildRadiometerFootprints:
    """build_radiometer_footprints ray-traces each L1B record from its ephemeris."""

    @staticmethod
    def _l1b(**overrides):
        inputs = {
            "RADIOMETER_TIME": np.array(["2026-06-11T00:00:00", "2026-06-11T00:00:01"], dtype="datetime64[ns]"),
            "latitude": np.array([10.0, np.nan]),
            "longitude": np.array([20.0, np.nan]),
            "viewing_zenith_angle": np.array([15.0, np.nan]),
            "subsatellite_latitude": np.array([11.5, 11.5]),
            "subsatellite_longitude": np.array([20.0, 20.0]),
            "satellite_radius_km": np.array([7213.0, 7213.0]),
            "cone_angle_rate": np.array([-1.0, np.nan]),
        }
        inputs.update(overrides)
        return inputs

    def test_carries_ephemeris_time_and_flags_off_earth_views(self):
        on_earth, off_earth = build_radiometer_footprints(self._l1b())
        assert not on_earth.off_limb
        assert on_earth.bbox.lat_min < 10.0 < on_earth.bbox.lat_max
        assert on_earth.satellite_radius_km == 7213.0
        assert on_earth.cone_angle_rate == -1.0
        assert on_earth.time == np.datetime64("2026-06-11T00:00:00", "ns")
        # The off-Earth record keeps its index, flagged, with no scan rate.
        assert off_earth.off_limb
        assert off_earth.bbox is None
        assert off_earth.cone_angle_rate is None
        assert off_earth.time == np.datetime64("2026-06-11T00:00:01", "ns")

    def test_off_earth_view_aggregates_to_fill(self, fake_sources):
        from libera_utils.footprint_matching.weighting import AngularPSFWeigher

        # An L1B space view has no box and a NaN viewing zenith; aggregation leaves it at
        # fill with zero coverage instead of building its geometry.
        footprints = build_radiometer_footprints(
            self._l1b(
                latitude=np.array([0.5, np.nan]),
                longitude=np.array([0.5, np.nan]),
                subsatellite_latitude=np.array([1.5, 1.5]),
                subsatellite_longitude=np.array([0.5, 0.5]),
            )
        )
        values, coverage = aggregate_external_variables(
            OperationalMode.CAM, footprints, fake_sources, weigher=AngularPSFWeigher(), return_coverage=True
        )
        np.testing.assert_allclose(values["_fake_agg_mean_field_a"], [5.0, np.nan])
        assert coverage[1] == 0.0

    def test_missing_ephemeris_raises(self):
        inputs = self._l1b()
        del inputs["satellite_radius_km"]
        with pytest.raises(ValueError, match="satellite_radius_km"):
            build_radiometer_footprints(inputs)

    def test_each_footprint_gets_a_box_enclosing_its_boresight(self):
        l1b = _with_ephemeris(
            {
                "latitude": np.array([1.0, -30.0], dtype=np.float32),
                "longitude": np.array([11.0, 100.0], dtype=np.float32),
                "viewing_zenith_angle": np.array([0.0, 40.0], dtype=np.float32),
            },
            sub_lat=[1.5, -28.0],
            sub_lon=[11.0, 100.0],
        )
        footprints = build_radiometer_footprints(l1b)

        assert len(footprints) == 2
        for footprint, lat, lon in zip(footprints, l1b["latitude"], l1b["longitude"], strict=True):
            assert footprint.bbox.lat_min <= lat <= footprint.bbox.lat_max
            assert footprint.bbox.lon_min <= lon <= footprint.bbox.lon_max
            assert footprint.satellite_radius_km == _SATELLITE_RADIUS_KM
            # No Cone_Angle_Rate supplied -> unknown scan rate.
            assert footprint.cone_angle_rate is None

    def test_oblique_view_gives_a_larger_box(self):
        boresight = {"latitude": np.array([0.0]), "longitude": np.array([0.0]), "viewing_zenith_angle": np.array([0.0])}
        (nadir,) = build_radiometer_footprints(_with_ephemeris(boresight, sub_lat=0.0, sub_lon=0.0))
        (oblique,) = build_radiometer_footprints(_with_ephemeris(boresight, sub_lat=8.0, sub_lon=0.0))
        assert (oblique.bbox.lat_max - oblique.bbox.lat_min) > (nadir.bbox.lat_max - nadir.bbox.lat_min)

    def test_ephemeris_and_scan_rate_are_carried_onto_the_footprint(self):
        l1b = _with_ephemeris(
            {
                "latitude": np.array([1.0], dtype=np.float32),
                "longitude": np.array([11.0], dtype=np.float32),
                "viewing_zenith_angle": np.array([40.0], dtype=np.float32),
                "cone_angle_rate": np.array([12.0], dtype=np.float64),
            },
            sub_lat=1.5,
            sub_lon=11.0,
        )
        (footprint,) = build_radiometer_footprints(l1b)

        assert footprint.bbox.lat_min <= 1.0 <= footprint.bbox.lat_max
        assert footprint.subsatellite_latitude == 1.5
        assert footprint.subsatellite_longitude == 11.0
        assert footprint.satellite_radius_km == _SATELLITE_RADIUS_KM
        assert footprint.cone_angle_rate == 12.0

    def test_oblique_ray_traced_box_is_not_centered_on_the_boresight(self):
        """A ray-traced oblique footprint is skewed along the scan plane, not centered on the boresight."""
        # Boresight at the equator, satellite offset to the north along the same meridian.
        l1b = _with_ephemeris(
            {
                "latitude": np.array([0.0], dtype=np.float32),
                "longitude": np.array([0.0], dtype=np.float32),
                "viewing_zenith_angle": np.array([55.0], dtype=np.float32),
                "cone_angle_rate": np.array([np.nan], dtype=np.float64),
            },
            sub_lat=10.0,
            sub_lon=0.0,
        )
        (ray_traced,) = build_radiometer_footprints(l1b)

        assert ray_traced.bbox.lat_min <= 0.0 <= ray_traced.bbox.lat_max
        assert abs(ray_traced.bbox.lat_max + ray_traced.bbox.lat_min) > 1e-2
        # A fill cone-angle rate is normalized to None.
        assert ray_traced.cone_angle_rate is None

    def test_footprint_beyond_the_horizon_raises(self):
        # 30 deg from the subsatellite point is past the horizon of an ~835 km orbit.
        l1b = _with_ephemeris(
            {"latitude": np.array([0.0]), "longitude": np.array([0.0]), "viewing_zenith_angle": np.array([0.0])},
            sub_lat=30.0,
            sub_lon=0.0,
        )
        with pytest.raises(MissingGeolocationError):
            build_radiometer_footprints(l1b)

    def test_missing_time_raises(self):
        inputs = self._l1b()
        del inputs["RADIOMETER_TIME"]
        with pytest.raises(ValueError, match="RADIOMETER_TIME"):
            build_radiometer_footprints(inputs)


class TestAggregateReturnsCoverage:
    """return_coverage adds a per-footprint coverage array without changing the values."""

    def test_covered_footprints_report_full_coverage(self, fake_sources):
        # The fake readers serve a uniform 10 km grid that fills the footprint box, so every
        # reference bin of the radial weigher's contour holds a sample -> coverage 1.
        values, coverage = aggregate_external_variables(
            OperationalMode.CAM, _footprints(), fake_sources, return_coverage=True
        )
        assert coverage.shape == (2,)
        np.testing.assert_allclose(coverage, [1.0, 1.0], atol=1e-6)
        # The values dict is exactly what the no-coverage call returns.
        assert set(values) == set(aggregate_external_variables(OperationalMode.CAM, _footprints(), fake_sources))

    def test_uncovered_footprint_reports_zero_coverage(self, fake_sources):
        far = [fake_radiometer_footprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5)]
        _values, coverage = aggregate_external_variables(OperationalMode.CAM, far, fake_sources, return_coverage=True)
        np.testing.assert_allclose(coverage, [0.0])
