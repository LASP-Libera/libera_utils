"""Integration tests for product.aggregate_external_variables.

These wire the whole tiles -> footprint path together: fake readers behind a
TileManager, the RadialWeigher, and the aggregation engine. The fake readers return
*uniform* grids so the expected aggregated values are independent of the (stand-in)
PSF weight kernel and can be asserted exactly:

* a weighted mean of a constant field is that constant (weights cancel);
* a weighted mode of a single-class field is that class.

They also lock in the output contract: variable naming, per-footprint array shapes,
dtypes, the standard-deviation companions, the ranked-scene extras, and the NaN
"no data" fill (integer variables are returned as float64 so they can carry it).
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.product import aggregate_external_variables, build_radiometer_footprints
from libera_utils.footprint_matching.tiling import TileManager
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode
from tests.unit.test_footprint_matching.fake_reader_fixtures import (
    FakeCatReader,
    FakeFootprint,
    FakeMeanReader,
    make_manager,
)


def _footprints() -> list[FakeFootprint]:
    return [
        FakeFootprint(BoundingBox(0.2, 0.8, 0.2, 0.8), 0.5, 0.5),
        FakeFootprint(BoundingBox(0.2, 0.8, 0.3, 0.9), 0.5, 0.6),  # overlapping -> cache reuse
    ]


class TestAggregateExternalVariables:
    def test_output_keys_and_shapes(self, tmp_path):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), make_manager(tmp_path))
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

    def test_weighted_mean_of_constant_field_is_that_constant(self, tmp_path):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), make_manager(tmp_path))
        np.testing.assert_allclose(result["_fake_agg_mean_field_a"], [5.0, 5.0])
        np.testing.assert_allclose(result["_fake_agg_mean_field_b"], [2.0, 2.0])
        # Constant field -> zero within-footprint standard deviation.
        np.testing.assert_allclose(result["_fake_agg_mean_field_a_standard_deviation"], [0.0, 0.0], atol=1e-6)

    def test_categorical_mode_and_ranked_scenes(self, tmp_path):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), make_manager(tmp_path))
        # Single-class field -> mode and primary are that class, secondary does not exist.
        # Integer variables come back as float64 so a missing value can stay NaN until the
        # product's integer fill replaces it.
        assert result["_fake_agg_cat_surface"].dtype == np.float64
        np.testing.assert_array_equal(result["_fake_agg_cat_surface"], [3, 3])
        np.testing.assert_array_equal(result["_fake_agg_cat_surface_primary"], [3, 3])
        # No second class -> NaN.
        assert np.isnan(result["_fake_agg_cat_surface_secondary"]).all()

    def test_missing_data_yields_nan_fill(self, tmp_path):
        # A footprint whose box is far from the readers' grid gets an empty/uncovered
        # merged tile -> no usable data -> NaN for float and integer variables alike.
        far = [FakeFootprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5)]
        result = aggregate_external_variables(OperationalMode.CAM, far, make_manager(tmp_path))
        assert np.isnan(result["_fake_agg_mean_field_a"][0])
        # The integer categorical stays NaN; the product's -1 integer fill replaces it on write.
        assert np.isnan(result["_fake_agg_cat_surface"][0])

    def test_requires_manager_or_paths(self):
        with pytest.raises(ValueError, match="either a tile_manager or source_file_paths"):
            aggregate_external_variables(OperationalMode.CAM, _footprints())

    def test_prefetch_produces_identical_results(self, tmp_path):
        # Background tile prefetch is a pure cache warm-up: enabling it must not change
        # any aggregated value (it only overlaps reader I/O with compute).
        footprints = _footprints()
        baseline = aggregate_external_variables(OperationalMode.CAM, footprints, make_manager(tmp_path))

        prefetch_manager = TileManager(
            {
                FakeMeanReader.READER_KEY: FakeMeanReader(tmp_path / "mean_pf.bin"),
                FakeCatReader.READER_KEY: FakeCatReader(tmp_path / "cat_pf.bin"),
            },
            OperationalMode.CAM,
            prefetch_lookahead=3,
        )
        try:
            prefetched = aggregate_external_variables(OperationalMode.CAM, footprints, prefetch_manager)
        finally:
            prefetch_manager.shutdown()

        assert set(prefetched) == set(baseline)
        for key, expected in baseline.items():
            np.testing.assert_array_equal(prefetched[key], expected)

    def test_runs_end_to_end_with_angular_psf_weigher(self, tmp_path):
        from libera_utils.footprint_matching.weighting import AngularPSFWeigher

        # Footprints carrying the geometry the CERES-PSF weigher needs.
        footprints = [
            FakeFootprint(
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
            OperationalMode.CAM, footprints, make_manager(tmp_path), weigher=AngularPSFWeigher()
        )
        # Constant input fields -> the weighted mean is that constant regardless of the
        # (PSF) weights, so the value is exact even with the real angular weighting.
        np.testing.assert_allclose(result["_fake_agg_mean_field_a"], [5.0])
        np.testing.assert_allclose(result["_fake_agg_mean_field_b"], [2.0])
        np.testing.assert_array_equal(result["_fake_agg_cat_surface"], [3])

    def test_angular_psf_weigher_rejects_footprints_without_ephemeris(self, tmp_path):
        from libera_utils.footprint_matching.weighting import AngularPSFWeigher

        # A camera-style footprint carries no subsatellite point or satellite radius, so it has
        # no viewing frame to weight by.
        with pytest.raises(ValueError, match="ephemeris"):
            aggregate_external_variables(
                OperationalMode.CAM, _footprints(), make_manager(tmp_path), weigher=AngularPSFWeigher()
            )


class TestBuildRadiometerFootprints:
    """build_radiometer_footprints ray-traces each L1B record from its ephemeris."""

    @staticmethod
    def _l1b(**overrides):
        inputs = {
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

    def test_carries_ephemeris_and_flags_off_earth_views(self):
        on_earth, off_earth = build_radiometer_footprints(self._l1b())
        assert not on_earth.off_limb
        assert on_earth.bbox.lat_min < 10.0 < on_earth.bbox.lat_max
        assert on_earth.satellite_radius_km == 7213.0
        assert on_earth.cone_angle_rate == -1.0
        # The off-Earth record keeps its index, flagged, with no scan rate.
        assert off_earth.off_limb
        assert off_earth.cone_angle_rate is None

    def test_missing_ephemeris_raises(self):
        inputs = self._l1b()
        del inputs["satellite_radius_km"]
        with pytest.raises(ValueError, match="satellite_radius_km"):
            build_radiometer_footprints(inputs)


class TestAggregateReturnsCoverage:
    """return_coverage adds a per-footprint coverage array without changing the values."""

    def test_covered_footprints_report_full_coverage(self, tmp_path):
        # The fake readers return a uniform grid that fills the footprint box, so the
        # radial weigher's whole in-contour energy is backed by data -> coverage ~1.
        values, coverage = aggregate_external_variables(
            OperationalMode.CAM, _footprints(), make_manager(tmp_path), return_coverage=True
        )
        assert coverage.shape == (2,)
        np.testing.assert_allclose(coverage, [1.0, 1.0], atol=1e-6)
        # The values dict is exactly what the no-coverage call returns.
        assert set(values) == set(
            aggregate_external_variables(OperationalMode.CAM, _footprints(), make_manager(tmp_path))
        )

    def test_uncovered_footprint_reports_zero_coverage(self, tmp_path):
        far = [FakeFootprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5)]
        _values, coverage = aggregate_external_variables(
            OperationalMode.CAM, far, make_manager(tmp_path), return_coverage=True
        )
        np.testing.assert_allclose(coverage, [0.0])
