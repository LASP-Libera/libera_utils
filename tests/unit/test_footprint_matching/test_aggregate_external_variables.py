"""Integration tests for product.aggregate_external_variables.

These wire the whole tiles -> footprint path together: fake readers behind a
TileManager, the RadialWeigher, and the aggregation engine. The fake readers return
*uniform* grids so the expected aggregated values are independent of the (stand-in)
PSF weight kernel and can be asserted exactly:

* a weighted mean of a constant field is that constant (weights cancel);
* a weighted mode of a single-class field is that class.

They also lock in the output contract: variable naming, per-footprint array shapes,
dtypes, the standard-deviation companions, the ranked-scene extras, and the
NaN(float)/0(int) "no data" fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

# Importing the readers subpackage registers the built-in readers (used by the
# _map_product_specs lock-in test below).
import libera_utils.footprint_matching.readers  # noqa: F401
from libera_utils.footprint_matching.aggregation import _map_product_specs
from libera_utils.footprint_matching.product import aggregate_external_variables
from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.tiling import TileManager
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# A small uniform grid returned by every fake reader, sitting inside the footprint
# bounding box used by the tests so the radial weigher gives all cells positive weight.
_GRID_LATS = np.array([0.3, 0.5, 0.7], dtype=np.float64)
_GRID_LONS = np.array([0.3, 0.5, 0.7], dtype=np.float64)


class _FakeMeanReader(GriddedDataReader):
    """Multi-variable reader: two continuous fields with constant values 5 and 2."""

    READER_KEY = "_fake_agg_mean"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    VARIABLES = (
        VariableSpec(name="field_a", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
        VariableSpec(name="field_b", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
    )

    def _load_spatial_region(self, bbox: BoundingBox):
        # 3-D (n_var, n_lat, n_lon): plane 0 == 5.0 everywhere, plane 1 == 2.0.
        data = np.stack(
            [
                np.full((_GRID_LATS.size, _GRID_LONS.size), 5.0, dtype=np.float32),
                np.full((_GRID_LATS.size, _GRID_LONS.size), 2.0, dtype=np.float32),
            ]
        )
        return data, _GRID_LATS, _GRID_LONS


class _FakeCatReader(GriddedDataReader):
    """Single-variable categorical reader: every cell is class 3, plus ranked extras."""

    READER_KEY = "_fake_agg_cat"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    VARIABLES = (
        VariableSpec(
            name="surface",
            dtype="int16",
            aggregation="weighted_mode",
            required_mode=OperationalMode.CAM,
            n_categories=6,
        ),
    )
    ADDITIONAL_PRODUCT_VARIABLES = (
        VariableSpec(
            name="surface_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            required_mode=OperationalMode.CAM,
            n_categories=6,
        ),
        VariableSpec(
            name="surface_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            required_mode=OperationalMode.CAM,
            n_categories=6,
        ),
    )

    def _load_spatial_region(self, bbox: BoundingBox):
        # 2-D single-variable grid, all class 3.
        data = np.full((_GRID_LATS.size, _GRID_LONS.size), 3.0, dtype=np.float32)
        return data, _GRID_LATS, _GRID_LONS


@dataclass
class _FakeFootprint:
    """Minimal footprint exposing what the weigher and TileManager need."""

    bbox: BoundingBox
    latitude: float
    longitude: float
    # Optional geometry consumed by the angular-frame weigher (ignored by RadialWeigher).
    subsatellite_latitude: float | None = None
    subsatellite_longitude: float | None = None
    cone_angle_rate: float | None = None
    viewing_zenith_angle: float = 0.0


def _manager(tmp_path: Path) -> TileManager:
    """A TileManager holding just the two fake readers, in CAM mode."""
    readers = {
        _FakeMeanReader.READER_KEY: _FakeMeanReader(tmp_path / "mean.bin"),
        _FakeCatReader.READER_KEY: _FakeCatReader(tmp_path / "cat.bin"),
    }
    return TileManager(readers, OperationalMode.CAM)


def _footprints() -> list[_FakeFootprint]:
    return [
        _FakeFootprint(BoundingBox(0.2, 0.8, 0.2, 0.8), 0.5, 0.5),
        _FakeFootprint(BoundingBox(0.2, 0.8, 0.3, 0.9), 0.5, 0.6),  # overlapping -> cache reuse
    ]


class TestAggregateExternalVariables:
    def test_output_keys_and_shapes(self, tmp_path):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), _manager(tmp_path))
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
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), _manager(tmp_path))
        np.testing.assert_allclose(result["_fake_agg_mean_field_a"], [5.0, 5.0])
        np.testing.assert_allclose(result["_fake_agg_mean_field_b"], [2.0, 2.0])
        # Constant field -> zero within-footprint standard deviation.
        np.testing.assert_allclose(result["_fake_agg_mean_field_a_standard_deviation"], [0.0, 0.0], atol=1e-6)

    def test_categorical_mode_and_ranked_scenes(self, tmp_path):
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(), _manager(tmp_path))
        # Single-class field -> mode and primary are that class, secondary does not exist.
        assert result["_fake_agg_cat_surface"].dtype == np.int16
        np.testing.assert_array_equal(result["_fake_agg_cat_surface"], [3, 3])
        np.testing.assert_array_equal(result["_fake_agg_cat_surface_primary"], [3, 3])
        # No second class -> NaN -> integer fill 0.
        np.testing.assert_array_equal(result["_fake_agg_cat_surface_secondary"], [0, 0])

    def test_missing_data_yields_nan_float_fill(self, tmp_path):
        # A footprint whose box is far from the readers' grid gets an empty/uncovered
        # merged tile -> no usable data -> NaN for float variables.
        far = [_FakeFootprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5)]
        result = aggregate_external_variables(OperationalMode.CAM, far, _manager(tmp_path))
        assert np.isnan(result["_fake_agg_mean_field_a"][0])
        # Integer categorical falls back to the 0 fill (cannot hold NaN).
        assert result["_fake_agg_cat_surface"][0] == 0

    def test_requires_manager_or_paths(self):
        with pytest.raises(ValueError, match="either a tile_manager or source_file_paths"):
            aggregate_external_variables(OperationalMode.CAM, _footprints())

    def test_runs_end_to_end_with_angular_psf_weigher(self, tmp_path):
        from libera_utils.footprint_matching.weighting import AngularPSFWeigher

        # Footprints carrying the geometry the CERES-PSF weigher needs.
        footprints = [
            _FakeFootprint(
                BoundingBox(0.2, 0.8, 0.2, 0.8),
                latitude=0.5,
                longitude=0.5,
                subsatellite_latitude=1.5,
                subsatellite_longitude=0.5,
                cone_angle_rate=-1.0,
                viewing_zenith_angle=20.0,
            )
        ]
        result = aggregate_external_variables(
            OperationalMode.CAM, footprints, _manager(tmp_path), weigher=AngularPSFWeigher()
        )
        # Constant input fields -> the weighted mean is that constant regardless of the
        # (PSF) weights, so the value is exact even with the real angular weighting.
        np.testing.assert_allclose(result["_fake_agg_mean_field_a"], [5.0])
        np.testing.assert_allclose(result["_fake_agg_mean_field_b"], [2.0])
        np.testing.assert_array_equal(result["_fake_agg_cat_surface"], [3])


class TestProductSpecPlaneMapping:
    """_map_product_specs must associate every product spec with a read plane."""

    def test_igbp_ranked_scenes_map_to_the_surface_type_plane(self):
        igbp = ReaderRegistry.get("igbp")
        mapping = dict(_map_product_specs(igbp))
        # IGBP reads a single plane (surface_type at index 0); every product spec
        # (surface_type + its ranked primary/secondary/tertiary extras) maps to it.
        by_name = {spec.name: idx for spec, idx in mapping.items()}
        assert by_name["surface_type"] == 0
        assert by_name["surface_type_primary"] == 0
        assert by_name["surface_type_secondary"] == 0
        assert by_name["surface_type_tertiary"] == 0

    def test_std_companions_map_to_their_parent_plane(self):
        era5 = ReaderRegistry.get("era5")
        mapping = {spec.name: idx for spec, idx in _map_product_specs(era5)}
        # wind_u10 is read plane 0, wind_v10 is read plane 1; each std companion
        # shares its parent's plane.
        assert mapping["wind_u10"] == mapping["wind_u10_standard_deviation"]
        assert mapping["wind_v10"] == mapping["wind_v10_standard_deviation"]
        assert mapping["wind_u10"] != mapping["wind_v10"]


class TestAggregateReturnsCoverage:
    """return_coverage adds a per-footprint coverage array without changing the values."""

    def test_covered_footprints_report_full_coverage(self, tmp_path):
        # The fake readers return a uniform grid that fills the footprint box, so the
        # radial weigher's whole in-contour energy is backed by data -> coverage ~1.
        values, coverage = aggregate_external_variables(
            OperationalMode.CAM, _footprints(), _manager(tmp_path), return_coverage=True
        )
        assert coverage.shape == (2,)
        np.testing.assert_allclose(coverage, [1.0, 1.0], atol=1e-6)
        # The values dict is exactly what the no-coverage call returns.
        assert set(values) == set(aggregate_external_variables(OperationalMode.CAM, _footprints(), _manager(tmp_path)))

    def test_uncovered_footprint_reports_zero_coverage(self, tmp_path):
        far = [_FakeFootprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5)]
        _values, coverage = aggregate_external_variables(
            OperationalMode.CAM, far, _manager(tmp_path), return_coverage=True
        )
        np.testing.assert_allclose(coverage, [0.0])
