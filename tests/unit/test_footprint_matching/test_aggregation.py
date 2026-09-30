"""Unit tests for the PSF aggregation engine (aggregation.py).

These exercise the pure strategy functions against hand-computed values, the
NaN / zero-weight exclusion rules, the categorical mode / ranked-mode logic, the
coverage bookkeeping, and the dispatcher's error handling. They do not touch the
TileManager or readers -- those are covered by the integration test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from libera_utils.footprint_matching import aggregation as agg
from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridTile,
    OperationalMode,
    VariableSpec,
    with_standard_deviation_companions,
)
from libera_utils.footprint_matching.weighting import WeightField


class TestWeightedMean:
    """weighted_mean = sum(w*x)/sum(w), NaN-excluded."""

    def test_matches_hand_computed_value(self):
        values = np.array([1.0, 3.0])
        weights = np.array([1.0, 3.0])
        # (1*1 + 3*3) / (1 + 3) = 10/4 = 2.5
        assert agg.weighted_mean(values, weights)["mean"] == pytest.approx(2.5)

    def test_nan_values_are_excluded(self):
        values = np.array([2.0, np.nan, 4.0])
        weights = np.array([1.0, 5.0, 1.0])
        # NaN cell dropped -> (2 + 4)/2 = 3.0
        assert agg.weighted_mean(values, weights)["mean"] == pytest.approx(3.0)

    def test_zero_and_negative_weights_dropped(self):
        values = np.array([2.0, 100.0, 4.0])
        weights = np.array([1.0, 0.0, 1.0])  # zero-weight cell outside contour
        assert agg.weighted_mean(values, weights)["mean"] == pytest.approx(3.0)

    def test_no_usable_data_returns_nan(self):
        assert math.isnan(agg.weighted_mean(np.array([np.nan]), np.array([1.0]))["mean"])


class TestWeightedStd:
    """weighted_std = sqrt(sum(w*x^2)/sum(w) - mean^2)."""

    def test_uniform_weight_matches_population_std(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        # population std of [1,2,3] = sqrt(2/3)
        assert agg.weighted_std(values, weights)["standard_deviation"] == pytest.approx(math.sqrt(2.0 / 3.0))

    def test_constant_values_give_zero_not_negative(self):
        # Floating round-off could push a zero variance slightly negative; the clip
        # must keep the result at exactly 0.0 (not NaN from sqrt of a negative).
        values = np.array([5.0, 5.0, 5.0])
        weights = np.array([0.3, 0.6, 0.1])
        assert agg.weighted_std(values, weights)["standard_deviation"] == pytest.approx(0.0, abs=1e-12)


class TestWeightedLogMean:
    """weighted_log_mean = exp(sum(w*ln x)/sum(w)); values <= 0 excluded."""

    def test_geometric_mean_of_positive_values(self):
        values = np.array([1.0, 100.0])
        weights = np.array([1.0, 1.0])
        # geometric mean sqrt(1*100) = 10
        assert agg.weighted_log_mean(values, weights)["log_mean"] == pytest.approx(10.0)

    def test_nonpositive_values_excluded(self):
        values = np.array([-1.0, 0.0, 4.0, 9.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        # only 4 and 9 survive -> sqrt(36) = 6
        assert agg.weighted_log_mean(values, weights)["log_mean"] == pytest.approx(6.0)

    def test_all_nonpositive_returns_nan(self):
        assert math.isnan(agg.weighted_log_mean(np.array([-1.0, 0.0]), np.array([1.0, 1.0]))["log_mean"])


class TestWeightedMedian:
    def test_weighted_median_picks_halfway_value(self):
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 5.0])
        # cumulative weight 1,2,7; half = 3.5 -> reached at value 3
        assert agg.weighted_median(values, weights)["median"] == pytest.approx(3.0)


class TestWeightedMode:
    """Categorical mode + per-class coverage fractions."""

    def test_mode_is_highest_weight_category(self):
        values = np.array([0.0, 1.0, 1.0])
        weights = np.array([5.0, 1.0, 1.0])  # class 0 has more weight
        result = agg.weighted_mode(values, weights, n_categories=2)
        assert result["mode"] == pytest.approx(0.0)
        assert result["coverage_fractions"][0] == pytest.approx(5.0 / 7.0)
        assert result["coverage_fractions"][1] == pytest.approx(2.0 / 7.0)

    def test_infers_categories_when_count_unknown(self):
        # n_categories=None -> categories inferred from the data (SSF scene codes).
        values = np.array([3.0, 3.0, 7.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert agg.weighted_mode(values, weights, n_categories=None)["mode"] == pytest.approx(3.0)

    def test_tie_breaks_to_lower_category(self):
        values = np.array([2.0, 5.0])
        weights = np.array([1.0, 1.0])  # tie
        assert agg.weighted_mode(values, weights, n_categories=None)["mode"] == pytest.approx(2.0)

    def test_no_data_returns_nan_mode(self):
        assert math.isnan(agg.weighted_mode(np.array([np.nan]), np.array([1.0]), n_categories=3)["mode"])

    def test_top_of_range_code_not_dropped_for_one_based_scheme(self):
        # Regression: CLDPIX cloud phase uses 1-based codes 1..5 and declares
        # n_categories=5. The category domain must include code 5 (it lands at
        # bincount index 5, one past the declared count) rather than truncating it.
        values = np.array([5.0, 5.0, 1.0])
        weights = np.array([2.0, 2.0, 1.0])  # code 5 has the most weight
        result = agg.weighted_mode(values, weights, n_categories=5)
        assert result["mode"] == pytest.approx(5.0)
        assert result["coverage_fractions"][5] == pytest.approx(4.0 / 5.0)


class TestRankedMode:
    """primary/secondary/tertiary = 1st/2nd/3rd most common class by weight."""

    def test_ranked_ordering(self):
        # class 5 weight 4, class 2 weight 3, class 1 weight 1
        values = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 1.0])
        weights = np.ones_like(values)
        assert agg.weighted_mode_primary(values, weights, n_categories=6)["mode"] == pytest.approx(5.0)
        assert agg.weighted_mode_secondary(values, weights, n_categories=6)["mode"] == pytest.approx(2.0)
        assert agg.weighted_mode_tertiary(values, weights, n_categories=6)["mode"] == pytest.approx(1.0)

    def test_missing_rank_is_nan(self):
        # Only one class present -> secondary/tertiary do not exist.
        values = np.array([4.0, 4.0])
        weights = np.array([1.0, 1.0])
        assert agg.weighted_mode_primary(values, weights, n_categories=6)["mode"] == pytest.approx(4.0)
        assert math.isnan(agg.weighted_mode_secondary(values, weights, n_categories=6)["mode"])
        assert math.isnan(agg.weighted_mode_tertiary(values, weights, n_categories=6)["mode"])


class TestCloudRadianceStrategies:
    """The cloud/radiance-path strategies (provided for future readers)."""

    def test_coverage_fraction(self):
        values = np.array([0.0, 1.0, 1.0])
        weights = np.array([2.0, 1.0, 1.0])
        # weight where x>0 is 2; total 4 -> 0.5
        assert agg.coverage_fraction(values, weights)["fraction"] == pytest.approx(0.5)

    def test_cloud_weighted_mean_zeros_clear_cells(self):
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 1.0])
        cloud_mask = np.array([True, False, True])
        result = agg.cloud_weighted_mean(values, weights, cloud_mask=cloud_mask)
        assert result["mean"] == pytest.approx(20.0)  # (10 + 30)/2
        assert result["n_cloudy"] == pytest.approx(2.0)

    def test_overcast_fraction(self):
        values = np.array([0.0, 0.5, 0.99])
        weights = np.array([1.0, 1.0, 1.0])
        # cloudy weight (x>0) = 2; overcast (x>0.95) = 1 -> 0.5
        assert agg.overcast_fraction(values, weights)["overcast_fraction"] == pytest.approx(0.5)

    def test_broadband_radiance_total(self):
        values = np.array([1.0, 3.0])
        weights = np.array([1.0, 1.0])
        assert agg.aggregate_broadband_radiance(values, weights)["total_radiance"] == pytest.approx(2.0)

    def test_report_all_returns_usable_arrays(self):
        values = np.array([1.0, np.nan, 3.0])
        weights = np.array([1.0, 1.0, 0.0])
        result = agg.report_all(values, weights)
        np.testing.assert_array_equal(result["values"], np.array([1.0]))
        np.testing.assert_array_equal(result["weights"], np.array([1.0]))


class TestCheckCoverage:
    def test_bands(self):
        assert agg.check_coverage(0.96, 1.0) == (True, False)  # full
        assert agg.check_coverage(0.80, 1.0) == (True, True)  # partial
        assert agg.check_coverage(0.50, 1.0) == (False, False)  # discard
        assert agg.check_coverage(1.0, 0.0) == (False, False)  # no PSF energy


class TestDispatcher:
    def test_aggregate_wraps_result_with_coverage(self):
        values = np.array([2.0, 4.0])
        weights = np.array([1.0, 1.0])
        result = agg.aggregate("weighted_mean", values, weights, total_energy=2.0)
        assert result.values["mean"] == pytest.approx(3.0)
        assert result.coverage == pytest.approx(1.0)
        assert result.n_valid == 2
        assert result.n_total == 2
        assert result.partial_flag is False

    def test_coverage_drops_with_uncovered_cells(self):
        # Total PSF weight 4, but only 3 of it backed by data -> coverage 0.75 (partial).
        values = np.array([1.0, 1.0, 1.0, np.nan])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        result = agg.aggregate("weighted_mean", values, weights, total_energy=4.0)
        assert result.coverage == pytest.approx(0.75)
        assert result.partial_flag is True

    def test_unknown_strategy_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            agg.aggregate("not_a_strategy", np.array([1.0]), np.array([1.0]))


class _ParityReader:
    """Reader stand-in exercising every shared-plane case aggregate_tile_variables optimizes.

    ``field_a``/``field_b`` are continuous (each gains a ``_standard_deviation`` companion
    sharing its plane), and ``surface`` is categorical with three ranked-mode extras that
    all derive from the one ``surface`` plane. Only the attributes
    ``aggregate_tile_variables`` / ``_map_product_specs`` touch are defined.
    """

    __name__ = "_ParityReader"
    VARIABLES = (
        VariableSpec(name="field_a", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
        VariableSpec(
            name="field_b", dtype="float32", aggregation="weighted_log_mean", required_mode=OperationalMode.CAM
        ),
        VariableSpec(
            name="surface",
            dtype="int16",
            aggregation="weighted_mode",
            required_mode=OperationalMode.CAM,
            n_categories=5,
        ),
    )
    ADDITIONAL_PRODUCT_VARIABLES = (
        VariableSpec(
            name="surface_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            required_mode=OperationalMode.CAM,
            n_categories=5,
        ),
        VariableSpec(
            name="surface_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            required_mode=OperationalMode.CAM,
            n_categories=5,
        ),
        VariableSpec(
            name="surface_tertiary",
            dtype="int16",
            aggregation="weighted_mode_tertiary",
            required_mode=OperationalMode.CAM,
            n_categories=5,
        ),
    )

    @classmethod
    def product_variable_specs(cls) -> tuple[VariableSpec, ...]:
        return with_standard_deviation_companions(cls.VARIABLES) + cls.ADDITIONAL_PRODUCT_VARIABLES


def _reference_tile_variables(reader_cls, tile: GridTile, weight_field: WeightField) -> dict[str, float]:
    """Per-variable reference: what aggregate_tile_variables did before the shared-plane refactor.

    Calls the public dispatcher once per product spec (no per-plane sharing), so any
    divergence in the optimized path shows up as a value mismatch here.
    """
    data = np.asarray(tile.data)
    data_3d = data[np.newaxis, ...] if data.ndim == 2 else data
    planes = [data_3d[i].ravel() for i in range(data_3d.shape[0])]
    weights = np.asarray(weight_field.weights, dtype=float).ravel()
    out: dict[str, float] = {}
    for spec, read_index in agg._map_product_specs(reader_cls):
        plane = planes[read_index] if read_index < len(planes) else np.empty(0, dtype=float)
        n_categories = spec.n_categories
        if n_categories is None:
            n_categories = reader_cls.VARIABLES[read_index].n_categories
        result = agg.aggregate(
            spec.aggregation, plane, weights, total_energy=weight_field.total_energy, n_categories=n_categories
        )
        out[spec.name] = float(result.values[agg._SCALAR_KEY[spec.aggregation]])
    return out


class TestAggregateTileVariablesParity:
    """The shared-plane aggregate_tile_variables must match the per-variable reference exactly."""

    def _tile_and_weights(self, rng: np.random.Generator) -> tuple[GridTile, WeightField]:
        n_lat, n_lon = 5, 6
        field_a = rng.normal(10.0, 3.0, size=(n_lat, n_lon))
        field_b = np.abs(rng.normal(4.0, 2.0, size=(n_lat, n_lon))) + 0.01  # positive-ish for log_mean
        surface = rng.integers(0, 5, size=(n_lat, n_lon)).astype(float)
        # Sprinkle NaNs / a non-positive log value so the usable + positive masks bite.
        field_a[0, 0] = np.nan
        field_b[1, 2] = -1.0
        field_b[2, 3] = np.nan
        data = np.stack([field_a, field_b, surface])

        lats = np.linspace(0.2, 0.8, n_lat)
        lons = np.linspace(0.2, 0.9, n_lon)
        weights = rng.random(size=(n_lat, n_lon))
        weights[weights < 0.15] = 0.0  # zero-weight cells (outside the contour)
        weights[4, 5] = np.nan
        tile = GridTile(data=data, lats=lats, lons=lons, bounds=BoundingBox(0.0, 1.0, 0.0, 1.0), source="_parity")
        return tile, WeightField(weights=weights, total_energy=float(np.nansum(weights)), max_radius_km=100.0)

    def test_matches_per_variable_reference(self):
        rng = np.random.default_rng(20260826)
        for _ in range(20):
            tile, weight_field = self._tile_and_weights(rng)
            got = agg.aggregate_tile_variables(_ParityReader, tile, weight_field)
            expected = _reference_tile_variables(_ParityReader, tile, weight_field)
            assert got.keys() == expected.keys()
            for name in expected:
                g, e = got[name], expected[name]
                if math.isnan(e):
                    assert math.isnan(g), name
                else:
                    # Same operations in the same order -> bit-for-bit identical.
                    assert g == e, name

    def test_empty_tile_yields_nan_everywhere(self):
        # A failed/missing region arrives as a zero-cell tile -> NaN for every variable.
        data = np.empty((3, 0, 0), dtype=np.float64)
        tile = GridTile(
            data=data,
            lats=np.empty(0),
            lons=np.empty(0),
            bounds=BoundingBox(0.0, 1.0, 0.0, 1.0),
            source="_parity",
        )
        weight_field = WeightField(weights=np.empty((0, 0)), total_energy=0.0, max_radius_km=100.0)
        got = agg.aggregate_tile_variables(_ParityReader, tile, weight_field)
        assert all(math.isnan(v) for v in got.values())
