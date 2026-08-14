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
