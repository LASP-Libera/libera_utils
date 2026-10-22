"""Unit tests for the PSF aggregation engine (aggregation.py).

These exercise the pure strategy functions against hand-computed values, the
NaN / zero-weight exclusion rules, the categorical mode / ranked-mode logic, the
coverage bookkeeping, and the dispatcher's error handling. They do not query a reader
or read data -- that is covered by the integration test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from libera_utils.footprint_matching import aggregation as agg
from libera_utils.footprint_matching.types import (
    BoundingBox,
    OperationalMode,
    SampleSet,
    VariableSpec,
    with_standard_deviation_companions,
)
from libera_utils.footprint_matching.weighting import WeightField


class TestWeightedMean:
    """weighted_mean = sum(w*x)/sum(w), NaN-excluded."""

    def test_matches_hand_computed_value(self):
        """``weighted_mean`` returns ``sum(w * x) / sum(w)``, the CERES PSF-weighted mean.

        The weighted mean is the statistic most product variables (temperatures, albedos, AOD companions) are reduced
        with, so a wrong formula would bias every footprint value the product stores.

        Feeds two values with unequal weights, where the plain mean (2.0) and the weighted mean (2.5) differ, and
        asserts the result equals the hand-computed 2.5.
        """
        values = np.array([1.0, 3.0])
        weights = np.array([1.0, 3.0])
        # (1*1 + 3*3) / (1 + 3) = 10/4 = 2.5
        assert agg.weighted_mean(values, weights)["mean"] == pytest.approx(2.5)

    def test_nan_values_are_excluded(self):
        """``weighted_mean`` drops NaN samples from both the numerator and the denominator.

        Readers mark missing or fill data as NaN. If a NaN sample's weight still counted in the denominator, a
        footprint over a data gap would report a mean pulled toward zero rather than the mean of the data present.

        Puts the largest weight on a NaN sample between two finite values and asserts the result is 3.0, the mean
        of the two finite values alone.
        """
        values = np.array([2.0, np.nan, 4.0])
        weights = np.array([1.0, 5.0, 1.0])
        # NaN cell dropped -> (2 + 4)/2 = 3.0
        assert agg.weighted_mean(values, weights)["mean"] == pytest.approx(3.0)

    def test_zero_and_negative_weights_dropped(self):
        """``weighted_mean`` ignores a sample whose PSF weight is zero.

        The weighers give zero weight to samples outside the PSF's 95%-energy contour. Those samples lie in the
        footprint's bounding box but not in what the radiometer saw, so their value must not reach the mean however
        extreme it is.

        Gives an outlier value of 100 a zero weight between two unit-weight values and asserts the mean is 3.0, the
        mean of the other two. Only the zero-weight case is exercised; a negative weight is not.
        """
        values = np.array([2.0, 100.0, 4.0])
        weights = np.array([1.0, 0.0, 1.0])  # zero-weight cell outside contour
        assert agg.weighted_mean(values, weights)["mean"] == pytest.approx(3.0)

    def test_no_usable_data_returns_nan(self):
        """``weighted_mean`` returns NaN when no sample is usable.

        NaN is the product's signal that a source did not cover a footprint. Returning 0.0, or raising, would either
        publish a fake measurement or abort the whole granule over one uncovered footprint.

        Passes a single NaN value and asserts the mean is NaN.
        """
        assert math.isnan(agg.weighted_mean(np.array([np.nan]), np.array([1.0]))["mean"])


class TestWeightedStd:
    """weighted_std = sqrt(sum(w*x^2)/sum(w) - mean^2)."""

    def test_uniform_weight_matches_population_std(self):
        """``weighted_std`` with equal weights equals the population standard deviation.

        Each continuous product variable has a ``_standard_deviation`` companion describing the spread of the source
        within the footprint. Checking the equal-weight case pins the formula to the familiar population (divide by
        N) definition rather than the sample (divide by N - 1) one.

        Passes ``[1, 2, 3]`` with unit weights and asserts the result equals ``sqrt(2/3)``.
        """
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 1.0])
        # population std of [1,2,3] = sqrt(2/3)
        assert agg.weighted_std(values, weights)["standard_deviation"] == pytest.approx(math.sqrt(2.0 / 3.0))

    def test_constant_values_give_zero_not_negative(self):
        """``weighted_std`` of a constant field is 0.0, not NaN.

        The formula subtracts two nearly equal numbers, so floating-point round-off can leave a tiny negative variance
        whose square root is NaN. A uniform scene (open ocean, an ice sheet) would then report a missing spread instead
        of zero spread.

        Passes three identical values with uneven weights, chosen so round-off can occur, and asserts the result is
        0.0 to within 1e-12.
        """
        values = np.array([5.0, 5.0, 5.0])
        weights = np.array([0.3, 0.6, 0.1])
        assert agg.weighted_std(values, weights)["standard_deviation"] == pytest.approx(0.0, abs=1e-12)


class TestWeightedLogMean:
    """weighted_log_mean = exp(sum(w*ln x)/sum(w)); values <= 0 excluded."""

    def test_geometric_mean_of_positive_values(self):
        """``weighted_log_mean`` returns the weighted geometric mean ``exp(sum(w * ln x) / sum(w))``.

        Quantities that span orders of magnitude, such as aerosol optical depth, are averaged geometrically, following
        the CERES convention, so that one large value does not dominate the footprint.

        Passes 1 and 100 with equal weights and asserts the result is 10, their geometric mean. The arithmetic mean
        would be 50.5.
        """
        values = np.array([1.0, 100.0])
        weights = np.array([1.0, 1.0])
        # geometric mean sqrt(1*100) = 10
        assert agg.weighted_log_mean(values, weights)["log_mean"] == pytest.approx(10.0)

    def test_nonpositive_values_excluded(self):
        """``weighted_log_mean`` drops zero and negative values, whose logarithm is undefined.

        Without the exclusion, one zero or negative sample would turn the whole footprint's log mean into NaN or
        ``-inf``.

        Mixes -1 and 0 with 4 and 9 at equal weight and asserts the result is 6, the geometric mean of 4 and 9 alone.
        """
        values = np.array([-1.0, 0.0, 4.0, 9.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        # only 4 and 9 survive -> sqrt(36) = 6
        assert agg.weighted_log_mean(values, weights)["log_mean"] == pytest.approx(6.0)

    def test_all_nonpositive_returns_nan(self):
        """``weighted_log_mean`` returns NaN when every value is zero or negative.

        With nothing left to average, the footprint has no valid value, and NaN is the product's signal for that.

        Passes only -1 and 0 and asserts the result is NaN.
        """
        assert math.isnan(agg.weighted_log_mean(np.array([-1.0, 0.0]), np.array([1.0, 1.0]))["log_mean"])


class TestWeightedMedian:
    def test_weighted_median_picks_halfway_value(self):
        """``weighted_median`` returns the first sorted value at which cumulative weight reaches half the total.

        A weighted median is a central value that resists outliers. It must follow the PSF weights, not the sample
        count, so a heavily weighted sample can be the median even when it is not the middle value.

        Gives the largest of three sorted values most of the weight (cumulative weights 1, 2, 7; half of 7 is 3.5)
        and asserts the median is that value, 3, not the unweighted middle value, 2.
        """
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, 1.0, 5.0])
        # cumulative weight 1,2,7; half = 3.5 -> reached at value 3
        assert agg.weighted_median(values, weights)["median"] == pytest.approx(3.0)


class TestWeightedMode:
    """Categorical mode + per-class coverage fractions."""

    def test_mode_is_highest_weight_category(self):
        """``weighted_mode`` picks the category carrying the most PSF weight and reports each category's share.

        Categorical variables such as surface type cannot be averaged. The footprint takes the class that covers most
        of what the instrument saw, which is the class with the most weight and not necessarily the most samples. The
        per-class fractions record the scene mix.

        Gives class 0 one sample of weight 5 and class 1 two samples of weight 1. Asserts the mode is 0, although
        class 1 has more samples, and that the coverage fractions are 5/7 and 2/7.
        """
        values = np.array([0.0, 1.0, 1.0])
        weights = np.array([5.0, 1.0, 1.0])  # class 0 has more weight
        result = agg.weighted_mode(values, weights, n_categories=2)
        assert result["mode"] == pytest.approx(0.0)
        assert result["coverage_fractions"][0] == pytest.approx(5.0 / 7.0)
        assert result["coverage_fractions"][1] == pytest.approx(2.0 / 7.0)

    def test_infers_categories_when_count_unknown(self):
        """``weighted_mode`` with ``n_categories=None`` takes the categories from the distinct values present.

        Some categorical sources, such as SSF scene-type codes, declare no fixed number of classes. The mode must still
        work on them without a preset domain.

        Passes codes 3, 3 and 7 with equal weights and no category count, and asserts the mode is 3.
        """
        # n_categories=None -> categories inferred from the data (SSF scene codes).
        values = np.array([3.0, 3.0, 7.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert agg.weighted_mode(values, weights, n_categories=None)["mode"] == pytest.approx(3.0)

    def test_tie_breaks_to_lower_category(self):
        """``weighted_mode`` breaks a weight tie in favor of the lower category code.

        A deterministic tie-break means the same input always yields the same product value, which keeps reprocessing
        reproducible.

        Gives codes 2 and 5 equal weight and asserts the mode is 2.
        """
        values = np.array([2.0, 5.0])
        weights = np.array([1.0, 1.0])  # tie
        assert agg.weighted_mode(values, weights, n_categories=None)["mode"] == pytest.approx(2.0)

    def test_no_data_returns_nan_mode(self):
        """``weighted_mode`` returns a NaN mode when no sample is usable.

        As for the continuous strategies, NaN marks a footprint the source did not cover, rather than a made-up
        class 0.

        Passes a single NaN value and asserts the mode is NaN.
        """
        assert math.isnan(agg.weighted_mode(np.array([np.nan]), np.array([1.0]), n_categories=3)["mode"])

    def test_top_of_range_code_not_dropped_for_one_based_scheme(self):
        """``weighted_mode`` keeps a code equal to ``n_categories`` when a scheme numbers its classes from 1.

        CLDPIX cloud phase uses codes 1 to 5 and declares ``n_categories=5``. A histogram sized to 0..4 would discard
        every phase-5 sample, so a footprint dominated by that phase would report another phase.

        Gives code 5 most of the weight (4 of 5) with ``n_categories=5``. Asserts the mode is 5 and that the coverage
        fraction for 5 is 4/5.
        """
        values = np.array([5.0, 5.0, 1.0])
        weights = np.array([2.0, 2.0, 1.0])  # code 5 has the most weight
        result = agg.weighted_mode(values, weights, n_categories=5)
        assert result["mode"] == pytest.approx(5.0)
        assert result["coverage_fractions"][5] == pytest.approx(4.0 / 5.0)


class TestRankedMode:
    """primary/secondary/tertiary = 1st/2nd/3rd most common class by weight."""

    def test_ranked_ordering(self):
        """``weighted_mode_primary`` / ``_secondary`` / ``_tertiary`` return the 1st, 2nd and 3rd classes by weight.

        Readers such as IGBP and AOD publish the three dominant classes of a mixed footprint, for example forest,
        cropland and water. A wrong rank order mislabels the scene mix.

        Builds three classes with weights 4, 3 and 1 (codes 5, 2 and 1, deliberately not in code order) and asserts
        each ranked function returns 5, 2 and 1 respectively.
        """
        # class 5 weight 4, class 2 weight 3, class 1 weight 1
        values = np.array([5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 1.0])
        weights = np.ones_like(values)
        assert agg.weighted_mode_primary(values, weights, n_categories=6)["mode"] == pytest.approx(5.0)
        assert agg.weighted_mode_secondary(values, weights, n_categories=6)["mode"] == pytest.approx(2.0)
        assert agg.weighted_mode_tertiary(values, weights, n_categories=6)["mode"] == pytest.approx(1.0)

    def test_missing_rank_is_nan(self):
        """A ranked mode with no class at that rank returns NaN.

        A footprint over a single class has no second or third class. Reporting a class with zero weight there, such
        as code 0, would invent a land cover that is not present.

        Passes one class only and asserts the primary mode is that class while the secondary and tertiary modes are
        NaN.
        """
        values = np.array([4.0, 4.0])
        weights = np.array([1.0, 1.0])
        assert agg.weighted_mode_primary(values, weights, n_categories=6)["mode"] == pytest.approx(4.0)
        assert math.isnan(agg.weighted_mode_secondary(values, weights, n_categories=6)["mode"])
        assert math.isnan(agg.weighted_mode_tertiary(values, weights, n_categories=6)["mode"])


class TestCloudRadianceStrategies:
    """The cloud/radiance-path strategies (provided for future readers)."""

    def test_coverage_fraction(self):
        """``coverage_fraction`` returns the share of PSF weight on samples whose value is above zero.

        This is how a per-pixel presence flag, such as cloudy or snow-covered, becomes a footprint cloud fraction or
        snow fraction. No current reader selects it, so this test is its only check.

        Puts weight 2 on a zero sample and weight 1 on each of two positive samples, and asserts the fraction is 0.5
        (weight 2 of 4).
        """
        values = np.array([0.0, 1.0, 1.0])
        weights = np.array([2.0, 1.0, 1.0])
        # weight where x>0 is 2; total 4 -> 0.5
        assert agg.coverage_fraction(values, weights)["fraction"] == pytest.approx(0.5)

    def test_cloud_weighted_mean_zeros_clear_cells(self):
        """``cloud_weighted_mean`` averages only the samples the cloud mask marks cloudy.

        Cloud properties such as optical depth or cloud-top height exist only where there is cloud. Averaging in
        clear-sky samples would dilute the in-cloud value.

        Masks the middle of three equal-weight samples as clear. Asserts the mean is 20, the mean of the two cloudy
        values 10 and 30, and that ``n_cloudy`` is 2.
        """
        values = np.array([10.0, 20.0, 30.0])
        weights = np.array([1.0, 1.0, 1.0])
        cloud_mask = np.array([True, False, True])
        result = agg.cloud_weighted_mean(values, weights, cloud_mask=cloud_mask)
        assert result["mean"] == pytest.approx(20.0)  # (10 + 30)/2
        assert result["n_cloudy"] == pytest.approx(2.0)

    def test_overcast_fraction(self):
        """``overcast_fraction`` returns the share of cloudy PSF weight whose cloud fraction exceeds 0.95.

        Following CERES, overcast fraction is measured against the cloudy part of the footprint, not the whole of it,
        so that clear sky does not lower the overcast share of the cloud that is present.

        Passes cloud fractions 0, 0.5 and 0.99 at equal weight. Two samples are cloudy and one of those is overcast,
        so the test asserts 0.5. Over the whole footprint the share would have been 1/3.
        """
        values = np.array([0.0, 0.5, 0.99])
        weights = np.array([1.0, 1.0, 1.0])
        # cloudy weight (x>0) = 2; overcast (x>0.95) = 1 -> 0.5
        assert agg.overcast_fraction(values, weights)["overcast_fraction"] == pytest.approx(0.5)

    def test_broadband_radiance_total(self):
        """``aggregate_broadband_radiance`` reports the PSF-weighted mean radiance as ``total_radiance``.

        The strategy is a stand-in until the cloud-layer split is implemented. This pins the part that exists, the
        total radiance, to the weighted mean.

        Passes radiances 1 and 3 at equal weight and asserts ``total_radiance`` is 2. The clear and cloudy split is
        not checked.
        """
        values = np.array([1.0, 3.0])
        weights = np.array([1.0, 1.0])
        assert agg.aggregate_broadband_radiance(values, weights)["total_radiance"] == pytest.approx(2.0)

    def test_report_all_returns_usable_arrays(self):
        """``report_all`` returns the usable values and weights unaggregated, minus NaN-valued and zero-weight samples.

        Some downstream algorithms aggregate the samples themselves. They need the same exclusions the other
        strategies apply, so that they never see fill values or samples from outside the PSF contour.

        Passes one good sample, one NaN-valued sample and one zero-weight sample, and asserts that only the good
        sample's value and weight come back.
        """
        values = np.array([1.0, np.nan, 3.0])
        weights = np.array([1.0, 1.0, 0.0])
        result = agg.report_all(values, weights)
        np.testing.assert_array_equal(result["values"], np.array([1.0]))
        np.testing.assert_array_equal(result["weights"], np.array([1.0]))


class TestCheckCoverage:
    def test_bands(self):
        """``check_coverage`` applies the CERES rule: accept at 75% coverage or more, and flag as partial below 95%.

        These thresholds decide whether a footprint is kept, kept with a partial-coverage flag, or discarded. Putting
        a footprint in the wrong band either publishes values backed by too little data or throws good ones away.

        Asserts the ``(accept, partial)`` pair for one coverage in each band: 0.96 is accepted in full, 0.80 is
        accepted as partial and 0.50 is rejected. Also asserts that a footprint with zero total PSF energy is rejected
        rather than dividing by zero. The exact boundaries of 0.75 and 0.95 are not tested.
        """
        assert agg.check_coverage(0.96, 1.0) == (True, False)  # full
        assert agg.check_coverage(0.80, 1.0) == (True, True)  # partial
        assert agg.check_coverage(0.50, 1.0) == (False, False)  # discard
        assert agg.check_coverage(1.0, 0.0) == (False, False)  # no PSF energy


class TestDispatcher:
    def test_aggregate_wraps_result_with_coverage(self):
        """``aggregate`` runs the named strategy and wraps its result with coverage bookkeeping.

        ``aggregate`` is the general entry point for running a strategy by name. Its ``AggregationResult`` must carry
        both the statistic and how much of the PSF energy the data backed, so that a caller can judge the value.

        Runs ``"weighted_mean"`` on two fully covered samples with ``total_energy`` equal to their weight sum. Asserts
        the mean is 3, the coverage is 1.0, both samples count as valid out of 2 in total, and the partial flag is
        off.
        """
        values = np.array([2.0, 4.0])
        weights = np.array([1.0, 1.0])
        result = agg.aggregate("weighted_mean", values, weights, total_energy=2.0)
        assert result.values["mean"] == pytest.approx(3.0)
        assert result.coverage == pytest.approx(1.0)
        assert result.n_valid == 2
        assert result.n_total == 2
        assert result.partial_flag is False

    def test_coverage_drops_with_uncovered_cells(self):
        """``aggregate`` coverage is the share of ``total_energy`` on usable samples; 75-95% sets the partial flag.

        A sample with no data inside the PSF contour means part of what the instrument saw is not represented. The
        coverage must fall by that sample's weight, so that the footprint is flagged or discarded under the CERES
        rule.

        Gives four samples weight 1 each with one NaN value and ``total_energy=4``. Asserts the coverage is 0.75 and
        that the result is flagged partial.
        """
        values = np.array([1.0, 1.0, 1.0, np.nan])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        result = agg.aggregate("weighted_mean", values, weights, total_energy=4.0)
        assert result.coverage == pytest.approx(0.75)
        assert result.partial_flag is True

    def test_unknown_strategy_raises_value_error(self):
        """``aggregate`` raises ``ValueError`` for an unregistered strategy name.

        A strategy name comes from a reader's variable specification. A typo there must stop the run with a clear
        error rather than silently fall back to another statistic.

        Calls ``aggregate`` with ``"not_a_strategy"`` and asserts a ``ValueError`` matching "Unknown aggregation
        strategy".
        """
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            agg.aggregate("not_a_strategy", np.array([1.0]), np.array([1.0]))


class _ParityReader:
    """Reader stand-in exercising every shared-plane case aggregate_sample_variables optimizes.

    ``field_a``/``field_b`` are continuous (each gains a ``_standard_deviation`` companion
    sharing its plane), and ``surface`` is categorical with three ranked-mode extras that
    all derive from the one ``surface`` plane. Only the attributes
    ``aggregate_sample_variables`` / ``_map_product_specs`` touch are defined.
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


def _reference_sample_variables(reader_cls, samples: SampleSet, weight_field: WeightField) -> dict[str, float]:
    """Per-variable reference: the public dispatcher called once per product spec.

    No per-plane sharing, so any divergence in the optimized path shows up as a value
    mismatch here.
    """
    planes = [samples.values[i] for i in range(samples.values.shape[0])]
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


class TestAggregateSampleVariablesParity:
    """The shared-plane aggregate_sample_variables must match the per-variable reference exactly."""

    def _samples_and_weights(self, rng: np.random.Generator) -> tuple[SampleSet, WeightField]:
        n = 30
        field_a = rng.normal(10.0, 3.0, size=n)
        field_b = np.abs(rng.normal(4.0, 2.0, size=n)) + 0.01  # positive-ish for log_mean
        surface = rng.integers(0, 5, size=n).astype(float)
        # Sprinkle NaNs / a non-positive log value so the usable + positive masks bite.
        field_a[0] = np.nan
        field_b[8] = -1.0
        field_b[15] = np.nan
        values = np.stack([field_a, field_b, surface]).astype(np.float32)

        weights = rng.random(size=n)
        weights[weights < 0.15] = 0.0  # zero-weight samples (outside the contour)
        weights[29] = np.nan
        samples = SampleSet(
            lats=rng.uniform(0.2, 0.8, n),
            lons=rng.uniform(0.2, 0.9, n),
            values=values,
            times=None,
            source="_parity",
            bounds=BoundingBox(0.0, 1.0, 0.0, 1.0),
        )
        field = WeightField(weights=weights, total_energy=float(np.nansum(weights)), max_radius_km=100.0, coverage=1.0)
        return samples, field

    def test_matches_per_variable_reference(self):
        """``aggregate_sample_variables`` gives exactly the values of aggregating each product variable on its own.

        For speed, the production path computes the usable-sample mask and the category histogram once per data plane
        and shares them across every product variable derived from that plane: a variable, its
        ``_standard_deviation`` companion, and its ranked modes. That shortcut must not change any number the product
        stores.

        Over 20 random footprints that include NaN values, a non-positive log-mean input, zero weights and a NaN
        weight, compares the shared path on ``_ParityReader`` with a reference that calls the public ``aggregate`` once
        per product spec. Asserts both return the same variable names and identical values: bit-for-bit equal, or NaN
        in both.
        """
        rng = np.random.default_rng(20260826)
        for _ in range(20):
            samples, weight_field = self._samples_and_weights(rng)
            got = agg.aggregate_sample_variables(_ParityReader, samples, weight_field)
            expected = _reference_sample_variables(_ParityReader, samples, weight_field)
            assert got.keys() == expected.keys()
            for name in expected:
                g, e = got[name], expected[name]
                if math.isnan(e):
                    assert math.isnan(g), name
                else:
                    # Same operations in the same order -> bit-for-bit identical.
                    assert g == e, name

    def test_empty_samples_yield_nan_everywhere(self):
        """``aggregate_sample_variables`` returns NaN for every product variable when the sample set is empty.

        A footprint outside a source's coverage, such as a regional dataset or a swath gap, arrives as a zero-sample
        set. Every one of that source's product variables must then read NaN, meaning "not covered", rather than
        crash or report a default value.

        Aggregates an empty ``SampleSet`` with an empty weight field for ``_ParityReader`` and asserts every returned
        value is NaN.
        """
        samples = SampleSet.empty("_parity", BoundingBox(0.0, 1.0, 0.0, 1.0), 3)
        weight_field = WeightField(weights=np.empty(0), total_energy=0.0, max_radius_km=100.0, coverage=0.0)
        got = agg.aggregate_sample_variables(_ParityReader, samples, weight_field)
        assert all(math.isnan(v) for v in got.values())
