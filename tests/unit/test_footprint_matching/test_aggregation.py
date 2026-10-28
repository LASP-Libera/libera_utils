"""Unit tests for the PSF aggregation engine (aggregation.py).

These exercise the pure strategy functions against hand-computed values, the
NaN / zero-weight exclusion rules, the categorical mode / ranked-mode logic, the
coverage rule, and the dispatcher's error handling. They do not query a reader
or read data -- that is covered by the integration test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from libera_utils.footprint_matching import aggregation as agg
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
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

    def test_positive_std_uses_the_log_means_samples(self):
        """``weighted_positive_std`` is the spread of the same positive samples ``weighted_log_mean`` averages.

        A zero is in range for an optical depth but has no logarithm, so the log mean drops it. If its
        ``_standard_deviation`` companion kept the zero, the mean and its spread would describe different samples.

        Passes 0, 4 and 9 at equal weight. Asserts the log mean is 6 (4 and 9 only) and the positive std is 2.5, the
        spread of 4 and 9, where ``weighted_std`` over all three would be about 3.68.
        """
        values = np.array([0.0, 4.0, 9.0])
        weights = np.array([1.0, 1.0, 1.0])
        assert agg.weighted_log_mean(values, weights)["log_mean"] == pytest.approx(6.0)
        assert agg.weighted_positive_std(values, weights)["standard_deviation"] == pytest.approx(2.5)


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

        Some categorical sources, such as FLASHFlux scene-type codes, declare no fixed number of classes. The mode must still
        work on them without a preset domain.

        Passes codes 3, 3 and 7 with equal weights and no category count, and asserts the mode is 3.
        """
        # n_categories=None -> categories inferred from the data (FLASHFlux scene codes).
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


class TestCheckCoverage:
    def test_bands(self):
        """``check_coverage`` applies the CERES rule: accept at 75% coverage or more, and flag as partial below 95%.

        These thresholds decide whether a footprint is kept, kept with a partial-coverage flag, or discarded. Putting
        a footprint in the wrong band either publishes values backed by too little data or throws good ones away.

        Asserts the ``(accept, partial)`` pair for one coverage in each band: 0.96 is accepted in full, 0.80 is
        accepted as partial, and 0.50 and 0.0 (no usable data) are rejected. The exact boundaries of 0.75 and 0.95 are
        not tested.
        """
        assert agg.check_coverage(0.96) == (True, False)  # full
        assert agg.check_coverage(0.80) == (True, True)  # partial
        assert agg.check_coverage(0.50) == (False, False)  # discard
        assert agg.check_coverage(0.0) == (False, False)  # no usable data


class TestDispatcher:
    def test_aggregate_wraps_result_with_sample_counts(self):
        """``aggregate`` runs the named strategy and wraps its result with the samples it used.

        ``aggregate`` is the general entry point for running a strategy by name. Its ``AggregationResult`` carries the
        statistic and how many samples backed it.

        Runs ``"weighted_mean"`` on two samples and asserts the mean is 3 and both samples count as valid out of 2.
        """
        values = np.array([2.0, 4.0])
        weights = np.array([1.0, 1.0])
        result = agg.aggregate("weighted_mean", values, weights)
        assert result.values["mean"] == pytest.approx(3.0)
        assert result.n_valid == 2
        assert result.n_total == 2

    def test_nan_samples_do_not_count_as_valid(self):
        """``aggregate`` counts a NaN sample in ``n_total`` but not in ``n_valid``.

        A NaN is the reader's "no value here", so it must not be reported as a sample that contributed.

        Gives four samples weight 1 each with one NaN value and asserts ``n_valid`` is 3 of ``n_total`` 4.
        """
        values = np.array([1.0, 1.0, 1.0, np.nan])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        result = agg.aggregate("weighted_mean", values, weights)
        assert result.n_valid == 3
        assert result.n_total == 4

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
        VariableSpec(
            name="field_a", dtype="float32", aggregation="weighted_mean", modes=frozenset({OperationalMode.CAM})
        ),
        VariableSpec(
            name="field_b", dtype="float32", aggregation="weighted_log_mean", modes=frozenset({OperationalMode.CAM})
        ),
        VariableSpec(
            name="surface",
            dtype="int16",
            aggregation="weighted_mode",
            modes=frozenset({OperationalMode.CAM}),
            n_categories=5,
        ),
    )
    ADDITIONAL_PRODUCT_VARIABLES = (
        VariableSpec(
            name="surface_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            modes=frozenset({OperationalMode.CAM}),
            n_categories=5,
        ),
        VariableSpec(
            name="surface_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            modes=frozenset({OperationalMode.CAM}),
            n_categories=5,
        ),
        VariableSpec(
            name="surface_tertiary",
            dtype="int16",
            aggregation="weighted_mode_tertiary",
            modes=frozenset({OperationalMode.CAM}),
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
    weights = np.asarray(weight_field.weights, dtype=float).ravel()
    out: dict[str, float] = {}
    for spec, read_index in agg._map_product_specs(reader_cls):
        n_categories = spec.n_categories
        if n_categories is None:
            n_categories = reader_cls.VARIABLES[read_index].n_categories
        result = agg.aggregate(spec.aggregation, samples.values[read_index], weights, n_categories=n_categories)
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


def _sample_set(values: np.ndarray) -> SampleSet:
    """A SampleSet carrying ``values`` (``(n_var, n)``) at arbitrary in-box positions."""
    n = values.shape[1]
    return SampleSet(
        lats=np.linspace(0.2, 0.8, n),
        lons=np.linspace(0.2, 0.8, n),
        values=values,
        times=None,
        source="_test",
        bounds=BoundingBox(0.0, 1.0, 0.0, 1.0),
    )


def _weight_field(weights: np.ndarray) -> WeightField:
    return WeightField(weights=weights, total_energy=float(weights.sum()), max_radius_km=100.0, coverage=1.0)


class TestAggregateSampleVariablesValues:
    def test_hand_computed_footprint(self):
        """``aggregate_sample_variables`` puts the hand-computed value of each product variable under its name.

        The parity test checks the shared path against a reference that uses the same spec-to-row mapping, so a
        variable read from the wrong row would pass it. Known numbers pin each variable to its own row.

        Four samples with weights 1, 1, 1 and 2:

        - ``field_a`` = 1, 3, NaN, 5: mean (1 + 3 + 10) / 4 = 3.5, std sqrt(15 - 3.5^2) = sqrt(2.75).
        - ``field_b`` = 0, 4, 9, 4: log mean over 4, 9, 4 is 4^0.75 * 9^0.25 = sqrt(24); its std over the same
          samples is sqrt(32.25 - 5.25^2) = sqrt(4.6875).
        - ``surface`` = 1, 2, 2, 1: class 1 weighs 3 and class 2 weighs 2, so the mode and primary are 1, the secondary
          is 2 and the tertiary is NaN.
        """
        values = np.array(
            [
                [1.0, 3.0, np.nan, 5.0],
                [0.0, 4.0, 9.0, 4.0],
                [1.0, 2.0, 2.0, 1.0],
            ]
        )
        got = agg.aggregate_sample_variables(
            _ParityReader, _sample_set(values), _weight_field(np.array([1.0, 1.0, 1.0, 2.0]))
        )
        assert got["field_a"] == pytest.approx(3.5)
        assert got["field_a_standard_deviation"] == pytest.approx(math.sqrt(2.75))
        assert got["field_b"] == pytest.approx(math.sqrt(24.0))
        assert got["field_b_standard_deviation"] == pytest.approx(math.sqrt(4.6875))
        assert got["surface"] == 1.0
        assert got["surface_primary"] == 1.0
        assert got["surface_secondary"] == 2.0
        assert math.isnan(got["surface_tertiary"])


class _UnmappableReader:
    """Reader stand-in whose derived product variable is not named after any read variable."""

    __name__ = "_UnmappableReader"
    VARIABLES = (
        VariableSpec(
            name="field_a", dtype="float32", aggregation="weighted_mean", modes=frozenset({OperationalMode.CAM})
        ),
    )

    @classmethod
    def product_variable_specs(cls) -> tuple[VariableSpec, ...]:
        return cls.VARIABLES + (
            VariableSpec(
                name="other_thing", dtype="float32", aggregation="weighted_mean", modes=frozenset({OperationalMode.CAM})
            ),
        )


class _UnregisteredStrategyReader:
    """Reader stand-in whose product variable names an unregistered aggregation strategy."""

    __name__ = "_UnregisteredStrategyReader"
    VARIABLES = (
        VariableSpec(
            name="field_a", dtype="float32", aggregation="not_a_strategy", modes=frozenset({OperationalMode.CAM})
        ),
    )

    @classmethod
    def product_variable_specs(cls) -> tuple[VariableSpec, ...]:
        return cls.VARIABLES


class TestAggregateSampleVariablesErrors:
    def test_unmappable_product_variable_raises(self):
        """``aggregate_sample_variables`` raises ``ValueError`` for a product variable it cannot map to a read row.

        A derived product variable is read from the row of the read variable its name starts with. One named after
        none of them would otherwise have no row to read, and the run must stop rather than store a wrong value.

        Uses a reader whose extra product variable is ``other_thing`` and asserts a ``ValueError`` matching "cannot be
        mapped".
        """
        values = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="cannot be mapped"):
            agg.aggregate_sample_variables(_UnmappableReader, _sample_set(values), _weight_field(np.ones(2)))

    def test_unregistered_aggregation_raises(self):
        """``aggregate_sample_variables`` raises ``ValueError`` naming the variable when its strategy is unregistered.

        A product variable labelled with a misspelled strategy has no statistic to compute. The run must stop with an
        error naming the reader and the variable so the spec can be fixed.

        Uses a reader whose only variable is ``not_a_strategy`` and asserts a ``ValueError`` naming ``field_a``.
        """
        values = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="'field_a'.*'not_a_strategy'"):
            agg.aggregate_sample_variables(_UnregisteredStrategyReader, _sample_set(values), _weight_field(np.ones(2)))

    def test_missing_value_row_raises(self):
        """``aggregate_sample_variables`` raises ``ValueError`` when the samples carry fewer rows than the reader's
        variables.

        A missing row would otherwise read as NaN, the "not covered" signal, or fail inside numpy with a message that
        names neither the reader nor the cause.

        Passes two rows for ``_ParityReader``'s three variables and asserts a ``ValueError`` matching "expected".
        """
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="_ParityReader.*expected"):
            agg.aggregate_sample_variables(_ParityReader, _sample_set(values), _weight_field(np.ones(2)))

    def test_weight_length_mismatch_raises(self):
        """``aggregate_sample_variables`` raises ``ValueError`` when the weights do not match the samples one to one.

        Weights computed for a different sample set would pair each value with another sample's weight.

        Passes three samples with two weights and asserts a ``ValueError`` matching "2 weights for 3 samples".
        """
        values = np.ones((3, 3))
        with pytest.raises(ValueError, match="2 weights for 3 samples"):
            agg.aggregate_sample_variables(_ParityReader, _sample_set(values), _weight_field(np.ones(2)))


class TestStackProductValues:
    """stack_product_values shapes a reader's per-spec scalars into its product variables."""

    @staticmethod
    def _scalars() -> dict[str, float]:
        # A distinct value per spec, so a member stacked out of order or under the wrong stack shows up.
        specs = ERA5PressureLevelReader.product_variable_specs()
        return {spec.name: float(i) for i, spec in enumerate(specs)}

    def test_each_stack_is_its_members_in_coordinate_order(self):
        """Every stacked product variable is the array of its members' scalars, in member (level) order.

        Element ``k`` is published at the stack's ``k``-th coordinate value; out of order, one pressure level's
        value would be published at another's.

        Stacks distinct scalars for every ERA5 pressure-level spec and compares each stack's array with its
        members' scalars taken in order.
        """
        scalars = self._scalars()
        stacks = ERA5PressureLevelReader.product_stacks()
        got = agg.stack_product_values(ERA5PressureLevelReader, scalars)
        assert set(got) == {stack.name for stack in stacks}
        for stack in stacks:
            np.testing.assert_array_equal(got[stack.name], [scalars[member] for member in stack.members])

    def test_unstacked_scalars_pass_through(self):
        """A scalar that is not a stack member is returned unchanged under its own name.

        A reader may contribute 1-D product variables alongside its stacks; dropping them would leave those
        variables empty.

        Adds an extra ``surface_pressure`` scalar to the ERA5 scalars and asserts it comes back as is.
        """
        scalars = self._scalars() | {"surface_pressure": 1013.0}
        assert agg.stack_product_values(ERA5PressureLevelReader, scalars)["surface_pressure"] == 1013.0

    def test_missing_member_raises(self):
        """A missing member scalar raises ``ValueError`` rather than publishing a stack with a level missing.

        Without the check, a stacked variable would be built one element short and misaligned with its coordinate.

        Drops ``temperature_500hPa`` from the scalars and asserts the error names it.
        """
        scalars = self._scalars()
        del scalars["temperature_500hPa"]
        with pytest.raises(ValueError, match="temperature_500hPa"):
            agg.stack_product_values(ERA5PressureLevelReader, scalars)
