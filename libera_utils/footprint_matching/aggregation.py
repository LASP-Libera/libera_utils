"""PSF aggregation engine: collapse a footprint's gridded pixels to one value.

Where this sits in the pipeline
-------------------------------
This is the last core-processing stage of footprint matching. Upstream, the
:class:`~libera_utils.footprint_matching.tiling.TileManager` has already served a
single merged :class:`~libera_utils.footprint_matching.types.GridTile` covering a
footprint's PSF bounding box, and the
:mod:`~libera_utils.footprint_matching.weighting` module has assigned each grid
cell a PSF weight. This module takes those (values, weights) and produces a single
statistic per variable -- the number that lands in the SSF-style output product.

Two layers
----------
1. **Pure strategy functions** (``weighted_mean``, ``weighted_mode``,
   ``weighted_log_mean`` ...). Each takes 1-D ``values`` and ``weights`` arrays and
   returns a small ``dict`` of named sub-results, exactly as tabulated in design
   doc section 2.8.1.4. These are reusable by scientists directly and are what the
   unit tests exercise. NaN values are excluded from every computation (doc: "NaN
   values are excluded from all computations").
2. **A product-facing projection** (:func:`aggregate_tile_variables`) that walks a
   reader's *product* variable specifications, picks the right data plane for each,
   dispatches to the correct strategy, and returns the single scalar the product
   definition stores for that variable. This is the bit
   :func:`libera_utils.footprint_matching.product.aggregate_external_variables`
   calls once per footprint per source.

CERES heritage
--------------
The weighted statistics implement CERES ATBD v2.2 Eq. 4.4-17
``xbar = sum(w*x) / sum(w)`` and its companions. The cloud-path strategies follow
CERES SSF conventions (geometric mean for optical depth, cloud-conditional means,
overcast fraction). See
https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.footprint_matching.types import STANDARD_DEVIATION_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Callable

    from libera_utils.footprint_matching.types import GridTile, VariableSpec
    from libera_utils.footprint_matching.weighting import WeightField

# CERES coverage thresholds (design doc section 2.4.2.7 / 2.8.1.1.3). The coverage
# metric is the fraction of the PSF's 95%-energy weight that had usable data:
#   >= 95%           -> fully accepted
#   75% .. < 95%     -> accepted but flagged as *partial* coverage
#   < 75%            -> insufficient data (the orchestrator discards the footprint)
# We only *compute and flag* here; the discard decision is the orchestrator's, so
# these are exposed as named constants rather than hard-coded at the call site.
ACCEPT_COVERAGE_THRESHOLD: float = 0.75
PARTIAL_COVERAGE_THRESHOLD: float = 0.95


@dataclass
class AggregationResult:
    """Output of a single aggregation, with coverage bookkeeping.

    Mirrors the design doc's ``AggregationResult`` (section 2.8.1.4.2). The
    ``values`` dict holds the strategy's named sub-results (e.g. ``{"mean": ...}``,
    ``{"mode": ..., "coverage_fractions": ...}``); the remaining fields describe how
    much of the PSF footprint actually had data so the footprint can be scored as
    full / partial / discard downstream.

    Attributes
    ----------
    values : dict[str, Any]
        Named results produced by the strategy function.
    coverage : float
        Fraction of the PSF's total in-contour weight backed by usable data, in
        ``[0, 1]``. ``0.0`` when there was no data or no PSF weight.
    n_valid : int
        Number of grid cells that contributed (finite value, positive weight).
    n_total : int
        Number of grid cells considered (the merged tile's cell count).
    partial_flag : bool
        ``True`` when ``0.75 <= coverage < 0.95`` (CERES partial-coverage band).
    """

    values: dict[str, Any]
    coverage: float
    n_valid: int
    n_total: int
    partial_flag: bool = field(default=False)


def check_coverage(
    sampled_weight: float,
    total_energy: float,
    *,
    accept_threshold: float = ACCEPT_COVERAGE_THRESHOLD,
    partial_threshold: float = PARTIAL_COVERAGE_THRESHOLD,
) -> tuple[bool, bool]:
    """Apply the CERES 75%/95% coverage rule.

    Parameters
    ----------
    sampled_weight : float
        Sum of PSF weights over cells that had usable data.
    total_energy : float
        Total PSF weight within the 95%-energy contour (the full-coverage
        denominator). ``<= 0`` means the footprint had no PSF weight at all.
    accept_threshold, partial_threshold : float, optional
        Lower bound for acceptance (default 0.75) and for full (non-partial)
        coverage (default 0.95).

    Returns
    -------
    tuple[bool, bool]
        ``(accept, partial)``. ``accept`` is ``False`` below ``accept_threshold``;
        ``partial`` is ``True`` in the ``[accept_threshold, partial_threshold)``
        band.
    """
    if total_energy <= 0.0:
        return False, False
    fraction = sampled_weight / total_energy
    accept = fraction >= accept_threshold
    partial = accept_threshold <= fraction < partial_threshold
    return accept, partial


# ---------------------------------------------------------------------------
# Shared masking helper
# ---------------------------------------------------------------------------


def _usable(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Boolean mask of grid cells usable for a weighted statistic.

    A cell is usable when its value is finite (NaN cells are excluded -- both the
    reader fill convention and uncovered merged-tile cells arrive as NaN) and its
    PSF weight is finite and strictly positive (a zero-weight cell lies outside the
    energy contour and must not contribute).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)


@dataclass(frozen=True)
class _WeightedPartials:
    """The usable ``(values, weights)`` of one data plane, masked once.

    The continuous surface strategies (:func:`weighted_mean`, :func:`weighted_std`,
    :func:`weighted_log_mean`, :func:`weighted_median`) each independently apply the
    :func:`_usable` mask before doing their arithmetic. When several product variables
    derive from the *same* data plane -- e.g. a variable and its
    ``<name>_standard_deviation`` companion -- that mask (and the array copy it drives)
    is identical across them. :func:`_prepare` computes it once and the ``*_from_partials``
    cores consume the result, so :func:`aggregate_tile_variables` pays the masking cost
    once per plane rather than once per variable. The public strategy functions are thin
    wrappers over the same cores, so the shared and the standalone paths are numerically
    identical by construction.

    Attributes
    ----------
    values, weights : np.ndarray
        1-D arrays of the plane's usable cell values and PSF weights (already masked
        by :func:`_usable`), in grid order.
    """

    values: np.ndarray
    weights: np.ndarray


def _prepare(values: np.ndarray, weights: np.ndarray) -> _WeightedPartials:
    """Mask ``(values, weights)`` to the usable cells once (see :class:`_WeightedPartials`)."""
    mask = _usable(values, weights)
    return _WeightedPartials(
        values=np.asarray(values, dtype=float)[mask],
        weights=np.asarray(weights, dtype=float)[mask],
    )


# ---------------------------------------------------------------------------
# Surface-path strategies (continuous)
# ---------------------------------------------------------------------------
#
# Each public strategy is a thin wrapper that masks the plane via _prepare and then
# calls its ``*_from_partials`` core. The core is what aggregate_tile_variables calls
# directly with a plane's shared partials, so the two paths cannot numerically diverge.


def _mean_from_partials(p: _WeightedPartials) -> dict[str, float]:
    """Weighted mean from pre-masked partials (see :func:`weighted_mean`)."""
    if p.weights.size == 0:
        return {"mean": float("nan")}
    return {"mean": float(np.sum(p.weights * p.values) / np.sum(p.weights))}


def _std_from_partials(p: _WeightedPartials) -> dict[str, float]:
    """Weighted standard deviation from pre-masked partials (see :func:`weighted_std`)."""
    if p.weights.size == 0:
        return {"standard_deviation": float("nan")}
    w_sum = np.sum(p.weights)
    mean = np.sum(p.weights * p.values) / w_sum
    variance = np.sum(p.weights * p.values * p.values) / w_sum - mean * mean
    return {"standard_deviation": float(np.sqrt(max(variance, 0.0)))}


def _log_mean_from_partials(p: _WeightedPartials) -> dict[str, float]:
    """Weighted geometric mean from pre-masked partials (see :func:`weighted_log_mean`)."""
    positive = p.values > 0.0
    if not np.any(positive):
        return {"log_mean": float("nan")}
    v = p.values[positive]
    w = p.weights[positive]
    return {"log_mean": float(np.exp(np.sum(w * np.log(v)) / np.sum(w)))}


def _median_from_partials(p: _WeightedPartials) -> dict[str, float]:
    """Weighted median from pre-masked partials (see :func:`weighted_median`)."""
    if p.weights.size == 0:
        return {"median": float("nan")}
    order = np.argsort(p.values)
    v_sorted = p.values[order]
    cumulative = np.cumsum(p.weights[order])
    half = 0.5 * cumulative[-1]
    # searchsorted finds the first cell whose cumulative weight reaches the halfway
    # point -- that value is the weighted median.
    idx = int(np.searchsorted(cumulative, half))
    idx = min(idx, v_sorted.size - 1)
    return {"median": float(v_sorted[idx])}


def weighted_mean(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, float]:
    """PSF-weighted mean ``xbar = sum(w*x)/sum(w)`` (CERES ATBD Eq. 4.4-17).

    Used for continuous surface variables (winds, BRDF kernels, cloud-top height...).
    Returns ``{"mean": NaN}`` when no cell is usable.
    """
    return _mean_from_partials(_prepare(values, weights))


def weighted_std(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, float]:
    """PSF-weighted standard deviation ``sqrt(sum(w*x^2)/sum(w) - xbar^2)``.

    This is the within-footprint spread that pairs with :func:`weighted_mean`; it is
    the value stored in every ``<name>_standard_deviation`` companion variable (see
    :data:`libera_utils.footprint_matching.types.STANDARD_DEVIATION_SUFFIX`). The
    variance is clipped at zero before the square root to absorb floating-point
    round-off that could otherwise make a near-zero variance slightly negative.
    """
    return _std_from_partials(_prepare(values, weights))


def weighted_log_mean(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, float]:
    """PSF-weighted geometric mean ``exp(sum(w*ln x)/sum(w))`` (CERES SSF convention).

    Used for optical depths, where the distribution is closer to log-normal so a
    geometric mean is the physically meaningful average. Only strictly positive
    values contribute -- ``ln`` is undefined at and below zero, and a zero/negative
    optical depth is not physical -- so those cells are dropped in addition to the
    usual NaN exclusion.
    """
    return _log_mean_from_partials(_prepare(values, weights))


def weighted_median(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, float]:
    """PSF-weighted median (the 50th percentile of the weight-sorted values).

    Sorts the usable values, walks the cumulative weight, and returns the value at
    which the cumulative weight first reaches half of the total. Robust to outliers,
    hence useful for skewed surface distributions.
    """
    return _median_from_partials(_prepare(values, weights))


# ---------------------------------------------------------------------------
# Surface-path strategies (categorical)
# ---------------------------------------------------------------------------


def _weighted_category_weights(
    values: np.ndarray, weights: np.ndarray, n_categories: int | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(categories, category_weight)`` -- total PSF weight per class.

    The categorical strategies all reduce to "sum the PSF weight landing in each
    class, then pick from that histogram". This shared helper builds that histogram
    once. When ``n_categories`` is known (e.g. IGBP's 20 classes) the categories are
    ``0 .. n_categories-1``; otherwise (e.g. SSF scene-type codes with no declared
    count) they are inferred from the distinct rounded values present.
    """
    mask = _usable(values, weights)
    v = np.rint(np.asarray(values, dtype=float)[mask]).astype(int)
    w = np.asarray(weights, dtype=float)[mask]
    if v.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    if n_categories is not None:
        # ``n_categories`` gives the size of a fixed 0-based code domain (e.g. IGBP's
        # 20 classes 0..19). Some categorical schemes are 1-based, though: CLDPIX cloud
        # phase uses codes 1..5, so the highest valid code (5) lands at ``bincount``
        # index 5 -- one past the declared count. Extend the histogram to cover the
        # largest code actually present so a valid top-of-range code is never silently
        # dropped; any unused low index (e.g. 0 for a 1-based scheme) simply gets zero
        # weight and is excluded from the mode/coverage outputs. Values are already
        # ``valid_range``-masked by the readers, so ``v.max()`` is bounded.
        domain = max(int(n_categories), int(v.max()) + 1)
        categories = np.arange(domain, dtype=int)
        # np.bincount sums the weights for each integer category in one pass.
        category_weight = np.bincount(v, weights=w, minlength=domain)
    else:
        categories, inverse = np.unique(v, return_inverse=True)
        category_weight = np.bincount(inverse, weights=w, minlength=categories.size)
    return categories, category_weight


def _order_categories(categories: np.ndarray, category_weight: np.ndarray) -> np.ndarray:
    """Categories ordered by descending PSF weight (most-common class first).

    Consumes a precomputed ``(categories, category_weight)`` histogram (see
    :func:`_weighted_category_weights`) so the ranked-mode variants and
    :func:`aggregate_tile_variables` can share one histogram per plane. Ties are
    broken by the ``np.argsort`` stable order (ascending category index), giving a
    deterministic result -- the same policy the doc's "CERES majority-rule" tie
    handling implies.
    """
    if categories.size == 0:
        return np.empty(0, dtype=int)
    # Negate so a stable ascending argsort yields descending weight; ties keep the
    # lower category index, which is deterministic.
    order = np.argsort(-category_weight, kind="stable")
    # Drop categories that received zero weight -- they are not present at all.
    order = [i for i in order if category_weight[i] > 0.0]
    return categories[order]


def _ranked_categories(values: np.ndarray, weights: np.ndarray, n_categories: int | None) -> np.ndarray:
    """Categories ordered by descending PSF weight -- builds the histogram then orders it."""
    return _order_categories(*_weighted_category_weights(values, weights, n_categories))


def _mode_from_histogram(categories: np.ndarray, category_weight: np.ndarray) -> dict[str, Any]:
    """PSF-weighted mode from a precomputed category histogram (see :func:`weighted_mode`)."""
    if categories.size == 0 or np.sum(category_weight) <= 0.0:
        return {"mode": float("nan"), "coverage_fractions": {}}
    total = float(np.sum(category_weight))
    winner = int(categories[int(np.argmax(category_weight))])
    fractions = {int(c): float(cw / total) for c, cw in zip(categories, category_weight, strict=True) if cw > 0.0}
    return {"mode": float(winner), "coverage_fractions": fractions}


def _ranked_mode_from_histogram(categories: np.ndarray, category_weight: np.ndarray, rank: int) -> dict[str, float]:
    """The ``rank``-th most common category from a precomputed histogram (see :func:`_ranked_mode`)."""
    ranked = _order_categories(categories, category_weight)
    if rank >= ranked.size:
        return {"mode": float("nan")}
    return {"mode": float(int(ranked[rank]))}


def weighted_mode(
    values: np.ndarray, weights: np.ndarray, *, n_categories: int | None = None, **_: Any
) -> dict[str, Any]:
    """PSF-weighted mode ``argmax_c sum(w*[x==c])`` for categorical variables.

    Returns the dominant class plus the per-class PSF-weight fractions, which record
    the scene mix within the footprint (design doc section 2.8.1.4.3). The mode is
    NaN when the footprint had no usable categorical data.
    """
    return _mode_from_histogram(*_weighted_category_weights(values, weights, n_categories))


def _ranked_mode(values: np.ndarray, weights: np.ndarray, rank: int, n_categories: int | None) -> dict[str, float]:
    """The ``rank``-th most common category (0 = primary, 1 = secondary, ...).

    Backs IGBP's ``surface_type_primary/secondary/tertiary`` ranked-scene outputs.
    Returns NaN when fewer than ``rank + 1`` classes are present, so a footprint that
    is entirely one land-cover type reports NaN for its (non-existent) secondary and
    tertiary scenes rather than a spurious class.
    """
    return _ranked_mode_from_histogram(*_weighted_category_weights(values, weights, n_categories), rank)


def weighted_mode_primary(
    values: np.ndarray, weights: np.ndarray, *, n_categories: int | None = None, **_: Any
) -> dict[str, float]:
    """Most common category by PSF weight (rank 0). See :func:`_ranked_mode`."""
    return _ranked_mode(values, weights, 0, n_categories)


def weighted_mode_secondary(
    values: np.ndarray, weights: np.ndarray, *, n_categories: int | None = None, **_: Any
) -> dict[str, float]:
    """Second most common category by PSF weight (rank 1). See :func:`_ranked_mode`."""
    return _ranked_mode(values, weights, 1, n_categories)


def weighted_mode_tertiary(
    values: np.ndarray, weights: np.ndarray, *, n_categories: int | None = None, **_: Any
) -> dict[str, float]:
    """Third most common category by PSF weight (rank 2). See :func:`_ranked_mode`."""
    return _ranked_mode(values, weights, 2, n_categories)


# ---------------------------------------------------------------------------
# Cloud / radiance-path strategies
# ---------------------------------------------------------------------------
#
# These are declared in the design doc (section 2.8.1.4) and provided here so the
# dispatcher is complete, but note that no *current* reader labels a variable with
# them -- VIIRS/CLDPIX cloud variables are presently aggregated with weighted_mean /
# weighted_log_mean. They are wired and unit-tested so a future reader (or a
# variables.yaml re-labelling) can select them without touching this module.


def coverage_fraction(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, float]:
    """PSF-weighted fraction of cells whose value exceeds zero ``sum(w*[x>0])/sum(w)``.

    Used for cloud fraction and snow fraction, where the input is a per-pixel
    presence indicator and the footprint statistic is the fractional area covered.
    """
    mask = _usable(values, weights)
    if not np.any(mask):
        return {"fraction": float("nan")}
    v = np.asarray(values, dtype=float)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    return {"fraction": float(np.sum(w * (v > 0.0)) / np.sum(w))}


def cloud_weighted_mean(
    values: np.ndarray, weights: np.ndarray, *, cloud_mask: np.ndarray | None = None, **_: Any
) -> dict[str, float]:
    """PSF-weighted mean over cloudy cells only (design doc section 2.8.1.4.3).

    Clear-sky cells are given zero weight before averaging, so the result is the
    in-cloud mean of a cloud-conditional quantity (optical depth, cloud-top height).
    ``cloud_mask`` is a per-cell boolean/0-1 array; when omitted, every finite cell
    counts as cloudy (the strategy then degenerates to :func:`weighted_mean`).
    """
    weights = np.asarray(weights, dtype=float)
    if cloud_mask is not None:
        # Zero the weight of every non-cloudy cell so it drops out of the mean.
        weights = np.where(np.asarray(cloud_mask, dtype=bool), weights, 0.0)
    mask = _usable(values, weights)
    n_cloudy = int(np.count_nonzero(mask))
    if n_cloudy == 0:
        return {"mean": float("nan"), "n_cloudy": 0.0}
    v = np.asarray(values, dtype=float)[mask]
    w = weights[mask]
    return {"mean": float(np.sum(w * v) / np.sum(w)), "n_cloudy": float(n_cloudy)}


def overcast_fraction(
    values: np.ndarray, weights: np.ndarray, *, threshold: float = 0.95, **_: Any
) -> dict[str, float]:
    """PSF-weighted overcast fraction ``sum(w*[cf>threshold])/sum(w_cloudy)``.

    Per CERES section 4.4.2.7: of the cloudy PSF weight, what fraction is fully
    overcast (cloud fraction above ``threshold``). Cloudy weight is that of cells
    with a positive cloud value; NaN when the footprint has no cloudy weight.
    """
    mask = _usable(values, weights)
    if not np.any(mask):
        return {"overcast_fraction": float("nan")}
    v = np.asarray(values, dtype=float)[mask]
    w = np.asarray(weights, dtype=float)[mask]
    cloudy_weight = np.sum(w * (v > 0.0))
    if cloudy_weight <= 0.0:
        return {"overcast_fraction": float("nan")}
    return {"overcast_fraction": float(np.sum(w * (v > threshold)) / cloudy_weight)}


def aggregate_broadband_radiance(
    values: np.ndarray, weights: np.ndarray, *, cloud_mask: np.ndarray | None = None, **_: Any
) -> dict[str, float]:
    """Aggregate camera broadband radiance (simplified stand-in).

    The full CERES treatment separates radiance by cloud-layer type within each
    angular bin (design doc section 2.8.1.4.3). That scene-dependent bin weighting
    needs the per-bin cloud layering, which the current stand-in weighting layer does
    not yet expose, so this provides the total PSF-weighted radiance now and reports
    NaN for the clear/cloudy split.

    TODO[LIBSDC-785]: implement the cloud-layer-aware bin weighting once the angular
    PSF projection (see :mod:`libera_utils.footprint_matching.weighting`) is in place.
    """
    total = weighted_mean(values, weights)["mean"]
    if cloud_mask is None:
        return {"total_radiance": total, "clear_radiance": float("nan"), "cloudy_radiance": float("nan")}
    cloud_mask = np.asarray(cloud_mask, dtype=bool)
    cloudy = cloud_weighted_mean(values, weights, cloud_mask=cloud_mask)["mean"]
    clear = cloud_weighted_mean(values, weights, cloud_mask=~cloud_mask)["mean"]
    return {"total_radiance": total, "clear_radiance": clear, "cloudy_radiance": cloudy}


def report_all(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, np.ndarray]:
    """Return the raw usable ``(values, weights)`` without aggregating.

    Used for pass-through variables (e.g. FlashFlux reference fluxes, camera-CF
    pixel provenance) that downstream algorithms aggregate themselves. Not a scalar
    strategy, so it has no entry in :data:`_SCALAR_KEY` and is never used on the
    per-footprint product path.
    """
    mask = _usable(values, weights)
    return {"values": np.asarray(values, dtype=float)[mask], "weights": np.asarray(weights, dtype=float)[mask]}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Registry of strategy name -> implementing function. New strategies are added here
# (and, if they yield a product scalar, in _SCALAR_KEY below). This is the single
# lookup the dispatcher and the product path share, mirroring the doc's "internal
# registry" (section 2.8.1.4.4).
_STRATEGIES: dict[str, Callable[..., dict[str, Any]]] = {
    "weighted_mean": weighted_mean,
    "weighted_std": weighted_std,
    "weighted_log_mean": weighted_log_mean,
    "weighted_median": weighted_median,
    "weighted_mode": weighted_mode,
    "weighted_mode_primary": weighted_mode_primary,
    "weighted_mode_secondary": weighted_mode_secondary,
    "weighted_mode_tertiary": weighted_mode_tertiary,
    "coverage_fraction": coverage_fraction,
    "cloud_weighted_mean": cloud_weighted_mean,
    "overcast_fraction": overcast_fraction,
    "aggregate_broadband_radiance": aggregate_broadband_radiance,
    "report_all": report_all,
}

# For each strategy that produces a single product value, which key of its result
# dict is that value. report_all is intentionally absent (it returns arrays, not a
# scalar), so selecting a scalar for it raises -- a reader must not label a product
# variable report_all.
_SCALAR_KEY: dict[str, str] = {
    "weighted_mean": "mean",
    "weighted_std": "standard_deviation",
    "weighted_log_mean": "log_mean",
    "weighted_median": "median",
    "weighted_mode": "mode",
    "weighted_mode_primary": "mode",
    "weighted_mode_secondary": "mode",
    "weighted_mode_tertiary": "mode",
    "coverage_fraction": "fraction",
    "cloud_weighted_mean": "mean",
    "overcast_fraction": "overcast_fraction",
    "aggregate_broadband_radiance": "total_radiance",
}

# Cores that consume a single plane's shared partials, keyed by strategy name.
# :func:`aggregate_tile_variables` dispatches through these so several product
# variables derived from one data plane (e.g. a mean-aggregated variable and its
# ``_standard_deviation`` companion) reuse a single :func:`_usable` mask instead of
# recomputing it per variable. Each core returns the same result dict as its public
# strategy (which are thin wrappers over the same cores), so the shared and the
# per-variable paths are numerically identical.
_CONTINUOUS_CORES: dict[str, Callable[[_WeightedPartials], dict[str, Any]]] = {
    "weighted_mean": _mean_from_partials,
    "weighted_std": _std_from_partials,
    "weighted_log_mean": _log_mean_from_partials,
    "weighted_median": _median_from_partials,
}

# Rank each ranked-mode strategy selects from the shared category histogram (0 =
# primary/most common). ``weighted_mode`` (the full mode + coverage fractions) is
# handled separately because it returns extra keys.
_CATEGORICAL_RANK: dict[str, int] = {
    "weighted_mode_primary": 0,
    "weighted_mode_secondary": 1,
    "weighted_mode_tertiary": 2,
}


def aggregate(
    strategy: str,
    values: np.ndarray,
    weights: np.ndarray,
    *,
    total_energy: float | None = None,
    **kwargs: Any,
) -> AggregationResult:
    """Run a named aggregation strategy and wrap it with coverage bookkeeping.

    Looks ``strategy`` up in the registry, invokes it on ``(values, weights)``, and
    packages the result in an :class:`AggregationResult` whose coverage is the
    fraction of the PSF's total weight (``total_energy``) that was backed by usable
    data. When ``total_energy`` is not supplied it defaults to the total finite
    weight, so coverage collapses to 1.0 for a fully-covered footprint.

    Parameters
    ----------
    strategy : str
        Registered strategy name (a key of :data:`_STRATEGIES`).
    values, weights : np.ndarray
        Per-cell data values and PSF weights (any shape; flattened internally by the
        strategies via the usable-mask helper).
    total_energy : float, optional
        Total PSF weight within the 95% contour, the coverage denominator. Defaults
        to the summed finite positive weight.
    **kwargs
        Extra strategy arguments (e.g. ``n_categories``, ``cloud_mask``,
        ``threshold``).

    Returns
    -------
    AggregationResult

    Raises
    ------
    ValueError
        If ``strategy`` is not registered.
    """
    try:
        func = _STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown aggregation strategy {strategy!r}. Known strategies: {sorted(_STRATEGIES)}."
        ) from exc

    result = func(values, weights, **kwargs)

    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mask = _usable(values_arr, weights_arr)
    valid_weight = float(np.sum(weights_arr[mask])) if np.any(mask) else 0.0
    denom = total_energy if total_energy is not None else valid_weight
    coverage = (valid_weight / denom) if denom and denom > 0.0 else 0.0
    _, partial = check_coverage(valid_weight, denom if denom else 0.0)

    return AggregationResult(
        values=result,
        coverage=float(coverage),
        n_valid=int(np.count_nonzero(mask)),
        n_total=int(values_arr.size),
        partial_flag=bool(partial),
    )


# ---------------------------------------------------------------------------
# Product-facing projection: reader product specs -> one scalar each
# ---------------------------------------------------------------------------


@cache
def _map_product_specs(reader_cls: Any) -> list[tuple[VariableSpec, int]]:
    """Associate every product variable spec with the read-plane it derives from.

    Cached per reader class (:func:`functools.lru_cache`): the spec-to-plane mapping is a
    static property of the reader, so it is computed once rather than rebuilt for every
    footprint by :func:`aggregate_tile_variables`. Callers only iterate the returned list
    (never mutate it), so sharing one instance across calls is safe.

    A reader's *product* variables (``product_variable_specs()``) are a superset of
    the *read* variables (``VARIABLES``): each read variable, plus a
    ``<name>_standard_deviation`` companion for continuous ones, plus reader-specific
    derived extras (IGBP's ranked scenes). All of those are computed from the *same*
    source data plane as their parent read variable, whose position on axis 0 of a
    multi-variable :class:`GridTile` we need in order to slice out its values.

    This resolves each product spec to that axis-0 index by name:

    * an exact read-variable name -> that read plane;
    * a ``..._standard_deviation`` companion -> the read plane of its stem;
    * otherwise a derived extra (e.g. ``surface_type_primary``) -> the read plane
      whose name is the longest prefix of the spec name.

    Returns
    -------
    list[tuple[VariableSpec, int]]
        ``(product_spec, read_plane_index)`` for every product variable, in product
        order.

    Raises
    ------
    ValueError
        If a product spec cannot be associated with any read variable (a
        misconfigured reader), so the failure is loud rather than a silent bad plane.
    """
    read_names = [spec.name for spec in reader_cls.VARIABLES]
    index_by_name = {name: i for i, name in enumerate(read_names)}

    mapping: list[tuple[VariableSpec, int]] = []
    for spec in reader_cls.product_variable_specs():
        if spec.name in index_by_name:
            mapping.append((spec, index_by_name[spec.name]))
            continue
        if spec.name.endswith(STANDARD_DEVIATION_SUFFIX):
            stem = spec.name[: -len(STANDARD_DEVIATION_SUFFIX)]
            if stem in index_by_name:
                mapping.append((spec, index_by_name[stem]))
                continue
        # Derived extra: match the read variable whose name is the longest prefix.
        prefixes = [name for name in read_names if spec.name.startswith(name)]
        if not prefixes:
            raise ValueError(
                f"Reader {reader_cls.__name__!r}: product variable {spec.name!r} cannot be mapped to "
                f"any read variable {read_names!r}. Derived product variables must be named after the "
                f"read variable they are computed from."
            )
        parent = max(prefixes, key=len)
        mapping.append((spec, index_by_name[parent]))
    return mapping


def aggregate_tile_variables(reader_cls: Any, tile: GridTile, weight_field: WeightField) -> dict[str, float]:
    """Aggregate one footprint's merged tile to one scalar per product variable.

    For the given reader, this slices the correct data plane for each product
    variable (see :func:`_map_product_specs`), dispatches to that variable's
    aggregation strategy with the PSF ``weight_field``, and returns the single scalar
    the product stores -- keyed by the *bare* spec name (the caller prefixes it with
    ``<source>_<instrument>_`` to form the product variable name).

    An empty or all-NaN tile yields NaN for every variable (the aggregation strategies
    return NaN when nothing is usable), which is exactly the partial-coverage signal
    the TileManager produces for a failed/missing region.

    Parameters
    ----------
    reader_cls : type[GriddedDataReader]
        The reader class whose product variables to compute.
    tile : GridTile
        The merged tile covering this footprint's PSF bounding box.
    weight_field : WeightField
        Per-cell PSF weights aligned to ``tile.lats`` x ``tile.lons``.

    Returns
    -------
    dict[str, float]
        ``{spec.name: scalar}`` for every variable in
        ``reader_cls.product_variable_specs()``.

    Notes
    -----
    Only the scalar each strategy yields is returned; the coverage bookkeeping that
    :func:`aggregate` computes is not needed here (the orchestrator derives coverage
    separately via :func:`~libera_utils.footprint_matching.product._tile_coverage`), so
    this dispatches to the strategy cores directly rather than through :func:`aggregate`.
    Because many product variables derive from the same data plane -- a variable and its
    ``_standard_deviation`` companion, or IGBP's four ranked-mode scenes off one
    ``surface_type`` plane -- the per-plane :func:`_usable` mask (continuous) and category
    histogram (categorical) are computed once and reused across those variables.
    """
    # Normalize to a 3-D (n_var, n_lat, n_lon) view so single- and multi-variable
    # readers share one code path. Axis 0 order matches VARIABLES (the reader
    # contract), which is what _map_product_specs indexes into.
    data = np.asarray(tile.data)
    data_3d = data[np.newaxis, ...] if data.ndim == 2 else data
    n_planes = data_3d.shape[0]

    weights = np.asarray(weight_field.weights, dtype=float).ravel()

    # Lazily computed, per-plane shared work: the usable-masked partials (reused by every
    # continuous variable on the plane) and the category histogram per (plane,
    # n_categories) (reused by a plane's mode + ranked-mode variants). Cached within this
    # call only -- weights differ per footprint, so nothing survives across footprints.
    def plane(read_index: int) -> np.ndarray:
        return data_3d[read_index].ravel() if read_index < n_planes else np.empty(0, dtype=float)

    partials_cache: dict[int, _WeightedPartials] = {}

    def partials(read_index: int) -> _WeightedPartials:
        cached = partials_cache.get(read_index)
        if cached is None:
            cached = _prepare(plane(read_index), weights)
            partials_cache[read_index] = cached
        return cached

    histogram_cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray]] = {}

    def histogram(read_index: int, n_categories: int | None) -> tuple[np.ndarray, np.ndarray]:
        key = (read_index, n_categories)
        cached = histogram_cache.get(key)
        if cached is None:
            cached = _weighted_category_weights(plane(read_index), weights, n_categories)
            histogram_cache[key] = cached
        return cached

    out: dict[str, float] = {}
    for spec, read_index in _map_product_specs(reader_cls):
        strategy = spec.aggregation
        if strategy in _CONTINUOUS_CORES:
            result = _CONTINUOUS_CORES[strategy](partials(read_index))
        else:
            # Prefer the product spec's own category count; fall back to the parent read
            # spec's (a std companion carries n_categories=None but its parent may not).
            n_categories = spec.n_categories
            if n_categories is None:
                n_categories = reader_cls.VARIABLES[read_index].n_categories
            if strategy == "weighted_mode":
                result = _mode_from_histogram(*histogram(read_index, n_categories))
            elif strategy in _CATEGORICAL_RANK:
                result = _ranked_mode_from_histogram(*histogram(read_index, n_categories), _CATEGORICAL_RANK[strategy])
            else:
                # Uncommon strategies (cloud / radiance path) that no current reader
                # labels on the product path: fall back to a direct strategy call. Still
                # bypasses aggregate() (no coverage needed here).
                result = _STRATEGIES[strategy](plane(read_index), weights, n_categories=n_categories)
        out[spec.name] = float(result[_SCALAR_KEY[strategy]])
    return out
