"""PSF aggregation engine: collapse a footprint's native samples to one value.

Where this sits in the pipeline
-------------------------------
This is the last core-processing stage of footprint matching. Upstream, a reader
has served the native :class:`~libera_utils.footprint_matching.types.SampleSet`
inside a footprint's PSF bounding box, and the
:mod:`~libera_utils.footprint_matching.weighting` module has assigned each sample
a PSF weight. This module takes those (values, weights) and produces a single
statistic per variable -- the number that lands in the SSF-style output product.
It is the only place a sample's value is combined with another's.

Two layers
----------
1. **Pure strategy functions** (``weighted_mean``, ``weighted_mode``,
   ``weighted_log_mean`` ...). Each takes 1-D ``values`` and ``weights`` arrays and
   returns a small ``dict`` of named sub-results, following the methods in design
   doc section 2.8.1.4. These are reusable by scientists directly and are what the
   unit tests exercise. NaN values are excluded from every computation (doc: "NaN
   values are excluded from all computations").
2. **A product-facing projection** (:func:`aggregate_sample_variables`) that walks a
   reader's *product* variable specifications, picks the right value row for each,
   dispatches to the correct strategy, and returns the single scalar the product
   definition stores for that variable, once per footprint per source.

CERES heritage
--------------
The weighted statistics implement CERES ATBD v2.2 Eq. 4.4-17
``xbar = sum(w*x) / sum(w)`` and its companions, and follow the CERES SSF
convention of log-space averaging for pixel cloud optical depth. See
https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf

Optical depth is averaged in log space only where the samples are imager pixels, as
CERES does when it convolves pixels into a footprint (see :func:`weighted_log_mean`).
An optical depth a source has already averaged over its own footprint or grid cell,
and every aerosol optical depth, is combined with the arithmetic mean: CERES takes "an
arithmetic mean ... of all imager pixels in the angular bin before they are weighted
by the PSF" for aerosol optical depth (SSF-137, CERES SSF Collection Guide, Release 2,
p. 101, https://ceres.larc.nasa.gov/documents/collect_guide/pdf/SSF_CG_R2V1.pdf).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.footprint_matching.types import STANDARD_DEVIATION_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Callable

    from libera_utils.footprint_matching.readers.base import AncillaryReader
    from libera_utils.footprint_matching.types import SampleSet, VariableSpec
    from libera_utils.footprint_matching.weighting import WeightField

# CERES coverage thresholds (design doc section 2.4.2.7 / 2.8.1.1.3). The coverage
# metric is the fraction of the PSF's 95%-energy contour backed by usable data, as the
# weigher measures it on its reference grid (WeightField.coverage):
#   >= 95%           -> fully accepted
#   75% .. < 95%     -> accepted but flagged as *partial* coverage
#   < 75%            -> insufficient data (the orchestrator discards the footprint)
# We only *compute and flag* here; the discard decision is the orchestrator's, so
# these are exposed as named constants rather than hard-coded at the call site.
ACCEPT_COVERAGE_THRESHOLD: float = 0.75
PARTIAL_COVERAGE_THRESHOLD: float = 0.95


@dataclass
class AggregationResult:
    """Output of a single aggregation, with sample bookkeeping.

    The ``values`` dict holds the strategy's named sub-results (e.g. ``{"mean": ...}``,
    ``{"mode": ..., "coverage_fractions": ...}``). Footprint coverage is not
    recorded here: a swath gap has no samples, so it can only be measured on the
    weigher's reference grid (``WeightField.coverage``), which :func:`check_coverage`
    scores.

    Attributes
    ----------
    values : dict[str, Any]
        Named results produced by the strategy function.
    n_valid : int
        Number of samples that contributed (finite value, positive weight).
    n_total : int
        Number of samples considered (the sample set's size).
    """

    values: dict[str, Any]
    n_valid: int
    n_total: int


def check_coverage(
    coverage: float,
    *,
    accept_threshold: float = ACCEPT_COVERAGE_THRESHOLD,
    partial_threshold: float = PARTIAL_COVERAGE_THRESHOLD,
) -> tuple[bool, bool]:
    """Apply the CERES 75%/95% coverage rule.

    Parameters
    ----------
    coverage : float
        Fraction in ``[0, 1]`` of the PSF's in-contour energy backed by usable data,
        as the weigher reports it (``WeightField.coverage``).
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
    accept = coverage >= accept_threshold
    partial = accept_threshold <= coverage < partial_threshold
    return accept, partial


# ---------------------------------------------------------------------------
# Shared masking helper
# ---------------------------------------------------------------------------


def _usable(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Boolean mask of samples usable for a weighted statistic.

    A sample is usable when its value is finite (NaN samples are excluded -- the
    reader fill convention) and its PSF weight is finite and strictly positive (a
    zero-weight sample lies outside the energy contour and must not contribute).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)


@dataclass(frozen=True)
class _WeightedPartials:
    """The usable ``(values, weights)`` of one data plane, masked once.

    The continuous surface strategies (:func:`weighted_mean`, :func:`weighted_std`,
    :func:`weighted_log_mean`, :func:`weighted_positive_std`) each independently apply the
    ``_usable`` mask before doing their arithmetic. When several product variables
    derive from the *same* data plane -- e.g. a variable and its
    ``<name>_standard_deviation`` companion -- that mask (and the array copy it drives)
    is identical across them. :func:`_prepare` computes it once and the ``*_from_partials``
    cores consume the result, so :func:`aggregate_sample_variables` pays the masking cost
    once per plane rather than once per variable. The public strategy functions are thin
    wrappers over the same cores, so the shared and the standalone paths are numerically
    identical by construction.

    Attributes
    ----------
    values, weights : np.ndarray
        1-D arrays of the plane's usable cell values and PSF weights (already masked
        by ``_usable``), in grid order.
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
# calls its ``*_from_partials`` core. The core is what aggregate_sample_variables calls
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


def _positive(p: _WeightedPartials) -> _WeightedPartials:
    """The strictly positive samples of pre-masked partials, the set a geometric mean is defined on."""
    positive = p.values > 0.0
    return _WeightedPartials(values=p.values[positive], weights=p.weights[positive])


def _log_mean_from_partials(p: _WeightedPartials) -> dict[str, float]:
    """Weighted geometric mean from pre-masked partials (see :func:`weighted_log_mean`)."""
    q = _positive(p)
    if q.weights.size == 0:
        return {"log_mean": float("nan")}
    return {"log_mean": float(np.exp(np.sum(q.weights * np.log(q.values)) / np.sum(q.weights)))}


def _positive_std_from_partials(p: _WeightedPartials) -> dict[str, float]:
    """Weighted standard deviation over the strictly positive samples (see :func:`weighted_positive_std`)."""
    return _std_from_partials(_positive(p))


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
    """PSF-weighted geometric mean ``exp(sum(w*ln x)/sum(w))``.

    Used for pixel-level cloud optical depth, which CERES averages in log space:
    "Cloud optical depth is averaged in logarithm form, since the logarithm of the
    cloud optical depth is proportional to the visible radiance" (CERES FluxByCldTyp
    Ed4A Data Quality Summary, p. 9,
    https://ceres.larc.nasa.gov/documents/DQ_summaries/CERES_FluxByCldTyp_Ed4A_DQS.pdf),
    and the SSF records the PSF-weighted mean of the natural logarithms of the pixels'
    visible optical depth (SSF-85, CERES SSF Collection Guide, Release 2, p. 76,
    https://ceres.larc.nasa.gov/documents/collect_guide/pdf/SSF_CG_R2V1.pdf). This
    strategy returns the exponential of that mean, so the value stays in optical-depth
    units. Only strictly positive values contribute -- ``ln`` is undefined at and
    below zero -- so those samples are dropped in addition to the usual NaN exclusion.
    Its ``_standard_deviation`` companion is :func:`weighted_positive_std`, over the
    same positive samples.
    """
    return _log_mean_from_partials(_prepare(values, weights))


def weighted_positive_std(values: np.ndarray, weights: np.ndarray, **_: Any) -> dict[str, float]:
    """PSF-weighted standard deviation over the strictly positive samples.

    The ``<name>_standard_deviation`` companion of a :func:`weighted_log_mean`
    variable, so the spread describes the same samples as the mean it pairs with. In
    the variable's own units, as :func:`weighted_std` is. NaN when no sample is
    positive.
    """
    return _positive_std_from_partials(_prepare(values, weights))


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
    ``0 .. n_categories-1``; otherwise (e.g. FLASHFlux scene-type codes with no declared
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
    :func:`aggregate_sample_variables` can share one histogram per plane. Ties are
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
    """The ``rank``-th most common category from a precomputed histogram (see ``_ranked_mode``)."""
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
    """Most common category by PSF weight (rank 0). See ``_ranked_mode``."""
    return _ranked_mode(values, weights, 0, n_categories)


def weighted_mode_secondary(
    values: np.ndarray, weights: np.ndarray, *, n_categories: int | None = None, **_: Any
) -> dict[str, float]:
    """Second most common category by PSF weight (rank 1). See ``_ranked_mode``."""
    return _ranked_mode(values, weights, 1, n_categories)


def weighted_mode_tertiary(
    values: np.ndarray, weights: np.ndarray, *, n_categories: int | None = None, **_: Any
) -> dict[str, float]:
    """Third most common category by PSF weight (rank 2). See ``_ranked_mode``."""
    return _ranked_mode(values, weights, 2, n_categories)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# Registry of strategy name -> (implementing function, key of its result dict that is
# the single value the product stores). This is the single lookup the dispatcher and
# the product path share, mirroring the doc's "internal registry" (section
# 2.8.1.4.4). A new strategy is added here and in the product path's dispatch table
# for its kind: _CONTINUOUS_CORES, _CATEGORICAL_RANK, or the ``weighted_mode`` branch
# of :func:`aggregate_sample_variables`.
_STRATEGIES: dict[str, tuple[Callable[..., dict[str, Any]], str]] = {
    "weighted_mean": (weighted_mean, "mean"),
    "weighted_std": (weighted_std, "standard_deviation"),
    "weighted_log_mean": (weighted_log_mean, "log_mean"),
    "weighted_positive_std": (weighted_positive_std, "standard_deviation"),
    "weighted_mode": (weighted_mode, "mode"),
    "weighted_mode_primary": (weighted_mode_primary, "mode"),
    "weighted_mode_secondary": (weighted_mode_secondary, "mode"),
    "weighted_mode_tertiary": (weighted_mode_tertiary, "mode"),
}

_SCALAR_KEY: dict[str, str] = {name: scalar_key for name, (_, scalar_key) in _STRATEGIES.items()}

# Cores that consume a single plane's shared partials, keyed by strategy name.
# :func:`aggregate_sample_variables` dispatches through these so several product
# variables derived from one data plane (e.g. a mean-aggregated variable and its
# ``_standard_deviation`` companion) reuse a single ``_usable`` mask instead of
# recomputing it per variable. Each core returns the same result dict as its public
# strategy (which are thin wrappers over the same cores), so the shared and the
# per-variable paths are numerically identical.
_CONTINUOUS_CORES: dict[str, Callable[[_WeightedPartials], dict[str, Any]]] = {
    "weighted_mean": _mean_from_partials,
    "weighted_std": _std_from_partials,
    "weighted_log_mean": _log_mean_from_partials,
    "weighted_positive_std": _positive_std_from_partials,
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
    **kwargs: Any,
) -> AggregationResult:
    """Run a named aggregation strategy and wrap it with sample bookkeeping.

    Looks ``strategy`` up in the registry, invokes it on ``(values, weights)``, and
    packages the result in an :class:`AggregationResult` with the number of samples
    that contributed.

    Parameters
    ----------
    strategy : str
        Registered strategy name (a key of ``_STRATEGIES``).
    values, weights : np.ndarray
        Per-cell data values and PSF weights (any shape; flattened internally by the
        strategies via the usable-mask helper).
    **kwargs
        Extra strategy arguments (``n_categories`` for the mode strategies).

    Returns
    -------
    AggregationResult

    Raises
    ------
    ValueError
        If ``strategy`` is not registered.
    """
    try:
        func, _ = _STRATEGIES[strategy]
    except KeyError as exc:
        raise ValueError(
            f"Unknown aggregation strategy {strategy!r}. Known strategies: {sorted(_STRATEGIES)}."
        ) from exc

    result = func(values, weights, **kwargs)

    values_arr = np.asarray(values, dtype=float)
    mask = _usable(values_arr, np.asarray(weights, dtype=float))
    return AggregationResult(
        values=result,
        n_valid=int(np.count_nonzero(mask)),
        n_total=int(values_arr.size),
    )


# ---------------------------------------------------------------------------
# Product-facing projection: reader product specs -> one scalar each
# ---------------------------------------------------------------------------


@cache
def _map_product_specs(reader_cls: type[AncillaryReader]) -> list[tuple[VariableSpec, int]]:
    """Associate every product variable spec with the read-plane it derives from.

    Cached per reader class (:func:`functools.lru_cache`): the spec-to-plane mapping is a
    static property of the reader, so it is computed once rather than rebuilt for every
    footprint by :func:`aggregate_sample_variables`. Callers only iterate the returned list
    (never mutate it), so sharing one instance across calls is safe.

    A reader's *product* variables (``product_variable_specs()``) are a superset of
    the *read* variables (``VARIABLES``): each read variable, plus a
    ``<name>_standard_deviation`` companion for continuous ones, plus reader-specific
    derived extras (IGBP's ranked scenes). All of those are computed from the *same*
    source data plane as their parent read variable, whose position on axis 0 of a
    multi-variable ``values`` array we need in order to pick out its row.

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
        If a product spec cannot be associated with any read variable, or names an
        unregistered aggregation strategy (a misconfigured reader), so the failure is
        loud at the first footprint rather than a silent bad plane.
    """
    read_names = [spec.name for spec in reader_cls.VARIABLES]
    index_by_name = {name: i for i, name in enumerate(read_names)}

    mapping: list[tuple[VariableSpec, int]] = []
    for spec in reader_cls.product_variable_specs():
        if spec.aggregation not in _SCALAR_KEY:
            raise ValueError(
                f"Reader {reader_cls.__name__!r}: product variable {spec.name!r} names aggregation "
                f"{spec.aggregation!r}, which is not a registered aggregation strategy. Set its "
                f"VariableSpec.aggregation to one of {sorted(_SCALAR_KEY)}."
            )
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


def aggregate_sample_variables(
    reader_cls: type[AncillaryReader], samples: SampleSet, weight_field: WeightField
) -> dict[str, float]:
    """Aggregate one footprint's samples from one source to one scalar per product variable.

    For the given reader, this picks the correct value row for each product
    variable (see ``_map_product_specs``), dispatches to that variable's
    aggregation strategy with the PSF ``weight_field``, and returns the single scalar
    the product stores -- keyed by the *bare* spec name (the caller prefixes it with
    ``<source>_`` to form the product variable name).

    An empty or all-NaN sample set yields NaN for every variable (the aggregation
    strategies return NaN when nothing is usable), which is the signal for a region
    the source does not cover.

    Parameters
    ----------
    reader_cls : type[AncillaryReader]
        The reader class whose product variables to compute.
    samples : SampleSet
        The native samples inside this footprint's PSF bounding box.
    weight_field : WeightField
        Per-sample PSF weights aligned to ``samples.lats`` / ``samples.lons``.

    Returns
    -------
    dict[str, float]
        ``{spec.name: scalar}`` for every variable in
        ``reader_cls.product_variable_specs()``.

    Raises
    ------
    ValueError
        If a product variable cannot be mapped to a read variable or names an
        unregistered aggregation (see ``_map_product_specs``); if
        ``samples.values`` is not ``(len(reader_cls.VARIABLES), samples.n_samples)``; or
        if ``weight_field.weights`` does not hold one weight per sample.

    Notes
    -----
    Only the scalar each strategy yields is returned, so this dispatches to the strategy
    cores directly rather than through :func:`aggregate`. Coverage is the weigher's
    (``WeightField.coverage``, scored by :func:`check_coverage`), not computed here.
    Because many product variables derive from the same data plane -- a variable and its
    ``_standard_deviation`` companion, or IGBP's four ranked-mode scenes off one
    ``surface_type`` plane -- the per-plane ``_usable`` mask (continuous) and category
    histogram (categorical) are computed once and reused across those variables.
    """
    mapping = _map_product_specs(reader_cls)

    # (n_var, n) value rows; axis 0 order matches VARIABLES (the reader contract),
    # which is what _map_product_specs indexes into.
    data_rows = np.asarray(samples.values)
    expected_shape = (len(reader_cls.VARIABLES), samples.n_samples)
    if data_rows.shape != expected_shape:
        raise ValueError(
            f"Reader {reader_cls.__name__!r}: samples.values has shape {data_rows.shape}, expected "
            f"{expected_shape} (one row per VARIABLES entry, one column per sample). The reader served "
            f"values that do not match its declared variables."
        )

    weights = np.asarray(weight_field.weights, dtype=float).ravel()
    if weights.size != samples.n_samples:
        raise ValueError(
            f"Reader {reader_cls.__name__!r}: the weight field holds {weights.size} weights for "
            f"{samples.n_samples} samples. Weigh the same SampleSet that is being aggregated."
        )

    # Lazily computed, per-plane shared work: the usable-masked partials (reused by every
    # continuous variable on the plane) and the category histogram per (plane,
    # n_categories) (reused by a plane's mode + ranked-mode variants). Cached within this
    # call only -- weights differ per footprint, so nothing survives across footprints.
    partials_cache: dict[int, _WeightedPartials] = {}

    def partials(read_index: int) -> _WeightedPartials:
        cached = partials_cache.get(read_index)
        if cached is None:
            cached = _prepare(data_rows[read_index], weights)
            partials_cache[read_index] = cached
        return cached

    histogram_cache: dict[tuple[int, int | None], tuple[np.ndarray, np.ndarray]] = {}

    def histogram(read_index: int, n_categories: int | None) -> tuple[np.ndarray, np.ndarray]:
        key = (read_index, n_categories)
        cached = histogram_cache.get(key)
        if cached is None:
            cached = _weighted_category_weights(data_rows[read_index], weights, n_categories)
            histogram_cache[key] = cached
        return cached

    out: dict[str, float] = {}
    for spec, read_index in mapping:
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
            else:
                result = _ranked_mode_from_histogram(*histogram(read_index, n_categories), _CATEGORICAL_RANK[strategy])
        out[spec.name] = float(result[_SCALAR_KEY[strategy]])
    return out
