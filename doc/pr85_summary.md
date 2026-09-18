# PR #85 — FMATCH 5: Aggregation

`fmatch-5-weighting-aggregation` → `fmatch-4-tiling` (stacked on #84) · +2345 / −5 across 6 files

## Summary

Adds the last core-processing stage of footprint matching: PSF weighting of a footprint's merged tile and aggregation of those weighted pixels into one statistic per variable (the value that lands in the SSF-style output product). Design doc §2.8.1.1 / §2.8.1.4.

## `weighting.py` — per-pixel PSF weights

Interface `PixelWeigher` + `WeightField`, with two implementations:

| Weigher | Behavior |
|---------|----------|
| `AngularPSFWeigher` | CERES-faithful — projects each grid cell into the radiometer along-scan/cross-scan angular frame `(delta, beta)` and evaluates the analytic CERES PSF, honoring asymmetric along-scan response, scan direction (L1B `Cone_Angle_Rate`), and the stationary-scanner static FOV |
| `RadialWeigher` | Simple boresight-centered Gaussian fallback — needs only the boresight, not full viewing geometry |

## `aggregation.py` — collapse pixels to one value

Two layers:

- **Pure strategy functions** — 1-D `(values, weights)` → named sub-results, one per method in the design-doc table: `weighted_mean`, `weighted_std`, `weighted_log_mean`, `weighted_median`, `weighted_mode` (primary/secondary/tertiary), `coverage_fraction`, `cloud_weighted_mean`, `overcast_fraction`, `aggregate_broadband_radiance`, `report_all`. NaNs excluded from all computations.
- **Product-facing projection** — `aggregate_tile_variables(reader_cls, tile, weight_field)` walks a reader's product `VariableSpec`s, picks the right data plane for each, and applies its aggregation method. Plus `aggregate()`, `check_coverage()`, `AggregationResult`.

## Notebook & tests

- `notebooks/05_weighting_aggregation.ipynb` — weighting + aggregation walkthrough (minor `04_tiling.ipynb` touch-up).
- Unit tests: `test_weighting.py`, `test_aggregation.py`.
