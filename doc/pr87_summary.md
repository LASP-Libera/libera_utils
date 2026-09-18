# PR #87 — FMATCH 7: Product Assembly

`fmatch-7-product-assembly` → `fmatch-6-product-definitions` (stacked on #86) · +1877 / −0 across 2 files

## Summary

Adds `product.py`, the seam between the footprint-matching engine (readers, PSF aggregation, geometry) and the Libera data-product machinery (`LiberaDataProductDefinition` / `write_libera_data_product`). Turns matched footprints into a conformant NetCDF file for all five operational modes.

## How each variable is filled (`libera_utils/footprint_matching/product.py`)

| Route | Source |
|-------|--------|
| **Input pass-through** | Geolocation + viewing angles straight from L1B Daily (`_RADIOMETER_L1B_VARIABLES` for radiometer modes, `_CAMTIME_SEGMENTATION_VARIABLES` for camera-segmented modes) |
| **Computed** | Derived viewing geometry (`sunglint_angle`) always; external aggregated variables + coverage/QA (`psf_coverage_fraction`, `q_flags`) whenever staged ancillary inputs are supplied — the full PSF aggregation over tiling/weighting/geometry (design doc §2.8.1) |
| **Placeholder** | What remains (external columns with no ancillary data, or future-work refs like FLASHFlux) — structurally correct but numerically meaningless: NaN for float, 0 for int |

## Key entry points

- `assemble_fmatch_dataset()` / `write_fmatch_product(mode, ...)` — top-level assembly + write per mode.
- `build_radiometer_footprints()`, `aggregate_external_variables()`, `compute_derived_viewing_geometry()`.
- Mode helpers: `load_fmatch_definition()`, `fmatch_time_variable()`, `is_camera_timescale_mode()`; radiometer vs. camtime assembly split (`_assemble_radiometer_dataset` / `_assemble_camtime_dataset`).

## Tests

- `tests/unit/test_footprint_matching/test_aggregate_external_variables.py` — external-variable aggregation path.
