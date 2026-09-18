# PR #86 — FMATCH 6: Product Definitions

`fmatch-6-product-definitions` → `fmatch-5-weighting-aggregation` (stacked on #85) · +5097 / −0 across 8 files

## Summary

Adds the Pydantic/YAML product definitions for the FMATCH output products, plus the missing `FMATCH-IMAGER-FLASH` product identifier. Data definitions only — no processing code.

## Product definitions (`libera_utils/data/product_definitions/`)

| File | Product | Timescale / notes |
|------|---------|-------------------|
| `fmatch_cam.yml` | FMATCH-CAM | radiometer timescale; `cloud_fraction_camera` in percent `[0, 100]` to match CF-CAM input |
| `fmatch_cam_camtime.yml` | FMATCH-CAM-CAMTIME | camera timescale |
| `fmatch_imager.yml` | FMATCH-IMAGER | radiometer timescale (RBSP Climate Quality); carries RBSP CLDPIX/SSF cloud fields + full ERA5 single-level and 37-level pressure fields + VIIRS imager fields + extended SSF cloud/aerosol/albedo vars (multi-axis flattened to 1-D per footprint, gated via `VariableSpec.only_modes`) |
| `fmatch_imager_camtime.yml` | FMATCH-IMAGER-CAMTIME | camera timescale |
| `fmatch_imager_flash.yml` | FMATCH-IMAGER-FLASH | FLASHFlux stream |

## Constants (`libera_utils/constants.py`)

- New `DataProductIdentifier.aux_fmatch_imager_flash` (`FMATCH-IMAGER-FLASH`, `DataLevel.AUX`).
- New `ProcessingStepIdentifier.aux_fmatch_imager_flash`.

## Notes

- Reader sets per mode are declared in `FMATCH_MODE_READERS` (`readers/registry.py`), not in the yml.
- Changelog entry for the FMATCH/SCENE-ID product definitions added under 5.11.0.

## Tests

- `tests/unit/test_constants.py` extended for the new identifier.
