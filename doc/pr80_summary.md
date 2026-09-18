# PR #80 — FMATCH 0: Scene ID

`fmatch-scene-id` → `main` · Version 5.11.0 · +2904 / −805 across 28 files

## Summary

Adds the **SCENE-ID-CAM** algorithm family and moves scene identification into its own package. Introduces two camera-based runners on distinct timescales, viewing-geometry classification, and an `unfiltering` scene definition.

## Changes

| Area | Change |
|------|--------|
| **Package move** ⚠️ | `scene_id` / `scene_definitions` moved to `libera_utils.scene_identification` package (**BREAKING** import change) |
| **New runners** | `SCENE-ID-CAM` (radiometer timescale, on `RADIOMETER_TIME`) and `SCENE-ID-CAM-CAMTIME` (camera timescale) — each with product def, Dockerfile, and reader |
| **Readers** | New `FootprintData.from_fmatch_cam` / `from_fmatch_cam_camtime`; shared manifest/dropbox plumbing via `_runner.py` |
| **Camtime axis** | CAM-CAMTIME stores one image subsection per record on `FOOTPRINT`; passes FMATCH footprint identifiers straight through (camera time, pixel-index ranges, PSF bbox, boresight geolocation) |
| **Coordinates** | `CAMERA_TIME` is a non-unique coordinate on `FOOTPRINT` (subsections share an image's time); `camera_pixel_x`/`y` are inclusive `(min,max)` over size-2 `CAMERA_PIXEL_BOUNDS` |
| **Scene defs** | Add `unfiltering`; defaults now TRMM + ERBE + unfiltering. New `standard_scene_definitions` / `default_scene_definitions`; `identify_scenes()` with no args uses defaults |
| **Viewing geometry** | Adds `solar_zenith_angle`, `viewing_zenith_angle`, `relative_azimuth_angle` as classification vars with `scene_bin_{type}_{angle}_min/max` bounds (bins span full range — placeholder, so scene IDs unchanged) |
| **Time products** | `FootprintData.to_time_product()` (+ `to_radiometer_time_product()` wrapper) promote observation-time var to coordinate and finalize dataset for writing |

## Notes

- Content matches prior review; this iteration adds **efficiency upgrades** (more compact dtypes/fill values, early typing in scene-id processing).
- Viewing-geometry bins are placeholders (full physical range) — no effect on scene IDs yet.

## Tests

- New integration tests for CAM-family runners (manifest-in/out + strict product write): `tests/integration/test_scene_id_runner.py`
- Expanded unit tests for scene definitions and scene ID; regenerated oracle `.nc` fixtures
