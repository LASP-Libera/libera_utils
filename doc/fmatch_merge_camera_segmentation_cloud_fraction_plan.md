# Plan: Consolidate camera-segmentation into cloud-fraction; FMATCH consumes CF-CAM-CAMTIME

> **STATUS: code IMPLEMENTED + tested (working branch `wip-camseg-cf-refactor`, off `fmatch-10`).**
> Items A–G done; unit suite green (1375 passed; the only failures are 3 pre-existing environmental
> `mkspk`/kernel-manager tests unrelated to this work); `ruff check` + `ruff format` clean.
> The PR split (combined 8' + rebased runners 9') is NOT yet done — it rewrites the existing named
> stack, so it awaits explicit go-ahead. Key design refinements discovered during implementation:
>
> 1. **IMAGER-CAMTIME does NOT read CF-CAM-CAMTIME** (per user). Only FMATCH-CAM-CAMTIME does.
>    IMAGER-CAMTIME carries no cloud fraction, so it runs `segment_l1b_camera` itself, in its own
>    module (`fmatch_imager_camtime.py`), on L1B-CAM. So each camera runner obtains footprints
>    differently.
> 2. **Injected `camera_footprint_loader`** on `FmatchRunnerConfig` is how that stays clean: the
>    shared `_runner.py` never imports `camera_segmentation`; each camera runner supplies its own
>    loader — `fmatch_cam_camtime.py` reconstructs from the CF product, `fmatch_imager_camtime.py`
>    segments L1B. Verified: `_runner.py` is free of any `camera_segmentation` reference.
> 3. **`BoundingBox` was already shared** in `types.py`; only `PseudoFootprint` +
>    `CameraFootprintQualityFlag` moved there.
> 4. Serialize/deserialize pair lives in `product.py`: `build_camtime_pseudofootprint_columns`
>    (write) + `pseudofootprints_from_camtime_dataset` (read) + `camtime_real_cell_mask` (shared
>    padded-cell selection). `psf_bbox_flags` int packs wraps_dateline/is_polar/truncated (Trap A);
>    `q_flags` stores base segmentation bits (Trap B).
> 5. Field name renamed `l1b_input_product_id` -> `input_product_id` (FMATCH config only; the
>    cloud-fraction algorithm's own `CfRunnerConfig` keeps its L1B name).


## Goal & decision

Put the `camera_segmentation` introduction **and** its only real consumer (the cloud-fraction
placeholder) in a single PR, ahead of the FMATCH L1B runners. The enabling architectural
decision (chosen 2026-09-17):

> **Segmentation occurs solely in the cloud-fraction algorithm.** `cloud_fraction` segments
> the L1B camera image *once* and writes complete pseudo-footprints (geometry **and** cloud
> fraction) into the `CF-CAM-CAMTIME` product. `FMATCH-CAM-CAMTIME` **reads them back** from
> that product and never calls `segment_l1b_camera`. FMATCH no longer imports
> `camera_segmentation`.

This is a real refactor, not a branch move. The branch/PR grouping falls out of it naturally.

## Current state (verified at `fmatch-10-cloud-fraction`, tip `039db4b`)

Stack: `7 product-assembly → 8 camera-seg (a8f16b7) → 9 runners (4314f0b) → 10 cloud-fraction (039db4b)`.

- FMATCH camtime runner `run_footprint_matching` ([_runner.py:654-659](libera_utils/footprint_matching/_runner.py#L654))
  calls `segment_l1b_camera(load_l1b_camera_dataset(l1b))` to build pseudo-footprints, and
  separately reads `cloud_fraction` from `CF-CAM-CAMTIME` ([_runner.py:326](libera_utils/footprint_matching/_runner.py#L326)).
  `fmatch_cam_camtime.py` declares both `l1b_input_product_id=l1b_cam` and
  `cloud_fraction_product_id=l2_cf_cam_camtime`.
- `PseudoFootprint`, `CameraFootprintQualityFlag`, `BoundingBox` live in `camera_segmentation.py`
  ([camera_segmentation.py:118-182](libera_utils/footprint_matching/camera_segmentation.py#L118)).
  `PseudoFootprint` carries: `time, slice_x, slice_y, center_ix, center_iy, latitude, longitude,
  altitude, solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle, bbox, q_flags`.
- `CF-CAM-CAMTIME` (`cf_cam_camtime.yml`) stores only `camera_pixel_{x,y}_{min,max}`,
  `cloud_fraction`, `cloud_fraction_standard_deviation` on the `(CAMERA_TIME, PSEUDOFOOTPRINT)`
  grid — **not** the geolocation/viewing geometry FMATCH needs.
- `CamtimeGrid` / `build_camtime_grid` / `build_camtime_provenance` in `product.py` were **added
  by the cloud-fraction commit** (039db4b) as a refactor of FMATCH's own camtime assembly; both
  FMATCH camtime and CF camtime use them.
- `DataProductIdentifier` members (`l2_cf_cam_camtime`, `aux_fmatch_cam_camtime`, …) already exist
  in `constants.py` — untouched by the cloud-fraction commit.
- No `cli.py` / root-`Dockerfile` / `product.py` **conflicts** between runners and cloud-fraction:
  runners touch only `footprint_matching/Dockerfile`; cloud-fraction owns the `cli.py` handlers,
  root `Dockerfile` CF stages, and the `product.py` camtime-grid additions.

## Target architecture

```
fmatch-7-product-assembly
   └─ fmatch-8  (COMBINED: camera_segmentation + cloud_fraction + shared modules + expanded CF product)
        └─ fmatch-9-runners  (rebased: reads CF-CAM-CAMTIME; no camera_segmentation import)
```

- **`camera_segmentation.py`** (`segment_l1b_camera`) — imported **only** by `cloud_fraction`.
- **New shared type module** `footprint_matching/pseudofootprint.py` — holds `PseudoFootprint`,
  `CameraFootprintQualityFlag`, `BoundingBox`. Imported by `cloud_fraction`, `product.py`, and
  FMATCH runners. Dependency-light (dataclasses + numpy only).
- **New shared runner-helpers module** `footprint_matching/_runner_common.py` — holds the six
  symbols cloud-fraction currently pulls from `footprint_matching._runner`
  (`FMATCH_RADIOMETER_TIME_COORDINATE`, `_as_local_path`, `algorithm_version`,
  `load_l1b_camera_dataset`, `load_l1b_radiometer_inputs`, `select_manifest_files_by_product_id`).
  Both `cloud_fraction._runner` and FMATCH `_runner` import from here. (Still required: cloud-fraction
  sits below runners, so it cannot import from runners' `_runner.py`.)
- **`CF-CAM-CAMTIME`** — expanded to carry the full pseudo-footprint geometry.
- **FMATCH camtime runner** — reconstructs `list[PseudoFootprint]` from `CF-CAM-CAMTIME` instead of
  segmenting; drops `l1b_input_product_id` for the camtime modes (input becomes the CF product).

`fmatch-10-cloud-fraction` is retired; runners become the top of the stack.

## Work items

### A. Relocate the shared `PseudoFootprint` type (in combined PR)

> **Audit correction:** `BoundingBox` is **already** shared in `types.py:99` — do NOT move it.
> Only `PseudoFootprint` + `CameraFootprintQualityFlag` need relocating.

1. Move `PseudoFootprint` + `CameraFootprintQualityFlag` from `camera_segmentation.py` into a
   dependency-light home. Simplest: put them in the existing `footprint_matching/types.py` next to
   `BoundingBox` (which `PseudoFootprint` references anyway), avoiding a new module.
2. `camera_segmentation.py` imports them from `types.py` (keeps `segment_l1b_camera` and the
   block-detection internals).
3. Repoint every importer: `cloud_fraction/*`, `product.py` (the `TYPE_CHECKING` import and
   `build_camtime_grid` signature), FMATCH `_runner.py`/entrypoints, `types.py`, `tiling.py`,
   `test_camera_segmentation.py`. (fmatch-7's `product.py` reference is `TYPE_CHECKING`-only, so
   introducing the module in the combined PR is import-safe.)

### B. Extract the six shared runner helpers (in combined PR)

4. Create `libera_utils/footprint_matching/_runner_common.py`; move the six symbols out of runners'
   `_runner.py` (they currently originate in commit 4314f0b). Bring only the imports/constants they
   need. NumPy-style module docstring.
5. `cloud_fraction/_runner.py`: import the six from `_runner_common`.
6. Runners' `_runner.py` (rebased in item J): import the still-needed helpers from `_runner_common`
   (`load_l1b_radiometer_inputs`, `_as_local_path`, `select_manifest_files_by_product_id`,
   `FMATCH_RADIOMETER_TIME_COORDINATE`, `algorithm_version`) — `load_l1b_camera_dataset` is no longer
   used by FMATCH after item H.

### C. Expand the `CF-CAM-CAMTIME` product definition (in combined PR)

> **DONE — see [doc/fmatch_cf_cam_camtime_field_audit.md](doc/fmatch_cf_cam_camtime_field_audit.md).**
> Result: all 13 `PseudoFootprint` fields are consumed; store all of them. Two fidelity traps.

7. Add to `cf_cam_camtime.yml` on the `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid, mirroring the
   identically-named FMATCH-CAM-CAMTIME output dtypes/units (`fmatch_cam_camtime.yml`):
   `latitude`, `longitude`, `altitude`, `solar_zenith_angle`, `viewing_zenith_angle`,
   `relative_azimuth_angle`, `center_pixel_x`, `center_pixel_y`, `q_flags`, and the footprint bbox.
   (`CAMERA_TIME`, `PSEUDOFOOTPRINT`, `camera_pixel_{x,y}_{min,max}`, `cloud_fraction*` already
   present.) These carry **real** segmentation values — do **not** tag them PLACEHOLDER.
8. **Trap A (bbox longitude is lossy):** the FMATCH output normalizes `psf_bbox_lon_*` to
   `[-180,180)`, but aggregation feeds the raw `bbox` (with `wraps_dateline`/`is_polar`) to the
   TileManager. Store bbox lon **raw / un-normalized** plus `wraps_dateline` + `is_polar`
   (+ `truncated`) so the box reconstructs exactly. Do NOT reuse the normalized output convention.
9. **Trap B (`q_flags` is derived):** store the **base** `CameraFootprintQualityFlag` (0–3), not
   the FMATCH output `q_flags` (which OR-s coverage bits incl. `LIMB_TRUNCATED` from
   `bbox.truncated`). Carry `bbox.truncated` separately; FMATCH re-derives coverage flags itself.

### D. Write the full geometry into CF-CAM-CAMTIME (in combined PR)

9. Extend `generate_placeholder_cloud_fraction_camtime(footprints)`
   ([cloud_fraction.py:108](libera_utils/cloud_fraction/cloud_fraction.py#L108)) to emit the new
   geometry arrays from the `footprints` it already receives (it has full `PseudoFootprint`s), via
   the existing `build_camtime_grid` gridding so provenance stays aligned.

### E. Add an FMATCH reader for CF-CAM-CAMTIME (in runners PR)

10. Add `read_cf_cam_camtime_pseudofootprints(path) -> list[PseudoFootprint]` (FMATCH side): open the
    CF product, reconstruct `PseudoFootprint`s (geometry + provenance) from the stored variables,
    reusing `_as_local_path` for S3. Cloud-fraction values are read as before.

### F. Rewire the FMATCH camtime runner (in runners PR)

11. In `run_footprint_matching`, replace the camera-timescale branch: instead of
    `segment_l1b_camera(load_l1b_camera_dataset(...))`, read the pseudo-footprints (and cloud
    fraction) from `CF-CAM-CAMTIME`. Remove the `camera_segmentation` import from `_runner.py`.
12. Update `fmatch_cam_camtime.py` / `fmatch_imager_camtime.py`: drop `l1b_input_product_id=l1b_cam`
    (camtime input is now the CF product via `cloud_fraction_product_id`), update docstrings that
    reference `segment_l1b_camera`.
13. Confirm the radiometer-timescale path (`FMATCH-CAM`, `FMATCH-IMAGER`, `-FLASH`) is unchanged —
    it still uses `load_l1b_radiometer_inputs`.
14. Update `footprint_matching/Dockerfile` comment/extras note that claimed the runners import
    `camera_segmentation` (line ~83).

### G. Notebooks / tests

15. Notebook `08_camera_segmentation.ipynb` — both a8f16b7 (create) and 039db4b (extend) edit it;
    they collapse into one in the combined PR. Re-execute end-to-end.
16. Move/keep tests with their code: `test_camera_segmentation.py`, `test_cloud_fraction/*`,
    `test_cf_runner.py`, CF `test_cli.py` → combined PR. Add a **round-trip test**: footprints →
    CF-CAM-CAMTIME → `read_cf_cam_camtime_pseudofootprints` reproduces every field FMATCH uses.
    `test_product.py` / `test_l1b_inputs.py` → runners PR (updated for the new reader path).
17. Add an FMATCH-camtime runner test asserting output is byte-for-byte/value-identical to the
    pre-refactor result for the same input day (segmentation moved, not changed).

### H–J. Branch / stack reorg

18. Back up tips: `git branch backup/fmatch-8-premerge fmatch-8-camera-segmentation` and
    `backup/fmatch-10-premerge fmatch-10-cloud-fraction`.
19. Build combined branch from `fmatch-8`: apply items A–D and G (cherry-pick 039db4b's
    cloud-fraction content, then layer the type relocation, helper extraction, product expansion,
    geometry write). Title: "FMATCH camera segmentation + CF-CAM cloud fraction."
20. Rebase `fmatch-9-runners` onto the combined branch and apply items B(6), E, F, G(16-17).
21. Re-parent anything based on `fmatch-10` onto the new runners tip; delete `fmatch-10` (keep backup).

## Verification

- **Import boundary:** `git grep camera_segmentation libera_utils/footprint_matching/_runner.py
  libera_utils/footprint_matching/fmatch_*.py` returns nothing at the runners tip (docstrings
  included). `python -c "import libera_utils.footprint_matching._runner"` works without importing
  `camera_segmentation`.
- **Round-trip fidelity (the crux):** the new round-trip test proves CF-CAM-CAMTIME stores every
  `PseudoFootprint` field FMATCH assembly reads. The camtime runner regression test proves the
  FMATCH-CAM-CAMTIME output is unchanged by the source swap.
- **Per-tip imports:** combined branch imports `camera_segmentation` + `cloud_fraction` without
  `footprint_matching._runner`; runners tip resolves the six helpers via `_runner_common`.
- `pytest -m "not integration" tests/`, `ruff check`, `ruff format --check`, pre-commit — clean on
  both tips. Watch F401 after both extractions.
- **No-drop check:** `git diff backup/fmatch-10-premerge <new runners tip>` shows only the intended
  refactor deltas — no lost `cloud_fraction` content.

## Risks / open items

- **Field-completeness audit — DONE** ([doc/fmatch_cf_cam_camtime_field_audit.md](doc/fmatch_cf_cam_camtime_field_audit.md)).
  All 13 `PseudoFootprint` fields are consumed, so all must be stored. Two silent-corruption traps
  found: (A) bbox longitude is lossy under output normalization — store raw + dateline/polar flags;
  (B) `q_flags` must be the base segmentation value, not the coverage-augmented output. Round-trip
  test must cover a dateline-wrapping bbox, a polar bbox, and a `truncated`+nonzero-`q_flags` case.
- **Placeholder semantics:** cloud-fraction is a placeholder; the *geometry* it writes is real
  (from segmentation), only the cloud-fraction values are random. Ensure the geometry variables are
  not marked PLACEHOLDER in `cf_cam_camtime.yml`.
- **`algorithm_version`:** CF reuses FMATCH's `algorithm_version()`; preserved via `_runner_common`.
  Distinct CF versioning is out of scope.
- **Product growth:** CF-CAM-CAMTIME roughly doubles in variable count; confirm no downstream
  consumer assumes the old minimal schema.
- **Video-mode `CAMERA_TIME` uniqueness** (`TODO[LIBSDC-847]`) is pre-existing and unaffected.
