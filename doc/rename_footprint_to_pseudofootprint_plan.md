# Rename camtime coordinate `FOOTPRINT` → `PSEUDOFOOTPRINT`

## Goal

The image-subsection coordinate on the camera-timescale (camtime) data products is currently
named `FOOTPRINT`. It is not a true instrument footprint — it is a *pseudo*-footprint (an image
subsection index that may overlap and does not tile the image). Rename the coordinate **and its
underlying dimension** from `FOOTPRINT` to `PSEUDOFOOTPRINT` everywhere it appears, so the name
reflects the physics. The grid becomes `(CAMERA_TIME, PSEUDOFOOTPRINT)`.

This is a pure rename of an identifier token — no data model, dtype, shape, or generation logic
changes.

## Scope / key finding

`FOOTPRINT` is used **only** by the camtime product on this branch; the radiometer-timescale
products use `RADIOMETER_TIME` and never reference it (`scene_id_cam.yml` has zero occurrences).
So renaming the shared dimension entry is safe.

Critically, **no runtime library code hardcodes the string `"FOOTPRINT"`** as a dimension name.
The coordinate is materialized generically in
[_runner.py:288-297](libera_utils/scene_identification/_runner.py#L288-L297), which derives
coordinate names from the product-definition YAML (`np.arange(sizes[name])` keyed by the declared
coordinate). Renaming the YAML declaration is therefore sufficient to change actual output; every
other Python occurrence is a docstring/comment, and the tests build synthetic grids with the dim
name spelled out literally.

## Files to change (this branch)

Occurrence counts of the bare token `FOOTPRINT` (excluding `CAMERA_PIXEL_*`, which is unrelated):

| File | Count | Nature |
| --- | --- | --- |
| [libera_dimensions.yml](libera_utils/data/libera_dimensions.yml) | 2 | dimension registry entry + prose |
| [scene_id_cam_camtime.yml](libera_utils/data/product_definitions/scene_id_cam_camtime.yml) | 49 | coordinate def + every variable's `dimensions:` + prose |
| [scene_id.py](libera_utils/scene_identification/scene_id.py) | 9 | docstrings/comments only |
| [scene_definitions.py](libera_utils/scene_identification/scene_definitions.py) | 1 | comment only |
| [cam_camtime/scene_id_cam_camtime.py](libera_utils/scene_identification/cam_camtime/scene_id_cam_camtime.py) | 2 | module docstring only |
| [tests/unit/test_scene_id.py](tests/unit/test_scene_id.py) | 2 | `grid_dims` literal + docstring |
| [tests/unit/test_scene_definitions.py](tests/unit/test_scene_definitions.py) | 4 | `grid_dims` literal + docstrings |
| [tests/integration/test_scene_id_runner.py](tests/integration/test_scene_id_runner.py) | 17 | `grid_dims` literal + assertions on `"FOOTPRINT"` |

### 1. `libera_utils/data/libera_dimensions.yml`

- Rename the `FOOTPRINT:` dimension key (line 23) to `PSEUDOFOOTPRINT:`.
- Update the `long_name` prose (lines 24-27) so the composite reads
  `(CAMERA_TIME, PSEUDOFOOTPRINT)`. Keep the "Pseudo-footprint (image subsection)" wording — it
  is already correct.

### 2. `libera_utils/data/product_definitions/scene_id_cam_camtime.yml`

This is the bulk of the change and must stay internally consistent:

- **Coordinate declaration** (lines 17-23): rename the `FOOTPRINT:` coordinate key to
  `PSEUDOFOOTPRINT:` and its `dimensions: ["FOOTPRINT"]` → `dimensions: ["PSEUDOFOOTPRINT"]`.
- **Every variable's grid**: replace all `dimensions: ["CAMERA_TIME", "FOOTPRINT"]`
  → `dimensions: ["CAMERA_TIME", "PSEUDOFOOTPRINT"]` (~45 occurrences).
- **Prose/comments** (lines 10-16, 22-23, 39, 240): update the `(CAMERA_TIME, FOOTPRINT)`
  composite references and "every FOOTPRINT of that image" phrasing to `PSEUDOFOOTPRINT`.
- Leave `long_name` descriptive text ("Pseudo-footprint (image subsection) index...") intact —
  only the axis identifier changes.

### 3. `libera_utils/scene_identification/*.py` (docstrings/comments only)

Update the `(CAMERA_TIME, FOOTPRINT)` / bare `FOOTPRINT` references in the docstrings and
comments of `scene_id.py` (9), `scene_definitions.py` (1), and
`cam_camtime/scene_id_cam_camtime.py` (2) to `PSEUDOFOOTPRINT`. No executable statement in these
files depends on the token, so this is documentation-only but should be done for consistency.

### 4. Tests

- `tests/unit/test_scene_id.py`, `tests/unit/test_scene_definitions.py`: update the
  `grid_dims = ("CAMERA_TIME", "FOOTPRINT")` literals and surrounding docstrings.
- `tests/integration/test_scene_id_runner.py`: update the `grid_dims` literal, the
  `("CAMERA_TIME", "FOOTPRINT")` build helpers, **and** the runtime assertions that read the
  written product (`"FOOTPRINT" in reopened.sizes`, `reopened["FOOTPRINT"]`,
  `reopened[name].dims == ("CAMERA_TIME", "FOOTPRINT")`, etc., lines 205-217). These assertions
  will genuinely fail if the YAML rename is applied without them, so they double as the
  regression check that the rename is wired end-to-end.

## Out of scope / follow-ups

- **FMATCH-CAM-CAMTIME product** (`fmatch_cam_camtime.yml` and its reader
  `FootprintData.from_fmatch_cam_camtime`): not present on this branch (the reader is a
  `NotImplementedError` stub at [scene_id.py:804](libera_utils/scene_identification/scene_id.py#L804)).
  On the stacked FMATCH branches this product carries the *identical* axis and passes the
  identifier variables through to SCENE-ID-CAM-CAMTIME, so the same rename must be applied there
  when those branches land — the two products must agree bit-for-bit on the axis name. Flag this
  in the PR so the stacked branches are updated in lockstep and don't reintroduce `FOOTPRINT`.
- Doc/notebook prose (`doc/source/changelog.md`, PR-summary markdown under `doc/`): optional
  cosmetic cleanup; not required for correctness.

## Verification

1. `ruff check` and `ruff format` (pre-commit) — no rule impact expected.
2. `pytest -m "not integration" tests/unit/test_scene_id.py tests/unit/test_scene_definitions.py`
3. `pytest tests/integration/test_scene_id_runner.py` — confirms the coordinate is written as
   `PSEUDOFOOTPRINT` and conformance-checks against the renamed product definition.
4. Sanity grep: `grep -rwI FOOTPRINT libera_utils/ tests/` should return **only**
   `CAMERA_PIXEL_*`-adjacent lines (none left as a bare `FOOTPRINT` token).

## Risk

Low. Single self-consistent rename; the generic runner means the YAML declaration drives real
output, and the integration test asserts the written axis name, so a partial rename fails loudly
rather than silently producing a mismatched product.
