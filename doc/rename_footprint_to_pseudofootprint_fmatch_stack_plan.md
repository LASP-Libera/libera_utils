# Rename camtime axis `FOOTPRINT` → `PSEUDOFOOTPRINT` across the FMATCH stack

Companion to [rename_footprint_to_pseudofootprint_plan.md](rename_footprint_to_pseudofootprint_plan.md),
which covers the SCENE-ID side already applied on `fmatch-scene-id`. This plan covers the FMATCH
product side, which lives on the stacked branches `fmatch-1 … fmatch-9`.

## Stack topology (verified)

`fmatch-scene-id` is an **ancestor of the whole `fmatch-1 … fmatch-9` chain** (confirmed via
`git merge-base --is-ancestor`). The chain is (bottom → top):

```
main → fmatch-scene-id → fmatch-1-geometry-psf → fmatch-2-readers-gridded → … → fmatch-9-runners
                ▲
        (rename already applied here)
```

**Consequence:** the rename already committed on `fmatch-scene-id` — the shared dimension entry in
`libera_utils/data/libera_dimensions.yml` plus all the SCENE-ID files
(`scene_id_cam_camtime.yml`, `scene_identification/*.py`, `tests/**/test_scene_*.py`) —
**propagates up the stack automatically when each branch is rebased onto the updated
`fmatch-scene-id`.** Do **NOT** re-rename those files on the upper branches; doing so risks merge
conflicts or (worse) re-introducing `FOOTPRINT` after the base already renamed it.

> Local-branch caveat: `git merge-base` currently reports the `fmatch-4 → 5 → 6` links as
> "not linear" — the local pointers have drifted from a clean rebased chain (there are `backup/*`
> branches from in-progress restacking). Before applying anything below, restack the branches so
> each sits cleanly on its parent (and ultimately on the updated `fmatch-scene-id`). The renames
> below assume a clean chain.

## What still needs renaming (FMATCH-only files)

These files do **not** exist on `fmatch-scene-id`, so they are not covered by the base rename.
Apply the rename **on the branch that first introduces each file**; it then flows upward to every
branch above via the normal rebase of the stack. Counts are bare `FOOTPRINT` tokens (excluding the
unrelated `CAMERA_PIXEL_*` axis), measured at the stack tip `fmatch-9-runners`.

| File | Introduced on | Count | Load-bearing? |
| --- | --- | --- | --- |
| `tests/test_data/footprint_matching/fixtures.py` | `fmatch-2-readers-gridded` | 4 | **Yes** (1 exec, 3 comment) |
| `libera_utils/data/product_definitions/fmatch_cam_camtime.yml` | `fmatch-6-product-definitions` | 66 | declarative (drives output) |
| `libera_utils/data/product_definitions/fmatch_imager_camtime.yml` | `fmatch-6-product-definitions` | 110 | declarative (drives output) |
| `libera_utils/footprint_matching/product.py` | `fmatch-7-product-assembly` | 11 | **Yes** (2 exec, 9 doc) |
| `tests/unit/test_footprint_matching/test_camera_segmentation.py` | `fmatch-8-camera-segmentation` | 4 | **Yes** (2 exec, 2 comment) |
| `tests/unit/test_footprint_matching/test_product.py` | `fmatch-9-runners` | 18 | **Yes** (~9 exec, rest doc) |

### Critical difference from the SCENE-ID side

On the SCENE-ID side the runner materialized the coordinate generically (name read from the YAML),
so only the YAML was load-bearing. **On the FMATCH side, `product.py` hardcodes the string
`"FOOTPRINT"` in executable code**, so the YAML rename and the code rename MUST land together
(they are on different branches — `fmatch-6` and `fmatch-7` — so once the YAML is renamed on
`fmatch-6`, `product.py` on `fmatch-7` will `KeyError` on
`definition.coordinates["FOOTPRINT"]` until it too is renamed). Rebase-and-test the two together.

## Per-branch changes

For every file, the mechanical transform is the same whole-word rename used on the base:
`FOOTPRINT` → `PSEUDOFOOTPRINT`. It is safe because the token is uppercase and distinct — it does
not collide with the lowercase prose "pseudo-footprint", the `camera_pixel_*` variables, or Python
identifiers like `n_footprints_per_image` / `footprint_dtype` (which stay as-is; they are not the
axis identifier).

### `fmatch-2-readers-gridded` — `tests/test_data/footprint_matching/fixtures.py`

- **Executable** (line ~1025): the `dimension_sizes` dict key
  `"FOOTPRINT": n_footprints_per_image` → `"PSEUDOFOOTPRINT": n_footprints_per_image`. This dict
  drives the synthetic dataset's dimension sizes, so it must match the renamed coordinate.
- Comments (lines ~1015-1019): update `(CAMERA_TIME, FOOTPRINT)` prose.

### `fmatch-6-product-definitions` — the two FMATCH product definitions

Both `fmatch_cam_camtime.yml` and `fmatch_imager_camtime.yml` use the identical structure to
`scene_id_cam_camtime.yml`:

- The `FOOTPRINT:` coordinate declaration and its `dimensions: ["FOOTPRINT"]` → `PSEUDOFOOTPRINT`.
- Every variable's `dimensions: ["CAMERA_TIME", "FOOTPRINT"]` → `["CAMERA_TIME", "PSEUDOFOOTPRINT"]`.
- The `(CAMERA_TIME, FOOTPRINT)` prose in the header comments.
- Leave the descriptive lowercase "pseudo-footprint (image subsection)" `long_name` text intact.

### `fmatch-7-product-assembly` — `libera_utils/footprint_matching/product.py`

- **Executable** (line ~1285): `definition.coordinates["FOOTPRINT"]`
  → `definition.coordinates["PSEUDOFOOTPRINT"]`.
- **Executable** (line ~1288): the data-dict key `"FOOTPRINT": np.arange(n_footprints_per_image, …)`
  → `"PSEUDOFOOTPRINT": np.arange(n_footprints_per_image, …)`.
- Docstrings/comments (lines ~868, 873, 1071, 1104, 1236, 1240, 1257, 1283, 1362): update the
  `(CAMERA_TIME, FOOTPRINT)` and "padded along FOOTPRINT" references.
- Local variable names (`footprint_dtype`, `n_footprints_per_image`, `footprints`) are semantic and
  stay unchanged.

### `fmatch-8-camera-segmentation` — `tests/unit/test_footprint_matching/test_camera_segmentation.py`

- **Executable** (lines ~355-356): `dataset["latitude"].dims == ("CAMERA_TIME", "FOOTPRINT")` and
  `dataset.sizes["FOOTPRINT"]` → `PSEUDOFOOTPRINT`.
- Comments (lines ~353-354): prose update.

### `fmatch-9-runners` — `tests/unit/test_footprint_matching/test_product.py`

- **Executable** assertions/setup (lines ~231, 347, 481, 514, 537-541, 576) that reference the
  literal `"FOOTPRINT"` string / `("CAMERA_TIME", "FOOTPRINT")` dims / `dimension_sizes` dict key →
  `PSEUDOFOOTPRINT`. These assert the written product's coordinate name and thus double as the
  end-to-end regression check that the YAML+`product.py` rename is wired through.
- Docstrings/comments (lines ~190, 229, 345, 495, 504, 547): prose update.

## Suggested execution order

1. Restack `fmatch-1 … fmatch-9` cleanly on the already-renamed `fmatch-scene-id` (resolve the
   `fmatch-4/5/6` drift first). This alone pulls the base rename into every branch.
2. `fmatch-2`: rename in `fixtures.py`, run
   `pytest tests/unit/test_footprint_matching -m "not integration"` (as far as those tests exist at
   this branch), then rebase 3-9 forward.
3. `fmatch-6`: rename both product-definition YAMLs, rebase 7-9 forward.
4. `fmatch-7`: rename `product.py` (the two executable lines + docstrings). Now the YAML + code
   agree; run the footprint_matching product/assembly tests. Rebase 8-9 forward.
5. `fmatch-8`: rename `test_camera_segmentation.py`; run its module.
6. `fmatch-9`: rename `test_product.py`; run the full footprint_matching suite + the SCENE-ID
   integration test (which already asserts `PSEUDOFOOTPRINT` from the base) to confirm both
   products agree bit-for-bit on the axis name.

## Verification (per branch after its rename)

- `ruff check` + `ruff format --check` on the changed files.
- The branch's own `pytest tests/unit/test_footprint_matching/…` and, at `fmatch-9`, the full
  `pytest -m "not integration"` plus `pytest tests/integration/test_scene_id_runner.py`.
- Sanity grep on the tip once the whole stack is done:
  `git grep -wI FOOTPRINT fmatch-9-runners -- '*.py' '*.yml'` should return only
  `CAMERA_PIXEL_*`-adjacent lines — zero bare `FOOTPRINT` tokens.

## Risk

Low-to-moderate. The rename itself is mechanical and self-consistent. The moderate part is purely
stack mechanics: the YAML (fmatch-6) and its consuming code (fmatch-7) are on different branches,
so an incomplete rebase can leave a transient `KeyError` state between them — always rebase and
test the pair together, and never re-rename the base-owned files on an upper branch.
