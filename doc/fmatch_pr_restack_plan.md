# FMATCH / Scene-ID PR Re-stack Plan

> **UPDATE (2026-09-11, later session)**: the demo notebooks are being relocated from
> `doc/notebooks/` to `notebooks/` (per request), with external-documentation links added
> to every notebook and a product-scope chart added to notebook 06. Each branch gets one
> additional commit moving/patching **its own** notebook (no history rewriting — inherited
> notebooks on later branches catch up through the normal stacked-PR base-update flow as
> PRs merge). Branches fmatch-1/2/3 are done; run `sh doc/notebook_move/run_remaining.sh`
> to finish fmatch-4 → scene-id-imager-family (the permission classifier blocked agent-run
> `git commit` mid-way, so the remaining commits are user-run). Pre-change branch tips are
> recorded in the session scratchpad (`pre_notebook_move_shas.txt`) and below:
> fmatch-4 7a00968 · fmatch-5 079fe51 · fmatch-6 139a4cd · fmatch-7 8db8df2 ·
> fmatch-8 40b2f2b · fmatch-9 b8390ac · scene-id-imager-family 1a49f49 ·
> fmatch-1 379c439 · fmatch-2 0e974de · fmatch-3 a2259c9.

> **STATUS: IMPLEMENTED (2026-09-11).** All ten branches are built locally, one commit
> each, with executed demo notebooks in `doc/notebooks/`. Deviations from the plan
> below, discovered during the build:
>
> 1. **Camera segmentation moved from position 2 to position 8** (after product
>    assembly): `test_camera_segmentation.py` imports camtime helpers from
>    `product.py`. `product.py` only needs `camera_segmentation` under
>    `TYPE_CHECKING`, so the reorder is import-clean.
> 2. **The `fmatch` extras + CI `libhdf4`/`--all-extras` change moved to PR 1**:
>    the final `geometry.py` imports `pyproj` at module level.
> 3. **`_swath.py` ships with the gridded readers (PR 2)**, not the swath PR:
>    `igbp`/`nsidc` use its `rasterize_points_to_grid`.
> 4. **PR 2 carries an intermediate `readers/__init__.py`** registering the six
>    readers present; PR 3 replaces it with the final nine-reader version
>    (`test_registry`, which locks the full listing, lands in PR 3).
> 5. **`test_product.py` lands in PR 9** (its camtime helpers import `_runner`'s
>    L1B loaders) and **`tests/integration/test_fmatch_runner.py` lands in PR 10**
>    (it imports the relocated Scene-ID modules).
> 6. Built order: 1 geometry-psf, 2 readers-gridded, 3 readers-swath, 4 tiling,
>    5 weighting-aggregation, 6 product-definitions, 7 product-assembly,
>    8 camera-segmentation, 9 runners, 10 scene-id-imager-family.
>
> Final sanity check passed: `git diff origin/fmatch-efficiency scene-id-imager-family`
> is exactly the ten notebooks. Known non-blocking failures: 3 kernel-manager tests
> fail in this devcontainer (missing NAIF `mkspk` binary, environmental, fail on every
> branch including main-era ones); 1 scene-id-runner test fails on the base branch
> #34 (stale manifest API usage) through PR 9 and is fixed by PR 10's final-state
> tests.

## Problem

The current stack is too large to review effectively, and the efficiency work arrives as a
separate final PR, so every earlier PR shows a superseded implementation:

| PR | Branch | Base | Size |
| --- | --- | --- | --- |
| #34 | `LIBSDC-673-scene-id-algorithm` | `main` | ~2.9k / −0.8k |
| #54 | `LIBSDC-794-fmatch-runners-modules` | #34 | **~20.7k** / −0.3k |
| #57 | `LIBSDC-807-tile-manager` | #54 | ~5.0k / −0.1k |
| #61 | `fmatch-efficiency` | #57 | ~2.4k / −0.2k |

Goals of the re-stack:

1. **Topic-focused PRs** a subject-matter specialist can review end-to-end (e.g. tiling in
   its own PR).
2. **Efficiency folded in**: each topic PR shows the *final* (vectorized) state of its
   modules, not the initial implementation later rewritten by #61.
3. **Stacked PRs**: integration tests may fail mid-stack (features incomplete until the last
   PR merges) — acceptable. Unit tests must pass on every branch and must test the *end-state*
   behavior only — no tests of placeholders or intermediate scaffolding.
4. **A demo notebook per PR** so reviewers can run and see the topic's concept in action.

## Key insight: harvest final state, don't cherry-pick

`origin/fmatch-efficiency` (tip of the stack) already contains the final state of every file.
Instead of untangling ~25 interleaved commits across three branches, each new topic branch is
built by **checking out the final versions of that topic's files from the stack tip**:

```bash
git checkout <previous-stack-branch> -b <new-topic-branch>
git checkout origin/fmatch-efficiency -- <topic file paths> <topic test paths>
# add demo notebook, run ruff + targeted pytest, commit
```

This automatically satisfies goal 2 (efficiency included), goal 3 (tests are the final-state
tests), and avoids rebase conflicts entirely. Commit granularity of the original branches is
sacrificed, but PRs are squash-merged anyway and the originals are preserved
(`backup/fmatch-efficiency-f64c98d`, `backup/807-tile-manager-eeaba64`, plus the live remote
branches until the new stack merges).

## Module dependency graph (bottom-up, determines stack order)

```
types.py, psf.py                      (leaves)
geometry.py        → psf, types
camera_segmentation.py → geometry, psf, types
readers/*          → types
tiling.py          → readers.registry, types
weighting.py       → geometry, psf, types
aggregation.py     → weighting, types
product.py         → aggregation, geometry, readers, tiling, weighting, camera_segmentation
_runner.py         → product, readers, camera_segmentation
fmatch_<product>.py → _runner
scene_id imager    → consumes FMATCH products at runtime (no import-time dependency)
```

## The new stack

PR #34 (Scene ID CAM / CAM-CAMTIME → `main`) stays as-is and remains the base of the stack.
PRs #54, #57, and #61 are closed with a comment pointing at the replacement stack once it is up.

Each PR below lists: topic, target reviewer, contents (all files at their
`fmatch-efficiency` final state), and the demo notebook.

### PR 1 — Footprint geometry & PSF core

- **Branch**: `fmatch-1-geometry-psf` (base: `LIBSDC-673-scene-id-algorithm`)
- **Topic**: The Libera radiometer point spread function and footprint viewing geometry —
  how an L1B viewing vector becomes a ground footprint, angular projection, and lat/lon
  bounding boxes. Includes the vectorization work from #61: batched
  `compute_footprint_bounding_boxes` (~66×), shared `ViewingFrame`, closed-form ECEF
  intersection.
- **Reviewer profile**: instrument geometry / geolocation specialist.
- **Files**: `footprint_matching/__init__.py`, `types.py`, `psf.py`, `geometry.py`;
  tests: `test_psf.py`, `test_geometry.py`, `tests/unit/test_footprint_matching/__init__.py`.
- **Note**: `types.py` final state includes `GridTile`/`TileKey` (used later by tiling).
  That is intentional — end-state code, unit-tested when tiling lands.
- **Notebook**: PSF shape and half-angle; project a footprint from synthetic L1B geometry;
  draw footprint bounding boxes on a lat/lon map; before/after timing of the batched
  bounding-box computation.

### PR 2 — Camera segmentation

- **Branch**: `fmatch-2-camera-segmentation` (base: PR 1)
- **Topic**: Segmenting Libera camera L1B images into radiometer-footprint-sized
  pseudo-footprints (`segment_l1b_camera`, `PseudoFootprint`), with the #61 vectorized
  segmentation.
- **Reviewer profile**: camera / imager processing specialist.
- **Files**: `camera_segmentation.py`; tests: `test_camera_segmentation.py`.
- **Notebook**: segment a synthetic camera granule; visualize pixel-to-pseudo-footprint
  assignment and the resulting footprint grid overlaid on the camera image.

### PR 3 — External data readers: framework + gridded ancillary

- **Branch**: `fmatch-3-readers-gridded` (base: PR 2)
- **Topic**: The reader abstraction (`readers/base.py`, `registry.py`, HDF4/HDF5/pyproj IO
  helpers) and readers for *gridded* ancillary datasets: ERA5 single-level and pressure-level
  reanalysis, IGBP surface type, NSIDC NISE snow/ice (both hemispheres), MODIS AOD, BRDF.
- **Reviewer profile**: ancillary/reanalysis data specialist.
- **Files**: `readers/__init__.py`, `base.py`, `registry.py`, `_hdf4_io.py`, `_hdf5_io.py`,
  `_pyproj_io.py`, `era5.py`, `era5_pressure.py`, `igbp.py`, `nsidc.py`, `aod.py`, `brdf.py`;
  tests: `test_readers/` for those modules + `fixtures.py`.
- **Watch-out**: `readers/base.py` final state includes the tiling hooks added in #57
  (+76 lines). They reference only `types` (already landed in PR 1), so they ship here as
  end-state code; the tiling PR exercises them.
- **Notebook**: open each gridded dataset (synthetic or bundled sample), show the registry
  lookup by `DataProductIdentifier`/file pattern, plot one field per reader on a global map.

### PR 4 — External data readers: swath products

- **Branch**: `fmatch-4-readers-swath` (base: PR 3)
- **Topic**: Readers for *swath* datasets on satellite ground tracks: CERES SSF, CLDPIX
  cloud-pixel products, VIIRS — plus the shared swath machinery (`_swath.py`).
- **Reviewer profile**: CERES/imager swath product specialist.
- **Files**: `readers/_swath.py`, `ssf.py`, `cldpix.py`, `viirs.py`;
  tests: `test_swath.py`, `test_ssf.py`, `test_cldpix.py`, `test_viirs.py`.
- **Notebook**: read a synthetic SSF/CLDPIX granule, show swath geometry, time matching, and
  the flattening of multi-axis SSF variables to the 1-D footprint axis.

*(PRs 3 and 4 can be collapsed into one if the same person reviews both; kept separate here
because the data formats and the specialists differ.)*

### PR 5 — Tile manager

- **Branch**: `fmatch-5-tiling` (base: PR 4)
- **Topic**: Spatial tiling of ancillary grids — `GridTile`/`TileKey` partitioning, tile
  loading/caching through the reader interface, dateline and polar handling. Final state
  includes the #61 tiling efficiency work (tiling.py grew 599 → ~950 lines across #57+#61).
- **Reviewer profile**: performance / spatial-indexing specialist.
- **Files**: `tiling.py`; tests: `test_tiling.py` (final combined ~640 lines).
- **Notebook**: build a tile grid over a synthetic global dataset; visualize which tiles a
  footprint bounding box touches (including a dateline-crossing case); demonstrate cache
  hit/miss behavior and the efficiency gain versus whole-grid loads.

### PR 6 — PSF weighting & aggregation

- **Branch**: `fmatch-6-weighting-aggregation` (base: PR 5)
- **Topic**: PSF-weighted statistics: computing per-point PSF weights over a footprint
  (`weighting.py`) and aggregating external variables into footprint means/standard
  deviations (`aggregation.py`), with the #61 vectorized aggregation (2.64 s → 1.8 s per
  granule benchmark).
- **Reviewer profile**: radiation-budget science / CERES PSF specialist.
- **Files**: `weighting.py`, `aggregation.py`;
  tests: `test_weighting.py`, `test_aggregation.py`, and
  `test_aggregate_external_variables.py` if its imports resolve at this point (it needs
  readers + weighting + aggregation, all present); otherwise it moves to PR 8.
- **Notebook**: plot the PSF weight field over a single footprint; aggregate a synthetic
  ancillary field and compare weighted vs. unweighted means; show the `_lower`/std-dev
  variable naming convention.

### PR 7 — FMATCH product definitions (metadata only)

- **Branch**: `fmatch-7-product-definitions` (base: PR 6)
- **Topic**: The five FMATCH data product definition YAMLs (`fmatch_cam`,
  `fmatch_cam_camtime`, `fmatch_imager`, `fmatch_imager_camtime`, `fmatch_imager_flash`) and
  the matching `DataProductIdentifier` additions in `constants.py`. Pure
  metadata/science-content review: variable names, dims, units (e.g. cloud fraction in
  percent), long names, SSF-derived variable set.
- **Reviewer profile**: data product / metadata & science content specialist (no algorithm
  code in this PR — deliberately reviewable by a non-developer).
- **Files**: `data/product_definitions/fmatch_*.yml`, `constants.py`.
- **Notebook**: load each YAML through `LiberaDataProductDefinition`, render a variable
  inventory table (name, dims, dtype, units) per product, and diff the CAM vs IMAGER
  variable sets so reviewers see product scope at a glance.

### PR 8 — Product assembly

- **Branch**: `fmatch-8-product-assembly` (base: PR 7)
- **Topic**: `product.py` — orchestrating readers, tiling, weighting, and aggregation into a
  populated product dataset: coverage handling, FOOTPRINT coordinate, camtime axes, fill
  values, per-mode variable gating (`only_modes`).
- **Reviewer profile**: FMATCH algorithm developer.
- **Files**: `product.py`; tests: `test_product.py`, `test_l1b_inputs.py` (if its imports
  resolve here; it exercises L1B input handling used by product assembly).
- **Notebook**: assemble a small FMATCH product from synthetic L1B + synthetic ancillary
  inputs (reusing `tests` fixture generators), inspect the resulting `xarray.Dataset`, and
  show coverage/fill behavior for a footprint with partial ancillary coverage.

### PR 9 — Runners, CLI, and Docker (integration point)

- **Branch**: `fmatch-9-runners` (base: PR 8)
- **Topic**: End-to-end orchestration: `_runner.py` (`FmatchRunnerConfig`, `run_algorithm`,
  manifest handling, NetCDF writing), the five `fmatch_<product>.py` entry-point modules,
  the FMATCH `Dockerfile`, and CLI registration. This is where the feature becomes complete:
  the integration tests (`tests/integration/test_fmatch_runner.py`) and any end-to-end unit
  tests from #61 land here and must pass.
- **Reviewer profile**: pipeline / SDC infrastructure specialist.
- **Files**: `_runner.py`, `fmatch_cam.py`, `fmatch_cam_camtime.py`, `fmatch_imager.py`,
  `fmatch_imager_camtime.py`, `fmatch_imager_flash.py`, `footprint_matching/Dockerfile`,
  `.github/actions/setup-env` change, `pyproject.toml` entry points;
  tests: `tests/integration/test_fmatch_runner.py` + remaining e2e tests.
- **Notebook**: the full synthetic-pipeline demo (per the existing
  `generate_example_products` work): generate a synthetic day of L1B + ancillary inputs,
  run every FMATCH runner end-to-end, and open the resulting NetCDF products.

### PR 10 — Scene-ID imager family

- **Branch**: `scene-id-imager-family` (base: PR 9)
- **Topic**: Extending Scene-ID beyond CAM: `scene_id_imager.py`, `scene_id_imager_flash.py`,
  the `scene_id.py`/`scene_definitions.py` updates, the consolidated
  `scene_identification/Dockerfile` (replacing the per-product `cam/` and `cam_camtime/`
  subpackage Dockerfiles), and the `scene_id_imager*.yml` product definitions. Consumes
  FMATCH-IMAGER products at runtime, hence last in the stack.
- **Reviewer profile**: scene classification specialist.
- **Files**: `scene_identification/*` diffs from #54 and #57 at final state,
  `data/product_definitions/scene_id_imager.yml`, `scene_id_imager_flash.yml`, the small
  `scene_id_cam*.yml` touch-ups; tests: `test_scene_id.py`, `test_scene_id_cam.py`,
  `test_scene_definitions.py` updates, `tests/integration/test_scene_id*.py`.
- **Notebook**: run Scene-ID over a synthetic FMATCH-IMAGER product; visualize the resulting
  scene classifications and how imager-flash differs from imager.

## Notebook conventions

- Location: `doc/notebooks/<nn>_<topic>.ipynb` (e.g. `doc/notebooks/05_tiling.ipynb`), one
  per PR, added in that PR.
- **Self-contained**: every notebook generates its own inputs by reusing the synthetic
  fixture generators in `tests/unit/test_footprint_matching/fixtures.py` and the tests
  plugins — no external data downloads, runnable by a reviewer with only the repo checkout.
- Committed **with outputs saved** so a reviewer can read the demo without executing it.
- Not executed in CI (they are review/demo artifacts, not tests). If we later want rot
  protection, a follow-up ticket can add an `nbclient` smoke job.

## Testing rules per branch

- Every branch: `ruff check`, `ruff format --check`, and
  `pytest -m "not integration" tests/` must pass.
- Test files are harvested at final state alongside their module — a test file joins the
  stack in the same PR as the module it tests. Because tests are final-state, nothing tests
  placeholders (goal 3). If a harvested test imports a module not yet in the stack, that is
  a signal the file belongs later in the stack, not that the test should be weakened.
- Integration tests exist only from PR 9 onward; before that nothing integration-shaped is
  present to fail. From PR 9 the full stack is functionally complete for FMATCH, and PR 10
  completes Scene-ID.

## Build procedure

1. Confirm `origin/fmatch-efficiency` is the true final state (it is the stack tip; spot-check
   that no newer local work exists on `LIBSDC-807-tile-manager` or elsewhere).
2. For each PR *n* in order: branch from PR *n−1*'s branch, `git checkout
   origin/fmatch-efficiency -- <files>`, write the notebook, run lint + unit tests, fix any
   file-assignment issues (imports that don't resolve), commit, push, open the PR against
   PR *n−1*'s branch.
3. Sanity check when done: `git diff fmatch-efficiency <PR-10-branch>` over `libera_utils/`
   and `tests/` should be empty except for the added notebooks and any deliberate omissions —
   this proves the re-stack dropped nothing.
4. Close #54, #57, #61 with a comment linking the new stack. Keep the old branches until the
   new stack fully merges.
5. Merge bottom-up; after each merge, retarget the next PR's base (GitHub retargets
   automatically on branch deletion).

## Open questions

- JIRA tickets: one per PR (branch names above are placeholders — substitute
  `LIBSDC-<ticket>-` prefixes once tickets exist), or reuse the existing 794/807 tickets
  for the PRs that cover their scope.
- Whether PRs 3/4 (readers) collapse into one, and whether PR 7 (YAML metadata) folds into
  PR 8 — recommended to keep all separate given the distinct reviewer profiles.
