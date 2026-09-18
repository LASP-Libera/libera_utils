# Plan: Placeholder Camera Cloud Fraction (CF-CAM / CF-CAM-CAMTIME) algorithms

## Goal

Turn the `camera_segmentation` module (added on `fmatch-8-camera-segmentation`) from an
internal FMATCH helper into the engine of a real, standalone **Camera Cloud Fraction**
algorithm family, in the exact style of the FMATCH runners:

- **CF-CAM-CAMTIME** — camera timescale. Reads an **L1B CAM** file, segments it into
  pseudo-footprints with `camera_segmentation.segment_l1b_camera`, and writes a product whose
  record axis / pixel provenance is **identical** to `FMATCH-CAM-CAMTIME`, plus a
  `cloud_fraction` and `cloud_fraction_standard_deviation` per pseudo-footprint. This is the
  input FMATCH-CAM-CAMTIME consumes as `cloud_fraction_camera`. **The pseudo-footprint is
  created here, not in footprint matching.**
- **CF-CAM** — radiometer timescale (the non-camtime sibling, mirroring FMATCH-CAM). Reads an
  **L1B RAD-4CH** file and writes one `cloud_fraction` / `cloud_fraction_standard_deviation`
  per `RADIOMETER_TIME` footprint. This is the input FMATCH-CAM consumes as
  `cloud_fraction_camera`. No camera segmentation — the footprints are the radiometer's own.

Both products carry **placeholder** values: a reproducible random draw in `[0, 100]` percent,
pending the real Libera WFOV cloud-fraction retrieval. Everything else (record axis,
coordinates, pixel provenance) is derived deterministically so each CF product aligns 1:1 with
its FMATCH sibling built from the same L1B file.

> **Design decision (confirmed with user, 2026-09-17):** CF-CAM is the *radiometer-timescale
> mirror* of FMATCH-CAM (input L1B RAD-4CH, records on `RADIOMETER_TIME`, no camera
> segmentation). Only CF-CAM-CAMTIME uses `camera_segmentation`.

## Branch / stack placement

This work sits **on top of the FMATCH runners** (`fmatch-9-runners`), because it reuses:
`footprint_matching._runner` (`load_l1b_camera_dataset`, `load_l1b_radiometer_inputs`,
`_as_local_path`, `select_manifest_files_by_product_id`), `camera_segmentation.segment_l1b_camera`,
and the camtime grid assembly in `footprint_matching/product.py`.

Create a new stacked branch off `fmatch-9-runners` (e.g. `fmatch-10-cloud-fraction` /
`LIBSDC-XXXX-cloud-fraction-placeholder`). A prior spike of the camtime half exists on the
`cloud-fraction-placeholder` branch (commit `7d9ce47`) and is a useful reference, **but it is
stale in two ways and must not be copied verbatim** — see "Reference spike" below.

## Current state / what already exists

- **Constants — already present on `fmatch-8`** (no change needed):
  - `DataProductIdentifier.l2_cf_cam = ("CF-CAM", DataLevel.L2)` and
    `l2_cf_cam_camtime = ("CF-CAM-CAMTIME", DataLevel.L2)` (`constants.py:271-272`)
  - `ProcessingStepIdentifier.l2_cf_cam` / `l2_cf_cam_camtime` (`constants.py:463-464`)
  - `PROCESSING_STEP_NODE_NAMES[...] = "L2-CloudFraction"` (`constants.py:626-627`)
  - Verify `tests/unit/test_constants.py` already lists both (the reference expects them).
- **FMATCH already declares the consumer side.** `fmatch_cam.py` sets
  `cloud_fraction_product_id=DataProductIdentifier.l2_cf_cam`; `fmatch_cam_camtime.py` sets
  `l2_cf_cam_camtime`. Today `footprint_matching/_runner.py` selects those files from the
  manifest, records them as provenance, and writes `cloud_fraction_camera` as a **placeholder**
  (`TODO[LIBSDC-785]`). Producing the CF products makes those inputs real; *ingesting* their
  values into FMATCH is optional follow-on (see "Out of scope").
- **Camtime provenance model (current, on this branch).** `fmatch_cam_camtime.yml` uses a **2-D
  `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid**: `CAMERA_TIME` (unique, sorted) and `PSEUDOFOOTPRINT`
  (0-based subsection index) coordinates, and **four separate int32 coordinates**
  `camera_pixel_x_min`, `camera_pixel_x_max`, `camera_pixel_y_min`, `camera_pixel_y_max`. The
  flat footprint list is scattered onto this grid by `_assemble_camtime_dataset` in
  `product.py:1163` (the `to_grid` closure at `product.py:1256`). **CF-CAM-CAMTIME must produce
  this exact model** so it aligns record-for-record with FMATCH-CAM-CAMTIME.

### Reference spike (`cloud-fraction-placeholder`, commit `7d9ce47`) — what to keep / fix

Keep the shape: `libera_utils/cloud_fraction/{__init__,_runner,cf_cam_camtime}.py`,
`CfRunnerConfig` mirroring the FMATCH/SCENE-ID runners, `generate_placeholder_cloud_fraction`
core, and the unit + integration tests. **Do not** keep the reference's standalone
`libera_utils/cloud_fraction/Dockerfile` or its `python -m`-style `main()` entry — this repo uses
one root Dockerfile with per-algorithm CLI entrypoints (items 2 and 6).

**Fix before reuse:**
1. **Stale product model.** The reference `cf_cam_camtime.yml` uses the old
   `FOOTPRINT` + `CAMERA_PIXEL_BOUNDS` axis with `camera_pixel_x`/`camera_pixel_y` as `(min,max)`
   pairs. That predates the current 2-D `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid with four
   `_min`/`_max` coordinates. **Rewrite against the current model** (mirror
   `fmatch_cam_camtime.yml`). In particular, the subsection axis **must be named
   `PSEUDOFOOTPRINT`, not `FOOTPRINT`**, to stay consistent with the FOOTPRINT→PSEUDOFOOTPRINT
   rename applied across the rest of the FMATCH stack — CF-CAM-CAMTIME and FMATCH-CAM-CAMTIME
   must use the identical `PSEUDOFOOTPRINT` axis name so their records line up.
2. **CF-CAM missing.** The spike only built the camtime half. This plan adds the
   radiometer-timescale **CF-CAM** and a shared runner that handles both.

## Work items

### 1. New package `libera_utils/cloud_fraction/`

```
libera_utils/cloud_fraction/
    __init__.py            # exports the two placeholder core builders
    cloud_fraction.py      # pure, I/O-free core (random draws + provenance assembly)
    _runner.py             # shared CfRunnerConfig + run_algorithm (both input modes)
    cf_cam.py              # radiometer runner  (L1B RAD-4CH -> CF-CAM); exposes algorithm()
    cf_cam_camtime.py      # camera runner      (L1B CAM     -> CF-CAM-CAMTIME); exposes algorithm()
```

**No per-package Dockerfile and no `python -m` entry.** Operationally these algorithms run through
the `libera-utils` CLI (item 2), and containerize as target stages of the single repo-root
`Dockerfile` (item 6) — exactly how SCENE-ID-CAM / SCENE-ID-CAM-CAMTIME already work on this branch.
The reference spike's standalone `Dockerfile` and `main()`/`argparse`/`__main__` blocks are
therefore **dropped**.

#### `cloud_fraction.py` — pure core

Two public builders, both returning a `dict[str, np.ndarray]` keyed by the product-definition
variable/coordinate names, ready for `write_libera_data_product`. Shared random-draw helper so
both use the same seeded generator and the same `[0, 100]` percent semantics.

- `generate_placeholder_cloud_fraction_camtime(footprints, *, rng=None, max_std_percent=15.0)`
  - Input: `Sequence[PseudoFootprint]` from `segment_l1b_camera`.
  - Draws one random `cloud_fraction` and `cloud_fraction_standard_deviation` per footprint,
    then scatters both — **plus the CAMERA_TIME / PSEUDOFOOTPRINT / camera_pixel_{x,y}_{min,max}
    provenance** — onto the rectangular `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid using the **same**
    transform FMATCH uses. See item 4: reuse a shared `build_camtime_grid` helper rather than
    re-deriving the ragged→rectangular mapping, so alignment is guaranteed by construction.
- `generate_placeholder_cloud_fraction_radiometer(radiometer_time, *, rng=None, max_std_percent=15.0)`
  - Input: the `RADIOMETER_TIME` datetime64 array from `load_l1b_radiometer_inputs`
    (`_runner.py` returns it keyed `"RADIOMETER_TIME"`).
  - Returns `{RADIOMETER_TIME, cloud_fraction, cloud_fraction_standard_deviation}`, one value per
    radiometer footprint on the 1-D `RADIOMETER_TIME` axis.
- Randomness: `np.random.default_rng(seed)` (numpy Generator, **not** stdlib `random` — avoids
  Bandit S311), fixed default seed for reproducibility, `rng` injectable for tests. Values cast
  to `float32`. `cloud_fraction ~ U(0, 100)`; `std ~ U(0, max_std_percent)`.
- Raise `ValueError` on empty input (no axis to write) and on out-of-range `max_std_percent`.

#### `_runner.py` — shared manifest-driven runner

Mirror `footprint_matching/_runner.py` and `scene_identification/_runner.py` exactly (same 6
logged steps: read manifest → collect inputs → build+write product(s) → output manifest). One
`CfRunnerConfig` frozen dataclass captures the per-runner differences; each concrete runner is a
thin module forwarding `algorithm` to `run_algorithm`. `run_algorithm` accepts either a manifest
path or an `argparse.Namespace` (so the CLI handler can pass the parsed args straight through),
exactly as the SCENE-ID / FMATCH runners already do.

`CfRunnerConfig` fields:

| field | CF-CAM | CF-CAM-CAMTIME |
| --- | --- | --- |
| `output_product_id` | `l2_cf_cam` | `l2_cf_cam_camtime` |
| `l1b_input_product_id` | `l1b_rad` | `l1b_cam` |
| `product_definition_path` | `cf_cam.yml` | `cf_cam_camtime.yml` |
| `time_variable` | `"RADIOMETER_TIME"` | `"CAMERA_TIME"` |
| `input_kind` | `RADIOMETER` | `CAMERA` (drives which loader + core builder run) |
| `log_prefix` | `"cf_cam"` | `"cf_cam_camtime"` |

- `run_algorithm`: read `PROCESSING_PATH` (raise if unset), collect input files by product id
  (reuse `select_manifest_files_by_product_id` — do not re-implement the reference's
  `collect_input_files`), and for each input build+write the product.
- `create_and_write_data_product`: branch on `input_kind`:
  - **CAMERA**: `_as_local_path` → `load_l1b_camera_dataset` → `segment_l1b_camera` →
    `generate_placeholder_cloud_fraction_camtime`.
  - **RADIOMETER**: `_as_local_path` → `load_l1b_radiometer_inputs` → take `RADIOMETER_TIME` →
    `generate_placeholder_cloud_fraction_radiometer`.
  - Then `write_libera_data_product(definition, data, output_path, time_variable=...,
    dynamic_product_attributes={"algorithm_version": ..., "input_files": <name>}, strict=True)`.
- Reuse the FMATCH runner's helpers where they already exist and are importable
  (`_as_local_path`, `load_l1b_*`, `select_manifest_files_by_product_id`, and an
  `algorithm_version()` equivalent) rather than copying them, to avoid drift. If any are
  currently module-private in `footprint_matching/_runner.py`, promote them to importable names
  (single source of truth) as part of this change.

#### `cf_cam.py` / `cf_cam_camtime.py` — thin runners

Copy the **`scene_id_cam.py` / `scene_id_cam_camtime.py`** shape (not the reference's `python -m`
shape): module docstring, `RUNNER_CONFIG`, and a single `algorithm(manifest_path)` that forwards to
`run_algorithm`. **No `main()` / `argparse` / `if __name__ == "__main__"`** — the CLI owns argument
parsing (item 2). Product definition path resolved via the package data dir, as SCENE-ID does.

#### `__init__.py`

Package docstring (family overview, placeholder caveat) + export the two core builders.

### 2. CLI wiring (`libera_utils/cli.py`)

Register a `cloud-fraction` subcommand with two sub-subcommands, mirroring the existing `scene-id`
block (`cli.py:110-123`) and its Docker/CLI convention:

- Two handlers next to `scene_id_cam_cli_handler` / `scene_id_cam_camtime_cli_handler`:
  - `cloud_fraction_cam_cli_handler(parsed_args)` — imports
    `libera_utils.cloud_fraction.cf_cam.algorithm` and returns `algorithm(parsed_args)`.
  - `cloud_fraction_cam_camtime_cli_handler(parsed_args)` — imports
    `libera_utils.cloud_fraction.cf_cam_camtime.algorithm` likewise.
  - Import inside the handler (lazy), as SCENE-ID does, so `libera-utils --version` etc. don't pull
    in the heavy geospatial stack.
- In `parse_cli_args`, add:
  ```python
  cloud_fraction_parser = subparsers.add_parser(
      "cloud-fraction", help="run a Libera Camera Cloud Fraction algorithm from a manifest file")
  cf_subparsers = cloud_fraction_parser.add_subparsers(...)
  cf_cam_parser = cf_subparsers.add_parser("cam", help="run the CF-CAM algorithm (radiometer timescale) ...")
  cf_cam_parser.set_defaults(func=cloud_fraction_cam_cli_handler)
  cf_cam_parser.add_argument("manifest", type=str, help="path to the input manifest file")
  cf_cam_camtime_parser = cf_subparsers.add_parser("cam-camtime", help="run the CF-CAM-CAMTIME algorithm (camera timescale) ...")
  cf_cam_camtime_parser.set_defaults(func=cloud_fraction_cam_camtime_cli_handler)
  cf_cam_camtime_parser.add_argument("manifest", type=str, help="path to the input manifest file")
  ```

Operational invocations become `libera-utils cloud-fraction cam <manifest>` and
`libera-utils cloud-fraction cam-camtime <manifest>`.

### 3. Product definitions (`libera_utils/data/product_definitions/`)

#### `cf_cam.yml` (new) — radiometer timescale

```yaml
attributes:
  ProductID: CF-CAM
  algorithm_version: null   # dynamic
  input_files: null         # dynamic (source L1B RAD-4CH filename)
coordinates:
  RADIOMETER_TIME:
    dtype: datetime64[ns]
    dimensions: ["RADIOMETER_TIME"]
    # long_name + encoding copied from fmatch_cam.yml's RADIOMETER_TIME so the axes match exactly
variables:
  cloud_fraction:
    dtype: float32
    dimensions: ["RADIOMETER_TIME"]
    attributes: {long_name: "... (Libera WFOV, PLACEHOLDER)", units: percent, valid_range: [0.0, 100.0]}
  cloud_fraction_standard_deviation:
    dtype: float32
    dimensions: ["RADIOMETER_TIME"]
    attributes: {long_name: "...", units: percent, valid_range: [0.0, 100.0]}
```

Copy `RADIOMETER_TIME`'s `long_name`/`encoding` verbatim from `fmatch_cam.yml` so CF-CAM and
FMATCH-CAM share an identical axis.

#### `cf_cam_camtime.yml` (new) — camera timescale, **current grid model**

Mirror `fmatch_cam_camtime.yml`'s coordinate block exactly (do **not** reuse the stale reference
yml):

```yaml
attributes:
  ProductID: CF-CAM-CAMTIME
  algorithm_version: null
  input_files: null
coordinates:
  CAMERA_TIME:            # dimensions: ["CAMERA_TIME"] — unique, sorted; identical decl to FMATCH-CAM-CAMTIME
  PSEUDOFOOTPRINT:        # dimensions: ["PSEUDOFOOTPRINT"] — 0-based subsection index
  camera_pixel_x_min:     # int32, dimensions: ["CAMERA_TIME", "PSEUDOFOOTPRINT"]
  camera_pixel_x_max:     # int32, same dims
  camera_pixel_y_min:     # int32, same dims
  camera_pixel_y_max:     # int32, same dims
variables:
  cloud_fraction:                     # float32, dims ["CAMERA_TIME","PSEUDOFOOTPRINT"], percent [0,100]
  cloud_fraction_standard_deviation:  # float32, same dims, percent [0,100]
```

Copy the four `camera_pixel_*` and both coordinate declarations byte-for-byte from
`fmatch_cam_camtime.yml` (including `long_name` and the `PSEUDOFOOTPRINT`/`CAMERA_TIME` notes)
so a reader can treat the provenance columns of the two products interchangeably.

### 4. Shared camtime grid helper (refactor in `footprint_matching/product.py`)

To guarantee CF-CAM-CAMTIME's `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid and pixel provenance are
**identical** to FMATCH-CAM-CAMTIME (and to keep the ragged→rectangular logic in one place),
factor the grid-construction preamble out of `_assemble_camtime_dataset` (`product.py:1236-1305`):

- New helper, e.g. `build_camtime_grid(footprints, definition) -> CamtimeGrid`, returning the
  `unique_times`, `to_grid` closure, and the base `data` dict already populated with
  `CAMERA_TIME`, `PSEUDOFOOTPRINT`, and the four `camera_pixel_{x,y}_{min,max}` coordinates.
- `_assemble_camtime_dataset` calls it, then continues to scatter its FMATCH-only columns.
- `cloud_fraction.generate_placeholder_cloud_fraction_camtime` calls it against the
  **CF-CAM-CAMTIME** definition, then scatters only `cloud_fraction` / `cloud_fraction_standard_deviation`.

This edit lives in a lower stack layer (`product.py`, on `fmatch-7`/`fmatch-9`). If keeping the
change out of the FMATCH layer is preferred, the fallback is to replicate the ~15-line transform
inside `cloud_fraction.py` — **but then add a mandatory parity test** (item 7) asserting the two
products' `CAMERA_TIME` / `PSEUDOFOOTPRINT` / `camera_pixel_*` grids are element-wise equal for
the same L1B CAM fixture. Recommended: the shared helper.

### 5. Notebook — extend `notebooks/08_camera_segmentation.ipynb`

Today the notebook covers the module and ends with "Where this goes next in the stack" pointing
at the runners PR. Rework it so it **also reviews the two CF algorithms** the module now powers.
Retitle to reflect the broader scope (e.g. "Camera segmentation → Camera Cloud Fraction").

Add, after the existing segmentation sections:

1. **From segmentation to CF-CAM-CAMTIME.** Call
   `generate_placeholder_cloud_fraction_camtime(footprints)` on the synthetic footprints already
   built in the notebook; show the returned `cloud_fraction` grid, and that the
   `camera_pixel_{x,y}_{min,max}` provenance matches what FMATCH-CAM-CAMTIME emits for the same
   footprints (side-by-side or an `assert`-backed equality cell). Plot the random cloud_fraction
   over the pseudo-footprint grid.
2. **CF-CAM (radiometer timescale).** Build a tiny synthetic `RADIOMETER_TIME` array (reuse the
   fmatch radiometer fixture helper) and call
   `generate_placeholder_cloud_fraction_radiometer(...)`; show the 1-D per-footprint output.
   Contrast the two record axes (camera image cadence vs radiometer times) in one markdown cell.
3. **How they feed FMATCH.** Markdown: CF-CAM → FMATCH-CAM `cloud_fraction_camera`;
   CF-CAM-CAMTIME → FMATCH-CAM-CAMTIME `cloud_fraction_camera`; placeholder caveat; the footprint
   originates in CF-CAM-CAMTIME, not FMATCH. Keep the "next in the stack" pointer, updated.

Follow the repo's notebook conventions: keep it runnable top-to-bottom with only synthetic data
(no AWS, no external files), re-execute so committed outputs are current, and add the
documentation cross-links used by the other moved notebooks.

### 6. Packaging / ops — single repo-root `Dockerfile`

There is **one Dockerfile for the whole repo** (`./Dockerfile`), with one build stage per
algorithm that only overrides `ENTRYPOINT` to a `libera-utils` CLI subcommand. Do **not** add a
`libera_utils/cloud_fraction/Dockerfile`. Append two target stages, right after the existing
`libera-utils-scene-id-cam-camtime` stage, following that exact pattern:

```dockerfile
# CLI for the CF-CAM algorithm (radiometer timescale) from a manifest file.
# -------------------------------------------------------------------------
FROM libera-utils AS libera-utils-cloud-fraction-cam

ENTRYPOINT ["libera-utils", "cloud-fraction", "cam"]


# CLI for the CF-CAM-CAMTIME algorithm (camera timescale) from a manifest file.
# -----------------------------------------------------------------------------
FROM libera-utils AS libera-utils-cloud-fraction-cam-camtime

ENTRYPOINT ["libera-utils", "cloud-fraction", "cam-camtime"]
```

- These inherit the base `libera-utils` image, so the system deps (netCDF/HDF5/GDAL/udunits, CSPICE)
  and the `poetry sync --only main` install already cover the CF runtime — nothing image-specific to
  add. Build one with `docker build --target libera-utils-cloud-fraction-cam-camtime .`.
- Confirm `libera_utils/cloud_fraction/` and the geospatial deps CF needs (`pyproj`, `xarray`,
  `netCDF4`) are installed into the base image. They are already required by FMATCH/SCENE-ID; if any
  live behind an optional extra, ensure the base image installs it. No new `[project.scripts]`
  (the sole console script stays `libera-utils`), and no separate Dockerfile.
- Confirm `data/product_definitions/*.yml` ship as package data (they already do).

### 7. Tests

- `tests/unit/test_cloud_fraction/test_cloud_fraction.py` — core builders: both return all
  declared variables on the right axis; values in `[0, 100]`; `std` in `[0, max]`; float32;
  reproducible with a fixed seed; camtime provenance (`camera_pixel_*`) matches the footprints'
  inclusive slices; empty input raises. (Extend the reference file.)
- `tests/unit/test_cloud_fraction/test_cf_cam_product.py` and `test_cf_cam_camtime_product.py` —
  write each product through `write_libera_data_product` against its yml and assert it validates
  (`strict=True`), dims/coords/dtypes/units, and (camtime) the 2-D grid shape + int32 pixel
  coordinates.
- **Parity test (critical):** for one synthetic L1B CAM fixture, assemble FMATCH-CAM-CAMTIME and
  CF-CAM-CAMTIME and assert their `CAMERA_TIME`, `PSEUDOFOOTPRINT`, and four `camera_pixel_*`
  arrays are element-wise equal. This is what "records align 1:1" means operationally.
- `tests/integration/test_cf_cam_runner.py` + `test_cf_cam_camtime_runner.py` — manifest in →
  product + output manifest out, with `PROCESSING_PATH` set to `tmp_path` and moto/local paths
  only (never real AWS). Mirror `tests/integration/test_fmatch_runner.py`. Reuse/extend
  `tests/test_data/footprint_matching/fixtures.py` (`make_l1b_camera_fixture`, radiometer
  equivalent, manifest builders).
- `tests/unit/test_cli.py` — extend the SCENE-ID cases (`test_cli.py:56-73`, `:414-422`): assert
  `cloud-fraction cam <manifest>` / `cloud-fraction cam-camtime <manifest>` parse to the right
  handler + `manifest`, and that each handler dispatches to its runner's `algorithm` with the parsed
  args (patch `libera_utils.cloud_fraction.cf_cam` / `cf_cam_camtime`).
- Ensure `tests/unit/test_constants.py` covers both product ids (already added in the reference).
- Full gate: `ruff check`, `ruff format --check`, `pytest -m "not integration" tests/`, then the
  new integration tests; pre-commit (incl. Bandit — the `default_rng` choice keeps S311 clean).

## Out of scope (call out explicitly; do not silently include)

- **Ingesting CF values into FMATCH.** FMATCH currently writes `cloud_fraction_camera` as a
  placeholder and only records the CF file as provenance (`TODO[LIBSDC-785]`). The reference
  spike went further and wired real ingestion (`read_cloud_fraction_camera` /
  `_ingest_camera_cloud_fraction` + a `_merge_cloud_fraction_camera` hook already present at
  `product.py:1317`). **This plan produces the CF products but leaves FMATCH ingestion as
  follow-on** unless the user asks to include it. If included, add it as a final optional phase
  with its own provenance-alignment guard and tests.
- The real WFOV cloud-fraction retrieval (this is a placeholder by design).

## Risks / watch-items

- **Provenance drift.** If the camtime model changes again (e.g. `LIBSDC-847` video mode breaks
  `CAMERA_TIME` uniqueness), CF-CAM-CAMTIME must move with FMATCH-CAM-CAMTIME. The shared
  `build_camtime_grid` helper (item 4) is the mitigation — one place to change.
- **Empty / all-fill images.** `segment_l1b_camera` can drop blocks (all-fill corners). Decide
  and test behavior when an image yields zero footprints (per-image vs whole-file). Match FMATCH.
- **Stack ordering.** The `product.py` refactor lands in a lower layer than the new package;
  sequence the rebase so `fmatch-9` carries the helper before the cloud-fraction branch consumes
  it, or take the inline-replica + parity-test fallback.
- Keep `libera_utils.io` free of runner plumbing (the `_as_local_path` pattern) — reuse the
  FMATCH helper, don't fork it.

## Suggested commit sequence (one stacked branch)

1. `product.py`: factor `build_camtime_grid` out of `_assemble_camtime_dataset` (no behavior
   change; existing FMATCH tests stay green).
2. Add `cf_cam.yml` + `cf_cam_camtime.yml` product definitions (+ product-write unit tests).
3. Add `libera_utils/cloud_fraction/` core + `_runner` + two thin `algorithm()` runners (+ core
   unit tests + parity test).
4. Wire the `cloud-fraction cam` / `cam-camtime` subcommands into `cli.py` (+ `test_cli.py` cases).
5. Add the two `libera-utils-cloud-fraction-*` target stages to the root `Dockerfile`; integration
   runner tests.
6. Extend `notebooks/08_camera_segmentation.ipynb` to review CF-CAM / CF-CAM-CAMTIME; re-execute.
