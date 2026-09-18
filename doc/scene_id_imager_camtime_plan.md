# Implementation Plan — Add the `SCENE-ID-IMAGER-CAMTIME` product (LIBSDC-672)

## Goal

Add a new SCENE-ID product, **`SCENE-ID-IMAGER-CAMTIME`**, the camera-timescale sibling of
`SCENE-ID-IMAGER`. It:

- runs on the **camera time** `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid (not `RADIOMETER_TIME`),
- takes **`FMATCH-IMAGER-CAMTIME`** as its operational input,
- produces the **same scientific variables as `SCENE-ID-IMAGER`** — the ERBE, unfiltering, and TRMM
  scene classifications with their input/derived scene properties and property-bin bounds, and
- additionally **passes through the pseudofootprint provenance** (the `PSEUDOFOOTPRINT` coordinate,
  boresight `latitude`/`longitude`/`altitude`, PSF bounding box, and camera pixel-block bounds) the
  way `SCENE-ID-CAM-CAMTIME` does.

In one sentence: it is `SCENE-ID-IMAGER`'s science re-based onto `SCENE-ID-CAM-CAMTIME`'s grid +
provenance passthrough.

## Why this is small

The heavy lifting already exists on this branch:

- **Constants** — both [`DataProductIdentifier.aux_fmatch_imager_camtime`](../libera_utils/constants.py#L325)
  and [`aux_scene_id_imager_camtime`](../libera_utils/constants.py#L337), plus their `ManifestType`
  entries ([constants.py:488](../libera_utils/constants.py#L488),
  [constants.py:491](../libera_utils/constants.py#L491)), are already defined. **No change needed.**
- **FMATCH input** — [`fmatch_imager_camtime.yml`](../libera_utils/data/product_definitions/fmatch_imager_camtime.yml)
  exists and already carries every classification input on the 2-D grid: the RBSP CLDPIX
  (`cldpix_cloud_optical_depth`, `cldpix_cloud_particle_phase`), CERES SSF (`ssf_clear_coverage`),
  ERA5 winds (`era5_wind_u10`/`v10`), and `igbp_surface_type`, plus the passthrough identifiers
  (`latitude`/`longitude`/`altitude`, `psf_bbox_*`, `camera_pixel_{x,y}_{min,max}`).
- **Runner** — [`_runner.py`](../libera_utils/scene_identification/_runner.py) is fully generic and
  parameterized by [`SceneIdRunnerConfig`](../libera_utils/scene_identification/_runner.py#L43). Its
  `create_and_write_data_product` already handles the camera-timescale write path (drops undeclared
  vars, materializes the `PSEUDOFOOTPRINT` coordinate via `np.arange`), proven by `SCENE-ID-CAM-CAMTIME`.
- **Extractor** — [`_extract_data_from_fmatch`](../libera_utils/scene_identification/scene_id.py#L1106)
  already accepts the exact combination we need: a `column_map`, 2-D `record_dimensions`, and
  `passthrough_variables` — all in one call.
- **Fixture** — `make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_CAMTIME)` is already
  wired (see [test_scene_id_runner.py](../tests/integration/test_scene_id_runner.py) usages of
  `OperationalMode.CAM_CAMTIME` / `IMAGER`).

So the work is: **one reader method, one product-definition YAML, one runner module, one Dockerfile
target, and tests.** Everything else is reuse.

---

## Changes

### 1. Reader: `FootprintData.from_fmatch_imager_camtime` (`scene_identification/scene_id.py`)

Add a classmethod that is the **union** of the two existing readers:

- record grid + passthrough from
  [`from_fmatch_cam_camtime`](../libera_utils/scene_identification/scene_id.py#L972)
  (`record_dimensions=(CAMERA_TIME_VARIABLE, PSEUDOFOOTPRINT_DIMENSION)`, `time_variable=CAMERA_TIME_VARIABLE`,
  `passthrough_variables=...`), and
- classification column map + up-front validation from
  [`from_fmatch_imager`](../libera_utils/scene_identification/scene_id.py#L1042)
  (`column_map=_FMATCH_IMAGER_COLUMN_MAP`, a `context=` string).

```python
@classmethod
def from_fmatch_imager_camtime(cls, fmatch_path: pathlib.Path) -> "FootprintData":
    extracted_data = cls._extract_data_from_fmatch(
        fmatch_path,
        record_dimensions=(CAMERA_TIME_VARIABLE, PSEUDOFOOTPRINT_DIMENSION),
        time_variable=CAMERA_TIME_VARIABLE,
        column_map=_FMATCH_IMAGER_COLUMN_MAP,
        passthrough_variables=_FMATCH_CAMTIME_PASSTHROUGH_VARIABLES,
        context=(
            "SCENE-ID-IMAGER-CAMTIME reader (FMATCH-IMAGER-CAMTIME); the FMATCH file lacks the "
            "RBSP ssf/cldpix variables required for scene identification"
        ),
    )
    return cls(extracted_data)
```

This reuses `_FMATCH_IMAGER_COLUMN_MAP`
([scene_id.py:832](../libera_utils/scene_identification/scene_id.py#L832)) unchanged — the same RBSP
CLDPIX optical-depth + `map_cldpix_phase_to_trmm` cloud-phase mapping that gives `SCENE-ID-IMAGER`
its full TRMM classification. Because the extractor emits each column on its *source* variable's own
dims, the identical column map flows through on the 2-D grid with no changes (exactly as
`_FMATCH_CAM_COLUMN_MAP` does for `from_fmatch_cam_camtime`).

**Small cleanup (recommended):** the passthrough tuple is currently named
[`_FMATCH_CAM_CAMTIME_PASSTHROUGH_VARIABLES`](../libera_utils/scene_identification/scene_id.py#L47)
but its contents (lat/lon/alt, psf bbox, camera pixel bounds) are generic camtime provenance shared by
both camtime readers. Rename it to `_FMATCH_CAMTIME_PASSTHROUGH_VARIABLES` and update the one existing
reference in `from_fmatch_cam_camtime`, or add an alias. Update the module-level comment to note it now
serves both `CAM-CAMTIME` and `IMAGER-CAMTIME`.

Write a Numpy-style docstring modeled on the two parent readers, stating that this is the
camera-timescale IMAGER reader: full ERBE/unfiltering/TRMM inputs on the `(CAMERA_TIME, PSEUDOFOOTPRINT)`
grid, carrying the footprint identifier variables straight through.

### 2. Product definition: `data/product_definitions/scene_id_imager_camtime.yml`

Build by combining the two existing YAMLs:

- **Scientific body** — take every `variables:` block from
  [`scene_id_imager.yml`](../libera_utils/data/product_definitions/scene_id_imager.yml): input scene
  properties (`igbp_surface_type`, the three angles), derived properties (`surface_type`,
  `cloud_fraction`, `surface_wind`, `cloud_phase`, `optical_depth`), the three scene IDs
  (`scene_id_erbe`, `scene_id_unfiltering`, `scene_id_trmm` — note `scene_id_trmm` is **uint16**), all
  ERBE / unfiltering / **TRMM** property-bin bounds, and `Quality_Flag`. **Rebase every variable's
  `dimensions:` from `["RADIOMETER_TIME"]` to `["CAMERA_TIME", "PSEUDOFOOTPRINT"]`.**
- **Coordinates** — replace the `RADIOMETER_TIME` coordinate with the `CAMERA_TIME` + `PSEUDOFOOTPRINT`
  coordinate pair copied verbatim from
  [`scene_id_cam_camtime.yml`](../libera_utils/data/product_definitions/scene_id_cam_camtime.yml#L9-L43)
  (including the CCSDS-epoch encoding and the `TODO[LIBSDC-847]` video-mode note).
- **Passthrough identifier block** — append the "Footprint identifiers (passed straight through)"
  section from
  [`scene_id_cam_camtime.yml`](../libera_utils/data/product_definitions/scene_id_cam_camtime.yml#L234-L320):
  `latitude`, `longitude`, `altitude`, `psf_bbox_lat/lon_min/max`, and
  `camera_pixel_{x,y}_{min,max}`, all on `["CAMERA_TIME", "PSEUDOFOOTPRINT"]`, as **variables** (matching
  `scene_id_cam_camtime.yml`, which keeps these under `variables:`, not `coordinates:`). Keep the dtypes
  and `long_name`s bit-for-bit identical to the FMATCH source so the products agree.
- `attributes:` — `ProductID: SCENE-ID-IMAGER-CAMTIME`, `algorithm_version: null`,
  `InputGranules: null` (same `TODO[LIBSDC-822]`).

Net difference vs `scene_id_cam_camtime.yml`: this product adds `surface_wind`, `cloud_phase`,
`optical_depth`, `scene_id_trmm`, and the full TRMM bin-bounds block (all present in
`scene_id_imager.yml`). Net difference vs `scene_id_imager.yml`: the 2-D grid, the `CAMERA_TIME`/
`PSEUDOFOOTPRINT` coordinates, and the passthrough identifier block.

### 3. Runner module: `scene_identification/scene_id_imager_camtime.py`

New module mirroring [`scene_id_imager.py`](../libera_utils/scene_identification/scene_id_imager.py)
and [`scene_id_cam_camtime.py`](../libera_utils/scene_identification/scene_id_cam_camtime.py). Its
`RUNNER_CONFIG`:

```python
SCENE_ID_IMAGER_CAMTIME_SCENE_TYPES = ["erbe", "unfiltering", "trmm"]

PRODUCT_DEFINITION_PATH = (
    Path(__import__("libera_utils").__file__).parent
    / "data" / "product_definitions" / "scene_id_imager_camtime.yml"
)

RUNNER_CONFIG = SceneIdRunnerConfig(
    input_product_id=DataProductIdentifier.aux_fmatch_imager_camtime,
    output_product_id=DataProductIdentifier.aux_scene_id_imager_camtime,
    reader=FootprintData.from_fmatch_imager_camtime,
    product_definition_path=PRODUCT_DEFINITION_PATH,
    time_variable="CAMERA_TIME",
    scene_types=SCENE_ID_IMAGER_CAMTIME_SCENE_TYPES,
    log_prefix="scene_id_imager_camtime",
)
```

Provide the same thin wrappers the siblings expose (needed by the integration tests, which import them):
`algorithm`, `collect_fmatch_imager_camtime_input_files`, `run_scene_identification_imager_camtime`,
`create_and_write_data_product_imager_camtime`, and `main`/`__main__`. Each is a one-line forward to the
shared `_runner` helper, copied from `scene_id_cam_camtime.py` with the names/labels swapped.

### 4. Dockerfile target (`scene_identification/Dockerfile`)

Add a runtime target next to the existing ones
([Dockerfile:70-80](../libera_utils/scene_identification/Dockerfile#L70)):

```dockerfile
FROM scene-id-base AS scene-id-imager-camtime
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/scene_id_imager_camtime.py"]
```

Update the header comment's "Runtime targets:" list (line ~10) to include `scene-id-imager-camtime`.

### 5. CLI — decision point

The **imager family currently has no `libera-utils scene-id` subcommand** — only `cam` and
`cam-camtime` are wired ([cli.py:239-249](../libera_utils/cli.py#L239)); `SCENE-ID-IMAGER` and
`-IMAGER-FLASH` run only via their Docker entrypoints. To match the imager siblings, **do not add a CLI
subcommand** for `imager-camtime`; ship it as a Docker entrypoint + module `main()` only. (If CLI parity
across the whole family is later wanted, that is a separate, family-wide change.)

> **Pre-existing bug, out of scope:** the two `scene-id` CLI handlers import from the old subpackage
> paths `scene_identification.cam.scene_id_cam` and `...cam_camtime.scene_id_cam_camtime`
> ([cli.py:39](../libera_utils/cli.py#L39), [cli.py:57](../libera_utils/cli.py#L57)), but the modules
> now live flat at `scene_identification/scene_id_cam*.py` (the `cam/` and `cam_camtime/` dirs hold only
> stale `__pycache__`). Those imports will `ModuleNotFoundError` when the subcommands run. Flag to the
> user; fix separately.

### 6. Tests

**Unit — `tests/unit/test_scene_id.py`:**
- Add a reader test alongside
  [`test_from_fmatch_cam_camtime_reads_records_on_camera_grid`](../tests/integration/test_scene_id_runner.py#L160)-style
  cases: `FootprintData.from_fmatch_imager_camtime(make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_CAMTIME))`
  reads onto the 2-D `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid, maps the RBSP inputs including a real
  (mapped) `cloud_phase`, and carries the passthrough identifiers (`latitude`, `camera_pixel_x_min`, …).
- Extend the two `@pytest.mark.parametrize("definition_filename", ["scene_id_imager.yml",
  "scene_id_imager_flash.yml"])` cases ([test_scene_id.py:901](../tests/unit/test_scene_id.py#L901),
  [:910](../tests/unit/test_scene_id.py#L910)) to also include `"scene_id_imager_camtime.yml"` — this
  asserts `scene_id_trmm` is uint16 and the bin-bounds dtypes are correct for the new definition.

**Integration — `tests/integration/test_scene_id_runner.py`:**
- Import the new wrappers and add an end-to-end test modeled on the CAM-CAMTIME cases
  ([test_scene_id_runner.py:254-330](../tests/integration/test_scene_id_runner.py#L254)):
  `run_scene_identification_imager_camtime` → `create_and_write_data_product_imager_camtime` → reopen and
  assert (a) conformance passes, (b) `scene_id_erbe`/`scene_id_unfiltering`/**`scene_id_trmm`** are present
  and hang on `("CAMERA_TIME", "PSEUDOFOOTPRINT")`, (c) the passthrough identifiers survive to the written
  product, and (d) undeclared FMATCH-only vars (e.g. `center_pixel_x`, the raw `era5_*`/`cldpix_*`) are
  dropped.
- Add a `collect_input_files` selection test (keeps only `aux_fmatch_imager_camtime`), mirroring
  [`test_product_mode_keeps_only_matching_product`](../tests/integration/test_scene_id_runner.py#L118).

**Verification the tests must lock in (the one real risk):** confirm the **full TRMM classification runs
on the 2-D grid**. `SCENE-ID-CAM-CAMTIME` already proves ERBE + unfiltering matching and the derived
`surface_type`/`cloud_fraction` calculators work on `(CAMERA_TIME, PSEUDOFOOTPRINT)`; the extra TRMM
derived variables (`surface_wind` = √(u²+v²), `cloud_phase`, `optical_depth`) are element-wise and thus
dimension-agnostic, and TRMM is just another `SceneDefinition` fed through the same matcher. The
integration test above is what guarantees this end-to-end rather than by inspection.

### 7. Notebook & docs (optional, match branch convention)

- The branch ships per-product notebooks (`notebooks/10_scene_id_imager.ipynb`). Optionally add a short
  `SCENE-ID-IMAGER-CAMTIME` walkthrough or a cell in the imager notebook; not required for correctness.
- If a `doc/pr*_summary.md` is produced for this change (as with PR #91/#92), note the new product,
  reader, YAML, runner, and Docker target.

---

## Suggested order

1. Rename/reuse the passthrough tuple + add `from_fmatch_imager_camtime` (§1).
2. Write `scene_id_imager_camtime.yml` (§2).
3. Add `scene_id_imager_camtime.py` (§3).
4. Add the Dockerfile target (§4).
5. Tests (§6); run `pytest -m "not integration" tests/` then the integration subset.
6. `ruff check` + `ruff format`; pre-commit.

## Acceptance checklist

- [ ] `from_fmatch_imager_camtime` reads a `FMATCH-IMAGER-CAMTIME` fixture onto the 2-D grid with real
      `cloud_phase`/`optical_depth`/`surface_wind` inputs and the passthrough identifiers.
- [ ] `scene_id_imager_camtime.yml` validates; a written product passes the conformance check with
      `strict=True`.
- [ ] End-to-end runner produces `scene_id_erbe`/`scene_id_unfiltering`/`scene_id_trmm` (+ all bin bounds)
      **and** the pseudofootprint `latitude`/`longitude`/`camera_pixel_*`/`psf_bbox_*` provenance, all on
      `(CAMERA_TIME, PSEUDOFOOTPRINT)`.
- [ ] `docker build --target scene-id-imager-camtime` succeeds; entrypoint runs a manifest.
- [ ] `ruff check`, `ruff format`, and pre-commit pass; no new TODOs without a JIRA tag.

## Files touched

| File | Change |
|------|--------|
| `libera_utils/scene_identification/scene_id.py` | add `from_fmatch_imager_camtime`; rename passthrough tuple to `_FMATCH_CAMTIME_PASSTHROUGH_VARIABLES` |
| `libera_utils/data/product_definitions/scene_id_imager_camtime.yml` | **new** — imager science on the camtime grid + provenance passthrough |
| `libera_utils/scene_identification/scene_id_imager_camtime.py` | **new** runner module |
| `libera_utils/scene_identification/Dockerfile` | new `scene-id-imager-camtime` target + header comment |
| `tests/unit/test_scene_id.py` | reader test + parametrize the trmm-dtype cases |
| `tests/integration/test_scene_id_runner.py` | end-to-end + input-selection tests |

No change required to `constants.py` (identifiers already exist) or `_runner.py` (already generic).
