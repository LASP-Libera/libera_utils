# Plan: Fold the scene-id Dockerfiles into the root `libera-utils` Dockerfile

## Goal

Delete the two standalone scene-id Dockerfiles and replace them with per-algorithm
**entrypoint stages** in the root `Dockerfile`, exactly mirroring the existing
`libera-utils-make-kernel-jpss` / `libera-utils-make-kernel-azel` pattern. Every algorithm
then ships from **one** multi-stage build sharing a single base layer, selected at build time
with `--target`.

## Current state (why this is the right shape)

The root `Dockerfile` already implements "one image, one entrypoint per algorithm":

```dockerfile
FROM libera-utils AS libera-utils-make-kernel-jpss
ENTRYPOINT ["libera-utils", "make-kernel", "jpss"]

FROM libera-utils AS libera-utils-make-kernel-azel
ENTRYPOINT ["libera-utils", "make-kernel", "azel"]
```

Each is a ~2-line stage inheriting the fully-built `libera-utils` base and overriding only the
`ENTRYPOINT`. The scene-id images do **not** follow this. They are two standalone files
([libera_utils/scene_identification/cam/Dockerfile](../libera_utils/scene_identification/cam/Dockerfile),
[libera_utils/scene_identification/cam_camtime/Dockerfile](../libera_utils/scene_identification/cam_camtime/Dockerfile))
that each **re-build the whole world** from scratch and diverge from the root image in several ways:

| Aspect | Root `Dockerfile` base (`libera-utils`) | Standalone scene-id Dockerfiles |
|--------|-----------------------------------------|---------------------------------|
| Base image | `public.ecr.aws/docker/library/python:3.12-slim` (ECR mirror) | `python:3.11-slim` (Docker Hub) |
| Structure | Multi-stage, shared base | Single-stage, fully duplicated |
| System deps | build-essential, pkg-config, udunits, gdal, **hdf5**, netcdf, python3-dev/numpy, CSPICE | libpq-dev, curl, gcc, pkg-config, **hdf5** |
| Install | `poetry lock && poetry sync --only main` | `poetry install --only main` |
| Entrypoint | `libera-utils` console script + subcommand | `python /opt/.../scene_id_cam.py` (file path) |
| curryer leapsecond env | set (`LEAPSECOND_FILE_ENV`) | not set |

Key facts confirming the fold is safe:

- The scene-id **runner code lives inside the `libera_utils` package**
  ([scene_id_cam.py](../libera_utils/scene_identification/cam/scene_id_cam.py),
  [scene_id_cam_camtime.py](../libera_utils/scene_identification/cam_camtime/scene_id_cam_camtime.py)),
  so the `libera-utils` base stage — which already does `poetry sync --only main` on the whole
  package — installs everything the runners need. **No extra apt or Python deps are required.**
- The runners currently read **CERES SSF / FMATCH** NetCDF inputs (netCDF4 / h5netcdf → HDF5).
  The base already installs `libhdf5-dev` + `libnetcdf-dev`, a superset of the scene-id deps.
  They do **not** use pyhdf/pyproj/HDF4, so no `fmatch`-extra concerns apply (there is no
  `fmatch` extra in `pyproject.toml` on this branch anyway).
- No CI workflow builds these images, and nothing in-repo references the scene-id Dockerfile
  paths except the files themselves. Image build + ECR push is manual
  (`libera-utils ecr-upload <step> <image>`), so the blast radius inside this repo is tiny.

## Decision to confirm: entrypoint style

There is exactly **one** open design choice.

### Option B — CLI subcommands *(recommended)*

Wire the runners into the `libera-utils` CLI as a `scene-id` subcommand group and make the
stages match the make-kernel convention:

```dockerfile
FROM libera-utils AS libera-utils-scene-id-cam
ENTRYPOINT ["libera-utils", "scene-id", "cam"]

FROM libera-utils AS libera-utils-scene-id-cam-camtime
ENTRYPOINT ["libera-utils", "scene-id", "cam-camtime"]
```

- **Pros:** uniform with `make-kernel`; entrypoints are the installed console script (survives
  package moves/renames of the runner module); testable end-to-end via `parse_cli_args`; the
  runners are then discoverable from `libera-utils --help`.
- **Cons:** requires a small CLI addition + a unit test.

### Option A — file-path entrypoint (minimal)

Keep the current script invocation, just relocate the stages into the root Dockerfile:

```dockerfile
FROM libera-utils AS libera-utils-scene-id-cam
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/cam/scene_id_cam.py"]
```

- **Pros:** zero Python changes; entrypoint byte-identical to today.
- **Cons:** perpetuates the inconsistency; entrypoint is a brittle absolute path coupled to the
  package layout; not reachable via the CLI.

**Recommendation: Option B.** It costs ~30 lines of CLI wiring and a test, and leaves every
containerized algorithm on the same `["libera-utils", <subcommand>...]` convention. The rest of
this plan is written for Option B, with Option A callouts where they differ.

## Implementation steps

### 1. (Option B only) Add a `scene-id` subcommand group to the CLI

In [libera_utils/cli.py](../libera_utils/cli.py), add a `scene-id` parser with `cam` and
`cam-camtime` subparsers, mirroring the `make-kernel` block. Each takes a single `manifest`
positional and dispatches to a thin handler. Import the runner modules **lazily inside the
handlers** (not at module top) to keep `libera-utils --version` and the AWS subcommands from
paying the xarray/netCDF import cost.

```python
def _scene_id_cam_cli_handler(args):
    from libera_utils.scene_identification.cam.scene_id_cam import algorithm
    return algorithm(args)  # run_algorithm reads args.manifest

def _scene_id_cam_camtime_cli_handler(args):
    from libera_utils.scene_identification.cam_camtime.scene_id_cam_camtime import algorithm
    return algorithm(args)
```

```python
scene_id_parser = subparsers.add_parser("scene-id", help="run a Libera SCENE-ID algorithm from a manifest")
scene_id_subparsers = scene_id_parser.add_subparsers(description="sub-commands for scene-id sub-command")

cam_parser = scene_id_subparsers.add_parser("cam", help="run SCENE-ID-CAM (radiometer timescale)")
cam_parser.set_defaults(func=_scene_id_cam_cli_handler)
cam_parser.add_argument("manifest", type=str, help="path to the input manifest file")

cam_camtime_parser = scene_id_subparsers.add_parser("cam-camtime", help="run SCENE-ID-CAM-CAMTIME (camera timescale)")
cam_camtime_parser.set_defaults(func=_scene_id_cam_camtime_cli_handler)
cam_camtime_parser.add_argument("manifest", type=str, help="path to the input manifest file")
```

Notes:
- Both runners' existing `algorithm(...)` already accept the argparse `Namespace` (they forward
  to `run_algorithm`, which reads `.manifest`), so no signature changes are needed. The runners'
  own `main()` / `__main__` blocks can stay for backward compatibility (Option A fallback) or be
  removed later — leaving them costs nothing.
- `PROCESSING_PATH` is read at runtime by `_runner.run_algorithm`; nothing to change.

### 2. Append the entrypoint stages to the root `Dockerfile`

After the make-kernel stages in [Dockerfile](../Dockerfile):

```dockerfile
# CLI for the Libera SCENE-ID-CAM algorithm (radiometer timescale) from a manifest.
# ---------------------------------------------------------------------------------
FROM libera-utils AS libera-utils-scene-id-cam

ENTRYPOINT ["libera-utils", "scene-id", "cam"]


# CLI for the Libera SCENE-ID-CAM-CAMTIME algorithm (camera timescale) from a manifest.
# -------------------------------------------------------------------------------------
FROM libera-utils AS libera-utils-scene-id-cam-camtime

ENTRYPOINT ["libera-utils", "scene-id", "cam-camtime"]
```

(Option A: same two stages, but with the `python .../scene_id_*.py` entrypoints instead.)

### 3. Delete the standalone Dockerfiles

- `rm libera_utils/scene_identification/cam/Dockerfile`
- `rm libera_utils/scene_identification/cam_camtime/Dockerfile`

### 4. (Optional) Add compose services for local build/run

In [docker-compose.yml](../docker-compose.yml), add services targeting the new stages so
developers can `docker compose build scene-id-cam` locally, consistent with the existing `sdp`
service:

```yaml
  scene-id-cam:
    platform: linux/amd64
    image: libera-utils-scene-id-cam:latest
    build:
      context: .
      target: libera-utils-scene-id-cam

  scene-id-cam-camtime:
    platform: linux/amd64
    image: libera-utils-scene-id-cam-camtime:latest
    build:
      context: .
      target: libera-utils-scene-id-cam-camtime
```

### 5. Tests

- **Unit** (Option B): extend [tests/unit/test_cli.py](../tests/unit/test_cli.py) to assert
  `parse_cli_args(["scene-id", "cam", "m.json"])` sets `func` to the CAM handler and
  `manifest == "m.json"` (and likewise for `cam-camtime`). Optionally assert the handler calls
  `algorithm` via a mock. This is the standard pattern already used for other subcommands.
- **Integration**: [tests/integration/test_scene_id_runner.py](../tests/integration/test_scene_id_runner.py)
  exercises the runner directly and is unaffected.
- **Manual smoke** (not CI): build and run each target locally —
  `docker build --target libera-utils-scene-id-cam -t sid-cam .` then
  `docker run -e PROCESSING_PATH=/out -v ... sid-cam /path/to/manifest.json`.

### 6. Run local quality gates

`ruff check && ruff format`, then `pytest -m "not integration" tests/`, plus the manual docker
smoke above. Pre-commit hooks as usual (never `--no-verify`).

## Build & deploy impact (coordinate outside this repo)

The ECR push path is unchanged in code — `libera-utils ecr-upload aux-scene-id-cam <image>:<tag>`
still works, because `aux_scene_id_cam` / `aux_scene_id_cam_camtime` are already valid
`ProcessingStepIdentifier` choices and `ecr_name` → `aux-scene-id-cam-docker-repo` as before.

**What changes for whoever builds the images** (SDC deployment tooling / runbooks, external to
this repo):

- Old: `docker build -f libera_utils/scene_identification/cam/Dockerfile -t <img> .`
  (build target implicitly `scene-id-cam`).
- New: `docker build --target libera-utils-scene-id-cam -t <img> .`
  (single root Dockerfile; **stage name gains the `libera-utils-` prefix** to match convention).

Action item: audit the SDC build/deploy scripts (outside this repo) for references to the old
Dockerfile paths and the old target names `scene-id-cam` / `scene-id-cam-camtime`, and switch
them to `--target libera-utils-scene-id-cam[-camtime]`. If keeping the old bare target names
matters for those scripts, that's a reason to name the stages `scene-id-cam` /
`scene-id-cam-camtime` instead — but the make-kernel precedent favors the `libera-utils-` prefix.

## Risks & mitigations

1. **Python 3.11 → 3.12.** The base defaults to `BASE_IMAGE_PYTHON_VERSION=3.12`; the scene-id
   images pinned 3.11. The package supports `>=3.11,<4` and the test matrix covers 3.11–3.14, so
   3.12 is expected to be fine. Mitigation: the manual docker smoke run validates it; if 3.11
   parity is required for any reason, build with `--build-arg BASE_IMAGE_PYTHON_VERSION=3.11`.
2. **Larger image.** The shared base adds CSPICE + gdal/udunits layers the scene-id images did
   not have. This is the accepted cost of one-image consolidation (identical to what
   make-kernel images already carry). If image size later matters, factor a lighter shared base;
   out of scope here.
3. **`libpq-dev` dropped.** The old scene-id Dockerfiles installed `libpq-dev` "for psycopg2".
   The base image's `poetry sync --only main` already succeeds without it, so psycopg2 is either
   not a resolved main dep or ships as a manylinux wheel. Mitigation: the docker smoke run
   imports/executes the runner; if any psycopg2 link error appears, add `libpq-dev` to the base
   `apt-get install` list (one line).

## File-change summary

| File | Change |
|------|--------|
| `libera_utils/cli.py` | *(Option B)* add `scene-id` subparser group + 2 lazy handlers |
| `Dockerfile` | add `libera-utils-scene-id-cam` + `-camtime` entrypoint stages |
| `libera_utils/scene_identification/cam/Dockerfile` | **delete** |
| `libera_utils/scene_identification/cam_camtime/Dockerfile` | **delete** |
| `docker-compose.yml` | *(optional)* add 2 build services |
| `tests/unit/test_cli.py` | *(Option B)* add scene-id subcommand parse/dispatch tests |
```
