# PR #81 — FMATCH 1: Geometry and PSF

`fmatch-1-geometry-psf` → `fmatch-scene-id` (stacked on #80) · +3402 / −2 across 10 files

## Summary

First implementation slice of the footprint-matching subsystem: given an L1B radiometer footprint's geolocation and viewing geometry, compute the lat/lon bounding box of Earth it senses (used downstream to select ancillary tiles). Adds coordinate geometry, a partial CERES PSF, shared FMATCH data types, and a demo notebook.

## New modules (`libera_utils/footprint_matching/`)

| Module | Contents |
|--------|----------|
| **`geometry.py`** | L1B viewing geometry → lat/lon box via WGS84 ECEF ray-trace. Public: `compute_footprint_bounding_box` (single) and `compute_footprint_bounding_boxes` (batched/vectorized), `bounding_box_from_points`, `bounding_box_from_boresight`, `psf_ground_radius_km`, `surface_cell_ecef_km`. Errors: `GeometryError`, `OffLimbError`, `PartialFootprintError` |
| **`psf.py`** | Partial PSF module — analytic CERES PSF `psf_weight` (ATBD Eq. 4.4-1/2) and `psf_95_energy_extent` (95%-energy angular reach, along/cross-scan). CERES stands in until the Libera PSF is delivered; CERES-specific constants isolated for a single-file swap |
| **`types.py`** | Dependency-free shared classes/enums: `OperationalMode`, `FmatchCoverageFlag`, `BoundingBox`, `TileKey`, `GridTile`, `RadiometerFootprint`, `VariableSpec` (+ `spec_active_in_mode`, `with_standard_deviation_companions`) |

## Key behaviors

- Box is a **safe superset** of the footprint — rounds outward; over-covering (extra tiles) is fine, under-covering (dropped data) is not.
- Handles hard cases: Earth curvature, dateline crossings, pole-enclosing footprints, off-limb viewing, and severe-angle truncation (boresight on Earth but box corner off limb → `PartialFootprintError`).
- Footprint stretches along-scan at large viewing zenith angles (tens of km at nadir → hundreds near the limb).

## Packaging

- New optional `fmatch` extras (`pip install libera_utils[fmatch]`): `pyproj`, `pyhdf`, `h5py` — heavy geospatial/HDF stack used only by the gridded readers, imported lazily.
- `pyhdf` requires system HDF4 (`libhdf4-dev`); documented in `pyproject.toml`.
- CI `setup-env` action updated for the extras.

## Notebook & tests

- `notebooks/01_geometry_psf.ipynb` — geometry + PSF walkthrough.
- Unit tests: `tests/unit/test_footprint_matching/test_geometry.py`, `test_psf.py`.
