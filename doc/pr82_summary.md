# PR #82 — FMATCH 2: Readers, Gridded

`fmatch-2-readers-gridded` → `fmatch-1-geometry-psf` (stacked on #81) · +7083 / −0 across 25 files

## Summary

Adds the gridded ancillary-data reader plugin system for footprint matching. A registry of self-registering readers each load a bounding-box region of an external ancillary product (land cover, sea ice, aerosol, BRDF, ERA5, etc.) so a footprint's tile can be filled from many heterogeneous source grids through one interface.

## Architecture (`libera_utils/footprint_matching/readers/`)

| Module | Role |
|--------|------|
| **`base.py`** | Abstract `GriddedDataReader` (Template Method: subclass implements `_load_spatial_region(bbox)`, base provides `load_tile(key)`). Auto-registration via `__init_subclass__` on `READER_KEY`. Exports `TILE_SIZE_DEG` |
| **`registry.py`** | `ReaderRegistry` — readers register on import; `ReaderRegistry.get(name)` / `list_readers()` |
| **`_swath.py`** | Shared helpers to rasterize swath/point products (lat/lon as data variables, not axes — e.g. CERES SSF footprints, CLDPIX pixels) onto a grid |
| **`_hdf4_io.py` / `_hdf5_io.py` / `_pyproj_io.py`** | Low-level I/O helpers (pyhdf, h5py, pyproj reprojection) |

## Readers

| Reader | Product | Format / grid |
|--------|---------|---------------|
| `IGBPReader` | MODIS MCD12Q1 land cover (17 IGBP classes) | HDF4, sinusoidal MODIS tiles |
| `NISEReader` | NSIDC NISE / NISE_A2 sea ice | HDF-EOS4; both EASE-Grid North (EPSG:3408) & South (3409) read and merged into one point cloud |
| `VIIRSAODReader` | NOAA-20 VIIRS Deep Blue aerosol (AERDB_D3) | NetCDF4, 1°×1° global |
| `VIIRSBRDFReader` | VIIRS VJ143C1 BRDF/albedo (9 RTLS vars) | HDF5/HDF-EOS5, 0.05° CMG (lat flipped to ascending) |
| `ERA5Reader` | ECMWF ERA5 single-level surface fields | NetCDF4, ~0.25° |
| `ERA5PressureLevelReader` | ECMWF ERA5 pressure-level fields (up to 37 levels) | NetCDF4, separate CDS dataset |

## Notes

- All readers normalize to ascending latitude for consistency.
- Heavy deps (`pyproj`, `pyhdf`, `h5py`) imported lazily; require the `fmatch` extra (README updated with install + `libhdf4-dev` note).

## Notebook & tests

- `notebooks/02_readers_gridded.ipynb` — gridded reader walkthrough (before/after per reader).
- Per-reader unit tests under `tests/unit/test_footprint_matching/test_readers/` plus base/swath tests; shared synthetic source fixtures in `tests/test_data/footprint_matching/fixtures.py`.
