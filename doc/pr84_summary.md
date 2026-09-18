# PR #84 — FMATCH 4: Tile Management

`fmatch-4-tiling` → `fmatch-3-readers-swath` (stacked on #83) · +2195 / −0 across 3 files

## Summary

Adds `TileManager`, the data-access layer between footprint bounding boxes and the ancillary readers (design doc §2.8.1.3). Resolves a `BoundingBox` to the 2° tiles it touches, caches loaded tiles, merges multi-tile requests into one contiguous grid, and optionally prefetches upcoming footprints' tiles.

## Why

- Orchestrator processes ~1,200 footprints/hour; each may touch >1,000 ancillary pixels from datasets of hundreds of MB to tens of GB.
- Footprints are processed in along-track order → adjacent footprints need the same tiles → LRU cache turns most repeat requests into memory hits.

## Behavior (`libera_utils/footprint_matching/tiling.py`)

| Concern | Handling |
|---------|----------|
| **Key resolution** | `BoundingBox` → set of 2° `TileKey`s; dateline-wrapping boxes split into two sub-requests |
| **Caching** | Byte-budgeted LRU (`_LRUTileCache` on `OrderedDict`) of loaded `GridTile`s; separate merge-result cache for footprints resolving to the same tile set |
| **Merging** | Multi-tile requests merged into one contiguous grid (`_union_bounds`) |
| **Prefetch** | Optional background worker prefetches upcoming footprints' tiles to overlap reader I/O with main-thread compute |
| **Thread safety** | All raw reader loads serialized behind one global lock — native netCDF4/h5py stack is not thread-safe |

Public API: `TileManager`, `build_tile_manager()`, `gather_footprint_tiles()`.

## Notebook & tests

- `notebooks/04_tiling.ipynb` — key resolution with tile overlay, multi-tile merges, dateline split, cache-hit behavior along a synthetic ground track, prefetch warm-up.
- Unit tests: `tests/unit/test_footprint_matching/test_tiling.py`.
