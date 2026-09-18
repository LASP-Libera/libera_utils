# PR #83 — FMATCH 3: Readers, Swath

`fmatch-3-readers-swath` → `fmatch-2-readers-gridded` (stacked on #82) · +2425 / −6 across 10 files

## Summary

Completes the ancillary reader set with the three CERES/VIIRS swath-and-pixel readers built on the `_swath` rasterization helpers from #82. Registry now lists all nine built-in readers.

## New readers (`libera_utils/footprint_matching/readers/`)

| Reader | Product | Format / layout |
|--------|---------|-----------------|
| `SSFReader` | CERES SSF (+ FLASHFlux) footprint fluxes/clouds | NetCDF4, 1-D `Footprints` dim in thematic groups; per-footprint geolocation (~20 km at nadir). Serves both SSF (→ FMATCH-IMAGER) and FLASHFlux (→ FMATCH-IMAGER-FLASH) — same file format, caller supplies stream |
| `CLDPIXReader` | CERES CLDPIX imager-pixel cloud | NetCDF4, flat 2-D imager swath `(Scanlines, Pixels)`; per-pixel 2-D lat/lon (~1 km) |
| `VIIRSCloudReader` | NOAA-20 VIIRS L3 cloud properties (CLDPROP_D3, Collection 011) | NetCDF4 nested groups, 1°×1° global; `(lon, lat)` layout transposed to `(lat, lon)` |

## Notes

- CERES geolocation conventions normalized here: `Latitude` holds **colatitude** (0° N pole → 180° S pole) and longitude is 0..360.
- Registry (`ReaderRegistry.list_readers()`) now returns all nine: `cldpix`, `era5`, `era5_pressure`, `igbp`, `nise`, `ssf`, `viirs_aod`, `viirs_brdf`, `viirs_cloud`. `__init__` exports `SSFReader`, `CLDPIXReader`, `VIIRSCloudReader`.
- Changelog entry for the full FMATCH reader subsystem added under 5.11.0.

## Notebook & tests

- `notebooks/03_readers_swath.ipynb` — swath reader walkthrough.
- Unit tests for each new reader plus expanded registry tests (`test_cldpix.py`, `test_ssf.py`, `test_viirs.py`, `test_registry.py`).
