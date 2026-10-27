# Shortened real ancillary granules

Small files cut from real ancillary granules, so the footprint matching reader unit tests
run on the formats as distributed. The `fmatch_real_data_path` fixture points here.

Every value in a file is copied unchanged from its source granule. Only the extent and
the variable list are reduced. Where a reader checks a file's grid metadata against its
arrays, that metadata is rewritten to describe the window. Each file was checked when it
was cut: its reader serves the same samples from it as from the source granule. The
exception is cell-center coordinates recomputed from rewritten sinusoidal corners, which
agree to 1e-9 degrees.

| File                                                         | Reader          | Source granule                                                              | Kept                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------ | --------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `era5_single_levels_20250101T00_subset.nc`                   | `era5_surface`  | ERA5 single levels, CDS NetCDF4 download, 2025-01-01T00:00                  | Latitude 50N..40N, longitude 250..260 (110W..100W) on the download's 0..359.75 axis, 41 x 41 cells. All variables (`u10`, `v10`, `d2m`, `t2m`, `sp`, `fal`, `z`) and attributes.                                                                                                                                                         |
| `era5_pressure_levels_20250101T00_subset.nc`                 | `era5_pressure` | ERA5 pressure levels, CDS NetCDF4 download, 2025-01-01T00:00, all 37 levels | The same window, 41 x 41 cells, on the download's -180..179.75 axis. All variables (`t`, `z`, `o3`, `q`, `r`) at all 37 levels.                                                                                                                                                                                                          |
| `AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051_subset.nc` | `viirs_aod`     | `AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc`                       | The full 180 x 360 grid. `Latitude_1D`, `Longitude_1D`, `Aerosol_Optical_Thickness_550_Land_Ocean_Mean`, `Aerosol_Type_Land_Ocean_Mode`, and the global attributes.                                                                                                                                                                      |
| `VJ143C1.A2026153.002.2026161161054_subset.h5`               | `viirs_brdf`    | `VJ143C1.A2026153.002.2026161161054.h5`                                     | Rows 1060..1099 and columns 1140..1179 of the CMG grid (37N..35N, 123W..121W, the Monterey Bay coast), 40 x 40 cells. The nine `BRDF_Albedo_Parameter{1,2,3}_{shortwave,vis,nir}` fields with their attributes, and `lat`/`lon`. `StructMetadata.0` is rewritten: `XDim`/`YDim`, the dimension sizes, the corners, and the field list.   |
| `MCD12Q1.A2024001.h21v07.061.2025205222320_subset.hdf`       | `igbp`          | `MCD12Q1.A2024001.h21v07.061.2025205222320.hdf`                             | Rows 1860..1979 and columns 1680..1799 of the tile (near 12N, 38E), 120 x 120 pixels. `LC_Type1` with its attributes and dimension names, plus `HDFEOSVersion` and the DOI attributes. `StructMetadata.0` is rewritten: `XDim`/`YDim`, the sinusoidal corners, and the field list. `CoreMetadata.0` and `ArchiveMetadata.0` are dropped. |
| `NISE_AMSR2_20260610_subset.HDFEOS`                          | `nise`          | `NISE_AMSR2_20260610.HDFEOS`                                                | Both full 721 x 721 `Extent` grids, Northern first, with their dimension names. The reader fixes the EASE-Grid geometry, so the grids are not cut. The `Age` fields and `age_grid_key` are dropped, and `StructMetadata.0` no longer lists `Age`.                                                                                        |

Sources:

- ERA5: Copernicus Climate Data Store, `reanalysis-era5-single-levels` and
  `reanalysis-era5-pressure-levels`.
- AERDB_D3_VIIRS_NOAA20: NASA LAADS DAAC, Deep Blue Level 3 daily aerosol.
- VJ143C1 and MCD12Q1: NASA LP DAAC.
- NISE: NSIDC, Near-real-time Ice and Snow Extent (NISE), AMSR-2.
