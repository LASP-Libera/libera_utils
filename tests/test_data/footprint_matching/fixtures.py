"""Factory functions for synthetic test fixtures used by footprint matching reader tests.

These factories create minimal in-memory or on-disk representations of real
ancillary data files so that unit tests can run without network access or
proprietary data. The fixtures are intentionally small (e.g., 4 × 4 grids)
while preserving the exact format and encoding of the real files.

Real Source Data — Download Locations
--------------------------------------
IGBP / MODIS MCD12Q1 (HDF4):
    LP DAAC AppEEARS portal: https://appeears.earthdatacloud.nasa.gov/
    LP DAAC Data Pool: https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/
    Example filename: MCD12Q1.A2023001.h09v05.061.hdf
    EarthData login required: https://urs.earthdata.nasa.gov/

NSIDC IMS Sea Ice (ASCII raster):
    NOAA NSIDC FTP: https://noaadata.apps.nsidc.org/NOAA/G02156/24km/
    Example filename: ims2023001_24km_v1.3.asc.gz
    No login required; files are publicly accessible.
    User guide: https://nsidc.org/sites/default/files/g02156-v001-userguide_1_1.pdf

ERA5 Wind (NetCDF4):
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    CDS API script example (Python):
        import cdsapi
        c = cdsapi.Client()
        c.retrieve('reanalysis-era5-single-levels', {
            'product_type': 'reanalysis',
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
            'year': '2023', 'month': '01', 'day': '01',
            'time': '00:00', 'format': 'netcdf',
        }, 'era5_wind_20230101.nc')
    CDS login required: https://cds.climate.copernicus.eu/user/register

VIIRS Cloud (HDF4):
    NOAA CLASS: https://www.avl.class.noaa.gov/saa/products/welcome
    Product: VIIRS Cloud Properties (CLDPX)
    NCEI CDR: https://www.ncei.noaa.gov/products/climate-data-records/cloud-properties-viirs
    NOAA login required for CLASS; NCEI CDR may be publicly accessible.
"""
from __future__ import annotations

import gzip
import pathlib
from pathlib import Path

import numpy as np
import xarray as xr


def make_nsidc_ascii_fixture(
    tmp_path: Path,
    grid_rows: int = 4,
    grid_cols: int = 4,
    data: np.ndarray | None = None,
    gzipped: bool = False,
) -> Path:
    """Write a synthetic IMS ASCII raster file to ``tmp_path``.

    The real IMS 24-km files begin with a short variable-length header (product
    name, dates, grid dimensions) followed by rows of concatenated single-digit
    integer category codes (0–4). This factory produces the same format with a
    minimal 3-line header.

    The real file format reference:
    https://nsidc.org/sites/default/files/g02156-v001-userguide_1_1.pdf

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    grid_rows : int
        Number of data rows. Default 4.
    grid_cols : int
        Number of columns per row. Default 4.
    data : np.ndarray, optional
        Integer array of shape ``(grid_rows, grid_cols)`` with values in 0–4.
        If None, a default checkerboard of 1s and 2s is used.
    gzipped : bool
        If True, write the file as gzip-compressed (suffix ``.asc.gz``).
        Default False (plain text, suffix ``.asc``).

    Returns
    -------
    Path
        Path to the created fixture file.
    """
    if data is None:
        # Checkerboard of ocean (1) and sea ice (2) categories.
        base = np.ones((grid_rows, grid_cols), dtype=np.int8)
        base[1::2, ::2] = 2
        base[::2, 1::2] = 2
        data = base

    # Build file content: header lines followed by data rows.
    # The header here mimics the real IMS header format (see user guide above).
    lines = [
        "IMS_IMAGE\n",
        f"Rows: {grid_rows}, Columns: {grid_cols}\n",
        "End_of_Header\n",
    ]
    for row in data:
        # Each row is concatenated digits with no separator — e.g., "1212\n".
        lines.append("".join(str(int(v)) for v in row) + "\n")

    content = "".join(lines).encode("ascii")

    if gzipped:
        out_path = tmp_path / "ims_fixture_24km.asc.gz"
        with gzip.open(out_path, "wb") as fh:
            fh.write(content)
    else:
        out_path = tmp_path / "ims_fixture_24km.asc"
        out_path.write_bytes(content)

    return out_path


def make_era5_netcdf_fixture(
    tmp_path: Path,
    lat_min: float = 0.0,
    lat_max: float = 2.0,
    lon_min: float = 10.0,
    lon_max: float = 12.0,
    n_lat: int = 4,
    n_lon: int = 4,
    u10_fill: float = 2.5,
    v10_fill: float = -1.5,
) -> Path:
    """Write a synthetic ERA5 NetCDF4 file to ``tmp_path``.

    The real ERA5 files from CDS contain u10 and v10 variables on a global
    lat/lon grid with the latitude dimension in DESCENDING order (90 → -90).
    This factory reproduces that convention in a small grid for testing.

    The real file format reference:
    https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lat_min, lat_max : float
        Latitude range. Latitudes are stored in DESCENDING order in the file
        (matching real ERA5 convention). Default: 0 → 2° N.
    lon_min, lon_max : float
        Longitude range. Default: 10 → 12° E.
    n_lat, n_lon : int
        Number of grid points in each dimension. Default 4.
    u10_fill : float
        Constant fill value for the u10 variable. Default 2.5 m/s.
    v10_fill : float
        Constant fill value for the v10 variable. Default -1.5 m/s.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    # Real ERA5 has latitudes in DESCENDING order; replicate here.
    lats = np.linspace(lat_max, lat_min, n_lat)  # descending
    lons = np.linspace(lon_min, lon_max, n_lon)

    u10_data = np.full((n_lat, n_lon), u10_fill, dtype=np.float32)
    v10_data = np.full((n_lat, n_lon), v10_fill, dtype=np.float32)

    ds = xr.Dataset(
        {
            "u10": xr.DataArray(
                u10_data,
                dims=["latitude", "longitude"],
                attrs={"units": "m s**-1", "long_name": "10 metre U wind component"},
            ),
            "v10": xr.DataArray(
                v10_data,
                dims=["latitude", "longitude"],
                attrs={"units": "m s**-1", "long_name": "10 metre V wind component"},
            ),
        },
        coords={
            "latitude": xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"}),
            "longitude": xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"}),
        },
    )

    out_path = tmp_path / "era5_fixture.nc"
    ds.to_netcdf(out_path)
    return out_path
