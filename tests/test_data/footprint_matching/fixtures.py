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

NISE Sea Ice (HDF-EOS4):
    NSIDC HTTPS: https://n5eil01u.ecs.nsidc.org/NISE/
    Example filename: NISE_SSMISF18_20260115.HDFEOS
    EarthData login required: https://urs.earthdata.nasa.gov/

ERA5 Wind (NetCDF4):
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register

VIIRS Cloud CLDPROP_D3 (NetCDF4 with groups):
    NCEI CDR: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
    Example filename: CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc

VIIRS BRDF VJ143C1 (HDF5/HDF-EOS5):
    LP DAAC: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/
    Example filename: VJ143C1.A2026153.002.2026161161054.h5
    EarthData login required: https://urs.earthdata.nasa.gov/
"""
from __future__ import annotations

import pathlib
from pathlib import Path

import numpy as np
import xarray as xr


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


def make_era5_valid_time_fixture(
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
    """Write a synthetic ERA5 NetCDF4 file with a ``valid_time`` time dimension.

    Identical to :func:`make_era5_netcdf_fixture` except that the u10/v10
    variables have an extra ``valid_time`` dimension of length 1, matching the
    format produced by the new CDS API. This exercises the reader's time-dim
    detection logic which uses a substring match on ``"time"``.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lat_min, lat_max : float
        Latitude range (DESCENDING in file). Default: 0 → 2° N.
    lon_min, lon_max : float
        Longitude range. Default: 10 → 12° E.
    n_lat, n_lon : int
        Number of grid points in each dimension. Default 4.
    u10_fill : float
        Constant fill value for u10. Default 2.5 m/s.
    v10_fill : float
        Constant fill value for v10. Default -1.5 m/s.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    lats = np.linspace(lat_max, lat_min, n_lat)  # descending
    lons = np.linspace(lon_min, lon_max, n_lon)

    # Shape: (1, n_lat, n_lon) with a leading valid_time dimension.
    u10_data = np.full((1, n_lat, n_lon), u10_fill, dtype=np.float32)
    v10_data = np.full((1, n_lat, n_lon), v10_fill, dtype=np.float32)

    ds = xr.Dataset(
        {
            "u10": xr.DataArray(
                u10_data,
                dims=["valid_time", "latitude", "longitude"],
                attrs={"units": "m s**-1", "long_name": "10 metre U wind component"},
            ),
            "v10": xr.DataArray(
                v10_data,
                dims=["valid_time", "latitude", "longitude"],
                attrs={"units": "m s**-1", "long_name": "10 metre V wind component"},
            ),
        },
        coords={
            "valid_time": xr.DataArray(
                np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
                dims=["valid_time"],
            ),
            "latitude": xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"}),
            "longitude": xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"}),
        },
    )

    out_path = tmp_path / "era5_valid_time_fixture.nc"
    ds.to_netcdf(out_path)
    return out_path


def make_viirs_cloud_d3_fixture(
    tmp_path: Path,
    n_lat: int = 4,
    n_lon: int = 8,
    lat_min: float = 0.5,
    lat_max: float = 3.5,
    lon_min: float = 10.5,
    lon_max: float = 17.5,
    cf_fill: float = 0.6,
    cot_fill: float = 4.0,
    ctp_fill: float = 700.0,
) -> Path:
    """Write a synthetic CLDPROP_D3 VIIRS cloud properties NetCDF4 file.

    Replicates the group structure of the real CLDPROP_D3 product:
    - Root-level ``latitude`` (n_lat,) and ``longitude`` (n_lon,) coordinate arrays
    - Three groups with a ``Mean`` variable each: ``Cloud_Fraction``,
      ``Cloud_Optical_Thickness_Combined``, ``Cloud_Top_Pressure``
    - **Variable dimension order is (longitude, latitude)** — transposed from
      the conventional (lat, lon) order — to match the real product format and
      exercise the transpose logic in ``VIIRSCloudReader``.
    - Fill values (−9999.0) are NOT used by default; callers can pass a data
      array with −9999.0 elements to test fill handling.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    n_lat, n_lon : int
        Grid dimensions. Default 4 × 8.
    lat_min, lat_max : float
        Latitude range for coordinate array (ascending). Default 0.5 → 3.5°.
    lon_min, lon_max : float
        Longitude range for coordinate array. Default 10.5 → 17.5°.
    cf_fill : float
        Constant fill value for cloud fraction. Default 0.6.
    cot_fill : float
        Constant fill value for cloud optical thickness. Default 4.0.
    ctp_fill : float
        Constant fill value for cloud top pressure. Default 700.0 hPa.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    import netCDF4  # noqa: PLC0415

    out_path = tmp_path / "cldprop_d3_fixture.nc"
    lats = np.linspace(lat_min, lat_max, n_lat)  # ascending
    lons = np.linspace(lon_min, lon_max, n_lon)

    with netCDF4.Dataset(str(out_path), "w") as ds:
        # Root-level dimensions and coordinate variables.
        ds.createDimension("latitude", n_lat)
        ds.createDimension("longitude", n_lon)

        lat_var = ds.createVariable("latitude", "f4", ("latitude",))
        lat_var[:] = lats.astype(np.float32)
        lat_var.units = "degrees_north"

        lon_var = ds.createVariable("longitude", "f4", ("longitude",))
        lon_var[:] = lons.astype(np.float32)
        lon_var.units = "degrees_east"

        # Data variables in groups with (longitude, latitude) dimension order.
        group_specs = [
            ("Cloud_Fraction", cf_fill),
            ("Cloud_Optical_Thickness_Combined", cot_fill),
            ("Cloud_Top_Pressure", ctp_fill),
        ]
        for grp_name, fill in group_specs:
            grp = ds.createGroup(grp_name)
            mean_var = grp.createVariable("Mean", "f4", ("longitude", "latitude"))
            # Fill with the constant value; shape is (n_lon, n_lat) in file.
            mean_var[:] = np.full((n_lon, n_lat), fill, dtype=np.float32)

    return out_path


def make_viirs_brdf_hdf5_fixture(
    tmp_path: Path,
    n_lat: int = 4,
    n_lon: int = 8,
    lat_min: float = 0.05,
    lat_max: float = 0.20,
    lon_min: float = 10.05,
    lon_max: float = 10.40,
    param_fill: int = 200,
    fill_sentinel: int = 32767,
) -> Path:
    """Write a synthetic VJ143C1 HDF5 BRDF fixture file.

    Replicates the HDF-EOS5 group structure of the real VJ143C1 product:
    - Group path ``HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/``
    - Coordinate arrays ``lat`` (n_lat,) in **descending** order (90 → -90,
      matching the real product) and ``lon`` (n_lon,) in ascending order
    - Nine int16 BRDF parameter datasets (3 bands × 3 kernel weights) with
      ``scale_factor=0.001`` and ``_FillValue=32767`` attributes

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    n_lat, n_lon : int
        Grid dimensions. Default 4 × 8.
    lat_min, lat_max : float
        Latitude range (stored in DESCENDING order in file). Default 0.05–0.20°.
    lon_min, lon_max : float
        Longitude range. Default 10.05–10.40°.
    param_fill : int
        Raw int16 fill value for all 9 BRDF parameter datasets.
        Default 200 (scales to 0.200 via scale_factor=0.001).
    fill_sentinel : int
        int16 fill sentinel written to the ``_FillValue`` attribute and used to
        mark NaN pixels in the last pixel of each dataset. Default 32767.

    Returns
    -------
    Path
        Path to the created HDF5 fixture file.
    """
    import h5py  # noqa: PLC0415

    out_path = tmp_path / "vj143c1_fixture.h5"

    # Latitude stored DESCENDING in real files (90 → -90).
    lats_desc = np.linspace(lat_max, lat_min, n_lat)[::-1]  # descending
    lons_asc = np.linspace(lon_min, lon_max, n_lon)

    field_names = [
        "BRDF_Albedo_Parameter1_shortwave",
        "BRDF_Albedo_Parameter2_shortwave",
        "BRDF_Albedo_Parameter3_shortwave",
        "BRDF_Albedo_Parameter1_vis",
        "BRDF_Albedo_Parameter2_vis",
        "BRDF_Albedo_Parameter3_vis",
        "BRDF_Albedo_Parameter1_nir",
        "BRDF_Albedo_Parameter2_nir",
        "BRDF_Albedo_Parameter3_nir",
    ]

    with h5py.File(str(out_path), "w") as f:
        grp = f.require_group("HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields")

        lat_ds = grp.create_dataset("lat", data=lats_desc.astype(np.float64))
        lat_ds.attrs["units"] = "degrees_north"

        lon_ds = grp.create_dataset("lon", data=lons_asc.astype(np.float64))
        lon_ds.attrs["units"] = "degrees_east"

        for name in field_names:
            raw = np.full((n_lat, n_lon), param_fill, dtype=np.int16)
            # Make the last element a fill sentinel so tests can verify NaN handling.
            raw[-1, -1] = fill_sentinel
            ds_obj = grp.create_dataset(name, data=raw)
            ds_obj.attrs["scale_factor"] = np.float32(0.001)
            ds_obj.attrs["_FillValue"] = np.int16(fill_sentinel)

    return out_path
