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

ERA5 single-level fields (NetCDF4):
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register

ERA5 pressure-level fields (NetCDF4):
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register

VIIRS Cloud CLDPROP_D3 (NetCDF4 with groups):
    NCEI CDR: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
    Example filename: CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc

VIIRS BRDF VJ143C1 (HDF5/HDF-EOS5):
    LP DAAC: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/
    Example filename: VJ143C1.A2026153.002.2026161161054.h5
    EarthData login required: https://urs.earthdata.nasa.gov/

VIIRS Deep Blue aerosol — AOD + type (AERDB_D3_VIIRS_NOAA20, NetCDF4, root-level):
    NASA Deep Blue: https://deepblue.gsfc.nasa.gov/
    Example filename: AERDB_D3_VIIRS_NOAA20.A2026001.002.2026005001030.nc

CERES SSF / FLASHFlux (NetCDF4, per-footprint swath):
    NASA CERES: https://ceres.larc.nasa.gov/data/#ssf-level-2
    Example filename: CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc

CERES CLDPIX (NetCDF4, imager-pixel swath):
    NASA CERES: https://ceres.larc.nasa.gov/data/
    Example filename: CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    # Imported for type annotations only; the concrete classes are imported lazily inside the factories that use
    # them (product.py pulls in the heavier footprint-matching stack) to keep fixture import cheap.
    from libera_utils.footprint_matching.product import OperationalMode

# Constant fill values for the five non-wind ERA5 single-level variables written
# by the ERA5 fixtures. Chosen to be physically plausible AND mutually distinct
# so tests can assert that each output layer came from the right file variable.
# Winds keep their historical u10_fill/v10_fill keyword parameters.
ERA5_SINGLE_LEVEL_FILLS: dict[str, float] = {
    "t2m": 285.0,  # 2 m temperature (K)
    "d2m": 280.0,  # 2 m dewpoint temperature (K)
    "sp": 101325.0,  # surface pressure (Pa)
    "z": 500.0,  # surface geopotential (m^2/s^2)
    "fal": 0.3,  # forecast albedo (0-1)
}

# Metadata written on each single-level fixture variable, mirroring real CDS files.
_ERA5_SINGLE_LEVEL_ATTRS: dict[str, dict[str, str]] = {
    "u10": {"units": "m s**-1", "long_name": "10 metre U wind component"},
    "v10": {"units": "m s**-1", "long_name": "10 metre V wind component"},
    "t2m": {"units": "K", "long_name": "2 metre temperature"},
    "d2m": {"units": "K", "long_name": "2 metre dewpoint temperature"},
    "sp": {"units": "Pa", "long_name": "Surface pressure"},
    "z": {"units": "m**2 s**-2", "long_name": "Geopotential"},
    "fal": {"units": "(0 - 1)", "long_name": "Forecast albedo"},
}


def _era5_single_level_dataset(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat: int,
    n_lon: int,
    u10_fill: float,
    v10_fill: float,
    with_valid_time: bool,
) -> xr.Dataset:
    """Build the in-memory Dataset shared by the two single-level fixture makers.

    Contains all seven single-level variables the ERA5Reader expects (u10, v10,
    t2m, d2m, sp, z, fal), each a constant-valued grid, with latitudes stored
    DESCENDING (real ERA5 convention). When ``with_valid_time`` is True, every
    variable gains a leading length-1 ``valid_time`` dimension, matching files
    produced by the new CDS API.
    """
    lats = np.linspace(lat_max, lat_min, n_lat)  # descending, real ERA5 convention
    lons = np.linspace(lon_min, lon_max, n_lon)

    fills = {"u10": u10_fill, "v10": v10_fill, **ERA5_SINGLE_LEVEL_FILLS}
    shape = (1, n_lat, n_lon) if with_valid_time else (n_lat, n_lon)
    dims = ["valid_time", "latitude", "longitude"] if with_valid_time else ["latitude", "longitude"]

    coords = {
        "latitude": xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"}),
        "longitude": xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"}),
    }
    if with_valid_time:
        coords["valid_time"] = xr.DataArray(
            np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            dims=["valid_time"],
        )

    return xr.Dataset(
        {
            name: xr.DataArray(
                np.full(shape, fill, dtype=np.float32),
                dims=dims,
                attrs=_ERA5_SINGLE_LEVEL_ATTRS[name],
            )
            for name, fill in fills.items()
        },
        coords=coords,
    )


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
    """Write a synthetic ERA5 single-levels NetCDF4 file to ``tmp_path``.

    The real ERA5 files from CDS contain the single-level variables on a global
    lat/lon grid with the latitude dimension in DESCENDING order (90 → -90).
    This factory reproduces that convention in a small grid for testing, writing
    all seven variables the ERA5Reader expects (u10, v10, t2m, d2m, sp, z, fal).
    The non-wind variables carry the constant ``ERA5_SINGLE_LEVEL_FILLS`` values.

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
    ds = _era5_single_level_dataset(
        lat_min, lat_max, lon_min, lon_max, n_lat, n_lon, u10_fill, v10_fill, with_valid_time=False
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
    """Write a synthetic ERA5 single-levels NetCDF4 file with a ``valid_time`` dimension.

    Identical to :func:`make_era5_netcdf_fixture` except that every variable has
    an extra ``valid_time`` dimension of length 1, matching the format produced
    by the new CDS API. This exercises the reader's time-dim detection logic
    which uses a substring match on ``"time"``.

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
    ds = _era5_single_level_dataset(
        lat_min, lat_max, lon_min, lon_max, n_lat, n_lon, u10_fill, v10_fill, with_valid_time=True
    )
    out_path = tmp_path / "era5_valid_time_fixture.nc"
    ds.to_netcdf(out_path)
    return out_path


def era5_pressure_fixture_value(var_index: int, level: float) -> float:
    """Deterministic fixture value for pressure-level variable ``var_index`` at ``level``.

    ``var_index`` is the position in ``_ERA5_PRESSURE_LEVEL_VARIABLES`` (0 = t,
    1 = z, ...). The formula keeps every (variable, level) layer distinct so
    tests can assert exact layer ordering in the stacked output array.
    """
    return float(var_index * 10000 + level)


def make_era5_pressure_netcdf_fixture(
    tmp_path: Path,
    lat_min: float = 0.0,
    lat_max: float = 2.0,
    lon_min: float = 10.0,
    lon_max: float = 12.0,
    n_lat: int = 4,
    n_lon: int = 4,
    levels: tuple[int, ...] | None = None,
    with_valid_time: bool = True,
) -> Path:
    """Write a synthetic ERA5 pressure-levels NetCDF4 file to ``tmp_path``.

    Replicates the structure of a real CDS pressure-levels download: variables
    ``t``, ``z``, ``o3``, ``q``, ``r`` on ``(valid_time, pressure_level,
    latitude, longitude)`` with latitudes DESCENDING. Each (variable, level)
    layer is filled with the deterministic constant
    :func:`era5_pressure_fixture_value` so tests can verify exact layer ordering
    in the reader's stacked output.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lat_min, lat_max : float
        Latitude range (stored DESCENDING). Default: 0 → 2° N.
    lon_min, lon_max : float
        Longitude range. Default: 10 → 12° E.
    n_lat, n_lon : int
        Number of grid points in each dimension. Default 4.
    levels : tuple[int, ...] | None
        Pressure levels (hPa) to write, stored descending like real CDS files
        (1000 first). Defaults to the reader's configured ``_ERA5_PRESSURE_LEVELS``
        so the fixture always satisfies the reader; pass a subset to exercise the
        missing-level error path.
    with_valid_time : bool
        Write the leading length-1 ``valid_time`` dimension (new CDS API format).
        Default True; False writes plain (pressure_level, latitude, longitude)
        variables.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    # Import here so the fixtures module does not require the reader at import
    # time (and so the default always tracks the reader's configured levels).
    from libera_utils.footprint_matching.readers.era5_pressure import (  # noqa: PLC0415
        _ERA5_PRESSURE_LEVEL_VARIABLES,
        _ERA5_PRESSURE_LEVELS,
    )

    if levels is None:
        levels = _ERA5_PRESSURE_LEVELS
    # Real CDS files store pressure levels descending (1000 → 1); replicate that
    # to prove the reader selects by value, not by position.
    stored_levels = np.array(sorted(levels, reverse=True), dtype=np.float64)

    lats = np.linspace(lat_max, lat_min, n_lat)  # descending, real ERA5 convention
    lons = np.linspace(lon_min, lon_max, n_lon)

    n_lev = stored_levels.size
    shape = (1, n_lev, n_lat, n_lon) if with_valid_time else (n_lev, n_lat, n_lon)
    dims = (
        ["valid_time", "pressure_level", "latitude", "longitude"]
        if with_valid_time
        else ["pressure_level", "latitude", "longitude"]
    )

    data_vars = {}
    for var_index, (_, nc_name) in enumerate(_ERA5_PRESSURE_LEVEL_VARIABLES):
        data = np.empty(shape, dtype=np.float32)
        for level_index, level in enumerate(stored_levels):
            value = era5_pressure_fixture_value(var_index, float(level))
            if with_valid_time:
                data[0, level_index, :, :] = value
            else:
                data[level_index, :, :] = value
        data_vars[nc_name] = xr.DataArray(data, dims=dims)

    coords = {
        "pressure_level": xr.DataArray(stored_levels, dims=["pressure_level"], attrs={"units": "hPa"}),
        "latitude": xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"}),
        "longitude": xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"}),
    }
    if with_valid_time:
        coords["valid_time"] = xr.DataArray(
            np.array(["2026-01-01T00:00:00"], dtype="datetime64[ns]"),
            dims=["valid_time"],
        )

    ds = xr.Dataset(data_vars, coords=coords)
    out_path = tmp_path / "era5_pressure_fixture.nc"
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

    # Latitude stored DESCENDING in real files (90 → -90); linspace(max, min)
    # is already descending, so the reader's flip-to-ascending branch is exercised.
    lats_desc = np.linspace(lat_max, lat_min, n_lat)  # descending
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


def make_aod_noaa20_fixture(
    tmp_path: Path,
    n_lat: int = 4,
    n_lon: int = 8,
    lat_min: float = 0.5,
    lat_max: float = 3.5,
    lon_min: float = 10.5,
    lon_max: float = 17.5,
    aod_fill: float = 0.2,
    aerosol_type_value: int = 1,
    include_fill_pixel: bool = True,
    write_histogram: bool = True,
) -> Path:
    """Write a synthetic AERDB_D3_VIIRS_NOAA20 Deep Blue aerosol NetCDF4 file.

    Replicates the relevant structure of the real single-sensor NOAA-20 VIIRS
    Deep Blue Level-3 daily product as consumed by ``VIIRSAODReader`` — all
    variables at the **root** (no groups), stored in (Latitude, Longitude) order
    with ascending latitude (no transpose needed):
    - Root ``Latitude_1D`` (ascending) and ``Longitude_1D`` coordinate arrays.
    - Root ``Aerosol_Optical_Thickness_550_Land_Ocean_Mean`` (AOD @ 550 nm),
      ``_FillValue = -999.0``.
    - Root ``Aerosol_Type_Land_Ocean_Mode`` (modal aerosol type, ``int32``,
      ``_FillValue = -999``, categories 0..7).
    - Optionally root ``Aerosol_Type_Land_Ocean_Histogram`` (per-type counts)
      for realism; the reader does not read it.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    n_lat, n_lon : int
        Grid dimensions. Default 4 × 8.
    lat_min, lat_max : float
        Latitude range (ascending). Default 0.5 → 3.5°.
    lon_min, lon_max : float
        Longitude range (−180..180 convention). Default 10.5 → 17.5°.
    aod_fill : float
        Constant AOD value written to every non-fill pixel. Default 0.2.
    aerosol_type_value : int
        Constant aerosol-type category (0..7) written to every non-fill pixel.
        Default 1 (smoke).
    include_fill_pixel : bool
        If True, set pixel [0, 0] of both the AOD and the aerosol-type fields to
        the −999 fill sentinel so tests can verify fill → NaN conversion.
        Default True.
    write_histogram : bool
        If True, additionally write the 8-category ``Aerosol_Type_Land_Ocean_Histogram``
        variable. The reader ignores it; it is present only for structural
        fidelity. Default True.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    import netCDF4  # noqa: PLC0415

    n_types = 8
    out_path = tmp_path / "aerdb_d3_viirs_noaa20_fixture.nc"
    lats = np.linspace(lat_min, lat_max, n_lat)  # ascending
    lons = np.linspace(lon_min, lon_max, n_lon)

    with netCDF4.Dataset(str(out_path), "w") as ds:
        # Note the ``_1D`` coordinate names — they differ from the merged product.
        ds.createDimension("Latitude_1D", n_lat)
        ds.createDimension("Longitude_1D", n_lon)

        lat_var = ds.createVariable("Latitude_1D", "f4", ("Latitude_1D",))
        lat_var[:] = lats.astype(np.float32)
        lat_var.units = "degrees_north"

        lon_var = ds.createVariable("Longitude_1D", "f4", ("Longitude_1D",))
        lon_var[:] = lons.astype(np.float32)
        lon_var.units = "degrees_east"

        # AOD daily-mean field at the root, (Latitude, Longitude) order.
        aod_var = ds.createVariable(
            "Aerosol_Optical_Thickness_550_Land_Ocean_Mean",
            "f4",
            ("Latitude_1D", "Longitude_1D"),
            fill_value=-999.0,
        )
        aod_data = np.full((n_lat, n_lon), aod_fill, dtype=np.float32)
        if include_fill_pixel:
            aod_data[0, 0] = -999.0
        aod_var[:] = aod_data

        # Modal aerosol-type field at the root (int32 in the real product).
        type_var = ds.createVariable(
            "Aerosol_Type_Land_Ocean_Mode",
            "i4",
            ("Latitude_1D", "Longitude_1D"),
            fill_value=-999,
        )
        type_data = np.full((n_lat, n_lon), aerosol_type_value, dtype=np.int32)
        if include_fill_pixel:
            type_data[0, 0] = -999
        type_var[:] = type_data
        type_var.valid_range = np.array([0, n_types - 1], dtype=np.int32)

        # Optional per-type histogram (Aerosol_Types, Latitude, Longitude). Not
        # read by the reader; written only to mirror the real granule's layout.
        if write_histogram:
            ds.createDimension("Aerosol_Types", n_types)
            hist_var = ds.createVariable(
                "Aerosol_Type_Land_Ocean_Histogram",
                "i4",
                ("Aerosol_Types", "Latitude_1D", "Longitude_1D"),
                fill_value=-999,
            )
            hist = np.zeros((n_types, n_lat, n_lon), dtype=np.int32)
            # Put all "counts" in the modal category so the histogram is
            # self-consistent. Guarded so tests can pass an out-of-range modal
            # value (to exercise fill/valid-range masking) without indexing error.
            if 0 <= aerosol_type_value < n_types:
                hist[aerosol_type_value, :, :] = 10
            hist_var[:] = hist

    return out_path


def make_ssf_fixture(
    tmp_path: Path,
    lats: np.ndarray | None = None,
    lons_0360: np.ndarray | None = None,
    aerosol_optical_depth: np.ndarray | None = None,
    clear_coverage: np.ndarray | None = None,
    cloud_optical_depth_lower: np.ndarray | None = None,
    cloud_water_particle_radius_lower: np.ndarray | None = None,
    cloud_ice_particle_radius_lower: np.ndarray | None = None,
    cloud_classification: np.ndarray | None = None,
    shortwave_adm_type: np.ndarray | None = None,
    longwave_adm_type: np.ndarray | None = None,
) -> Path:
    """Write a synthetic CERES SSF (footprint/swath) NetCDF4 file.

    Replicates the grouped, per-footprint structure of the real SSF product:
    - 1-D ``Footprints`` dimension, a ``LowerUpper`` dimension of length 2, and the
      ``AeroTypePct`` (7) secondary axis the reader flattens
    - ``Time_and_Position/instrument_fov_latitude`` and ``…_longitude``
      (**longitude stored in the 0..360 convention**, matching the real file)
    - One variable per supported reader field across the corresponding groups,
      with float fill ``3.4028235e38`` and int16 fill ``32767``. This includes the
      FMATCH-IMAGER-only extended fields: the layered cloud fields (both layers,
      upper offset from lower), ``Assimilated_Aerosol_Properties`` (``match_aot``,
      ``aerosol_type_percentage``), the land/ocean 0.55 µm imager AODs,
      ``Auxillary_Properties/surface_albedo`` (1-D, 0..1), and
      ``Observed_TOA_Fluxes/toa_incoming_solar_radiation`` (1-D, W/m²).

    All arrays default to a small deterministic set of footprints clustered near
    lat ≈ 10–11°, lon ≈ −10° (written as 350° in the 0..360 file convention) so
    tests can verify longitude normalization and rasterization. Pass explicit
    arrays to override any field.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lats : np.ndarray, optional
        Per-footprint latitudes (−90..90).
    lons_0360 : np.ndarray, optional
        Per-footprint longitudes in the **0..360** convention.
    aerosol_optical_depth, clear_coverage, cloud_optical_depth_lower : np.ndarray, optional
        Per-footprint continuous values. ``cloud_optical_depth_lower`` fills the
        lower (index 0) layer of the 2-D ``cloud_optical_depth_mean`` variable.
    cloud_water_particle_radius_lower : np.ndarray, optional
        Per-footprint water cloud particle effective radius (μm) for the lower
        layer (index 0) of ``cloud_water_particle_radius_37um_mean``.
    cloud_ice_particle_radius_lower : np.ndarray, optional
        Per-footprint ice cloud particle effective radius (μm) for the lower
        layer (index 0) of ``cloud_ice_particle_radius_37um_mean``.
    cloud_classification, shortwave_adm_type, longwave_adm_type : np.ndarray, optional
        Per-footprint int16 categorical/encoded codes.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    import netCDF4  # noqa: PLC0415

    fill_f = np.float32(3.4028235e38)
    fill_i = np.int16(32767)

    # Default footprint set: five in the test cluster (lat 10–11, lon −10) plus
    # one far-away footprint that must be excluded by any local bbox.
    if lats is None:
        lats = np.array([10.2, 10.4, 10.6, 10.8, 11.0, -50.0], dtype=np.float32)
    if lons_0360 is None:
        # −10° in the 0..360 convention is 350°; last point is far away (100°).
        lons_0360 = np.array([350.0, 350.0, 350.0, 350.0, 350.0, 100.0], dtype=np.float32)
    n = lats.size

    if aerosol_optical_depth is None:
        aerosol_optical_depth = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60], dtype=np.float32)[:n]
    if clear_coverage is None:
        clear_coverage = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)[:n]
    if cloud_optical_depth_lower is None:
        cloud_optical_depth_lower = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=np.float32)[:n]
    # Water and ice particle radii use distinct deterministic values so each
    # variable can be asserted independently in tests.
    if cloud_water_particle_radius_lower is None:
        cloud_water_particle_radius_lower = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)[:n]
    if cloud_ice_particle_radius_lower is None:
        cloud_ice_particle_radius_lower = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0], dtype=np.float32)[:n]
    if cloud_classification is None:
        # Four of one code, one of another → modal code is 1001.
        cloud_classification = np.array([1001, 1001, 1001, 1001, 1191, 2000], dtype=np.int16)[:n]
    if shortwave_adm_type is None:
        shortwave_adm_type = np.array([50, 50, 50, fill_i, fill_i, 60], dtype=np.int16)[:n]
    if longwave_adm_type is None:
        longwave_adm_type = np.array([50, 50, 50, 50, 50, 60], dtype=np.int16)[:n]

    out_path = tmp_path / "cer_ssf_fixture.nc"

    def _add(grp, name, data, fill, valid_range=None):
        var = grp.createVariable(name, data.dtype, ("Footprints",), fill_value=fill)
        var[:] = data
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=data.dtype)

    def _add_lower_upper(grp, name, lower_values, fill, valid_range=None, upper_values=None):
        """Write a 2-D (Footprints, LowerUpper) variable with per-layer values.

        Index 0 is the lower cloud layer (selected by ``_CLOUD_LAYER_LOWER_INDEX``),
        index 1 the upper layer (``_CLOUD_LAYER_UPPER_INDEX``). ``upper_values``
        defaults to fill, so callers that only care about the lower layer are
        unaffected; the extended cloud fields pass distinct upper values so tests
        can tell the two layers apart.
        """
        var = grp.createVariable(name, "f4", ("Footprints", "LowerUpper"), fill_value=fill)
        data = np.full((n, 2), fill, dtype=np.float32)
        data[:, 0] = lower_values.astype(np.float32)
        if upper_values is not None:
            data[:, 1] = upper_values.astype(np.float32)
        var[:] = data
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=np.float32)

    def _add_2d(grp, name, values_2d, dim_name, fill, valid_range=None):
        """Write a 2-D (Footprints, <dim_name>) variable (e.g. AeroTypePct)."""
        var = grp.createVariable(name, "f4", ("Footprints", dim_name), fill_value=fill)
        var[:] = values_2d.astype(np.float32)
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=np.float32)

    # Deterministic per-footprint ramp used to fill the extended cloud/aerosol/albedo
    # source fields with distinct, assertable values (upper layer offset from lower).
    ramp = np.arange(n, dtype=np.float32)

    with netCDF4.Dataset(str(out_path), "w") as ds:
        ds.createDimension("Footprints", n)
        ds.createDimension("LowerUpper", 2)
        # Secondary axis flattened by the reader into _typeN 1-D variables.
        ds.createDimension("AeroTypePct", 7)

        tp = ds.createGroup("Time_and_Position")
        _add(tp, "instrument_fov_latitude", lats.astype(np.float32), fill_f, (-90.0, 90.0))
        _add(tp, "instrument_fov_longitude", lons_0360.astype(np.float32), fill_f, (0.0, 360.0))

        aux = ds.createGroup("Auxillary_Properties")
        _add(aux, "aerosol_optical_depth", aerosol_optical_depth.astype(np.float32), fill_f, (0.0, 8.0))

        clr = ds.createGroup("Clear_Footprint_Area")
        _add(clr, "clear_coverage", clear_coverage.astype(np.float32), fill_f, (0.0, 100.0))

        cif = ds.createGroup("Cloudy_Imager_Footprint_Layer")
        # The three fields whose lower layer feeds the base reader specs. Upper-layer
        # values (offset from lower) feed the FMATCH-IMAGER-only *_upper specs.
        _add_lower_upper(
            cif,
            "cloud_optical_depth_mean",
            cloud_optical_depth_lower,
            fill_f,
            (0.0, 512.0),
            upper_values=cloud_optical_depth_lower + 1.0,
        )
        # Phase-separated effective particle radii. SSF does not provide a single
        # blended radius — water and ice clouds are retrieved independently at 3.7 μm.
        _add_lower_upper(
            cif,
            "cloud_water_particle_radius_37um_mean",
            cloud_water_particle_radius_lower,
            fill_f,
            (2.0, 60.0),
            upper_values=cloud_water_particle_radius_lower + 1.0,
        )
        _add_lower_upper(
            cif,
            "cloud_ice_particle_radius_37um_mean",
            cloud_ice_particle_radius_lower,
            fill_f,
            (5.0, 90.0),
            upper_values=cloud_ice_particle_radius_lower + 2.0,
        )
        # Five genuinely-new layered fields carried by FMATCH-IMAGER (both layers).
        _add_lower_upper(
            cif, "layers_coverages", 10.0 + ramp * 5.0, fill_f, (0.0, 100.0), upper_values=15.0 + ramp * 5.0
        )
        _add_lower_upper(
            cif, "cloud_coverage_multilayer", 5.0 + ramp * 3.0, fill_f, (0.0, 100.0), upper_values=8.0 + ramp * 3.0
        )
        _add_lower_upper(
            cif, "cloud_top_pressure_mean", 300.0 + ramp * 10.0, fill_f, (0.0, 1100.0), upper_values=250.0 + ramp * 10.0
        )
        _add_lower_upper(
            cif,
            "cloud_base_pressure_mean",
            800.0 + ramp * 10.0,
            fill_f,
            (0.0, 1100.0),
            upper_values=700.0 + ramp * 10.0,
        )
        _add_lower_upper(
            cif, "cloud_particle_phase_37um_mean", 1.0 + ramp * 0.05, fill_f, (1.0, 2.0), upper_values=1.5 + ramp * 0.05
        )

        scn = ds.createGroup("Scene_Type")
        _add(scn, "cloud_classification", cloud_classification.astype(np.int16), fill_i, (0, 32766))
        _add(scn, "shortwave_adm_type", shortwave_adm_type.astype(np.int16), fill_i, (0, 5000))
        _add(scn, "longwave_adm_type", longwave_adm_type.astype(np.int16), fill_i, (0, 5000))

        # FMATCH-IMAGER-only aerosol fields.
        aap = ds.createGroup("Assimilated_Aerosol_Properties")
        _add(aap, "match_aot", (0.15 + ramp * 0.1).astype(np.float32), fill_f, (0.0, 8.0))
        # (Footprints, AeroTypePct=7): type t gets (t+1)*5 + footprint index.
        aerosol_pct = (np.arange(1, 8, dtype=np.float32)[None, :] * 5.0) + ramp[:, None]
        _add_2d(aap, "aerosol_type_percentage", aerosol_pct, "AeroTypePct", fill_f, (0.0, 100.0))

        ila = ds.createGroup("Imager_Land_Aerosols")
        _add(
            ila,
            "imager_dark_target_land_055um_corrected_aerosol_optical_depth",
            (0.2 + ramp * 0.1).astype(np.float32),
            fill_f,
            (0.0, 5.0),
        )

        ioa = ds.createGroup("Imager_Ocean_Aerosols")
        _add(
            ioa,
            "imager_deep_blue_ocean_055um_aerosol_optical_depth",
            (0.25 + ramp * 0.1).astype(np.float32),
            fill_f,
            (0.0, 5.0),
        )

        # FMATCH-IMAGER-only surface albedo: a single 1-D (Footprints,) broadband value
        # (0..1 fraction) in Auxillary_Properties, matching CER_SSF's surface_albedo.
        _add(aux, "surface_albedo", (0.1 + ramp * 0.05).astype(np.float32), fill_f, (0.0, 1.0))

        # FMATCH-IMAGER-only TOA incoming solar radiation: 1-D (Footprints,), W/m^2.
        otf = ds.createGroup("Observed_TOA_Fluxes")
        _add(otf, "toa_incoming_solar_radiation", (1360.0 + ramp).astype(np.float32), fill_f, (0.0, 1400.0))

    return out_path


def make_cldpix_fixture(
    tmp_path: Path,
    lats: np.ndarray | None = None,
    lons_0360: np.ndarray | None = None,
) -> Path:
    """Write a synthetic CERES CLDPIX (imager-pixel swath) NetCDF4 file.

    Replicates the flat, 2-D ``(Scanlines, Pixels)`` structure of the real
    CLDPIX product:
    - 2-D ``Latitude`` / ``Longitude`` arrays, written in the real product's
      geolocation conventions: ``Latitude`` holds **colatitude** (0° = North
      Pole → 180° = South Pole, ``valid_range`` ``[0, 180]``) and ``Longitude``
      is in the **0..360 convention**. The ``lats`` argument below is given as
      an ordinary −90..90 latitude and converted on write, so tests can express
      their expectations in normal latitudes.
    - A minimal set of float cloud variables (fill ``3.4028235e38``) and int8
      categorical variables (fill ``127``)
    - ``Eff_Cld_Pressure`` is written with a **descending** ``valid_range``
      ([1100, 10]) exactly as the real file does, so tests can verify the reader
      disables netCDF4 auto-masking (which would otherwise mask every value).

    Surface-type variables (``IGBP_Ecosystem``, ``Snow_Map_Value``,
    ``Ice_Map_Value``) are present in real CLDPIX files but are NOT written
    here — the reader does not extract them (see ``cldpix.py`` module docstring).

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lats, lons_0360 : np.ndarray, optional
        2-D ``(Scanlines, Pixels)`` geolocation arrays. ``lats`` is given as a
        true −90..90 latitude (converted to colatitude on write); ``lons_0360``
        is in the 0..360 convention. Defaults place all pixels near lat ≈ 40°,
        lon ≈ −15° (written as colatitude 50° and longitude 345° in the file).

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    import netCDF4  # noqa: PLC0415

    fill_f = np.float32(3.4028235e38)
    fill_i8 = np.int8(127)

    n_scan, n_pix = 3, 4
    if lats is None:
        lats = np.full((n_scan, n_pix), 40.0, dtype=np.float32)
        # Spread pixels slightly so they fall in a couple of 0.05° cells.
        lats += np.linspace(0.0, 0.10, n_scan * n_pix).reshape(n_scan, n_pix).astype(np.float32)
    if lons_0360 is None:
        # −15° in the 0..360 convention is 345°.
        lons_0360 = np.full((n_scan, n_pix), 345.0, dtype=np.float32)
    shape = lats.shape

    out_path = tmp_path / "cer_cldpix_fixture.nc"

    with netCDF4.Dataset(str(out_path), "w") as ds:
        ds.createDimension("Scanlines", shape[0])
        ds.createDimension("Pixels", shape[1])
        dims = ("Scanlines", "Pixels")

        def _add(name, data, fill, valid_range=None):
            var = ds.createVariable(name, data.dtype, dims, fill_value=fill)
            var[:] = data
            if valid_range is not None:
                var.valid_range = np.array(valid_range, dtype=data.dtype)

        # Real CLDPIX files store colatitude under the name "Latitude", with a
        # [0, 180] valid_range. Convert the caller's true latitudes so the
        # fixture exercises the reader's colatitude → latitude conversion.
        colatitudes = 90.0 - lats.astype(np.float32)
        _add("Latitude", colatitudes.astype(np.float32), fill_f, (0.0, 180.0))
        _add("Longitude", lons_0360.astype(np.float32), fill_f, (0.0, 360.0))

        # Continuous cloud properties (constant values for easy assertions).
        _add("Eff_Cld_Optical_Depth", np.full(shape, 4.0, np.float32), fill_f, (0.25, 150.0))
        _add("Cld_Water_Path", np.full(shape, 100.0, np.float32), fill_f, (0.0, 10000.0))
        _add("Eff_Cld_Temp", np.full(shape, 270.0, np.float32), fill_f, (190.0, 350.0))
        _add("Eff_Cld_Height", np.full(shape, 5.0, np.float32), fill_f, (0.0, 18.0))
        # Reversed valid_range exactly like the real file.
        _add("Eff_Cld_Pressure", np.full(shape, 800.0, np.float32), fill_f, (1100.0, 10.0))
        # Top_Cld_Height is the only field the reader configures with no
        # valid_range, so its fill pixels can only be caught by the fill-value
        # test. One pixel is set to the float32 fill sentinel to exercise that
        # path (see test_top_cloud_height_fill_is_dropped).
        top_height = np.full(shape, 6.0, np.float32)
        top_height.flat[0] = fill_f
        _add("Top_Cld_Height", top_height, fill_f)
        # Effective cloud particle radius (μm) — blended water+ice value from
        # the CERES retrieval algorithm. Constant 10.0 μm for easy assertions.
        _add("Cld_Radius", np.full(shape, 10.0, np.float32), fill_f, (2.0, 60.0))

        # Categorical (int8) fields; one pixel set to the 127 fill sentinel.
        phase = np.full(shape, 1, dtype=np.int8)
        phase.flat[-1] = fill_i8
        _add("Cloud_Particle_Phase", phase, fill_i8, (1, 5))
        _add("CERES_Cloud_Mask", np.full(shape, 1, np.int8), fill_i8, (0, 3))

    return out_path


def make_fmatch_product_fixture(
    directory: Path,
    mode: OperationalMode | None = None,
    n_footprints: int = 8,
) -> Path:
    """Write a synthetic, conformant FMATCH product NetCDF file (input to the SCENE-ID runners).

    Introspects the FMATCH product definition for ``mode`` so the written file always matches the shipped YAML (dtype
    and variable set), then writes it with :func:`libera_utils.io.netcdf.write_libera_data_product` under a proper
    Libera filename. This is the FMATCH-CAM / FMATCH-CAM-CAMTIME input that the SCENE-ID-CAM(-CAMTIME) readers consume,
    replacing the old CERES-SSF placeholder input.

    Every variable is filled with zeros except ``igbp_surface_type``, which is set to valid IGBP land-cover codes
    (1..17) so that ``calculate_trmm_surface_type`` accepts them when the reader-fed data is classified. The time
    coordinate is a monotonically increasing 100 Hz series (``write_libera_data_product`` stamps the filename's
    start/end from its first/last values).

    Parameters
    ----------
    directory : Path
        Directory to write the product file into (e.g. pytest ``tmp_path``).
    mode : OperationalMode, optional
        The FMATCH operational mode whose definition drives the file. Defaults to ``OperationalMode.CAM``.
    n_footprints : int, optional
        Number of footprints (length of the time axis). Defaults to 8.

    Returns
    -------
    Path
        Path to the written FMATCH product NetCDF file.
    """
    from libera_utils.footprint_matching.product import (  # noqa: PLC0415
        OperationalMode,
        fmatch_time_variable,
        load_fmatch_definition,
    )
    from libera_utils.io.netcdf import write_libera_data_product  # noqa: PLC0415

    if mode is None:
        mode = OperationalMode.CAM
    definition = load_fmatch_definition(mode)
    time_variable = fmatch_time_variable(mode)

    base_time = np.datetime64("2026-06-11T00:00:00", "ns")
    cadence = np.timedelta64(10_000_000, "ns")  # 100 Hz
    times = base_time + np.arange(n_footprints, dtype="int64") * cadence

    data: dict[str, np.ndarray] = {time_variable: times}

    # Fill any non-time coordinate the definition declares. On the camera-timescale products this is the pair of 2-D
    # camera_pixel_x/y range coordinates (FOOTPRINT x CAMERA_PIXEL_BOUNDS); the radiometer products have none. The
    # record axis length is n_footprints; CAMERA_PIXEL_BOUNDS is the fixed size-2 (min, max) pair.
    dimension_sizes = {"RADIOMETER_TIME": n_footprints, "FOOTPRINT": n_footprints, "CAMERA_PIXEL_BOUNDS": 2}
    for name, coord_def in definition.coordinates.items():
        if name == time_variable:
            continue
        shape = tuple(dimension_sizes[dimension] for dimension in coord_def.dimensions)
        data[name] = np.zeros(shape, dtype=coord_def.dtype)

    for name, var_def in definition.variables.items():
        if name == "igbp_surface_type":
            data[name] = (np.arange(n_footprints) % 17 + 1).astype(var_def.dtype)
        elif name == "ssf_clear_coverage":
            # Clear-sky percentage in [0, 100]; the imager scene-ID readers derive cloud_fraction = 100 -
            # clear_coverage, so a spread across the full range gives the classifier both cloudy and (near-)clear
            # footprints -- the latter exercise the clear/surface TRMM scenes that FLASH can match despite having no
            # cloud phase.
            data[name] = np.linspace(0, 100, n_footprints).astype(var_def.dtype)
        elif name == "cldpix_cloud_particle_phase":
            # Valid CLDPIX phase codes cycling liquid(1)/ice(2) so map_cldpix_phase_to_trmm yields usable 1/2
            # values (rather than all-NaN) for the TRMM classification.
            data[name] = (np.arange(n_footprints) % 2 + 1).astype(var_def.dtype)
        else:
            data[name] = np.zeros(n_footprints, dtype=var_def.dtype)

    dynamic_attrs = {
        "algorithm_version": "0.1.0",
        "input_files": "SYNTHETIC EXAMPLE - no real input files were used",
    }
    written = write_libera_data_product(
        definition,
        data,
        output_path=directory,
        time_variable=time_variable,
        dynamic_product_attributes=dynamic_attrs,
    )
    return Path(str(written.path))


def make_l1b_radiometer_fixture(
    directory: Path,
    n_footprints: int = 8,
    *,
    n_invalid: int = 0,
) -> Path:
    """Write a synthetic L1B RAD-4CH NetCDF file (input to the radiometer-timescale FMATCH runners).

    Contains exactly the variables that
    :func:`libera_utils.footprint_matching.l1b_inputs.load_l1b_radiometer_inputs` reads: the CF-encoded
    ``radiometer_time`` coordinate plus the geolocation and Sun-surface-sensor viewing angles that FMATCH passes
    through verbatim. The values are physically plausible but arbitrary; only the variable names, dtypes and time
    encoding are the contract under test.

    The file is written under a proper Libera ``L1B RAD-4CH`` filename so that the manifest-driven runners select it
    with :func:`libera_utils.footprint_matching._runner.select_manifest_files_by_product_id`.

    Parameters
    ----------
    directory : Path
        Directory to write the L1B file into (e.g. pytest ``tmp_path``).
    n_footprints : int, optional
        Total number of footprints written. Defaults to 8.
    n_invalid : int, optional
        How many leading footprints to fill with NaN geolocation, mimicking the real files' off-Earth samples. Those
        rows are expected to be dropped by the reader, so ``n_footprints - n_invalid`` records should survive.
        Defaults to 0.

    Returns
    -------
    Path
        Path to the written L1B RAD-4CH NetCDF file.
    """
    from libera_utils.constants import DataProductIdentifier  # noqa: PLC0415
    from libera_utils.footprint_matching.l1b_inputs import L1B_PASSTHROUGH_VARIABLES, L1B_TIME_VARIABLE  # noqa: PLC0415
    from libera_utils.io.filenaming import LiberaDataProductFilename  # noqa: PLC0415

    base_time = np.datetime64("2026-06-11T00:00:00", "ns")
    cadence = np.timedelta64(10_000_000, "ns")  # 100 Hz
    times = base_time + np.arange(n_footprints, dtype="int64") * cadence

    # Spread the footprints over a plausible range for each quantity so tests can tell the columns apart. The angles
    # stay inside the product definition's valid ranges (SZA [0,180], VZA [0,90], RAA [0,360]).
    ranges = {
        "latitude": (-60.0, 60.0),
        "longitude": (-170.0, 170.0),
        "solar_zenith_angle": (10.0, 80.0),
        "viewing_zenith_angle": (0.0, 45.0),
        "relative_azimuth_angle": (0.0, 350.0),
    }
    data_vars = {}
    for fmatch_name, l1b_name in L1B_PASSTHROUGH_VARIABLES.items():
        low, high = ranges[fmatch_name]
        values = np.linspace(low, high, n_footprints, dtype=np.float32)
        if n_invalid:
            # NaN is how the real L1B marks samples whose boresight misses the Earth.
            values[:n_invalid] = np.nan
        data_vars[l1b_name] = ((L1B_TIME_VARIABLE,), values)

    dataset = xr.Dataset(data_vars, coords={L1B_TIME_VARIABLE: times})
    # Match the real product's CF time encoding so xarray decodes it back to datetime64[ns] on read.
    dataset[L1B_TIME_VARIABLE].encoding = {
        "units": "nanoseconds since 1958-01-01",
        "calendar": "standard",
        "dtype": "int64",
    }

    filename = LiberaDataProductFilename.from_filename_parts(
        product_name=DataProductIdentifier.l1b_rad,
        version="V1-0-0",
        utc_start=_as_utc_datetime(times[0]),
        utc_end=_as_utc_datetime(times[-1]),
    ).path.name
    out_path = Path(directory) / filename
    dataset.to_netcdf(out_path)
    return out_path


def make_l1b_camera_fixture(
    directory: Path,
    n_images: int = 2,
    n_pixels_x: int = 6,
    n_pixels_y: int = 6,
) -> Path:
    """Write a synthetic L1B CAM NetCDF file (input to the camera-timescale FMATCH runners).

    Contains the geolocation and viewing-angle grids that
    :func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera` segments into pseudo-footprints,
    on the ``CAMERA_TIME`` x ``CAMERA_PIXEL_COUNT_X`` x ``CAMERA_PIXEL_COUNT_Y`` grid. The pixel spacing is chosen
    coarse enough (~0.5 degrees, roughly 55 km) that each pixel becomes its own pseudo-footprint, which keeps the
    expected footprint count simple: ``n_images * n_pixels_x * n_pixels_y``.

    The file is written under a proper Libera ``L1B CAM`` filename so that the manifest-driven runners select it with
    :func:`libera_utils.footprint_matching._runner.select_manifest_files_by_product_id`.

    Parameters
    ----------
    directory : Path
        Directory to write the L1B file into (e.g. pytest ``tmp_path``).
    n_images : int, optional
        Number of camera images (length of the ``CAMERA_TIME`` axis). Defaults to 2.
    n_pixels_x, n_pixels_y : int, optional
        Pixel-grid dimensions of each image. Default to 6 x 6.

    Returns
    -------
    Path
        Path to the written L1B CAM NetCDF file.
    """
    from libera_utils.constants import DataProductIdentifier  # noqa: PLC0415
    from libera_utils.footprint_matching import camera_segmentation as seg  # noqa: PLC0415
    from libera_utils.io.filenaming import LiberaDataProductFilename  # noqa: PLC0415

    dims = (seg.CAMERA_TIME_NAME, seg.PIXEL_X_DIM, seg.PIXEL_Y_DIM)
    shape = (n_images, n_pixels_x, n_pixels_y)

    base_time = np.datetime64("2026-06-11T00:00:00", "ns")
    # Camera images are seconds apart (unlike the 100 Hz radiometer), so successive images get distinct times.
    times = base_time + np.arange(n_images, dtype="int64") * np.timedelta64(1, "s")

    # ~0.5 degree pixel spacing keeps each pixel above the target footprint diameter, so segmentation emits one
    # pseudo-footprint per pixel. Each image is offset in latitude so the images do not overlap.
    lat = np.zeros(shape, dtype=float)
    lon = np.zeros(shape, dtype=float)
    for image_index in range(n_images):
        lat_axis = 10.0 + image_index * 5.0 + np.arange(n_pixels_x) * 0.5
        lon_axis = 20.0 + np.arange(n_pixels_y) * 0.5
        lat[image_index] = lat_axis[:, None]
        lon[image_index] = lon_axis[None, :]

    def const_grid(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=float)

    dataset = xr.Dataset(
        {
            seg.LATITUDE_NAME: (dims, lat),
            seg.LONGITUDE_NAME: (dims, lon),
            seg.ALTITUDE_NAME: (dims, const_grid(835_000.0)),
            seg.SOLAR_ZENITH_NAME: (dims, const_grid(30.0)),
            seg.VIEWING_ZENITH_NAME: (dims, const_grid(10.0)),
            seg.RELATIVE_AZIMUTH_NAME: (dims, const_grid(120.0)),
        },
        coords={seg.CAMERA_TIME_NAME: times},
    )
    dataset[seg.CAMERA_TIME_NAME].encoding = {
        "units": "nanoseconds since 1958-01-01",
        "calendar": "standard",
        "dtype": "int64",
    }

    filename = LiberaDataProductFilename.from_filename_parts(
        product_name=DataProductIdentifier.l1b_cam,
        version="V1-0-0",
        utc_start=_as_utc_datetime(times[0]),
        utc_end=_as_utc_datetime(times[-1]),
    ).path.name
    out_path = Path(directory) / filename
    dataset.to_netcdf(out_path)
    return out_path


def _as_utc_datetime(value: np.datetime64) -> datetime:
    """Convert a numpy datetime64 to a timezone-aware UTC datetime for Libera filename construction."""
    return datetime.fromisoformat(str(np.datetime64(value, "us"))).replace(tzinfo=UTC)
