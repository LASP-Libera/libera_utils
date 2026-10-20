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

VIIRS BRDF VJ143C1 (HDF5/HDF-EOS5):
    LP DAAC: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/
    Example filename: VJ143C1.A2026153.002.2026161161054.h5
    EarthData login required: https://urs.earthdata.nasa.gov/

VIIRS Deep Blue aerosol — AOD + type (AERDB_D3_VIIRS_NOAA20, NetCDF4, root-level):
    NASA Deep Blue: https://deepblue.gsfc.nasa.gov/
    Example filename: AERDB_D3_VIIRS_NOAA20.A2026001.002.2026005001030.nc

VIIRS Cloud CLDPROP_D3 (NetCDF4 with groups):
    NCEI CDR: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
    Example filename: CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc

CERES SSF / FLASHFlux (NetCDF4, per-footprint swath):
    NASA CERES: https://ceres.larc.nasa.gov/data/#ssf-level-2
    Example filename: CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc

CERES CLDPIX (NetCDF4, imager-pixel swath):
    NASA CERES: https://ceres.larc.nasa.gov/data/
    Example filename: CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# Constant fill values for the five non-wind ERA5 single-level variables written
# by the ERA5 fixtures. Chosen to be physically plausible AND mutually distinct
# so tests can assert that each output layer came from the right file variable.
# The winds take their values from the u10_fill / v10_fill keyword parameters instead.
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


ERA5_FIXTURE_TIME = np.datetime64("2026-01-01T00:00:00", "ns")
"""The single hourly step the default ERA5 fixtures carry; query them at this time."""


def _era5_single_level_dataset(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    n_lat: int,
    n_lon: int,
    u10_fill: float,
    v10_fill: float,
    time_dim: str,
    times: np.ndarray | None,
    u10_per_step: np.ndarray | None = None,
) -> xr.Dataset:
    """Build the in-memory Dataset shared by the two single-level fixture makers.

    Contains all seven single-level variables the ERA5Reader expects (u10, v10,
    t2m, d2m, sp, z, fal), each a constant-valued grid, with latitudes stored
    DESCENDING (real ERA5 convention) and a leading time dimension named
    ``time_dim`` (``"time"`` as older CDS downloads name it, or ``"valid_time"`` as
    the new CDS API does). ``times`` are the step times (default: the single
    :data:`ERA5_FIXTURE_TIME`); ``u10_per_step`` optionally gives u10 one constant per
    step so a test can tell which step the reader served.
    """
    lats = np.linspace(lat_max, lat_min, n_lat)  # descending, real ERA5 convention
    lons = np.linspace(lon_min, lon_max, n_lon)
    if times is None:
        times = np.array([ERA5_FIXTURE_TIME], dtype="datetime64[ns]")
    times = np.asarray(times).astype("datetime64[ns]")
    n_time = times.size

    fills = {"u10": u10_fill, "v10": v10_fill, **ERA5_SINGLE_LEVEL_FILLS}
    shape = (n_time, n_lat, n_lon)
    dims = [time_dim, "latitude", "longitude"]

    coords = {
        "latitude": xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"}),
        "longitude": xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"}),
        time_dim: xr.DataArray(times, dims=[time_dim]),
    }

    data_vars = {}
    for name, fill in fills.items():
        data = np.full(shape, fill, dtype=np.float32)
        if name == "u10" and u10_per_step is not None:
            for step, value in enumerate(np.asarray(u10_per_step, dtype=np.float32)):
                data[step] = value
        data_vars[name] = xr.DataArray(data, dims=dims, attrs=_ERA5_SINGLE_LEVEL_ATTRS[name])
    return xr.Dataset(data_vars, coords=coords)


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
    times: np.ndarray | None = None,
    u10_per_step: np.ndarray | None = None,
    filename: str = "era5_fixture.nc",
) -> Path:
    """Write a synthetic ERA5 single-levels NetCDF4 file to ``tmp_path``.

    The real ERA5 files from CDS contain the single-level variables on a global
    lat/lon grid with the latitude dimension in DESCENDING order (90 -> -90) and a
    leading hourly ``time`` dimension. This factory reproduces that convention in a
    small grid for testing, writing all seven variables the ERA5Reader expects
    (u10, v10, t2m, d2m, sp, z, fal). The non-wind variables carry the constant
    ``ERA5_SINGLE_LEVEL_FILLS`` values.

    The real file format reference:
    https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lat_min, lat_max : float
        Latitude range. Latitudes are stored in DESCENDING order in the file
        (matching real ERA5 convention). Default: 0 -> 2 N.
    lon_min, lon_max : float
        Longitude range. Default: 10 -> 12 E.
    n_lat, n_lon : int
        Number of grid points in each dimension. Default 4.
    u10_fill : float
        Constant fill value for the u10 variable. Default 2.5 m/s.
    v10_fill : float
        Constant fill value for the v10 variable. Default -1.5 m/s.
    times : np.ndarray, optional
        Step times (``datetime64``). Default: one step at :data:`ERA5_FIXTURE_TIME`.
    u10_per_step : np.ndarray, optional
        One u10 constant per step, overriding ``u10_fill`` step by step.
    filename : str, optional
        Name of the file written under ``tmp_path``.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    ds = _era5_single_level_dataset(
        lat_min, lat_max, lon_min, lon_max, n_lat, n_lon, u10_fill, v10_fill, "time", times, u10_per_step
    )
    out_path = tmp_path / filename
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

    Identical to :func:`make_era5_netcdf_fixture` except that the time dimension is
    named ``valid_time``, matching the format produced by the new CDS API. This
    exercises the reader's time-dimension detection, which matches on the substring
    ``"time"``.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    lat_min, lat_max : float
        Latitude range (DESCENDING in file). Default: 0 -> 2 N.
    lon_min, lon_max : float
        Longitude range. Default: 10 -> 12 E.
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
        lat_min, lat_max, lon_min, lon_max, n_lat, n_lon, u10_fill, v10_fill, "valid_time", None
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
        variables, which the reader rejects (it needs a time axis to match steps).

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
        coords["valid_time"] = xr.DataArray(np.array([ERA5_FIXTURE_TIME], dtype="datetime64[ns]"), dims=["valid_time"])

    ds = xr.Dataset(data_vars, coords=coords)
    out_path = tmp_path / "era5_pressure_fixture.nc"
    ds.to_netcdf(out_path)
    return out_path


# MODIS sinusoidal tile grid: 36 x 18 tiles of 1111950.519667 m on a 6371007.181 m sphere
# (MCD12Q1 v061 User Guide). Tile (h, v) has its upper-left corner at
# (-20015109.354 + h * size, 10007554.677 - v * size) meters.
MODIS_SPHERE_RADIUS_M: float = 6371007.181
MODIS_TILE_SIZE_M: float = 1111950.519667
_MODIS_X_MIN_M: float = -20015109.354
_MODIS_Y_MAX_M: float = 10007554.677


def modis_tile_corners(h: int, v: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the ``(upper_left, lower_right)`` corners of MODIS tile ``h``, ``v``, sinusoidal meters.

    Parameters
    ----------
    h, v : int
        Horizontal (0..35) and vertical (0..17) tile indices.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ``((x_ul, y_ul), (x_lr, y_lr))``.
    """
    x_ul = _MODIS_X_MIN_M + h * MODIS_TILE_SIZE_M
    y_ul = _MODIS_Y_MAX_M - v * MODIS_TILE_SIZE_M
    return (x_ul, y_ul), (x_ul + MODIS_TILE_SIZE_M, y_ul - MODIS_TILE_SIZE_M)


def make_modis_sinusoidal_hdf4_fixture(
    path: Path,
    data: np.ndarray,
    *,
    ul_m: tuple[float, float],
    lr_m: tuple[float, float],
    projection: str = "GCTP_SNSOID",
    proj_params: str = "6371007.181000,0,0,0,0,0,0,0,0,0,0,0,0",
    omit: tuple[str, ...] = (),
    xdim: int | None = None,
    sds_name: str = "LC_Type1",
) -> Path:
    """Write a MODIS sinusoidal HDF-EOS2 tile (one SDS plus ``StructMetadata.0``) with pyhdf.

    Mirrors an MCD12Q1 tile: the SDS holds the codes in their stored dtype, and the grid
    geometry is only in the ``StructMetadata.0`` global attribute.

    Parameters
    ----------
    path : Path
        Output ``.hdf`` path.
    data : np.ndarray
        ``(n_rows, n_cols)`` codes, written in their own dtype (uint8 for ``LC_Type1``).
    ul_m, lr_m : tuple[float, float]
        Upper-left and lower-right corners, sinusoidal meters (see :func:`modis_tile_corners`).
    projection : str, optional
        The ``Projection`` value written.
    proj_params : str, optional
        The ``ProjParams`` list written; its first value is the sphere radius.
    omit : tuple[str, ...], optional
        ``StructMetadata.0`` fields to leave out (e.g. ``("XDim",)``).
    xdim : int, optional
        The ``XDim`` written, when it should differ from ``data``'s column count.
    sds_name : str, optional
        The SDS name.

    Returns
    -------
    Path
        ``path``.
    """
    import pyhdf.SD as sd  # noqa: PLC0415

    n_rows, n_cols = data.shape
    fields = {
        "XDim": f"XDim={n_cols if xdim is None else xdim}",
        "YDim": f"YDim={n_rows}",
        "UpperLeftPointMtrs": f"UpperLeftPointMtrs=({ul_m[0]:.6f},{ul_m[1]:.6f})",
        "LowerRightMtrs": f"LowerRightMtrs=({lr_m[0]:.6f},{lr_m[1]:.6f})",
        "Projection": f"Projection={projection}",
        "ProjParams": f"ProjParams=({proj_params})",
    }
    lines = ["GROUP=SwathStructure", "END_GROUP=SwathStructure", "GROUP=GridStructure", "\tGROUP=GRID_1"]
    lines.append('\t\tGridName="MOD12Q1"')
    lines.extend(f"\t\t{text}" for name, text in fields.items() if name not in omit)
    lines.extend(["\t\tSphereCode=-1", "\tEND_GROUP=GRID_1", "END_GROUP=GridStructure", "END"])
    dtypes = {np.dtype(np.uint8): sd.SDC.UINT8, np.dtype(np.int16): sd.SDC.INT16, np.dtype(np.float32): sd.SDC.FLOAT32}

    hdf = sd.SD(str(path), sd.SDC.WRITE | sd.SDC.CREATE)
    try:
        sds = hdf.create(sds_name, dtypes[data.dtype], data.shape)
        sds[:] = data
        sds.endaccess()
        hdf.attr("StructMetadata.0").set(sd.SDC.CHAR8, "\n".join(lines) + "\n")
    finally:
        hdf.end()
    return path


NISE_HEMISPHERE_LABELS: tuple[str, str] = ("Northern Hemisphere", "Southern Hemisphere")


def make_nise_hdfeos_fixture(
    path: Path,
    north: np.ndarray,
    south: np.ndarray,
    *,
    upper_left_m: tuple[float, float] = (-9_036_842.7625, 9_036_842.7625),
    lower_right_m: tuple[float, float] = (9_036_842.7625, -9_036_842.7625),
    projection: str = "GCTP_LAMAZ",
    struct_metadata: bool = True,
) -> Path:
    """Write a NISE HDF-EOS4 granule with pyhdf: an ``Extent`` and an ``Age`` SDS per hemisphere.

    Mirrors the real layout: both hemispheres name their fields ``Extent`` and ``Age``,
    and only the dimension names (``"YDim:Northern Hemisphere"``, ...) tell them apart.
    Northern fields are written first. ``StructMetadata.0`` defines one grid per
    hemisphere, named by its label, with the arrays' dimensions and the corners given.

    Parameters
    ----------
    path : Path
        Output ``.HDFEOS`` path.
    north, south : np.ndarray
        uint8 Extent codes per hemisphere, ``(n_rows, n_cols)``.
    upper_left_m, lower_right_m : tuple[float, float], optional
        Both grids' corners in ``StructMetadata.0``, projection meters. The defaults are
        the 25 km EASE-Grid's.
    projection : str, optional
        The ``Projection`` both grids name.
    struct_metadata : bool, optional
        Whether to write ``StructMetadata.0`` at all.

    Returns
    -------
    Path
        ``path``.
    """
    import pyhdf.SD as sd  # noqa: PLC0415

    lines = ["GROUP=SwathStructure", "END_GROUP=SwathStructure", "GROUP=GridStructure"]
    for number, (label, extent) in enumerate(zip(NISE_HEMISPHERE_LABELS, (north, south), strict=True), start=1):
        center = "90000000" if label.startswith("Northern") else "-90000000"
        lines.extend(
            [
                f"\tGROUP=GRID_{number}",
                f'\t\tGridName="{label}"',
                f"\t\tXDim={extent.shape[1]}",
                f"\t\tYDim={extent.shape[0]}",
                f"\t\tUpperLeftPointMtrs=({upper_left_m[0]:.6f},{upper_left_m[1]:.6f})",
                f"\t\tLowerRightMtrs=({lower_right_m[0]:.6f},{lower_right_m[1]:.6f})",
                f"\t\tProjection={projection}",
                f"\t\tProjParams=(6371228,0,0,0,0,{center},0,0,0,0,0,0,0)",
                "\t\tSphereCode=19",
                f"\tEND_GROUP=GRID_{number}",
            ]
        )
    lines.extend(["END_GROUP=GridStructure", "END"])

    hdf = sd.SD(str(path), sd.SDC.WRITE | sd.SDC.CREATE)
    try:
        for label, extent in zip(NISE_HEMISPHERE_LABELS, (north, south), strict=True):
            for name, values in (("Extent", extent), ("Age", np.zeros_like(extent))):
                sds = hdf.create(name, sd.SDC.UINT8, extent.shape)
                sds.dim(0).setname(f"YDim:{label}")
                sds.dim(1).setname(f"XDim:{label}")
                sds[:] = np.asarray(values, dtype=np.uint8)
                sds.endaccess()
        if struct_metadata:
            hdf.attr("StructMetadata.0").set(sd.SDC.CHAR8, "\n".join(lines) + "\n")
    finally:
        hdf.end()
    return path


def _degrees_to_dms(value: float) -> float:
    """Encode decimal degrees as the HDF-EOS packed ``DDDMMMSSS.SS`` angle."""
    sign = -1.0 if value < 0 else 1.0
    total_seconds = round(abs(value) * 3600.0, 6)
    degrees, rest = divmod(total_seconds, 3600.0)
    minutes, seconds = divmod(rest, 60.0)
    return sign * (degrees * 1_000_000 + minutes * 1_000 + seconds)


def make_viirs_brdf_hdf5_fixture(
    tmp_path: Path,
    n_lat: int = 4,
    n_lon: int = 8,
    lat_min: float = 0.0,
    lat_max: float = 0.20,
    lon_min: float = 10.0,
    lon_max: float = 10.40,
    param_fill: int = 200,
    fill_sentinel: int = 32767,
    valid_range: tuple[int, int] = (0, 32766),
    omit_attributes: tuple[str, ...] = (),
    projection: str = "HE5_GCTP_GEO",
    metadata_corners: tuple[float, float, float, float] | None = None,
) -> Path:
    """Write a synthetic VJ143C1 HDF5 BRDF fixture file.

    Replicates the HDF-EOS5 layout of the real VJ143C1 product:
    - ``HDFEOS INFORMATION/StructMetadata.0`` defining grid ``VIIRS_CMG_BRDF`` with
      ``XDim``/``YDim``, packed-DMS ``UpperLeftPointMtrs``/``LowerRightMtrs`` corners and
      ``Projection=HE5_GCTP_GEO``.
    - Group ``HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/`` holding ``lat`` (n_lat,) and
      ``lon`` (n_lon,) as each cell's upper-left **edge**, latitude descending, as the
      real file stores them (the real CMG runs ``lat`` 90 -> -89.95, ``lon`` -180 -> 179.95).
    - Nine int16 BRDF parameter datasets (3 bands x 3 kernel weights), each with
      ``scale_factor=0.001``, ``add_offset=0``, ``_FillValue`` and ``valid_range``.

    Parameters
    ----------
    tmp_path : Path
        pytest ``tmp_path`` fixture directory.
    n_lat, n_lon : int
        Grid dimensions. Default 4 x 8.
    lat_min, lat_max, lon_min, lon_max : float
        The region's outer cell edges, degrees; the cell size is the extent over
        ``n_lat`` / ``n_lon``. Default 0.0-0.20 by 10.0-10.40 (0.05 degree cells).
    param_fill : int
        Raw int16 value of every BRDF parameter pixel. Default 200 (0.200 scaled).
    fill_sentinel : int
        The ``_FillValue``, also written to the last pixel of each dataset. Default 32767.
    valid_range : tuple[int, int]
        The ``valid_range`` attribute written on each field.
    omit_attributes : tuple[str, ...]
        Field attributes to leave out (e.g. ``("add_offset",)``).
    projection : str
        The ``Projection`` written for the grid.
    metadata_corners : tuple[float, float, float, float], optional
        ``(ul_lon, ul_lat, lr_lon, lr_lat)`` written to StructMetadata in place of the
        region's own corners.

    Returns
    -------
    Path
        Path to the created HDF5 fixture file.
    """
    import h5py  # noqa: PLC0415

    out_path = tmp_path / "vj143c1_fixture.h5"
    lat_cell = (lat_max - lat_min) / n_lat
    lon_cell = (lon_max - lon_min) / n_lon
    lat_edges = lat_max - np.arange(n_lat) * lat_cell  # upper edges, descending
    lon_edges = lon_min + np.arange(n_lon) * lon_cell  # western edges, ascending
    ul_lon, ul_lat, lr_lon, lr_lat = metadata_corners or (lon_min, lat_max, lon_max, lat_min)
    struct_metadata = "\n".join(
        [
            "GROUP=SwathStructure",
            "END_GROUP=SwathStructure",
            "GROUP=GridStructure",
            "\tGROUP=GRID_1",
            '\t\tGridName="VIIRS_CMG_BRDF"',
            f"\t\tXDim={n_lon}",
            f"\t\tYDim={n_lat}",
            f"\t\tUpperLeftPointMtrs=({_degrees_to_dms(ul_lon):.6f},{_degrees_to_dms(ul_lat):.6f})",
            f"\t\tLowerRightMtrs=({_degrees_to_dms(lr_lon):.6f},{_degrees_to_dms(lr_lat):.6f})",
            f"\t\tProjection={projection}",
            "\t\tSphereCode=12",
            "\tEND_GROUP=GRID_1",
            "END_GROUP=GridStructure",
            "",
        ]
    )

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
    attributes = {
        "scale_factor": np.array([0.001]),
        "add_offset": np.array([0.0]),
        "_FillValue": np.array([fill_sentinel], dtype=np.int16),
        "valid_range": np.array(valid_range, dtype=np.int16),
    }

    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_(struct_metadata))
        grp = f.require_group("HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields")
        grp.create_dataset("lat", data=lat_edges.astype(np.float64)).attrs["units"] = "degrees_north"
        grp.create_dataset("lon", data=lon_edges.astype(np.float64)).attrs["units"] = "degrees_east"
        for name in field_names:
            raw = np.full((n_lat, n_lon), param_fill, dtype=np.int16)
            # The last element is the fill sentinel so tests can verify NaN handling.
            raw[-1, -1] = fill_sentinel
            ds_obj = grp.create_dataset(name, data=raw)
            for attr, value in attributes.items():
                if attr not in omit_attributes:
                    ds_obj.attrs[attr] = value

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
    aod_valid_range: tuple[float, float] | None = (0.0, 10.0),
    aerosol_type_valid_range: tuple[int, int] | None = (0, 7),
    aod_fill_value: float | None = -999.0,
    aerosol_type_fill_value: int | None = -999,
) -> Path:
    """Write a synthetic AERDB_D3_VIIRS_NOAA20 Deep Blue aerosol NetCDF4 file.

    Replicates the relevant structure of the real single-sensor NOAA-20 VIIRS
    Deep Blue Level-3 daily product as consumed by ``VIIRSAODReader`` — all
    variables at the **root** (no groups), stored in (Latitude, Longitude) order
    with ascending latitude (no transpose needed):
    - Root ``Latitude_1D`` (ascending) and ``Longitude_1D`` coordinate arrays.
    - Root ``Aerosol_Optical_Thickness_550_Land_Ocean_Mean`` (AOD @ 550 nm),
      ``_FillValue = -999.0``, ``valid_range = [0, 10]``.
    - Root ``Aerosol_Type_Land_Ocean_Mode`` (modal aerosol type, ``int32``,
      ``_FillValue = -999``, ``valid_range = [0, 7]``, categories 0..7).
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
    aod_valid_range, aerosol_type_valid_range : tuple or None
        The ``valid_range`` attribute written on each field; ``None`` writes none.
    aod_fill_value, aerosol_type_fill_value : float, int or None
        The ``_FillValue`` attribute written on each field; ``None`` writes none.
        Default −999, as in the real product.

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
            fill_value=False if aod_fill_value is None else aod_fill_value,
        )
        aod_data = np.full((n_lat, n_lon), aod_fill, dtype=np.float32)
        if include_fill_pixel:
            aod_data[0, 0] = -999.0
        aod_var[:] = aod_data
        if aod_valid_range is not None:
            aod_var.valid_range = np.array(aod_valid_range, dtype=np.float32)

        # Modal aerosol-type field at the root (int32 in the real product).
        type_var = ds.createVariable(
            "Aerosol_Type_Land_Ocean_Mode",
            "i4",
            ("Latitude_1D", "Longitude_1D"),
            fill_value=False if aerosol_type_fill_value is None else aerosol_type_fill_value,
        )
        type_data = np.full((n_lat, n_lon), aerosol_type_value, dtype=np.int32)
        if include_fill_pixel:
            type_data[0, 0] = -999
        type_var[:] = type_data
        if aerosol_type_valid_range is not None:
            type_var.valid_range = np.array(aerosol_type_valid_range, dtype=np.int32)

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


def make_viirs_cldprop_d3_fixture(
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
    mean_fill_value: float | None = -9999.0,
) -> Path:
    """Write a synthetic CLDPROP_D3 VIIRS cloud properties NetCDF4 file.

    Replicates the group structure of the real CLDPROP_D3 product:
    - Root-level ``latitude`` (n_lat,) and ``longitude`` (n_lon,) coordinate arrays
    - Three groups with a ``Mean`` variable each: ``Cloud_Fraction``,
      ``Cloud_Optical_Thickness_Combined``, ``Cloud_Top_Pressure``
    - **Variable dimension order is (longitude, latitude)** — transposed from
      the conventional (lat, lon) order — to match the real product format and
      exercise the transpose logic in ``VIIRSCldpropD3Reader``.
    - Each ``Mean`` declares ``_FillValue = -9999.0``, as the distributed files do; the
      constant data values contain no fill.

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
    mean_fill_value : float or None
        ``_FillValue`` declared on each ``Mean``. ``None`` writes none, to test a file
        without one.

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
            if mean_fill_value is None:
                mean_var = grp.createVariable("Mean", "f4", ("longitude", "latitude"), fill_value=False)
            else:
                mean_var = grp.createVariable("Mean", "f4", ("longitude", "latitude"), fill_value=mean_fill_value)
            # Fill with the constant value; shape is (n_lon, n_lat) in file.
            mean_var[:] = np.full((n_lon, n_lat), fill, dtype=np.float32)

    return out_path


SWATH_FIXTURE_TIME = np.datetime64("2026-01-01T12:00:00", "ns")
"""The observation time the default SSF and CLDPIX fixtures carry; query them at this time."""
_SWATH_TIME_UNITS = "seconds since 2026-01-01 00:00:00"


def _seconds_since_fixture_epoch(times: np.ndarray) -> np.ndarray:
    """Encode ``datetime64`` times in the CF units the swath fixtures write."""
    epoch = np.datetime64("2026-01-01T00:00:00", "ns")
    return ((np.asarray(times).astype("datetime64[ns]") - epoch) / np.timedelta64(1, "s")).astype(np.float64)


def make_ssf_fixture(
    tmp_path: Path,
    lats: np.ndarray | None = None,
    lons_0360: np.ndarray | None = None,
    times: np.ndarray | None = None,
    aerosol_optical_depth: np.ndarray | None = None,
    clear_coverage: np.ndarray | None = None,
    cloud_optical_depth_lower: np.ndarray | None = None,
    cloud_water_particle_radius_lower: np.ndarray | None = None,
    cloud_ice_particle_radius_lower: np.ndarray | None = None,
    cloud_classification: np.ndarray | None = None,
    shortwave_adm_type: np.ndarray | None = None,
    longwave_adm_type: np.ndarray | None = None,
    latitude_units: str | None = "degrees_north",
    omit_fill_value: tuple[str, ...] = (),
) -> Path:
    """Write a synthetic CERES SSF (footprint/swath) NetCDF4 file.

    Replicates the grouped, per-footprint structure of the real SSF product:
    - 1-D ``Footprints`` dimension, a ``LowerUpper`` dimension of length 2, and the
      ``AeroTypePct`` (7) secondary axis the reader flattens
    - ``Time_and_Position/instrument_fov_latitude`` and ``..._longitude``
      (**longitude stored in the 0..360 convention**, matching the real file), and
      ``Time_and_Position/time``, the per-footprint observation time encoded with
      CF units (``seconds since 2026-01-01``), and ``sample_number``, the position of
      each footprint within its scan
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
    times : np.ndarray, optional
        Per-footprint observation times (``datetime64``). Default: every footprint at
        :data:`SWATH_FIXTURE_TIME`.
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
    latitude_units : str or None, optional
        ``units`` written on ``instrument_fov_latitude``; ``None`` writes none.
        Default ``"degrees_north"``, as the real product declares.
    omit_fill_value : tuple of str, optional
        Names of 1-D variables written without a ``_FillValue`` attribute.

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
    if times is None:
        times = np.full(n, SWATH_FIXTURE_TIME, dtype="datetime64[ns]")

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
        fill = False if name in omit_fill_value else fill
        var = grp.createVariable(name, data.dtype, ("Footprints",), fill_value=fill)
        var[:] = data
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=data.dtype)
        return var

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
        lat_var = _add(tp, "instrument_fov_latitude", lats.astype(np.float32), fill_f, (-90.0, 90.0))
        if latitude_units is not None:
            lat_var.units = latitude_units
        _add(tp, "instrument_fov_longitude", lons_0360.astype(np.float32), fill_f, (0.0, 360.0))
        time_var = tp.createVariable("time", np.float64, ("Footprints",))
        time_var[:] = _seconds_since_fixture_epoch(times)
        time_var.units = _SWATH_TIME_UNITS
        # One scan: footprints numbered 1..n along it, as the real product numbers them.
        _add(tp, "sample_number", (np.arange(n) % 660 + 1).astype(np.int16), fill_i, (1, 660))
        times_ns = np.asarray(times).astype("datetime64[ns]")
        ds.time_coverage_start = np.datetime_as_string(times_ns.min(), unit="us") + "Z"
        ds.time_coverage_end = np.datetime_as_string(times_ns.max(), unit="us") + "Z"

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
    times: np.ndarray | None = None,
    filename: str = "cer_cldpix_fixture.nc",
    latitude_valid_range: tuple[float, float] | None = (0.0, 180.0),
    omit_fill_value: tuple[str, ...] = (),
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
    - ``Observation_Time``, per pixel, in fractional UTC hours of the day (every
      pixel of a scanline shares its time), dated by the ``time_coverage_start``
      global attribute as the real product is
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
    times : np.ndarray, optional
        One observation time per scanline (``datetime64``). Default: every scanline
        at :data:`SWATH_FIXTURE_TIME`.
    filename : str, optional
        Name of the file written under ``tmp_path``.
    latitude_valid_range : tuple of (float, float) or None, optional
        ``valid_range`` written on ``Latitude``; ``None`` writes none. Default
        ``(0, 180)``, the colatitude range the real product declares.
    omit_fill_value : tuple of str, optional
        Names of variables written without a ``_FillValue`` attribute.

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
    if times is None:
        times = np.full(shape[0], SWATH_FIXTURE_TIME, dtype="datetime64[ns]")

    out_path = tmp_path / filename

    with netCDF4.Dataset(str(out_path), "w") as ds:
        ds.createDimension("Scanlines", shape[0])
        ds.createDimension("Pixels", shape[1])
        dims = ("Scanlines", "Pixels")
        # Real CLDPIX: per-pixel fractional UTC hours of the day, dated by the global
        # time_coverage_start attribute (one hour per granule, so one date serves all).
        times = np.asarray(times).astype("datetime64[ns]")
        day = times[0].astype("datetime64[D]")
        hours = ((times - day.astype("datetime64[ns]")) / np.timedelta64(1, "h")).astype(np.float32)
        time_var = ds.createVariable("Observation_Time", np.float32, dims, fill_value=fill_f)
        time_var[:] = np.repeat(hours[:, None], shape[1], axis=1)
        time_var.units = "hours"
        time_var.valid_range = np.array([0.0, 23.999998], dtype=np.float32)
        ds.time_coverage_start = np.datetime_as_string(times.min(), unit="us") + "Z"
        ds.time_coverage_end = np.datetime_as_string(times.max(), unit="us") + "Z"

        def _add(name, data, fill, valid_range=None):
            fill = False if name in omit_fill_value else fill
            var = ds.createVariable(name, data.dtype, dims, fill_value=fill)
            var[:] = data
            if valid_range is not None:
                var.valid_range = np.array(valid_range, dtype=data.dtype)

        # Real CLDPIX files store colatitude under the name "Latitude", with a
        # [0, 180] valid_range. Convert the caller's true latitudes so the
        # fixture exercises the reader's colatitude → latitude conversion.
        colatitudes = 90.0 - lats.astype(np.float32)
        _add("Latitude", colatitudes.astype(np.float32), fill_f, latitude_valid_range)
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


def make_l1b_radiometer_fixture(
    directory: Path,
    n_footprints: int = 8,
    *,
    n_invalid: int = 0,
) -> Path:
    """Write a synthetic L1B RAD-4CH NetCDF file (input to the radiometer-timescale FMATCH runners).

    Contains exactly the variables that
    :func:`libera_utils.footprint_matching.footprint_match_algorithm.load_l1b_radiometer_inputs` reads: the CF-encoded
    ``radiometer_time`` coordinate, the geolocation and Sun-surface-sensor viewing angles that FMATCH passes
    through verbatim, and the scan-reference geometry (subsatellite point, satellite radius and cone-angle rate) that feeds the
    ray-traced bounding box and angular PSF weigher. The values are physically plausible but arbitrary; only the
    variable names, dtypes and time encoding are the contract under test.

    The file is written under a proper Libera ``L1B RAD-4CH`` filename so that the manifest-driven runners select it
    with :func:`libera_utils.footprint_matching.footprint_match_algorithm.select_manifest_files_by_product_id`.

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
    from libera_utils.footprint_matching.footprint_match_algorithm import L1B_TIME_VARIABLE  # noqa: PLC0415
    from libera_utils.footprint_matching.product import (  # noqa: PLC0415
        L1B_CONE_ANGLE_RATE_VARIABLE,
        L1B_PASSTHROUGH_VARIABLES,
        L1B_SCAN_REFERENCE_VARIABLES,
    )
    from libera_utils.io.filenaming import LiberaDataProductFilename  # noqa: PLC0415

    base_time = np.datetime64("2026-06-11T00:00:00", "ns")
    cadence = np.timedelta64(10_000_000, "ns")  # 100 Hz
    times = base_time + np.arange(n_footprints, dtype="int64") * cadence

    # Spread the footprints over a plausible range for each quantity so tests can tell the columns apart. The angles
    # stay inside the product definition's valid ranges (SZA [0,180], VZA [0,90], RAA [0,360]). The subsatellite point
    # sits a few degrees off the footprint boresight so the ray-traced box has a real scan azimuth to orient by.
    ranges = {
        "latitude": (-60.0, 60.0),
        "longitude": (-170.0, 170.0),
        # Surface height of the boresight point above the WGS84 ellipsoid, metres.
        "altitude": (0.0, 4000.0),
        "solar_zenith_angle": (10.0, 80.0),
        "viewing_zenith_angle": (0.0, 45.0),
        "relative_azimuth_angle": (0.0, 350.0),
        "subsatellite_latitude": (-58.0, 62.0),
        "subsatellite_longitude": (-168.0, 172.0),
        # Spacecraft distance from Earth's center for an ~835 km orbit over these latitudes.
        "satellite_radius_km": (7205.0, 7215.0),
    }
    data_vars = {}
    # The pass-through columns are float32 to match the product; the scan-reference geometry is float64.
    for fmatch_name, l1b_name in L1B_PASSTHROUGH_VARIABLES.items():
        low, high = ranges[fmatch_name]
        values = np.linspace(low, high, n_footprints, dtype=np.float32)
        if n_invalid:
            # NaN is how the real L1B marks samples whose boresight misses the Earth.
            values[:n_invalid] = np.nan
        data_vars[l1b_name] = ((L1B_TIME_VARIABLE,), values)
    for fmatch_name, l1b_name in L1B_SCAN_REFERENCE_VARIABLES.items():
        low, high = ranges[fmatch_name]
        values = np.linspace(low, high, n_footprints, dtype=np.float64)
        if n_invalid:
            values[:n_invalid] = np.nan
        data_vars[l1b_name] = ((L1B_TIME_VARIABLE,), values)
    # Cone-angle rate sweeps through zero so both scanning and (near-)stationary samples are represented.
    cone_rate = np.linspace(-60.0, 60.0, n_footprints, dtype=np.float64)
    data_vars[L1B_CONE_ANGLE_RATE_VARIABLE] = ((L1B_TIME_VARIABLE,), cone_rate)

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


def make_cf_cam_camtime_dataset(
    n_images: int = 2,
    n_pixels_x: int = 3,
    n_pixels_y: int = 3,
    *,
    n_padded: int = 0,
    with_cloud_fraction: bool = True,
) -> xr.Dataset:
    """Build a synthetic CF-CAM-CAMTIME dataset carrying the variables FMATCH reads its pseudo-footprints from.

    Each L1B camera pixel is its own pseudo-footprint (a 1 x 1 pixel block on a ~0.5 degree grid), so a full image has
    ``n_pixels_x * n_pixels_y`` footprints laid out row-major along ``PSEUDOFOOTPRINT``. Each image is offset in
    latitude so the images do not overlap.

    Parameters
    ----------
    n_images : int, optional
        Number of camera images (length of the ``CAMERA_TIME`` axis). Defaults to 2.
    n_pixels_x, n_pixels_y : int, optional
        Pixel-grid dimensions of each image. Default to 3 x 3.
    n_padded : int, optional
        How many trailing cells of the last image are padding (fill) rather than real footprints, as when an image
        is segmented into fewer subsections than the widest one. Defaults to 0.
    with_cloud_fraction : bool, optional
        Whether to include the ``cloud_fraction`` variable (percent). Defaults to True.

    Returns
    -------
    xarray.Dataset
        The dataset, with ``n_images * n_pixels_x * n_pixels_y - n_padded`` real footprint cells.
    """
    n_cells = n_pixels_x * n_pixels_y
    shape = (n_images, n_cells)
    dims = ("CAMERA_TIME", "PSEUDOFOOTPRINT")
    times = np.datetime64("2026-06-11T00:00:00", "ns") + np.arange(n_images, dtype="int64") * np.timedelta64(1, "s")

    # Row-major pixel indices of each cell within its image.
    pixel_x = np.broadcast_to(np.repeat(np.arange(n_pixels_x), n_pixels_y), shape).copy()
    pixel_y = np.broadcast_to(np.tile(np.arange(n_pixels_y), n_pixels_x), shape).copy()
    image_offset = 5.0 * np.arange(n_images)[:, None]
    lat = 10.0 + image_offset + 0.5 * pixel_x
    lon = 20.0 + 0.5 * pixel_y

    def const(value: float) -> np.ndarray:
        return np.full(shape, value, dtype=float)

    floats = {
        "latitude": lat,
        "longitude": lon,
        "altitude": const(835_000.0),
        "solar_zenith_angle": const(30.0),
        "viewing_zenith_angle": const(10.0),
        "relative_azimuth_angle": const(120.0),
        "psf_bbox_lat_min": lat - 0.25,
        "psf_bbox_lat_max": lat + 0.25,
        "psf_bbox_lon_min": lon - 0.25,
        "psf_bbox_lon_max": lon + 0.25,
    }
    if with_cloud_fraction:
        floats["cloud_fraction"] = np.linspace(0.0, 100.0, n_images * n_cells).reshape(shape)
    ints = {
        "center_pixel_x": pixel_x,
        "center_pixel_y": pixel_y,
        "psf_bbox_flags": np.zeros(shape, dtype=np.int32),
        "q_flags": np.zeros(shape, dtype=np.int32),
    }
    pixel_bounds = {
        "camera_pixel_x_min": pixel_x,
        "camera_pixel_x_max": pixel_x.copy(),
        "camera_pixel_y_min": pixel_y,
        "camera_pixel_y_max": pixel_y.copy(),
    }
    if n_padded:
        for values in floats.values():
            values[-1, -n_padded:] = np.nan
        for values in (*ints.values(), *pixel_bounds.values()):
            values[-1, -n_padded:] = -1

    dataset = xr.Dataset(
        {name: (dims, values) for name, values in {**floats, **ints}.items()},
        coords={
            "CAMERA_TIME": times,
            "PSEUDOFOOTPRINT": np.arange(n_cells, dtype=np.int32),
            **{name: (dims, values.astype(np.int32)) for name, values in pixel_bounds.items()},
        },
    )
    dataset["CAMERA_TIME"].encoding = {
        "units": "nanoseconds since 1958-01-01",
        "calendar": "standard",
        "dtype": "int64",
    }
    return dataset


def make_cf_cam_camtime_fixture(directory: Path, **dataset_kwargs: Any) -> Path:
    """Write a synthetic CF-CAM-CAMTIME NetCDF file (input to the camera-timescale FMATCH runners).

    The file is written under a proper Libera ``CF-CAM-CAMTIME`` filename so that the manifest-driven runners select
    it with :func:`libera_utils.footprint_matching.footprint_match_algorithm.select_manifest_files_by_product_id`.

    Parameters
    ----------
    directory : Path
        Directory to write the file into (e.g. pytest ``tmp_path``).
    **dataset_kwargs
        Passed to :func:`make_cf_cam_camtime_dataset`.

    Returns
    -------
    Path
        Path to the written CF-CAM-CAMTIME NetCDF file.
    """
    from libera_utils.constants import DataProductIdentifier  # noqa: PLC0415
    from libera_utils.io.filenaming import LiberaDataProductFilename  # noqa: PLC0415

    dataset = make_cf_cam_camtime_dataset(**dataset_kwargs)
    times = dataset["CAMERA_TIME"].values
    filename = LiberaDataProductFilename.from_filename_parts(
        product_name=DataProductIdentifier.l2_cf_cam_camtime,
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
