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
    Example filename: NISE_AMSR2_20260610.HDFEOS
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
"""

from pathlib import Path

import numpy as np
import xarray as xr

# Constant fill values for the five non-wind ERA5 single-level variables written
# by the ERA5 fixtures. Chosen to be physically plausible AND mutually distinct
# so tests can assert that each output layer came from the right file variable.
# The winds take their values from the u10_fill / v10_fill keyword parameters instead.
ERA5_SURFACE_FILLS: dict[str, float] = {
    "t2m": 285.0,  # 2 m temperature (K)
    "d2m": 280.0,  # 2 m dewpoint temperature (K)
    "sp": 101325.0,  # surface pressure (Pa)
    "z": 500.0,  # surface geopotential (m^2/s^2)
    "fal": 0.3,  # forecast albedo (0-1)
}

# Metadata written on each single-level fixture variable, mirroring real CDS files.
_ERA5_SURFACE_ATTRS: dict[str, dict[str, str]] = {
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


def make_era5_surface_netcdf_fixture(
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
    filename: str = "era5_surface_fixture.nc",
    time_dim: str | None = "time",
) -> Path:
    """Write a synthetic ERA5 single-levels NetCDF4 file to ``tmp_path``.

    The real ERA5 files from CDS contain the single-level variables on a global
    lat/lon grid with the latitude dimension in DESCENDING order (90 -> -90) and a
    leading hourly time dimension. This factory reproduces that convention in a
    small grid for testing, writing all seven variables the ERA5SurfaceReader expects
    (u10, v10, t2m, d2m, sp, z, fal), each a constant-valued grid. The non-wind
    variables carry the constant ``ERA5_SURFACE_FILLS`` values.

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
    time_dim : str or None, optional
        Name of the leading time dimension: ``"time"`` (default) as older CDS
        downloads name it, or ``"valid_time"`` as the current CDS API does. ``None``
        writes the first step's grids with no time dimension at all.

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    lats = np.linspace(lat_max, lat_min, n_lat)  # descending, real ERA5 convention
    lons = np.linspace(lon_min, lon_max, n_lon)
    if times is None:
        times = np.array([ERA5_FIXTURE_TIME], dtype="datetime64[ns]")
    times = np.asarray(times).astype("datetime64[ns]")

    fills = {"u10": u10_fill, "v10": v10_fill, **ERA5_SURFACE_FILLS}
    coords = {
        "latitude": xr.DataArray(lats, dims=["latitude"], attrs={"units": "degrees_north"}),
        "longitude": xr.DataArray(lons, dims=["longitude"], attrs={"units": "degrees_east"}),
    }
    if time_dim is not None:
        coords[time_dim] = xr.DataArray(times, dims=[time_dim])

    data_vars = {}
    for name, fill in fills.items():
        data = np.full((times.size, n_lat, n_lon), fill, dtype=np.float32)
        if name == "u10" and u10_per_step is not None:
            for step, value in enumerate(np.asarray(u10_per_step, dtype=np.float32)):
                data[step] = value
        if time_dim is None:
            data_vars[name] = xr.DataArray(data[0], dims=["latitude", "longitude"], attrs=_ERA5_SURFACE_ATTRS[name])
        else:
            data_vars[name] = xr.DataArray(
                data, dims=[time_dim, "latitude", "longitude"], attrs=_ERA5_SURFACE_ATTRS[name]
            )
    out_path = tmp_path / filename
    xr.Dataset(data_vars, coords=coords).to_netcdf(out_path)
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
