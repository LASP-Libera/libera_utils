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

VIIRS / merged AOD (AERDB_D3_GEOLEO, NetCDF4 with per-sensor groups):
    NASA Deep Blue: https://deepblue.gsfc.nasa.gov/
    Example filename: AERDB_D3_GEOLEO_Merged.A2020121.001.2024121023016.nc

CERES SSF / FLASHFlux (NetCDF4, per-footprint swath):
    NASA CERES: https://ceres.larc.nasa.gov/data/#ssf-level-2
    Example filename: CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc

CERES CLDPIX (NetCDF4, imager-pixel swath):
    NASA CERES: https://ceres.larc.nasa.gov/data/
    Example filename: CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc
"""
from __future__ import annotations

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


def make_aod_noaa20_fixture(
    tmp_path: Path,
    n_lat: int = 4,
    n_lon: int = 8,
    lat_min: float = 0.5,
    lat_max: float = 3.5,
    lon_min: float = 10.5,
    lon_max: float = 17.5,
    aod_fill: float = 0.2,
    include_fill_pixel: bool = True,
    merged_decoy_value: float | None = None,
) -> Path:
    """Write a synthetic AERDB_D3_GEOLEO NOAA-20 VIIRS AOD NetCDF4 file.

    Replicates the relevant structure of the real Deep Blue GEO-LEO merged
    product as consumed by ``VIIRSAODReader`` (which reads the per-sensor
    NOAA-20 VIIRS group, not the cross-sensor ``Merged`` group):
    - Root-level ``Latitude`` (ascending) and ``Longitude`` coordinate arrays
    - A ``NOAA20_VIIRS`` group containing ``Aerosol_Optical_Thickness_550_Land_Ocean``
      with **(Latitude, Longitude)** dimension order (no transpose needed) and
      ``_FillValue = -999.0``

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
    include_fill_pixel : bool
        If True, set pixel [0, 0] to the −999.0 fill sentinel so tests can
        verify fill → NaN conversion. Default True.
    merged_decoy_value : float | None
        If not None, additionally write a decoy ``Merged`` group filled with this
        constant value. The reader must ignore it (it reads ``NOAA20_VIIRS``), so
        tests can assert the reader is reading the correct group. Default None
        (no ``Merged`` group written).

    Returns
    -------
    Path
        Path to the created NetCDF4 fixture file.
    """
    import netCDF4  # noqa: PLC0415

    out_path = tmp_path / "aerdb_d3_geoleo_merged_fixture.nc"
    lats = np.linspace(lat_min, lat_max, n_lat)  # ascending
    lons = np.linspace(lon_min, lon_max, n_lon)

    with netCDF4.Dataset(str(out_path), "w") as ds:
        ds.createDimension("Latitude", n_lat)
        ds.createDimension("Longitude", n_lon)

        lat_var = ds.createVariable("Latitude", "f4", ("Latitude",))
        lat_var[:] = lats.astype(np.float32)
        lat_var.units = "degrees_north"

        lon_var = ds.createVariable("Longitude", "f4", ("Longitude",))
        lon_var[:] = lons.astype(np.float32)
        lon_var.units = "degrees_east"

        grp = ds.createGroup("NOAA20_VIIRS")
        # Note: dimension order (Latitude, Longitude) — matches the real product.
        aod_var = grp.createVariable(
            "Aerosol_Optical_Thickness_550_Land_Ocean", "f4", ("Latitude", "Longitude"),
            fill_value=-999.0,
        )
        data = np.full((n_lat, n_lon), aod_fill, dtype=np.float32)
        if include_fill_pixel:
            data[0, 0] = -999.0
        aod_var[:] = data

        # Optionally write a decoy "Merged" group with a distinct constant value.
        # The reader reads NOAA20_VIIRS, so it must never return these values —
        # this lets a test guard against an accidental revert to the merged group.
        if merged_decoy_value is not None:
            decoy_grp = ds.createGroup("Merged")
            decoy_var = decoy_grp.createVariable(
                "Aerosol_Optical_Thickness_550_Land_Ocean", "f4", ("Latitude", "Longitude"),
                fill_value=-999.0,
            )
            decoy_var[:] = np.full((n_lat, n_lon), merged_decoy_value, dtype=np.float32)

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
    - 1-D ``Footprints`` dimension and a ``LowerUpper`` dimension of length 2
    - ``Time_and_Position/instrument_fov_latitude`` and ``…_longitude``
      (**longitude stored in the 0..360 convention**, matching the real file)
    - One variable per supported reader field across the corresponding groups,
      with float fill ``3.4028235e38`` and int16 fill ``32767``

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
        cloud_water_particle_radius_lower = np.array(
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)[:n]
    if cloud_ice_particle_radius_lower is None:
        cloud_ice_particle_radius_lower = np.array(
            [20.0, 25.0, 30.0, 35.0, 40.0, 45.0], dtype=np.float32)[:n]
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

    def _add_lower_upper(grp, name, lower_values, fill, valid_range=None):
        """Write a 2-D (Footprints, LowerUpper) variable with lower-layer values."""
        var = grp.createVariable(name, "f4", ("Footprints", "LowerUpper"), fill_value=fill)
        data = np.full((n, 2), fill, dtype=np.float32)
        # Index 0 is the lower cloud layer — the layer selected by _CLOUD_LAYER_INDEX.
        data[:, 0] = lower_values.astype(np.float32)
        var[:] = data
        if valid_range is not None:
            var.valid_range = np.array(valid_range, dtype=np.float32)

    with netCDF4.Dataset(str(out_path), "w") as ds:
        ds.createDimension("Footprints", n)
        ds.createDimension("LowerUpper", 2)

        tp = ds.createGroup("Time_and_Position")
        _add(tp, "instrument_fov_latitude", lats.astype(np.float32), fill_f, (-90.0, 90.0))
        _add(tp, "instrument_fov_longitude", lons_0360.astype(np.float32), fill_f, (0.0, 360.0))

        aux = ds.createGroup("Auxillary_Properties")
        _add(aux, "aerosol_optical_depth", aerosol_optical_depth.astype(np.float32), fill_f, (0.0, 8.0))

        clr = ds.createGroup("Clear_Footprint_Area")
        _add(clr, "clear_coverage", clear_coverage.astype(np.float32), fill_f, (0.0, 100.0))

        cif = ds.createGroup("Cloudy_Imager_Footprint_Layer")
        _add_lower_upper(cif, "cloud_optical_depth_mean",
                         cloud_optical_depth_lower, fill_f, (0.0, 512.0))
        # Phase-separated effective particle radii. SSF does not provide a single
        # blended radius — water and ice clouds are retrieved independently at 3.7 μm.
        _add_lower_upper(cif, "cloud_water_particle_radius_37um_mean",
                         cloud_water_particle_radius_lower, fill_f, (2.0, 60.0))
        _add_lower_upper(cif, "cloud_ice_particle_radius_37um_mean",
                         cloud_ice_particle_radius_lower, fill_f, (5.0, 90.0))

        scn = ds.createGroup("Scene_Type")
        _add(scn, "cloud_classification", cloud_classification.astype(np.int16), fill_i, (0, 32766))
        _add(scn, "shortwave_adm_type", shortwave_adm_type.astype(np.int16), fill_i, (0, 5000))
        _add(scn, "longwave_adm_type", longwave_adm_type.astype(np.int16), fill_i, (0, 5000))

    return out_path


def make_cldpix_fixture(
    tmp_path: Path,
    lats: np.ndarray | None = None,
    lons_0360: np.ndarray | None = None,
) -> Path:
    """Write a synthetic CERES CLDPIX (imager-pixel swath) NetCDF4 file.

    Replicates the flat, 2-D ``(Scanlines, Pixels)`` structure of the real
    CLDPIX product:
    - 2-D ``Latitude`` / ``Longitude`` arrays (**longitude in the 0..360
      convention**)
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
        2-D ``(Scanlines, Pixels)`` geolocation arrays. ``lons_0360`` is in the
        0..360 convention. Defaults place all pixels near lat ≈ 40°, lon ≈ −15°
        (written as 345° in the file).

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

        _add("Latitude", lats.astype(np.float32), fill_f, (-90.0, 90.0))
        _add("Longitude", lons_0360.astype(np.float32), fill_f, (0.0, 360.0))

        # Continuous cloud properties (constant values for easy assertions).
        _add("Eff_Cld_Optical_Depth", np.full(shape, 4.0, np.float32), fill_f, (0.25, 150.0))
        _add("Cld_Water_Path", np.full(shape, 100.0, np.float32), fill_f, (0.0, 10000.0))
        _add("Eff_Cld_Temp", np.full(shape, 270.0, np.float32), fill_f, (190.0, 350.0))
        _add("Eff_Cld_Height", np.full(shape, 5.0, np.float32), fill_f, (0.0, 18.0))
        # Reversed valid_range exactly like the real file.
        _add("Eff_Cld_Pressure", np.full(shape, 800.0, np.float32), fill_f, (1100.0, 10.0))
        _add("Top_Cld_Height", np.full(shape, 6.0, np.float32), fill_f)
        # Effective cloud particle radius (μm) — blended water+ice value from
        # the CERES retrieval algorithm. Constant 10.0 μm for easy assertions.
        _add("Cld_Radius", np.full(shape, 10.0, np.float32), fill_f, (2.0, 60.0))

        # Categorical (int8) fields; one pixel set to the 127 fill sentinel.
        phase = np.full(shape, 1, dtype=np.int8)
        phase.flat[-1] = fill_i8
        _add("Cloud_Particle_Phase", phase, fill_i8, (1, 5))
        _add("CERES_Cloud_Mask", np.full(shape, 1, np.int8), fill_i8, (0, 3))

    return out_path
