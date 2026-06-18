"""Unit tests for ERA5Reader.

Uses real synthetic NetCDF4 fixture files (created by make_era5_netcdf_fixture)
so that the xarray slicing and coordinate handling can be exercised without
calling real CDS services.

Real ERA5 files can be downloaded from:
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey
from tests.test_data.footprint_matching.fixtures import make_era5_netcdf_fixture


class TestERA5ReaderClassAttributes:
    def test_reader_key(self):
        assert ERA5Reader.READER_KEY == "era5"

    def test_resolution_km(self):
        assert ERA5Reader.RESOLUTION_KM == 25.0

    def test_required_mode_is_cam(self):
        assert ERA5Reader.REQUIRED_MODE == OperationalMode.CAM

    def test_variables_has_two_entries(self):
        assert len(ERA5Reader.VARIABLES) == 2

    def test_variable_names(self):
        names = [v.name for v in ERA5Reader.VARIABLES]
        assert "wind_u10" in names
        assert "wind_v10" in names

    def test_variables_have_no_categories(self):
        for v in ERA5Reader.VARIABLES:
            assert v.n_categories is None


class TestERA5ReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, lats, lons = reader._load_spatial_region(bbox)
        # 3D: (2 variables, n_lat, n_lon)
        assert data.ndim == 3
        assert data.shape[0] == 2

    def test_data_axis0_is_variables(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(tmp_path, u10_fill=2.5, v10_fill=-1.5, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = reader._load_spatial_region(bbox)
        # data[0] should be u10 (all 2.5) and data[1] should be v10 (all -1.5)
        assert np.allclose(data[0], 2.5, atol=1e-4)
        assert np.allclose(data[1], -1.5, atol=1e-4)

    def test_lats_are_ascending_order(self, tmp_path):
        # The fixture stores lats in DESCENDING order (ERA5 convention);
        # the reader must flip them to ASCENDING order on output.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        _, lats, _ = reader._load_spatial_region(bbox)
        assert np.all(np.diff(lats) >= 0), f"Lats should be ascending but got: {lats}"

    def test_data_dtype_is_float32(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(tmp_path, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = reader._load_spatial_region(bbox)
        assert data.dtype == np.float32

    def test_lats_dtype_is_float64(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(tmp_path, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        _, lats, lons = reader._load_spatial_region(bbox)
        assert lats.dtype == np.float64
        assert lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path):
        # Fixture covers lat 0–2, lon 10–12 with 8 points in each axis.
        # Request only the lower half of the lat range.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        # Only request the upper half of the lat range
        bbox = BoundingBox(2.0, 4.0, 10.0, 12.0)
        data, lats, _ = reader._load_spatial_region(bbox)
        # All returned lats should be >= 2.0
        assert np.all(lats >= 2.0 - 1e-6)


class TestERA5ReaderLoadTile:
    def test_load_tile_returns_grid_tile(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        # TileKey(lat_idx=45, lon_idx=90) → lat [0, 2°], lon [0, 2°]
        key = TileKey("era5", 45, 90)
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)

    def test_load_tile_source_is_era5(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        key = TileKey("era5", 45, 90)
        tile = reader.load_tile(key)
        assert tile.source == "era5"

    def test_load_tile_timestamp_source_is_none(self, tmp_path):
        # ERA5 is a static reanalysis product; no instrument timestamp.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        key = TileKey("era5", 45, 90)
        tile = reader.load_tile(key)
        assert tile.timestamp_source is None


class TestERA5ReaderValidTimeDimension:
    """ERA5 files downloaded via the new CDS API use 'valid_time' as the time dimension name.

    The reader must drop ANY dimension whose name contains the substring 'time',
    not just the exact name 'time'. This class verifies the fix for that edge case.
    """

    def test_valid_time_dimension_is_dropped(self, tmp_path):
        from tests.test_data.footprint_matching.fixtures import make_era5_valid_time_fixture

        fixture_path = make_era5_valid_time_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, lats, lons = reader._load_spatial_region(bbox)
        # Without the fix the shape would be (2, 1, n_lat, n_lon); the time dim must be gone.
        assert data.ndim == 3
        assert data.shape[0] == 2

    def test_valid_time_values_correct(self, tmp_path):
        from tests.test_data.footprint_matching.fixtures import make_era5_valid_time_fixture

        fixture_path = make_era5_valid_time_fixture(
            tmp_path,
            lat_min=0.0,
            lat_max=2.0,
            lon_min=10.0,
            lon_max=12.0,
            n_lat=4,
            n_lon=4,
            u10_fill=3.0,
            v10_fill=-2.0,
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = reader._load_spatial_region(bbox)
        assert np.allclose(data[0], 3.0, atol=1e-4)
        assert np.allclose(data[1], -2.0, atol=1e-4)
