"""Unit tests for ERA5PressureLevelReader.

Uses real synthetic NetCDF4 fixture files (created by
make_era5_pressure_netcdf_fixture) so the xarray slicing, pressure-level
selection, and coordinate handling can be exercised without calling real CDS
services.

Real ERA5 pressure-level files can be downloaded from:
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.era5_pressure import (
    _ERA5_PRESSURE_LEVEL_VARIABLES,
    _ERA5_PRESSURE_LEVELS,
    ERA5PressureLevelReader,
)
from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridTile,
    OperationalMode,
    TileKey,
)
from tests.test_data.footprint_matching.fixtures import (
    era5_pressure_fixture_value,
    make_era5_pressure_netcdf_fixture,
)

# Expected spec count: 5 base variables × 37 configured levels.
_N_SPECS = len(_ERA5_PRESSURE_LEVEL_VARIABLES) * len(_ERA5_PRESSURE_LEVELS)


class TestERA5PressureLevelReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "era5_pressure".
        assert ERA5PressureLevelReader.READER_KEY == "era5_pressure"

    def test_resolution_km(self):
        # Targets the reader's grid resolution attribute; asserts RESOLUTION_KM equals 25.0 km.
        assert ERA5PressureLevelReader.RESOLUTION_KM == 25.0

    def test_variable_count_is_variables_times_levels(self):
        # Targets the VARIABLES count; asserts it equals base variables × pressure levels (_N_SPECS).
        assert len(ERA5PressureLevelReader.VARIABLES) == _N_SPECS

    def test_variable_names_are_variable_major_level_ascending(self):
        # Targets the VARIABLES naming/order; asserts names are variable-major, level-ascending (e.g.
        # temperature_10hPa ... temperature_1000hPa, then geopotential_10hPa ...).
        expected = [
            f"{base}_{level}hPa" for base, _ in _ERA5_PRESSURE_LEVEL_VARIABLES for level in _ERA5_PRESSURE_LEVELS
        ]
        assert [v.name for v in ERA5PressureLevelReader.VARIABLES] == expected

    def test_all_variables_continuous_weighted_mean(self):
        # Targets each spec's continuous/IMAGER config; asserts every VARIABLES entry is float32,
        # weighted_mean, non-categorical, and required_mode IMAGER.
        for v in ERA5PressureLevelReader.VARIABLES:
            assert v.dtype == "float32"
            assert v.aggregation == "weighted_mean"
            assert v.n_categories is None
            assert v.required_mode == OperationalMode.IMAGER

    def test_every_spec_gets_standard_deviation_companion(self):
        # All specs are continuous, so the product variable set doubles; asserts product_variable_specs has
        # 2×_N_SPECS entries including each field and its "_standard_deviation" companion.
        product_specs = ERA5PressureLevelReader.product_variable_specs()
        assert len(product_specs) == 2 * _N_SPECS
        names = {s.name for s in product_specs}
        assert "temperature_500hPa" in names
        assert "temperature_500hPa_standard_deviation" in names


class TestERA5PressureLevelReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        # Targets _load_spatial_region output shape; asserts a 3D array whose axis 0 holds all _N_SPECS layers.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, lats, lons = reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))
        assert data.ndim == 3
        assert data.shape[0] == _N_SPECS

    def test_layer_order_matches_variables(self, tmp_path):
        # The fixture writes a distinct constant per (variable, level); the
        # stacked output layer i must contain exactly the value for VARIABLES[i].
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, _, _ = reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))
        layer = 0
        for var_index in range(len(_ERA5_PRESSURE_LEVEL_VARIABLES)):
            for level in _ERA5_PRESSURE_LEVELS:
                expected = era5_pressure_fixture_value(var_index, level)
                assert np.allclose(data[layer], expected, rtol=1e-6), (var_index, level)
                layer += 1

    def test_data_dtype_is_float32(self, tmp_path):
        # Targets the returned data array dtype; asserts _load_spatial_region yields float32 values.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, _, _ = reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))
        assert data.dtype == np.float32

    def test_lats_are_ascending_order(self, tmp_path):
        # The fixture stores lats in DESCENDING order (ERA5 convention); asserts the reader flips them
        # so that np.diff(lats) >= 0 (ascending) on output.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        _, lats, _ = reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))
        assert np.all(np.diff(lats) >= 0), f"Lats should be ascending but got: {lats}"

    def test_coordinate_dtypes_are_float64(self, tmp_path):
        # Targets the returned coordinate dtypes; asserts both lats and lons arrays are float64.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        _, lats, lons = reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))
        assert lats.dtype == np.float64
        assert lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path):
        # Targets bbox spatial subsetting; asserts a sub-range request returns only lats within that range.
        fixture_path = make_era5_pressure_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4
        )
        reader = ERA5PressureLevelReader(fixture_path)
        # Only request the upper half of the lat range.
        _, lats, _ = reader._load_spatial_region(BoundingBox(2.0, 4.0, 10.0, 12.0))
        assert np.all(lats >= 2.0 - 1e-6)

    def test_no_time_dimension_variant_supported(self, tmp_path):
        # Older CDS downloads may lack the valid_time dimension entirely; asserts such a file still loads
        # to a 3D array with all _N_SPECS layers.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, with_valid_time=False)
        reader = ERA5PressureLevelReader(fixture_path)
        data, _, _ = reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))
        assert data.ndim == 3
        assert data.shape[0] == _N_SPECS


class TestERA5PressureLevelReaderErrors:
    def test_missing_level_raises_value_error(self, tmp_path):
        # Write a file that omits some configured levels: asserts the reader refuses it by raising
        # ValueError("missing configured pressure") rather than substituting neighboring levels.
        partial_levels = tuple(_ERA5_PRESSURE_LEVELS[:-2])  # drop 975 and 1000 hPa
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, levels=partial_levels)
        reader = ERA5PressureLevelReader(fixture_path)
        with pytest.raises(ValueError, match="missing configured pressure"):
            reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))

    def test_missing_variable_raises_key_error(self, tmp_path):
        # Targets loud failure on a missing required variable; asserts _load_spatial_region raises
        # KeyError("missing variable") when a needed field (o3) is stripped.
        import xarray as xr

        # Strip one required variable from an otherwise-valid fixture file.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        with xr.open_dataset(fixture_path) as ds:
            stripped = ds.drop_vars("o3").load()
        stripped_path = tmp_path / "era5_pressure_missing_o3.nc"
        stripped.to_netcdf(stripped_path)

        reader = ERA5PressureLevelReader(stripped_path)
        with pytest.raises(KeyError, match="missing variable"):
            reader._load_spatial_region(BoundingBox(0.0, 2.0, 10.0, 12.0))


class TestERA5PressureLevelReaderLoadTile:
    def test_load_tile_returns_grid_tile(self, tmp_path):
        # Targets load_tile's return type and source label; asserts it returns a GridTile with source "era5_pressure".
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0)
        reader = ERA5PressureLevelReader(fixture_path)
        # TileKey(lat_idx=45, lon_idx=90) → lat [0, 2°], lon [0, 2°]
        tile = reader.load_tile(TileKey("era5_pressure", 45, 90))
        assert isinstance(tile, GridTile)
        assert tile.source == "era5_pressure"

    def test_load_tile_timestamp_source_is_none(self, tmp_path):
        # ERA5 is a reanalysis product; asserts load_tile leaves tile.timestamp_source None (no instrument timestamp).
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0)
        reader = ERA5PressureLevelReader(fixture_path)
        tile = reader.load_tile(TileKey("era5_pressure", 45, 90))
        assert tile.timestamp_source is None
