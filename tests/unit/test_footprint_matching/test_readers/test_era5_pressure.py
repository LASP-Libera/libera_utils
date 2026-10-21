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
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode
from tests.test_data.footprint_matching.fixtures import (
    ERA5_FIXTURE_TIME,
    era5_pressure_fixture_value,
    make_era5_pressure_netcdf_fixture,
)
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = ERA5_FIXTURE_TIME

# Expected spec count: 5 base variables × 37 configured levels.
_N_SPECS = len(_ERA5_PRESSURE_LEVEL_VARIABLES) * len(_ERA5_PRESSURE_LEVELS)


class TestERA5PressureLevelReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "era5_pressure".
        assert ERA5PressureLevelReader.READER_KEY == "era5_pressure"

    def test_resolution_km(self):
        # Targets the reader's grid resolution attribute; asserts RESOLUTION_KM equals the 0.25 degree spacing, 28 km.
        assert ERA5PressureLevelReader.RESOLUTION_KM == 28.0

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
        # Targets samples_in output shape; asserts a 3D array whose axis 0 holds all _N_SPECS layers.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, lats, lons = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert data.ndim == 3
        assert data.shape[0] == _N_SPECS

    def test_layer_order_matches_variables(self, tmp_path):
        # The fixture writes a distinct constant per (variable, level); the
        # stacked output layer i must contain exactly the value for VARIABLES[i].
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, _, _ = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        layer = 0
        for var_index in range(len(_ERA5_PRESSURE_LEVEL_VARIABLES)):
            for level in _ERA5_PRESSURE_LEVELS:
                expected = era5_pressure_fixture_value(var_index, level)
                assert np.allclose(data[layer], expected, rtol=1e-6), (var_index, level)
                layer += 1

    def test_data_dtype_is_float32(self, tmp_path):
        # Targets the returned data array dtype; asserts samples_in yields float32 values.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, _, _ = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert data.dtype == np.float32

    def test_lats_are_ascending_order(self, tmp_path):
        # The fixture stores lats in DESCENDING order (ERA5 convention); asserts the reader flips them
        # so that np.diff(lats) >= 0 (ascending) on output.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        _, lats, _ = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert np.all(np.diff(lats) >= 0), f"Lats should be ascending but got: {lats}"

    def test_coordinate_dtypes_are_float64(self, tmp_path):
        # Targets the returned coordinate dtypes; asserts both lats and lons arrays are float64.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        _, lats, lons = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert lats.dtype == np.float64
        assert lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path):
        # Targets bbox spatial subsetting; asserts a sub-range request returns only lats within that range.
        fixture_path = make_era5_pressure_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4
        )
        reader = ERA5PressureLevelReader(fixture_path)
        # Only request the upper half of the lat range.
        _, lats, _ = _region(reader, BoundingBox(2.0, 4.0, 10.0, 12.0), _T)
        assert np.all(lats >= 2.0 - 1e-6)

    def test_no_time_dimension_variant_is_rejected(self, tmp_path):
        # A download with no time axis cannot be matched to a footprint time; the reader refuses it.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, with_valid_time=False)
        reader = ERA5PressureLevelReader(fixture_path)
        with pytest.raises(ValueError, match="time dimension"):
            reader.samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), _T)


class TestERA5PressureLevelReaderErrors:
    def test_missing_level_raises_value_error(self, tmp_path):
        # Write a file that omits some configured levels: asserts the reader refuses it by raising
        # ValueError("missing configured pressure") rather than substituting neighboring levels.
        partial_levels = tuple(_ERA5_PRESSURE_LEVELS[:-2])  # drop 975 and 1000 hPa
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, levels=partial_levels)
        reader = ERA5PressureLevelReader(fixture_path)
        with pytest.raises(ValueError, match="missing configured pressure"):
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)

    def test_missing_variable_raises_key_error(self, tmp_path):
        # Targets loud failure on a missing required variable; asserts samples_in raises
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
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)


class TestERA5PressureLevelReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0)
        samples = ERA5PressureLevelReader(fixture_path).samples_in(BoundingBox(0.0, 2.0, 0.0, 2.0), _T)
        assert samples.source == "era5_pressure"
        assert samples.timestamp_source is None
        assert samples.values.shape == (_N_SPECS, 16)


class TestERA5PressureLevelReaderRealSubset:
    """A real 37-level ERA5 pressure-level download cut to 50N..40N, 110W..100W at 2025-01-01T00.

    Unlike the single-level download, this one already stores longitude -180..180, so
    the window is served without a longitude shift. Pinned values are the source
    granule's own at 45N, 105W, 500 hPa.
    """

    _TIME = np.datetime64("2025-01-01T00:00:00", "ns")

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return ERA5PressureLevelReader(fmatch_real_data_path / "era5_pressure_levels_20250101T00_subset.nc")

    def test_every_level_of_the_window_is_served(self, reader):
        grid = reader._load_step(0)
        assert (grid.lat_axis[0], grid.lat_axis[-1]) == (40.0, 50.0)
        assert (grid.lon_axis[0], grid.lon_axis[-1]) == (-110.0, -100.0)
        assert grid.data.shape == (_N_SPECS, 41, 41)
        assert np.isfinite(grid.data).all()

    def test_values_at_a_grid_point_are_the_granules(self, reader):
        samples = reader.samples_in(BoundingBox(44.9, 45.1, -105.1, -104.9), self._TIME)
        names = [v.name for v in ERA5PressureLevelReader.VARIABLES]
        values = sample_at(samples, 45.0, -105.0)
        assert values[names.index("temperature_500hPa")] == pytest.approx(240.2764, abs=1e-3)
        assert values[names.index("geopotential_500hPa")] == pytest.approx(53364.29, abs=0.1)
