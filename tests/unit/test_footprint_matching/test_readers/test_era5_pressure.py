"""Unit tests for ERA5PressureLevelReader.

Uses real synthetic NetCDF4 fixture files (created by
make_era5_pressure_netcdf_fixture) so the xarray slicing, pressure-level
selection, and coordinate handling can be exercised without calling real CDS
services. What both ERA5 readers share is tested once, for both, in ``test_era5_surface.py``.

Real ERA5 pressure-level files can be downloaded from:
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register
"""

import numpy as np
import pytest
import xarray as xr

from libera_utils.footprint_matching.readers.era5_pressure import (
    _ERA5_PRESSURE_LEVEL_VARIABLES,
    _ERA5_PRESSURE_LEVELS,
    ERA5PressureLevelReader,
)
from libera_utils.footprint_matching.types import BoundingBox
from tests.plugins.footprint_matching_fixtures import (
    ERA5_FIXTURE_TIME,
    era5_pressure_fixture_value,
    make_era5_pressure_netcdf_fixture,
)
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = ERA5_FIXTURE_TIME

# Expected spec count: 5 base variables × 37 configured levels.
_N_SPECS = len(_ERA5_PRESSURE_LEVEL_VARIABLES) * len(_ERA5_PRESSURE_LEVELS)


class TestERA5PressureLevelReaderPlanes:
    def test_planes_hold_their_variable_and_level(self, tmp_path):
        """Plane ``i`` of the served data holds, as float32, the variable and pressure level ``VARIABLES[i]`` names.

        The pressure-level dimension is flattened into axis 0 with the time axis removed; a plane read from the wrong
        level or field publishes, for example, 850 hPa temperature as 500 hPa temperature, and a wider dtype would
        double an already large step. The source key tells the samples apart from the single-level reader's, and
        ``timestamp_source`` is reserved for cloud-property products tied to a radiometer or camera time.

        The synthetic 4 x 4 file stores levels descending and fills each (variable, level) layer with its own constant
        from ``era5_pressure_fixture_value``; a full-extent query must fold back to one float32 plane per spec, each
        the constant for its spec (so levels are selected by value, not by position), with source ``"era5_pressure"``
        and no timestamp source.
        """
        reader = ERA5PressureLevelReader([make_era5_pressure_netcdf_fixture(tmp_path)])
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = _region(reader, bbox, _T)
        assert data.shape == (_N_SPECS, 4, 4)
        assert data.dtype == np.float32
        layer = 0
        for var_index in range(len(_ERA5_PRESSURE_LEVEL_VARIABLES)):
            for level in _ERA5_PRESSURE_LEVELS:
                expected = era5_pressure_fixture_value(var_index, level)
                assert np.allclose(data[layer], expected, rtol=1e-6), (var_index, level)
                layer += 1
        samples = reader.samples_in(bbox, _T)
        assert samples.source == "era5_pressure"
        assert samples.timestamp_source is None


class TestERA5PressureLevelReaderErrors:
    def test_missing_level_raises_value_error(self, tmp_path):
        """``ERA5PressureLevelReader`` raises ``ValueError`` when a file lacks a configured pressure level.

        A partial download must be fixed when it is acquired; substituting a neighboring level would publish values from
        the wrong altitude under that level's name.

        Writes a file without the 975 and 1000 hPa levels and asserts a query raises ``ValueError`` naming the missing
        configured levels.
        """
        # Write a file that omits some configured levels: asserts the reader refuses it by raising
        # ValueError("missing configured pressure") rather than substituting neighboring levels.
        partial_levels = tuple(_ERA5_PRESSURE_LEVELS[:-2])  # drop 975 and 1000 hPa
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, levels=partial_levels)
        reader = ERA5PressureLevelReader([fixture_path])
        with pytest.raises(ValueError, match="missing configured pressure"):
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)

    def test_missing_variable_raises_key_error(self, tmp_path):
        """``ERA5PressureLevelReader`` raises ``KeyError`` when a file lacks one of the five pressure-level fields.

        A download configured without a field would otherwise leave its planes unfilled, and the product would carry
        values that were never read.

        Drops ``o3`` from a valid synthetic file and asserts a query raises ``KeyError`` naming the missing variable.
        """
        # Strip one required variable from an otherwise-valid fixture file.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        with xr.open_dataset(fixture_path) as ds:
            stripped = ds.drop_vars("o3").load()
        stripped_path = tmp_path / "era5_pressure_missing_o3.nc"
        stripped.to_netcdf(stripped_path)

        reader = ERA5PressureLevelReader([stripped_path])
        with pytest.raises(KeyError, match="missing variable"):
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)


class TestERA5PressureLevelReaderRealSubset:
    """A real 37-level ERA5 pressure-level download cut to 50N..40N, 110W..100W at 2025-01-01T00.

    Unlike the single-level download, this one already stores longitude -180..180, so
    the window is served without a longitude shift. Pinned values are the source
    granule's own at 45N, 105W, 500 hPa.
    """

    _TIME = np.datetime64("2025-01-01T00:00:00", "ns")

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return ERA5PressureLevelReader([fmatch_real_data_path / "era5_pressure_levels_20250101T00_subset.nc"])

    def test_every_level_of_the_window_is_served(self, reader):
        """``ERA5PressureLevelReader._load_step`` serves every configured level of a real 37-level download.

        The reader must accept a CDS download as distributed and produce a plane for each (variable, level) pair, with
        no gaps.

        Loads the one step of a real 41 x 41 subset already on -180..180 longitudes and asserts the axes span 40 to 50 N
        and -110 to -100, one plane per spec, and every value finite.
        """
        grid = reader._load_step(0)
        assert (grid.lat_axis[0], grid.lat_axis[-1]) == (40.0, 50.0)
        assert (grid.lon_axis[0], grid.lon_axis[-1]) == (-110.0, -100.0)
        assert grid.data.shape == (_N_SPECS, 41, 41)
        assert np.isfinite(grid.data).all()

    def test_values_at_a_grid_point_are_the_granules(self, reader):
        """``ERA5PressureLevelReader`` serves a real download's 500 hPa values unchanged at a grid point.

        A value must reach the product exactly as the source granule stores it, under the name of its own variable and
        level.

        Queries a small box around 45N, 105W of the real subset and compares ``temperature_500hPa`` and
        ``geopotential_500hPa`` with the source granule's values there.
        """
        samples = reader.samples_in(BoundingBox(44.9, 45.1, -105.1, -104.9), self._TIME)
        names = [v.name for v in ERA5PressureLevelReader.VARIABLES]
        values = sample_at(samples, 45.0, -105.0)
        assert values[names.index("temperature_500hPa")] == pytest.approx(240.2764, abs=1e-3)
        assert values[names.index("geopotential_500hPa")] == pytest.approx(53364.29, abs=0.1)
