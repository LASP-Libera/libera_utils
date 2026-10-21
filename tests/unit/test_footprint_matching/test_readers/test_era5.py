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
import pytest

from libera_utils.footprint_matching.readers.era5 import _ERA5_SINGLE_LEVEL_VARIABLES, ERA5Reader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, TimeMatch
from tests.test_data.footprint_matching.fixtures import (
    ERA5_FIXTURE_TIME,
    ERA5_SINGLE_LEVEL_FILLS,
    make_era5_netcdf_fixture,
)
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = ERA5_FIXTURE_TIME

# The five additional FMATCH-IMAGER single-level fields alongside the winds.
# (They sit in the FMATCH-IMAGER product alongside the RBSP fields; they are
# mode-gated to IMAGER.)
_IMAGER_SINGLE_LEVEL_SPEC_NAMES = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "surface_pressure",
    "surface_geopotential",
    "forecast_albedo",
)


class TestERA5ReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "era5".
        assert ERA5Reader.READER_KEY == "era5"

    def test_resolution_km(self):
        # Targets the reader's grid resolution attribute; asserts RESOLUTION_KM equals the 0.25 degree spacing, 28 km.
        assert ERA5Reader.RESOLUTION_KM == 28.0

    def test_variables_has_seven_entries(self):
        # Targets the VARIABLES tuple size; asserts the reader exposes exactly 7 single-level fields.
        assert len(ERA5Reader.VARIABLES) == 7

    def test_variable_names(self):
        # Targets the VARIABLES field names/order; asserts they are the two winds then the five IMAGER fields.
        names = [v.name for v in ERA5Reader.VARIABLES]
        assert names == ["wind_u10", "wind_v10", *_IMAGER_SINGLE_LEVEL_SPEC_NAMES]

    def test_variables_order_matches_file_mapping(self):
        # Axis 0 of the data array follows _ERA5_SINGLE_LEVEL_VARIABLES, which
        # must therefore agree with the VARIABLES tuple order exactly.
        assert [v.name for v in ERA5Reader.VARIABLES] == [name for name, _ in _ERA5_SINGLE_LEVEL_VARIABLES]

    def test_wind_variables_are_unrestricted(self):
        # Targets the two wind variables' mode gating; asserts the first two VARIABLES require CAM mode.
        for v in ERA5Reader.VARIABLES[:2]:
            assert v.required_mode == OperationalMode.CAM

    def test_imager_single_level_variables_are_mode_gated(self):
        # The additional single-level fields exist for the FMATCH-IMAGER product; asserts each of
        # VARIABLES[2:] has required_mode == IMAGER (mode-gated to IMAGER).
        for v in ERA5Reader.VARIABLES[2:]:
            assert v.required_mode == OperationalMode.IMAGER

    def test_variables_have_no_categories(self):
        # Targets that every field is continuous, not categorical; asserts each VARIABLES entry has n_categories None.
        for v in ERA5Reader.VARIABLES:
            assert v.n_categories is None


class TestERA5ReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        # Targets samples_in output shape; asserts a 3D array whose axis 0 holds the 7 variable layers.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, lats, lons = _region(reader, bbox, _T)
        # 3D: (7 variables, n_lat, n_lon)
        assert data.ndim == 3
        assert data.shape[0] == 7

    def test_data_axis0_is_variables(self, tmp_path):
        fixture_path = make_era5_netcdf_fixture(tmp_path, u10_fill=2.5, v10_fill=-1.5, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = _region(reader, bbox, _T)
        # data[0] should be u10 (all 2.5) and data[1] should be v10 (all -1.5);
        # the remaining layers follow the file-variable mapping order with the
        # distinct constant fills the fixture writes for each variable.
        assert np.allclose(data[0], 2.5, atol=1e-4)
        assert np.allclose(data[1], -1.5, atol=1e-4)
        for i, (_, nc_name) in enumerate(_ERA5_SINGLE_LEVEL_VARIABLES[2:], start=2):
            assert np.allclose(data[i], ERA5_SINGLE_LEVEL_FILLS[nc_name], rtol=1e-6), nc_name

    def test_lats_are_ascending_order(self, tmp_path):
        # The fixture stores lats in DESCENDING order (ERA5 convention); asserts the reader flips them
        # so that np.diff(lats) >= 0 (ascending) on output.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        _, lats, _ = _region(reader, bbox, _T)
        assert np.all(np.diff(lats) >= 0), f"Lats should be ascending but got: {lats}"

    def test_data_dtype_is_float32(self, tmp_path):
        # Targets the returned data array dtype; asserts samples_in yields float32 values.
        fixture_path = make_era5_netcdf_fixture(tmp_path, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = _region(reader, bbox, _T)
        assert data.dtype == np.float32

    def test_lats_dtype_is_float64(self, tmp_path):
        # Targets the returned coordinate dtypes; asserts both lats and lons arrays are float64.
        fixture_path = make_era5_netcdf_fixture(tmp_path, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        _, lats, lons = _region(reader, bbox, _T)
        assert lats.dtype == np.float64
        assert lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path):
        # Targets bbox spatial subsetting; asserts a sub-range request returns only lats within that range.
        # Fixture covers lat 0–2, lon 10–12 with 8 points in each axis.
        # Request only the lower half of the lat range.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        # Only request the upper half of the lat range
        bbox = BoundingBox(2.0, 4.0, 10.0, 12.0)
        data, lats, _ = _region(reader, bbox, _T)
        # All returned lats should be >= 2.0
        assert np.all(lats >= 2.0 - 1e-6)


class TestERA5ReaderValidTimeDimension:
    """ERA5 files downloaded via the new CDS API use 'valid_time' as the time dimension name.

    The reader must find ANY dimension whose name contains the substring 'time',
    not just the exact name 'time'.
    """

    def test_valid_time_dimension_is_the_step_axis(self, tmp_path):
        # Targets a 'valid_time' step axis; asserts one step serves a (7, n_lat, n_lon) grid.
        from tests.test_data.footprint_matching.fixtures import make_era5_valid_time_fixture

        fixture_path = make_era5_valid_time_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, lats, lons = _region(reader, bbox, _T)
        # Without the fix the shape would be (7, 1, n_lat, n_lon); the time dim must be gone.
        assert data.ndim == 3
        assert data.shape[0] == 7

    def test_valid_time_values_correct(self, tmp_path):
        # Targets value integrity after collapsing valid_time; asserts u10/v10 layers equal the fixture fills.
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
        data, _, _ = _region(reader, bbox, _T)
        assert np.allclose(data[0], 3.0, atol=1e-4)
        assert np.allclose(data[1], -2.0, atol=1e-4)


class TestERA5ReaderMissingVariables:
    """A partial CDS download (missing required variables) must fail loudly."""

    def test_file_missing_variables_raises_key_error(self, tmp_path):
        # Targets loud failure on incomplete files; asserts samples_in raises KeyError("missing variable").
        import pytest
        import xarray as xr

        # A legacy wind-only download: valid ERA5 format but missing the five
        # additional single-level variables (t2m, d2m, sp, z, fal).
        lats = np.linspace(2.0, 0.0, 4)
        lons = np.linspace(10.0, 12.0, 4)
        ds = xr.Dataset(
            {
                "u10": xr.DataArray(np.zeros((1, 4, 4), dtype=np.float32), dims=["time", "latitude", "longitude"]),
                "v10": xr.DataArray(np.zeros((1, 4, 4), dtype=np.float32), dims=["time", "latitude", "longitude"]),
            },
            coords={
                "time": xr.DataArray(np.array([ERA5_FIXTURE_TIME]), dims=["time"]),
                "latitude": xr.DataArray(lats, dims=["latitude"]),
                "longitude": xr.DataArray(lons, dims=["longitude"]),
            },
        )
        path = tmp_path / "era5_winds_only.nc"
        ds.to_netcdf(path)

        reader = ERA5Reader(path)
        with pytest.raises(KeyError, match="missing variable"):
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)


class TestERA5ReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        # ERA5 is a reanalysis; its samples carry the reader key and no instrument timestamp source.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0, n_lat=4, n_lon=4
        )
        samples = ERA5Reader(fixture_path).samples_in(BoundingBox(0.0, 2.0, 0.0, 2.0), _T)
        assert samples.source == "era5"
        assert samples.timestamp_source is None
        assert samples.times is None
        assert samples.n_samples == 16


class TestERA5ReaderTimeSteps:
    """The hourly step nearest the footprint time is served, one step at a time."""

    @staticmethod
    def _three_steps(tmp_path, filename="era5_steps.nc"):
        times = ERA5_FIXTURE_TIME + np.arange(3) * np.timedelta64(1, "h")
        return make_era5_netcdf_fixture(
            tmp_path, times=times, u10_per_step=np.array([10.0, 20.0, 30.0]), filename=filename
        )

    def test_policy_is_nearest_step(self):
        assert ERA5Reader.TIME_MATCH is TimeMatch.NEAREST_STEP
        assert ERA5Reader.MAX_TIME_OFFSET == np.timedelta64(1, "h")

    def test_nearest_step_is_served(self, tmp_path):
        reader = ERA5Reader(self._three_steps(tmp_path))
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        at_0010 = reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(10, "m"))
        at_0150 = reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(110, "m"))
        assert np.allclose(at_0010.values[0], 10.0)
        assert np.allclose(at_0150.values[0], 30.0)

    def test_step_beyond_max_offset_gives_no_samples(self, tmp_path):
        reader = ERA5Reader(self._three_steps(tmp_path))
        samples = reader.samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), ERA5_FIXTURE_TIME + np.timedelta64(9, "h"))
        assert samples.n_samples == 0

    def test_steps_span_several_granules(self, tmp_path):
        first = make_era5_netcdf_fixture(
            tmp_path, times=[ERA5_FIXTURE_TIME], u10_per_step=np.array([1.0]), filename="a.nc"
        )
        second = make_era5_netcdf_fixture(
            tmp_path, times=[ERA5_FIXTURE_TIME + np.timedelta64(1, "h")], u10_per_step=np.array([2.0]), filename="b.nc"
        )
        # Handed out of time order: the step index sorts them.
        reader = ERA5Reader([second, first])
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        assert np.allclose(reader.samples_in(box, ERA5_FIXTURE_TIME).values[0], 1.0)
        assert np.allclose(reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(1, "h")).values[0], 2.0)

    def test_file_without_time_axis_is_rejected(self, tmp_path):
        import pytest
        import xarray as xr

        lats = np.linspace(2.0, 0.0, 4)
        lons = np.linspace(10.0, 12.0, 4)
        ds = xr.Dataset(
            {
                name: xr.DataArray(np.zeros((4, 4), dtype=np.float32), dims=["latitude", "longitude"])
                for name in ("u10",)
            },
            coords={
                "latitude": xr.DataArray(lats, dims=["latitude"]),
                "longitude": xr.DataArray(lons, dims=["longitude"]),
            },
        )
        path = tmp_path / "era5_no_time.nc"
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="time dimension"):
            ERA5Reader(path).samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), ERA5_FIXTURE_TIME)


class TestERA5ReaderRealSubset:
    """A real ERA5 single-level download cut to 50N..40N, 110W..100W at 2025-01-01T00.

    The download stores longitude 0..359.75, so serving it runs the normalization to
    -180..180. Pinned values are the source granule's own at 45N, 105W.
    """

    _TIME = np.datetime64("2025-01-01T00:00:00", "ns")

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return ERA5Reader(fmatch_real_data_path / "era5_single_levels_20250101T00_subset.nc")

    def test_window_is_served_on_normalized_longitudes(self, reader):
        grid = reader._load_step(0)
        assert (grid.lat_axis[0], grid.lat_axis[-1]) == (40.0, 50.0)
        assert (grid.lon_axis[0], grid.lon_axis[-1]) == (-110.0, -100.0)
        assert grid.data.shape == (len(ERA5Reader.VARIABLES), 41, 41)
        assert np.isfinite(grid.data).all()

    def test_values_at_a_grid_point_are_the_granules(self, reader):
        samples = reader.samples_in(BoundingBox(44.9, 45.1, -105.1, -104.9), self._TIME)
        values = dict(zip([v.name for v in ERA5Reader.VARIABLES], sample_at(samples, 45.0, -105.0), strict=True))
        assert values["temperature_2m"] == pytest.approx(268.0293, abs=1e-3)
        assert values["surface_pressure"] == pytest.approx(88917.95, abs=0.1)
        assert values["wind_u10"] == pytest.approx(1.84288, abs=1e-4)

    def test_a_time_beyond_the_offset_serves_nothing(self, reader):
        samples = reader.samples_in(BoundingBox(44.0, 46.0, -106.0, -104.0), self._TIME + np.timedelta64(2, "h"))
        assert samples.n_samples == 0
