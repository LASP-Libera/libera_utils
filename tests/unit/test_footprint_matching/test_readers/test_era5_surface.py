"""Unit tests for ERA5SurfaceReader and the ERA5 reader base both ERA5 readers share.

Uses real synthetic NetCDF4 fixture files (created by make_era5_surface_netcdf_fixture and
make_era5_pressure_netcdf_fixture) so that the xarray slicing and coordinate handling
can be exercised without calling real CDS services.

Real ERA5 files can be downloaded from:
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register
"""

import gc
import tracemalloc
import weakref

import numpy as np
import pytest
import xarray as xr

from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.era5_surface import _ERA5_SURFACE_VARIABLES, ERA5SurfaceReader
from libera_utils.footprint_matching.types import BoundingBox
from tests.plugins.footprint_matching_fixtures import (
    ERA5_FIXTURE_TIME,
    ERA5_SURFACE_FILLS,
    make_era5_pressure_netcdf_fixture,
    make_era5_surface_netcdf_fixture,
)
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = ERA5_FIXTURE_TIME

# Both ERA5 readers, each with its fixture factory, the factory's keywords for a file with no
# time axis, and its keywords for a grid large enough to measure a step load's peak memory.
_ERA5_READERS = [
    pytest.param(
        ERA5SurfaceReader,
        make_era5_surface_netcdf_fixture,
        {"time_dim": None},
        {"n_lat": 400, "n_lon": 800},
        id="single-level",
    ),
    pytest.param(
        ERA5PressureLevelReader,
        make_era5_pressure_netcdf_fixture,
        {"with_valid_time": False},
        {"n_lat": 60, "n_lon": 120, "lat_min": -15.0, "lat_max": 15.0, "lon_min": -30.0, "lon_max": 30.0},
        id="pressure",
    ),
]


@pytest.mark.parametrize(("reader_cls", "make_fixture", "no_time", "large_grid"), _ERA5_READERS)
class TestERA5ReaderBase:
    """What the shared ERA5 reader base does for both ERA5 readers."""

    def test_served_grid_is_ascending_and_float64(self, tmp_path, reader_cls, make_fixture, no_time, large_grid):
        """An ERA5 reader serves latitudes ascending, as float64, although ERA5 files store them descending.

        The box query slices the latitude axis with ``searchsorted``, which needs an ascending axis, and
        ``SampleSet`` defines its coordinates as float64, which the PSF weigher places each sample by; a descending
        axis would select the wrong rows, and a narrower dtype would shift the samples it weighs.

        Queries a synthetic file whose latitudes are stored 2 to 0 over its full extent and asserts the served,
        row-major latitudes never decrease and both coordinate arrays are float64.
        """
        samples = reader_cls(make_fixture(tmp_path)).samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert np.all(np.diff(samples.lats) >= 0), f"Lats should be ascending but got: {samples.lats}"
        assert samples.lats.dtype == np.float64
        assert samples.lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path, reader_cls, make_fixture, no_time, large_grid):
        """An ERA5 reader serves only the grid points inside the query box, not the whole step.

        A footprint must be aggregated from the cells around it; serving cells from outside the box would mix in values
        from elsewhere and waste memory on every query.

        Builds an 8 x 4 grid over latitudes 0 to 4 and longitudes 10 to 12, queries a box inside it on every side
        (latitudes 1 to 3, longitudes 10.5 to 11.5, padded by about 0.13 degree), and asserts exactly the four interior
        rows and two interior columns are served.
        """
        fixture_path = make_fixture(tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4)
        data, lats, lons = _region(reader_cls(fixture_path), BoundingBox(1.0, 3.0, 10.5, 11.5), _T)
        np.testing.assert_allclose(lats, np.linspace(0.0, 4.0, 8)[2:6])
        np.testing.assert_allclose(lons, np.linspace(10.0, 12.0, 4)[1:3])
        assert data.shape[1:] == (4, 2)

    def test_file_without_time_axis_is_rejected(self, tmp_path, reader_cls, make_fixture, no_time, large_grid):
        """An ERA5 reader raises ``ValueError`` for a file with no time dimension.

        Without a time axis a step cannot be matched to a footprint's time, so the file is rejected rather than served
        for every time.

        Writes a file whose grids have only latitude and longitude dimensions and asserts a query raises
        ``ValueError`` naming the time dimension.
        """
        reader = reader_cls(make_fixture(tmp_path, **no_time))
        with pytest.raises(ValueError, match="time dimension"):
            reader.samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), _T)

    def test_step_load_peak_memory_is_one_step(self, tmp_path, reader_cls, make_fixture, no_time, large_grid):
        """An ERA5 reader's ``_load_step`` reads a step with peak memory under 1.5 times the step's own size.

        Each plane is read into one preallocated array; building per-variable arrays and stacking them would hold the
        step twice, and the full-resolution global step is large (about 0.8 GB for every pressure layer).

        Loads one step of a larger synthetic grid under ``tracemalloc`` and compares the traced peak with the size of
        the returned data array.
        """
        reader = reader_cls(make_fixture(tmp_path, **large_grid))
        _ = reader._steps
        tracemalloc.start()
        try:
            grid = reader._load_step(0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 1.5 * grid.data.nbytes


class TestERA5ReaderPlanes:
    @pytest.mark.parametrize("time_dim", ["time", "valid_time"])
    def test_planes_hold_their_variables(self, tmp_path, time_dim):
        """``ERA5SurfaceReader`` serves one float32 plane per variable, plane ``i`` holding the field ``VARIABLES[i]`` names.

        CDS downloads name the time dimension ``time`` or ``valid_time``; a reader that missed either would reject the
        file or serve a step with the time axis still attached. Each plane is copied from a named file variable, so a
        mismatch publishes, for example, surface pressure under the 2 m temperature name. ERA5 is a reanalysis whose
        whole step is chosen by time, so its samples carry the reader key, no timestamp source and no per-sample times.

        Parametrized over the two time-dimension names. The synthetic 4 x 4 file gives every variable its own constant
        (``u10`` 3.0, ``v10`` -2.0, the rest from ``ERA5_SURFACE_FILLS``); a full-extent query must fold back to
        ``(7, 4, 4)`` float32 planes, each the constant of the variable at that position, with the samples' source,
        timestamp source and times as above.
        """
        fixture_path = make_era5_surface_netcdf_fixture(tmp_path, u10_fill=3.0, v10_fill=-2.0, time_dim=time_dim)
        reader = ERA5SurfaceReader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = _region(reader, bbox, _T)
        assert data.shape == (7, 4, 4)
        assert data.dtype == np.float32
        assert np.allclose(data[0], 3.0, atol=1e-4)
        assert np.allclose(data[1], -2.0, atol=1e-4)
        for i, (_, nc_name, _) in enumerate(_ERA5_SURFACE_VARIABLES[2:], start=2):
            assert np.allclose(data[i], ERA5_SURFACE_FILLS[nc_name], rtol=1e-6), nc_name
        samples = reader.samples_in(bbox, _T)
        assert samples.source == "era5_surface"
        assert samples.timestamp_source is None
        assert samples.times is None


class TestERA5ReaderMissingVariables:
    """A partial CDS download (missing required variables) must fail loudly."""

    def test_file_missing_variables_raises_key_error(self, tmp_path):
        """``ERA5SurfaceReader`` raises ``KeyError`` when a file lacks any of the seven single-level variables.

        A CDS download configured with only some of the fields would otherwise leave planes unfilled, and the product
        would carry values that were never read.

        Writes a file holding only ``u10`` and ``v10`` and asserts a query raises ``KeyError`` naming the missing
        variables.
        """
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

        reader = ERA5SurfaceReader(path)
        with pytest.raises(KeyError, match="missing variable"):
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)


class TestERA5ReaderDateline:
    def test_dateline_box_on_a_0_to_360_file_serves_both_ends(self, tmp_path):
        """A dateline-crossing box on a 0..360 file serves the columns on both sides of 180 degrees.

        A 0..360 file is moved onto an ascending -180..180 axis, so a box across the antimeridian must be split and read
        from both ends of that axis; otherwise footprints there lose half their cells.

        Writes a one-degree global file whose ``u10`` holds each column's stored longitude, queries a box from 178.6 to
        181.4 degrees, and asserts the three served columns are -180, -179 and 179 with ``u10`` 180, 181 and 179.
        """
        # A 0..360 file is put on an ascending -180..180 axis, so the two halves of a split box
        # query both ends of one axis. u10 carries each column's stored 0..360 longitude.
        constant = make_era5_surface_netcdf_fixture(
            tmp_path, lon_min=0.0, lon_max=359.0, n_lon=360, filename="constant.nc"
        )
        ds = xr.load_dataset(constant)
        ds["u10"] = ds["u10"].copy(data=np.broadcast_to(ds["longitude"].values, ds["u10"].shape).astype(np.float32))
        path = tmp_path / "era5_0_360.nc"
        ds.to_netcdf(path)

        samples = ERA5SurfaceReader(path).samples_in(BoundingBox(0.0, 2.0, 178.6, 181.4, wraps_dateline=True), _T)

        assert samples.n_samples == 4 * 3
        assert sorted(set(zip(samples.lons.tolist(), samples.values[0].tolist(), strict=True))) == [
            (-180.0, 180.0),
            (-179.0, 181.0),
            (179.0, 179.0),
        ]


class TestERA5ReaderTimeSteps:
    """The hourly step nearest the footprint time is served, one step at a time."""

    @staticmethod
    def _three_steps(tmp_path, filename="era5_steps.nc"):
        times = ERA5_FIXTURE_TIME + np.arange(3) * np.timedelta64(1, "h")
        return make_era5_surface_netcdf_fixture(
            tmp_path, times=times, u10_per_step=np.array([10.0, 20.0, 30.0]), filename=filename
        )

    def test_nearest_step_is_served(self, tmp_path):
        """``ERA5SurfaceReader.samples_in`` serves the step nearest the footprint time.

        Serving any other step pairs a footprint with weather up to hours away from its observation.

        Writes three hourly steps whose ``u10`` is 10, 20 and 30, queries at 00:10 and 01:50, and asserts ``u10`` is 10
        and 30.
        """
        reader = ERA5SurfaceReader(self._three_steps(tmp_path))
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        at_0010 = reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(10, "m"))
        at_0150 = reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(110, "m"))
        assert np.allclose(at_0010.values[0], 10.0)
        assert np.allclose(at_0150.values[0], 30.0)

    def test_only_the_nearest_step_is_held(self, tmp_path):
        """``ERA5SurfaceReader`` releases the previous step when a query moves to another step.

        A day of hourly steps does not fit in memory at once, so only the step in use may be held.

        Queries a three-step file at the first and then the last step, holds a weak reference to the first step's data,
        and asserts it is collected and the loaded step index is 2.
        """
        reader = ERA5SurfaceReader(self._three_steps(tmp_path))
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(10, "m"))
        first_step = weakref.ref(reader._loaded_step[1].data)
        reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(110, "m"))
        gc.collect()
        # The first step was released when the second was loaded, not kept beside it.
        assert first_step() is None
        assert reader._loaded_step[0] == 2

    def test_steps_span_several_granules(self, tmp_path):
        """``ERA5SurfaceReader`` builds one time-ordered step index across several staged granules.

        A processing period can be staged as several files in any order; each footprint must still get the step nearest
        it, whichever file holds it.

        Stages a 01:00 granule before a 00:00 granule, queries at each hour and asserts the ``u10`` constant of the
        matching granule.
        """
        first = make_era5_surface_netcdf_fixture(
            tmp_path, times=[ERA5_FIXTURE_TIME], u10_per_step=np.array([1.0]), filename="a.nc"
        )
        second = make_era5_surface_netcdf_fixture(
            tmp_path, times=[ERA5_FIXTURE_TIME + np.timedelta64(1, "h")], u10_per_step=np.array([2.0]), filename="b.nc"
        )
        # Handed out of time order: the step index sorts them.
        reader = ERA5SurfaceReader([second, first])
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        assert np.allclose(reader.samples_in(box, ERA5_FIXTURE_TIME).values[0], 1.0)
        assert np.allclose(reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(1, "h")).values[0], 2.0)


class TestERA5ReaderRealSubset:
    """A real ERA5 single-level download cut to 50N..40N, 110W..100W at 2025-01-01T00.

    The download stores longitude 0..359.75, so serving it runs the normalization to
    -180..180. Pinned values are the source granule's own at 45N, 105W.
    """

    _TIME = np.datetime64("2025-01-01T00:00:00", "ns")

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return ERA5SurfaceReader(fmatch_real_data_path / "era5_single_levels_20250101T00_subset.nc")

    def test_window_is_served_on_normalized_longitudes(self, reader):
        """``ERA5SurfaceReader._load_step`` serves a real 0..360 CDS download on -180..180 longitudes.

        The box query works in -180..180, so an unnormalized download would answer queries with the wrong cells or none.

        Loads the one step of a real 41 x 41 single-level subset stored on 250..260 degrees east and asserts the axes
        span 40 to 50 N and -110 to -100, all seven planes are present and every value is finite.
        """
        grid = reader._load_step(0)
        assert (grid.lat_axis[0], grid.lat_axis[-1]) == (40.0, 50.0)
        assert (grid.lon_axis[0], grid.lon_axis[-1]) == (-110.0, -100.0)
        assert grid.data.shape == (len(ERA5SurfaceReader.VARIABLES), 41, 41)
        assert np.isfinite(grid.data).all()

    def test_values_at_a_grid_point_are_the_granules(self, reader):
        """``ERA5SurfaceReader`` serves a real download's values unchanged at a grid point.

        A value must reach the product exactly as the source granule stores it, under the right variable name.

        Queries a small box around 45N, 105W of the real subset and compares ``temperature_2m``, ``surface_pressure``
        and ``wind_u10`` with the source granule's values there.
        """
        samples = reader.samples_in(BoundingBox(44.9, 45.1, -105.1, -104.9), self._TIME)
        values = dict(zip([v.name for v in ERA5SurfaceReader.VARIABLES], sample_at(samples, 45.0, -105.0), strict=True))
        assert values["temperature_2m"] == pytest.approx(268.0293, abs=1e-3)
        assert values["surface_pressure"] == pytest.approx(88917.95, abs=0.1)
        assert values["wind_u10"] == pytest.approx(1.84288, abs=1e-4)
