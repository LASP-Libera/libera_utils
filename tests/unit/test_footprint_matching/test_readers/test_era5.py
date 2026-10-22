"""Unit tests for ERA5Reader.

Uses real synthetic NetCDF4 fixture files (created by make_era5_netcdf_fixture)
so that the xarray slicing and coordinate handling can be exercised without
calling real CDS services.

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
        """``ERA5Reader`` registers under the key ``"era5_single_level"``.

        The key selects the reader in ``ReaderRegistry`` and prefixes every product variable name it contributes
        (``era5_single_level_wind_u10``), so changing it renames those variables and breaks every lookup by key.

        Asserts the ``READER_KEY`` class attribute equals ``"era5_single_level"``.
        """
        assert ERA5Reader.READER_KEY == "era5_single_level"

    def test_resolution_km(self):
        """``ERA5Reader.RESOLUTION_KM`` is 28 km, the 0.25 degree ERA5 grid spacing at the equator.

        The query box is padded by half this cell size and the weigher sizes its coverage measure from it, so a wrong
        value serves the wrong rim cells and misreports how much of a footprint the grid covers.

        Asserts the class attribute equals 28.0.
        """
        assert ERA5Reader.RESOLUTION_KM == 28.0

    def test_variables_has_seven_entries(self):
        """``ERA5Reader.VARIABLES`` declares exactly seven single-level fields.

        The product definition derives the reader's variables from this tuple and each served step has one data plane
        per entry, so an entry added or dropped changes the product schema and the plane count together.

        Asserts the tuple has length 7.
        """
        assert len(ERA5Reader.VARIABLES) == 7

    def test_variable_names(self):
        """``ERA5Reader.VARIABLES`` names the two 10 m winds first, then the five IMAGER single-level fields.

        The spec order is the order of the data planes the reader serves; if the names and the planes disagree, a value
        is published under another variable's name.

        Compares the spec names, in order, with ``wind_u10``, ``wind_v10`` and ``_IMAGER_SINGLE_LEVEL_SPEC_NAMES``.
        """
        names = [v.name for v in ERA5Reader.VARIABLES]
        assert names == ["wind_u10", "wind_v10", *_IMAGER_SINGLE_LEVEL_SPEC_NAMES]

    def test_wind_variables_are_unrestricted(self):
        """The two wind specs are gated at ``OperationalMode.CAM``, so every FMATCH product carries them.

        A spec is carried when its ``required_mode`` rank is at or below the active mode's, and CAM is the
        lowest-latency mode; a higher gate would drop the winds from the CAM products that run from mission start.

        Asserts ``required_mode`` is CAM for the first two entries of ``VARIABLES``.
        """
        for v in ERA5Reader.VARIABLES[:2]:
            assert v.required_mode == OperationalMode.CAM

    def test_imager_single_level_variables_are_mode_gated(self):
        """The five non-wind single-level specs are gated at ``OperationalMode.IMAGER``.

        These fields belong to the FMATCH-IMAGER products and must stay out of the lower-latency CAM products; a CAM
        gate would add them to products they are not defined for.

        Asserts ``required_mode`` is IMAGER for every entry of ``VARIABLES`` after the two winds.
        """
        # The additional single-level fields exist for the FMATCH-IMAGER product; asserts each of
        # VARIABLES[2:] has required_mode == IMAGER (mode-gated to IMAGER).
        for v in ERA5Reader.VARIABLES[2:]:
            assert v.required_mode == OperationalMode.IMAGER

    def test_variables_have_no_categories(self):
        """Every ``ERA5Reader`` variable is continuous, with ``n_categories`` left as ``None``.

        ERA5 fields are physical quantities aggregated as a weighted mean; a category count would declare one of them a
        categorical variable.

        Asserts ``n_categories`` is ``None`` on every entry of ``VARIABLES``.
        """
        for v in ERA5Reader.VARIABLES:
            assert v.n_categories is None


class TestERA5ReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        """``ERA5Reader.samples_in`` serves a grid that folds back to ``(7, n_lat, n_lon)``.

        Axis 0 must hold one plane per variable with the time axis removed; an extra or missing axis means the samples
        no longer line up with ``VARIABLES``.

        Queries a 4 x 4 synthetic file over its full extent, folds the samples back into a grid with ``region`` and
        asserts three dimensions with 7 planes on axis 0.
        """
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
        """Plane ``i`` of the served data holds the ERA5 variable that ``VARIABLES[i]`` names.

        Each plane is copied from a named file variable; a mismatch publishes, for example, surface pressure under the 2
        m temperature name.

        The synthetic file gives every variable its own constant (``u10`` 2.5, ``v10`` -1.5, the rest from
        ``ERA5_SINGLE_LEVEL_FILLS``), and each plane is compared with the constant of the variable at that position.
        """
        fixture_path = make_era5_netcdf_fixture(tmp_path, u10_fill=2.5, v10_fill=-1.5, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = _region(reader, bbox, _T)
        # data[0] should be u10 (all 2.5) and data[1] should be v10 (all -1.5);
        # the remaining layers follow the file-variable mapping order with the
        # distinct constant fills the fixture writes for each variable.
        assert np.allclose(data[0], 2.5, atol=1e-4)
        assert np.allclose(data[1], -1.5, atol=1e-4)
        for i, (_, nc_name, _) in enumerate(_ERA5_SINGLE_LEVEL_VARIABLES[2:], start=2):
            assert np.allclose(data[i], ERA5_SINGLE_LEVEL_FILLS[nc_name], rtol=1e-6), nc_name

    def test_lats_are_ascending_order(self, tmp_path):
        """``ERA5Reader`` serves latitudes ascending although ERA5 files store them descending.

        The box query slices the latitude axis with ``searchsorted``, which needs an ascending axis; a descending axis
        would select the wrong rows or none.

        Queries a synthetic file whose latitudes are stored 2 to 0 and asserts the served latitudes never decrease.
        """
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
        """``ERA5Reader.samples_in`` serves its values as float32.

        ``VARIABLES`` declares every field float32, the dtype the product stores; a wider dtype would double the memory
        of every served step.

        Queries a 4 x 4 synthetic file and asserts the folded data array is float32.
        """
        fixture_path = make_era5_netcdf_fixture(tmp_path, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, _, _ = _region(reader, bbox, _T)
        assert data.dtype == np.float32

    def test_lats_dtype_is_float64(self, tmp_path):
        """``ERA5Reader.samples_in`` serves latitudes and longitudes as float64.

        ``SampleSet`` defines its coordinates as float64, and the PSF weigher places each sample by them; a narrower
        dtype would shift the samples it weighs.

        Queries a 4 x 4 synthetic file and asserts both coordinate arrays are float64.
        """
        fixture_path = make_era5_netcdf_fixture(tmp_path, n_lat=4, n_lon=4)
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        _, lats, lons = _region(reader, bbox, _T)
        assert lats.dtype == np.float64
        assert lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path):
        """``ERA5Reader.samples_in`` serves only the grid points inside the query box, not the whole step.

        A footprint must be aggregated from the cells around it; serving cells from outside the box would mix in values
        from elsewhere and waste memory on every query.

        Builds an 8 x 4 grid over latitudes 0 to 4 and longitudes 10 to 12, queries a box inside it on every side
        (latitudes 1 to 3, longitudes 10.5 to 11.5, padded by about 0.13 degree), and asserts exactly the four interior
        rows and two interior columns are served.
        """
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        data, lats, lons = _region(reader, BoundingBox(1.0, 3.0, 10.5, 11.5), _T)
        np.testing.assert_allclose(lats, np.linspace(0.0, 4.0, 8)[2:6])
        np.testing.assert_allclose(lons, np.linspace(10.0, 12.0, 4)[1:3])
        assert data.shape[1:] == (4, 2)

    def test_step_load_peak_memory_is_one_step(self, tmp_path):
        """``ERA5Reader._load_step`` reads a step with peak memory under 1.5 times the step's own size.

        The variables are read plane by plane into one preallocated array; building per-variable arrays and stacking
        them would hold the step twice, and the full-resolution global step is large.

        Loads one step of a 400 x 800 synthetic grid under ``tracemalloc`` and compares the traced peak with the size of
        the returned data array.
        """
        reader = ERA5Reader(make_era5_netcdf_fixture(tmp_path, n_lat=400, n_lon=800))
        _ = reader._steps
        tracemalloc.start()
        try:
            grid = reader._load_step(0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The variables are read plane by plane into one array; stacking per-variable arrays
        # would hold at least twice the step.
        assert peak < 1.5 * grid.data.nbytes


class TestERA5ReaderValidTimeDimension:
    """ERA5 files downloaded via the new CDS API use 'valid_time' as the time dimension name.

    The reader must find ANY dimension whose name contains the substring 'time',
    not just the exact name 'time'.
    """

    def test_valid_time_dimension_is_the_step_axis(self, tmp_path):
        """``ERA5Reader`` treats a ``valid_time`` dimension as its hourly step axis.

        CDS downloads name the time dimension ``valid_time`` or ``time``; a reader that missed ``valid_time`` would
        reject those files or serve a step with the time axis still attached.

        Queries a synthetic file whose time dimension is ``valid_time`` and asserts the served grid is three-dimensional
        with 7 planes.
        """
        from tests.test_data.footprint_matching.fixtures import make_era5_valid_time_fixture

        fixture_path = make_era5_valid_time_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=4, n_lon=4
        )
        reader = ERA5Reader(fixture_path)
        bbox = BoundingBox(0.0, 2.0, 10.0, 12.0)
        data, lats, lons = _region(reader, bbox, _T)
        # A time axis left on the step would make the shape (7, 1, n_lat, n_lon).
        assert data.ndim == 3
        assert data.shape[0] == 7

    def test_valid_time_values_correct(self, tmp_path):
        """``ERA5Reader`` serves the stored wind values from a file with a ``valid_time`` dimension.

        ERA5 files from the CDS may name their time axis ``valid_time`` instead of ``time``; a reader that found that
        axis but read the wrong slice of it would publish winds that were never in the file.

        Writes a ``valid_time`` file with ``u10`` 3.0 and ``v10`` -2.0 and asserts the first two served planes hold
        those constants.
        """
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
        """``ERA5Reader`` raises ``KeyError`` when a file lacks any of the seven single-level variables.

        A CDS download configured with only some of the fields would otherwise leave planes unfilled, and the product
        would carry values that were never read.

        Writes a file holding only ``u10`` and ``v10`` and asserts a query raises ``KeyError`` whose message says a
        variable is missing.
        """
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
        """``ERA5Reader`` samples carry the source ``"era5_single_level"``, no timestamp source and no per-sample times.

        ERA5 is a reanalysis: the whole step is chosen by time, so its samples need no times of their own, and
        ``timestamp_source`` is reserved for cloud-property products tied to a radiometer or camera time.

        Queries a 4 x 4 synthetic file over its full extent and asserts ``source``, ``timestamp_source``, ``times`` and
        a count of 16 samples.
        """
        # ERA5 is a reanalysis; its samples carry the reader key and no instrument timestamp source.
        fixture_path = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0, n_lat=4, n_lon=4
        )
        samples = ERA5Reader(fixture_path).samples_in(BoundingBox(0.0, 2.0, 0.0, 2.0), _T)
        assert samples.source == "era5_single_level"
        assert samples.timestamp_source is None
        assert samples.times is None
        assert samples.n_samples == 16

    def test_dateline_box_on_a_0_to_360_file_serves_both_ends(self, tmp_path):
        """A dateline-crossing box on a 0..360 file serves the columns on both sides of 180 degrees.

        A 0..360 file is moved onto an ascending -180..180 axis, so a box across the antimeridian must be split and read
        from both ends of that axis; otherwise footprints there lose half their cells.

        Writes a one-degree global file whose ``u10`` holds each column's stored longitude, queries a box from 178.6 to
        181.4 degrees, and asserts the three served columns are -180, -179 and 179 with ``u10`` 180, 181 and 179.
        """
        # A 0..360 file is put on an ascending -180..180 axis, so the two halves of a split box
        # query both ends of one axis. u10 carries each column's stored 0..360 longitude.
        constant = make_era5_netcdf_fixture(tmp_path, lon_min=0.0, lon_max=359.0, n_lon=360, filename="constant.nc")
        ds = xr.load_dataset(constant)
        ds["u10"] = ds["u10"].copy(data=np.broadcast_to(ds["longitude"].values, ds["u10"].shape).astype(np.float32))
        path = tmp_path / "era5_0_360.nc"
        ds.to_netcdf(path)

        samples = ERA5Reader(path).samples_in(BoundingBox(0.0, 2.0, 178.6, 181.4, wraps_dateline=True), _T)

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
        return make_era5_netcdf_fixture(
            tmp_path, times=times, u10_per_step=np.array([10.0, 20.0, 30.0]), filename=filename
        )

    def test_policy_is_nearest_step(self):
        """``ERA5Reader`` matches time by ``NEAREST_STEP`` with a one-hour ``MAX_TIME_OFFSET``.

        The reader serves the hourly step nearest a footprint's time without interpolation, and a footprint more than an
        hour from every staged step gets no ERA5 samples and is flagged.

        Asserts the ``TIME_MATCH`` and ``MAX_TIME_OFFSET`` class attributes.
        """
        assert ERA5Reader.TIME_MATCH is TimeMatch.NEAREST_STEP
        assert ERA5Reader.MAX_TIME_OFFSET == np.timedelta64(1, "h")

    def test_nearest_step_is_served(self, tmp_path):
        """``ERA5Reader.samples_in`` serves the step nearest the footprint time.

        Serving any other step pairs a footprint with weather up to hours away from its observation.

        Writes three hourly steps whose ``u10`` is 10, 20 and 30, queries at 00:10 and 01:50, and asserts ``u10`` is 10
        and 30.
        """
        reader = ERA5Reader(self._three_steps(tmp_path))
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        at_0010 = reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(10, "m"))
        at_0150 = reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(110, "m"))
        assert np.allclose(at_0010.values[0], 10.0)
        assert np.allclose(at_0150.values[0], 30.0)

    def test_only_the_nearest_step_is_held(self, tmp_path):
        """``ERA5Reader`` releases the previous step when a query moves to another step.

        A day of hourly steps does not fit in memory at once, so only the step in use may be held.

        Queries a three-step file at the first and then the last step, holds a weak reference to the first step's data,
        and asserts it is collected and the loaded step index is 2.
        """
        reader = ERA5Reader(self._three_steps(tmp_path))
        box = BoundingBox(0.0, 2.0, 10.0, 12.0)
        reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(10, "m"))
        first_step = weakref.ref(reader._loaded_step[1].data)
        reader.samples_in(box, ERA5_FIXTURE_TIME + np.timedelta64(110, "m"))
        gc.collect()
        # The first step was released when the second was loaded, not kept beside it.
        assert first_step() is None
        assert reader._loaded_step[0] == 2

    def test_step_beyond_max_offset_gives_no_samples(self, tmp_path):
        """``ERA5Reader.samples_in`` serves no samples when the nearest step is beyond ``MAX_TIME_OFFSET``.

        A footprint far from every staged step must be scored as uncovered and flagged, not filled from a step hours
        away.

        Queries a file with steps at 00:00 to 02:00 at 09:00 and asserts zero samples.
        """
        reader = ERA5Reader(self._three_steps(tmp_path))
        samples = reader.samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), ERA5_FIXTURE_TIME + np.timedelta64(9, "h"))
        assert samples.n_samples == 0

    def test_steps_span_several_granules(self, tmp_path):
        """``ERA5Reader`` builds one time-ordered step index across several staged granules.

        A processing period can be staged as several files in any order; each footprint must still get the step nearest
        it, whichever file holds it.

        Stages a 01:00 granule before a 00:00 granule, queries at each hour and asserts the ``u10`` constant of the
        matching granule.
        """
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
        """``ERA5Reader`` raises ``ValueError`` for a file with no time dimension.

        Without a time axis a step cannot be matched to a footprint's time, so the file is rejected rather than served
        for every time.

        Writes a ``u10`` grid with only latitude and longitude dimensions and asserts a query raises ``ValueError``
        naming the time dimension.
        """
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
        """``ERA5Reader._load_step`` serves a real 0..360 CDS download on -180..180 longitudes.

        The box query works in -180..180, so an unnormalized download would answer queries with the wrong cells or none.

        Loads the one step of a real 41 x 41 single-level subset stored on 250..260 degrees east and asserts the axes
        span 40 to 50 N and -110 to -100, all seven planes are present and every value is finite.
        """
        grid = reader._load_step(0)
        assert (grid.lat_axis[0], grid.lat_axis[-1]) == (40.0, 50.0)
        assert (grid.lon_axis[0], grid.lon_axis[-1]) == (-110.0, -100.0)
        assert grid.data.shape == (len(ERA5Reader.VARIABLES), 41, 41)
        assert np.isfinite(grid.data).all()

    def test_values_at_a_grid_point_are_the_granules(self, reader):
        """``ERA5Reader`` serves a real download's values unchanged at a grid point.

        A value must reach the product exactly as the source granule stores it, under the right variable name.

        Queries a small box around 45N, 105W of the real subset and compares ``temperature_2m``, ``surface_pressure``
        and ``wind_u10`` with the source granule's values there.
        """
        samples = reader.samples_in(BoundingBox(44.9, 45.1, -105.1, -104.9), self._TIME)
        values = dict(zip([v.name for v in ERA5Reader.VARIABLES], sample_at(samples, 45.0, -105.0), strict=True))
        assert values["temperature_2m"] == pytest.approx(268.0293, abs=1e-3)
        assert values["surface_pressure"] == pytest.approx(88917.95, abs=0.1)
        assert values["wind_u10"] == pytest.approx(1.84288, abs=1e-4)

    def test_a_time_beyond_the_offset_serves_nothing(self, reader):
        """A real single-step download serves no samples two hours after its step.

        The one-hour ``MAX_TIME_OFFSET`` must hold on a real file, so a footprint outside the staged hours is flagged as
        uncovered.

        Queries the real subset two hours after its only step and asserts zero samples.
        """
        samples = reader.samples_in(BoundingBox(44.0, 46.0, -106.0, -104.0), self._TIME + np.timedelta64(2, "h"))
        assert samples.n_samples == 0
