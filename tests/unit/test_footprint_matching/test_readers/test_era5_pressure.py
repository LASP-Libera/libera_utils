"""Unit tests for ERA5PressureLevelReader.

Uses real synthetic NetCDF4 fixture files (created by
make_era5_pressure_netcdf_fixture) so the xarray slicing, pressure-level
selection, and coordinate handling can be exercised without calling real CDS
services.

Real ERA5 pressure-level files can be downloaded from:
    Copernicus CDS: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
    CDS login required: https://cds.climate.copernicus.eu/user/register
"""

import tracemalloc

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.era5_pressure import (
    _ERA5_PRESSURE_LEVEL_VARIABLES,
    _ERA5_PRESSURE_LEVELS,
    ERA5PressureLevelReader,
)
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode
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


class TestERA5PressureLevelReaderClassAttributes:
    def test_reader_key(self):
        """``ERA5PressureLevelReader`` registers under the key ``"era5_pressure"``.

        The key selects the reader in ``ReaderRegistry`` and prefixes every product variable name it contributes
        (``era5_pressure_temperature_500hPa``), so changing it renames those variables and breaks every lookup by key.

        Asserts the ``READER_KEY`` class attribute equals ``"era5_pressure"``.
        """
        assert ERA5PressureLevelReader.READER_KEY == "era5_pressure"

    def test_resolution_km(self):
        """``ERA5PressureLevelReader.RESOLUTION_KM`` is 28 km, the 0.25 degree ERA5 grid spacing at the equator.

        The query box is padded by half this cell size and the weigher sizes its coverage measure from it, so a wrong
        value serves the wrong rim cells and misreports how much of a footprint the grid covers.

        Asserts the class attribute equals 28.0.
        """
        assert ERA5PressureLevelReader.RESOLUTION_KM == 28.0

    def test_variable_count_is_variables_times_levels(self):
        """``ERA5PressureLevelReader.VARIABLES`` has one spec per (variable, pressure level) pair.

        FMATCH products are one-dimensional per footprint, so each level of each field is its own variable; a missing
        pair drops a level from the product.

        Asserts the tuple length equals the five base variables times the configured levels.
        """
        assert len(ERA5PressureLevelReader.VARIABLES) == _N_SPECS

    def test_variable_names_are_variable_major_level_ascending(self):
        """Pressure-level spec names run variable by variable, levels ascending within each.

        The spec order is the order of the data planes the reader writes; if names and planes disagree, a value is
        published under another level's or another field's name.

        Compares the names with ``<variable>_<level>hPa`` built from ``_ERA5_PRESSURE_LEVEL_VARIABLES`` and
        ``_ERA5_PRESSURE_LEVELS`` in that order.
        """
        expected = [
            f"{base}_{level}hPa" for base, _ in _ERA5_PRESSURE_LEVEL_VARIABLES for level in _ERA5_PRESSURE_LEVELS
        ]
        assert [v.name for v in ERA5PressureLevelReader.VARIABLES] == expected

    def test_all_variables_continuous_weighted_mean(self):
        """Every pressure-level spec is a float32, weighted-mean, IMAGER-gated continuous field.

        The pressure-level fields are physical quantities that belong only to the FMATCH-IMAGER products; a different
        aggregation, dtype or gate would change how they are averaged, stored or which products carry them.

        Asserts ``dtype``, ``aggregation``, ``n_categories`` and ``required_mode`` on every entry of ``VARIABLES``.
        """
        for v in ERA5PressureLevelReader.VARIABLES:
            assert v.dtype == "float32"
            assert v.aggregation == "weighted_mean"
            assert v.n_categories is None
            assert v.required_mode == OperationalMode.IMAGER

    def test_every_spec_gets_standard_deviation_companion(self):
        """``product_variable_specs`` adds a ``_standard_deviation`` companion for every pressure-level spec.

        Each continuous field also reports its spread within the footprint, so the product carries twice as many
        pressure-level variables as the reader reads.

        Asserts the product spec count is twice ``VARIABLES`` and that ``temperature_500hPa`` and its companion are both
        present.
        """
        # All specs are continuous, so the product variable set doubles; asserts product_variable_specs has
        # 2×_N_SPECS entries including each field and its "_standard_deviation" companion.
        product_specs = ERA5PressureLevelReader.product_variable_specs()
        assert len(product_specs) == 2 * _N_SPECS
        names = {s.name for s in product_specs}
        assert "temperature_500hPa" in names
        assert "temperature_500hPa_standard_deviation" in names


class TestERA5PressureLevelReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        """``ERA5PressureLevelReader.samples_in`` serves a grid that folds back to ``(n_specs, n_lat, n_lon)``.

        The pressure-level dimension is flattened into axis 0 with the time axis removed; any other shape means the
        samples no longer line up with ``VARIABLES``.

        Queries a 4 x 4 synthetic file over its full extent, folds the samples back with ``region`` and asserts three
        dimensions with one plane per spec.
        """
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, lats, lons = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert data.ndim == 3
        assert data.shape[0] == _N_SPECS

    def test_layer_order_matches_variables(self, tmp_path):
        """Plane ``i`` of the served data holds the variable and pressure level that ``VARIABLES[i]`` names.

        A plane read from the wrong level or field publishes, for example, 850 hPa temperature as 500 hPa temperature.

        The synthetic file stores levels descending and fills each (variable, level) layer with its own constant from
        ``era5_pressure_fixture_value``; each served plane is compared with the constant for its spec, so levels must be
        selected by value, not by position.
        """
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
        """``ERA5PressureLevelReader.samples_in`` serves its values as float32.

        ``VARIABLES`` declares every field float32, and a step of every pressure layer is already large; a wider dtype
        would double it.

        Queries a synthetic file and asserts the folded data array is float32.
        """
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        data, _, _ = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert data.dtype == np.float32

    def test_lats_are_ascending_order(self, tmp_path):
        """``ERA5PressureLevelReader`` serves latitudes ascending although ERA5 files store them descending.

        The box query slices the latitude axis with ``searchsorted``, which needs an ascending axis; a descending axis
        would select the wrong rows or none.

        Queries a synthetic file whose latitudes are stored descending and asserts the served latitudes never decrease.
        """
        # The fixture stores lats in DESCENDING order (ERA5 convention); asserts the reader flips them
        # so that np.diff(lats) >= 0 (ascending) on output.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        _, lats, _ = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert np.all(np.diff(lats) >= 0), f"Lats should be ascending but got: {lats}"

    def test_coordinate_dtypes_are_float64(self, tmp_path):
        """``ERA5PressureLevelReader.samples_in`` serves latitudes and longitudes as float64.

        ``SampleSet`` defines its coordinates as float64, and the PSF weigher places each sample by them; a narrower
        dtype would shift the samples it weighs.

        Queries a synthetic file and asserts both coordinate arrays are float64.
        """
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path)
        reader = ERA5PressureLevelReader(fixture_path)
        _, lats, lons = _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)
        assert lats.dtype == np.float64
        assert lons.dtype == np.float64

    def test_partial_bbox_subsets_grid(self, tmp_path):
        """``ERA5PressureLevelReader.samples_in`` serves only the grid points inside the query box.

        A footprint must be aggregated from the cells around it; serving cells from outside the box would mix in values
        from elsewhere and multiply the memory of every query across all pressure layers.

        Builds an 8 x 4 grid over latitudes 0 to 4 and longitudes 10 to 12, queries a box inside it on every side
        (latitudes 1 to 3, longitudes 10.5 to 11.5, padded by about 0.13 degree), and asserts exactly the four interior
        rows and two interior columns are served.
        """
        fixture_path = make_era5_pressure_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=4.0, lon_min=10.0, lon_max=12.0, n_lat=8, n_lon=4
        )
        reader = ERA5PressureLevelReader(fixture_path)
        data, lats, lons = _region(reader, BoundingBox(1.0, 3.0, 10.5, 11.5), _T)
        np.testing.assert_allclose(lats, np.linspace(0.0, 4.0, 8)[2:6])
        np.testing.assert_allclose(lons, np.linspace(10.0, 12.0, 4)[1:3])
        assert data.shape[1:] == (4, 2)

    def test_no_time_dimension_variant_is_rejected(self, tmp_path):
        """``ERA5PressureLevelReader`` raises ``ValueError`` for a file with no time dimension.

        Without a time axis a step cannot be matched to a footprint's time, so the file is rejected rather than served
        for every time.

        Writes a pressure-level file without its ``valid_time`` dimension and asserts a query raises ``ValueError``
        naming the time dimension.
        """
        # A download with no time axis cannot be matched to a footprint time; the reader refuses it.
        fixture_path = make_era5_pressure_netcdf_fixture(tmp_path, with_valid_time=False)
        reader = ERA5PressureLevelReader(fixture_path)
        with pytest.raises(ValueError, match="time dimension"):
            reader.samples_in(BoundingBox(0.0, 2.0, 10.0, 12.0), _T)

    def test_step_load_peak_memory_is_one_step(self, tmp_path):
        """``ERA5PressureLevelReader._load_step`` reads a step with peak memory under 1.5 times its own size.

        Each (variable, level) plane is written into one preallocated array; stacking separate planes would hold the
        step twice, and a full-resolution global step of every layer is about 0.8 GB.

        Loads one step of a 60 x 120 synthetic grid under ``tracemalloc`` and compares the traced peak with the size of
        the returned data array.
        """
        fixture_path = make_era5_pressure_netcdf_fixture(
            tmp_path, n_lat=60, n_lon=120, lat_min=-15.0, lat_max=15.0, lon_min=-30.0, lon_max=30.0
        )
        reader = ERA5PressureLevelReader(fixture_path)
        _ = reader._steps
        tracemalloc.start()
        try:
            grid = reader._load_step(0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Each pressure plane is written into one preallocated array; stacking the planes
        # would hold at least twice the step.
        assert peak < 1.5 * grid.data.nbytes


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
        reader = ERA5PressureLevelReader(fixture_path)
        with pytest.raises(ValueError, match="missing configured pressure"):
            _region(reader, BoundingBox(0.0, 2.0, 10.0, 12.0), _T)

    def test_missing_variable_raises_key_error(self, tmp_path):
        """``ERA5PressureLevelReader`` raises ``KeyError`` when a file lacks one of the five pressure-level fields.

        A download configured without a field would otherwise leave its planes unfilled, and the product would carry
        values that were never read.

        Drops ``o3`` from a valid synthetic file and asserts a query raises ``KeyError`` naming the missing variable.
        """
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
        """``ERA5PressureLevelReader`` samples carry the source ``"era5_pressure"`` and no timestamp source.

        ERA5 is a reanalysis chosen by step time, and ``timestamp_source`` is reserved for cloud-property products tied
        to a radiometer or camera time; the source key tells the samples apart from the single-level reader's.

        Queries a 4 x 4 synthetic file over its full extent and asserts ``source``, ``timestamp_source`` and a values
        array of one row per spec by 16 samples.
        """
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
