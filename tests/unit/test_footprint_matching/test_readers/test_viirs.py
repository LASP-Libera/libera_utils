"""Unit tests for VIIRSCldpropD3Reader.

Uses synthetic CLDPROP_D3 NetCDF4 fixture files created by
``make_viirs_cldprop_d3_fixture`` so that the group navigation, fill handling,
and (lon, lat) → (lat, lon) transpose logic can be exercised without real data.

Real CLDPROP_D3 files can be downloaded from:
    NCEI CDR: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
    Example: CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.viirs import VIIRSCldpropD3Reader
from libera_utils.footprint_matching.types import BoundingBox, TimeMatch
from tests.test_data.footprint_matching.fixtures import make_viirs_cldprop_d3_fixture
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = np.datetime64("2026-01-01T00:00:00", "ns")

# Grid dimensions used in most tests.
_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.5
_LAT_MAX = 3.5
_LON_MIN = 10.5
_LON_MAX = 17.5


def _make_reader(tmp_path, **kwargs) -> tuple[VIIRSCldpropD3Reader, object]:
    """Return (reader, fixture_path) using default or overridden fixture params."""
    fixture_path = make_viirs_cldprop_d3_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        **kwargs,
    )
    return VIIRSCldpropD3Reader(fixture_path), fixture_path


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.1, _LAT_MAX + 0.1, _LON_MIN - 0.1, _LON_MAX + 0.1)


class TestVIIRSCldpropD3ReaderClassAttributes:
    def test_reader_key(self):
        """``VIIRSCldpropD3Reader.READER_KEY`` is ``"viirs_cldprop_d3"``.

        The key is how the pipeline looks the reader up and the prefix of every output variable name, so a changed key
        would break configuration lookups and rename the cloud variables in the product.

        Asserts the class attribute equals ``"viirs_cldprop_d3"``.
        """
        assert VIIRSCldpropD3Reader.READER_KEY == "viirs_cldprop_d3"

    def test_resolution_km(self):
        """``VIIRSCldpropD3Reader.RESOLUTION_KM`` is 111 km, the size of a 1 degree cell at the equator.

        The resolution sets how far a footprint's box is widened and how coverage is scored; a wrong value would drop
        grid cells that overlap a footprint's rim or misreport how much of the footprint has cloud data.

        Asserts the class attribute equals 111.0.
        """
        assert VIIRSCldpropD3Reader.RESOLUTION_KM == 111.0

    def test_variables_has_two_entries(self):
        """``VIIRSCldpropD3Reader.VARIABLES`` declares exactly two variables.

        The leading axis of every served array follows ``VARIABLES``, so an extra or missing spec would misalign the
        loaded fields with the product variables they are written to.

        Asserts ``len(VARIABLES) == 2``.
        """
        assert len(VIIRSCldpropD3Reader.VARIABLES) == 2

    def test_variable_names_in_order(self):
        """``VIIRSCldpropD3Reader.VARIABLES`` lists ``cloud_optical_thickness`` then ``cloud_top_pressure``.

        Values are matched to variables by position, so swapping the order would publish optical thickness as cloud top
        pressure and vice versa.

        Asserts the spec names equal ``["cloud_optical_thickness", "cloud_top_pressure"]`` in that order.
        """
        names = [v.name for v in VIIRSCldpropD3Reader.VARIABLES]
        assert names == ["cloud_optical_thickness", "cloud_top_pressure"]

    def test_variables_have_no_categories(self):
        """``VIIRSCldpropD3Reader.VARIABLES`` declares both cloud variables continuous (``n_categories`` None).

        Optical thickness and top pressure are averaged over a footprint; treating either as categorical would aggregate
        it as class fractions rather than a mean.

        Asserts ``n_categories is None`` for every spec.
        """
        for v in VIIRSCldpropD3Reader.VARIABLES:
            assert v.n_categories is None


class TestVIIRSCldpropD3ReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves both cloud variables for every grid point in the box.

        Downstream aggregation expects one value row per declared variable; a missing row would leave a cloud variable
        unfilled in the product.

        Queries the synthetic 4 x 8 fixture with a box covering the whole grid, folds the samples into a grid and
        asserts a 3-D array with 2 entries on the variable axis.
        """
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.ndim == 3
        assert data.shape[0] == 2

    def test_variable_stacking_order(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` stacks cloud optical thickness first and cloud top pressure second.

        Each row is written to the product variable at the same position, so reading the wrong group or stacking out of
        order would publish one cloud property under another's name.

        Builds a fixture whose cloud fraction, optical thickness and top pressure groups hold 0.6, 4.0 and 700.0 and
        asserts row 0 is 4.0 and row 1 is 700.0, so the unread cloud fraction group does not appear in either row.
        """
        reader, _ = _make_reader(tmp_path, cf_fill=0.6, cot_fill=4.0, ctp_fill=700.0)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[0], 4.0, equal_nan=True, atol=1e-4)  # cloud_optical_thickness
        assert np.allclose(data[1], 700.0, equal_nan=True, atol=1e-3)  # cloud_top_pressure

    def test_data_dtype_is_float32(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves float32 values.

        The variables are declared float32 and fill is carried as NaN, which an integer array could not hold.

        Queries the full fixture grid and asserts the served data dtype is ``float32``.
        """
        reader, _ = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert data.dtype == np.float32

    def test_output_lat_count_matches_n_lat(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves every latitude row of the grid inside a box that covers it.

        A dropped row would leave part of a footprint without cloud data and lower its reported coverage.

        Queries a box just larger than the 4 x 8 fixture and asserts 4 distinct latitudes are served.
        """
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert lats.size == _N_LAT

    def test_output_lon_count_matches_n_lon(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves every longitude column of the grid inside a box that covers it.

        A dropped column would leave part of a footprint without cloud data and lower its reported coverage.

        Queries a box just larger than the 4 x 8 fixture and asserts 8 distinct longitudes are served.
        """
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert lons.size == _N_LON

    def test_empty_result_outside_bbox(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves no samples, shaped for two variables, for a box off the grid.

        A footprint with no cloud data must come back empty, so it is scored as uncovered, and keep the variable axis so
        the caller can handle it like any other result.

        Queries a box at 60S, 171E, far from the fixture grid, and asserts the data is empty with 2 entries on the
        variable axis.
        """
        reader, _ = _make_reader(tmp_path)
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = _region(reader, bbox, _T)
        assert data.size == 0
        assert data.shape[0] == 2

    def test_fill_value_becomes_nan(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves cells holding the -9999.0 fill value as NaN.

        A fill sentinel left as a number would be averaged into a footprint's cloud top pressure as if it were data.

        Builds a fixture whose cloud top pressure field is all -9999.0, the declared ``_FillValue``, and asserts every
        served cloud top pressure is NaN.
        """
        fixture_path = make_viirs_cldprop_d3_fixture(
            tmp_path,
            n_lat=_N_LAT,
            n_lon=_N_LON,
            lat_min=_LAT_MIN,
            lat_max=_LAT_MAX,
            lon_min=_LON_MIN,
            lon_max=_LON_MAX,
            ctp_fill=-9999.0,
        )
        reader = VIIRSCldpropD3Reader(fixture_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # cloud_top_pressure (index 1) should be all NaN since fill_value = -9999.0
        assert np.all(np.isnan(data[1]))

    def test_fill_value_comes_from_the_field(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` masks the fill value each field declares, not a hard-coded -9999.0.

        A file that declares a different sentinel would otherwise have its fill averaged in as data, while a genuine
        value that happens to equal the usual sentinel would be discarded.

        Builds a fixture declaring ``_FillValue`` -999.0 with optical thickness all -9999.0 and top pressure all -999.0,
        and asserts optical thickness is served as -9999.0 and top pressure as NaN.
        """
        reader, _ = _make_reader(tmp_path, mean_fill_value=-999.0, cot_fill=-9999.0, ctp_fill=-999.0)
        data, _, _ = _region(reader, _full_bbox(), _T)
        np.testing.assert_array_equal(data[0], -9999.0)
        assert np.all(np.isnan(data[1]))

    def test_field_without_fill_value_raises(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` raises ``ValueError`` when a field declares no ``_FillValue``.

        Without a declared sentinel the reader cannot tell fill from data, and guessing one could publish fill as cloud
        values, so it stops instead.

        Builds a fixture whose ``Mean`` fields carry no ``_FillValue`` and asserts the query raises ``ValueError``
        matching "has no _FillValue attribute".
        """
        reader, _ = _make_reader(tmp_path, mean_fill_value=None)
        with pytest.raises(ValueError, match="has no _FillValue attribute"):
            _region(reader, _full_bbox(), _T)


class TestVIIRSCldpropD3ReaderTranspose:
    """Verify the (lon, lat) → (lat, lon) dimension transpose is applied."""

    def test_spatial_shape_after_transpose(self, tmp_path):
        """``VIIRSCldpropD3Reader`` transposes the file's (longitude, latitude) fields to (latitude, longitude).

        CLDPROP_D3 stores its fields longitude first; without the transpose the grid would not match its latitude and
        longitude axes and values would be placed at the wrong locations.

        Stores the 4 lat x 8 lon fixture as (8, 4), queries the full grid and asserts the served grid is 4 latitudes by
        8 longitudes.
        """
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        # With the default 4 lat × 8 lon fixture, axis 1 must be n_lat=4
        # and axis 2 must be n_lon=8.
        n_lat_out, n_lon_out = data.shape[1], data.shape[2]
        assert n_lat_out == _N_LAT
        assert n_lon_out == _N_LON

    def test_partial_lat_subset_selects_correct_rows(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves fewer rows but every column for a box over part of the latitudes.

        A footprint should be served only the cells near it; slicing the wrong axis of the transposed grid would cut
        columns instead of rows.

        Queries a box over the lower half of the fixture's latitudes and the full longitude range, and asserts fewer
        than 4 latitudes and all 8 longitudes are served.
        """
        reader, _ = _make_reader(tmp_path)
        half_lat = (_LAT_MIN + _LAT_MAX) / 2
        bbox = BoundingBox(_LAT_MIN - 0.1, half_lat, _LON_MIN - 0.1, _LON_MAX + 0.1)
        data, lats, lons = _region(reader, bbox, _T)
        assert lats.size < _N_LAT
        assert lons.size == _N_LON


class TestVIIRSCldpropD3ReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` tags samples with source ``"viirs_cldprop_d3"`` and no timestamp source.

        A daily composite has no radiometer or camera observation time, so claiming one would misstate the cloud data's
        provenance; the source label ties the samples to this reader.

        Queries the full fixture grid and asserts ``timestamp_source`` is None and ``source`` is ``"viirs_cldprop_d3"``.
        """
        reader, _ = _make_reader(tmp_path)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.timestamp_source is None
        assert samples.source == "viirs_cldprop_d3"

    def test_daily_composite_ignores_time(self, tmp_path):
        """``VIIRSCldpropD3Reader`` is ``TimeMatch.DAILY``, so ``samples_in`` serves the same values at any hour.

        Every footprint in the processing day matches the one daily composite; a time selection would drop footprints
        late in the day from cloud matching.

        Asserts ``TIME_MATCH`` is ``TimeMatch.DAILY`` and that queries at midnight and 23 hours later serve identical
        values.
        """
        reader, _ = _make_reader(tmp_path)
        assert VIIRSCldpropD3Reader.TIME_MATCH is TimeMatch.DAILY
        a = reader.samples_in(_full_bbox(), _T)
        b = reader.samples_in(_full_bbox(), _T + np.timedelta64(23, "h"))
        np.testing.assert_array_equal(a.values, b.values)


class TestVIIRSCldpropD3ReaderRealSubset:
    """The full 1 degree grid of a real CLDPROP_D3_VIIRS_NOAA20 granule (2026-05-27).

    Only the coordinates and the two Mean fields the reader reads are kept, stored as
    in the product with dimensions (longitude, latitude). Pinned values are the source
    granule's own.
    """

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return VIIRSCldpropD3Reader(
            fmatch_real_data_path / "CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710_subset.nc"
        )

    def test_grid_is_transposed_to_latitude_by_longitude(self, reader):
        """``VIIRSCldpropD3Reader._load_native`` loads a real granule as a 180 x 360 latitude-by-longitude grid.

        The distributed product stores fields longitude first; the real granule confirms the transpose and the 1 degree
        bin centers, so values land at the right locations.

        Loads the real subset granule and asserts the data shape is (2, 180, 360), latitudes run -89.5 to 89.5 and
        longitudes -179.5 to 179.5.
        """
        native = reader._load_native()
        assert native.data.shape == (2, 180, 360)
        assert (native.lat_axis[0], native.lat_axis[-1]) == (-89.5, 89.5)
        assert (native.lon_axis[0], native.lon_axis[-1]) == (-179.5, 179.5)

    def test_fill_cells_are_nan(self, reader):
        """``VIIRSCldpropD3Reader._load_native`` turns every fill cell of a real granule into NaN.

        Cells with no retrieval carry -9999 in the product; any left as numbers would be averaged into footprints as
        cloud values.

        Loads the real subset granule and asserts the NaN counts equal the granule's fill counts: 16035 for optical
        thickness and 885 for top pressure.
        """
        cot, ctp = reader._load_native().data
        # -9999 fill: 16 035 of the 64 800 cells have no optical thickness, 885 no top pressure.
        assert np.isnan(cot).sum() == 16035
        assert np.isnan(ctp).sum() == 885

    def test_values_at_cells_are_the_granules(self, reader):
        """``VIIRSCldpropD3Reader.samples_in`` serves a real granule's values at the cell they belong to.

        A misplaced or rescaled value would attach another location's cloud properties to a footprint.

        Queries two 1 degree boxes and asserts the cell at (-29.5, 120.5) holds the granule's optical thickness and top
        pressure, and the cell at (30.5, 20.5) has NaN optical thickness and the granule's top pressure.
        """
        samples = reader.samples_in(BoundingBox(-30.0, -29.0, 120.0, 121.0), _T)
        cot, ctp = sample_at(samples, -29.5, 120.5)
        assert cot == pytest.approx(9.74419, abs=1e-4)
        assert ctp == pytest.approx(332.3532, abs=1e-3)
        samples = reader.samples_in(BoundingBox(30.0, 31.0, 20.0, 21.0), _T)
        cot, ctp = sample_at(samples, 30.5, 20.5)
        assert np.isnan(cot)
        assert ctp == pytest.approx(849.7842, abs=1e-3)
