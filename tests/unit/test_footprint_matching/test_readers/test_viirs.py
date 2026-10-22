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
        # Targets the reader's registry key; asserts READER_KEY equals the expected "viirs_cldprop_d3".
        assert VIIRSCldpropD3Reader.READER_KEY == "viirs_cldprop_d3"

    def test_resolution_km(self):
        # Targets the declared grid resolution; asserts RESOLUTION_KM equals 111.0 km.
        assert VIIRSCldpropD3Reader.RESOLUTION_KM == 111.0

    def test_variables_has_two_entries(self):
        # Targets the VARIABLES count; asserts exactly two variable specs are declared.
        assert len(VIIRSCldpropD3Reader.VARIABLES) == 2

    def test_variable_names_in_order(self):
        # Targets VARIABLES ordering; asserts names are ["cloud_optical_thickness", "cloud_top_pressure"].
        names = [v.name for v in VIIRSCldpropD3Reader.VARIABLES]
        assert names == ["cloud_optical_thickness", "cloud_top_pressure"]

    def test_variables_have_no_categories(self):
        # Targets that cloud variables are continuous; asserts every spec's n_categories is None.
        for v in VIIRSCldpropD3Reader.VARIABLES:
            assert v.n_categories is None


class TestVIIRSCldpropD3ReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        # Targets samples_in output; asserts a 3-D array whose leading axis holds the 2 variables.
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.ndim == 3
        assert data.shape[0] == 2

    def test_variable_stacking_order(self, tmp_path):
        # Targets variable stacking order; asserts data[0] is cloud_optical_thickness and data[1] is cloud_top_pressure,
        # while the still-written-but-unread Cloud_Fraction group (cf_fill) stays absent from the output.
        reader, _ = _make_reader(tmp_path, cf_fill=0.6, cot_fill=4.0, ctp_fill=700.0)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[0], 4.0, equal_nan=True, atol=1e-4)  # cloud_optical_thickness
        assert np.allclose(data[1], 700.0, equal_nan=True, atol=1e-3)  # cloud_top_pressure

    def test_data_dtype_is_float32(self, tmp_path):
        # Targets output dtype; asserts the loaded data array is float32.
        reader, _ = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert data.dtype == np.float32

    def test_output_lat_count_matches_n_lat(self, tmp_path):
        # Targets latitude-axis sizing; asserts the returned lats array has _N_LAT entries.
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert lats.size == _N_LAT

    def test_output_lon_count_matches_n_lon(self, tmp_path):
        # Targets longitude-axis sizing; asserts the returned lons array has _N_LON entries.
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert lons.size == _N_LON

    def test_empty_result_outside_bbox(self, tmp_path):
        # Targets querying a bbox with no data; asserts the data array is empty but keeps its leading 2-variable axis.
        reader, _ = _make_reader(tmp_path)
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = _region(reader, bbox, _T)
        assert data.size == 0
        assert data.shape[0] == 2

    def test_fill_value_becomes_nan(self, tmp_path):
        # Targets D3 fill-value masking; builds a fixture with ctp_fill=-9999.0 (the D3 fill) so the field is all fill.
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
        # Targets reading the field's own _FillValue; a file declaring -999.0 masks -999.0 and keeps -9999.0.
        reader, _ = _make_reader(tmp_path, mean_fill_value=-999.0, cot_fill=-9999.0, ctp_fill=-999.0)
        data, _, _ = _region(reader, _full_bbox(), _T)
        np.testing.assert_array_equal(data[0], -9999.0)
        assert np.all(np.isnan(data[1]))

    def test_field_without_fill_value_raises(self, tmp_path):
        # Targets a file whose Mean fields declare no _FillValue; the reader refuses to guess one.
        reader, _ = _make_reader(tmp_path, mean_fill_value=None)
        with pytest.raises(ValueError, match="has no _FillValue attribute"):
            _region(reader, _full_bbox(), _T)


class TestVIIRSCldpropD3ReaderTranspose:
    """Verify the (lon, lat) → (lat, lon) dimension transpose is applied."""

    def test_spatial_shape_after_transpose(self, tmp_path):
        # The fixture stores data as (n_lon, n_lat); after transpose + subset the
        # output shape must be (3, n_lat, n_lon), NOT (3, n_lon, n_lat).
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        # With the default 4 lat × 8 lon fixture, axis 1 must be n_lat=4
        # and axis 2 must be n_lon=8.
        n_lat_out, n_lon_out = data.shape[1], data.shape[2]
        assert n_lat_out == _N_LAT
        assert n_lon_out == _N_LON

    def test_partial_lat_subset_selects_correct_rows(self, tmp_path):
        # Request only the lower half of the lat range; the result should
        # have fewer lat points but the full lon range.
        reader, _ = _make_reader(tmp_path)
        half_lat = (_LAT_MIN + _LAT_MAX) / 2
        bbox = BoundingBox(_LAT_MIN - 0.1, half_lat, _LON_MIN - 0.1, _LON_MAX + 0.1)
        data, lats, lons = _region(reader, bbox, _T)
        assert lats.size < _N_LAT
        assert lons.size == _N_LON


class TestVIIRSCldpropD3ReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        # A daily composite has no observation time, so the samples claim no timestamp source.
        reader, _ = _make_reader(tmp_path)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.timestamp_source is None
        assert samples.source == "viirs_cldprop_d3"

    def test_daily_composite_ignores_time(self, tmp_path):
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
        native = reader._load_native()
        assert native.data.shape == (2, 180, 360)
        assert (native.lat_axis[0], native.lat_axis[-1]) == (-89.5, 89.5)
        assert (native.lon_axis[0], native.lon_axis[-1]) == (-179.5, 179.5)

    def test_fill_cells_are_nan(self, reader):
        cot, ctp = reader._load_native().data
        # -9999 fill: 16 035 of the 64 800 cells have no optical thickness, 885 no top pressure.
        assert np.isnan(cot).sum() == 16035
        assert np.isnan(ctp).sum() == 885

    def test_values_at_cells_are_the_granules(self, reader):
        samples = reader.samples_in(BoundingBox(-30.0, -29.0, 120.0, 121.0), _T)
        cot, ctp = sample_at(samples, -29.5, 120.5)
        assert cot == pytest.approx(9.74419, abs=1e-4)
        assert ctp == pytest.approx(332.3532, abs=1e-3)
        samples = reader.samples_in(BoundingBox(30.0, 31.0, 20.0, 21.0), _T)
        cot, ctp = sample_at(samples, 30.5, 20.5)
        assert np.isnan(cot)
        assert ctp == pytest.approx(849.7842, abs=1e-3)
