"""Unit tests for VIIRSCloudReader.

Uses synthetic CLDPROP_D3 NetCDF4 fixture files created by
``make_viirs_cloud_d3_fixture`` so that the group navigation, fill handling,
and (lon, lat) → (lat, lon) transpose logic can be exercised without real data.

Real CLDPROP_D3 files can be downloaded from:
    NCEI CDR: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
    Example: CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.viirs import VIIRSCloudReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, TileKey
from tests.test_data.footprint_matching.fixtures import make_viirs_cloud_d3_fixture

# Grid dimensions used in most tests.
_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.5
_LAT_MAX = 3.5
_LON_MIN = 10.5
_LON_MAX = 17.5


def _make_reader(tmp_path, **kwargs) -> tuple[VIIRSCloudReader, object]:
    """Return (reader, fixture_path) using default or overridden fixture params."""
    fixture_path = make_viirs_cloud_d3_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        **kwargs,
    )
    return VIIRSCloudReader(fixture_path), fixture_path


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.1, _LAT_MAX + 0.1, _LON_MIN - 0.1, _LON_MAX + 0.1)


class TestVIIRSCloudReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "viirs_cloud".
        assert VIIRSCloudReader.READER_KEY == "viirs_cloud"

    def test_resolution_km(self):
        # Targets the declared grid resolution; asserts RESOLUTION_KM equals 111.0 km.
        assert VIIRSCloudReader.RESOLUTION_KM == 111.0

    def test_variables_has_two_entries(self):
        # Targets the VARIABLES count; asserts exactly two variable specs are declared.
        assert len(VIIRSCloudReader.VARIABLES) == 2

    def test_variable_names_in_order(self):
        # Targets VARIABLES ordering; asserts names are ["cloud_optical_thickness", "cloud_top_pressure"].
        names = [v.name for v in VIIRSCloudReader.VARIABLES]
        assert names == ["cloud_optical_thickness", "cloud_top_pressure"]

    def test_variables_have_no_categories(self):
        # Targets that cloud variables are continuous; asserts every spec's n_categories is None.
        for v in VIIRSCloudReader.VARIABLES:
            assert v.n_categories is None


class TestVIIRSCloudReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        # Targets _load_spatial_region output; asserts a 3-D array whose leading axis holds the 2 variables.
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert data.ndim == 3
        assert data.shape[0] == 2

    def test_variable_stacking_order(self, tmp_path):
        # Targets variable stacking order; asserts data[0] is cloud_optical_thickness and data[1] is cloud_top_pressure,
        # while the still-written-but-unread Cloud_Fraction group (cf_fill) stays absent from the output.
        reader, _ = _make_reader(tmp_path, cf_fill=0.6, cot_fill=4.0, ctp_fill=700.0)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.allclose(data[0], 4.0, equal_nan=True, atol=1e-4)  # cloud_optical_thickness
        assert np.allclose(data[1], 700.0, equal_nan=True, atol=1e-3)  # cloud_top_pressure

    def test_data_dtype_is_float32(self, tmp_path):
        # Targets output dtype; asserts the loaded data array is float32.
        reader, _ = _make_reader(tmp_path)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert data.dtype == np.float32

    def test_output_lat_count_matches_n_lat(self, tmp_path):
        # Targets latitude-axis sizing; asserts the returned lats array has _N_LAT entries.
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert lats.size == _N_LAT

    def test_output_lon_count_matches_n_lon(self, tmp_path):
        # Targets longitude-axis sizing; asserts the returned lons array has _N_LON entries.
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert lons.size == _N_LON

    def test_empty_result_outside_bbox(self, tmp_path):
        # Targets querying a bbox with no data; asserts the data array is empty but keeps its leading 2-variable axis.
        reader, _ = _make_reader(tmp_path)
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = reader._load_spatial_region(bbox)
        assert data.size == 0
        assert data.shape[0] == 2

    def test_fill_value_becomes_nan(self, tmp_path):
        # Targets D3 fill-value masking; builds a fixture with ctp_fill=-9999.0 (the D3 fill) so the field is all fill.
        fixture_path = make_viirs_cloud_d3_fixture(
            tmp_path,
            n_lat=_N_LAT,
            n_lon=_N_LON,
            lat_min=_LAT_MIN,
            lat_max=_LAT_MAX,
            lon_min=_LON_MIN,
            lon_max=_LON_MAX,
            ctp_fill=-9999.0,
        )
        reader = VIIRSCloudReader(fixture_path)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        # cloud_top_pressure (index 1) should be all NaN since fill_value = -9999.0
        assert np.all(np.isnan(data[1]))


class TestVIIRSCloudReaderTranspose:
    """Verify the (lon, lat) → (lat, lon) dimension transpose is applied."""

    def test_spatial_shape_after_transpose(self, tmp_path):
        # The fixture stores data as (n_lon, n_lat); after transpose + subset the
        # output shape must be (3, n_lat, n_lon), NOT (3, n_lon, n_lat).
        reader, _ = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
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
        data, lats, lons = reader._load_spatial_region(bbox)
        assert lats.size < _N_LAT
        assert lons.size == _N_LON


class TestVIIRSCloudReaderLoadTile:
    def test_timestamp_source_is_radiometer(self, tmp_path):
        # Targets tile timestamp provenance; asserts the loaded tile is a GridTile with timestamp_source "radiometer".
        reader, _ = _make_reader(tmp_path)
        # TileKey that maps to lat [0, 2°], lon [10, 12°]
        key = TileKey("viirs_cloud", 45, 95)
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.timestamp_source == "radiometer"

    def test_source_is_viirs_cloud(self, tmp_path):
        # Targets tile provenance labeling; asserts the loaded tile's source is "viirs_cloud".
        reader, _ = _make_reader(tmp_path)
        key = TileKey("viirs_cloud", 45, 95)
        tile = reader.load_tile(key)
        assert tile.source == "viirs_cloud"
