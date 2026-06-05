"""Unit tests for NSIDCReader.

Uses real synthetic ASCII fixture files (created by make_nsidc_ascii_fixture)
with a small 4 × 4 grid to keep tests fast. NSIDCReader is initialized with
matching grid dimensions and a simple pyproj transformer.

Real IMS 24-km files can be downloaded from:
    NOAA NSIDC FTP: https://noaadata.apps.nsidc.org/NOAA/G02156/24km/
    No login required; files are publicly accessible.
"""
from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.nsidc import NSIDCReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, TileKey
from tests.test_data.footprint_matching.fixtures import make_nsidc_ascii_fixture

# Small test grid parameters — passed to NSIDCReader to override 1024×1024 defaults.
# These values are chosen so the pyproj-converted lat/lons produce a reasonable
# geographic extent for the bounding box tests below.
_TEST_ROWS = 4
_TEST_COLS = 4
# Reduce resolution so the 4×4 grid covers a usable geographic extent.
_TEST_RESOLUTION_M = 500_000.0  # 500 km — very coarse, but fine for unit tests
# x_origin and y_origin position the test grid in the Northern Hemisphere.
_TEST_X_ORIGIN = -1_000_000.0  # meters
_TEST_Y_ORIGIN = 1_000_000.0   # meters


def _make_reader(tmp_path, data=None, gzipped=False):
    """Helper: write fixture file and return an NSIDCReader with test grid params."""
    fixture_path = make_nsidc_ascii_fixture(
        tmp_path,
        grid_rows=_TEST_ROWS,
        grid_cols=_TEST_COLS,
        data=data,
        gzipped=gzipped,
    )
    return NSIDCReader(
        fixture_path,
        grid_rows=_TEST_ROWS,
        grid_cols=_TEST_COLS,
        resolution_m=_TEST_RESOLUTION_M,
        x_origin=_TEST_X_ORIGIN,
        y_origin=_TEST_Y_ORIGIN,
    )


class TestNSIDCReaderClassAttributes:
    def test_reader_key(self):
        assert NSIDCReader.READER_KEY == "nsidc"

    def test_resolution_km(self):
        assert NSIDCReader.RESOLUTION_KM == 25.0

    def test_required_mode_is_cam(self):
        assert NSIDCReader.REQUIRED_MODE == OperationalMode.CAM

    def test_variables_has_one_entry(self):
        assert len(NSIDCReader.VARIABLES) == 1

    def test_variable_name_is_sea_ice_type(self):
        assert NSIDCReader.VARIABLES[0].name == "sea_ice_type"

    def test_n_categories_is_5(self):
        assert NSIDCReader.VARIABLES[0].n_categories == 5


class TestNSIDCReaderParseAscii:
    def test_parse_plain_text_file(self, tmp_path):
        reader = _make_reader(tmp_path)
        grid = reader._parse_ascii_grid()
        assert grid.shape == (_TEST_ROWS, _TEST_COLS)

    def test_parse_gzipped_file(self, tmp_path):
        reader = _make_reader(tmp_path, gzipped=True)
        grid = reader._parse_ascii_grid()
        assert grid.shape == (_TEST_ROWS, _TEST_COLS)

    def test_values_are_in_valid_range(self, tmp_path):
        reader = _make_reader(tmp_path)
        grid = reader._parse_ascii_grid()
        assert np.all((grid >= 0) & (grid <= 4))

    def test_specific_values_match_fixture(self, tmp_path):
        # Use a known data array and verify round-trip fidelity.
        known_data = np.array([[1, 2, 1, 2], [2, 1, 2, 1], [0, 3, 0, 3], [4, 0, 4, 0]], dtype=np.int8)
        reader = _make_reader(tmp_path, data=known_data)
        grid = reader._parse_ascii_grid()
        np.testing.assert_array_equal(grid, known_data)

    def test_wrong_row_count_raises_value_error(self, tmp_path):
        # Write a fixture with fewer rows than the reader expects.
        fixture = make_nsidc_ascii_fixture(tmp_path, grid_rows=2, grid_cols=4)
        reader = NSIDCReader(
            fixture,
            grid_rows=8,  # expects 8 rows but file only has 2
            grid_cols=4,
            resolution_m=_TEST_RESOLUTION_M,
            x_origin=_TEST_X_ORIGIN,
            y_origin=_TEST_Y_ORIGIN,
        )
        with pytest.raises(ValueError, match="Expected 8 data rows"):
            reader._parse_ascii_grid()


class TestNSIDCReaderLatLonGrid:
    def test_lat_lon_grid_shape(self, tmp_path):
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        assert lats_2d.shape == (_TEST_ROWS, _TEST_COLS)
        assert lons_2d.shape == (_TEST_ROWS, _TEST_COLS)

    def test_lat_lon_values_are_in_valid_range(self, tmp_path):
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        # All lats should be in [-90, 90]; all lons in [-180, 180]
        assert np.all((lats_2d >= -90) & (lats_2d <= 90))
        assert np.all((lons_2d >= -180) & (lons_2d <= 180))

    def test_northern_hemisphere_coverage(self, tmp_path):
        # With the test grid origin in the Northern Hemisphere, most lat values
        # should be positive (north of equator).
        reader = _make_reader(tmp_path)
        lats_2d, _ = reader._compute_latlon_grid()
        # At least half the pixels should be in the Northern Hemisphere.
        assert np.sum(lats_2d > 0) >= _TEST_ROWS * _TEST_COLS // 2


class TestNSIDCReaderLoadSpatialRegion:
    def test_returns_data_lats_lons(self, tmp_path):
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        # Use the actual geographic extent of the test grid as the bbox.
        bbox = BoundingBox(
            lat_min=float(lats_2d.min()) - 0.1,
            lat_max=float(lats_2d.max()) + 0.1,
            lon_min=float(lons_2d.min()) - 0.1,
            lon_max=float(lons_2d.max()) + 0.1,
        )
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        assert data_sub.size > 0
        assert lats_sub.ndim == 1
        assert lons_sub.ndim == 1

    def test_empty_result_outside_bbox(self, tmp_path):
        reader = _make_reader(tmp_path)
        # Request a bbox in the Southern Hemisphere far from the test grid.
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        assert data_sub.size == 0

    def test_data_dtype_is_int16(self, tmp_path):
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            lat_min=float(lats_2d.min()) - 0.1,
            lat_max=float(lats_2d.max()) + 0.1,
            lon_min=float(lons_2d.min()) - 0.1,
            lon_max=float(lons_2d.max()) + 0.1,
        )
        data_sub, _, _ = reader._load_spatial_region(bbox)
        assert data_sub.dtype == np.int16
