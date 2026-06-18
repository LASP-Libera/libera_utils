"""Unit tests for GriddedDataReader base class and TileKey → BoundingBox conversion.

Tests confirm:
- _tile_key_to_bbox correctly maps integer tile indices to geographic bounds
- Abstract base class cannot be instantiated
- load_tile delegates to _load_spatial_region and wraps the result in GridTile
- TILE_SIZE_DEG constant is 2.0
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG, GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey, VariableSpec

# ---------------------------------------------------------------------------
# Minimal concrete reader used only in tests
# ---------------------------------------------------------------------------


class _FakeReader(GriddedDataReader):
    """Minimal concrete subclass for testing base class behavior."""

    # Use a key that won't collide with production readers.
    READER_KEY = "_fake_test_reader"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    VARIABLES = (
        VariableSpec(
            name="fake_var",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
        ),
    )

    def _load_spatial_region(self, bbox: BoundingBox):
        # Return 2×2 grid of ones within the bbox.
        lats = np.array([bbox.lat_min + 0.5, bbox.lat_min + 1.5])
        lons = np.array([bbox.lon_min + 0.5, bbox.lon_min + 1.5])
        data = np.ones((2, 2), dtype=np.float32)
        return data, lats, lons


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTileSizeConstant:
    def test_tile_size_is_two_degrees(self):
        assert TILE_SIZE_DEG == 2.0


class TestTileKeyToBoundingBox:
    def test_origin_tile(self):
        # TileKey(source, lat_idx=0, lon_idx=0) → lat [-90, -88], lon [-180, -178]
        key = TileKey("igbp", 0, 0)
        bbox = GriddedDataReader._tile_key_to_bbox(key)
        assert bbox.lat_min == pytest.approx(-90.0)
        assert bbox.lat_max == pytest.approx(-88.0)
        assert bbox.lon_min == pytest.approx(-180.0)
        assert bbox.lon_max == pytest.approx(-178.0)

    def test_equatorial_tile(self):
        # lat_idx=45 → lat 0°–2°; lon_idx=90 → lon 0°–2°
        key = TileKey("igbp", 45, 90)
        bbox = GriddedDataReader._tile_key_to_bbox(key)
        assert bbox.lat_min == pytest.approx(0.0)
        assert bbox.lat_max == pytest.approx(2.0)
        assert bbox.lon_min == pytest.approx(0.0)
        assert bbox.lon_max == pytest.approx(2.0)

    def test_polar_tile_sets_is_polar(self):
        # lat_idx=88 → lat_min = -90 + 88*2 = 86°; lat_max = 88° → is_polar=True
        key = TileKey("nsidc", 88, 90)
        bbox = GriddedDataReader._tile_key_to_bbox(key)
        assert bbox.is_polar is True

    def test_non_polar_tile_is_not_polar(self):
        key = TileKey("era5", 45, 90)
        bbox = GriddedDataReader._tile_key_to_bbox(key)
        assert bbox.is_polar is False

    def test_bbox_is_bounding_box_type(self):
        key = TileKey("igbp", 45, 90)
        bbox = GriddedDataReader._tile_key_to_bbox(key)
        assert isinstance(bbox, BoundingBox)

    def test_tile_width_is_tile_size_deg(self):
        for lat_idx, lon_idx in [(0, 0), (45, 90), (89, 179)]:
            key = TileKey("igbp", lat_idx, lon_idx)
            bbox = GriddedDataReader._tile_key_to_bbox(key)
            assert bbox.lat_max - bbox.lat_min == pytest.approx(TILE_SIZE_DEG)
            assert bbox.lon_max - bbox.lon_min == pytest.approx(TILE_SIZE_DEG)


class TestAbstractBaseClass:
    def test_cannot_instantiate_abstract_class_directly(self):
        # GriddedDataReader has _load_spatial_region as abstractmethod.
        with pytest.raises(TypeError):
            GriddedDataReader(file_path="anything")  # type: ignore[abstract]


class TestLoadTile:
    def test_load_tile_returns_grid_tile(self, tmp_path):
        reader = _FakeReader(tmp_path / "dummy.nc")
        key = TileKey("_fake_test_reader", 45, 90)
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)

    def test_load_tile_source_matches_reader_key(self, tmp_path):
        reader = _FakeReader(tmp_path / "dummy.nc")
        key = TileKey("_fake_test_reader", 45, 90)
        tile = reader.load_tile(key)
        assert tile.source == "_fake_test_reader"

    def test_load_tile_data_shape(self, tmp_path):
        reader = _FakeReader(tmp_path / "dummy.nc")
        key = TileKey("_fake_test_reader", 45, 90)
        tile = reader.load_tile(key)
        assert tile.data.shape == (2, 2)

    def test_load_tile_bounds_match_key(self, tmp_path):
        reader = _FakeReader(tmp_path / "dummy.nc")
        key = TileKey("_fake_test_reader", 45, 90)
        tile = reader.load_tile(key)
        expected_bbox = GriddedDataReader._tile_key_to_bbox(key)
        assert tile.bounds == expected_bbox

    def test_load_tile_default_timestamp_source_is_none(self, tmp_path):
        reader = _FakeReader(tmp_path / "dummy.nc")
        key = TileKey("_fake_test_reader", 45, 90)
        tile = reader.load_tile(key)
        assert tile.timestamp_source is None

    def test_file_path_property(self, tmp_path):
        path = tmp_path / "dummy.nc"
        reader = _FakeReader(path)
        assert reader.file_path == path
