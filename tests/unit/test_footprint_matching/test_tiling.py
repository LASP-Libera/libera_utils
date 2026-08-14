"""Unit tests for the tiled data manager and LRU tile cache (``tiling.py``).

Covers:
- resolve_tile_keys: single-tile, multi-tile, and per-reader tile sizing
- LRU cache hit/miss accounting and byte-budget eviction
- merge_tiles: 2-D and 3-D stitching, NaN-filled gaps, single-tile fast path,
  all-empty fallback
- dateline splitting in get_data
- reader-error path (catch -> empty tile, not cached, segment continues)
- unknown-source KeyError
- build_tile_manager wiring and gather_footprint_tiles seam

No real data files or AWS are touched: a minimal in-test reader returns synthetic
grids, matching the pattern used in ``test_readers/test_base.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.tiling import (
    DEFAULT_MAX_CACHE_BYTES,
    TileManager,
    build_tile_manager,
    gather_footprint_tiles,
)
from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridTile,
    OperationalMode,
    TileKey,
    VariableSpec,
)

# ---------------------------------------------------------------------------
# Minimal controllable reader used only in these tests
# ---------------------------------------------------------------------------


class _FakeTilingReader(GriddedDataReader):
    """Concrete reader returning synthetic 2×2 grids; can be told to raise.

    ``TILE_SIZE_DEG`` is overridable per instance so the per-reader tile-sizing
    behaviour can be exercised without touching a production reader.
    """

    READER_KEY = "_fake_tiling"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    VARIABLES = (
        VariableSpec(name="fake_var", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
    )

    def __init__(self, file_path, tile_size_deg: float = 2.0) -> None:
        super().__init__(file_path)
        # Instance-level override so a single test can compare tile sizes.
        self.TILE_SIZE_DEG = tile_size_deg
        self.load_calls = 0
        self.raise_on_load = False

    def _load_spatial_region(self, bbox: BoundingBox):
        self.load_calls += 1
        if self.raise_on_load:
            raise RuntimeError("simulated reader failure")
        # Two cell centres inside the tile on each axis.
        lats = np.array([bbox.lat_min + 0.5, bbox.lat_min + 1.5], dtype=np.float64)
        lons = np.array([bbox.lon_min + 0.5, bbox.lon_min + 1.5], dtype=np.float64)
        data = np.ones((2, 2), dtype=np.float32)
        return data, lats, lons


def _reader(tmp_path, tile_size_deg: float = 2.0) -> _FakeTilingReader:
    return _FakeTilingReader(tmp_path / "fake.nc", tile_size_deg=tile_size_deg)


def _manager(reader: _FakeTilingReader, max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES) -> TileManager:
    return TileManager({reader.READER_KEY: reader}, OperationalMode.CAM, max_cache_bytes=max_cache_bytes)


def _tile(lats, lons, data, *, source="_fake_tiling", timestamp_source=None) -> GridTile:
    """Hand-build a GridTile for direct merge_tiles tests."""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    data = np.asarray(data)
    bounds = BoundingBox(
        float(lats.min()) - 0.5, float(lats.max()) + 0.5, float(lons.min()) - 0.5, float(lons.max()) + 0.5
    )
    return GridTile(data=data, lats=lats, lons=lons, bounds=bounds, source=source, timestamp_source=timestamp_source)


# ---------------------------------------------------------------------------
# resolve_tile_keys
# ---------------------------------------------------------------------------


class TestResolveTileKeys:
    def test_single_tile_box(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        keys = tm.resolve_tile_keys("_fake_tiling", BoundingBox(0.2, 0.8, 0.2, 0.8))
        assert keys == [TileKey("_fake_tiling", 45, 90)]

    def test_multi_tile_box_spans_two_lon_tiles(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        keys = tm.resolve_tile_keys("_fake_tiling", BoundingBox(0.2, 0.8, 0.5, 2.5))
        assert keys == [TileKey("_fake_tiling", 45, 90), TileKey("_fake_tiling", 45, 91)]

    def test_box_ending_on_tile_boundary_does_not_pull_next_tile(self, tmp_path):
        # lon_max exactly 2.0° must not add tile 91 (epsilon guard).
        tm = _manager(_reader(tmp_path))
        keys = tm.resolve_tile_keys("_fake_tiling", BoundingBox(0.2, 0.8, 0.2, 2.0))
        assert keys == [TileKey("_fake_tiling", 45, 90)]

    def test_degenerate_box_on_tile_boundary_still_returns_a_key(self, tmp_path):
        # A zero-area box (e.g. a single-pixel camera footprint) whose min edge lands
        # exactly on a tile boundary must still resolve to the tile containing it --
        # otherwise the max-edge epsilon leaves stop below start and the key list is
        # empty, which would raise "merge_tiles requires at least one tile" downstream.
        tm = _manager(_reader(tmp_path))
        keys = tm.resolve_tile_keys("_fake_tiling", BoundingBox(0.0, 0.0, 0.0, 0.0))
        assert keys == [TileKey("_fake_tiling", 45, 90)]

    def test_per_reader_tile_size_changes_key_count(self, tmp_path):
        # Same 1.8°-wide box: 2° tiles -> 1 key; 1° tiles -> 2 keys.
        box = BoundingBox(0.1, 0.9, 0.1, 1.9)
        coarse = _manager(_reader(tmp_path, tile_size_deg=2.0))
        fine = _manager(_reader(tmp_path, tile_size_deg=1.0))
        assert len(coarse.resolve_tile_keys("_fake_tiling", box)) == 1
        assert len(fine.resolve_tile_keys("_fake_tiling", box)) == 2

    def test_unknown_source_raises(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        with pytest.raises(KeyError):
            tm.resolve_tile_keys("nope", BoundingBox(0.2, 0.8, 0.2, 0.8))


# ---------------------------------------------------------------------------
# get_data + cache
# ---------------------------------------------------------------------------


class TestGetDataAndCache:
    def test_cache_hit_avoids_second_read(self, tmp_path):
        reader = _reader(tmp_path)
        tm = _manager(reader)
        box = BoundingBox(0.2, 0.8, 0.2, 0.8)
        tm.get_data("_fake_tiling", box)
        tm.get_data("_fake_tiling", box)
        assert reader.load_calls == 1  # second request served from cache
        stats = tm.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_multi_tile_merge_reads_each_tile_once(self, tmp_path):
        reader = _reader(tmp_path)
        tm = _manager(reader)
        merged = tm.get_data("_fake_tiling", BoundingBox(0.2, 0.8, 0.5, 2.5))
        assert reader.load_calls == 2
        # Union of two 2×2 tiles -> 2 lats × 4 lons.
        assert merged.data.shape == (2, 4)

    def test_byte_budget_evicts_lru(self, tmp_path):
        reader = _reader(tmp_path)
        # One synthetic tile is 48 bytes (16 data + 16 lats + 16 lons); budget of
        # 60 holds exactly one, so a second insertion evicts the first.
        tm = _manager(reader, max_cache_bytes=60)
        box_a = BoundingBox(0.2, 0.8, 0.2, 0.8)  # tile (45, 90)
        box_b = BoundingBox(0.2, 0.8, 2.2, 2.8)  # tile (45, 91)
        tm.get_data("_fake_tiling", box_a)  # load A
        tm.get_data("_fake_tiling", box_b)  # load B, evict A
        tm.get_data("_fake_tiling", box_a)  # A gone -> load again
        assert reader.load_calls == 3
        stats = tm.get_cache_stats()
        assert stats["evictions"] >= 1
        assert stats["n_tiles"] == 1

    def test_clear_cache_empties_cache(self, tmp_path):
        reader = _reader(tmp_path)
        tm = _manager(reader)
        tm.get_data("_fake_tiling", BoundingBox(0.2, 0.8, 0.2, 0.8))
        assert tm.get_cache_stats()["n_tiles"] == 1
        tm.clear_cache()
        assert tm.get_cache_stats()["n_tiles"] == 0

    def test_unknown_source_raises(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        with pytest.raises(KeyError):
            tm.get_data("nope", BoundingBox(0.2, 0.8, 0.2, 0.8))


# ---------------------------------------------------------------------------
# Dateline handling
# ---------------------------------------------------------------------------


class TestDateline:
    def test_dateline_box_splits_and_merges(self, tmp_path):
        reader = _reader(tmp_path)
        tm = _manager(reader)
        box = BoundingBox(0.0, 2.0, 179.0, -179.0, wraps_dateline=True)
        merged = tm.get_data("_fake_tiling", box)
        # Two sub-requests -> two tile reads.
        assert reader.load_calls == 2
        # Longitudes from both sides of the antimeridian appear, sorted ascending.
        assert merged.lons.min() < 0 < merged.lons.max()


# ---------------------------------------------------------------------------
# Error path (catch -> empty/partial)
# ---------------------------------------------------------------------------


class TestReaderErrorPath:
    def test_reader_exception_returns_empty_tile(self, tmp_path):
        reader = _reader(tmp_path)
        reader.raise_on_load = True
        tm = _manager(reader)
        tile = tm.get_data("_fake_tiling", BoundingBox(0.2, 0.8, 0.2, 0.8))
        assert tile.data.size == 0  # empty -> partial coverage downstream
        assert tile.source == "_fake_tiling"
        assert tm.get_cache_stats()["errors"] == 1

    def test_error_tiles_are_not_cached(self, tmp_path):
        reader = _reader(tmp_path)
        reader.raise_on_load = True
        tm = _manager(reader)
        box = BoundingBox(0.2, 0.8, 0.2, 0.8)
        tm.get_data("_fake_tiling", box)
        tm.get_data("_fake_tiling", box)
        # Retried (not served from cache) so the reader was invoked twice.
        assert reader.load_calls == 2
        assert tm.get_cache_stats()["errors"] == 2
        assert tm.get_cache_stats()["n_tiles"] == 0

    def test_segment_continues_after_error(self, tmp_path):
        # An error on one footprint must not stop a later good read.
        reader = _reader(tmp_path)
        tm = _manager(reader)
        reader.raise_on_load = True
        tm.get_data("_fake_tiling", BoundingBox(0.2, 0.8, 0.2, 0.8))
        reader.raise_on_load = False
        good = tm.get_data("_fake_tiling", BoundingBox(0.2, 0.8, 2.2, 2.8))
        assert good.data.shape == (2, 2)


# ---------------------------------------------------------------------------
# merge_tiles (direct)
# ---------------------------------------------------------------------------


class TestMergeTiles:
    def test_single_tile_fast_path_returns_input(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        tile = _tile([0.5, 1.5], [0.5, 1.5], np.ones((2, 2), dtype=np.float32))
        assert tm.merge_tiles([tile]) is tile

    def test_merge_two_2d_tiles(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        left = _tile([0.5, 1.5], [0.5, 1.5], np.ones((2, 2)))
        right = _tile([0.5, 1.5], [2.5, 3.5], np.full((2, 2), 2.0))
        merged = tm.merge_tiles([left, right])
        assert merged.data.shape == (2, 4)
        np.testing.assert_array_equal(merged.lons, [0.5, 1.5, 2.5, 3.5])
        assert np.all(merged.data[:, :2] == 1.0)
        assert np.all(merged.data[:, 2:] == 2.0)

    def test_merge_three_dimensional_tiles(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        left = _tile([0.5, 1.5], [0.5, 1.5], np.ones((2, 2, 2)))  # (n_var, n_lat, n_lon)
        right = _tile([0.5, 1.5], [2.5, 3.5], np.full((2, 2, 2), 2.0))
        merged = tm.merge_tiles([left, right])
        assert merged.data.shape == (2, 2, 4)
        assert np.all(merged.data[:, :, :2] == 1.0)
        assert np.all(merged.data[:, :, 2:] == 2.0)

    def test_merge_fills_gaps_with_nan(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        # Tiles occupy disjoint lat AND lon blocks, leaving two corners uncovered.
        a = _tile([0.5], [0.5, 1.5], np.ones((1, 2)))
        b = _tile([1.5], [2.5, 3.5], np.full((1, 2), 2.0))
        merged = tm.merge_tiles([a, b])
        assert merged.data.shape == (2, 4)
        assert np.isnan(merged.data[0, 2])  # row of A, cols of B -> gap
        assert np.isnan(merged.data[1, 0])  # row of B, cols of A -> gap

    def test_merge_all_empty_returns_empty(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        empty = GridTile(
            data=np.empty((0, 0), dtype=np.float32),
            lats=np.empty(0),
            lons=np.empty(0),
            bounds=BoundingBox(0.0, 2.0, 0.0, 2.0),
            source="_fake_tiling",
        )
        merged = tm.merge_tiles([empty, empty])
        assert merged.data.size == 0
        assert merged.source == "_fake_tiling"

    def test_merge_requires_at_least_one_tile(self, tmp_path):
        tm = _manager(_reader(tmp_path))
        with pytest.raises(ValueError, match="at least one tile"):
            tm.merge_tiles([])


# ---------------------------------------------------------------------------
# build_tile_manager + gather_footprint_tiles
# ---------------------------------------------------------------------------


class TestBuildTileManager:
    def test_builds_readers_for_mode(self, tmp_path):
        active = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        paths = {key: tmp_path / f"{key}.nc" for key in active}
        tm = build_tile_manager(OperationalMode.CAM, paths)
        assert set(tm.sources) == set(active)
        assert tm.active_mode is OperationalMode.CAM

    def test_missing_file_path_raises(self, tmp_path):
        active = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        paths = {key: tmp_path / f"{key}.nc" for key in active}
        paths.pop(next(iter(paths)))  # drop one required source
        with pytest.raises(KeyError):
            build_tile_manager(OperationalMode.CAM, paths)


@dataclass
class _FakeFootprint:
    """Minimal stand-in exposing the ``bbox`` attribute gather expects."""

    bbox: BoundingBox


class TestGatherFootprintTiles:
    def test_gathers_one_tile_per_footprint_per_source(self, tmp_path):
        reader = _reader(tmp_path)
        tm = _manager(reader)
        footprints = [
            _FakeFootprint(BoundingBox(0.2, 0.8, 0.2, 0.8)),
            _FakeFootprint(BoundingBox(0.2, 0.8, 0.3, 0.9)),  # same tile -> cache hit
        ]
        tiles_by_source = gather_footprint_tiles(tm, footprints)
        assert set(tiles_by_source) == {"_fake_tiling"}
        assert len(tiles_by_source["_fake_tiling"]) == 2
        # Both footprints hit the same tile, so only one physical read occurred.
        assert reader.load_calls == 1
