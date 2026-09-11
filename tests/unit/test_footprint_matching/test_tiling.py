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

import threading
import time
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

    def test_wrapped_box_in_360_representation_does_not_read_near_global(self, tmp_path):
        # Regression: bounding_box_from_points reports a wrapped box in the [0, 360)
        # representation, so a narrow 179/-179 crossing has lon_max == 181 (not -179).
        # The east sub-request must normalize that max back into [-180, 180]; otherwise
        # it selects every longitude tile from -180 through 181 -- a ~180-tile
        # near-global read -- instead of the two-degree sliver east of the antimeridian.
        reader = _reader(tmp_path)  # 2 deg tiles
        tm = _manager(reader)
        box = BoundingBox(0.0, 2.0, 179.0, 181.0, wraps_dateline=True)
        tm.get_data("_fake_tiling", box)
        # West sliver [179, 180] + east sliver [-180, -179] -> exactly two tile reads.
        assert reader.load_calls == 2


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


# ---------------------------------------------------------------------------
# Look-ahead prefetch
# ---------------------------------------------------------------------------


class _CountingPrefetchReader(GriddedDataReader):
    """Reader that counts loads (thread-safely) and records the loading thread name.

    Used by the prefetch tests to assert *which* thread performed each load and that a
    tile is loaded exactly once. Loads are trivially fast; determinism comes from the
    tests always calling ``shutdown(wait=True)`` before inspecting state.
    """

    READER_KEY = "_prefetch_reader"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    TILE_SIZE_DEG = 2.0
    VARIABLES = (
        VariableSpec(name="pf_var", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
    )

    def __init__(self, file_path) -> None:
        super().__init__(file_path)
        self._lock = threading.Lock()
        self.loaded_keys: list[tuple[float, float]] = []
        self.load_threads: set[str] = set()
        self.raise_on_load = False

    def _load_spatial_region(self, bbox: BoundingBox):
        with self._lock:
            self.loaded_keys.append((round(bbox.lat_min, 3), round(bbox.lon_min, 3)))
            self.load_threads.add(threading.current_thread().name)
        if self.raise_on_load:
            raise RuntimeError("simulated reader failure")
        lats = np.array([bbox.lat_min + 0.5, bbox.lat_min + 1.5], dtype=np.float64)
        lons = np.array([bbox.lon_min + 0.5, bbox.lon_min + 1.5], dtype=np.float64)
        return np.ones((2, 2), dtype=np.float32), lats, lons

    @property
    def load_calls(self) -> int:
        with self._lock:
            return len(self.loaded_keys)


def _prefetch_manager(tmp_path, lookahead: int, workers: int = 1) -> tuple[TileManager, _CountingPrefetchReader]:
    reader = _CountingPrefetchReader(tmp_path / "pf.nc")
    tm = TileManager(
        {reader.READER_KEY: reader},
        OperationalMode.CAM,
        prefetch_lookahead=lookahead,
        prefetch_workers=workers,
    )
    return tm, reader


class _ConcurrencyMonitor:
    """Tracks the peak number of loads running inside ``_load_spatial_region`` at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active -= 1


class _ConcurrencySpyReader(GriddedDataReader):
    """Reader whose loads record their peak concurrency in a shared monitor.

    Every instance reports to the *same* monitor, so the recorded peak captures overlap
    across different sources -- exactly what the global reader lock must prevent, since
    the real reader backends (netCDF4 / h5py over libhdf5) segfault under concurrent
    access. Each load sleeps briefly to widen the window so a genuine overlap is caught.
    """

    READER_KEY = "_concurrency_spy"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    TILE_SIZE_DEG = 2.0
    VARIABLES = (
        VariableSpec(name="spy_var", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
    )

    def __init__(self, file_path, monitor: _ConcurrencyMonitor) -> None:
        super().__init__(file_path)
        self._monitor = monitor

    def _load_spatial_region(self, bbox: BoundingBox):
        self._monitor.enter()
        try:
            time.sleep(0.01)  # widen the window so any real concurrency would overlap here
            lats = np.array([bbox.lat_min + 0.5, bbox.lat_min + 1.5], dtype=np.float64)
            lons = np.array([bbox.lon_min + 0.5, bbox.lon_min + 1.5], dtype=np.float64)
            return np.ones((2, 2), dtype=np.float32), lats, lons
        finally:
            self._monitor.leave()


class TestPrefetch:
    def test_enabled_by_default(self, tmp_path):
        # Prefetch is on by default (lookahead 1) so reader I/O overlaps compute without
        # the caller opting in; see TileManager / build_tile_manager docstrings.
        reader = _reader(tmp_path)
        tm = _manager(reader)
        assert tm.prefetch_enabled is True
        assert tm.prefetch_lookahead == 1

    def test_explicit_disable_is_a_noop(self, tmp_path):
        reader = _reader(tmp_path)
        tm = TileManager({reader.READER_KEY: reader}, OperationalMode.CAM, prefetch_lookahead=0)
        assert tm.prefetch_enabled is False
        assert tm.prefetch_lookahead == 0
        # prefetch is a no-op when disabled: nothing is loaded.
        tm.prefetch([BoundingBox(0.2, 0.8, 0.2, 0.8)])
        assert reader.load_calls == 0

    def test_prefetch_populates_cache_from_background_thread(self, tmp_path):
        tm, reader = _prefetch_manager(tmp_path, lookahead=2)
        assert tm.prefetch_enabled is True
        box = BoundingBox(0.2, 0.8, 0.2, 0.8)  # one tile
        tm.prefetch([box])
        tm.shutdown(wait=True)  # deterministic: all background loads have finished

        # The tile was loaded once, on a background prefetch thread (not the main thread).
        assert reader.load_calls == 1
        assert reader.load_threads
        assert all("prefetch" in name for name in reader.load_threads)
        assert tm.get_cache_stats()["prefetched"] == 1

        # A synchronous request for the same box is now a pure cache hit: no extra load.
        tile = tm.get_data(reader.READER_KEY, box)
        assert reader.load_calls == 1
        assert tile.data.shape == (2, 2)
        assert tm.get_cache_stats()["hits"] == 1

    def test_prefetch_dedupes_repeated_and_cached_keys(self, tmp_path):
        tm, reader = _prefetch_manager(tmp_path, lookahead=2)
        box = BoundingBox(0.2, 0.8, 0.2, 0.8)
        # Submit the same box twice before the worker drains: the second submit must be
        # deduped (in-flight or cached), so the tile is loaded exactly once.
        tm.prefetch([box])
        tm.prefetch([box])
        tm.shutdown(wait=True)
        assert reader.load_calls == 1
        assert tm.get_cache_stats()["prefetched"] == 1

    def test_prefetch_result_matches_synchronous_load(self, tmp_path):
        # A tile served from the prefetch cache is identical to one loaded synchronously.
        box = BoundingBox(0.2, 0.8, 0.5, 2.5)  # spans two lon tiles
        tm_pf, reader_pf = _prefetch_manager(tmp_path, lookahead=3)
        tm_pf.prefetch([box])
        tm_pf.shutdown(wait=True)
        prefetched = tm_pf.get_data(reader_pf.READER_KEY, box)

        sync_reader = _CountingPrefetchReader(tmp_path / "sync.nc")
        tm_sync = TileManager({sync_reader.READER_KEY: sync_reader}, OperationalMode.CAM)
        synchronous = tm_sync.get_data(sync_reader.READER_KEY, box)

        np.testing.assert_array_equal(prefetched.data, synchronous.data)
        np.testing.assert_array_equal(prefetched.lats, synchronous.lats)
        np.testing.assert_array_equal(prefetched.lons, synchronous.lons)

    def test_prefetch_reader_failure_is_swallowed(self, tmp_path):
        tm, reader = _prefetch_manager(tmp_path, lookahead=2)
        reader.raise_on_load = True
        box = BoundingBox(0.2, 0.8, 0.2, 0.8)
        # A failing prefetch must not crash the worker/pool; the tile is simply not
        # cached, so a later synchronous get re-attempts and gets an empty (NaN) tile.
        tm.prefetch([box])
        tm.shutdown(wait=True)
        reader.raise_on_load = False
        tile = tm.get_data(reader.READER_KEY, box)  # sync path recovers
        assert tile.data.shape == (2, 2)
        assert not np.isnan(tile.data).all()  # reader now succeeds

    def test_loads_never_overlap_across_sources(self, tmp_path):
        # Regression guard for the segfault fix: the reader backends (netCDF4 / h5py
        # over libhdf5) are not thread-safe, so the single global reader lock must
        # serialize *every* load -- including loads of different sources dispatched to
        # different prefetch workers. Three sources, a 3-worker pool, and boxes hitting
        # all three: the peak observed in-load concurrency must stay 1. (With a
        # per-source lock these would run three-wide and the peak would be 3.)
        monitor = _ConcurrencyMonitor()
        readers = {f"spy_{name}": _ConcurrencySpyReader(tmp_path / f"{name}.nc", monitor) for name in ("a", "b", "c")}
        tm = TileManager(readers, OperationalMode.CAM, prefetch_lookahead=1, prefetch_workers=3)
        try:
            tm.prefetch([BoundingBox(0.2, 0.8, 0.2, 0.8)])  # one tile per source -> three queued loads
            tm.shutdown(wait=True)  # deterministic: all background loads have finished
        finally:
            tm.shutdown()
        assert monitor.peak == 1

    def test_shutdown_is_idempotent(self, tmp_path):
        tm, _ = _prefetch_manager(tmp_path, lookahead=1)
        tm.shutdown()
        tm.shutdown()  # second call is a harmless no-op
        assert tm.prefetch_enabled is False

    def test_context_manager_shuts_down(self, tmp_path):
        reader = _CountingPrefetchReader(tmp_path / "ctx.nc")
        with TileManager({reader.READER_KEY: reader}, OperationalMode.CAM, prefetch_lookahead=2) as tm:
            assert tm.prefetch_enabled is True
        assert tm.prefetch_enabled is False


class TestMergeCache:
    """Merged multi-tile results are reused (same object) for a repeated key-set."""

    # A box spanning two 2° tiles in latitude (tiles start at -90°, so a boundary sits at
    # 2°); lon stays within one tile -> exactly two covering tile keys -> a real merge.
    _MULTI = BoundingBox(1.0, 3.0, 0.5, 0.5)

    def test_repeated_multitile_box_returns_same_object(self, tmp_path):
        reader = _reader(tmp_path)
        mgr = _manager(reader)
        assert len(mgr.resolve_tile_keys(reader.READER_KEY, self._MULTI)) == 2  # genuinely a merge

        first = mgr.get_data(reader.READER_KEY, self._MULTI)
        second = mgr.get_data(reader.READER_KEY, self._MULTI)
        assert second is first  # served from the merge cache
        assert mgr.get_cache_stats()["merge_hits"] == 1

    def test_cached_merge_is_byte_identical_to_uncached(self, tmp_path):
        cached = _manager(_reader(tmp_path))
        uncached = TileManager({"_fake_tiling": _reader(tmp_path)}, OperationalMode.CAM, merge_cache_size=0)
        c = cached.get_data("_fake_tiling", self._MULTI)
        u0 = uncached.get_data("_fake_tiling", self._MULTI)
        u1 = uncached.get_data("_fake_tiling", self._MULTI)
        assert u1 is not u0  # cache disabled -> fresh object each call
        assert uncached.get_cache_stats()["merge_hits"] == 0
        # Same coordinates and data whether cached or not.
        np.testing.assert_array_equal(c.lats, u0.lats)
        np.testing.assert_array_equal(c.lons, u0.lons)
        np.testing.assert_array_equal(np.asarray(c.data), np.asarray(u0.data))

    def test_distinct_keysets_do_not_collide(self, tmp_path):
        reader = _reader(tmp_path)
        mgr = _manager(reader)
        other = BoundingBox(5.0, 7.0, 0.5, 0.5)  # a different pair of tiles
        a = mgr.get_data(reader.READER_KEY, self._MULTI)
        b = mgr.get_data(reader.READER_KEY, other)
        assert a is not b
        assert not np.array_equal(a.lats, b.lats)

    def test_failed_component_merge_is_not_cached(self, tmp_path):
        # A merge that folded in a failed (empty) component must not be cached, so the
        # region recovers once the reader works again.
        reader = _reader(tmp_path)
        mgr = _manager(reader)
        reader.raise_on_load = True
        first = mgr.get_data(reader.READER_KEY, self._MULTI)
        assert first.data.size == 0  # partial-coverage empty tile
        assert mgr.get_cache_stats()["merge_hits"] == 0  # not served/!stored from cache

        reader.raise_on_load = False
        recovered = mgr.get_data(reader.READER_KEY, self._MULTI)
        assert recovered.data.size > 0  # recovered rather than serving a stale empty merge

    def test_clear_cache_drops_merged_results(self, tmp_path):
        reader = _reader(tmp_path)
        mgr = _manager(reader)
        first = mgr.get_data(reader.READER_KEY, self._MULTI)
        mgr.clear_cache()
        second = mgr.get_data(reader.READER_KEY, self._MULTI)
        assert second is not first  # merge cache cleared alongside the tile cache
