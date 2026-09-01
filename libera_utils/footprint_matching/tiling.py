"""Tiled data manager and LRU tile cache for footprint matching.

This module implements the *reader → cache* seam described in §2.8.1.3 ("Tiled
Data Manager") of the Footprint Matching and Scene ID design doc
(``instructions/documentation/Footprint Matching and Scene ID PDF``). It is the
component that sits **between** the plugin readers
(:class:`~libera_utils.footprint_matching.readers.base.GriddedDataReader`) and the
PSF aggregation engine.

Why a cache at all
------------------
The orchestrator processes ~1,200 radiometer footprints per hour, each of which
may touch more than a thousand fine-resolution ancillary pixels, drawn from input
datasets that range from hundreds of megabytes to tens of gigabytes. Naively
re-reading a source file for every footprint would be ruinously slow. Because
footprints are processed in along-track order, spatially adjacent footprints tend
to need the *same* tiles, so an LRU cache turns almost every repeat request into a
memory hit. See https://realpython.com/lru-cache-python/ for the LRU pattern and
https://docs.python.org/3/library/collections.html#collections.OrderedDict for the
``OrderedDict`` we build it on.

What the TileManager does
-------------------------
1. Converts a footprint's geographic :class:`BoundingBox` into the set of tile
   keys that cover it, using *that reader's* tile size (per-reader tiling).
2. Serves each tile from the LRU cache, calling the reader only on a miss.
3. Merges the covering tiles (and, for antimeridian-crossing boxes, two
   sub-requests) into a single contiguous :class:`GridTile`.
4. Bounds memory with a **byte budget**: when inserting a tile would push the
   cache over ``max_cache_bytes``, the least-recently-used tiles are evicted until
   it fits.

Error handling (catch → partial coverage)
-----------------------------------------
If a reader raises while loading a tile, or a source is otherwise unable to supply
data for a region, the TileManager does **not** abort the segment. It logs the
failure, counts it, and substitutes an *empty* tile (see :func:`_empty_tile`).
Because the downstream aggregation engine treats absent cells as zero PSF weight
and excludes NaN, an empty tile simply drives that footprint's sampled weight down
so it is scored as partial coverage — or discarded by the 75 %/95 % coverage rule
— rather than silently substituting a different source. This keeps the *segment*
running while the affected *footprint* is honestly marked, honoring the mode
contract's "no silent fallback" rule. Empty/error tiles are deliberately **not**
cached, so a transient failure can recover on the next segment.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import BoundingBox, GridTile, TileKey

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from libera_utils.footprint_matching.readers.base import GriddedDataReader
    from libera_utils.footprint_matching.types import OperationalMode

# Module-level logger. The public entry points accept an optional injected logger
# (dependency injection, per the project logging guidance); when none is supplied
# we fall back to this module logger so library code always has somewhere to write.
logger = logging.getLogger(__name__)

# Default cache byte budget: 2 GiB. Ancillary inputs can total tens of GB, but the
# working set for a one-hour along-track segment is far smaller; 2 GiB comfortably
# holds the handful of tiles that adjacent footprints reuse while leaving headroom
# for the rest of the process. Tunable via ``build_tile_manager`` / the config once
# real files are profiled. TODO[LIBSDC-807]: expose as a config key.
DEFAULT_MAX_CACHE_BYTES: int = 2 * 1024 * 1024 * 1024

# Tiny epsilon (degrees) subtracted from a bounding box's max edge before flooring
# to a tile index. Without it, a box whose max edge lands *exactly* on a tile
# boundary would pull in the next (empty) tile — harmless but wasteful. Far smaller
# than any real cell size, so it never drops a tile that actually holds data.
_EDGE_EPSILON_DEG: float = 1e-9


def _empty_tile(source: str, bounds: BoundingBox, n_var: int, timestamp_source: str | None) -> GridTile:
    """Build a zero-pixel :class:`GridTile` used as the partial-coverage signal.

    Returned when a reader fails to load a region (exception) or genuinely has no
    data there. The coordinate arrays are length zero, so the tile contributes no
    sampled PSF weight downstream and the footprint is scored as partial / low
    coverage rather than aborting the segment (see the module docstring).

    Parameters
    ----------
    source : str
        Reader registry key this tile stands in for.
    bounds : BoundingBox
        Geographic region the (missing) tile was meant to cover. Preserved so the
        merged result still reports where the gap is.
    n_var : int
        Number of variables the owning reader supplies. ``1`` yields a 2-D
        ``(0, 0)`` data array (single-variable contract); ``>1`` yields a 3-D
        ``(n_var, 0, 0)`` array so a merge with real multi-variable tiles is
        shape-consistent.
    timestamp_source : str or None
        ``'radiometer'`` / ``'camera'`` for cloud products, else ``None``.

    Returns
    -------
    GridTile
        Empty tile with valid ``source``/``bounds`` metadata.
    """
    # Match the dimensionality real tiles from this reader would have so that
    # merge_tiles can concatenate without special-casing width.
    data = np.empty((0, 0), dtype=np.float32) if n_var == 1 else np.empty((n_var, 0, 0), dtype=np.float32)
    return GridTile(
        data=data,
        lats=np.empty(0, dtype=np.float64),
        lons=np.empty(0, dtype=np.float64),
        bounds=bounds,
        source=source,
        timestamp_source=timestamp_source,
    )


def _union_bounds(tiles: list[GridTile]) -> BoundingBox:
    """Return the smallest BoundingBox enclosing every tile's bounds.

    Used to describe the geographic extent of a merged (multi-tile) result. Flags
    (``wraps_dateline``, ``is_polar``, ``truncated``) are OR-ed across inputs so the
    merged tile does not lose a special-case marker set on any contributing tile.
    """
    lat_min = min(t.bounds.lat_min for t in tiles)
    lat_max = max(t.bounds.lat_max for t in tiles)
    lon_min = min(t.bounds.lon_min for t in tiles)
    lon_max = max(t.bounds.lon_max for t in tiles)
    wraps = any(t.bounds.wraps_dateline for t in tiles)
    polar = any(t.bounds.is_polar for t in tiles)
    truncated = any(t.bounds.truncated for t in tiles)
    return BoundingBox(lat_min, lat_max, lon_min, lon_max, wraps, polar, truncated)


class _LRUTileCache:
    """Least-recently-used tile cache bounded by a total byte budget.

    Wraps an :class:`~collections.OrderedDict` whose insertion order tracks recency:
    the most-recently-used key is kept at the end and the least-recently-used at the
    front. Sizing is by *bytes* (summed :attr:`GridTile.nbytes`) rather than a tile
    count, because tiles vary enormously in memory footprint — a coarse 25 km wind
    tile is kilobytes while a fine imager tile can be tens of megabytes — so a fixed
    tile count would either waste memory or evict too aggressively.

    This class is intentionally private to the module; callers interact with it only
    through :class:`TileManager`.

    Parameters
    ----------
    max_cache_bytes : int
        Upper bound on the summed byte size of cached tiles. After each insertion,
        least-recently-used tiles are evicted until the total fits.
    """

    def __init__(self, max_cache_bytes: int) -> None:
        if max_cache_bytes <= 0:
            raise ValueError(f"max_cache_bytes must be positive, got {max_cache_bytes!r}")
        self._max_cache_bytes = int(max_cache_bytes)
        # Ordered by recency: front = least recently used, back = most recent.
        self._tiles: OrderedDict[TileKey, GridTile] = OrderedDict()
        self._current_bytes = 0
        # Diagnostics returned by TileManager.get_cache_stats().
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._errors = 0

    def get(self, key: TileKey) -> GridTile | None:
        """Return the cached tile for ``key`` (marking it most-recently-used), or None."""
        tile = self._tiles.get(key)
        if tile is None:
            self._misses += 1
            return None
        # Move to the back to mark it most-recently-used.
        self._tiles.move_to_end(key)
        self._hits += 1
        return tile

    def put(self, key: TileKey, tile: GridTile) -> None:
        """Insert ``tile`` under ``key`` and evict LRU tiles until within budget.

        A single tile larger than the whole budget is still stored for the duration
        of the caller's request (it cannot be served otherwise), then evicted on the
        next insertion because it alone exceeds the budget — an intentional, bounded
        worst case rather than an infinite eviction loop.
        """
        if key in self._tiles:
            # Replacing an existing key: subtract the old size before re-accounting.
            self._current_bytes -= self._tiles[key].nbytes
        self._tiles[key] = tile
        self._tiles.move_to_end(key)
        self._current_bytes += tile.nbytes
        self._evict_to_budget(keep=key)

    def _evict_to_budget(self, keep: TileKey) -> None:
        """Evict least-recently-used tiles until the total fits the byte budget.

        ``keep`` is never evicted during this call — it is the tile the caller just
        inserted and is about to use, so evicting it would defeat the request.
        """
        while self._current_bytes > self._max_cache_bytes and len(self._tiles) > 1:
            lru_key, lru_tile = next(iter(self._tiles.items()))
            if lru_key == keep:
                # The just-inserted tile is (currently) the LRU only when it is the
                # sole oversized entry; stop rather than evict what we need now.
                break
            self._tiles.popitem(last=False)
            self._current_bytes -= lru_tile.nbytes
            self._evictions += 1

    def clear(self) -> None:
        """Drop all cached tiles and reset the byte total (counters are preserved)."""
        self._tiles.clear()
        self._current_bytes = 0

    def record_error(self) -> None:
        """Increment the reader-error counter (a load raised and was caught)."""
        self._errors += 1

    def stats(self) -> dict[str, Any]:
        """Return a diagnostics snapshot for TileManager.get_cache_stats()."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "current_bytes": self._current_bytes,
            "n_tiles": len(self._tiles),
            "evictions": self._evictions,
            "errors": self._errors,
        }


class TileManager:
    """Central data-access layer: resolves, caches, and merges gridded tiles.

    Every request from the orchestrator for a footprint's ancillary data passes
    through the TileManager. It is constructed with **only the readers active for
    the current operational mode** (readers for unavailable sources are simply never
    instantiated), so requesting a source that is not active for the mode is a hard
    error rather than a silent no-op.

    Parameters
    ----------
    readers : dict[str, GriddedDataReader]
        Mapping of reader registry key to an *instantiated* reader, already filtered
        to the active mode (see :func:`build_tile_manager`).
    active_mode : OperationalMode
        The operational mode this manager serves. Stored for provenance / diagnostics
        and to make the mode explicit at the data-access boundary.
    max_cache_bytes : int, optional
        Byte budget for the LRU cache. Defaults to :data:`DEFAULT_MAX_CACHE_BYTES`.
    logger : logging.Logger, optional
        Injected logger; defaults to this module's logger. Reader-load failures are
        reported here at ``WARNING`` level.
    """

    def __init__(
        self,
        readers: dict[str, GriddedDataReader],
        active_mode: OperationalMode,
        max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
        logger: logging.Logger | None = None,
    ) -> None:
        self._readers = dict(readers)
        self._active_mode = active_mode
        self._cache = _LRUTileCache(max_cache_bytes)
        self._logger = logger if logger is not None else globals()["logger"]

    @property
    def active_mode(self) -> OperationalMode:
        """The operational mode this manager was initialized for."""
        return self._active_mode

    @property
    def sources(self) -> list[str]:
        """Sorted registry keys of the readers active in this manager."""
        return sorted(self._readers.keys())

    def get_data(self, source: str, bbox: BoundingBox) -> GridTile:
        """Return a single merged tile of ``source`` data covering ``bbox``.

        This is the primary access method. It resolves the bounding box into tile
        keys, serves each from the cache (loading on a miss), and merges the results
        into one contiguous :class:`GridTile`. Antimeridian-crossing boxes are split
        into two sub-requests and stitched.

        Parameters
        ----------
        source : str
            Registry key of a reader active in this mode.
        bbox : BoundingBox
            Geographic region to cover (typically a footprint's PSF bounding box).

        Returns
        -------
        GridTile
            Merged data covering ``bbox``. Regions the reader could not supply are
            NaN / absent, so the footprint is scored as partial coverage downstream.

        Raises
        ------
        KeyError
            If ``source`` is not one of the readers active for this mode. This is a
            mode-configuration error, distinct from a transient read failure (which
            is caught and turned into an empty tile).
        """
        if source not in self._readers:
            raise KeyError(
                f"Source {source!r} is not active for mode {self._active_mode.value!r}. Active sources: {self.sources}"
            )

        # Antimeridian crossing: split into two boxes on either side of ±180° and
        # merge the two results (see BoundingBox.wraps_dateline).
        if bbox.wraps_dateline:
            # A wrapped box is in the [0, 360) representation with lon_min < 180 < lon_max
            # (see bounding_box_from_points). The west half [lon_min, 180] is already in
            # range, but the east max must be normalized back into [-180, 180] by
            # subtracting 360 -- otherwise the non-wrapping east sub-request would select
            # every longitude tile from -180 through lon_max (a near-global read) instead
            # of just the -180..(lon_max-360) sliver east of the antimeridian.
            east_max = bbox.lon_max - 360.0
            west = BoundingBox(bbox.lat_min, bbox.lat_max, bbox.lon_min, 180.0, False, bbox.is_polar, bbox.truncated)
            east = BoundingBox(bbox.lat_min, bbox.lat_max, -180.0, east_max, False, bbox.is_polar, bbox.truncated)
            return self.merge_tiles([self.get_data(source, west), self.get_data(source, east)])

        keys = self.resolve_tile_keys(source, bbox)
        tiles = [self._get_tile(key) for key in keys]
        return self.merge_tiles(tiles)

    def _get_tile(self, key: TileKey) -> GridTile:
        """Serve one tile from cache, loading it via the reader on a miss.

        Reader exceptions are caught here and converted to an empty tile so a single
        bad tile never aborts the whole segment. Empty/error tiles are not cached, so
        a transient failure can recover next time the tile is requested.
        """
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        reader = self._readers[key.source]
        try:
            tile = reader.load_tile(key)
        except Exception:  # noqa: BLE001 - deliberately broad: any reader failure → partial coverage.
            # Log with the key so the failing region is identifiable, but keep going.
            self._logger.warning("Reader %r failed to load tile %r; treating as partial coverage.", key.source, key)
            self._cache.record_error()
            n_var = len(reader.VARIABLES)
            timestamp_source = reader.get_timestamp_source() if hasattr(reader, "get_timestamp_source") else None
            bounds = reader._tile_key_to_bbox(key, reader.TILE_SIZE_DEG)  # noqa: SLF001 - same module family
            return _empty_tile(key.source, bounds, n_var, timestamp_source)

        self._cache.put(key, tile)
        return tile

    def resolve_tile_keys(self, source: str, bbox: BoundingBox) -> list[TileKey]:
        """Convert a bounding box into the tile keys covering it for ``source``.

        Uses the reader's own :attr:`~GriddedDataReader.TILE_SIZE_DEG` so per-reader
        tile grids resolve correctly. Typically returns one or two keys at nadir and
        three or more for high-elevation-angle footprints whose boxes span more tiles.
        Polar boxes span the full longitude range (set upstream in geometry), which
        yields a full latitude-row of tiles near the pole — expected and acceptable.

        Parameters
        ----------
        source : str
            Registry key of an active reader.
        bbox : BoundingBox
            Geographic region to cover. Must not be dateline-wrapping (``get_data``
            splits those before calling this).

        Returns
        -------
        list[TileKey]
            Tile keys covering ``bbox``, in row-major (lat then lon) order.

        Raises
        ------
        KeyError
            If ``source`` is not active for this mode.
        """
        if source not in self._readers:
            raise KeyError(f"Source {source!r} is not active for mode {self._active_mode.value!r}.")
        size = self._readers[source].TILE_SIZE_DEG

        # Number of whole tiles spanning each global axis (origin anchored at
        # -90° lat / -180° lon). ceil so a size that does not divide evenly still
        # covers the last partial strip.
        n_lat = int(np.ceil(180.0 / size))
        n_lon = int(np.ceil(360.0 / size))

        # Index of the tile containing an edge = floor((edge - origin) / size). The
        # max edges get a tiny epsilon subtracted so a box ending exactly on a tile
        # boundary does not pull in the next (empty) tile.
        lat_start = int(np.floor((bbox.lat_min + 90.0) / size))
        lat_stop = int(np.floor((bbox.lat_max + 90.0 - _EDGE_EPSILON_DEG) / size))
        lon_start = int(np.floor((bbox.lon_min + 180.0) / size))
        lon_stop = int(np.floor((bbox.lon_max + 180.0 - _EDGE_EPSILON_DEG) / size))

        # Clamp to the valid global index range so a box grazing a pole or the
        # ±180° edge never produces an out-of-range key.
        lat_start = max(0, min(lat_start, n_lat - 1))
        lat_stop = max(0, min(lat_stop, n_lat - 1))
        lon_start = max(0, min(lon_start, n_lon - 1))
        lon_stop = max(0, min(lon_stop, n_lon - 1))

        # A zero-height/zero-width box (e.g. a single-pixel camera footprint) whose min
        # edge sits exactly on a tile boundary would otherwise leave stop one below
        # start once the max-edge epsilon is applied, yielding an empty key list and a
        # `merge_tiles requires at least one tile` error downstream. Pin stop to at
        # least start so the tile containing the min edge is always covered.
        lat_stop = max(lat_stop, lat_start)
        lon_stop = max(lon_stop, lon_start)

        return [
            TileKey(source, lat_idx, lon_idx)
            for lat_idx in range(lat_start, lat_stop + 1)
            for lon_idx in range(lon_start, lon_stop + 1)
        ]

    def merge_tiles(self, tiles: list[GridTile]) -> GridTile:
        """Stitch adjacent tiles into one contiguous :class:`GridTile`.

        Handles both single-variable 2-D ``(n_lat, n_lon)`` and multi-variable 3-D
        ``(n_var, n_lat, n_lon)`` tiles. All tiles are assumed to come from the same
        source (same underlying grid), so their cell centres align; the union lat/lon
        axes are formed from the tiles' coordinates and each tile's data is placed
        into the corresponding block. Cells not covered by any tile are ``NaN``.

        Parameters
        ----------
        tiles : list[GridTile]
            Tiles to merge, all from one source.

        Returns
        -------
        GridTile
            Single merged tile spanning the union of the inputs. If every input is
            empty (all readers failed / no data), an empty tile spanning the union of
            the input bounds is returned (partial-coverage signal).
        """
        if not tiles:
            raise ValueError("merge_tiles requires at least one tile.")

        # Drop empties (0-length coord arrays) — they carry no data to place, only a
        # region marker. Keep track so an all-empty request still returns an empty
        # tile with sensible bounds.
        non_empty = [t for t in tiles if t.lats.size > 0 and t.lons.size > 0 and t.data.size > 0]
        if not non_empty:
            first = tiles[0]
            n_var = 1 if first.data.ndim <= 2 else first.data.shape[0]
            return _empty_tile(first.source, _union_bounds(tiles), n_var, first.timestamp_source)

        # Single real tile → nothing to stitch. (Common at nadir.)
        if len(non_empty) == 1 and len(tiles) == 1:
            return non_empty[0]

        source = non_empty[0].source
        timestamp_source = non_empty[0].timestamp_source
        is_3d = non_empty[0].data.ndim == 3
        n_var = non_empty[0].data.shape[0] if is_3d else 1

        # Build the union coordinate axes. Rounding guards against float noise so
        # cell centres shared between adjacent tiles collapse to one axis entry.
        lat_axis = np.unique(np.round(np.concatenate([t.lats for t in non_empty]), 9))
        lon_axis = np.unique(np.round(np.concatenate([t.lons for t in non_empty]), 9))

        # Allocate the merged grid filled with NaN (uncovered cells stay NaN).
        shape = (n_var, lat_axis.size, lon_axis.size) if is_3d else (lat_axis.size, lon_axis.size)
        merged = np.full(shape, np.nan, dtype=np.float64)

        for tile in non_empty:
            # Map each tile coordinate to its position in the union axes.
            lat_pos = np.searchsorted(lat_axis, np.round(tile.lats, 9))
            lon_pos = np.searchsorted(lon_axis, np.round(tile.lons, 9))
            # np.ix_ builds the open mesh that scatters this tile's block into place.
            if is_3d:
                merged[:, lat_pos[:, None], lon_pos[None, :]] = tile.data
            else:
                merged[np.ix_(lat_pos, lon_pos)] = tile.data

        return GridTile(
            data=merged,
            lats=lat_axis,
            lons=lon_axis,
            bounds=_union_bounds(non_empty),
            source=source,
            timestamp_source=timestamp_source,
        )

    def evict_lru(self) -> None:
        """Evict the single least-recently-used tile (diagnostic / manual use).

        Normal operation evicts automatically inside the cache on insertion; this is
        exposed for tests and explicit memory management.
        """
        if self._cache._tiles:  # noqa: SLF001 - same-module cooperative access
            _, tile = self._cache._tiles.popitem(last=False)  # noqa: SLF001
            self._cache._current_bytes -= tile.nbytes  # noqa: SLF001
            self._cache._evictions += 1  # noqa: SLF001

    def clear_cache(self) -> None:
        """Remove all tiles from the cache (counters are preserved)."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache diagnostics: hits, misses, current_bytes, n_tiles, evictions, errors."""
        return self._cache.stats()


def build_tile_manager(
    mode: OperationalMode,
    source_file_paths: dict[str, Path],
    max_cache_bytes: int = DEFAULT_MAX_CACHE_BYTES,
    logger: logging.Logger | None = None,
) -> TileManager:
    """Construct a :class:`TileManager` with the readers active for ``mode``.

    Selects the active reader classes via
    :meth:`ReaderRegistry.get_readers_for_mode`, instantiates each with the file path
    supplied in ``source_file_paths``, and wraps them in a TileManager. This is the
    single construction helper the orchestrator / product code calls so the reader
    wiring lives in one place.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode whose active readers should be built.
    source_file_paths : dict[str, Path]
        Mapping of reader registry key to the on-disk file for that source. Must
        contain an entry for every reader active in ``mode`` (in development we
        assume all ancillary data is available, per the design doc).
    max_cache_bytes : int, optional
        Byte budget for the LRU cache. Defaults to :data:`DEFAULT_MAX_CACHE_BYTES`.
    logger : logging.Logger, optional
        Injected logger passed through to the TileManager.

    Returns
    -------
    TileManager
        A manager ready to serve tiles for ``mode``.

    Raises
    ------
    KeyError
        If ``source_file_paths`` is missing a file for any active reader.
    """
    reader_classes = ReaderRegistry.get_readers_for_mode(mode)
    readers: dict[str, GriddedDataReader] = {}
    for key, reader_cls in reader_classes.items():
        if key not in source_file_paths:
            raise KeyError(
                f"No file path supplied for active source {key!r} (mode {mode.value!r}). "
                f"Required sources: {sorted(reader_classes)}"
            )
        readers[key] = reader_cls(source_file_paths[key])
    return TileManager(readers, mode, max_cache_bytes=max_cache_bytes, logger=logger)


def gather_footprint_tiles(
    tile_manager: TileManager,
    footprints: Iterable[Any],
) -> dict[str, list[GridTile]]:
    """Fetch, for every active source, the merged tile covering each footprint's box.

    This is the concrete *data-access seam* the aggregation engine will build on: it
    walks the footprints (which must be sorted along-track by the caller for cache
    locality) and, for each active reader, asks the TileManager for the merged tile
    covering that footprint's PSF bounding box. Repeated/adjacent boxes are served
    from the LRU cache, so this exercises the caching layer end to end.

    Parameters
    ----------
    tile_manager : TileManager
        The manager whose active sources to gather from.
    footprints : Iterable[Any]
        Footprint objects exposing a ``bbox`` attribute of type
        :class:`~libera_utils.footprint_matching.types.BoundingBox` (e.g.
        :class:`~libera_utils.footprint_matching.camera_segmentation.PseudoFootprint`).

    Returns
    -------
    dict[str, list[GridTile]]
        Mapping of source key to a list of merged tiles, one per footprint in input
        order. The PSF aggregation engine will consume this per source/footprint.
    """
    tiles_by_source: dict[str, list[GridTile]] = {source: [] for source in tile_manager.sources}
    for footprint in footprints:
        bbox = footprint.bbox
        for source in tile_manager.sources:
            tiles_by_source[source].append(tile_manager.get_data(source, bbox))
    return tiles_by_source
