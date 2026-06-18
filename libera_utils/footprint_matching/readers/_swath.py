"""Shared helpers for rasterizing swath / point ancillary products onto a grid.

Most footprint-matching ancillary readers serve data that is already on a
regular latitude/longitude grid (IGBP, ERA5, VIIRS cloud/BRDF/AOD). Those
readers subclass :class:`~libera_utils.footprint_matching.readers.base.GriddedDataReader`
and simply slice their native grid.

The CERES heritage products are different: SSF is *footprint* data (a 1-D list
of CERES footprints, each with its own lat/lon) and CLDPIX is *pixel* data (a
2-D imager swath, each pixel with its own lat/lon). In both cases latitude and
longitude are **data variables**, not axes — so a 2° tile is an irregular,
variable-length collection of points rather than a rectangular grid.

To keep these products inside the existing ``GriddedDataReader`` / ``GridTile``
contract (so the downstream TileManager, PSF engine, and aggregation engine do
not need a second code path), we **rasterize**: each reader bins its points into
a regular sub-grid that covers the requested 2° tile and aggregates the points
that fall in each cell. The output is an ordinary gridded ``(n_var, n_lat,
n_lon)`` array, exactly like the native-grid readers produce.

This module holds the format-agnostic pieces shared by the SSF and CLDPIX
readers:

- :func:`normalize_longitude` — convert a 0..360 longitude convention to
  −180..180 (CERES products store longitude as 0..360).
- :func:`apply_fill_and_valid_range` — convert a raw masked/integer array to a
  float array with fill values and out-of-range values replaced by ``NaN``.
- :func:`rasterize_points_to_grid` — the core point-to-grid binning routine.

Why rasterize rather than add a separate "point tile" abstraction?
------------------------------------------------------------------
A parallel point abstraction would force every downstream layer (cache,
aggregation, plotting) to branch on grid-vs-point. Rasterizing at read time
keeps a single, simple data model. The cost is that per-footprint / per-pixel
identity is lost inside a cell; that is acceptable because the footprint
matching engine ultimately wants a *spatially averaged* value under each Libera
radiometer footprint anyway.
"""

from __future__ import annotations

import numpy as np

# Aggregation strategy names understood by :func:`rasterize_points_to_grid`.
# These mirror the ``aggregation`` strings used in ``VariableSpec`` so a reader
# can pass its variables' aggregation names straight through.
AGG_MEAN: str = "weighted_mean"
AGG_LOG_MEAN: str = "weighted_log_mean"
AGG_MODE: str = "weighted_mode"


def normalize_longitude(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from the 0..360 convention to −180..180.

    CERES SSF and CLDPIX store longitude in degrees-east on a 0..360 scale,
    whereas the footprint-matching tile grid (and every other reader) uses
    −180..180. The transform ``((lon + 180) % 360) - 180`` maps both
    conventions onto −180..180 and is a no-op for values already in that range.

    Parameters
    ----------
    lon : np.ndarray
        Longitude values in degrees (any convention).

    Returns
    -------
    np.ndarray
        Longitudes wrapped to the half-open interval [−180, 180).
    """
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def apply_fill_and_valid_range(
    raw: np.ndarray,
    fill_value: float | int | None = None,
    valid_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return ``raw`` as float64 with fill / out-of-range values set to ``NaN``.

    netCDF variables in the CERES products flag missing data in two ways that we
    must both honour: an explicit ``_FillValue`` sentinel (e.g. ``3.4028235e38``
    for floats, ``32767`` for int16, ``127`` for int8, ``-1`` for the snow/ice
    maps) *and* a ``valid_range`` attribute outside of which values are invalid.
    Reading as float64 first lets us represent ``NaN`` uniformly regardless of
    the source integer dtype.

    Parameters
    ----------
    raw : np.ndarray
        Raw values read from the file. May be a ``numpy.ma.MaskedArray``.
    fill_value : float or int, optional
        Sentinel value to treat as missing. Compared before the array is cast,
        so integer fills match exactly. ``None`` skips the fill check.
    valid_range : tuple of (float, float), optional
        Inclusive ``(low, high)`` bounds; values outside are set to ``NaN``.
        The order is normalized internally, so ``(1100, 10)`` (as CLDPIX stores
        for pressure) is treated the same as ``(10, 1100)``. ``None`` skips the
        range check.

    Returns
    -------
    np.ndarray
        float64 array, same shape as ``raw``, with invalid entries as ``NaN``.
    """
    # Start from a plain float64 array. ``np.ma.filled`` collapses any existing
    # mask to NaN; np.asarray handles the already-unmasked case.
    if isinstance(raw, np.ma.MaskedArray):
        arr = np.ma.filled(raw.astype(np.float64), np.nan)
    else:
        arr = np.asarray(raw, dtype=np.float64)

    if fill_value is not None:
        # Use np.isclose for floats (the float max sentinel is exact in IEEE-754
        # but isclose is harmless) and exact equality for integers via the same
        # path — both are represented in float64 here.
        arr[arr == float(fill_value)] = np.nan

    if valid_range is not None:
        low, high = float(valid_range[0]), float(valid_range[1])
        if low > high:
            low, high = high, low
        # NaN comparisons are always False, so already-NaN entries are untouched.
        arr[(arr < low) | (arr > high)] = np.nan

    return arr


def rasterize_points_to_grid(
    point_lats: np.ndarray,
    point_lons: np.ndarray,
    values: np.ndarray,
    bbox: tuple[float, float, float, float],
    cell_size_deg: float,
    aggregations: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin scattered points into a regular sub-grid covering ``bbox``.

    Builds a regular grid over ``bbox`` at ``cell_size_deg`` spacing, assigns
    every input point to the cell that contains it, and aggregates the points in
    each cell according to the per-variable strategy in ``aggregations``. Cells
    with no valid points become ``NaN``.

    Parameters
    ----------
    point_lats : np.ndarray
        1-D array of point latitudes in degrees (−90..90). Length ``n_points``.
    point_lons : np.ndarray
        1-D array of point longitudes in degrees, already normalized to
        −180..180 (see :func:`normalize_longitude`). Length ``n_points``.
    values : np.ndarray
        Variable values for the points, shape ``(n_var, n_points)``. Missing
        values must already be ``NaN`` (see :func:`apply_fill_and_valid_range`).
    bbox : tuple of (lat_min, lat_max, lon_min, lon_max)
        Geographic bounds of the target tile, in degrees.
    cell_size_deg : float
        Edge length of each output grid cell in degrees. The number of rows and
        columns is ``round((extent) / cell_size_deg)`` so the tile is covered
        exactly when its extent is a multiple of ``cell_size_deg``.
    aggregations : list of str
        One entry per variable (same order as ``values`` axis 0). One of
        :data:`AGG_MEAN`, :data:`AGG_LOG_MEAN`, or :data:`AGG_MODE`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(data, lats, lons)`` where ``data`` is float32 of shape
        ``(n_var, n_lat, n_lon)`` (cell-center grid), and ``lats`` / ``lons``
        are float64 1-D cell-center coordinate arrays in ascending order.

    Notes
    -----
    All variables — continuous and categorical alike — are returned in a single
    float32 array. Categorical codes are carried as floats with ``NaN`` fill,
    matching the single-array convention used by the multi-variable grid readers
    (the ``VariableSpec.dtype`` records the intended type for the aggregation
    engine to cast later).
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    n_var = values.shape[0]

    # --- build the output grid dimensions -----------------------------------
    # round() (not int()) so a 2° tile at 0.05° gives exactly 40 cells rather
    # than 39 from floating-point truncation.
    n_lat = max(1, int(round((lat_max - lat_min) / cell_size_deg)))
    n_lon = max(1, int(round((lon_max - lon_min) / cell_size_deg)))

    # Cell-center coordinate arrays (ascending), consistent with grid readers.
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)
    lats_out = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lons_out = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    data = np.full((n_var, n_lat, n_lon), np.nan, dtype=np.float32)

    point_lats = np.asarray(point_lats, dtype=np.float64)
    point_lons = np.asarray(point_lons, dtype=np.float64)

    # --- keep only points that fall inside the tile and have finite coords ---
    in_box = (
        np.isfinite(point_lats)
        & np.isfinite(point_lons)
        & (point_lats >= lat_min)
        & (point_lats <= lat_max)
        & (point_lons >= lon_min)
        & (point_lons <= lon_max)
    )
    if not np.any(in_box):
        # No coverage in this tile; return the all-NaN grid.
        return data, lats_out.astype(np.float64), lons_out.astype(np.float64)

    sel_lats = point_lats[in_box]
    sel_lons = point_lons[in_box]
    sel_vals = values[:, in_box]

    # --- assign each point to a cell ----------------------------------------
    # floor((coord - origin) / cell) gives the 0-based cell index; clip the
    # exact upper-edge points (coord == max) back into the last cell.
    lat_idx = np.floor((sel_lats - lat_min) / cell_size_deg).astype(np.int64)
    lon_idx = np.floor((sel_lons - lon_min) / cell_size_deg).astype(np.int64)
    np.clip(lat_idx, 0, n_lat - 1, out=lat_idx)
    np.clip(lon_idx, 0, n_lon - 1, out=lon_idx)

    # Flatten (lat, lon) cell coordinates to a single linear cell id so we can
    # use fast np.add.at / np.bincount accumulation.
    flat_cell = lat_idx * n_lon + lon_idx
    n_cells = n_lat * n_lon

    for v in range(n_var):
        agg = aggregations[v]
        vals = sel_vals[v]
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue

        cells_v = flat_cell[finite]
        vals_v = vals[finite]

        if agg in (AGG_MEAN, AGG_LOG_MEAN):
            grid_flat = _aggregate_mean(cells_v, vals_v, n_cells, log=(agg == AGG_LOG_MEAN))
        elif agg == AGG_MODE:
            grid_flat = _aggregate_mode(cells_v, vals_v, n_cells)
        else:
            raise ValueError(f"Unknown aggregation {agg!r} for rasterization.")

        data[v] = grid_flat.reshape(n_lat, n_lon).astype(np.float32)

    return data, lats_out.astype(np.float64), lons_out.astype(np.float64)


def _aggregate_mean(cells: np.ndarray, vals: np.ndarray, n_cells: int, *, log: bool) -> np.ndarray:
    """Per-cell arithmetic (or geometric) mean, empty cells as ``NaN``.

    Parameters
    ----------
    cells : np.ndarray
        Linear cell id for each point.
    vals : np.ndarray
        Finite values for each point.
    n_cells : int
        Total number of grid cells.
    log : bool
        If True compute the geometric mean (``exp(mean(log(v)))``) for
        positive-definite, log-normally distributed quantities such as optical
        depth and AOD. Non-positive values are dropped from the log average.

    Returns
    -------
    np.ndarray
        Flat array of length ``n_cells`` with the per-cell mean (``NaN`` where
        no points contributed).
    """
    if log:
        # Geometric mean is only defined for strictly positive values.
        positive = vals > 0
        cells = cells[positive]
        vals = np.log(vals[positive])

    counts = np.bincount(cells, minlength=n_cells)
    sums = np.bincount(cells, weights=vals, minlength=n_cells)

    out = np.full(n_cells, np.nan, dtype=np.float64)
    nonempty = counts > 0
    out[nonempty] = sums[nonempty] / counts[nonempty]
    if log:
        out[nonempty] = np.exp(out[nonempty])
    return out


def _aggregate_mode(cells: np.ndarray, vals: np.ndarray, n_cells: int) -> np.ndarray:
    """Per-cell most-common value (mode), empty cells as ``NaN``.

    Used for categorical variables (e.g. cloud phase, IGBP ecosystem). Works for
    arbitrary integer-coded category spaces — including the large encoded CERES
    scene/ADM identifiers — by grouping on the composite ``(cell, value)`` key
    and selecting, per cell, the value with the highest count.

    Parameters
    ----------
    cells : np.ndarray
        Linear cell id for each point.
    vals : np.ndarray
        Finite (integer-coded) category values for each point.
    n_cells : int
        Total number of grid cells.

    Returns
    -------
    np.ndarray
        Flat array of length ``n_cells`` with the per-cell modal value
        (``NaN`` where no points contributed).
    """
    out = np.full(n_cells, np.nan, dtype=np.float64)

    # Round to integer codes; categorical data is integer in the source files.
    codes = np.rint(vals).astype(np.int64)

    # Count occurrences of each (cell, code) pair. Sorting by (cell, code) groups
    # identical pairs together so we can find run lengths with one diff pass.
    order = np.lexsort((codes, cells))
    s_cells = cells[order]
    s_codes = codes[order]

    # Boundaries between distinct (cell, code) groups.
    pair_changes = np.empty(s_cells.shape[0], dtype=bool)
    pair_changes[0] = True
    pair_changes[1:] = (s_cells[1:] != s_cells[:-1]) | (s_codes[1:] != s_codes[:-1])
    group_starts = np.flatnonzero(pair_changes)

    group_cells = s_cells[group_starts]
    group_codes = s_codes[group_starts]
    # Count for each group = distance to the next group start.
    group_counts = np.diff(np.append(group_starts, s_cells.shape[0]))

    # For each cell, keep the (code) of the group with the largest count.
    # Iterate over cells that actually have data; the number of distinct groups
    # is bounded by n_points so this stays cheap.
    best_count: dict[int, int] = {}
    for cell_id, code, count in zip(group_cells, group_codes, group_counts):
        prev = best_count.get(cell_id)
        if prev is None or count > prev:
            best_count[cell_id] = count
            out[cell_id] = code
    return out
