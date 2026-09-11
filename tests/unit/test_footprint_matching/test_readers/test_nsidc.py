"""Unit tests for NISEReader.

The NISE product is distributed as an HDF-EOS4 file requiring pyhdf, which
in turn requires the HDF4 C library. To keep tests environment-independent,
``_read_extent_sds`` is mocked to return a synthetic numpy array directly.
The lat/lon grid computation (``_compute_latlon_grid``) exercises the real
pyproj EPSG:3408/3409 → EPSG:4326 transforms, which are always available.

An optional integration test (``TestNISEReaderRealGranule``) runs against a real
NISE granule staged under ``external_data/`` when both the file and pyhdf are
available; it is skipped otherwise (e.g. in CI without the HDF4 C library).

Real NISE files can be downloaded from:
    NSIDC HTTPS: https://n5eil01u.ecs.nsidc.org/NISE/
    Earthdata login required: https://urs.earthdata.nasa.gov/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.nsidc import (
    _HEMISPHERE_LABEL_NORTH,
    _HEMISPHERE_LABEL_SOUTH,
    NISEReader,
    _HemisphereGrid,
)
from libera_utils.footprint_matching.types import BoundingBox, GridTile, TileKey

# Expected output variables, in the canonical order the reader stacks them
# (axis 0 of the returned data array). Kept here so the tests assert the
# ordering contract independently of the reader's internal constants.
_EXPECTED_VARIABLES = (
    "sea_ice_concentration",
    "no_ice_or_snow",
    "permanent_ice",
    "dry_snow_on_land",
    "snow_ice_missing",
)


def _layer_index(name: str) -> int:
    """Return the axis-0 index of variable ``name`` in the reader output."""
    return _EXPECTED_VARIABLES.index(name)


# Small test grid parameters — passed to NISEReader to override 721×721 defaults.
# 500 km cells give a 4×4 grid a usable geographic extent in Northern Hemisphere.
_TEST_ROWS = 4
_TEST_COLS = 4
_TEST_RESOLUTION_M = 500_000.0  # 500 km — very coarse, enough for unit tests
_TEST_X_ORIGIN = -1_000_000.0  # meters (EPSG:3408)
_TEST_Y_ORIGIN = 1_000_000.0  # meters (EPSG:3408)


def _make_reader(tmp_path: Path) -> NISEReader:
    """Return a NISEReader pointing at a dummy file with small test grid params."""
    return NISEReader(
        tmp_path / "NISE_fixture.HDFEOS",
        grid_rows=_TEST_ROWS,
        grid_cols=_TEST_COLS,
        resolution_m=_TEST_RESOLUTION_M,
        x_origin=_TEST_X_ORIGIN,
        y_origin=_TEST_Y_ORIGIN,
    )


def _mock_extent(rows: int, cols: int, data: np.ndarray | None = None) -> np.ndarray:
    """Return a uint8 array for use as a fake Extent SDS."""
    if data is not None:
        return data.astype(np.uint8)
    return np.full((rows, cols), 50, dtype=np.uint8)  # 50% concentration by default


class TestNISEReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader registry key; asserts READER_KEY equals the expected "nise".
        assert NISEReader.READER_KEY == "nise"

    def test_resolution_km(self):
        # Targets the native NISE grid resolution; asserts RESOLUTION_KM equals 25.0 km.
        assert NISEReader.RESOLUTION_KM == 25.0

    def test_output_cell_deg(self):
        # Targets the output raster cell size; asserts OUTPUT_CELL_DEG equals 0.25 degrees.
        assert NISEReader.OUTPUT_CELL_DEG == 0.25

    def test_variables_has_five_entries(self):
        # Targets the VARIABLES spec count; asserts exactly five coverage layers are defined.
        assert len(NISEReader.VARIABLES) == 5

    def test_variable_names_and_order(self):
        # Targets the VARIABLES naming/order contract; asserts the names tuple matches _EXPECTED_VARIABLES.
        names = tuple(v.name for v in NISEReader.VARIABLES)
        assert names == _EXPECTED_VARIABLES

    def test_all_variables_are_float32(self):
        # Targets the output dtype contract; asserts every VARIABLES entry declares float32.
        assert all(v.dtype == "float32" for v in NISEReader.VARIABLES)

    def test_all_variables_use_weighted_mean(self):
        # Targets the aggregation contract; asserts every variable uses "weighted_mean".
        assert all(v.aggregation == "weighted_mean" for v in NISEReader.VARIABLES)

    def test_all_n_categories_are_none(self):
        # Fractional coverage layers, not discrete classes; asserts every variable's n_categories is None.
        assert all(v.n_categories is None for v in NISEReader.VARIABLES)


class TestNISEReaderLayerMapping:
    """Verify each NISE Extent code routes to the correct float32 coverage layer."""

    def _run(self, tmp_path: Path, monkeypatch, data: np.ndarray, bbox: BoundingBox | None = None):
        """Return the full ``(5, n_lat, n_lon)`` stack for an all-``data`` grid."""
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds", lambda token=None: data)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        if bbox is None:
            bbox = BoundingBox(
                float(lats_2d.min()) - 0.1,
                float(lats_2d.max()) + 0.1,
                float(lons_2d.min()) - 0.1,
                float(lons_2d.max()) + 0.1,
            )
        stack, _, _ = reader._load_spatial_region(bbox)
        return stack

    def _layer(self, stack: np.ndarray, name: str) -> np.ndarray:
        """Return the finite (covered) cells of one rasterized layer.

        Rasterization leaves uncovered cells as NaN, so the per-code assertions
        below operate on the covered cells only. A uniform input code produces a
        single value across every covered cell.
        """
        layer = stack[_layer_index(name)]
        finite = layer[np.isfinite(layer)]
        assert finite.size > 0, f"layer {name!r} has no covered cells"
        return finite

    def _filled(self, code: int) -> np.ndarray:
        return np.full((_TEST_ROWS, _TEST_COLS), code, dtype=np.uint8)

    def test_code_50_is_half_sea_ice(self, tmp_path, monkeypatch):
        # Targets Extent code 50 → sea-ice mapping; asserts the sea_ice layer is 0.5 and all other layers are 0.
        stack = self._run(tmp_path, monkeypatch, self._filled(50))
        assert np.allclose(self._layer(stack, "sea_ice_concentration"), 0.5, atol=1e-5)
        # All other layers must be zero for a pure sea-ice tile.
        for name in _EXPECTED_VARIABLES:
            if name != "sea_ice_concentration":
                assert np.all(self._layer(stack, name) == 0.0)

    def test_code_1_maps_to_0_01(self, tmp_path, monkeypatch):
        # Targets Extent code 1 → 1% sea ice; asserts the sea_ice_concentration layer equals 0.01.
        stack = self._run(tmp_path, monkeypatch, self._filled(1))
        assert np.allclose(self._layer(stack, "sea_ice_concentration"), 0.01, atol=1e-5)

    def test_code_100_maps_to_1_0(self, tmp_path, monkeypatch):
        # Targets Extent code 100 → 100% sea ice; asserts the sea_ice_concentration layer equals 1.0.
        stack = self._run(tmp_path, monkeypatch, self._filled(100))
        assert np.allclose(self._layer(stack, "sea_ice_concentration"), 1.0, atol=1e-5)

    def test_code_0_is_no_ice_or_snow(self, tmp_path, monkeypatch):
        # Targets Extent code 0 → no-ice/snow; asserts the no_ice_or_snow layer is 1.0 and sea ice is 0.
        stack = self._run(tmp_path, monkeypatch, self._filled(0))
        assert np.all(self._layer(stack, "no_ice_or_snow") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_101_is_permanent_ice(self, tmp_path, monkeypatch):
        # Code 101 = permanent ice (Greenland, Antarctica ice shelves); asserts permanent_ice is 1.0 and sea ice 0.
        stack = self._run(tmp_path, monkeypatch, self._filled(101))
        assert np.all(self._layer(stack, "permanent_ice") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_103_is_dry_snow_on_land(self, tmp_path, monkeypatch):
        # Code 103 is within the 103–110 dry-snow-on-land range; asserts dry_snow_on_land is 1.0 and sea ice 0.
        stack = self._run(tmp_path, monkeypatch, self._filled(103))
        assert np.all(self._layer(stack, "dry_snow_on_land") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_110_is_dry_snow_on_land(self, tmp_path, monkeypatch):
        # Upper bound of the dry-snow range is inclusive; asserts code 110 sets dry_snow_on_land to 1.0.
        stack = self._run(tmp_path, monkeypatch, self._filled(110))
        assert np.all(self._layer(stack, "dry_snow_on_land") == 1.0)

    def test_code_255_is_missing(self, tmp_path, monkeypatch):
        # Targets Extent code 255 → missing; asserts snow_ice_missing is 1.0 and sea ice is 0.
        stack = self._run(tmp_path, monkeypatch, self._filled(255))
        assert np.all(self._layer(stack, "snow_ice_missing") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_102_belongs_to_no_layer(self, tmp_path, monkeypatch):
        # Code 102 ("not used") must be 0.0 in every covered cell of every layer
        # (covered cells exist because the pixels are geolocated; their values
        # are all zero).
        stack = self._run(tmp_path, monkeypatch, self._filled(102))
        finite = stack[np.isfinite(stack)]
        assert finite.size > 0
        assert np.all(finite == 0.0)


class TestNISEReaderLatLonGrid:
    def test_lat_lon_grid_shape(self, tmp_path):
        # Targets the lat/lon grid builder shape; asserts both 2-D arrays are (rows, cols).
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        assert lats_2d.shape == (_TEST_ROWS, _TEST_COLS)
        assert lons_2d.shape == (_TEST_ROWS, _TEST_COLS)

    def test_lat_values_in_valid_range(self, tmp_path):
        # Targets geographic validity of the reprojected grid; asserts lats in [-90,90] and lons in [-180,180].
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        assert np.all((lats_2d >= -90) & (lats_2d <= 90))
        assert np.all((lons_2d >= -180) & (lons_2d <= 180))

    def test_northern_hemisphere_coverage(self, tmp_path):
        # Test grid is centered in the Northern Hemisphere (EPSG:3408 near-pole); asserts most cells have lat > 0.
        reader = _make_reader(tmp_path)
        lats_2d, _ = reader._compute_latlon_grid()
        assert np.sum(lats_2d > 0) >= _TEST_ROWS * _TEST_COLS // 2


class TestNISEReaderLoadSpatialRegion:
    def test_returns_3d_data_and_1d_coords(self, tmp_path, monkeypatch):
        # Targets the region loader output shape; asserts a 3-D (vars, lat, lon) array with 1-D lat/lon coords.
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(
            reader, "_read_extent_sds", lambda token=None: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8)
        )
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1,
            float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1,
            float(lons_2d.max()) + 0.1,
        )
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        assert data_sub.ndim == 3
        # Axis 0 is the variable axis and must match the VARIABLES count.
        assert data_sub.shape[0] == len(NISEReader.VARIABLES)
        assert data_sub.shape[1:] == (lats_sub.size, lons_sub.size)
        assert lats_sub.ndim == 1
        assert lons_sub.ndim == 1

    def test_empty_result_outside_bbox(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(
            reader, "_read_extent_sds", lambda token=None: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8)
        )
        # Bbox in the Southern Hemisphere far from test grid (near pole).
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        # Like the swath readers, an uncovered tile is an all-NaN grid whose
        # leading axis still reports the variable count.
        assert data_sub.shape[0] == len(NISEReader.VARIABLES)
        assert np.all(np.isnan(data_sub))

    def test_data_dtype_is_float32(self, tmp_path, monkeypatch):
        # Targets output dtype after rasterization; asserts the loaded region array is float32.
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(
            reader, "_read_extent_sds", lambda token=None: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8)
        )
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1,
            float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1,
            float(lons_2d.max()) + 0.1,
        )
        data_sub, _, _ = reader._load_spatial_region(bbox)
        assert data_sub.dtype == np.float32

    def test_data_values_in_0_to_1_range(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        # Mix of all meaningful codes to verify all map to [0, 1].
        mixed = np.array([[0, 50, 100, 101], [0, 25, 75, 103], [0, 1, 99, 101], [0, 0, 0, 0]], dtype=np.uint8)
        monkeypatch.setattr(reader, "_read_extent_sds", lambda token=None: mixed)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1,
            float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1,
            float(lons_2d.max()) + 0.1,
        )
        data_sub, _, _ = reader._load_spatial_region(bbox)
        # Covered cells must be in [0, 1]; uncovered cells are NaN.
        finite = data_sub[np.isfinite(data_sub)]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite <= 1.0))

    def test_load_tile_returns_grid_tile(self, tmp_path, monkeypatch):
        # Targets the public load_tile API; asserts it returns a GridTile with source "nise" and one layer per variable.
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(
            reader, "_read_extent_sds", lambda token=None: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8)
        )
        lats_2d, lons_2d = reader._compute_latlon_grid()
        # Build a TileKey that overlaps the test grid's geographic extent.
        lat_center = float(lats_2d.mean())
        lon_center = float(lons_2d.mean())
        import math

        from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG

        lat_idx = max(0, min(int(math.floor((lat_center + 90.0) / TILE_SIZE_DEG)), 89))
        lon_idx = max(0, min(int(math.floor((lon_center + 180.0) / TILE_SIZE_DEG)), 179))
        key = TileKey("nise", lat_idx, lon_idx)
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.source == "nise"
        # Tile data carries one layer per NISE variable on the leading axis.
        assert tile.data.shape[0] == len(NISEReader.VARIABLES)


class TestNISEExtentToCategoryMasks:
    """Directly exercise the Extent-code → five-layer split helper."""

    def test_stack_shape_and_dtype(self, tmp_path):
        # Targets the Extent→masks helper output; asserts shape (nvars, rows, cols) and float32 dtype.
        reader = _make_reader(tmp_path)
        raw = np.zeros((_TEST_ROWS, _TEST_COLS), dtype=np.uint8)
        masks = reader._extent_to_category_masks(raw)
        assert masks.shape == (len(NISEReader.VARIABLES), _TEST_ROWS, _TEST_COLS)
        assert masks.dtype == np.float32

    def test_each_code_lands_in_expected_layer(self, tmp_path):
        reader = _make_reader(tmp_path)
        # One pixel per code group, laid out across a 2×3 grid:
        #   60  -> sea ice 0.60      0   -> no_ice_or_snow
        #   101 -> permanent ice     105 -> dry snow on land
        #   255 -> missing           102 -> belongs to no layer
        raw = np.array([[60, 0, 101], [105, 255, 102]], dtype=np.uint8)
        masks = reader._extent_to_category_masks(raw)

        sea_ice = masks[_layer_index("sea_ice_concentration")]
        no_ice = masks[_layer_index("no_ice_or_snow")]
        perm = masks[_layer_index("permanent_ice")]
        snow = masks[_layer_index("dry_snow_on_land")]
        missing = masks[_layer_index("snow_ice_missing")]

        assert np.isclose(sea_ice[0, 0], 0.60, atol=1e-5)
        assert no_ice[0, 1] == 1.0
        assert perm[0, 2] == 1.0
        assert snow[1, 0] == 1.0
        assert missing[1, 1] == 1.0
        # Code 102 pixel is zero in every layer.
        assert np.all(masks[:, 1, 2] == 0.0)

    def test_layers_are_mutually_exclusive_per_pixel(self, tmp_path):
        # For non-sea-ice codes the five indicator layers must not double-count:
        # at most one layer is 1.0 at any pixel (sea-ice excluded since it is a
        # fractional value, not a 0/1 indicator).
        reader = _make_reader(tmp_path)
        raw = np.array([[0, 101, 105], [255, 102, 0]], dtype=np.uint8)
        masks = reader._extent_to_category_masks(raw)
        indicator_layers = [
            masks[_layer_index(n)] for n in ("no_ice_or_snow", "permanent_ice", "dry_snow_on_land", "snow_ice_missing")
        ]
        indicator_sum = np.sum(indicator_layers, axis=0)
        assert np.all(indicator_sum <= 1.0)


def _small_hemi(epsg: int, token: str) -> _HemisphereGrid:
    """A ``_HemisphereGrid`` on the coarse 4×4 test grid, for hemisphere tests."""
    return _HemisphereGrid(
        epsg=epsg,
        hemisphere_label=token,
        grid_rows=_TEST_ROWS,
        grid_cols=_TEST_COLS,
        resolution_m=_TEST_RESOLUTION_M,
        x_origin=_TEST_X_ORIGIN,
        y_origin=_TEST_Y_ORIGIN,
    )


class TestNISEReaderBothHemispheres:
    """The reader reads and merges both the Northern and Southern EASE-Grids."""

    def test_southern_hemisphere_reprojection(self, tmp_path):
        # A Southern (EPSG:3409) small grid must reproject to negative latitudes,
        # mirroring TestNISEReaderLatLonGrid.test_northern_hemisphere_coverage.
        reader = NISEReader(
            tmp_path / "NISE.HDFEOS",
            hemispheres=(_small_hemi(3409, _HEMISPHERE_LABEL_SOUTH),),
        )
        lats_2d, _ = reader._compute_latlon_grid(reader._hemispheres[0])
        assert np.all(lats_2d < 0)

    def test_both_hemispheres_are_concatenated(self, tmp_path, monkeypatch):
        # Targets merging of both hemispheres into one point cloud; asserts lats span +/- and sizes equal 2x the grid.
        reader = NISEReader(
            tmp_path / "NISE.HDFEOS",
            hemispheres=(
                _small_hemi(3408, _HEMISPHERE_LABEL_NORTH),
                _small_hemi(3409, _HEMISPHERE_LABEL_SOUTH),
            ),
        )
        per_token = {
            _HEMISPHERE_LABEL_NORTH: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8),
            _HEMISPHERE_LABEL_SOUTH: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8),
        }
        monkeypatch.setattr(reader, "_read_extent_sds", lambda token: per_token[token])
        lats, lons, values = reader._load_points
        # Point cloud spans both hemispheres and keeps every pixel (no code 254 to drop).
        assert np.any(lats > 0)
        assert np.any(lats < 0)
        expected = 2 * _TEST_ROWS * _TEST_COLS
        assert lats.size == expected
        assert lons.size == expected
        assert values.shape == (len(NISEReader.VARIABLES), expected)

    def test_southern_tile_is_populated(self, tmp_path, monkeypatch):
        # Regression for the reported gap: a Southern-hemisphere tile now carries finite
        # coverage where a North-only reader returned an all-NaN grid.
        reader = NISEReader(
            tmp_path / "NISE.HDFEOS",
            hemispheres=(_small_hemi(3409, _HEMISPHERE_LABEL_SOUTH),),
        )
        monkeypatch.setattr(
            reader,
            "_read_extent_sds",
            lambda token=None: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8),
        )
        lats_2d, lons_2d = reader._compute_latlon_grid(reader._hemispheres[0])
        assert np.all(lats_2d < 0)  # sanity: really the Southern grid
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1,
            float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1,
            float(lons_2d.max()) + 0.1,
        )
        data_sub, _, _ = reader._load_spatial_region(bbox)
        finite = data_sub[np.isfinite(data_sub)]
        assert finite.size > 0
        # Code 50 → 0.5 sea ice concentration across covered cells of that layer.
        sea_ice = data_sub[_layer_index("sea_ice_concentration")]
        assert np.allclose(sea_ice[np.isfinite(sea_ice)], 0.5, atol=1e-5)

    def test_off_earth_code_254_is_dropped_not_zeroed(self, tmp_path, monkeypatch):
        # Off-Earth corner pixels (254) must be removed from the point cloud entirely,
        # not carried as all-zero layers (which would dilute the opposite hemisphere).
        reader = _make_reader(tmp_path)  # single Northern hemisphere, 4×4
        raw = np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8)
        raw[0, 0] = 254
        raw[-1, -1] = 254
        monkeypatch.setattr(reader, "_read_extent_sds", lambda token=None: raw)
        lats, lons, values = reader._load_points
        expected = _TEST_ROWS * _TEST_COLS - 2  # two 254 pixels dropped
        assert lats.size == expected
        assert lons.size == expected
        assert values.shape == (len(NISEReader.VARIABLES), expected)


# Real granule staged under the repo's external_data/ tree (untracked; present locally).
_REAL_NISE = (
    Path(__file__).resolve().parents[4] / "external_data" / "external_data" / "NSDIC" / "NISE_SSMISF18_20260111.HDFEOS"
)


def _pyhdf_available() -> bool:
    try:
        import pyhdf.SD  # noqa: F401,PLC0415

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _REAL_NISE.exists(), reason="real NISE granule not staged under external_data/")
@pytest.mark.skipif(not _pyhdf_available(), reason="pyhdf/HDF4 C library not available")
class TestNISEReaderRealGranule:
    """End-to-end checks against a real staged NISE granule (both hemispheres flow through).

    Skipped automatically when the granule is not staged or pyhdf/HDF4 is unavailable
    (e.g. in CI), which is why the rest of the suite mocks ``_read_extent_sds``.
    """

    def test_reads_both_extent_sds_by_hemisphere(self):
        # Targets per-hemisphere SDS reads on a real granule; asserts both are 721x721 and are distinct arrays.
        reader = NISEReader(_REAL_NISE)
        north = reader._read_extent_sds(_HEMISPHERE_LABEL_NORTH)
        south = reader._read_extent_sds(_HEMISPHERE_LABEL_SOUTH)
        assert north.shape == (721, 721)
        assert south.shape == (721, 721)
        # Two genuinely different grids, not the same SDS resolved twice.
        assert not np.array_equal(north, south)

    def test_northern_and_southern_tiles_both_covered(self):
        # Targets both-hemisphere coverage on a real granule; asserts Arctic and Antarctic bboxes each yield finite cells.
        reader = NISEReader(_REAL_NISE)
        north_tile, _, _ = reader._load_spatial_region(BoundingBox(70.0, 80.0, -10.0, 10.0))
        south_tile, _, _ = reader._load_spatial_region(BoundingBox(-80.0, -70.0, -10.0, 10.0))
        assert np.isfinite(north_tile).any()
        assert np.isfinite(south_tile).any()

    def test_point_cloud_spans_both_hemispheres(self):
        # Targets full-globe point cloud on a real granule; asserts lat max > 60 (Arctic) and min < -60 (Antarctic).
        reader = NISEReader(_REAL_NISE)
        lats, _, _ = reader._load_points
        finite = lats[np.isfinite(lats)]
        assert finite.max() > 60.0  # Arctic coverage
        assert finite.min() < -60.0  # Antarctic coverage
