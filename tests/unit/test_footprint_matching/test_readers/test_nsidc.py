"""Unit tests for NISEReader.

The NISE product is distributed as an HDF-EOS4 file requiring pyhdf, which
in turn requires the HDF4 C library. To keep tests environment-independent,
``_read_extent_sds`` is mocked to return a synthetic numpy array directly.
The lat/lon grid computation (``_compute_latlon_grid``) exercises the real
pyproj EPSG:3408 → EPSG:4326 transform, which is always available.

Real NISE files can be downloaded from:
    NSIDC HTTPS: https://n5eil01u.ecs.nsidc.org/NISE/
    Earthdata login required: https://urs.earthdata.nasa.gov/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey

# Expected output variables, in the canonical order the reader stacks them
# (axis 0 of the returned data array). Kept here so the tests assert the
# ordering contract independently of the reader's internal constants.
_EXPECTED_VARIABLES = (
    "sea_ice_concentration",
    "no_ice_or_snow",
    "permanent_ice",
    "dry_snow_on_land",
    "missing",
)


def _layer_index(name: str) -> int:
    """Return the axis-0 index of variable ``name`` in the reader output."""
    return _EXPECTED_VARIABLES.index(name)

# Small test grid parameters — passed to NISEReader to override 721×721 defaults.
# 500 km cells give a 4×4 grid a usable geographic extent in Northern Hemisphere.
_TEST_ROWS = 4
_TEST_COLS = 4
_TEST_RESOLUTION_M = 500_000.0   # 500 km — very coarse, enough for unit tests
_TEST_X_ORIGIN = -1_000_000.0   # meters (EPSG:3408)
_TEST_Y_ORIGIN = 1_000_000.0    # meters (EPSG:3408)


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
        assert NISEReader.READER_KEY == "nise"

    def test_resolution_km(self):
        assert NISEReader.RESOLUTION_KM == 25.0

    def test_required_mode_is_cam(self):
        assert NISEReader.REQUIRED_MODE == OperationalMode.CAM

    def test_output_cell_deg(self):
        assert NISEReader.OUTPUT_CELL_DEG == 0.25

    def test_variables_has_five_entries(self):
        assert len(NISEReader.VARIABLES) == 5

    def test_variable_names_and_order(self):
        names = tuple(v.name for v in NISEReader.VARIABLES)
        assert names == _EXPECTED_VARIABLES

    def test_all_variables_are_float32(self):
        assert all(v.dtype == "float32" for v in NISEReader.VARIABLES)

    def test_all_variables_use_weighted_mean(self):
        assert all(v.aggregation == "weighted_mean" for v in NISEReader.VARIABLES)

    def test_all_n_categories_are_none(self):
        # Fractional coverage layers, not discrete classes.
        assert all(v.n_categories is None for v in NISEReader.VARIABLES)


class TestNISEReaderLayerMapping:
    """Verify each NISE Extent code routes to the correct float32 coverage layer."""

    def _run(self, tmp_path: Path, monkeypatch, data: np.ndarray, bbox: BoundingBox | None = None):
        """Return the full ``(5, n_lat, n_lon)`` stack for an all-``data`` grid."""
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds", lambda: data)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        if bbox is None:
            bbox = BoundingBox(
                float(lats_2d.min()) - 0.1, float(lats_2d.max()) + 0.1,
                float(lons_2d.min()) - 0.1, float(lons_2d.max()) + 0.1,
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
        stack = self._run(tmp_path, monkeypatch, self._filled(50))
        assert np.allclose(self._layer(stack, "sea_ice_concentration"), 0.5, atol=1e-5)
        # All other layers must be zero for a pure sea-ice tile.
        for name in _EXPECTED_VARIABLES:
            if name != "sea_ice_concentration":
                assert np.all(self._layer(stack, name) == 0.0)

    def test_code_1_maps_to_0_01(self, tmp_path, monkeypatch):
        stack = self._run(tmp_path, monkeypatch, self._filled(1))
        assert np.allclose(self._layer(stack, "sea_ice_concentration"), 0.01, atol=1e-5)

    def test_code_100_maps_to_1_0(self, tmp_path, monkeypatch):
        stack = self._run(tmp_path, monkeypatch, self._filled(100))
        assert np.allclose(self._layer(stack, "sea_ice_concentration"), 1.0, atol=1e-5)

    def test_code_0_is_no_ice_or_snow(self, tmp_path, monkeypatch):
        stack = self._run(tmp_path, monkeypatch, self._filled(0))
        assert np.all(self._layer(stack, "no_ice_or_snow") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_101_is_permanent_ice(self, tmp_path, monkeypatch):
        # Code 101 = permanent ice (Greenland, Antarctica ice shelves).
        stack = self._run(tmp_path, monkeypatch, self._filled(101))
        assert np.all(self._layer(stack, "permanent_ice") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_103_is_dry_snow_on_land(self, tmp_path, monkeypatch):
        # Code 103 is within the 103–110 dry-snow-on-land range.
        stack = self._run(tmp_path, monkeypatch, self._filled(103))
        assert np.all(self._layer(stack, "dry_snow_on_land") == 1.0)
        assert np.all(self._layer(stack, "sea_ice_concentration") == 0.0)

    def test_code_110_is_dry_snow_on_land(self, tmp_path, monkeypatch):
        # Upper bound of the dry-snow range is inclusive.
        stack = self._run(tmp_path, monkeypatch, self._filled(110))
        assert np.all(self._layer(stack, "dry_snow_on_land") == 1.0)

    def test_code_255_is_missing(self, tmp_path, monkeypatch):
        stack = self._run(tmp_path, monkeypatch, self._filled(255))
        assert np.all(self._layer(stack, "missing") == 1.0)
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
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        assert lats_2d.shape == (_TEST_ROWS, _TEST_COLS)
        assert lons_2d.shape == (_TEST_ROWS, _TEST_COLS)

    def test_lat_values_in_valid_range(self, tmp_path):
        reader = _make_reader(tmp_path)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        assert np.all((lats_2d >= -90) & (lats_2d <= 90))
        assert np.all((lons_2d >= -180) & (lons_2d <= 180))

    def test_northern_hemisphere_coverage(self, tmp_path):
        # Test grid is centered in the Northern Hemisphere (EPSG:3408 near-pole).
        reader = _make_reader(tmp_path)
        lats_2d, _ = reader._compute_latlon_grid()
        assert np.sum(lats_2d > 0) >= _TEST_ROWS * _TEST_COLS // 2


class TestNISEReaderLoadSpatialRegion:
    def test_returns_3d_data_and_1d_coords(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds",
                            lambda: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8))
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1, float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1, float(lons_2d.max()) + 0.1,
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
        monkeypatch.setattr(reader, "_read_extent_sds",
                            lambda: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8))
        # Bbox in the Southern Hemisphere far from test grid (near pole).
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        # Like the swath readers, an uncovered tile is an all-NaN grid whose
        # leading axis still reports the variable count.
        assert data_sub.shape[0] == len(NISEReader.VARIABLES)
        assert np.all(np.isnan(data_sub))

    def test_data_dtype_is_float32(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds",
                            lambda: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8))
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1, float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1, float(lons_2d.max()) + 0.1,
        )
        data_sub, _, _ = reader._load_spatial_region(bbox)
        assert data_sub.dtype == np.float32

    def test_data_values_in_0_to_1_range(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        # Mix of all meaningful codes to verify all map to [0, 1].
        mixed = np.array([[0, 50, 100, 101],
                          [0, 25, 75, 103],
                          [0, 1, 99, 101],
                          [0, 0, 0, 0]], dtype=np.uint8)
        monkeypatch.setattr(reader, "_read_extent_sds", lambda: mixed)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1, float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1, float(lons_2d.max()) + 0.1,
        )
        data_sub, _, _ = reader._load_spatial_region(bbox)
        # Covered cells must be in [0, 1]; uncovered cells are NaN.
        finite = data_sub[np.isfinite(data_sub)]
        assert finite.size > 0
        assert np.all((finite >= 0.0) & (finite <= 1.0))

    def test_load_tile_returns_grid_tile(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds",
                            lambda: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8))
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
        raw = np.array([[60, 0, 101],
                        [105, 255, 102]], dtype=np.uint8)
        masks = reader._extent_to_category_masks(raw)

        sea_ice = masks[_layer_index("sea_ice_concentration")]
        no_ice = masks[_layer_index("no_ice_or_snow")]
        perm = masks[_layer_index("permanent_ice")]
        snow = masks[_layer_index("dry_snow_on_land")]
        missing = masks[_layer_index("missing")]

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
        raw = np.array([[0, 101, 105],
                        [255, 102, 0]], dtype=np.uint8)
        masks = reader._extent_to_category_masks(raw)
        indicator_layers = [
            masks[_layer_index(n)]
            for n in ("no_ice_or_snow", "permanent_ice", "dry_snow_on_land", "missing")
        ]
        indicator_sum = np.sum(indicator_layers, axis=0)
        assert np.all(indicator_sum <= 1.0)
