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

    def test_variables_has_one_entry(self):
        assert len(NISEReader.VARIABLES) == 1

    def test_variable_name_is_sea_ice_concentration(self):
        assert NISEReader.VARIABLES[0].name == "sea_ice_concentration"

    def test_variable_dtype_is_float32(self):
        assert NISEReader.VARIABLES[0].dtype == "float32"

    def test_variable_aggregation_is_weighted_mean(self):
        assert NISEReader.VARIABLES[0].aggregation == "weighted_mean"

    def test_n_categories_is_none(self):
        assert NISEReader.VARIABLES[0].n_categories is None


class TestNISEReaderConcentrationMapping:
    """Verify the NISE Extent code → float32 concentration value mapping."""

    def _run(self, tmp_path: Path, monkeypatch, data: np.ndarray, bbox: BoundingBox | None = None):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds", lambda: data)
        lats_2d, lons_2d = reader._compute_latlon_grid()
        if bbox is None:
            bbox = BoundingBox(
                float(lats_2d.min()) - 0.1, float(lats_2d.max()) + 0.1,
                float(lons_2d.min()) - 0.1, float(lons_2d.max()) + 0.1,
            )
        conc, _, _ = reader._load_spatial_region(bbox)
        return conc

    def test_code_50_maps_to_0_5(self, tmp_path, monkeypatch):
        data = np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8)
        conc = self._run(tmp_path, monkeypatch, data)
        assert np.allclose(conc[conc > 0], 0.5, atol=1e-5)

    def test_code_1_maps_to_0_01(self, tmp_path, monkeypatch):
        data = np.full((_TEST_ROWS, _TEST_COLS), 1, dtype=np.uint8)
        conc = self._run(tmp_path, monkeypatch, data)
        assert np.allclose(conc[conc > 0], 0.01, atol=1e-5)

    def test_code_100_maps_to_1_0(self, tmp_path, monkeypatch):
        data = np.full((_TEST_ROWS, _TEST_COLS), 100, dtype=np.uint8)
        conc = self._run(tmp_path, monkeypatch, data)
        assert np.allclose(conc[conc > 0], 1.0, atol=1e-5)

    def test_code_101_maps_to_1_0(self, tmp_path, monkeypatch):
        # Code 101 = permanent ice (Greenland, Antarctica ice shelves) → 100%
        data = np.full((_TEST_ROWS, _TEST_COLS), 101, dtype=np.uint8)
        conc = self._run(tmp_path, monkeypatch, data)
        assert np.allclose(conc[conc > 0], 1.0, atol=1e-5)

    def test_code_0_maps_to_0_0(self, tmp_path, monkeypatch):
        # Code 0 = outside domain / snow-free land → 0.0
        data = np.full((_TEST_ROWS, _TEST_COLS), 0, dtype=np.uint8)
        conc = self._run(tmp_path, monkeypatch, data)
        assert np.all(conc == 0.0)

    def test_code_103_maps_to_0_0(self, tmp_path, monkeypatch):
        # Code 103 = dry snow on land → treated as non-ocean → 0.0
        data = np.full((_TEST_ROWS, _TEST_COLS), 103, dtype=np.uint8)
        conc = self._run(tmp_path, monkeypatch, data)
        assert np.all(conc == 0.0)


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
    def test_returns_data_lats_lons(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds",
                            lambda: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8))
        lats_2d, lons_2d = reader._compute_latlon_grid()
        bbox = BoundingBox(
            float(lats_2d.min()) - 0.1, float(lats_2d.max()) + 0.1,
            float(lons_2d.min()) - 0.1, float(lons_2d.max()) + 0.1,
        )
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        assert data_sub.size > 0
        assert lats_sub.ndim == 1
        assert lons_sub.ndim == 1

    def test_empty_result_outside_bbox(self, tmp_path, monkeypatch):
        reader = _make_reader(tmp_path)
        monkeypatch.setattr(reader, "_read_extent_sds",
                            lambda: np.full((_TEST_ROWS, _TEST_COLS), 50, dtype=np.uint8))
        # Bbox in the Southern Hemisphere far from test grid (near pole).
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)
        assert data_sub.size == 0

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
        assert np.all((data_sub >= 0.0) & (data_sub <= 1.0))

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
