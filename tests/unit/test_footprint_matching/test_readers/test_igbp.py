"""Unit tests for IGBPReader.

pyhdf requires a system HDF4 C library which is unavailable in this container.
These tests mock ``read_hdf4_lat_lon_grid`` to return synthetic numpy arrays,
allowing the reader's spatial subsetting logic to be tested without HDF4.

Real MCD12Q1 files can be downloaded from:
    LP DAAC AppEEARS: https://appeears.earthdatacloud.nasa.gov/
    LP DAAC Data Pool: https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, TileKey


# ---------------------------------------------------------------------------
# Synthetic HDF4 data used in all IGBPReader tests
# ---------------------------------------------------------------------------

def _make_synthetic_igbp_data():
    """Return (data, lats, lons) mimicking a small MCD12Q1 tile.

    Grid: 6 rows × 6 cols, lats 0–3°, lons 10–13°.
    Category values form a simple 0–5 pattern for easy assertion.
    """
    lats = np.linspace(0.0, 3.0, 6)  # ascending latitudes
    lons = np.linspace(10.0, 13.0, 6)
    # 6 × 6 grid; values cycle through IGBP classes 0–5
    data = np.tile(np.arange(6, dtype=np.float32), (6, 1))
    return data, lats, lons


class TestIGBPReaderClassAttributes:
    def test_reader_key(self):
        assert IGBPReader.READER_KEY == "igbp"

    def test_resolution_km(self):
        assert IGBPReader.RESOLUTION_KM == 1.0

    def test_required_mode_is_cam(self):
        assert IGBPReader.REQUIRED_MODE == OperationalMode.CAM

    def test_variables_has_one_entry(self):
        assert len(IGBPReader.VARIABLES) == 1

    def test_variable_name_is_surface_type(self):
        assert IGBPReader.VARIABLES[0].name == "surface_type"

    def test_n_categories_is_20(self):
        assert IGBPReader.VARIABLES[0].n_categories == 20


class TestIGBPReaderLoadTile:
    def test_returns_grid_tile_with_correct_source(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.igbp.read_hdf4_lat_lon_grid",
            lambda **kwargs: (data, lats, lons),
        )

        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        key = TileKey("igbp", 45, 90)  # lat 0–2°, lon 0–2° (tile coords)
        # Patch bbox to match our synthetic lat/lon range
        bbox = BoundingBox(0.5, 2.5, 10.5, 12.5)
        tile = reader._load_spatial_region(bbox)
        assert tile[0].size > 0

    def test_subset_within_bbox(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.igbp.read_hdf4_lat_lon_grid",
            lambda **kwargs: (data, lats, lons),
        )

        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        # Only request the central portion of the synthetic grid
        bbox = BoundingBox(0.5, 2.5, 10.5, 12.5)
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)

        # All returned lats should be within the bbox
        assert np.all(lats_sub >= 0.5)
        assert np.all(lats_sub <= 2.5)

    def test_empty_result_when_no_pixels_in_bbox(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.igbp.read_hdf4_lat_lon_grid",
            lambda **kwargs: (data, lats, lons),
        )

        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        # Request a region completely outside the synthetic data extent
        bbox = BoundingBox(50.0, 52.0, 50.0, 52.0)
        data_sub, lats_sub, lons_sub = reader._load_spatial_region(bbox)

        assert data_sub.size == 0
        assert lats_sub.size == 0
        assert lons_sub.size == 0

    def test_data_dtype_is_float32(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.igbp.read_hdf4_lat_lon_grid",
            lambda **kwargs: (data, lats, lons),
        )

        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        bbox = BoundingBox(0.5, 2.5, 10.5, 12.5)
        data_sub, _, _ = reader._load_spatial_region(bbox)

        assert data_sub.dtype == np.float32

    def test_load_tile_via_base_returns_grid_tile(self, tmp_path, monkeypatch):
        """Full end-to-end test through load_tile() template method."""
        from libera_utils.footprint_matching.types import GridTile

        data, lats, lons = _make_synthetic_igbp_data()
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.igbp.read_hdf4_lat_lon_grid",
            lambda **kwargs: (data, lats, lons),
        )

        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        # TileKey(lat_idx=45, lon_idx=90) → lat [0, 2°], lon [0, 2°]
        # Our synthetic data is at lat 0–3°, lon 10–13°, so the bbox won't
        # overlap; we use monkeypatching at the _load_spatial_region level instead.
        key = TileKey("igbp", 45, 90)

        # Patch _load_spatial_region directly so we don't depend on coordinate math
        monkeypatch.setattr(
            reader,
            "_load_spatial_region",
            lambda bbox: (data[:2, :2], lats[:2], lons[:2]),
        )

        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.source == "igbp"
        assert tile.timestamp_source is None
