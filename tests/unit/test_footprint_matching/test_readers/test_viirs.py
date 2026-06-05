"""Unit tests for VIIRSL2L3Reader.

pyhdf requires a system HDF4 C library which is unavailable in this container.
These tests mock ``read_hdf4_lat_lon_grid`` to return synthetic numpy arrays.

Real VIIRS CLDPX files can be downloaded from:
    NOAA CLASS: https://www.avl.class.noaa.gov/saa/products/welcome
    NCEI CDR:   https://www.ncei.noaa.gov/products/climate-data-records/cloud-properties-viirs
    NOAA STAR:  https://www.star.nesdis.noaa.gov/jpss/clouds.php
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.viirs import VIIRSL2L3Reader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey


# ---------------------------------------------------------------------------
# Synthetic HDF4 data used in VIIRSL2L3Reader tests
# ---------------------------------------------------------------------------

def _make_viirs_data():
    """Return (data, lats, lons) mimicking a small VIIRS cloud granule.

    Grid: 5 rows × 5 cols, lats 0–4°, lons 10–14°.
    """
    lats = np.linspace(0.0, 4.0, 5)
    lons = np.linspace(10.0, 14.0, 5)
    # Each call returns unique fill values so tests can verify stacking order.
    return {
        "cloud_fraction": np.full((5, 5), 0.3, dtype=np.float32),
        "cloud_optical_thickness": np.full((5, 5), 5.0, dtype=np.float32),
        "cloud_top_pressure": np.full((5, 5), 500.0, dtype=np.float32),
        "lats": lats,
        "lons": lons,
    }


def _mock_read_hdf4(file_path, data_sds_name, lat_sds_name, lon_sds_name, fill_value=None):
    """Mock for read_hdf4_lat_lon_grid that returns variable-specific data."""
    d = _make_viirs_data()
    return d[data_sds_name], d["lats"], d["lons"]


class TestVIIRSL2L3ReaderClassAttributes:
    def test_reader_key(self):
        assert VIIRSL2L3Reader.READER_KEY == "viirs_l2l3"

    def test_resolution_km(self):
        assert VIIRSL2L3Reader.RESOLUTION_KM == 0.75

    def test_required_mode_is_cam(self):
        assert VIIRSL2L3Reader.REQUIRED_MODE == OperationalMode.CAM

    def test_variables_has_three_entries(self):
        assert len(VIIRSL2L3Reader.VARIABLES) == 3

    def test_variable_names(self):
        names = [v.name for v in VIIRSL2L3Reader.VARIABLES]
        assert names == ["cloud_fraction", "cloud_optical_thickness", "cloud_top_pressure"]


class TestVIIRSL2L3ReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.viirs.read_hdf4_lat_lon_grid",
            _mock_read_hdf4,
        )
        reader = VIIRSL2L3Reader(tmp_path / "VIIRS_CLDPX.hdf")
        bbox = BoundingBox(0.5, 3.5, 10.5, 13.5)
        data, lats, lons = reader._load_spatial_region(bbox)
        # 3D: (3 variables, n_lat, n_lon)
        assert data.ndim == 3
        assert data.shape[0] == 3

    def test_variable_stacking_order(self, tmp_path, monkeypatch):
        """Axis 0 must match VARIABLES ordering: cloud_fraction, cot, cloud_top_pressure."""
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.viirs.read_hdf4_lat_lon_grid",
            _mock_read_hdf4,
        )
        reader = VIIRSL2L3Reader(tmp_path / "VIIRS_CLDPX.hdf")
        bbox = BoundingBox(0.5, 3.5, 10.5, 13.5)
        data, _, _ = reader._load_spatial_region(bbox)

        # Verify the fill values match what _mock_read_hdf4 returns per-variable
        assert np.allclose(data[0], 0.3, atol=1e-4)    # cloud_fraction
        assert np.allclose(data[1], 5.0, atol=1e-4)    # cloud_optical_thickness
        assert np.allclose(data[2], 500.0, atol=1e-4)  # cloud_top_pressure

    def test_empty_result_outside_bbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.viirs.read_hdf4_lat_lon_grid",
            _mock_read_hdf4,
        )
        reader = VIIRSL2L3Reader(tmp_path / "VIIRS_CLDPX.hdf")
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = reader._load_spatial_region(bbox)
        assert data.size == 0
        assert data.shape[0] == 3  # Still 3 variables even when empty

    def test_data_dtype_is_float32(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.viirs.read_hdf4_lat_lon_grid",
            _mock_read_hdf4,
        )
        reader = VIIRSL2L3Reader(tmp_path / "VIIRS_CLDPX.hdf")
        bbox = BoundingBox(0.5, 3.5, 10.5, 13.5)
        data, _, _ = reader._load_spatial_region(bbox)
        assert data.dtype == np.float32


class TestVIIRSL2L3ReaderLoadTile:
    def test_load_tile_timestamp_source_is_radiometer(self, tmp_path, monkeypatch):
        """VIIRS cloud data is collocated with the radiometer; timestamp_source must be 'radiometer'."""
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.viirs.read_hdf4_lat_lon_grid",
            _mock_read_hdf4,
        )
        reader = VIIRSL2L3Reader(tmp_path / "VIIRS_CLDPX.hdf")
        key = TileKey("viirs_l2l3", 45, 90)

        # Patch _load_spatial_region to return known data within the TileKey's bbox
        d = _make_viirs_data()
        monkeypatch.setattr(
            reader,
            "_load_spatial_region",
            lambda bbox: (
                np.stack([d["cloud_fraction"][:2, :2], d["cloud_optical_thickness"][:2, :2],
                          d["cloud_top_pressure"][:2, :2]], axis=0),
                d["lats"][:2],
                d["lons"][:2],
            ),
        )

        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.timestamp_source == "radiometer"

    def test_load_tile_source_is_viirs_l2l3(self, tmp_path, monkeypatch):
        reader = VIIRSL2L3Reader(tmp_path / "VIIRS_CLDPX.hdf")
        d = _make_viirs_data()
        monkeypatch.setattr(
            reader,
            "_load_spatial_region",
            lambda bbox: (
                np.stack([d["cloud_fraction"][:2, :2], d["cloud_optical_thickness"][:2, :2],
                          d["cloud_top_pressure"][:2, :2]], axis=0),
                d["lats"][:2],
                d["lons"][:2],
            ),
        )
        key = TileKey("viirs_l2l3", 45, 90)
        tile = reader.load_tile(key)
        assert tile.source == "viirs_l2l3"
