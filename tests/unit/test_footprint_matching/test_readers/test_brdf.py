"""Unit tests for VIIRSBRDFReader.

Uses synthetic VJ143C1 HDF5 fixture files created by
``make_viirs_brdf_hdf5_fixture``. h5py is a project dependency, so these
tests can write and read real HDF5 files without any system C library caveat
(the HDF5 C library ships bundled with h5py on most platforms).

Real VJ143C1 files can be downloaded from:
    LP DAAC: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/
    EarthData login required: https://urs.earthdata.nasa.gov/
    Example: VJ143C1.A2026153.002.2026161161054.h5
"""

from __future__ import annotations

import math

import numpy as np

from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey
from tests.test_data.footprint_matching.fixtures import make_viirs_brdf_hdf5_fixture

# Default fixture grid covers a tiny region to keep tests fast.
_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.05
_LAT_MAX = 0.20
_LON_MIN = 10.05
_LON_MAX = 10.40
_PARAM_FILL = 200  # raw int16 → 0.200 after scale_factor=0.001
_FILL_SENTINEL = 32767  # last pixel; should become NaN


def _make_reader(tmp_path, **kwargs) -> VIIRSBRDFReader:
    fixture_path = make_viirs_brdf_hdf5_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        param_fill=_PARAM_FILL,
        fill_sentinel=_FILL_SENTINEL,
        **kwargs,
    )
    return VIIRSBRDFReader(fixture_path)


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.01, _LAT_MAX + 0.01, _LON_MIN - 0.01, _LON_MAX + 0.01)


class TestVIIRSBRDFReaderClassAttributes:
    def test_reader_key(self):
        assert VIIRSBRDFReader.READER_KEY == "viirs_brdf"

    def test_resolution_km(self):
        assert VIIRSBRDFReader.RESOLUTION_KM == 5.6

    def test_required_mode_is_cam(self):
        assert VIIRSBRDFReader.REQUIRED_MODE == OperationalMode.CAM

    def test_variables_count_is_nine(self):
        assert len(VIIRSBRDFReader.VARIABLES) == 9

    def test_variable_names_cover_three_bands_three_params(self):
        names = {v.name for v in VIIRSBRDFReader.VARIABLES}
        for band in ("shortwave", "vis", "nir"):
            for param in ("fiso", "fvol", "fgeo"):
                assert f"brdf_{band}_{param}" in names

    def test_all_variables_are_float32_weighted_mean(self):
        for v in VIIRSBRDFReader.VARIABLES:
            assert v.dtype == "float32"
            assert v.aggregation == "weighted_mean"
            assert v.n_categories is None


class TestVIIRSBRDFReaderScaleAndFill:
    """Verify scale_factor application and fill → NaN conversion."""

    def test_scale_factor_applied_to_param_fill(self, tmp_path):
        # param_fill=200 with scale_factor=0.001 → 0.2
        reader = _make_reader(tmp_path)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        # All non-fill pixels should equal 0.200
        non_fill = data[~np.isnan(data)]
        assert non_fill.size > 0
        assert np.allclose(non_fill, 0.200, atol=1e-4)

    def test_fill_sentinel_becomes_nan(self, tmp_path):
        # make_viirs_brdf_hdf5_fixture writes fill_sentinel to [-1, -1] of each dataset.
        reader = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        # The fixture has fill at position [-1, -1] for every band.
        assert np.isnan(data[:, -1, -1]).all()

    def test_non_fill_pixels_are_not_nan(self, tmp_path):
        reader = _make_reader(tmp_path)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        # At least one pixel (all except [-1, -1]) should be non-NaN.
        assert not np.all(np.isnan(data))


class TestVIIRSBRDFReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        reader = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert data.ndim == 3
        assert data.shape[0] == 9

    def test_data_dtype_is_float32(self, tmp_path):
        reader = _make_reader(tmp_path)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert data.dtype == np.float32

    def test_lat_is_ascending(self, tmp_path):
        # The fixture stores lats in DESCENDING order; read_viirs_brdf_hdf5 must flip them.
        reader = _make_reader(tmp_path)
        _, lats, _ = reader._load_spatial_region(_full_bbox())
        if lats.size > 1:
            assert np.all(np.diff(lats) >= 0), f"lats not ascending: {lats}"

    def test_spatial_shape_matches_fixture(self, tmp_path):
        reader = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert data.shape[1] == _N_LAT
        assert data.shape[2] == _N_LON

    def test_empty_result_outside_bbox(self, tmp_path):
        reader = _make_reader(tmp_path)
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = reader._load_spatial_region(bbox)
        assert data.size == 0
        assert data.shape[0] == 9

    def test_partial_bbox_subsets_correctly(self, tmp_path):
        reader = _make_reader(tmp_path)
        # Request only the first half of the lon range.
        lon_mid = (_LON_MIN + _LON_MAX) / 2
        bbox = BoundingBox(_LAT_MIN - 0.01, _LAT_MAX + 0.01, _LON_MIN - 0.01, lon_mid)
        data, lats, lons = reader._load_spatial_region(bbox)
        assert lons.size < _N_LON


class TestVIIRSBRDFReaderLoadTile:
    def test_load_tile_returns_grid_tile(self, tmp_path):
        reader = _make_reader(tmp_path)
        lat_center = (_LAT_MIN + _LAT_MAX) / 2
        lon_center = (_LON_MIN + _LON_MAX) / 2
        lat_idx = max(0, min(int(math.floor((lat_center + 90.0) / TILE_SIZE_DEG)), 89))
        lon_idx = max(0, min(int(math.floor((lon_center + 180.0) / TILE_SIZE_DEG)), 179))
        key = TileKey("viirs_brdf", lat_idx, lon_idx)
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)

    def test_source_is_viirs_brdf(self, tmp_path):
        reader = _make_reader(tmp_path)
        lat_center = (_LAT_MIN + _LAT_MAX) / 2
        lon_center = (_LON_MIN + _LON_MAX) / 2
        lat_idx = max(0, min(int(math.floor((lat_center + 90.0) / TILE_SIZE_DEG)), 89))
        lon_idx = max(0, min(int(math.floor((lon_center + 180.0) / TILE_SIZE_DEG)), 179))
        key = TileKey("viirs_brdf", lat_idx, lon_idx)
        tile = reader.load_tile(key)
        assert tile.source == "viirs_brdf"

    def test_timestamp_source_is_none(self, tmp_path):
        # BRDF is a static surface property product; no instrument timestamp.
        reader = _make_reader(tmp_path)
        lat_center = (_LAT_MIN + _LAT_MAX) / 2
        lon_center = (_LON_MIN + _LON_MAX) / 2
        lat_idx = max(0, min(int(math.floor((lat_center + 90.0) / TILE_SIZE_DEG)), 89))
        lon_idx = max(0, min(int(math.floor((lon_center + 180.0) / TILE_SIZE_DEG)), 179))
        key = TileKey("viirs_brdf", lat_idx, lon_idx)
        tile = reader.load_tile(key)
        assert tile.timestamp_source is None
