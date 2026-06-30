"""Unit tests for VIIRSAODReader.

Uses synthetic AERDB_D3_GEOLEO NOAA-20 VIIRS AOD NetCDF4 fixtures created by
``make_aod_noaa20_fixture``.

Real AOD files come from the NASA Deep Blue GEO-LEO merged product, e.g.
``AERDB_D3_GEOLEO_Merged.A2020121.001.2024121023016.nc``; the reader pulls the
per-sensor ``NOAA20_VIIRS`` group from that file.
"""

from __future__ import annotations

import math

import numpy as np

from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey
from tests.test_data.footprint_matching.fixtures import make_aod_noaa20_fixture

_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.5
_LAT_MAX = 3.5
_LON_MIN = 10.5
_LON_MAX = 17.5
_AOD_FILL = 0.2


def _make_reader(tmp_path, **kwargs) -> VIIRSAODReader:
    kwargs.setdefault("aod_fill", _AOD_FILL)
    fixture_path = make_aod_noaa20_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        **kwargs,
    )
    return VIIRSAODReader(fixture_path)


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.1, _LAT_MAX + 0.1, _LON_MIN - 0.1, _LON_MAX + 0.1)


class TestVIIRSAODReaderClassAttributes:
    def test_reader_key(self):
        assert VIIRSAODReader.READER_KEY == "viirs_aod"

    def test_resolution_km(self):
        assert VIIRSAODReader.RESOLUTION_KM == 111.0

    def test_required_mode_is_imager(self):
        assert VIIRSAODReader.REQUIRED_MODE == OperationalMode.IMAGER

    def test_single_aod_variable(self):
        assert len(VIIRSAODReader.VARIABLES) == 1
        var = VIIRSAODReader.VARIABLES[0]
        assert var.name == "aod_550"
        assert var.dtype == "float32"
        assert var.aggregation == "weighted_log_mean"
        assert var.n_categories is None


class TestVIIRSAODReaderLoadSpatialRegion:
    def test_returns_2d_data(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert data.ndim == 2
        assert data.shape == (_N_LAT, _N_LON)
        assert data.dtype == np.float32

    def test_lat_is_ascending(self, tmp_path):
        reader = _make_reader(tmp_path)
        _, lats, _ = reader._load_spatial_region(_full_bbox())
        assert np.all(np.diff(lats) >= 0)

    def test_non_fill_values_preserved(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.allclose(data, _AOD_FILL, atol=1e-5)

    def test_fill_pixel_becomes_nan(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=True)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.isnan(data[0, 0])
        # Every other pixel remains valid.
        assert np.isfinite(data).sum() == _N_LAT * _N_LON - 1

    def test_out_of_range_becomes_nan(self, tmp_path):
        reader = _make_reader(tmp_path, aod_fill=99.0, include_fill_pixel=False)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        # AOD valid max is 5.0, so 99.0 must be masked.
        assert np.all(np.isnan(data))

    def test_empty_result_outside_bbox(self, tmp_path):
        reader = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(BoundingBox(-60.0, -58.0, 170.0, 172.0))
        assert data.size == 0

    def test_reads_noaa20_group_not_merged(self, tmp_path):
        # Build a file with BOTH a NOAA20_VIIRS group (the real AOD value) and a
        # decoy Merged group filled with a distinct sentinel. The reader must
        # return the NOAA-20 values, guarding against a revert to the merged group.
        reader = _make_reader(tmp_path, include_fill_pixel=False, merged_decoy_value=1.23)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.allclose(data, _AOD_FILL, atol=1e-5)
        # The decoy value (1.23) must never appear.
        assert not np.any(np.isclose(data, 1.23, atol=1e-5))


class TestVIIRSAODReaderLoadTile:
    def _key(self):
        lat_center = (_LAT_MIN + _LAT_MAX) / 2
        lon_center = (_LON_MIN + _LON_MAX) / 2
        lat_idx = max(0, min(int(math.floor((lat_center + 90.0) / TILE_SIZE_DEG)), 89))
        lon_idx = max(0, min(int(math.floor((lon_center + 180.0) / TILE_SIZE_DEG)), 179))
        return TileKey("viirs_aod", lat_idx, lon_idx)

    def test_load_tile_returns_grid_tile(self, tmp_path):
        reader = _make_reader(tmp_path)
        tile = reader.load_tile(self._key())
        assert isinstance(tile, GridTile)

    def test_source_is_viirs_aod(self, tmp_path):
        reader = _make_reader(tmp_path)
        assert reader.load_tile(self._key()).source == "viirs_aod"

    def test_timestamp_source_is_none(self, tmp_path):
        reader = _make_reader(tmp_path)
        assert reader.load_tile(self._key()).timestamp_source is None
