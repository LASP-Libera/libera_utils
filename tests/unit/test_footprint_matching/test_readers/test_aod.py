"""Unit tests for VIIRSAODReader.

Uses synthetic AERDB_D3_VIIRS_NOAA20 Deep Blue aerosol NetCDF4 fixtures created
by ``make_aod_noaa20_fixture``.

Real files come from the NASA Deep Blue single-sensor Level-3 daily product, e.g.
``AERDB_D3_VIIRS_NOAA20.A2026001.002.2026005001030.nc``; the reader reads the
root-level ``Aerosol_Optical_Thickness_550_Land_Ocean_Mean`` (AOD) and
``Aerosol_Type_Land_Ocean_Mode`` (aerosol type) fields.
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
_AEROSOL_TYPE = 1  # smoke

# Axis-0 indices into the stacked (n_var, n_lat, n_lon) array, in VARIABLES order.
_AOD_AXIS = 0
_TYPE_AXIS = 1


def _make_reader(tmp_path, **kwargs) -> VIIRSAODReader:
    kwargs.setdefault("aod_fill", _AOD_FILL)
    kwargs.setdefault("aerosol_type_value", _AEROSOL_TYPE)
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

    def test_reads_aod_and_aerosol_type(self):
        assert len(VIIRSAODReader.VARIABLES) == 2
        aod, aerosol_type = VIIRSAODReader.VARIABLES
        assert aod.name == "aod_550"
        assert aod.dtype == "float32"
        assert aod.aggregation == "weighted_log_mean"
        assert aod.n_categories is None
        assert aerosol_type.name == "aerosol_type"
        assert aerosol_type.dtype == "int16"
        assert aerosol_type.aggregation == "weighted_mode"
        assert aerosol_type.n_categories == 8

    def test_ranked_aerosol_type_companions(self):
        # The ranked types are derived product outputs, not read variables.
        additional = {spec.name: spec for spec in VIIRSAODReader.ADDITIONAL_PRODUCT_VARIABLES}
        assert set(additional) == {
            "aerosol_type_primary",
            "aerosol_type_secondary",
            "aerosol_type_tertiary",
        }
        for name, agg in (
            ("aerosol_type_primary", "weighted_mode_primary"),
            ("aerosol_type_secondary", "weighted_mode_secondary"),
            ("aerosol_type_tertiary", "weighted_mode_tertiary"),
        ):
            assert additional[name].aggregation == agg
            assert additional[name].dtype == "int16"
            assert additional[name].n_categories == 8

    def test_categorical_variables_get_no_standard_deviation_companion(self):
        # aod_550 is continuous -> gains a std-dev companion; the aerosol-type
        # (weighted_mode) variables must not.
        names = {spec.name for spec in VIIRSAODReader.product_variable_specs()}
        assert "aod_550_standard_deviation" in names
        for categorical in (
            "aerosol_type",
            "aerosol_type_primary",
            "aerosol_type_secondary",
            "aerosol_type_tertiary",
        ):
            assert f"{categorical}_standard_deviation" not in names


class TestVIIRSAODReaderLoadSpatialRegion:
    def test_returns_3d_data_stacked_in_variables_order(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, lats, lons = reader._load_spatial_region(_full_bbox())
        assert data.ndim == 3
        assert data.shape == (2, _N_LAT, _N_LON)
        assert data.dtype == np.float32

    def test_lat_is_ascending(self, tmp_path):
        reader = _make_reader(tmp_path)
        _, lats, _ = reader._load_spatial_region(_full_bbox())
        assert np.all(np.diff(lats) >= 0)

    def test_non_fill_aod_values_preserved(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)

    def test_non_fill_aerosol_type_values_preserved(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)

    def test_fill_pixel_becomes_nan_in_both_fields(self, tmp_path):
        reader = _make_reader(tmp_path, include_fill_pixel=True)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.isnan(data[_AOD_AXIS, 0, 0])
        assert np.isnan(data[_TYPE_AXIS, 0, 0])
        # Every other pixel of each field remains valid.
        assert np.isfinite(data[_AOD_AXIS]).sum() == _N_LAT * _N_LON - 1
        assert np.isfinite(data[_TYPE_AXIS]).sum() == _N_LAT * _N_LON - 1

    def test_aod_out_of_range_becomes_nan(self, tmp_path):
        reader = _make_reader(tmp_path, aod_fill=99.0, include_fill_pixel=False)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        # AOD valid max is 5.0, so 99.0 must be masked; aerosol type is unaffected.
        assert np.all(np.isnan(data[_AOD_AXIS]))
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)

    def test_out_of_range_aerosol_type_becomes_nan(self, tmp_path):
        # Category 99 is outside the valid [0, 7] range -> NaN; AOD is unaffected.
        reader = _make_reader(tmp_path, aerosol_type_value=99, include_fill_pixel=False)
        data, _, _ = reader._load_spatial_region(_full_bbox())
        assert np.all(np.isnan(data[_TYPE_AXIS]))
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)

    def test_empty_result_outside_bbox(self, tmp_path):
        reader = _make_reader(tmp_path)
        data, lats, lons = reader._load_spatial_region(BoundingBox(-60.0, -58.0, 170.0, 172.0))
        assert data.shape == (2, 0, 0)
        assert data.size == 0


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
