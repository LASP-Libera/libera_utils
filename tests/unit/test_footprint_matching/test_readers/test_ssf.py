"""Unit tests for SSFReader (CERES SSF / FLASHFlux footprint reader).

Uses synthetic SSF NetCDF4 fixtures created by ``make_ssf_fixture``. The fixture
clusters five footprints near lat ≈ 10–11°, lon ≈ −10° (stored as 350° in the
0..360 convention) plus one far-away footprint, so tests can verify longitude
normalization, fill handling, layer selection, and rasterization onto the 2°
tile grid.

Real SSF/FLASHFlux files come from NASA CERES, e.g.
``CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc``.
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey
from tests.test_data.footprint_matching.fixtures import make_ssf_fixture

# Local tile that contains the default footprint cluster (lon −10° == 350°).
_BBOX = BoundingBox(10.0, 12.0, -11.0, -9.0)


def _var_index(name: str) -> int:
    return [v.name for v in SSFReader.VARIABLES].index(name)


def _finite_values(data: np.ndarray, name: str) -> np.ndarray:
    arr = data[_var_index(name)]
    return np.sort(arr[np.isfinite(arr)])


class TestSSFReaderClassAttributes:
    def test_reader_key(self):
        assert SSFReader.READER_KEY == "ssf"

    def test_resolution_km(self):
        assert SSFReader.RESOLUTION_KM == 20.0

    def test_required_mode_is_imager_flash(self):
        # IMAGER_FLASH rank is low enough to be active for Flash, Imager, and
        # Imager-camera-time modes.
        assert SSFReader.REQUIRED_MODE == OperationalMode.IMAGER_FLASH

    def test_expected_variable_names(self):
        names = {v.name for v in SSFReader.VARIABLES}
        assert names == {
            "aerosol_optical_depth",
            "clear_coverage",
            "cloud_optical_depth",
            "cloud_water_particle_radius",
            "cloud_ice_particle_radius",
            "cloud_classification",
            "shortwave_adm_type",
            "longwave_adm_type",
        }


class TestSSFReaderLoadSpatialRegion:
    def test_returns_3d_array_in_variable_order(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, lats, lons = reader._load_spatial_region(_BBOX)
        assert data.ndim == 3
        assert data.shape[0] == len(SSFReader.VARIABLES)
        assert data.dtype == np.float32

    def test_longitude_normalization_places_points(self, tmp_path):
        # Footprints stored at 350° must be found in the −10° tile ...
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.isfinite(data).any()

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        # ... and absent from a tile at +170°, proving the 0..360 longitude was
        # converted rather than taken literally.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(BoundingBox(10.0, 12.0, 169.0, 171.0))
        assert not np.isfinite(data).any()

    def test_aerosol_optical_depth_values(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "aerosol_optical_depth"), [0.10, 0.20, 0.30, 0.40, 0.50], atol=1e-5)

    def test_cloud_optical_depth_uses_lower_layer(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "cloud_optical_depth"), [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)

    def test_cloud_classification_codes_preserved(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        codes = set(_finite_values(data, "cloud_classification").astype(int))
        assert codes == {1001, 1191}

    def test_cloud_water_particle_radius_uses_lower_layer(self, tmp_path):
        # Fixture defaults: lower-layer water radii are [5, 6, 7, 8, 9] μm for
        # the five clustered footprints; upper layer is fill → rasterized cells
        # contain only the lower-layer values, sorted for assertion stability.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        values = _finite_values(data, "cloud_water_particle_radius")
        assert values.size > 0
        assert np.all((values >= 5.0) & (values <= 9.0))

    def test_cloud_ice_particle_radius_uses_lower_layer(self, tmp_path):
        # Fixture defaults: lower-layer ice radii are [20, 25, 30, 35, 40] μm
        # for the five clustered footprints; upper layer is fill.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        values = _finite_values(data, "cloud_ice_particle_radius")
        assert values.size > 0
        assert np.all((values >= 20.0) & (values <= 40.0))

    def test_fill_values_dropped_for_shortwave_adm(self, tmp_path):
        # Two of the five clustered footprints have the int16 fill for the SW
        # ADM type; only three valid values remain.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        sw = _finite_values(data, "shortwave_adm_type")
        lw = _finite_values(data, "longwave_adm_type")
        assert sw.size == 3
        assert lw.size == 5
        assert np.allclose(sw, 50.0)


class TestSSFReaderLoadTileAndCache:
    def test_load_tile_source_and_timestamp(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        # Tile index for lat 11, lon -10 in the 2° global grid.
        key = TileKey("ssf", int((11.0 + 90.0) // 2), int((-10.0 + 180.0) // 2))
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.source == "ssf"
        assert tile.timestamp_source is None

    def test_points_parsed_once_and_cached(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        first = reader._load_points()
        second = reader._load_points()
        # Same cached object is reused across calls.
        assert first is second
