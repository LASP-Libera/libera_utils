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
        # Targets the reader registry key; asserts READER_KEY equals the expected "ssf".
        assert SSFReader.READER_KEY == "ssf"

    def test_resolution_km(self):
        # Targets the SSF footprint resolution; asserts RESOLUTION_KM equals 20.0 km.
        assert SSFReader.RESOLUTION_KM == 20.0

    def test_base_variable_names_present(self):
        # The original FLASH+ base fields (only_modes=None) are unchanged; asserts their name set matches the expected 8.
        names = {v.name for v in SSFReader.VARIABLES if v.only_modes is None}
        assert names == {
            "aerosol_optical_depth",
            "clear_coverage",
            "cloud_optical_depth_lower",
            "cloud_water_particle_radius_lower",
            "cloud_ice_particle_radius_lower",
            "cloud_classification",
            "shortwave_adm_type",
            "longwave_adm_type",
        }

    def test_extended_fields_are_imager_only(self):
        # The extended cloud/aerosol/surface/TOA fields are pinned to FMATCH-IMAGER via
        # only_modes; spot-check representative names, including the two 1-D scalar
        # fields (surface_albedo, toa_incoming_solar_radiation).
        names = {v.name for v in SSFReader.VARIABLES}
        for name in (
            "layer_coverage_lower",
            "layer_coverage_upper",
            "cloud_top_pressure_lower",
            "cloud_optical_depth_upper",
            "match_aot",
            "aerosol_type_percentage_type0",
            "aerosol_type_percentage_type6",
            "surface_albedo",
            "toa_incoming_solar_radiation",
        ):
            assert name in names, f"missing extended field {name}"
        imager_only = [v for v in SSFReader.VARIABLES if v.name in names and v.only_modes is not None]
        assert all(v.only_modes == (OperationalMode.IMAGER,) for v in imager_only)
        # The aerosol-type axis is fully flattened (7 types); surface_albedo and the TOA
        # field are genuinely 1-D, so each appears exactly once (no _bandN flattening).
        assert sum(n.startswith("aerosol_type_percentage_type") for n in names) == 7
        assert sum(n == "surface_albedo" for n in names) == 1
        assert sum(n == "toa_incoming_solar_radiation" for n in names) == 1


class TestSSFReaderLoadSpatialRegion:
    def test_returns_3d_array_in_variable_order(self, tmp_path):
        # Targets region loader output shape/order; asserts a 3-D float32 array with axis 0 = VARIABLES count.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, lats, lons = reader._load_spatial_region(_BBOX)
        assert data.ndim == 3
        assert data.shape[0] == len(SSFReader.VARIABLES)
        assert data.dtype == np.float32

    def test_longitude_normalization_places_points(self, tmp_path):
        # Footprints stored at 350° must be found in the −10° tile; asserts the loaded tile has finite (present) data.
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
        # Targets AOD passthrough; asserts the five clustered footprints rasterize to [0.10..0.50].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "aerosol_optical_depth"), [0.10, 0.20, 0.30, 0.40, 0.50], atol=1e-5)

    def test_cloud_optical_depth_uses_lower_layer(self, tmp_path):
        # Targets that cloud_optical_depth_lower reads the lower layer; asserts the finite values equal [1,2,4,8,16].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "cloud_optical_depth_lower"), [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)

    def test_cloud_classification_codes_preserved(self, tmp_path):
        # Targets cloud_classification code passthrough; asserts the finite codes are exactly {1001, 1191}.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        codes = set(_finite_values(data, "cloud_classification").astype(int))
        assert codes == {1001, 1191}

    def test_cloud_water_particle_radius_uses_lower_layer(self, tmp_path):
        # Fixture defaults: lower-layer water radii are [5, 6, 7, 8, 9] μm for the
        # five clustered footprints. The base `cloud_water_particle_radius_lower` spec
        # reads only the lower layer (index 0); the upper layer is exposed separately as
        # the IMAGER-only `cloud_water_particle_radius_upper` spec.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        values = _finite_values(data, "cloud_water_particle_radius_lower")
        assert values.size > 0
        assert np.all((values >= 5.0) & (values <= 9.0))

    def test_cloud_ice_particle_radius_uses_lower_layer(self, tmp_path):
        # Fixture defaults: lower-layer ice radii are [20, 25, 30, 35, 40] μm for the
        # five clustered footprints. The base spec reads only the lower layer.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        values = _finite_values(data, "cloud_ice_particle_radius_lower")
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


class TestSSFReaderExtendedImagerFields:
    """The FMATCH-IMAGER-only flattened fields read the correct second-axis index.

    All assertions use the deterministic fixture defaults for the five clustered
    footprints (per-footprint ``ramp`` = 0..4). Each clustered footprint rasterizes
    to its own 0.2° cell, so the sorted finite cell values equal the per-footprint
    source values.
    """

    def test_upper_layer_reads_index_one(self, tmp_path):
        # cloud_optical_depth lower = [1,2,4,8,16]; upper = lower + 1 = [2,3,5,9,17].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        lower = _finite_values(data, "cloud_optical_depth_lower")
        upper = _finite_values(data, "cloud_optical_depth_upper")
        assert np.allclose(lower, [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)
        assert np.allclose(upper, [2.0, 3.0, 5.0, 9.0, 17.0], atol=1e-4)

    def test_new_layered_field_both_layers(self, tmp_path):
        # cloud_top_pressure lower = 300 + 10*ramp; upper = 250 + 10*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "cloud_top_pressure_lower"), [300, 310, 320, 330, 340], atol=1e-3)
        assert np.allclose(_finite_values(data, "cloud_top_pressure_upper"), [250, 260, 270, 280, 290], atol=1e-3)

    def test_scalar_field_read(self, tmp_path):
        # match_aot is a 1-D (Footprints,) field: 0.15 + 0.1*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "match_aot"), [0.15, 0.25, 0.35, 0.45, 0.55], atol=1e-5)

    def test_aerosol_type_percentage_selects_type_index(self, tmp_path):
        # aerosol_type_percentage[:, t] = (t+1)*5 + ramp. Type 3 -> 20 + ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "aerosol_type_percentage_type3"), [20, 21, 22, 23, 24], atol=1e-3)
        assert np.allclose(_finite_values(data, "aerosol_type_percentage_type0"), [5, 6, 7, 8, 9], atol=1e-3)

    def test_surface_albedo_scalar_read(self, tmp_path):
        # surface_albedo is a 1-D (Footprints,) fraction (0..1): 0.1 + 0.05*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite_values(data, "surface_albedo"), [0.10, 0.15, 0.20, 0.25, 0.30], atol=1e-5)

    def test_toa_incoming_solar_radiation_scalar_read(self, tmp_path):
        # toa_incoming_solar_radiation is a 1-D (Footprints,) field (W/m^2): 1360 + ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(
            _finite_values(data, "toa_incoming_solar_radiation"), [1360, 1361, 1362, 1363, 1364], atol=1e-3
        )


class TestSSFReaderLoadTileAndCache:
    def test_load_tile_source_and_timestamp(self, tmp_path):
        # Targets the public load_tile API; asserts it returns a GridTile with source "ssf" and no timestamp_source.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        # Tile index for lat 11, lon -10 in the 2° global grid.
        key = TileKey("ssf", int((11.0 + 90.0) // 2), int((-10.0 + 180.0) // 2))
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.source == "ssf"
        assert tile.timestamp_source is None

    def test_points_parsed_once_and_cached(self, tmp_path):
        # Targets that _load_points parses once and caches; asserts repeated calls return the same cached object.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        first = reader._load_points()
        second = reader._load_points()
        # Same cached object is reused across calls.
        assert first is second
