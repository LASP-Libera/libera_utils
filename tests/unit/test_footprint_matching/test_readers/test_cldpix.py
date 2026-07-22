"""Unit tests for CLDPIXReader (CERES CLDPIX imager-pixel reader).

Uses synthetic CLDPIX NetCDF4 fixtures created by ``make_cldpix_fixture``. The
fixture places all pixels near lat ≈ 40°, lon ≈ −15° (stored as 345° in the
0..360 convention) and deliberately reproduces the real product's *descending*
``valid_range`` on ``Eff_Cld_Pressure`` so the reader's auto-mask handling is
exercised.

Real CLDPIX files come from NASA CERES, e.g.
``CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc``.
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.types import BoundingBox, FmatchVariant, GridTile, OperationalMode, TileKey
from tests.test_data.footprint_matching.fixtures import make_cldpix_fixture

# Local tile containing the default pixel cluster (lon −15° == 345°).
_BBOX = BoundingBox(39.0, 41.0, -16.0, -14.0)


def _var_index(name: str) -> int:
    return [v.name for v in CLDPIXReader.VARIABLES].index(name)


def _finite(data: np.ndarray, name: str) -> np.ndarray:
    arr = data[_var_index(name)]
    return arr[np.isfinite(arr)]


class TestCLDPIXReaderClassAttributes:
    def test_reader_key(self):
        assert CLDPIXReader.READER_KEY == "cldpix"

    def test_resolution_km(self):
        assert CLDPIXReader.RESOLUTION_KM == 1.0

    def test_required_mode_is_imager(self):
        assert CLDPIXReader.REQUIRED_MODE == OperationalMode.IMAGER

    def test_required_variant_is_post_year_one(self):
        # RBSP CLDPIX products are unavailable in year one, so this reader must
        # only be active post-year-one (year-one IMAGER substitutes ERA5 fields).
        assert CLDPIXReader.REQUIRED_VARIANT is FmatchVariant.POST_YEAR_ONE

    def test_expected_variable_names(self):
        names = {v.name for v in CLDPIXReader.VARIABLES}
        # Surface-type variables (igbp_ecosystem, snow_map, ice_map) are
        # intentionally absent — the IGBP and NISE readers are authoritative.
        assert {
            "cloud_optical_depth",
            "cloud_effective_pressure",
            "cloud_particle_phase",
            "cloud_mask",
            "cloud_particle_radius",
        } <= names
        assert "igbp_ecosystem" not in names
        assert "snow_map" not in names
        assert "ice_map" not in names


class TestCLDPIXReaderLoadSpatialRegion:
    def test_returns_3d_array(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert data.ndim == 3
        assert data.shape[0] == len(CLDPIXReader.VARIABLES)
        assert data.dtype == np.float32

    def test_longitude_normalization_places_points(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.isfinite(data).any()

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(BoundingBox(39.0, 41.0, 14.0, 16.0))
        assert not np.isfinite(data).any()

    def test_continuous_values(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite(data, "cloud_optical_depth"), 4.0, atol=1e-4)
        assert np.allclose(_finite(data, "cloud_water_path"), 100.0, atol=1e-3)
        assert np.allclose(_finite(data, "cloud_effective_temperature"), 270.0, atol=1e-3)

    def test_pressure_survives_descending_valid_range(self, tmp_path):
        # Regression: Eff_Cld_Pressure has valid_range [1100, 10]; netCDF4
        # auto-masking would mask every value. The reader disables auto-masking
        # and normalizes the range, so the constant 800 hPa must survive.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        pressure = _finite(data, "cloud_effective_pressure")
        assert pressure.size > 0
        assert np.allclose(pressure, 800.0, atol=1e-3)

    def test_cloud_particle_radius_values(self, tmp_path):
        # Fixture writes Cld_Radius = 10.0 μm for all pixels; rasterized cells
        # should retain that constant after weighted_mean aggregation.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        radius = _finite(data, "cloud_particle_radius")
        assert radius.size > 0
        assert np.allclose(radius, 10.0, atol=1e-3)

    def test_categorical_values(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        assert np.allclose(_finite(data, "cloud_mask"), 1.0)
        assert np.allclose(_finite(data, "cloud_particle_phase"), 1.0)

    def test_southern_hemisphere_pixels_are_kept(self, tmp_path):
        # Regression: CLDPIX stores COLATITUDE under the name "Latitude", so a
        # southern-hemisphere pixel is written as colatitude > 90. Reading the
        # field as a true latitude and filtering on −90..90 silently dropped the
        # entire southern hemisphere. Pixels placed at 40° S must survive and
        # land in the southern tile, not the northern one.
        lats = np.full((3, 4), -40.0, dtype=np.float32)
        fixture = make_cldpix_fixture(tmp_path, lats=lats)
        reader = CLDPIXReader(fixture)

        southern, _, _ = reader._load_spatial_region(BoundingBox(-41.0, -39.0, -16.0, -14.0))
        assert np.isfinite(southern).any()

        # The mirrored northern tile must be empty — a sign-flip bug would put
        # the pixels there instead.
        northern, _, _ = reader._load_spatial_region(_BBOX)
        assert not np.isfinite(northern).any()

    def test_top_cloud_height_fill_is_dropped(self, tmp_path):
        # Regression: Top_Cld_Height is the one field with no valid_range, so
        # the float32 fill (3.4028235e38) had to be caught by the fill test
        # alone. Widening the stored float32 fill to float64 does not equal the
        # decimal literal the reader declares, so the fill leaked through as a
        # ~3.4e38 "cloud top height". Every retained value must be the
        # fixture's 6.0 km.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data, _, _ = reader._load_spatial_region(_BBOX)
        heights = _finite(data, "cloud_top_height")
        assert heights.size > 0
        assert np.allclose(heights, 6.0, atol=1e-3)


class TestCLDPIXReaderLoadTile:
    def test_load_tile_source_and_timestamp(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        key = TileKey("cldpix", int((40.0 + 90.0) // 2), int((-15.0 + 180.0) // 2))
        tile = reader.load_tile(key)
        assert isinstance(tile, GridTile)
        assert tile.source == "cldpix"
        assert tile.timestamp_source is None
