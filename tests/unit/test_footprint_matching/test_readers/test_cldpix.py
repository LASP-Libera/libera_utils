"""Unit tests for CLDPIXReader (CERES CLDPIX imager-pixel reader).

Uses synthetic CLDPIX NetCDF4 fixtures created by ``make_cldpix_fixture``. The
fixture places all pixels near lat ≈ 40°, lon ≈ −15° (stored as 345° in the
0..360 convention) and deliberately reproduces the real product's *descending*
``valid_range`` on ``Eff_Cld_Pressure`` so the reader's auto-mask handling is
exercised.

Real CLDPIX files come from NASA CERES, e.g.
``CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc``.
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.types import BoundingBox, TimeMatch
from tests.test_data.footprint_matching.fixtures import SWATH_FIXTURE_TIME, make_cldpix_fixture

# Box containing the default pixel cluster (lon −15° == 345°).
_BBOX = BoundingBox(39.0, 41.0, -16.0, -14.0)
_T = SWATH_FIXTURE_TIME


def _values(reader: CLDPIXReader, bbox: BoundingBox = _BBOX, time=_T) -> np.ndarray:
    """The ``(n_var, n)`` value rows of the samples in ``bbox`` at ``time``."""
    return reader.samples_in(bbox, time).values


def _var_index(name: str) -> int:
    return [v.name for v in CLDPIXReader.VARIABLES].index(name)


def _finite(data: np.ndarray, name: str) -> np.ndarray:
    arr = data[_var_index(name)]
    return arr[np.isfinite(arr)]


class TestCLDPIXReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "cldpix".
        assert CLDPIXReader.READER_KEY == "cldpix"

    def test_resolution_km(self):
        # Targets the declared native resolution; asserts RESOLUTION_KM equals the imager-pixel 1.0 km value.
        assert CLDPIXReader.RESOLUTION_KM == 1.0

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

    def test_weighted_mode_variables_are_int16(self):
        # Targets the dtype rule; asserts mode-aggregated categories are int16 and every other variable float32.
        for v in CLDPIXReader.VARIABLES:
            assert v.dtype == ("int16" if v.aggregation == "weighted_mode" else "float32"), v.name


class TestCLDPIXReaderSamples:
    def test_returns_one_row_per_variable(self, tmp_path):
        # Targets the sample layout/dtype; asserts a (n_var, n_pixels) float32 array, every pixel served.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.values.ndim == 2
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), 12)
        assert samples.values.dtype == np.float32
        assert samples.times is not None
        assert samples.times.shape == (12,)

    def test_longitude_normalization_places_points(self, tmp_path):
        # Targets 0..360 lon normalization; asserts pixels at 345° land in the −15° bbox as finite values.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.isfinite(data).any()

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        # Targets spatial filtering; asserts a bbox away from the pixel cluster yields no finite values.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, BoundingBox(39.0, 41.0, 14.0, 16.0))
        assert not np.isfinite(data).any()

    def test_continuous_values(self, tmp_path):
        # Targets continuous-field scaling; asserts optical depth, water path, and eff. temp match
        # the fixture constants pixel by pixel.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite(data, "cloud_optical_depth"), 4.0, atol=1e-4)
        assert np.allclose(_finite(data, "cloud_water_path"), 100.0, atol=1e-3)
        assert np.allclose(_finite(data, "cloud_effective_temperature"), 270.0, atol=1e-3)

    def test_pressure_survives_descending_valid_range(self, tmp_path):
        # Regression: Eff_Cld_Pressure has valid_range [1100, 10]; netCDF4
        # auto-masking would mask every value. The reader disables auto-masking
        # and normalizes the range, so the constant 800 hPa must survive.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        pressure = _finite(data, "cloud_effective_pressure")
        assert pressure.size > 0
        assert np.allclose(pressure, 800.0, atol=1e-3)

    def test_cloud_particle_radius_values(self, tmp_path):
        # Fixture writes Cld_Radius = 10.0 μm for all pixels; every served pixel keeps it.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        radius = _finite(data, "cloud_particle_radius")
        assert radius.size > 0
        assert np.allclose(radius, 10.0, atol=1e-3)

    def test_categorical_values(self, tmp_path):
        # Targets categorical-field handling; asserts cloud_mask and cloud_particle_phase retain the fixture's 1.0.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
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

        southern = _values(reader, BoundingBox(-41.0, -39.0, -16.0, -14.0))
        assert np.isfinite(southern).any()

        # The mirrored northern tile must be empty — a sign-flip bug would put
        # the pixels there instead.
        northern = _values(reader, _BBOX)
        assert not np.isfinite(northern).any()

    def test_top_cloud_height_fill_is_dropped(self, tmp_path):
        # Regression: Top_Cld_Height is the one field with no valid_range, so
        # the float32 fill (3.4028235e38) had to be caught by the fill test
        # alone. Widening the stored float32 fill to float64 does not equal the
        # decimal literal the reader declares, so the fill leaked through as a
        # ~3.4e38 "cloud top height". Every retained value must be the
        # fixture's 6.0 km.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        heights = _finite(data, "cloud_top_height")
        assert heights.size > 0
        assert np.allclose(heights, 6.0, atol=1e-3)


class TestCLDPIXReaderTimeMatching:
    def test_policy_is_nearest_pass(self):
        assert CLDPIXReader.TIME_MATCH is TimeMatch.NEAREST_PASS
        assert CLDPIXReader.PASS_WINDOW == np.timedelta64(15, "m")
        assert CLDPIXReader.MAX_TIME_OFFSET == np.timedelta64(3, "h")

    def test_samples_carry_source_and_scanline_times(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.source == "cldpix"
        assert samples.timestamp_source is None
        assert np.all(samples.times == _T)

    def test_nearest_pass_is_served(self, tmp_path):
        # Two granules over the same pixels twelve hours apart: a footprint near the
        # first pass gets only the first granule's pixels, and vice versa.
        n_scan = 3
        early = make_cldpix_fixture(tmp_path, times=np.full(n_scan, _T, dtype="datetime64[ns]"), filename="early.nc")
        late_time = _T + np.timedelta64(12, "h")
        late = make_cldpix_fixture(
            tmp_path, times=np.full(n_scan, late_time, dtype="datetime64[ns]"), filename="late.nc"
        )
        reader = CLDPIXReader([early, late])
        near_early = reader.samples_in(_BBOX, _T + np.timedelta64(30, "m"))
        assert near_early.n_samples == 12
        assert np.all(near_early.times == _T)
        near_late = reader.samples_in(_BBOX, late_time - np.timedelta64(30, "m"))
        assert near_late.n_samples == 12
        assert np.all(near_late.times == late_time)

    def test_beyond_max_offset_gives_no_samples(self, tmp_path):
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T + np.timedelta64(6, "h"))
        assert samples.n_samples == 0
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), 0)


class TestCLDPIXReaderGranuleDate:
    def test_missing_time_coverage_start_raises(self, tmp_path):
        import netCDF4

        path = make_cldpix_fixture(tmp_path)
        with netCDF4.Dataset(str(path), "r+") as ds:
            ds.delncattr("time_coverage_start")
        with pytest.raises(ValueError, match="time_coverage_start"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    def test_times_are_dated_by_the_granule(self, tmp_path):
        # The fixture writes fractional hours of 2026-01-01 and dates them in the attribute;
        # every served pixel decodes back to the fixture instant.
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert np.all(samples.times == _T)


class TestCLDPIXReaderGeolocationChecks:
    def test_latitude_valid_range_other_than_colatitude_raises(self, tmp_path):
        # Targets a granule whose Latitude declares a true -90..90 range; converting it as colatitude
        # would mirror the data, so the reader raises instead.
        path = make_cldpix_fixture(tmp_path, latitude_valid_range=(-90.0, 90.0))
        with pytest.raises(ValueError, match=r"Latitude declares valid_range"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    def test_latitude_without_valid_range_raises(self, tmp_path):
        # Targets a granule whose Latitude declares no valid_range; the convention cannot be confirmed.
        path = make_cldpix_fixture(tmp_path, latitude_valid_range=None)
        with pytest.raises(ValueError, match=r"Latitude declares valid_range None"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize("name", ["Latitude", "Eff_Cld_Optical_Depth", "Cloud_Particle_Phase"])
    def test_field_without_fill_value_raises(self, tmp_path, name):
        # Targets a geolocation, continuous and categorical field written with no _FillValue.
        path = make_cldpix_fixture(tmp_path, omit_fill_value=(name,))
        with pytest.raises(ValueError, match=f"{name} has no _FillValue attribute"):
            CLDPIXReader(path).samples_in(_BBOX, _T)
