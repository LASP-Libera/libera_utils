"""Unit tests for SSFReader (CERES SSF / FLASHFlux footprint reader).

Uses synthetic SSF NetCDF4 fixtures created by ``make_ssf_fixture``. The fixture
clusters five footprints near lat ≈ 10–11°, lon ≈ −10° (stored as 350° in the
0..360 convention) plus one far-away footprint, so tests can verify longitude
normalization, fill handling, layer selection, and time matching.

Real SSF/FLASHFlux files come from NASA CERES, e.g.
``CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc``.
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, TimeMatch
from tests.test_data.footprint_matching.fixtures import SWATH_FIXTURE_TIME, make_ssf_fixture

# Box that contains the default footprint cluster (lon −10° == 350°).
_BBOX = BoundingBox(10.0, 12.0, -11.0, -9.0)
_T = SWATH_FIXTURE_TIME


def _values(reader: SSFReader, bbox: BoundingBox = _BBOX, time=_T) -> np.ndarray:
    """The ``(n_var, n)`` value rows of the samples in ``bbox`` at ``time``."""
    return reader.samples_in(bbox, time).values


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

    def test_weighted_mode_variables_are_int16(self):
        # Targets the dtype rule; the scene and ADM type codes are mode-aggregated int16, as in the source.
        for v in SSFReader.VARIABLES:
            assert v.dtype == ("int16" if v.aggregation == "weighted_mode" else "float32"), v.name

    def test_fields_are_the_base_set_then_the_imager_only_set(self):
        # Targets the field order; asserts VARIABLES lists every base field before the FMATCH-IMAGER-only ones.
        pinned = [v.only_modes is not None for v in SSFReader.VARIABLES]
        first_imager_only = pinned.index(True)
        assert not any(pinned[:first_imager_only])
        assert all(pinned[first_imager_only:])


class TestSSFReaderSamples:
    def test_returns_one_row_per_variable_in_variable_order(self, tmp_path):
        # Targets the sample layout/order; asserts a (n_var, n) float32 array with axis 0 = VARIABLES count
        # and one sample per clustered footprint.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.values.ndim == 2
        assert samples.values.shape == (len(SSFReader.VARIABLES), 5)
        assert samples.values.dtype == np.float32

    def test_longitude_normalization_places_points(self, tmp_path):
        # Footprints stored at 350° must be found in the −10° tile; asserts the loaded tile has finite (present) data.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.isfinite(data).any()

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        # ... and absent from a tile at +170°, proving the 0..360 longitude was
        # converted rather than taken literally.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, BoundingBox(10.0, 12.0, 169.0, 171.0))
        assert not np.isfinite(data).any()

    def test_aerosol_optical_depth_values(self, tmp_path):
        # Targets AOD passthrough; asserts the five clustered footprints are served as [0.10..0.50].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "aerosol_optical_depth"), [0.10, 0.20, 0.30, 0.40, 0.50], atol=1e-5)

    def test_cloud_optical_depth_uses_lower_layer(self, tmp_path):
        # Targets that cloud_optical_depth_lower reads the lower layer; asserts the finite values equal [1,2,4,8,16].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "cloud_optical_depth_lower"), [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)

    def test_cloud_classification_codes_preserved(self, tmp_path):
        # Targets cloud_classification code passthrough; asserts the finite codes are exactly {1001, 1191}.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        codes = set(_finite_values(data, "cloud_classification").astype(int))
        assert codes == {1001, 1191}

    def test_cloud_water_particle_radius_uses_lower_layer(self, tmp_path):
        # Fixture defaults: lower-layer water radii are [5, 6, 7, 8, 9] μm for the
        # five clustered footprints. The base `cloud_water_particle_radius_lower` spec
        # reads only the lower layer (index 0); the upper layer is exposed separately as
        # the IMAGER-only `cloud_water_particle_radius_upper` spec.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        values = _finite_values(data, "cloud_water_particle_radius_lower")
        assert values.size > 0
        assert np.all((values >= 5.0) & (values <= 9.0))

    def test_cloud_ice_particle_radius_uses_lower_layer(self, tmp_path):
        # Fixture defaults: lower-layer ice radii are [20, 25, 30, 35, 40] μm for the
        # five clustered footprints. The base spec reads only the lower layer.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        values = _finite_values(data, "cloud_ice_particle_radius_lower")
        assert values.size > 0
        assert np.all((values >= 20.0) & (values <= 40.0))

    def test_fill_values_dropped_for_shortwave_adm(self, tmp_path):
        # Two of the five clustered footprints have the int16 fill for the SW
        # ADM type; only three valid values remain.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        sw = _finite_values(data, "shortwave_adm_type")
        lw = _finite_values(data, "longwave_adm_type")
        assert sw.size == 3
        assert lw.size == 5
        assert np.allclose(sw, 50.0)


class TestSSFReaderExtendedImagerFields:
    """The FMATCH-IMAGER-only flattened fields read the correct second-axis index.

    All assertions use the deterministic fixture defaults for the five clustered
    footprints (per-footprint ``ramp`` = 0..4). Each clustered footprint is served as
    its own sample, so the sorted finite values equal the per-footprint source values.
    """

    def test_upper_layer_reads_index_one(self, tmp_path):
        # cloud_optical_depth lower = [1,2,4,8,16]; upper = lower + 1 = [2,3,5,9,17].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        lower = _finite_values(data, "cloud_optical_depth_lower")
        upper = _finite_values(data, "cloud_optical_depth_upper")
        assert np.allclose(lower, [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)
        assert np.allclose(upper, [2.0, 3.0, 5.0, 9.0, 17.0], atol=1e-4)

    def test_new_layered_field_both_layers(self, tmp_path):
        # cloud_top_pressure lower = 300 + 10*ramp; upper = 250 + 10*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "cloud_top_pressure_lower"), [300, 310, 320, 330, 340], atol=1e-3)
        assert np.allclose(_finite_values(data, "cloud_top_pressure_upper"), [250, 260, 270, 280, 290], atol=1e-3)

    def test_scalar_field_read(self, tmp_path):
        # match_aot is a 1-D (Footprints,) field: 0.15 + 0.1*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "match_aot"), [0.15, 0.25, 0.35, 0.45, 0.55], atol=1e-5)

    def test_aerosol_type_percentage_selects_type_index(self, tmp_path):
        # aerosol_type_percentage[:, t] = (t+1)*5 + ramp. Type 3 -> 20 + ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "aerosol_type_percentage_type3"), [20, 21, 22, 23, 24], atol=1e-3)
        assert np.allclose(_finite_values(data, "aerosol_type_percentage_type0"), [5, 6, 7, 8, 9], atol=1e-3)

    def test_surface_albedo_scalar_read(self, tmp_path):
        # surface_albedo is a 1-D (Footprints,) fraction (0..1): 0.1 + 0.05*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "surface_albedo"), [0.10, 0.15, 0.20, 0.25, 0.30], atol=1e-5)

    def test_toa_incoming_solar_radiation_scalar_read(self, tmp_path):
        # toa_incoming_solar_radiation is a 1-D (Footprints,) field (W/m^2): 1360 + ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(
            _finite_values(data, "toa_incoming_solar_radiation"), [1360, 1361, 1362, 1363, 1364], atol=1e-3
        )


class TestSSFReaderTimeMatchingAndCache:
    def test_policy_is_nearest_pass(self):
        assert SSFReader.TIME_MATCH is TimeMatch.NEAREST_PASS
        assert SSFReader.PASS_WINDOW == np.timedelta64(15, "m")
        assert SSFReader.MAX_TIME_OFFSET == np.timedelta64(3, "h")

    def test_samples_carry_source_and_footprint_times(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.source == "ssf"
        assert samples.timestamp_source is None
        assert np.all(samples.times == _T)

    def test_nearest_pass_is_served(self, tmp_path):
        # Six footprints: the first three at the fixture time, the last three twelve hours
        # later, all at the same cluster location. A footprint near either pass gets that
        # pass's three footprints only.
        lats = np.array([10.2, 10.4, 10.6, 10.2, 10.4, 10.6], dtype=np.float32)
        lons = np.full(6, 350.0, dtype=np.float32)
        late_time = _T + np.timedelta64(12, "h")
        times = np.array([_T, _T, _T, late_time, late_time, late_time], dtype="datetime64[ns]")
        reader = SSFReader(make_ssf_fixture(tmp_path, lats=lats, lons_0360=lons, times=times))
        near_early = reader.samples_in(_BBOX, _T + np.timedelta64(20, "m"))
        assert near_early.n_samples == 3
        assert np.allclose(_finite_values(near_early.values, "aerosol_optical_depth"), [0.10, 0.20, 0.30], atol=1e-4)
        near_late = reader.samples_in(_BBOX, late_time + np.timedelta64(20, "m"))
        assert near_late.n_samples == 3
        assert np.allclose(_finite_values(near_late.values, "aerosol_optical_depth"), [0.40, 0.50, 0.60], atol=1e-4)

    def test_beyond_max_offset_gives_no_samples(self, tmp_path):
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T + np.timedelta64(5, "h"))
        assert samples.n_samples == 0

    def test_granule_loaded_once_while_in_window(self, tmp_path):
        # Targets that a granule in the footprint's time window is parsed and indexed once;
        # asserts repeated queries reuse it, and that it is released once the window moves on.
        path = make_ssf_fixture(tmp_path)
        reader = SSFReader(path)
        reader.samples_in(_BBOX, _T)
        first = reader._loaded[path]
        reader.samples_in(_BBOX, _T + np.timedelta64(1, "h"))
        assert reader._loaded[path] is first
        reader.samples_in(_BBOX, _T + np.timedelta64(10, "h"))
        assert path not in reader._loaded


class TestSSFReaderSpacing:
    def test_served_resolution_is_the_measured_footprint_spacing(self, tmp_path):
        # The default fixture lays five footprints 0.2 deg (~22 km) apart along one scan,
        # so the measured along-scan spacing, not the nominal 20 km, is served.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.resolution_km == pytest.approx(22.2, rel=0.03)


class TestSSFReaderGeolocationChecks:
    @pytest.mark.parametrize("units", ["degrees", None])
    def test_latitude_units_other_than_degrees_north_raises(self, tmp_path, units):
        # Targets a granule whose footprint latitude does not declare degrees_north (or declares no units).
        path = make_ssf_fixture(tmp_path, latitude_units=units)
        with pytest.raises(ValueError, match="instrument_fov_latitude has units"):
            SSFReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize("name", ["instrument_fov_latitude", "aerosol_optical_depth", "cloud_classification"])
    def test_field_without_fill_value_raises(self, tmp_path, name):
        # Targets a geolocation, continuous and code field written with no _FillValue.
        path = make_ssf_fixture(tmp_path, omit_fill_value=(name,))
        with pytest.raises(ValueError, match=f"{name} has no _FillValue attribute"):
            SSFReader(path).samples_in(_BBOX, _T)
