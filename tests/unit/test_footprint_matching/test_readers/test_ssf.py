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
        """``SSFReader.READER_KEY`` is ``"ssf"``.

        The registry, the product reader sets and every sample's ``source`` tag find this reader by that key, so a
        changed key would leave the SSF fields unread in every product that lists ``"ssf"``.

        Asserts the class attribute equals ``"ssf"``.
        """
        assert SSFReader.READER_KEY == "ssf"

    def test_resolution_km(self):
        """``SSFReader.RESOLUTION_KM`` is the 20 km CERES footprint scale at nadir.

        It is the spacing the coverage measure falls back on when a granule's own footprint spacing cannot be measured,
        so a wrong value would over- or under-credit how much of a Libera footprint SSF covers.

        Asserts the class attribute equals 20.0.
        """
        assert SSFReader.RESOLUTION_KM == 20.0

    def test_base_variable_names_present(self):
        """``SSFReader.VARIABLES`` holds exactly the eight base SSF fields that are not pinned to a product.

        The base fields are carried by every product SSF feeds (FMATCH-IMAGER-FLASH, FMATCH-IMAGER and FMATCH-IMAGER-
        CAMTIME); a field missing or added here changes the variable set of all three products.

        Collects the names of the specs whose ``only_modes`` is None and asserts they equal the expected set of eight.
        """
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
        """``SSFReader.VARIABLES`` pins the extended cloud, aerosol, surface and TOA fields to FMATCH-IMAGER.

        These fields belong to FMATCH-IMAGER alone; one leaking into FMATCH-IMAGER-FLASH or FMATCH-IMAGER-CAMTIME would
        put variables in those products that their definitions do not carry, and a mis-flattened multi-valued field
        would drop or duplicate an aerosol type.

        Asserts representative extended names are present, including the 1-D ``surface_albedo`` and
        ``toa_incoming_solar_radiation``, that every one of them has ``only_modes == (IMAGER,)``, that seven
        ``aerosol_type_percentage_typeN`` fields exist and that each 1-D field appears exactly once.
        """
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
        """``SSFReader.VARIABLES`` declares the mode-aggregated fields as int16 and every other field as float32.

        The scene and ADM type fields are encoded CERES integer codes; written as float they would no longer match the
        source codes exactly, and a continuous field written as an integer would lose its fraction.

        Asserts, for every spec, ``dtype`` is ``"int16"`` when ``aggregation`` is ``"weighted_mode"`` and ``"float32"``
        otherwise.
        """
        for v in SSFReader.VARIABLES:
            assert v.dtype == ("int16" if v.aggregation == "weighted_mode" else "float32"), v.name

    def test_fields_are_the_base_set_then_the_imager_only_set(self):
        """``SSFReader.VARIABLES`` lists every base field before every FMATCH-IMAGER-only field.

        The reader stacks value rows in this order, so the field order is part of the contract that pairs each value row
        with its variable name.

        Finds the first spec with ``only_modes`` set and asserts no spec before it is pinned and every spec from it on
        is.
        """
        pinned = [v.only_modes is not None for v in SSFReader.VARIABLES]
        first_imager_only = pinned.index(True)
        assert not any(pinned[:first_imager_only])
        assert all(pinned[first_imager_only:])


class TestSSFReaderSamples:
    def test_returns_one_row_per_variable_in_variable_order(self, tmp_path):
        """``SSFReader.samples_in`` returns float32 ``(n_var, n)`` values: a row per variable, a column per footprint.

        Each footprint is served as its own native sample for the PSF weigher; a wrong shape would misalign values with
        variable names or merge footprints before they are weighted.

        Reads the default fixture over the cluster box and asserts the values are 2-D, shaped ``(len(VARIABLES), 5)``
        for the five clustered footprints, and float32, and that the rows at the positions of three fields with
        distinct fixture values (a base field near the start, a layered field and an imager-only field near the end)
        hold those fields' values.
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.values.ndim == 2
        assert samples.values.shape == (len(SSFReader.VARIABLES), 5)
        assert samples.values.dtype == np.float32
        names = [v.name for v in SSFReader.VARIABLES]
        for name, expected in (
            ("aerosol_optical_depth", [0.10, 0.20, 0.30, 0.40, 0.50]),
            ("cloud_optical_depth_lower", [1.0, 2.0, 4.0, 8.0, 16.0]),
            ("toa_incoming_solar_radiation", [1360.0, 1361.0, 1362.0, 1363.0, 1364.0]),
        ):
            assert np.allclose(np.sort(samples.values[names.index(name)]), expected, atol=1e-3), name

    def test_longitude_normalization_places_points(self, tmp_path):
        """``SSFReader.samples_in`` finds footprints stored at 350 degrees longitude in a box at -10 degrees.

        SSF stores longitude in 0..360 while boxes use -180..180; without normalization every western-hemisphere
        footprint would be missed and those Libera footprints would get no SSF match.

        Reads the default fixture over the cluster box at -11..-9 degrees and asserts all five clustered footprints are
        served, each with its longitude converted to -10 degrees.
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.n_samples == 5
        assert np.allclose(samples.lons, -10.0, atol=1e-4)

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        """``SSFReader.samples_in`` serves nothing in a box at +170 degrees for footprints stored at 350 degrees.

        If longitude were not converted, or were converted wrongly, footprints could be served at the wrong place on
        Earth and attributed to unrelated Libera footprints.

        Reads the default fixture over a box at 169..171 degrees longitude and asserts no footprint is served.
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        assert reader.samples_in(BoundingBox(10.0, 12.0, 169.0, 171.0), _T).n_samples == 0

    def test_aerosol_optical_depth_values(self, tmp_path):
        """``SSFReader.samples_in`` serves ``aerosol_optical_depth`` with the footprints' source values unchanged.

        Aerosol optical depth is passed straight through to the PSF weigher, so any scaling or mix-up between footprints
        would bias the product's aerosol field.

        Reads the default fixture over the cluster box and asserts the sorted finite values equal [0.10, 0.20, 0.30,
        0.40, 0.50].
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "aerosol_optical_depth"), [0.10, 0.20, 0.30, 0.40, 0.50], atol=1e-5)

    def test_cloud_optical_depth_uses_lower_layer(self, tmp_path):
        """``SSFReader.samples_in`` fills ``cloud_optical_depth_lower`` from the lower layer (index 0).

        SSF stores cloud optical depth per cloud layer; reading the upper layer would report the wrong cloud's optical
        depth under the lower-layer name.

        Reads the default fixture, whose upper layer is the lower plus one, and asserts the finite values equal the
        lower-layer [1, 2, 4, 8, 16].
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "cloud_optical_depth_lower"), [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)

    def test_cloud_classification_codes_preserved(self, tmp_path):
        """``SSFReader.samples_in`` serves ``cloud_classification`` codes unchanged.

        The classification is an encoded CERES identifier that downstream users decode; a code altered by scaling or
        rounding would decode to a different scene.

        Reads the default fixture over the cluster box and asserts the set of finite codes is exactly {1001, 1191}.
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        codes = set(_finite_values(data, "cloud_classification").astype(int))
        assert codes == {1001, 1191}

    def test_cloud_water_particle_radius_uses_lower_layer(self, tmp_path):
        """``SSFReader.samples_in`` fills ``cloud_water_particle_radius_lower`` from the lower cloud layer.

        Water and ice radii are stored per layer; reading the upper layer would report the wrong cloud's droplet size
        under the lower-layer name.

        Reads the default fixture, whose upper layer is the lower plus one, and asserts the finite values equal the
        lower-layer [5, 6, 7, 8, 9] um.
        """
        # Fixture defaults: lower-layer water radii are [5, 6, 7, 8, 9] μm for the
        # five clustered footprints. The base `cloud_water_particle_radius_lower` spec
        # reads only the lower layer (index 0); the upper layer is exposed separately as
        # the IMAGER-only `cloud_water_particle_radius_upper` spec.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        values = _finite_values(data, "cloud_water_particle_radius_lower")
        assert np.allclose(values, [5.0, 6.0, 7.0, 8.0, 9.0], atol=1e-4)

    def test_cloud_ice_particle_radius_uses_lower_layer(self, tmp_path):
        """``SSFReader.samples_in`` fills ``cloud_ice_particle_radius_lower`` from the lower cloud layer.

        Reading the upper layer would report the wrong cloud's ice particle size under the lower-layer name.

        Reads the default fixture, whose upper layer is the lower plus two, and asserts the finite values equal the
        lower-layer [20, 25, 30, 35, 40] um.
        """
        # Fixture defaults: lower-layer ice radii are [20, 25, 30, 35, 40] μm for the
        # five clustered footprints. The base spec reads only the lower layer.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        values = _finite_values(data, "cloud_ice_particle_radius_lower")
        assert np.allclose(values, [20.0, 25.0, 30.0, 35.0, 40.0], atol=1e-4)

    def test_fill_values_dropped_for_shortwave_adm(self, tmp_path):
        """``SSFReader.samples_in`` turns the int16 fill in ``shortwave_adm_type`` into NaN.

        A fill sentinel left as a number would be counted in the weighted mode as if it were a real ADM type and could
        become a footprint's dominant code.

        Reads the default fixture, where two of the five clustered footprints hold the shortwave fill, and asserts three
        finite shortwave codes, all 50, remain while all five longwave codes are kept.
        """
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
        """``SSFReader.samples_in`` fills ``cloud_optical_depth_upper`` from layer index 1 and the lower from index 0.

        The upper and lower layers describe different clouds; swapped or repeated layers would give FMATCH-IMAGER a
        wrong two-layer cloud picture.

        Reads the default fixture and asserts the lower values equal [1, 2, 4, 8, 16] and the upper values equal [2, 3,
        5, 9, 17].
        """
        # cloud_optical_depth lower = [1,2,4,8,16]; upper = lower + 1 = [2,3,5,9,17].
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        lower = _finite_values(data, "cloud_optical_depth_lower")
        upper = _finite_values(data, "cloud_optical_depth_upper")
        assert np.allclose(lower, [1.0, 2.0, 4.0, 8.0, 16.0], atol=1e-4)
        assert np.allclose(upper, [2.0, 3.0, 5.0, 9.0, 17.0], atol=1e-4)

    def test_cloud_top_pressure_reads_both_layers(self, tmp_path):
        """``SSFReader.samples_in`` serves ``cloud_top_pressure_lower`` and ``_upper`` each from its own layer.

        A layered field flattened into ``_lower`` and ``_upper`` variables must take each from its own layer, or the two
        cloud-top pressures would be swapped or equal.

        Reads the default fixture and asserts the lower values equal [300..340] and the upper values equal [250..290] in
        steps of 10.
        """
        # cloud_top_pressure lower = 300 + 10*ramp; upper = 250 + 10*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "cloud_top_pressure_lower"), [300, 310, 320, 330, 340], atol=1e-3)
        assert np.allclose(_finite_values(data, "cloud_top_pressure_upper"), [250, 260, 270, 280, 290], atol=1e-3)

    def test_scalar_field_read(self, tmp_path):
        """``SSFReader.samples_in`` reads the 1-D ``match_aot`` field without indexing a second axis.

        A 1-D field must be read whole; indexing it as if it had layers would fail or serve the wrong footprint's value.

        Reads the default fixture and asserts the finite ``match_aot`` values equal [0.15, 0.25, 0.35, 0.45, 0.55].
        """
        # match_aot is a 1-D (Footprints,) field: 0.15 + 0.1*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "match_aot"), [0.15, 0.25, 0.35, 0.45, 0.55], atol=1e-5)

    def test_aerosol_type_percentage_selects_type_index(self, tmp_path):
        """``SSFReader.samples_in`` fills each ``aerosol_type_percentage_typeN`` from column N of the aerosol-type axis.

        Each column is the percentage of a different aerosol type; a wrong column would attribute one type's share to
        another.

        Reads the default fixture, where column t holds (t + 1) * 5 plus the footprint index, and asserts type 3 serves
        [20..24] and type 0 serves [5..9].
        """
        # aerosol_type_percentage[:, t] = (t+1)*5 + ramp. Type 3 -> 20 + ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "aerosol_type_percentage_type3"), [20, 21, 22, 23, 24], atol=1e-3)
        assert np.allclose(_finite_values(data, "aerosol_type_percentage_type0"), [5, 6, 7, 8, 9], atol=1e-3)

    def test_surface_albedo_scalar_read(self, tmp_path):
        """``SSFReader.samples_in`` reads the 1-D ``surface_albedo`` fraction unchanged.

        Surface albedo is a 0..1 fraction passed to FMATCH-IMAGER; a misread or rescaled value would bias the surface
        reflectance carried in the product.

        Reads the default fixture and asserts the finite values equal [0.10, 0.15, 0.20, 0.25, 0.30].
        """
        # surface_albedo is a 1-D (Footprints,) fraction (0..1): 0.1 + 0.05*ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite_values(data, "surface_albedo"), [0.10, 0.15, 0.20, 0.25, 0.30], atol=1e-5)

    def test_toa_incoming_solar_radiation_scalar_read(self, tmp_path):
        """``SSFReader.samples_in`` reads the 1-D ``toa_incoming_solar_radiation`` field unchanged, in W/m^2.

        TOA insolation is the reference against which reflected shortwave flux is compared; a misread value would skew
        any albedo derived from the product.

        Reads the default fixture and asserts the finite values equal [1360..1364] W/m^2.
        """
        # toa_incoming_solar_radiation is a 1-D (Footprints,) field (W/m^2): 1360 + ramp.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(
            _finite_values(data, "toa_incoming_solar_radiation"), [1360, 1361, 1362, 1363, 1364], atol=1e-3
        )


class TestSSFReaderTimeMatchingAndCache:
    def test_policy_is_nearest_pass(self):
        """``SSFReader`` matches by nearest pass, with a 15 minute pass window and a 3 hour maximum offset.

        SSF sees a location on several passes a day; the window keeps one pass together without admitting the next, and
        the maximum offset stops a pass hours away being matched to a Libera footprint.

        Asserts ``TIME_MATCH`` is ``NEAREST_PASS``, ``PASS_WINDOW`` is 15 minutes and ``MAX_TIME_OFFSET`` is 3 hours.
        """
        assert SSFReader.TIME_MATCH is TimeMatch.NEAREST_PASS
        assert SSFReader.PASS_WINDOW == np.timedelta64(15, "m")
        assert SSFReader.MAX_TIME_OFFSET == np.timedelta64(3, "h")

    def test_samples_carry_source_and_footprint_times(self, tmp_path):
        """``SSFReader.samples_in`` tags samples with source ``"ssf"``, no timestamp source and footprint times.

        The per-sample times are what the nearest-pass selection and any time-offset reporting rely on; a missing or
        wrong time would match footprints from the wrong pass.

        Reads the default fixture, whose footprints are all at the fixture time, and asserts the source is ``"ssf"``,
        ``timestamp_source`` is None and every sample time equals the fixture time.
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.source == "ssf"
        assert samples.timestamp_source is None
        assert np.all(samples.times == _T)

    def test_nearest_pass_is_served(self, tmp_path):
        """``SSFReader.samples_in`` serves only the pass nearest the query time when two passes cover the box.

        Two passes over the same place can both fall inside ``MAX_TIME_OFFSET``; mixing the farther one into a Libera
        footprint would average clouds and aerosols observed hours apart into one match.

        Builds a fixture with three footprints at the fixture time and three two hours later at the same location, so
        both passes are within the three-hour ``MAX_TIME_OFFSET`` of either query, queries 20 minutes after each pass,
        and asserts each query returns only that pass's three samples and aerosol optical depths.
        """
        # Six footprints: the first three at the fixture time, the last three two hours
        # later, all at the same cluster location. A footprint near either pass gets that
        # pass's three footprints only.
        lats = np.array([10.2, 10.4, 10.6, 10.2, 10.4, 10.6], dtype=np.float32)
        lons = np.full(6, 350.0, dtype=np.float32)
        late_time = _T + np.timedelta64(2, "h")
        times = np.array([_T, _T, _T, late_time, late_time, late_time], dtype="datetime64[ns]")
        reader = SSFReader(make_ssf_fixture(tmp_path, lats=lats, lons_0360=lons, times=times))
        near_early = reader.samples_in(_BBOX, _T + np.timedelta64(20, "m"))
        assert near_early.n_samples == 3
        assert np.allclose(_finite_values(near_early.values, "aerosol_optical_depth"), [0.10, 0.20, 0.30], atol=1e-4)
        near_late = reader.samples_in(_BBOX, late_time + np.timedelta64(20, "m"))
        assert near_late.n_samples == 3
        assert np.allclose(_finite_values(near_late.values, "aerosol_optical_depth"), [0.40, 0.50, 0.60], atol=1e-4)

    def test_beyond_max_offset_gives_no_samples(self, tmp_path):
        """``SSFReader.samples_in`` returns no samples when every footprint is beyond ``MAX_TIME_OFFSET``.

        A footprint with no SSF pass within three hours must be scored as having no match rather than given stale cloud
        and aerosol values.

        Queries the default fixture five hours after its footprint time and asserts ``n_samples`` is 0.
        """
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T + np.timedelta64(5, "h"))
        assert samples.n_samples == 0

    def test_granule_loaded_once_while_in_window(self, tmp_path):
        """``SSFReader`` parses a granule once while it is in the time window and releases it after.

        Footprints arrive in time order across a day of hourly granules; reparsing a granule per query would multiply
        run time, and keeping every granule would grow memory without bound.

        Queries the fixture at its time and one hour later and asserts the same loaded index object is reused, then
        queries ten hours later and asserts the granule is no longer held in ``_loaded``.
        """
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
        """``SSFReader.samples_in`` serves the granule's measured footprint spacing as ``resolution_km``.

        The coverage measure credits each sample with this spacing; using the nominal value for a sparser swath would
        overstate how completely SSF covers a Libera footprint.

        Reads the default fixture, whose footprints lie 0.2 degrees (about 22 km) apart along one scan, and asserts
        ``resolution_km`` is 22.2 km within 3 percent.
        """
        # The default fixture lays five footprints 0.2 deg (~22 km) apart along one scan,
        # so the measured along-scan spacing, not the nominal 20 km, is served.
        reader = SSFReader(make_ssf_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.resolution_km == pytest.approx(22.2, rel=0.03)


class TestSSFReaderGeolocationChecks:
    @pytest.mark.parametrize("units", ["degrees", None])
    def test_latitude_units_other_than_degrees_north_raises(self, tmp_path, units):
        """``SSFReader.samples_in`` raises ``ValueError`` when footprint latitude is not in ``degrees_north``.

        The reader assumes a true -90..90 latitude; a granule using another convention, such as a colatitude, would
        place every footprint at the wrong latitude without an error.

        Parametrized over a ``"degrees"`` units attribute and no units attribute; asserts the error names
        ``instrument_fov_latitude`` and its units.
        """
        path = make_ssf_fixture(tmp_path, latitude_units=units)
        with pytest.raises(ValueError, match="instrument_fov_latitude has units"):
            SSFReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize("name", ["instrument_fov_latitude", "aerosol_optical_depth", "cloud_classification"])
    def test_field_without_fill_value_raises(self, tmp_path, name):
        """``SSFReader.samples_in`` raises ``ValueError`` when a field it reads has no ``_FillValue`` attribute.

        The reader masks each field with its own fill value; with none declared, fill sentinels would pass through as
        real numbers and be averaged into footprints.

        Parametrized over a geolocation field, a continuous field and a code field, each written without ``_FillValue``;
        asserts the error names the field.
        """
        path = make_ssf_fixture(tmp_path, omit_fill_value=(name,))
        with pytest.raises(ValueError, match=f"{name} has no _FillValue attribute"):
            SSFReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize("name", ["aerosol_optical_depth", "cloud_ice_particle_radius_37um_mean"])
    def test_field_without_valid_range_raises(self, tmp_path, name):
        """``SSFReader.samples_in`` raises ``ValueError`` when a field it reads has no ``valid_range`` attribute.

        The reader takes each field's valid range from the file; with none declared it cannot tell a retrieval from
        an out-of-range value, and guessing one could silently drop or keep the wrong values.

        Parametrized over a 1-D field and a layered field, each written without ``valid_range``; asserts the error
        names the field.
        """
        path = make_ssf_fixture(tmp_path, omit_valid_range=(name,))
        with pytest.raises(ValueError, match=f"{name} has no valid_range attribute"):
            SSFReader(path).samples_in(_BBOX, _T)


class TestSSFReaderValidRange:
    def test_valid_range_comes_from_the_file(self, tmp_path):
        """``SSFReader`` range-checks each field with the ``valid_range`` the file declares.

        A hard-coded range that differs from the file silently drops valid retrievals or publishes invalid ones; the
        real alpha4 granule declares the 3.7 µm ice radius [0, 300] µm and water radius [0, 40] µm.

        Writes lower-layer ice radii of 150 µm (inside the file's range) and water radii of 50 µm (outside it) on the
        five clustered footprints, and asserts all five ice radii are served and no water radius is.
        """
        n = 6
        path = make_ssf_fixture(
            tmp_path,
            cloud_ice_particle_radius_lower=np.full(n, 150.0, dtype=np.float32),
            cloud_water_particle_radius_lower=np.full(n, 50.0, dtype=np.float32),
        )
        data = _values(SSFReader(path))
        assert np.allclose(_finite_values(data, "cloud_ice_particle_radius_lower"), 150.0)
        assert _finite_values(data, "cloud_ice_particle_radius_lower").size == 5
        assert _finite_values(data, "cloud_water_particle_radius_lower").size == 0


class TestSSFReaderLayoutChecks:
    def test_renamed_layer_dimension_raises(self, tmp_path):
        """``SSFReader`` raises ``ValueError`` naming the field and ``product_version`` when a layer axis is renamed.

        The reader picks a cloud layer by position on the second axis; in a layout whose second axis means something
        else, such as Edition1C's ``Conditions``, it would serve the wrong quantity under the layer's name.

        Builds the fixture with the layered fields' second dimension named ``Conditions`` and asserts a ``ValueError``
        naming the first layered field read and ``alpha4``.
        """
        path = make_ssf_fixture(tmp_path, layer_dim_name="Conditions")
        with pytest.raises(ValueError, match=r"read as cloud_optical_depth_lower\) has dimensions.*'alpha4'"):
            SSFReader(path).samples_in(_BBOX, _T)

    def test_missing_variable_raises(self, tmp_path):
        """``SSFReader`` raises ``ValueError``, not ``KeyError``, naming a variable the granule lacks.

        A granule in another layout should fail with a message saying what is missing and what to do, rather than an
        opaque ``KeyError`` from inside the netCDF library.

        Builds the fixture without ``Scene_Type/shortwave_adm_type`` and asserts a ``ValueError`` naming it and
        ``alpha4``.
        """
        path = make_ssf_fixture(tmp_path, omit_variables=(("Scene_Type", "shortwave_adm_type"),))
        with pytest.raises(ValueError, match=r"no variable Scene_Type/shortwave_adm_type .*'alpha4'"):
            SSFReader(path).samples_in(_BBOX, _T)

    def test_missing_group_raises(self, tmp_path):
        """``SSFReader`` raises ``ValueError``, not ``KeyError``, naming a group the granule lacks.

        A granule without ``Auxillary_Properties`` should fail with a message naming the missing group, not an opaque
        ``KeyError``, so the reader of the error knows the granule is in a layout the reader does not read.

        Builds the fixture with both ``Auxillary_Properties`` variables left out, so the group is not written, and
        asserts a ``ValueError`` naming the group.
        """
        omit = (("Auxillary_Properties", "aerosol_optical_depth"), ("Auxillary_Properties", "surface_albedo"))
        path = make_ssf_fixture(tmp_path, omit_variables=omit)
        with pytest.raises(ValueError, match="no group 'Auxillary_Properties'"):
            SSFReader(path).samples_in(_BBOX, _T)
