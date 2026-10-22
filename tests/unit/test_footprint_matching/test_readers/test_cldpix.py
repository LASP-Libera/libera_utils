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
        """``CLDPIXReader.READER_KEY`` is ``"cldpix"``.

        The key is how a configuration names the reader and how its samples are labeled; a changed key would leave a
        configuration that asks for ``"cldpix"`` with no reader and no CLDPIX cloud properties.

        Asserts ``READER_KEY`` equals ``"cldpix"``.
        """
        assert CLDPIXReader.READER_KEY == "cldpix"

    def test_resolution_km(self):
        """``CLDPIXReader.RESOLUTION_KM`` declares the 1 km VIIRS imager-pixel scale.

        The resolution pads each footprint's query box and is the fallback pixel spacing when a granule's own spacing
        cannot be measured; a wrong value would serve too few or too many pixels to a footprint.

        Asserts ``RESOLUTION_KM`` equals 1.0.
        """
        assert CLDPIXReader.RESOLUTION_KM == 1.0

    def test_expected_variable_names(self):
        """``CLDPIXReader.VARIABLES`` holds the CLDPIX cloud properties and none of the surface-type fields.

        The cloud properties are what the imager products carry from CLDPIX, while land cover and snow/ice come from
        the IGBP and NISE readers; a CLDPIX copy of a surface field would put two competing values in the product.

        Asserts the five core cloud variable names are a subset of ``VARIABLES`` and that ``igbp_ecosystem``,
        ``snow_map`` and ``ice_map`` are absent.
        """
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
        """``CLDPIXReader.VARIABLES`` declares mode-aggregated categories as int16 and every other variable as float32.

        A category such as cloud phase is a class label stored as an integer in the product, while a continuous
        property needs a float; the wrong dtype would truncate a mean or write a label as a fractional value.

        Asserts, for every variable, ``int16`` when its aggregation is ``weighted_mode`` and ``float32`` otherwise.
        """
        for v in CLDPIXReader.VARIABLES:
            assert v.dtype == ("int16" if v.aggregation == "weighted_mode" else "float32"), v.name


class TestCLDPIXReaderSamples:
    def test_returns_one_row_per_variable(self, tmp_path):
        """``CLDPIXReader.samples_in`` returns a float32 ``(n_var, n_pixels)`` value array with one time per pixel.

        The PSF weigher indexes values by variable row and pixel column; a transposed, missing or differently typed
        array would attach a value to the wrong variable or pixel.

        Builds the default 3 x 4 pixel fixture, queries the box holding it and asserts a 2-D float32 array of shape
        ``(len(VARIABLES), 12)`` and a ``times`` array of shape ``(12,)``.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.values.ndim == 2
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), 12)
        assert samples.values.dtype == np.float32
        assert samples.times is not None
        assert samples.times.shape == (12,)

    def test_longitude_normalization_places_points(self, tmp_path):
        """``CLDPIXReader`` normalizes the file's 0..360 longitudes so pixels land in a -180..180 query box.

        Footprint boxes are given in -180..180; a pixel left at 345° would never match a footprint at -15° and the
        western-hemisphere cloud properties would be lost.

        Builds the fixture with pixels at 345° longitude, queries a box spanning -16° to -14° and asserts some value is
        finite.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.isfinite(data).any()

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves no pixels to a box that does not contain them.

        A footprint must receive only the imager pixels under it; pixels leaking into a distant box would assign one
        region's clouds to another.

        Builds the fixture with pixels near 40° N, -15° E, queries a box at 14° to 16° E and asserts no value is
        finite.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, BoundingBox(39.0, 41.0, 14.0, 16.0))
        assert not np.isfinite(data).any()

    def test_continuous_values(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves continuous cloud properties with the values stored in the file.

        A decoding error such as a wrong scale or a misread field would publish a plausible-looking but wrong optical
        depth, water path or temperature.

        Builds the fixture with constant fields and asserts every finite optical depth is 4.0, water path 100.0 and
        effective temperature 270.0.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite(data, "cloud_optical_depth"), 4.0, atol=1e-4)
        assert np.allclose(_finite(data, "cloud_water_path"), 100.0, atol=1e-3)
        assert np.allclose(_finite(data, "cloud_effective_temperature"), 270.0, atol=1e-3)

    def test_pressure_survives_descending_valid_range(self, tmp_path):
        """``CLDPIXReader`` keeps effective cloud pressure although the file declares its ``valid_range`` descending.

        The real ``Eff_Cld_Pressure`` declares ``valid_range`` [1100, 10], which netCDF4 automatic masking reads as an
        empty range and masks every value, so the product would carry no cloud pressure at all.

        Builds the fixture with a constant 800 hPa and the descending range, and asserts some pressure is finite and
        every finite one is 800.0.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        pressure = _finite(data, "cloud_effective_pressure")
        assert pressure.size > 0
        assert np.allclose(pressure, 800.0, atol=1e-3)

    def test_cloud_particle_radius_values(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves the effective cloud particle radius from ``Cld_Radius``.

        The particle radius is read from the blended ``Cld_Radius`` field; reading the wrong field or masking it would
        publish a wrong or missing radius.

        Builds the fixture with ``Cld_Radius`` of 10.0 μm everywhere and asserts some radius is finite and every finite
        one is 10.0.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        radius = _finite(data, "cloud_particle_radius")
        assert radius.size > 0
        assert np.allclose(radius, 10.0, atol=1e-3)

    def test_categorical_values(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves the cloud mask and particle phase categories with their stored values.

        A category is a class label, so any change in decoding would report a different cloud class, such as ice in
        place of water.

        Builds the fixture with both fields set to category 1 and asserts every finite ``cloud_mask`` and
        ``cloud_particle_phase`` value is 1.0.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        assert np.allclose(_finite(data, "cloud_mask"), 1.0)
        assert np.allclose(_finite(data, "cloud_particle_phase"), 1.0)

    def test_southern_hemisphere_pixels_are_kept(self, tmp_path):
        """``CLDPIXReader`` converts the colatitude stored as ``Latitude`` so southern-hemisphere pixels are kept.

        CLDPIX stores colatitude under the name ``Latitude``, so a southern pixel holds a value above 90; reading it as
        a true latitude and filtering on -90..90 would drop the whole southern hemisphere, and a sign error would
        mirror it north.

        Builds the fixture with every pixel at 40° S and asserts the box at 39° to 41° S holds finite values while the
        mirrored box at 39° to 41° N holds none.
        """
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
        """``CLDPIXReader`` masks the float32 fill sentinel in ``Top_Cld_Height``, the one field with no valid range.

        With no ``valid_range`` the fill test alone must catch the sentinel 3.4028235e38; if the stored float32 fill
        and the declared value compare unequal, the sentinel would be published as a cloud top height of ~3.4e38 km.

        Builds the fixture with one ``Top_Cld_Height`` pixel at the fill and the rest at 6.0 km, and asserts some
        height is finite and every finite one is 6.0.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        heights = _finite(data, "cloud_top_height")
        assert heights.size > 0
        assert np.allclose(heights, 6.0, atol=1e-3)


class TestCLDPIXReaderTimeMatching:
    def test_policy_is_nearest_pass(self):
        """``CLDPIXReader`` matches by nearest pass with a 15 minute pass window and a 3 hour maximum offset.

        A ground location is seen on more than one pass a day, so the reader must serve the pass nearest the
        footprint time and refuse one too far away; other limits would mix passes or match clouds hours apart.

        Asserts ``TIME_MATCH`` is ``NEAREST_PASS``, ``PASS_WINDOW`` is 15 minutes and ``MAX_TIME_OFFSET`` is 3 hours.
        """
        assert CLDPIXReader.TIME_MATCH is TimeMatch.NEAREST_PASS
        assert CLDPIXReader.PASS_WINDOW == np.timedelta64(15, "m")
        assert CLDPIXReader.MAX_TIME_OFFSET == np.timedelta64(3, "h")

    def test_samples_carry_source_and_scanline_times(self, tmp_path):
        """``CLDPIXReader.samples_in`` labels its samples ``"cldpix"`` with no timestamp source and per-pixel times.

        The source label and each pixel's observation time are what the product records about where and when a cloud
        property came from; a wrong label or time would misattribute the data.

        Builds the fixture with every scanline at the fixture time and asserts ``source`` is ``"cldpix"``,
        ``timestamp_source`` is None and every sample time equals the fixture time.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.source == "cldpix"
        assert samples.timestamp_source is None
        assert np.all(samples.times == _T)

    def test_nearest_pass_is_served(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves only the granule nearest the footprint time when two cover the same pixels.

        Two passes over one location are hours apart in cloud state; mixing them would average clouds from different
        times into one footprint.

        Builds two granules over the same pixels twelve hours apart and asserts a query 30 minutes after the first gets
        its 12 pixels all at the first time, and a query 30 minutes before the second gets 12 pixels at the second.
        """
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
        """``CLDPIXReader.samples_in`` returns zero samples when the footprint time is beyond ``MAX_TIME_OFFSET``.

        A cloud observation hours away from the footprint no longer describes the scene, so the footprint must be
        scored as having no CLDPIX match rather than given stale clouds.

        Builds the fixture at the fixture time, queries 6 hours later and asserts zero samples and a value array of
        shape ``(len(VARIABLES), 0)``.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T + np.timedelta64(6, "h"))
        assert samples.n_samples == 0
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), 0)


class TestCLDPIXReaderGranuleDate:
    def test_missing_time_coverage_start_raises(self, tmp_path):
        """``CLDPIXReader`` raises ``ValueError`` naming ``time_coverage_start`` when a granule lacks that attribute.

        Observation times are stored as hours of the day, and the date comes only from ``time_coverage_start``; without
        it every pixel time would be undated or guessed.

        Deletes the attribute from the fixture and asserts ``samples_in`` raises ``ValueError`` matching
        ``time_coverage_start``.
        """
        import netCDF4

        path = make_cldpix_fixture(tmp_path)
        with netCDF4.Dataset(str(path), "r+") as ds:
            ds.delncattr("time_coverage_start")
        with pytest.raises(ValueError, match="time_coverage_start"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    def test_times_are_dated_by_the_granule(self, tmp_path):
        """``CLDPIXReader`` dates the fractional-hour ``Observation_Time`` by the granule's ``time_coverage_start``.

        A wrong date on the hours of day would shift every pixel by whole days and match it to the wrong footprints.

        Builds the fixture with fractional hours of 2026-01-01 dated by the attribute and asserts every served pixel
        time equals the fixture instant.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert np.all(samples.times == _T)


class TestCLDPIXReaderGeolocationChecks:
    def test_latitude_valid_range_other_than_colatitude_raises(self, tmp_path):
        """``CLDPIXReader`` raises ``ValueError`` when ``Latitude`` declares a valid range other than colatitude [0, 180].

        The reader converts ``Latitude`` as colatitude; applied to a granule that holds true -90..90 latitudes, the
        conversion would mirror every pixel across the equator.

        Builds the fixture with ``Latitude`` declaring (-90, 90) and asserts ``samples_in`` raises ``ValueError``
        matching "Latitude declares valid_range".
        """
        path = make_cldpix_fixture(tmp_path, latitude_valid_range=(-90.0, 90.0))
        with pytest.raises(ValueError, match=r"Latitude declares valid_range"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    def test_latitude_without_valid_range_raises(self, tmp_path):
        """``CLDPIXReader`` raises ``ValueError`` when ``Latitude`` declares no valid range.

        Without the [0, 180] range the colatitude convention cannot be confirmed, and converting an unconfirmed field
        could misplace every pixel.

        Builds the fixture with no ``valid_range`` on ``Latitude`` and asserts ``samples_in`` raises ``ValueError``
        matching "Latitude declares valid_range None".
        """
        path = make_cldpix_fixture(tmp_path, latitude_valid_range=None)
        with pytest.raises(ValueError, match=r"Latitude declares valid_range None"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize("name", ["Latitude", "Eff_Cld_Optical_Depth", "Cloud_Particle_Phase"])
    def test_field_without_fill_value_raises(self, tmp_path, name):
        """``CLDPIXReader`` raises ``ValueError`` when a field it reads has no ``_FillValue`` attribute.

        The reader takes each field's fill sentinel from the file; with none declared it cannot tell missing pixels
        from data, and a sentinel would be averaged in as a real value.

        Parametrized over a geolocation (``Latitude``), a continuous (``Eff_Cld_Optical_Depth``) and a categorical
        (``Cloud_Particle_Phase``) field; writes the fixture without that field's fill and asserts ``samples_in``
        raises ``ValueError`` matching "<name> has no _FillValue attribute".
        """
        path = make_cldpix_fixture(tmp_path, omit_fill_value=(name,))
        with pytest.raises(ValueError, match=f"{name} has no _FillValue attribute"):
            CLDPIXReader(path).samples_in(_BBOX, _T)
