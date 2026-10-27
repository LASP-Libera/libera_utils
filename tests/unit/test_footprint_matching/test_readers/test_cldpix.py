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

from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader, _bad_data_flags
from libera_utils.footprint_matching.types import BoundingBox, TimeMatch
from tests.plugins.footprint_matching_fixtures import SWATH_FIXTURE_TIME, make_cldpix_fixture

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

        Builds the fixture with its 12 pixels at 345° longitude, queries a box spanning -16° to -14° and asserts all 12
        are served, each at -15°.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        samples = reader.samples_in(_BBOX, _T)
        assert samples.n_samples == 12
        assert np.allclose(samples.lons, -15.0, atol=1e-4)

    def test_points_absent_from_unrelated_tile(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves no pixels to a box that does not contain them.

        A footprint must receive only the imager pixels under it; pixels leaking into a distant box would assign one
        region's clouds to another.

        Builds the fixture with pixels near 40° N, -15° E, queries a box at 14° to 16° E and asserts no pixel is
        served.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        assert reader.samples_in(BoundingBox(39.0, 41.0, 14.0, 16.0), _T).n_samples == 0

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
        """``CLDPIXReader.samples_in`` serves the cloud mask and particle phase categories as stored, fill as NaN.

        A category is a class label, so any change in decoding would report a different cloud class, such as ice in
        place of water, and an int8 fill of 127 left in place would be counted as a class of its own.

        Builds the fixture with both fields set to category 1 on its 12 pixels, except one ``Cloud_Particle_Phase``
        pixel at the fill 127, and asserts all 12 ``cloud_mask`` values are 1.0 and the 11 finite
        ``cloud_particle_phase`` values are 1.0.
        """
        reader = CLDPIXReader(make_cldpix_fixture(tmp_path))
        data = _values(reader, _BBOX)
        mask = _finite(data, "cloud_mask")
        phase = _finite(data, "cloud_particle_phase")
        assert mask.size == 12
        assert np.allclose(mask, 1.0)
        assert phase.size == 11
        assert np.allclose(phase, 1.0)

    def test_southern_hemisphere_pixels_are_kept(self, tmp_path):
        """``CLDPIXReader`` converts the colatitude stored as ``Latitude`` so southern-hemisphere pixels are kept.

        CLDPIX stores colatitude under the name ``Latitude``, so a southern pixel holds a value above 90; reading it as
        a true latitude and filtering on -90..90 would drop the whole southern hemisphere, and a sign error would
        mirror it north.

        Builds the fixture with its 12 pixels at 40° S and asserts the box at 39° to 41° S serves all 12, at 40° S,
        while the mirrored box at 39° to 41° N serves none.
        """
        lats = np.full((3, 4), -40.0, dtype=np.float32)
        fixture = make_cldpix_fixture(tmp_path, lats=lats)
        reader = CLDPIXReader(fixture)

        southern = reader.samples_in(BoundingBox(-41.0, -39.0, -16.0, -14.0), _T)
        assert southern.n_samples == 12
        assert np.allclose(southern.lats, -40.0, atol=1e-4)

        # The mirrored northern tile must be empty — a sign-flip bug would put
        # the pixels there instead.
        assert reader.samples_in(_BBOX, _T).n_samples == 0

    def test_top_cloud_height_fill_is_dropped(self, tmp_path):
        """``CLDPIXReader`` masks the float32 fill sentinel in ``Top_Cld_Height`` by its ``_FillValue``.

        If the stored float32 fill and the declared value compared unequal, the sentinel 3.4028235e38 would be
        published as a cloud top height of ~3.4e38 km.

        Builds the fixture with one ``Top_Cld_Height`` pixel at the fill and the rest at 6.0 km, widens the field's
        ``valid_range`` to include the sentinel so only the fill check can drop it, and asserts some height is finite
        and every finite one is 6.0.
        """
        import netCDF4

        path = make_cldpix_fixture(tmp_path)
        with netCDF4.Dataset(str(path), "r+") as ds:
            ds.variables["Top_Cld_Height"].valid_range = np.array([0.0, 3.4028235e38], dtype=np.float32)
        reader = CLDPIXReader(path)
        data = _values(reader, _BBOX)
        heights = _finite(data, "cloud_top_height")
        assert heights.size > 0
        assert np.allclose(heights, 6.0, atol=1e-3)

    @pytest.mark.parametrize(
        ("var", "out_name", "value", "kept"),
        [
            # Inside the file's [2, 78] µm.
            ("Cld_Radius", "cloud_particle_radius", 70.0, True),
            # Outside the file's [0, 18] km.
            ("Top_Cld_Height", "cloud_top_height", 20.0, False),
        ],
    )
    def test_valid_range_comes_from_the_file(self, tmp_path, var, out_name, value, kept):
        """``CLDPIXReader`` range-checks each field with the ``valid_range`` the file declares.

        A hard-coded range that differs from the file silently drops valid retrievals or publishes invalid ones; the
        real granules declare ``Cld_Radius`` [2, 78] µm and ``Top_Cld_Height`` [0, 18] km.

        Parametrized over a 70 µm radius (inside the file's range) and a 20 km top height (outside it); writes the
        value into every pixel of the fixture and asserts all 12 are served or none is.
        """
        import netCDF4

        path = make_cldpix_fixture(tmp_path)
        with netCDF4.Dataset(str(path), "r+") as ds:
            ds.variables[var][:] = np.float32(value)
        served = _finite(_values(CLDPIXReader(path)), out_name)
        if kept:
            assert served.size == 12
            assert np.allclose(served, value)
        else:
            assert served.size == 0

    @pytest.mark.parametrize(
        ("var", "out_name", "bad_code", "kept_code"),
        [("Cloud_Particle_Phase", "cloud_particle_phase", 5, 4), ("CERES_Cloud_Mask", "cloud_mask", 3, 2)],
    )
    def test_bad_data_codes_do_not_vote(self, tmp_path, var, out_name, bad_code, kept_code):
        """``CLDPIXReader`` drops the categorical codes the file's ``flag_meanings`` label bad data.

        Phase 5 (``bad_data``) and mask 3 (``Bad_data``) carry no retrieval; served as categories, a footprint of mostly
        bad pixels would publish "bad data" as its mode. Mask 2 (``TBD``) is not labelled bad data and is kept.

        Parametrized over phase and mask; sets four pixels to the bad-data code and four to another code, and asserts
        the bad-data code is never served while the other is served on its four pixels.
        """
        import netCDF4

        path = make_cldpix_fixture(tmp_path)
        with netCDF4.Dataset(str(path), "r+") as ds:
            codes = np.full((3, 4), 1, dtype=np.int8)
            codes[0, :] = bad_code
            codes[1, :] = kept_code
            ds.variables[var][:] = codes
        served = _finite(_values(CLDPIXReader(path)), out_name)
        assert not np.any(served == bad_code)
        assert np.count_nonzero(served == kept_code) == 4


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

        Two passes over one location can both fall inside ``MAX_TIME_OFFSET``, and their cloud states differ; mixing
        them would average clouds observed hours apart into one footprint.

        Builds two granules over the same pixels two hours apart, so both are within the three-hour ``MAX_TIME_OFFSET``
        of either query, and asserts a query 30 minutes after the first gets only its 12 pixels, all at the first time,
        and a query 30 minutes before the second gets only its 12 pixels, at the second.
        """
        n_scan = 3
        early = make_cldpix_fixture(tmp_path, times=np.full(n_scan, _T, dtype="datetime64[ns]"), filename="early.nc")
        late_time = _T + np.timedelta64(2, "h")
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

        ``Observation_Time`` holds only hours of the day, so the date must come from the attribute; a date taken from
        anywhere else would shift every pixel by whole days and match it to the wrong footprints.

        Builds the fixture, then moves its ``time_coverage_start`` and ``time_coverage_end`` one day later while leaving
        the hours unchanged, and asserts a query at the fixture time a day later serves the 12 pixels at that time.
        """
        import netCDF4

        path = make_cldpix_fixture(tmp_path)
        next_day = _T + np.timedelta64(1, "D")
        with netCDF4.Dataset(str(path), "r+") as ds:
            for attr in ("time_coverage_start", "time_coverage_end"):
                value = np.datetime64(ds.getncattr(attr).rstrip("Z")) + np.timedelta64(1, "D")
                ds.setncattr(attr, np.datetime_as_string(value, unit="us") + "Z")
        samples = CLDPIXReader(path).samples_in(_BBOX, next_day)
        assert samples.n_samples == 12
        assert np.all(samples.times == next_day)

    def test_scanline_before_midnight_keeps_its_day(self, tmp_path):
        """``CLDPIXReader`` dates a scanline observed just before midnight to that day in a granule starting after it.

        The real granules' first scanlines precede ``time_coverage_start`` by about two seconds; in a 00 UTC granule
        such a line is stored as ~23.9999 h, and dated by the start's calendar day it would land 24 h late and drop
        out of the nearest pass.

        Builds a granule with scanlines at 23:59:59.6 the previous day and 00:00:10, 00:00:20 with
        ``time_coverage_start`` 00:00:01.4, queries at 23:59:59.6 and asserts that scanline's four pixels are served at
        that time (to float32 resolution).
        """
        midnight = np.datetime64("2026-01-02T00:00:00", "ns")
        before = midnight - np.timedelta64(400, "ms")
        times = np.array([before, midnight + np.timedelta64(10, "s"), midnight + np.timedelta64(20, "s")])
        path = make_cldpix_fixture(tmp_path, times=times, time_coverage_start=midnight + np.timedelta64(1400, "ms"))
        samples = CLDPIXReader(path).samples_in(_BBOX, before)
        early = samples.times[samples.times < midnight]
        assert early.size == 4
        assert np.all(np.abs(early - before) < np.timedelta64(10, "ms"))


class TestCLDPIXReaderTimeShape:
    def test_one_time_per_scanline_is_broadcast(self, tmp_path):
        """``CLDPIXReader`` serves a per-scanline ``Observation_Time`` as each pixel of the line at the line's time.

        A granule may store one time per scanline rather than per pixel; misaligned broadcasting would give pixels
        another line's time.

        Builds the fixture with a 1-D ``(Scanlines,)`` time, scanlines 0, 15 and 30 s after the fixture time, and
        asserts the 12 served times are each line's time four times over (to float32 resolution).
        """
        times = _T + np.array([0, 15, 30], dtype="timedelta64[s]").astype("timedelta64[ns]")
        path = make_cldpix_fixture(tmp_path, times=times, time_dims=("Scanlines",))
        samples = CLDPIXReader(path).samples_in(_BBOX, _T)
        assert samples.n_samples == 12
        assert np.all(np.abs(np.sort(samples.times) - np.repeat(times, 4)) < np.timedelta64(10, "ms"))

    def test_time_of_another_shape_raises(self, tmp_path):
        """``CLDPIXReader`` raises ``ValueError`` when ``Observation_Time`` is neither per scanline nor per pixel.

        A time variable of another shape cannot be attributed to pixels, and guessing would date every pixel wrongly.

        Builds the fixture with a 4-long ``(Pixels,)`` time against the 3 x 4 swath and asserts ``samples_in`` raises
        ``ValueError`` matching "expected one time per scanline or per pixel".
        """
        path = make_cldpix_fixture(tmp_path, time_dims=("Pixels",))
        with pytest.raises(ValueError, match="expected one time per scanline or per pixel"):
            CLDPIXReader(path).samples_in(_BBOX, _T)


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
        matching "Latitude has no valid_range attribute".
        """
        path = make_cldpix_fixture(tmp_path, latitude_valid_range=None)
        with pytest.raises(ValueError, match=r"Latitude has no valid_range attribute"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize(
        "name", ["Latitude", "Longitude", "Observation_Time", "Eff_Cld_Optical_Depth", "Cloud_Particle_Phase"]
    )
    def test_field_without_fill_value_raises(self, tmp_path, name):
        """``CLDPIXReader`` raises ``ValueError`` when a field it reads has no ``_FillValue`` attribute.

        The reader takes each field's fill sentinel from the file; with none declared it cannot tell missing pixels
        from data, and a sentinel would be averaged in as a real value.

        Parametrized over the geolocation fields (``Latitude``, ``Longitude``), the pixel time (``Observation_Time``),
        a continuous (``Eff_Cld_Optical_Depth``) and a categorical (``Cloud_Particle_Phase``) field; writes the fixture
        without that field's fill and asserts ``samples_in`` raises ``ValueError`` matching "<name> has no _FillValue
        attribute".
        """
        path = make_cldpix_fixture(tmp_path, omit_fill_value=(name,))
        with pytest.raises(ValueError, match=f"{name} has no _FillValue attribute"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    @pytest.mark.parametrize(
        "name", ["Longitude", "Observation_Time", "Eff_Cld_Optical_Depth", "Top_Cld_Height", "Cloud_Particle_Phase"]
    )
    def test_field_without_valid_range_raises(self, tmp_path, name):
        """``CLDPIXReader`` raises ``ValueError`` when a field it reads has no ``valid_range`` attribute.

        The reader takes each field's valid range from the file; with none declared it cannot tell a retrieval from
        an out-of-range value, and guessing one could silently drop or keep the wrong values.

        Parametrized over ``Longitude``, the pixel time (``Observation_Time``), two continuous fields and a categorical
        one; writes the fixture without that field's range and asserts ``samples_in`` raises ``ValueError`` matching
        "<name> has no valid_range attribute".
        """
        path = make_cldpix_fixture(tmp_path, omit_valid_range=(name,))
        with pytest.raises(ValueError, match=f"{name} has no valid_range attribute"):
            CLDPIXReader(path).samples_in(_BBOX, _T)

    def test_longitude_valid_range_comes_from_the_file(self, tmp_path):
        """``CLDPIXReader`` range-checks pixel longitude with the ``valid_range`` the file declares.

        Geolocation is held to the file's own range like every other field, so a pixel outside it is dropped rather
        than placed on the map.

        Writes ``Longitude`` with a ``valid_range`` of [0, 340], which excludes the fixture's 345 degrees, and asserts
        no pixel is served in the fixture box.
        """
        path = make_cldpix_fixture(tmp_path, longitude_valid_range=(0.0, 340.0))
        assert CLDPIXReader(path).samples_in(_BBOX, _T).n_samples == 0

    @pytest.mark.parametrize("name", ["Cloud_Particle_Phase", "CERES_Cloud_Mask"])
    def test_categorical_field_without_flags_raises(self, tmp_path, name):
        """``CLDPIXReader`` raises ``ValueError`` when a categorical field has no ``flag_values`` / ``flag_meanings``.

        The reader drops the codes the file labels bad data; without the labels it cannot tell which those are, and a
        bad-data code would vote in the mode.

        Parametrized over phase and mask; writes the fixture without that field's flags and asserts ``samples_in``
        raises ``ValueError`` matching "<name> has no flag_values / flag_meanings".
        """
        path = make_cldpix_fixture(tmp_path, omit_flags=(name,))
        with pytest.raises(ValueError, match=f"{name} has no flag_values / flag_meanings"):
            CLDPIXReader(path).samples_in(_BBOX, _T)


class TestBadDataFlags:
    @staticmethod
    def _variable(tmp_path, flag_values, flag_meanings):
        import netCDF4

        ds = netCDF4.Dataset(str(tmp_path / "flags.nc"), "w")
        ds.createDimension("n", 2)
        var = ds.createVariable("phase", np.int8, ("n",))
        if flag_values is not None:
            var.flag_values = np.array(flag_values, dtype=np.int8)
        if flag_meanings is not None:
            var.flag_meanings = flag_meanings
        return ds, var

    @pytest.mark.parametrize(
        ("flag_values", "flag_meanings", "expected"),
        [
            ([1, 2, 3, 4, 5], "water_cloud ice_cloud no_cloud_retrieval clear bad_data", (5,)),
            ([0, 1, 2, 3], "clear cloud TBD Bad_data", (3,)),
            ([0, 1], "clear cloud", ()),
        ],
    )
    def test_returns_the_values_labelled_bad_data(self, tmp_path, flag_values, flag_meanings, expected):
        """``_bad_data_flags`` returns the flag values whose ``flag_meanings`` entry is ``bad_data``, any case.

        A pixel flagged bad data carries no retrieval; counted as a category, a footprint of mostly bad pixels would
        publish "bad data" as its cloud phase or mask.

        Parametrized over the real CLDPIX phase flags (5 is ``bad_data``), mask flags (3 is ``Bad_data``, 2 ``TBD`` is
        kept) and a field labelling none; asserts the returned tuple.
        """
        ds, var = self._variable(tmp_path, flag_values, flag_meanings)
        try:
            assert _bad_data_flags(var, source="test") == expected
        finally:
            ds.close()

    @pytest.mark.parametrize(
        ("flag_values", "flag_meanings", "match"),
        [
            (None, "clear cloud", "no flag_values / flag_meanings"),
            ([0, 1], None, "no flag_values / flag_meanings"),
            ([0, 1, 2], "clear cloud", "3 flag_values but 2 flag_meanings"),
        ],
    )
    def test_missing_or_mismatched_flags_raise(self, tmp_path, flag_values, flag_meanings, match):
        """``_bad_data_flags`` raises ``ValueError`` when the flag attributes are absent or disagree in length.

        Without both lists the reader cannot tell which codes are bad data, and a mismatched pair would label the
        wrong code.

        Parametrized over a field missing ``flag_values``, one missing ``flag_meanings`` and one listing three values
        against two meanings; asserts a ``ValueError`` saying which.
        """
        ds, var = self._variable(tmp_path, flag_values, flag_meanings)
        try:
            with pytest.raises(ValueError, match=match):
                _bad_data_flags(var, source="test")
        finally:
            ds.close()
