"""Unit tests for CLDPIXReader (CERES CLDPIX imager-pixel reader).

Uses synthetic CLDPIX NetCDF4 fixtures created by ``make_cldpix_fixture``. The
fixture places all pixels near lat ≈ 40°, lon ≈ −15° (stored as 345° in the
0..360 convention) and deliberately reproduces the real product's *descending*
``valid_range`` on ``Eff_Cld_Pressure`` so the reader's auto-mask handling is
exercised.

Real CLDPIX files come from NASA CERES, e.g.
``CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc``.
"""

import netCDF4
import numpy as np
import pytest

from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader, _bad_data_flags
from libera_utils.footprint_matching.types import BoundingBox
from tests.plugins.footprint_matching_fixtures import SWATH_FIXTURE_TIME, make_cldpix_fixture
from tests.unit.test_footprint_matching.test_readers.grid_helpers import raw_and_usable, served_at

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


# The served pixels of every field in the default fixture, as (count, value): every field is constant, effective
# pressure under its descending valid_range, and one Cloud_Particle_Phase pixel and one Top_Cld_Height pixel are fill.
_FIXTURE_VALUES = {
    "cloud_optical_depth": (12, 4.0),
    "cloud_water_path": (12, 100.0),
    "cloud_effective_temperature": (12, 270.0),
    "cloud_effective_height": (12, 5.0),
    "cloud_effective_pressure": (12, 800.0),
    "cloud_top_height": (11, 6.0),
    "cloud_particle_radius": (12, 10.0),
    "cloud_particle_phase": (11, 1.0),
    "cloud_mask": (12, 1.0),
}


class TestCLDPIXReaderSamples:
    def test_pixels_are_served_as_time_stamped_points(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves each pixel as its own float32 sample, at its position and time.

        The PSF weigher indexes values by variable row and pixel column, and the nearest-pass selection uses each
        pixel's own time. Footprint boxes are given in -180..180, so a pixel left at 345 degrees would never match a
        footprint at -15 degrees.

        Builds the default 3 x 4 pixel fixture at 345 degrees longitude, queries the box at -16..-14 degrees and asserts
        float32 values of shape ``(len(VARIABLES), 12)``, every longitude at -15 degrees, the source ``"cldpix"``, no
        timestamp source and every pixel time at the fixture time.
        """
        samples = CLDPIXReader([make_cldpix_fixture(tmp_path)]).samples_in(_BBOX, _T)
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), 12)
        assert samples.values.dtype == np.float32
        assert np.allclose(samples.lons, -15.0, atol=1e-4)
        assert samples.source == "cldpix"
        assert samples.timestamp_source is None
        assert np.all(samples.times == _T)

    def test_every_field_serves_its_source_values(self, tmp_path):
        """``CLDPIXReader.samples_in`` serves every field in its own row with the file's values, fill as NaN.

        Values are matched to variables by row, so a row out of ``VARIABLES`` order publishes one field under another's
        name. The real ``Eff_Cld_Pressure`` declares its ``valid_range`` descending ([1100, 10]), which netCDF4's
        automatic masking reads as empty, masking every pressure; and a category's int8 fill left in place would be
        counted as a class of its own.

        Reads the default fixture once and asserts, for every variable, the count and value of its finite pixels
        (``_FIXTURE_VALUES``).
        """
        data = _values(CLDPIXReader([make_cldpix_fixture(tmp_path)]))
        assert set(_FIXTURE_VALUES) == {v.name for v in CLDPIXReader.VARIABLES}
        for name, (count, value) in _FIXTURE_VALUES.items():
            served = _finite(data, name)
            assert served.size == count, name
            np.testing.assert_allclose(served, value, rtol=1e-6, err_msg=name)

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
        reader = CLDPIXReader([fixture])

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
        reader = CLDPIXReader([path])
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
        served = _finite(_values(CLDPIXReader([path])), out_name)
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
        served = _finite(_values(CLDPIXReader([path])), out_name)
        assert not np.any(served == bad_code)
        assert np.count_nonzero(served == kept_code) == 4


class TestCLDPIXReaderTimeMatching:
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
            CLDPIXReader([path]).samples_in(_BBOX, _T)

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
        samples = CLDPIXReader([path]).samples_in(_BBOX, next_day)
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
        samples = CLDPIXReader([path]).samples_in(_BBOX, before)
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
        samples = CLDPIXReader([path]).samples_in(_BBOX, _T)
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
            CLDPIXReader([path]).samples_in(_BBOX, _T)


class TestCLDPIXReaderLayoutChecks:
    def test_missing_variable_raises(self, tmp_path):
        """``CLDPIXReader`` raises ``ValueError``, not ``KeyError``, naming a variable the granule lacks.

        A granule in another layout should fail with a message saying what is missing and what to do, as the FLASHFlux reader
        does, rather than an opaque ``KeyError`` from inside the netCDF library.

        Builds the fixture, renames ``Cld_Radius`` so the granule lacks it, and asserts a ``ValueError`` naming
        ``Cld_Radius`` and saying to stage a granule in the layout the reader reads.
        """
        path = make_cldpix_fixture(tmp_path)
        with netCDF4.Dataset(str(path), "a") as ds:
            ds.renameVariable("Cld_Radius", "Cld_Radius_renamed")
        with pytest.raises(ValueError, match=r"no variable 'Cld_Radius' in this granule.*Stage a CER_CLDPIX granule"):
            CLDPIXReader([path]).samples_in(_BBOX, _T)


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
            CLDPIXReader([path]).samples_in(_BBOX, _T)

    def test_latitude_without_valid_range_raises(self, tmp_path):
        """``CLDPIXReader`` raises ``ValueError`` when ``Latitude`` declares no valid range.

        Without the [0, 180] range the colatitude convention cannot be confirmed, and converting an unconfirmed field
        could misplace every pixel.

        Builds the fixture with no ``valid_range`` on ``Latitude`` and asserts ``samples_in`` raises ``ValueError``
        matching "Latitude has no valid_range attribute".
        """
        path = make_cldpix_fixture(tmp_path, latitude_valid_range=None)
        with pytest.raises(ValueError, match=r"Latitude has no valid_range attribute"):
            CLDPIXReader([path]).samples_in(_BBOX, _T)

    @pytest.mark.parametrize(
        ("attribute", "name"),
        [
            *(
                ("_FillValue", name)
                for name in (
                    "Latitude",
                    "Longitude",
                    "Observation_Time",
                    "Eff_Cld_Optical_Depth",
                    "Cloud_Particle_Phase",
                )
            ),
            *(
                ("valid_range", name)
                for name in (
                    "Longitude",
                    "Observation_Time",
                    "Eff_Cld_Optical_Depth",
                    "Top_Cld_Height",
                    "Cloud_Particle_Phase",
                )
            ),
        ],
    )
    def test_field_without_fill_value_or_valid_range_raises(self, tmp_path, attribute, name):
        """``CLDPIXReader`` raises ``ValueError`` when a field it reads has no ``_FillValue`` or ``valid_range``.

        The reader takes each field's fill sentinel and valid range from the file; with either missing it cannot tell
        missing pixels or out-of-range values from data, and would average them in as real values.

        Parametrized over each attribute on the geolocation fields, the pixel time, continuous fields and a categorical
        field (``Latitude``'s range is covered by the colatitude checks above); writes the fixture without it and
        asserts ``samples_in`` raises ``ValueError`` naming the field and the attribute.
        """
        omit = {"omit_fill_value": (name,)} if attribute == "_FillValue" else {"omit_valid_range": (name,)}
        path = make_cldpix_fixture(tmp_path, **omit)
        with pytest.raises(ValueError, match=f"{name} has no {attribute} attribute"):
            CLDPIXReader([path]).samples_in(_BBOX, _T)

    def test_longitude_valid_range_comes_from_the_file(self, tmp_path):
        """``CLDPIXReader`` range-checks pixel longitude with the ``valid_range`` the file declares.

        Geolocation is held to the file's own range like every other field, so a pixel outside it is dropped rather
        than placed on the map.

        Writes ``Longitude`` with a ``valid_range`` of [0, 340], which excludes the fixture's 345 degrees, and asserts
        no pixel is served in the fixture box.
        """
        path = make_cldpix_fixture(tmp_path, longitude_valid_range=(0.0, 340.0))
        assert CLDPIXReader([path]).samples_in(_BBOX, _T).n_samples == 0

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
            CLDPIXReader([path]).samples_in(_BBOX, _T)


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


class TestCLDPIXReaderRealSubset:
    """A real CER_CLDPIX_NOAA20-VIIRS 1P9test granule (2020-04-10 15 UTC), cut to three blocks of scanlines.

    The blocks are spread across the hour (start, middle, end) with their values unchanged; see
    tests/test_data/footprint_matching/README.md. The tests pin the file conventions the reader relies on: the
    fractional-hour time variable, the colatitude geolocation, the hourly granule span, and that a query at a pixel's
    own time serves its pass with the values the file holds there.
    """

    @pytest.fixture
    def cldpix_path(self, fmatch_real_data_path):
        return fmatch_real_data_path / "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015_subset.nc"

    def test_times_are_dated_fractional_hours_spanning_one_hour(self, cldpix_path):
        """``CLDPIXReader._load_granule`` dates a real granule's fractional-hour ``Observation_Time`` onto its day.

        CLDPIX stores times as hours of the day with the date only in a global attribute, so a decoding that dropped or
        misread the date would put every pixel on the wrong day and no Libera footprint would ever match it.

        Loads the real CLDPIX cut and asserts it has set times spanning between 55 min and 1 h, and that they fall
        within 10 s of the granule's ``time_coverage_start``/``time_coverage_end`` as read by ``_granule_span``.
        """
        reader = CLDPIXReader([cldpix_path])
        points = reader._load_granule(cldpix_path)
        assert points.times is not None
        finite = points.times[~np.isnat(points.times)]
        assert finite.size > 0
        span = finite.max() - finite.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        # The granule's own coverage attributes bracket every decoded time to within a few
        # seconds: the first scanlines precede the declared start by about two seconds in the
        # source granule, which is why each hour is dated to the day nearest the start.
        start, end = reader._granule_span(cldpix_path)
        assert finite.min() >= start - np.timedelta64(10, "s")
        assert finite.max() <= end + np.timedelta64(10, "s")

    def test_latitudes_are_true_latitudes_in_both_hemispheres(self, cldpix_path):
        """``CLDPIXReader._load_granule`` returns latitudes converted from the file's colatitude, within [-90, 90].

        CLDPIX stores colatitude (0 at the North Pole to 180 at the South Pole); read as latitude, the southern
        hemisphere would fall outside the valid range and be discarded and the rest would be misplaced, so cloud values
        would land on the wrong footprints.

        Loads the real CLDPIX cut, which crosses the equator, and asserts every latitude lies in [-90, 90], that some
        are south and some north of the equator, and every longitude lies in [-180, 180).
        """
        reader = CLDPIXReader([cldpix_path])
        points = reader._load_granule(cldpix_path)
        assert np.nanmin(points.lats) >= -90.0
        assert np.nanmax(points.lats) <= 90.0
        assert np.nanmin(points.lats) < 0.0 < np.nanmax(points.lats)
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_pixel_time_serves_its_pass(self, cldpix_path):
        """``CLDPIXReader.samples_in`` serves a real CLDPIX pixel's own pass when queried at its time and place.

        This is the end-to-end path a Libera footprint takes through the cloud reader: a wrong geolocation or time
        convention would leave a footprint over a known pixel with no cloud values, or with values from another pass.

        Takes the first pixel from the middle of the real CLDPIX cut whose ``Cld_Radius`` is valid in the file, queries
        a 0.6-degree box around it at its time and asserts samples are served, all within ``PASS_WINDOW`` of that time,
        with one value row per declared variable, and that the sample at that pixel serves the ``Cld_Radius`` the file
        stores for it.
        """
        reader = CLDPIXReader([cldpix_path])
        points = reader._load_granule(cldpix_path)
        radius, radius_ok = raw_and_usable(cldpix_path, None, "Cld_Radius")
        usable = radius_ok.ravel() & np.isfinite(points.lats) & ~np.isnat(points.times)
        middle = points.lats.size // 2
        i = middle + int(np.flatnonzero(usable[middle:])[0])
        box = BoundingBox(points.lats[i] - 0.3, points.lats[i] + 0.3, points.lons[i] - 0.3, points.lons[i] + 0.3)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= CLDPIXReader.PASS_WINDOW)
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), samples.n_samples)
        at_i = served_at(samples, points.lats[i], points.lons[i], points.times[i])
        assert at_i.size > 0
        row = [v.name for v in CLDPIXReader.VARIABLES].index("cloud_particle_radius")
        assert np.allclose(samples.values[row, at_i], radius.ravel()[i])

    def test_granule_span_matches_the_coverage_attributes(self, cldpix_path):
        """``CLDPIXReader._granule_span`` reads an hour-long span from a real granule that ``granules_in_window`` uses.

        The span decides which granules are held for a footprint; a misread span would drop the granule a footprint
        falls in, losing its cloud values, or hold granules a day away.

        Reads the real CLDPIX cut's span, asserts it lasts between 55 min and 1 h, and asserts ``granules_in_window``
        returns the granule for a time 30 min into the span and nothing for a time 24 h later.
        """
        reader = CLDPIXReader([cldpix_path])
        start, end = reader._granule_span(cldpix_path)
        assert np.timedelta64(55, "m") < end - start <= np.timedelta64(1, "h")
        # A footprint inside the hour has the granule in its window; one a day away does not.
        assert reader.granules_in_window(start + np.timedelta64(30, "m")) == [cldpix_path]
        assert reader.granules_in_window(start + np.timedelta64(24, "h")) == []
