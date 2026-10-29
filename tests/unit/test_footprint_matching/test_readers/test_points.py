"""Unit tests for the shared reader decoding helpers.

These cover :mod:`libera_utils.footprint_matching.readers._points`: CF and hour-of-day
time decoding, granule time coverage, and swath sample spacing.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from libera_utils.footprint_matching.readers._points import (
    decode_cf_times,
    decode_hours_of_day,
    granule_time_coverage,
    neighbor_spacing_km,
    scan_to_scan_spacing_km,
)


class TestDecodeCfTimes:
    @staticmethod
    def _variable(tmp_path, values, units=None, mask_index=None, fill_value=-1.0, valid_range=(0.0, 1.0e9)):
        import netCDF4

        path = tmp_path / "t.nc"
        ds = netCDF4.Dataset(str(path), "w")
        ds.createDimension("n", len(values))
        var = ds.createVariable("time", np.float64, ("n",), fill_value=False if fill_value is None else fill_value)
        var[:] = np.asarray(values, dtype=np.float64)
        if units is not None:
            var.units = units
        if valid_range is not None:
            var.valid_range = np.array(valid_range)
        if mask_index is not None:
            var[mask_index] = -1.0
        return ds, var

    def test_seconds_since_epoch(self, tmp_path):
        """``decode_cf_times`` decodes values in CF ``seconds since`` units into ``datetime64[ns]`` instants.

        Sample times decide which footprint a measurement is matched to, so a wrong epoch or unit would attach
        measurements to the wrong footprints.

        Decodes [0, 3600] seconds since 2026-01-01 and asserts a ``datetime64[ns]`` result of midnight and 01:00 on
        that day.
        """
        ds, var = self._variable(tmp_path, [0.0, 3600.0], units="seconds since 2026-01-01 00:00:00")
        try:
            times = decode_cf_times(var, source="test")
        finally:
            ds.close()
        assert times.dtype == np.dtype("datetime64[ns]")
        assert times[0] == np.datetime64("2026-01-01T00:00:00", "ns")
        assert times[1] == np.datetime64("2026-01-01T01:00:00", "ns")

    def test_masked_value_is_nat(self, tmp_path):
        """``decode_cf_times`` decodes a masked time entry to ``NaT`` and leaves valid entries decoded.

        A missing time decoded as the epoch would give its sample a time it was never observed at; ``NaT`` lets the
        reader drop the sample instead.

        Writes the variable's -1 fill into the second of two times, so netCDF4 masks it, and asserts the first time is
        valid and the second is ``NaT``.
        """
        ds, var = self._variable(tmp_path, [0.0, 60.0], units="seconds since 2026-01-01", mask_index=1)
        try:
            times = decode_cf_times(var, source="test")
        finally:
            ds.close()
        assert not np.isnat(times[0])
        assert np.isnat(times[1])

    def test_missing_units_raises(self, tmp_path):
        """``decode_cf_times`` raises ``ValueError`` for a time variable with no ``units`` attribute.

        Without units the numbers have no epoch or scale, and any assumed one would silently shift every sample time.

        Builds a time variable without ``units`` and asserts a ``ValueError`` saying it has no ``'units'`` attribute.
        """
        ds, var = self._variable(tmp_path, [0.0])
        try:
            with pytest.raises(ValueError, match="no 'units' attribute"):
                decode_cf_times(var, source="ssf x.nc Time_and_Position/time")
        finally:
            ds.close()

    def test_undecodable_units_raise(self, tmp_path):
        """``decode_cf_times`` raises ``ValueError`` when the ``units`` are not a CF time unit netCDF4 can decode.

        A unit string the decoder does not understand must stop the read with a clear message rather than surface as an
        opaque library error or a wrong time.

        Builds a time variable with units ``"julian day"`` and asserts a ``ValueError`` saying it cannot decode them.
        """
        ds, var = self._variable(tmp_path, [0.0], units="julian day")
        try:
            with pytest.raises(ValueError, match="cannot decode"):
                decode_cf_times(var, source="test")
        finally:
            ds.close()

    def test_fill_and_out_of_range_values_are_nat(self, tmp_path):
        """``decode_cf_times`` decodes ``_FillValue`` and out-of-``valid_range`` entries to ``NaT`` with masking off.

        The CERES SSF time fill is about 1.8e308 days, which would overflow the calendar arithmetic, and the SSF reader
        turns netCDF4's automatic masking off, so the function must apply the fill and range itself.

        Writes 18353.5 days since 1970, the float64 maximum fill and 5.0 (below a ``valid_range`` of 10000..400000),
        reopens with ``set_auto_mask(False)``, and asserts 2020-04-01T12:00 followed by two ``NaT``.
        """
        import netCDF4

        path = tmp_path / "fill.nc"
        with netCDF4.Dataset(str(path), "w") as ds:
            ds.createDimension("n", 3)
            var = ds.createVariable("time", np.float64, ("n",), fill_value=1.7976931348623157e308)
            var.units = "days since 1970-01-01 00:00:00"
            var.valid_range = np.array([10000.0, 400000.0])
            var[:] = np.array([18353.5, 1.7976931348623157e308, 5.0])
        with netCDF4.Dataset(str(path)) as ds:
            ds.set_auto_mask(False)
            times = decode_cf_times(ds.variables["time"], source="test")
        assert times[0] == np.datetime64("2020-04-01T12:00:00", "ns")
        assert np.isnat(times[1])
        assert np.isnat(times[2])

    @pytest.mark.parametrize(
        ("fill_value", "valid_range", "match"),
        [
            (None, (0.0, 1.0e9), "time has no _FillValue attribute"),
            (-1.0, None, "time has no valid_range attribute"),
            (-1.0, (0.0, 1.0, 2.0), r"time declares valid_range \[0.0, 1.0, 2.0\]"),
        ],
    )
    def test_missing_or_malformed_fill_or_range_raises(self, tmp_path, fill_value, valid_range, match):
        """``decode_cf_times`` raises ``ValueError`` when the time variable lacks a fill or a (low, high) valid range.

        The fill and range come from the file as for every other field; with either missing, a fill sentinel or an
        out-of-range value would decode to a real-looking time and match the sample to the wrong footprints.

        Parametrized over no ``_FillValue``, no ``valid_range`` and a three-value ``valid_range``; asserts each raises
        ``ValueError`` naming the variable.
        """
        ds, var = self._variable(
            tmp_path, [0.0], units="seconds since 2026-01-01", fill_value=fill_value, valid_range=valid_range
        )
        try:
            with pytest.raises(ValueError, match=match):
                decode_cf_times(var, source="test")
        finally:
            ds.close()

    def test_descending_valid_range_is_accepted(self, tmp_path):
        """``decode_cf_times`` applies a ``valid_range`` declared high-to-low as the same interval.

        CERES files declare some ranges in descending order; read as written, such a range would mark every time
        invalid.

        Declares ``valid_range`` [100, 0] seconds, decodes 50 and 200 seconds since 2026-01-01, and asserts the first
        decodes and the second is ``NaT``.
        """
        ds, var = self._variable(tmp_path, [50.0, 200.0], units="seconds since 2026-01-01", valid_range=(100.0, 0.0))
        try:
            ds.set_auto_mask(False)
            times = decode_cf_times(var, source="test")
        finally:
            ds.close()
        assert times[0] == np.datetime64("2026-01-01T00:00:50", "ns")
        assert np.isnat(times[1])


class TestDecodeHoursOfDay:
    @staticmethod
    def _variable(tmp_path, values, units="hours", fill=True, valid_range=True):
        import netCDF4

        path = tmp_path / "hours.nc"
        ds = netCDF4.Dataset(str(path), "w")
        ds.createDimension("s", len(values))
        ds.createDimension("p", 2)
        fill_value = np.float32(3.4028235e38) if fill else False
        var = ds.createVariable("Observation_Time", np.float32, ("s", "p"), fill_value=fill_value)
        var[:] = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 2, axis=1)
        if units is not None:
            var.units = units
        if valid_range:
            var.valid_range = np.array([0.0, 23.999998], dtype=np.float32)
        return ds, var

    def test_hours_date_from_the_granule_day(self, tmp_path):
        """``decode_hours_of_day`` adds fractional UTC hours to the reference's midnight, keeping the variable's shape.

        CLDPIX stores only the hour of day, so the date must come from the granule; a wrong day or hour scale would
        shift every pixel away from the footprints it was observed with.

        Decodes a (2, 2) float32 variable holding 15.5 and 15.999 hours against a 2020-04-10T15:00:01.4 reference and
        asserts the shape, 15:30:00 exactly, and 15:59:56.4 to within 10 ms (float32 resolution).
        """
        ds, var = self._variable(tmp_path, [15.5, 15.999])
        try:
            times = decode_hours_of_day(var, np.datetime64("2020-04-10T15:00:01.4"), source="test")
        finally:
            ds.close()
        assert times.shape == (2, 2)
        assert times[0, 0] == np.datetime64("2020-04-10T15:30:00", "ns")
        # float32 hours resolve to a few milliseconds.
        assert abs(times[1, 1] - np.datetime64("2020-04-10T15:59:56.4", "ns")) < np.timedelta64(10, "ms")

    def test_fill_is_nat(self, tmp_path):
        """``decode_hours_of_day`` decodes fill entries to ``NaT`` and leaves valid hours decoded.

        A float32-maximum fill read as hours would land a pixel years past its granule; ``NaT`` lets the reader drop
        it.

        Overwrites the second row of a (2, 2) variable with the float32-maximum fill and asserts the first row has no
        ``NaT`` and the second row is all ``NaT``.
        """
        ds, var = self._variable(tmp_path, [1.0, 2.0])
        var[1, :] = np.float32(3.4028235e38)
        try:
            times = decode_hours_of_day(var, np.datetime64("2020-04-10T01:00"), source="test")
        finally:
            ds.close()
        assert not np.isnat(times[0]).any()
        assert np.isnat(times[1]).all()

    @pytest.mark.parametrize(
        ("reference", "hours", "expected"),
        [
            # A 00 UTC granule whose first scanline precedes midnight.
            ("2020-04-10T00:00:01.4", 23.9999, "2020-04-09T23:59:59.64"),
            # A 23 UTC granule whose last scanline follows midnight.
            ("2020-04-09T23:00:01", 0.0001, "2020-04-10T00:00:00.36"),
        ],
    )
    def test_hours_across_midnight_take_the_nearest_day(self, tmp_path, reference, hours, expected):
        """``decode_hours_of_day`` dates an hour on the far side of midnight to the day nearest the reference.

        A granule's scanlines can precede its ``time_coverage_start`` by a second or two; near midnight, dating them by
        the start's calendar day would put them 24 h late (or early) and drop them from the nearest pass.

        Parametrized over a 00 UTC reference with a 23.9999 h value and a 23 UTC reference with a 0.0001 h value;
        asserts each decodes to the previous or next day respectively, within 10 ms (float32 resolution).
        """
        ds, var = self._variable(tmp_path, [hours])
        try:
            times = decode_hours_of_day(var, np.datetime64(reference), source="test")
        finally:
            ds.close()
        assert abs(times[0, 0] - np.datetime64(expected, "ns")) < np.timedelta64(10, "ms")

    def test_wrong_units_raise(self, tmp_path):
        """``decode_hours_of_day`` raises ``ValueError`` when the variable's ``units`` are not hours.

        Reading a CF ``seconds since`` time as hours of day would put every sample at a meaningless time.

        Builds the variable with units ``"seconds since 1970-01-01"`` and asserts a ``ValueError`` that says fractional
        hours were expected.
        """
        ds, var = self._variable(tmp_path, [1.0], units="seconds since 1970-01-01")
        try:
            with pytest.raises(ValueError, match="fractional hours"):
                decode_hours_of_day(var, np.datetime64("2020-04-10"), source="test")
        finally:
            ds.close()

    @pytest.mark.parametrize(
        ("fill", "valid_range", "match"),
        [
            (False, True, "Observation_Time has no _FillValue attribute"),
            (True, False, "Observation_Time has no valid_range attribute"),
        ],
    )
    def test_missing_fill_or_range_raises(self, tmp_path, fill, valid_range, match):
        """``decode_hours_of_day`` raises ``ValueError`` when the variable has no ``_FillValue`` or ``valid_range``.

        The fill and range come from the file as for every other field; with either missing, a fill sentinel would be
        read as an hour and land the pixel far from its granule.

        Parametrized over a variable with no ``_FillValue`` and one with no ``valid_range``; asserts each raises
        ``ValueError`` naming the variable.
        """
        ds, var = self._variable(tmp_path, [1.0], fill=fill, valid_range=valid_range)
        try:
            with pytest.raises(ValueError, match=match):
                decode_hours_of_day(var, np.datetime64("2020-04-10"), source="test")
        finally:
            ds.close()


class TestNeighborSpacing:
    def test_swath_reports_the_larger_of_across_and_along(self):
        """``neighbor_spacing_km`` reports the larger of the across-track and along-track spacing of a 2-D swath.

        The coverage measure credits each sample with this spacing, so a swath subsampled in one direction would be
        credited with too small an area, and its coverage understated, if only the finer direction were measured.

        Builds a 10 x 5 equatorial grid with rows 0.018 deg (about 2 km) apart and pixels 0.063 deg (about 7 km) apart
        and asserts a spacing of 7 km within 3%.
        """
        # 2 km along-track rows, 7 km across-track pixels at the equator (0.063 deg).
        lats = np.arange(0.0, 0.18, 0.018)[:, None] + np.zeros((1, 5))
        lons = np.zeros((10, 1)) + np.arange(0.0, 0.315, 0.063)[None, :]
        spacing = neighbor_spacing_km(lats, lons)
        assert spacing == pytest.approx(7.0, rel=0.03)

    def test_track_ignores_large_jumps(self):
        """``neighbor_spacing_km`` leaves out 1-D steps larger than ten times the median before taking the median.

        A scan turnaround or data gap is not a sample spacing; letting it raise the estimate would credit each sample
        with too much area and overstate coverage.

        Builds a meridional track of four 10 km steps, three 20 km steps and three 1000 km jumps, whose plain median is
        20 km, and asserts a spacing of 10 km within 3%, the median once the jumps are left out.
        """
        # Sorted steps 10, 10, 10, 10, 20, 20, 20, 1000, 1000, 1000: the jumps push the plain
        # median to 20 km; without them the median is 10 km.
        steps_km = np.array([10.0] * 4 + [20.0] * 3 + [1000.0] * 3)
        lats = np.concatenate([[0.0], np.cumsum(steps_km / 111.195)])
        lons = np.zeros_like(lats)
        assert neighbor_spacing_km(lats, lons) == pytest.approx(10.0, rel=0.03)

    def test_non_finite_and_degenerate_inputs(self):
        """``neighbor_spacing_km`` returns NaN when there are fewer than two finite neighboring samples.

        With no measurable pair there is no spacing, and a made-up number such as zero would be credited to every
        sample as if it were measured.

        Asserts NaN for a single point and for two points whose latitudes are both NaN.
        """
        assert np.isnan(neighbor_spacing_km(np.array([0.0]), np.array([0.0])))
        assert np.isnan(neighbor_spacing_km(np.array([np.nan, np.nan]), np.array([0.0, 0.0])))


class TestScanToScanSpacing:
    def test_pairs_the_same_scan_position_on_consecutive_scans(self):
        """``scan_to_scan_spacing_km`` measures the distance between a scan position and the same position one scan on.

        SSF footprints are listed scan by scan, so consecutive footprints are only the along-scan spacing apart; the
        along-track spacing between scans, several times larger, is only seen by pairing like positions.

        Builds three scans of four footprints 0.108 deg (about 12 km) apart along-scan, with scans 0.396 deg (about 44
        km) apart, and asserts ``neighbor_spacing_km`` gives 12 km while ``scan_to_scan_spacing_km`` gives 44 km.
        """
        # Three scans of four footprints 12 km apart along-scan; scans 44 km apart along-track.
        n_scan, n_pos = 3, 4
        lats = (np.arange(n_scan)[:, None] * 0.396 + np.zeros((1, n_pos))).ravel()
        lons = (np.zeros((n_scan, 1)) + np.arange(n_pos)[None, :] * 0.108).ravel()
        times = np.repeat(np.arange(n_scan), n_pos).astype(float)
        sample_number = np.tile(np.arange(1, n_pos + 1), n_scan)
        assert neighbor_spacing_km(lats, lons) == pytest.approx(12.0, rel=0.05)
        assert scan_to_scan_spacing_km(lats, lons, times, sample_number) == pytest.approx(44.0, rel=0.03)

    def test_single_scan_is_nan(self):
        """``scan_to_scan_spacing_km`` returns NaN when no two footprints share a scan position.

        A single scan has no next scan to measure against, and any number returned would be a guess presented as a
        measurement.

        Passes four footprints with distinct sample numbers 1 to 4 and asserts NaN.
        """
        lats = np.zeros(4)
        lons = np.arange(4) * 0.1
        assert np.isnan(scan_to_scan_spacing_km(lats, lons, np.arange(4.0), np.arange(1, 5)))


class TestGranuleTimeCoverage:
    def test_reads_both_attributes_with_a_z_suffix(self):
        """``granule_time_coverage`` parses ``time_coverage_start`` and ``time_coverage_end`` with a trailing ``Z``.

        The granule's span decides which files are read for a footprint's time; failing on the UTC ``Z`` suffix CERES
        granules carry, or losing sub-second precision, would misplace the granule in time.

        Passes a stand-in dataset with both attributes ending in ``Z`` and asserts the start, with microseconds, and the
        end as ``datetime64[ns]``.
        """
        ds = SimpleNamespace(
            time_coverage_start="2020-04-10T15:00:01.414490Z", time_coverage_end="2020-04-10T15:59:59Z"
        )
        start, end = granule_time_coverage(ds, source="test")
        assert start == np.datetime64("2020-04-10T15:00:01.414490", "ns")
        assert end == np.datetime64("2020-04-10T15:59:59", "ns")

    def test_missing_attribute_raises(self):
        """``granule_time_coverage`` raises ``ValueError`` naming the attribute when one of the pair is missing.

        A granule with no stated end cannot be placed in time, and an assumed span could pull in or skip data.

        Passes a stand-in dataset with only ``time_coverage_start`` and asserts a ``ValueError`` naming
        ``time_coverage_end``.
        """
        ds = SimpleNamespace(time_coverage_start="2020-04-10T15:00:00Z")
        with pytest.raises(ValueError, match="no 'time_coverage_end' global attribute"):
            granule_time_coverage(ds, source="test")

    def test_non_iso_value_raises(self):
        """``granule_time_coverage`` raises ``ValueError`` when an attribute is not an ISO 8601 instant.

        An unparsable coverage time must stop the read with a message naming the value rather than surface as a
        bare NumPy error.

        Passes a stand-in dataset whose ``time_coverage_start`` is ``"not-a-time"`` and asserts a ``ValueError`` saying
        it is not an ISO 8601 instant.
        """
        ds = SimpleNamespace(time_coverage_start="not-a-time", time_coverage_end="2020-04-10T15:59:59Z")
        with pytest.raises(ValueError, match="is not an ISO 8601 instant"):
            granule_time_coverage(ds, source="test")
