"""Unit tests for the shared point-reader decoding helpers.

These cover the format-agnostic pieces in
:mod:`libera_utils.footprint_matching.readers._points` used by the SSF, CLDPIX,
IGBP and NISE readers: longitude normalization, fill / valid-range masking, and
CF time decoding.
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers._points import (
    apply_fill_and_valid_range,
    decode_cf_times,
    decode_hours_of_day,
    neighbour_spacing_km,
    normalize_longitude,
    scan_to_scan_spacing_km,
)


class TestNormalizeLongitude:
    def test_0_360_maps_to_minus180_180(self):
        # Targets normalize_longitude wrapping [0, 360) into [-180, 180); asserts each input maps to
        # its expected signed longitude (e.g. 270 -> -90, 350 -> -10).
        lon = np.array([0.0, 90.0, 180.0, 270.0, 350.0, 359.9])
        out = normalize_longitude(lon)
        assert np.allclose(out, [0.0, 90.0, -180.0, -90.0, -10.0, -0.1], atol=1e-6)

    def test_already_in_range_is_unchanged(self):
        # Targets normalize_longitude idempotence for in-range input; asserts values already in
        # [-180, 180) pass through unchanged.
        lon = np.array([-179.0, -10.0, 0.0, 45.0, 179.0])
        out = normalize_longitude(lon)
        assert np.allclose(out, lon, atol=1e-6)


class TestApplyFillAndValidRange:
    def test_fill_value_becomes_nan(self):
        # Targets fill-value masking in apply_fill_and_valid_range; asserts the -999 fill becomes NaN
        # while the remaining valid values are preserved.
        raw = np.array([1.0, 2.0, -999.0, 3.0])
        out = apply_fill_and_valid_range(raw, fill_value=-999.0)
        assert np.isnan(out[2])
        assert np.allclose(out[[0, 1, 3]], [1.0, 2.0, 3.0])

    def test_out_of_valid_range_becomes_nan(self):
        # Targets valid_range masking; asserts values below/above the (0, 5) range become NaN while
        # the inclusive endpoints are kept.
        raw = np.array([-1.0, 0.0, 5.0, 6.0])
        out = apply_fill_and_valid_range(raw, valid_range=(0.0, 5.0))
        assert np.isnan(out[0])
        assert np.isnan(out[3])
        assert np.allclose(out[[1, 2]], [0.0, 5.0])

    def test_descending_valid_range_is_normalized(self):
        # The CLDPIX pressure field stores valid_range as [1100, 10]; values in
        # between must be kept, not masked.
        raw = np.array([800.0, 50.0, 5.0, 2000.0])
        out = apply_fill_and_valid_range(raw, valid_range=(1100.0, 10.0))
        assert np.allclose(out[[0, 1]], [800.0, 50.0])
        assert np.isnan(out[2])
        assert np.isnan(out[3])

    def test_integer_fill_matches_exactly(self):
        # Targets exact integer fill matching (no float tolerance); asserts the int8 fill 127 becomes
        # NaN in the float output.
        raw = np.array([1, 2, 127, 3], dtype=np.int8)
        out = apply_fill_and_valid_range(raw, fill_value=127)
        assert np.isnan(out[2])

    def test_masked_array_input(self):
        # Targets numpy MaskedArray handling; asserts masked elements become NaN while unmasked
        # values pass through.
        raw = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        out = apply_fill_and_valid_range(raw)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 3.0])

    def test_float32_fill_matches_in_raw_dtype(self):
        # The CERES float fill is the float32 maximum. Stored as float32 and compared in
        # that dtype it matches exactly, even though its float64 widening differs from
        # the decimal literal the readers declare.
        raw = np.array([1.0, 3.4028235e38, 2.0], dtype=np.float32)
        out = apply_fill_and_valid_range(raw, fill_value=3.4028235e38)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 2.0])

    def test_int32_extreme_fill_matches_exactly(self):
        # An int32 sentinel float32 cannot represent exactly still matches, because the
        # comparison is made before any cast.
        sentinel = np.iinfo(np.int32).max
        raw = np.array([1, sentinel, 2], dtype=np.int32)
        out = apply_fill_and_valid_range(raw, fill_value=sentinel)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 2.0])


class TestDecodeCfTimes:
    @staticmethod
    def _variable(tmp_path, values, units=None, mask_index=None):
        import netCDF4

        path = tmp_path / "t.nc"
        ds = netCDF4.Dataset(str(path), "w")
        ds.createDimension("n", len(values))
        var = ds.createVariable("time", np.float64, ("n",), fill_value=-1.0)
        var[:] = np.asarray(values, dtype=np.float64)
        if units is not None:
            var.units = units
        if mask_index is not None:
            var[mask_index] = -1.0
        return ds, var

    def test_seconds_since_epoch(self, tmp_path):
        ds, var = self._variable(tmp_path, [0.0, 3600.0], units="seconds since 2026-01-01 00:00:00")
        try:
            times = decode_cf_times(var, source="test")
        finally:
            ds.close()
        assert times.dtype == np.dtype("datetime64[ns]")
        assert times[0] == np.datetime64("2026-01-01T00:00:00", "ns")
        assert times[1] == np.datetime64("2026-01-01T01:00:00", "ns")

    def test_masked_value_is_nat(self, tmp_path):
        ds, var = self._variable(tmp_path, [0.0, 60.0], units="seconds since 2026-01-01", mask_index=1)
        try:
            times = decode_cf_times(var, source="test")
        finally:
            ds.close()
        assert not np.isnat(times[0])
        assert np.isnat(times[1])

    def test_missing_units_raises(self, tmp_path):
        ds, var = self._variable(tmp_path, [0.0])
        try:
            with pytest.raises(ValueError, match="no 'units' attribute"):
                decode_cf_times(var, source="ssf x.nc Time_and_Position/time")
        finally:
            ds.close()

    def test_undecodable_units_raise(self, tmp_path):
        ds, var = self._variable(tmp_path, [0.0], units="julian day")
        try:
            with pytest.raises(ValueError, match="cannot decode"):
                decode_cf_times(var, source="test")
        finally:
            ds.close()

    def test_fill_and_out_of_range_values_are_nat(self, tmp_path):
        # The CERES SSF time fill is ~1.8e308 days, which would overflow the calendar
        # arithmetic; it and anything outside valid_range decode to NaT instead.
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


class TestDecodeHoursOfDay:
    @staticmethod
    def _variable(tmp_path, values, units="hours"):
        import netCDF4

        path = tmp_path / "hours.nc"
        ds = netCDF4.Dataset(str(path), "w")
        ds.createDimension("s", len(values))
        ds.createDimension("p", 2)
        var = ds.createVariable("Observation_Time", np.float32, ("s", "p"), fill_value=np.float32(3.4028235e38))
        var[:] = np.repeat(np.asarray(values, dtype=np.float32)[:, None], 2, axis=1)
        if units is not None:
            var.units = units
        var.valid_range = np.array([0.0, 23.999998], dtype=np.float32)
        return ds, var

    def test_hours_date_from_the_granule_day(self, tmp_path):
        ds, var = self._variable(tmp_path, [15.5, 15.999])
        try:
            times = decode_hours_of_day(var, np.datetime64("2020-04-10"), source="test")
        finally:
            ds.close()
        assert times.shape == (2, 2)
        assert times[0, 0] == np.datetime64("2020-04-10T15:30:00", "ns")
        # float32 hours resolve to a few milliseconds.
        assert abs(times[1, 1] - np.datetime64("2020-04-10T15:59:56.4", "ns")) < np.timedelta64(10, "ms")

    def test_fill_is_nat(self, tmp_path):
        ds, var = self._variable(tmp_path, [1.0, 2.0])
        var[1, :] = np.float32(3.4028235e38)
        try:
            times = decode_hours_of_day(var, np.datetime64("2020-04-10"), source="test")
        finally:
            ds.close()
        assert not np.isnat(times[0]).any()
        assert np.isnat(times[1]).all()

    def test_wrong_units_raise(self, tmp_path):
        ds, var = self._variable(tmp_path, [1.0], units="seconds since 1970-01-01")
        try:
            with pytest.raises(ValueError, match="fractional hours"):
                decode_hours_of_day(var, np.datetime64("2020-04-10"), source="test")
        finally:
            ds.close()


class TestNeighbourSpacing:
    def test_swath_reports_the_larger_of_across_and_along(self):
        # 2 km along-track rows, 7 km across-track pixels at the equator (0.063 deg).
        lats = np.arange(0.0, 0.18, 0.018)[:, None] + np.zeros((1, 5))
        lons = np.zeros((10, 1)) + np.arange(0.0, 0.315, 0.063)[None, :]
        spacing = neighbour_spacing_km(lats, lons)
        assert spacing == pytest.approx(7.0, rel=0.03)

    def test_track_ignores_large_jumps(self):
        # 12 km steps with one 500 km jump (a scan turnaround) that must not inflate the median.
        lats = np.concatenate([np.arange(0.0, 0.54, 0.108), np.arange(5.0, 5.54, 0.108)])
        lons = np.zeros_like(lats)
        assert neighbour_spacing_km(lats, lons) == pytest.approx(12.0, rel=0.03)

    def test_non_finite_and_degenerate_inputs(self):
        assert np.isnan(neighbour_spacing_km(np.array([0.0]), np.array([0.0])))
        assert np.isnan(neighbour_spacing_km(np.array([np.nan, np.nan]), np.array([0.0, 0.0])))


class TestScanToScanSpacing:
    def test_pairs_the_same_scan_position_on_consecutive_scans(self):
        # Three scans of four footprints 12 km apart along-scan; scans 44 km apart along-track.
        n_scan, n_pos = 3, 4
        lats = (np.arange(n_scan)[:, None] * 0.396 + np.zeros((1, n_pos))).ravel()
        lons = (np.zeros((n_scan, 1)) + np.arange(n_pos)[None, :] * 0.108).ravel()
        times = np.repeat(np.arange(n_scan), n_pos).astype(float)
        sample_number = np.tile(np.arange(1, n_pos + 1), n_scan)
        assert neighbour_spacing_km(lats, lons) == pytest.approx(12.0, rel=0.05)
        assert scan_to_scan_spacing_km(lats, lons, times, sample_number) == pytest.approx(44.0, rel=0.03)

    def test_single_scan_is_nan(self):
        lats = np.zeros(4)
        lons = np.arange(4) * 0.1
        assert np.isnan(scan_to_scan_spacing_km(lats, lons, np.arange(4.0), np.arange(1, 5)))
