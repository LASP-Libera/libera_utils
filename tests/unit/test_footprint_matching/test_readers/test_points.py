"""Unit tests for the shared reader decoding helpers.

These cover :mod:`libera_utils.footprint_matching.readers._points`: longitude
normalization and fill / valid-range masking.
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers._points import apply_fill_and_valid_range, normalize_longitude


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
