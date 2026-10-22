"""Unit tests for the shared reader decoding helpers.

These cover :mod:`libera_utils.footprint_matching.readers._points`: longitude
normalization and fill / valid-range masking.
"""

import numpy as np

from libera_utils.footprint_matching.readers._points import apply_fill_and_valid_range, normalize_longitude


class TestNormalizeLongitude:
    def test_0_360_maps_to_minus180_180(self):
        """``normalize_longitude`` maps longitudes on the 0..360 convention onto [-180, 180).

        ERA5 grids arrive on 0..360 while the readers work in -180..180, so a longitude above 180 left unconverted would
        never match a footprint box in the western hemisphere.

        Converts six longitudes from 0 to 359.9 and asserts each lands on its signed equivalent, with 180 mapping to
        -180 and 359.9 to -0.1.
        """
        lon = np.array([0.0, 90.0, 180.0, 270.0, 350.0, 359.9])
        out = normalize_longitude(lon)
        assert np.allclose(out, [0.0, 90.0, -180.0, -90.0, -10.0, -0.1], atol=1e-6)

    def test_already_in_range_is_unchanged(self):
        """``normalize_longitude`` leaves longitudes already in [-180, 180) unchanged.

        A caller can then apply it to a grid without first checking its convention; that is safe only if it is a no-op
        on longitudes already in range.

        Converts five longitudes between -179 and 179 and asserts the output equals the input.
        """
        lon = np.array([-179.0, -10.0, 0.0, 45.0, 179.0])
        out = normalize_longitude(lon)
        assert np.allclose(out, lon, atol=1e-6)


class TestApplyFillAndValidRange:
    def test_fill_value_becomes_nan(self):
        """``apply_fill_and_valid_range`` replaces the ``fill_value`` sentinel with NaN and keeps every other value.

        A sentinel such as -999 left as a number would be averaged into a footprint as if it were data.

        Passes a float array containing one -999 with ``fill_value=-999.0`` and asserts that element is NaN and the
        other three are unchanged.
        """
        raw = np.array([1.0, 2.0, -999.0, 3.0])
        out = apply_fill_and_valid_range(raw, fill_value=-999.0)
        assert np.isnan(out[2])
        assert np.allclose(out[[0, 1, 3]], [1.0, 2.0, 3.0])

    def test_out_of_valid_range_becomes_nan(self):
        """``apply_fill_and_valid_range`` sets values outside ``valid_range`` to NaN and keeps the inclusive endpoints.

        A value outside a netCDF ``valid_range`` is invalid, but one sitting exactly on a bound is valid and would be
        lost by an exclusive test.

        Applies a range of (0, 5) to [-1, 0, 5, 6] and asserts -1 and 6 become NaN while 0 and 5 are kept.
        """
        raw = np.array([-1.0, 0.0, 5.0, 6.0])
        out = apply_fill_and_valid_range(raw, valid_range=(0.0, 5.0))
        assert np.isnan(out[0])
        assert np.isnan(out[3])
        assert np.allclose(out[[1, 2]], [0.0, 5.0])

    def test_descending_valid_range_is_normalized(self):
        """``apply_fill_and_valid_range`` treats a ``valid_range`` given high to low the same as one given low to high.

        Some products store a range in descending order, such as a pressure field's [1100, 10]; taken literally, no
        value would satisfy both bounds and the whole field would be masked.

        Applies (1100, 10) to [800, 50, 5, 2000] and asserts 800 and 50 are kept while 5 and 2000 become NaN.
        """
        # The CLDPIX pressure field stores valid_range as [1100, 10]; values in
        # between must be kept, not masked.
        raw = np.array([800.0, 50.0, 5.0, 2000.0])
        out = apply_fill_and_valid_range(raw, valid_range=(1100.0, 10.0))
        assert np.allclose(out[[0, 1]], [800.0, 50.0])
        assert np.isnan(out[2])
        assert np.isnan(out[3])

    def test_integer_fill_matches_exactly(self):
        """``apply_fill_and_valid_range`` recognizes an integer fill value in an integer array.

        Integer fields mark missing cells with an integer sentinel, which would otherwise be served as a valid value.

        Passes an ``int8`` array containing 127 with ``fill_value=127`` and asserts that element is NaN in the float
        output.
        """
        raw = np.array([1, 2, 127, 3], dtype=np.int8)
        out = apply_fill_and_valid_range(raw, fill_value=127)
        assert np.isnan(out[2])

    def test_masked_array_input(self):
        """``apply_fill_and_valid_range`` turns the masked elements of a ``numpy.ma.MaskedArray`` into NaN.

        A masked array flags its missing elements in the mask rather than by a sentinel, and the data under the mask is
        not a valid value, so the mask must be honored even when no fill value is given.

        Passes a three-element masked array with its middle element masked and no fill or range, and asserts that
        element is NaN and the other two are unchanged.
        """
        raw = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        out = apply_fill_and_valid_range(raw)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 3.0])

    def test_float32_fill_matches_in_raw_dtype(self):
        """``apply_fill_and_valid_range`` matches a float32 fill equal to the float32 maximum by comparing in float32.

        Some products use the float32 maximum as fill; the decimal literal a reader declares, widened to float64,
        differs from the stored float32 value, so a float64 comparison would miss every fill and serve it as data.

        Passes a float32 array containing 3.4028235e38 with that literal as ``fill_value`` and asserts that element is
        NaN and the others are unchanged.
        """
        # The CERES float fill is the float32 maximum. Stored as float32 and compared in
        # that dtype it matches exactly, even though its float64 widening differs from
        # the decimal literal the readers declare.
        raw = np.array([1.0, 3.4028235e38, 2.0], dtype=np.float32)
        out = apply_fill_and_valid_range(raw, fill_value=3.4028235e38)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 2.0])

    def test_int32_extreme_fill_matches_exactly(self):
        """``apply_fill_and_valid_range`` matches an int32 fill float32 cannot represent, without masking its neighbor.

        The int32 maximum and the value one below it round to the same float32, so a comparison made after a cast to
        float32 would also mask a valid value; comparing in the raw int32 dtype tells them apart.

        Passes an int32 array holding the int32 maximum as fill and the maximum minus one as data, and asserts the fill
        is NaN while the maximum minus one and the other values stay finite.
        """
        # An int32 sentinel float32 cannot represent exactly still matches, because the
        # comparison is made before any cast. sentinel - 1 rounds to the same float32, so it
        # survives only if the comparison is made in the raw dtype.
        sentinel = np.iinfo(np.int32).max
        raw = np.array([1, sentinel, 2, sentinel - 1], dtype=np.int32)
        out = apply_fill_and_valid_range(raw, fill_value=sentinel)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 2.0])
        assert np.isfinite(out[3])
