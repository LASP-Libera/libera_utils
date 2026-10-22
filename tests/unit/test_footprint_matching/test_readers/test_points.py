"""Unit tests for the shared reader decoding helpers.

These cover :mod:`libera_utils.footprint_matching.readers._points`: longitude
normalization.
"""

import numpy as np

from libera_utils.footprint_matching.readers._points import normalize_longitude


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
