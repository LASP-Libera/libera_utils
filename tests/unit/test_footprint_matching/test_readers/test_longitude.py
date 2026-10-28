"""Unit tests for the readers' longitude-convention helper.

These cover :mod:`libera_utils.footprint_matching.readers._longitude`: longitude
normalization.
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers._longitude import normalize_longitude


@pytest.mark.parametrize(
    ("lon", "expected"),
    [
        pytest.param(
            [0.0, 90.0, 180.0, 270.0, 350.0, 359.9],
            [0.0, 90.0, -180.0, -90.0, -10.0, -0.1],
            id="0..360",
        ),
        pytest.param(
            [-179.0, -10.0, 0.0, 45.0, 179.0],
            [-179.0, -10.0, 0.0, 45.0, 179.0],
            id="already-signed",
        ),
    ],
)
def test_normalize_longitude(lon, expected):
    """``normalize_longitude`` maps longitudes onto [-180, 180) and leaves signed ones unchanged.

    ERA5 grids arrive on 0..360 while the readers work in -180..180, so a longitude above 180 left unconverted would
    never match a footprint box in the western hemisphere. A caller can apply the helper without first checking a
    grid's convention only if it is a no-op on longitudes already in range.

    Converts each set of longitudes and asserts each lands on its signed equivalent: on 0..360, 180 maps to -180 and
    359.9 to -0.1; already-signed longitudes come back unchanged.
    """
    assert np.allclose(normalize_longitude(np.array(lon)), expected, atol=1e-6)
