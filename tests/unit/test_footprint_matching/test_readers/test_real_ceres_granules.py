"""End-to-end checks of the SSF and CLDPIX readers against real CERES granules.

The granules are not in the repository. The tests look for them under the directory named
by the ``FMATCH_REAL_GRANULE_DIR`` environment variable, then under ``external_data/CERES``
at the repository root, and are skipped when neither holds a matching file. They pin the
file conventions the readers rely on: the time variables and their encodings, the
geolocation conventions, the hourly granule span, and that a footprint query at a
footprint's own time serves its pass.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.types import BoundingBox

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _find(pattern: str) -> Path | None:
    roots = [Path(p) for p in (os.environ.get("FMATCH_REAL_GRANULE_DIR"),) if p]
    roots.append(_REPO_ROOT / "external_data" / "CERES")
    for root in roots:
        if root.is_dir():
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


_SSF = _find("CER_SSF_NOAA20-*.nc")
_CLDPIX = _find("CER_CLDPIX_NOAA20-*.nc")


@pytest.mark.skipif(
    _SSF is None, reason="no CER_SSF_NOAA20 granule staged (FMATCH_REAL_GRANULE_DIR or external_data/CERES)"
)
class TestRealSSF:
    def test_times_cover_one_hour_and_every_footprint_is_served(self):
        reader = SSFReader(_SSF)
        points = reader._load_granule(_SSF)
        assert points.times is not None
        assert np.isnat(points.times).sum() == 0
        span = points.times.max() - points.times.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        assert points.values.shape == (len(SSFReader.VARIABLES), points.lats.size)
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_footprint_time_serves_its_pass(self):
        reader = SSFReader(_SSF)
        points = reader._load_granule(_SSF)
        i = points.lats.size // 2
        box = BoundingBox(points.lats[i] - 0.5, points.lats[i] + 0.5, points.lons[i] - 0.5, points.lons[i] + 0.5)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= SSFReader.PASS_WINDOW)
        # Four hours away nothing is served.
        assert reader.samples_in(box, points.times[i] + np.timedelta64(4, "h")).n_samples == 0


@pytest.mark.skipif(
    _CLDPIX is None, reason="no CER_CLDPIX_NOAA20 granule staged (FMATCH_REAL_GRANULE_DIR or external_data/CERES)"
)
class TestRealCLDPIX:
    def test_times_are_dated_fractional_hours_spanning_one_hour(self):
        reader = CLDPIXReader(_CLDPIX)
        points = reader._load_granule(_CLDPIX)
        assert points.times is not None
        finite = points.times[~np.isnat(points.times)]
        assert finite.size > 0
        span = finite.max() - finite.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        # The granule's own coverage attributes bracket every decoded time to within a few
        # seconds (the first scanlines precede the declared start by about two seconds in the
        # staged granule, so the dating by day is right and only the attribute is coarse).
        start, end = reader._granule_span(_CLDPIX)
        assert finite.min() >= start - np.timedelta64(10, "s")
        assert finite.max() <= end + np.timedelta64(10, "s")

    def test_latitudes_are_true_latitudes_in_both_hemispheres(self):
        reader = CLDPIXReader(_CLDPIX)
        points = reader._load_granule(_CLDPIX)
        assert np.nanmin(points.lats) >= -90.0
        assert np.nanmax(points.lats) <= 90.0
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_pixel_time_serves_its_pass(self):
        reader = CLDPIXReader(_CLDPIX)
        points = reader._load_granule(_CLDPIX)
        i = points.lats.size // 2
        box = BoundingBox(points.lats[i] - 0.3, points.lats[i] + 0.3, points.lons[i] - 0.3, points.lons[i] + 0.3)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= CLDPIXReader.PASS_WINDOW)
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), samples.n_samples)

    def test_granule_span_matches_the_coverage_attributes(self):
        reader = CLDPIXReader(_CLDPIX)
        start, end = reader._granule_span(_CLDPIX)
        assert np.timedelta64(55, "m") < end - start <= np.timedelta64(1, "h")
        # A footprint inside the hour has the granule in its window; one a day away does not.
        assert reader.granules_in_window(start + np.timedelta64(30, "m")) == [_CLDPIX]
        assert reader.granules_in_window(start + np.timedelta64(24, "h")) == []
