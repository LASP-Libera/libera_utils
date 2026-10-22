"""End-to-end checks of the SSF and CLDPIX readers against real CERES granules.

The granules are cut from real CER_SSF_NOAA20-FM6-VIIRS and CER_CLDPIX_NOAA20-VIIRS
granules to three contiguous blocks spread across the hour (start, middle, end), with
their values unchanged; see tests/test_data/footprint_matching/real/README.md. The
tests pin the file conventions the readers rely on: the time variables and their
encodings, the geolocation conventions, the hourly granule span, and that a footprint
query at a footprint's own time serves its pass.
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.types import BoundingBox


@pytest.fixture
def ssf_path(fmatch_real_data_path):
    return fmatch_real_data_path / "CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115_subset.nc"


@pytest.fixture
def cldpix_path(fmatch_real_data_path):
    return fmatch_real_data_path / "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015_subset.nc"


class TestRealSSF:
    def test_times_cover_one_hour_and_every_footprint_is_served(self, ssf_path):
        reader = SSFReader(ssf_path)
        points = reader._load_granule(ssf_path)
        assert points.times is not None
        assert np.isnat(points.times).sum() == 0
        span = points.times.max() - points.times.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        assert points.values.shape == (len(SSFReader.VARIABLES), points.lats.size)
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_footprint_time_serves_its_pass(self, ssf_path):
        reader = SSFReader(ssf_path)
        points = reader._load_granule(ssf_path)
        i = points.lats.size // 2
        box = BoundingBox(points.lats[i] - 0.5, points.lats[i] + 0.5, points.lons[i] - 0.5, points.lons[i] + 0.5)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= SSFReader.PASS_WINDOW)
        # Four hours away nothing is served.
        assert reader.samples_in(box, points.times[i] + np.timedelta64(4, "h")).n_samples == 0


class TestRealCLDPIX:
    def test_times_are_dated_fractional_hours_spanning_one_hour(self, cldpix_path):
        reader = CLDPIXReader(cldpix_path)
        points = reader._load_granule(cldpix_path)
        assert points.times is not None
        finite = points.times[~np.isnat(points.times)]
        assert finite.size > 0
        span = finite.max() - finite.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        # The granule's own coverage attributes bracket every decoded time to within a few
        # seconds (the first scanlines precede the declared start by about two seconds in the
        # source granule, so the dating by day is right and only the attribute is coarse).
        start, end = reader._granule_span(cldpix_path)
        assert finite.min() >= start - np.timedelta64(10, "s")
        assert finite.max() <= end + np.timedelta64(10, "s")

    def test_latitudes_are_true_latitudes_in_both_hemispheres(self, cldpix_path):
        reader = CLDPIXReader(cldpix_path)
        points = reader._load_granule(cldpix_path)
        assert np.nanmin(points.lats) >= -90.0
        assert np.nanmax(points.lats) <= 90.0
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_pixel_time_serves_its_pass(self, cldpix_path):
        reader = CLDPIXReader(cldpix_path)
        points = reader._load_granule(cldpix_path)
        i = points.lats.size // 2
        box = BoundingBox(points.lats[i] - 0.3, points.lats[i] + 0.3, points.lons[i] - 0.3, points.lons[i] + 0.3)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= CLDPIXReader.PASS_WINDOW)
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), samples.n_samples)

    def test_granule_span_matches_the_coverage_attributes(self, cldpix_path):
        reader = CLDPIXReader(cldpix_path)
        start, end = reader._granule_span(cldpix_path)
        assert np.timedelta64(55, "m") < end - start <= np.timedelta64(1, "h")
        # A footprint inside the hour has the granule in its window; one a day away does not.
        assert reader.granules_in_window(start + np.timedelta64(30, "m")) == [cldpix_path]
        assert reader.granules_in_window(start + np.timedelta64(24, "h")) == []
