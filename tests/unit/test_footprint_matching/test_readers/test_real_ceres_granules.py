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
    def test_every_footprint_has_a_time_and_the_times_span_one_hour(self, ssf_path):
        """``SSFReader._load_granule`` decodes a time for every footprint of a real SSF granule, spanning the hour.

        The pass matching selects SSF footprints by their observation time, so a footprint with an undecoded time would
        never match, and a time variable decoded with the wrong encoding would place the whole granule at the wrong
        hour.

        Loads the real SSF cut and asserts every time is set (no NaT), the times span between 55 min and 1 h, the values
        array holds one row per declared variable and one column per footprint, and every longitude lies in [-180, 180).
        """
        reader = SSFReader(ssf_path)
        points = reader._load_granule(ssf_path)
        assert points.times is not None
        assert np.isnat(points.times).sum() == 0
        span = points.times.max() - points.times.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        assert points.values.shape == (len(SSFReader.VARIABLES), points.lats.size)
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_footprint_time_serves_its_pass(self, ssf_path):
        """``SSFReader.samples_in`` serves a real SSF footprint's own pass when queried at its time and place.

        This is the end-to-end path a Libera footprint takes: if the geolocation or time conventions of the file were
        read wrong, a query at a known SSF footprint would find nothing, or a pass from the wrong time, and the matched
        footprint would carry no or wrong SSF values.

        Takes the middle footprint of the real SSF cut, queries a 1-degree box around it at its time and asserts samples
        are served, all within ``PASS_WINDOW`` of that time, and that the same box four hours later serves none.
        """
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
        """``CLDPIXReader._load_granule`` dates a real granule's fractional-hour ``Observation_Time`` onto its day.

        CLDPIX stores times as hours of the day with the date only in a global attribute, so a decoding that dropped or
        misread the date would put every pixel on the wrong day and no Libera footprint would ever match it.

        Loads the real CLDPIX cut and asserts it has set times spanning between 55 min and 1 h, and that they fall
        within 10 s of the granule's ``time_coverage_start``/``time_coverage_end`` as read by ``_granule_span``.
        """
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
        """``CLDPIXReader._load_granule`` returns latitudes converted from the file's colatitude, within [-90, 90].

        CLDPIX stores colatitude (0 at the North Pole to 180 at the South Pole); read as latitude, the southern
        hemisphere would fall outside the valid range and be discarded and the rest would be misplaced, so cloud values
        would land on the wrong footprints.

        Loads the real CLDPIX cut, which crosses the equator, and asserts every latitude lies in [-90, 90], that some
        are south and some north of the equator, and every longitude lies in [-180, 180).
        """
        reader = CLDPIXReader(cldpix_path)
        points = reader._load_granule(cldpix_path)
        assert np.nanmin(points.lats) >= -90.0
        assert np.nanmax(points.lats) <= 90.0
        assert np.nanmin(points.lats) < 0.0 < np.nanmax(points.lats)
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_pixel_time_serves_its_pass(self, cldpix_path):
        """``CLDPIXReader.samples_in`` serves a real CLDPIX pixel's own pass when queried at its time and place.

        This is the end-to-end path a Libera footprint takes through the cloud reader: a wrong geolocation or time
        convention would leave a footprint over a known pixel with no cloud values, or with values from another pass.

        Takes the middle pixel of the real CLDPIX cut, queries a 0.6-degree box around it at its time and asserts
        samples are served, all within ``PASS_WINDOW`` of that time, with one value row per declared variable.
        """
        reader = CLDPIXReader(cldpix_path)
        points = reader._load_granule(cldpix_path)
        i = points.lats.size // 2
        box = BoundingBox(points.lats[i] - 0.3, points.lats[i] + 0.3, points.lons[i] - 0.3, points.lons[i] + 0.3)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= CLDPIXReader.PASS_WINDOW)
        assert samples.values.shape == (len(CLDPIXReader.VARIABLES), samples.n_samples)

    def test_granule_span_matches_the_coverage_attributes(self, cldpix_path):
        """``CLDPIXReader._granule_span`` reads an hour-long span from a real granule that ``granules_in_window`` uses.

        The span decides which granules are held for a footprint; a misread span would drop the granule a footprint
        falls in, losing its cloud values, or hold granules a day away.

        Reads the real CLDPIX cut's span, asserts it lasts between 55 min and 1 h, and asserts ``granules_in_window``
        returns the granule for a time 30 min into the span and nothing for a time 24 h later.
        """
        reader = CLDPIXReader(cldpix_path)
        start, end = reader._granule_span(cldpix_path)
        assert np.timedelta64(55, "m") < end - start <= np.timedelta64(1, "h")
        # A footprint inside the hour has the granule in its window; one a day away does not.
        assert reader.granules_in_window(start + np.timedelta64(30, "m")) == [cldpix_path]
        assert reader.granules_in_window(start + np.timedelta64(24, "h")) == []
