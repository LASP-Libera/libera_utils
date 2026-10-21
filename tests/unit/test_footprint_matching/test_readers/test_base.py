"""Unit tests for the reader base classes: the box query, the point index, time selection.

Tests confirm:
- the point index returns exactly the points a brute-force box mask returns
- a grid is sliced inclusively on both edges and flattened row-major
- antimeridian and polar boxes are served whole
- ``DAILY`` ignores time, ``NEAREST_PASS`` keeps one pass, ``NEAREST_STEP`` serves one step
- the registration hook rejects a reader whose time attributes do not match its policy
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.base import (
    AncillaryReader,
    PassReader,
    StaticReader,
    SteppedGridReader,
    _PointIndex,
    split_dateline,
)
from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridSamples,
    OperationalMode,
    PointSamples,
    SampleSet,
    TimeMatch,
    VariableSpec,
)

_T0 = np.datetime64("2026-01-01T12:00:00", "ns")
_SPEC = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM)

# ---------------------------------------------------------------------------
# Minimal concrete readers used only in tests
# ---------------------------------------------------------------------------


class _GridReader(StaticReader):
    """A daily 1 degree global grid whose value is 100 * lat_index + lon_index."""

    READER_KEY = "_base_test_grid"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 111.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (_SPEC,)

    def _load_native(self) -> GridSamples:
        lats = np.arange(-89.5, 90.0, 1.0)
        lons = np.arange(-179.5, 180.0, 1.0)
        data = (np.arange(lats.size)[:, None] * 1000 + np.arange(lons.size)[None, :]).astype(np.float32)
        return GridSamples(lat_axis=lats, lon_axis=lons, data=data[np.newaxis])


class _PointReader(StaticReader):
    """A daily point cloud handed in at construction."""

    READER_KEY = "_base_test_points"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 1.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (_SPEC,)

    def __init__(self, points: PointSamples) -> None:
        super().__init__("unused.bin")
        self._points = points

    def _load_native(self) -> PointSamples:
        return self._points


class _PassReader(StaticReader):
    """A nearest-pass point cloud handed in at construction."""

    READER_KEY = "_base_test_pass"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 1.0
    TIME_MATCH = TimeMatch.NEAREST_PASS
    PASS_WINDOW = np.timedelta64(15, "m")
    MAX_TIME_OFFSET = np.timedelta64(3, "h")
    VARIABLES = (_SPEC,)

    def __init__(self, points: PointSamples) -> None:
        super().__init__("unused.bin")
        self._points = points

    def _load_native(self) -> PointSamples:
        return self._points


class _GranuleReader(PassReader):
    """Hourly granules handed in as ``{path: (start, end, PointSamples)}``; counts loads."""

    READER_KEY = "_base_test_granules"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 1.0
    PASS_WINDOW = np.timedelta64(15, "m")
    MAX_TIME_OFFSET = np.timedelta64(3, "h")
    VARIABLES = (_SPEC,)

    def __init__(self, granules: dict) -> None:
        super().__init__(list(granules))
        self._granules = granules
        self.loads: list = []

    def _granule_span(self, path):
        start, end, _ = self._granules[path]
        return start, end

    def _load_granule(self, path):
        self.loads.append(path)
        return self._granules[path][2]


class _StepReader(SteppedGridReader):
    """Hourly steps whose grid value is the step index; counts loads."""

    READER_KEY = "_base_test_steps"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 25.0
    MAX_TIME_OFFSET = np.timedelta64(1, "h")
    VARIABLES = (_SPEC,)

    def __init__(self) -> None:
        super().__init__("unused.nc")
        self.loads: list[int] = []

    def _step_times(self) -> np.ndarray:
        return _T0 + np.arange(4) * np.timedelta64(1, "h")

    def _load_step(self, index: int) -> GridSamples:
        self.loads.append(index)
        axis = np.array([0.0, 1.0])
        return GridSamples(lat_axis=axis, lon_axis=axis, data=np.full((1, 2, 2), float(index), dtype=np.float32))


def _random_points(rng: np.random.Generator, n: int, with_times: bool = False) -> PointSamples:
    lats = rng.uniform(-90.0, 90.0, n)
    lons = rng.uniform(-180.0, 180.0, n)
    values = np.arange(n, dtype=np.float32)[np.newaxis, :]
    times = None
    if with_times:
        times = _T0 + rng.integers(0, 24 * 3600, n).astype("timedelta64[s]")
    return PointSamples(lats=lats, lons=lons, values=values, times=times)


def _brute_force(points: PointSamples, bbox: BoundingBox) -> set[float]:
    inside = (
        (points.lats >= bbox.lat_min)
        & (points.lats <= bbox.lat_max)
        & (points.lons >= bbox.lon_min)
        & (points.lons <= bbox.lon_max)
    )
    return set(points.values[0, inside].tolist())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPointIndex:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_query_matches_brute_force_mask(self, seed):
        rng = np.random.default_rng(seed)
        points = _random_points(rng, 20_000)
        index = _PointIndex(points)
        for _ in range(25):
            lat0 = rng.uniform(-88.0, 85.0)
            lon0 = rng.uniform(-178.0, 175.0)
            bbox = BoundingBox(lat0, lat0 + rng.uniform(0.1, 3.0), lon0, lon0 + rng.uniform(0.1, 3.0))
            got = set(index.values[0, index.query(bbox)].tolist())
            assert got == _brute_force(points, bbox)

    def test_box_on_bin_edges_is_inclusive(self):
        points = PointSamples(lats=np.array([10.0, 11.0]), lons=np.array([20.0, 21.0]), values=np.array([[1.0, 2.0]]))
        index = _PointIndex(points)
        assert set(index.values[0, index.query(BoundingBox(10.0, 11.0, 20.0, 21.0))].tolist()) == {1.0, 2.0}

    def test_non_finite_coordinates_are_dropped(self):
        points = PointSamples(
            lats=np.array([0.5, np.nan, 0.6]), lons=np.array([0.5, 0.5, np.inf]), values=np.array([[1.0, 2.0, 3.0]])
        )
        index = _PointIndex(points)
        assert index.n_points == 1
        assert index.values.tolist() == [[1.0]]

    def test_nat_times_are_dropped(self):
        points = PointSamples(
            lats=np.array([0.5, 0.5]),
            lons=np.array([0.5, 0.5]),
            values=np.array([[1.0, 2.0]]),
            times=np.array([_T0, np.datetime64("NaT")], dtype="datetime64[ns]"),
        )
        assert _PointIndex(points).n_points == 1

    def test_empty_query(self):
        rng = np.random.default_rng(0)
        index = _PointIndex(_random_points(rng, 100))
        # A box with no points (the random cloud is sparse at this scale).
        assert index.query(BoundingBox(0.0, 0.001, 0.0, 0.001)).size == 0


class TestSplitDateline:
    def test_plain_box_is_returned_whole(self):
        bbox = BoundingBox(0.0, 1.0, 10.0, 11.0)
        assert split_dateline(bbox) == [bbox]

    def test_wrapping_box_splits_into_west_and_east(self):
        west, east = split_dateline(BoundingBox(0.0, 1.0, 179.0, 181.0, wraps_dateline=True))
        assert (west.lon_min, west.lon_max) == (179.0, 180.0)
        assert (east.lon_min, east.lon_max) == (-180.0, -179.0)


class TestGridQuery:
    def test_slice_is_inclusive_and_row_major(self, tmp_path):
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.5, 1.5, 10.5, 11.5), _T0)
        assert isinstance(samples, SampleSet)
        assert samples.n_samples == 4
        np.testing.assert_array_equal(samples.lats, [0.5, 0.5, 1.5, 1.5])
        np.testing.assert_array_equal(samples.lons, [10.5, 11.5, 10.5, 11.5])
        # value = 1000 * lat_index + lon_index; lat 0.5 is index 90, lon 10.5 is index 190.
        np.testing.assert_array_equal(samples.values[0], [90190, 90191, 91190, 91191])
        assert samples.values.dtype == np.float32
        assert samples.times is None
        assert samples.source == "_base_test_grid"

    def test_box_is_padded_by_half_a_cell(self, tmp_path):
        # A 111 km cell pads the box by 0.5 deg: the four grid points whose cells overlap a
        # box lying between them are served.
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.6, 1.4, 10.6, 11.4), _T0)
        assert samples.n_samples == 4
        np.testing.assert_array_equal(samples.lats, [0.5, 0.5, 1.5, 1.5])

    def test_box_beyond_the_padding_is_empty(self, tmp_path):
        # A 1 km-cell reader pads by 0.0045 deg: a box 0.4 deg from every point serves nothing,
        # a box around a point serves it.
        points = PointSamples(lats=np.array([0.5, 1.5]), lons=np.array([10.5, 11.5]), values=np.array([[1.0, 2.0]]))
        reader = _PointReader(points)
        assert reader.samples_in(BoundingBox(0.9, 1.1, 10.9, 11.1), _T0).n_samples == 0
        assert reader.samples_in(BoundingBox(0.4, 0.6, 10.4, 10.6), _T0).n_samples == 1

    def test_dateline_box_serves_both_halves(self, tmp_path):
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 179.0, 181.0, wraps_dateline=True), _T0)
        # Both halves, each padded by half a cell (0.5 deg) outward.
        assert set(samples.lons.tolist()) == {178.5, 179.5, -179.5, -178.5}

    def test_padding_across_the_antimeridian_wraps(self, tmp_path):
        # A plain box ending at 180.0 pads to 180.5: the padded box wraps and the first
        # column east of the line is served too.
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 179.2, 180.0), _T0)
        assert set(samples.lons.tolist()) == {179.5, -179.5}

    def test_polar_box_spans_every_longitude(self, tmp_path):
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(89.0, 90.0, -180.0, 180.0, is_polar=True), _T0)
        # Two latitude rows (89.5 and, through the padding, 88.5) at every longitude.
        assert samples.n_samples == 720

    def test_daily_reader_ignores_time(self, tmp_path):
        reader = _GridReader(tmp_path / "unused.nc")
        a = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0)
        b = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0 + np.timedelta64(20, "h"))
        np.testing.assert_array_equal(a.values, b.values)
        c = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), np.datetime64("NaT"))
        np.testing.assert_array_equal(a.values, c.values)


class TestNearestPass:
    @staticmethod
    def _two_passes() -> PointSamples:
        # Pass A at 12:00 (+0..60 s), pass B twelve hours later, same ground spot.
        n = 5
        lats = np.full(2 * n, 10.5)
        lons = np.full(2 * n, 20.5)
        values = np.concatenate([np.full(n, 1.0), np.full(n, 2.0)])[np.newaxis, :].astype(np.float32)
        times = np.concatenate(
            [
                _T0 + np.arange(n) * np.timedelta64(15, "s"),
                _T0 + np.timedelta64(12, "h") + np.arange(n) * np.timedelta64(15, "s"),
            ]
        )
        return PointSamples(lats=lats, lons=lons, values=values, times=times)

    def test_nearest_pass_only(self):
        reader = _PassReader(self._two_passes())
        box = BoundingBox(10.0, 11.0, 20.0, 21.0)
        near_a = reader.samples_in(box, _T0 + np.timedelta64(40, "m"))
        assert set(near_a.values[0].tolist()) == {1.0}
        assert near_a.n_samples == 5
        near_b = reader.samples_in(box, _T0 + np.timedelta64(11, "h"))
        assert set(near_b.values[0].tolist()) == {2.0}

    def test_beyond_max_offset_is_empty(self):
        reader = _PassReader(self._two_passes())
        samples = reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0 + np.timedelta64(6, "h"))
        assert samples.n_samples == 0

    def test_window_keeps_a_pass_together(self):
        # The pass spans 60 s; a footprint 10 min before its first sample keeps all of it.
        reader = _PassReader(self._two_passes())
        samples = reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0 - np.timedelta64(10, "m"))
        assert samples.n_samples == 5
        assert samples.times is not None

    def test_nat_footprint_time_raises(self):
        reader = _PassReader(self._two_passes())
        with pytest.raises(ValueError, match="NaT"):
            reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), np.datetime64("NaT"))

    def test_pass_reader_without_times_raises(self):
        points = PointSamples(lats=np.array([10.5]), lons=np.array([20.5]), values=np.array([[1.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="per-sample times"):
            _PassReader(points).samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0)


class TestNearestStep:
    def test_serves_nearest_step_and_reloads_on_change(self):
        reader = _StepReader()
        box = BoundingBox(0.0, 1.0, 0.0, 1.0)
        first = reader.samples_in(box, _T0 + np.timedelta64(10, "m"))
        assert set(first.values[0].tolist()) == {0.0}
        again = reader.samples_in(box, _T0 + np.timedelta64(20, "m"))
        assert set(again.values[0].tolist()) == {0.0}
        # Crossing the half-hour boundary moves to step 1, loaded once.
        later = reader.samples_in(box, _T0 + np.timedelta64(40, "m"))
        assert set(later.values[0].tolist()) == {1.0}
        assert reader.loads == [0, 1]

    def test_beyond_max_offset_is_empty(self):
        reader = _StepReader()
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0 + np.timedelta64(9, "h"))
        assert samples.n_samples == 0
        assert reader.loads == []

    def test_nearest_step_index_picks_the_closer_neighbor(self):
        reader = _StepReader()
        assert reader.nearest_step_index(_T0 + np.timedelta64(29, "m")) == 0
        assert reader.nearest_step_index(_T0 + np.timedelta64(31, "m")) == 1
        assert reader.nearest_step_index(_T0 - np.timedelta64(2, "h")) is None


class TestRegistrationValidation:
    def test_cannot_instantiate_abstract_classes(self):
        with pytest.raises(TypeError):
            AncillaryReader("anything")  # type: ignore[abstract]
        with pytest.raises(TypeError):
            StaticReader("anything")  # type: ignore[abstract]

    def test_missing_time_match_is_rejected_at_import(self):
        with pytest.raises(TypeError, match="TIME_MATCH"):

            class _NoPolicy(StaticReader):
                READER_KEY = "_base_test_no_policy"
                INSTRUMENT = "FAKE"
                RESOLUTION_KM = 1.0
                VARIABLES = (_SPEC,)

                def _load_native(self):
                    raise NotImplementedError

    def test_nearest_pass_without_window_is_rejected(self):
        with pytest.raises(TypeError, match="PASS_WINDOW"):

            class _NoWindow(StaticReader):
                READER_KEY = "_base_test_no_window"
                INSTRUMENT = "FAKE"
                RESOLUTION_KM = 1.0
                TIME_MATCH = TimeMatch.NEAREST_PASS
                MAX_TIME_OFFSET = np.timedelta64(1, "h")
                VARIABLES = (_SPEC,)

                def _load_native(self):
                    raise NotImplementedError

    def test_daily_with_window_is_rejected(self):
        with pytest.raises(TypeError, match="daily"):

            class _DailyWindow(StaticReader):
                READER_KEY = "_base_test_daily_window"
                INSTRUMENT = "FAKE"
                RESOLUTION_KM = 1.0
                TIME_MATCH = TimeMatch.DAILY
                MAX_TIME_OFFSET = np.timedelta64(1, "h")
                VARIABLES = (_SPEC,)

                def _load_native(self):
                    raise NotImplementedError


class TestFilePaths:
    def test_file_paths_and_single_file_path(self, tmp_path):
        one = _GridReader(tmp_path / "a.nc")
        assert one.file_paths == (tmp_path / "a.nc",)
        assert one.single_file_path == tmp_path / "a.nc"
        two = _GridReader([tmp_path / "a.nc", tmp_path / "b.nc"])
        assert len(two.file_paths) == 2
        with pytest.raises(ValueError, match="one granule per period"):
            _ = two.single_file_path

    def test_no_file_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            _GridReader([])


class TestPassReaderWindow:
    """A pass reader holds only the granules within MAX_TIME_OFFSET of the footprint time."""

    @staticmethod
    def _granules(tmp_path):
        granules = {}
        for hour in range(6):
            start = _T0 + np.timedelta64(hour, "h")
            end = start + np.timedelta64(59, "m")
            times = start + np.arange(4) * np.timedelta64(10, "m")
            points = PointSamples(
                lats=np.full(4, 10.5),
                lons=np.full(4, 20.5),
                values=np.full((1, 4), float(hour), dtype=np.float32),
                times=times,
            )
            granules[tmp_path / f"granule_{hour:02d}.nc"] = (start, end, points)
        return granules

    def test_only_granules_in_the_window_are_loaded(self, tmp_path):
        reader = _GranuleReader(self._granules(tmp_path))
        box = BoundingBox(10.0, 11.0, 20.0, 21.0)
        # Footprint at hour 0 + 5 min: granules 0..3 lie within +-3 h; 4 and 5 do not.
        samples = reader.samples_in(box, _T0 + np.timedelta64(5, "m"))
        assert sorted(p.name for p in reader.loads) == [f"granule_0{h}.nc" for h in range(4)]
        assert set(samples.values[0].tolist()) == {0.0}  # the nearest pass is hour 0's
        # Moving to hour 5 loads 4 and 5, releases 0 and 1, and never reloads 2 and 3.
        samples = reader.samples_in(box, _T0 + np.timedelta64(5, "h") + np.timedelta64(5, "m"))
        assert sorted(p.name for p in reader.loads) == [f"granule_0{h}.nc" for h in range(6)]
        assert sorted(p.name for p in reader._loaded) == [f"granule_0{h}.nc" for h in range(2, 6)]
        assert set(samples.values[0].tolist()) == {5.0}

    def test_no_granule_in_window_gives_no_samples(self, tmp_path):
        reader = _GranuleReader(self._granules(tmp_path))
        samples = reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0 + np.timedelta64(20, "h"))
        assert samples.n_samples == 0
        assert reader.loads == []

    def test_invalid_span_raises(self, tmp_path):
        granules = self._granules(tmp_path)
        path = next(iter(granules))
        start, end, points = granules[path]
        granules[path] = (end, start, points)
        with pytest.raises(ValueError, match="invalid time span"):
            _GranuleReader(granules).samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0)


class TestResolutionOfServedSamples:
    def test_declared_resolution_by_default(self, tmp_path):
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0)
        assert samples.resolution_km == 111.0
        empty = reader.samples_in(BoundingBox(89.0, 90.0, -180.0, 180.0, is_polar=True), np.datetime64("NaT"))
        assert empty.resolution_km == 111.0

    def test_measured_spacing_wins_over_the_declared_resolution(self):
        points = PointSamples(
            lats=np.array([0.5, 0.5]), lons=np.array([10.5, 10.6]), values=np.array([[1.0, 2.0]]), spacing_km=7.2
        )
        samples = _PointReader(points).samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0)
        assert samples.resolution_km == 7.2
        # The query padding follows the measured spacing too: 7.2 km is 0.032 deg.
        assert _PointReader(points).samples_in(BoundingBox(0.0, 1.0, 10.62, 10.63), _T0).n_samples == 1
