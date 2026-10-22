"""Unit tests for the reader base classes: the box query, the point index, time selection.

Tests confirm:
- the point index returns exactly the points a brute-force box mask returns
- a grid is sliced inclusively on both edges and flattened row-major
- antimeridian and polar boxes are served whole
- ``DAILY`` ignores time, ``NEAREST_PASS`` keeps one pass, ``NEAREST_STEP`` serves one step
- the registration hook rejects a reader whose time attributes do not match its policy
- a reader missing any one abstract method cannot be instantiated, and is never registered
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.base import (
    AncillaryReader,
    StaticReader,
    SteppedGridReader,
    _PointIndex,
    split_dateline,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
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


class _StepReader(SteppedGridReader):
    """Hourly steps whose grid value is the step index; counts loads."""

    READER_KEY = "_base_test_steps"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 25.0
    MAX_TIME_OFFSET = np.timedelta64(1, "h")
    VARIABLES = (_SPEC,)

    def __init__(self, planes: int = 1, times: np.ndarray | None = None) -> None:
        super().__init__("unused.nc")
        self.loads: list[int] = []
        self._planes = planes
        self._times = _T0 + np.arange(4) * np.timedelta64(1, "h") if times is None else times

    def _step_times(self) -> np.ndarray:
        return self._times

    def _load_step(self, index: int) -> GridSamples:
        self.loads.append(index)
        axis = np.array([0.0, 1.0])
        data = np.full((self._planes, 2, 2), float(index), dtype=np.float32)
        return GridSamples(lat_axis=axis, lon_axis=axis, data=data)


class _WrongPlanesGridReader(StaticReader):
    """A daily grid reader that loads two planes for its one declared variable."""

    READER_KEY = "_base_test_wrong_planes"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 111.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (_SPEC,)

    def _load_native(self) -> GridSamples:
        axis = np.array([0.0, 1.0])
        return GridSamples(lat_axis=axis, lon_axis=axis, data=np.zeros((2, 2, 2), dtype=np.float32))


class _BoxHolder:
    """A box-queried native: samples on a 0.5 degree lattice, generated per query."""

    def __init__(self, n_variables: int = 1) -> None:
        self.n_variables = n_variables
        self.queries: list[BoundingBox] = []

    def query_box(self, bbox: BoundingBox):
        self.queries.append(bbox)
        lat_axis = np.arange(-89.75, 90.0, 0.5)
        lon_axis = np.arange(-179.75, 180.0, 0.5)
        lats = lat_axis[(lat_axis >= bbox.lat_min) & (lat_axis <= bbox.lat_max)]
        lons = lon_axis[(lon_axis >= bbox.lon_min) & (lon_axis <= bbox.lon_max)]
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        values = np.tile(lon_grid.ravel().astype(np.float32), (self.n_variables, 1))
        return lat_grid.ravel(), lon_grid.ravel(), values, None


class _BoxReader(StaticReader):
    """A daily reader whose native data answers box queries itself."""

    READER_KEY = "_base_test_box"
    INSTRUMENT = "FAKE"
    RESOLUTION_KM = 1.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (_SPEC,)

    def __init__(self, native: object) -> None:
        super().__init__("unused.hdf")
        self._held = native

    def _load_native(self):
        return self._held


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
        """``_PointIndex.query`` returns exactly the points a brute-force box mask over the whole cloud selects.

        The index only looks inside the 1 degree bins a box overlaps; if its bin bookkeeping skipped or
        double-counted a bin, footprints would silently gain or lose samples and their aggregated values
        would be wrong.

        For each seed, 20,000 random points are indexed and 25 random boxes up to 3 degrees across are
        queried; the set of values returned must equal the set a direct mask on the unsorted points gives.
        """
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
        """``_PointIndex.query`` includes points lying exactly on the box's edges, even where they sit on a bin edge.

        The box test is inclusive on all four edges; a point on the upper edge falls in the next 1 degree bin, so
        a query that read only the bins strictly inside the box would drop it.

        Two points are placed on opposite corners of a box whose edges are whole degrees, and both must come back
        from the query.
        """
        points = PointSamples(lats=np.array([10.0, 11.0]), lons=np.array([20.0, 21.0]), values=np.array([[1.0, 2.0]]))
        index = _PointIndex(points)
        assert set(index.values[0, index.query(BoundingBox(10.0, 11.0, 20.0, 21.0))].tolist()) == {1.0, 2.0}

    def test_non_finite_coordinates_are_dropped(self):
        """``_PointIndex`` drops points whose latitude or longitude is NaN or infinite when it is built.

        A point with no valid location cannot be binned or placed under a footprint, and keeping it would put
        an unlocatable value into the index.

        Three points are indexed, one with a NaN latitude and one with an infinite longitude; only the valid
        point remains, with its value.
        """
        points = PointSamples(
            lats=np.array([0.5, np.nan, 0.6]), lons=np.array([0.5, 0.5, np.inf]), values=np.array([[1.0, 2.0, 3.0]])
        )
        index = _PointIndex(points)
        assert index.n_points == 1
        assert index.values.tolist() == [[1.0]]

    def test_nat_times_are_dropped(self):
        """``_PointIndex`` drops points whose time is NaT when the cloud carries per-sample times.

        A sample with no time cannot be matched to a footprint time, so it must not reach the time selection.

        Two co-located points are indexed, one with a valid time and one with NaT; the index holds one point.
        """
        points = PointSamples(
            lats=np.array([0.5, 0.5]),
            lons=np.array([0.5, 0.5]),
            values=np.array([[1.0, 2.0]]),
            times=np.array([_T0, np.datetime64("NaT")], dtype="datetime64[ns]"),
        )
        assert _PointIndex(points).n_points == 1

    def test_empty_query(self):
        """``_PointIndex.query`` returns an empty index array for a box that holds no points.

        An empty footprint must come back as zero samples rather than an error, so the caller can score it as
        having no coverage.

        A sparse cloud of 100 random points is indexed and a box 0.001 degree across is queried; the result
        has size zero.
        """
        rng = np.random.default_rng(0)
        index = _PointIndex(_random_points(rng, 100))
        # A box with no points (the random cloud is sparse at this scale).
        assert index.query(BoundingBox(0.0, 0.001, 0.0, 0.001)).size == 0


class TestSplitDateline:
    def test_plain_box_is_returned_whole(self):
        """``split_dateline`` returns a box that does not cross the antimeridian unchanged, as a one-element list.

        Only a wrapping box needs splitting; altering a plain box would change which samples a footprint
        receives.

        A box from 10 to 11 degrees longitude is split and the result must be a list holding that same box.
        """
        bbox = BoundingBox(0.0, 1.0, 10.0, 11.0)
        assert split_dateline(bbox) == [bbox]

    def test_wrapping_box_splits_into_west_and_east(self):
        """``split_dateline`` splits a wrapping box into a west half ending at 180 and an east half from -180.

        Readers hold longitudes in [-180, 180], so a box given in the [0, 360) form must be cut into two plain
        boxes before either can be matched against the data.

        A wrapping box from 179 to 181 degrees is split; the halves must span 179 to 180 and -180 to -179.
        """
        west, east = split_dateline(BoundingBox(0.0, 1.0, 179.0, 181.0, wraps_dateline=True))
        assert (west.lon_min, west.lon_max) == (179.0, 180.0)
        assert (east.lon_min, east.lon_max) == (-180.0, -179.0)


class TestGridQuery:
    def test_slice_is_row_major_and_paired(self, tmp_path):
        """``StaticReader.samples_in`` on a grid serves the grid points inside the box, flattened row by row.

        Each served value must stay paired with its own latitude and longitude, or the weigher would weight
        values at the wrong locations.

        A 1 degree global test grid encoding each cell's row and column in its value is queried with a box around
        four grid points; the test asserts the four latitudes, longitudes and values in row-major order, the
        float32 dtype, no times, and the reader key as ``source``.
        """
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

    def test_grid_slice_includes_points_on_the_box_edges(self, tmp_path):
        """The grid slice behind ``samples_in`` keeps grid points lying exactly on the box's edges.

        Footprint boxes are built edge to edge, so a grid point on a shared edge must be served to both boxes; an
        exclusive slice would drop it from every footprint.

        Calls the unpadded slice directly (``samples_in`` would pad the edge points inside the box) with a box whose
        four corners are grid points, and asserts all four are returned.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        lats, lons, _, _ = reader._box_samples(reader._native, BoundingBox(0.5, 1.5, 10.5, 11.5))
        np.testing.assert_array_equal(lats, [0.5, 0.5, 1.5, 1.5])
        np.testing.assert_array_equal(lons, [10.5, 11.5, 10.5, 11.5])

    def test_box_is_padded_by_half_a_cell(self, tmp_path):
        # A 111 km cell pads the box by 0.5 deg: the four grid points whose cells overlap a
        # box lying between them are served.
        """``samples_in`` widens the box by half a native cell, so grid points whose cells overlap it are served.

        A sample whose center lies just outside a footprint's box can still cover part of it; without the padding
        a coarse grid would never serve the cells on the box's rim and the rim would be scored as missing.

        On the 111 km test grid, a box lying strictly between four grid points (0.1 degree inside each) is
        queried, and all four points must be served.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.6, 1.4, 10.6, 11.4), _T0)
        assert samples.n_samples == 4
        np.testing.assert_array_equal(samples.lats, [0.5, 0.5, 1.5, 1.5])

    def test_box_beyond_the_padding_is_empty(self, tmp_path):
        # A 1 km-cell reader pads by 0.0045 deg: a box 0.4 deg from every point serves nothing,
        # a box around a point serves it.
        """The query padding is only half a native cell, so points farther than that from the box are not served.

        Padding wider than the cell would pull in samples whose cells do not touch the footprint.

        A 1 km point reader (padding about 0.0045 degree) with two points is queried with a box 0.4 degree from
        both, which serves nothing, and with a box around one point, which serves that one.
        """
        points = PointSamples(lats=np.array([0.5, 1.5]), lons=np.array([10.5, 11.5]), values=np.array([[1.0, 2.0]]))
        reader = _PointReader(points)
        assert reader.samples_in(BoundingBox(0.9, 1.1, 10.9, 11.1), _T0).n_samples == 0
        assert reader.samples_in(BoundingBox(0.4, 0.6, 10.4, 10.6), _T0).n_samples == 1

    def test_dateline_box_serves_both_halves(self, tmp_path):
        """``samples_in`` serves grid points from both sides of the antimeridian for a box that wraps it.

        A footprint straddling the dateline covers ground on both ends of the [-180, 180] longitude axis, and
        serving only one half would lose the other half of its samples.

        A wrapping box from 179 to 181 degrees is queried on the 1 degree test grid; the served longitudes must
        be the two columns on each side of the line, including the half-cell padding.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 179.0, 181.0, wraps_dateline=True), _T0)
        # Both halves, each padded by half a cell (0.5 deg) outward.
        assert set(samples.lons.tolist()) == {178.5, 179.5, -179.5, -178.5}

    def test_padding_across_the_antimeridian_wraps(self, tmp_path):
        # A plain box ending at 180.0 pads to 180.5: the padded box wraps and the first
        # column east of the line is served too.
        """When padding pushes a plain box past 180 degrees, ``samples_in`` wraps it and serves columns east of it.

        A box that ends at the antimeridian still overlaps cells centered just across it; dropping the part
        of the padded box beyond 180 would lose those cells.

        A plain box from 179.2 to 180 degrees is queried on the 1 degree test grid; the served longitudes must
        be 179.5 and -179.5.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 179.2, 180.0), _T0)
        assert set(samples.lons.tolist()) == {179.5, -179.5}

    def test_polar_box_spans_every_longitude(self, tmp_path):
        """``samples_in`` serves every longitude for a box flagged as enclosing a pole.

        A footprint that covers a pole surrounds it in longitude, so every grid column near the pole lies under
        it.

        A polar box from 89 to 90 degrees latitude is queried on the 1 degree test grid; the padding reaches the
        88.5 row, so two full rows of 360 points (720 samples) must be served.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(89.0, 90.0, -180.0, 180.0, is_polar=True), _T0)
        # Two latitude rows (89.5 and, through the padding, 88.5) at every longitude.
        assert samples.n_samples == 720

    def test_daily_reader_ignores_time(self, tmp_path):
        """A ``DAILY`` reader serves the same samples for any footprint time, including NaT.

        A daily source has one value per location for the whole period, so the footprint time must neither
        change the result nor be required.

        The same box is queried at three times (noon, 20 hours later and NaT) and the served values must be
        identical.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        a = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0)
        b = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0 + np.timedelta64(20, "h"))
        np.testing.assert_array_equal(a.values, b.values)
        c = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), np.datetime64("NaT"))
        np.testing.assert_array_equal(a.values, c.values)


class TestBoxQueriedNative:
    def test_served_through_samples_in_padded_and_split(self):
        """A box-queried native gets padded, antimeridian-split boxes from ``samples_in`` and its samples are served.

        A projected source answers box queries itself and expects plain boxes inside [-180, 180]; it must still
        get the same padding and dateline handling as a grid or point cloud.

        A wrapping box is queried on a reader holding a recording box-queried stub; the test asserts two plain
        queries were made, the latitude range was widened, the samples from both halves are served with their
        values, and the declared resolution is reported.
        """
        holder = _BoxHolder()
        reader = _BoxReader(holder)
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 179.0, 181.0, wraps_dateline=True), _T0)
        # Two halves queried, each inside [-180, 180] and padded by half a 1 km cell.
        assert len(holder.queries) == 2
        assert all(-180.0 <= q.lon_min <= q.lon_max <= 180.0 for q in holder.queries)
        assert holder.queries[0].lat_min < 0.0 < 1.0 < holder.queries[0].lat_max
        assert set(samples.lons.tolist()) == {179.25, 179.75, -179.75, -179.25}
        np.testing.assert_array_equal(samples.values[0], samples.lons.astype(np.float32))
        assert samples.resolution_km == 1.0

    def test_native_is_loaded_once_and_held_as_returned(self, monkeypatch):
        """A ``StaticReader`` loads its box-queried native once and holds that object, unwrapped, across queries.

        The projected form exists so the source is never flattened to one point per pixel; reloading it per query
        would repeat the read for every footprint, and wrapping or copying it would bring back the memory cost it
        avoids.

        Counts calls to the reader's loader across two box queries and asserts it ran once and that the held native
        is the very stub object it returned.
        """
        holder = _BoxHolder()
        reader = _BoxReader(holder)
        loads = []
        monkeypatch.setattr(reader, "_load_native", lambda: loads.append(1) or holder)
        reader.samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)
        reader.samples_in(BoundingBox(5.0, 6.0, 5.0, 6.0), _T0)
        assert len(loads) == 1
        assert reader._native is holder

    def test_wrong_variable_count_raises(self):
        """A box-queried native whose variable count differs from the reader's ``VARIABLES`` raises ``ValueError``.

        Each row of served values is attributed to a declared variable, so a count mismatch would publish
        values under the wrong variable names.

        A stub declaring two variables is held by a reader declaring one, and the first query must raise with a
        message naming the box-queried native and its two variables.
        """
        with pytest.raises(ValueError, match="box-queried native with 2 variables"):
            _BoxReader(_BoxHolder(n_variables=2)).samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)

    def test_unknown_native_type_raises(self):
        """A ``StaticReader`` whose loader returns an unsupported type raises ``TypeError`` on the first query.

        The base class only knows how to query grids, point clouds and box-queried natives; anything else is a
        programming error in the reader and must stop the run rather than serve nothing.

        A reader whose loader returns a ``dict`` is queried and must raise a message naming the type ``dict``.
        """
        with pytest.raises(TypeError, match="native data of type dict"):
            _BoxReader({}).samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)

    def test_unknown_native_type_from_native_for_raises(self, monkeypatch):
        # A reader whose _native_for hands the query an unrecognized object.
        """The box query itself raises ``TypeError`` for a native of unsupported type handed to it by ``_native_for``.

        The load-time type check lives in ``StaticReader._native``; a native reaching the query by any other
        route must still stop the run rather than serve nothing.

        ``_native_for`` is patched to return a bare ``object``, and the query must raise a message naming the
        type ``object``.
        """
        reader = _BoxReader(_BoxHolder())
        monkeypatch.setattr(reader, "_native_for", lambda time: (object(),))
        with pytest.raises(TypeError, match="native data of type object"):
            reader.samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)


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
        """A ``NEAREST_PASS`` reader serves only the pass nearest the footprint time.

        Two passes over the same spot can be hours apart and see different conditions, so mixing them would
        blend observations that do not describe the footprint's moment.

        Two passes of five samples, 12 hours apart, cover one spot; a query 40 minutes after the first pass gets
        exactly its five samples, and a query an hour before the second pass gets only the second pass.
        """
        reader = _PassReader(self._two_passes())
        box = BoundingBox(10.0, 11.0, 20.0, 21.0)
        near_a = reader.samples_in(box, _T0 + np.timedelta64(40, "m"))
        assert set(near_a.values[0].tolist()) == {1.0}
        assert near_a.n_samples == 5
        near_b = reader.samples_in(box, _T0 + np.timedelta64(11, "h"))
        assert set(near_b.values[0].tolist()) == {2.0}

    def test_beyond_max_offset_is_empty(self):
        """A ``NEAREST_PASS`` query farther than ``MAX_TIME_OFFSET`` from every sample serves no samples.

        A pass too far from the footprint time no longer describes it; returning nothing lets the footprint be
        scored as having no coverage instead of taking a stale value.

        The two-pass fixture is queried 6 hours after the first pass, past the 3 hour limit to either pass, and
        the result must be empty.
        """
        reader = _PassReader(self._two_passes())
        samples = reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0 + np.timedelta64(6, "h"))
        assert samples.n_samples == 0

    def test_window_keeps_a_pass_together(self):
        # The pass spans 60 s; a footprint 10 min before its first sample keeps all of it.
        """``PASS_WINDOW`` keeps every sample of the nearest pass, not just the single closest one.

        A pass spans some seconds of observation, and keeping only the closest sample would discard most of the
        pass that covers the footprint.

        The first pass spans 60 seconds; a query 10 minutes before it must serve all five of its samples, with
        their times.
        """
        reader = _PassReader(self._two_passes())
        samples = reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0 - np.timedelta64(10, "m"))
        assert samples.n_samples == 5
        assert samples.times is not None

    def test_nat_footprint_time_raises(self):
        """A reader that selects in time raises ``ValueError`` when the footprint time is NaT.

        Without a valid time no pass can be chosen, and serving an arbitrary or empty selection would hide the
        missing time.

        A ``NEAREST_PASS`` reader is queried with a NaT time and must raise a message mentioning NaT.
        """
        reader = _PassReader(self._two_passes())
        with pytest.raises(ValueError, match="NaT"):
            reader.samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), np.datetime64("NaT"))

    def test_pass_reader_without_times_raises(self):
        """A ``NEAREST_PASS`` reader whose loader returns points without times raises ``ValueError``.

        Nearest-pass selection needs a time on every sample; without one the reader could not select a pass
        and would serve the wrong data.

        A pass reader is given a one-point cloud with no times and the first query must raise a message about
        per-sample times.
        """
        points = PointSamples(lats=np.array([10.5]), lons=np.array([20.5]), values=np.array([[1.0]], dtype=np.float32))
        with pytest.raises(ValueError, match="per-sample times"):
            _PassReader(points).samples_in(BoundingBox(10.0, 11.0, 20.0, 21.0), _T0)


class TestNearestStep:
    def test_serves_nearest_step_and_reloads_on_change(self):
        """``SteppedGridReader`` serves the step nearest the footprint time and loads a step only when it changes.

        Footprints arrive in time order, so reloading the same step for every footprint would multiply read time,
        while serving a stale step would pair a footprint with the wrong hour.

        On hourly steps whose values equal their index, queries at 10 and 20 minutes serve step 0 and a query at
        40 minutes serves step 1; the reader's load log must be exactly ``[0, 1]``.
        """
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
        """A ``NEAREST_STEP`` query farther than ``MAX_TIME_OFFSET`` from every step serves nothing and loads no step.

        A step too far from the footprint time no longer describes it, and loading a large grid only to discard
        it would waste read time.

        Hourly steps span 3 hours from noon; a query 9 hours after noon must return no samples and leave the
        load log empty.
        """
        reader = _StepReader()
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0 + np.timedelta64(9, "h"))
        assert samples.n_samples == 0
        assert reader.loads == []

    def test_step_with_wrong_plane_count_raises(self):
        """A stepped reader whose loaded step has a different number of planes than ``VARIABLES`` raises ``ValueError``.

        Each plane is attributed to a declared variable, so a mismatch would publish values under the wrong
        variable names.

        A test reader loading two planes for its one declared variable is queried and must raise a message
        naming the step.
        """
        reader = _StepReader(planes=2)
        with pytest.raises(ValueError, match="for step"):
            reader.samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)

    @pytest.mark.parametrize(
        ("times", "match"),
        [
            (np.array([], dtype="datetime64[ns]"), "no time steps"),
            (np.array([_T0, np.datetime64("NaT")], dtype="datetime64[ns]"), "missing value"),
            (np.array([_T0, _T0 + np.timedelta64(1, "h"), _T0 + np.timedelta64(1, "h")]), "strictly ascending"),
        ],
    )
    def test_invalid_step_times_raise(self, times, match):
        """``SteppedGridReader`` raises ``ValueError`` when the staged granules' step times are unusable.

        The nearest-step search assumes a non-empty, fully valid, strictly ascending time axis; any other axis
        would pick the wrong step without notice.

        The parameters supply an empty axis, an axis with a NaT step, and an axis with a repeated time; calling
        ``nearest_step_index`` must raise the matching message for each.
        """
        reader = _StepReader(times=times)
        with pytest.raises(ValueError, match=match):
            reader.nearest_step_index(_T0)

    def test_nearest_step_index_picks_the_closer_neighbor(self):
        """``nearest_step_index`` picks the closer of the steps around a time, or ``None`` beyond ``MAX_TIME_OFFSET``.

        A footprint must be paired with the step closest to its own time, and a time too far from every step
        must not be paired at all.

        On hourly steps with a 1 hour limit, 29 minutes past the first step maps to step 0, 31 minutes maps to
        step 1, and 2 hours before the first step maps to ``None``.
        """
        reader = _StepReader()
        assert reader.nearest_step_index(_T0 + np.timedelta64(29, "m")) == 0
        assert reader.nearest_step_index(_T0 + np.timedelta64(31, "m")) == 1
        assert reader.nearest_step_index(_T0 - np.timedelta64(2, "h")) is None


_ABSTRACT_CASES = [
    pytest.param(base, policy, name, id=f"{base.__name__}-{name}")
    for base, policy in (
        (StaticReader, {"TIME_MATCH": TimeMatch.DAILY}),
        (SteppedGridReader, {"MAX_TIME_OFFSET": np.timedelta64(1, "h")}),
    )
    for name in sorted(base.__abstractmethods__)
]


class TestRegistrationValidation:
    def test_cannot_instantiate_abstract_classes(self):
        """The abstract bases ``AncillaryReader`` and ``StaticReader`` cannot be instantiated.

        Readers must subclass a base and supply its loader; an instance of a base would have no data to serve.

        Instantiating each base directly must raise ``TypeError``.
        """
        with pytest.raises(TypeError):
            AncillaryReader("anything")  # type: ignore[abstract]
        with pytest.raises(TypeError):
            StaticReader("anything")  # type: ignore[abstract]

    @pytest.mark.parametrize(("base", "policy", "missing"), _ABSTRACT_CASES)
    def test_reader_missing_one_abstract_method_cannot_be_instantiated(self, base, policy, missing):
        """A reader that omits any one abstract method fails at instantiation, before the method could be called.

        A reader missing its loader would otherwise fail only on its first query, far from the definition at
        fault; it must also stay out of the registry so it is never selected for processing.

        The parameters cover each abstract method of ``StaticReader`` and ``SteppedGridReader``; a subclass
        defining all the others is built, and the test asserts only the omitted method remains abstract, the
        class is not registered, and instantiating it raises a ``TypeError`` naming the method.
        """
        namespace = {
            "READER_KEY": f"_base_test_missing{missing}",
            "INSTRUMENT": "FAKE",
            "RESOLUTION_KM": 1.0,
            "VARIABLES": (_SPEC,),
            **policy,
            **{name: lambda self, *args: None for name in base.__abstractmethods__ if name != missing},
        }
        incomplete = type(base)(f"_Missing{missing}", (base,), namespace)

        assert incomplete.__abstractmethods__ == frozenset({missing})
        assert namespace["READER_KEY"] not in ReaderRegistry.list_readers()
        with pytest.raises(TypeError, match=missing):
            incomplete("unused.nc")

    def test_missing_time_match_is_rejected_at_import(self):
        """Defining a concrete ``StaticReader`` without ``TIME_MATCH`` raises ``TypeError`` when the class is created.

        Without a time policy the reader cannot select samples in time; rejecting it when its module is imported
        surfaces the mistake before any footprint is processed.

        A reader class with every other required attribute is defined inside ``pytest.raises`` and must fail
        with a message naming ``TIME_MATCH``.
        """
        with pytest.raises(TypeError, match="TIME_MATCH"):

            class _NoPolicy(StaticReader):
                READER_KEY = "_base_test_no_policy"
                INSTRUMENT = "FAKE"
                RESOLUTION_KM = 1.0
                VARIABLES = (_SPEC,)

                def _load_native(self):
                    raise NotImplementedError

    def test_missing_instrument_is_rejected_at_import(self):
        """Defining a concrete reader without ``INSTRUMENT`` raises ``TypeError`` when the class is created.

        The instrument is recorded in every output variable's ``long_name`` for provenance, so a reader without
        it would publish variables with no stated source instrument.

        A reader class lacking ``INSTRUMENT`` is defined inside ``pytest.raises`` and must fail with a message
        naming ``INSTRUMENT``.
        """
        with pytest.raises(TypeError, match="INSTRUMENT"):

            class _NoInstrument(StaticReader):
                READER_KEY = "_base_test_no_instrument"
                RESOLUTION_KM = 1.0
                TIME_MATCH = TimeMatch.DAILY
                VARIABLES = (_SPEC,)

                def _load_native(self):
                    raise NotImplementedError

    def test_nearest_step_without_max_offset_is_rejected(self):
        """Defining a ``SteppedGridReader`` without ``MAX_TIME_OFFSET`` raises ``TypeError`` when the class is created.

        Without a limit, a footprint far from every step would still be paired with the nearest one, however
        distant in time.

        A stepped reader class lacking ``MAX_TIME_OFFSET`` is defined inside ``pytest.raises`` and must fail
        with a message naming that attribute.
        """
        with pytest.raises(TypeError, match="MAX_TIME_OFFSET"):

            class _NoOffset(SteppedGridReader):
                READER_KEY = "_base_test_no_offset"
                INSTRUMENT = "FAKE"
                RESOLUTION_KM = 1.0
                VARIABLES = (_SPEC,)

                def _step_times(self):
                    raise NotImplementedError

                def _load_step(self, index):
                    raise NotImplementedError

    def test_nearest_pass_without_window_is_rejected(self):
        """Defining a ``NEAREST_PASS`` reader without ``PASS_WINDOW`` raises ``TypeError`` when the class is created.

        The window is what keeps a pass's samples together; without it the reader could not tell which samples
        form the nearest pass.

        A nearest-pass reader class with ``MAX_TIME_OFFSET`` but no ``PASS_WINDOW`` is defined inside
        ``pytest.raises`` and must fail with a message naming ``PASS_WINDOW``.
        """
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
        """Defining a ``DAILY`` reader that sets a time-window attribute raises ``TypeError`` when the class is created.

        A daily reader applies no time selection, so a declared window would be silently ignored and mislead
        anyone reading the reader's definition.

        A daily reader class setting ``MAX_TIME_OFFSET`` is defined inside ``pytest.raises`` and must fail with
        a message mentioning daily.
        """
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


class TestLoadedShapeValidation:
    def test_point_cloud_with_wrong_variable_rows_raises(self):
        """A point cloud whose value rows differ in number from the reader's ``VARIABLES`` raises ``ValueError``.

        Each value row is attributed to a declared variable, so a mismatch would publish values under the wrong
        variable names.

        A one-variable point reader is handed a cloud with two value rows and the first query must raise a
        message about variable rows.
        """
        points = PointSamples(lats=np.array([0.5]), lons=np.array([0.5]), values=np.zeros((2, 1), dtype=np.float32))
        with pytest.raises(ValueError, match="variable rows"):
            _PointReader(points).samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)

    def test_grid_with_wrong_variable_planes_raises(self, tmp_path):
        """A grid whose planes differ in number from the reader's ``VARIABLES`` raises ``ValueError``.

        Each plane is attributed to a declared variable, so a mismatch would publish values under the wrong
        variable names.

        A one-variable grid reader whose loader returns two planes is queried and must raise a message about
        variable planes.
        """
        with pytest.raises(ValueError, match="variable planes"):
            _WrongPlanesGridReader(tmp_path / "unused.nc").samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0)


class TestFilePaths:
    def test_file_paths_and_single_file_path(self, tmp_path):
        """``file_paths`` holds the staged granules; ``single_file_path`` returns the one granule or raises if several.

        A reader that reads one granule per period must not silently pick one of several staged granules and
        ignore the rest.

        A reader built from one path exposes it in both properties; a reader built from two paths reports two
        ``file_paths`` and raises ``ValueError`` from ``single_file_path``.
        """
        one = _GridReader(tmp_path / "a.nc")
        assert one.file_paths == (tmp_path / "a.nc",)
        assert one.single_file_path == tmp_path / "a.nc"
        two = _GridReader([tmp_path / "a.nc", tmp_path / "b.nc"])
        assert len(two.file_paths) == 2
        with pytest.raises(ValueError, match="one granule per period"):
            _ = two.single_file_path

    def test_no_file_raises(self):
        """Constructing a reader with no staged granule raises ``ValueError``.

        A reader with nothing to read would otherwise fail later, or serve nothing, far from the staging step
        at fault.

        A reader built from an empty list must raise a message saying at least one granule is needed.
        """
        with pytest.raises(ValueError, match="at least one"):
            _GridReader([])


class TestResolutionOfServedSamples:
    def test_declared_resolution_by_default(self, tmp_path):
        """Served samples carry the reader's declared ``RESOLUTION_KM`` when the native data measures no spacing.

        The weigher sizes its coverage measure from the resolution each sample set reports, so it must be
        the reader's own cell size.

        A query on the 111 km test grid reports 111 km, and an hourly 25 km stepped reader queried 9 hours from
        every step, which serves an empty set without loading a step, still reports 25 km.
        """
        reader = _GridReader(tmp_path / "unused.nc")
        samples = reader.samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0)
        assert samples.resolution_km == 111.0
        empty = _StepReader().samples_in(BoundingBox(0.0, 1.0, 0.0, 1.0), _T0 + np.timedelta64(9, "h"))
        assert empty.n_samples == 0
        assert empty.resolution_km == 25.0

    def test_measured_spacing_wins_over_the_declared_resolution(self):
        """A spacing measured on the loaded point cloud overrides ``RESOLUTION_KM`` for the served samples and padding.

        A source subsampled on delivery stands for larger cells than the instrument's nominal one, and must be
        credited with the cell it really covers.

        A 1 km point reader loads points measured at 7.2 km spacing; the served set reports 7.2 km, and a box
        0.02 degree from a point still serves it, which only the wider 7.2 km padding reaches.
        """
        points = PointSamples(
            lats=np.array([0.5, 0.5]), lons=np.array([10.5, 10.6]), values=np.array([[1.0, 2.0]]), spacing_km=7.2
        )
        samples = _PointReader(points).samples_in(BoundingBox(0.0, 1.0, 10.0, 11.0), _T0)
        assert samples.resolution_km == 7.2
        # The query padding follows the measured spacing too: 7.2 km is 0.032 deg.
        assert _PointReader(points).samples_in(BoundingBox(0.0, 1.0, 10.62, 10.63), _T0).n_samples == 1
