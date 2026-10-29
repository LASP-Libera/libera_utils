"""Abstract base classes for the ancillary data reader plugins.

A reader turns one external ancillary dataset into *native samples*: the source's own
pixels, footprints or grid cells, each at its own location, with no averaging, binning
or interpolation applied. The PSF weigher then weights every sample where it lies and
the aggregation engine reduces the weighted samples to one value per variable, so a
value is transformed exactly once on its way into a FMATCH product.

Three concrete contracts
------------------------
* :class:`StaticReader` -- the source is one body of data for the staged period: a
  daily composite, a yearly map, or swath data carrying a time per sample. The reader
  loads it once (:meth:`StaticReader._load_native`) as :class:`~libera_utils.footprint_matching.types.GridSamples`,
  :class:`~libera_utils.footprint_matching.types.PointSamples`, or a native that answers box queries itself (see
  ``_BoxNative``); the base class indexes a point cloud and answers every query
  from memory.
* :class:`SteppedGridReader` -- the source is a time series of global grids (ERA5
  hourly steps). The reader lists the step times (:meth:`SteppedGridReader._step_times`)
  and loads one step on demand (:meth:`SteppedGridReader._load_step`); the base class
  keeps only the step nearest the footprint time in memory.
* :class:`PassReader` -- the source is swath data in hourly granules, each with a time
  per sample (CERES SSF, CLDPIX). The reader reads a granule's time span
  (:meth:`PassReader._granule_span`) and loads one granule as points
  (:meth:`PassReader._load_granule`); the base class holds only the granules whose span
  lies within ``MAX_TIME_OFFSET`` of the current footprint time.

Geographic query
----------------
:meth:`AncillaryReader.samples_in` takes a footprint's :class:`~libera_utils.footprint_matching.types.BoundingBox` and time.
A grid is sliced on its axes with ``searchsorted``. A source whose pixels are regular
in a projection, not in latitude/longitude (IGBP's tiles on the MODIS sinusoidal grid), is held in its
projected form and geolocates only the block a box selects (``_BoxNative``). A
point cloud is sorted once, at
load, into 1 degree latitude/longitude bins held as offset ranges (one permuted copy
of the data, the unsorted one released), so a box query touches only the bins the box
overlaps before applying the exact box mask. A box that crosses the antimeridian is
split into its two halves and the halves' samples are concatenated. A box that
encloses a pole spans every longitude, as the geometry module builds it.

Time selection
--------------
After the box query, the reader's :class:`~libera_utils.footprint_matching.types.TimeMatch`
policy selects in time: nothing for a ``DAILY`` source, the nearest step for a
``NEAREST_STEP`` source, and the nearest pass (every sample within ``PASS_WINDOW`` of
the observation time of the sample nearest the footprint time) for a ``NEAREST_PASS``
source. Beyond
``MAX_TIME_OFFSET`` the query returns no samples, which the orchestrator scores as zero
coverage and flags.

Auto-registration via ``__init_subclass__``
-------------------------------------------
Every concrete subclass of :class:`AncillaryReader` is added to
``ReaderRegistry._registry`` under its ``READER_KEY`` when its module is imported;
no manual registration call is needed. Abstract intermediates are skipped.
"""

import abc
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridSamples,
    PointSamples,
    SampleSet,
    TimeMatch,
    VariableSpec,
    with_standard_deviation_companions,
)

# Edge length of the latitude/longitude bins the point index sorts a point cloud into.
# 1 degree keeps the bin table small (180 x 360 offsets) while a footprint box, at most a
# few degrees across, touches only a handful of bins.
POINT_INDEX_BIN_DEG: float = 1.0
# Kilometers per degree of latitude, for turning a native cell size into query padding.
_KM_PER_DEG: float = 111.0
_N_LAT_BINS: int = int(round(180.0 / POINT_INDEX_BIN_DEG))
_N_LON_BINS: int = int(round(360.0 / POINT_INDEX_BIN_DEG))


def split_dateline(bbox: BoundingBox) -> list[BoundingBox]:
    """Split an antimeridian-crossing box into its west and east halves.

    A wrapping box is given in the [0, 360) form with ``lon_min < 180 < lon_max`` (see
    :class:`~libera_utils.footprint_matching.types.BoundingBox`). The west half
    ``[lon_min, 180]`` is already a plain box; the east half runs from -180 to
    ``lon_max - 360``.

    Parameters
    ----------
    bbox : BoundingBox
        Any box.

    Returns
    -------
    list[BoundingBox]
        ``[bbox]`` for a box that does not wrap, else ``[west, east]``.
    """
    if not bbox.wraps_dateline:
        return [bbox]
    east_max = bbox.lon_max - 360.0
    west = BoundingBox(bbox.lat_min, bbox.lat_max, bbox.lon_min, 180.0, False, bbox.is_polar, bbox.truncated)
    east = BoundingBox(bbox.lat_min, bbox.lat_max, -180.0, east_max, False, bbox.is_polar, bbox.truncated)
    return [west, east]


def in_box(lats: np.ndarray, lons: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    """Mask of the points inside a plain (non-wrapping) box, edges included.

    Parameters
    ----------
    lats, lons : np.ndarray
        Point coordinates in degrees, longitude in -180..180, of the same shape.
    bbox : BoundingBox
        A plain box, such as one half from :func:`split_dateline`.

    Returns
    -------
    np.ndarray
        Boolean array the shape of ``lats``, true where the point lies inside ``bbox``.
    """
    return (lats >= bbox.lat_min) & (lats <= bbox.lat_max) & (lons >= bbox.lon_min) & (lons <= bbox.lon_max)


class _PointIndex:
    """A point cloud sorted into bins of :data:`POINT_INDEX_BIN_DEG` degrees for bounding-box queries.

    Built once per loaded :class:`PointSamples`. Samples with a non-finite coordinate,
    or a NaT time when times are present, are dropped here; the rest are permuted into
    bin order and held as the only copy (the unsorted arrays are not kept).

    Parameters
    ----------
    points : PointSamples
        The reader's native points.

    Raises
    ------
    ValueError
        If a kept point lies outside latitude [-90, 90] or longitude [-180, 180].
    """

    def __init__(self, points: PointSamples) -> None:
        lats = np.asarray(points.lats, dtype=np.float64)
        lons = np.asarray(points.lons, dtype=np.float64)
        keep = np.isfinite(lats) & np.isfinite(lons)
        times = points.times
        if times is not None:
            times = np.asarray(times, dtype="datetime64[ns]")
            keep &= ~np.isnat(times)
        kept = np.flatnonzero(keep)
        outside = (np.abs(lats[kept]) > 90.0) | (np.abs(lons[kept]) > 180.0)
        if np.any(outside):
            first = kept[np.flatnonzero(outside)[0]]
            raise ValueError(
                f"{int(outside.sum())} of {kept.size} points lie outside latitude [-90, 90] or longitude "
                f"[-180, 180], the first at lat {lats[first]}, lon {lons[first]}. A reader's _load_native must "
                "return longitudes in [-180, 180]: convert a 0..360 source with normalize_longitude first."
            )

        # The clip files the upper edges, latitude 90 and longitude 180, in the last bin.
        lat_bin = np.clip(np.floor((lats[kept] + 90.0) / POINT_INDEX_BIN_DEG).astype(np.int64), 0, _N_LAT_BINS - 1)
        lon_bin = np.clip(np.floor((lons[kept] + 180.0) / POINT_INDEX_BIN_DEG).astype(np.int64), 0, _N_LON_BINS - 1)
        cell = lat_bin * _N_LON_BINS + lon_bin
        order = np.argsort(cell, kind="stable")
        # One gather per array: the kept samples, in bin order.
        idx = kept[order]
        self.lats = lats[idx]
        self.lons = lons[idx]
        self.values = np.asarray(points.values, dtype=np.float32)[:, idx]
        self.times = times[idx] if times is not None else None
        self.spacing_km = points.spacing_km
        # offsets[c] .. offsets[c + 1] is the index range of bin c in the sorted arrays.
        self._offsets = np.searchsorted(cell[order], np.arange(_N_LAT_BINS * _N_LON_BINS + 1))

    @property
    def n_points(self) -> int:
        """Number of indexed points."""
        return int(self.lats.size)

    def query(self, bbox: BoundingBox) -> np.ndarray:
        """Return the sorted-array indices of the points inside ``bbox`` (inclusive edges).

        Parameters
        ----------
        bbox : BoundingBox
            A non-wrapping box (callers split wrapping boxes first).

        Returns
        -------
        np.ndarray
            1-D int64 indices into :attr:`lats` / :attr:`lons` / :attr:`values`.
        """
        lat0 = int(np.clip(np.floor((bbox.lat_min + 90.0) / POINT_INDEX_BIN_DEG), 0, _N_LAT_BINS - 1))
        lat1 = int(np.clip(np.floor((bbox.lat_max + 90.0) / POINT_INDEX_BIN_DEG), 0, _N_LAT_BINS - 1))
        lon0 = int(np.clip(np.floor((bbox.lon_min + 180.0) / POINT_INDEX_BIN_DEG), 0, _N_LON_BINS - 1))
        lon1 = int(np.clip(np.floor((bbox.lon_max + 180.0) / POINT_INDEX_BIN_DEG), 0, _N_LON_BINS - 1))
        ranges = []
        for row in range(lat0, lat1 + 1):
            start = self._offsets[row * _N_LON_BINS + lon0]
            stop = self._offsets[row * _N_LON_BINS + lon1 + 1]
            if stop > start:
                ranges.append(np.arange(start, stop))
        if not ranges:
            return np.empty(0, dtype=np.int64)
        candidates = np.concatenate(ranges)
        return candidates[in_box(self.lats[candidates], self.lons[candidates], bbox)]


@runtime_checkable
class _BoxNative(Protocol):
    """A native body of data that answers a box query itself.

    For a source held in a projected form (IGBP's tiles on the MODIS sinusoidal grid), where flattening
    every pixel to a latitude/longitude point would multiply its size. The holder
    geolocates only the block a query selects.

    Attributes
    ----------
    n_variables : int
        Number of variables in every ``values`` array it returns.
    """

    n_variables: int

    def query_box(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(lats, lons, values, times)`` of the samples inside one plain box.

        Parameters
        ----------
        bbox : BoundingBox
            A non-wrapping box inside [-180, 180] (callers split wrapping boxes first).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray or None]
            ``(n,)`` float64 latitudes and longitudes, ``(n_variables, n)`` float32
            values (inclusive box edges), and ``(n,)`` times or ``None``.
        """
        ...


_Native = GridSamples | _PointIndex | _BoxNative


class AncillaryReader(abc.ABC):
    """Abstract base class for all ancillary data readers.

    Each concrete subclass reads one external data source (IGBP land cover, NSIDC
    sea ice, ERA5 wind, ...) and serves the native samples inside a footprint's
    bounding box, selected in time by its :class:`~libera_utils.footprint_matching.types.TimeMatch` policy. Subclass
    :class:`StaticReader` or :class:`SteppedGridReader`, not this class directly.

    Class-Level Attributes (required on every concrete subclass)
    -------------------------------------------------------------
    READER_KEY : str
        Unique string key used to register the reader in ``ReaderRegistry``.
        Examples: ``"igbp"``, ``"nise"``, ``"era5_surface"``, ``"viirs_aod"``.
    INSTRUMENT : str
        Instrument / platform token recorded in each output variable's ``long_name``
        for provenance, e.g. ``"NOAA20"``, ``"MODIS"``, ``"ECMWF"``. The variable
        *name* is ``f"{READER_KEY}_{spec.name}"``; the instrument is appended to the
        product-definition ``long_name`` as ``"... ({INSTRUMENT})"``.
    RESOLUTION_KM : float
        Native sample spacing of the source in km. The weigher sizes its coverage
        measure from it, unless the reader measures the spacing on the granule itself
        (``PointSamples.spacing_km``), which then takes precedence.
    VARIABLES : tuple[VariableSpec, ...]
        Ordered tuple of the variables this reader *reads from its source*. Axis 0 of
        every ``values`` array follows this order.
    TIME_MATCH : TimeMatch
        The time-selection policy (see the module docstring).
    PASS_WINDOW : np.timedelta64 or None
        ``NEAREST_PASS`` only: samples within this window of the observation time of the
        sample nearest the footprint time are the nearest pass. Required for that policy;
        ``None`` otherwise.
    MAX_TIME_OFFSET : np.timedelta64 or None
        ``NEAREST_STEP`` and ``NEAREST_PASS``: the largest sample-to-footprint time
        offset accepted; beyond it the query returns no samples. Required for those
        policies; ``None`` for ``DAILY``.
    TIMESTAMP_SOURCE : str or None
        For cloud-property products, ``'radiometer'`` or ``'camera'``; recorded on every
        :class:`~libera_utils.footprint_matching.types.SampleSet` served. ``None`` (the default) for surface and static products.
    ADDITIONAL_PRODUCT_VARIABLES : tuple[VariableSpec, ...]
        Extra variables this reader contributes to the *product definition* that are
        derived during PSF aggregation rather than read (e.g. IGBP's ranked scenes).
        Standard-deviation companions are added automatically (see
        ``product_variable_specs``).

    Parameters
    ----------
    file_paths : Sequence[Path]
        The staged granules of this source, on the local filesystem; each element is
        converted with ``Path``. A reader that reads one granule per period uses
        :attr:`single_file_path`; one that concatenates granules uses :attr:`file_paths`.

    Raises
    ------
    TypeError
        If ``file_paths`` is a single ``str`` or ``Path`` rather than a sequence of them.
    ValueError
        If ``file_paths`` is empty.
    FileNotFoundError
        If any path in ``file_paths`` is not an existing file; the message names the
        reader and every such path.
    """

    READER_KEY: str
    INSTRUMENT: str
    RESOLUTION_KM: float
    VARIABLES: tuple[VariableSpec, ...]
    TIME_MATCH: TimeMatch
    PASS_WINDOW: np.timedelta64 | None = None
    MAX_TIME_OFFSET: np.timedelta64 | None = None
    TIMESTAMP_SOURCE: str | None = None
    ADDITIONAL_PRODUCT_VARIABLES: tuple[VariableSpec, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate and register every concrete subclass in ReaderRegistry.

        Abstract intermediates (classes that still have ``abc.abstractmethod`` members)
        are skipped. A concrete reader missing ``INSTRUMENT``, ``TIME_MATCH``, or the
        time-window attributes its policy needs fails here, at import, rather than on
        the first footprint. So does a second class registering a ``READER_KEY`` that
        another class already holds; re-executing a reader module (``importlib.reload``,
        IPython ``%autoreload``) defines a new class under the same key and raises too.

        Raises
        ------
        TypeError
            If a concrete subclass lacks a required class attribute, declares a window
            attribute inconsistent with its ``TIME_MATCH``, or reuses a ``READER_KEY``
            registered to a different class.
        """
        super().__init_subclass__(**kwargs)
        # ABCMeta sets __abstractmethods__ only after type.__new__ returns, and this hook runs inside
        # type.__new__, so abstractness is read from the members, as ABCMeta itself computes it.
        if any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls)):
            return
        if getattr(cls, "READER_KEY", None) is None:
            return

        key = cls.READER_KEY
        if getattr(cls, "INSTRUMENT", None) is None:
            raise TypeError(
                f"Reader {cls.__name__!r} (READER_KEY={key!r}) must define an INSTRUMENT class attribute; "
                "it is recorded in each output variable's long_name as '... ({INSTRUMENT})'."
            )
        policy = getattr(cls, "TIME_MATCH", None)
        if not isinstance(policy, TimeMatch):
            raise TypeError(
                f"Reader {cls.__name__!r} (READER_KEY={key!r}) must set TIME_MATCH to a TimeMatch member; "
                f"got {policy!r}."
            )
        needs_offset = policy in (TimeMatch.NEAREST_STEP, TimeMatch.NEAREST_PASS)
        if needs_offset and cls.MAX_TIME_OFFSET is None:
            raise TypeError(
                f"Reader {cls.__name__!r} uses TIME_MATCH={policy.value!r} and must set MAX_TIME_OFFSET "
                "(a numpy timedelta64), the largest sample-to-footprint time offset it accepts."
            )
        if policy is TimeMatch.NEAREST_PASS and cls.PASS_WINDOW is None:
            raise TypeError(
                f"Reader {cls.__name__!r} uses TIME_MATCH='nearest_pass' and must set PASS_WINDOW "
                "(a numpy timedelta64), the span that keeps one pass together."
            )
        if policy is TimeMatch.DAILY and (cls.MAX_TIME_OFFSET is not None or cls.PASS_WINDOW is not None):
            raise TypeError(
                f"Reader {cls.__name__!r} uses TIME_MATCH='daily', which applies no time selection; "
                "remove its MAX_TIME_OFFSET / PASS_WINDOW."
            )

        from libera_utils.footprint_matching.readers.registry import ReaderRegistry  # noqa: PLC0415

        registered = ReaderRegistry._registry.get(key)
        if registered is not None and registered is not cls:
            raise TypeError(
                f"Reader {cls.__module__}.{cls.__qualname__} uses READER_KEY={key!r}, which is already registered "
                f"to {registered.__module__}.{registered.__qualname__}. Give each reader a unique READER_KEY."
            )
        ReaderRegistry._registry[key] = cls

    @classmethod
    def product_variable_specs(cls) -> tuple[VariableSpec, ...]:
        """Return every variable this reader contributes to the product definition.

        This is the full set of *output* variables -- what appears in the FMATCH
        product definition YAMLs and what the product/test cross-check is built
        against. It is distinct from ``VARIABLES`` (the smaller set actually read
        from the source file) and is composed of, in order:

        1. each read variable in ``VARIABLES``;
        2. a ``<name>_standard_deviation`` companion for every *continuous*
           (mean-aggregated) read variable, inserted right after its parent by
           :func:`~libera_utils.footprint_matching.types.with_standard_deviation_companions`;
        3. any reader-specific derived outputs declared in
           ``ADDITIONAL_PRODUCT_VARIABLES`` (e.g. IGBP ranked scenes).
        """
        return with_standard_deviation_companions(cls.VARIABLES) + cls.ADDITIONAL_PRODUCT_VARIABLES

    def __init__(self, file_paths: Sequence[Path]) -> None:
        if isinstance(file_paths, (str, Path)):
            raise TypeError(
                f"{type(self).__name__} takes a sequence of staged granule paths; got the single path "
                f"{str(file_paths)!r}. Wrap it in a list."
            )
        paths = tuple(Path(p) for p in file_paths)
        if not paths:
            raise ValueError(f"{type(self).__name__} needs at least one staged granule; got none.")
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{type(self).__name__} was given staged granule path(s) that are not existing files: {missing}. "
                "Stage each granule before constructing the reader, or correct the path."
            )
        self._file_paths = paths

    @property
    def file_paths(self) -> tuple[Path, ...]:
        """The staged granule paths passed at construction, in the order given."""
        return self._file_paths

    @property
    def single_file_path(self) -> Path:
        """The one granule of a reader that reads exactly one per period.

        Raises
        ------
        ValueError
            If more than one granule was staged for this reader.
        """
        if len(self._file_paths) != 1:
            raise ValueError(
                f"{type(self).__name__} ({self.READER_KEY!r}) reads one granule per period but "
                f"{len(self._file_paths)} were staged: {[p.name for p in self._file_paths]}. "
                "Stage exactly the granule that covers the processing period."
            )
        return self._file_paths[0]

    @property
    def n_variables(self) -> int:
        """Number of read variables (the first axis of every ``values`` array)."""
        return len(self.VARIABLES)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def samples_in(self, bbox: BoundingBox, time: np.datetime64) -> SampleSet:
        """Return the native samples inside ``bbox`` that match ``time``.

        Parameters
        ----------
        bbox : BoundingBox
            The footprint's geographic box, widened by half a native cell
            (``RESOLUTION_KM``) on every side so a sample whose cell overlaps the box is
            served. A box that wraps the antimeridian (given in the [0, 360) form) is
            split and both halves served.
        time : np.datetime64
            The footprint's observation time (radiometer or camera time). Ignored by a
            ``DAILY`` reader.

        Returns
        -------
        SampleSet
            The selected samples; zero samples when the box holds none, or when no
            sample lies within ``MAX_TIME_OFFSET`` of ``time``.

        Raises
        ------
        ValueError
            If ``time`` is NaT and this reader selects in time; if more than one granule
            is staged for a reader that reads one per period; if the loaded data carries
            a different number of variables than ``VARIABLES``; if a stepped reader's
            step times are empty, hold a NaT or are not strictly ascending; or if a
            ``NEAREST_PASS`` reader serves samples without per-sample times.
        TypeError
            If the reader's native data is none of ``GridSamples``, ``PointSamples`` or
            a box-queried native.
        """
        time = np.datetime64(time, "ns")
        if self.TIME_MATCH is not TimeMatch.DAILY:
            self._require_valid_time(time)
        natives = self._native_for(time)
        resolution_km = self._resolution_km(natives)
        if not natives:
            return SampleSet.empty(self.READER_KEY, bbox, self.n_variables, self.TIMESTAMP_SOURCE, resolution_km)

        boxes = split_dateline(self._padded(bbox, resolution_km))
        parts = [self._box_samples(native, part) for native in natives for part in boxes]
        lats = np.concatenate([p[0] for p in parts])
        lons = np.concatenate([p[1] for p in parts])
        values = np.concatenate([p[2] for p in parts], axis=1)
        times = None
        if parts[0][3] is not None:
            times = np.concatenate([p[3] for p in parts])

        if self.TIME_MATCH is TimeMatch.NEAREST_PASS:
            if times is None:
                raise ValueError(
                    f"Reader {self.READER_KEY!r} uses TIME_MATCH='nearest_pass' but its native data served no "
                    "per-sample times, so no pass can be selected. A nearest-pass reader must load PointSamples "
                    "with a times array, or a box-queried native whose query_box returns times."
                )
            if times.size:
                inside = np.zeros(times.shape, dtype=bool)
                for part in split_dateline(bbox):
                    inside |= in_box(lats, lons, part)
                keep = self._nearest_pass_mask(times, time, inside)
                lats, lons, values, times = lats[keep], lons[keep], values[:, keep], times[keep]

        return SampleSet(
            lats=lats,
            lons=lons,
            values=values,
            times=times,
            source=self.READER_KEY,
            bounds=bbox,
            timestamp_source=self.TIMESTAMP_SOURCE,
            resolution_km=resolution_km,
        )

    def _require_valid_time(self, time: np.datetime64) -> None:
        """Raise when a footprint time a time-selecting query needs is NaT.

        Raises
        ------
        ValueError
            If ``time`` is NaT.
        """
        if np.isnat(time):
            raise ValueError(
                f"Reader {self.READER_KEY!r} selects samples by time ({self.TIME_MATCH.value}) but the footprint "
                "time is NaT. Every footprint handed to this reader needs a valid observation time."
            )

    def _resolution_km(self, natives: tuple[_Native, ...]) -> float:
        """The native cell size the served samples stand for, km.

        The largest spacing a loaded point cloud measured on itself, else the reader's
        declared ``RESOLUTION_KM``. A measured spacing wins because a swath subsampled
        on delivery (e.g. every eighth pixel) must be credited
        with the cell it really stands for, not the instrument's nominal one.
        """
        measured = [n.spacing_km for n in natives if isinstance(n, _PointIndex) and n.spacing_km is not None]
        measured = [v for v in measured if np.isfinite(v) and v > 0.0]
        return float(max(measured)) if measured else float(self.RESOLUTION_KM)

    def _padded(self, bbox: BoundingBox, resolution_km: float) -> BoundingBox:
        """Widen ``bbox`` by half a native cell (``resolution_km``) on every side.

        A sample whose cell overlaps the footprint's box belongs to the footprint even when
        its center lies just outside: without the padding a coarse grid (ERA5 at 0.25
        degree) never serves the points whose cells cover the rim of the contour, and the
        weigher scores the rim as missing. The PSF weight of a sample outside the contour
        is zero, so padding changes no aggregated value, only the coverage.
        """
        half_deg = 0.5 * resolution_km / _KM_PER_DEG
        lat_min = max(-90.0, bbox.lat_min - half_deg)
        lat_max = min(90.0, bbox.lat_max + half_deg)
        lon_span = bbox.lon_max - bbox.lon_min
        if lon_span >= 360.0:
            return BoundingBox(lat_min, lat_max, bbox.lon_min, bbox.lon_max, bbox.wraps_dateline, True, bbox.truncated)
        # Half a cell spans more degrees of longitude toward the poles, so the pad is taken at
        # the box's most poleward edge. A padded box that reaches all the way around serves
        # every longitude.
        lon_pad = half_deg / np.cos(np.radians(max(abs(bbox.lat_min), abs(bbox.lat_max))))
        if lon_span + 2.0 * lon_pad >= 360.0:
            return BoundingBox(lat_min, lat_max, -180.0, 180.0, False, True, bbox.truncated)
        lon_min, lon_max = bbox.lon_min - lon_pad, bbox.lon_max + lon_pad
        wraps = bbox.wraps_dateline
        if not wraps and lon_max > 180.0:
            # The padding pushed a plain box across the antimeridian: use the [0, 360) form.
            lon_min, wraps = lon_min % 360.0, True
        elif not wraps and lon_min < -180.0:
            lon_min, lon_max, wraps = bbox.lon_min - lon_pad + 360.0, bbox.lon_max + lon_pad + 360.0, True
        return BoundingBox(lat_min, lat_max, lon_min, lon_max, wraps, bbox.is_polar, bbox.truncated)

    def _nearest_pass_mask(self, times: np.ndarray, time: np.datetime64, inside: np.ndarray) -> np.ndarray:
        """Boolean mask of the samples that belong to the pass nearest ``time``.

        The nearest pass is every sample, padding included, within ``PASS_WINDOW`` of
        the observation time of the sample nearest ``time`` among the samples ``inside``
        the footprint's own (unpadded) box, so a pass that only clips the padding, where
        the PSF weight is zero, cannot displace one that covers the footprint. The window
        is centred on that sample's own time rather than on its offset, so two passes on
        either side of the footprint time are never served together. When no sample lies
        inside the box (a footprint smaller than one native cell), the nearest sample is
        taken from every served sample; on a tie, the first in array order. When its
        offset exceeds ``MAX_TIME_OFFSET`` no sample is kept.
        """
        offsets = np.abs(times - time)
        candidates = inside if inside.any() else np.ones(times.shape, dtype=bool)
        nearest_time = times[candidates][np.argmin(offsets[candidates])]
        if np.abs(nearest_time - time) > self.MAX_TIME_OFFSET:
            return np.zeros(times.shape, dtype=bool)
        return np.abs(times - nearest_time) <= self.PASS_WINDOW

    def _box_samples(
        self, native: _Native, bbox: BoundingBox
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return ``(lats, lons, values, times)`` of the native samples inside one plain box.

        Raises
        ------
        TypeError
            If ``native`` is none of the three native forms.
        """
        if isinstance(native, GridSamples):
            lat0 = int(np.searchsorted(native.lat_axis, bbox.lat_min, side="left"))
            lat1 = int(np.searchsorted(native.lat_axis, bbox.lat_max, side="right"))
            lon0 = int(np.searchsorted(native.lon_axis, bbox.lon_min, side="left"))
            lon1 = int(np.searchsorted(native.lon_axis, bbox.lon_max, side="right"))
            lat_axis = native.lat_axis[lat0:lat1]
            lon_axis = native.lon_axis[lon0:lon1]
            block = native.data[:, lat0:lat1, lon0:lon1]
            lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
            values = np.ascontiguousarray(block.reshape(block.shape[0], -1), dtype=np.float32)
            return lat_grid.ravel().astype(np.float64), lon_grid.ravel().astype(np.float64), values, None
        if isinstance(native, _PointIndex):
            idx = native.query(bbox)
            times = native.times[idx] if native.times is not None else None
            return native.lats[idx], native.lons[idx], native.values[:, idx], times
        if isinstance(native, _BoxNative):
            return native.query_box(bbox)
        raise TypeError(self._unknown_native_message(native))

    def _unknown_native_message(self, native: object) -> str:
        return (
            f"Reader {self.READER_KEY!r} holds native data of type {type(native).__name__}; a reader's native data "
            "must be GridSamples, PointSamples, or an object with n_variables and query_box(bbox)."
        )

    @abc.abstractmethod
    def _native_for(self, time: np.datetime64) -> tuple[_Native, ...]:
        """Return the native data a query at ``time`` draws from; empty for no match.

        :class:`~libera_utils.footprint_matching.readers.base.StaticReader` returns its one indexed body of data regardless of
        ``time``; :class:`~libera_utils.footprint_matching.readers.base.SteppedGridReader` the step nearest ``time``, or nothing when
        that step is farther than ``MAX_TIME_OFFSET``.
        """

    def _index_points(self, native: PointSamples) -> _PointIndex:
        """Validate a loaded point cloud against the reader's contract and index it."""
        if native.values.shape[0] != self.n_variables:
            raise ValueError(
                f"Reader {self.READER_KEY!r} loaded {native.values.shape[0]} variable rows but declares "
                f"{self.n_variables} VARIABLES."
            )
        if self.TIME_MATCH is TimeMatch.NEAREST_PASS and native.times is None:
            raise ValueError(
                f"Reader {self.READER_KEY!r} uses TIME_MATCH='nearest_pass' but loaded no per-sample times; "
                "a nearest-pass reader must return PointSamples with a times array."
            )
        return _PointIndex(native)


class StaticReader(AncillaryReader):
    """A reader whose source is one body of data for the staged period.

    Covers the ``DAILY`` sources (one composite or map per period), and a
    ``NEAREST_PASS`` source small enough to hold whole. The subclass loads the whole
    source once in :meth:`_load_native`; this class indexes it (a point cloud into 1
    degree bins) and serves every query from memory.
    """

    @cached_property
    def _native(self) -> _Native:
        """The loaded, indexed native data (loaded on first access, then held).

        Raises
        ------
        ValueError
            If the loaded data carries a different number of variables than ``VARIABLES``.
        TypeError
            If :meth:`_load_native` returned none of the three native forms.
        """
        native = self._load_native()
        if isinstance(native, GridSamples):
            if native.data.shape[0] != self.n_variables:
                raise ValueError(
                    f"Reader {self.READER_KEY!r} loaded {native.data.shape[0]} variable planes but declares "
                    f"{self.n_variables} VARIABLES."
                )
            return native
        if isinstance(native, PointSamples):
            return self._index_points(native)
        if isinstance(native, _BoxNative):
            if native.n_variables != self.n_variables:
                raise ValueError(
                    f"Reader {self.READER_KEY!r} loaded a box-queried native with {native.n_variables} variables "
                    f"but declares {self.n_variables} VARIABLES."
                )
            return native
        raise TypeError(self._unknown_native_message(native))

    def _native_for(self, time: np.datetime64) -> tuple[_Native, ...]:
        return (self._native,)

    @abc.abstractmethod
    def _load_native(self) -> GridSamples | PointSamples | _BoxNative:
        """Open the staged granule(s) and return the whole source as native samples.

        Called once per reader instance. Implementers handle the source's own fill and
        valid-range conventions, returning NaN where the source has no value, and
        return longitudes in [-180, 180].

        Returns
        -------
        GridSamples or PointSamples or _BoxNative
            ``GridSamples`` for a regular latitude/longitude grid, a ``_BoxNative``
            for a source held in a projected form that geolocates the block a query
            selects, ``PointSamples`` for anything else (with ``times`` for a
            ``NEAREST_PASS`` reader).
        """


class SteppedGridReader(AncillaryReader):
    """A reader whose source is a time series of global grids, served one step at a time.

    The subclass lists every step's time across the staged granules
    (:meth:`_step_times`) and loads one step (:meth:`_load_step`). This class resolves a
    footprint time to the nearest step, loads it on change, and keeps only that step in
    memory, so a day of hourly 0.25 degree pressure-level fields never has to be held
    at once. Footprints arrive in time order, so a step changes about once an hour.
    """

    TIME_MATCH: TimeMatch = TimeMatch.NEAREST_STEP

    @cached_property
    def _steps(self) -> np.ndarray:
        """Every step's time, ``datetime64[ns]``, ascending, across all granules."""
        steps = np.asarray(self._step_times()).astype("datetime64[ns]")
        if steps.size == 0:
            raise ValueError(f"Reader {self.READER_KEY!r} found no time steps in {[p.name for p in self.file_paths]}.")
        if np.any(np.isnat(steps)):
            raise ValueError(
                f"Reader {self.READER_KEY!r} found a NaT time step in {[p.name for p in self.file_paths]}: "
                "the granule's time axis has a missing value; re-download that period."
            )
        if steps.size > 1 and np.any(np.diff(steps) <= np.timedelta64(0, "ns")):
            raise ValueError(
                f"Reader {self.READER_KEY!r}: step times must be strictly ascending across the staged granules; "
                "stage each period once."
            )
        return steps

    def __init__(self, file_paths: Sequence[Path]) -> None:
        super().__init__(file_paths)
        self._loaded_step: tuple[int, GridSamples] | None = None

    def nearest_step_index(self, time: np.datetime64) -> int | None:
        """Index of the step nearest ``time``, or ``None`` beyond ``MAX_TIME_OFFSET``.

        Parameters
        ----------
        time : np.datetime64
            A footprint time.

        Returns
        -------
        int or None
            The index into the step times of the step nearest ``time``, or ``None`` when
            that step is farther than ``MAX_TIME_OFFSET``.

        Raises
        ------
        ValueError
            If ``time`` is NaT, or the staged granules hold no time steps, a NaT step, or
            step times that are not strictly ascending.
        """
        time = np.datetime64(time, "ns")
        self._require_valid_time(time)
        steps = self._steps
        pos = int(np.searchsorted(steps, time))
        candidates = [i for i in (pos - 1, pos) if 0 <= i < steps.size]
        best = min(candidates, key=lambda i: abs(steps[i] - time))
        if abs(steps[best] - time) > self.MAX_TIME_OFFSET:
            return None
        return best

    def _native_for(self, time: np.datetime64) -> tuple[GridSamples, ...]:
        index = self.nearest_step_index(time)
        if index is None:
            return ()
        if self._loaded_step is None or self._loaded_step[0] != index:
            grid = self._load_step(index)
            if grid.data.shape[0] != self.n_variables:
                raise ValueError(
                    f"Reader {self.READER_KEY!r} loaded {grid.data.shape[0]} variable planes for step {index} "
                    f"but declares {self.n_variables} VARIABLES."
                )
            # Replace, never accumulate: the previous step is released here.
            self._loaded_step = (index, grid)
        return (self._loaded_step[1],)

    @abc.abstractmethod
    def _step_times(self) -> np.ndarray:
        """Return every step's time across the staged granules, ``datetime64``, ascending."""

    @abc.abstractmethod
    def _load_step(self, index: int) -> GridSamples:
        """Load the grid of step ``index`` (an index into :meth:`_step_times`)."""


class PassReader(AncillaryReader):
    """A reader whose source is swath data in time-spanning granules, served by time window.

    CERES SSF and CLDPIX come as hourly granules, and a CLDPIX hour alone is several
    million pixels, so a day cannot be held at once. The subclass reports each granule's
    time span (:meth:`_granule_span`) and loads one granule as time-stamped points
    (:meth:`_load_granule`); this class keeps indexed only the granules whose span lies
    within ``MAX_TIME_OFFSET`` of the current footprint time, loading a granule when the
    footprints reach its window and releasing it when they leave. Footprints arrive in
    time order, so the window slides forward and each granule is loaded once.

    The nearest-pass selection then runs over every sample the loaded granules hold in
    the box, so a pass is found whichever granule it falls in. The pass is chosen from
    the samples inside the footprint's own box and then served across the padded box.
    """

    TIME_MATCH: TimeMatch = TimeMatch.NEAREST_PASS
    # The window for a swath from a polar orbiter, such as the NOAA-20 CERES host. One pass
    # crosses a footprint box in seconds; 15 min holds one pass's samples together; passes
    # about 100 min apart are separated by the nearest-pass selection.
    # Libera and the swath's host are different spacecraft, so the nearest pass can be tens
    # of minutes away; 3 h is the largest offset accepted before the footprint is scored as
    # having no match. A reader for another orbit overrides both.
    PASS_WINDOW: np.timedelta64 | None = np.timedelta64(15, "m")
    MAX_TIME_OFFSET: np.timedelta64 | None = np.timedelta64(3, "h")

    def __init__(self, file_paths: Sequence[Path]) -> None:
        super().__init__(file_paths)
        # Granule path -> its indexed points, for the granules inside the current window.
        self._loaded: dict[Path, _PointIndex] = {}

    @cached_property
    def _spans(self) -> list[tuple[np.datetime64, np.datetime64, Path]]:
        """``(start, end, path)`` per staged granule, ascending by start."""
        spans = []
        for path in self.file_paths:
            start, end = self._granule_span(path)
            start, end = np.datetime64(start, "ns"), np.datetime64(end, "ns")
            if np.isnat(start) or np.isnat(end) or end < start:
                raise ValueError(
                    f"Reader {self.READER_KEY!r}: granule {path.name} has an invalid time span {start}..{end}."
                )
            spans.append((start, end, path))
        spans.sort(key=lambda item: item[0])
        return spans

    def granules_in_window(self, time: np.datetime64) -> list[Path]:
        """The staged granules whose span lies within ``MAX_TIME_OFFSET`` of ``time``.

        Parameters
        ----------
        time : np.datetime64
            A footprint time.

        Returns
        -------
        list[Path]
            The granules whose ``(start, end)`` span overlaps ``time`` +/- ``MAX_TIME_OFFSET``,
            ascending by start; empty when none does.

        Raises
        ------
        ValueError
            If a staged granule's span is invalid: a start or end that is NaT, or an end
            before its start.
        """
        low, high = time - self.MAX_TIME_OFFSET, time + self.MAX_TIME_OFFSET
        return [path for start, end, path in self._spans if end >= low and start <= high]

    def _native_for(self, time: np.datetime64) -> tuple[_PointIndex, ...]:
        wanted = self.granules_in_window(time)
        for path in list(self._loaded):
            if path not in wanted:
                # Release a granule the footprints have moved past.
                del self._loaded[path]
        for path in wanted:
            if path not in self._loaded:
                self._loaded[path] = self._index_points(self._load_granule(path))
        return tuple(self._loaded[path] for path in wanted)

    @abc.abstractmethod
    def _granule_span(self, path: Path) -> tuple[np.datetime64, np.datetime64]:
        """Return the ``(start, end)`` observation times a granule covers, from its metadata."""

    @abc.abstractmethod
    def _load_granule(self, path: Path) -> PointSamples:
        """Open one granule and return its samples as time-stamped points.

        Implementers handle the granule's own fill and valid-range conventions, returning
        NaN where it has no value, longitudes in [-180, 180], and a time per sample.
        """
