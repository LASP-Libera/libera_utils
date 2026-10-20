"""Shared data classes and enumerations for the footprint matching subsystem.

All classes in this module are intentionally dependency-free so they can be
imported by any layer without creating circular dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class OperationalMode(enum.Enum):
    """Five operational modes for the FMATCH algorithm.

    Each mode produces a distinct data product with its own identifier.
    Members are declared in order of increasing data latency.

    Attributes
    ----------
    CAM : str
        Radiometer timescale, camera/NRT latency. Runs continuously from mission start.
    CAM_CAMTIME : str
        Camera timescale, camera/NRT latency. Runs continuously from mission start.
    IMAGER_FLASH : str
        Radiometer timescale, RBSP Flash latency. Available post-Year 1.
    IMAGER : str
        Radiometer timescale, RBSP Climate Quality latency. Available post-Year 1.
    IMAGER_CAMTIME : str
        Camera timescale, RBSP Climate Quality latency. Available post-Year 1.

    Notes
    -----
    Mode strings match the product identifiers used in the pipeline and in output
    NetCDF global attributes (e.g., ``FMATCH-CAM``).
    """

    CAM = "FMATCH-CAM"
    CAM_CAMTIME = "FMATCH-CAM-CAMTIME"
    IMAGER_FLASH = "FMATCH-IMAGER-FLASH"
    IMAGER = "FMATCH-IMAGER"
    IMAGER_CAMTIME = "FMATCH-IMAGER-CAMTIME"


class FmatchCoverageFlag(enum.IntFlag):
    """Bitwise coverage/QA flags folded into a footprint's ``q_flags`` variable.

    Set by the product orchestrator from the CERES 75%/95% PSF-energy coverage rule once
    the external-variable aggregation has run. The values are OR-combined and tested
    bitwise, and an empty (zero) value means "no coverage issue". The
    ``PARTIAL_COVERAGE`` bit deliberately shares bit 0 with
    :class:`CameraFootprintQualityFlag.PARTIAL_COVERAGE`
    so the camera segmentation's own partial-coverage flag and the PSF-energy
    partial-coverage flag OR together into one consistent "partial" bit; the other
    bits use positions the camera flag does not (it uses bit 1 for
    ``CENTER_PIXEL_SUBSTITUTED``).

    Attributes
    ----------
    PARTIAL_COVERAGE
        75% <= coverage < 95% of the PSF's 95%-energy weight was backed by usable
        ancillary data (accepted, but flagged partial). Bit 0.
    INSUFFICIENT_COVERAGE
        coverage < 75% -- below the CERES acceptance threshold. The footprint is
        retained but flagged (a flag-not-discard policy; see the product module). Bit 2.
    LIMB_TRUNCATED
        The footprint's bounding box was clipped at the Earth's limb
        (``BoundingBox.truncated``), so it only ever had partial ground coverage. Bit 3.
    OFF_LIMB
        The view's centroid is off the Earth's limb (a space / calibration look), so
        :func:`~libera_utils.footprint_matching.geometry.compute_footprint_bounding_box` reports *no* Earth footprint at all. The
        record is retained for index alignment but its external variables are left at
        their fill sentinel and its coverage is forced to zero -- it must never be
        scored as a real observation. Bit 4.
    """

    PARTIAL_COVERAGE = 0b0001
    INSUFFICIENT_COVERAGE = 0b0100
    LIMB_TRUNCATED = 0b1000
    OFF_LIMB = 0b10000


class BoundingBox(NamedTuple):
    """Geographic bounding box for a footprint's PSF contour.

    An immutable, hashable seven-element tuple of (lat_min, lat_max, lon_min,
    lon_max, wraps_dateline, is_polar, truncated).

    Attributes
    ----------
    lat_min : float
        Lower latitude bound in degrees. ``lat_min < lat_max``.
    lat_max : float
        Upper latitude bound in degrees.
    lon_min : float
        Lower longitude bound in degrees, always with ``lon_min <= lon_max``. A box that
        does not cross the antimeridian is in [-180, 180]. A box that crosses it is
        given in [0, 360), so ``lon_max > 180`` (e.g. ``179.9`` to ``180.2``); subtract
        360 from the part above 180 to get signed longitudes. A box enclosing a pole
        spans the full ``-180`` to ``180``.
    lon_max : float
        Upper longitude bound in degrees.
    wraps_dateline : bool
        True when the box crosses the antimeridian, i.e. it is given in the [0, 360)
        form with ``lon_max > 180``. The reader base class queries both halves.
    is_polar : bool
        True when the box reaches 85° latitude or beyond, or encloses a pole (then
        latitude is pinned to +/-90 and longitude spans -180 to 180). Advisory: a
        lat/lon rectangle is only a coarse fit at such latitudes.
    truncated : bool
        True when the box was clipped at the Earth's limb because the footprint
        ran partly off the edge of the Earth at a severe viewing angle (see
        ``geometry.compute_footprint_bounding_box``). The footprint therefore has
        only **partial coverage**, and the orchestrator should set the
        corresponding QA flag.
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    wraps_dateline: bool = False
    is_polar: bool = False
    truncated: bool = False

    def __repr__(self) -> str:
        return (
            f"BoundingBox(lat=[{self.lat_min}, {self.lat_max}], "
            f"lon=[{self.lon_min}, {self.lon_max}], "
            f"wraps_dateline={self.wraps_dateline}, is_polar={self.is_polar}, "
            f"truncated={self.truncated})"
        )


class TimeMatch(enum.Enum):
    """How a reader selects, in time, the samples it serves for one footprint.

    A staged 24 h period can hold several observations of the same ground location: a
    swath product sees it on two or more passes, a reanalysis holds 24 hourly steps.
    Each reader declares one policy, applied after the geographic (bounding box) query.
    Time selection is a *selection*, never an average or an interpolation, so it does
    not change any value a sample carries.

    Attributes
    ----------
    DAILY
        The staged granule is a daily composite, climatology or static map (IGBP land
        cover, NISE snow/ice, VIIRS BRDF, VIIRS Deep Blue AOD, VIIRS L3 cloud). No
        per-sample selection; every sample in the box is served regardless of time.
    NEAREST_STEP
        The source is a time series of global grids (ERA5 hourly steps). The single
        step whose time is nearest the footprint time is served, with no interpolation
        between steps. A footprint farther than ``MAX_TIME_OFFSET`` from every step gets
        no samples.
    NEAREST_PASS
        The source is swath or point data with a time per sample (CERES FLASHFlux, CLDPIX).
        Of the samples in the box, those within ``PASS_WINDOW`` of the observation time
        of the sample nearest the footprint time are served: the nearest pass, whole,
        and no other pass. A footprint whose nearest sample is farther than
        ``MAX_TIME_OFFSET`` gets no samples.
    """

    DAILY = "daily"
    NEAREST_STEP = "nearest_step"
    NEAREST_PASS = "nearest_pass"  # noqa: S105 - an enum label (a satellite pass), not a credential


@dataclass(frozen=True)
class GridSamples:
    """A reader's native data on a regular latitude/longitude grid.

    The native form of a gridded source (ERA5, VIIRS L3 products). The reader base class
    answers a bounding-box query by slicing the axes, so the array is never flattened
    to per-sample coordinates until a query selects a (small) region.

    Attributes
    ----------
    lat_axis : np.ndarray
        1-D ascending latitudes of the rows, degrees.
    lon_axis : np.ndarray
        1-D ascending longitudes of the columns, degrees, in [-180, 180].
    data : np.ndarray
        ``(n_var, n_lat, n_lon)`` float32 values, NaN where the source has no value.
        Axis 0 follows the reader's ``VARIABLES`` order.
    """

    lat_axis: np.ndarray
    lon_axis: np.ndarray
    data: np.ndarray

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"GridSamples.data must be (n_var, n_lat, n_lon); got shape {self.data.shape}.")
        if self.data.shape[1:] != (self.lat_axis.size, self.lon_axis.size):
            raise ValueError(
                f"GridSamples.data shape {self.data.shape} does not match the axes "
                f"({self.lat_axis.size} latitudes, {self.lon_axis.size} longitudes)."
            )
        for name, axis in (("lat_axis", self.lat_axis), ("lon_axis", self.lon_axis)):
            if axis.size > 1 and np.any(np.diff(axis) <= 0):
                raise ValueError(f"GridSamples.{name} must be strictly ascending.")


@dataclass(frozen=True)
class PointSamples:
    """A reader's native data as geolocated points.

    The native form of swath, footprint and projected-grid sources (CERES FLASHFlux and
    CLDPIX, IGBP tiles on the MODIS sinusoidal grid, NISE EASE-Grid pixels): each sample carries its
    own latitude and longitude, and for a ``NEAREST_PASS`` source its own time.

    Attributes
    ----------
    lats : np.ndarray
        1-D ``(n,)`` float64 sample latitudes, degrees.
    lons : np.ndarray
        1-D ``(n,)`` float64 sample longitudes, degrees, in [-180, 180].
        Samples with a non-finite coordinate are dropped by the reader base class.
    values : np.ndarray
        ``(n_var, n)`` float32 values, NaN where the source has no value. Axis 0 follows
        the reader's ``VARIABLES`` order.
    times : np.ndarray or None
        1-D ``(n,)`` ``datetime64[ns]`` observation time per sample. Required for a
        ``NEAREST_PASS`` reader; ``None`` otherwise.
    spacing_km : float or None
        The samples' native spacing, km, measured on this body of data.
        ``None`` means the reader's declared ``RESOLUTION_KM`` applies. A swath that is
        subsampled on delivery is served with its true spacing, so the coverage measure
        credits each sample with the cell it really stands for.
    """

    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray
    times: np.ndarray | None = None
    spacing_km: float | None = None

    def __post_init__(self) -> None:
        n = self.lats.size
        if self.lons.shape != (n,) or self.lats.ndim != 1:
            raise ValueError(f"PointSamples.lats {self.lats.shape} and lons {self.lons.shape} must both be (n,).")
        if self.values.ndim != 2 or self.values.shape[1] != n:
            raise ValueError(f"PointSamples.values must be (n_var, {n}); got shape {self.values.shape}.")
        if self.times is not None and self.times.shape != (n,):
            raise ValueError(f"PointSamples.times must be ({n},); got shape {self.times.shape}.")


@dataclass(frozen=True)
class SampleSet:
    """The native samples of one source that fall in one footprint's query.

    What a reader hands the PSF weighting: every sample inside the footprint's bounding
    box that also passes the reader's :class:`TimeMatch` selection, at its own native
    location, with no averaging, binning or interpolation applied. The weighting assigns
    each sample a PSF weight and the aggregation engine reduces the weighted samples to
    one value per variable.

    Attributes
    ----------
    lats : np.ndarray
        1-D ``(n,)`` float64 sample latitudes, degrees.
    lons : np.ndarray
        1-D ``(n,)`` float64 sample longitudes, degrees.
    values : np.ndarray
        ``(n_var, n)`` float32 values in the reader's ``VARIABLES`` order, NaN where the
        source has no value at that sample.
    times : np.ndarray or None
        1-D ``(n,)`` ``datetime64[ns]`` per-sample observation times for a
        ``NEAREST_PASS`` source; ``None`` for the others.
    source : str
        Reader registry key that produced the samples.
    bounds : BoundingBox
        The query box the samples were selected for.
    timestamp_source : str or None
        For cloud-property products: ``'radiometer'`` or ``'camera'``, the instrument
        whose observation time governs the data. ``None`` for surface and static
        ancillary products.
    resolution_km : float
        The native cell size the samples stand for, km: the reader's declared
        ``RESOLUTION_KM``, or the spacing measured on the granule when the reader
        measures it. The weighting sizes its coverage measure from it.
    """

    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray
    times: np.ndarray | None
    source: str
    bounds: BoundingBox
    timestamp_source: str | None = None
    resolution_km: float = float("nan")

    @property
    def n_samples(self) -> int:
        """Number of samples in the set (zero for an empty query)."""
        return int(self.lats.size)

    @classmethod
    def empty(
        cls,
        source: str,
        bounds: BoundingBox,
        n_var: int,
        timestamp_source: str | None = None,
        resolution_km: float = float("nan"),
    ) -> SampleSet:
        """Return a zero-sample set, the result of a query that selected nothing.

        Parameters
        ----------
        source : str
            Reader registry key.
        bounds : BoundingBox
            The query box.
        n_var : int
            Number of variables the reader serves, so ``values`` keeps its first axis.
        timestamp_source : str, optional
            See the class attributes.
        resolution_km : float, optional
            See the class attributes.
        """
        return cls(
            lats=np.empty(0, dtype=np.float64),
            lons=np.empty(0, dtype=np.float64),
            values=np.empty((n_var, 0), dtype=np.float32),
            times=None,
            source=source,
            bounds=bounds,
            timestamp_source=timestamp_source,
            resolution_km=resolution_km,
        )


class CameraFootprintQualityFlag(enum.IntFlag):
    """Bitwise quality flags for a camera pseudo-footprint.

    Set per footprint by the Camera Cloud Fraction algorithm, which segments the L1B camera
    image, in the CF-CAM-CAMTIME ``q_flags`` variable; FMATCH-CAM-CAMTIME and
    FMATCH-IMAGER-CAMTIME carry them forward and OR their own :class:`FmatchCoverageFlag`
    bits on top. ``IntFlag`` lets the flags be OR-combined and tested bitwise, and an empty
    (zero) value means "no issues".

    Attributes
    ----------
    PARTIAL_COVERAGE
        At least one -- but not all -- corner pixels of the block were off-Earth
        (fill), so the bounding box was shrunk to the valid corners and covers only
        part of the nominal block.
    CENTER_PIXEL_SUBSTITUTED
        The geometric center pixel was off-Earth (fill), so the nearest valid pixel
        in the block was substituted as the footprint boresight.
    """

    PARTIAL_COVERAGE = 0b0001
    CENTER_PIXEL_SUBSTITUTED = 0b0010


@dataclass(frozen=True)
class PseudoFootprint:
    """One camera pseudo-footprint: a block of L1B camera pixels reduced to a footprint record.

    The camera-timescale FMATCH modes are built from these. The pixel blocks are segmented by
    the Camera Cloud Fraction algorithm and read from its CF-CAM-CAMTIME product.

    Attributes
    ----------
    time : np.datetime64
        The ``CAMERA_TIME`` of the image this footprint came from. All footprints
        segmented from the same image share this timestamp.
    slice_x, slice_y : slice
        The block's half-open extent in the ``CAMERA_PIXEL_COUNT_X`` /
        ``CAMERA_PIXEL_COUNT_Y`` pixel grid.
    center_ix, center_iy : int
        Pixel indices of the footprint's center (the boresight stand-in), after any
        nearest-valid-pixel substitution.
    latitude, longitude : float
        Center-pixel geodetic latitude/longitude, degrees.
    altitude : float
        Center-pixel altitude, metres (as stored in L1B).
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : float
        Center-pixel viewing-geometry angles, degrees.
    bbox : BoundingBox
        Geographic box enclosing the block's valid corner pixels.
    q_flags : CameraFootprintQualityFlag
        Bitwise quality flags for this footprint (0 == clean).
    """

    time: np.datetime64
    slice_x: slice
    slice_y: slice
    center_ix: int
    center_iy: int
    latitude: float
    longitude: float
    altitude: float
    solar_zenith_angle: float
    viewing_zenith_angle: float
    relative_azimuth_angle: float
    bbox: BoundingBox
    q_flags: CameraFootprintQualityFlag


@dataclass(frozen=True)
class RadiometerFootprint:
    """One radiometer-timescale footprint, ready for external-variable aggregation.

    The radiometer-timescale FMATCH modes take their footprints straight from the L1B
    Daily radiometer product. This is the minimal object the aggregation path needs:
    a geographic bounding box (the region the readers serve ancillary samples from)
    plus the boresight geolocation and viewing zenith the PSF weighting reads. It is the
    radiometer-path analog of the camera path's
    :class:`PseudoFootprint`, kept
    deliberately small so it stays dependency-free here.

    The subsatellite point and satellite radius are required: L1B RAD populates them
    from the ephemeris for every record, and together with the boresight they fix the
    satellite viewing frame the ray-traced bounding box
    (:func:`~libera_utils.footprint_matching.geometry.compute_footprint_bounding_box`) and
    the PSF weighting are built on.

    Attributes
    ----------
    bbox : BoundingBox or None
        Geographic box enclosing the footprint's PSF ground contour, or ``None`` exactly
        when ``off_limb`` (an off-Earth view has no ground contour).
    time : np.datetime64
        Radiometer observation time (L1B ``RADIOMETER_TIME``). The readers' time
        selection (:class:`TimeMatch`) matches ancillary samples to it.
    latitude : float
        Boresight centroid latitude (L1B ``Latitude``), degrees.
    longitude : float
        Boresight centroid longitude (L1B ``Longitude``), degrees.
    viewing_zenith_angle : float
        Viewing zenith angle (L1B ``Viewing_Zenith_Surface``), degrees.
    subsatellite_latitude : float
        Geodetic sub-point latitude of the spacecraft (L1B ``Subsatellite_Latitude``), degrees.
    subsatellite_longitude : float
        Geodetic sub-point longitude of the spacecraft (L1B ``Subsatellite_Longitude``), degrees.
    satellite_radius_km : float
        Spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km. Not L1B ``Altitude``, which is
        the height of the boresight's ellipsoid intersection.
    cone_angle_rate : float or None
        Instrument cone-angle rate (L1B ``Cone_Angle_Rate``), degrees per second:
        negative toward nadir, positive away, 0 when the scanner is stationary. ``None``
        when unavailable.
        TODO[LIBSDC-884]: adopt libera_rad PR #34's (LIBSDC-739) definition and fill
        convention once it merges, and handle a footprint with no usable rate in one
        stated way: kept, weighted direction-independently, and QA-flagged.
    off_limb : bool
        ``True`` for a space / calibration view whose centroid is off the Earth's limb,
        i.e. one with no Earth footprint (:class:`~libera_utils.footprint_matching.geometry.OffLimbError`). ``bbox`` is then
        ``None``; the aggregation path skips these records (fill values, zero coverage)
        and flags them ``OFF_LIMB``.

    Raises
    ------
    ValueError
        If ``bbox`` is ``None`` while ``off_limb`` is ``False``, or a ``bbox`` is given
        while ``off_limb`` is ``True``.
    """

    bbox: BoundingBox | None
    time: np.datetime64
    latitude: float
    longitude: float
    viewing_zenith_angle: float
    subsatellite_latitude: float
    subsatellite_longitude: float
    satellite_radius_km: float
    cone_angle_rate: float | None = None
    off_limb: bool = False

    def __post_init__(self) -> None:
        if self.bbox is None and not self.off_limb:
            raise ValueError(
                "RadiometerFootprint has no bbox but off_limb is False: pass the BoundingBox from "
                "compute_footprint_bounding_boxes, or set off_limb=True for a space/calibration view "
                "(whose entry there is None)."
            )
        if self.bbox is not None and self.off_limb:
            raise ValueError(
                "RadiometerFootprint is off_limb but was given a bbox: an off-limb view has no Earth "
                "footprint, so pass bbox=None."
            )


@dataclass(frozen=True)
class VariableSpec:
    """Metadata describing a single output variable provided by a reader.

    Used by the reader base class and the aggregation engine to know what data a
    reader supplies, how to aggregate it, and which products carry it.

    Attributes
    ----------
    name : str
        Variable name used throughout the footprint matching pipeline
        (e.g., ``"surface_type"``, ``"cloud_optical_thickness"``).
    dtype : str
        NumPy dtype string for the variable's data array (e.g., ``"int16"``).
    aggregation : str
        PSF aggregation strategy name (e.g., ``"weighted_mean"``,
        ``"weighted_mode"``, ``"weighted_log_mean"``). The strategy is resolved
        by the aggregation engine from the variables.yaml configuration.
    modes : frozenset[OperationalMode]
        Every operational mode (product) that carries this variable. A variable
        belongs to a product exactly when ``mode in spec.modes``, and a reader
        feeds a product when any of its variables names that product
        (``readers/registry.py`` derives ``FMATCH_MODE_READERS`` from these sets).
    n_categories : int or None
        For categorical variables only: number of distinct category values
        (e.g., 20 for IGBP surface type). ``None`` for continuous variables, and
        for a mode-aggregated code with no fixed count (e.g., the CERES FLASHFlux scene
        and ADM type codes), which the aggregation bins by the codes it sees.

    Raises
    ------
    TypeError
        If ``modes`` is not a ``frozenset`` of :class:`OperationalMode` members.
    ValueError
        If ``modes`` is empty.
    """

    name: str
    dtype: str
    aggregation: str
    modes: frozenset[OperationalMode]
    n_categories: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.modes, frozenset) or not all(isinstance(m, OperationalMode) for m in self.modes):
            raise TypeError(
                f"VariableSpec {self.name!r} has modes={self.modes!r}: pass a frozenset of OperationalMode "
                "members, e.g. frozenset({OperationalMode.IMAGER})."
            )
        if not self.modes:
            raise ValueError(
                f"VariableSpec {self.name!r} has empty modes: name at least one product that carries it, "
                "e.g. frozenset({OperationalMode.IMAGER})."
            )


# The aggregation of the standard-deviation companion for each mean-aggregated parent:
# a geometric mean is defined only on positive values, so its spread is taken over the
# same positive samples.
_STANDARD_DEVIATION_AGGREGATION: dict[str, str] = {
    "weighted_mean": "weighted_std",
    "weighted_log_mean": "weighted_positive_std",
}

# Suffix appended to a continuous variable's name to form its std-dev companion.
# Kept as a module constant so the readers, product definitions, and tests all
# agree on the exact spelling (e.g. ``era5_surface_wind_u10_standard_deviation``).
STANDARD_DEVIATION_SUFFIX: str = "_standard_deviation"


def with_standard_deviation_companions(specs: tuple[VariableSpec, ...]) -> tuple[VariableSpec, ...]:
    """Return ``specs`` plus a standard-deviation companion for each continuous spec.

    For every mean-aggregated (continuous) variable in ``specs`` this appends a
    ``<name>_standard_deviation`` companion describing the spread of that
    variable's values within the footprint. Categorical / mode-aggregated
    variables are passed through unchanged (no companion), because a standard
    deviation of a most-common category is not physically meaningful.

    The companion is declared here so the reader ``VariableSpec`` tuple stays the
    single source of truth for the FMATCH product variables (the product
    definition YAMLs and the cross-check test both derive variable names from
    these specs). The companion's ``aggregation`` is ``"weighted_std"`` for a
    ``weighted_mean`` parent and ``"weighted_positive_std"`` for a
    ``weighted_log_mean`` parent, so the spread covers the same samples as the mean.

    Parameters
    ----------
    specs : tuple[VariableSpec, ...]
        The reader's base variable specifications, in output order.

    Returns
    -------
    tuple[VariableSpec, ...]
        The original specs, each immediately followed by its standard-deviation
        companion when the spec is mean-aggregated. Ordering is preserved so the
        companion sits next to its parent in the product definition.
    """
    expanded: list[VariableSpec] = []
    for spec in specs:
        expanded.append(spec)
        if spec.aggregation in _STANDARD_DEVIATION_AGGREGATION:
            # A standard deviation is always a non-negative real number, so it is
            # stored as float32 regardless of the parent's dtype and carries no
            # category count. The companion inherits the parent's ``modes``, so it
            # appears in exactly the same product definitions as its parent.
            expanded.append(
                VariableSpec(
                    name=f"{spec.name}{STANDARD_DEVIATION_SUFFIX}",
                    dtype="float32",
                    aggregation=_STANDARD_DEVIATION_AGGREGATION[spec.aggregation],
                    modes=spec.modes,
                    n_categories=None,
                )
            )
    return tuple(expanded)


@dataclass(frozen=True)
class ProductStack:
    """A product variable that stacks several read planes along a second product dimension.

    A reader serves each plane of a multi-level field as its own :class:`VariableSpec`,
    and the aggregation engine reduces each plane to its own scalar. A stack declares that
    the product stores those scalars as one variable on ``(time, dimension)``: element
    ``k`` of the product variable is the scalar of ``members[k]``, at coordinate value
    ``coordinate_values[k]``.

    Attributes
    ----------
    name : str
        Bare product variable name (the caller prefixes it with ``<source>_``, as for any
        product variable), e.g. ``"temperature"``.
    dimension : str
        The product dimension the stack runs along, as named in the Libera standard
        dimensions file, e.g. ``"PRESSURE_LEVEL"``.
    coordinate_values : tuple[float, ...]
        The coordinate value of each element along ``dimension``, in the units the
        product definition's coordinate of that name declares.
    members : tuple[str, ...]
        The product spec names stacked, one per coordinate value, in coordinate order.

    Raises
    ------
    ValueError
        If ``members`` is empty or names a spec twice, or if it has a different length
        than ``coordinate_values``.
    """

    name: str
    dimension: str
    coordinate_values: tuple[float, ...]
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError(f"ProductStack {self.name!r} has no members; list the spec names it stacks.")
        if len(set(self.members)) != len(self.members):
            raise ValueError(
                f"ProductStack {self.name!r} lists a member more than once: {list(self.members)}. "
                "Each element along the stack's dimension must come from its own spec."
            )
        if len(self.members) != len(self.coordinate_values):
            raise ValueError(
                f"ProductStack {self.name!r} has {len(self.members)} members but {len(self.coordinate_values)} "
                f"coordinate values along {self.dimension!r}; give one coordinate value per member."
            )
