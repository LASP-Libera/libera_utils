"""Shared data classes and enumerations for the footprint matching subsystem.


All classes in this module are intentionally dependency-free so they can be
imported by any layer without creating circular dependencies.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class OperationalMode(enum.Enum):
    """Five operational modes for the FMATCH algorithm.

    Each mode produces a distinct data product with its own identifier.
    Modes are ordered by increasing data latency; the ``rank`` property
    exposes this ordering for reader filtering.

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

    @property
    def rank(self) -> int:
        """Integer ordering of modes by data latency (0 = lowest latency).

        Used to compare a :class:`VariableSpec`'s ``required_mode`` against the active
        mode (a spec is carried when ``spec.required_mode.rank <= mode.rank``), which
        gates variables *within* a reader -- e.g. the ERA5 single-level winds (CAM) vs
        its IMAGER-only substitute fields. Reader-to-product membership is declared
        separately by the per-product reader sets in ``readers/registry.py``.
        """
        return list(OperationalMode).index(self)


class FmatchCoverageFlag(enum.IntFlag):
    """Bitwise coverage/QA flags folded into a footprint's ``q_flags`` variable.

    Set by the product orchestrator from the CERES 75%/95% PSF-energy coverage rule
    (design doc section 2.4.2.7) once the external-variable aggregation has run. The
    values are OR-combined and tested bitwise, and an empty (zero) value means "no
    coverage issue". The ``PARTIAL_COVERAGE`` bit deliberately shares bit 0 with
    :class:`~libera_utils.footprint_matching.camera_segmentation.CameraFootprintQualityFlag.PARTIAL_COVERAGE`
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
        :func:`compute_footprint_bounding_box` reports *no* Earth footprint at all. The
        record is retained for index alignment but its external variables are left at
        their fill sentinel and its coverage is forced to zero -- it must never be
        scored as a real observation. Bit 4.
    """

    PARTIAL_COVERAGE = 0b0001
    INSUFFICIENT_COVERAGE = 0b0100
    LIMB_TRUNCATED = 0b1000
    OFF_LIMB = 0b10000


class BoundingBox(tuple):
    """Geographic bounding box for a footprint's PSF contour.

    An immutable, hashable seven-element tuple of (lat_min, lat_max, lon_min,
    lon_max, wraps_dateline, is_polar, truncated), constructed directly.

    Attributes
    ----------
    lat_min, lat_max : float
        Latitude extent in degrees. ``lat_min < lat_max``.
    lon_min, lon_max : float
        Longitude extent in degrees. For standard (non-dateline-crossing) boxes,
        ``lon_min < lon_max``.
    wraps_dateline : bool
        True when the box straddles the antimeridian (detected when
        ``lon_max - lon_min > 180°``). The TileManager issues two sub-requests.
    is_polar : bool
        True when the boresight latitude exceeds 85°, triggering a great-circle
        distance test instead of rectangular bounding box logic.
    truncated : bool
        True when the box was clipped at the Earth's limb because the footprint
        ran partly off the edge of the Earth at a severe viewing angle (see
        ``geometry.compute_footprint_bounding_box``). The footprint therefore has
        only **partial coverage**, and the orchestrator should set the
        corresponding QA flag. Always ``False`` for tile bounding boxes.
    """

    __slots__ = ()

    def __new__(
        cls,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        wraps_dateline: bool = False,
        is_polar: bool = False,
        truncated: bool = False,
    ) -> BoundingBox:
        return super().__new__(cls, (lat_min, lat_max, lon_min, lon_max, wraps_dateline, is_polar, truncated))

    @property
    def lat_min(self) -> float:
        """Minimum latitude in degrees."""
        return self[0]

    @property
    def lat_max(self) -> float:
        """Maximum latitude in degrees."""
        return self[1]

    @property
    def lon_min(self) -> float:
        """Minimum longitude in degrees."""
        return self[2]

    @property
    def lon_max(self) -> float:
        """Maximum longitude in degrees."""
        return self[3]

    @property
    def wraps_dateline(self) -> bool:
        """True when the box straddles the antimeridian."""
        return bool(self[4])

    @property
    def is_polar(self) -> bool:
        """True when the boresight is poleward of 85°."""
        return bool(self[5])

    @property
    def truncated(self) -> bool:
        """True when the box was clipped at the Earth's limb (partial coverage)."""
        return bool(self[6])

    def __repr__(self) -> str:
        return (
            f"BoundingBox(lat=[{self.lat_min}, {self.lat_max}], "
            f"lon=[{self.lon_min}, {self.lon_max}], "
            f"wraps_dateline={self.wraps_dateline}, is_polar={self.is_polar}, "
            f"truncated={self.truncated})"
        )


class TileKey(tuple):
    """Immutable, hashable cache key for a spatial tile.

    A three-element tuple of (source, lat_idx, lon_idx) used by the TileManager
    as a dict key in the LRU cache and by readers to determine which geographic
    region to load.

    Attributes
    ----------
    source : str
        Registry key of the reader that owns this tile (e.g., ``"igbp"``).
    lat_idx : int
        Tile row index in the global tile grid (0-based).
    lon_idx : int
        Tile column index in the global tile grid (0-based).
    """

    __slots__ = ()

    def __new__(cls, source: str, lat_idx: int, lon_idx: int) -> TileKey:
        return super().__new__(cls, (source, lat_idx, lon_idx))

    @property
    def source(self) -> str:
        """Reader registry key for this tile."""
        return self[0]

    @property
    def lat_idx(self) -> int:
        """Tile row index in the global tile grid."""
        return self[1]

    @property
    def lon_idx(self) -> int:
        """Tile column index in the global tile grid."""
        return self[2]

    def __repr__(self) -> str:
        return f"TileKey(source={self.source!r}, lat_idx={self.lat_idx}, lon_idx={self.lon_idx})"


@dataclass
class GridTile:
    """Rectangular region of gridded data with coordinate arrays.

    The data array is 2-D (n_lats, n_lons) for single-variable readers and
    3-D (n_variables, n_lats, n_lons) for multi-variable readers (ERA5, VIIRS).
    The variable ordering in the first axis of a 3-D tile matches the order
    of ``VariableSpec`` entries in the reader's ``VARIABLES`` class attribute.

    Attributes
    ----------
    data : np.ndarray
        Gridded values. Shape (n_lat, n_lon) or (n_var, n_lat, n_lon).
    lats : np.ndarray
        1-D latitude coordinate array in degrees for the row axis of ``data``.
    lons : np.ndarray
        1-D longitude coordinate array in degrees for the column axis of ``data``.
    bounds : BoundingBox
        Geographic extent of this tile.
    source : str
        Reader registry key that produced this tile.
    timestamp_source : str or None
        For cloud-property products: ``'radiometer'`` or ``'camera'``, indicating
        which instrument's observation time governs the data. ``None`` for surface
        and static ancillary products.
    """

    data: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    bounds: BoundingBox
    source: str
    timestamp_source: str | None = None

    @property
    def nbytes(self) -> int:
        """Estimated memory footprint in bytes, used for LRU cache eviction decisions."""
        return int(self.data.nbytes + self.lats.nbytes + self.lons.nbytes)


@dataclass(frozen=True)
class RadiometerFootprint:
    """One radiometer-timescale footprint, ready for external-variable aggregation.

    The radiometer-timescale FMATCH modes take their footprints straight from the L1B
    Daily radiometer product. This is the minimal object the aggregation path needs:
    a geographic bounding box (which ancillary tiles to load) plus the boresight
    geolocation and viewing zenith the PSF weigher reads. It is the radiometer-path
    analogue of the camera path's
    :class:`~libera_utils.footprint_matching.camera_segmentation.PseudoFootprint`,
    kept deliberately small so it stays dependency-free here.

    The scan-frame fields the CERES-faithful ``AngularPSFWeigher`` needs
    (subsatellite point and cone-angle rate) are optional. When they are populated -
    which the production reader now does from the L1B ``Subsatellite_Latitude`` /
    ``Subsatellite_Longitude`` / ``Cone_Angle_Rate`` fields - the footprint carries a
    true ray-traced bounding box (:func:`compute_footprint_bounding_box`) and the
    angular weigher orients the PSF along the real scan plane. When they are ``None``
    (e.g. a minimal caller-built dict), the box degrades to the boresight-centred
    approximation and the angular weigher falls back to a nadir frame.

    Attributes
    ----------
    bbox : BoundingBox
        Geographic box enclosing the footprint's PSF ground contour.
    latitude, longitude : float
        Boresight centroid (L1B ``Latitude``/``Longitude``), degrees.
    spacecraft_altitude_km : float
        Spacecraft altitude above the surface, km (nominal orbit altitude when the L1B
        field is unavailable). Read only by the PSF geometry; this is the satellite
        altitude, not a surface/terrain height -- kept explicitly in km and separate
        from the camera path's metres-valued surface-height output field.
    viewing_zenith_angle : float
        Viewing zenith angle (L1B ``Viewing_Zenith_Surface``), degrees.
    subsatellite_latitude, subsatellite_longitude : float or None
        Subsatellite ground point (L1B ``Subsatellite_Latitude`` /
        ``Subsatellite_Longitude``), degrees. ``None`` when unavailable. Read by the
        angular weigher (and the ray-traced box) to orient the scan plane.
    cone_angle_rate : float or None
        Instrument cone-angle rate (L1B ``Cone_Angle_Rate``), degrees per second.
        ``None`` when unavailable. Its sign sets the along-scan PSF orientation and a
        near-zero magnitude flags the stationary-scanner (uniform-FOV) case.
    off_limb : bool
        ``True`` for a space / calibration view whose centroid is off the Earth's limb,
        i.e. one with no Earth footprint (:class:`OffLimbError`). ``bbox`` is then only a
        boresight placeholder kept for index alignment; the aggregation path skips these
        records (fill values, zero coverage) and flags them ``OFF_LIMB`` rather than
        aggregating ancillary data against a fabricated geographic box.
    """

    bbox: BoundingBox
    latitude: float
    longitude: float
    spacecraft_altitude_km: float
    viewing_zenith_angle: float
    subsatellite_latitude: float | None = None
    subsatellite_longitude: float | None = None
    cone_angle_rate: float | None = None
    off_limb: bool = False


@dataclass(frozen=True)
class VariableSpec:
    """Metadata describing a single output variable provided by a reader.

    Used by the TileManager and aggregation engine to know what data a reader
    supplies, how to aggregate it, and in which operational modes it is active.

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
    required_mode : OperationalMode
        Minimum operational mode in which this variable is processed. Readers
        and variables with a rank higher than the active mode are excluded.
        Ignored when ``only_modes`` is set (see below).
    n_categories : int or None
        For categorical variables only: number of distinct category values
        (e.g., 20 for IGBP surface type). ``None`` for continuous variables.
    only_modes : tuple[OperationalMode, ...] or None
        Exact set of operational modes (products) this variable belongs to. When
        ``None`` (the default), the variable uses the ``required_mode`` *minimum
        latency* rule: it is carried by every product whose rank is at least
        ``required_mode.rank``. When set, the variable is carried by **exactly**
        the listed modes and the rank rule is bypassed -- this is how a spec is
        pinned to a single product (e.g. a field that must appear only in
        FMATCH-IMAGER, not in the higher-ranked FMATCH-IMAGER-CAMTIME). Membership
        is resolved by :func:`spec_active_in_mode`.
    """

    name: str
    dtype: str
    aggregation: str
    required_mode: OperationalMode
    n_categories: int | None = None
    only_modes: tuple[OperationalMode, ...] | None = None


def spec_active_in_mode(spec: VariableSpec, mode: OperationalMode) -> bool:
    """Return whether a variable spec is carried by a given operational mode.

    The single source of truth for the spec-level product gate. It resolves the
    two mutually exclusive gating rules a :class:`VariableSpec` can carry:

    * ``only_modes`` set -> exact membership: the spec belongs to a product iff
      ``mode`` is one of the listed modes. This can pin a spec to a single product
      even when a higher-ranked product also activates the reader (rank alone
      could not express that, since a higher-latency mode always has a larger
      rank).
    * ``only_modes`` unset -> the minimum-latency rule: the spec is carried once
      the active mode's rank reaches the spec's ``required_mode`` rank.

    Reader-to-product *membership* (which readers feed which product) is a
    separate, coarser gate declared in ``readers/registry.py``; both gates must
    pass for a variable to appear in a product.

    Parameters
    ----------
    spec : VariableSpec
        The variable specification to test.
    mode : OperationalMode
        The operational mode (product) to test against.

    Returns
    -------
    bool
        True when ``spec`` is part of ``mode``'s product definition.
    """
    if spec.only_modes is not None:
        return mode in spec.only_modes
    return spec.required_mode.rank <= mode.rank


# Aggregation strategies that collapse a footprint's pixels to a *mean* value.
# These are the only ones for which a within-footprint standard deviation is
# meaningful: a std-dev quantifies the spread of values around their mean, so it
# pairs with a mean-type aggregation. A std-dev of a categorical "mode" (most
# common value) has no physical meaning, so ``weighted_mode`` variables are
# deliberately excluded below. Note this is *stricter* than "n_categories is
# None": some readers (e.g. SSF's encoded scene-type codes) carry no category
# count yet are still mode-aggregated, and those must NOT get a std-dev companion.
_MEAN_AGGREGATIONS: frozenset[str] = frozenset({"weighted_mean", "weighted_log_mean"})

# Suffix appended to a continuous variable's name to form its std-dev companion.
# Kept as a module constant so the readers, product definitions, and tests all
# agree on the exact spelling (e.g. ``era5_wind_u10_standard_deviation``).
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
    these specs). The companion's ``aggregation`` is set to ``"weighted_std"`` --
    a strategy name that the PSF aggregation engine does not yet implement; like
    the parent variables, the companion is *declared* now and *computed* once the
    aggregation engine is built (see ``product.aggregate_external_variables``).

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
        if spec.aggregation in _MEAN_AGGREGATIONS:
            # A standard deviation is always a non-negative real number, so it is
            # stored as float32 regardless of the parent's dtype and carries no
            # category count. The companion inherits the parent's mode gating --
            # both ``required_mode`` and ``only_modes`` -- so it appears in exactly
            # the same product definitions as its parent.
            expanded.append(
                VariableSpec(
                    name=f"{spec.name}{STANDARD_DEVIATION_SUFFIX}",
                    dtype="float32",
                    aggregation="weighted_std",
                    required_mode=spec.required_mode,
                    n_categories=None,
                    only_modes=spec.only_modes,
                )
            )
    return tuple(expanded)
