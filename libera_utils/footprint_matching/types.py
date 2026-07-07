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

        Used by ReaderRegistry.get_readers_for_mode() to filter the reader
        set to those whose REQUIRED_MODE rank is <= the active mode's rank.
        """
        return list(OperationalMode).index(self)


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
    n_categories : int or None
        For categorical variables only: number of distinct category values
        (e.g., 20 for IGBP surface type). ``None`` for continuous variables.
    """

    name: str
    dtype: str
    aggregation: str
    required_mode: OperationalMode
    n_categories: int | None = None


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
# agree on the exact spelling (e.g. ``era5_ECMWF_wind_u10_standard_deviation``).
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
            # category count.
            expanded.append(
                VariableSpec(
                    name=f"{spec.name}{STANDARD_DEVIATION_SUFFIX}",
                    dtype="float32",
                    aggregation="weighted_std",
                    required_mode=spec.required_mode,
                    n_categories=None,
                )
            )
    return tuple(expanded)
