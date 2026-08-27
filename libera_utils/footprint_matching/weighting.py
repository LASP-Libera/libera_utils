"""Per-pixel PSF weighting for footprint aggregation.

What this module provides
-------------------------
The aggregation engine needs, for every grid cell of a footprint's merged tile, a
*weight* proportional to how strongly that ground location contributes to what the
radiometer measured -- i.e. the instrument Point Spread Function (PSF) sampled at
that cell. This module defines the interface for producing those weights
(:class:`PixelWeigher`) and two implementations:

* :class:`AngularPSFWeigher` -- the CERES-faithful weigher. It projects each grid
  cell into the radiometer's along-scan/cross-scan angular frame ``(delta, beta)``
  (via :func:`~libera_utils.footprint_matching.geometry.project_to_angular`) and
  evaluates the analytic CERES PSF
  :func:`~libera_utils.footprint_matching.psf.psf_weight` there, honouring the
  asymmetric along-scan response, the scan direction (L1B ``Cone_Angle_Rate``), and
  the stationary-scanner static FOV (design doc sections 2.4.2.5-2.4.2.7, 2.8.1.1).
* :class:`RadialWeigher` -- a simple, boresight-centred Gaussian stand-in kept as a
  dependency-light fallback (it needs only the boresight, not the full viewing
  geometry).

Why the interface matters
-------------------------
Everything the aggregation engine sees is the :class:`WeightField` returned here, so
the two weighers are interchangeable with **no change to**
:mod:`~libera_utils.footprint_matching.aggregation`. Both make the *coverage* metric
meaningful -- uncovered cells (NaN data inside the contour) drop out of the numerator
while still counting toward the denominator, exactly as the CERES 75%/95% rule
expects.

References
----------
* Design doc ``instructions/documentation/Footprint Matching and Scene ID PDF``,
  sections 2.4.2.5-2.4.2.7 and 2.8.1.1.
* CERES ATBD v2.2 section 4.4:
  https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf
"""

from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from pyproj import Geod

from libera_utils.footprint_matching.geometry import (
    NOMINAL_ALTITUDE_KM,
    POLAR_LATITUDE_THRESHOLD_DEG,
    project_to_angular,
    psf_ground_radius_km,
    surface_cell_ecef_km,
)
from libera_utils.footprint_matching.psf import (
    LIBERA_FOV_HALFANGLE_DEG,
    psf_95_energy_extent,
    psf_weight,
)
from libera_utils.footprint_matching.types import GridTile

# Sigma of the radial Gaussian kernel as a fraction of the truncation radius. With
# sigma = r_max / 2 the kernel falls to ~13.5% of its peak at the truncation radius,
# i.e. the bulk of the weight sits well inside the PSF's 95%-energy ground radius --
# a reasonable stand-in shape for a centrally-peaked response.
_SIGMA_FRACTION_OF_RMAX: float = 0.5

# WGS84 ellipsoid parameters (km) for the local-tangent-plane distance fast path.
_WGS84_SEMI_MAJOR_KM: float = 6378.137
_WGS84_ECCENTRICITY_SQ: float = 6.694379990141316e-3  # first eccentricity squared, e^2

# Coordinate span (deg) above which a tile is treated as "large" and the RadialWeigher
# falls back to the exact ellipsoidal Geod.inv distance instead of the local-tangent-plane
# fast path. A normal footprint box spans well under a degree; this threshold also catches
# the wide lon axis a dateline-merged tile produces, so those and polar tiles keep the
# exact distance where the flat-Earth approximation would drift.
_FAST_DISTANCE_MAX_SPAN_DEG: float = 10.0


def _local_tangent_plane_km(
    origin_lat_deg: float, origin_lon_deg: float, lat_deg: np.ndarray, lon_deg: np.ndarray
) -> np.ndarray:
    """Distance (km) from one point to an array of points on the WGS84 local tangent plane.

    A closed-form, pyproj-free distance used by :class:`RadialWeigher`'s fast path. It
    projects each cell's ``(dlat, dlon)`` offset from the boresight onto the local
    north/east plane using the ellipsoid's meridional (M) and prime-vertical (N) radii of
    curvature evaluated at the boresight latitude, then takes the planar hypotenuse. Over a
    footprint's ~tens-of-km span at non-polar latitudes this agrees with the WGS84 geodesic
    to sub-metre -- far below the Gaussian stand-in's own approximation -- while avoiding
    pyproj's iterative :meth:`pyproj.Geod.inv` once per footprint. Polar / large boxes fall
    back to Geod.inv (see :data:`_FAST_DISTANCE_MAX_SPAN_DEG`), where the flat-plane model
    and the constant-``cos(lat)`` east scaling break down.
    """
    lat0 = np.radians(origin_lat_deg)
    sin_lat0_sq = np.sin(lat0) ** 2
    denom = 1.0 - _WGS84_ECCENTRICITY_SQ * sin_lat0_sq
    # Meridional (north-south) and prime-vertical (east-west) radii of curvature at the
    # boresight latitude, km.
    meridional_km = _WGS84_SEMI_MAJOR_KM * (1.0 - _WGS84_ECCENTRICITY_SQ) / denom**1.5
    prime_vertical_km = _WGS84_SEMI_MAJOR_KM / np.sqrt(denom)

    north_km = meridional_km * np.radians(np.asarray(lat_deg, dtype=float) - origin_lat_deg)
    east_km = prime_vertical_km * np.cos(lat0) * np.radians(np.asarray(lon_deg, dtype=float) - origin_lon_deg)
    return np.hypot(north_km, east_km)


# Alignment offset (degrees) between the reported L1B footprint centroid -- which we
# take as the boresight direction, i.e. delta = 0 -- and the PSF's own centroid.
# ASSUMED ZERO for now: the projected delta is fed straight to psf_weight (which
# still applies the CERES-intrinsic centroid shift internally). Set here in one place
# so it is easy to find.
# TODO[LIBSDC-794]: replace 0.0 with the measured Libera centroid offset once the
# ground-characterized PSF/geometry is delivered.
_CENTROID_OFFSET_DEG: float = 0.0

# |cone-angle rate| (deg/s) at or below which the scanner is treated as *stationary*
# (a scan turnaround): the uniform static FOV response is used instead of the dynamic
# PSF, per design doc section 2.4.2.2. Above it, the sign of the rate sets scan
# direction. The exact turnaround value is instrument-specific; this small threshold
# absorbs numerical noise around zero.
# TODO[LIBSDC-794]: confirm the turnaround threshold against real L1B Cone_Angle_Rate.
_STATIC_CONE_RATE_EPS_DEG_PER_S: float = 1.0e-3


@dataclass(frozen=True)
class WeightField:
    """Per-cell PSF weights for one footprint's merged tile.

    Attributes
    ----------
    weights : np.ndarray
        2-D array aligned to the tile's ``(lats, lons)`` grid. Each entry is the PSF
        weight of that cell; cells outside the PSF ground contour are ``0``.
    total_energy : float
        Sum of ``weights`` over every cell in the covered region -- the denominator
        of the coverage metric. Because uncovered cells (NaN data) still carry PSF
        weight here but contribute no *data*, coverage = sampled/total naturally
        drops for partially-covered footprints.
    max_radius_km : float
        Ground radius (km) at which the kernel was truncated, for diagnostics.
    """

    weights: np.ndarray
    total_energy: float
    max_radius_km: float


class PixelWeigher(abc.ABC):
    """Interface: turn a footprint's tile + viewing geometry into per-cell PSF weights.

    Implementations return a :class:`WeightField` whose ``weights`` array is aligned
    to ``tile.lats`` x ``tile.lons``. Keeping this a narrow ABC is the whole point:
    the aggregation engine depends only on this contract, so the radial stand-in and
    the future angular-frame weigher are interchangeable.
    """

    #: Whether this weigher consumes a precomputed per-footprint viewing frame (the
    #: ``frame`` argument of :meth:`weight_field`). The orchestrator batches the frames
    #: up front only when this is ``True``, so a weigher that ignores ``frame`` (the
    #: radial stand-in) pays nothing for the machinery.
    uses_viewing_frame: bool = False

    @abc.abstractmethod
    def weight_field(
        self,
        tile: GridTile,
        boresight_lat_deg: float,
        boresight_lon_deg: float,
        *,
        altitude_km: float | None = None,
        viewing_zenith_deg: float = 0.0,
        subsatellite_lat_deg: float | None = None,
        subsatellite_lon_deg: float | None = None,
        cone_angle_rate: float | None = None,
        frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> WeightField:
        """Return the per-cell PSF :class:`WeightField` for ``tile``.

        Parameters
        ----------
        tile : GridTile
            The merged tile covering the footprint's PSF bounding box.
        boresight_lat_deg, boresight_lon_deg : float
            Footprint boresight centroid, degrees (the PSF centre).
        altitude_km : float, optional
            Satellite altitude above the surface, km. Used to convert the PSF's
            angular extent into a ground radius. Defaults to the nominal orbit
            altitude when not supplied.
        viewing_zenith_deg : float, optional
            Viewing zenith angle, degrees. Elongates the ground footprint at oblique
            views.
        subsatellite_lat_deg, subsatellite_lon_deg : float, optional
            Subsatellite point, degrees. Needed by weighers that project cells into
            the angular frame (:class:`AngularPSFWeigher`) to orient the scan plane;
            ignored by weighers that only need the boresight (:class:`RadialWeigher`).
        cone_angle_rate : float, optional
            L1B ``Cone_Angle_Rate`` (deg/s). Its sign sets the scan direction and a
            (near-)zero value marks a stationary scanner. Ignored by weighers that do
            not model the asymmetric along-scan PSF.
        frame : tuple of np.ndarray, optional
            A precomputed per-footprint ``(satellite, b_hat, c_hat, n_hat)`` viewing
            frame (see
            :func:`~libera_utils.footprint_matching.geometry.compute_viewing_frames`).
            Weighers that project into the angular frame use it to skip the
            per-footprint viewing-geometry reconstruction; weighers that do not need it
            (the radial stand-in) ignore it.
        """


class RadialWeigher(PixelWeigher):
    """Boresight-centred radial Gaussian PSF weight -- the swappable stand-in.

    The weight of a cell is a Gaussian in its great-circle distance from the
    boresight, truncated to zero beyond the PSF's ground radius. This is *not* the
    true asymmetric CERES PSF; it is a smooth, centrally-peaked placeholder that lets
    the aggregation path run end to end while the angular-frame PSF projection is
    built (see the module docstring and :class:`AngularPSFWeigher`).

    Distances from the boresight use a fast local-tangent-plane formula
    (:func:`_local_tangent_plane_km`) for normal footprint boxes -- within sub-metre of
    the WGS84 geodesic over a footprint's span -- and fall back to the exact ellipsoidal
    :class:`pyproj.Geod` distance for polar or large / dateline-merged tiles, where the
    flat-plane approximation would drift.

    Parameters
    ----------
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle used as the floor on the ground radius. Defaults to
        :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.
    """

    def __init__(self, fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG) -> None:
        self._fov_halfangle_deg = fov_halfangle_deg
        # One WGS84 Geod reused for every footprint (thread-safe, cheap to keep).
        self._geod = Geod(ellps="WGS84")

    def weight_field(
        self,
        tile: GridTile,
        boresight_lat_deg: float,
        boresight_lon_deg: float,
        *,
        altitude_km: float | None = None,
        viewing_zenith_deg: float = 0.0,
        subsatellite_lat_deg: float | None = None,
        subsatellite_lon_deg: float | None = None,
        cone_angle_rate: float | None = None,
        frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> WeightField:
        """Compute the radial Gaussian :class:`WeightField` for ``tile``.

        See :meth:`PixelWeigher.weight_field`. ``subsatellite_*``, ``cone_angle_rate``
        and ``frame`` are accepted for interface compatibility but unused -- the radial
        stand-in only needs the boresight. An empty tile (no coordinate cells) yields an
        empty weight array and zero total energy, which the aggregation strategies read
        as "no coverage".
        """
        lats = np.asarray(tile.lats, dtype=float)
        lons = np.asarray(tile.lons, dtype=float)

        altitude = altitude_km if (altitude_km is not None and altitude_km > 0.0) else NOMINAL_ALTITUDE_KM
        max_radius_km = psf_ground_radius_km(altitude, viewing_zenith_deg, self._fov_halfangle_deg)

        # Empty tile (failed / missing region): return an empty, zero-energy field.
        if lats.size == 0 or lons.size == 0:
            shape = (lats.size, lons.size)
            return WeightField(weights=np.zeros(shape, dtype=float), total_energy=0.0, max_radius_km=max_radius_km)

        # Full 2-D mesh of cell centres, then distance from the boresight to every cell.
        lon_grid, lat_grid = np.meshgrid(lons, lats)  # both shape (n_lat, n_lon)
        flat_lat = lat_grid.ravel()
        flat_lon = lon_grid.ravel()

        # Fast path: a local-tangent-plane distance (pure numpy) is within sub-metre of
        # the WGS84 geodesic over a normal footprint's span and avoids pyproj's iterative
        # Geod.inv per footprint. Polar or unusually large / dateline-merged tiles keep
        # the exact ellipsoidal distance, where the flat-plane approximation would drift.
        lat_span = float(lats.max() - lats.min())
        lon_span = float(lons.max() - lons.min())
        near_pole = float(np.max(np.abs(lats))) >= POLAR_LATITUDE_THRESHOLD_DEG
        if near_pole or lat_span > _FAST_DISTANCE_MAX_SPAN_DEG or lon_span > _FAST_DISTANCE_MAX_SPAN_DEG:
            # Geod.inv needs equal-length arrays, so broadcast the scalar boresight.
            boresight_lats = np.full(flat_lat.shape, boresight_lat_deg, dtype=float)
            boresight_lons = np.full(flat_lon.shape, boresight_lon_deg, dtype=float)
            _, _, distance_m = self._geod.inv(boresight_lons, boresight_lats, flat_lon, flat_lat)
            distance_km = np.asarray(distance_m, dtype=float) / 1000.0
        else:
            distance_km = _local_tangent_plane_km(boresight_lat_deg, boresight_lon_deg, flat_lat, flat_lon)

        # Gaussian kernel, truncated at the PSF ground radius.
        sigma_km = max(max_radius_km * _SIGMA_FRACTION_OF_RMAX, 1.0e-6)
        weights_flat = np.exp(-0.5 * (distance_km / sigma_km) ** 2)
        weights_flat[distance_km > max_radius_km] = 0.0

        weights = weights_flat.reshape(lat_grid.shape)
        return WeightField(
            weights=weights,
            total_energy=float(np.sum(weights)),
            max_radius_km=max_radius_km,
        )


class AngularPSFWeigher(PixelWeigher):
    """Science-accurate PSF weighting via the radiometer angular frame.

    This is the CERES-faithful replacement for :class:`RadialWeigher`. For each
    footprint it projects every merged-tile cell's ``(lat, lon)`` into the
    along-scan/cross-scan angular frame ``(delta, beta)`` (via
    :func:`libera_utils.footprint_matching.geometry.project_to_angular`, the inverse
    of the bounding-box ray-trace) and evaluates the analytic CERES PSF
    :func:`libera_utils.footprint_matching.psf.psf_weight` at each cell -- so the
    weight a cell receives is the real, asymmetric instrument response, not a
    boresight-distance kernel. Because it returns the same :class:`WeightField`, the
    aggregation engine is unchanged.

    Scan direction and stationary scanner
    -------------------------------------
    The CERES PSF is asymmetric along-scan (a trailing detector-response tail), so the
    orientation of ``+delta`` depends on the scan direction. That is read from the L1B
    ``Cone_Angle_Rate`` (``cone_angle_rate``): positive means scanning *away* from
    nadir (outward), which reverses the sign of ``delta`` (design doc section
    2.8.1.2 step 5); a (near-)zero rate means the scanner is stationary at a scan
    turnaround, so the uniform static FOV is used instead of the dynamic PSF.

    Centroid offset
    ---------------
    The projected ``delta`` is anchored at the reported footprint centroid (the
    boresight). Any residual alignment offset to the PSF centroid is
    :data:`_CENTROID_OFFSET_DEG`, currently assumed 0 (see its TODO).

    Direct evaluation vs pre-integrated bins
    ----------------------------------------
    This uses *direct per-cell* PSF evaluation, which yields one weight per cell that
    plugs straight into :class:`WeightField`. The CERES pre-integrated angular-bin
    scheme (ATBD Eq. 4.4-18) is a possible later optimization but needs a bin-aware
    aggregation path, so it is intentionally out of scope here.

    Parameters
    ----------
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle used for the stationary-scanner static response and
        as a diagnostic radius floor. Defaults to
        :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.
    energy_fraction : float, optional
        PSF energy fraction defining the truncation contour. Default 0.95 (CERES
        heritage).
    cell_geometry_cache_size : int, optional
        Number of tiles whose per-cell surface ECEF is memoized (default 32). A tile's
        cell geometry is independent of the viewing frame, and because footprints are
        processed in along-track order the same merged tile is reused by many consecutive
        footprints (~18x in a realistic segment); caching the closed-form geodetic->ECEF
        transform per tile turns the dominant weighting cost into one computation per
        distinct tile instead of one per footprint. The cache is a small identity-keyed LRU
        (holding a reference to each cached tile), so a handful of entries covers the active
        working set across all sources. Pass ``0`` to disable it (always recompute).
    """

    # Consumes the precomputed per-footprint viewing frame (see project_to_angular).
    uses_viewing_frame: bool = True

    def __init__(
        self,
        fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
        energy_fraction: float = 0.95,
        cell_geometry_cache_size: int = 32,
    ) -> None:
        self._fov_halfangle_deg = fov_halfangle_deg
        # 95%-energy angular half-extents (delta_back, delta_front, beta_max), a
        # static instrument property computed once and cached inside psf.
        self._extent = psf_95_energy_extent(energy_fraction)
        # Identity-keyed LRU of per-tile cell ECEF (see _cell_ecef_km). Keyed by id(tile)
        # with the tile kept alive by the stored reference, so ids never collide among
        # cached entries. Bounded by cell_geometry_cache_size (0 disables caching).
        self._cell_geometry_cache_size = max(0, int(cell_geometry_cache_size))
        self._cell_ecef_cache: OrderedDict[int, tuple[GridTile, np.ndarray]] = OrderedDict()

    def _cell_ecef_km(self, tile: GridTile, lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
        """Return the tile cells' surface ECEF (km), memoized per tile.

        The value is byte-identical to what :func:`project_to_angular` computes inline (both
        go through :func:`surface_cell_ecef_km`); caching only skips recomputing it for the
        many consecutive footprints that reuse the same merged tile. The ``is tile`` check
        guards against the (harmless) possibility of an ``id`` reused after eviction.
        """
        if self._cell_geometry_cache_size == 0:
            return surface_cell_ecef_km(lat_grid, lon_grid)
        key = id(tile)
        cached = self._cell_ecef_cache.get(key)
        if cached is not None and cached[0] is tile:
            self._cell_ecef_cache.move_to_end(key)
            return cached[1]
        ecef = surface_cell_ecef_km(lat_grid, lon_grid)
        self._cell_ecef_cache[key] = (tile, ecef)
        self._cell_ecef_cache.move_to_end(key)
        while len(self._cell_ecef_cache) > self._cell_geometry_cache_size:
            self._cell_ecef_cache.popitem(last=False)
        return ecef

    def weight_field(
        self,
        tile: GridTile,
        boresight_lat_deg: float,
        boresight_lon_deg: float,
        *,
        altitude_km: float | None = None,
        viewing_zenith_deg: float = 0.0,
        subsatellite_lat_deg: float | None = None,
        subsatellite_lon_deg: float | None = None,
        cone_angle_rate: float | None = None,
        frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> WeightField:
        """Compute the CERES-PSF :class:`WeightField` for ``tile``. See class docstring.

        When ``frame`` is supplied (the orchestrator batched the viewing frames up
        front), it is passed straight to :func:`project_to_angular`, skipping the
        per-footprint viewing-geometry reconstruction; the result is identical.
        """
        lats = np.asarray(tile.lats, dtype=float)
        lons = np.asarray(tile.lons, dtype=float)

        altitude = altitude_km if (altitude_km is not None and altitude_km > 0.0) else NOMINAL_ALTITUDE_KM
        # max_radius_km is diagnostic only (the true contour lives in angle space); we
        # reuse the radial ground-radius estimate so the WeightField field stays populated.
        max_radius_km = psf_ground_radius_km(altitude, viewing_zenith_deg, self._fov_halfangle_deg)

        # Empty tile (failed / missing region): empty, zero-energy field.
        if lats.size == 0 or lons.size == 0:
            return WeightField(
                weights=np.zeros((lats.size, lons.size), dtype=float),
                total_energy=0.0,
                max_radius_km=max_radius_km,
            )

        # Without a subsatellite point we cannot orient the scan plane; fall back to
        # the boresight (a nadir-ish frame). This is a degraded mode -- the orchestrator
        # should supply the L1B subsatellite geolocation whenever available.
        sub_lat = subsatellite_lat_deg if subsatellite_lat_deg is not None else boresight_lat_deg
        sub_lon = subsatellite_lon_deg if subsatellite_lon_deg is not None else boresight_lon_deg

        lon_grid, lat_grid = np.meshgrid(lons, lats)  # both (n_lat, n_lon)
        # The cells' surface ECEF is frame-independent, so it is memoized per tile and reused
        # across the many consecutive footprints that share this merged tile (see _cell_ecef_km).
        cell_ecef = self._cell_ecef_km(tile, lat_grid, lon_grid)
        delta, beta = project_to_angular(
            lat_grid,
            lon_grid,
            boresight_lat_deg,
            boresight_lon_deg,
            sub_lat,
            sub_lon,
            viewing_zenith_deg,
            altitude_km=altitude,
            frame=frame,
            cell_ecef_km=cell_ecef,
        )
        # Alignment offset between the reported centroid (delta = 0) and the PSF
        # centroid. Currently 0 -- see _CENTROID_OFFSET_DEG.
        delta = delta + _CENTROID_OFFSET_DEG

        stationary = cone_angle_rate is not None and abs(cone_angle_rate) <= _STATIC_CONE_RATE_EPS_DEG_PER_S
        if stationary:
            # Scan turnaround: uniform response inside the circular optical FOV, zero
            # outside (design doc section 2.4.2.2). Scan direction is irrelevant.
            angular_radius = np.hypot(delta, beta)
            weights = (angular_radius <= self._fov_halfangle_deg).astype(float)
        else:
            # Outward scan (positive cone-angle rate) reverses the along-scan axis so
            # the PSF's trailing tail points the correct way (design doc 2.8.1.2 step 5).
            delta_psf = -delta if (cone_angle_rate is not None and cone_angle_rate > 0.0) else delta
            weights = psf_weight(delta_psf, beta)
            # Truncate at the 95%-energy contour: zero any cell outside the PSF's
            # angular half-extents (evaluated in psf_weight's own input-delta frame,
            # which is the frame delta_psf lives in).
            inside = (
                (delta_psf >= -self._extent.delta_back_deg)
                & (delta_psf <= self._extent.delta_front_deg)
                & (np.abs(beta) <= self._extent.beta_max_deg)
            )
            weights = np.where(inside, weights, 0.0)

        # PSF response is non-negative; clip guards against tiny negative round-off.
        weights = np.clip(weights, 0.0, None)
        return WeightField(
            weights=weights,
            total_energy=float(np.sum(weights)),
            max_radius_km=max_radius_km,
        )
