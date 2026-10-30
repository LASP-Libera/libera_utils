"""Per-sample PSF weighting and coverage for footprint aggregation.

What this module provides
-------------------------
The aggregation engine needs, for every native sample a reader served for a
footprint, a *weight* proportional to how strongly that ground location contributes
to what the radiometer measured -- i.e. the instrument Point Spread Function (PSF)
evaluated at that sample. This module defines the interface for producing those
weights (:class:`PixelWeigher`) and two implementations:

* :class:`AngularPSFWeigher` -- the CERES-faithful weigher. It projects each sample
  into the radiometer's along-scan/cross-scan angular frame ``(delta, beta)`` (via
  :func:`~libera_utils.footprint_matching.geometry.project_to_angular`) and evaluates
  the analytic CERES PSF :func:`~libera_utils.footprint_matching.psf.psf_weight` there,
  honouring the asymmetric along-scan response, the scan direction (L1B
  ``Cone_Angle_Rate``), and the stationary-scanner static FOV (design doc sections
  2.4.2.5-2.4.2.7, 2.8.1.1).
* :class:`RadialWeigher` -- a boresight-centered Gaussian stand-in kept as a
  dependency-light fallback for footprints with no ephemeris (the camera
  pseudo-footprints).

The weight is evaluated at the sample's own location: no sample is averaged, binned
or interpolated before it is weighted, and the aggregation engine is the only place
the weighted samples are combined.

Coverage
--------
Each :class:`WeightField` also carries the footprint's *coverage*: the fraction of
the PSF's energy that is backed by a usable sample. Because a swath gap simply has
no samples, coverage cannot be read off the samples alone. Each weigher therefore
lays a fine reference grid over the PSF contour in its own frame (angular for the
angular weigher, ground-plane for the radial one) holding the PSF energy of every
cell, and treats each finite sample as covering the square centered on it whose half-width
is the circumradius of the source's native cell (``RESOLUTION_KM / sqrt(2)``), so a regular
lattice of cells leaves no gap at any rotation relative to the reference axes. In the
angular frame the square is foreshortened along-scan by ``cos(VZA)``, as the ground cell
is when seen at an oblique view. Coverage is
the share of the reference energy inside at least one such square. A complete source
scores 1.0; a source missing half the contour scores about 0.5; a source with no samples
scores 0.0 -- exactly the quantity the CERES 75 % / 95 % rule is applied to. A source whose cell
is wider than the contour (the 1 degree VIIRS composites) is complete as soon as one
sample lies inside it.

References
----------
* CERES ATBD v2.2 section 4.4:
  https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from curryer.compute.constants import WGS84_ECCENTRICITY2, WGS84_SEMI_MAJOR_AXIS_KM
from pyproj import Geod

from libera_utils.footprint_matching.geometry import (
    POLAR_LATITUDE_THRESHOLD_DEG,
    ViewingFrame,
    project_to_angular,
    psf_ground_radius_km,
    slant_range_km,
)
from libera_utils.footprint_matching.psf import (
    LIBERA_FOV_HALFANGLE_DEG,
    PSFAngularExtent,
    psf_95_energy_extent,
    psf_weight,
)
from libera_utils.footprint_matching.types import SampleSet

# Sigma of the radial Gaussian kernel as a fraction of the truncation radius. With
# sigma = r_max / 2 the kernel falls to ~13.5% of its peak at the truncation radius,
# i.e. the bulk of the weight sits well inside the PSF's 95%-energy ground radius --
# a reasonable stand-in shape for a centrally-peaked response.
_SIGMA_FRACTION_OF_RMAX: float = 0.5

# Coordinate span (deg) above which a sample set is treated as "large" and the
# RadialWeigher falls back to the exact ellipsoidal Geod.inv distance instead of the
# local-tangent-plane fast path. A normal footprint box spans well under a degree; this
# threshold also catches the wide lon span of a dateline-crossing box, so those and
# polar boxes keep the exact distance where the flat-Earth approximation would drift.
_FAST_DISTANCE_MAX_SPAN_DEG: float = 10.0

# Step of the angular coverage reference grid, degrees of angle at the satellite. Matches
# the step psf_95_energy_extent resolves the contour with; a finer step would only add cost.
_COVERAGE_STEP_DEG: float = 0.02
# Step of the ground-plane coverage reference grid (km) for the radial weigher.
_COVERAGE_STEP_KM: float = 0.25
# Radii are rounded to this many decimals so the radial reference grids (which depend only
# on the truncation radius) are shared through a small cache.
_COVERAGE_RADIUS_DECIMALS: int = 3


def _local_tangent_plane_km(
    origin_lat_deg: float, origin_lon_deg: float, lat_deg: np.ndarray, lon_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """North and east offsets (km) from one point to an array of points on the WGS84 local tangent plane.

    A closed-form, pyproj-free offset used by :class:`RadialWeigher`'s fast path. It
    projects each sample's ``(dlat, dlon)`` offset from the boresight onto the local
    north/east plane using the ellipsoid's meridional (M) and prime-vertical (N) radii of
    curvature evaluated at the boresight latitude. Over a footprint's ~tens-of-km span at
    non-polar latitudes the hypotenuse agrees with the WGS84 geodesic to sub-meter --
    far below the Gaussian stand-in's own approximation -- while avoiding pyproj's
    iterative :meth:`pyproj.Geod.inv` once per footprint. Polar / large boxes fall back
    to Geod.inv (see :data:`_FAST_DISTANCE_MAX_SPAN_DEG`), where the flat-plane model
    and the constant-``cos(lat)`` east scaling break down.
    """
    lat0 = np.radians(origin_lat_deg)
    sin_lat0_sq = np.sin(lat0) ** 2
    denom = 1.0 - WGS84_ECCENTRICITY2 * sin_lat0_sq
    # Meridional (north-south) and prime-vertical (east-west) radii of curvature at the
    # boresight latitude, km.
    meridional_km = WGS84_SEMI_MAJOR_AXIS_KM * (1.0 - WGS84_ECCENTRICITY2) / denom**1.5
    prime_vertical_km = WGS84_SEMI_MAJOR_AXIS_KM / np.sqrt(denom)

    north_km = meridional_km * np.radians(np.asarray(lat_deg, dtype=float) - origin_lat_deg)
    # Wrap the longitude offset to [-180, 180) so samples across the antimeridian from the
    # boresight (or given in another longitude convention) land at their true short offset.
    dlon_deg = (np.asarray(lon_deg, dtype=float) - origin_lon_deg + 180.0) % 360.0 - 180.0
    east_km = prime_vertical_km * np.cos(lat0) * np.radians(dlon_deg)
    return north_km, east_km


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
    """Per-sample PSF weights and coverage for one footprint and one source.

    Attributes
    ----------
    weights : np.ndarray
        1-D array aligned to the sample set's ``lats`` / ``lons``. Each entry is the
        PSF weight of that sample; samples outside the PSF contour are ``0``.
    total_energy : float
        Sum of ``weights`` over every sample in the set.
    max_radius_km : float
        Ground radius (km) at which the kernel was truncated, for diagnostics.
    coverage : float
        Fraction in ``[0, 1]`` of the PSF's in-contour energy backed by a finite
        sample, measured on the weigher's reference grid (see the module docstring).
    """

    weights: np.ndarray
    total_energy: float
    max_radius_km: float
    coverage: float


def _finite_sample_mask(samples: SampleSet) -> np.ndarray:
    """Boolean mask of samples carrying a finite value on any variable plane."""
    values = np.asarray(samples.values)
    if values.ndim == 1:
        return np.isfinite(values)
    return np.any(np.isfinite(values), axis=0)


def _covered_fraction(
    reference: np.ndarray, row: np.ndarray, col: np.ndarray, row_radius: int, col_radius: int
) -> float:
    """Share of ``reference`` energy within the box of a marked ``(row, col)`` cell.

    Each marked cell is dilated into the ``(2 * row_radius + 1) x (2 * col_radius + 1)``
    box around it (a box filter over the mark grid, computed with a 2-D cumulative sum),
    which is the rectangle the sample's native cell covers in the reference frame. Marks
    may lie up to the radius outside the grid: a sample just outside the contour still
    covers the part of its cell that is inside. Marks farther out are ignored.
    """
    total = float(reference.sum())
    if total <= 0.0 or row.size == 0:
        return 0.0
    n_row, n_col = reference.shape
    keep = (row >= -row_radius) & (row < n_row + row_radius) & (col >= -col_radius) & (col < n_col + col_radius)
    if not np.any(keep):
        return 0.0
    marks = np.zeros((n_row + 2 * row_radius, n_col + 2 * col_radius), dtype=np.int64)
    marks[row[keep] + row_radius, col[keep] + col_radius] = 1
    p_row, p_col = marks.shape
    padded = np.zeros((p_row + 1, p_col + 1), dtype=np.int64)
    padded[1:, 1:] = np.cumsum(np.cumsum(marks, axis=0), axis=1)
    rows = np.arange(row_radius, row_radius + n_row)
    cols = np.arange(col_radius, col_radius + n_col)
    r0, r1 = rows - row_radius, rows + row_radius + 1
    c0, c1 = cols - col_radius, cols + col_radius + 1
    box = padded[r1[:, None], c1[None, :]] - padded[r0[:, None], c1[None, :]]
    box -= padded[r1[:, None], c0[None, :]] - padded[r0[:, None], c0[None, :]]
    return float(reference[box > 0].sum()) / total


# A sample covers the square of half-width ``resolution * _COVER_HALF_WIDTH_FACTOR``: the
# circumradius of its native cell, so a lattice of cells rotated at any angle to the
# reference axes is covered without gaps (a lattice of spacing s needs radius s / sqrt(2)).
_COVER_HALF_WIDTH_FACTOR: float = 0.5 * math.sqrt(2.0)


def _dilation_cells(half_width: float, step: float) -> int:
    """Number of reference cells a sample's covering half-width spans (rounded up, never negative)."""
    return max(0, int(math.ceil(half_width / step - 1e-9)))


class PixelWeigher(abc.ABC):
    """Interface: turn a footprint's samples + viewing geometry into per-sample PSF weights.

    Implementations return a :class:`WeightField` whose ``weights`` array is aligned
    to the samples. Keeping this a narrow ABC is the whole point: the aggregation
    engine depends only on this contract, so the radial stand-in and the angular-frame
    weigher are interchangeable.
    """

    #: Whether this weigher consumes a precomputed per-footprint viewing frame (the
    #: ``frame`` argument of :meth:`weight_field`). The orchestrator batches the frames
    #: up front only when this is ``True``, so a weigher that ignores ``frame`` (the
    #: radial stand-in) pays nothing for the machinery.
    uses_viewing_frame: bool = False

    @abc.abstractmethod
    def weight_field(
        self,
        samples: SampleSet,
        boresight_lat_deg: float,
        boresight_lon_deg: float,
        *,
        subsatellite_lat_deg: float,
        subsatellite_lon_deg: float,
        satellite_radius_km: float,
        sample_resolution_km: float,
        viewing_zenith_deg: float = 0.0,
        cone_angle_rate: float | None = None,
        frame: ViewingFrame | None = None,
    ) -> WeightField:
        """Return the per-sample PSF :class:`WeightField` for ``samples``.

        Parameters
        ----------
        samples : SampleSet
            The native samples a reader served for the footprint's bounding box.
        boresight_lat_deg, boresight_lon_deg : float
            Footprint boresight centroid, degrees (the PSF center).
        subsatellite_lat_deg, subsatellite_lon_deg : float
            Geodetic spacecraft sub-point (L1B ``Subsatellite_Latitude`` /
            ``Subsatellite_Longitude``), degrees. Weighers that project samples into
            the angular frame (:class:`AngularPSFWeigher`) use it to orient the scan
            plane; weighers that only need the boresight (:class:`RadialWeigher`)
            ignore it.
        satellite_radius_km : float
            Spacecraft distance from Earth's center (L1B
            ``Radius_of_Satellite_from_Center_of_Earth``), km. Used to convert the PSF's
            angular extent into a ground radius and, for the angular frame, to place the
            satellite.
        sample_resolution_km : float
            The source's native sample spacing (the reader's ``RESOLUTION_KM``). Sets
            the coverage reference bin so one native sample marks about one bin.
        viewing_zenith_deg : float, optional
            Viewing zenith angle, degrees. Elongates the ground footprint at oblique
            views.
        cone_angle_rate : float, optional
            L1B ``Cone_Angle_Rate`` (deg/s). Its sign sets the scan direction and a
            (near-)zero value marks a stationary scanner. Required by
            :class:`AngularPSFWeigher`; ignored by weighers that do not model the
            asymmetric along-scan PSF.
        frame : ViewingFrame, optional
            A precomputed single-footprint viewing frame (see
            :func:`~libera_utils.footprint_matching.geometry.compute_viewing_frames` and
            :meth:`~libera_utils.footprint_matching.geometry.ViewingFrame.for_index`).
            Weighers that project into the angular frame use it to skip the
            per-footprint viewing-geometry reconstruction; weighers that do not need it
            (the radial stand-in) ignore it.
        """


@lru_cache(maxsize=64)
def _radial_reference(max_radius_km: float) -> np.ndarray:
    """Gaussian kernel energy on the ground-plane reference grid of :data:`_COVERAGE_STEP_KM` cells.

    Square grid over ``[-max_radius_km, max_radius_km]`` in north and east, kernel
    truncated at ``max_radius_km`` exactly as :class:`RadialWeigher` truncates its
    sample weights. Cached on the (rounded) radius.
    """
    n = max(1, int(math.ceil(2.0 * max_radius_km / _COVERAGE_STEP_KM)))
    centers = -max_radius_km + (np.arange(n) + 0.5) * _COVERAGE_STEP_KM
    east, north = np.meshgrid(centers, centers)
    distance = np.hypot(north, east)
    sigma = max(max_radius_km * _SIGMA_FRACTION_OF_RMAX, 1.0e-6)
    energy = np.exp(-0.5 * (distance / sigma) ** 2)
    energy[distance > max_radius_km] = 0.0
    return energy


class RadialWeigher(PixelWeigher):
    """Boresight-centered radial Gaussian PSF weight -- the swappable stand-in.

    The weight of a sample is a Gaussian in its distance from the boresight,
    truncated to zero beyond the PSF's ground radius. This is *not* the true
    asymmetric CERES PSF; it is a smooth, centrally-peaked placeholder for footprints
    that carry no ephemeris (the camera pseudo-footprints), where the angular frame
    cannot be built (see :class:`AngularPSFWeigher`).

    Distances from the boresight use a fast local-tangent-plane formula
    (:func:`_local_tangent_plane_km`) for normal footprint boxes -- within sub-meter of
    the WGS84 geodesic over a footprint's span -- and fall back to the exact ellipsoidal
    :class:`pyproj.Geod` distance for polar or large / dateline-crossing boxes, where the
    flat-plane approximation would drift.

    Coverage is measured on a ground-plane reference grid over the truncation disc, each
    finite sample covering the square of half-width ``RESOLUTION_KM / sqrt(2)`` around
    it (see the module
    docstring).

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
        samples: SampleSet,
        boresight_lat_deg: float,
        boresight_lon_deg: float,
        *,
        subsatellite_lat_deg: float,
        subsatellite_lon_deg: float,
        satellite_radius_km: float,
        sample_resolution_km: float,
        viewing_zenith_deg: float = 0.0,
        cone_angle_rate: float | None = None,
        frame: ViewingFrame | None = None,
    ) -> WeightField:
        """Compute the radial Gaussian :class:`WeightField` for ``samples``.

        See :meth:`PixelWeigher.weight_field`. ``subsatellite_*``, ``cone_angle_rate``
        and ``frame`` are accepted for interface compatibility but unused -- the radial
        stand-in only needs the boresight. An empty sample set yields an empty weight
        array, zero total energy and zero coverage.
        """
        lats = np.asarray(samples.lats, dtype=float)
        lons = np.asarray(samples.lons, dtype=float)

        max_radius_km = psf_ground_radius_km(satellite_radius_km, viewing_zenith_deg, self._fov_halfangle_deg)

        if lats.size == 0:
            return WeightField(
                weights=np.zeros(0, dtype=float), total_energy=0.0, max_radius_km=max_radius_km, coverage=0.0
            )

        # Fast path: a local-tangent-plane offset (pure numpy) is within sub-meter of
        # the WGS84 geodesic over a normal footprint's span and avoids pyproj's iterative
        # Geod.inv per footprint. Polar or unusually large / dateline boxes keep the
        # exact ellipsoidal distance, where the flat-plane approximation would drift.
        lat_span = float(lats.max() - lats.min())
        lon_span = float(lons.max() - lons.min())
        near_pole = float(np.max(np.abs(lats))) >= POLAR_LATITUDE_THRESHOLD_DEG
        if near_pole or lat_span > _FAST_DISTANCE_MAX_SPAN_DEG or lon_span > _FAST_DISTANCE_MAX_SPAN_DEG:
            # Geod.inv needs equal-length arrays, so broadcast the scalar boresight. The
            # forward azimuth splits the geodesic distance into north/east components
            # for the coverage grid.
            boresight_lats = np.full(lats.shape, boresight_lat_deg, dtype=float)
            boresight_lons = np.full(lons.shape, boresight_lon_deg, dtype=float)
            azimuth_deg, _, distance_m = self._geod.inv(boresight_lons, boresight_lats, lons, lats)
            distance_km = np.asarray(distance_m, dtype=float) / 1000.0
            azimuth = np.radians(np.asarray(azimuth_deg, dtype=float))
            north_km = distance_km * np.cos(azimuth)
            east_km = distance_km * np.sin(azimuth)
        else:
            north_km, east_km = _local_tangent_plane_km(boresight_lat_deg, boresight_lon_deg, lats, lons)
            distance_km = np.hypot(north_km, east_km)

        # Gaussian kernel, truncated at the PSF ground radius.
        sigma_km = max(max_radius_km * _SIGMA_FRACTION_OF_RMAX, 1.0e-6)
        weights = np.exp(-0.5 * (distance_km / sigma_km) ** 2)
        weights[distance_km > max_radius_km] = 0.0

        # Coverage on a ground-plane reference grid: each finite sample covers the square of
        # the source's native cell centered on it.
        radius_key = round(float(max_radius_km), _COVERAGE_RADIUS_DECIMALS)
        reference = _radial_reference(radius_key)
        finite = _finite_sample_mask(samples)
        row = np.floor((north_km[finite] + radius_key) / _COVERAGE_STEP_KM).astype(int)
        col = np.floor((east_km[finite] + radius_key) / _COVERAGE_STEP_KM).astype(int)
        radius = _dilation_cells(_COVER_HALF_WIDTH_FACTOR * float(sample_resolution_km), _COVERAGE_STEP_KM)
        coverage = _covered_fraction(reference, row, col, radius, radius)

        return WeightField(
            weights=weights,
            total_energy=float(np.sum(weights)),
            max_radius_km=max_radius_km,
            coverage=coverage,
        )


@lru_cache(maxsize=8)
def _angular_reference(extent: PSFAngularExtent, stationary: bool) -> np.ndarray:
    """PSF energy on the angular reference grid of :data:`_COVERAGE_STEP_DEG` cells over ``extent``.

    Rows run over beta from ``-beta_max`` and columns over delta from
    ``-delta_back``. The energy is ``psf_weight * cos(beta)`` (the solid-angle
    Jacobian, as in :func:`~libera_utils.footprint_matching.psf.psf_95_energy_extent`)
    inside the contour and zero outside; the stationary scanner uses the uniform disc
    response instead. A static instrument property, cached per extent.
    """
    span_delta = extent.delta_back_deg + extent.delta_front_deg
    span_beta = 2.0 * extent.beta_max_deg
    n_delta = max(1, int(math.ceil(span_delta / _COVERAGE_STEP_DEG)))
    n_beta = max(1, int(math.ceil(span_beta / _COVERAGE_STEP_DEG)))
    delta_centers = -extent.delta_back_deg + (np.arange(n_delta) + 0.5) * _COVERAGE_STEP_DEG
    beta_centers = -extent.beta_max_deg + (np.arange(n_beta) + 0.5) * _COVERAGE_STEP_DEG
    delta_grid, beta_grid = np.meshgrid(delta_centers, beta_centers)
    if stationary:
        energy = (np.hypot(delta_grid, beta_grid) <= extent.beta_max_deg).astype(float)
    else:
        inside = (
            (delta_grid >= -extent.delta_back_deg)
            & (delta_grid <= extent.delta_front_deg)
            & (np.abs(beta_grid) <= extent.beta_max_deg)
        )
        energy = np.where(inside, psf_weight(delta_grid, beta_grid) * np.cos(np.radians(beta_grid)), 0.0)
    return np.clip(energy, 0.0, None)


class AngularPSFWeigher(PixelWeigher):
    """Science-accurate PSF weighting via the radiometer angular frame.

    This is the CERES-faithful weigher. For each footprint it projects every
    sample's ``(lat, lon)`` into the along-scan/cross-scan angular frame
    ``(delta, beta)`` (via
    :func:`libera_utils.footprint_matching.geometry.project_to_angular`, the inverse
    of the bounding-box ray-trace) and evaluates the analytic CERES PSF
    :func:`libera_utils.footprint_matching.psf.psf_weight` at each sample -- so the
    weight a sample receives is the real, asymmetric instrument response, not a
    boresight-distance kernel. Because it returns the same :class:`WeightField`, the
    aggregation engine is unchanged.

    Scan direction and stationary scanner
    -------------------------------------
    The CERES PSF is asymmetric along-scan (a trailing detector-response tail), so the
    orientation of ``+delta`` depends on the scan direction. That is read from the L1B
    ``Cone_Angle_Rate`` (``cone_angle_rate``): positive means scanning *away* from
    nadir (outward), which reverses the sign of ``delta`` (design doc section
    2.8.1.2 step 5); a (near-)zero rate means the scanner is stationary at a scan
    turnaround, so the uniform static FOV is used instead of the dynamic PSF. The rate
    is therefore required, and :meth:`weight_field` raises ``ValueError`` without a
    finite one or with a non-finite viewing zenith.

    Centroid offset
    ---------------
    The projected ``delta`` is anchored at the reported footprint centroid (the
    boresight). Any residual alignment offset to the PSF centroid is
    :data:`_CENTROID_OFFSET_DEG`, currently assumed 0 (see its LIBSDC-794 note).

    Coverage
    --------
    Measured on a fine angular reference grid over the 95 %-energy contour: each
    finite sample covers a rectangle seen from the satellite,
    ``atan(RESOLUTION_KM / sqrt(2) / slant range)`` to each side cross-scan -- the
    circumradius of its native cell (see the module docstring) -- and ``cos(VZA)`` of
    that along-scan, where the scan plane holds the view direction and the ground cell
    is foreshortened. A source whose cell is wider than the contour (the 1 degree VIIRS
    composites) therefore scores 1.0 as soon as one sample lies in it.

    Direct evaluation vs pre-integrated bins
    ----------------------------------------
    This uses *direct per-sample* PSF evaluation, which yields one weight per sample
    that plugs straight into :class:`WeightField`. The CERES pre-integrated angular-bin
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
    """

    # Consumes the precomputed per-footprint viewing frame (see project_to_angular).
    uses_viewing_frame: bool = True

    def __init__(
        self,
        fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
        energy_fraction: float = 0.95,
    ) -> None:
        self._fov_halfangle_deg = fov_halfangle_deg
        # 95%-energy angular half-extents (delta_back, delta_front, beta_max), a
        # static instrument property computed once and cached inside psf.
        self._extent = psf_95_energy_extent(energy_fraction)
        # The stationary scanner's uniform disc of the FOV half-angle.
        self._static_extent = PSFAngularExtent(
            delta_back_deg=fov_halfangle_deg, delta_front_deg=fov_halfangle_deg, beta_max_deg=fov_halfangle_deg
        )

    def weight_field(
        self,
        samples: SampleSet,
        boresight_lat_deg: float,
        boresight_lon_deg: float,
        *,
        subsatellite_lat_deg: float,
        subsatellite_lon_deg: float,
        satellite_radius_km: float,
        sample_resolution_km: float,
        viewing_zenith_deg: float = 0.0,
        cone_angle_rate: float | None = None,
        frame: ViewingFrame | None = None,
    ) -> WeightField:
        """Compute the CERES-PSF :class:`WeightField` for ``samples``. See class docstring.

        When ``frame`` is supplied (the orchestrator batched the viewing frames up
        front), it is passed straight to :func:`project_to_angular`, skipping the
        per-footprint viewing-geometry reconstruction; the result is identical.

        Raises
        ------
        ValueError
            If ``cone_angle_rate`` is ``None`` or not finite (the scan direction, and so
            the side the PSF's trailing tail lies on, is unknown), or if
            ``viewing_zenith_deg`` is not finite.
        """
        if cone_angle_rate is None or not math.isfinite(cone_angle_rate):
            raise ValueError(
                f"AngularPSFWeigher needs the footprint's L1B Cone_Angle_Rate to orient the asymmetric PSF; "
                f"got {cone_angle_rate!r}. Pass the rate from the L1B record, or use RadialWeigher for a "
                f"footprint without one."
            )
        if not math.isfinite(viewing_zenith_deg):
            raise ValueError(
                f"AngularPSFWeigher needs a finite viewing zenith angle; got {viewing_zenith_deg!r}. "
                f"Pass the footprint's L1B viewing zenith in degrees."
            )
        lats = np.asarray(samples.lats, dtype=float)
        lons = np.asarray(samples.lons, dtype=float)

        # max_radius_km is diagnostic only (the true contour lives in angle space); we
        # reuse the radial ground-radius estimate so the WeightField field stays populated.
        max_radius_km = psf_ground_radius_km(satellite_radius_km, viewing_zenith_deg, self._fov_halfangle_deg)

        if lats.size == 0:
            return WeightField(
                weights=np.zeros(0, dtype=float), total_energy=0.0, max_radius_km=max_radius_km, coverage=0.0
            )

        delta, beta = project_to_angular(
            lats,
            lons,
            frame=frame,
            boresight_lat_deg=boresight_lat_deg,
            boresight_lon_deg=boresight_lon_deg,
            subsatellite_lat_deg=subsatellite_lat_deg,
            subsatellite_lon_deg=subsatellite_lon_deg,
            satellite_radius_km=satellite_radius_km,
        )
        # Alignment offset between the reported centroid (delta = 0) and the PSF
        # centroid. Currently 0 -- see _CENTROID_OFFSET_DEG.
        delta = delta + _CENTROID_OFFSET_DEG

        stationary = abs(cone_angle_rate) <= _STATIC_CONE_RATE_EPS_DEG_PER_S
        if stationary:
            # Scan turnaround: uniform response inside the circular optical FOV, zero
            # outside (design doc section 2.4.2.2). Scan direction is irrelevant.
            delta_psf = delta
            extent = self._static_extent
            inside = np.hypot(delta, beta) <= self._fov_halfangle_deg
            weights = inside.astype(float)
        else:
            # Outward scan (positive cone-angle rate) reverses the along-scan axis so
            # the PSF's trailing tail points the correct way (design doc 2.8.1.2 step 5).
            delta_psf = -delta if cone_angle_rate > 0.0 else delta
            extent = self._extent
            # Truncate at the 95%-energy contour: zero any sample outside the PSF's
            # angular half-extents (evaluated in psf_weight's own input-delta frame,
            # which is the frame delta_psf lives in).
            inside = (
                (delta_psf >= -extent.delta_back_deg)
                & (delta_psf <= extent.delta_front_deg)
                & (np.abs(beta) <= extent.beta_max_deg)
            )
            weights = np.where(inside, psf_weight(delta_psf, beta), 0.0)

        # PSF response is non-negative; clip guards against tiny negative round-off.
        weights = np.clip(weights, 0.0, None)

        # Coverage: each finite sample inside the contour covers its source's native cell as
        # seen from the satellite, centered on its (delta, beta). The scan plane holds the
        # view direction, so along-scan (delta) the cell is foreshortened by cos(VZA);
        # cross-scan (beta) it is not.
        half_cell_rad = (
            _COVER_HALF_WIDTH_FACTOR
            * float(sample_resolution_km)
            / slant_range_km(satellite_radius_km, viewing_zenith_deg)
        )
        half_beta_deg = math.degrees(math.atan(half_cell_rad))
        half_delta_deg = math.degrees(math.atan(half_cell_rad * math.cos(math.radians(viewing_zenith_deg))))
        reference = _angular_reference(extent, stationary)
        finite = _finite_sample_mask(samples)
        row = np.floor((beta[finite] + extent.beta_max_deg) / _COVERAGE_STEP_DEG).astype(int)
        col = np.floor((delta_psf[finite] + extent.delta_back_deg) / _COVERAGE_STEP_DEG).astype(int)
        coverage = _covered_fraction(
            reference,
            row,
            col,
            _dilation_cells(half_beta_deg, _COVERAGE_STEP_DEG),
            _dilation_cells(half_delta_deg, _COVERAGE_STEP_DEG),
        )

        return WeightField(
            weights=weights,
            total_energy=float(np.sum(weights)),
            max_radius_km=max_radius_km,
            coverage=coverage,
        )
