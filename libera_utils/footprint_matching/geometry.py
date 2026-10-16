"""Coordinate geometry for footprint matching: L1B viewing geometry -> lat/lon box.

Purpose
-------
Given a footprint's geolocation and viewing geometry from an L1B file,
compute the geographic (latitude/longitude) bounding box of the patch of Earth the
radiometer is sensing. Downstream, that box tells the tile manager which
ancillary-data tiles to load for this footprint. The box is therefore a *safe
superset*: it must always enclose the footprint, and over-covering slightly is fine
(a few extra tiles) while under-covering is not (dropped data). We deliberately
round outward for this reason.

The footprint is not a fixed shape -- at large viewing zenith angles it stretches
dramatically along the scan direction (tens of km at nadir, hundreds of km near the
limb), because a fixed angular offset at the satellite projects to a much larger
ground distance when looking toward the horizon. This module handles that, plus the
hard edge cases: Earth curvature, dateline crossings, footprints that enclose a
pole, and viewing geometry that runs off the limb of the Earth -- including the case
where the boresight is on Earth but the limb-ward corner of the box is not (severe-
angle truncation), which raises :class:`PartialFootprintError`.

Earth model: WGS84 ellipsoid (ECEF ray-trace)
---------------------------------------------
The footprint is anchored at the reported L1B footprint latitude/longitude (the
box center). The satellite is placed in Earth-Centered-Earth-Fixed (ECEF) coordinates
from the L1B subsatellite point and ``Radius_of_Satellite_from_Center_of_Earth``, and
the line from it to the footprint is the boresight. We rotate the boresight by the
PSF's angular half-extents and intersect each resulting ray with the WGS84
ellipsoid. The intersection points, converted back to geodetic latitude/longitude,
give the box. Because the model is the true ellipsoid:

  * flattening (equatorial 6378.137 km vs polar 6356.752 km) is exact, and
  * geodetic latitude is honored exactly -- the local vertical at each point is the
    ellipsoid normal, not the geocentric radial direction.

Off-limb handling falls out of the ray-trace: rays that miss the ellipsoid are off
the Earth. A boresight with no Earth intersection (L1B writes NaN latitude/longitude
for a space or calibration view) raises :class:`OffLimbError`; a boresight on Earth
whose box *corner* rays miss the ellipsoid is partial coverage (truncated/flagged, or
raised as :class:`PartialFootprintError`). Unusable ephemeris fields, or a footprint the
satellite could not have seen, raise :class:`MissingGeolocationError`.

Batched implementation
----------------------
Every helper operates on arrays of footprints. :func:`compute_footprint_bounding_boxes`
and :func:`compute_viewing_frames` are the implementations; the single-footprint
:func:`compute_footprint_bounding_box` and :func:`project_to_angular` run them on a
batch of one.

The geodetic<->ECEF transforms and the ray-ellipsoid intersection are curryer's
(:mod:`curryer.compute.spatial`); :mod:`pyproj` is used only for the surface distances in
the pole-enclosure test (``pyproj.Geod``).

References
----------
* CERES ATBD v2.2, Section 4.4 (viewing geometry):
  https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf
* WGS84 ellipsoid parameters (NIMA TR8350.2).

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from curryer.compute import spatial
from curryer.compute.constants import WGS84_SEMI_MAJOR_AXIS_KM as WGS84_SEMI_MAJOR_AXIS_KM
from curryer.compute.constants import WGS84_SEMI_MINOR_AXIS_KM as WGS84_SEMI_MINOR_AXIS_KM
from numpy.typing import ArrayLike
from pyproj import Geod

from libera_utils.footprint_matching.psf import (
    LIBERA_FOV_HALFANGLE_DEG,
    conservative_along_scan_extent,
    psf_95_energy_extent,
)
from libera_utils.footprint_matching.types import BoundingBox

# The WGS84 ellipsoid axes (km) are imported from curryer, the single source of truth for
# the ellipsoid across the pipeline, and re-exported for this module's callers.

# Outward safety margin applied to the PSF angular half-extents before the box is
# projected. Absorbs any small slop in the PSF extent and guarantees the box is a
# true superset of the footprint.
BBOX_MARGIN_FRACTION: float = 0.05

# Latitude beyond which we flag the box as "polar" so downstream code knows the
# rectangular lat/lon box is a coarse over-approximation (meridians converge). This
# mirrors the design doc's 85 deg threshold.
POLAR_LATITUDE_THRESHOLD_DEG: float = 85.0

# How far (km) the boresight ray's first ground intersection may fall from the reported
# footprint before the footprint counts as not visible from the satellite. L1B
# Latitude/Longitude is that intersection to float32 precision (~metres), so a larger
# miss means the footprint lies beyond the satellite's horizon: inconsistent geolocation.
_VISIBILITY_TOLERANCE_KM: float = 1.0

# L1B fill value. Footprints (or camera pixels) that did not intersect the Earth
# (space/cal views) are stored as this sentinel; we treat such inputs as "no
# footprint" / "off-Earth pixel". Public because the camera-segmentation tool
# (which reads the same L1B fill convention) shares it -- keeping a single source of
# truth so a change to the L1B fill value propagates to both call sites.
L1B_FILL_VALUE: float = -999.0

# A box that does not cross 0 or 180 deg has the same longitude span in [-180, 180) and
# [0, 360); the wrapped representation must be smaller by more than this before it is
# chosen, so float rounding in ``lon % 360`` cannot flag such a box as dateline-wrapping.
_LON_SPAN_TIE_TOL_DEG: float = 1e-9

# Number of samples around the PSF angular-extent rectangle perimeter. 72 == every 5 deg;
# a multiple of 8 so that samples land exactly on the rectangle's four corners.
_N_PERIMETER_SAMPLES: int = 72

# Bisection steps that pull a limb-missing perimeter ray back to the grazing direction.
_LIMB_BISECTION_STEPS: int = 40

# Viewing-zenith angle (deg) past which the ground-footprint elongation factor is
# clamped. At grazing angles ``1/cos(vza)`` diverges; clamping keeps the projected
# ground radius finite and bounded. Real limb handling is the ray-trace's job
# (:class:`OffLimbError`); this only bounds the analytic radius estimate.
_MAX_VZA_FOR_ELONGATION_DEG: float = 80.0


class GeometryError(Exception):
    """Base class for geometry errors raised by this module."""


class OffLimbError(GeometryError):
    """Raised when the boresight does not intersect the Earth's surface.

    L1B RAD writes NaN (or a fill value) for the boresight latitude/longitude of a space
    or calibration view, where the boresight ray misses the ellipsoid. A missing or
    out-of-range boresight latitude or longitude, with the spacecraft geolocation
    present, therefore means there is no Earth footprint at all. The orchestrator is
    expected to catch this and flag/discard the footprint rather than silently
    substituting data.
    """


class MissingGeolocationError(GeometryError):
    """Raised when the spacecraft geolocation needed to build the viewing geometry is unusable.

    The subsatellite point and ``Radius_of_Satellite_from_Center_of_Earth`` are
    populated from the ephemeris for every L1B record, on or off the Earth, so a fill,
    non-finite or out-of-range value there means the record has no geolocation at all,
    not that the instrument looked at space. Also raised when the reported footprint
    could not be seen from the satellite those fields place (it lies beyond the
    horizon), i.e. the inputs disagree. Deliberately *not* an :class:`OffLimbError`, so
    a caller catching off-limb views does not also swallow bad input.
    """


class PartialFootprintError(OffLimbError):
    """Raised when the boresight is on Earth but part of the bounding box is not.

    At severe viewing zenith angles the footprint stretches so far that the
    limb-ward *corner* of its bounding box projects past the Earth's horizon, even
    though the boresight still intersects the surface. The box would otherwise
    silently include a region that is off the Earth.

    By default this condition is *flagged* rather than raised:
    :func:`compute_footprint_bounding_box` truncates the offending rays at the limb
    and sets ``BoundingBox.truncated = True`` (partial coverage). This exception is
    raised only when the caller opts in with ``on_limb="raise"``.

    This is a subclass of :class:`OffLimbError`, so callers that simply
    ``except OffLimbError`` keep working; callers that want to distinguish "no
    footprint at all" (centroid off-limb) from "footprint clipped by the limb" can
    catch this subclass specifically.
    """


# ---------------------------------------------------------------------------
# pyproj singleton (created once, reused for every footprint)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _wgs84_geod() -> Geod:
    """WGS84 :class:`pyproj.Geod` for ellipsoidal surface distances."""
    return Geod(ellps="WGS84")


def _pyproj_seq(array: np.ndarray) -> list[float] | np.ndarray:
    """Coerce a coordinate array into a form pyproj treats as a sequence.

    pyproj dispatches a length-1 NumPy array to its scalar-point code path, which then
    triggers a NumPy>=1.25 ``DeprecationWarning`` (ndim>0 -> scalar). Passing a Python
    list of length 1 keeps it on the array path; arrays of length >= 2 are returned
    unchanged (no copy). Callers normalize the returned coordinates with ``np.asarray``.
    """
    array = np.asarray(array, dtype=float)
    return array.tolist() if array.size == 1 else array


# ---------------------------------------------------------------------------
# Vector / coordinate helpers. Points and directions are ``(..., 3)`` arrays; per-point
# scalars (latitudes, angles) are ``(...)`` arrays.
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return each ``(..., 3)`` vector scaled to unit length."""
    return vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def _geodetic_to_ecef(lat_deg: ArrayLike, lon_deg: ArrayLike, height_km: ArrayLike) -> np.ndarray:
    """Geodetic (lat, lon, ellipsoidal height) -> ``(N, 3)`` ECEF positions in km (curryer)."""
    lat, lon, height = np.broadcast_arrays(
        np.atleast_1d(np.asarray(lat_deg, dtype=float)),
        np.atleast_1d(np.asarray(lon_deg, dtype=float)),
        np.atleast_1d(np.asarray(height_km, dtype=float)),
    )
    return spatial.geodetic_to_ecef(np.stack([lon, lat, height], axis=-1))


def _ecef_to_geodetic(xyz_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(..., 3)`` ECEF km -> flat geodetic ``(lat_deg, lon_deg, height_km)`` arrays (curryer).

    Longitude is in [-180, 180].
    """
    lon_lat_alt = spatial.ecef_to_geodetic(np.asarray(xyz_km, dtype=float).reshape(-1, 3))
    return lon_lat_alt[:, 1], lon_lat_alt[:, 0], lon_lat_alt[:, 2]


def _ellipsoid_normal(lat_deg: ArrayLike, lon_deg: ArrayLike) -> np.ndarray:
    """Outward unit normals (local geodetic zenith) at surface points, ``(N,)`` -> ``(N, 3)``.

    This is the geodetic vertical, ``(cos lat cos lon, cos lat sin lon, sin lat)`` --
    *not* the geocentric radial direction. Honoring this difference is the whole
    point of using the ellipsoid rather than a sphere.
    """
    lat = np.radians(np.atleast_1d(np.asarray(lat_deg, dtype=float)))
    lon = np.radians(np.atleast_1d(np.asarray(lon_deg, dtype=float)))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _arbitrary_tangent(normal: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to each ``(N, 3)`` ``normal`` (direction is arbitrary).

    Used only at nadir, where the scan azimuth is irrelevant (the footprint is a
    near-circular disc), so any tangent direction will do.
    """
    reference = np.where(
        (np.abs(normal[..., 2]) < 0.9)[..., None],
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    return _unit(np.cross(normal, reference))


def _rotate(vec: np.ndarray, axis: np.ndarray, angle_rad: ArrayLike) -> np.ndarray:
    """Rotate ``vec`` about ``axis`` by ``angle_rad`` (Rodrigues' formula).

    ``vec`` and ``axis`` are ``(..., 3)``; ``angle_rad`` broadcasts against the leading
    dims. ``axis`` is normalized to unit length.
    """
    axis = _unit(axis)
    cos_a = np.cos(angle_rad)[..., None]
    sin_a = np.sin(angle_rad)[..., None]
    dot = np.sum(axis * vec, axis=-1, keepdims=True)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * dot * (1.0 - cos_a)


def _ray_ellipsoid_intersection(origin_km: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest intersection of each ray with the WGS84 ellipsoid (curryer).

    Parameters
    ----------
    origin_km : np.ndarray
        ``(..., 3)`` ray starts (the satellite), ECEF km; broadcast against ``direction``.
    direction : np.ndarray
        ``(..., 3)`` ray directions (need not be unit length).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(points, hit)``: ``points`` has ``direction``'s shape and holds the ECEF
        intersection (km) at the smallest positive ray parameter, NaN where the ray
        misses; ``hit`` is the matching boolean mask over the leading dims.
    """
    direction = np.asarray(direction, dtype=float)
    origin = np.broadcast_to(np.asarray(origin_km, dtype=float), direction.shape)
    points = spatial.ray_intersect_ellipsoid(direction.reshape(-1, 3), np.ascontiguousarray(origin).reshape(-1, 3))
    points = points.reshape(direction.shape)
    return points, np.all(np.isfinite(points), axis=-1)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _footprint_arrays(*arrays: ArrayLike) -> list[np.ndarray]:
    """Per-footprint inputs as float arrays of one common 1-D shape.

    Scalars broadcast against the arrays (e.g. one satellite radius for a whole batch).

    Raises
    ------
    ValueError
        If an input has more than one dimension, or the inputs do not share a length.
    """
    converted = [np.atleast_1d(np.asarray(a, dtype=float)) for a in arrays]
    if any(a.ndim != 1 for a in converted):
        raise ValueError(
            f"Per-footprint inputs must be scalars or 1-D arrays, got shapes {[a.shape for a in converted]}."
        )
    try:
        return np.broadcast_arrays(*converted)
    except ValueError as err:
        raise ValueError(
            f"Per-footprint inputs must share one length, got shapes {[a.shape for a in converted]}."
        ) from err


def _usable(lat_deg: ArrayLike, *others: ArrayLike) -> np.ndarray:
    """True where a latitude and its companion values are usable: finite, not the L1B fill, |lat| <= 90.

    xarray decodes ``_FillValue`` to NaN on read, so NaN is the main signal; the explicit
    :data:`L1B_FILL_VALUE` check covers raw reads, and the latitude range check catches
    other fill conventions (e.g. the NetCDF default fill).
    """
    lat = np.asarray(lat_deg, dtype=float)
    values = [lat, *(np.asarray(v, dtype=float) for v in others)]
    valid = np.ones(np.broadcast(*values).shape, dtype=bool)
    for v in values:
        valid &= np.isfinite(v) & (v != L1B_FILL_VALUE)
    return valid & (np.abs(lat) <= 90.0)


def _valid_boresight(boresight_lat_deg: ArrayLike, boresight_lon_deg: ArrayLike) -> np.ndarray:
    """True where the boresight geolocation is usable (missing there means an off-Earth view)."""
    return _usable(boresight_lat_deg, boresight_lon_deg)


def _require_spacecraft(sub_lat: np.ndarray, sub_lon: np.ndarray, satellite_radius_km: np.ndarray) -> None:
    """Raise unless every footprint's ephemeris fields are usable.

    The radius must put the satellite above the ellipsoid (greater than the semi-major
    axis), which also rejects its -9999 fill and any non-physical value.

    Raises
    ------
    MissingGeolocationError
        If any subsatellite point or radius is fill, non-finite or out of range.
    """
    radius = np.asarray(satellite_radius_km, dtype=float)
    valid = _usable(sub_lat, sub_lon) & np.isfinite(radius) & (radius > WGS84_SEMI_MAJOR_AXIS_KM)
    if not np.all(valid):
        raise MissingGeolocationError(
            f"{int(np.count_nonzero(~valid))} footprint(s) have a fill/non-finite/out-of-range "
            "subsatellite point or radius."
        )


# ---------------------------------------------------------------------------
# Viewing geometry: anchor the boresight at the footprint and locate the satellite
# ---------------------------------------------------------------------------


def _satellite_from_subpoint(
    subsatellite_lat_deg: np.ndarray, subsatellite_lon_deg: np.ndarray, satellite_radius_km: np.ndarray
) -> np.ndarray:
    """Satellite ECEF positions (km) from the L1B sub-point and radius, ``(N,)`` -> ``(N, 3)``.

    L1B ``Subsatellite_Latitude``/``Longitude`` is the geodetic sub-point of the
    spacecraft (curryer's ``subobserver_point``, the geodetic coordinates of the
    spacecraft position), so the spacecraft lies on the ellipsoid normal ``n`` through
    the surface point ``s`` there: ``sat = s + h n`` for its geodetic height ``h``. L1B
    carries ``|sat|`` (``Radius_of_Satellite_from_Center_of_Earth``) rather than ``h``,
    so ``h`` is the positive root of ``|s + h n|^2 = R^2``::

        h = -(s . n) + sqrt((s . n)^2 - |s|^2 + R^2)

    Closed form; the root is real and positive because ``R`` exceeds the ellipsoid's
    semi-major axis (:func:`_require_spacecraft`).
    """
    surface = _geodetic_to_ecef(subsatellite_lat_deg, subsatellite_lon_deg, 0.0)
    normal = _ellipsoid_normal(subsatellite_lat_deg, subsatellite_lon_deg)
    s_dot_n = np.sum(surface * normal, axis=1)
    radius = np.asarray(satellite_radius_km, dtype=float)
    height = -s_dot_n + np.sqrt(s_dot_n * s_dot_n - np.sum(surface * surface, axis=1) + radius * radius)
    return surface + height[:, None] * normal


def _viewing_geometry(
    lat: np.ndarray,
    lon: np.ndarray,
    sub_lat: np.ndarray,
    sub_lon: np.ndarray,
    satellite_radius_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate each satellite in ECEF and build the boresight line of sight to its footprint.

    The satellite is placed from the L1B ephemeris fields (:func:`_satellite_from_subpoint`),
    and the boresight is the unit vector from it to the reported footprint ground point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(satellite_ecef_km, boresight_direction)``, each ``(N, 3)``; the direction
        points from the satellite toward the ground.

    Raises
    ------
    MissingGeolocationError
        If any footprint is not visible from its satellite: the boresight ray's first
        ground intersection lands more than :data:`_VISIBILITY_TOLERANCE_KM` from the
        reported footprint, i.e. the footprint lies beyond the satellite's horizon.
    """
    satellite = _satellite_from_subpoint(sub_lat, sub_lon, satellite_radius_km)
    ground = _geodetic_to_ecef(lat, lon, 0.0)
    boresight_direction = _unit(ground - satellite)

    first_hit, hit = _ray_ellipsoid_intersection(satellite, boresight_direction)
    with np.errstate(invalid="ignore"):
        visible = hit & (np.linalg.norm(first_hit - ground, axis=1) <= _VISIBILITY_TOLERANCE_KM)
    if not np.all(visible):
        raise MissingGeolocationError(
            f"{int(np.count_nonzero(~visible))} footprint(s) are not visible from the satellite placed by their "
            "subsatellite point and radius (beyond its horizon): inconsistent geolocation."
        )
    return satellite, boresight_direction


def _cross_scan_axis(boresight_direction: np.ndarray, satellite_km: np.ndarray) -> np.ndarray:
    """Unit cross-scan rotation axes (perpendicular to the scan plane), ``(N, 3)``.

    The scan plane contains the boresight and the satellite's geocentric nadir
    (``-satellite / |satellite|``), the direction curryer measures the cone angle from, so
    the along-scan axis matches the plane L1B ``Cone_Angle``/``Cone_Angle_Rate`` are
    defined in. (The geodetic nadir differs from it by up to ~0.2 deg.) Rotating
    the boresight about this axis slides the look-point along the scan (a cone-angle
    perturbation); rotating about an axis in the plane tilts it cross-scan. A boresight
    at nadir leaves the plane undefined; any tangent then serves, since the footprint is
    a near-circular disc.
    """
    cross_axis = np.cross(-_unit(satellite_km), boresight_direction)
    norm = np.linalg.norm(cross_axis, axis=-1)
    degenerate = norm < 1.0e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = cross_axis / norm[..., None]
    return np.where(degenerate[..., None], _arbitrary_tangent(boresight_direction), unit)


# ---------------------------------------------------------------------------
# PSF perimeter ray-trace
# ---------------------------------------------------------------------------


def _offset_ray_direction(
    boresight_direction: np.ndarray, cross_axis: np.ndarray, delta_deg: ArrayLike, beta_deg: ArrayLike
) -> np.ndarray:
    """Boresight directions rotated by an along-scan (delta) and cross-scan (beta) angle."""
    along = _rotate(boresight_direction, cross_axis, np.radians(delta_deg))
    inplane_axis = np.cross(cross_axis, along)  # in the scan plane, perpendicular to `along`
    return _rotate(along, inplane_axis, np.radians(beta_deg))


def _perimeter_angles(
    along_extent_deg: float, cross_extent_deg: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Along-scan (delta) and cross-scan (beta) offsets tracing the PSF extent rectangle.

    The PSF energy contour is bounded by its delta and beta half-extents, i.e. by a
    rectangle in the angular frame, and its corners reach into that rectangle's corners.
    The rectangle itself is therefore sampled, not the ellipse inscribed in it, which
    would cut the corners off and, at scan azimuths ~30-45 deg from north, leave part of
    the footprint outside the lat/lon box. Each unit-circle direction is pushed out to
    the unit square, then scaled by the extents.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(delta_deg, beta_deg)``, each of shape ``(n_samples,)``.
    """
    t = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    cos_t, sin_t = np.cos(t), np.sin(t)
    to_square = 1.0 / np.maximum(np.abs(cos_t), np.abs(sin_t))
    return along_extent_deg * cos_t * to_square, cross_extent_deg * sin_t * to_square


def _box_angular_extents(fov_halfangle_deg: float) -> tuple[float, float]:
    """Along-scan and cross-scan angular half-extents (deg) the box perimeter is traced at.

    The larger of the dynamic 95%-energy extent and the static FOV, so the box is never
    smaller than the optical field of view (the FOV also stands in for the
    stationary-scanner case, since the box takes no cone-angle rate), enlarged by the
    outward safety margin :data:`BBOX_MARGIN_FRACTION`.
    """
    psf_extent = psf_95_energy_extent()
    scale = 1.0 + BBOX_MARGIN_FRACTION
    along_deg = max(conservative_along_scan_extent(psf_extent), fov_halfangle_deg) * scale
    cross_deg = max(psf_extent.beta_max_deg, fov_halfangle_deg) * scale
    return along_deg, cross_deg


def _limb_clip(
    satellite: np.ndarray,
    boresight_direction: np.ndarray,
    cross_axis: np.ndarray,
    delta_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> np.ndarray:
    """Grazing ground points of perimeter rays that missed the ellipsoid, ``(K, 3)`` ECEF km.

    Bisects each ray's angular fraction back toward the boresight (which is on Earth)
    until the ray just grazes the limb, so the point sits on the horizon and the box
    stays a conservative superset up to the limb.
    """
    lo = np.zeros(satellite.shape[0])
    hi = np.ones(satellite.shape[0])
    for _ in range(_LIMB_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        probe = _offset_ray_direction(boresight_direction, cross_axis, delta_deg * mid, beta_deg * mid)
        _, hit = _ray_ellipsoid_intersection(satellite, probe)
        lo = np.where(hit, mid, lo)
        hi = np.where(hit, hi, mid)
    grazing = _offset_ray_direction(boresight_direction, cross_axis, delta_deg * lo, beta_deg * lo)
    points, _ = _ray_ellipsoid_intersection(satellite, grazing)
    return points


def _trace_perimeter(
    satellite: np.ndarray,
    boresight_direction: np.ndarray,
    cross_axis: np.ndarray,
    delta_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground intersections of every footprint's perimeter rays, clipped to the limb.

    Parameters
    ----------
    satellite, boresight_direction, cross_axis : np.ndarray
        ``(M, 3)`` per-footprint viewing geometry.
    delta_deg, beta_deg : np.ndarray
        ``(S,)`` perimeter offsets shared by every footprint.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(lat_deg, lon_deg, missed)``, each ``(M, S)``. ``missed`` marks rays whose full
        offset runs off the Earth; their point is the grazing point on the horizon.
    """
    directions = _offset_ray_direction(
        boresight_direction[:, None, :], cross_axis[:, None, :], delta_deg[None, :], beta_deg[None, :]
    )
    points, hit = _ray_ellipsoid_intersection(satellite[:, None, :], directions)
    missed = ~hit
    if np.any(missed):
        shape = directions.shape
        points[missed] = _limb_clip(
            np.broadcast_to(satellite[:, None, :], shape)[missed],
            np.broadcast_to(boresight_direction[:, None, :], shape)[missed],
            np.broadcast_to(cross_axis[:, None, :], shape)[missed],
            np.broadcast_to(delta_deg[None, :], missed.shape)[missed],
            np.broadcast_to(beta_deg[None, :], missed.shape)[missed],
        )
    lat, lon, _ = _ecef_to_geodetic(points)
    return lat.reshape(missed.shape), lon.reshape(missed.shape), missed


# ---------------------------------------------------------------------------
# Bounding-box assembly (perimeter lat/lon samples -> lat/lon box)
# ---------------------------------------------------------------------------


def bounding_box_from_points_batch(
    center_lat_deg: np.ndarray,
    center_lon_deg: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    truncated: np.ndarray,
) -> list[BoundingBox]:
    """Build one lat/lon bounding box per footprint from its boundary points.

    This is the box-assembly step shared by two producers:

    * the radiometer path (:func:`compute_footprint_bounding_boxes`), which passes the
      ray-traced PSF *perimeter* samples, and
    * the camera path (:mod:`libera_utils.footprint_matching.camera_segmentation`),
      which passes the four *corner pixels* of a pseudo-footprint's pixel block.

    ``center_lat_deg``/``center_lon_deg`` is each footprint's anchor (the radiometer
    boresight, or the camera block's center pixel) and is used only for the
    pole-enclosure reach test, not for the box extent.

    Handles the three structural edge cases:

    * **Pole enclosure**: if the footprint reaches a pole, all meridians are inside
      it, so longitude spans the full [-180, 180] and the bounding latitude is pinned
      to +/- 90. Detected when the surface distance from the anchor to the pole is
      within the footprint's reach (the max anchor-to-boundary distance).
    * **Dateline crossing**: detected by comparing the longitude span in [-180, 180]
      vs in [0, 360); the smaller span wins. When it wraps, we return the [0, 360)
      representation (``lon_max`` > 180) and set ``wraps_dateline``.
    * **Polar advisory**: boxes touching very high latitudes are flagged ``is_polar``
      so downstream code knows the rectangular box is a coarse over-approximation.

    Parameters
    ----------
    center_lat_deg, center_lon_deg : np.ndarray
        ``(N,)`` footprint anchors, degrees.
    lats, lons : np.ndarray
        ``(N, S)`` boundary points, degrees.
    truncated : np.ndarray
        ``(N,)`` limb-truncation flags, carried onto the boxes.

    Returns
    -------
    list[BoundingBox]
        One box per footprint, in input order.
    """
    geod = _wgs84_geod()
    center_lat = np.asarray(center_lat_deg, dtype=float)
    center_lon = np.asarray(center_lon_deg, dtype=float)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    n, s = lats.shape

    # Pole-enclosure reach: geodesic distance from each center to every boundary point.
    _, _, reach = geod.inv(
        _pyproj_seq(np.repeat(center_lon, s)),
        _pyproj_seq(np.repeat(center_lat, s)),
        _pyproj_seq(lons.reshape(-1)),
        _pyproj_seq(lats.reshape(-1)),
    )
    max_reach = np.asarray(reach, dtype=float).reshape(n, s).max(axis=1)
    _, _, dist_north = geod.inv(
        _pyproj_seq(center_lon), _pyproj_seq(center_lat), _pyproj_seq(center_lon), _pyproj_seq(np.full(n, 90.0))
    )
    _, _, dist_south = geod.inv(
        _pyproj_seq(center_lon), _pyproj_seq(center_lat), _pyproj_seq(center_lon), _pyproj_seq(np.full(n, -90.0))
    )
    encloses_north = np.asarray(dist_north, dtype=float) <= max_reach
    encloses_south = np.asarray(dist_south, dtype=float) <= max_reach

    lat_min = lats.min(axis=1)
    lat_max = lats.max(axis=1)

    # Dateline: choose the [-180,180) or [0,360) representation with the smaller span.
    span_signed = lons.max(axis=1) - lons.min(axis=1)
    lons_360 = lons % 360.0
    span_360 = lons_360.max(axis=1) - lons_360.min(axis=1)
    use_360 = span_360 < span_signed - _LON_SPAN_TIE_TOL_DEG
    lon_min = np.where(use_360, lons_360.min(axis=1), lons.min(axis=1))
    lon_max = np.where(use_360, lons_360.max(axis=1), lons.max(axis=1))
    wraps = use_360 & (lon_max > 180.0)

    is_polar = (np.abs(lat_min) >= POLAR_LATITUDE_THRESHOLD_DEG) | (np.abs(lat_max) >= POLAR_LATITUDE_THRESHOLD_DEG)
    truncated = np.broadcast_to(np.asarray(truncated, dtype=bool), (n,))

    boxes: list[BoundingBox] = []
    for m in range(n):
        if encloses_north[m]:
            boxes.append(BoundingBox(float(lat_min[m]), 90.0, -180.0, 180.0, False, True, bool(truncated[m])))
        elif encloses_south[m]:
            boxes.append(BoundingBox(-90.0, float(lat_max[m]), -180.0, 180.0, False, True, bool(truncated[m])))
        else:
            boxes.append(
                BoundingBox(
                    float(lat_min[m]),
                    float(lat_max[m]),
                    float(lon_min[m]),
                    float(lon_max[m]),
                    bool(wraps[m]),
                    bool(is_polar[m]),
                    bool(truncated[m]),
                )
            )
    return boxes


def bounding_box_from_points(
    center_lat_deg: float,
    center_lon_deg: float,
    lats: ArrayLike,
    lons: ArrayLike,
    *,
    truncated: bool = False,
) -> BoundingBox:
    """Build the lat/lon bounding box of one footprint from its boundary points.

    Single-footprint form of :func:`bounding_box_from_points_batch`, which documents the
    pole, dateline and polar handling.

    Parameters
    ----------
    center_lat_deg, center_lon_deg : float
        Footprint anchor, degrees; used only for the pole-enclosure test.
    lats, lons : array_like
        Boundary-point latitudes and longitudes, degrees.
    truncated : bool, optional
        Limb-truncation flag carried onto the box.

    Returns
    -------
    BoundingBox
        The enclosing box.
    """
    return bounding_box_from_points_batch(
        np.array([center_lat_deg], dtype=float),
        np.array([center_lon_deg], dtype=float),
        np.asarray(lats, dtype=float)[None, :],
        np.asarray(lons, dtype=float)[None, :],
        np.array([truncated]),
    )[0]


def psf_ground_radius_km(
    satellite_radius_km: float,
    viewing_zenith_deg: float,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
) -> float:
    """Project the PSF angular half-extent to an isotropic ground radius, in km.

    A small angular offset ``theta`` at the satellite spans ``rho * tan(theta)``
    perpendicular to the line of sight, where ``rho`` is the slant range. On a spherical
    Earth of radius ``R``, a satellite at distance ``R_s`` from Earth's center sees a
    point at viewing zenith ``v`` from ``rho = sqrt(R_s^2 - (R sin v)^2) - R cos v``. Along
    the view direction the line of sight strikes the surface obliquely, stretching that
    span by a further ``1/cos(v)``; across it there is no stretch. We return the
    along-view (larger) value so the radius bounds the footprint in every direction. At
    nadir this reduces to ``(R_s - R) * tan(theta)``.

    Unlike the rest of this module, which ray-traces the WGS84 ellipsoid, this is an
    analytic estimate on a sphere of the WGS84 equatorial radius
    (:data:`WGS84_SEMI_MAJOR_AXIS_KM`).

    The angular half-extent is the diagonal of the PSF's along-scan and cross-scan
    95%-energy half-extents, because the energy contour reaches toward the corners of
    that rectangle (see :func:`_perimeter_angles`). It is floored at the optical FOV
    half-angle so the radius is never smaller than the field of view.

    This is the single home for the angular-extent -> ground-radius projection, used by
    the per-cell PSF weighers (:mod:`libera_utils.footprint_matching.weighting`).

    Parameters
    ----------
    satellite_radius_km : float
        Spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.
    viewing_zenith_deg : float
        Viewing zenith angle, degrees (clamped at
        :data:`_MAX_VZA_FOR_ELONGATION_DEG` for the elongation factor).
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle, degrees -- the floor on the angular extent.
        Defaults to :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.

    Returns
    -------
    float
        Ground radius in km.
    """
    extent = psf_95_energy_extent()
    # Diagonal angular half-extent -> isotropic radius. conservative_along_scan_extent
    # picks the larger of the asymmetric front/back along-scan reaches.
    angular_deg = max(math.hypot(conservative_along_scan_extent(extent), extent.beta_max_deg), fov_halfangle_deg)
    vza_rad = math.radians(min(abs(viewing_zenith_deg), _MAX_VZA_FOR_ELONGATION_DEG))
    earth_radius_km = WGS84_SEMI_MAJOR_AXIS_KM
    slant_range_km = math.sqrt(
        satellite_radius_km**2 - (earth_radius_km * math.sin(vza_rad)) ** 2
    ) - earth_radius_km * math.cos(vza_rad)
    return slant_range_km * math.tan(math.radians(angular_deg)) / math.cos(vza_rad)


# ---------------------------------------------------------------------------
# Public bounding-box entry points
# ---------------------------------------------------------------------------


def compute_footprint_bounding_boxes(
    boresight_lat_deg: ArrayLike,
    boresight_lon_deg: ArrayLike,
    subsatellite_lat_deg: ArrayLike,
    subsatellite_lon_deg: ArrayLike,
    *,
    satellite_radius_km: ArrayLike,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    n_samples: int = _N_PERIMETER_SAMPLES,
) -> list[BoundingBox | None]:
    """Compute the lat/lon bounding box of each of ``N`` radiometer footprints on the surface.

    Chains: locate each satellite and build the boresight line of sight -> get the PSF
    angular extent -> rotate each boresight by that extent and ray-trace each ray onto
    the WGS84 ellipsoid -> assemble a lat/lon box with pole and dateline handling.

    Inputs come straight from L1B RAD fields: footprint ``Latitude``/``Longitude``,
    ``Subsatellite_Latitude``/``Longitude`` and
    ``Radius_of_Satellite_from_Center_of_Earth``. The last three place the satellite
    exactly; the line from it to the footprint is the boresight, so the viewing zenith
    angle and the scan orientation follow from the geometry rather than being inputs.

    Off-Earth views (a fill, non-finite or out-of-range boresight latitude/longitude) are
    returned as ``None`` at their index rather than raising, so the caller can flag them
    per footprint. Corner rays that run off the limb at a severe angle are truncated at
    the horizon and the box's ``truncated`` flag is set. Missing ephemeris fields or a
    footprint not visible from its satellite raises :class:`MissingGeolocationError` for
    the whole call instead of hiding among the ``None`` entries.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : array_like
        Per-footprint boresight centroids (L1B ``Latitude``/``Longitude``), degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : array_like
        Per-footprint geodetic spacecraft sub-points (L1B ``Subsatellite_Latitude`` /
        ``Subsatellite_Longitude``), degrees.
    satellite_radius_km : array_like
        Per-footprint spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle, degrees. Used as a floor on the PSF extent so the
        box is never smaller than the optical field of view. Defaults to
        :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.
    n_samples : int, optional
        Number of PSF-perimeter samples. Defaults to :data:`_N_PERIMETER_SAMPLES`.

    Returns
    -------
    list[BoundingBox | None]
        One entry per input footprint, in input order. ``None`` marks an off-Earth view.

    Raises
    ------
    ValueError
        If an input is more than 1-D, or the inputs do not share a length (scalars
        broadcast).
    MissingGeolocationError
        If any footprint's subsatellite point or radius is unusable, or any footprint is
        not visible from its satellite.
    """
    lat, lon, sub_lat, sub_lon, radius = _footprint_arrays(
        boresight_lat_deg, boresight_lon_deg, subsatellite_lat_deg, subsatellite_lon_deg, satellite_radius_km
    )
    # Missing ephemeris is bad input, not an off-Earth view: fail loudly rather than hide
    # it among the Nones. A missing boresight with the spacecraft present is a
    # space/calibration view: no Earth intersection.
    _require_spacecraft(sub_lat, sub_lon, radius)
    idx = np.nonzero(_valid_boresight(lat, lon))[0]

    boxes: list[BoundingBox | None] = [None] * lat.size
    if idx.size == 0:
        return boxes

    satellite, boresight_direction = _viewing_geometry(lat[idx], lon[idx], sub_lat[idx], sub_lon[idx], radius[idx])
    cross_axis = _cross_scan_axis(boresight_direction, satellite)
    delta_deg, beta_deg = _perimeter_angles(*_box_angular_extents(fov_halfangle_deg), n_samples)
    perimeter_lat, perimeter_lon, missed = _trace_perimeter(
        satellite, boresight_direction, cross_axis, delta_deg, beta_deg
    )

    valid_boxes = bounding_box_from_points_batch(
        lat[idx], lon[idx], perimeter_lat, perimeter_lon, np.any(missed, axis=1)
    )
    for i, box in zip(idx, valid_boxes, strict=True):
        boxes[int(i)] = box
    return boxes


def compute_footprint_bounding_box(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    *,
    satellite_radius_km: float,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    on_limb: Literal["flag", "raise"] = "flag",
) -> BoundingBox:
    """Compute the lat/lon bounding box of one radiometer footprint on the surface.

    Single-footprint form of :func:`compute_footprint_bounding_boxes`, which documents
    the method and inputs, that raises instead of returning ``None`` for an off-Earth
    view and can raise on limb truncation.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : float
        Footprint centroid (L1B ``Latitude``/``Longitude``), degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : float
        Geodetic sub-point of the spacecraft (L1B ``Subsatellite_Latitude``/``Longitude``),
        degrees.
    satellite_radius_km : float
        Distance of the spacecraft from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle, degrees; the floor on the PSF extent. Defaults to
        :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.
    on_limb : {"flag", "raise"}, optional
        Behavior when a box *corner* ray runs off the Earth limb at a severe angle
        while the boresight is still on Earth. ``"flag"`` (default) truncates those
        rays at the horizon and marks the box ``BoundingBox.truncated = True`` so the
        orchestrator can record partial coverage; ``"raise"`` raises
        :class:`PartialFootprintError` instead. Note this does NOT cover the
        *centroid* being off-limb (or fill inputs) -- those mean there is no footprint
        at all and always raise :class:`OffLimbError`.

    Returns
    -------
    BoundingBox
        Geographic bounding box enclosing the footprint. ``BoundingBox.truncated`` is
        ``True`` when the box was clipped at the limb (partial coverage).

    Raises
    ------
    MissingGeolocationError
        If the subsatellite point or satellite radius is fill, non-finite or out of
        range, or if the footprint is not visible from the satellite they place (it lies
        beyond the horizon). Not an :class:`OffLimbError`: this is bad input, not a view
        off the Earth.
    OffLimbError
        If the boresight latitude or longitude is fill, non-finite or out of range:
        L1B's encoding of a space/calibration view, whose boresight misses the Earth.
        There is no footprint to build, so this raises regardless of ``on_limb``.
    PartialFootprintError
        A subclass of :class:`OffLimbError`. Raised only when the boresight is on Earth
        but a corner ray of the box is off it (severe-angle truncation) *and*
        ``on_limb="raise"``.
    ValueError
        If ``on_limb`` is not ``"flag"`` or ``"raise"``.
    """
    if on_limb not in ("flag", "raise"):
        raise ValueError(f"on_limb must be 'flag' or 'raise', got {on_limb!r}")

    (box,) = compute_footprint_bounding_boxes(
        boresight_lat_deg,
        boresight_lon_deg,
        subsatellite_lat_deg,
        subsatellite_lon_deg,
        satellite_radius_km=satellite_radius_km,
        fov_halfangle_deg=fov_halfangle_deg,
    )
    if box is None:
        raise OffLimbError("Footprint boresight has no Earth intersection (fill/non-finite geolocation).")
    if box.truncated and on_limb == "raise":
        raise PartialFootprintError(
            "A bounding-box corner ray is off the Earth limb: part of the box is off the Earth. "
            "The default on_limb='flag' truncates the box at the horizon and marks it as partial "
            "coverage instead of raising."
        )
    return box


# ---------------------------------------------------------------------------
# Viewing frame: computed once per footprint, reused by the PSF weigher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewingFrame:
    """Radiometer viewing frames of ``N`` footprints.

    The angular-frame PSF projection (:func:`project_to_angular`) needs, per footprint,
    the satellite ECEF position and the three orthonormal look axes ``(b_hat, c_hat,
    n_hat)``. :func:`compute_viewing_frames` builds all ``N`` together so the orchestrator
    can hand each footprint its single-footprint frame via :meth:`for_index`.

    Attributes
    ----------
    satellite : np.ndarray
        ``(N, 3)`` satellite ECEF positions, km.
    b_hat : np.ndarray
        ``(N, 3)`` boresight look-direction unit vectors (``delta = beta = 0``).
    c_hat : np.ndarray
        ``(N, 3)`` cross-scan unit axes (perpendicular to the scan plane).
    n_hat : np.ndarray
        ``(N, 3)`` in-scan-plane unit axes (``= c_hat x b_hat``).
    """

    satellite: np.ndarray
    b_hat: np.ndarray
    c_hat: np.ndarray
    n_hat: np.ndarray

    def __len__(self) -> int:
        return int(self.satellite.shape[0])

    def for_index(self, i: int) -> ViewingFrame:
        """Return the single-footprint frame at index ``i`` (arrays of shape ``(1, 3)``)."""
        rows = [i]
        return ViewingFrame(self.satellite[rows], self.b_hat[rows], self.c_hat[rows], self.n_hat[rows])


def compute_viewing_frames(
    boresight_lat_deg: ArrayLike,
    boresight_lon_deg: ArrayLike,
    subsatellite_lat_deg: ArrayLike,
    subsatellite_lon_deg: ArrayLike,
    *,
    satellite_radius_km: ArrayLike,
) -> ViewingFrame:
    """Build every footprint's viewing frame in one batched call.

    Uses the same viewing geometry and scan-plane axes as the bounding box, so the
    angles :func:`project_to_angular` reads off a frame are consistent with the box.
    Off-Earth views have no frame: drop them (e.g. the ``None`` entries of
    :func:`compute_footprint_bounding_boxes`) before calling.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : array_like
        Per-footprint boresight centroids, degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : array_like
        Per-footprint geodetic spacecraft sub-points (L1B ``Subsatellite_Latitude`` /
        ``Subsatellite_Longitude``), degrees.
    satellite_radius_km : array_like
        Per-footprint spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.

    Returns
    -------
    ViewingFrame
        The ``N`` batched frames.

    Raises
    ------
    ValueError
        If an input is more than 1-D, or the inputs do not share a length.
    MissingGeolocationError
        If any footprint's ephemeris fields are unusable or the footprint is not visible
        from its satellite.
    OffLimbError
        If any boresight latitude or longitude is fill, non-finite or out of range (an
        off-Earth view).
    """
    lat, lon, sub_lat, sub_lon, radius = _footprint_arrays(
        boresight_lat_deg, boresight_lon_deg, subsatellite_lat_deg, subsatellite_lon_deg, satellite_radius_km
    )
    _require_spacecraft(sub_lat, sub_lon, radius)
    off_earth = ~_valid_boresight(lat, lon)
    if np.any(off_earth):
        raise OffLimbError(
            f"{int(np.count_nonzero(off_earth))} footprint(s) have no Earth intersection (fill/non-finite "
            "boresight); drop off-Earth views before building viewing frames."
        )

    satellite, b_hat = _viewing_geometry(lat, lon, sub_lat, sub_lon, radius)
    c_hat = _cross_scan_axis(b_hat, satellite)
    # Unit, because c_hat is a unit vector perpendicular to the unit b_hat.
    n_hat = np.cross(c_hat, b_hat)
    return ViewingFrame(satellite=satellite, b_hat=b_hat, c_hat=c_hat, n_hat=n_hat)


# ---------------------------------------------------------------------------
# Inverse projection: ground lat/lon -> radiometer angular frame (delta, beta)
# ---------------------------------------------------------------------------


def surface_cell_ecef_km(cell_lats_deg: np.ndarray, cell_lons_deg: np.ndarray) -> np.ndarray:
    """ECEF (km) of surface grid cells, raveled to ``(N, 3)`` in the input's row-major order.

    Thin public wrapper over curryer's closed-form ``geodetic_to_ecef`` that flattens the (possibly
    2-D) cell coordinate grids the same way :func:`project_to_angular` does. It is the single
    source of truth for a tile's per-cell surface ECEF so the value
    :class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher` caches per tile is
    byte-identical to the one :func:`project_to_angular` would compute inline. Because a tile's
    cell geometry is independent of the viewing frame, the result can be reused across every
    footprint that shares the tile.

    Parameters
    ----------
    cell_lats_deg, cell_lons_deg : np.ndarray
        Grid-cell latitudes/longitudes in degrees (any common shape). Assumed on the surface
        (ellipsoidal height 0).

    Returns
    -------
    np.ndarray
        ``(N, 3)`` ECEF positions in km, where ``N = cell_lats_deg.size`` in row-major order.
    """
    lats = np.asarray(cell_lats_deg, dtype=float).ravel()
    lons = np.asarray(cell_lons_deg, dtype=float).ravel()
    return _geodetic_to_ecef(lats, lons, 0.0)


def project_to_angular(
    cell_lats_deg: np.ndarray,
    cell_lons_deg: np.ndarray,
    *,
    frame: ViewingFrame | None = None,
    boresight_lat_deg: float | None = None,
    boresight_lon_deg: float | None = None,
    subsatellite_lat_deg: float | None = None,
    subsatellite_lon_deg: float | None = None,
    satellite_radius_km: float | None = None,
    cell_ecef_km: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ground grid cells into the radiometer's angular frame ``(delta, beta)``.

    This is the **inverse** of the forward rotation :func:`_offset_ray_direction`
    used to build the footprint bounding box, and the piece the PSF aggregation
    weighting needs: given where each ancillary grid cell sits on the ground, how far
    off the boresight is it in the along-scan (``delta``) and cross-scan (``beta``)
    angular directions as *seen from the satellite*. Those angles are exactly the
    coordinates the CERES PSF :func:`libera_utils.footprint_matching.psf.psf_weight`
    is a function of (CERES ATBD v2.2 §4.4.2.3, Eq. 4.4-8).

    Pass either a precomputed single-footprint ``frame`` or the footprint's geolocation
    (all of ``boresight_lat_deg``, ``boresight_lon_deg``, ``subsatellite_lat_deg``,
    ``subsatellite_lon_deg`` and ``satellite_radius_km``), from which the frame is built.

    Method
    ------
    The viewing frame (:func:`compute_viewing_frames`) gives three orthonormal axes:

    * ``b_hat`` -- the boresight look direction (``delta = beta = 0``),
    * ``c_hat`` -- the cross-scan axis (perpendicular to the scan plane),
    * ``n_hat = c_hat x b_hat`` -- the in-scan-plane axis (``b_hat``'s ``+delta`` side).

    For each cell we form the unit look vector from the satellite to the cell and read
    off the angles by projecting onto that basis. Because
    :func:`_offset_ray_direction` produces a direction
    ``d = (b_hat cos(delta) + n_hat sin(delta)) cos(beta) - c_hat sin(beta)`` (the
    beta rotation is about ``c_hat x along``), the inverse is exact::

        beta  = asin(-look . c_hat)
        delta = atan2(look . n_hat, look . b_hat)

    The elongation of the footprint at oblique viewing is captured by projecting the
    *actual* ground cells to satellite angles, so no viewing-zenith stretch factor is
    needed.

    Parameters
    ----------
    cell_lats_deg, cell_lons_deg : np.ndarray
        Grid-cell latitudes and longitudes in degrees (any common shape). Assumed on
        the surface (ellipsoidal height 0).
    frame : ViewingFrame, optional
        A precomputed single-footprint frame, as returned by
        :meth:`ViewingFrame.for_index`. When given, the geolocation arguments are ignored.
    boresight_lat_deg, boresight_lon_deg : float, optional
        Footprint boresight centroid, degrees (``delta = beta = 0`` anchor).
    subsatellite_lat_deg, subsatellite_lon_deg : float, optional
        Geodetic spacecraft sub-point (L1B ``Subsatellite_Latitude``/``Longitude``),
        degrees.
    satellite_radius_km : float, optional
        Spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.
    cell_ecef_km : np.ndarray, optional
        Precomputed surface ECEF (km) of the cells, ``(N, 3)`` in ``cell_lats_deg.ravel()``
        order (i.e. exactly :func:`surface_cell_ecef_km`). A tile's cell geometry is
        independent of the viewing frame, so when several footprints share a tile the caller
        (:class:`~libera_utils.footprint_matching.weighting.AngularPSFWeigher`) computes this
        once and passes it in, skipping the per-footprint closed-form geodetic->ECEF transform
        (the dominant weighting cost). When omitted it is computed from the cell coordinates;
        the result is identical either way.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(delta_deg, beta_deg)`` arrays, each the shape of ``cell_lats_deg``.

    Raises
    ------
    ValueError
        If neither ``frame`` nor the full geolocation is given, or ``frame`` does not
        hold exactly one footprint.
    MissingGeolocationError
        If the ephemeris fields are unusable or the footprint is not visible from the
        satellite.
    OffLimbError
        If the boresight latitude or longitude is fill, non-finite or out of range.
    """
    lats = np.asarray(cell_lats_deg, dtype=float)
    lons = np.asarray(cell_lons_deg, dtype=float)
    shape = lats.shape

    if frame is None:
        geolocation = {
            "boresight_lat_deg": boresight_lat_deg,
            "boresight_lon_deg": boresight_lon_deg,
            "subsatellite_lat_deg": subsatellite_lat_deg,
            "subsatellite_lon_deg": subsatellite_lon_deg,
            "satellite_radius_km": satellite_radius_km,
        }
        missing = [name for name, value in geolocation.items() if value is None]
        if missing:
            raise ValueError(f"project_to_angular needs a frame or the footprint geolocation; missing {missing}.")
        frame = compute_viewing_frames(
            boresight_lat_deg,
            boresight_lon_deg,
            subsatellite_lat_deg,
            subsatellite_lon_deg,
            satellite_radius_km=satellite_radius_km,
        )
    if len(frame) != 1:
        raise ValueError(f"frame must hold exactly one footprint (use ViewingFrame.for_index), got {len(frame)}.")
    satellite, b_hat, c_hat, n_hat = frame.satellite[0], frame.b_hat[0], frame.c_hat[0], frame.n_hat[0]

    # Geodetic -> ECEF (km) for every cell center (cells sit on the ellipsoid). This
    # depends only on the cell coordinates, not the frame, so an already-computed value
    # (shared across footprints on the same tile) is used when supplied.
    cells_km = cell_ecef_km if cell_ecef_km is not None else surface_cell_ecef_km(lats, lons)  # (M, 3)

    # Unit look vector from the satellite toward each cell.
    look = _unit(cells_km - satellite)

    # Read the angles off the orthonormal basis (the exact inverse of the forward map).
    # The forward beta-rotation in _offset_ray_direction is about (c_hat x along), which
    # makes the offset direction d = along*cos(beta) - c_hat*sin(beta); hence the beta
    # recovery carries a minus sign on the c_hat component.
    along_scan = look @ b_hat
    in_plane = look @ n_hat
    cross_scan = look @ c_hat
    beta = np.degrees(np.arcsin(np.clip(-cross_scan, -1.0, 1.0)))
    delta = np.degrees(np.arctan2(in_plane, along_scan))
    return delta.reshape(shape), beta.reshape(shape)
