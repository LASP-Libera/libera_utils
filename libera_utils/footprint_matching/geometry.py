"""Coordinate geometry for footprint matching: L1B viewing geometry -> lat/lon box.

Purpose
-------
Given a single radiometer footprint's geolocation and viewing geometry from an
L1B file, compute the geographic (latitude/longitude) bounding box of the patch of
Earth the radiometer is sensing. Downstream, that box tells the tile manager which
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
box center). From the viewing zenith angle and the bearing toward the subsatellite
point we build the boresight line of sight at that ground point, locate the
satellite in Earth-Centered-Earth-Fixed (ECEF) coordinates, then rotate the boresight
by the PSF's angular half-extents and intersect each resulting ray with the WGS84
ellipsoid. The intersection points, converted back to geodetic latitude/longitude,
give the box. Because the model is the true ellipsoid:

  * flattening (equatorial 6378.137 km vs polar 6356.752 km) is exact, and
  * geodetic latitude is honoured exactly -- the local vertical at each point is the
    ellipsoid normal, not the geocentric radial direction.

Off-limb handling falls out of the ray-trace: rays that miss the ellipsoid are off
the Earth. A boresight at viewing zenith >= 90 deg is on (or past) the limb, so it
raises :class:`OffLimbError`; a boresight on Earth whose box *corner* rays miss the
ellipsoid is partial coverage (truncated/flagged, or raised as
:class:`PartialFootprintError`).

We use :mod:`pyproj` for the geodetic<->ECEF transforms (``pyproj.Transformer``) and
for the surface distances in the pole-enclosure test (``pyproj.Geod``).

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

import numpy as np
from pyproj import Geod, Transformer

from libera_utils.footprint_matching.psf import (
    LIBERA_FOV_HALFANGLE_DEG,
    conservative_along_scan_extent,
    psf_95_energy_extent,
)
from libera_utils.footprint_matching.types import BoundingBox

# WGS84 ellipsoid parameters. The semi-major axis and inverse flattening are the
# defining constants (NIMA TR8350.2); the semi-minor axis and first-eccentricity
# follow from them. All lengths in km to match the rest of the module.
WGS84_SEMI_MAJOR_AXIS_KM: float = 6378.137
WGS84_FLATTENING: float = 1.0 / 298.257223563
WGS84_SEMI_MINOR_AXIS_KM: float = WGS84_SEMI_MAJOR_AXIS_KM * (1.0 - WGS84_FLATTENING)
# First eccentricity squared, e^2 = f(2 - f), for the closed-form geodetic->ECEF used on
# the hot per-footprint cell projection (:func:`_geodetic_to_ecef_surface`).
WGS84_FIRST_ECCENTRICITY_SQ: float = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)

# Fallback satellite altitude, used only when the altitude cannot be derived from
# the inputs (no Altitude field AND the footprint is essentially at nadir, where the
# altitude-recovery geometry is numerically degenerate). Value is the median altitude
# recovered from the example L1B file (834.6 km), i.e. the JPSS orbit.
# TODO[LIBSDC-794]: read the nominal altitude from mission config rather than
# hard-coding it.
NOMINAL_ALTITUDE_KM: float = 835.0

# Outward safety margin applied to the PSF angular half-extents before the box is
# projected. Absorbs any small slop in the PSF extent and guarantees the box is a
# true superset of the footprint.
BBOX_MARGIN_FRACTION: float = 0.05

# Latitude beyond which we flag the box as "polar" so downstream code knows the
# rectangular lat/lon box is a coarse over-approximation (meridians converge). This
# mirrors the design doc's 85 deg threshold.
POLAR_LATITUDE_THRESHOLD_DEG: float = 85.0

# Sentinel below which a footprint is treated as essentially at nadir: the scan
# azimuth and along-scan asymmetry become ill-defined, so we use the nominal altitude
# and an arbitrary scan orientation. 1e-6 deg is far smaller than any real footprint.
_NADIR_CONE_ANGLE_EPS_DEG: float = 1e-6

# L1B fill value. Footprints (or camera pixels) that did not intersect the Earth
# (space/cal views) are stored as this sentinel; we treat such inputs as "no
# footprint" / "off-Earth pixel". Public because the camera-segmentation tool
# (which reads the same L1B fill convention) shares it -- keeping a single source of
# truth so a change to the L1B fill value propagates to both call sites.
L1B_FILL_VALUE: float = -999.0

# Private alias of the public fill-value constant, used internally below.
_L1B_FILL_VALUE: float = L1B_FILL_VALUE

# Number of samples around the PSF angular-extent ellipse perimeter. 72 == every 5 deg.
_N_PERIMETER_SAMPLES: int = 72

# Viewing-zenith angle (deg) past which the ground-footprint elongation factor is
# clamped. At grazing angles ``1/cos(vza)`` diverges; clamping keeps the projected
# ground radius finite and bounded. Real limb handling is the ray-trace's job
# (:class:`OffLimbError`); this only bounds the analytic radius estimate.
_MAX_VZA_FOR_ELONGATION_DEG: float = 80.0


class GeometryError(Exception):
    """Base class for geometry errors raised by this module."""


class OffLimbError(GeometryError):
    """Raised when the viewing geometry does not intersect the Earth's surface.

    This happens when the boresight viewing zenith angle is at or beyond 90 deg (the
    line of sight is tangent to / past the limb), or when the input footprint is a
    fill value (a non-Earth view). The orchestrator is expected to catch this and
    flag/discard the footprint rather than silently substituting data.
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
# pyproj singletons (created once, reused for every footprint)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _geodetic_to_ecef_transformer() -> Transformer:
    """Transformer from geodetic 3D (EPSG:4979) to geocentric ECEF (EPSG:4978)."""
    return Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


@lru_cache(maxsize=1)
def _ecef_to_geodetic_transformer() -> Transformer:
    """Transformer from geocentric ECEF (EPSG:4978) to geodetic 3D (EPSG:4979)."""
    return Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)


@lru_cache(maxsize=1)
def _wgs84_geod() -> Geod:
    """WGS84 :class:`pyproj.Geod` for ellipsoidal surface distances."""
    return Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Vector / coordinate helpers
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit length."""
    return vec / np.linalg.norm(vec)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height_km: float) -> np.ndarray:
    """Geodetic (lat, lon, ellipsoidal height) -> ECEF position vector, in km."""
    x, y, z = _geodetic_to_ecef_transformer().transform(lon_deg, lat_deg, height_km * 1000.0)
    return np.array([x, y, z], dtype=float) / 1000.0


def _ecef_to_geodetic(xyz_km: np.ndarray) -> tuple[float, float, float]:
    """ECEF position (km) -> geodetic ``(lat_deg, lon_deg, height_km)``.

    Longitude is normalized to [-180, 180].
    """
    lon, lat, height_m = _ecef_to_geodetic_transformer().transform(
        xyz_km[0] * 1000.0, xyz_km[1] * 1000.0, xyz_km[2] * 1000.0
    )
    lon = (lon + 540.0) % 360.0 - 180.0
    return lat, lon, height_m / 1000.0


def _ellipsoid_normal(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Outward unit normal (local geodetic zenith) at a surface point.

    This is the geodetic vertical, ``(cos lat cos lon, cos lat sin lon, sin lat)`` --
    *not* the geocentric radial direction. Honouring this difference is the whole
    point of using the ellipsoid rather than a sphere.
    """
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=float,
    )


def _arbitrary_tangent(normal: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to ``normal`` (direction is arbitrary).

    Used only at nadir, where the scan azimuth is irrelevant (the footprint is a
    near-circular disc), so any tangent direction will do.
    """
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    return _unit(np.cross(normal, reference))


def _rotate(vec: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate ``vec`` about a unit ``axis`` by ``angle_rad`` (Rodrigues' formula)."""
    axis = _unit(axis)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * np.dot(axis, vec) * (1.0 - cos_a)


def _ray_ellipsoid_intersection(origin_km: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Nearest intersection of a ray with the WGS84 ellipsoid, or ``None`` if it misses.

    The ray is ``origin + s * direction`` for ``s > 0``. Scaling each axis by the
    reciprocal semi-axis turns the ellipsoid into the unit sphere, so the
    intersection reduces to a quadratic ``|D*origin + s*D*direction|^2 = 1``.

    Parameters
    ----------
    origin_km : np.ndarray
        Ray start (the satellite), ECEF km.
    direction : np.ndarray
        Ray direction (need not be unit length).

    Returns
    -------
    np.ndarray or None
        The ECEF intersection point (km) at the smallest positive ray parameter, or
        ``None`` when the ray does not meet the ellipsoid.
    """
    scale = np.array([1.0 / WGS84_SEMI_MAJOR_AXIS_KM, 1.0 / WGS84_SEMI_MAJOR_AXIS_KM, 1.0 / WGS84_SEMI_MINOR_AXIS_KM])
    o = origin_km * scale
    d = direction * scale
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(o, d))
    c = float(np.dot(o, o)) - 1.0
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    roots = ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a))
    positive = [s for s in roots if s > 0.0]
    if not positive:
        return None
    return origin_km + min(positive) * direction


# ---------------------------------------------------------------------------
# Viewing geometry: anchor the boresight at the footprint and locate the satellite
# ---------------------------------------------------------------------------


def _satellite_along_ray(ground_km: np.ndarray, up_direction: np.ndarray, altitude_km: float) -> np.ndarray:
    """Point along the upward ray from a ground point at a given geodetic height.

    Walks outward from ``ground_km`` along ``up_direction`` (the unit vector toward
    the satellite) until the geodetic ellipsoidal height equals ``altitude_km``.
    Geodetic height increases monotonically along an upward-going ray, so a simple
    bisection converges.

    Parameters
    ----------
    ground_km : np.ndarray
        Footprint ground point, ECEF km.
    up_direction : np.ndarray
        Unit direction from the ground toward the satellite.
    altitude_km : float
        Target ellipsoidal height of the satellite, km.

    Returns
    -------
    np.ndarray
        Satellite ECEF position, km.
    """
    lo = 0.0
    hi = max(altitude_km, 1.0)
    # Grow the upper bracket until the height overshoots the target (oblique rays
    # need a long slant range to gain altitude).
    while _ecef_to_geodetic(ground_km + hi * up_direction)[2] < altitude_km and hi < 1.0e6:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _ecef_to_geodetic(ground_km + mid * up_direction)[2] < altitude_km:
            lo = mid
        else:
            hi = mid
    return ground_km + hi * up_direction


def _viewing_geometry(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    viewing_zenith_deg: float,
    altitude_km: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the boresight line of sight and locate the satellite in ECEF.

    The footprint ground point ``P`` (the box center) is the reported L1B
    latitude/longitude. The boresight ray to the satellite makes the viewing zenith
    angle with the local vertical at ``P``, in the vertical plane that also contains
    the subsatellite point. The satellite ``S`` lies along that ray and on the
    geodetic normal through the subsatellite point.

    * If ``altitude_km`` is given, ``S`` is the point on the boresight ray at that
      ellipsoidal height.
    * Otherwise ``S`` is recovered as the (least-squares) intersection of the
      boresight ray from ``P`` and the vertical through the subsatellite point.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(satellite_ecef_km, boresight_direction, subsatellite_normal)`` where the
        boresight direction is the unit vector from the satellite toward the ground.
    """
    ground = _geodetic_to_ecef(boresight_lat_deg, boresight_lon_deg, 0.0)
    normal_p = _ellipsoid_normal(boresight_lat_deg, boresight_lon_deg)
    subsat_ground = _geodetic_to_ecef(subsatellite_lat_deg, subsatellite_lon_deg, 0.0)
    normal_sub = _ellipsoid_normal(subsatellite_lat_deg, subsatellite_lon_deg)

    # Horizontal direction at the footprint pointing toward the subsatellite point.
    toward_subsat = subsat_ground - ground
    horizontal = toward_subsat - np.dot(toward_subsat, normal_p) * normal_p
    horizontal_norm = float(np.linalg.norm(horizontal))
    near_nadir = viewing_zenith_deg < _NADIR_CONE_ANGLE_EPS_DEG or horizontal_norm < 1.0e-9
    horizontal_unit = _arbitrary_tangent(normal_p) if near_nadir else horizontal / horizontal_norm

    theta = math.radians(viewing_zenith_deg)
    # Unit direction from the ground point toward the satellite.
    to_satellite = math.cos(theta) * normal_p + math.sin(theta) * horizontal_unit
    boresight_direction = -to_satellite  # satellite -> ground

    if altitude_km is not None and altitude_km > 0.0:
        satellite = _satellite_along_ray(ground, to_satellite, altitude_km)
    elif near_nadir:
        satellite = _satellite_along_ray(ground, to_satellite, NOMINAL_ALTITUDE_KM)
    else:
        # S = ground + rho * to_satellite = subsat_ground + h * normal_sub.
        # Solve the over-determined system for (rho, h); rho is the slant range.
        matrix = np.column_stack([to_satellite, -normal_sub])
        solution, *_ = np.linalg.lstsq(matrix, toward_subsat, rcond=None)
        rho = float(solution[0])
        if rho <= 0.0:
            satellite = _satellite_along_ray(ground, to_satellite, NOMINAL_ALTITUDE_KM)
        else:
            satellite = ground + rho * to_satellite

    return satellite, boresight_direction, normal_sub


# ---------------------------------------------------------------------------
# PSF perimeter ray-trace
# ---------------------------------------------------------------------------


def _scan_frame_axes(boresight_direction: np.ndarray, subsatellite_normal: np.ndarray) -> np.ndarray:
    """Cross-scan rotation axis (perpendicular to the scan plane).

    The scan plane contains the boresight and the satellite nadir direction. Rotating
    the boresight about this axis slides the look-point along the scan (a cone-angle
    perturbation); rotating about an axis in the plane tilts it cross-scan.

    Returns
    -------
    np.ndarray
        Unit cross-scan rotation axis.
    """
    nadir = -subsatellite_normal
    cross_axis = np.cross(nadir, boresight_direction)
    if float(np.linalg.norm(cross_axis)) < 1.0e-9:
        # Boresight ~ nadir: orientation is irrelevant for the near-circular disc.
        return _arbitrary_tangent(boresight_direction)
    return _unit(cross_axis)


def _offset_ray_direction(
    boresight_direction: np.ndarray, cross_axis: np.ndarray, delta_deg: float, beta_deg: float
) -> np.ndarray:
    """Boresight direction rotated by an along-scan (delta) and cross-scan (beta) angle."""
    along = _rotate(boresight_direction, cross_axis, math.radians(delta_deg))
    inplane_axis = np.cross(cross_axis, along)  # in the scan plane, perpendicular to `along`
    return _rotate(along, inplane_axis, math.radians(beta_deg))


def _perimeter_point(
    satellite_km: np.ndarray,
    boresight_direction: np.ndarray,
    cross_axis: np.ndarray,
    delta_deg: float,
    beta_deg: float,
) -> tuple[float, float, bool]:
    """Ground intersection of one PSF-perimeter ray, clipping to the limb if it misses.

    Returns ``(lat_deg, lon_deg, missed)``. When the full-extent ray misses the
    ellipsoid (``missed=True``) the angular offset is bisected back toward the
    boresight until the ray just grazes the limb, so the returned point sits on the
    horizon (keeping the box a conservative superset up to the limb).
    """
    direction = _offset_ray_direction(boresight_direction, cross_axis, delta_deg, beta_deg)
    hit = _ray_ellipsoid_intersection(satellite_km, direction)
    if hit is not None:
        lat, lon, _ = _ecef_to_geodetic(hit)
        return lat, lon, False

    # Bisect the angular fraction down to the grazing direction (largest that hits).
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        probe = _offset_ray_direction(boresight_direction, cross_axis, delta_deg * mid, beta_deg * mid)
        if _ray_ellipsoid_intersection(satellite_km, probe) is not None:
            lo = mid
        else:
            hi = mid
    grazing = _offset_ray_direction(boresight_direction, cross_axis, delta_deg * lo, beta_deg * lo)
    hit = _ray_ellipsoid_intersection(satellite_km, grazing)
    lat, lon, _ = _ecef_to_geodetic(hit)
    return lat, lon, True


# ---------------------------------------------------------------------------
# Bounding-box assembly (perimeter lat/lon samples -> lat/lon box)
# ---------------------------------------------------------------------------


def bounding_box_from_points(
    center_lat_deg: float,
    center_lon_deg: float,
    lats: list[float],
    lons: list[float],
    *,
    truncated: bool = False,
) -> BoundingBox:
    """Build a lat/lon bounding box from a set of footprint boundary points.

    This is the shared box-assembly step used by two producers:

    * the radiometer path (:func:`compute_footprint_bounding_box`), which passes the
      ray-traced PSF *perimeter* samples, and
    * the camera path (:mod:`libera_utils.footprint_matching.camera_segmentation`),
      which passes the four *corner pixels* of a pseudo-footprint's pixel block.

    Both need identical pole/dateline handling, so keeping one implementation avoids
    two subtly different box builders. ``center_lat_deg``/``center_lon_deg`` is the
    footprint anchor (the radiometer boresight, or the camera block's center pixel)
    and is used only for the pole-enclosure reach test, not for the box extent.

    Handles the three structural edge cases:

    * **Pole enclosure**: if the footprint reaches a pole, all meridians are inside
      it, so longitude spans the full [-180, 180] and the bounding latitude is pinned
      to +/- 90. Detected when the surface distance from the boresight to the pole is
      within the footprint's reach (the max boresight-to-perimeter distance).
    * **Dateline crossing**: detected by comparing the longitude span in [-180, 180]
      vs in [0, 360); the smaller span wins. When it wraps, we return the [0, 360)
      representation (``lon_max`` > 180) and set ``wraps_dateline``.
    * **Polar advisory**: boxes touching very high latitudes are flagged ``is_polar``
      so downstream code knows the rectangular box is a coarse over-approximation.
    """
    geod = _wgs84_geod()

    # Pole enclosure: compare the footprint's reach to the distance to each pole.
    perimeter_distances_m = [
        geod.inv(center_lon_deg, center_lat_deg, lon, lat)[2] for lat, lon in zip(lats, lons, strict=True)
    ]
    max_reach_m = max(perimeter_distances_m)
    dist_north_pole_m = geod.inv(center_lon_deg, center_lat_deg, center_lon_deg, 90.0)[2]
    dist_south_pole_m = geod.inv(center_lon_deg, center_lat_deg, center_lon_deg, -90.0)[2]

    if dist_north_pole_m <= max_reach_m:
        return BoundingBox(min(lats), 90.0, -180.0, 180.0, wraps_dateline=False, is_polar=True, truncated=truncated)
    if dist_south_pole_m <= max_reach_m:
        return BoundingBox(-90.0, max(lats), -180.0, 180.0, wraps_dateline=False, is_polar=True, truncated=truncated)

    lat_min, lat_max = min(lats), max(lats)

    # Dateline handling: choose the representation with the smaller longitude span.
    lons_arr = np.asarray(lons)
    span_signed = float(lons_arr.max() - lons_arr.min())  # span in [-180, 180]
    lons_360 = lons_arr % 360.0
    span_360 = float(lons_360.max() - lons_360.min())  # span in [0, 360)

    if span_360 < span_signed:
        lon_min = float(lons_360.min())
        lon_max = float(lons_360.max())  # may exceed 180 -> signals the wrap
        wraps_dateline = lon_max > 180.0
    else:
        lon_min = float(lons_arr.min())
        lon_max = float(lons_arr.max())
        wraps_dateline = False

    # Advisory polar flag for boxes that reach very high latitudes.
    is_polar = abs(lat_min) >= POLAR_LATITUDE_THRESHOLD_DEG or abs(lat_max) >= POLAR_LATITUDE_THRESHOLD_DEG

    return BoundingBox(
        lat_min, lat_max, lon_min, lon_max, wraps_dateline=wraps_dateline, is_polar=is_polar, truncated=truncated
    )


def psf_ground_radius_km(
    altitude_km: float,
    viewing_zenith_deg: float,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
) -> float:
    """Project the PSF angular half-extent to an isotropic ground radius, in km.

    A fixed angular offset at the satellite projects to a ground distance that grows
    with altitude and, at oblique views, with ``1/cos(vza)`` because the line of sight
    strikes the surface at a shallow angle (the footprint "elongates on the ground" at
    high viewing zenith angles). We take the larger of the PSF's along-scan and
    cross-scan 95%-energy half-extents -- an isotropic over-estimate -- and floor it at
    the optical FOV half-angle so the radius is never smaller than the field of view.

    This is the single home for the angular-extent -> ground-radius projection, shared
    by the boresight-centred bounding box (:func:`bounding_box_from_boresight`) and the
    per-cell PSF weighers
    (:mod:`libera_utils.footprint_matching.weighting`), so the two never drift.

    Parameters
    ----------
    altitude_km : float
        Satellite altitude above the surface, km.
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
    # Largest angular half-extent -> isotropic radius. conservative_along_scan_extent
    # picks the larger of the asymmetric front/back along-scan reaches.
    angular_deg = max(conservative_along_scan_extent(extent), extent.beta_max_deg, fov_halfangle_deg)
    nadir_radius_km = altitude_km * math.tan(math.radians(angular_deg))
    vza = min(abs(viewing_zenith_deg), _MAX_VZA_FOR_ELONGATION_DEG)
    return nadir_radius_km / math.cos(math.radians(vza))


def bounding_box_from_boresight(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    viewing_zenith_deg: float,
    *,
    altitude_km: float | None = None,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    n_samples: int = _N_PERIMETER_SAMPLES,
) -> BoundingBox:
    """Boresight-centred bounding box from the PSF ground radius (no subsatellite point).

    A lighter-weight companion to :func:`compute_footprint_bounding_box` for footprints
    that carry only a boresight and a viewing zenith angle. Rather than ray-tracing the
    asymmetric PSF perimeter (which needs the subsatellite point to orient the scan
    plane), it draws a circle of the PSF's isotropic ground radius
    (:func:`psf_ground_radius_km`) around the boresight and boxes those points with the
    same pole/dateline handling as :func:`bounding_box_from_points`.

    Because the box is centred on the boresight and symmetric, it is always a safe
    superset of the true (elongated, asymmetric) footprint for the purpose of deciding
    which ancillary tiles to load -- exactly what the bounding box is for. It is never
    limb-truncated (the circle is drawn on the surface, not ray-traced), so
    ``BoundingBox.truncated`` is always ``False`` here.

    The production radiometer path now reads the L1B subsatellite geolocation and
    cone-angle rate, so it builds the true asymmetric box via
    :func:`compute_footprint_bounding_box` and weights with the :class:`AngularPSFWeigher`
    (see :func:`libera_utils.footprint_matching.product.build_radiometer_footprints`).
    This boresight-circle box remains the fallback for footprints with no scan reference
    (a minimal caller-built dict, or a record whose ray-trace runs off the limb).

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : float
        Footprint boresight centroid (L1B ``Latitude``/``Longitude``), degrees.
    viewing_zenith_deg : float
        Viewing zenith angle (L1B ``Viewing_Zenith_Surface``), degrees. Elongates the
        ground radius at oblique views.
    altitude_km : float or None, optional
        Satellite altitude above the surface, km. Defaults to
        :data:`NOMINAL_ALTITUDE_KM` when not supplied or non-positive.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle, degrees -- floor on the PSF extent.
    n_samples : int, optional
        Number of points sampled around the ground circle. Defaults to
        :data:`_N_PERIMETER_SAMPLES`.

    Returns
    -------
    BoundingBox
        Geographic box enclosing the boresight-centred PSF ground circle.
    """
    altitude = altitude_km if (altitude_km is not None and altitude_km > 0.0) else NOMINAL_ALTITUDE_KM
    # Apply the same outward safety margin the ray-traced box uses, so this box is an
    # equally-safe (slightly conservative) superset.
    radius_km = psf_ground_radius_km(altitude, viewing_zenith_deg, fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)

    geod = _wgs84_geod()
    bearings = np.linspace(0.0, 360.0, n_samples, endpoint=False)
    lon0 = np.full(n_samples, boresight_lon_deg, dtype=float)
    lat0 = np.full(n_samples, boresight_lat_deg, dtype=float)
    distances_m = np.full(n_samples, radius_km * 1000.0, dtype=float)
    # Geod.fwd walks the WGS84 geodesic from the boresight along each bearing, giving
    # the ground circle's perimeter points.
    perimeter_lons, perimeter_lats, _ = geod.fwd(lon0, lat0, bearings, distances_m)

    return bounding_box_from_points(boresight_lat_deg, boresight_lon_deg, list(perimeter_lats), list(perimeter_lons))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_footprint_bounding_box(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    viewing_zenith_deg: float,
    *,
    altitude_km: float | None = None,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    on_limb: str = "flag",
) -> BoundingBox:
    """Compute the lat/lon bounding box of one radiometer footprint on the surface.

    This is the public, per-footprint entry point. It chains: build the boresight
    line of sight and locate the satellite -> get the PSF angular extent -> rotate
    the boresight by that extent and ray-trace each ray onto the WGS84 ellipsoid ->
    assemble a lat/lon box with pole and dateline handling.

    Inputs come straight from the L1B fields that are actually populated:
    footprint ``Latitude``/``Longitude``, ``Subsatellite_Latitude``/``Longitude``,
    and ``Viewing_Zenith_Surface``. (The solar zenith and relative azimuth angles
    describe sun geometry and do not affect which ground patch the radiometer sees,
    so they are intentionally not parameters here.)

    The box is always centered on the reported footprint latitude/longitude; the
    viewing zenith angle and the bearing toward the subsatellite point set the
    footprint's size and orientation. When ``altitude_km`` is supplied it fixes the
    satellite range along the boresight; otherwise the range is recovered from the
    footprint/subsatellite geometry.

    TODO[LIBSDC-794]: this is scalar (one footprint per call) for clarity. To meet
    the real-time latency budget, vectorize the helpers over NumPy arrays of
    footprints; the math is all elementwise except the perimeter sampling, which can
    be batched.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : float
        Footprint centroid (L1B ``Latitude``/``Longitude``), degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : float
        Subsatellite point (L1B ``Subsatellite_Latitude``/``Longitude``), degrees.
        Sets the scan azimuth and, without an altitude field, the satellite range.
    viewing_zenith_deg : float
        Viewing zenith angle (L1B ``Viewing_Zenith_Surface``), degrees.
    altitude_km : float or None, optional
        Satellite altitude above the surface, km. Used if provided and positive;
        otherwise recovered from the positions + VZA. Default None.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle, degrees. Used as a floor on the PSF extent so the
        box is never smaller than the optical field of view. Defaults to
        :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.
    on_limb : {"flag", "raise"}, optional
        Behaviour when a box *corner* ray runs off the Earth limb at a severe angle
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
    OffLimbError
        If any input is an L1B fill value (a non-Earth view), or if the *centroid*
        viewing zenith angle is at or beyond 90 deg (on/past the limb). These mean
        there is no footprint at all, so they raise regardless of ``on_limb``.
    PartialFootprintError
        A subclass of :class:`OffLimbError`. Raised only when the boresight is on Earth
        but a corner ray of the box is off it (severe-angle truncation) *and*
        ``on_limb="raise"``.
    ValueError
        If ``on_limb`` is not ``"flag"`` or ``"raise"``.
    """
    if on_limb not in ("flag", "raise"):
        raise ValueError(f"on_limb must be 'flag' or 'raise', got {on_limb!r}")

    # Reject fill-valued / non-finite inputs: these are space or calibration views
    # with no Earth intersection. Treat them like an off-limb footprint.
    for value in (boresight_lat_deg, boresight_lon_deg, viewing_zenith_deg):
        if not math.isfinite(value) or value == _L1B_FILL_VALUE:
            raise OffLimbError("Footprint has fill/non-finite geolocation (non-Earth view).")

    # A viewing zenith angle at or beyond 90 deg means the boresight is on or past
    # the limb: there is no footprint at all.
    if viewing_zenith_deg >= 90.0:
        raise OffLimbError(f"Viewing zenith angle {viewing_zenith_deg:.3f} deg is at or beyond the limb (90 deg).")

    # 1. Build the boresight ray and locate the satellite in ECEF.
    satellite, boresight_direction, subsatellite_normal = _viewing_geometry(
        boresight_lat_deg,
        boresight_lon_deg,
        subsatellite_lat_deg,
        subsatellite_lon_deg,
        viewing_zenith_deg,
        altitude_km,
    )

    # 2. Get the PSF angular extent and apply the FOV floor. We use the larger of the
    #    dynamic 95%-energy extent and the static FOV so the box is never smaller than
    #    the optical field of view (the FOV also stands in for the stationary-scanner
    #    case, which we cannot currently detect without the cone-angle rate). The
    #    outward safety margin is applied to the angular extents.
    psf_extent = psf_95_energy_extent()
    along_extent_deg = max(conservative_along_scan_extent(psf_extent), fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)
    cross_extent_deg = max(psf_extent.beta_max_deg, fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)

    # 3. Rotate the boresight by the PSF angular extent and ray-trace each perimeter
    #    sample onto the ellipsoid. Rays that miss are clipped to the limb (flag) or
    #    raise PartialFootprintError (raise).
    cross_axis = _scan_frame_axes(boresight_direction, subsatellite_normal)
    lats: list[float] = []
    lons: list[float] = []
    truncated = False
    for t in np.linspace(0.0, 2.0 * math.pi, _N_PERIMETER_SAMPLES, endpoint=False):
        delta_deg = along_extent_deg * math.cos(t)
        beta_deg = cross_extent_deg * math.sin(t)
        lat, lon, missed = _perimeter_point(satellite, boresight_direction, cross_axis, delta_deg, beta_deg)
        if missed:
            if on_limb == "raise":
                raise PartialFootprintError(
                    "A bounding-box corner ray is off the Earth limb: part of the box is off the Earth. "
                    "The default on_limb='flag' truncates the box at the horizon and marks it as partial "
                    "coverage instead of raising."
                )
            truncated = True
        lats.append(lat)
        lons.append(lon)

    # 4. Assemble the lat/lon box (handles poles and dateline), carrying the
    #    partial-coverage truncation flag onto the returned box.
    return bounding_box_from_points(boresight_lat_deg, boresight_lon_deg, lats, lons, truncated=truncated)


# ---------------------------------------------------------------------------
# Vectorized bounding box: the whole pipeline batched over N footprints
# ---------------------------------------------------------------------------
#
# These functions reproduce compute_footprint_bounding_box exactly, but operate on
# arrays of N footprints so the per-footprint pyproj / rotation work is paid once per
# *segment* instead of once per footprint (design: doc/fmatch_vectorization_plan.md,
# Phase 1). The scalar path above is retained as the readable parity reference; the
# batch path is validated against it cell-for-cell in the unit tests. Only the
# on_limb="flag" semantics are implemented here (corner misses truncate); the "raise"
# behaviour stays on the scalar entry point.


def _pyproj_seq(array: np.ndarray) -> list[float] | np.ndarray:
    """Coerce a batch coordinate array into a form pyproj treats as a sequence.

    pyproj dispatches a length-1 NumPy array to its scalar-point code path, which then
    triggers a NumPy>=1.25 ``DeprecationWarning`` (ndim>0 -> scalar). Passing a Python
    list of length 1 keeps it on the array path; arrays of length >= 2 are returned
    unchanged (no copy). Callers normalize the returned coordinates with ``np.asarray``.
    """
    array = np.asarray(array, dtype=float)
    return array.tolist() if array.size == 1 else array


def _geodetic_to_ecef_surface(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Closed-form WGS84 geodetic -> ECEF (km) for surface points (ellipsoidal height 0).

    Numerically identical to the pyproj transform (agreement ~nanometres) but without its
    fixed per-call overhead, which dominates the per-footprint cell projection in
    :func:`project_to_angular`. Returns an ``(M, 3)`` array of ECEF positions in km.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    # Radius of curvature in the prime vertical, N(lat) = a / sqrt(1 - e^2 sin^2 lat).
    prime_vertical = WGS84_SEMI_MAJOR_AXIS_KM / np.sqrt(1.0 - WGS84_FIRST_ECCENTRICITY_SQ * sin_lat * sin_lat)
    x = prime_vertical * cos_lat * np.cos(lon)
    y = prime_vertical * cos_lat * np.sin(lon)
    z = prime_vertical * (1.0 - WGS84_FIRST_ECCENTRICITY_SQ) * sin_lat
    return np.stack([x, y, z], axis=1)


def _geodetic_to_ecef_batch(lat_deg: np.ndarray, lon_deg: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_geodetic_to_ecef`: ``(N,)`` geodetic -> ``(N, 3)`` ECEF km."""
    x, y, z = _geodetic_to_ecef_transformer().transform(
        _pyproj_seq(lon_deg), _pyproj_seq(lat_deg), _pyproj_seq(np.asarray(height_km, dtype=float) * 1000.0)
    )
    return np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=1) / 1000.0


def _ecef_to_geodetic_batch(xyz_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`_ecef_to_geodetic`: ``(K, 3)`` ECEF km -> ``(lat, lon, height_km)`` arrays."""
    xyz = np.asarray(xyz_km, dtype=float)
    lon, lat, height_m = _ecef_to_geodetic_transformer().transform(
        _pyproj_seq(xyz[:, 0] * 1000.0), _pyproj_seq(xyz[:, 1] * 1000.0), _pyproj_seq(xyz[:, 2] * 1000.0)
    )
    lon = (np.asarray(lon) + 540.0) % 360.0 - 180.0
    return np.asarray(lat), lon, np.asarray(height_m) / 1000.0


def _ellipsoid_normal_batch(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_ellipsoid_normal`: ``(N,)`` geodetic -> ``(N, 3)`` outward normals."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def _arbitrary_tangent_batch(normal: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_arbitrary_tangent`: a unit vector perpendicular to each ``normal``."""
    reference = np.where(
        (np.abs(normal[:, 2]) < 0.9)[:, None],
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    tangent = np.cross(normal, reference)
    return tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)


def _rotate_batch(vec: np.ndarray, axis: np.ndarray, angle_rad: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation (:func:`_rotate`) over broadcastable leading dims.

    ``vec`` and ``axis`` are ``(..., 3)``; ``angle_rad`` broadcasts against the leading
    dims. ``axis`` is normalized to unit length, matching the scalar helper.
    """
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    cos_a = np.cos(angle_rad)[..., None]
    sin_a = np.sin(angle_rad)[..., None]
    dot = np.sum(axis * vec, axis=-1, keepdims=True)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * dot * (1.0 - cos_a)


def _ray_ellipsoid_intersection_batch(origin_km: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`_ray_ellipsoid_intersection`.

    Parameters
    ----------
    origin_km, direction : np.ndarray
        ``(K, 3)`` ray origins (satellite) and directions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(points, hit)`` where ``points`` is ``(K, 3)`` (the nearest positive-ray
        intersection, garbage where ``hit`` is False) and ``hit`` is a ``(K,)`` bool
        mask marking rays that meet the ellipsoid at a positive ray parameter.
    """
    scale = np.array([1.0 / WGS84_SEMI_MAJOR_AXIS_KM, 1.0 / WGS84_SEMI_MAJOR_AXIS_KM, 1.0 / WGS84_SEMI_MINOR_AXIS_KM])
    o = origin_km * scale
    d = direction * scale
    a = np.sum(d * d, axis=-1)
    b = 2.0 * np.sum(o * d, axis=-1)
    c = np.sum(o * o, axis=-1) - 1.0
    discriminant = b * b - 4.0 * a * c
    real = discriminant >= 0.0
    sqrt_disc = np.sqrt(np.where(real, discriminant, 0.0))
    root_minus = (-b - sqrt_disc) / (2.0 * a)
    root_plus = (-b + sqrt_disc) / (2.0 * a)
    # Smallest strictly-positive root (matches min([s for s in roots if s > 0])).
    cand_minus = np.where(root_minus > 0.0, root_minus, np.inf)
    cand_plus = np.where(root_plus > 0.0, root_plus, np.inf)
    s = np.minimum(cand_minus, cand_plus)
    hit = real & np.isfinite(s)
    points = origin_km + np.where(hit, s, 0.0)[..., None] * direction
    return points, hit


def _slant_range_batch(
    to_satellite: np.ndarray, subsatellite_normal: np.ndarray, toward_subsat: np.ndarray
) -> np.ndarray:
    """Vectorized least-squares slant range ``rho`` (the ``altitude=None`` recovery).

    Solves, per footprint, the over-determined ``[to_satellite, -normal_sub] @ [rho, h] =
    toward_subsat`` via the 2x2 normal equations. For a full-rank system this equals the
    ``np.linalg.lstsq`` minimum-norm solution the scalar path uses.
    """
    matrix = np.stack([to_satellite, -subsatellite_normal], axis=2)  # (K, 3, 2)
    ata = np.einsum("kij,kil->kjl", matrix, matrix)  # (K, 2, 2)
    atb = np.einsum("kij,ki->kj", matrix, toward_subsat)  # (K, 2)
    solution = np.linalg.solve(ata, atb)  # (K, 2)
    # One step of iterative refinement recovers the accuracy the normal equations lose
    # relative to the scalar path's SVD lstsq when A is ill-conditioned (near-nadir, where
    # the boresight is nearly parallel to the subsatellite vertical). Cheap and keeps the
    # batch path bit-parity with the scalar box to well below tile granularity.
    residual = atb - np.einsum("kjl,kl->kj", ata, solution)
    solution = solution + np.linalg.solve(ata, residual)
    return solution[:, 0]


def _satellite_along_ray_batch(ground_km: np.ndarray, up_direction: np.ndarray, altitude_km: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_satellite_along_ray` (bisection to a target geodetic height).

    Same bracket-grow-then-bisect algorithm as the scalar helper, run in lock-step over
    every ray with the height evaluated through one batched ``_ecef_to_geodetic`` per
    iteration instead of one pyproj call per ray per iteration.
    """
    altitude = np.asarray(altitude_km, dtype=float)
    lo = np.zeros_like(altitude)
    hi = np.maximum(altitude, 1.0).astype(float)
    # Grow the upper bracket until the geodetic height overshoots the target (oblique
    # rays need a long slant range). 64 doublings comfortably reach the 1e6 km cap.
    for _ in range(64):
        heights = _ecef_to_geodetic_batch(ground_km + hi[:, None] * up_direction)[2]
        grow = (heights < altitude) & (hi < 1.0e6)
        if not np.any(grow):
            break
        hi = np.where(grow, hi * 2.0, hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        heights = _ecef_to_geodetic_batch(ground_km + mid[:, None] * up_direction)[2]
        below = heights < altitude
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return ground_km + hi[:, None] * up_direction


def _locate_satellite_batch(
    ground: np.ndarray,
    to_satellite: np.ndarray,
    subsatellite_normal: np.ndarray,
    toward_subsat: np.ndarray,
    near_nadir: np.ndarray,
    altitude_km: float | np.ndarray | None,
) -> np.ndarray:
    """Vectorized satellite location: the three branches of :func:`_viewing_geometry`."""
    n = ground.shape[0]
    if altitude_km is not None and np.ndim(altitude_km) > 0:
        # Per-footprint altitude array (the weighting frame path). Non-positive entries
        # fall back to the nominal orbit altitude, matching the weigher's own guard.
        alt = np.asarray(altitude_km, dtype=float)
        alt = np.where(alt > 0.0, alt, NOMINAL_ALTITUDE_KM)
        return _satellite_along_ray_batch(ground, to_satellite, alt)
    if altitude_km is not None and altitude_km > 0.0:
        return _satellite_along_ray_batch(ground, to_satellite, np.full(n, float(altitude_km)))

    # altitude None -> recover the slant range from the footprint/subsatellite geometry.
    # Only the non-nadir systems are full rank; near-nadir ones are degenerate and take
    # the bisection fallback (rho left 0 keeps them in use_bisect below).
    rho = np.zeros(n)
    non_nadir = ~near_nadir
    if np.any(non_nadir):
        rho[non_nadir] = _slant_range_batch(
            to_satellite[non_nadir], subsatellite_normal[non_nadir], toward_subsat[non_nadir]
        )
    use_bisect = near_nadir | (rho <= 0.0)
    satellite = ground + rho[:, None] * to_satellite
    if np.any(use_bisect):
        idx = np.nonzero(use_bisect)[0]
        satellite[idx] = _satellite_along_ray_batch(
            ground[idx], to_satellite[idx], np.full(idx.size, NOMINAL_ALTITUDE_KM)
        )
    return satellite


def _viewing_geometry_batch(
    lat: np.ndarray,
    lon: np.ndarray,
    sub_lat: np.ndarray,
    sub_lon: np.ndarray,
    vza: np.ndarray,
    altitude_km: float | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`_viewing_geometry` over ``N`` footprints.

    Returns ``(satellite (N,3), boresight_direction (N,3), subsatellite_normal (N,3))``.
    """
    ground = _geodetic_to_ecef_batch(lat, lon, np.zeros_like(lat))
    normal_p = _ellipsoid_normal_batch(lat, lon)
    subsat_ground = _geodetic_to_ecef_batch(sub_lat, sub_lon, np.zeros_like(sub_lat))
    normal_sub = _ellipsoid_normal_batch(sub_lat, sub_lon)

    toward_subsat = subsat_ground - ground
    horizontal = toward_subsat - np.sum(toward_subsat * normal_p, axis=1, keepdims=True) * normal_p
    horizontal_norm = np.linalg.norm(horizontal, axis=1)
    near_nadir = (vza < _NADIR_CONE_ANGLE_EPS_DEG) | (horizontal_norm < 1.0e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        horizontal_unit_raw = horizontal / horizontal_norm[:, None]
    horizontal_unit = np.where(near_nadir[:, None], _arbitrary_tangent_batch(normal_p), horizontal_unit_raw)

    theta = np.radians(vza)
    to_satellite = np.cos(theta)[:, None] * normal_p + np.sin(theta)[:, None] * horizontal_unit
    boresight_direction = -to_satellite

    satellite = _locate_satellite_batch(ground, to_satellite, normal_sub, toward_subsat, near_nadir, altitude_km)
    return satellite, boresight_direction, normal_sub


def _scan_frame_axes_batch(boresight_direction: np.ndarray, subsatellite_normal: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_scan_frame_axes`: the ``(N, 3)`` cross-scan rotation axes."""
    nadir = -subsatellite_normal
    cross_axis = np.cross(nadir, boresight_direction)
    norm = np.linalg.norm(cross_axis, axis=1)
    degenerate = norm < 1.0e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = cross_axis / norm[:, None]
    return np.where(degenerate[:, None], _arbitrary_tangent_batch(boresight_direction), unit)


def _offset_ray_direction_batch(
    boresight_direction: np.ndarray, cross_axis: np.ndarray, delta_deg: np.ndarray, beta_deg: np.ndarray
) -> np.ndarray:
    """Vectorized :func:`_offset_ray_direction` (along-scan then cross-scan rotation)."""
    along = _rotate_batch(boresight_direction, cross_axis, np.radians(delta_deg))
    inplane_axis = np.cross(cross_axis, along)
    return _rotate_batch(along, inplane_axis, np.radians(beta_deg))


def _grazing_points_batch(
    satellite: np.ndarray,
    boresight_direction: np.ndarray,
    cross_axis: np.ndarray,
    delta_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> np.ndarray:
    """Vectorized limb-clip: the miss-bisection branch of :func:`_perimeter_point`.

    For each ray that missed the ellipsoid at its full extent, bisects the angular
    fraction back toward the boresight until it just grazes the limb, returning the
    grazing ground intersection.
    """
    k = satellite.shape[0]
    lo = np.zeros(k)
    hi = np.ones(k)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        probe = _offset_ray_direction_batch(boresight_direction, cross_axis, delta_deg * mid, beta_deg * mid)
        _, hit = _ray_ellipsoid_intersection_batch(satellite, probe)
        lo = np.where(hit, mid, lo)
        hi = np.where(hit, hi, mid)
    grazing = _offset_ray_direction_batch(boresight_direction, cross_axis, delta_deg * lo, beta_deg * lo)
    points, _ = _ray_ellipsoid_intersection_batch(satellite, grazing)
    return points


def bounding_box_from_points_batch(
    center_lat_deg: np.ndarray,
    center_lon_deg: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    truncated: np.ndarray,
) -> list[BoundingBox]:
    """Vectorized :func:`bounding_box_from_points` over ``N`` footprints.

    ``center_lat_deg``/``center_lon_deg`` are ``(N,)``; ``lats``/``lons`` are ``(N, S)``
    perimeter samples; ``truncated`` is ``(N,)``. Returns one :class:`BoundingBox` per
    footprint, with identical pole-enclosure, dateline, and polar-advisory handling to
    the scalar assembler.
    """
    geod = _wgs84_geod()
    n, s = lats.shape

    # Pole-enclosure reach: geodesic distance from each centre to every perimeter point.
    center_lat_rep = np.repeat(center_lat_deg, s)
    center_lon_rep = np.repeat(center_lon_deg, s)
    _, _, reach = geod.inv(
        _pyproj_seq(center_lon_rep),
        _pyproj_seq(center_lat_rep),
        _pyproj_seq(lons.reshape(-1)),
        _pyproj_seq(lats.reshape(-1)),
    )
    max_reach = np.asarray(reach, dtype=float).reshape(n, s).max(axis=1)
    north = np.full(n, 90.0)
    south = np.full(n, -90.0)
    _, _, dist_north = geod.inv(
        _pyproj_seq(center_lon_deg), _pyproj_seq(center_lat_deg), _pyproj_seq(center_lon_deg), _pyproj_seq(north)
    )
    _, _, dist_south = geod.inv(
        _pyproj_seq(center_lon_deg), _pyproj_seq(center_lat_deg), _pyproj_seq(center_lon_deg), _pyproj_seq(south)
    )
    encloses_north = np.asarray(dist_north, dtype=float) <= max_reach
    encloses_south = np.asarray(dist_south, dtype=float) <= max_reach

    lat_min = lats.min(axis=1)
    lat_max = lats.max(axis=1)

    # Dateline: choose the [-180,180) or [0,360) representation with the smaller span.
    span_signed = lons.max(axis=1) - lons.min(axis=1)
    lons_360 = lons % 360.0
    span_360 = lons_360.max(axis=1) - lons_360.min(axis=1)
    use_360 = span_360 < span_signed
    lon_min = np.where(use_360, lons_360.min(axis=1), lons.min(axis=1))
    lon_max = np.where(use_360, lons_360.max(axis=1), lons.max(axis=1))
    wraps = use_360 & (lon_max > 180.0)

    is_polar = (np.abs(lat_min) >= POLAR_LATITUDE_THRESHOLD_DEG) | (np.abs(lat_max) >= POLAR_LATITUDE_THRESHOLD_DEG)
    truncated = np.asarray(truncated, dtype=bool)

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


def compute_footprint_bounding_boxes(
    boresight_lat_deg: np.ndarray,
    boresight_lon_deg: np.ndarray,
    subsatellite_lat_deg: np.ndarray,
    subsatellite_lon_deg: np.ndarray,
    viewing_zenith_deg: np.ndarray,
    *,
    altitude_km: float | None = None,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    n_samples: int = _N_PERIMETER_SAMPLES,
) -> list[BoundingBox | None]:
    """Batched :func:`compute_footprint_bounding_box` over ``N`` footprints.

    Vectorizes the entire scalar pipeline (viewing geometry, PSF perimeter ray-trace,
    box assembly) over arrays of ``N`` footprints so the per-footprint pyproj / rotation
    work is amortized across the whole segment. Numerically equivalent to calling the
    scalar entry point per footprint with ``on_limb="flag"`` (validated in the tests).

    Off-limb footprints -- those the scalar path raises :class:`OffLimbError` for (a fill
    or non-finite ``latitude``/``longitude``/``viewing_zenith``, or a centroid viewing
    zenith at/beyond 90 deg) -- are returned as ``None`` at their index rather than
    raising, so the caller can substitute a fallback box per footprint (see
    :func:`libera_utils.footprint_matching.product.build_radiometer_footprints`). Corner
    rays that run off the limb at a severe angle are truncated at the horizon and the
    box's ``truncated`` flag is set, exactly like ``on_limb="flag"``.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : np.ndarray
        Per-footprint boresight centroids (L1B ``Latitude``/``Longitude``), degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : np.ndarray
        Per-footprint subsatellite points, degrees.
    viewing_zenith_deg : np.ndarray
        Per-footprint viewing zenith angles, degrees.
    altitude_km : float or None, optional
        Satellite altitude above the surface, km. Used for every footprint when supplied
        and positive; otherwise recovered per footprint from the geometry.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle floor on the PSF extent, degrees.
    n_samples : int, optional
        Number of PSF-perimeter samples. Defaults to :data:`_N_PERIMETER_SAMPLES`.

    Returns
    -------
    list[BoundingBox | None]
        One entry per input footprint, in input order. ``None`` marks an off-limb
        centroid (the scalar :class:`OffLimbError` case).
    """
    lat = np.asarray(boresight_lat_deg, dtype=float)
    lon = np.asarray(boresight_lon_deg, dtype=float)
    sub_lat = np.asarray(subsatellite_lat_deg, dtype=float)
    sub_lon = np.asarray(subsatellite_lon_deg, dtype=float)
    vza = np.asarray(viewing_zenith_deg, dtype=float)
    n_total = lat.size

    # Off-limb centroids are exactly the scalar OffLimbError conditions: a fill / non-
    # finite lat/lon/vza, or a viewing zenith at or beyond the limb (90 deg).
    finite = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(vza)
    not_fill = (lat != _L1B_FILL_VALUE) & (lon != _L1B_FILL_VALUE) & (vza != _L1B_FILL_VALUE)
    on_earth = vza < 90.0
    valid = finite & not_fill & on_earth

    boxes: list[BoundingBox | None] = [None] * n_total
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return boxes

    la, lo_, sla, slo, vz = lat[idx], lon[idx], sub_lat[idx], sub_lon[idx], vza[idx]
    m = idx.size

    satellite, boresight_direction, subsatellite_normal = _viewing_geometry_batch(la, lo_, sla, slo, vz, altitude_km)

    # PSF angular extents: static across footprints (same as the scalar path), with the
    # FOV floor and outward safety margin applied.
    psf_extent = psf_95_energy_extent()
    along_extent_deg = max(conservative_along_scan_extent(psf_extent), fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)
    cross_extent_deg = max(psf_extent.beta_max_deg, fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)

    cross_axis = _scan_frame_axes_batch(boresight_direction, subsatellite_normal)

    t = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    delta_deg = along_extent_deg * np.cos(t)  # (S,)
    beta_deg = cross_extent_deg * np.sin(t)  # (S,)

    # (M, S, 3) perimeter ray directions and their ellipsoid intersections.
    directions = _offset_ray_direction_batch(
        boresight_direction[:, None, :], cross_axis[:, None, :], delta_deg[None, :], beta_deg[None, :]
    )
    origin = np.broadcast_to(satellite[:, None, :], directions.shape)
    points, hit = _ray_ellipsoid_intersection_batch(origin.reshape(-1, 3), directions.reshape(-1, 3))
    hit = hit.reshape(m, n_samples)
    points = points.reshape(m, n_samples, 3)
    missed = ~hit

    # Limb-clip the rays that missed (their box corner is off the Earth): bisect each to
    # its grazing direction. Handled as one flattened sub-batch of the missed entries.
    if np.any(missed):
        missed_flat = missed.reshape(-1)
        sel = np.nonzero(missed_flat)[0]
        bore_full = np.broadcast_to(boresight_direction[:, None, :], (m, n_samples, 3)).reshape(-1, 3)[sel]
        cross_full = np.broadcast_to(cross_axis[:, None, :], (m, n_samples, 3)).reshape(-1, 3)[sel]
        sat_full = np.broadcast_to(satellite[:, None, :], (m, n_samples, 3)).reshape(-1, 3)[sel]
        delta_full = np.broadcast_to(delta_deg[None, :], (m, n_samples)).reshape(-1)[sel]
        beta_full = np.broadcast_to(beta_deg[None, :], (m, n_samples)).reshape(-1)[sel]
        grazing = _grazing_points_batch(sat_full, bore_full, cross_full, delta_full, beta_full)
        flat_points = points.reshape(-1, 3)
        flat_points[sel] = grazing
        points = flat_points.reshape(m, n_samples, 3)

    perimeter_lat, perimeter_lon, _ = _ecef_to_geodetic_batch(points.reshape(-1, 3))
    perimeter_lat = perimeter_lat.reshape(m, n_samples)
    perimeter_lon = perimeter_lon.reshape(m, n_samples)
    truncated = np.any(missed, axis=1)

    sub_boxes = bounding_box_from_points_batch(la, lo_, perimeter_lat, perimeter_lon, truncated)
    for k, i in enumerate(idx):
        boxes[int(i)] = sub_boxes[k]
    return boxes


# ---------------------------------------------------------------------------
# Shared viewing frame: computed once per footprint, reused by the PSF weigher
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewingFrame:
    """Per-footprint radiometer viewing frame, batched over ``N`` footprints.

    The angular-frame PSF projection (:func:`project_to_angular`) needs, per footprint,
    the satellite ECEF position and the three orthonormal look axes ``(b_hat, c_hat,
    n_hat)``. Building them repeats the (bisection-heavy) :func:`_viewing_geometry`, so
    the weighting path recomputes it once per footprint. This struct holds all ``N``
    frames computed together (:func:`compute_viewing_frames`) so the orchestrator can
    build them in one batched call and hand each footprint its slice via
    :meth:`for_index`, keeping the per-footprint projection frame-free.

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

    def for_index(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the single-footprint ``(satellite, b_hat, c_hat, n_hat)`` at index ``i``."""
        return self.satellite[i], self.b_hat[i], self.c_hat[i], self.n_hat[i]


def compute_viewing_frames(
    boresight_lat_deg: np.ndarray,
    boresight_lon_deg: np.ndarray,
    subsatellite_lat_deg: np.ndarray,
    subsatellite_lon_deg: np.ndarray,
    viewing_zenith_deg: np.ndarray,
    altitude_km: float | np.ndarray,
) -> ViewingFrame:
    """Build every footprint's :class:`ViewingFrame` in one batched call.

    Vectorized companion to the frame construction inside :func:`project_to_angular`.
    Reproduces it exactly (same viewing geometry, same orthonormal basis) so a frame
    fed back into :func:`project_to_angular` yields identical ``(delta, beta)`` to the
    inline path -- validated in the tests. Unlike the bounding-box batch, the altitude
    is always supplied here (the weigher passes the nominal orbit altitude), so the
    satellite is located by the along-ray bisection, batched across all footprints.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : np.ndarray
        Per-footprint boresight centroids, degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : np.ndarray
        Per-footprint subsatellite points, degrees. (Callers substitute the boresight
        when the L1B subsatellite point is unavailable, matching the weigher's fallback.)
    viewing_zenith_deg : np.ndarray
        Per-footprint viewing zenith angles, degrees.
    altitude_km : float or np.ndarray
        Satellite altitude(s) above the surface, km (scalar or per-footprint).

    Returns
    -------
    ViewingFrame
        The ``N`` batched frames.
    """
    lat = np.asarray(boresight_lat_deg, dtype=float)
    lon = np.asarray(boresight_lon_deg, dtype=float)
    sub_lat = np.asarray(subsatellite_lat_deg, dtype=float)
    sub_lon = np.asarray(subsatellite_lon_deg, dtype=float)
    vza = np.asarray(viewing_zenith_deg, dtype=float)

    satellite, boresight_direction, subsatellite_normal = _viewing_geometry_batch(
        lat, lon, sub_lat, sub_lon, vza, altitude_km
    )
    b_hat = boresight_direction / np.linalg.norm(boresight_direction, axis=1, keepdims=True)
    c_hat = _scan_frame_axes_batch(boresight_direction, subsatellite_normal)
    c_hat = c_hat / np.linalg.norm(c_hat, axis=1, keepdims=True)
    n_hat = np.cross(c_hat, b_hat)
    return ViewingFrame(satellite=satellite, b_hat=b_hat, c_hat=c_hat, n_hat=n_hat)


# ---------------------------------------------------------------------------
# Inverse projection: ground lat/lon -> radiometer angular frame (delta, beta)
# ---------------------------------------------------------------------------


def project_to_angular(
    cell_lats_deg: np.ndarray,
    cell_lons_deg: np.ndarray,
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    viewing_zenith_deg: float,
    *,
    altitude_km: float | None = None,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project ground grid cells into the radiometer's angular frame ``(delta, beta)``.

    This is the **inverse** of the forward rotation :func:`_offset_ray_direction`
    used to build the footprint bounding box, and the piece the PSF aggregation
    weighting needs: given where each ancillary grid cell sits on the ground, how far
    off the boresight is it in the along-scan (``delta``) and cross-scan (``beta``)
    angular directions as *seen from the satellite*. Those angles are exactly the
    coordinates the CERES PSF :func:`libera_utils.footprint_matching.psf.psf_weight`
    is a function of (CERES ATBD v2.2 §4.4.2.3, Eq. 4.4-8).

    Method
    ------
    We reconstruct the same satellite viewing frame the bounding-box code builds
    (:func:`_viewing_geometry` + :func:`_scan_frame_axes`), giving three orthonormal
    axes:

    * ``b_hat`` -- the boresight look direction (``delta = beta = 0``),
    * ``c_hat`` -- the cross-scan axis (perpendicular to the scan plane),
    * ``n_hat = c_hat x b_hat`` -- the in-scan-plane axis (``b_hat``'s ``+delta`` side).

    For each cell we form the unit look vector from the satellite to the cell and read
    off the angles by projecting onto that basis. Because
    :func:`_offset_ray_direction` produces a direction
    ``d = (b_hat cos(delta) + n_hat sin(delta)) cos(beta) + c_hat sin(beta)``, the
    inverse is exact::

        beta  = asin(look . c_hat)
        delta = atan2(look . n_hat, look . b_hat)

    Since the elongation of the footprint at oblique viewing is captured by projecting
    the *actual* ground cells to satellite angles, no separate viewing-zenith stretch
    factor is needed (unlike the radial stand-in).

    Parameters
    ----------
    cell_lats_deg, cell_lons_deg : np.ndarray
        Grid-cell latitudes and longitudes in degrees (any common shape). Assumed on
        the surface (ellipsoidal height 0).
    boresight_lat_deg, boresight_lon_deg : float
        Footprint boresight centroid, degrees (``delta = beta = 0`` anchor).
    subsatellite_lat_deg, subsatellite_lon_deg : float
        Subsatellite point, degrees -- sets the scan azimuth / plane orientation.
    viewing_zenith_deg : float
        Viewing zenith angle at the boresight, degrees.
    altitude_km : float, optional
        Satellite altitude above the surface, km. Recovered from the geometry when
        not supplied (see :func:`_viewing_geometry`). Ignored when ``frame`` is given.
    frame : tuple of np.ndarray, optional
        A precomputed ``(satellite, b_hat, c_hat, n_hat)`` viewing frame for this
        footprint (each a length-3 array), as produced per footprint by
        :func:`compute_viewing_frames` / :meth:`ViewingFrame.for_index`. When supplied,
        the per-footprint :func:`_viewing_geometry` reconstruction (and its satellite
        bisection) is skipped and the frame is used directly -- the boresight /
        subsatellite / viewing-zenith / altitude arguments are then unused. Passing the
        frame the orchestrator already batched avoids recomputing it once per footprint.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(delta_deg, beta_deg)`` arrays, each the shape of ``cell_lats_deg``.
    """
    lats = np.asarray(cell_lats_deg, dtype=float)
    lons = np.asarray(cell_lons_deg, dtype=float)
    shape = lats.shape

    if frame is not None:
        # Use the frame the orchestrator batched (identical to the inline construction).
        satellite, b_hat, c_hat, n_hat = frame
    else:
        # Rebuild the satellite viewing frame for this footprint (same construction the
        # bounding-box path uses, so the projection is consistent with it).
        satellite, boresight_direction, subsatellite_normal = _viewing_geometry(
            boresight_lat_deg,
            boresight_lon_deg,
            subsatellite_lat_deg,
            subsatellite_lon_deg,
            viewing_zenith_deg,
            altitude_km,
        )
        b_hat = _unit(boresight_direction)
        c_hat = _unit(_scan_frame_axes(boresight_direction, subsatellite_normal))
        # In-plane axis: unit because c_hat is perpendicular to b_hat (cross-scan axis is
        # built as a cross product with the boresight), and both are unit vectors.
        n_hat = np.cross(c_hat, b_hat)

    # Geodetic -> ECEF (km) for every cell centre. Uses the closed-form surface transform
    # (cells sit on the ellipsoid) rather than pyproj: this runs once per footprint, so the
    # pyproj per-call overhead would otherwise dominate the weighting path.
    cells_km = _geodetic_to_ecef_surface(lats.ravel(), lons.ravel())  # (M, 3)

    # Unit look vector from the satellite toward each cell.
    look = cells_km - satellite
    look /= np.linalg.norm(look, axis=1, keepdims=True)

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
