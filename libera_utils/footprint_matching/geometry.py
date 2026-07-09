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

# Backwards-compatible private alias. The value was originally module-private; keep
# the old name working for any existing importers while the public name is adopted.
_L1B_FILL_VALUE: float = L1B_FILL_VALUE

# Number of samples around the PSF angular-extent ellipse perimeter. 72 == every 5 deg.
_N_PERIMETER_SAMPLES: int = 72


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
