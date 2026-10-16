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

import numpy as np
from curryer.compute import spatial
from curryer.compute.constants import WGS84_SEMI_MAJOR_AXIS_KM as WGS84_SEMI_MAJOR_AXIS_KM
from curryer.compute.constants import WGS84_SEMI_MINOR_AXIS_KM as WGS84_SEMI_MINOR_AXIS_KM
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

# Private alias of the public fill-value constant, used internally below.
_L1B_FILL_VALUE: float = L1B_FILL_VALUE

# A box that does not cross 0 or 180 deg has the same longitude span in [-180, 180) and
# [0, 360); the wrapped representation must be smaller by more than this before it is
# chosen, so float rounding in ``lon % 360`` cannot flag such a box as dateline-wrapping.
_LON_SPAN_TIE_TOL_DEG: float = 1e-9

# Number of samples around the PSF angular-extent rectangle perimeter. 72 == every 5 deg;
# a multiple of 8 so that samples land exactly on the rectangle's four corners.
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
    line of sight is tangent to / past the limb), or when the boresight latitude,
    longitude or viewing zenith is missing while the spacecraft geolocation is present:
    L1B RAD writes NaN there for a space or calibration view, where the boresight ray
    misses the ellipsoid. The orchestrator is expected to catch this and flag/discard
    the footprint rather than silently substituting data.
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


# ---------------------------------------------------------------------------
# Vector / coordinate helpers
# ---------------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return ``vec`` scaled to unit length."""
    return vec / np.linalg.norm(vec)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, height_km: float) -> np.ndarray:
    """Geodetic (lat, lon, ellipsoidal height) -> ECEF position vector, in km (curryer)."""
    return spatial.geodetic_to_ecef(np.array([lon_deg, lat_deg, height_km], dtype=float))


def _ecef_to_geodetic(xyz_km: np.ndarray) -> tuple[float, float, float]:
    """ECEF position (km) -> geodetic ``(lat_deg, lon_deg, height_km)`` (curryer).

    Longitude is in [-180, 180].
    """
    lon, lat, height_km = spatial.ecef_to_geodetic(np.asarray(xyz_km, dtype=float))
    return float(lat), float(lon), float(height_km)


def _ellipsoid_normal(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Outward unit normal (local geodetic zenith) at a surface point.

    This is the geodetic vertical, ``(cos lat cos lon, cos lat sin lon, sin lat)`` --
    *not* the geocentric radial direction. Honoring this difference is the whole
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

    Wraps :func:`curryer.compute.spatial.ray_intersect_ellipsoid`, which marks a miss
    with NaN, as the ``None`` the scalar path tests for.

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
    point = spatial.ray_intersect_ellipsoid(np.asarray(direction, dtype=float), np.asarray(origin_km, dtype=float))
    return None if not np.all(np.isfinite(point)) else point


# ---------------------------------------------------------------------------
# Viewing geometry: anchor the boresight at the footprint and locate the satellite
# ---------------------------------------------------------------------------


def _viewing_geometry(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    satellite_radius_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate the satellite in ECEF and build the boresight line of sight to the footprint.

    The satellite is placed exactly from the L1B ephemeris fields
    (:func:`_satellite_from_subpoint_batch`), and the boresight is the unit vector from
    it to the reported footprint ground point. No viewing-zenith input or altitude
    recovery is needed, and nadir is not a special case.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(satellite_ecef_km, boresight_direction)``; the direction points from the
        satellite toward the ground.

    Raises
    ------
    MissingGeolocationError
        If the footprint is not visible from the satellite (beyond its horizon).
    """
    satellite, boresight_direction = _viewing_geometry_batch(
        np.array([boresight_lat_deg], dtype=float),
        np.array([boresight_lon_deg], dtype=float),
        np.array([subsatellite_lat_deg], dtype=float),
        np.array([subsatellite_lon_deg], dtype=float),
        np.array([satellite_radius_km], dtype=float),
    )
    return satellite[0], boresight_direction[0]


# ---------------------------------------------------------------------------
# PSF perimeter ray-trace
# ---------------------------------------------------------------------------


def _scan_frame_axes(boresight_direction: np.ndarray, satellite_km: np.ndarray) -> np.ndarray:
    """Cross-scan rotation axis (perpendicular to the scan plane).

    The scan plane contains the boresight and the satellite's geocentric nadir
    (``-satellite / |satellite|``), the direction curryer measures the cone angle from, so
    the along-scan axis matches the plane L1B ``Cone_Angle``/``Cone_Angle_Rate`` are
    defined in. (The geodetic nadir differs from it by up to ~0.2 deg.) Rotating
    the boresight about this axis slides the look-point along the scan (a cone-angle
    perturbation); rotating about an axis in the plane tilts it cross-scan.

    Returns
    -------
    np.ndarray
        Unit cross-scan rotation axis.
    """
    nadir = -_unit(satellite_km)
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


def _perimeter_angles(
    along_extent_deg: float, cross_extent_deg: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Along-scan (delta) and cross-scan (beta) offsets tracing the PSF extent rectangle.

    The PSF energy contour is bounded by its delta and beta half-extents, i.e. by a
    rectangle in the angular frame, and its corners reach into that rectangle's corners.
    Sampling the inscribed ellipse instead would cut those corners off, so a scan azimuth
    rotated ~30-45 deg from north would leave part of the footprint outside the lat/lon box.
    Each unit-circle direction is pushed out to the unit square, then scaled by the extents.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(delta_deg, beta_deg)``, each of shape ``(n_samples,)``.
    """
    t = np.linspace(0.0, 2.0 * math.pi, n_samples, endpoint=False)
    cos_t, sin_t = np.cos(t), np.sin(t)
    to_square = 1.0 / np.maximum(np.abs(cos_t), np.abs(sin_t))
    return along_extent_deg * cos_t * to_square, cross_extent_deg * sin_t * to_square


def _usable(lat_deg: float | np.ndarray, *others: float | np.ndarray) -> np.ndarray:
    """True where a latitude and its companion values are usable: finite, not the L1B fill, |lat| <= 90.

    xarray decodes ``_FillValue`` to NaN on read, so NaN is the main signal; the explicit
    :data:`L1B_FILL_VALUE` check covers raw reads, and the latitude range check catches
    other fill conventions (e.g. the NetCDF default fill).
    """
    lat = np.asarray(lat_deg, dtype=float)
    values = [lat, *(np.asarray(v, dtype=float) for v in others)]
    valid = np.ones(np.broadcast(*values).shape, dtype=bool)
    for v in values:
        valid &= np.isfinite(v) & (v != _L1B_FILL_VALUE)
    return valid & (np.abs(lat) <= 90.0)


def _valid_boresight(boresight_lat_deg: float | np.ndarray, boresight_lon_deg: float | np.ndarray) -> np.ndarray:
    """True where the boresight geolocation is usable (missing there means an off-Earth view)."""
    return _usable(boresight_lat_deg, boresight_lon_deg)


def _valid_spacecraft(
    subsatellite_lat_deg: float | np.ndarray,
    subsatellite_lon_deg: float | np.ndarray,
    satellite_radius_km: float | np.ndarray,
) -> np.ndarray:
    """True where the ephemeris fields are usable (missing there means no geolocation at all).

    The radius must put the satellite above the ellipsoid (greater than the semi-major
    axis), which also rejects its -9999 fill and any non-physical value.
    """
    radius = np.asarray(satellite_radius_km, dtype=float)
    return (
        _usable(subsatellite_lat_deg, subsatellite_lon_deg) & np.isfinite(radius) & (radius > WGS84_SEMI_MAJOR_AXIS_KM)
    )


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

    if span_360 < span_signed - _LON_SPAN_TIE_TOL_DEG:
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
    spacecraft_altitude_km: float,
    viewing_zenith_deg: float,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
) -> float:
    """Project the PSF angular half-extent to an isotropic ground radius, in km.

    A small angular offset ``theta`` at the satellite spans ``rho * tan(theta)``
    perpendicular to the line of sight, where ``rho`` is the slant range. On a spherical
    Earth of radius ``R``, a satellite at altitude ``h`` sees a point at viewing zenith
    ``v`` from ``rho = sqrt((R + h)^2 - (R sin v)^2) - R cos v``. Along the view
    direction the line of sight strikes the surface obliquely, stretching that span by
    a further ``1/cos(v)``; across it there is no stretch. We return the along-view
    (larger) value so the radius bounds the footprint in every direction. At nadir this
    reduces to ``h * tan(theta)``.

    The angular half-extent is the diagonal of the PSF's along-scan and cross-scan
    95%-energy half-extents, because the energy contour reaches toward the corners of
    that rectangle (see :func:`_perimeter_angles`). It is floored at the optical FOV
    half-angle so the radius is never smaller than the field of view.

    This is the single home for the angular-extent -> ground-radius projection, used by
    the per-cell PSF weighers (:mod:`libera_utils.footprint_matching.weighting`).

    Parameters
    ----------
    spacecraft_altitude_km : float
        Spacecraft geodetic height above the ellipsoid, km. Not L1B ``Altitude``, which
        is the height of the boresight's ellipsoid intersection (~0).
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
        (earth_radius_km + spacecraft_altitude_km) ** 2 - (earth_radius_km * math.sin(vza_rad)) ** 2
    ) - earth_radius_km * math.cos(vza_rad)
    return slant_range_km * math.tan(math.radians(angular_deg)) / math.cos(vza_rad)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_footprint_bounding_box(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    *,
    satellite_radius_km: float,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    on_limb: str = "flag",
) -> BoundingBox:
    """Compute the lat/lon bounding box of one radiometer footprint on the surface.

    This is the public, per-footprint entry point. It chains: build the boresight
    line of sight and locate the satellite -> get the PSF angular extent -> rotate
    the boresight by that extent and ray-trace each ray onto the WGS84 ellipsoid ->
    assemble a lat/lon box with pole and dateline handling.

    Inputs come straight from L1B RAD fields: footprint ``Latitude``/``Longitude``,
    ``Subsatellite_Latitude``/``Longitude`` and
    ``Radius_of_Satellite_from_Center_of_Earth``. The last three place the satellite
    exactly; the line from it to the footprint is the boresight, so the viewing zenith
    angle and the scan orientation follow from the geometry rather than being inputs.
    (The solar angles describe sun geometry and do not affect which ground patch the
    radiometer sees.)

    TODO[LIBSDC-794]: this is scalar (one footprint per call) for clarity. To meet
    the real-time latency budget, vectorize the helpers over NumPy arrays of
    footprints; the math is all elementwise except the perimeter sampling, which can
    be batched.

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
        Instrument FOV half-angle, degrees. Used as a floor on the PSF extent so the
        box is never smaller than the optical field of view. Defaults to
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

    # The ephemeris fields are populated for every record, so missing values there mean
    # no geolocation at all. A missing boresight with the spacecraft present is a
    # space/calibration view: no Earth intersection.
    if not _valid_spacecraft(subsatellite_lat_deg, subsatellite_lon_deg, satellite_radius_km):
        raise MissingGeolocationError("Footprint has a fill/non-finite/out-of-range subsatellite point or radius.")
    if not _valid_boresight(boresight_lat_deg, boresight_lon_deg):
        raise OffLimbError("Footprint boresight has no Earth intersection (fill/non-finite geolocation).")

    # 1. Locate the satellite in ECEF and build the boresight ray to the footprint.
    satellite, boresight_direction = _viewing_geometry(
        boresight_lat_deg, boresight_lon_deg, subsatellite_lat_deg, subsatellite_lon_deg, satellite_radius_km
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
    cross_axis = _scan_frame_axes(boresight_direction, satellite)
    lats: list[float] = []
    lons: list[float] = []
    truncated = False
    for delta_deg, beta_deg in zip(
        *_perimeter_angles(along_extent_deg, cross_extent_deg, _N_PERIMETER_SAMPLES), strict=True
    ):
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
# behavior stays on the scalar entry point.


def _pyproj_seq(array: np.ndarray) -> list[float] | np.ndarray:
    """Coerce a batch coordinate array into a form pyproj treats as a sequence.

    pyproj dispatches a length-1 NumPy array to its scalar-point code path, which then
    triggers a NumPy>=1.25 ``DeprecationWarning`` (ndim>0 -> scalar). Passing a Python
    list of length 1 keeps it on the array path; arrays of length >= 2 are returned
    unchanged (no copy). Callers normalize the returned coordinates with ``np.asarray``.
    """
    array = np.asarray(array, dtype=float)
    return array.tolist() if array.size == 1 else array


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
    return _geodetic_to_ecef_batch(lats, lons, np.zeros_like(lats))


def _geodetic_to_ecef_batch(lat_deg: np.ndarray, lon_deg: np.ndarray, height_km: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_geodetic_to_ecef`: ``(N,)`` geodetic -> ``(N, 3)`` ECEF km."""
    lon_lat_alt = np.stack(
        [np.asarray(lon_deg, dtype=float), np.asarray(lat_deg, dtype=float), np.asarray(height_km, dtype=float)],
        axis=1,
    )
    return spatial.geodetic_to_ecef(lon_lat_alt)


def _ecef_to_geodetic_batch(xyz_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :func:`_ecef_to_geodetic`: ``(K, 3)`` ECEF km -> ``(lat, lon, height_km)`` arrays."""
    lon_lat_alt = spatial.ecef_to_geodetic(np.asarray(xyz_km, dtype=float).reshape(-1, 3))
    return lon_lat_alt[:, 1], lon_lat_alt[:, 0], lon_lat_alt[:, 2]


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
        intersection, NaN where ``hit`` is False) and ``hit`` is a ``(K,)`` bool
        mask marking rays that meet the ellipsoid at a positive ray parameter.
    """
    origin = np.broadcast_to(np.asarray(origin_km, dtype=float), np.shape(direction))
    points = spatial.ray_intersect_ellipsoid(np.asarray(direction, dtype=float), np.ascontiguousarray(origin))
    hit = np.all(np.isfinite(points), axis=-1)
    return points, hit


def _satellite_from_subpoint_batch(
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
    semi-major axis (:func:`_valid_spacecraft`).
    """
    surface = _geodetic_to_ecef_batch(subsatellite_lat_deg, subsatellite_lon_deg, np.zeros_like(subsatellite_lat_deg))
    normal = _ellipsoid_normal_batch(subsatellite_lat_deg, subsatellite_lon_deg)
    s_dot_n = np.sum(surface * normal, axis=1)
    radius = np.asarray(satellite_radius_km, dtype=float)
    height = -s_dot_n + np.sqrt(s_dot_n * s_dot_n - np.sum(surface * surface, axis=1) + radius * radius)
    return surface + height[:, None] * normal


def _viewing_geometry_batch(
    lat: np.ndarray,
    lon: np.ndarray,
    sub_lat: np.ndarray,
    sub_lon: np.ndarray,
    satellite_radius_km: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`_viewing_geometry` over ``N`` footprints.

    Returns ``(satellite (N,3), boresight_direction (N,3))``.

    Raises
    ------
    MissingGeolocationError
        If any footprint is not visible from its satellite: the boresight ray's first
        ground intersection lands more than :data:`_VISIBILITY_TOLERANCE_KM` from the
        reported footprint, i.e. the footprint lies beyond the satellite's horizon.
    """
    satellite = _satellite_from_subpoint_batch(sub_lat, sub_lon, satellite_radius_km)
    ground = _geodetic_to_ecef_batch(lat, lon, np.zeros_like(lat))
    to_ground = ground - satellite
    boresight_direction = to_ground / np.linalg.norm(to_ground, axis=1, keepdims=True)

    first_hit, hit = _ray_ellipsoid_intersection_batch(satellite, boresight_direction)
    with np.errstate(invalid="ignore"):
        visible = hit & (np.linalg.norm(first_hit - ground, axis=1) <= _VISIBILITY_TOLERANCE_KM)
    if not np.all(visible):
        raise MissingGeolocationError(
            f"{int(np.count_nonzero(~visible))} footprint(s) are not visible from the satellite placed by their "
            "subsatellite point and radius (beyond its horizon): inconsistent geolocation."
        )
    return satellite, boresight_direction


def _scan_frame_axes_batch(boresight_direction: np.ndarray, satellite_km: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_scan_frame_axes`: the ``(N, 3)`` cross-scan rotation axes."""
    nadir = -satellite_km / np.linalg.norm(satellite_km, axis=1, keepdims=True)
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

    # Pole-enclosure reach: geodesic distance from each center to every perimeter point.
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
    use_360 = span_360 < span_signed - _LON_SPAN_TIE_TOL_DEG
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
    *,
    satellite_radius_km: np.ndarray,
    fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG,
    n_samples: int = _N_PERIMETER_SAMPLES,
) -> list[BoundingBox | None]:
    """Batched :func:`compute_footprint_bounding_box` over ``N`` footprints.

    Vectorizes the entire scalar pipeline (viewing geometry, PSF perimeter ray-trace,
    box assembly) over arrays of ``N`` footprints so the per-footprint pyproj / rotation
    work is amortized across the whole segment. Numerically equivalent to calling the
    scalar entry point per footprint with ``on_limb="flag"`` (validated in the tests).

    Off-Earth views -- those the scalar path raises :class:`OffLimbError` for (a fill,
    non-finite or out-of-range boresight latitude/longitude) -- are returned as ``None``
    at their index rather than raising, so the caller can flag them per footprint. Corner
    rays that run off the limb at a severe angle are truncated at the horizon and the
    box's ``truncated`` flag is set, exactly like ``on_limb="flag"``. Missing ephemeris
    fields or a footprint not visible from its satellite raises
    :class:`MissingGeolocationError` for the whole call instead of hiding among the
    ``None`` entries.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : np.ndarray
        Per-footprint boresight centroids (L1B ``Latitude``/``Longitude``), degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : np.ndarray
        Per-footprint geodetic spacecraft sub-points (L1B ``Subsatellite_Latitude`` /
        ``Subsatellite_Longitude``), degrees.
    satellite_radius_km : np.ndarray
        Per-footprint spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle floor on the PSF extent, degrees.
    n_samples : int, optional
        Number of PSF-perimeter samples. Defaults to :data:`_N_PERIMETER_SAMPLES`.

    Returns
    -------
    list[BoundingBox | None]
        One entry per input footprint, in input order. ``None`` marks an off-Earth view
        (the scalar :class:`OffLimbError` case).

    Raises
    ------
    MissingGeolocationError
        If any footprint's subsatellite point or radius is unusable, or any footprint is
        not visible from its satellite.
    """
    lat = np.asarray(boresight_lat_deg, dtype=float)
    lon = np.asarray(boresight_lon_deg, dtype=float)
    sub_lat = np.asarray(subsatellite_lat_deg, dtype=float)
    sub_lon = np.asarray(subsatellite_lon_deg, dtype=float)
    radius = np.broadcast_to(np.asarray(satellite_radius_km, dtype=float), lat.shape)
    n_total = lat.size

    # Missing ephemeris is bad input, not an off-Earth view: fail loudly, as the scalar
    # path's MissingGeolocationError does, rather than hide it among the Nones.
    spacecraft_ok = _valid_spacecraft(sub_lat, sub_lon, radius)
    if not np.all(spacecraft_ok):
        raise MissingGeolocationError(
            f"{int(np.count_nonzero(~spacecraft_ok))} footprint(s) have a fill/non-finite/out-of-range "
            "subsatellite point or radius."
        )
    # Off-Earth views are exactly the scalar OffLimbError condition: a missing boresight.
    valid = _valid_boresight(lat, lon)

    boxes: list[BoundingBox | None] = [None] * n_total
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return boxes

    la, lo_ = lat[idx], lon[idx]
    m = idx.size

    satellite, boresight_direction = _viewing_geometry_batch(la, lo_, sub_lat[idx], sub_lon[idx], radius[idx])

    # PSF angular extents: static across footprints (same as the scalar path), with the
    # FOV floor and outward safety margin applied.
    psf_extent = psf_95_energy_extent()
    along_extent_deg = max(conservative_along_scan_extent(psf_extent), fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)
    cross_extent_deg = max(psf_extent.beta_max_deg, fov_halfangle_deg) * (1.0 + BBOX_MARGIN_FRACTION)

    cross_axis = _scan_frame_axes_batch(boresight_direction, satellite)

    delta_deg, beta_deg = _perimeter_angles(along_extent_deg, cross_extent_deg, n_samples)  # (S,), (S,)

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
    n_hat)``. Building them per footprint repeats :func:`_viewing_geometry`. This struct holds all ``N``
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
    *,
    satellite_radius_km: np.ndarray,
) -> ViewingFrame:
    """Build every footprint's :class:`ViewingFrame` in one batched call.

    Vectorized companion to the frame construction inside :func:`project_to_angular`.
    Reproduces it exactly (same viewing geometry, same orthonormal basis) so a frame
    fed back into :func:`project_to_angular` yields identical ``(delta, beta)`` to the
    inline path -- validated in the tests.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : np.ndarray
        Per-footprint boresight centroids, degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : np.ndarray
        Per-footprint geodetic spacecraft sub-points (L1B ``Subsatellite_Latitude`` /
        ``Subsatellite_Longitude``), degrees.
    satellite_radius_km : np.ndarray
        Per-footprint spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km.

    Returns
    -------
    ViewingFrame
        The ``N`` batched frames.

    Raises
    ------
    MissingGeolocationError
        If any footprint's ephemeris fields are unusable or the footprint is not visible
        from its satellite.
    """
    lat = np.asarray(boresight_lat_deg, dtype=float)
    lon = np.asarray(boresight_lon_deg, dtype=float)
    sub_lat = np.asarray(subsatellite_lat_deg, dtype=float)
    sub_lon = np.asarray(subsatellite_lon_deg, dtype=float)
    radius = np.broadcast_to(np.asarray(satellite_radius_km, dtype=float), lat.shape)
    if not np.all(_valid_spacecraft(sub_lat, sub_lon, radius)):
        raise MissingGeolocationError("Fill/non-finite/out-of-range subsatellite point or radius.")

    satellite, boresight_direction = _viewing_geometry_batch(lat, lon, sub_lat, sub_lon, radius)
    b_hat = boresight_direction / np.linalg.norm(boresight_direction, axis=1, keepdims=True)
    c_hat = _scan_frame_axes_batch(boresight_direction, satellite)
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
    *,
    satellite_radius_km: float | None = None,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
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
    ``d = (b_hat cos(delta) + n_hat sin(delta)) cos(beta) - c_hat sin(beta)`` (the
    beta rotation is about ``c_hat x along``), the inverse is exact::

        beta  = asin(-look . c_hat)
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
        Geodetic spacecraft sub-point (L1B ``Subsatellite_Latitude``/``Longitude``),
        degrees.
    satellite_radius_km : float, optional
        Spacecraft distance from Earth's center (L1B
        ``Radius_of_Satellite_from_Center_of_Earth``), km. Required unless ``frame`` is
        given, and ignored when it is.
    frame : tuple of np.ndarray, optional
        A precomputed ``(satellite, b_hat, c_hat, n_hat)`` viewing frame for this
        footprint (each a length-3 array), as produced per footprint by
        :func:`compute_viewing_frames` / :meth:`ViewingFrame.for_index`. When supplied,
        the per-footprint :func:`_viewing_geometry` reconstruction is skipped and the
        frame is used directly -- the boresight / subsatellite / radius arguments are
        then unused. Passing the
        frame the orchestrator already batched avoids recomputing it once per footprint.
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
        If neither ``frame`` nor ``satellite_radius_km`` is given.
    MissingGeolocationError
        If the ephemeris fields are unusable or the footprint is not visible from the
        satellite.
    """
    lats = np.asarray(cell_lats_deg, dtype=float)
    lons = np.asarray(cell_lons_deg, dtype=float)
    shape = lats.shape

    if frame is not None:
        # Use the frame the orchestrator batched (identical to the inline construction).
        satellite, b_hat, c_hat, n_hat = frame
    else:
        if satellite_radius_km is None:
            raise ValueError("project_to_angular needs satellite_radius_km when no frame is given.")
        if not _valid_spacecraft(subsatellite_lat_deg, subsatellite_lon_deg, satellite_radius_km):
            raise MissingGeolocationError("Fill/non-finite/out-of-range subsatellite point or radius.")
        # Rebuild the satellite viewing frame for this footprint (same construction the
        # bounding-box path uses, so the projection is consistent with it).
        satellite, boresight_direction = _viewing_geometry(
            boresight_lat_deg, boresight_lon_deg, subsatellite_lat_deg, subsatellite_lon_deg, satellite_radius_km
        )
        b_hat = _unit(boresight_direction)
        c_hat = _unit(_scan_frame_axes(boresight_direction, satellite))
        # In-plane axis: unit because c_hat is perpendicular to b_hat (cross-scan axis is
        # built as a cross product with the boresight), and both are unit vectors.
        n_hat = np.cross(c_hat, b_hat)

    # Geodetic -> ECEF (km) for every cell center (cells sit on the ellipsoid). This
    # depends only on the cell coordinates, not
    # the frame, so an already-computed value (shared across footprints on the same tile) is
    # used when supplied and recomputed otherwise -- identical either way.
    cells_km = cell_ecef_km if cell_ecef_km is not None else surface_cell_ecef_km(lats, lons)  # (M, 3)

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
