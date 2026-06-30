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

Earth model: SPHERE
-------------------
We approximate the Earth as a sphere of radius :data:`EARTH_RADIUS_KM` (the CERES
ATBD value), matching the heritage PSF geometry. This was validated against the
example L1B file: recovering the satellite altitude from VZA + footprint/subsatellite
positions on this sphere gave 834.6 +/- 5.8 km (~0.7% scatter) across ~399k
footprints. That ~0.7% is the real WGS84/terrain departure from the sphere, and it
is comfortably absorbed by the outward rounding of a bounding box.

Caveats of the spherical approximation (all benign for a bounding box, more relevant
for the later pixel-level projection step):
  * Ellipsoid flattening: WGS84 equatorial 6378.1 km vs polar 6356.8 km (~0.3%).
  * Geodetic vs geocentric latitude differ by up to ~0.2 deg at mid-latitudes;
    L1B latitudes are geodetic and we treat them as spherical.
TODO[LIBSDC-794]: migrate to the WGS84 ellipsoid using pyproj (already a project
dependency: pyproj.Geod for ellipsoidal geodesics, pyproj.Transformer for
geodetic<->ECEF) if pixel-accurate geometry is later required.


References
----------
* CERES ATBD v2.2, Section 4.4 (spherical viewing geometry, Eqs. 4.4-4/4.4-5):
  https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf
* Great-circle "destination point", bearing, and distance formulae:
  https://www.movable-type.co.uk/scripts/latlong.html

"""

from __future__ import annotations

import math

import numpy as np

from libera_utils.footprint_matching.psf import (
    LIBERA_FOV_HALFANGLE_DEG,
    conservative_along_scan_extent,
    psf_95_energy_extent,
)
from libera_utils.footprint_matching.types import BoundingBox

# Spherical Earth radius, in km. This is the CERES ATBD value (ATBD Table 4.4-1),
# chosen so our geometry stays consistent with the heritage PSF math. See the
# module docstring for the validation of this approximation against real L1B data.
# TODO[LIBSDC-794]: replace with the WGS84 ellipsoid (via pyproj) if pixel-accurate
# geometry becomes necessary.
EARTH_RADIUS_KM: float = 6367.0

# Fallback satellite altitude, used only when the altitude cannot be derived from
# the inputs (no Altitude field AND the footprint is essentially at nadir, where the
# altitude-recovery formula is numerically degenerate). Value is the median altitude
# recovered from the example L1B file (834.6 km), i.e. the JPSS orbit.
# TODO[LIBSDC-794]: read the nominal altitude from mission config rather than
# hard-coding it.
NOMINAL_ALTITUDE_KM: float = 835.0

# Outward safety margin applied to the footprint half-extents before building the
# box. Absorbs the spherical-Earth approximation error (~0.7%, see module docstring)
# and any small slop in the PSF extent, guaranteeing the box is a true superset.
BBOX_MARGIN_FRACTION: float = 0.05

# Latitude beyond which we flag the box as "polar" so downstream code knows the
# rectangular lat/lon box is a coarse over-approximation (meridians converge). This
# mirrors the design doc's 85 deg threshold.
POLAR_LATITUDE_THRESHOLD_DEG: float = 85.0

# Sentinel below which a footprint is treated as essentially at nadir: the
# along-scan asymmetry and scan azimuth become ill-defined, so we use a symmetric
# box. 1e-6 deg of cone angle is far smaller than any real footprint.
_NADIR_CONE_ANGLE_EPS_DEG: float = 1e-6

# L1B fill value. Footprints that did not intersect the Earth (space/cal views) are
# stored as this sentinel; we treat such inputs as "no footprint".
_L1B_FILL_VALUE: float = -999.0


class GeometryError(Exception):
    """Base class for geometry errors raised by this module."""


class OffLimbError(GeometryError):
    """Raised when the viewing geometry does not intersect the Earth's surface.

    This happens when the line of sight misses the Earth entirely (cone angle at or
    beyond the Earth's angular radius as seen from the satellite), or when the input
    footprint is a fill value (a non-Earth view). The orchestrator is expected to
    catch this and flag/discard the footprint rather than silently substituting data.
    """


class PartialFootprintError(OffLimbError):
    """Raised when the boresight is on Earth but part of the bounding box is not.

    At severe viewing zenith angles the footprint stretches so far that the
    limb-ward *corner* of its bounding box (maximum along-scan offset toward the limb
    combined with the maximum cross-scan offset) projects past the Earth's horizon,
    even though the boresight -- and even the pure along-scan and cross-scan edges --
    still intersect the surface. The box would otherwise silently include a region
    that is off the Earth.

    By default this condition is *flagged* rather than raised:
    :func:`compute_footprint_bounding_box` truncates the box at the limb and sets
    ``BoundingBox.truncated = True`` (partial coverage). This exception is raised only
    when the caller opts in with ``on_limb="raise"``.

    This is a subclass of :class:`OffLimbError`, so callers that simply
    ``except OffLimbError`` keep working; callers that want to distinguish "no
    footprint at all" (centroid off-limb) from "footprint clipped by the limb" can
    catch this subclass specifically.
    """


# ---------------------------------------------------------------------------
# Spherical trigonometry helpers (great-circle math on a sphere of radius R)
# These follow https://www.movable-type.co.uk/scripts/latlong.html
# ---------------------------------------------------------------------------


def _great_circle_distance_km(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float, earth_radius_km: float
) -> float:
    """Great-circle (haversine) distance between two lat/lon points, in km.

    The haversine form is used (rather than the simpler spherical law of cosines)
    because it stays numerically accurate for the small angular separations typical
    between a footprint and its subsatellite point.

    Parameters
    ----------
    lat1_deg, lon1_deg, lat2_deg, lon2_deg : float
        The two points, in degrees.
    earth_radius_km : float
        Sphere radius in km.

    Returns
    -------
    float
        Surface distance in km.
    """
    phi1, phi2 = math.radians(lat1_deg), math.radians(lat2_deg)
    dphi = math.radians(lat2_deg - lat1_deg)
    dlambda = math.radians(lon2_deg - lon1_deg)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return earth_radius_km * 2.0 * math.asin(min(1.0, math.sqrt(a)))


def _initial_bearing_deg(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees clockwise from north.

    Used to find the scan azimuth: the along-scan ground direction is the great
    circle joining the footprint and the subsatellite point (changing the cone angle
    slides the look-point along that great circle), so the bearing from the footprint
    toward the subsatellite point gives the along-scan axis orientation.

    Parameters
    ----------
    lat1_deg, lon1_deg, lat2_deg, lon2_deg : float
        Start (1) and end (2) points in degrees.

    Returns
    -------
    float
        Bearing in [0, 360) degrees.
    """
    phi1, phi2 = math.radians(lat1_deg), math.radians(lat2_deg)
    dlambda = math.radians(lon2_deg - lon1_deg)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.degrees(math.atan2(x, y)) % 360.0


def _destination_point(
    lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float, earth_radius_km: float
) -> tuple[float, float]:
    """Point reached by travelling ``distance_km`` along ``bearing_deg`` from a start point.

    This is the spherical "destination point" formula. We use it instead of the
    flat-Earth ``km / 111`` and ``km / (111 * cos(lat))`` approximation because it is
    curvature-correct everywhere and degrades gracefully toward the poles (where the
    flat-Earth longitude scaling blows up).

    Parameters
    ----------
    lat_deg, lon_deg : float
        Start point in degrees.
    bearing_deg : float
        Travel direction, degrees clockwise from north.
    distance_km : float
        Surface distance to travel, km.
    earth_radius_km : float
        Sphere radius, km.

    Returns
    -------
    tuple[float, float]
        Destination (latitude, longitude) in degrees, longitude normalized to
        [-180, 180].
    """
    angular_distance = distance_km / earth_radius_km  # central angle, radians
    phi1 = math.radians(lat_deg)
    lambda1 = math.radians(lon_deg)
    theta = math.radians(bearing_deg)

    sin_phi2 = math.sin(phi1) * math.cos(angular_distance) + math.cos(phi1) * math.sin(angular_distance) * math.cos(
        theta
    )
    sin_phi2 = max(-1.0, min(1.0, sin_phi2))
    phi2 = math.asin(sin_phi2)

    y = math.sin(theta) * math.sin(angular_distance) * math.cos(phi1)
    x = math.cos(angular_distance) - math.sin(phi1) * sin_phi2
    lambda2 = lambda1 + math.atan2(y, x)

    lat2 = math.degrees(phi2)
    # Normalize longitude to [-180, 180].
    lon2 = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return lat2, lon2


# ---------------------------------------------------------------------------
# Viewing-geometry triangle (satellite, Earth-centre, ground point)
# ---------------------------------------------------------------------------


def _solve_viewing_triangle(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    subsatellite_lat_deg: float,
    subsatellite_lon_deg: float,
    viewing_zenith_deg: float,
    altitude_km: float | None,
    earth_radius_km: float,
) -> tuple[float, float, float]:
    """Solve the satellite/Earth-centre/ground-point triangle for this footprint.

    The triangle (CERES ATBD Fig. 4.4-2) has vertices at the Earth's centre O, the
    satellite S, and the ground point P, with:
      * cone angle ``alpha`` = nadir angle of the view vector at S,
      * viewing zenith angle ``theta`` = angle at P (given, from L1B),
      * Earth-central angle ``gamma`` = angle at O, and gamma = theta - alpha,
      * law of sines: Re / sin(alpha) = (Re + h) / sin(theta).

    There are two ways to close the triangle depending on what is available:

    * If ``altitude_km`` is given, ``Re + h`` is known and we get
      ``alpha = asin(Re * sin(theta) / (Re + h))`` directly.
    * Otherwise (the current L1B product has no usable altitude), we get ``gamma``
      from the *positions* (the great-circle distance from footprint to subsatellite
      point is ``Re * gamma``), then ``alpha = theta - gamma`` and recover
      ``Re + h = Re * sin(theta) / sin(alpha)``. This was validated to ~0.7% against
      the example file.

    Near nadir (alpha -> 0) the position-based recovery of ``Re + h`` is
    numerically ill-conditioned (it divides by sin(alpha)), so we fall back to
    :data:`NOMINAL_ALTITUDE_KM`. The footprint is a tiny near-circular disc there, so
    the exact altitude is immaterial.

    Returns
    -------
    tuple[float, float, float]
        ``(alpha_deg, rho_km, earth_plus_alt_km)`` -- the cone angle, the slant range
        (satellite-to-ground distance), and ``Re + h``.

    Raises
    ------
    OffLimbError
        If the resulting geometry does not intersect the Earth (see
        :func:`_check_on_limb`).
    """
    theta = math.radians(viewing_zenith_deg)

    if altitude_km is not None and altitude_km > 0.0:
        # Altitude known: close the triangle via the law of sines.
        earth_plus_alt = earth_radius_km + altitude_km
        sin_alpha = earth_radius_km * math.sin(theta) / earth_plus_alt
        alpha = math.asin(max(-1.0, min(1.0, sin_alpha)))
    else:
        # Altitude unknown: derive the Earth-central angle from the positions.
        gamma = (
            _great_circle_distance_km(
                boresight_lat_deg, boresight_lon_deg, subsatellite_lat_deg, subsatellite_lon_deg, earth_radius_km
            )
            / earth_radius_km
        )
        alpha = theta - gamma
        if alpha <= math.radians(_NADIR_CONE_ANGLE_EPS_DEG):
            # Degenerate near nadir: positions can't recover altitude robustly.
            earth_plus_alt = earth_radius_km + NOMINAL_ALTITUDE_KM
            sin_alpha = earth_radius_km * math.sin(theta) / earth_plus_alt
            alpha = math.asin(max(-1.0, min(1.0, sin_alpha)))
        else:
            # Law of sines, solved for the satellite radius Re + h.
            earth_plus_alt = earth_radius_km * math.sin(theta) / math.sin(alpha)

    _check_on_limb(alpha, earth_plus_alt, earth_radius_km)

    # Earth-central angle and slant range (law of cosines, ATBD Eq. 4.4-5).
    gamma = theta - alpha
    rho = math.sqrt(earth_radius_km**2 + earth_plus_alt**2 - 2.0 * earth_radius_km * earth_plus_alt * math.cos(gamma))
    return math.degrees(alpha), rho, earth_plus_alt


def _limb_cone_angle_deg(earth_plus_alt_km: float, earth_radius_km: float) -> float:
    """Maximum cone angle that still hits the Earth, in degrees.

    Beyond this angle the line of sight is tangent to / misses the Earth: it is the
    Earth's angular radius as seen from the satellite, ``asin(Re / (Re + h))``.
    """
    return math.degrees(math.asin(earth_radius_km / earth_plus_alt_km))


def _check_on_limb(alpha_rad: float, earth_plus_alt_km: float, earth_radius_km: float) -> None:
    """Raise :class:`OffLimbError` if a cone angle is at or beyond the Earth limb.

    Parameters
    ----------
    alpha_rad : float
        Cone angle in radians.
    earth_plus_alt_km : float
        Satellite radius Re + h, km.
    earth_radius_km : float
        Sphere radius, km.
    """
    limb = math.radians(_limb_cone_angle_deg(earth_plus_alt_km, earth_radius_km))
    if alpha_rad >= limb:
        raise OffLimbError(
            f"Cone angle {math.degrees(alpha_rad):.3f} deg reaches the Earth limb "
            f"({math.degrees(limb):.3f} deg); the line of sight misses the surface."
        )


def _effective_cone_angle_deg(inplane_cone_angle_deg: float, cross_offset_deg: float) -> float:
    """Cone angle of a view ray given its in-plane and cross-scan angular components.

    A footprint point is offset from the boresight by an along-scan angle (which lies
    in the scan plane -- the plane that also contains the nadir direction) and a
    cross-scan angle (perpendicular to that plane). The along-scan offset therefore
    adds *in-plane* to the boresight's own cone angle, giving an in-plane cone angle;
    the cross-scan offset is the perpendicular leg.

    Because the two legs are orthogonal, the resulting cone angle (the total
    off-nadir angle of the ray) follows the spherical Pythagorean theorem for a right
    spherical triangle, ``cos(c) = cos(a) * cos(b)``:

        cos(alpha_eff) = cos(inplane_cone_angle) * cos(cross_offset)

    Parameters
    ----------
    inplane_cone_angle_deg : float
        In-plane cone angle (centroid cone angle plus the along-scan offset), degrees.
    cross_offset_deg : float
        Cross-scan angular offset, degrees.

    Returns
    -------
    float
        Effective cone angle (off-nadir angle of the ray) in degrees.
    """
    cos_eff = math.cos(math.radians(inplane_cone_angle_deg)) * math.cos(math.radians(cross_offset_deg))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_eff))))


def _check_box_within_limb(
    alpha0_deg: float,
    along_extent_deg: float,
    cross_extent_deg: float,
    earth_plus_alt_km: float,
    earth_radius_km: float,
    on_limb: str,
) -> tuple[float, float, bool]:
    """Check whether the whole bounding box (including its corners) stays on Earth.

    The bounding box is a rectangle in the instrument angular frame, so its most
    extreme point is the limb-ward **corner**: the maximum along-scan offset toward
    the limb *combined with* the maximum cross-scan offset. Because the effective
    cone angle (:func:`_effective_cone_angle_deg`) is monotonic in both the limb-ward
    along-scan offset and the cross-scan offset, that single corner is the worst case
    for the entire box -- it goes off-limb before any pure edge does.

    This is the key check the centroid test (:func:`_check_on_limb`) and the old
    pure-along-scan test miss: at severe angles the corner can be off the Earth while
    the boresight and both axis edges are still on it.

    Parameters
    ----------
    alpha0_deg : float
        Centroid cone angle, degrees.
    along_extent_deg, cross_extent_deg : float
        Along-scan and cross-scan angular half-extents of the box, degrees.
    earth_plus_alt_km : float
        Satellite radius Re + h, km.
    earth_radius_km : float
        Sphere radius, km.
    on_limb : str
        ``"flag"`` (default policy) to truncate the limb-ward along-scan extent so the
        corner sits just inside the limb and report the truncation, or ``"raise"`` to
        raise :class:`PartialFootprintError` instead.

    Returns
    -------
    tuple[float, float, bool]
        ``(along_extent_deg, cross_extent_deg, truncated)``. When the box is fully on
        the Earth the extents are unchanged and ``truncated`` is ``False``. When the
        corner was off-limb and ``on_limb="flag"``, the along-scan extent is reduced
        to the horizon and ``truncated`` is ``True``.

    Raises
    ------
    PartialFootprintError
        If the limb-ward corner is off the Earth and ``on_limb="raise"``.
    """
    limb_deg = _limb_cone_angle_deg(earth_plus_alt_km, earth_radius_km)
    # Limb-ward corner: in-plane cone angle alpha0 + along_extent, plus cross_extent.
    corner_cone_deg = _effective_cone_angle_deg(alpha0_deg + along_extent_deg, cross_extent_deg)

    if corner_cone_deg < limb_deg:
        return along_extent_deg, cross_extent_deg, False

    if on_limb == "raise":
        raise PartialFootprintError(
            f"Bounding-box corner reaches cone angle {corner_cone_deg:.3f} deg "
            f"(Earth limb {limb_deg:.3f} deg) at centroid cone angle {alpha0_deg:.3f} deg: "
            f"part of the box is off the Earth limb. The default on_limb='flag' truncates "
            f"the box at the horizon and marks it as partial coverage instead of raising."
        )

    # flag: truncate. Find the largest in-plane cone angle whose corner (with the same
    # cross-scan extent) still sits just inside the limb, then back out the
    # corresponding along-scan extent. Solve, from the Pythagorean relation,
    #   cos(limb*(1-eps)) = cos(inplane_max) * cos(cross)
    # for inplane_max, then along' = inplane_max - alpha0 (clamped at >= 0).
    target = math.radians(limb_deg * (1.0 - 1e-9))
    cos_ratio = math.cos(target) / math.cos(math.radians(cross_extent_deg))
    inplane_max_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ratio))))
    clamped_along_deg = max(0.0, inplane_max_deg - alpha0_deg)
    return clamped_along_deg, cross_extent_deg, True


def _ground_arc_from_cone_angle(alpha_deg: float, earth_plus_alt_km: float, earth_radius_km: float) -> float:
    """Ground arc distance (subsatellite point -> look-point) for a cone angle.

    Forward spherical geometry (CERES ATBD, notebook ``get_geometry_from_alpha``):
    from the cone angle ``alpha`` and satellite radius, find the viewing zenith
    angle, the Earth-central angle ``gamma = theta - alpha``, and hence the ground
    arc ``l = Re * gamma``. Used to convert an along-scan *angular* offset into an
    along-scan *ground* distance by perturbing alpha.

    Returns
    -------
    float
        Ground arc distance in km.

    Raises
    ------
    OffLimbError
        If ``alpha`` is at or beyond the Earth limb.
    """
    alpha = math.radians(alpha_deg)
    sin_theta = earth_plus_alt_km * math.sin(alpha) / earth_radius_km
    if sin_theta >= 1.0:
        raise OffLimbError(f"Perturbed cone angle {alpha_deg:.3f} deg projects past the Earth limb.")
    theta = math.asin(sin_theta)
    gamma = theta - alpha
    return earth_radius_km * gamma


def _along_scan_ground_extent_km(
    alpha0_deg: float,
    delta_extent_deg: float,
    earth_plus_alt_km: float,
    earth_radius_km: float,
) -> float:
    """Project the along-scan PSF angular half-extent to a ground half-extent (km).

    The scan sweeps in cone angle, so an along-scan angular offset ``delta`` is a
    perturbation of the centroid cone angle ``alpha0`` (notebook cell 2:
    ``alpha = alpha0 - delta``). Because the cone-angle -> ground-arc relationship is
    nonlinear, the same angular offset projects to *different* ground distances on
    the limb side vs the nadir side -- this is the along-scan stretch that grows
    severe near the limb.

    For a bounding box we want a single conservative half-extent, so we perturb in
    BOTH directions and take the larger ground excursion. The limb-ward direction
    dominates at high VZA.

    Off-limb safety is enforced upstream: :func:`compute_footprint_bounding_box`
    calls :func:`_check_box_within_limb` *before* this projection, so by the time we
    perturb here the edges are guaranteed to stay inside the limb. The guard in
    :func:`_ground_arc_from_cone_angle` remains only as a defensive backstop.

    Parameters
    ----------
    alpha0_deg : float
        Centroid cone angle, degrees.
    delta_extent_deg : float
        Along-scan angular half-extent of the box, degrees.
    earth_plus_alt_km : float
        Satellite radius Re + h, km.
    earth_radius_km : float
        Sphere radius, km.

    Returns
    -------
    float
        Conservative along-scan ground half-extent in km.
    """
    l0 = _ground_arc_from_cone_angle(alpha0_deg, earth_plus_alt_km, earth_radius_km)

    excursions = []
    # +delta -> toward nadir (smaller alpha); -delta -> toward limb (larger alpha).
    for signed_delta in (delta_extent_deg, -delta_extent_deg):
        # Cone angle is unsigned; reflect through nadir if the perturbation overshoots.
        alpha_edge = abs(alpha0_deg - signed_delta)
        l_edge = _ground_arc_from_cone_angle(alpha_edge, earth_plus_alt_km, earth_radius_km)
        excursions.append(abs(l_edge - l0))

    return max(excursions)


def _cross_scan_ground_extent_km(slant_range_km: float, beta_max_deg: float) -> float:
    """Project the cross-scan PSF angular half-extent to a ground half-extent (km).

    The cross-scan direction is perpendicular to the scan plane, so the angular
    offset ``beta`` does not change the cone angle; it is simply projected through the
    slant range: ``perpendicular distance = rho * tan(beta)`` (notebook cell 1).

    Parameters
    ----------
    slant_range_km : float
        Satellite-to-ground distance rho, km.
    beta_max_deg : float
        Cross-scan angular half-extent of the PSF, degrees.

    Returns
    -------
    float
        Cross-scan ground half-extent in km.
    """
    return slant_range_km * math.tan(math.radians(beta_max_deg))


# ---------------------------------------------------------------------------
# Bounding-box assembly (footprint half-extents in km -> lat/lon box)
# ---------------------------------------------------------------------------


def _assemble_bounding_box(
    boresight_lat_deg: float,
    boresight_lon_deg: float,
    along_half_km: float,
    cross_half_km: float,
    scan_azimuth_deg: float,
    earth_radius_km: float,
    truncated: bool = False,
    n_perimeter_samples: int = 72,
) -> BoundingBox:
    """Build a lat/lon bounding box from the footprint's ground half-extents.

    The footprint is modelled as an ellipse with the long axis along the scan
    azimuth. We sample its perimeter, map each sample to a (bearing, distance) from
    the boresight, project to lat/lon with the curvature-correct destination-point
    formula, and take the min/max -- handling the three structural edge cases:

    * **Pole enclosure**: if the footprint reaches a pole, all meridians are inside
      it, so longitude must span the full [-180, 180] and the bounding latitude is
      pinned to +/- 90. Detected geometrically (distance from boresight to the pole
      <= footprint radius).
    * **Dateline crossing**: detected by comparing the longitude span in [-180, 180]
      vs in [0, 360); the smaller span wins. When it wraps, we return the [0, 360)
      representation (``lon_max`` > 180), matching the convention used elsewhere in
      the codebase (see ``readers/base.py``), and set ``wraps_dateline``.
    * **Polar advisory**: boxes touching very high latitudes are flagged ``is_polar``
      so downstream code knows the rectangular box is a coarse over-approximation.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : float
        Footprint centroid, degrees.
    along_half_km, cross_half_km : float
        Ground half-extents along and across scan, km.
    scan_azimuth_deg : float
        Orientation of the along-scan axis, degrees clockwise from north.
    earth_radius_km : float
        Sphere radius, km.
    truncated : bool, optional
        Whether the box was clipped at the Earth's limb (partial coverage). Stored on
        the returned :class:`BoundingBox`. Default False.
    n_perimeter_samples : int, optional
        Number of ellipse perimeter samples. Default 72 (every 5 degrees).

    Returns
    -------
    BoundingBox
        The geographic bounding box.
    """
    # The maximum reach in any direction; used for the pole-enclosure test.
    max_radius_km = max(along_half_km, cross_half_km)

    # --- Pole enclosure: does the footprint contain the N or S pole? ---
    # Great-circle distance from the boresight to a pole is Re * radians(90 -/+ lat).
    dist_to_north_pole_km = earth_radius_km * math.radians(90.0 - boresight_lat_deg)
    dist_to_south_pole_km = earth_radius_km * math.radians(90.0 + boresight_lat_deg)

    # --- Sample the elliptical perimeter and project each point to lat/lon. ---
    lats: list[float] = []
    lons: list[float] = []
    for t in np.linspace(0.0, 2.0 * math.pi, n_perimeter_samples, endpoint=False):
        # Point on the ellipse in the local (along, cross) frame.
        along = along_half_km * math.cos(t)
        cross = cross_half_km * math.sin(t)
        distance_km = math.hypot(along, cross)
        # Bearing of this perimeter point relative to the scan axis, then rotated
        # into the geographic frame by the scan azimuth.
        local_bearing_deg = math.degrees(math.atan2(cross, along))
        bearing_deg = scan_azimuth_deg + local_bearing_deg
        lat, lon = _destination_point(boresight_lat_deg, boresight_lon_deg, bearing_deg, distance_km, earth_radius_km)
        lats.append(lat)
        lons.append(lon)

    if dist_to_north_pole_km <= max_radius_km:
        # Footprint encloses the North pole: every longitude is covered.
        return BoundingBox(min(lats), 90.0, -180.0, 180.0, wraps_dateline=False, is_polar=True, truncated=truncated)
    if dist_to_south_pole_km <= max_radius_km:
        # Footprint encloses the South pole.
        return BoundingBox(-90.0, max(lats), -180.0, 180.0, wraps_dateline=False, is_polar=True, truncated=truncated)

    lat_min, lat_max = min(lats), max(lats)

    # --- Dateline handling: choose the representation with the smaller span. ---
    lons_arr = np.asarray(lons)
    span_signed = float(lons_arr.max() - lons_arr.min())  # span in [-180, 180]
    lons_360 = lons_arr % 360.0
    span_360 = float(lons_360.max() - lons_360.min())  # span in [0, 360)

    if span_360 < span_signed:
        # The footprint is tighter when expressed in [0, 360): it wraps the dateline.
        lon_min = float(lons_360.min())
        lon_max = float(lons_360.max())  # may exceed 180 -> signals the wrap
        wraps_dateline = lon_max > 180.0
    else:
        lon_min = float(lons_arr.min())
        lon_max = float(lons_arr.max())
        wraps_dateline = False

    # Advisory polar flag for boxes that reach very high latitudes (meridians
    # converge, so the rectangular box over-covers); mirrors the design doc threshold.
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
    earth_radius_km: float = EARTH_RADIUS_KM,
    on_limb: str = "flag",
) -> BoundingBox:
    """Compute the lat/lon bounding box of one radiometer footprint on the surface.

    This is the public, per-footprint entry point. It chains: solve the viewing
    triangle -> get the PSF angular extent -> project that extent to ground km
    (asymmetric along-scan stretch + cross-scan projection) -> assemble a lat/lon box
    with curvature, pole, and dateline handling.

    Inputs come straight from the L1B fields that are actually populated:
    footprint ``Latitude``/``Longitude``, ``Subsatellite_Latitude``/``Longitude``,
    and ``Viewing_Zenith_Surface``. (The solar zenith and relative azimuth angles
    describe sun geometry and do not affect which ground patch the radiometer sees,
    so they are intentionally not parameters here.)

    TODO[LIBSDC-794]: this is scalar (one footprint per call) for clarity. To meet
    the real-time latency budget, vectorize the helpers over
    NumPy arrays of footprints; the math is all elementwise except the perimeter
    sampling, which can be batched.

    Parameters
    ----------
    boresight_lat_deg, boresight_lon_deg : float
        Footprint centroid (L1B ``Latitude``/``Longitude``), degrees.
    subsatellite_lat_deg, subsatellite_lon_deg : float
        Subsatellite point (L1B ``Subsatellite_Latitude``/``Longitude``), degrees.
        Used to recover geometry without an altitude field and to set the scan
        azimuth.
    viewing_zenith_deg : float
        Viewing zenith angle (L1B ``Viewing_Zenith_Surface``), degrees.
    altitude_km : float or None, optional
        Satellite altitude above the surface, km. Used if provided and positive;
        otherwise recovered from the positions + VZA. Default None.
    fov_halfangle_deg : float, optional
        Instrument FOV half-angle, degrees. Used as a floor on the PSF extent so the
        box is never smaller than the optical field of view. Defaults to
        :data:`~libera_utils.footprint_matching.psf.LIBERA_FOV_HALFANGLE_DEG`.
    earth_radius_km : float, optional
        Spherical Earth radius, km. Defaults to :data:`EARTH_RADIUS_KM`.
    on_limb : {"flag", "raise"}, optional
        Behaviour when the box *corner* runs off the Earth limb at a severe angle
        while the boresight is still on Earth. ``"flag"`` (default) truncates the box
        at the horizon and marks it ``BoundingBox.truncated = True`` so the
        orchestrator can record partial coverage; ``"raise"`` raises
        :class:`PartialFootprintError` instead. Note this does NOT cover the
        *centroid* being off-limb (or fill inputs) -- those mean there is no footprint
        at all and always raise :class:`OffLimbError`, regardless of ``on_limb``.

    Returns
    -------
    BoundingBox
        Geographic bounding box enclosing the footprint. ``BoundingBox.truncated`` is
        ``True`` when the box was clipped at the limb (partial coverage).

    Raises
    ------
    OffLimbError
        If any input is an L1B fill value (a non-Earth view), or if the *centroid*
        viewing geometry does not intersect the surface. These mean there is no
        footprint at all, so they raise regardless of ``on_limb``.
    PartialFootprintError
        A subclass of :class:`OffLimbError`. Raised only when the boresight is on Earth
        but the limb-ward corner of the box is off it (severe-angle truncation) *and*
        ``on_limb="raise"``.
    ValueError
        If ``on_limb`` is not ``"flag"`` or ``"raise"``.
    """
    if on_limb not in ("flag", "raise"):
        raise ValueError(f"on_limb must be 'flag' or 'raise', got {on_limb!r}")

    # Reject fill-valued / non-finite inputs: these are space or calibration views
    # with no Earth intersection. Treat them like an off-limb footprint so the
    # caller's flag/discard path handles them uniformly.
    for value in (boresight_lat_deg, boresight_lon_deg, viewing_zenith_deg):
        if not math.isfinite(value) or value == _L1B_FILL_VALUE:
            raise OffLimbError("Footprint has fill/non-finite geolocation (non-Earth view).")

    # 1. Solve the viewing triangle for the cone angle, slant range, and Re + h.
    alpha0_deg, slant_range_km, earth_plus_alt_km = _solve_viewing_triangle(
        boresight_lat_deg,
        boresight_lon_deg,
        subsatellite_lat_deg,
        subsatellite_lon_deg,
        viewing_zenith_deg,
        altitude_km,
        earth_radius_km,
    )

    # 2. Get the PSF angular extent and apply the FOV floor. We use the larger of the
    #    dynamic 95%-energy extent and the static FOV so the box is never smaller than
    #    the optical field of view (the FOV also stands in for the stationary-scanner
    #    case, which we cannot currently detect without the cone-angle rate).
    psf_extent = psf_95_energy_extent()
    along_extent_deg = max(conservative_along_scan_extent(psf_extent), fov_halfangle_deg)
    cross_extent_deg = max(psf_extent.beta_max_deg, fov_halfangle_deg)

    # 2b. Verify the WHOLE box stays on the Earth. At severe angles the limb-ward
    #     corner (max along-scan toward the limb + max cross-scan) can be off-limb
    #     even though the centroid and the pure axis edges are not. By default
    #     (on_limb="flag") this truncates the along-scan extent at the horizon and
    #     flags the box as partial coverage; on_limb="raise" raises instead.
    along_extent_deg, cross_extent_deg, truncated = _check_box_within_limb(
        alpha0_deg, along_extent_deg, cross_extent_deg, earth_plus_alt_km, earth_radius_km, on_limb
    )

    # 3. Project the angular extents to ground half-extents (km).
    along_half_km = _along_scan_ground_extent_km(alpha0_deg, along_extent_deg, earth_plus_alt_km, earth_radius_km)
    cross_half_km = _cross_scan_ground_extent_km(slant_range_km, cross_extent_deg)

    # 4. Round outward by the safety margin (see BBOX_MARGIN_FRACTION rationale).
    along_half_km *= 1.0 + BBOX_MARGIN_FRACTION
    cross_half_km *= 1.0 + BBOX_MARGIN_FRACTION

    # 5. Scan azimuth: the along-scan axis points along the great circle toward the
    #    subsatellite point. (Ill-defined exactly at nadir, but there the footprint is
    #    a near-circular disc and the orientation is irrelevant.)
    scan_azimuth_deg = _initial_bearing_deg(
        boresight_lat_deg, boresight_lon_deg, subsatellite_lat_deg, subsatellite_lon_deg
    )

    # 6. Assemble the lat/lon box (handles curvature, poles, dateline), carrying the
    #    partial-coverage truncation flag onto the returned box.
    return _assemble_bounding_box(
        boresight_lat_deg,
        boresight_lon_deg,
        along_half_km,
        cross_half_km,
        scan_azimuth_deg,
        earth_radius_km,
        truncated=truncated,
    )
