"""Module for performing geolocation tasks"""
# Standard
import logging
# Installed
import numpy as np
import spiceypy as spice
# Local
from libera_utils import spice_utils

logger = logging.getLogger(__name__)


vec_pxform = np.vectorize(spice.pxform,
                          excluded=['fromstr', 'tostr'],
                          signature='(),(),()->(3,3)',
                          otypes=[np.float64])
"""
Vectorized form of spice.pxform function.

"Return the matrix that transforms position vectors from one specified frame to another at a specified epoch."
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/pxform_c.html

======== ==== ===============================================
VARIABLE  I/O  DESCRIPTION
======== ==== ===============================================
from       I   Name of the frame to transform from.
to         I   Name of the frame to transform to.
et         I   Epoch of the rotation matrix.
rotate     O   A rotation matrix.
======== ==== ===============================================

Note: For time-independent reference frames transformations, the ET value passed doesn't matter,
as long as it is a valid ephemeris time.
"""


vec_subpnt = np.vectorize(spice.subpnt,
                          excluded=['method', 'target', 'fixref', 'abcorr', 'obsrvr'],
                          signature='(),(),(),(),(),()->(3),(),(3)',
                          otypes=[np.float64, np.float64, np.float64])
"""
Vectorized form of spice.subpnt function

"Compute the rectangular coordinates of the sub-observer point on a target body at a specified epoch, optionally
corrected for light time and stellar aberration."
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/subpnt_c.html

======== ==== ===============================================
Variable  I/O  Description
======== ==== ===============================================
method     I   Computation method.
target     I   Name of target body.
et         I   Epoch in TDB seconds past J2000 TDB.
fixref     I   Body-fixed, body-centered target body frame.
abcorr     I   Aberration correction flag.
obsrvr     I   Name of observing body.
spoint     O   Sub-observer point on the target body.
trgepc     O   Sub-observer point epoch.
srfvec     O   Vector from observer to sub-observer point.
======== ==== ===============================================
"""


vec_subslr = np.vectorize(spice.subslr,
                          excluded=['method', 'target', 'fixref', 'abcorr', 'obsrvr'],
                          signature='(),(),(),(),(),()->(3),(),(3)',
                          otypes=[np.float64, np.float64, np.float64])
"""
Vectorized form of spice.subslr function

"Compute the rectangular coordinates of the sub-solar point on a target body at a specified epoch, optionally
corrected for light time and stellar aberration."
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/subslr_c.html

======== ==== ===============================================
Variable  I/O  Description
======== ==== ===============================================
method     I   Computation method.
target     I   Name of target body.
et         I   Epoch in ephemeris seconds past J2000 TDB.
fixref     I   Body-fixed, body-centered target body frame.
abcorr     I   Aberration correction.
obsrvr     I   Name of observing body.
spoint     O   Sub-solar point on the target body.
trgepc     O   Sub-solar point epoch.
srfvec     O   Vector from observer to sub-solar point.
======== ==== ===============================================
"""

vec_npedln = np.vectorize(spice.npedln,
                          excluded=['a', 'b', 'c'],
                          signature='(),(),(),(3),(3)->(3),()',
                          otypes=[np.float64, np.float64])
"""
Vectorized form of spice.subslr function

"Compute the rectangular coordinates of the sub-solar point on a target body at a specified epoch, optionally
corrected for light time and stellar aberration."
https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/subslr_c.html

======== ==== ===============================================
Variable  I/O  Description
======== ==== ===============================================
method     I   Computation method.
target     I   Name of target body.
et         I   Epoch in ephemeris seconds past J2000 TDB.
fixref     I   Body-fixed, body-centered target body frame.
abcorr     I   Aberration correction.
obsrvr     I   Name of observing body.
spoint     O   Sub-solar point on the target body.
trgepc     O   Sub-solar point epoch.
srfvec     O   Vector from observer to sub-solar point.
======== ==== ===============================================
"""

# TODO: Try moving the ensure_spice decorators
#  off of the higher level functions below and onto the vectorized spice functions above.


@spice_utils.ensure_spice
def get_earth_radii():
    """
    Retrieve Earth radii values from SPICE

    Returns
    -------
    : tuple
        (re, rp, flat) tuple of equatorial and polar ellipsoid radii and a flattening coefficient
    """
    # Retrieve Earth radii
    _, radii = spice.bodvrd(spice_utils.SpiceBody.EARTH.value.strid, 'RADII', 3)
    re = radii[0]  # Equatorial
    rp = radii[2]  # Polar
    flat = (re - rp) / re
    return re, rp, flat


@spice_utils.ensure_spice
def target_position(target: spice_utils.SpiceBody, et: float or np.ndarray,
                    frame: spice_utils.SpiceFrame, observer: spice_utils.SpiceBody,
                    abcorr: str = 'NONE', normalize: bool = False):
    """Calculates the position and velocity of the `target` at ephemeris time `et` relative to `observer` in
    reference frame `frame`.
    Also calculates the light travel time between `target` and `observer` at time `et`.

    Parameters
    ----------
    target : spice_utils.SpiceBody
        Target body for which to calculate position and velocity relative to observer
    et : float or numpy.ndarray
        Ephemeris time(s)
    frame : spice_utils.SpiceFrame
        Reference frame (unit vectors)
    observer : spice_utils.SpiceBody
        The observer of the target. Resulting coordinates point from observer to target.
    abcorr : str
        A scalar string that indicates the aberration corrections to apply to the database of the target body to account
        for one-way light time and stellar aberration. Default is 'NONE'.
    normalize : bool, Optional
        Return unit vectors for position and velocity (light time output is unchanged)

    Returns
    -------
    : tuple
        (x: numpy.ndarray, v: numpy.ndarray, lt: numpy.ndarray) or (x: float, v: float, lt: float)
        Rectangular position and velocity vectors (x, y, z), (v_x, v_y, v_z) where position
        points from the planet center of mass location at ``et`` to the aberration-corrected location of the target.
        Light time (lt) between planetary body and target.
    """
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/spkezr_c.html
    position_velocity, light_time_between = spice.spkezr(target.value.strid, et,
                                                         frame.value.strid, abcorr, observer.value.strid)

    if isinstance(position_velocity, list):
        position = np.array([el[0:3] for el in position_velocity])
        velocity = np.array([el[3:6] for el in position_velocity])
        if normalize:
            position = np.array([v / np.linalg.norm(v) for v in position])
            velocity = np.array([v / np.linalg.norm(v) for v in velocity])
    else:
        position = np.array(position_velocity[0:3])
        velocity = np.array(position_velocity[3:6])
        if normalize:
            position = position / np.linalg.norm(position)
            velocity = velocity / np.linalg.norm(velocity)
    return position, velocity, light_time_between


@spice_utils.ensure_spice
def sub_observer_point(target: spice_utils.SpiceBody, et: float or np.ndarray,
                       frame: spice_utils.SpiceFrame, observer: spice_utils.SpiceBody,
                       abcorr: str = 'NONE', method: str = 'NEAR POINT/ELLIPSOID'):
    """Computes the cartesian coordinates of the sub-observer point at time `et` and the observer altitude
    above the point. Units in km.

    Parameters
    ----------
    target : spice_utils.SpiceBody
        Body on which the sub point will be calculated (usually a planetary body).
    et : float or numpy.ndarray
        Ephemeris time of observation.
    frame : spice_utils.SpiceFrame
        Reference frame for returned vectors.
    observer : spice_utils.SpiceBody
        The object from which to calculate the sub point (e.g. a spacecraft). A-B correction is applied based on the
        distance between observer and sub observer point.
    abcorr : str, Optional
        String delineating what kind of A-B light time correction to perform. Default is 'NONE'.
    method : str, Optional
        String specifying what kind of method to use to find the vector between the observer and the target.
        Default is `NEAR POINT/ELLIPSOID`, which uses the nearest point on the ellipsoid rather than drawing a line
        through the center of the ellipsoid.

    Returns
    -------
    : numpy.ndarray, float
        Cartesian point on the target body surface in the specified reference frame
        and also the euclidean distance between the observer and the sub point in order: [x, y, z], obs_alt
    """

    spoint, _, obs_tgt_vec = vec_subpnt(method, target.value.strid, et, frame.value.strid,
                                             abcorr, observer.value.strid)
    if hasattr(et, '__len__'):
        observer_alt = np.linalg.norm(obs_tgt_vec, axis=1)  # Calculate distance from obs to spoint as observer altitude
    else:
        observer_alt = np.linalg.norm(obs_tgt_vec)
    return spoint, observer_alt


@spice_utils.ensure_spice
def sub_solar_point(target: spice_utils.SpiceBody, et: float or np.ndarray,
                    frame: spice_utils.SpiceFrame, observer: spice_utils.SpiceBody,
                    abcorr='LT+S', method='NEAR POINT/ELLIPSOID'):
    """Computes the cartesian coordinates of the subsolar point at ephemeris time et.

    Parameters
    ----------
    target : spice_utils.SpiceBody
        Body on which the sub point will be calculated (usually a planetary body).
    et : float or numpy.ndarray
        Ephemeris time of observation.
    frame : spice_utils.SpiceFrame
        Reference frame for returned vectors.
    observer : spice_utils.SpiceBody
        The object from which to calculate the subsolar point (e.g. a spacecraft).
        A-B correction is applied based on the distance between observer and subsolar point.
    abcorr : str
        String delineating what kind of A-B lighttime correction to perform. Default is 'LT+S'.
    method : str
        String specifying what kind of method to use to find the vector between the observer and the target.
        Default is `NEAR POINT/ELLIPSOID`, which uses the nearest point on the ellipsoid rather than drawing a line
        through the center of the ellipsoid.

    Returns
    -------
    : numpy.ndarray, numpy.ndarray, numpy.ndarray
        Subsolar point on the ellipsoid surface in the specified reference frame, apparent epoch at that point
        (depending on specified light time correction), and vector from observer to subsolar point.
    """
    fixref = frame.value.strid
    obsrvr = observer.value.strid
    spoint, trgepc, srfvec = vec_subslr(method, target.value.strid, et, fixref, abcorr, obsrvr)

    return spoint, trgepc, srfvec


@spice_utils.ensure_spice
def frame_transform(from_frame: spice_utils.SpiceFrame, to_frame: spice_utils.SpiceFrame,
                    et: float or np.ndarray, position: np.ndarray,
                    normalize: bool = False) -> np.ndarray:
    """
    Transform a position <x, y, z> vector between reference frames, optionally normalizing the result.

    Parameters
    ----------
    from_frame : spice_utils.SpiceFrame
        Reference frame of position
    to_frame : spice_utils.SpiceFrame
        Reference frame of output
    et : numpy.float64 or numpy.ndarray
        Ephemeris time(s) corresponding to position(s).
        For time-independent transformations, this can by any valid ephemeris time.
    position : numpy.ndarray
        <x, y, z> vector or array of vectors in reference frame `from_frame`
    normalize : bool, Optional
        Optionally normalize the output vector

    Returns
    -------
    : numpy.ndarray
        3d position vector(s) in reference frame `to_frame`
    """
    if ((position.ndim == 1 and not (len(position) == 3 and isinstance(et, float))) or
            (position.ndim == 2 and not (len(position) == len(et) and len(position[0]) == 3))):
        raise ValueError("Incorrect dimensions.")

    rotate = vec_pxform(from_frame.value.strid, to_frame.value.strid, et)

    if hasattr(rotate[0][0], '__len__'):
        result = np.array([np.matmul(rotate, pos).astype(np.float64) for rotate, pos in zip(rotate, position)])
        if normalize:
            result = np.array([r / np.linalg.norm(r) for r in result])
    else:
        result = np.matmul(rotate, position).astype(np.float64)
        if normalize:
            result = result / np.linalg.norm(result)

    return result


def angle_between(v1: np.ndarray, v2: np.ndarray, degrees: bool = False):
    """
    Returns angle between vectors v1 and v2, in units of radians (default) or degrees.
    N is the number of vectors
    D is the dimension of the space

    Parameters
    ----------
    v1 : numpy.ndarray
        Vector(s) 1. May be shape (D,) or (N, D).
    v2 : numpy.ndarray
        Vector(s) 2. May be shape (D,) or (N, D).
    degrees : bool
        Specify True to return result in degrees. Default is False (returns radians).

    Returns
    -------
    : float or numpy.ndarray
        Angle between v1 and v2 in radians (optionally in degrees)
    """
    if v1.ndim == 1 and v2.ndim == 1:
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        theta = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
    elif v1.ndim == 2 and v2.ndim == 2:
        u1 = v1 / np.array([np.linalg.norm(v1, axis=1)]).T
        u2 = v2 / np.array([np.linalg.norm(v2, axis=1)]).T
        theta = np.array([np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)) for a, b in zip(u1, u2)])
    else:
        raise ValueError(f"Function {__name__} only accepts inputs of shape (3,) or (3, N). "
                         f"Got v1.shape={v1.shape}, v2.shape={v2.shape}.")

    if degrees:
        theta = np.rad2deg(theta)
    return theta


@spice_utils.ensure_spice
def cartesian_to_planetographic(cartesian_coords: np.ndarray, degrees: bool = True):
    """
    Convert cartesian coordinates in the ITRF93 frame to planetographic latitude and longitude.
    Longitude runs 0-360 such that longitude appears to increase
    as the planet rotates when viewed by an observer and latitude is calculated from a surface normal vector rather
    than a line through the planet center. See
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/17_frames_and_coordinate_systems.pdf
    for reference.

    Parameters
    ----------
    cartesian_coords : numpy.ndarray
        Rectangular coordinates in ITRF93 frame.
    degrees : bool
        Default true. If False, returns angles in radians.

    Returns
    -------
    : numpy.ndarray
        Each coordinate is returned as (longitude, latitude, altitude).
    """
    re, _, flat = get_earth_radii()

    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/recpgr_c.html
    if hasattr(cartesian_coords[0], '__len__'):
        nan_locations = np.any(np.isnan(cartesian_coords), axis=1)
        result = np.array([
            spice.recpgr(spice_utils.SpiceBody.EARTH.value.strid, p.astype(np.float64), re, flat)
            for p in cartesian_coords
        ])
        longitude = result[:, 0]
        latitude = result[:, 1]
        altitude = result[:, 2]
        longitude[nan_locations] = np.nan
        latitude[nan_locations] = np.nan
        altitude[nan_locations] = np.nan
    else:
        if np.any(np.isnan(cartesian_coords.astype(np.float64))):
            longitude, latitude, altitude = (np.nan, np.nan, np.nan)
        else:
            longitude, latitude, altitude = spice.recpgr(
                spice_utils.SpiceBody.EARTH.value.strid, cartesian_coords.astype(np.float64), re, flat)

    if degrees:
        return np.rad2deg(longitude), np.rad2deg(latitude), altitude

    return longitude, latitude, altitude


@spice_utils.ensure_spice
def surface_intercept_point(sc_location: np.ndarray, look_vector: np.ndarray,
                            look_frame: spice_utils.SpiceFrame, et: float or np.ndarray = None):
    """
    Returns rectangular coordinates of the point of interception of a look direction from the spacecraft onto
    the Earth ellipsoid. If the look vector misses the planet, then the distance returned will be non-zero and the
    point returned is the point on the look_vector ray that is closest to the ellipsoid.

    This routine assumes that the location of the spacecraft and the location of the instrument are the same
    because we don't have ephemeris data for the instrument but we _do_ have ephemeris for the spacecraft. Over
    the scale of distances involved, the offset between spacecraft and instrument (meters) should be negligible
    in affecting the near-point calculation.

    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/npedln_c.html

    Parameters
    ----------
    sc_location : numpy.ndarray
        The location of the observing body (i.e. the spacecraft body) with respect to Earth
    look_vector : numpy.ndarray
        Look direction unit vector (e.g. an instrument look direction)
    look_frame : spice_utils.SpiceFrame
        Reference frame of `look_vector`
    et : float or numpy.ndarray or None, Optional
        Ephemeris time (at spacecraft at photon detection time). Only required if look_frame is not ITRF93.

    Returns
    -------
    : tuple
        (pnear, alt) Rectangular coordinates of nearest point to reference surface ellipsoid and distance
        between the line and the near point.
    """
    # Transform look unit vector(s) from instrument frame into the ITRF93 frame
    if look_frame != spice_utils.SpiceFrame.ITRF93:
        if not et:
            raise ValueError("You specified a look_frame other than ITRF93 but no ephemeris time. "
                             "ET is required to transform your look vector to ITRF93 frame.")
        look_vector_ecef = frame_transform(look_frame, spice_utils.SpiceFrame.ITRF93, et, look_vector, normalize=True)
    else:
        look_vector_ecef = look_vector

    re_earth, rp_earth, _ = get_earth_radii()

    if sc_location.ndim == 2:  # More than one sc_location and look_vector
        pnear_dist = [spice.npedln(re_earth, re_earth, rp_earth, loc, look)
                      for loc, look in zip(sc_location, look_vector_ecef)]
        # [([x, y, z], d), ...]
        pnear = np.array([p for p, d in pnear_dist])  # [[x1, y1, z1], [x2, y2, z2], ...]
        dist = np.array([d for p, d in pnear_dist])  # [d1, d2, ...]
    else:  # Just one sc_location and look_vector
        pnear, dist = spice.npedln(re_earth, re_earth, rp_earth, sc_location, look_vector_ecef)
    return pnear, dist
