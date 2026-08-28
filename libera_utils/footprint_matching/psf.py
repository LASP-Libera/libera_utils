"""Point Spread Function (PSF) support for footprint matching -- PARTIAL module.

What this module is (and is not)
--------------------------------
This is a **partial** PSF module. It implements only the slice of PSF behaviour
that the geographic bounding-box calculation in :mod:`geometry` actually needs:

1. the analytic CERES PSF value :func:`psf_weight` (ATBD Eq. 4.4-1/4.4-2), and
2. the **95%-energy angular extent** of that PSF (:func:`psf_95_energy_extent`),
   i.e. how far the footprint reaches in the along-scan (delta) and cross-scan
   (beta) angular directions before 5% of the response is left in the tails.


Why the CERES PSF (for now)
---------------------------
The Libera instrument PSF has not been measured/delivered yet, so we stand in the
heritage CERES analytic PSF, which the reference notebook
``instructions/CERES PSF PIPELINE-2.ipynb`` reproduces and validates against the
CERES ATBD. Everything CERES-specific lives in the clearly marked constants block
below so that swapping in the real Libera PSF is a single-file change.

References
----------
CERES ATBD v2.2, Section 4.4 "Convolution of Imager Cloud Properties with CERES
Footprint Point Spread Function":
https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# =============================================================================
# INSTRUMENT RESPONSE PARAMETERS  --  CERES STAND-IN
#
# TODO[LIBSDC-794]: Everything in THIS block is a CERES placeholder. When the
# measured Libera PSF lookup table and ground-measured FOV are available, replace
# this block ONLY. Nothing in geometry.py should need to change, because geometry
# consumes the PSF solely through the PSFAngularExtent returned by the functions
# below -- it never sees these constants directly.
#
# The CERES analytic PSF is defined in ATBD v2.2 Eq. 4.4-1 (the PSF P(delta',beta))
# and Eq. 4.4-2 (the detector time-response function F(xi)).
# https://ceres.larc.nasa.gov/documents/ATBD/pdf/r2_2/ceres-atbd2.2-s4.4.pdf
# =============================================================================

# Half-width of the CERES optical field-of-view hexagon, parameter "a" in
# ATBD Eq. 4.4-1. This sets the width of the PSF "core" before the response tail.
CERES_FOV_HALF_WIDTH_DEG: float = 0.65

# Radiometric centroid shift Delta-delta (ATBD Eq. 4.4-1). The PSF is evaluated
# at delta' = delta + Delta-delta: the optical axis (delta = 0) is offset from the
# PSF energy centroid because the detector keeps integrating as the scan moves on.
CERES_CENTROID_SHIFT_DEG: float = 1.0

# Bolometer time-response coefficients, ATBD Eq. 4.4-2. These describe how the
# detector keeps recording energy after the optical FOV has swept past a point on
# Earth, which is what makes the along-scan PSF asymmetric (a trailing tail).
_A1: float = 1.84205
_A2: float = -0.22502
_B1: float = 1.47034
_B2: float = 0.45904
_C1: float = 1.98412
_K2_EXP: float = 6.35465
_K2_FREQ: float = 1.90282
_K3_EXP: float = 4.61598
_K3_FREQ: float = 5.83072

# Libera radiometer field-of-view HALF-angle, in degrees.
#
# Source: the Libera instrument kernel
# ``libera_utils/data/spice/jpss4/libera_v01.instruments.ik.ti`` defines every
# radiometer (SW/SSW/LW/TOT) with ``FOV_SHAPE = 'CIRCLE'`` and
# ``FOV_REF_ANGLE = 1.0``. Per the SPICE getfov convention, for a CIRCLE/ANGLES
# field of view the reference angle is the half-angle (angular radius from the
# boresight to the cone edge):
# https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/getfov_c.html
#
# This is THE single source of truth for the Libera FOV across the pipeline.
#
# TODO[LIBSDC-601]: 1.0 deg is a PLACEHOLDER. The kernel itself flags two open
# questions: (a) whether FOV_REF_ANGLE is a half- or full-angle, and (b) that the
# ground-measured value is still pending. Confirm both, and ideally read this
# value directly from the IK via spiceypy.getfov instead of hard-coding it here.
LIBERA_FOV_HALFANGLE_DEG: float = 1.0


@dataclass(frozen=True)
class PSFAngularExtent:
    """Half-extents of the PSF response in the instrument angular frame (degrees).

    The footprint's angular reach is split into three numbers because the PSF is
    *not* a simple disc:

    * The along-scan (delta) extent is **asymmetric** -- the detector time-response
      tail (ATBD Eq. 4.4-2) trails the scan, so the response reaches further on one
      side of the centroid than the other. ``delta_back_deg`` and
      ``delta_front_deg`` are the (positive) half-extents toward -delta and +delta
      respectively.
    * The cross-scan (beta) extent is **symmetric** because the PSF obeys
      P(delta, beta) = P(delta, -beta) (ATBD Eq. 4.4-1), so a single
      ``beta_max_deg`` describes both sides.

    Attributes
    ----------
    delta_back_deg : float
        Half-extent toward negative along-scan angle (degrees, >= 0).
    delta_front_deg : float
        Half-extent toward positive along-scan angle (degrees, >= 0).
    beta_max_deg : float
        Symmetric cross-scan half-extent (degrees, >= 0).
    """

    delta_back_deg: float
    delta_front_deg: float
    beta_max_deg: float


def _f_response(xi: np.ndarray) -> np.ndarray:
    """Detector time-response function F(xi) -- CERES ATBD Eq. 4.4-2.

    Physically this is the integrated detector response to a step input as a
    function of the dimensionless time variable ``xi``. It captures the fact that a
    bolometer keeps recording energy after the optical FOV has swept past a point,
    because of its finite thermal response time -- the origin of the PSF's
    along-scan tail.

    Parameters
    ----------
    xi : np.ndarray
        Dimensionless time/angle variable (the distance into the response, in the
        same angular units as delta).

    Returns
    -------
    np.ndarray
        The response F(xi), same shape as ``xi``.
    """
    xi = np.asarray(xi, dtype=float)
    # Three additive terms from ATBD Eq. 4.4-2: a rising exponential approach to
    # unity (term1) plus two damped oscillations (term2, term3) that model the
    # bolometer's ringing as it thermally settles.
    term1 = 1.0 - (1.0 + _A1 + _A2) * np.exp(-_C1 * xi)
    term2 = np.exp(-_K2_EXP * xi) * (_A1 * np.cos(_K2_FREQ * xi) + _B1 * np.sin(_K2_FREQ * xi))
    term3 = np.exp(-_K3_EXP * xi) * (_A2 * np.cos(_K3_FREQ * xi) + _B2 * np.sin(_K3_FREQ * xi))
    return term1 + term2 + term3


def psf_weight(delta_deg: np.ndarray, beta_deg: np.ndarray) -> np.ndarray:
    """Evaluate the CERES PSF P(delta', beta) -- ATBD Eq. 4.4-1.

    This is a direct port of ``compute_psf_weight`` from cell 1 of the reference
    notebook. ``delta_deg`` is the *geometric* along-scan angle (e.g. what
    :mod:`geometry` produces by perturbing the cone angle); the radiometric
    centroid shift is applied internally as delta' = delta + Delta-delta, and the
    PSF is evaluated at delta'.

    The PSF has a flat-ish "core" region (set by the optical FOV half-width ``a``)
    convolved with the detector time response :func:`_f_response`, which produces a
    sharp leading edge and a long trailing tail. Cross-scan symmetry
    P(delta, beta) = P(delta, -beta) is enforced by taking ``abs(beta)``.

    Parameters
    ----------
    delta_deg : np.ndarray
        Along-scan angle(s) in degrees (geometric, before the centroid shift).
    beta_deg : np.ndarray
        Cross-scan angle(s) in degrees. Sign is ignored (symmetry).

    Returns
    -------
    np.ndarray
        PSF response, broadcast to the shape of ``delta_deg``/``beta_deg``.
        Zero outside the optical FOV in the cross-scan direction.
    """
    # Apply the radiometric centroid shift: the PSF equation is written in delta'.
    delta = np.asarray(delta_deg, dtype=float) + CERES_CENTROID_SHIFT_DEG
    beta_abs = np.abs(np.asarray(beta_deg, dtype=float))

    # Work on the common broadcast shape so scalars and arrays both behave.
    out_shape = np.broadcast(delta, beta_abs).shape
    P = np.zeros(out_shape, dtype=float)
    delta = np.broadcast_to(delta, out_shape)
    beta_abs = np.broadcast_to(beta_abs, out_shape)

    a = CERES_FOV_HALF_WIDTH_DEG

    # Outside +/- 2a in the cross-scan direction the PSF is identically zero: the
    # hexagonal FOV has no response there.
    valid = beta_abs <= (2.0 * a)
    if not np.any(valid):
        return P

    # The along-scan integration limits (df, db) depend on the cross-scan position
    # because the FOV is a hexagon: near the centre (|beta| < a) the chord is full
    # width [-a, a]; toward the edges (|beta| >= a) the chord narrows linearly.
    df = np.zeros(out_shape)
    db = np.zeros(out_shape)
    inner = valid & (beta_abs < a)
    outer = valid & (beta_abs >= a)
    df[inner] = -a
    db[inner] = a
    df[outer] = -2.0 * a + beta_abs[outer]
    db[outer] = 2.0 * a - beta_abs[outer]

    # Core region: the FOV is still over the point, response is F(xi) measured from
    # the leading edge df.
    core = valid & (delta >= df) & (delta < db)
    if np.any(core):
        P[core] = _f_response(delta[core] - df[core])

    # Tail region: the FOV has swept past (delta >= db). The remaining response is
    # the difference of the time-response evaluated from the leading and trailing
    # edges -- this is the asymmetric trailing tail.
    tail = valid & (delta >= db)
    if np.any(tail):
        P[tail] = _f_response(delta[tail] - df[tail]) - _f_response(delta[tail] - db[tail])

    return P


@lru_cache(maxsize=8)
def psf_95_energy_extent(
    energy_fraction: float = 0.95,
    step_deg: float = 0.02,
    delta_span_deg: float = 3.0,
    beta_span_deg: float = 1.4,
) -> PSFAngularExtent:
    """Angular half-extents enclosing ``energy_fraction`` of the PSF energy.

    This finds the smallest set of grid cells (by descending energy density) whose
    cumulative PSF energy reaches ``energy_fraction``, then reports how far that set
    reaches in delta and beta. It is the analogue of cell 9's ``compute_psf_95_mask``
    in the reference notebook, but it sizes the *extent* directly from a fine,
    fully-vectorised grid evaluation rather than from the slow per-bin
    ``dblquad`` pre-integration (which is only needed later for PSF-weighted
    aggregation, not for the bounding box).

    "Energy" here is P(delta, beta) * cos(beta): the cos(beta) factor is the
    spherical-area Jacobian from ATBD Eq. 4.4-18, weighting each angular cell by the
    solid angle it subtends. The grid step cancels in the cumulative fraction, so it
    only needs to be fine enough to resolve the contour.

    The result is cached (``lru_cache``) because, for a fixed PSF, the 95% extent is
    a static instrument property -- it is computed once and reused for every
    footprint.

    Parameters
    ----------
    energy_fraction : float, optional
        Fraction of total PSF energy the contour must enclose. Default 0.95
        (CERES heritage truncation level).
    step_deg : float, optional
        Angular grid resolution in degrees. Default 0.02 (well finer than the
        contour we are measuring).
    delta_span_deg, beta_span_deg : float, optional
        Half-widths of the evaluation grid in delta and beta. Defaults comfortably
        contain the CERES PSF support (beta is identically zero beyond
        +/- 2 * CERES_FOV_HALF_WIDTH_DEG = 1.3 deg).

    Returns
    -------
    PSFAngularExtent
        Asymmetric along-scan and symmetric cross-scan half-extents (degrees).
    """
    # Build a regular delta-beta grid and evaluate the PSF energy density on it.
    delta_axis = np.arange(-delta_span_deg, delta_span_deg + step_deg, step_deg)
    beta_axis = np.arange(-beta_span_deg, beta_span_deg + step_deg, step_deg)
    delta_grid, beta_grid = np.meshgrid(delta_axis, beta_axis)

    # Energy density per cell: PSF value times the solid-angle Jacobian cos(beta).
    density = psf_weight(delta_grid, beta_grid) * np.cos(np.radians(beta_grid))

    # Select the highest-density cells until we have accumulated energy_fraction of
    # the total. For a unimodal PSF this set is a single connected blob, so its
    # delta/beta min/max give the contour extent.
    flat = density.ravel()
    order = np.argsort(flat)[::-1]  # cells from most to least energetic
    cumulative = np.cumsum(flat[order])
    total = cumulative[-1]
    cutoff_index = int(np.searchsorted(cumulative, energy_fraction * total))

    mask = np.zeros_like(flat, dtype=bool)
    mask[order[: cutoff_index + 1]] = True
    mask = mask.reshape(density.shape)

    delta_in_contour = delta_grid[mask]
    beta_in_contour = beta_grid[mask]

    # max(0, ...) guards against a degenerate contour that does not straddle delta=0.
    return PSFAngularExtent(
        delta_back_deg=float(max(0.0, -delta_in_contour.min())),
        delta_front_deg=float(max(0.0, delta_in_contour.max())),
        beta_max_deg=float(np.abs(beta_in_contour).max()),
    )


def static_fov_extent(fov_halfangle_deg: float = LIBERA_FOV_HALFANGLE_DEG) -> PSFAngularExtent:
    """Angular extent of the uniform "static" FOV used when the scanner is stationary.

    When the cone-angle rate is zero (scan turnarounds), there is no time-response
    tail -- the instrument simply integrates uniformly over its circular optical
    field of view (design doc section 2.4.2.2). The extent is then just the FOV
    half-angle in every direction.

    Parameters
    ----------
    fov_halfangle_deg : float, optional
        FOV half-angle in degrees. Defaults to :data:`LIBERA_FOV_HALFANGLE_DEG`.

    Returns
    -------
    PSFAngularExtent
        Symmetric extent equal to the FOV half-angle on all three axes.
    """
    return PSFAngularExtent(
        delta_back_deg=fov_halfangle_deg,
        delta_front_deg=fov_halfangle_deg,
        beta_max_deg=fov_halfangle_deg,
    )


def conservative_along_scan_extent(extent: PSFAngularExtent) -> float:
    """Return a single, scan-direction-independent along-scan half-extent.

    The true along-scan footprint is asymmetric and which side is "leading" depends
    on the scan direction (the sign of the cone-angle rate). In the current L1B
    product the cone-angle rate is not populated, and for a *bounding box* (a safe
    superset used only to decide which data tiles to load) we do not need the exact
    asymmetry anyway. Taking the larger of the two half-extents guarantees the box
    encloses the footprint regardless of scan direction.

    TODO[LIBSDC-794]: once the cone-angle rate is available and the directional
    pixel-projection step is implemented, the orchestrator can switch to the true
    (asymmetric) front/back extents instead of this conservative maximum.

    Parameters
    ----------
    extent : PSFAngularExtent
        The PSF angular extent.

    Returns
    -------
    float
        ``max(delta_back_deg, delta_front_deg)`` in degrees.
    """
    return max(extent.delta_back_deg, extent.delta_front_deg)
