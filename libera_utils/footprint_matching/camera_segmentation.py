"""Segment L1B Daily Camera images into radiometer-sized pseudo-footprints.

Purpose
-------
The camera-timescale FMATCH products (``FMATCH-CAM-CAMTIME``,
``FMATCH-IMAGER-CAMTIME``) are built on the *camera* image cadence rather than the
radiometer observation times. The L1B Daily Camera product is a stack of 2-D
images: for each ``CAMERA_TIME`` there is a ``CAMERA_PIXEL_COUNT_X`` x
``CAMERA_PIXEL_COUNT_Y`` grid of geolocated pixels (see
``l1b_example/l1b_cam.yml``). That per-pixel grid is far finer than a radiometer
footprint, so before the footprint-matching machinery can treat the camera data
"as if" it were a set of radiometer footprints, each image must be **segmented
into pseudo-footprints**.

A *pseudo-footprint* is a contiguous block of camera pixels, sized so that its
extent on the ground is comparable to a real radiometer footprint (design doc
sections 1.3.2 and 2.3.3: "footprints are derived for all data available within
each image frame"). Every block becomes one per-footprint record on the
``CAMERA_TIME`` axis of the FMATCH-CAM-CAMTIME product, exactly like a radiometer
footprint is one record on the ``RADIOMETER_TIME`` axis of FMATCH-CAM.

What this module computes for each pseudo-footprint
---------------------------------------------------
1. A geographic **bounding box** built *only from the four corner pixels* of the
   block. The camera pixels are already geolocated (unlike the radiometer, which
   needs the viewing-geometry ray-trace in :mod:`geometry`), so the box that
   encloses the block is just the lat/lon envelope of its corners -- we do not
   need to inspect every interior pixel. Pole/dateline handling is delegated to
   :func:`geometry.bounding_box_from_points`, the same assembler the radiometer
   path uses, so both products get identical edge-case behaviour.
2. The per-footprint scalar geolocation/geometry (latitude, longitude, altitude,
   solar/viewing zenith and relative azimuth angles) taken from the block's
   **centre pixel** -- the pseudo-footprint's stand-in for the radiometer
   boresight.

Off-Earth pixels
----------------
Pixels that did not intersect the Earth (space or calibration views) are stored
with the L1B fill sentinel :data:`~libera_utils.footprint_matching.geometry.L1B_FILL_VALUE`
(``-999``). Corners can therefore be fill even when the block straddles the Earth's
limb. Following the chosen policy:

* some (but not all) corners fill -> the box is shrunk to the *valid* corners and
  the footprint is flagged :data:`CameraFootprintQualityFlag.PARTIAL_COVERAGE`;
* all four corners fill -> there is effectively no footprint, so the block is
  dropped entirely;
* the centre pixel itself fill -> the nearest valid pixel in the block is
  substituted for the boresight and the footprint is flagged
  :data:`CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED`.

References
----------
* Design doc: ``instructions/documentation/Footprint Matching and Scene ID PDF``,
  sections 1.3.2 (timescale dimension) and 2.3.3 (FMATCH-CAM-CAMTIME).
* L1B Camera product definition: ``l1b_example/l1b_cam.yml``.

"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from pyproj import Geod

if TYPE_CHECKING:
    import xarray as xr

from libera_utils.footprint_matching.geometry import (
    L1B_FILL_VALUE,
    NOMINAL_ALTITUDE_KM,
    bounding_box_from_points,
)
from libera_utils.footprint_matching.psf import LIBERA_FOV_HALFANGLE_DEG
from libera_utils.footprint_matching.types import BoundingBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# L1B Camera variable / dimension names (from l1b_example/l1b_cam.yml).
# Centralised here so the reader never hard-codes them inline -- if the L1B
# product definition renames a field, it changes in exactly one place.
# ---------------------------------------------------------------------------
CAMERA_TIME_NAME: str = "CAMERA_TIME"
PIXEL_X_DIM: str = "CAMERA_PIXEL_COUNT_X"
PIXEL_Y_DIM: str = "CAMERA_PIXEL_COUNT_Y"

# Per-pixel geolocation fields reduced to the footprint's centre-pixel scalars.
# We use the plain (non terrain-corrected) latitude/longitude to match the
# "Footprint boresight" convention used elsewhere in the FMATCH pipeline
# (see scripts/generate_fmatch_example_products.py).
LATITUDE_NAME: str = "Latitude"
LONGITUDE_NAME: str = "Longitude"
ALTITUDE_NAME: str = "Altitude"
SOLAR_ZENITH_NAME: str = "Solar_Zenith_Surface"
VIEWING_ZENITH_NAME: str = "Viewing_Zenith_Surface"
RELATIVE_AZIMUTH_NAME: str = "Relative_Azimuth_Surface"

# Target on-the-ground diameter of one pseudo-footprint, in km. We size the pixel
# blocks so their ground extent approximates a real radiometer footprint. Rather
# than hard-code a number, we derive it from the single sources of truth already in
# the codebase: the radiometer FOV half-angle and the nominal orbit altitude. At
# nadir a footprint's ground radius is altitude * tan(half-angle), so the diameter
# is 2 * altitude * tan(half-angle) (~29 km for 1.0 deg at 835 km). Deriving it this
# way means a change to the FOV or the nominal altitude propagates automatically.
# TODO[LIBSDC-794]: read the true footprint size from mission config once available.
TARGET_FOOTPRINT_DIAMETER_KM: float = 2.0 * NOMINAL_ALTITUDE_KM * math.tan(math.radians(LIBERA_FOV_HALFANGLE_DEG))

# Fallback ground-sampling distance (km per pixel) used only when the per-image GSD
# cannot be estimated (e.g. an image whose centre pixels are all fill). Chosen so a
# block is at least a handful of pixels; it only affects degenerate images.
_FALLBACK_GSD_KM: float = 1.0


class CameraFootprintQualityFlag(enum.IntFlag):
    """Bitwise quality flags for a camera pseudo-footprint.

    Stored in the FMATCH-CAM-CAMTIME ``q_flags`` variable. ``IntFlag`` lets the
    flags be OR-combined and tested bitwise, and an empty (zero) value means "no
    issues".

    Attributes
    ----------
    PARTIAL_COVERAGE
        At least one -- but not all -- corner pixels of the block were off-Earth
        (fill), so the bounding box was shrunk to the valid corners and covers only
        part of the nominal block.
    CENTER_PIXEL_SUBSTITUTED
        The geometric centre pixel was off-Earth (fill), so the nearest valid pixel
        in the block was substituted as the footprint boresight.
    """

    PARTIAL_COVERAGE = 0b0001
    CENTER_PIXEL_SUBSTITUTED = 0b0010


@dataclass(frozen=True)
class PseudoFootprint:
    """One camera pseudo-footprint: a pixel block reduced to a footprint record.

    Attributes
    ----------
    time : np.datetime64
        The ``CAMERA_TIME`` of the image this footprint came from. All footprints
        segmented from the same image share this timestamp (see the module note on
        ``CAMERA_TIME`` non-uniqueness).
    slice_x, slice_y : slice
        The block's extent in the ``CAMERA_PIXEL_COUNT_X`` / ``CAMERA_PIXEL_COUNT_Y``
        pixel grid. Kept for provenance and for the (future) PSF-weighted
        aggregation of the block's pixels.
    center_ix, center_iy : int
        Pixel indices of the footprint's centre (the boresight stand-in), after any
        nearest-valid-pixel substitution.
    latitude, longitude : float
        Centre-pixel geodetic latitude/longitude, degrees.
    altitude : float
        Centre-pixel altitude, metres (as stored in L1B).
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : float
        Centre-pixel viewing-geometry angles, degrees.
    bbox : BoundingBox
        Geographic box enclosing the block's valid corner pixels, with pole/dateline
        handling from :func:`geometry.bounding_box_from_points`.
    q_flags : CameraFootprintQualityFlag
        Bitwise quality flags for this footprint (0 == clean).
    """

    time: np.datetime64
    slice_x: slice
    slice_y: slice
    center_ix: int
    center_iy: int
    latitude: float
    longitude: float
    altitude: float
    solar_zenith_angle: float
    viewing_zenith_angle: float
    relative_azimuth_angle: float
    bbox: BoundingBox
    q_flags: CameraFootprintQualityFlag


@lru_cache(maxsize=1)
def _wgs84_geod() -> Geod:
    """WGS84 :class:`pyproj.Geod` for ellipsoidal surface distances (created once)."""
    return Geod(ellps="WGS84")


def _is_valid(value: float) -> bool:
    """Return True if a geolocation value is finite and not the L1B fill sentinel."""
    return math.isfinite(value) and value != L1B_FILL_VALUE


def _estimate_ground_sampling_distance_km(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    """Estimate the ground distance between adjacent pixels, in km.

    We sample near the image centre (where the grid is most likely populated and
    least distorted) and measure the geodesic distance to the neighboring pixel in
    each grid direction, averaging whatever samples are valid. This tells us how many
    pixels span the target footprint diameter.

    Parameters
    ----------
    lat2d, lon2d : np.ndarray
        2-D ``(x, y)`` latitude/longitude grids for a single image, degrees.

    Returns
    -------
    float
        Mean adjacent-pixel ground distance in km, or :data:`_FALLBACK_GSD_KM` if no
        valid adjacent pair could be measured.
    """
    geod = _wgs84_geod()
    nx, ny = lat2d.shape
    cx, cy = nx // 2, ny // 2

    distances_km: list[float] = []
    # Compare the centre pixel with its +x and +y neighbours where those exist.
    for dx, dy in ((1, 0), (0, 1)):
        ax, ay = cx, cy
        bx, by = min(cx + dx, nx - 1), min(cy + dy, ny - 1)
        if (ax, ay) == (bx, by):
            continue
        lat_a, lon_a, lat_b, lon_b = lat2d[ax, ay], lon2d[ax, ay], lat2d[bx, by], lon2d[bx, by]
        if not (_is_valid(lat_a) and _is_valid(lon_a) and _is_valid(lat_b) and _is_valid(lon_b)):
            continue
        # Geod.inv returns (forward_azimuth, back_azimuth, distance_m).
        _, _, distance_m = geod.inv(lon_a, lat_a, lon_b, lat_b)
        distances_km.append(distance_m / 1000.0)

    if not distances_km:
        return _FALLBACK_GSD_KM
    return float(np.mean(distances_km))


def _block_size_pixels(gsd_km: float) -> int:
    """Number of pixels per block side to approximate the target footprint diameter.

    ``block = round(target_diameter / gsd)``, clamped to at least 1 pixel so a
    degenerate (very coarse) image still yields single-pixel footprints rather than
    an empty result.
    """
    if gsd_km <= 0.0:
        gsd_km = _FALLBACK_GSD_KM
    return max(1, int(round(TARGET_FOOTPRINT_DIAMETER_KM / gsd_km)))


def _iter_blocks(nx: int, ny: int, block: int) -> list[tuple[slice, slice]]:
    """Tile an ``(nx, ny)`` pixel grid into contiguous ``block`` x ``block`` slices.

    Edge blocks that do not divide evenly are simply smaller (the last row/column of
    blocks may be a partial block). Returns the slices in row-major order.
    """
    blocks: list[tuple[slice, slice]] = []
    for x0 in range(0, nx, block):
        for y0 in range(0, ny, block):
            blocks.append((slice(x0, min(x0 + block, nx)), slice(y0, min(y0 + block, ny))))
    return blocks


def _corner_indices(slice_x: slice, slice_y: slice) -> list[tuple[int, int]]:
    """Return the four corner pixel indices of a block, as ``(ix, iy)`` tuples."""
    x0, x1 = slice_x.start, slice_x.stop - 1
    y0, y1 = slice_y.start, slice_y.stop - 1
    # Deduplicate so a 1-pixel-wide block does not report the same corner twice.
    corners = {(x0, y0), (x0, y1), (x1, y0), (x1, y1)}
    return sorted(corners)


def _select_center_pixel(slice_x: slice, slice_y: slice, valid: np.ndarray) -> tuple[int, int, bool] | None:
    """Pick the block's boresight pixel, substituting the nearest valid one if needed.

    Prefers the geometric centre of the block. If that pixel is off-Earth (fill), the
    nearest valid pixel (by Chebyshev/grid distance) within the block is used instead.

    Parameters
    ----------
    slice_x, slice_y : slice
        The block extent in the pixel grid.
    valid : np.ndarray
        2-D boolean grid (whole image) marking pixels with valid geolocation.

    Returns
    -------
    tuple[int, int, bool] or None
        ``(center_ix, center_iy, substituted)``; ``None`` if the block has no valid
        pixel at all.
    """
    x0, x1 = slice_x.start, slice_x.stop - 1
    y0, y1 = slice_y.start, slice_y.stop - 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2

    if valid[cx, cy]:
        return cx, cy, False

    # Centre pixel is fill: search the block for the valid pixel closest to the
    # geometric centre. The block is small (a few pixels on a side), so a direct
    # scan is cheap and clearer than anything fancier.
    best: tuple[int, int] | None = None
    best_distance = math.inf
    for ix in range(x0, x1 + 1):
        for iy in range(y0, y1 + 1):
            if not valid[ix, iy]:
                continue
            distance = max(abs(ix - cx), abs(iy - cy))
            if distance < best_distance:
                best, best_distance = (ix, iy), distance

    if best is None:
        return None
    return best[0], best[1], True


def _segment_image(
    time: np.datetime64,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    alt2d: np.ndarray,
    sza2d: np.ndarray,
    vza2d: np.ndarray,
    raa2d: np.ndarray,
) -> list[PseudoFootprint]:
    """Segment a single camera image into pseudo-footprints.

    Parameters
    ----------
    time : np.datetime64
        The image's ``CAMERA_TIME``.
    lat2d, lon2d, alt2d, sza2d, vza2d, raa2d : np.ndarray
        2-D ``(x, y)`` per-pixel fields for this image.

    Returns
    -------
    list[PseudoFootprint]
        One entry per non-empty pixel block (empty/all-fill blocks are dropped).
    """
    nx, ny = lat2d.shape

    # A pixel is usable only if BOTH its lat and lon are valid; either being fill
    # means the pixel did not intersect the Earth.
    valid = np.vectorize(_is_valid)(lat2d) & np.vectorize(_is_valid)(lon2d)

    gsd_km = _estimate_ground_sampling_distance_km(lat2d, lon2d)
    block = _block_size_pixels(gsd_km)

    footprints: list[PseudoFootprint] = []
    for slice_x, slice_y in _iter_blocks(nx, ny, block):
        footprint = _build_footprint(time, slice_x, slice_y, valid, lat2d, lon2d, alt2d, sza2d, vza2d, raa2d)
        if footprint is not None:
            footprints.append(footprint)
    return footprints


def _build_footprint(
    time: np.datetime64,
    slice_x: slice,
    slice_y: slice,
    valid: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    alt2d: np.ndarray,
    sza2d: np.ndarray,
    vza2d: np.ndarray,
    raa2d: np.ndarray,
) -> PseudoFootprint | None:
    """Build one pseudo-footprint from a pixel block, or ``None`` if it has no footprint.

    Applies the off-Earth policy: all corners fill -> drop; some corners fill ->
    partial-coverage flag; centre pixel fill -> nearest-valid substitution flag.
    """
    q_flags = CameraFootprintQualityFlag(0)

    # --- Bounding box from the (valid) corner pixels only -------------------
    corner_lats: list[float] = []
    corner_lons: list[float] = []
    n_corners = 0
    for ix, iy in _corner_indices(slice_x, slice_y):
        n_corners += 1
        if valid[ix, iy]:
            corner_lats.append(float(lat2d[ix, iy]))
            corner_lons.append(float(lon2d[ix, iy]))

    # All corners off-Earth => there is effectively no footprint here; drop it.
    if not corner_lats:
        return None
    # Some corners off-Earth => the box only covers part of the nominal block.
    if len(corner_lats) < n_corners:
        q_flags |= CameraFootprintQualityFlag.PARTIAL_COVERAGE

    # --- Centre pixel (boresight stand-in), with nearest-valid substitution -
    center = _select_center_pixel(slice_x, slice_y, valid)
    if center is None:
        # No valid pixel anywhere in the block (should not happen if a corner was
        # valid, but guard defensively rather than emit a footprint with no anchor).
        return None
    center_ix, center_iy, substituted = center
    if substituted:
        q_flags |= CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED

    center_lat = float(lat2d[center_ix, center_iy])
    center_lon = float(lon2d[center_ix, center_iy])

    # The box assembler is shared with the radiometer path; passing the corners as
    # the boundary points gives us identical pole/dateline handling. Camera corner
    # boxes are never limb-truncated (the pixels are already on the surface), so
    # truncated=False.
    bbox = bounding_box_from_points(center_lat, center_lon, corner_lats, corner_lons, truncated=False)

    return PseudoFootprint(
        time=time,
        slice_x=slice_x,
        slice_y=slice_y,
        center_ix=center_ix,
        center_iy=center_iy,
        latitude=center_lat,
        longitude=center_lon,
        altitude=float(alt2d[center_ix, center_iy]),
        solar_zenith_angle=float(sza2d[center_ix, center_iy]),
        viewing_zenith_angle=float(vza2d[center_ix, center_iy]),
        relative_azimuth_angle=float(raa2d[center_ix, center_iy]),
        bbox=bbox,
        q_flags=q_flags,
    )


def segment_l1b_camera(dataset: xr.Dataset, *, log: logging.Logger | None = None) -> list[PseudoFootprint]:
    """Segment an L1B Daily Camera dataset into camera pseudo-footprints.

    Iterates every camera image (``CAMERA_TIME``) and segments its pixel grid into
    radiometer-sized pseudo-footprints, returning them as a single flat list in
    (image, block) order -- the order in which they will be written to the
    FMATCH-CAM-CAMTIME product.

    Note on ``CAMERA_TIME`` uniqueness: because an image is segmented into *many*
    pseudo-footprints, all footprints from one image share that image's
    ``CAMERA_TIME``. The resulting product therefore has repeated ``CAMERA_TIME``
    values (unlike a strictly-increasing radiometer timeline). This is intentional
    for the camera timescale -- the time identifies the source image, not a unique
    footprint.

    TODO[LIBSDC-794]: this scalar, per-block loop is written for clarity. To meet the
    real-time latency budget it can be vectorised over the pixel grid (the corner and
    centre selections are all pure index math).

    Parameters
    ----------
    dataset : xarray.Dataset
        An open L1B Daily Camera dataset conforming to ``l1b_example/l1b_cam.yml``
        (dimensions ``CAMERA_TIME`` x ``CAMERA_PIXEL_COUNT_X`` x
        ``CAMERA_PIXEL_COUNT_Y``).
    log : logging.Logger, optional
        Logger for progress messages. Defaults to this module's logger.

    Returns
    -------
    list[PseudoFootprint]
        All pseudo-footprints across all images, in write order.
    """
    log = log or logger

    # Pull the fields we need into numpy up front. Dimension order in the L1B product
    # is (CAMERA_TIME, PIXEL_X, PIXEL_Y), so indexing [t] yields the 2-D image grid.
    times = np.asarray(dataset[CAMERA_TIME_NAME].values)
    lat = np.asarray(dataset[LATITUDE_NAME].values, dtype=float)
    lon = np.asarray(dataset[LONGITUDE_NAME].values, dtype=float)
    alt = np.asarray(dataset[ALTITUDE_NAME].values, dtype=float)
    sza = np.asarray(dataset[SOLAR_ZENITH_NAME].values, dtype=float)
    vza = np.asarray(dataset[VIEWING_ZENITH_NAME].values, dtype=float)
    raa = np.asarray(dataset[RELATIVE_AZIMUTH_NAME].values, dtype=float)

    n_images = times.shape[0]
    log.info("Segmenting %d camera image(s) into pseudo-footprints", n_images)

    footprints: list[PseudoFootprint] = []
    for t in range(n_images):
        image_footprints = _segment_image(times[t], lat[t], lon[t], alt[t], sza[t], vza[t], raa[t])
        footprints.extend(image_footprints)

    log.info("Produced %d pseudo-footprint(s) from %d image(s)", len(footprints), n_images)
    return footprints
