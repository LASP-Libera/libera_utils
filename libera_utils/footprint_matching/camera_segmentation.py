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
   **center pixel** -- the pseudo-footprint's stand-in for the radiometer
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
* the center pixel itself fill -> the nearest valid pixel in the block is
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
    bounding_box_from_points_batch,
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

# Per-pixel geolocation fields reduced to the footprint's center-pixel scalars.
# We use the plain (non terrain-corrected) latitude/longitude to match the
# "Footprint boresight" convention used elsewhere in the FMATCH pipeline
# (see notebooks/generate_example_products.ipynb).
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
# cannot be estimated (e.g. an image whose center pixels are all fill). Chosen so a
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
        The geometric center pixel was off-Earth (fill), so the nearest valid pixel
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
        Pixel indices of the footprint's center (the boresight stand-in), after any
        nearest-valid-pixel substitution.
    latitude, longitude : float
        center-pixel geodetic latitude/longitude, degrees.
    altitude : float
        center-pixel altitude, metres (as stored in L1B).
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : float
        center-pixel viewing-geometry angles, degrees.
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


# A block reduced to everything a PseudoFootprint needs *except* the geographic box.
# The box is deferred so every block's box can be assembled in one batched pyproj call
# (see segment_l1b_camera / bounding_box_from_points_batch): the corner/pole geodesic
# distances that bounding_box_from_points computes per block are the dominant camera-
# segmentation cost, and pyproj evaluates them far faster over one big array than block
# by block. ``corner_lats``/``corner_lons`` are the block's valid corner pixels padded
# to four by repeating a valid corner -- duplicates leave the box min/max/span/pole-reach
# unchanged, so padding is exact and lets every block share one fixed-width batch.
_CORNERS_PER_BLOCK: int = 4


@dataclass(frozen=True)
class _BlockRecord:
    """One segmented block minus its bounding box (assembled in a later batch)."""

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
    corner_lats: np.ndarray
    corner_lons: np.ndarray
    q_flags: CameraFootprintQualityFlag


@lru_cache(maxsize=1)
def _wgs84_geod() -> Geod:
    """WGS84 :class:`pyproj.Geod` for ellipsoidal surface distances (created once)."""
    return Geod(ellps="WGS84")


def _is_valid(value: float) -> bool:
    """Return True if a geolocation value is finite and not the L1B fill sentinel."""
    return math.isfinite(value) and value != L1B_FILL_VALUE


def _valid_mask(array: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_is_valid` over a whole array (finite and not fill).

    The elementwise equivalent of :func:`_is_valid`, used to build the per-image
    validity grid in one pass instead of a Python-level ``np.vectorize`` loop.
    """
    array = np.asarray(array, dtype=float)
    return np.isfinite(array) & (array != L1B_FILL_VALUE)


def _estimate_ground_sampling_distance_km(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    """Estimate the ground distance between adjacent pixels, in km.

    We sample near the image center (where the grid is most likely populated and
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
    # Compare the center pixel with its +x and +y neighbors where those exist.
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

    Prefers the geometric center of the block. If that pixel is off-Earth (fill), the
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

    # center pixel is fill: find the valid pixel in the block closest to the geometric
    # center by Chebyshev (grid) distance. Vectorized over the block's valid pixels;
    # np.nonzero yields them in row-major (ix asc, iy asc) order and np.argmin returns
    # the first minimum, which reproduces the original nested-loop tie-break exactly
    # (smallest ix, then smallest iy, among equidistant pixels).
    sub_valid = valid[x0 : x1 + 1, y0 : y1 + 1]
    local_ix, local_iy = np.nonzero(sub_valid)
    if local_ix.size == 0:
        return None
    abs_ix = local_ix + x0
    abs_iy = local_iy + y0
    distance = np.maximum(np.abs(abs_ix - cx), np.abs(abs_iy - cy))
    nearest = int(np.argmin(distance))
    return int(abs_ix[nearest]), int(abs_iy[nearest]), True


def _segment_image(
    time: np.datetime64,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    alt2d: np.ndarray,
    sza2d: np.ndarray,
    vza2d: np.ndarray,
    raa2d: np.ndarray,
) -> list[_BlockRecord]:
    """Segment a single camera image into per-block records (boxes assembled later).

    Parameters
    ----------
    time : np.datetime64
        The image's ``CAMERA_TIME``.
    lat2d, lon2d, alt2d, sza2d, vza2d, raa2d : np.ndarray
        2-D ``(x, y)`` per-pixel fields for this image.

    Returns
    -------
    list[_BlockRecord]
        One entry per non-empty pixel block (empty/all-fill blocks are dropped). The
        geographic box is deferred to :func:`segment_l1b_camera`, which assembles every
        block's box across the whole product in one batched pyproj call.
    """
    nx, ny = lat2d.shape

    # A pixel is usable only if BOTH its lat and lon are valid; either being fill
    # means the pixel did not intersect the Earth. Vectorized over the whole image
    # grid (see _valid_mask) rather than a per-pixel np.vectorize loop.
    valid = _valid_mask(lat2d) & _valid_mask(lon2d)

    gsd_km = _estimate_ground_sampling_distance_km(lat2d, lon2d)
    block = _block_size_pixels(gsd_km)

    records: list[_BlockRecord] = []
    for slice_x, slice_y in _iter_blocks(nx, ny, block):
        record = _block_record(time, slice_x, slice_y, valid, lat2d, lon2d, alt2d, sza2d, vza2d, raa2d)
        if record is not None:
            records.append(record)
    return records


def _block_record(
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
) -> _BlockRecord | None:
    """Reduce a pixel block to a :class:`_BlockRecord`, or ``None`` if it has no footprint.

    Applies the off-Earth policy: all corners fill -> drop; some corners fill ->
    partial-coverage flag; center pixel fill -> nearest-valid substitution flag. Does
    no pyproj work -- the block's valid corner pixels are gathered (and padded to
    :data:`_CORNERS_PER_BLOCK`) so the box is assembled later in one batch.
    """
    q_flags = CameraFootprintQualityFlag(0)

    # --- Valid corner pixels (the box's boundary points) --------------------
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

    # --- center pixel (boresight stand-in), with nearest-valid substitution -
    center = _select_center_pixel(slice_x, slice_y, valid)
    if center is None:
        # No valid pixel anywhere in the block (should not happen if a corner was
        # valid, but guard defensively rather than emit a footprint with no anchor).
        return None
    center_ix, center_iy, substituted = center
    if substituted:
        q_flags |= CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED

    # Pad the valid corners to a fixed width by repeating the first valid corner. The
    # box's lat/lon min/max, longitude span, and pole-reach are all order-independent
    # reductions that duplicates cannot change, so this is exact and lets every block
    # feed one fixed-width batched box assembly (bounding_box_from_points_batch).
    corner_lats_arr = np.asarray(corner_lats, dtype=float)
    corner_lons_arr = np.asarray(corner_lons, dtype=float)
    if corner_lats_arr.size < _CORNERS_PER_BLOCK:
        pad = _CORNERS_PER_BLOCK - corner_lats_arr.size
        corner_lats_arr = np.concatenate([corner_lats_arr, np.full(pad, corner_lats_arr[0])])
        corner_lons_arr = np.concatenate([corner_lons_arr, np.full(pad, corner_lons_arr[0])])

    return _BlockRecord(
        time=time,
        slice_x=slice_x,
        slice_y=slice_y,
        center_ix=center_ix,
        center_iy=center_iy,
        latitude=float(lat2d[center_ix, center_iy]),
        longitude=float(lon2d[center_ix, center_iy]),
        altitude=float(alt2d[center_ix, center_iy]),
        solar_zenith_angle=float(sza2d[center_ix, center_iy]),
        viewing_zenith_angle=float(vza2d[center_ix, center_iy]),
        relative_azimuth_angle=float(raa2d[center_ix, center_iy]),
        corner_lats=corner_lats_arr,
        corner_lons=corner_lons_arr,
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

    The per-block corner and center selections are pure index math; the one geodesic
    cost -- the corner/pole distances that turn a block's corners into a
    :class:`~libera_utils.footprint_matching.types.BoundingBox` -- is deferred and run
    for *every* block of *every* image in a single batched pyproj call
    (:func:`~libera_utils.footprint_matching.geometry.bounding_box_from_points_batch`)
    rather than block by block.

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

    # Segment every image into boxless block records first (pure index/mask math), in
    # (image, block) write order.
    records: list[_BlockRecord] = []
    for t in range(n_images):
        records.extend(_segment_image(times[t], lat[t], lon[t], alt[t], sza[t], vza[t], raa[t]))

    if not records:
        log.info("Produced 0 pseudo-footprint(s) from %d image(s)", n_images)
        return []

    # Assemble every block's bounding box in one batched pyproj call. Camera corner
    # boxes are never limb-truncated (the pixels are already on the surface), so
    # truncated=False for all. The record order is preserved, so boxes align 1:1.
    center_lats = np.fromiter((r.latitude for r in records), dtype=float, count=len(records))
    center_lons = np.fromiter((r.longitude for r in records), dtype=float, count=len(records))
    corner_lats = np.stack([r.corner_lats for r in records])
    corner_lons = np.stack([r.corner_lons for r in records])
    truncated = np.zeros(len(records), dtype=bool)
    boxes = bounding_box_from_points_batch(center_lats, center_lons, corner_lats, corner_lons, truncated)

    footprints = [
        PseudoFootprint(
            time=record.time,
            slice_x=record.slice_x,
            slice_y=record.slice_y,
            center_ix=record.center_ix,
            center_iy=record.center_iy,
            latitude=record.latitude,
            longitude=record.longitude,
            altitude=record.altitude,
            solar_zenith_angle=record.solar_zenith_angle,
            viewing_zenith_angle=record.viewing_zenith_angle,
            relative_azimuth_angle=record.relative_azimuth_angle,
            bbox=bbox,
            q_flags=record.q_flags,
        )
        for record, bbox in zip(records, boxes, strict=True)
    ]

    log.info("Produced %d pseudo-footprint(s) from %d image(s)", len(footprints), n_images)
    return footprints
