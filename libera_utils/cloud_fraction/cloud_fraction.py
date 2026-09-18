"""Placeholder camera cloud-fraction core algorithm.

This module holds the core of the Camera Cloud Fraction (CF-CAM) product family:
given the record axis of a camera cloud-fraction product, it builds the per-record variable arrays
declared by ``cf_cam.yml`` / ``cf_cam_camtime.yml``.

Two builders share the same random-draw core but differ by timescale:

* :func:`generate_placeholder_cloud_fraction_camtime` -- camera timescale (CF-CAM-CAMTIME). Given the
  camera pseudo-footprints produced by
  :func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`, it reuses the
  *exact* FMATCH camera-timescale grid assembly
  (:func:`~libera_utils.footprint_matching.product.build_camtime_grid` /
  :func:`~libera_utils.footprint_matching.product.build_camtime_provenance`) so the product's
  ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid and ``camera_pixel_{x,y}_{min,max}`` provenance match
  FMATCH-CAM-CAMTIME record-for-record.
* :func:`generate_placeholder_cloud_fraction_radiometer` -- radiometer timescale (CF-CAM). Given the
  ``RADIOMETER_TIME`` coordinate of an L1B radiometer file, it emits one value per footprint on the
  1-D ``RADIOMETER_TIME`` axis, matching FMATCH-CAM.

The cloud-fraction values are a **placeholder**: a reproducible random, in-range value per record for
``cloud_fraction`` and ``cloud_fraction_standard_deviation``, pending the real Libera WFOV
cloud-fraction retrieval. Everything else (the record axis and, for the camera timescale, the
pixel-block provenance) is derived deterministically from the input, exactly as FMATCH derives it, so
each CF product aligns 1:1 with its FMATCH sibling built from the same L1B file.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from libera_utils.footprint_matching.product import (
    build_camtime_grid,
    build_camtime_provenance,
    build_camtime_pseudofootprint_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from libera_utils.footprint_matching.types import PseudoFootprint
    from libera_utils.io.product_definition import LiberaDataProductDefinition

logger = logging.getLogger(__name__)

# Product-definition variable names (must match cf_cam.yml / cf_cam_camtime.yml).
CLOUD_FRACTION_VARIABLE = "cloud_fraction"
CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE = "cloud_fraction_standard_deviation"

# Name of the radiometer-timescale record coordinate (CF-CAM), matching FMATCH-CAM.
RADIOMETER_TIME_COORDINATE = "RADIOMETER_TIME"

# Cloud fraction is reported in percent over [0, 100]; the placeholder draws uniformly across the
# whole valid range. The standard-deviation placeholder is capped well below the full range so the
# values read as a plausible within-footprint spread rather than noise spanning 0-100%.
_CLOUD_FRACTION_MIN_PERCENT: float = 0.0
_CLOUD_FRACTION_MAX_PERCENT: float = 100.0
_DEFAULT_MAX_STANDARD_DEVIATION_PERCENT: float = 15.0

# Fixed default seed so the placeholder output is reproducible across runs and in tests. Uses the
# numpy Generator API (not the stdlib ``random`` module) so the draw is not flagged as a security
# concern (Bandit S311).
_DEFAULT_RANDOM_SEED: int = 20260611


def _draw_placeholder_values(
    n_records: int,
    rng: np.random.Generator | None,
    max_standard_deviation_percent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_records`` reproducible placeholder ``(cloud_fraction, standard_deviation)`` values.

    Parameters
    ----------
    n_records : int
        Number of records to draw one value each for. Must be positive.
    rng : numpy.random.Generator or None
        Random generator; a generator seeded with :data:`_DEFAULT_RANDOM_SEED` is used when None.
    max_standard_deviation_percent : float
        Inclusive upper bound of the uniformly-drawn standard-deviation values, in percent.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(cloud_fraction, cloud_fraction_standard_deviation)``, both ``float32`` of length
        ``n_records`` and in percent.

    Raises
    ------
    ValueError
        If ``max_standard_deviation_percent`` is outside ``[0, 100]``.
    """
    if not 0.0 <= max_standard_deviation_percent <= _CLOUD_FRACTION_MAX_PERCENT:
        raise ValueError(
            f"max_standard_deviation_percent must be within [0, 100], got {max_standard_deviation_percent}."
        )
    if rng is None:
        rng = np.random.default_rng(_DEFAULT_RANDOM_SEED)

    cloud_fraction = rng.uniform(_CLOUD_FRACTION_MIN_PERCENT, _CLOUD_FRACTION_MAX_PERCENT, size=n_records).astype(
        np.float32
    )
    cloud_fraction_standard_deviation = rng.uniform(0.0, max_standard_deviation_percent, size=n_records).astype(
        np.float32
    )
    return cloud_fraction, cloud_fraction_standard_deviation


def generate_placeholder_cloud_fraction_camtime(
    footprints: Sequence[PseudoFootprint],
    definition: LiberaDataProductDefinition,
    *,
    rng: np.random.Generator | None = None,
    max_standard_deviation_percent: float = _DEFAULT_MAX_STANDARD_DEVIATION_PERCENT,
) -> dict[str, np.ndarray]:
    """Build the CF-CAM-CAMTIME variable arrays from camera pseudo-footprints.

    Reuses the FMATCH camera-timescale grid assembly so the returned ``CAMERA_TIME`` / ``PSEUDOFOOTPRINT``
    axes and the four ``camera_pixel_{x,y}_{min,max}`` provenance coordinates are identical to
    FMATCH-CAM-CAMTIME built from the same footprints. The two placeholder cloud-fraction variables are
    drawn per footprint and scattered onto the same rectangular grid (padded cells take ``NaN``).

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in segmentation (write) order, as returned by
        :func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
    definition : LiberaDataProductDefinition
        The parsed CF-CAM-CAMTIME product definition (declares the ``PSEUDOFOOTPRINT`` and
        ``camera_pixel_*`` coordinate dtypes/fills honoured by the grid assembly).
    rng : numpy.random.Generator, optional
        Random generator used to draw the placeholder values. Defaults to a generator seeded with
        :data:`_DEFAULT_RANDOM_SEED`, so repeated calls on the same footprints are reproducible.
    max_standard_deviation_percent : float, optional
        Upper bound (inclusive) of the uniformly-drawn ``cloud_fraction_standard_deviation`` values,
        in percent. Defaults to ``15.0``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping keyed by the ``cf_cam_camtime.yml`` variable/coordinate names, ready to hand to
        :func:`libera_utils.io.netcdf.write_libera_data_product`:

        * ``CAMERA_TIME`` -- ``datetime64[ns]``, unique and sorted;
        * ``PSEUDOFOOTPRINT`` -- ``int32`` 0-based subsection index;
        * ``camera_pixel_{x,y}_{min,max}`` -- ``int32`` inclusive bounds on the 2-D grid;
        * ``cloud_fraction`` / ``cloud_fraction_standard_deviation`` -- ``float32`` percent on the grid.

    Raises
    ------
    ValueError
        If ``footprints`` is empty (there would be no ``CAMERA_TIME`` axis to write), or if
        ``max_standard_deviation_percent`` is outside ``[0, 100]``.
    """
    footprints = list(footprints)
    if not footprints:
        raise ValueError("Cannot build a CF-CAM-CAMTIME product from zero pseudo-footprints.")

    n_footprints = len(footprints)
    logger.info("Generating placeholder cloud fraction for %d camera pseudo-footprints", n_footprints)

    cloud_fraction, cloud_fraction_standard_deviation = _draw_placeholder_values(
        n_footprints, rng, max_standard_deviation_percent
    )

    # Reuse the exact FMATCH ragged->rectangular grid and provenance so records align 1:1.
    grid = build_camtime_grid(footprints)
    data = build_camtime_provenance(footprints, grid, definition, time_variable="CAMERA_TIME")

    # Write the COMPLETE pseudo-footprint record (geolocation, viewing geometry, PSF bounding box and
    # quality flags), not just cloud fraction: FMATCH-CAM-CAMTIME reconstructs its footprints from this
    # product (see product.pseudofootprints_from_camtime_dataset) rather than re-segmenting the L1B image.
    data.update(build_camtime_pseudofootprint_columns(footprints, grid, definition))

    # Scatter the flat per-footprint placeholder cloud-fraction columns onto the same grid; padded cells
    # take NaN (matching how FMATCH pads its float32 camera-timescale variables).
    data[CLOUD_FRACTION_VARIABLE] = grid.to_grid(cloud_fraction, np.dtype(np.float32), np.nan)
    data[CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE] = grid.to_grid(
        cloud_fraction_standard_deviation, np.dtype(np.float32), np.nan
    )
    return data


def generate_placeholder_cloud_fraction_radiometer(
    radiometer_time: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    max_standard_deviation_percent: float = _DEFAULT_MAX_STANDARD_DEVIATION_PERCENT,
) -> dict[str, np.ndarray]:
    """Build the CF-CAM variable arrays on the radiometer-time record axis.

    Parameters
    ----------
    radiometer_time : numpy.ndarray
        The 1-D ``RADIOMETER_TIME`` coordinate (datetime64) of an L1B RAD-4CH file, one entry per
        radiometer footprint, as returned under ``"RADIOMETER_TIME"`` by
        :func:`libera_utils.footprint_matching._runner_common.load_l1b_radiometer_inputs`.
    rng : numpy.random.Generator, optional
        Random generator used to draw the placeholder values. Defaults to a generator seeded with
        :data:`_DEFAULT_RANDOM_SEED`.
    max_standard_deviation_percent : float, optional
        Upper bound (inclusive) of the uniformly-drawn ``cloud_fraction_standard_deviation`` values,
        in percent. Defaults to ``15.0``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping keyed by the ``cf_cam.yml`` variable/coordinate names:

        * ``RADIOMETER_TIME`` -- ``datetime64[ns]``;
        * ``cloud_fraction`` / ``cloud_fraction_standard_deviation`` -- ``float32`` percent.

    Raises
    ------
    ValueError
        If ``radiometer_time`` is empty (no record axis to write), or if
        ``max_standard_deviation_percent`` is outside ``[0, 100]``.
    """
    radiometer_time = np.asarray(radiometer_time)
    n_records = radiometer_time.shape[0]
    if n_records == 0:
        raise ValueError("Cannot build a CF-CAM product from zero radiometer footprints.")

    logger.info("Generating placeholder cloud fraction for %d radiometer footprints", n_records)

    cloud_fraction, cloud_fraction_standard_deviation = _draw_placeholder_values(
        n_records, rng, max_standard_deviation_percent
    )

    return {
        RADIOMETER_TIME_COORDINATE: radiometer_time.astype("datetime64[ns]"),
        CLOUD_FRACTION_VARIABLE: cloud_fraction,
        CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE: cloud_fraction_standard_deviation,
    }
