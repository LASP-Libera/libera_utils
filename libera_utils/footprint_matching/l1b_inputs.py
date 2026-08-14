"""Reading the L1B Daily inputs that the FMATCH products are built on.

Every FMATCH product starts from an L1B Daily file, and which one depends on the
mode's timescale:

* **Radiometer-timescale** modes (``CAM``, ``IMAGER_FLASH``, ``IMAGER``) start from
  the L1B Daily *radiometer* product (``RAD-4CH``). A handful of its per-footprint
  columns are not computed by footprint matching at all - they are carried through
  to the FMATCH product verbatim (see :data:`L1B_PASSTHROUGH_VARIABLES`). This is
  the "(a) Geolocation inputs (from L1B Daily)" block at the top of every
  ``fmatch_*.yml``.
* **Camera-timescale** modes (``CAM_CAMTIME``, ``IMAGER_CAMTIME``) start from the
  L1B Daily *camera* product (``CAM``), whose pixel grid is segmented into
  radiometer-sized pseudo-footprints by
  :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.

Why this lives in the package
-----------------------------
The pass-through logic originated in the example-product generator (now
``notebooks/generate_example_products.ipynb``), but a notebook is not installed with
``libera_utils`` and is not copied into the algorithm container images (the Dockerfiles
``COPY libera_utils`` only). Production runners therefore need it inside the package;
the example-product notebook now imports it from here so there is a single
implementation of "what FMATCH takes from L1B".

See Also
--------
libera_utils.footprint_matching.product : Assembles these inputs into a product Dataset.
libera_utils.footprint_matching._runner : Calls these readers from the manifest-driven runners.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping of FMATCH product variable name -> L1B RAD-4CH variable name, for the
# per-footprint quantities FMATCH copies straight out of L1B rather than computing.
#
# Viewing angles: the FMATCH solar/viewing zenith and relative azimuth map to the
# L1B "_Surface" angles (geodetic angles at the Earth point), whose units (degrees)
# and ranges line up with the FMATCH definition - in particular L1B
# ``Relative_Azimuth_Surface`` spans [0, 360], matching relative_azimuth_angle's
# valid_range. Note we do NOT pass through the derived ``sunglint_angle``; that is a
# computed product variable (see product.compute_derived_viewing_geometry), not an
# L1B input.
#
# Every variable listed here is declared float32 in the product definition, which is
# why the reader below can cast them all to float32 generically. The L1B time
# coordinate is handled separately because it maps to the product's time
# *coordinate* (RADIOMETER_TIME), not a data variable.
L1B_PASSTHROUGH_VARIABLES: dict[str, str] = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "solar_zenith_angle": "Solar_Zenith_Surface",
    "viewing_zenith_angle": "Viewing_Zenith_Surface",
    "relative_azimuth_angle": "Relative_Azimuth_Surface",
}

# Name of the time coordinate variable inside the L1B radiometer file. xarray decodes
# its CF "nanoseconds since 1958-01-01" units into datetime64[ns], which is exactly
# the dtype the FMATCH RADIOMETER_TIME coordinate declares.
L1B_TIME_VARIABLE: str = "radiometer_time"

# Name of the FMATCH product's radiometer time coordinate (the key this module
# returns the decoded L1B times under).
FMATCH_RADIOMETER_TIME_COORDINATE: str = "RADIOMETER_TIME"


def load_l1b_radiometer_inputs(l1b_file: Path) -> dict[str, np.ndarray]:
    """Read the per-footprint L1B inputs that FMATCH passes through verbatim.

    Pulls every quantity the FMATCH product contract takes straight from L1B Daily:
    the ``RADIOMETER_TIME`` coordinate plus all variables in
    :data:`L1B_PASSTHROUGH_VARIABLES` (footprint ``latitude``/``longitude`` and the
    solar/viewing zenith and relative-azimuth angles).

    Footprints with non-finite values are dropped. The L1B geolocation and angles are
    NaN wherever the boresight has no valid Earth intersection (e.g. the first samples
    of a file, and any gaps), and such rows carry no usable values. We keep only
    footprints where *every* pass-through variable is finite - a logical AND of the
    per-variable finite masks. In practice the geolocation and the "_Surface" angles
    share the same gaps, but AND-ing all of them is robust if they ever diverge.

    Parameters
    ----------
    l1b_file : pathlib.Path
        Path to a local L1B RAD-4CH NetCDF file. Remote inputs must be materialized
        locally first (see :class:`libera_utils.footprint_matching._runner._as_local_path`), because
        :func:`xarray.open_dataset` seeks within the file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping keyed by FMATCH variable name: ``"RADIOMETER_TIME"`` (datetime64[ns])
        plus each key of :data:`L1B_PASSTHROUGH_VARIABLES` (float32). All arrays are
        1-D and the same length.

    Raises
    ------
    ValueError
        If no footprint has finite values for every pass-through variable, which would
        leave nothing to build a product from.
    """
    # Open with default decoding so the CF-encoded time coordinate ("nanoseconds since
    # 1958-01-01") is decoded into datetime64[ns] for us.
    with xr.open_dataset(l1b_file) as l1b:
        radiometer_time = l1b[L1B_TIME_VARIABLE].values
        # Read every pass-through variable, keyed by its FMATCH (output) name.
        passthrough = {fmatch_name: l1b[l1b_name].values for fmatch_name, l1b_name in L1B_PASSTHROUGH_VARIABLES.items()}

    # Keep only footprints where every pass-through variable is finite. Start from an
    # all-True mask and AND in each variable's finite mask, so a NaN in ANY variable
    # drops that footprint.
    finite = np.ones(radiometer_time.shape, dtype=bool)
    for values in passthrough.values():
        finite &= np.isfinite(values)

    n_finite = int(finite.sum())
    if n_finite == 0:
        raise ValueError(
            f"No usable footprints in L1B file {l1b_file}: every record has a non-finite value in at least one of "
            f"the pass-through variables ({', '.join(sorted(L1B_PASSTHROUGH_VARIABLES))})."
        )
    if n_finite < finite.size:
        logger.info(
            "Dropped %d of %d L1B footprints with non-finite geolocation/viewing angles",
            finite.size - n_finite,
            finite.size,
        )

    radiometer_time = radiometer_time[finite]
    passthrough = {name: values[finite] for name, values in passthrough.items()}

    # Cast to the exact dtypes the FMATCH definition declares so conformance checking
    # passes without an auto-cast. Every pass-through variable is float32 in the
    # definition and the decoded time is datetime64[ns]; the casts are belt-and-braces
    # over already-correct dtypes.
    result: dict[str, np.ndarray] = {
        FMATCH_RADIOMETER_TIME_COORDINATE: radiometer_time.astype("datetime64[ns]"),
    }
    result.update({name: values.astype(np.float32) for name, values in passthrough.items()})
    return result


def load_l1b_camera_dataset(l1b_file: Path) -> xr.Dataset:
    """Open an L1B Daily Camera file for segmentation into pseudo-footprints.

    The camera-timescale FMATCH modes do not pass L1B columns through; they segment the
    camera pixel grid with
    :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`,
    which needs the whole dataset (geolocation grids, altitude, and viewing angles on
    the ``CAMERA_TIME`` x ``CAMERA_PIXEL_COUNT_X`` x ``CAMERA_PIXEL_COUNT_Y`` grid).

    The dataset is loaded eagerly into memory (``.load()``) and detached from the file
    handle, so callers may use it after the source file goes away - which matters when
    the input was materialized into a temporary directory by
    :class:`libera_utils.footprint_matching._runner._as_local_path`.

    Parameters
    ----------
    l1b_file : pathlib.Path
        Path to a local L1B CAM NetCDF file.

    Returns
    -------
    xarray.Dataset
        The opened, fully-loaded L1B camera dataset.
    """
    with xr.open_dataset(l1b_file) as l1b:
        return l1b.load()
