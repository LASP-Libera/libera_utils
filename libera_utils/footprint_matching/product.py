"""FMATCH-CAM data product assembly and writing.

This module is the seam between the footprint-matching *engine* (readers, PSF
aggregation, geometry) and the Libera *data-product* machinery
(``LiberaDataProductDefinition`` / ``write_libera_data_product``). It owns the
FMATCH-CAM product definition and the eventual flow that turns a day of matched
footprints into a conformant NetCDF file.

Milestone scope
---------------
Only :func:`load_fmatch_cam_definition` is implemented. The aggregation,
geometry, assembly, and write functions are intentionally **stubs** that raise
``NotImplementedError`` - they document the intended pipeline so the product
definition has an obvious home and its future producers are visible, but the
actual computation is deferred to later milestones (see the design doc:
``instructions/documentation/Footprint Matching and Scene ID PDF``).

Why a thin seam here
--------------------
The product definition (``libera_utils/data/product_definitions/fmatch_cam.yml``)
is the contract every downstream consumer (Scene ID, Camera Cloud Fraction)
reads against. Keeping the loader next to the (future) writer means there is a
single place that knows how a FMATCH-CAM file is produced, while the reader
plugins stay decoupled from product I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from libera_utils.config import config
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    import numpy as np
    from xarray import Dataset

    from libera_utils.footprint_matching.types import OperationalMode

# Filename of the FMATCH-CAM product definition within the configured
# LIBERA_PRODUCT_DEFINITIONS_PATH directory. Kept as a module constant so tests
# and callers reference one source of truth rather than hard-coding the string.
FMATCH_CAM_DEFINITION_FILENAME = "fmatch_cam.yml"

# Name of the coordinate/dimension that identifies each footprint by its
# radiometer observation time. This is the ``time_variable`` handed to
# ``write_libera_data_product`` for start/end-time filename generation.
FMATCH_CAM_TIME_VARIABLE = "RADIOMETER_TIME"


def load_fmatch_cam_definition() -> LiberaDataProductDefinition:
    """Load and validate the FMATCH-CAM product definition.

    Resolves ``fmatch_cam.yml`` under the configured product-definitions
    directory and parses it into a validated :class:`LiberaDataProductDefinition`.

    Returns
    -------
    LiberaDataProductDefinition
        The validated FMATCH-CAM product definition, ready for use with
        ``create_product_dataset`` / ``enforce_dataset_conformance`` /
        ``check_dataset_conformance``.

    Notes
    -----
    The directory is read from ``config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")``
    so packaging/test overrides are honored, matching how L1A product
    definitions are resolved elsewhere in the codebase.
    """
    definitions_dir = Path(str(config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")))
    definition_path = definitions_dir / FMATCH_CAM_DEFINITION_FILENAME
    return LiberaDataProductDefinition.from_yaml(definition_path)


def aggregate_external_variables(
    mode: OperationalMode,
    *args: Any,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    """Aggregate every active reader's gridded data to one value per footprint.

    For the given operational mode this will select the active readers via
    ``ReaderRegistry.get_readers_for_mode(mode)``, load the tiles overlapping
    each footprint, and apply each variable's PSF-weighted aggregation strategy
    (weighted mean / mode / log-mean) to collapse the fine-resolution pixels to a
    single value per footprint. The returned dict is keyed by the variable names
    declared in ``fmatch_cam.yml`` (``surface_type``, ``sea_ice_concentration``,
    ``wind_u10``, ``wind_v10``, ``cloud_fraction``, ``cloud_optical_thickness``,
    ``cloud_top_pressure``).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of aggregated-variable name to a 1-D array indexed by footprint.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. The PSF/aggregation engine is future work.
    """
    # TODO[LIBSDC-785]: implement PSF-weighted aggregation over active readers.
    raise NotImplementedError(
        "External-variable aggregation is not implemented yet. This is a placeholder "
        "for the FMATCH PSF aggregation engine (future milestone)."
    )


def compute_derived_viewing_geometry(
    solar_zenith_angle: np.ndarray,
    viewing_zenith_angle: np.ndarray,
    relative_azimuth_angle: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute derived viewing-geometry variables from the geolocation angles.

    Produces the ``scattering_angle`` and ``sunglint_angle`` variables defined in
    ``fmatch_cam.yml``. The intended (CERES/SSF-heritage) formulas, with all
    angles in degrees:

    - Scattering angle ``Theta``::

          cos(Theta) = -cos(SZA) * cos(VZA)
                       + sin(SZA) * sin(VZA) * cos(RAA)

    - Sun glint angle: the angle between the sensor view direction and the
      specular reflection of the solar beam; small values indicate potential
      sun glint contamination.

    Parameters
    ----------
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : np.ndarray
        Per-footprint geolocation angles in degrees.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"scattering_angle": ..., "sunglint_angle": ...}``.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. The geometry module is future work.
    """
    # TODO[LIBSDC-785]: implement scattering and sun-glint angle calculations.
    raise NotImplementedError(
        "Derived viewing-geometry computation is not implemented yet. This is a "
        "placeholder for the FMATCH geometry module (future milestone)."
    )


def assemble_fmatch_cam_dataset(*args: Any, **kwargs: Any) -> Dataset:
    """Assemble a conformant FMATCH-CAM :class:`xarray.Dataset`.

    Will combine the L1B geolocation inputs, the derived viewing geometry from
    :func:`compute_derived_viewing_geometry`, and the aggregated external
    variables from :func:`aggregate_external_variables` into the variable dict
    expected by the product definition, then build a Dataset via
    ``LiberaDataProductDefinition.create_product_dataset`` and bring it into
    conformance with ``enforce_dataset_conformance``.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. Dataset assembly is future work.
    """
    # TODO[LIBSDC-785]: assemble the per-footprint Dataset from all inputs.
    raise NotImplementedError(
        "FMATCH-CAM dataset assembly is not implemented yet. This is a placeholder "
        "for the footprint-matching orchestrator (future milestone)."
    )


def write_fmatch_cam_product(*args: Any, **kwargs: Any) -> Any:
    """Write a FMATCH-CAM NetCDF data product to disk.

    Will delegate to ``libera_utils.io.netcdf.write_libera_data_product`` using
    the definition from :func:`load_fmatch_cam_definition`, the assembled Dataset
    from :func:`assemble_fmatch_cam_dataset`, and
    ``time_variable=FMATCH_CAM_TIME_VARIABLE`` so the output filename encodes the
    footprint time span.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. The writer entry point is future work.
    """
    # TODO[LIBSDC-785]: wire assembly + write_libera_data_product together.
    raise NotImplementedError(
        "FMATCH-CAM product writing is not implemented yet. This is a placeholder "
        "for the footprint-matching orchestrator entry point (future milestone)."
    )
