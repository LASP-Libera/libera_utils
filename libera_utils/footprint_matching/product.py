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
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    import numpy as np
    from xarray import Dataset

# Product definition YAML filename for each FMATCH operational mode. Every mode
# has its own SSF-style product definition (the mode *is* the product), and the
# active reader set / variables differ by mode. Kept as one source of truth so
# callers and tests never hard-code filenames.
FMATCH_DEFINITION_FILENAMES: dict[OperationalMode, str] = {
    OperationalMode.CAM: "fmatch_cam.yml",
    OperationalMode.CAM_CAMTIME: "fmatch_cam_camtime.yml",
    OperationalMode.IMAGER_FLASH: "fmatch_imager_flash.yml",
    OperationalMode.IMAGER: "fmatch_imager.yml",
    OperationalMode.IMAGER_CAMTIME: "fmatch_imager_camtime.yml",
}

# Camera-timescale modes index footprints by camera image time; all other modes
# index by radiometer observation time. This is the dimension/coordinate name and
# the ``time_variable`` handed to ``write_libera_data_product`` for filename
# start/end-time generation.
_CAMERA_TIMESCALE_MODES = frozenset({OperationalMode.CAM_CAMTIME, OperationalMode.IMAGER_CAMTIME})

# Back-compat aliases for the CAM product (the first one delivered).
FMATCH_CAM_DEFINITION_FILENAME = FMATCH_DEFINITION_FILENAMES[OperationalMode.CAM]
FMATCH_CAM_TIME_VARIABLE = "RADIOMETER_TIME"


def fmatch_time_variable(mode: OperationalMode) -> str:
    """Return the per-footprint time coordinate name for an operational mode.

    Camera-timescale modes (``CAM_CAMTIME``, ``IMAGER_CAMTIME``) use
    ``CAMERA_TIME``; all radiometer-timescale modes use ``RADIOMETER_TIME``.
    """
    return "CAMERA_TIME" if mode in _CAMERA_TIMESCALE_MODES else "RADIOMETER_TIME"


def load_fmatch_definition(mode: OperationalMode) -> LiberaDataProductDefinition:
    """Load and validate the FMATCH product definition for an operational mode.

    Resolves the mode's YAML under the configured product-definitions directory
    and parses it into a validated :class:`LiberaDataProductDefinition`.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode whose product definition to load.

    Returns
    -------
    LiberaDataProductDefinition
        The validated product definition, ready for use with
        ``create_product_dataset`` / ``enforce_dataset_conformance`` /
        ``check_dataset_conformance``.

    Notes
    -----
    The directory is read from ``config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")``
    so packaging/test overrides are honored, matching how L1A product
    definitions are resolved elsewhere in the codebase.
    """
    definitions_dir = Path(str(config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")))
    definition_path = definitions_dir / FMATCH_DEFINITION_FILENAMES[mode]
    return LiberaDataProductDefinition.from_yaml(definition_path)


def load_fmatch_cam_definition() -> LiberaDataProductDefinition:
    """Load and validate the FMATCH-CAM product definition.

    Thin convenience wrapper around :func:`load_fmatch_definition` for the
    lowest-latency CAM product (the first one delivered).

    Returns
    -------
    LiberaDataProductDefinition
        The validated FMATCH-CAM product definition.
    """
    return load_fmatch_definition(OperationalMode.CAM)


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
    single value per footprint. The active reader set - and therefore the keys of
    the returned dict - grows with the mode's latency (e.g. CAM has era5, igbp,
    nise, viirs_brdf, viirs_cloud; IMAGER additionally has ssf, cldpix, viirs_aod).

    Every output variable is prefixed with the reader source key for provenance
    (e.g. ``era5_wind_u10``, ``igbp_surface_type``, ``cldpix_cloud_mask``),
    matching the product definition variable names.

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

    Produces the ``scattering_angle`` and ``sunglint_angle`` variables present in
    every FMATCH product definition. The intended (CERES/SSF-heritage) formulas,
    with all angles in degrees:

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


def assemble_fmatch_dataset(mode: OperationalMode, *args: Any, **kwargs: Any) -> Dataset:
    """Assemble a conformant FMATCH :class:`xarray.Dataset` for an operational mode.

    Will combine the L1B geolocation inputs, the derived viewing geometry from
    :func:`compute_derived_viewing_geometry`, and the aggregated external
    variables from :func:`aggregate_external_variables` into the variable dict
    expected by the mode's product definition (from :func:`load_fmatch_definition`),
    then build a Dataset via ``LiberaDataProductDefinition.create_product_dataset``
    and bring it into conformance with ``enforce_dataset_conformance``.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. Dataset assembly is future work.
    """
    # TODO[LIBSDC-785]: assemble the per-footprint Dataset from all inputs.
    raise NotImplementedError(
        "FMATCH dataset assembly is not implemented yet. This is a placeholder "
        "for the footprint-matching orchestrator (future milestone)."
    )


def write_fmatch_product(mode: OperationalMode, *args: Any, **kwargs: Any) -> Any:
    """Write a FMATCH NetCDF data product to disk for an operational mode.

    Will delegate to ``libera_utils.io.netcdf.write_libera_data_product`` using
    the definition from :func:`load_fmatch_definition`, the assembled Dataset from
    :func:`assemble_fmatch_dataset`, and ``time_variable=fmatch_time_variable(mode)``
    (``RADIOMETER_TIME`` or ``CAMERA_TIME``) so the output filename encodes the
    footprint time span.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. The writer entry point is future work.
    """
    # TODO[LIBSDC-785]: wire assembly + write_libera_data_product together.
    raise NotImplementedError(
        "FMATCH product writing is not implemented yet. This is a placeholder "
        "for the footprint-matching orchestrator entry point (future milestone)."
    )
