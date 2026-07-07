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

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.config import config
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    from collections.abc import Sequence

    from xarray import Dataset

    from libera_utils.footprint_matching.camera_segmentation import PseudoFootprint
    from libera_utils.io.filenaming import LiberaDataProductFilename

# Product-definition variable names that the camera-segmentation tool fills with
# *real* per-footprint values (centre-pixel geolocation/geometry, the corner-derived
# PSF bounding box, and the QA flags). Every other declared variable belongs to the
# not-yet-implemented aggregation / derived-geometry engines (see
# :func:`aggregate_external_variables` / :func:`compute_derived_viewing_geometry`)
# and is written as a conformant placeholder for now. Kept as one set so the
# assembly and its tests agree on exactly which variables are "real" this milestone.
_CAMTIME_SEGMENTATION_VARIABLES: frozenset[str] = frozenset(
    {
        "latitude",
        "longitude",
        "altitude",
        "solar_zenith_angle",
        "viewing_zenith_angle",
        "relative_azimuth_angle",
        "psf_bbox_lat_min",
        "psf_bbox_lat_max",
        "psf_bbox_lon_min",
        "psf_bbox_lon_max",
        "q_flags",
    }
)

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

    Every output variable is named ``<source_key>_<instrument>_<spec_name>`` for
    provenance, where the instrument token comes from the reader's ``INSTRUMENT``
    attribute (e.g. ``era5_ECMWF_wind_u10``, ``igbp_MODIS_surface_type``,
    ``cldpix_NOAA20_cloud_mask``), matching the product definition variable names.

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

    Produces the ``sunglint_angle`` variable present in every FMATCH product
    definition. The intended (CERES/SSF-heritage) formula, with all angles in
    degrees:

    - Sun glint angle: the angle between the sensor view direction and the
      specular reflection of the solar beam; small values indicate potential
      sun glint contamination.

    (The ``scattering_angle`` quantity was previously emitted here too, but has
    been dropped from the FMATCH product contract; downstream code that needs it
    can derive it on demand from the geolocation angles that remain in every
    product.)

    Parameters
    ----------
    solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle : np.ndarray
        Per-footprint geolocation angles in degrees.

    Returns
    -------
    dict[str, np.ndarray]
        ``{"sunglint_angle": ...}``.

    Raises
    ------
    NotImplementedError
        Always, in this milestone. The geometry module is future work.
    """
    # TODO[LIBSDC-785]: implement the sun-glint angle calculation.
    raise NotImplementedError(
        "Derived viewing-geometry computation is not implemented yet. This is a "
        "placeholder for the FMATCH geometry module (future milestone)."
    )


def assemble_fmatch_dataset(
    mode: OperationalMode,
    *args: Any,
    cloud_fraction_camera: np.ndarray | None = None,
    **kwargs: Any,
) -> Dataset:
    """Assemble a conformant FMATCH :class:`xarray.Dataset` for an operational mode.

    Combines the per-footprint geolocation inputs, the derived viewing geometry from
    :func:`compute_derived_viewing_geometry`, and the aggregated external
    variables from :func:`aggregate_external_variables` into the variable dict
    expected by the mode's product definition (from :func:`load_fmatch_definition`),
    then builds a Dataset via ``LiberaDataProductDefinition.create_product_dataset``
    and brings it into conformance with ``enforce_dataset_conformance``.

    Only the camera-timescale CAM product (``CAM_CAMTIME``) is implemented in this
    milestone; it is assembled from the camera pseudo-footprints produced by
    :func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
    The other modes remain future work and raise ``NotImplementedError``.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being assembled.
    *args, **kwargs
        Mode-specific inputs. For ``CAM_CAMTIME`` the first positional argument is
        the sequence of :class:`PseudoFootprint` objects; see
        :func:`_assemble_camtime_dataset` for the accepted keyword arguments.
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint cloud fraction from the Camera Cloud Fraction (CF-CAM)
        algorithm (Libera WFOV camera), as a 1-D array indexed by footprint in
        the same order as the time coordinate. This is an *internal* algorithm
        output - it does not come from a reader and is already aggregated to one
        value per footprint - so it is merged directly into the ``cloud_fraction_camera``
        variable rather than going through :func:`aggregate_external_variables`.
        Only the CAM modes (``CAM``, ``CAM_CAMTIME``) declare this variable; it is
        ``None`` for the IMAGER modes.

    Raises
    ------
    NotImplementedError
        For every mode except ``CAM_CAMTIME`` in this milestone.
    """
    if mode is OperationalMode.CAM_CAMTIME:
        return _assemble_camtime_dataset(*args, cloud_fraction_camera=cloud_fraction_camera, **kwargs)

    # TODO[LIBSDC-785]: assemble the per-footprint Dataset for the remaining modes.
    raise NotImplementedError(
        f"FMATCH dataset assembly is not implemented yet for mode {mode.value}. Only "
        f"{OperationalMode.CAM_CAMTIME.value} is supported in this milestone."
    )


def _placeholder_variable_array(variable_definition: Any, n_footprints: int) -> np.ndarray:
    """Build a conformant placeholder array for a not-yet-computed product variable.

    Variables owned by the aggregation / derived-geometry engines (not built yet)
    still have to appear in the output file with the right dtype and shape so the
    product conforms to its definition. We fill them with the variable's declared
    ``_FillValue`` when it has one, and otherwise with ``NaN`` for floating-point
    variables or ``0`` for integer variables. The magnitudes are meaningless; only
    the dtype/shape/attributes form the product contract (the same stance the
    example-product generator takes in ``scripts/generate_fmatch_example_products.py``).

    Parameters
    ----------
    variable_definition : LiberaVariableDefinition
        The product-definition entry for the variable.
    n_footprints : int
        Length of the footprint (``CAMERA_TIME``) axis.

    Returns
    -------
    np.ndarray
        A 1-D array of length ``n_footprints`` of the variable's declared dtype.
    """
    dtype = np.dtype(variable_definition.dtype)
    fill_value = variable_definition.attributes.get("_FillValue")
    if fill_value is None:
        # No declared fill: NaN reads as "missing" for floats; 0 is the neutral
        # integer stand-in (integers cannot represent NaN).
        fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
    return np.full(n_footprints, fill_value, dtype=dtype)


def _normalize_longitude(longitude_deg: float) -> float:
    """Wrap a longitude into [-180, 180).

    Corner-derived bounding boxes that straddle the antimeridian can report a
    ``lon_max`` greater than 180 (the :class:`BoundingBox` dateline convention).
    The product definition's ``psf_bbox_lon_*`` variables declare a [-180, 180]
    valid range, so we wrap the stored bounds back into that convention. Downstream
    consumers can still detect a dateline-crossing box because it then has
    ``lon_min > lon_max``.
    """
    return (longitude_deg + 180.0) % 360.0 - 180.0


def _assemble_camtime_dataset(
    footprints: Sequence[PseudoFootprint],
    *,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
) -> Dataset:
    """Assemble the FMATCH-CAM-CAMTIME Dataset from camera pseudo-footprints.

    Builds the per-footprint variable arrays declared by the FMATCH-CAM-CAMTIME
    product definition. The centre-pixel geolocation/geometry, the corner-derived
    PSF bounding box, and the QA flags come straight from the pseudo-footprints
    (:data:`_CAMTIME_SEGMENTATION_VARIABLES`); every other declared variable is a
    conformant placeholder pending the aggregation / derived-geometry engines
    (``TODO[LIBSDC-785]``).

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in write order, as returned by
        :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
    definition : LiberaDataProductDefinition, optional
        The FMATCH-CAM-CAMTIME product definition. Loaded via
        :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute
        (typically the source L1B camera filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). When
        omitted the ``cloud_fraction_camera`` variable is written as a placeholder.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the FMATCH-CAM-CAMTIME definition.

    Raises
    ------
    ValueError
        If ``footprints`` is empty (there would be no time axis to write).
    """
    if definition is None:
        definition = load_fmatch_definition(OperationalMode.CAM_CAMTIME)

    footprints = list(footprints)
    if not footprints:
        raise ValueError("Cannot assemble a FMATCH-CAM-CAMTIME product from zero pseudo-footprints.")
    n_footprints = len(footprints)

    time_variable = fmatch_time_variable(OperationalMode.CAM_CAMTIME)  # "CAMERA_TIME"

    # The real, segmentation-derived columns. Longitudes of the PSF box are wrapped
    # into [-180, 180) to satisfy the product definition's valid range.
    real_columns: dict[str, list[float]] = {
        "latitude": [f.latitude for f in footprints],
        "longitude": [f.longitude for f in footprints],
        "altitude": [f.altitude for f in footprints],
        "solar_zenith_angle": [f.solar_zenith_angle for f in footprints],
        "viewing_zenith_angle": [f.viewing_zenith_angle for f in footprints],
        "relative_azimuth_angle": [f.relative_azimuth_angle for f in footprints],
        "psf_bbox_lat_min": [f.bbox.lat_min for f in footprints],
        "psf_bbox_lat_max": [f.bbox.lat_max for f in footprints],
        "psf_bbox_lon_min": [_normalize_longitude(f.bbox.lon_min) for f in footprints],
        "psf_bbox_lon_max": [_normalize_longitude(f.bbox.lon_max) for f in footprints],
        "q_flags": [int(f.q_flags) for f in footprints],
    }

    # Start the data dict with the time coordinate (nanosecond datetimes; note the
    # values repeat within an image -- see segment_l1b_camera's docstring).
    data: dict[str, np.ndarray] = {
        time_variable: np.array([f.time for f in footprints], dtype="datetime64[ns]"),
    }

    # Cast each real column to the exact dtype the definition declares.
    for name, values in real_columns.items():
        data[name] = np.asarray(values, dtype=np.dtype(definition.variables[name].dtype))

    # Optional internal (non-reader) Camera Cloud Fraction values.
    if cloud_fraction_camera is not None:
        data["cloud_fraction_camera"] = np.asarray(
            cloud_fraction_camera, dtype=np.dtype(definition.variables["cloud_fraction_camera"].dtype)
        )

    # Every remaining declared variable is a placeholder until its engine exists.
    for name, variable_definition in definition.variables.items():
        if name not in data:
            data[name] = _placeholder_variable_array(variable_definition, n_footprints)

    # Build the Dataset and bring it into conformance. Dynamic (per-run) global
    # attributes are set directly, mirroring run_scene_id_cam.py; they are declared
    # (as null) in the definition, so enforce_dataset_conformance keeps them.
    dataset = definition.create_product_dataset(data)
    dataset = definition.enforce_dataset_conformance(dataset)
    dataset.attrs["date_created"] = datetime.now(UTC).isoformat()
    if input_files is not None:
        dataset.attrs["input_files"] = input_files
    if algorithm_version is not None:
        dataset.attrs["algorithm_version"] = algorithm_version

    return dataset


def write_fmatch_product(mode: OperationalMode, *args: Any, **kwargs: Any) -> Any:
    """Write a FMATCH NetCDF data product to disk for an operational mode.

    Delegates to ``libera_utils.io.netcdf.write_libera_data_product`` using the
    definition from :func:`load_fmatch_definition`, the assembled Dataset from
    :func:`assemble_fmatch_dataset`, and ``time_variable=fmatch_time_variable(mode)``
    (``RADIOMETER_TIME`` or ``CAMERA_TIME``) so the output filename encodes the
    footprint time span.

    Only ``CAM_CAMTIME`` is implemented in this milestone; the other modes remain
    future work.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    *args, **kwargs
        Mode-specific inputs forwarded to the writer. For ``CAM_CAMTIME`` see
        :func:`_write_camtime_product`.

    Raises
    ------
    NotImplementedError
        For every mode except ``CAM_CAMTIME`` in this milestone.
    """
    if mode is OperationalMode.CAM_CAMTIME:
        return _write_camtime_product(*args, **kwargs)

    # TODO[LIBSDC-785]: wire assembly + write for the remaining modes.
    raise NotImplementedError(
        f"FMATCH product writing is not implemented yet for mode {mode.value}. Only "
        f"{OperationalMode.CAM_CAMTIME.value} is supported in this milestone."
    )


def _write_camtime_product(
    footprints: Sequence[PseudoFootprint],
    output_path: str | Path,
    *,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    strict: bool = True,
) -> LiberaDataProductFilename:
    """Assemble and write the FMATCH-CAM-CAMTIME NetCDF product.

    Loads the product definition once, assembles the pseudo-footprints into a
    conformant Dataset via :func:`_assemble_camtime_dataset`, and writes it with
    ``write_libera_data_product`` (which generates the standardized Libera filename
    from the ``CAMERA_TIME`` span).

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in write order.
    output_path : str or pathlib.Path
        Directory (or S3 prefix) to write the product file into.
    algorithm_version : str, optional
        Value for the ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the ``input_files`` global attribute.
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV).
    strict : bool, optional
        When True (default), fail if the assembled Dataset does not conform.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.
    """
    definition = load_fmatch_definition(OperationalMode.CAM_CAMTIME)
    dataset = _assemble_camtime_dataset(
        footprints,
        definition=definition,
        algorithm_version=algorithm_version,
        input_files=input_files,
        cloud_fraction_camera=cloud_fraction_camera,
    )
    return write_libera_data_product(
        data_product_definition=definition,
        data=dataset,
        output_path=output_path,
        time_variable=fmatch_time_variable(OperationalMode.CAM_CAMTIME),
        strict=strict,
    )
