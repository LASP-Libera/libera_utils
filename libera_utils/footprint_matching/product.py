"""FMATCH data product assembly and writing.

This module is the seam between the footprint-matching *engine* (readers, PSF
aggregation, geometry) and the Libera *data-product* machinery
(``LiberaDataProductDefinition`` / ``write_libera_data_product``). It owns the
FMATCH product definitions and the flow that turns matched footprints into a
conformant NetCDF file, for all five operational modes.

Milestone scope
---------------
Assembly and writing are wired for **every** operational mode, but the values
they carry are only partly computed:

* **Real** per-footprint values come from the L1B Daily inputs - the geolocation
  and viewing angles for the radiometer-timescale modes
  (:data:`_RADIOMETER_L1B_VARIABLES`), and the camera segmentation results for
  the camera-timescale modes (:data:`_CAMTIME_SEGMENTATION_VARIABLES`).
* **Placeholders** fill every other declared variable. Those belong to the PSF
  aggregation engine (:func:`aggregate_external_variables`) and the derived
  geometry (:func:`compute_derived_viewing_geometry`), which remain
  ``NotImplementedError`` stubs pending ``TODO[LIBSDC-785]``.

A placeholder is *structurally* correct (declared dtype, shape and attributes)
but numerically meaningless. Floating-point placeholders are filled with ``NaN``
and integer placeholders with ``0``; see :func:`_fill_placeholder_variables`.

Why a thin seam here
--------------------
The product definitions (``libera_utils/data/product_definitions/fmatch_*.yml``)
are the contract every downstream consumer (Scene ID, Camera Cloud Fraction, ADM
binning) reads against. Keeping the loaders next to the writers means there is a
single place that knows how a FMATCH file is produced, while the reader plugins
stay decoupled from product I/O.

See Also
--------
libera_utils.footprint_matching._runner : Manifest-driven runners that call into this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from libera_utils.config import config
from libera_utils.footprint_matching.l1b_inputs import L1B_PASSTHROUGH_VARIABLES
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition

if TYPE_CHECKING:
    # Imported only for type hints to avoid pulling heavy deps at import time.
    from collections.abc import Sequence

    from xarray import Dataset

    from libera_utils.footprint_matching.camera_segmentation import PseudoFootprint
    from libera_utils.io.filenaming import LiberaDataProductFilename

logger = logging.getLogger(__name__)

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
        # Boresight (centre) pixel provenance: which L1B camera pixel this pseudo-footprint's
        # boresight stand-in falls on. Real per-footprint integers straight off the
        # PseudoFootprint. The block's inclusive (min, max) pixel extent is emitted separately
        # as the 2-D camera_pixel_x/y range COORDINATES, which this set (real data *variables*)
        # deliberately does not list.
        "center_pixel_x",
        "center_pixel_y",
    }
)

# The radiometer-timescale counterpart of _CAMTIME_SEGMENTATION_VARIABLES: the
# product-definition variables filled with *real* values passed straight through
# from the L1B Daily radiometer product, rather than computed by footprint
# matching. These are the "(a) Geolocation inputs (from L1B Daily)" block of each
# radiometer-timed fmatch_*.yml, and they are exactly the keys (other than the
# RADIOMETER_TIME coordinate) that
# ``l1b_inputs.load_l1b_radiometer_inputs`` returns. Kept as one set so the
# assembly and its tests agree on which variables are "real" this milestone.
_RADIOMETER_L1B_VARIABLES: frozenset[str] = frozenset(L1B_PASSTHROUGH_VARIABLES)

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
    return "CAMERA_TIME" if is_camera_timescale_mode(mode) else "RADIOMETER_TIME"


def is_camera_timescale_mode(mode: OperationalMode) -> bool:
    """Return whether a mode indexes its footprints by camera image time.

    The timescale determines what a mode is built *from*: camera-timescale modes are
    assembled from camera pseudo-footprints (segmented from the L1B camera grid),
    while radiometer-timescale modes are assembled from L1B radiometer pass-through
    inputs. Runners and assembly both branch on this.

    Parameters
    ----------
    mode : OperationalMode
        The operational mode to test.

    Returns
    -------
    bool
        True for ``CAM_CAMTIME`` and ``IMAGER_CAMTIME``; False otherwise.
    """
    return mode in _CAMERA_TIMESCALE_MODES


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
    filename = FMATCH_DEFINITION_FILENAMES[mode]
    definitions_dir = Path(str(config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")))
    return LiberaDataProductDefinition.from_yaml(definitions_dir / filename)


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
    ``ReaderRegistry.get_readers_for_mode(mode)``, load the tiles overlapping each
    footprint, and apply each variable's PSF-weighted aggregation strategy (weighted
    mean / mode / log-mean) to collapse the fine-resolution pixels to a single value
    per footprint. The active reader set - and therefore the keys of the returned
    dict - grows with the mode's latency (e.g. CAM has era5, igbp, nise, viirs_brdf,
    viirs_cloud; IMAGER additionally has era5_pressure, viirs_aod, and the RBSP ssf
    and cldpix fields).
    Per-spec gating also applies: only specs whose ``required_mode`` rank is
    <= the mode's rank are aggregated.

    Every output variable is named ``<source_key>_<spec_name>`` (e.g.
    ``era5_wind_u10``, ``igbp_surface_type``, ``cldpix_cloud_mask``), matching the
    product definition variable names. The reader's ``INSTRUMENT`` is recorded in
    each variable's ``long_name`` (``"... (ECMWF)"``) rather than in the name.

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
    expected by the mode's product definition (from
    :func:`load_fmatch_definition`), then builds a Dataset via
    ``LiberaDataProductDefinition.create_product_dataset`` and brings it into
    conformance with ``enforce_dataset_conformance``.

    Dispatch is by timescale, because that determines what the mode is built
    *from*:

    * **Camera-timescale** (``CAM_CAMTIME``, ``IMAGER_CAMTIME``) - assembled from
      the camera pseudo-footprints produced by
      :func:`libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
      The first positional argument is that sequence of
      :class:`PseudoFootprint` objects. See :func:`_assemble_camtime_dataset`.
    * **Radiometer-timescale** (``CAM``, ``IMAGER_FLASH``, ``IMAGER``) - assembled
      from the L1B pass-through arrays produced by
      :func:`libera_utils.footprint_matching.l1b_inputs.load_l1b_radiometer_inputs`.
      The first positional argument is that dict. See
      :func:`_assemble_radiometer_dataset`.

    In both cases the variables owned by the not-yet-implemented aggregation and
    derived-geometry engines are written as conformant placeholders (structurally
    valid but numerically meaningless) pending ``TODO[LIBSDC-785]``.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being assembled.
    *args, **kwargs
        Mode-specific inputs, forwarded to the timescale's assembler (see above
        for the leading positional argument of each).
    cloud_fraction_camera : np.ndarray, optional
        Per-footprint cloud fraction from the Camera Cloud Fraction (CF-CAM)
        algorithm (Libera WFOV camera), as a 1-D array indexed by footprint in
        the same order as the time coordinate. This is an *internal* algorithm
        output - it does not come from a reader and is already aggregated to one
        value per footprint - so it is merged directly into the ``cloud_fraction_camera``
        variable rather than going through :func:`aggregate_external_variables`.
        Only the CAM modes (``CAM``, ``CAM_CAMTIME``) declare this variable; it is
        ``None`` for the IMAGER modes.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's product definition.
    """
    if mode in _CAMERA_TIMESCALE_MODES:
        return _assemble_camtime_dataset(*args, mode=mode, cloud_fraction_camera=cloud_fraction_camera, **kwargs)
    return _assemble_radiometer_dataset(*args, mode=mode, cloud_fraction_camera=cloud_fraction_camera, **kwargs)


def _placeholder_variable_array(variable_definition: Any, n_footprints: int) -> np.ndarray:
    """Build a conformant placeholder array for a not-yet-computed product variable.

    Variables owned by the aggregation / derived-geometry engines (not built yet)
    still have to appear in the output file with the right dtype and shape so the
    product conforms to its definition. We fill them with the variable's declared
    ``_FillValue`` when it has one, and otherwise with ``NaN`` for floating-point
    variables or ``0`` for integer variables. The magnitudes are meaningless; only
    the dtype/shape/attributes form the product contract (the same stance the
    example-product generator takes in ``notebooks/generate_example_products.ipynb``).

    Parameters
    ----------
    variable_definition : LiberaVariableDefinition
        The product-definition entry for the variable.
    n_footprints : int
        Length of the footprint (time) axis.

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


def _fill_placeholder_variables(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    n_footprints: int,
) -> None:
    """Fill every declared variable missing from ``data`` with a conformant placeholder.

    Mutates ``data`` in place, adding one array per not-yet-computed variable. The
    product definition requires every declared variable to be present, so this is
    what lets a product conform while the aggregation / derived-geometry engines are
    still ``TODO[LIBSDC-785]`` stubs. The placeholder values are structurally valid
    (declared dtype/shape/attributes) but numerically meaningless.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        The variable arrays assembled so far (the real, input-derived columns).
    definition : LiberaDataProductDefinition
        The product definition naming every variable the file must contain.
    n_footprints : int
        Length of the footprint (time) axis.
    """
    for name, variable_definition in definition.variables.items():
        if name not in data:
            data[name] = _placeholder_variable_array(variable_definition, n_footprints)


def _finalize_product_dataset(
    definition: LiberaDataProductDefinition,
    data: dict[str, np.ndarray],
    *,
    algorithm_version: str | None,
    input_files: str | None,
) -> Dataset:
    """Build a conformant Dataset from assembled arrays and set the dynamic global attributes.

    Shared tail of both assembly paths. Dynamic (per-run) global attributes are set
    directly on the Dataset; they are declared (as null) in the definition, so
    ``enforce_dataset_conformance`` keeps them rather than stripping them as extras.

    Parameters
    ----------
    definition : LiberaDataProductDefinition
        The product definition to build against.
    data : dict[str, np.ndarray]
        Every coordinate and variable array the product declares.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute.

    Returns
    -------
    xarray.Dataset
        The conformance-enforced dataset, ready to write.
    """
    dataset = definition.create_product_dataset(data)
    dataset = definition.enforce_dataset_conformance(dataset)
    if input_files is not None:
        dataset.attrs["input_files"] = input_files
    if algorithm_version is not None:
        dataset.attrs["algorithm_version"] = algorithm_version
    return dataset


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
    mode: OperationalMode = OperationalMode.CAM_CAMTIME,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
) -> Dataset:
    """Assemble a camera-timescale FMATCH Dataset from camera pseudo-footprints.

    Serves both camera-timescale modes (``CAM_CAMTIME`` and ``IMAGER_CAMTIME``).
    They are built the same way - from the same L1B camera segmentation - and
    differ only in the set of aggregated variables their definitions declare,
    which the placeholder fill handles generically.

    Builds the per-footprint variable arrays declared by the mode's product
    definition. The centre-pixel geolocation/geometry, the corner-derived PSF
    bounding box, and the QA flags come straight from the pseudo-footprints
    (:data:`_CAMTIME_SEGMENTATION_VARIABLES`); every other declared variable is a
    conformant placeholder pending the aggregation / derived-geometry engines
    (``TODO[LIBSDC-785]``).

    Parameters
    ----------
    footprints : Sequence[PseudoFootprint]
        Camera pseudo-footprints in write order, as returned by
        :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`.
    mode : OperationalMode, optional
        Which camera-timescale mode to assemble. Defaults to ``CAM_CAMTIME``.
    definition : LiberaDataProductDefinition, optional
        The product definition. Loaded via :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute
        (typically the source L1B camera filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). Only the
        CAM modes declare this variable; for ``IMAGER_CAMTIME`` it is ignored. When
        omitted (or undeclared) the variable is written as a placeholder.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's definition.

    Raises
    ------
    ValueError
        If ``mode`` is not a camera-timescale mode, or ``footprints`` is empty
        (there would be no time axis to write).
    """
    if mode not in _CAMERA_TIMESCALE_MODES:
        raise ValueError(
            f"{mode.value} is not a camera-timescale mode; camera pseudo-footprints only assemble "
            f"{', '.join(sorted(m.value for m in _CAMERA_TIMESCALE_MODES))}."
        )
    if definition is None:
        definition = load_fmatch_definition(mode)

    footprints = list(footprints)
    if not footprints:
        raise ValueError(f"Cannot assemble a {mode.value} product from zero pseudo-footprints.")
    n_footprints = len(footprints)

    time_variable = fmatch_time_variable(mode)  # "CAMERA_TIME"

    # The real, segmentation-derived 1-D columns. Longitudes of the PSF box are wrapped
    # into [-180, 180) to satisfy the product definition's valid range. center_pixel_x/y
    # are the boresight stand-in pixel; they are FMATCH-only provenance (the block's
    # inclusive extent is emitted as the camera_pixel_x/y range coordinates below).
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
        "center_pixel_x": [f.center_ix for f in footprints],
        "center_pixel_y": [f.center_iy for f in footprints],
    }

    # Start the data dict with the time coordinate (nanosecond datetimes; note the values
    # repeat within an image -- see segment_l1b_camera's docstring). CAMERA_TIME is a
    # coordinate on the FOOTPRINT record axis: create_product_dataset routes it to .coords
    # because the definition declares it under coordinates:.
    data: dict[str, np.ndarray] = {
        time_variable: np.array([f.time for f in footprints], dtype="datetime64[ns]"),
    }

    # Camera pixel-index ranges as inclusive (min, max) pairs on the CAMERA_PIXEL_BOUNDS
    # axis. slice_x/slice_y are half-open [start, stop), so the inclusive maximum is
    # stop - 1. These are 2-D coordinates (FOOTPRINT x CAMERA_PIXEL_BOUNDS) that both
    # camtime products declare and that pass straight through to SCENE-ID-CAM-CAMTIME.
    pixel_ranges: dict[str, list[tuple[int, int]]] = {
        "camera_pixel_x": [(f.slice_x.start, f.slice_x.stop - 1) for f in footprints],
        "camera_pixel_y": [(f.slice_y.start, f.slice_y.stop - 1) for f in footprints],
    }
    for coordinate_name, ranges in pixel_ranges.items():
        coordinate_definition = definition.coordinates[coordinate_name]
        data[coordinate_name] = np.asarray(ranges, dtype=np.dtype(coordinate_definition.dtype))

    # Cast each real 1-D column to the exact dtype the definition declares. Only columns
    # the definition actually declares are written; both camtime products declare the same
    # segmentation variables, but the guard keeps the assembly robust to definition drift.
    for name, values in real_columns.items():
        variable_definition = definition.variables.get(name)
        if variable_definition is None:
            continue
        data[name] = np.asarray(values, dtype=np.dtype(variable_definition.dtype))

    # Optional internal (non-reader) Camera Cloud Fraction values. Guarded on the
    # definition declaring the variable, because IMAGER-CAMTIME does not.
    _merge_cloud_fraction_camera(data, definition, cloud_fraction_camera)

    # Every remaining declared variable is filled with a placeholder until its engine exists.
    _fill_placeholder_variables(data, definition, n_footprints)

    return _finalize_product_dataset(
        definition,
        data,
        algorithm_version=algorithm_version,
        input_files=input_files,
    )


def _merge_cloud_fraction_camera(
    data: dict[str, np.ndarray],
    definition: LiberaDataProductDefinition,
    cloud_fraction_camera: np.ndarray | None,
) -> None:
    """Merge the CF-CAM cloud fraction into ``data`` when it is supplied and declared.

    ``cloud_fraction_camera`` is an internal Libera algorithm output (from the WFOV
    Camera Cloud Fraction algorithm), already one value per footprint, so it bypasses
    the reader/aggregation path and is merged straight in. Only the CAM-family
    definitions declare it; passing values for an IMAGER mode is ignored rather than
    raising, so a caller can hand the same inputs to any mode.
    """
    if cloud_fraction_camera is None:
        return
    variable_definition = definition.variables.get("cloud_fraction_camera")
    if variable_definition is None:
        logger.warning(
            "cloud_fraction_camera values were supplied but %s does not declare that variable; ignoring them.",
            definition.attributes.get("ProductID", "this product"),
        )
        return
    data["cloud_fraction_camera"] = np.asarray(cloud_fraction_camera, dtype=np.dtype(variable_definition.dtype))


def _assemble_radiometer_dataset(
    l1b_inputs: dict[str, np.ndarray],
    *,
    mode: OperationalMode,
    definition: LiberaDataProductDefinition | None = None,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
) -> Dataset:
    """Assemble a radiometer-timescale FMATCH Dataset from L1B pass-through inputs.

    Serves the three radiometer-timescale modes (``CAM``, ``IMAGER_FLASH``,
    ``IMAGER``). Their footprints are the L1B radiometer footprints themselves, so
    the time coordinate and the geolocation/viewing-angle columns
    (:data:`_RADIOMETER_L1B_VARIABLES`) are carried through verbatim from L1B; every
    other declared variable belongs to the aggregation / derived-geometry engines and
    is written as a conformant placeholder pending ``TODO[LIBSDC-785]``.

    Parameters
    ----------
    l1b_inputs : dict[str, np.ndarray]
        The pass-through arrays from
        :func:`libera_utils.footprint_matching.l1b_inputs.load_l1b_radiometer_inputs`:
        the ``RADIOMETER_TIME`` coordinate plus each of
        :data:`_RADIOMETER_L1B_VARIABLES`, all the same length.
    mode : OperationalMode
        Which radiometer-timescale mode to assemble.
    definition : LiberaDataProductDefinition, optional
        The product definition. Loaded via :func:`load_fmatch_definition` when omitted.
    algorithm_version : str, optional
        Value for the required dynamic ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the required dynamic ``input_files`` global attribute
        (typically the source L1B radiometer filename).
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV), in the same
        footprint order as the time coordinate. Only ``CAM`` declares this variable.

    Returns
    -------
    xarray.Dataset
        A dataset brought into conformance with the mode's definition.

    Raises
    ------
    ValueError
        If ``mode`` is a camera-timescale mode, if a required pass-through input is
        missing, if the inputs have inconsistent lengths, or if they are empty.
    """
    if mode in _CAMERA_TIMESCALE_MODES:
        raise ValueError(
            f"{mode.value} is a camera-timescale mode and cannot be assembled from L1B radiometer pass-through "
            f"inputs; use the camera pseudo-footprint path instead."
        )
    if definition is None:
        definition = load_fmatch_definition(mode)

    time_variable = fmatch_time_variable(mode)  # "RADIOMETER_TIME"

    # Fail with the specific missing names rather than a bare KeyError deep in the
    # loop below, because a partial pass-through dict is the most likely caller error.
    required = {time_variable, *_RADIOMETER_L1B_VARIABLES}
    missing = sorted(required - set(l1b_inputs))
    if missing:
        raise ValueError(f"L1B pass-through inputs for {mode.value} are missing required key(s): {', '.join(missing)}")

    n_footprints = len(l1b_inputs[time_variable])
    if n_footprints == 0:
        raise ValueError(f"Cannot assemble a {mode.value} product from zero footprints.")
    inconsistent = sorted(name for name in required if len(l1b_inputs[name]) != n_footprints)
    if inconsistent:
        raise ValueError(
            f"L1B pass-through inputs for {mode.value} have inconsistent lengths; expected {n_footprints} footprints "
            f"(from {time_variable}) but got a different length for: {', '.join(inconsistent)}"
        )

    # Start with the time coordinate, then the real L1B columns cast to the exact
    # dtype the definition declares.
    data: dict[str, np.ndarray] = {
        time_variable: np.asarray(l1b_inputs[time_variable], dtype="datetime64[ns]"),
    }
    for name in sorted(_RADIOMETER_L1B_VARIABLES):
        data[name] = np.asarray(l1b_inputs[name], dtype=np.dtype(definition.variables[name].dtype))

    # Optional internal (non-reader) Camera Cloud Fraction values (CAM only).
    _merge_cloud_fraction_camera(data, definition, cloud_fraction_camera)

    # Every remaining declared variable is filled with a placeholder until its engine exists.
    _fill_placeholder_variables(data, definition, n_footprints)

    return _finalize_product_dataset(
        definition,
        data,
        algorithm_version=algorithm_version,
        input_files=input_files,
    )


def write_fmatch_product(mode: OperationalMode, *args: Any, **kwargs: Any) -> Any:
    """Write a FMATCH NetCDF data product to disk for an operational mode.

    Delegates to ``libera_utils.io.netcdf.write_libera_data_product`` using the
    definition from :func:`load_fmatch_definition`, the assembled Dataset from
    :func:`assemble_fmatch_dataset`, and ``time_variable=fmatch_time_variable(mode)``
    (``RADIOMETER_TIME`` or ``CAMERA_TIME``) so the output filename encodes the
    footprint time span.

    Every operational mode is supported. The mode's timescale selects what the
    leading positional argument must be - camera pseudo-footprints for the
    camera-timescale modes, L1B pass-through arrays for the radiometer-timescale
    ones - exactly as in :func:`assemble_fmatch_dataset`.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    *args, **kwargs
        Mode-specific inputs forwarded to :func:`assemble_fmatch_dataset`, followed
        by ``output_path``. See :func:`_write_fmatch_product` for the accepted
        keyword arguments.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.
    """
    return _write_fmatch_product(mode, *args, **kwargs)


def _write_fmatch_product(
    mode: OperationalMode,
    inputs: Sequence[PseudoFootprint] | dict[str, np.ndarray],
    output_path: str | Path,
    *,
    algorithm_version: str | None = None,
    input_files: str | None = None,
    cloud_fraction_camera: np.ndarray | None = None,
    strict: bool = True,
) -> LiberaDataProductFilename:
    """Assemble and write one FMATCH NetCDF product.

    Loads the product definition once (so assembly and writing cannot disagree about
    it), assembles a conformant Dataset via :func:`assemble_fmatch_dataset`, and
    writes it with ``write_libera_data_product``, which generates the standardized
    Libera filename from the product's time span.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode to write.
    inputs : Sequence[PseudoFootprint] | dict[str, np.ndarray]
        The mode's assembly inputs: camera pseudo-footprints for the camera-timescale
        modes, or the L1B pass-through dict for the radiometer-timescale modes.
    output_path : str or pathlib.Path
        Directory (or S3 prefix) to write the product file into.
    algorithm_version : str, optional
        Value for the ``algorithm_version`` global attribute.
    input_files : str, optional
        Provenance string for the ``input_files`` global attribute.
    cloud_fraction_camera : np.ndarray, optional
        Optional per-footprint Camera Cloud Fraction values (Libera WFOV). Only the
        CAM modes declare this variable.
    strict : bool, optional
        When True (default), fail if the assembled Dataset does not conform.

    Returns
    -------
    LiberaDataProductFilename
        The written product filename object.
    """
    definition = load_fmatch_definition(mode)
    dataset = assemble_fmatch_dataset(
        mode,
        inputs,
        definition=definition,
        algorithm_version=algorithm_version,
        input_files=input_files,
        cloud_fraction_camera=cloud_fraction_camera,
    )
    return write_libera_data_product(
        data_product_definition=definition,
        data=dataset,
        output_path=output_path,
        time_variable=fmatch_time_variable(mode),
        strict=strict,
    )
