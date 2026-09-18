"""Shared, algorithm-agnostic runner helpers for the camera/radiometer product families.

These helpers are used by more than one algorithm's runner -- the FMATCH runners
(:mod:`libera_utils.footprint_matching._runner`) and the Camera Cloud Fraction runners
(:mod:`libera_utils.cloud_fraction._runner`) both need to select input files from a manifest,
materialize a possibly-remote L1B file locally, load the L1B radiometer pass-through inputs or the
L1B camera dataset, and stamp the running package version onto a product.

They live here, in the lower ``footprint_matching`` layer, rather than in either algorithm's
``_runner`` module so that the cloud-fraction algorithm can reuse them without importing FMATCH's
runner (which would otherwise pull the two algorithms into a dependency cycle and force them into the
same review/PR). Keeping them dependency-light -- no FMATCH- or CF-specific config types -- is what
lets both runners share one implementation.
"""

from __future__ import annotations

import logging
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, S3Path

from libera_utils.footprint_matching.product import (
    FMATCH_CONE_ANGLE_RATE_KEY,
    L1B_CONE_ANGLE_RATE_VARIABLE,
    L1B_PASSTHROUGH_VARIABLES,
    L1B_SCAN_REFERENCE_VARIABLES,
)
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.smart_open import is_s3, smart_copy_file

if TYPE_CHECKING:
    from libera_utils.constants import DataProductIdentifier
    from libera_utils.io.manifest import Manifest

logger = logging.getLogger(__name__)

# Name of the time coordinate variable inside the L1B radiometer file. xarray decodes
# its CF "nanoseconds since 1958-01-01" units into datetime64[ns], which is exactly
# the dtype the FMATCH RADIOMETER_TIME coordinate declares.
L1B_TIME_VARIABLE: str = "radiometer_time"

# Name of the FMATCH product's radiometer time coordinate (the key the L1B reader
# returns the decoded L1B times under).
FMATCH_RADIOMETER_TIME_COORDINATE: str = "RADIOMETER_TIME"


def select_manifest_files_by_product_id(manifest: Manifest, *product_ids: DataProductIdentifier) -> list[str]:
    """Select the files referenced by a manifest that belong to the given Libera data product(s).

    A runner receives one manifest listing every file staged for the processing step, which in
    general contains more than the runner consumes (an L1B input, an optional cloud-fraction product,
    ancillary granules, sibling products). This helper keeps exactly the records whose parsed Libera
    product id is one of ``product_ids``, in manifest order.

    Files whose names do not parse as a :class:`~libera_utils.io.filenaming.LiberaDataProductFilename`
    are skipped rather than raising: manifests legitimately carry non-Libera files (for example the
    externally-sourced ancillary granules the readers consume), and a runner should ignore them, not
    fail on them.

    This lives in the runner-support layer, alongside its callers, so that ``libera_utils.io`` stays a
    generic I/O package with no dependency on Libera product identifiers. The SCENE-ID runner keeps
    its own single-product equivalent (``collect_input_files``) for the same reason.

    Parameters
    ----------
    manifest : Manifest
        The manifest to inspect.
    *product_ids : DataProductIdentifier
        One or more Libera product ids to keep. Passing several is useful when a runner takes more
        than one Libera input from the same manifest (e.g. FMATCH-CAM reads both an L1B product and an
        optional cloud-fraction one).

    Returns
    -------
    list[str]
        The matching manifest filenames, in manifest order. Empty when the manifest references none of
        the products.

    Raises
    ------
    ValueError
        If no ``product_ids`` are given, which would silently match nothing and is always a caller bug.
    """
    if not product_ids:
        raise ValueError("At least one DataProductIdentifier must be given to select manifest files.")

    wanted = set(product_ids)
    wanted_labels = ", ".join(sorted(product_id.value for product_id in wanted))
    selected: list[str] = []
    for file_record in manifest.files:
        filename = file_record.filename
        try:
            libera_filename = LiberaDataProductFilename.from_file_path(filename)
        except Exception:
            # Not a Libera product name, so it cannot be one of the requested products.
            logger.info("Skipping non-Libera-product file (not a %s input): %s", wanted_labels, filename)
            continue
        # Parsed as a Libera product; keep it only if it is one of the requested products.
        if libera_filename.data_product_id in wanted:
            logger.info("Recording %s input file: %s", libera_filename.data_product_id.value, filename)
            selected.append(filename)
        else:
            logger.info(
                "Skipping Libera product '%s' (not %s): %s",
                libera_filename.data_product_id.value,
                wanted_labels,
                filename,
            )
    return selected


def load_l1b_radiometer_inputs(l1b_file: Path) -> dict[str, np.ndarray]:
    """Read the per-footprint L1B inputs that FMATCH passes through verbatim.

    Pulls every quantity the FMATCH product contract takes straight from L1B Daily:
    the ``RADIOMETER_TIME`` coordinate plus all variables in
    :data:`~libera_utils.footprint_matching.product.L1B_PASSTHROUGH_VARIABLES`
    (footprint ``latitude``/``longitude`` and the solar/viewing zenith and
    relative-azimuth angles).

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
        locally first (see :class:`_as_local_path`), because
        :func:`xarray.open_dataset` seeks within the file.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping keyed by FMATCH variable name: ``"RADIOMETER_TIME"`` (datetime64[ns])
        plus each key of :data:`~libera_utils.footprint_matching.product.L1B_PASSTHROUGH_VARIABLES`
        (float32). It additionally carries the scan-reference geometry used to build
        footprints (not product columns): each key of
        :data:`~libera_utils.footprint_matching.product.L1B_SCAN_REFERENCE_VARIABLES` and
        :data:`~libera_utils.footprint_matching.product.FMATCH_CONE_ANGLE_RATE_KEY`
        (float64). All arrays are 1-D and the same length.

    Raises
    ------
    ValueError
        If no footprint has finite values for every pass-through / subsatellite
        variable, which would leave nothing to build a product from.
    """
    # Open with default decoding so the CF-encoded time coordinate ("nanoseconds since
    # 1958-01-01") is decoded into datetime64[ns] for us, and _FillValue-tagged fields
    # (the L1B fills are -999 / -9999) are masked to NaN so the finite check below works.
    with xr.open_dataset(l1b_file) as l1b:
        radiometer_time = l1b[L1B_TIME_VARIABLE].values
        # Read every pass-through variable, keyed by its FMATCH (output) name.
        passthrough = {fmatch_name: l1b[l1b_name].values for fmatch_name, l1b_name in L1B_PASSTHROUGH_VARIABLES.items()}
        # Scan-reference geometry (subsatellite point + cone-angle rate), read at
        # float64 for the ray-trace. Not written to the product; feeds footprint build.
        scan_reference = {
            fmatch_name: np.asarray(l1b[l1b_name].values, dtype=np.float64)
            for fmatch_name, l1b_name in L1B_SCAN_REFERENCE_VARIABLES.items()
        }
        cone_angle_rate = np.asarray(l1b[L1B_CONE_ANGLE_RATE_VARIABLE].values, dtype=np.float64)

    # Keep only footprints where every pass-through variable AND the subsatellite point
    # are finite. Start from an all-True mask and AND in each variable's finite mask, so
    # a NaN in ANY of them drops that footprint. The cone-angle rate is intentionally
    # excluded: an unknown scan rate must not discard an otherwise-good footprint.
    finite = np.ones(radiometer_time.shape, dtype=bool)
    for values in (*passthrough.values(), *scan_reference.values()):
        finite &= np.isfinite(values)

    n_finite = int(finite.sum())
    if n_finite == 0:
        raise ValueError(
            f"No usable footprints in L1B file {l1b_file}: every record has a non-finite value in at least one of "
            f"the pass-through or subsatellite variables "
            f"({', '.join(sorted({*L1B_PASSTHROUGH_VARIABLES, *L1B_SCAN_REFERENCE_VARIABLES}))})."
        )
    if n_finite < finite.size:
        logger.info(
            "Dropped %d of %d L1B footprints with non-finite geolocation/viewing angles",
            finite.size - n_finite,
            finite.size,
        )

    radiometer_time = radiometer_time[finite]
    passthrough = {name: values[finite] for name, values in passthrough.items()}
    scan_reference = {name: values[finite] for name, values in scan_reference.items()}
    cone_angle_rate = cone_angle_rate[finite]

    # This loader is the input boundary, so it is where the FMATCH dtypes are allocated:
    # cast every pass-through variable to the float32 the definition declares (L1B may store
    # them at wider precision) and the time coordinate to datetime64[ns]. Typing here means
    # everything downstream inherits the right dtypes and conformance passes without an auto-cast.
    result: dict[str, np.ndarray] = {
        FMATCH_RADIOMETER_TIME_COORDINATE: radiometer_time.astype("datetime64[ns]"),
    }
    result.update({name: values.astype(np.float32) for name, values in passthrough.items()})
    result.update(scan_reference)
    result[FMATCH_CONE_ANGLE_RATE_KEY] = cone_angle_rate
    return result


def load_l1b_camera_dataset(l1b_file: Path) -> xr.Dataset:
    """Open an L1B Daily Camera file for segmentation into pseudo-footprints.

    The Camera Cloud Fraction algorithm segments the camera pixel grid with
    :func:`~libera_utils.footprint_matching.camera_segmentation.segment_l1b_camera`,
    which needs the whole dataset (geolocation grids, altitude, and viewing angles on
    the ``CAMERA_TIME`` x ``CAMERA_PIXEL_COUNT_X`` x ``CAMERA_PIXEL_COUNT_Y`` grid).

    The dataset is loaded eagerly into memory (``.load()``) and detached from the file
    handle, so callers may use it after the source file goes away - which matters when
    the input was materialized into a temporary directory by :class:`_as_local_path`.

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


def algorithm_version() -> str:
    """Return the version recorded in a product's ``algorithm_version`` attribute.

    Sourced from the installed ``libera_utils`` package metadata, so a written product
    always names the exact release that produced it. Falls back to ``"0.0.0"`` only
    when the package is not installed (e.g. running from a source tree without an
    editable install), which should not happen inside the algorithm container.

    Returns
    -------
    str
        The installed ``libera_utils`` version.
    """
    try:
        return version("libera_utils")
    except PackageNotFoundError:  # pragma: no cover - not reachable in an installed environment
        logger.warning("libera_utils package metadata not found; recording algorithm_version as 0.0.0")
        return "0.0.0"


class _as_local_path:
    """Context manager yielding a real local filesystem path for a possibly-remote input file.

    Reading an L1B product with :func:`xarray.open_dataset` (and the HDF4/HDF5 libraries underneath it)
    requires a real file on a local filesystem, because those libraries seek within the file rather
    than streaming it. A runner's manifest, however, may reference either local paths or S3 object
    urls. For S3 inputs this downloads the object to a temporary directory that is removed on exit;
    local inputs are yielded unchanged with no copy.

    Kept in the runner-support layer so that ``libera_utils.io`` stays free of runner-specific
    plumbing; the SCENE-ID runner keeps its own identical helper for the same reason.

    Parameters
    ----------
    source_path : str | pathlib.Path | cloudpathlib.S3Path
        Path to the input file, local or in S3.
    """

    def __init__(self, source_path: str | Path | S3Path):
        self._source_path = AnyPath(source_path)
        self._tempdir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> Path:
        if is_s3(self._source_path):
            # Materialize the S3 object locally so file-based readers (netCDF4, HDF4/HDF5) can open it.
            self._tempdir = tempfile.TemporaryDirectory()
            local_path = Path(self._tempdir.name) / self._source_path.name
            smart_copy_file(self._source_path, local_path)
            return local_path
        # Already local; hand back a plain pathlib.Path.
        return Path(str(self._source_path))

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
