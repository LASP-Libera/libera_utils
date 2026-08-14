"""Locating the staged ancillary granules that the FMATCH readers consume.

Footprint matching aggregates external, non-Libera datasets (ERA5 reanalysis, IGBP
land cover, NISE snow/ice, VIIRS products, and - post year one - CERES CLDPIX/SSF)
onto each radiometer footprint. Those granules are *not* Libera data products: they
have third-party filenames, so they cannot be selected out of a manifest the way
:func:`libera_utils.footprint_matching._runner.select_manifest_files_by_product_id` selects L1B
inputs, and the reader classes deliberately know nothing about filenames (each takes
one already-resolved ``file_path``).

The contract this module defines
--------------------------------
The pipeline stages ancillary granules into a directory tree, and the runner is told
where that tree is via the ``FMATCH_ANCILLARY_PATH`` environment variable - the same
shape as the ``PROCESSING_PATH`` (dropbox) variable every Libera runner already
takes. Within the tree, **one subdirectory per reader, named by the reader's registry
key**::

    $FMATCH_ANCILLARY_PATH/
        era5/           ERA5 single-level granules
        era5_pressure/  ERA5 pressure-level granules
        igbp/           MCD12Q1 land-cover granules
        nise/           NISE snow/ice granules
        viirs_brdf/     VIIRS BRDF granules
        viirs_cloud/    VIIRS cloud granules
        viirs_aod/      VIIRS Deep Blue aerosol granules (AOD + aerosol type)
        cldpix/         CERES CLDPIX granules   (IMAGER-family modes)
        ssf/            CERES SSF granules      (IMAGER-family modes)

Using the registry keys as directory names means the set of directories required for
a run is *derived* from ``ReaderRegistry.get_readers_for_mode(mode)`` rather than
hard-coded here. A new reader, or a change to a reader's mode gating, automatically
changes what this module looks for - there is no second list to keep in sync.

See Also
--------
libera_utils.footprint_matching.readers.registry.ReaderRegistry : Source of the active reader set.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from cloudpathlib import AnyPath, S3Path

# Importing the readers subpackage is what populates ReaderRegistry: every concrete
# reader self-registers at class-definition time via GriddedDataReader.__init_subclass__,
# which only happens once its module has been imported. Without this import the
# registry would be empty and every mode would resolve to zero readers.
import libera_utils.footprint_matching.readers  # noqa: F401  (imported for its registration side effect)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode

logger = logging.getLogger(__name__)

# Environment variable naming the root of the staged ancillary tree. Mirrors the
# PROCESSING_PATH convention used for the output dropbox.
ANCILLARY_PATH_ENV: str = "FMATCH_ANCILLARY_PATH"


def resolve_ancillary_inputs(
    mode: OperationalMode,
    root: str | Path | S3Path | None = None,
    *,
    strict: bool = False,
) -> dict[str, list[Path | S3Path]]:
    """Map each reader active for ``mode`` to its staged granule files.

    The active reader set comes from
    :meth:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry.get_readers_for_mode`,
    so this automatically tracks the mode's latency rank: FMATCH-CAM looks for five
    readers, while FMATCH-IMAGER additionally looks for ``era5_pressure``,
    ``viirs_aod``, and the RBSP ``cldpix``/``ssf`` inputs.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH operational mode being run.
    root : str | pathlib.Path | cloudpathlib.S3Path, optional
        Root of the staged ancillary tree. Defaults to the ``FMATCH_ANCILLARY_PATH``
        environment variable.
    strict : bool, optional
        How to treat missing inputs. When False (the default) a missing root, a
        missing reader subdirectory, or an empty one is logged as a warning and
        yields an empty list. When True each of those raises.

        The default is False **in this milestone** because the PSF aggregation engine
        that consumes these granules is not implemented yet
        (``product.aggregate_external_variables`` is still a ``TODO[LIBSDC-785]``
        stub): a runner must not fail on inputs that nothing reads. Flip this default
        to True in the same change that implements the aggregation engine, at which
        point a missing granule genuinely invalidates the product.

    Returns
    -------
    dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        Mapping of reader registry key to the granule files staged for it, sorted by
        name so repeated runs process files in a deterministic order. Readers with no
        staged files map to an empty list (when ``strict`` is False).

    Raises
    ------
    ValueError
        If ``root`` is not given and ``FMATCH_ANCILLARY_PATH`` is unset, and ``strict``.
    FileNotFoundError
        If the root or a required reader subdirectory does not exist, or a required
        subdirectory is empty, and ``strict``.
    """
    active_readers = ReaderRegistry.get_readers_for_mode(mode)

    ancillary_root = _resolve_root(root, mode=mode, strict=strict)
    if ancillary_root is None:
        return {key: [] for key in sorted(active_readers)}

    resolved: dict[str, list[Path | S3Path]] = {}
    for reader_key in sorted(active_readers):
        resolved[reader_key] = _resolve_reader_directory(ancillary_root, reader_key, strict=strict)
    return resolved


def _resolve_root(
    root: str | Path | S3Path | None,
    *,
    mode: OperationalMode,
    strict: bool,
) -> Path | S3Path | None:
    """Resolve and validate the ancillary tree root, or return None when absent and not strict."""
    if root is None:
        root = os.getenv(ANCILLARY_PATH_ENV)
    if not root:
        message = (
            f"{ANCILLARY_PATH_ENV} environment variable is not set, so no ancillary inputs can be located for "
            f"{mode.value}."
        )
        if strict:
            raise ValueError(message)
        logger.warning("%s Continuing with no ancillary inputs.", message)
        return None

    # AnyPath gives a local Path or an S3Path depending on the string, so a staged
    # tree works equally well on local disk and in S3.
    ancillary_root = AnyPath(root)
    if not ancillary_root.exists():
        message = f"Ancillary root directory does not exist: {ancillary_root}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s Continuing with no ancillary inputs.", message)
        return None
    return ancillary_root


def _resolve_reader_directory(
    ancillary_root: Path | S3Path,
    reader_key: str,
    *,
    strict: bool,
) -> list[Path | S3Path]:
    """List the granule files staged under one reader's subdirectory."""
    reader_directory = ancillary_root / reader_key
    if not reader_directory.exists():
        message = f"No staged ancillary directory for reader '{reader_key}': {reader_directory}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s (expected %s/%s/)", message, ANCILLARY_PATH_ENV, reader_key)
        return []

    # Sorted for determinism; skip nested directories so a reader subdirectory can
    # hold per-day subfolders in the future without those being mistaken for granules.
    files: list[Path | S3Path] = sorted((entry for entry in reader_directory.iterdir() if entry.is_file()), key=str)
    if not files:
        message = f"Ancillary directory for reader '{reader_key}' is empty: {reader_directory}"
        if strict:
            raise FileNotFoundError(message)
        logger.warning("%s", message)
    return files


def log_ancillary_inventory(ancillary_inputs: dict[str, list[Path | S3Path]]) -> None:
    """Log a one-line-per-reader inventory of the staged ancillary inputs.

    Written so an operator can diagnose a staging problem from the task log alone,
    without shelling into the container: every active reader appears, and the ones
    with nothing staged are called out explicitly rather than being silently absent.

    Parameters
    ----------
    ancillary_inputs : dict[str, list[pathlib.Path | cloudpathlib.S3Path]]
        The mapping returned by :func:`resolve_ancillary_inputs`.
    """
    if not ancillary_inputs:
        logger.warning("No ancillary readers are active for this run.")
        return
    for reader_key, files in sorted(ancillary_inputs.items()):
        if files:
            logger.info("Ancillary input '%s': %d file(s)", reader_key, len(files))
        else:
            logger.warning("Ancillary input '%s': 0 file(s) -- MISSING", reader_key)
