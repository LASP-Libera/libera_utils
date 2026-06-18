"""FMATCH product metadata: mapping operational modes to data product identifiers.

This module connects each ``OperationalMode`` value to its corresponding
``DataProductIdentifier``, so the FMATCH pipeline knows which output product
to write for a given run mode.

Background
----------
FMATCH (Footprint Matching) runs in five distinct operational modes that differ
in data latency and ancillary data sources. Each mode produces a separate data
product with a unique product ID defined in the Libera Data Product Information
(DPI) PDF v87 (2026-06-18). The mapping here is the single authoritative place
that encodes mode ↔ product ID and mode ↔ active reader set relationships.

Product definition YAML files (which define the NetCDF variable schemas) are
managed separately in ``libera_utils/data/product_definitions/`` and will be
linked here in a future branch once created.

References
----------
DPI v87 Table of Data Product Identifiers (FMATCH section):
    instructions/documentation/Data Product Information-v87-20260618_005103.pdf
FMATCH reader mode assignments (ACTIVE_MODES per reader):
    libera_utils/footprint_matching/readers/
"""

from __future__ import annotations

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.types import OperationalMode

# ---------------------------------------------------------------------------
# Mode → DataProductIdentifier
# ---------------------------------------------------------------------------

# Maps each of the five FMATCH operational modes to its DataProductIdentifier.
# This is the single authoritative source for the mode ↔ product ID relationship
# within the pipeline. The pipeline uses this to construct output filenames,
# set NetCDF global attributes, and route data to the correct S3 prefix.
#
# Product IDs come directly from the DPI v87 Table of Data Product Identifiers.
# Keep this in sync with DataProductIdentifier in libera_utils/constants.py.
PRODUCT_FOR_MODE: dict[OperationalMode, DataProductIdentifier] = {
    # Camera/NRT tier — available from mission start, no RBSP dependency.
    OperationalMode.CAM: DataProductIdentifier.fmatch_cam,
    OperationalMode.CAM_CAMTIME: DataProductIdentifier.fmatch_cam_camtime,
    # RBSP FlashFlux tier — available ~5 days after observation.
    OperationalMode.IMAGER_FLASH: DataProductIdentifier.fmatch_imager_flash,
    # RBSP Climate Quality tier — available post-Year 1, highest fidelity.
    OperationalMode.IMAGER: DataProductIdentifier.fmatch_imager,
    OperationalMode.IMAGER_CAMTIME: DataProductIdentifier.fmatch_imager_camtime,
}


def get_product_identifier(mode: OperationalMode) -> DataProductIdentifier:
    """Return the DataProductIdentifier for the given operational mode.

    Parameters
    ----------
    mode : OperationalMode
        The active FMATCH operational mode.

    Returns
    -------
    DataProductIdentifier
        The product identifier used for output file naming and pipeline routing.

    Examples
    --------
    >>> from libera_utils.footprint_matching.types import OperationalMode
    >>> get_product_identifier(OperationalMode.CAM)
    <DataProductIdentifier.fmatch_cam: 'FMATCHCAM'>
    >>> str(get_product_identifier(OperationalMode.IMAGER))
    'FMATCHIMAGER'
    """
    return PRODUCT_FOR_MODE[mode]
