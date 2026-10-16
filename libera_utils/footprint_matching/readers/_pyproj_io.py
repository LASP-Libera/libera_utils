"""Shared pyproj lazy-import guard for the footprint matching reader plugin system.

pyproj is used by the NISE sea ice reader (NISEReader) to reproject the NSIDC
EASE-Grid pixel centers to geographic lat/lon (EPSG:4326). It ships as a regular
dependency of libera_utils (manylinux wheels, so no system library is required).

Why lazy import?
----------------
pyproj is a regular dependency, but importing it at module load time would make
the entire footprint_matching package un-importable if the geospatial stack ever
fails to import. Deferring the import to call time via
``_require_pyproj()`` keeps the rest of the package importable and surfaces a
clear, actionable error only when a NISE reprojection is actually attempted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyproj as pyproj_type


def _require_pyproj() -> pyproj_type:
    """Import pyproj and return the module, raising ImportError with instructions if absent.

    Returns
    -------
    module
        The ``pyproj`` module.

    Raises
    ------
    ImportError
        If pyproj is not installed, with instructions for installing it.
    """
    try:
        import pyproj  # noqa: PLC0415

        return pyproj
    except ImportError as exc:
        raise ImportError(
            "pyproj is required to reproject the NISE EASE-Grid grid (NSIDC sea ice). "
            "Install instructions:\n"
            "  pip:   pip install pyproj\n"
            "  conda: conda install -c conda-forge pyproj\n"
            "See https://pyproj4.github.io/pyproj/stable/ for details."
        ) from exc
