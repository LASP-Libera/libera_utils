"""Shared HDF4 I/O helper for the footprint matching reader plugin system.

HDF4 files are used by the IGBP land cover reader (MCD12Q1) and the VIIRS
cloud property reader. Both use pyhdf, which wraps the HDF4 C library.

Why lazy import?
----------------
pyhdf requires the HDF4 C library (``libhdf4-dev`` on Debian/Ubuntu) to be
installed as a system package before pip/poetry can build its wheel. In
environments where the library is absent (e.g., the CI container without HDF4),
importing ``pyhdf`` at module load time would make the *entire* footprint_matching
package un-importable. By deferring the import to function call time via
``_require_pyhdf()``, the rest of the package stays importable and only the HDF4
I/O calls themselves fail with a clear, actionable error message.
"""
from __future__ import annotations

import numpy as np


def _require_pyhdf() -> object:
    """Import pyhdf.SD and return the module, raising ImportError with clear instructions if absent.

    Returns
    -------
    module
        The ``pyhdf.SD`` module.

    Raises
    ------
    ImportError
        If pyhdf is not installed or the HDF4 C library is missing, with
        instructions for installing both.
    """
    try:
        import pyhdf.SD as sd  # noqa: PLC0415
        return sd
    except ImportError as exc:
        raise ImportError(
            "pyhdf is required to read HDF4 files (IGBP MCD12Q1 and VIIRS cloud products). "
            "Install instructions:\n"
            "  conda: conda install -c conda-forge pyhdf\n"
            "  pip:   sudo apt-get install libhdf4-dev && pip install pyhdf\n"
            "See https://fhs.github.io/pyhdf/ for details."
        ) from exc


def read_hdf4_lat_lon_grid(
    file_path: str,
    data_sds_name: str,
    lat_sds_name: str,
    lon_sds_name: str,
    fill_value: float = 255.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open an HDF4 file and extract a named data variable plus lat/lon grids.

    This function is shared by IGBPReader and VIIRSL2L3Reader. It opens the
    file, reads three named Scientific Data Sets (SDS), casts fill values to
    NaN (or the caller-specified sentinel), and returns numpy arrays.

    Parameters
    ----------
    file_path : str
        Absolute path to the HDF4 (.hdf) file.
    data_sds_name : str
        Name of the SDS holding the primary data variable. For IGBP this is
        ``"Land_Cover_Type_1"``; for VIIRS the caller iterates per-variable.
    lat_sds_name : str
        Name of the latitude SDS (e.g., ``"latitude"``).
    lon_sds_name : str
        Name of the longitude SDS (e.g., ``"longitude"``).
    fill_value : float, optional
        Pixel value used as the HDF4 fill/missing sentinel. These pixels are
        returned as-is in the data array; the caller is responsible for masking.
        Default is 255 (used by IGBP MCD12Q1).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(data, lats, lons)`` with dtypes matching the on-disk representation.
        No coordinate trimming is applied — the returned arrays cover the full
        extent stored in the HDF4 file.

    Raises
    ------
    ImportError
        If pyhdf is not installed (see :func:`_require_pyhdf`).
    KeyError
        If any of the three SDS names is not found in the file.
    OSError
        If the file cannot be opened (file not found, corrupt HDF4 header, etc.).

    Notes
    -----
    HDF4 Scientific Data Sets are accessed via ``pyhdf.SD.SD.select()``.
    The SDS data is read into memory in full; spatial subsetting is left to the
    caller (``_load_spatial_region``).

    References
    ----------
    pyhdf documentation: https://fhs.github.io/pyhdf/
    HDF4 format spec: https://support.hdfgroup.org/products/hdf4/
    """
    sd = _require_pyhdf()
    # Open the HDF4 file in read-only mode (sd.SDC.READ = 1).
    hdf = sd.SD(str(file_path), sd.SDC.READ)
    try:
        data_sds = hdf.select(data_sds_name)
        lat_sds = hdf.select(lat_sds_name)
        lon_sds = hdf.select(lon_sds_name)

        # Read the full arrays into memory.
        # pyhdf .get() returns Python lists; numpy converts them efficiently.
        data_arr = np.array(data_sds.get(), dtype=np.float32)
        lats_arr = np.array(lat_sds.get(), dtype=np.float64)
        lons_arr = np.array(lon_sds.get(), dtype=np.float64)
    finally:
        # Closing the SD object also closes all its SDS handles.
        hdf.end()

    return data_arr, lats_arr, lons_arr
