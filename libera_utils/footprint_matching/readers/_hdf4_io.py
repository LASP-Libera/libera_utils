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

import re

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
        Name of the SDS holding the primary data variable (e.g., for VIIRS the
        caller iterates per-variable).
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


def read_modis_sinusoidal_hdf4(
    file_path: str,
    data_sds_name: str,
    fill_value: float = 255.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a MODIS sinusoidal-projection HDF4 tile, computing lat/lon from StructMetadata.

    MODIS gridded products (e.g., MCD12Q1) do not store latitude/longitude as
    SDS arrays. Instead, the tile geometry is encoded in the HDF-EOS
    ``StructMetadata.0`` attribute as upper-left/lower-right corners in
    sinusoidal-projection metres. This function parses those corners, derives
    pixel-centre coordinates, and converts them to geographic degrees.

    Parameters
    ----------
    file_path : str
        Absolute path to the MODIS HDF4 tile file.
    data_sds_name : str
        Name of the SDS to read (e.g., ``"LC_Type1"`` for MCD12Q1 IGBP).
    fill_value : float, optional
        Fill/missing sentinel value in the data SDS. Returned as-is; the caller
        is responsible for masking. Default is 255 (MCD12Q1 standard).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(data, lats, lons)`` where:

        - ``data`` is float32, shape ``(nrows, ncols)``.
        - ``lats`` is float64, shape ``(nrows, ncols)`` — pixel-centre latitudes.
        - ``lons`` is float64, shape ``(nrows, ncols)`` — pixel-centre longitudes.

    Raises
    ------
    ImportError
        If pyhdf is not installed (see :func:`_require_pyhdf`).
    KeyError
        If ``data_sds_name`` is not found in the file.
    ValueError
        If ``StructMetadata.0`` cannot be parsed (missing tile geometry fields).

    Notes
    -----
    Sinusoidal projection conversion:

    .. code-block:: text

        lat = Y / R
        lon = X / (R * cos(lat))

    where ``R`` is the sphere radius from ``ProjParams`` (default 6 371 007.181 m,
    the MODIS standard). Both ``lats`` and ``lons`` are returned as 2-D arrays so
    the caller can perform pixel-level bounding-box tests without special-casing the
    non-rectangular sinusoidal footprint.
    """
    sd = _require_pyhdf()
    hdf = sd.SD(str(file_path), sd.SDC.READ)
    try:
        data_arr = np.array(hdf.select(data_sds_name).get(), dtype=np.float32)
        struct_meta: str = hdf.attributes().get("StructMetadata.0", "")
    finally:
        hdf.end()

    # --- parse tile geometry from HDF-EOS StructMetadata ---
    def _parse(pattern: str, text: str) -> re.Match:  # type: ignore[type-arg]
        m = re.search(pattern, text)
        if m is None:
            raise ValueError(f"Cannot parse '{pattern}' from StructMetadata.0 in {file_path}")
        return m

    xdim = int(_parse(r"\bXDim\s*=\s*(\d+)", struct_meta).group(1))
    ydim = int(_parse(r"\bYDim\s*=\s*(\d+)", struct_meta).group(1))
    ul = _parse(r"UpperLeftPointMtrs\s*=\s*\(([^,]+),([^)]+)\)", struct_meta)
    lr = _parse(r"LowerRightMtrs\s*=\s*\(([^,]+),([^)]+)\)", struct_meta)
    x_ul, y_ul = float(ul.group(1)), float(ul.group(2))
    x_lr, y_lr = float(lr.group(1)), float(lr.group(2))

    r_match = re.search(r"ProjParams\s*=\s*\(\s*([^,)]+)", struct_meta)
    sphere_radius: float = (
        float(r_match.group(1))
        if r_match and float(r_match.group(1)) > 0
        else 6371007.181
    )

    # --- pixel-centre coordinates in sinusoidal metres ---
    xs = x_ul + (np.arange(xdim) + 0.5) * ((x_lr - x_ul) / xdim)  # (xdim,)
    ys = y_ul + (np.arange(ydim) + 0.5) * ((y_lr - y_ul) / ydim)  # (ydim,)

    # --- sinusoidal → geographic (degrees) ---
    lat_rad = ys / sphere_radius  # (ydim,)
    # Broadcast to 2-D (ydim, xdim) for pixel-level lon computation
    lats_2d = np.degrees(np.broadcast_to(lat_rad[:, np.newaxis], (ydim, xdim)))
    lons_2d = np.degrees(
        np.broadcast_to(xs[np.newaxis, :], (ydim, xdim))
        / (sphere_radius * np.cos(lat_rad[:, np.newaxis]))
    )

    return data_arr, np.array(lats_2d, dtype=np.float64), np.array(lons_2d, dtype=np.float64)
