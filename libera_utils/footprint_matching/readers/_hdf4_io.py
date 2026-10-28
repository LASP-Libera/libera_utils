"""HDF4 I/O helpers for the footprint matching reader plugin system.

HDF4 files are used by the IGBP land cover reader (MCD12Q1 tiles on the MODIS sinusoidal
grid) and the NSIDC NISE sea-ice reader (HDF-EOS2). Both read them with pyhdf, which
wraps the HDF4 C library; the grid definition they carry is parsed by ``_eos_metadata``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyhdf.SD as sd

from libera_utils.footprint_matching.readers._eos_metadata import parse_eos_grid, struct_metadata_error


def read_hdf4_struct_metadata(file_path: str | Path) -> str:
    """Return the HDF-EOS2 ``StructMetadata.0`` attribute of an HDF4 file, or ``""`` when it has none.

    Parameters
    ----------
    file_path : str or Path
        Path to the HDF4 file.

    Returns
    -------
    str
        The attribute's text.

    Raises
    ------
    pyhdf.error.HDF4Error
        If the file cannot be opened.
    """
    hdf = sd.SD(str(file_path), sd.SDC.READ)
    try:
        return hdf.attributes().get("StructMetadata.0", "")
    finally:
        hdf.end()


@dataclass(frozen=True)
class _SinusoidalGeometry:
    """Pixel geometry of one tile on the MODIS sinusoidal grid, from its HDF-EOS2 ``StructMetadata.0``.

    Pixel ``(r, c)`` has its center at sinusoidal ``x = x_ul + (c + 0.5) * dx``,
    ``y = y_ul + (r + 0.5) * dy`` meters, which is latitude ``y / radius_m`` and
    longitude ``x / (radius_m * cos(latitude))``, both in radians.

    Attributes
    ----------
    x_ul, y_ul : float
        Upper-left corner of the tile in sinusoidal meters (``UpperLeftPointMtrs``).
    dx, dy : float
        Pixel size in meters along columns and rows; ``dy`` is negative (rows run
        north to south).
    radius_m : float
        Sphere radius of the projection, meters (``ProjParams`` first value).
    """

    x_ul: float
    y_ul: float
    dx: float
    dy: float
    radius_m: float


def read_modis_sinusoidal_hdf4(file_path: str, data_sds_name: str) -> tuple[np.ndarray, _SinusoidalGeometry]:
    """Read one HDF-EOS2 tile on the sinusoidal grid MODIS land products use, and its pixel geometry.

    Products on this grid (e.g., MCD12Q1) do not store latitude/longitude as SDS
    arrays: the tile geometry is the HDF-EOS2 ``StructMetadata.0`` attribute's
    upper-left/lower-right corners in sinusoidal meters, its ``XDim``/``YDim``, and
    its projection and sphere radius. This function returns the data as stored and
    that geometry; it computes no per-pixel coordinates, so a caller can geolocate
    only the pixels it needs.

    Parameters
    ----------
    file_path : str
        Path to the HDF-EOS2 tile file.
    data_sds_name : str
        Name of the SDS to read (e.g., ``"LC_Type1"`` for MCD12Q1 IGBP).

    Returns
    -------
    tuple[np.ndarray, _SinusoidalGeometry]
        ``(data, geometry)``: ``data`` is the SDS in its stored dtype, shape
        ``(n_rows, n_cols)``; ``geometry`` locates every pixel.

    Raises
    ------
    pyhdf.error.HDF4Error
        If the file cannot be opened or has no SDS named ``data_sds_name``.
    ValueError
        If ``StructMetadata.0`` lacks a tile geometry field or ``ProjParams``, names a
        projection other than ``GCTP_SNSOID``, carries no positive sphere radius in
        ``ProjParams``, or its ``XDim``/``YDim`` do not match the SDS shape.

    Notes
    -----
    Sinusoidal projection conversion:

    .. code-block:: text

        lat = Y / R
        lon = X / (R * cos(lat))

    where ``R`` is the sphere radius from ``ProjParams`` (6 371 007.181 m on the MODIS sinusoidal grid).
    """
    hdf = sd.SD(str(file_path), sd.SDC.READ)
    try:
        data = np.array(hdf.select(data_sds_name).get())
        struct_meta: str = hdf.attributes().get("StructMetadata.0", "")
    finally:
        hdf.end()

    advice = (
        "A tile on the MODIS sinusoidal grid (e.g. MCD12Q1) carries its grid geometry there; check that the file "
        "is an unmodified HDF-EOS2 tile as distributed."
    )
    grid = parse_eos_grid(struct_meta, file_path, advice)
    if grid.projection != "GCTP_SNSOID":
        raise struct_metadata_error(
            file_path,
            f"names Projection={grid.projection}, not the sinusoidal GCTP_SNSOID projection the pixels are "
            "geolocated with",
            advice,
        )
    if grid.proj_params is None:
        raise struct_metadata_error(file_path, "has no ProjParams field", advice)
    radius_text = grid.proj_params[0]
    radius_m = float(radius_text)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise struct_metadata_error(
            file_path,
            f"gives ProjParams sphere radius {radius_text!r}, which is not a positive length",
            advice,
        )
    if data.shape != (grid.ydim, grid.xdim):
        raise struct_metadata_error(
            file_path,
            f"gives YDim={grid.ydim}, XDim={grid.xdim} but the {data_sds_name} SDS has shape {data.shape}",
            advice,
        )

    (x_ul, y_ul), (x_lr, y_lr) = grid.upper_left, grid.lower_right
    geometry = _SinusoidalGeometry(
        x_ul=x_ul,
        y_ul=y_ul,
        dx=(x_lr - x_ul) / grid.xdim,
        dy=(y_lr - y_ul) / grid.ydim,
        radius_m=radius_m,
    )
    return data, geometry
