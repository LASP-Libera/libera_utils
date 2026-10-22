"""Shared HDF4 I/O helper for the footprint matching reader plugin system.

HDF4 files are used by the IGBP land cover reader (MCD12Q1 sinusoidal tiles) and the
NSIDC NISE sea-ice reader (HDF-EOS4). Both read them with pyhdf, which wraps the HDF4
C library.
"""

import re
from dataclasses import dataclass

import numpy as np
import pyhdf.SD as sd

# How much of StructMetadata.0 an error message quotes, so a reader can see what the file
# does carry without the message reproducing the whole (tens of kB) attribute.
_STRUCT_METADATA_EXCERPT_CHARS: int = 400


@dataclass(frozen=True)
class _SinusoidalGeometry:
    """Pixel geometry of one MODIS sinusoidal tile, from its HDF-EOS ``StructMetadata.0``.

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
    """Read one MODIS sinusoidal-projection HDF4 tile and its pixel geometry.

    MODIS gridded products (e.g., MCD12Q1) do not store latitude/longitude as SDS
    arrays: the tile geometry is the HDF-EOS ``StructMetadata.0`` attribute's
    upper-left/lower-right corners in sinusoidal meters, its ``XDim``/``YDim``, and
    its projection and sphere radius. This function returns the data as stored and
    that geometry; it computes no per-pixel coordinates, so a caller can geolocate
    only the pixels it needs.

    Parameters
    ----------
    file_path : str
        Path to the MODIS HDF4 tile file.
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
        If ``StructMetadata.0`` lacks a tile geometry field, names a projection other
        than ``GCTP_SNSOID``, carries no positive sphere radius in ``ProjParams``, or
        its ``XDim``/``YDim`` do not match the SDS shape.

    Notes
    -----
    Sinusoidal projection conversion:

    .. code-block:: text

        lat = Y / R
        lon = X / (R * cos(lat))

    where ``R`` is the sphere radius from ``ProjParams`` (6 371 007.181 m for MODIS).
    """
    hdf = sd.SD(str(file_path), sd.SDC.READ)
    try:
        data = np.array(hdf.select(data_sds_name).get())
        struct_meta: str = hdf.attributes().get("StructMetadata.0", "")
    finally:
        hdf.end()

    def _fail(what: str) -> ValueError:
        excerpt = struct_meta[:_STRUCT_METADATA_EXCERPT_CHARS]
        return ValueError(
            f"{file_path}: StructMetadata.0 {what}. A MODIS sinusoidal tile (e.g. MCD12Q1) carries its grid "
            f"geometry there; check that the file is an unmodified HDF-EOS2 MODIS tile. "
            f"StructMetadata.0 begins: {excerpt!r}"
        )

    def _parse(pattern: str, field: str) -> re.Match:  # type: ignore[type-arg]
        m = re.search(pattern, struct_meta)
        if m is None:
            raise _fail(f"has no {field} field")
        return m

    xdim = int(_parse(r"\bXDim\s*=\s*(\d+)", "XDim").group(1))
    ydim = int(_parse(r"\bYDim\s*=\s*(\d+)", "YDim").group(1))
    ul = _parse(r"UpperLeftPointMtrs\s*=\s*\(([^,]+),([^)]+)\)", "UpperLeftPointMtrs")
    lr = _parse(r"LowerRightMtrs\s*=\s*\(([^,]+),([^)]+)\)", "LowerRightMtrs")
    projection = _parse(r"\bProjection\s*=\s*(\w+)", "Projection").group(1)
    if projection != "GCTP_SNSOID":
        raise _fail(f"names Projection={projection}, not the sinusoidal GCTP_SNSOID this reader geolocates")
    radius_match = _parse(r"ProjParams\s*=\s*\(\s*([^,)]+)", "ProjParams")
    radius_m = float(radius_match.group(1))
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise _fail(f"gives ProjParams sphere radius {radius_match.group(1)!r}, which is not a positive length")
    if data.shape != (ydim, xdim):
        raise _fail(f"gives YDim={ydim}, XDim={xdim} but the {data_sds_name} SDS has shape {data.shape}")

    x_ul, y_ul = float(ul.group(1)), float(ul.group(2))
    x_lr, y_lr = float(lr.group(1)), float(lr.group(2))
    geometry = _SinusoidalGeometry(
        x_ul=x_ul,
        y_ul=y_ul,
        dx=(x_lr - x_ul) / xdim,
        dy=(y_lr - y_ul) / ydim,
        radius_m=radius_m,
    )
    return data, geometry
