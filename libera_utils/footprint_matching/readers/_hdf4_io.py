"""Shared HDF4 I/O helper for the footprint matching reader plugin system.

HDF4 files are used by the IGBP land cover reader (MCD12Q1 sinusoidal tiles) and the
NSIDC NISE sea-ice reader (HDF-EOS4). Both read them with pyhdf, which wraps the HDF4
C library.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyhdf.SD as sd

# How much of StructMetadata.0 an error message quotes, so a reader can see what the file
# does carry without the message reproducing the whole (tens of kB) attribute.
_STRUCT_METADATA_EXCERPT_CHARS: int = 400


@dataclass(frozen=True)
class _EosGrid:
    """One grid's definition in an HDF-EOS2 ``StructMetadata.0`` attribute.

    Attributes
    ----------
    xdim, ydim : int
        Grid columns and rows (``XDim``, ``YDim``).
    upper_left_m, lower_right_m : tuple[float, float]
        Outer corners of the grid, ``(x, y)`` in projection meters (``UpperLeftPointMtrs``,
        ``LowerRightMtrs``).
    projection : str
        The GCTP projection name (``Projection``), e.g. ``GCTP_SNSOID`` or ``GCTP_LAMAZ``.
    proj_params : tuple[str, ...]
        The ``ProjParams`` values as written; the first is the sphere radius in meters.
    """

    xdim: int
    ydim: int
    upper_left_m: tuple[float, float]
    lower_right_m: tuple[float, float]
    projection: str
    proj_params: tuple[str, ...]


def struct_metadata_error(source: str | Path, struct_meta: str, what: str, advice: str) -> ValueError:
    """Build the error for a ``StructMetadata.0`` that does not describe the grid a reader expects.

    Parameters
    ----------
    source : str or Path
        The file the attribute was read from.
    struct_meta : str
        The ``StructMetadata.0`` text; the message quotes its start.
    what : str
        What is wrong, as a predicate of "StructMetadata.0", e.g. ``"has no XDim field"``.
    advice : str
        What the person staging the file should check, naming the product.

    Returns
    -------
    ValueError
        The error, for the caller to raise.
    """
    excerpt = struct_meta[:_STRUCT_METADATA_EXCERPT_CHARS]
    return ValueError(f"{source}: StructMetadata.0 {what}. {advice} StructMetadata.0 begins: {excerpt!r}")


def parse_eos_grid(struct_meta: str, source: str | Path, advice: str, grid_name: str | None = None) -> _EosGrid:
    """Parse one grid's dimensions, corners and projection from HDF-EOS2 ``StructMetadata.0``.

    Parameters
    ----------
    struct_meta : str
        The ``StructMetadata.0`` text.
    source : str or Path
        The file it was read from, for error messages.
    advice : str
        What to check when the attribute is unusable, naming the product (see
        :func:`struct_metadata_error`).
    grid_name : str, optional
        The ``GridName`` of the grid to parse. ``None`` reads the first value of each
        field anywhere in the attribute, for a file that holds one grid.

    Returns
    -------
    _EosGrid
        The grid's definition, as written.

    Raises
    ------
    ValueError
        If no grid is named ``grid_name``, or the grid lacks ``XDim``, ``YDim``,
        ``UpperLeftPointMtrs``, ``LowerRightMtrs``, ``Projection`` or ``ProjParams``.
    """
    block = struct_meta
    if grid_name is not None:
        match = re.search(
            r'GROUP=(GRID_\d+)\s+GridName="' + re.escape(grid_name) + r'"(.*?)END_GROUP=\1', struct_meta, re.DOTALL
        )
        if match is None:
            raise struct_metadata_error(source, struct_meta, f"has no grid named {grid_name!r}", advice)
        block = match.group(2)

    def _parse(pattern: str, field: str) -> re.Match:  # type: ignore[type-arg]
        m = re.search(pattern, block)
        if m is None:
            raise struct_metadata_error(source, struct_meta, f"has no {field} field", advice)
        return m

    xdim = int(_parse(r"\bXDim\s*=\s*(\d+)", "XDim").group(1))
    ydim = int(_parse(r"\bYDim\s*=\s*(\d+)", "YDim").group(1))
    ul = _parse(r"UpperLeftPointMtrs\s*=\s*\(([^,]+),([^)]+)\)", "UpperLeftPointMtrs")
    lr = _parse(r"LowerRightMtrs\s*=\s*\(([^,]+),([^)]+)\)", "LowerRightMtrs")
    projection = _parse(r"\bProjection\s*=\s*(\w+)", "Projection").group(1)
    proj_params = _parse(r"ProjParams\s*=\s*\(([^)]*)\)", "ProjParams").group(1)
    return _EosGrid(
        xdim=xdim,
        ydim=ydim,
        upper_left_m=(float(ul.group(1)), float(ul.group(2))),
        lower_right_m=(float(lr.group(1)), float(lr.group(2))),
        projection=projection,
        proj_params=tuple(value.strip() for value in proj_params.split(",")),
    )


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

    advice = (
        "A MODIS sinusoidal tile (e.g. MCD12Q1) carries its grid geometry there; check that the file is an "
        "unmodified HDF-EOS2 MODIS tile."
    )
    grid = parse_eos_grid(struct_meta, file_path, advice)
    if grid.projection != "GCTP_SNSOID":
        raise struct_metadata_error(
            file_path,
            struct_meta,
            f"names Projection={grid.projection}, not the sinusoidal GCTP_SNSOID this reader geolocates",
            advice,
        )
    radius_text = grid.proj_params[0]
    radius_m = float(radius_text)
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise struct_metadata_error(
            file_path,
            struct_meta,
            f"gives ProjParams sphere radius {radius_text!r}, which is not a positive length",
            advice,
        )
    if data.shape != (grid.ydim, grid.xdim):
        raise struct_metadata_error(
            file_path,
            struct_meta,
            f"gives YDim={grid.ydim}, XDim={grid.xdim} but the {data_sds_name} SDS has shape {data.shape}",
            advice,
        )

    (x_ul, y_ul), (x_lr, y_lr) = grid.upper_left_m, grid.lower_right_m
    geometry = _SinusoidalGeometry(
        x_ul=x_ul,
        y_ul=y_ul,
        dx=(x_lr - x_ul) / grid.xdim,
        dy=(y_lr - y_ul) / grid.ydim,
        radius_m=radius_m,
    )
    return data, geometry
