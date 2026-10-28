"""HDF-EOS ``StructMetadata.0`` grid parsing for the footprint matching readers.

``StructMetadata.0`` is the ODL text in which HDF-EOS2 (HDF4) and HDF-EOS5 (HDF5) files
define each grid's dimensions, corners and projection. Reading the attribute is
format-specific (``_hdf4_io``, ``_hdf5_io``); parsing its text is not, and lives here.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _EosGrid:
    """One grid's definition in an HDF-EOS ``StructMetadata.0`` attribute.

    Attributes
    ----------
    xdim, ydim : int
        Grid columns and rows (``XDim``, ``YDim``).
    upper_left, lower_right : tuple[float, float]
        Outer corners of the grid, ``(x, y)`` as written (``UpperLeftPointMtrs``,
        ``LowerRightMtrs``), in the projection's units: meters for a projected grid,
        packed ``DDDMMMSSS.SS`` degrees for a geographic one.
    projection : str
        The GCTP projection name (``Projection``), e.g. ``GCTP_SNSOID`` or ``GCTP_LAMAZ``.
    proj_params : tuple[str, ...] or None
        The ``ProjParams`` values as written, or ``None`` when the grid has none; for a
        projected grid the first is the sphere radius in meters.
    """

    xdim: int
    ydim: int
    upper_left: tuple[float, float]
    lower_right: tuple[float, float]
    projection: str
    proj_params: tuple[str, ...] | None


def struct_metadata_error(source: str | Path, what: str, advice: str) -> ValueError:
    """Build the error for a ``StructMetadata.0`` that does not describe the grid a reader expects.

    Parameters
    ----------
    source : str or Path
        The file the attribute was read from.
    what : str
        What is wrong, as a predicate of "StructMetadata.0", e.g. ``"has no XDim field"``.
    advice : str
        What the person staging the file should check, naming the product.

    Returns
    -------
    ValueError
        The error, for the caller to raise.
    """
    return ValueError(f"{source}: StructMetadata.0 {what}. {advice}")


def parse_eos_grid(struct_meta: str, source: str | Path, advice: str, grid_name: str | None = None) -> _EosGrid:
    """Parse one grid's dimensions, corners and projection from HDF-EOS ``StructMetadata.0``.

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
        ``UpperLeftPointMtrs``, ``LowerRightMtrs`` or ``Projection``.
    """
    block = struct_meta
    for_grid = ""
    if grid_name is not None:
        match = re.search(
            r'GROUP=(GRID_\d+)\s+GridName="' + re.escape(grid_name) + r'"(.*?)END_GROUP=\1', struct_meta, re.DOTALL
        )
        if match is None:
            raise struct_metadata_error(source, f"has no grid named {grid_name!r}", advice)
        block = match.group(2)
        for_grid = f" for grid {grid_name!r}"

    def _parse(pattern: str, field: str) -> re.Match:  # type: ignore[type-arg]
        m = re.search(pattern, block)
        if m is None:
            raise struct_metadata_error(source, f"has no {field} field{for_grid}", advice)
        return m

    xdim = int(_parse(r"\bXDim\s*=\s*(\d+)", "XDim").group(1))
    ydim = int(_parse(r"\bYDim\s*=\s*(\d+)", "YDim").group(1))
    ul = _parse(r"UpperLeftPointMtrs\s*=\s*\(([^,]+),([^)]+)\)", "UpperLeftPointMtrs")
    lr = _parse(r"LowerRightMtrs\s*=\s*\(([^,]+),([^)]+)\)", "LowerRightMtrs")
    projection = _parse(r"\bProjection\s*=\s*(\w+)", "Projection").group(1)
    proj_params = re.search(r"ProjParams\s*=\s*\(([^)]*)\)", block)
    return _EosGrid(
        xdim=xdim,
        ydim=ydim,
        upper_left=(float(ul.group(1)), float(ul.group(2))),
        lower_right=(float(lr.group(1)), float(lr.group(2))),
        projection=projection,
        proj_params=None if proj_params is None else tuple(value.strip() for value in proj_params.group(1).split(",")),
    )
