"""HDF-EOS metadata parsing for the footprint matching readers.

``StructMetadata.0`` is the ODL text in which HDF-EOS2 (HDF4) and HDF-EOS5 (HDF5) files
define each grid's dimensions, corners and projection. ``coremetadata.0`` is the ODL text
holding a granule's ECS inventory metadata, among it the ``RANGEDATETIME`` group that dates
the granule. Reading an attribute is format-specific (``_hdf4_io``, ``_hdf5_io``); parsing
its text is not, and lives here.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The ECS inventory metadata fields that date a granule, in ``(start, end)`` order: each a date
# (YYYY-MM-DD) and a time of day (hh:mm:ss[.f], optionally ending in Z), in UTC.
_ECS_RANGE_FIELDS: tuple[str, ...] = ("RANGEBEGINNINGDATE", "RANGEBEGINNINGTIME", "RANGEENDINGDATE", "RANGEENDINGTIME")


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


def ecs_range_span(
    beginning_date: str, beginning_time: str, ending_date: str, ending_time: str, *, source: str | Path
) -> tuple[np.datetime64, np.datetime64]:
    """Combine a granule's ECS range dates and times into its ``(start, end)`` span.

    Parameters
    ----------
    beginning_date, beginning_time, ending_date, ending_time : str
        The ``RangeBeginningDate``, ``RangeBeginningTime``, ``RangeEndingDate`` and
        ``RangeEndingTime`` values as written: a ``YYYY-MM-DD`` date and an
        ``hh:mm:ss[.f]`` UTC time of day, optionally ending in ``Z``.
    source : str or Path
        The file they were read from, for error messages.

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        ``(start, end)`` as UTC ``datetime64[ns]``.

    Raises
    ------
    ValueError
        If a date and time do not combine into an ISO 8601 instant, or the end is before
        the start.
    """
    span = []
    for date, time in ((beginning_date, beginning_time), (ending_date, ending_time)):
        instant = f"{date.strip()}T{time.strip().rstrip('Z')}"
        try:
            span.append(np.datetime64(instant, "ns"))
        except ValueError as exc:
            raise ValueError(
                f"{source}: the range date {date!r} and time {time!r} do not form an ISO 8601 instant, so the "
                "granule cannot be dated. Check that the file is the unmodified granule as distributed."
            ) from exc
    if span[1] < span[0]:
        raise ValueError(f"{source}: the granule's range ends ({span[1]}) before it begins ({span[0]}).")
    return span[0], span[1]


def parse_core_metadata_range(core_meta: str, source: str | Path) -> tuple[np.datetime64, np.datetime64]:
    """Read a granule's ``(start, end)`` span from the ``RANGEDATETIME`` group of its ``coremetadata.0``.

    Parameters
    ----------
    core_meta : str
        The ``coremetadata.0`` ODL text.
    source : str or Path
        The file it was read from, for error messages.

    Returns
    -------
    tuple[np.datetime64, np.datetime64]
        ``(start, end)`` as UTC ``datetime64[ns]``.

    Raises
    ------
    ValueError
        If any of ``RANGEBEGINNINGDATE``, ``RANGEBEGINNINGTIME``, ``RANGEENDINGDATE`` or
        ``RANGEENDINGTIME`` has no ``VALUE``, or they do not form a valid span (see
        :func:`ecs_range_span`).
    """
    values = []
    for field in _ECS_RANGE_FIELDS:
        match = re.search(
            r"OBJECT\s*=\s*" + field + r'\b.*?VALUE\s*=\s*"([^"]*)".*?END_OBJECT\s*=\s*' + field + r"\b",
            core_meta,
            re.DOTALL,
        )
        if match is None:
            raise ValueError(
                f"{source}: coremetadata.0 has no {field} value, so the granule cannot be dated. Check that the "
                "file is the unmodified granule as distributed."
            )
        values.append(match.group(1))
    return ecs_range_span(*values, source=source)
