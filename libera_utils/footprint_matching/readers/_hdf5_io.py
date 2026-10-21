"""Shared HDF5 I/O helper for the footprint matching reader plugin system.

HDF5 files are used by the VIIRS BRDF reader (VJ143C1). This module provides
a lazy-import guard for h5py (mirrors the pattern in _hdf4_io.py for pyhdf)
so that the rest of the footprint_matching package remains importable in
environments where h5py is not installed.

Why lazy import?
----------------
h5py links against the HDF5 C library. In environments where that library is
absent, importing h5py at module load time would make the entire
footprint_matching package un-importable. Deferring the import to function
call time via ``_require_h5py()`` keeps the rest of the package importable and
surfaces a clear, actionable error only when a BRDF read is actually attempted.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py as h5py_type

# The HDF-EOS5 metadata dataset that defines every grid's dimensions, corners and projection.
_HE5_STRUCT_METADATA_PATH: str = "HDFEOS INFORMATION/StructMetadata.0"
# How much of StructMetadata.0 an error message quotes.
_STRUCT_METADATA_EXCERPT_CHARS: int = 400
# The per-field attributes a VJ143C1 BRDF field carries and the read applies.
_BRDF_FIELD_ATTRIBUTES: tuple[str, ...] = ("_FillValue", "valid_range", "scale_factor", "add_offset")


def _require_h5py() -> h5py_type:
    """Import h5py and return the module, raising ImportError with instructions if absent.

    Returns
    -------
    module
        The ``h5py`` module.

    Raises
    ------
    ImportError
        If h5py is not installed or the HDF5 C library is missing, with
        instructions for installing both.
    """
    try:
        import h5py  # noqa: PLC0415

        return h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read HDF5 files (VIIRS BRDF VJ143C1). "
            "Install instructions:\n"
            "  pip:   pip install h5py\n"
            "  conda: conda install -c conda-forge h5py\n"
            "See https://docs.h5py.org/en/stable/build.html for details."
        ) from exc


def _dms_to_degrees(packed: float) -> float:
    """Decode an HDF-EOS packed ``DDDMMMSSS.SS`` angle to decimal degrees."""
    sign = -1.0 if packed < 0 else 1.0
    value = abs(packed)
    degrees = value // 1_000_000
    minutes = (value % 1_000_000) // 1_000
    seconds = value % 1_000
    return sign * (degrees + minutes / 60.0 + seconds / 3600.0)


def _parse_he5_grid_metadata(struct_metadata: str, grid_name: str, source: str) -> dict[str, float | int]:
    """Return one geographic grid's dimensions and corners from HDF-EOS5 ``StructMetadata.0``.

    Parameters
    ----------
    struct_metadata : str
        The decoded ``StructMetadata.0`` text.
    grid_name : str
        The grid's ``GridName`` (e.g. ``"VIIRS_CMG_BRDF"``).
    source : str
        The file, for error messages.

    Returns
    -------
    dict[str, float | int]
        ``xdim``, ``ydim``, and the corners in decimal degrees: ``ul_lon``,
        ``ul_lat``, ``lr_lon``, ``lr_lat``.

    Raises
    ------
    ValueError
        If the grid or one of its ``XDim``, ``YDim``, ``UpperLeftPointMtrs``,
        ``LowerRightMtrs`` or ``Projection`` fields is absent, or the projection is
        not the geographic ``HE5_GCTP_GEO``.
    """

    def _fail(what: str) -> ValueError:
        excerpt = struct_metadata[:_STRUCT_METADATA_EXCERPT_CHARS]
        return ValueError(
            f"{source}: StructMetadata.0 {what}. A VJ143C1 granule defines its CMG grid there; check that the file "
            f"is an unmodified HDF-EOS5 VJ143C1 granule. StructMetadata.0 begins: {excerpt!r}"
        )

    block = re.search(rf'GridName="{re.escape(grid_name)}"(.*?)END_GROUP=GRID_\d+', struct_metadata, flags=re.DOTALL)
    if block is None:
        raise _fail(f"has no grid named {grid_name!r}")
    text = block.group(1)

    def _field(pattern: str, field: str) -> re.Match:  # type: ignore[type-arg]
        m = re.search(pattern, text)
        if m is None:
            raise _fail(f"has no {field} field for grid {grid_name!r}")
        return m

    xdim = int(_field(r"\bXDim\s*=\s*(\d+)", "XDim").group(1))
    ydim = int(_field(r"\bYDim\s*=\s*(\d+)", "YDim").group(1))
    ul = _field(r"UpperLeftPointMtrs\s*=\s*\(([^,]+),([^)]+)\)", "UpperLeftPointMtrs")
    lr = _field(r"LowerRightMtrs\s*=\s*\(([^,]+),([^)]+)\)", "LowerRightMtrs")
    projection = _field(r"\bProjection\s*=\s*(\w+)", "Projection").group(1)
    if projection != "HE5_GCTP_GEO":
        raise _fail(f"gives grid {grid_name!r} Projection={projection}, not the geographic HE5_GCTP_GEO")
    return {
        "xdim": xdim,
        "ydim": ydim,
        # Geographic grids store their corners as packed DDDMMMSSS.SS degrees, longitude first.
        "ul_lon": _dms_to_degrees(float(ul.group(1))),
        "ul_lat": _dms_to_degrees(float(ul.group(2))),
        "lr_lon": _dms_to_degrees(float(lr.group(1))),
        "lr_lat": _dms_to_degrees(float(lr.group(2))),
    }


def _cell_centres(edges: np.ndarray, first_edge: float, last_edge: float, axis_name: str, source: str) -> np.ndarray:
    """Turn a VJ143C1 coordinate axis of leading cell edges into cell centres.

    The file's ``lat``/``lon`` hold each cell's edge on the side the axis starts from
    (the grid's upper-left corner): ``lat`` 90 -> -89.95 and ``lon`` -180 -> 179.95 for
    the 0.05 degree CMG. The centre is half a cell further along the axis.

    Raises
    ------
    ValueError
        If the axis spacing is not uniform, or the axis does not run from
        ``first_edge`` to ``last_edge`` (the StructMetadata corners).
    """
    steps = np.diff(edges)
    cell = float(np.mean(np.abs(steps))) if steps.size else 0.0
    if steps.size == 0 or not np.allclose(np.abs(steps), cell):
        raise ValueError(
            f"{source}: the {axis_name} axis is not uniformly spaced, so it is not the regular CMG grid this reader "
            "serves. Check that the file is an unmodified VJ143C1 granule."
        )
    direction = float(np.sign(steps[0]))
    if not (np.isclose(edges[0], first_edge) and np.isclose(edges[-1] + direction * cell, last_edge)):
        raise ValueError(
            f"{source}: the {axis_name} axis runs {edges[0]} .. {edges[-1]} in steps of {cell}, which does not span "
            f"the StructMetadata corners {first_edge} .. {last_edge} as cell edges. Check that the file is an "
            "unmodified VJ143C1 granule."
        )
    return edges + direction * 0.5 * cell


def read_viirs_brdf_hdf5(
    file_path: str,
    field_names: list[str],
    grid_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a VJ143C1 HDF5 file and read named BRDF parameter fields.

    The VJ143C1 product stores BRDF kernel parameters as int16, each field with its
    own ``_FillValue``, ``valid_range``, ``scale_factor`` and ``add_offset``. This
    function reads the requested fields, replaces fill and out-of-range pixels with
    NaN, applies the scale and offset, and returns the cell-centre coordinates in
    ascending latitude order.

    Parameters
    ----------
    file_path : str
        Path to a VJ143C1 HDF5 (HDF-EOS5) file.
    field_names : list[str]
        Names of the data fields to read from the grid's ``Data Fields`` group
        (e.g., ``["BRDF_Albedo_Parameter1_shortwave"]``). Fields are stacked
        in the order given along axis 0 of the returned data array.
    grid_name : str
        The HDF-EOS5 grid (``"VIIRS_CMG_BRDF"`` for VJ143C1). Its group
        ``HDFEOS/GRIDS/<grid_name>/Data Fields`` holds the fields and the
        ``lat``/``lon`` axes, and its ``StructMetadata.0`` entry the grid definition.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(data, lats, lons)`` where:

        - ``data`` is float32, shape ``(n_fields, n_lat, n_lon)``.  Fill and
          out-of-range pixels are NaN.  Latitude axis is in **ascending** order
          (flipped from the file's descending storage).
        - ``lats`` is float64, shape ``(n_lat,)``, cell centres, **ascending**.
        - ``lons`` is float64, shape ``(n_lon,)``, cell centres, ascending.

    Raises
    ------
    ImportError
        If h5py is not installed (see :func:`_require_h5py`).
    KeyError
        If the grid's group, a requested field, the lat/lon datasets or
        ``StructMetadata.0`` are not found in the file.
    OSError
        If the file cannot be opened (not found, corrupt HDF5 header, etc.).
    ValueError
        If a field lacks one of ``_FillValue``, ``valid_range``, ``scale_factor`` or
        ``add_offset``; if ``StructMetadata.0`` is not text or does not describe the grid as a
        geographic (``HE5_GCTP_GEO``) grid of the axes' size; or if the axes are not
        uniformly spaced cell edges spanning its corners.

    Notes
    -----
    The fill and valid-range test runs on the raw int16 values, before scaling, and
    is done here rather than with ``_points.apply_fill_and_valid_range``: that helper
    returns float64, about 207 MB of transient memory per global field next to the
    preallocated float32 array.

    References
    ----------
    VJ143C1 product page:
        https://www.earthdata.nasa.gov/data/catalog/lpcloud-vj143c1-002
    h5py documentation: https://docs.h5py.org/en/stable/
    HDF-EOS5 format: https://hdfeos.org/software/hdfeos5.php
    """
    h5py = _require_h5py()
    source = str(file_path)

    with h5py.File(source, "r") as f:
        raw_metadata = f[_HE5_STRUCT_METADATA_PATH][()]
        if not isinstance(raw_metadata, bytes):
            raise ValueError(
                f"{source}: {_HE5_STRUCT_METADATA_PATH} is {type(raw_metadata).__name__}, not the fixed-length string "
                "HDF-EOS5 stores it as. Check that the file is an unmodified VJ143C1 granule."
            )
        grid = _parse_he5_grid_metadata(raw_metadata.decode("ascii").rstrip("\x00"), grid_name, source)
        group = f[f"HDFEOS/GRIDS/{grid_name}/Data Fields"]

        lat_edges = np.array(group["lat"], dtype=np.float64)
        lon_edges = np.array(group["lon"], dtype=np.float64)
        if (lat_edges.size, lon_edges.size) != (grid["ydim"], grid["xdim"]):
            raise ValueError(
                f"{source}: the lat/lon axes have {lat_edges.size} x {lon_edges.size} values but StructMetadata.0 "
                f"gives grid {grid_name!r} YDim={grid['ydim']}, XDim={grid['xdim']}. Check that the file is an "
                "unmodified VJ143C1 granule."
            )
        lats = _cell_centres(lat_edges, grid["ul_lat"], grid["lr_lat"], "lat", source)
        lons = _cell_centres(lon_edges, grid["ul_lon"], grid["lr_lon"], "lon", source)

        # One array for every field, filled in place (nine global fields are ~0.9 GB).
        data = np.empty((len(field_names), lats.size, lons.size), dtype=np.float32)
        for i, name in enumerate(field_names):
            ds = group[name]
            missing = [attr for attr in _BRDF_FIELD_ATTRIBUTES if attr not in ds.attrs]
            if missing:
                raise ValueError(
                    f"{source}: BRDF field {name!r} has no {', '.join(missing)} attribute. Every VJ143C1 field "
                    "carries all four; stage the granule as distributed."
                )
            # HDF-EOS5 stores each attribute as a 1-element array (valid_range as 2).
            attrs = {attr: np.asarray(ds.attrs[attr]).ravel() for attr in _BRDF_FIELD_ATTRIBUTES}
            raw = np.array(ds)
            low, high = attrs["valid_range"][:2]
            invalid = (raw == attrs["_FillValue"][0]) | (raw < low) | (raw > high)
            np.multiply(raw, float(attrs["scale_factor"][0]), out=data[i], casting="unsafe")
            data[i] += float(attrs["add_offset"][0])
            data[i][invalid] = np.nan

    # Flip to ascending latitude order (the file stores 90 -> -90), as a view.
    if lats.size > 1 and lats[0] > lats[-1]:
        lats = lats[::-1]
        data = data[:, ::-1, :]

    return data, lats, lons
