"""HDF-EOS5 helpers for the footprint matching readers.

Reads an HDF-EOS5 file's ``StructMetadata.0`` grid definition with h5py, and parses it
with ``_eos_metadata``. What a product stores in its grids, and how its fields are
decoded, belongs to the product's reader (the VIIRS BRDF reader, VJ143C1, in ``brdf.py``).
"""

import h5py

from libera_utils.footprint_matching.readers._eos_metadata import parse_eos_grid, struct_metadata_error

# The HDF-EOS5 metadata dataset that defines every grid's dimensions, corners and projection.
_HE5_STRUCT_METADATA_PATH: str = "HDFEOS INFORMATION/StructMetadata.0"
# What to check when StructMetadata.0 does not describe the expected grid.
_HE5_STRUCT_METADATA_ADVICE: str = (
    "An HDF-EOS5 file defines each grid there; check that the file is the unmodified granule as distributed."
)


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
    grid = parse_eos_grid(struct_metadata, source, _HE5_STRUCT_METADATA_ADVICE, grid_name=grid_name)
    if grid.projection != "HE5_GCTP_GEO":
        raise struct_metadata_error(
            source,
            f"gives grid {grid_name!r} Projection={grid.projection}, not the geographic HE5_GCTP_GEO",
            _HE5_STRUCT_METADATA_ADVICE,
        )
    (ul_lon, ul_lat), (lr_lon, lr_lat) = grid.upper_left, grid.lower_right
    return {
        "xdim": grid.xdim,
        "ydim": grid.ydim,
        # Geographic grids store their corners as packed DDDMMMSSS.SS degrees, longitude first.
        "ul_lon": _dms_to_degrees(ul_lon),
        "ul_lat": _dms_to_degrees(ul_lat),
        "lr_lon": _dms_to_degrees(lr_lon),
        "lr_lat": _dms_to_degrees(lr_lat),
    }


def read_he5_struct_metadata(f: h5py.File, source: str) -> str:
    """Return an open HDF-EOS5 file's ``StructMetadata.0`` text.

    Parameters
    ----------
    f : h5py.File
        The open file.
    source : str
        The file, for error messages.

    Returns
    -------
    str
        The decoded text, trailing NUL padding removed.

    Raises
    ------
    KeyError
        If the file has no ``StructMetadata.0`` dataset.
    ValueError
        If ``StructMetadata.0`` is not the fixed-length string HDF-EOS5 stores it as.
    """
    raw_metadata = f[_HE5_STRUCT_METADATA_PATH][()]
    if not isinstance(raw_metadata, bytes):
        raise ValueError(
            f"{source}: {_HE5_STRUCT_METADATA_PATH} is {type(raw_metadata).__name__}, not the fixed-length string "
            "HDF-EOS5 stores it as. Check that the file is the unmodified granule as distributed."
        )
    return raw_metadata.decode("ascii").rstrip("\x00")
