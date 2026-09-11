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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py as h5py_type


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


def read_viirs_brdf_hdf5(
    file_path: str,
    field_names: list[str],
    hdf5_data_path: str = "HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields",
    fill_value: int = 32767,
    scale_factor: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open a VJ143C1 HDF5 file and read named BRDF parameter fields.

    The VJ143C1 product stores BRDF kernel parameters as int16 with a
    ``scale_factor`` attribute. This function reads the requested fields,
    applies the scale factor, replaces fill pixels with NaN, and returns the
    lat/lon coordinate arrays in ascending latitude order.

    Parameters
    ----------
    file_path : str
        Absolute path to a VJ143C1 HDF5 (HDF-EOS5) file.
    field_names : list[str]
        Names of the data fields to read from ``hdf5_data_path``
        (e.g., ``["BRDF_Albedo_Parameter1_shortwave"]``). Fields are stacked
        in the order given along axis 0 of the returned data array.
    hdf5_data_path : str, optional
        HDF5 path to the group containing the data fields and lat/lon arrays.
        Default matches the VJ143C1 product layout.
    fill_value : int, optional
        Raw int16 fill sentinel in the file. Pixels with this value are
        converted to NaN after scaling. Default 32767 (VJ143C1 standard).
    scale_factor : float, optional
        Multiplicative scale factor applied to raw int16 values to obtain
        physical units (dimensionless BRDF kernel parameters). Default 0.001.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(data, lats, lons)`` where:

        - ``data`` is float32, shape ``(n_fields, n_lat, n_lon)``.  Fill pixels
          are NaN.  Latitude axis is in **ascending** order (flipped from the
          file's descending storage).
        - ``lats`` is float64, shape ``(n_lat,)``, **ascending** order.
        - ``lons`` is float64, shape ``(n_lon,)``.

    Raises
    ------
    ImportError
        If h5py is not installed (see :func:`_require_h5py`).
    KeyError
        If a requested field name or the lat/lon datasets are not found in the
        file.
    OSError
        If the file cannot be opened (not found, corrupt HDF5 header, etc.).

    Notes
    -----
    The VJ143C1 latitude array is stored in descending order (90 → −90).
    This function flips both ``lats`` and the data spatial axis to ascending
    order so downstream callers can assume ascending latitudes.

    The ``scale_factor`` attribute is read from each dataset if present; the
    ``scale_factor`` argument is used as a fallback for synthetic test fixtures
    that omit the attribute.

    References
    ----------
    VJ143C1 product page:
        https://www.earthdata.nasa.gov/data/catalog/lpcloud-vj143c1-002
    h5py documentation: https://docs.h5py.org/en/stable/
    HDF-EOS5 format: https://hdfeos.org/software/hdfeos5.php
    """
    h5py = _require_h5py()

    with h5py.File(str(file_path), "r") as f:
        group = f[hdf5_data_path]

        # --- read coordinate arrays ---
        lats_raw = np.array(group["lat"], dtype=np.float64)  # descending (90 → -90)
        lons_raw = np.array(group["lon"], dtype=np.float64)  # ascending (-180 → 180)

        # --- read and scale each requested field ---
        arrays: list[np.ndarray] = []
        for name in field_names:
            ds = group[name]

            # Read as native int16 so fill detection is exact (integer comparison),
            # then scale to float32. Do NOT read as float32 first — that makes the
            # fill sentinel indistinguishable from a legitimate scaled value at
            # some scale factors.
            raw_int = np.array(ds, dtype=np.int16)
            fill_mask = raw_int == np.int16(fill_value)

            # Prefer the scale_factor stored in the dataset attributes; fall back
            # to the caller-supplied default (used by synthetic test fixtures).
            sf_attr = ds.attrs.get("scale_factor", None)
            sf = float(sf_attr.flat[0]) if sf_attr is not None else scale_factor

            scaled = raw_int.astype(np.float32) * sf
            scaled[fill_mask] = np.nan

            arrays.append(scaled)

        data = np.stack(arrays, axis=0)  # (n_fields, n_lat, n_lon)

    # Flip to ascending latitude order (file stores 90 → -90).
    if lats_raw.size > 1 and lats_raw[0] > lats_raw[-1]:
        lats_raw = lats_raw[::-1]
        data = data[:, ::-1, :]

    return data, lats_raw, lons_raw
