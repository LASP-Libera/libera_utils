"""VIIRS surface BRDF reader plugin for the footprint matching pipeline.

Data source: VIIRS/NOAA-20 BRDF/Albedo Model Parameters Daily L3 Global 0.05° CMG

- Product: VJ143C1 (Collection 002)
- Format: HDF5/HDF-EOS5 (.h5), opened via h5py
- Spatial resolution: 0.05° (~5.6 km at equator)
- Grid: 3600 × 7200 global Climate Model Grid (CMG). The file's ``lat``/``lon``
  hold each cell's upper-left edge, latitude DESCENDING (90 → −89.95, longitude
  −180 → 179.95); this reader serves cell centers (±89.975, ±179.975) with
  latitude ascending, like every other reader, after checking the axes against
  the grid's HDF-EOS5 ``StructMetadata.0`` corners.
- Temporal resolution: Daily (using 16 days of accumulated observations)
- Parameters: 3 RossThick-LiSparse (RTLS) kernel coefficients × 3 broadbands
  (shortwave, visible, near-infrared) = 9 output variables

BRDF kernel parameters
-----------------------
RTLS decomposition: R = f_iso + f_vol * k_vol + f_geo * k_geo

  fiso  — isotropic (Lambertian) scattering weight
  fvol  — volume scattering (Ross-Thick kernel) weight
  fgeo  — geometric (Li-Sparse) scattering weight

These are dimensionless; values range from 0.0 to ~3.0 for fiso and ~1.0 for
fvol/fgeo. Stored as int16; each field's own ``_FillValue`` (32767), ``valid_range``
(0..32766), ``scale_factor`` (0.001) and ``add_offset`` (0) are applied.

Output variable naming: ``brdf_{band}_{param}``

  ========= =================
  Band      HDF5 field suffix
  ========= =================
  shortwave shortwave
  vis       vis
  nir       nir
  ========= =================

  ========= ======================
  Parameter HDF5 field prefix
  ========= ======================
  fiso      BRDF_Albedo_Parameter1
  fvol      BRDF_Albedo_Parameter2
  fgeo      BRDF_Albedo_Parameter3
  ========= ======================

References
----------
Product page:
    https://www.earthdata.nasa.gov/data/catalog/lpcloud-vj143c1-002
User guide:
    https://lpdaac.usgs.gov/documents/194/VNP43_User_Guide_V2.pdf
Data access:
    https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/  (Earthdata login required)
File naming:
    VJ143C1.A{YYYYDDD}.{version}.{YYYYDDDHHMMSS}.h5
"""

import h5py
import numpy as np

from libera_utils.footprint_matching.readers._hdf5_io import _parse_he5_grid_metadata, read_he5_struct_metadata
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import GridSamples, OperationalMode, TimeMatch, VariableSpec

# The HDF-EOS5 grid holding the BRDF parameter fields and coordinate arrays.
_BRDF_GRID_NAME: str = "VIIRS_CMG_BRDF"

# Each output variable and the HDF5 field it reads from the grid's Data Fields group.
# This order is VARIABLES order and axis 0 of the reader's data array.
_BRDF_FIELDS: tuple[tuple[str, str], ...] = (
    ("brdf_shortwave_fiso", "BRDF_Albedo_Parameter1_shortwave"),
    ("brdf_shortwave_fvol", "BRDF_Albedo_Parameter2_shortwave"),
    ("brdf_shortwave_fgeo", "BRDF_Albedo_Parameter3_shortwave"),
    ("brdf_vis_fiso", "BRDF_Albedo_Parameter1_vis"),
    ("brdf_vis_fvol", "BRDF_Albedo_Parameter2_vis"),
    ("brdf_vis_fgeo", "BRDF_Albedo_Parameter3_vis"),
    ("brdf_nir_fiso", "BRDF_Albedo_Parameter1_nir"),
    ("brdf_nir_fvol", "BRDF_Albedo_Parameter2_nir"),
    ("brdf_nir_fgeo", "BRDF_Albedo_Parameter3_nir"),
)

# The per-field attributes a VJ143C1 BRDF field carries and the read applies.
_BRDF_FIELD_ATTRIBUTES: tuple[str, ...] = ("_FillValue", "valid_range", "scale_factor", "add_offset")


def _cell_centers(edges: np.ndarray, first_edge: float, last_edge: float, axis_name: str, source: str) -> np.ndarray:
    """Turn a VJ143C1 coordinate axis of leading cell edges into cell centers.

    The file's ``lat``/``lon`` hold each cell's edge on the side the axis starts from
    (the grid's upper-left corner): ``lat`` 90 -> -89.95 and ``lon`` -180 -> 179.95 for
    the 0.05 degree CMG. The center is half a cell further along the axis.

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


# The products that carry the VIIRS BRDF variables: every FMATCH product.
_MODES: frozenset[OperationalMode] = frozenset(OperationalMode)


class VIIRSBRDFReader(StaticReader):
    """Read VIIRS surface BRDF model parameters from a VJ143C1 HDF5 file.

    Loads nine RTLS kernel parameter fields (3 bands x 3 parameters) from a
    VJ143C1 VIIRS daily BRDF product as a ``(9, n_lat, n_lon)`` grid stacked in
    ``VARIABLES`` order. The product is a 16-day composite issued daily with no
    time axis, so the staged day is the time match (``TIME_MATCH = DAILY``).

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"viirs_brdf"``.
    RESOLUTION_KM : float
        5.6 km (0.05° CMG grid spacing at equator).
    VARIABLES : tuple[VariableSpec, ...]
        Nine BRDF kernel parameter variables (3 bands × 3 kernel weights),
        all float32 with ``weighted_mean`` aggregation.

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The one VJ143C1 HDF5 (HDF-EOS5) file for the processing day.
    """

    READER_KEY: str = "viirs_brdf"
    # VJ143C1 BRDF/Albedo is produced from VIIRS aboard NOAA-20 (JPSS-1).
    # TODO[LIBSDC-899]: decide whether NOAA-22 VIIRS replaces NOAA-20 here (a cutover of this reader) or gets a
    # second reader.
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 5.6  # 0.05 deg is about 5.6 km at the equator
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=name,
            dtype="float32",
            aggregation="weighted_mean",
            modes=_MODES,
            n_categories=None,
        )
        for name, _ in _BRDF_FIELDS
    )

    def _load_native(self) -> GridSamples:
        """Read all nine BRDF fields from the HDF5 file once.

        Each field is int16 with its own ``_FillValue``, ``valid_range``,
        ``scale_factor`` and ``add_offset``. Fill and out-of-range pixels become NaN,
        the scale and offset are applied, and the cell-edge ``lat``/``lon`` axes are
        checked against the ``StructMetadata.0`` corners and served as cell centers,
        latitude ascending.

        Returns
        -------
        GridSamples
            ``data`` is float32 shape ``(9, n_lat, n_lon)``, axis 0 in ``VARIABLES``
            order, fill and out-of-range as NaN; ``lat_axis`` and ``lon_axis`` are
            float64 cell centers, ascending.

        Raises
        ------
        KeyError
            If the grid's group, a field, the lat/lon datasets or ``StructMetadata.0``
            are not found in the file.
        OSError
            If the file cannot be opened (not found, corrupt HDF5 header, etc.).
        ValueError
            If a field lacks one of ``_FillValue``, ``valid_range``, ``scale_factor`` or
            ``add_offset``; if ``StructMetadata.0`` is not text or does not describe the
            grid as a geographic (``HE5_GCTP_GEO``) grid of the axes' size; or if the
            axes are not uniformly spaced cell edges spanning its corners.

        Notes
        -----
        The fill and valid-range test runs on the raw int16 values, before scaling, and
        each field is scaled straight into the preallocated float32 array.
        """
        source = str(self.single_file_path)
        with h5py.File(source, "r") as f:
            grid = _parse_he5_grid_metadata(read_he5_struct_metadata(f, source), _BRDF_GRID_NAME, source)
            group = f[f"HDFEOS/GRIDS/{_BRDF_GRID_NAME}/Data Fields"]

            lat_edges = np.array(group["lat"], dtype=np.float64)
            lon_edges = np.array(group["lon"], dtype=np.float64)
            if (lat_edges.size, lon_edges.size) != (grid["ydim"], grid["xdim"]):
                raise ValueError(
                    f"{source}: the lat/lon axes have {lat_edges.size} x {lon_edges.size} values but StructMetadata.0 "
                    f"gives grid {_BRDF_GRID_NAME!r} YDim={grid['ydim']}, XDim={grid['xdim']}. Check that the file "
                    "is an unmodified VJ143C1 granule."
                )
            lats = _cell_centers(lat_edges, grid["ul_lat"], grid["lr_lat"], "lat", source)
            lons = _cell_centers(lon_edges, grid["ul_lon"], grid["lr_lon"], "lon", source)

            # One array for every field, filled in place (nine global fields are ~0.9 GB).
            data = np.empty((len(_BRDF_FIELDS), lats.size, lons.size), dtype=np.float32)
            for i, (_, field) in enumerate(_BRDF_FIELDS):
                ds = group[field]
                missing = [attr for attr in _BRDF_FIELD_ATTRIBUTES if attr not in ds.attrs]
                if missing:
                    raise ValueError(
                        f"{source}: BRDF field {field!r} has no {', '.join(missing)} attribute. Every VJ143C1 field "
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
        return GridSamples(lat_axis=lats, lon_axis=lons, data=data)
