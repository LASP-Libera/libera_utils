"""VIIRS surface BRDF reader plugin for the footprint matching pipeline.

Data source: VIIRS/NOAA-20 BRDF/Albedo Model Parameters Daily L3 Global 0.05° CMG
- Product: VJ143C1 (Collection 002)
- Format: HDF5/HDF-EOS5 (.h5), opened via h5py
- Spatial resolution: 0.05° (~5.6 km at equator)
- Grid: 3600 × 7200 global Climate Model Grid (CMG), ascending lat (−90 → 90)
  NOTE: the file stores latitude in DESCENDING order (90 → −90); this reader
  flips it to ascending order for consistency with all other readers.
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
fvol/fgeo. Stored as int16 with scale_factor = 0.001; fill = 32767.

Output variable naming: ``brdf_{band}_{param}``

  Band     | HDF5 field suffix
  ---------|-------------------
  shortwave| shortwave
  vis      | vis
  nir      | nir

  Parameter | HDF5 field prefix
  ----------|------------------
  fiso      | BRDF_Albedo_Parameter1
  fvol      | BRDF_Albedo_Parameter2
  fgeo      | BRDF_Albedo_Parameter3

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

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from libera_utils.footprint_matching.readers._hdf5_io import read_viirs_brdf_hdf5
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import GridSamples, OperationalMode, TimeMatch, VariableSpec

# HDF5 group path containing the BRDF parameter fields and coordinate arrays.
_BRDF_DATA_PATH: str = "HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields"

# Mapping from VariableSpec name → HDF5 field name inside _BRDF_DATA_PATH.
# Order here must match VARIABLES tuple order (used for indexing data axis 0).
_BRDF_FIELD_MAP: dict[str, str] = {
    "brdf_shortwave_fiso": "BRDF_Albedo_Parameter1_shortwave",
    "brdf_shortwave_fvol": "BRDF_Albedo_Parameter2_shortwave",
    "brdf_shortwave_fgeo": "BRDF_Albedo_Parameter3_shortwave",
    "brdf_vis_fiso": "BRDF_Albedo_Parameter1_vis",
    "brdf_vis_fvol": "BRDF_Albedo_Parameter2_vis",
    "brdf_vis_fgeo": "BRDF_Albedo_Parameter3_vis",
    "brdf_nir_fiso": "BRDF_Albedo_Parameter1_nir",
    "brdf_nir_fvol": "BRDF_Albedo_Parameter2_nir",
    "brdf_nir_fgeo": "BRDF_Albedo_Parameter3_nir",
}


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
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 5.6  # 0.05 deg is about 5.6 km at the equator
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="brdf_shortwave_fiso",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_shortwave_fvol",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_shortwave_fgeo",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_vis_fiso",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_vis_fvol",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_vis_fgeo",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_nir_fiso",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_nir_fvol",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
        VariableSpec(
            name="brdf_nir_fgeo",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.CAM,
            n_categories=None,
        ),
    )

    def __init__(self, file_paths: Path | Sequence[Path]) -> None:
        super().__init__(file_paths)
        # Build ordered field name list matching VARIABLES tuple order.
        self._field_names: list[str] = [_BRDF_FIELD_MAP[v.name] for v in self.VARIABLES]

    def _load_native(self) -> GridSamples:
        """Read all 9 BRDF fields from the HDF5 file once.

        Reads via
        :func:`~libera_utils.footprint_matching.readers._hdf5_io.read_viirs_brdf_hdf5`,
        which returns data in ascending latitude order with fill as NaN.

        Returns
        -------
        GridSamples
            ``data`` is float32 shape ``(9, n_lat, n_lon)`` (axis 0 in ``VARIABLES``
            order, fill as NaN).
        """
        data, lats, lons = read_viirs_brdf_hdf5(
            file_path=str(self.single_file_path),
            field_names=self._field_names,
            hdf5_data_path=_BRDF_DATA_PATH,
        )
        return GridSamples(lat_axis=lats, lon_axis=lons, data=np.asarray(data, dtype=np.float32))
