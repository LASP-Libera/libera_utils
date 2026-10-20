"""IGBP land cover reader plugin for the footprint matching pipeline.

Data source: MODIS MCD12Q1 (Terra+Aqua Combined Land Cover Type Yearly L3 Global 500m)
- Product: MCD12Q1
- Version: 061 (current)
- Format: HDF4 (.hdf)
- Spatial resolution: ~500m (0.5 km), stored in this reader as 1.0 km nominal
- Temporal coverage: Annual, available from 2001 to near-present
- Spatial coverage: Global, sinusoidal MODIS tile grid (e.g., h09v05)
- Classification scheme: IGBP Land Cover (17 classes + water + fill)

References
----------
Product page: https://lpdaac.usgs.gov/products/mcd12q1v061/
User guide:   https://lpdaac.usgs.gov/documents/1409/MCD12_User_Guide_V61.pdf
LP DAAC Access: https://appeears.earthdatacloud.nasa.gov/ (EarthData login required)
SDS naming:   "LC_Type1" for IGBP classification scheme
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers._hdf4_io import read_modis_sinusoidal_hdf4
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import OperationalMode, PointSamples, TimeMatch, VariableSpec

# IGBP fill / no-data value in the MCD12Q1 HDF4 file.
# Value 255 is the standard fill for uint8 SDS fields in this product.
_IGBP_FILL_VALUE: float = 255.0

# MCD12Q1 ``LC_Type1`` valid IGBP land-cover codes are 1..17, where 17 = Water Bodies
# (255 is fill). These codes are 1-based and align 1:1 with the low 17 members of
# ``scene_identification.scene_id.IGBPSurfaceType`` (1 = Evergreen Needleleaf .. 17 =
# Water Bodies), so a modal class read here maps straight through
# ``calculate_trmm_surface_type`` without any renumbering. The enum's 18..20 (tundra /
# fresh-snow / sea-ice) belong to the 20-class TRMM/CERES IGBP variant and are never
# emitted by MODIS ``LC_Type1``; a value of 0 is likewise not a valid class (it only
# ever arises downstream as the no-data sentinel when a footprint has no usable IGBP
# pixels). Out-of-domain codes are masked to NaN at read time so this reader provably
# emits only enum-consistent classes.
_IGBP_VALID_MIN: int = 1
_IGBP_VALID_MAX: int = 17

# Histogram size for the categorical PSF aggregation. Sized to the full 20-class
# ``IGBPSurfaceType`` domain (an upper bound on ``LC_Type1``'s 1..17) so the bins line
# up with the enum; unused high bins simply receive zero weight.
_N_IGBP_CATEGORIES: int = 20


class IGBPReader(StaticReader):
    """Read IGBP land cover classification from MODIS MCD12Q1 HDF4 tile files.

    The MCD12Q1 product stores IGBP land cover as ``LC_Type1`` (uint8) in
    sinusoidal-projected HDF4 tiles. Geographic coordinates are not stored as SDS
    arrays; they are computed from the HDF-EOS ``StructMetadata.0`` tile-corner
    metadata via :func:`read_modis_sinusoidal_hdf4`, which returns **2-D
    per-pixel** latitude/longitude grids.

    Geolocation: native pixels, not a regular grid
    ----------------------------------------------
    The sinusoidal projection is *not* axis-aligned in lat/lon -- longitude varies
    along each row (``lon = X / (R cos(lat))``), so a single longitude per column
    (a column mean) places no real pixel and mis-geolocates the data. Every pixel is
    therefore served at its own file-derived position as a
    :class:`~libera_utils.footprint_matching.types.PointSamples` point, and the PSF
    weigher is the first place a pixel is weighted. Several tiles (one file each)
    staged for a day are concatenated.

    The yearly product has no time axis, so the staged granules are the time match
    (``TIME_MATCH = DAILY``).

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"igbp"``.
    RESOLUTION_KM : float
        Nominal resolution 1.0 km (actual MODIS 500 m native, rounded for PSF
        weighting calculations).
    VARIABLES : tuple[VariableSpec, ...]
        Single variable: ``"surface_type"`` (categorical, 20 IGBP classes).

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The MCD12Q1 HDF4 tile file(s) (e.g., ``MCD12Q1.A2023001.h09v05.061.hdf``).
    """

    READER_KEY: str = "igbp"
    # MCD12Q1 land cover is derived from Terra+Aqua MODIS.
    INSTRUMENT: str = "MODIS"
    RESOLUTION_KM: float = 1.0
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="surface_type",
            dtype="int16",
            aggregation="weighted_mode",
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
    )
    # A footprint spans several IGBP land-cover classes. ``surface_type`` above is the single
    # aggregated result; these three report the ranked mix (1st/2nd/3rd most-common class by
    # PSF-weighted area), computed during PSF aggregation from the same pixels rather than read
    # from a source field -- hence here rather than in VARIABLES.
    ADDITIONAL_PRODUCT_VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="surface_type_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
        VariableSpec(
            name="surface_type_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
        VariableSpec(
            name="surface_type_tertiary",
            dtype="int16",
            aggregation="weighted_mode_tertiary",
            required_mode=OperationalMode.CAM,
            n_categories=_N_IGBP_CATEGORIES,
        ),
    )

    def _load_native(self) -> PointSamples:
        """Read every staged tile and flatten it to geolocated land-cover points.

        Returns
        -------
        PointSamples
            ``lats``/``lons`` are ``(n_pixels,)`` float64 pixel-center coordinates
            derived from the sinusoidal projection metadata, and ``values`` is float32
            shape ``(1, n_pixels)`` holding the IGBP class code per pixel. Fill pixels
            (original value 255) and any code outside the valid IGBP domain (``1..17``)
            are NaN so the ``weighted_mode`` aggregation never selects fill or an
            out-of-domain code as the modal land-cover class.
        """
        tile_lats: list[np.ndarray] = []
        tile_lons: list[np.ndarray] = []
        tile_values: list[np.ndarray] = []
        for path in self.file_paths:
            data_2d, lats_2d, lons_2d = read_modis_sinusoidal_hdf4(
                file_path=str(path),
                data_sds_name="LC_Type1",
                fill_value=_IGBP_FILL_VALUE,
            )
            # Mask anything outside the valid IGBP land-cover domain (1..17) -- including
            # the 255 fill and any stray 0 -- to NaN before flattening, so the mode
            # aggregation never counts fill as a category and, downstream,
            # calculate_trmm_surface_type never sees a code it cannot map.
            in_domain = (data_2d >= _IGBP_VALID_MIN) & (data_2d <= _IGBP_VALID_MAX)
            tile_values.append(np.where(in_domain, data_2d, np.nan).astype(np.float32).ravel())
            tile_lats.append(np.asarray(lats_2d, dtype=np.float64).ravel())
            tile_lons.append(np.asarray(lons_2d, dtype=np.float64).ravel())
        return PointSamples(
            lats=np.concatenate(tile_lats),
            lons=np.concatenate(tile_lons),
            values=np.concatenate(tile_values)[np.newaxis, :],
        )
