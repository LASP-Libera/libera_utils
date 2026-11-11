"""Module for mapping radiometer footprints to scene IDs.

This module provides functionality for identifying and classifying atmospheric scenes
based on footprint data from satellite observations.

"""

import enum
import logging
import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import netCDF4 as nc
import numpy as np
import xarray as xr
from cloudpathlib import S3Path
from numpy.typing import NDArray

from libera_utils.config import config
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.smart_open import smart_open
from libera_utils.scene_identification.scene_definitions import SceneDefinition

logger = logging.getLogger(__name__)

# Names used throughout footprint data processing and when emitting a Libera SCENE-ID product. FootprintData
# carries data on the "RADIOMETER_TIME" dimension from creation onward: the radiometer-timescale scene-ID
# products (CAM, IMAGER) are written on that dimension with a "RADIOMETER_TIME" coordinate to exactly
# mirror the upstream L1B_RAD product, so scene IDs align 1:1 with L1B footprints. "RADIOMETER_TIME" names both
# the per-footprint dimension and the datetime coordinate written on it (a NetCDF / xarray dimension coordinate),
# so one constant is used throughout scene-ID processing. This constant is shared with the product runner (see
# libera_utils/scene_identification/scene_id_algorithm.py) and the product-definition YAML.
RADIOMETER_TIME_DIMENSION = "RADIOMETER_TIME"

# The camera-timescale counterpart of the constant above. The camera-timescale scene-ID product (CAM-CAMTIME) is
# written on a 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid mirroring its FMATCH-CAM-CAMTIME input: CAMERA_TIME is the unique,
# sorted image-acquisition axis (one 2048x2048 image = one entry) and PSEUDOFOOTPRINT is the 0-based subsection index within
# each image (subsections may overlap and do not tile the image). Both are dimension coordinates and every data
# variable hangs on both.
CAMERA_TIME_VARIABLE = "CAMERA_TIME"

# Product names of the identifier variables that the camera-timescale FMATCH products carry and the camera-timescale
# SCENE-ID products (SCENE-ID-CAM-CAMTIME, SCENE-ID-IMAGER-CAMTIME) pass straight through -- the four inclusive camera
# pixel-block bounds (Camera_Pixel_{X,Y}_{Min,Max}), the PSF bounding box, and the boresight geolocation -- each read
# from the FMATCH file and written to SCENE-ID under the same name, so a scene can be traced back
# to the exact camera pixels and ground footprint. These are copied verbatim (each keeping its own 2-D
# (CAMERA_TIME, PSEUDOFOOTPRINT) grid dimensions) by from_fmatch_cam_camtime / from_fmatch_imager_camtime and are not
# consumed by the classification; they simply ride along to the written product. The FMATCH-only Center_Pixel_X/Y
# (boresight pixel) is deliberately NOT listed: SCENE-ID does not carry it.
_FMATCH_CAMTIME_PASSTHROUGH_VARIABLES: tuple[str, ...] = (
    "Latitude",
    "Longitude",
    "Altitude",
    "PSF_Bbox_Lat_Min",
    "PSF_Bbox_Lat_Max",
    "PSF_Bbox_Lon_Min",
    "PSF_Bbox_Lon_Max",
    "Camera_Pixel_X_Min",
    "Camera_Pixel_X_Max",
    "Camera_Pixel_Y_Min",
    "Camera_Pixel_Y_Max",
)

# Name of the per-footprint data-quality bit-flag variable declared in the SCENE-ID product definitions
# (scene_id_cam.yml).
QUALITY_FLAG_VARIABLE = "Quality_Flag"

# Standard scene definitions shipped with libera_utils, keyed by scene type: the lowercased stem of the CSV
# filename. The value is the config.json key that resolves to the CSV path. Consumers can select a subset by type
# (e.g. the SCENE-ID-CAM product runs "erbe" and "unfiltering" but not "trmm"). The CSV filename stem, as spelled,
# names the output variables (Scene_ID_ERBE, Scene_Bin_ERBE_<variable>_Min); see SceneDefinition.
STANDARD_SCENE_DEFINITION_CONFIG_KEYS: dict[str, str] = {
    "trmm": "TRMM_SCENE_DEFINITION",
    "erbe": "ERBE_SCENE_DEFINITION",
    "unfiltering": "UNFILTERING_SCENE_DEFINITION",
}

# Sentinel marking "the caller did not pass scene_definitions, so use the defaults".
_USE_DEFAULT_SCENE_DEFINITIONS = object()


def standard_scene_definitions(scene_types: list[str] | None = None) -> list[SceneDefinition]:
    """Build the standard libera_utils scene definitions, optionally limited to specific types.

    Parameters
    ----------
    scene_types : list of str or None
        Scene types to include, drawn from the keys of ``STANDARD_SCENE_DEFINITION_CONFIG_KEYS``
        (``"trmm"``, ``"erbe"``, ``"unfiltering"``). If None (default), all standard definitions are returned in
        that dictionary's insertion order.

    Returns
    -------
    list[SceneDefinition]
        The requested standard scene definitions, loaded from the CSV paths configured in ``config.json``.

    Raises
    ------
    ValueError
        If a requested scene type is not a known standard definition.

    Notes
    -----
    These are only the definitions shipped with libera_utils. A user can supply a *custom* scene definition without
    editing library code: build a CSV with the same column layout as the shipped definitions in
    ``libera_utils/data/scene_definitions/`` (``scene_id`` plus ``<variable>_Min`` / ``<variable>_Max`` columns),
    wrap it in a :class:`~libera_utils.scene_identification.scene_definitions.SceneDefinition`, and pass it through
    the ``additional_scene_definitions_files`` argument of :meth:`FootprintData.identify_scenes` (or pass an explicit
    ``scene_definitions`` list). The CSV is the source of truth for the bins, which is what makes the classification
    reconfigurable in an analysis context. TODO[LIBSDC-857]: expand this into fuller end-user documentation of the
    custom-scene-definition workflow.
    """
    if scene_types is None:
        scene_types = list(STANDARD_SCENE_DEFINITION_CONFIG_KEYS)

    definitions: list[SceneDefinition] = []
    for scene_type in scene_types:
        if scene_type not in STANDARD_SCENE_DEFINITION_CONFIG_KEYS:
            raise ValueError(
                f"Unknown standard scene definition type '{scene_type}'. "
                f"Valid types are: {sorted(STANDARD_SCENE_DEFINITION_CONFIG_KEYS)}"
            )
        config_key = STANDARD_SCENE_DEFINITION_CONFIG_KEYS[scene_type]
        definitions.append(SceneDefinition(pathlib.Path(config.get(config_key))))
    return definitions


def add_placeholder_quality_flag(
    product: xr.Dataset, dimensions: str | Sequence[str] = RADIOMETER_TIME_DIMENSION
) -> xr.Dataset:
    """Add the SCENE-ID ``Quality_Flag`` variable as an all-zero placeholder.

    The SCENE-ID product definitions (e.g. ``scene_id_cam.yml``) declare a per-footprint ``Quality_Flag`` bit-flag
    variable. Per-footprint quality flagging is not implemented yet, so this fills every footprint with ``0``
    ("no flags set") so the product still conforms to its definition. The variable is added over the product's
    record grid, so the input must already carry those dimensions.

    Parameters
    ----------
    product : xr.Dataset
        A per-footprint product dataset to add the flag to.
    dimensions : str or sequence of str, optional
        Name(s) of the record grid dimension(s) to add the flag over. Defaults to ``RADIOMETER_TIME`` (the
        radiometer-timescale products, whose records are 1-D); the camera-timescale product passes the 2-D grid
        ``("CAMERA_TIME", "PSEUDOFOOTPRINT")`` so the flag spans one entry per (image, subsection).

    Returns
    -------
    xr.Dataset
        The same dataset with a ``Quality_Flag`` variable added (or overwritten).

    Notes
    -----
    The placeholder is emitted as ``uint32`` to match the ``Quality_Flag`` dtype in the product definition; the
    NetCDF writer's conformance step will re-cast if the definition ever changes.
    """
    # TODO[LIBSDC-810]: replace this all-zero placeholder with real per-footprint quality flagging (for example,
    # flagging unmatched footprints, fill/NaN inputs, and out-of-range viewing geometry). Until then every
    # footprint is reported as "no flags set" (0).
    grid_dimensions = (dimensions,) if isinstance(dimensions, str) else tuple(dimensions)
    grid_shape = tuple(product.sizes[dimension] for dimension in grid_dimensions)
    quality_flag = np.zeros(grid_shape, dtype=np.uint32)
    product[QUALITY_FLAG_VARIABLE] = (grid_dimensions, quality_flag)
    return product


def default_scene_definitions() -> list[SceneDefinition]:
    """Return the scene definitions applied by :meth:`FootprintData.identify_scenes` when none are specified.

    These are the full set of standard definitions: TRMM, ERBE, and unfiltering.

    Returns
    -------
    list[SceneDefinition]
        The default TRMM, ERBE, and unfiltering scene definitions.
    """
    return standard_scene_definitions()


class TRMMSurfaceType(enum.IntEnum):
    """Enumeration of TRMM surface types used in ERBE and TRMM scene classification.

    Attributes
    ----------
    OCEAN : int
        Ocean/water surfaces (value: 0)
    HI_SHRUB : int
        High vegetation/shrubland surfaces (value: 1)
    LOW_SHRUB : int
        Low vegetation/grassland surfaces (value: 2)
    DARK_DESERT : int
        Dark desert/bare soil surfaces (value: 3)
    BRIGHT_DESERT : int
        Bright desert/sand surfaces (value: 4)
    SNOW : int
        Snow/ice covered surfaces (value: 5)
    """

    OCEAN = 0
    HI_SHRUB = 1
    LOW_SHRUB = 2
    DARK_DESERT = 3
    BRIGHT_DESERT = 4
    SNOW = 5


class IGBPSurfaceType(enum.IntEnum):
    """Enumeration of surface types used in scene classification.

    These surface types are derived from IGBP (International Geosphere-Biosphere Programme)
    land cover classifications.

    Members are the 20 TRMM surface type categories, with values 1-20.
    """

    EVERGREEN_NEEDLELEAF_FOREST = 1
    EVERGREEN_BROADLEAF_FOREST = 2
    DECIDUOUS_NEEDLELEAF_FOREST = 3
    DECIDUOUS_BROADLEAF_FOREST = 4
    MIXED_FOREST = 5
    CLOSED_SHRUBLANDS = 6
    OPEN_SHRUBLANDS = 7
    WOODY_SAVANNAS = 8
    SAVANNAS = 9
    GRASSLANDS = 10
    PERMANENT_WETLANDS = 11
    CROPLANDS = 12
    URBAN = 13
    CROPLAND_MOSAICS = 14
    PERMANENT_SNOW_ICE = 15
    BARE_SOIL_ROCKS = 16
    WATER_BODIES = 17
    TUNDRA = 18
    FRESH_SNOW = 19
    SEA_ICE = 20

    @property
    def trmm_surface_type(self) -> TRMMSurfaceType:
        """Map IGBP surface type to corresponding TRMM surface type.

        Returns
        -------
        TRMMSurfaceType
            The corresponding TRMM surface type category

        Examples
        --------
        >>> IGBPSurfaceType.EVERGREEN_NEEDLELEAF_FOREST.trmm_surface_type
        <TRMMSurfaceType.HI_SHRUB: 1>
        >>> IGBPSurfaceType.WATER_BODIES.trmm_surface_type
        <TRMMSurfaceType.OCEAN: 0>
        """
        igbp_to_trmm_map = {
            1: TRMMSurfaceType.HI_SHRUB,
            2: TRMMSurfaceType.HI_SHRUB,
            3: TRMMSurfaceType.HI_SHRUB,
            4: TRMMSurfaceType.HI_SHRUB,
            5: TRMMSurfaceType.HI_SHRUB,
            6: TRMMSurfaceType.HI_SHRUB,
            7: TRMMSurfaceType.DARK_DESERT,
            8: TRMMSurfaceType.HI_SHRUB,
            9: TRMMSurfaceType.LOW_SHRUB,
            10: TRMMSurfaceType.LOW_SHRUB,
            11: TRMMSurfaceType.LOW_SHRUB,
            12: TRMMSurfaceType.LOW_SHRUB,
            13: TRMMSurfaceType.LOW_SHRUB,
            14: TRMMSurfaceType.LOW_SHRUB,
            15: TRMMSurfaceType.SNOW,
            16: TRMMSurfaceType.BRIGHT_DESERT,
            17: TRMMSurfaceType.OCEAN,
            18: TRMMSurfaceType.LOW_SHRUB,
            19: TRMMSurfaceType.SNOW,
            20: TRMMSurfaceType.SNOW,
        }
        return igbp_to_trmm_map[self.value]


class CLDPIXCloudPhase(enum.IntEnum):
    """CLDPIX cloud particle phase codes.

    FMATCH's ``CLDPIX_Cloud_Particle_Phase`` is the PSF-weighted mode of the CER_CLDPIX ``Cloud_Particle_Phase``
    code. The CER_CLDPIX_NOAA20-VIIRS files declare the codes in that variable's own CF ``flag_values`` and
    ``flag_meanings`` (as in the test granule
    ``tests/test_data/footprint_matching/CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015_subset.nc``):

    - 1 ``water_cloud``
    - 2 ``ice_cloud``
    - 3 ``no_cloud_retrieval``
    - 4 ``clear``
    - 5 ``bad_data``, which the CLDPIX reader drops

    Only the two codes that carry a cloud phase are named here. Codes 3 and 4 carry none (no retrieval, or no
    cloud), so :func:`map_cldpix_phase_to_trmm` maps them to NaN and those footprints stay unmatched for
    phase-gated scenes.

    Attributes
    ----------
    LIQUID : int
        ``water_cloud`` (code 1), mapped to the classifier's Cloud_Phase value 1.
    ICE : int
        ``ice_cloud`` (code 2), mapped to the classifier's Cloud_Phase value 2.
    """

    LIQUID = 1
    ICE = 2


def map_cldpix_phase_to_trmm(cldpix_phase: NDArray) -> NDArray[np.floating]:
    """Map CLDPIX cloud-particle-phase codes to the classifier's ``Cloud_Phase`` (1 = liquid, 2 = ice).

    The TRMM scene classifier bins ``Cloud_Phase`` as 1 (liquid) or 2 (ice) -- see the ``Cloud_Phase_Min/_Max``
    columns of ``TRMM.csv`` and :func:`calculate_cloud_phase`, which likewise emits 1 or 2. FMATCH's
    ``CLDPIX_Cloud_Particle_Phase`` instead carries the CER_CLDPIX ``Cloud_Particle_Phase`` integer code, of which the
    CLDPIX reader keeps 1-4 (it drops 5, which the file labels ``bad_data``). This is a *code remap*, deliberately
    unlike the continuous cloud-fraction-weighted :func:`calculate_cloud_phase`: each CLDPIX code is looked up and
    collapsed onto {1, 2}.

    Any code that is not a liquid/ice value (fill/NaN, or CLDPIX codes 3 ``no_cloud_retrieval`` and 4 ``clear``;
    see :class:`CLDPIXCloudPhase`) is mapped to ``NaN``. A ``NaN`` classification value leaves the footprint unmatched
    (scene ID 0) for any scene that bounds ``Cloud_Phase`` -- exactly the desired behavior when the phase is unknown.

    Parameters
    ----------
    cldpix_phase : ndarray
        CLDPIX cloud particle phase codes (any numeric dtype; NaN allowed for missing).

    Returns
    -------
    ndarray of float
        ``Cloud_Phase`` values (1.0 = liquid, 2.0 = ice, NaN = unknown), as ``float32`` so NaN is representable.
    """
    # Compare against the liquid and ice codes and build a float result so every other code can be NaN.
    phase = np.asarray(cldpix_phase, dtype=np.float32)
    result = np.full(phase.shape, np.nan, dtype=np.float32)
    result[phase == CLDPIXCloudPhase.LIQUID] = 1.0
    result[phase == CLDPIXCloudPhase.ICE] = 2.0
    return result


# Scene Property Calculations


def calculate_cloud_fraction(clear_area: float | NDArray[np.floating]) -> float | NDArray[np.floating]:
    """Calculate cloud fraction from clear sky area percentage.

    Parameters
    ----------
    clear_area : float or ndarray
        Clear area percentage (0-100)

    Returns
    -------
    float or ndarray
        Cloud fraction percentage (0-100), calculated as 100 - clear_area

    Raises
    ------
    ValueError
        If clear_area contains values less than 0 or greater than 100

    Examples
    --------
    >>> calculate_cloud_fraction(30.0)
    70.0
    >>> calculate_cloud_fraction(np.array([10, 25, 90]))
    array([90, 75, 10])
    """
    # Check if input is within valid range
    if np.any(clear_area < 0) or np.any(clear_area > 100):
        raise ValueError(f"Clear Area must be between 0 and 100. Got {clear_area}")

    cloud_fraction = 100.0 - clear_area
    return cloud_fraction


def calculate_surface_wind(
    surface_wind_u: float | NDArray[np.floating], surface_wind_v: float | NDArray[np.floating]
) -> float | NDArray[np.floating]:
    """Calculate total surface wind speed from u and v vector components.

    Parameters
    ----------
    surface_wind_u : float or ndarray
        U component of surface wind (m/s), indicating East/West direction
    surface_wind_v : float or ndarray
        V component of surface wind (m/s), indicating North/South direction

    Returns
    -------
    float or ndarray
        Total wind speed magnitude (m/s), or np.nan where input components are NaN

    Notes
    -----
    Wind speed is calculated using the Pythagorean theorem: sqrt(u^2 + v^2).
    NaN values in either component result in NaN output for that position.

    Examples
    --------
    >>> calculate_surface_wind(3.0, 4.0)
    5.0
    >>> calculate_surface_wind(np.array([3, np.nan]), np.array([4, 5]))
    array([5., nan])
    """
    surface_wind = np.sqrt(surface_wind_u**2 + surface_wind_v**2)
    # Handle NaN cases
    surface_wind = np.where(np.isnan(surface_wind_u) | np.isnan(surface_wind_v), np.nan, surface_wind)
    return surface_wind


def calculate_trmm_surface_type(igbp_surface_type: float | NDArray[np.number]) -> float | NDArray[np.floating]:
    """Convert an IGBP surface type code to its TRMM surface type classification.

    This maps IGBP -> TRMM (the direction used by scene identification): each IGBP land-cover code (1-20) is
    collapsed onto one of the six coarser TRMM surface types via :meth:`IGBPSurfaceType.trmm_surface_type`.
    A missing code (``NaN``, e.g. a footprint FMATCH found no IGBP pixels for) gives a missing surface type, which
    leaves that footprint unmatched (scene ID 0) for every scene that bounds the surface type.

    Parameters
    ----------
    igbp_surface_type : float or numpy.ndarray
        IGBP surface type codes (valid range 1-20), with ``NaN`` where the code is missing.

    Returns
    -------
    float or ndarray of float
        TRMM surface type codes (0=ocean ... 5=snow), with ``NaN`` where the input is missing.

    Raises
    ------
    ValueError
        If any non-missing input value is not a valid IGBP surface type (outside 1-20, or not a whole number).

    Notes
    -----
    The conversion uses a lookup table derived from the IGBP->TRMM map on :class:`IGBPSurfaceType`.

    Examples
    --------
    >>> calculate_trmm_surface_type(1)  # IGBP EVERGREEN_NEEDLELEAF_FOREST -> TRMM HI_SHRUB (1)
    1.0
    >>> calculate_trmm_surface_type(np.array([1, 17, np.nan]))  # 17 = WATER_BODIES -> OCEAN (0)
    array([ 1.,  0., nan])
    >>> calculate_trmm_surface_type(999)
    ValueError: Cannot convert IGBP surface type value to TRMM surface type: [999]
    """
    max_igbp = max(surface.value for surface in IGBPSurfaceType)
    lookup = np.full(max_igbp + 1, np.nan)
    for surface_type in IGBPSurfaceType:
        lookup[surface_type.value] = surface_type.trmm_surface_type

    igbp = np.asarray(igbp_surface_type, dtype=np.float64)
    missing = np.isnan(igbp)
    valid = ~missing & (igbp >= 1) & (igbp <= max_igbp) & (igbp == np.round(igbp))
    invalid = ~missing & ~valid
    if np.any(invalid):
        failed_values = np.atleast_1d(igbp)[np.atleast_1d(invalid)]
        failed_values = [
            int(value) if np.isfinite(value) and value == np.round(value) else float(value) for value in failed_values
        ]
        raise ValueError(f"Cannot convert IGBP surface type value to TRMM surface type: {failed_values}")

    result = np.where(valid, lookup[np.where(valid, igbp, 0).astype(int)], np.nan)
    return float(result) if np.ndim(result) == 0 else result


def calculate_cloud_fraction_weighted_optical_depth(
    optical_depth_lower: float | NDArray[np.floating],
    optical_depth_upper: float | NDArray[np.floating],
    cloud_fraction_lower: float | NDArray[np.floating],
    cloud_fraction_upper: float | NDArray[np.floating],
    cloud_fraction: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate weighted optical depth from upper and lower cloud layers.

    Combines optical depth measurements from two atmospheric layers using cloud fraction weighting to produce a single
    representative optical depth value.

    Parameters
    ----------
    optical_depth_lower : float or ndarray
        Optical depth for lower cloud layer (dimensionless)
    optical_depth_upper : float or ndarray
        Optical depth for upper cloud layer (dimensionless)
    cloud_fraction_lower : float or ndarray
        Cloud fraction for lower layer (0-100)
    cloud_fraction_upper : float or ndarray
        Cloud fraction for upper layer (0-100)
    cloud_fraction : float or ndarray
        Total cloud fraction (0-100)

    Returns
    -------
    float or ndarray
        Optical depth weighted by cloud fraction and summed across layers,
        or np.nan if no valid data or zero total cloud fraction

    """
    # Initialize result array
    result = np.zeros_like(optical_depth_lower, dtype=np.float64)

    # Check where cloud_fraction is non-zero
    no_clouds = cloud_fraction == 0

    # For each point with clouds, calculate weighted optical depth
    optical_temp_1 = np.where(
        (np.isnan(optical_depth_lower) | no_clouds), 0.0, (cloud_fraction_lower / cloud_fraction) * optical_depth_lower
    )

    optical_temp_2 = np.where(
        (np.isnan(optical_depth_upper) | no_clouds), 0.0, (cloud_fraction_upper / cloud_fraction) * optical_depth_upper
    )

    # Sum contributions
    weighted_optical_depth = optical_temp_1 + optical_temp_2

    # Set to NaN only if BOTH optical_depth values are NaN
    both_optical_nan = np.isnan(optical_depth_lower) & np.isnan(optical_depth_upper)

    # Apply the logic: NaN if no clouds OR both optical depths are NaN
    result = np.where(no_clouds | both_optical_nan, np.nan, weighted_optical_depth)

    return result


def calculate_cloud_phase(
    cloud_phase_lower: float | NDArray[np.floating],
    cloud_phase_upper: float | NDArray[np.floating],
    cloud_fraction_lower: float | NDArray[np.floating],
    cloud_fraction_upper: float | NDArray[np.floating],
    cloud_fraction: float | NDArray[np.floating],
    optical_depth_lower: float | NDArray[np.floating],
    optical_depth_upper: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate weighted cloud phase from upper and lower cloud layers.

    Computes the dominant cloud phase by weighting each layer's phase by its cloud fraction contribution and rounding
    to the nearest integer phase classification (1=liquid, 2=ice).

    Parameters
    ----------
    cloud_phase_lower : float or ndarray
        Cloud phase for lower layer (1=liquid, 2=ice)
    cloud_phase_upper : float or ndarray
        Cloud phase for upper layer (1=liquid, 2=ice)
    cloud_fraction_lower : float or ndarray
        Cloud fraction for lower layer (0-100)
    cloud_fraction_upper : float or ndarray
        Cloud fraction for upper layer (0-100)
    cloud_fraction : float or ndarray
        Total cloud fraction (0-100)
    optical_depth_lower : float or ndarray
        Optical depth for lower layer (used for NaN check)
    optical_depth_upper : float or ndarray
        Optical depth for upper layer (used for NaN check)

    Returns
    -------
    float or ndarray
        Cloud phase weighted by cloud fraction and rounded to nearest integer
        (1=liquid, 2=ice), or np.nan if no valid data
    """
    # Initialize result array
    result = np.zeros_like(cloud_phase_lower, dtype=np.float64)

    # Check where cloud_fraction is non-zero
    no_clouds = cloud_fraction == 0

    # For each point with clouds, calculate weighted phase
    phase_temp_1 = np.where(
        np.isnan(cloud_phase_lower) | no_clouds, 0.0, (cloud_fraction_lower * cloud_phase_lower) / cloud_fraction
    )
    phase_temp_2 = np.where(
        np.isnan(cloud_phase_upper) | no_clouds, 0.0, (cloud_fraction_upper * cloud_phase_upper) / cloud_fraction
    )
    weighted_phase = phase_temp_1 + phase_temp_2

    # Set to NaN only if BOTH optical_depth values are NaN
    both_optical_nan = np.isnan(optical_depth_lower) & np.isnan(optical_depth_upper)
    result = np.where(no_clouds | both_optical_nan, np.nan, weighted_phase)
    rounded_phase = np.round(result)

    # Final validation: cloud phase must be 1 or 2 (or NaN)
    result = np.where((rounded_phase < 0.5) | (rounded_phase > 2.5) | np.isnan(rounded_phase), np.nan, rounded_phase)

    return result


# Scene Property Column Names and Relationships


class FootprintVariables(enum.StrEnum):
    """Standardized variable names for footprint data processing.

    This class defines consistent naming conventions for all variables used in the scene identification workflow,
    including both input variables from satellite data products and calculated derived fields.

    Attributes
    ----------
    IGBP_SURFACE_TYPE : str
        IGBP land cover type code (input variable)
    SURFACE_WIND_U : str
        U-component of surface wind vector in m/s (input variable)
    SURFACE_WIND_V : str
        V-component of surface wind vector in m/s (input variable)
    CLEAR_AREA : str
        Clear sky area percentage, 0-100% (input variable)
    OPTICAL_DEPTH_LOWER : str
        Cloud optical depth for lower atmospheric layer (input variable)
    OPTICAL_DEPTH_UPPER : str
        Cloud optical depth for upper atmospheric layer (input variable)
    CLOUD_FRACTION_LOWER : str
        Cloud fraction for lower layer, 0-100% (input variable)
    CLOUD_FRACTION_UPPER : str
        Cloud fraction for upper layer, 0-100% (input variable)
    CLOUD_PHASE_LOWER : str
        Cloud phase for lower layer, 1=liquid, 2=ice (input variable)
    CLOUD_PHASE_UPPER : str
        Cloud phase for upper layer, 1=liquid, 2=ice (input variable)
    SOLAR_ZENITH_ANGLE : str
        Solar zenith angle at the surface, in degrees, 0-180 (input variable)
    VIEWING_ZENITH_ANGLE : str
        Instrument viewing zenith angle at the surface, in degrees, 0-90 (input variable)
    RELATIVE_AZIMUTH_ANGLE : str
        Relative azimuth angle (between solar and viewing directions) at the surface, in degrees, 0-360
        (input variable)
    CLOUD_FRACTION : str
        Total cloud fraction across all layers (calculated variable)
    OPTICAL_DEPTH : str
        Cloud-fraction-weighted optical depth (calculated variable)
    SURFACE_WIND : str
        Total surface wind speed magnitude in m/s (calculated variable)
    SURFACE_TYPE : str
        TRMM-compatible surface type classification (calculated variable)
    CLOUD_PHASE : str
        Cloud-fraction-weighted dominant cloud phase (calculated variable)
    """

    # Columns from input datasets. A variable the SCENE-ID products write carries its product name (Camel_Case,
    # as declared in scene_id_cam.yml); the intermediate inputs the classification consumes stay snake_case.
    IGBP_SURFACE_TYPE = "IGBP_Surface_Type"
    SURFACE_WIND_U = "surface_wind_u"
    SURFACE_WIND_V = "surface_wind_v"
    CLEAR_AREA = "clear_area"
    OPTICAL_DEPTH_LOWER = "optical_depth_lower"
    OPTICAL_DEPTH_UPPER = "optical_depth_upper"
    CLOUD_FRACTION_LOWER = "cloud_fraction_lower"
    CLOUD_FRACTION_UPPER = "cloud_fraction_upper"
    CLOUD_PHASE_LOWER = "cloud_phase_lower"
    CLOUD_PHASE_UPPER = "cloud_phase_upper"

    # Viewing-geometry angles (input variables), one value per footprint. They are required classification variables
    # for every scene definition (each scene CSV carries Solar_Zenith_Surface/Viewing_Zenith_Surface/
    # Relative_Azimuth_Surface min/max columns). The string
    # values here must match those CSV variable names exactly, otherwise SceneDefinition.required_columns will not
    # line up with the dataset variables. They are the L1B names of the same quantities.
    SOLAR_ZENITH_ANGLE = "Solar_Zenith_Surface"
    VIEWING_ZENITH_ANGLE = "Viewing_Zenith_Surface"
    RELATIVE_AZIMUTH_ANGLE = "Relative_Azimuth_Surface"

    # Calculated columns
    CLOUD_FRACTION = "Cloud_Fraction"
    OPTICAL_DEPTH = "Optical_Depth"
    SURFACE_WIND = "Surface_Wind"
    SURFACE_TYPE = "Surface_Type"
    CLOUD_PHASE = "Cloud_Phase"


@dataclass(frozen=True)
class CalculationSpec:
    """Specification for calculating a derived variable.

    Defines the parameters needed to calculate a derived variable from input data, including the calculation function,
    required inputs, and any dependencies on other calculated variables.

    Attributes
    ----------
    output_var : str
        Name of the output variable to be created
    function : Callable
        The function to call for calculation
    input_vars : list of str
        List of input variable names required by the function
    output_datatype : type
        Expected data type of the output (e.g., float, int)
    dependent_calculations : list of str or None, optional
        List of other calculated variables that must be computed first, or None if no dependencies exist.
        Default is None.

    Examples
    --------
    >>> spec = CalculationSpec(
    ...     output_var="Cloud_Fraction",
    ...     function=calculate_cloud_fraction,
    ...     input_vars=["clear_area"],
    ...     output_datatype=float
    ... )
    """

    output_var: str
    function: Callable
    input_vars: list[str]
    output_datatype: type
    dependent_calculations: list[str] | None = None


_CALCULATED_VARIABLE_MAP = {
    FootprintVariables.CLOUD_FRACTION: CalculationSpec(
        output_var=FootprintVariables.CLOUD_FRACTION,
        function=calculate_cloud_fraction,
        input_vars=[FootprintVariables.CLEAR_AREA],
        output_datatype=float,
    ),
    FootprintVariables.SURFACE_WIND: CalculationSpec(
        output_var=FootprintVariables.SURFACE_WIND,
        function=calculate_surface_wind,
        input_vars=[FootprintVariables.SURFACE_WIND_U, FootprintVariables.SURFACE_WIND_V],
        output_datatype=float,
    ),
    FootprintVariables.SURFACE_TYPE: CalculationSpec(
        output_var=FootprintVariables.SURFACE_TYPE,
        function=calculate_trmm_surface_type,
        input_vars=[FootprintVariables.IGBP_SURFACE_TYPE],
        output_datatype=float,
    ),
    FootprintVariables.OPTICAL_DEPTH: CalculationSpec(
        output_var=FootprintVariables.OPTICAL_DEPTH,
        function=calculate_cloud_fraction_weighted_optical_depth,
        input_vars=[
            FootprintVariables.OPTICAL_DEPTH_LOWER,
            FootprintVariables.OPTICAL_DEPTH_UPPER,
            FootprintVariables.CLOUD_FRACTION_LOWER,
            FootprintVariables.CLOUD_FRACTION_UPPER,
            FootprintVariables.CLOUD_FRACTION,
        ],
        output_datatype=float,
        dependent_calculations=[FootprintVariables.CLOUD_FRACTION],
    ),
    FootprintVariables.CLOUD_PHASE: CalculationSpec(
        output_var=FootprintVariables.CLOUD_PHASE,
        function=calculate_cloud_phase,
        input_vars=[
            FootprintVariables.CLOUD_PHASE_LOWER,
            FootprintVariables.CLOUD_PHASE_UPPER,
            FootprintVariables.CLOUD_FRACTION_LOWER,
            FootprintVariables.CLOUD_FRACTION_UPPER,
            FootprintVariables.CLOUD_FRACTION,
            FootprintVariables.OPTICAL_DEPTH_LOWER,
            FootprintVariables.OPTICAL_DEPTH_UPPER,
        ],
        output_datatype=float,
        dependent_calculations=[FootprintVariables.CLOUD_FRACTION],
    ),
}

# Scene Identification Data Processing


@dataclass(frozen=True)
class _FmatchColumn:
    """How one FMATCH source variable becomes one standardized :class:`FootprintVariables` classification column.

    The radiometer- and camera-timescale FMATCH products all share a flat, one-value-per-footprint layout but draw
    their classification inputs from *different* source variables (the CAM camera cloud fraction, the imager FLASHFlux
    clear coverage / CLDPIX fields, and so on). Rather than a bespoke ``_extract_data_from_fmatch`` per
    product, each reader passes a declarative ``{target FootprintVariables: _FmatchColumn}`` map and the shared
    helper applies it uniformly.

    Attributes
    ----------
    source_name : str
        Name of the source variable in the FMATCH product (its reader's ``<READER_KEY>_<spec_name>``, or an L1B
        pass-through name).
    dtype : type
        NumPy dtype the emitted column is cast to (also what the SCENE-ID product definition declares).
    transform : Callable or None, optional
        Elementwise transform applied to the source array (e.g. a code remap). Defaults to None.
    """

    source_name: str
    dtype: type
    transform: Callable | None = None


# CAM / CAM-CAMTIME classification inputs. The Libera WFOV camera cloud fraction is stored in percent [0, 100] in
# the FMATCH product (fmatch_cam.yml: units "percent", matching the CF-CAM input), which is the same scale as the
# scene-definition bins and the SCENE-ID "Cloud_Fraction" variable, so it is ingested as-is (no rescale).
# Cloud_Fraction is injected directly (there is no CERES "clear_area" on the CAM products to derive it from).
_FMATCH_CAM_COLUMN_MAP: dict[FootprintVariables, _FmatchColumn] = {
    FootprintVariables.IGBP_SURFACE_TYPE: _FmatchColumn("IGBP_Surface_Type", np.float32),
    FootprintVariables.CLOUD_FRACTION: _FmatchColumn("Cloud_Fraction_Camera", np.float32),
    FootprintVariables.SOLAR_ZENITH_ANGLE: _FmatchColumn("Solar_Zenith_Surface", np.float32),
    FootprintVariables.VIEWING_ZENITH_ANGLE: _FmatchColumn("Viewing_Zenith_Surface", np.float32),
    FootprintVariables.RELATIVE_AZIMUTH_ANGLE: _FmatchColumn("Relative_Azimuth_Surface", np.float32),
}

# FMATCH-IMAGER / FMATCH-IMAGER-CAMTIME classification inputs. Note two of these are *raw inputs* to existing
# derived-variable calculators rather than final classification columns, chosen so the imager path reuses the
# validated calculators instead of duplicating them:
#   * clear_area (from the FLASHFlux clear coverage, already in percent) -> Cloud_Fraction is derived by
#     identify_scenes as 100 - clear_area (calculate_cloud_fraction); and
#   * surface_wind_u / surface_wind_v (from the ERA5 10 m wind components) -> Surface_Wind is derived as
#     sqrt(u^2 + v^2) (calculate_surface_wind).
# Surface_Type is likewise derived from IGBP_Surface_Type via calculate_trmm_surface_type. The IGBP reader
# masks LC_Type1 to its valid 1..17 domain (17 = Water Bodies), which aligns 1:1 with the low IGBPSurfaceType
# members, so a valid modal class maps straight through. A footprint with no usable IGBP pixels (and every padded
# cell of a camera-timescale grid) carries FMATCH's declared integer _FillValue, which xarray decodes to NaN on read,
# leaving that footprint unmatched for the surface-bounded scenes. The three viewing angles are read straight
# through. The CLDPIX cloud particle phase code is remapped to the classifier's 1 = liquid / 2 = ice convention via
# map_cldpix_phase_to_trmm (see CLDPIXCloudPhase for the CER_CLDPIX code meanings).
_FMATCH_IMAGER_COLUMN_MAP: dict[FootprintVariables, _FmatchColumn] = {
    FootprintVariables.IGBP_SURFACE_TYPE: _FmatchColumn("IGBP_Surface_Type", np.float32),
    FootprintVariables.CLEAR_AREA: _FmatchColumn("FLASHFlux_Clear_Coverage", np.float32),
    FootprintVariables.SURFACE_WIND_U: _FmatchColumn("ERA5_Surface_Wind_U10", np.float32),
    FootprintVariables.SURFACE_WIND_V: _FmatchColumn("ERA5_Surface_Wind_V10", np.float32),
    FootprintVariables.SOLAR_ZENITH_ANGLE: _FmatchColumn("Solar_Zenith_Surface", np.float32),
    FootprintVariables.VIEWING_ZENITH_ANGLE: _FmatchColumn("Viewing_Zenith_Surface", np.float32),
    FootprintVariables.RELATIVE_AZIMUTH_ANGLE: _FmatchColumn("Relative_Azimuth_Surface", np.float32),
    FootprintVariables.OPTICAL_DEPTH: _FmatchColumn("CLDPIX_Cloud_Optical_Depth", np.float32),
    FootprintVariables.CLOUD_PHASE: _FmatchColumn(
        "CLDPIX_Cloud_Particle_Phase", np.float32, transform=map_cldpix_phase_to_trmm
    ),
}


def _warn_if_no_camera_cloud_fraction(footprint_data: xr.Dataset, fmatch_path: str | pathlib.Path | S3Path) -> None:
    """Log a warning when an FMATCH-CAM product's camera cloud fraction has no finite value.

    Every scene SCENE-ID-CAM classifies bounds cloud fraction, so such a product classifies no footprint: every
    scene ID is 0. FMATCH-CAM writes ``Cloud_Fraction_Camera`` as fill until it reads the radiometer-timescale CF-CAM
    product (LIBSDC-785), so the product is still written, and the warning says why it is empty.

    Parameters
    ----------
    footprint_data : xarray.Dataset
        The footprint data a CAM reader extracted, holding the ``Cloud_Fraction`` column.
    fmatch_path : str | pathlib.Path | cloudpathlib.S3Path
        The FMATCH product the data was read from, named in the warning.
    """
    if not np.isfinite(footprint_data[FootprintVariables.CLOUD_FRACTION].to_numpy()).any():
        logger.warning(
            "%s: Cloud_Fraction_Camera has no finite value, so every footprint is left unmatched (scene ID 0). "
            "FMATCH-CAM writes it as fill until the radiometer-timescale CF-CAM is read (LIBSDC-785).",
            fmatch_path,
        )


class FootprintData:
    """Container for footprint data with scene identification capabilities.

    Manages satellite footprint data through the complete scene identification workflow, including data extraction,
    preprocessing, derived field calculation, and scene classification.

    Parameters
    ----------
    data : xr.Dataset
        Input dataset containing required footprint variables

    Attributes
    ----------
    _data : xr.Dataset
        Internal dataset of footprint data. During scene identification, scene IDs
        are added as variables to this dataset.

    Notes
    -----
    This class handles the complete pipeline from a FMATCH product to scene
    identification, including:
    1. Data extraction from FMATCH NetCDF files
    2. Missing value handling
    3. Derived field calculation (cloud fraction, optical depth, etc.)
    4. Scene ID matching based on classification rules
    """

    def __init__(self, data: xr.Dataset):
        self._data = data

    @classmethod
    def from_fmatch_cam(cls, fmatch_path: str | pathlib.Path | S3Path) -> "FootprintData":
        """Read a FMATCH-CAM product into a FootprintData (radiometer timescale).

        FMATCH-CAM is the operational input to SCENE-ID-CAM: one footprint per ``RADIOMETER_TIME``, carrying the
        Libera-camera-derived cloud fraction and the scene properties needed for classification. A product whose
        camera cloud fraction has no finite value is read with a logged warning, since every footprint is then left
        unmatched.

        Parameters
        ----------
        fmatch_path : str | pathlib.Path | cloudpathlib.S3Path
            Path (local or S3) to a Libera FMATCH-CAM NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``RADIOMETER_TIME`` dimension, ready for :meth:`identify_scenes`.

        Raises
        ------
        FileNotFoundError
            If ``fmatch_path`` does not exist.
        ValueError
            If the file lacks a variable this reader reads, for example a FMATCH-IMAGER file.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            time_variable=RADIOMETER_TIME_DIMENSION,
            column_map=_FMATCH_CAM_COLUMN_MAP,
            expected_product=DataProductIdentifier.aux_fmatch_cam,
            variant="cam",
        )
        _warn_if_no_camera_cloud_fraction(extracted_data, fmatch_path)
        return cls(extracted_data)

    @classmethod
    def from_fmatch_cam_camtime(cls, fmatch_path: str | pathlib.Path | S3Path) -> "FootprintData":
        """Read a FMATCH-CAM-CAMTIME product into a FootprintData (camera timescale).

        FMATCH-CAM-CAMTIME is the operational input to SCENE-ID-CAM-CAMTIME. It is defined on a 2-D
        ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid: one ``CAMERA_TIME`` entry per 2048x2048 image and one ``PSEUDOFOOTPRINT`` entry
        per image subsection (each a contiguous, possibly overlapping block of L1B WFOV camera pixels). It carries
        both the scene properties and the footprint *identifier* variables (the inclusive pixel-block bounds
        ``Camera_Pixel_X_Min``/``Camera_Pixel_X_Max``/``Camera_Pixel_Y_Min``/``Camera_Pixel_Y_Max``, PSF bounding
        box, boresight geolocation) that the camera-timescale product passes straight through. A product whose camera
        cloud fraction has no finite value is read with a logged warning, since every footprint is then left unmatched.

        Parameters
        ----------
        fmatch_path : str | pathlib.Path | cloudpathlib.S3Path
            Path (local or S3) to a Libera FMATCH-CAM-CAMTIME NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid (one entry per image subsection), ready for
            :meth:`identify_scenes`.

        Raises
        ------
        FileNotFoundError
            If ``fmatch_path`` does not exist.
        ValueError
            If the file lacks a variable this reader reads, for example a FMATCH-IMAGER-CAMTIME file.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            time_variable=CAMERA_TIME_VARIABLE,
            column_map=_FMATCH_CAM_COLUMN_MAP,
            passthrough_variables=_FMATCH_CAMTIME_PASSTHROUGH_VARIABLES,
            expected_product=DataProductIdentifier.aux_fmatch_cam_camtime,
            variant="cam-camtime",
        )
        _warn_if_no_camera_cloud_fraction(extracted_data, fmatch_path)
        return cls(extracted_data)

    @classmethod
    def from_fmatch_imager(cls, fmatch_path: str | pathlib.Path | S3Path) -> "FootprintData":
        """Read a FMATCH-IMAGER product into a FootprintData (radiometer timescale).

        The FMATCH-IMAGER product is the operational input to SCENE-ID-IMAGER: one footprint per
        ``RADIOMETER_TIME``, carrying the FLASHFlux clear coverage (for cloud fraction), the CLDPIX cloud
        optical depth and particle phase (for the TRMM classification), the ERA5 winds (for surface wind), and
        the IGBP surface type. Unlike SCENE-ID-CAM this supports the full TRMM classification.

        Parameters
        ----------
        fmatch_path : str | pathlib.Path | cloudpathlib.S3Path
            Path (local or S3) to a Libera FMATCH-IMAGER NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``RADIOMETER_TIME`` dimension, ready for :meth:`identify_scenes`.

        Raises
        ------
        FileNotFoundError
            If ``fmatch_path`` does not exist.
        ValueError
            If the file lacks a variable this reader reads, for example a FMATCH-CAM file, which has no
            FLASHFlux or CLDPIX variables.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            time_variable=RADIOMETER_TIME_DIMENSION,
            column_map=_FMATCH_IMAGER_COLUMN_MAP,
            expected_product=DataProductIdentifier.aux_fmatch_imager,
            variant="imager",
        )
        return cls(extracted_data)

    @classmethod
    def from_fmatch_imager_camtime(cls, fmatch_path: str | pathlib.Path | S3Path) -> "FootprintData":
        """Read a FMATCH-IMAGER-CAMTIME product into a FootprintData (camera timescale).

        FMATCH-IMAGER-CAMTIME is the operational input to SCENE-ID-IMAGER-CAMTIME. It is the camera-timescale
        counterpart of FMATCH-IMAGER: it carries the *same* classification inputs -- the FLASHFlux clear coverage
        (for cloud fraction), the CLDPIX cloud optical depth and particle phase (for the full TRMM
        classification), the ERA5 winds (for surface wind), and the IGBP surface type -- but laid out on the 2-D
        ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid (one ``CAMERA_TIME`` entry per 2048x2048 image and one
        ``PSEUDOFOOTPRINT`` entry per image subsection). Like SCENE-ID-IMAGER, this supports ERBE, unfiltering, and
        the full TRMM classification.

        This reader is the union of :meth:`from_fmatch_imager` and :meth:`from_fmatch_cam_camtime`: it uses the
        IMAGER column map (so the FLASHFlux/CLDPIX inputs and the mapped cloud phase are read) on the camera-timescale
        record grid, and additionally carries the footprint *identifier* variables (the inclusive camera pixel-block
        bounds ``Camera_Pixel_{X,Y}_{Min,Max}``, the PSF bounding box, and the boresight geolocation) straight
        through so a classified scene can be traced back to the exact camera pixels and ground footprint.

        Parameters
        ----------
        fmatch_path : str | pathlib.Path | cloudpathlib.S3Path
            Path (local or S3) to a Libera FMATCH-IMAGER-CAMTIME NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid (one entry per image subsection), ready for
            :meth:`identify_scenes`.

        Raises
        ------
        FileNotFoundError
            If ``fmatch_path`` does not exist.
        ValueError
            If the file lacks a variable this reader reads, for example a FMATCH-CAM-CAMTIME file, which has no
            FLASHFlux or CLDPIX variables.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            time_variable=CAMERA_TIME_VARIABLE,
            column_map=_FMATCH_IMAGER_COLUMN_MAP,
            passthrough_variables=_FMATCH_CAMTIME_PASSTHROUGH_VARIABLES,
            expected_product=DataProductIdentifier.aux_fmatch_imager_camtime,
            variant="imager-camtime",
        )
        return cls(extracted_data)

    @staticmethod
    def _extract_data_from_fmatch(
        fmatch_path: str | pathlib.Path | S3Path,
        *,
        time_variable: str,
        column_map: dict[FootprintVariables, "_FmatchColumn"],
        expected_product: DataProductIdentifier,
        variant: str,
        passthrough_variables: tuple[str, ...] = (),
    ) -> xr.Dataset:
        """Extract the classification inputs (and any pass-through identifiers) from a FMATCH product.

        The radiometer-timescale FMATCH products (CAM, IMAGER) carry a flat one-value-per-footprint
        (1-D ``RADIOMETER_TIME``) layout; the camera-timescale product (CAM-CAMTIME) carries a 2-D
        ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid. Each classification column is emitted on its FMATCH source variable's own
        dimensions, so both layouts flow through unchanged. This helper is driven by a declarative ``column_map``
        supplied by each concrete reader:
        it reads each mapped source variable, applies its dtype/transform, and emits it under the standardized
        :class:`FootprintVariables` name the scene classifier consumes. Some emitted columns are *raw inputs* to the
        derived-variable calculators (e.g. ``clear_area`` -> ``Cloud_Fraction``, ``surface_wind_u``/``_v`` ->
        ``Surface_Wind``, ``IGBP_Surface_Type`` -> ``Surface_Type``) rather than final classification columns; those
        are derived later by :meth:`identify_scenes`.

        The time coordinate is carried as a plain ``time_variable`` data variable so it rides along through scene
        identification; the runner promotes it to a coordinate via :meth:`to_time_product` before writing.

        Parameters
        ----------
        fmatch_path : str | pathlib.Path | cloudpathlib.S3Path
            Path (local or S3) to a Libera FMATCH NetCDF product file, read through ``smart_open``.
        time_variable : str
            Name of the datetime time coordinate, read from the FMATCH file and emitted under the same name (it is
            the SCENE-ID product's declared time-coordinate name): ``RADIOMETER_TIME`` for SCENE-ID-CAM and
            SCENE-ID-IMAGER, or ``CAMERA_TIME`` for SCENE-ID-CAM-CAMTIME. Carried as a plain data variable so
            it rides through scene identification; the runner promotes it to a coordinate via
            :meth:`to_time_product` before writing.
        column_map : dict[FootprintVariables, _FmatchColumn]
            Mapping of standardized output column -> FMATCH source column spec (FMATCH variable name, dtype,
            transform).
        expected_product : DataProductIdentifier
            The FMATCH product this reader reads, named in the missing-variable error.
        variant : str
            The ``LIBERA_SCENE_ID_VARIANT`` that selects this reader, named in the missing-variable error.
        passthrough_variables : tuple of str, optional
            Names of FMATCH variables to copy verbatim onto the output under the same name (e.g. the camera-timescale
            identifier variables). Empty for the radiometer-timescale products.

        Returns
        -------
        xr.Dataset
            Footprint data on the product's record dimensions, ready for :meth:`identify_scenes`.

        Raises
        ------
        FileNotFoundError
            If ``fmatch_path`` does not exist.
        ValueError
            If the file lacks a variable the reader reads, most often a FMATCH file of another product.
        """
        logger.info("Reading FMATCH product %s", fmatch_path)
        # netCDF4 opens only a filesystem path or an in-memory buffer, so the bytes are read through smart_open (local,
        # S3 or gzip) and handed to xarray as an in-memory netCDF4 dataset, which xarray closes with the store.
        try:
            with smart_open(fmatch_path) as file_handle:
                fmatch_bytes = file_handle.read()
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Unable to open input file: {fmatch_path}") from err
        store = xr.backends.NetCDF4DataStore(nc.Dataset(pathlib.Path(str(fmatch_path)).name, memory=fmatch_bytes))
        with xr.open_dataset(store) as opened_dataset:
            fmatch_dataset = opened_dataset.load()

        missing = [
            name
            for name in (*(column.source_name for column in column_map.values()), *passthrough_variables)
            if name not in fmatch_dataset.variables
        ]
        if missing:
            raise ValueError(
                f"{fmatch_path} is missing required variable(s) {missing}; check that the manifest's input is a "
                f"{expected_product.value} file for LIBERA_SCENE_ID_VARIANT={variant}"
            )

        data_variables: dict[str, tuple[list[str], NDArray]] = {}
        for target, column in column_map.items():
            source_variable = fmatch_dataset[column.source_name]
            values = source_variable.to_numpy()
            # A transform (a code remap such as CLDPIX phase -> {liquid, ice}) is applied on the raw source values.
            if column.transform is not None:
                values = column.transform(values)
            # Emit on the source variable's own dimensions so radiometer inputs stay 1-D (RADIOMETER_TIME) while
            # camera-timescale inputs keep their 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid. Cast last so the emitted column
            # always matches the dtype the SCENE-ID product definition declares (np.where in a transform can widen to
            # float64).
            data_variables[target] = (list(source_variable.dims), np.asarray(values).astype(column.dtype, copy=False))

        # The FMATCH time coordinate is decoded to datetime64[ns] by xarray; carry it on its own dimension(s) -- the
        # 1-D RADIOMETER_TIME axis, or the 1-D CAMERA_TIME axis of the camera-timescale grid -- under the SCENE-ID
        # product's declared name (``time_variable``), so to_time_product can promote it to the product's time
        # coordinate.
        data_variables[time_variable] = (
            list(fmatch_dataset[time_variable].dims),
            fmatch_dataset[time_variable].to_numpy(),
        )

        # Copy each identifier variable verbatim, preserving its own dimensions so a multi-dimensional passthrough
        # (the 2-D Camera_Pixel_{X,Y}_{Min,Max} provenance coordinates on the (CAMERA_TIME, PSEUDOFOOTPRINT) grid) rides
        # through unchanged, not just the 1-D per-record identifiers.
        for variable_name in passthrough_variables:
            source_variable = fmatch_dataset[variable_name]
            data_variables[variable_name] = (list(source_variable.dims), source_variable.to_numpy())

        parsed_dataset = xr.Dataset(data_variables)
        logger.info("FMATCH product read successfully with sizes %s", dict(parsed_dataset.sizes))
        return parsed_dataset

    def identify_scenes(
        self,
        scene_definitions: list[SceneDefinition] | None = _USE_DEFAULT_SCENE_DEFINITIONS,  # type: ignore[assignment]
        additional_scene_definitions_files: list[pathlib.Path] | None = None,
        report_bin_bounds: bool = True,
    ):
        """Identify and assign scene IDs to all footprints based on scene definitions.

        Applies scene classification rules from one or more SceneDefinition objects to assign scene IDs to each
        footprint in the dataset.

        Parameters
        ----------
        scene_definitions : list[SceneDefinition] or None, optional
            List of SceneDefinition objects to apply. If omitted, the default standard definitions are used
            (TRMM, ERBE, and unfiltering; see :func:`default_scene_definitions`). Passing an explicit ``None`` or
            an empty list is treated as a caller error and raises ``ValueError``.
        additional_scene_definitions_files : list of pathlib.Path or None
            List of scene definition files containing classification rules for custom analysis.
        report_bin_bounds : bool, optional
            If True (default), also report, for each classification variable, the
            (min, max) bounds of the property bin that each footprint's matched
            scene occupies, as ``Scene_Bin_{type}_{variable}_Min`` / ``_Max``
            variables. Set to False to emit only the scene ID columns.

        Notes
        -----
        This method modifies self._data in place by adding scene IDs for each row of footprint data.

        For each SceneDefinition provided:
        1. Validates that all required variables exist in the footprint data
        2. Matches each footprint to a scene based on variable ranges
        3. Adds a new variable to the dataset with the scene IDs

        Footprints that don't match any scene are assigned a scene ID of 0.

        Examples
        --------
        >>> footprint_data = FootprintData(dataset)
        >>> footprint_data.identify_scenes()
        """
        # Resolve the default set only when the caller omitted the argument entirely (the sentinel). An explicit
        # None is still treated as a caller error, preserving the previous contract.
        if scene_definitions is _USE_DEFAULT_SCENE_DEFINITIONS:
            scene_definitions = default_scene_definitions()
        if scene_definitions is None:
            raise ValueError("No scene definitions provided.")
        if len(scene_definitions) == 0:
            raise ValueError("Scene definitions list is empty.")

        # Calculate required fields for each scene
        required_calculated_fields = list()
        if additional_scene_definitions_files:
            # Copy first so appending the caller-supplied extra definitions does not mutate the list the caller
            # passed in (which may be reused across calls, e.g. the shared standard_scene_definitions() result).
            scene_definitions = list(scene_definitions)
            for additional_scene_definition in additional_scene_definitions_files:
                scene_definitions.append(SceneDefinition(additional_scene_definition))
        for scene_definition in scene_definitions:
            required_calculated_fields += scene_definition.required_columns

        self._calculate_required_fields(required_calculated_fields)
        for scene_definition in scene_definitions:
            logger.info(f"Identifying {scene_definition.type} scenes...")
            self._data = scene_definition.identify_and_update(self._data, report_bin_bounds=report_bin_bounds)
            logger.info(f"Added Scene_ID_{scene_definition.type} to dataset")

    def _calculate_required_fields(self, result_fields: list[str]):
        """Calculate necessary derived fields on data from input FootprintVariables.

        Computes derived atmospheric variables needed for scene identification, handling dependencies between
        calculated fields automatically.

        Parameters
        ----------
        result_fields : list of str
            List of field names to calculate (e.g., 'Cloud_Fraction', 'Optical_Depth')

        Raises
        ------
        ValueError
            If an unknown field is requested or if circular dependencies exist

        Notes
        -----
        This method modifies self._data in place to conserve memory. It automatically
        resolves dependencies between calculated fields (e.g., optical depth depends
        on cloud fraction being calculated first).

        The calculation order is determined by dependency analysis and may require
        multiple passes. A maximum of 30 iterations is allowed to prevent infinite
        loops from circular dependencies.

        Available calculated fields are defined in _CALCULATED_VARIABLE_MAP.
        """
        # We could copy _data here, but instead we are modifying in place to save memory

        # Track calculated fields to handle dependencies
        calculated = set(self._data.variables)

        # Keep calculating until all requested fields are done
        remaining = set(result_fields) - calculated

        loop_check = 0
        while remaining:
            field_calculated = False

            for field in list(remaining):
                if field not in _CALCULATED_VARIABLE_MAP:
                    raise ValueError(f"Unknown calculated field: {field}")

                calc_spec = _CALCULATED_VARIABLE_MAP[field]
                if calc_spec.dependent_calculations:
                    for dependency in calc_spec.dependent_calculations:
                        if dependency not in calculated:
                            # Dependency needed to be calculated first
                            dependency_spec = _CALCULATED_VARIABLE_MAP[dependency]
                            self._calculate_single_field_from_spec(dependency_spec, calculated)
                            calculated.add(dependency)
                            if dependency in remaining:
                                remaining.remove(dependency)

                # Now calculate the target field
                self._calculate_single_field_from_spec(calc_spec, calculated)
                calculated.add(field)
                if field in remaining:
                    remaining.remove(field)
                field_calculated = True
            loop_check += 1
            if not field_calculated and remaining:
                raise ValueError(f"Cannot calculate fields {remaining} - missing dependencies")
            if loop_check > 30:
                raise ValueError(f"Cannot calculate fields {remaining} - dependencies not found")

    def _calculate_single_field_from_spec(self, spec: CalculationSpec, calculated: set[str]):
        """Calculate a single field from input FootprintVariables.

        Applies the calculation function specified in the CalculationSpec to the input variables, creating a new
        variable in the dataset.

        Parameters
        ----------
        spec : CalculationSpec
            Specification defining the calculation to perform
        calculated : set of str
            Set of variable names already available in the dataset

        Raises
        ------
        ValueError
            If required input variables are not available in the dataset

        """
        if all(var in calculated for var in spec.input_vars):
            inputs = [self._data[var] for var in spec.input_vars]

            result = xr.apply_ufunc(
                spec.function,
                *inputs,
                output_dtypes=[spec.output_datatype],
                keep_attrs=True,
            )
            self._data[spec.output_var] = result
        else:
            raise ValueError(f"Cannot calculate fields - missing dependencies {spec.input_vars}")

    def to_time_product(self, time_variable: str = RADIOMETER_TIME_DIMENSION) -> xr.Dataset:
        """Return the footprint data ready to write on its record grid.

        The radiometer-timescale scene-ID products (CAM/IMAGER) contain exactly one footprint per observation
        time and are written on the same 1-D time dimension as their upstream product, so downstream consumers can
        align scene IDs to the upstream records positionally. The camera-timescale product (CAM-CAMTIME) is instead
        written on a 2-D ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid: one entry per image subsection. :class:`FootprintData`
        already carries data on the correct dimensions, so this method only promotes the time variable to a
        coordinate so the result is ready to hand to :func:`libera_utils.io.netcdf.write_libera_data_product` with
        the matching ``time_variable``.

        Parameters
        ----------
        time_variable : str, optional
            Name of the time variable to promote to a coordinate. Defaults to ``RADIOMETER_TIME`` (the
            radiometer-timescale products); the camera-timescale product passes ``CAMERA_TIME``.

        Returns
        -------
        xr.Dataset
            A copy of the internal dataset with the time variable promoted to a coordinate and a placeholder
            ``Quality_Flag`` added over the record grid.

        Raises
        ------
        ValueError
            If ``time_variable`` is not present (e.g. the instance was built directly from a synthetic dataset
            that did not include observation times).
        """
        if time_variable not in self._data.variables:
            raise ValueError(
                f"Cannot build a time product: '{time_variable}' is not present in the footprint data. It is "
                "populated by the FootprintData reader (e.g. from_fmatch_cam()); datasets constructed directly "
                "must add it themselves."
            )
        # Work on a copy so callers that inspect FootprintData._data afterwards still see the internal
        # representation (set_coords otherwise mutates the shared dataset).
        product = self._data.set_coords(time_variable)
        # The time coordinate is 1-D on its own axis: RADIOMETER_TIME for the radiometer-timescale products (where
        # the time variable is a dimension coordinate), or CAMERA_TIME for the camera-timescale product.
        (time_dimension,) = product[time_variable].dims
        # The record grid is the set of dimensions the per-record data variables span: (RADIOMETER_TIME,) for the
        # radiometer-timescale products, or (CAMERA_TIME, PSEUDOFOOTPRINT) for the camera-timescale product. Derive it
        # from the widest data variable lying on the time dimension (rather than from the 1-D time coordinate alone)
        # so the placeholder Quality_Flag covers the full grid.
        record_variables = [product[name] for name in product.data_vars if time_dimension in product[name].dims]
        grid_dimensions = (
            max(record_variables, key=lambda variable: variable.ndim).dims if record_variables else (time_dimension,)
        )
        # TODO[LIBSDC-810]: Add real quality flag to the product
        product = add_placeholder_quality_flag(product, dimensions=grid_dimensions)
        return product

    def to_radiometer_time_product(self) -> xr.Dataset:
        """Return the footprint data ready to write on the Libera ``RADIOMETER_TIME`` axis.

        Backward-compatible wrapper around :meth:`to_time_product` pinned to the ``RADIOMETER_TIME`` variable.

        Returns
        -------
        xr.Dataset
            A copy of the internal dataset with the ``RADIOMETER_TIME`` variable promoted to a coordinate.
        """
        return self.to_time_product(RADIOMETER_TIME_DIMENSION)
