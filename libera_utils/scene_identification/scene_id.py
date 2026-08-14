"""Module for mapping radiometer footprints to scene IDs.

This module provides functionality for identifying and classifying atmospheric scenes
based on footprint data from satellite observations.

"""

import enum
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import netCDF4 as nc
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from libera_utils.config import config
from libera_utils.scene_identification.scene_definitions import SceneDefinition

logger = logging.getLogger(__name__)

# Names used throughout footprint data processing and when emitting a Libera SCENE-ID product. FootprintData
# carries data on the "RADIOMETER_TIME" dimension from creation onward: the radiometer-timescale scene-ID
# products (CAM, IMAGER, FLASH) are written on that dimension with a "RADIOMETER_TIME" coordinate to exactly
# mirror the upstream L1B_RAD product, so scene IDs align 1:1 with L1B footprints. "RADIOMETER_TIME" names both
# the per-footprint dimension and the datetime coordinate written on it (a NetCDF / xarray dimension coordinate),
# so one constant is used throughout scene-ID processing. This constant is shared with the product runner (see
# libera_utils/scene_identification/cam/scene_id_cam.py) and the product-definition YAML.
RADIOMETER_TIME_DIMENSION = "RADIOMETER_TIME"

# The camera-timescale counterparts of the constants above. The camera-timescale scene-ID product (CAM-CAMTIME) is
# written on the FOOTPRINT record axis (one record per image subsection / pseudo-footprint), with CAMERA_TIME carried
# as a NON-UNIQUE coordinate on FOOTPRINT (one 2048x2048 image is segmented into many, possibly overlapping,
# subsections that all share its time), mirroring its FMATCH-CAM-CAMTIME input. FOOTPRINT names the record dimension;
# CAMERA_TIME names the datetime coordinate riding on it (its name != its dimension, unlike RADIOMETER_TIME).
FOOTPRINT_DIMENSION = "FOOTPRINT"
CAMERA_TIME_VARIABLE = "CAMERA_TIME"

# Identifier variables that the camera-timescale FMATCH product carries and the SCENE-ID-CAM-CAMTIME product passes
# straight through -- the camera pixel-index ranges (camera_pixel_x/y, 2-D inclusive (min, max) pairs on the
# CAMERA_PIXEL_BOUNDS axis), the PSF bounding box, and the boresight geolocation -- so a scene can be traced back to
# the exact camera pixels and ground footprint. These are copied verbatim (each keeping its own dimensions) by
# from_fmatch_cam_camtime and are not consumed by the classification; they simply ride along to the written product.
# The FMATCH-only center_pixel_x/y (boresight pixel) is deliberately NOT listed: SCENE-ID does not carry it.
_FMATCH_CAM_CAMTIME_PASSTHROUGH_VARIABLES: tuple[str, ...] = (
    "latitude",
    "longitude",
    "altitude",
    "psf_bbox_lat_min",
    "psf_bbox_lat_max",
    "psf_bbox_lon_min",
    "psf_bbox_lon_max",
    "camera_pixel_x",
    "camera_pixel_y",
)

# Name of the per-footprint data-quality bit-flag variable declared in the SCENE-ID product definitions
# (scene_id_cam.yml).
QUALITY_FLAG_VARIABLE = "Quality_Flag"

# Standard scene definitions shipped with libera_utils, keyed by scene "type". The type is the lowercased stem of
# the CSV filename and also becomes the output column name (scene_id_{type}). The value is the config.json key that
# resolves to the CSV path. Consumers can select a subset by type (e.g. the SCENE-ID-CAM product runs "erbe" and
# "unfiltering" but not "trmm").
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
        Scene types to include, drawn from the keys of :data:`STANDARD_SCENE_DEFINITION_CONFIG_KEYS`
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


def add_placeholder_quality_flag(product: xr.Dataset, dimension: str = RADIOMETER_TIME_DIMENSION) -> xr.Dataset:
    """Add the SCENE-ID ``Quality_Flag`` variable as an all-zero placeholder.

    The SCENE-ID product definitions (e.g. ``scene_id_cam.yml``) declare a per-footprint ``Quality_Flag`` bit-flag
    variable. Per-footprint quality flagging is not implemented yet, so this fills every footprint with ``0``
    ("no flags set") so the product still conforms to its definition. The variable is added on the product's
    per-footprint time dimension, so the input must already be on that axis.

    Parameters
    ----------
    product : xr.Dataset
        A per-footprint product dataset to add the flag to.
    dimension : str, optional
        Name of the per-footprint dimension to add the flag along. Defaults to ``RADIOMETER_TIME`` (the
        radiometer-timescale products); the camera-timescale product passes ``FOOTPRINT`` (the axis its non-unique
        ``CAMERA_TIME`` coordinate lives on).

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
    number_of_footprints = product.sizes[dimension]
    quality_flag = np.zeros(number_of_footprints, dtype=np.uint32)
    product[QUALITY_FLAG_VARIABLE] = (dimension, quality_flag)
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

    Attributes
    ----------
    IGBP_1 through IGBP_20 : int
        TRMM surface type categories (values: 1-20)

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

    TODO[LIBSDC-001]: PLACEHOLDER MAPPING -- confirm against the CLDPIX data dictionary.
    The FMATCH ``cldpix_cloud_particle_phase`` variable is declared with ``valid_range [1, 5]``, but the
    meaning of each of the five codes is not yet confirmed. Only the two codes the scene classifier can use are
    named here (liquid, ice); until the encoding is confirmed, every other code is treated as "no usable phase"
    (mapped to NaN) so those footprints are simply left unmatched for phase-gated scenes rather than being
    misclassified.

    Attributes
    ----------
    LIQUID : int
        Liquid-water cloud (mapped to the classifier's cloud_phase value 1).
    ICE : int
        Ice cloud (mapped to the classifier's cloud_phase value 2).
    """

    # TODO[LIBSDC-817]: verify these code values (and the meaning of codes 3-5) against the CLDPIX data dictionary.
    LIQUID = 1
    ICE = 2


def map_cldpix_phase_to_trmm(cldpix_phase: NDArray) -> NDArray[np.floating]:
    """Map CLDPIX cloud-particle-phase codes to the classifier's ``cloud_phase`` (1 = liquid, 2 = ice).

    The TRMM scene classifier bins ``cloud_phase`` as 1 (liquid) or 2 (ice) -- see the ``cloud_phase_min/max``
    columns of ``trmm.csv`` and :func:`calculate_cloud_phase`, which likewise emits 1 or 2. The CLDPIX source
    variable (``cldpix_cloud_particle_phase``) instead uses its own integer code scheme (valid_range
    ``[1, 5]``). This is a *code remap*, deliberately unlike the continuous cloud-fraction-weighted
    :func:`calculate_cloud_phase`: each CLDPIX code is looked up and collapsed onto {1, 2}.

    Any code that is not a recognized liquid/ice value (including fill/NaN, or the not-yet-confirmed CLDPIX codes
    3-5) is mapped to ``NaN``. A ``NaN`` classification value leaves the footprint unmatched (scene ID 0) for any
    scene that bounds ``cloud_phase`` -- exactly the desired behavior when the phase is unknown.

    Parameters
    ----------
    cldpix_phase : ndarray
        CLDPIX cloud particle phase codes (any numeric dtype; NaN allowed for missing).

    Returns
    -------
    ndarray of float
        ``cloud_phase`` values (1.0 = liquid, 2.0 = ice, NaN = unknown), as ``float32`` so NaN is representable.

    Notes
    -----
    TODO[LIBSDC-001]: replace this placeholder mapping once the CLDPIX phase encoding is confirmed. See
    :class:`CLDPIXCloudPhase`.
    """
    # Compare against the (placeholder) recognized codes and build a float result so unknown codes can be NaN.
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


def calculate_trmm_surface_type(igbp_surface_type: int | NDArray[np.integer]) -> int | NDArray[np.integer]:
    """Convert an IGBP surface type code to its TRMM surface type classification.

    This maps IGBP -> TRMM (the direction used by scene identification): each IGBP land-cover code (1-20) is
    collapsed onto one of the six coarser TRMM surface types via :meth:`IGBPSurfaceType.trmm_surface_type`.

    Parameters
    ----------
    igbp_surface_type : int or ndarray of int
        IGBP surface type codes (valid range 1-20)

    Returns
    -------
    int or ndarray of int
        TRMM surface type codes (0=ocean ... 5=snow)

    Raises
    ------
    ValueError
        If any input value is not a valid IGBP surface type (outside 1-20)

    Notes
    -----
    The conversion uses a lookup table derived from the IGBP->TRMM map on :class:`IGBPSurfaceType`.
    Values that don't correspond to a valid IGBP surface type will raise a ValueError.

    Examples
    --------
    >>> calculate_trmm_surface_type(1)  # IGBP EVERGREEN_NEEDLELEAF_FOREST -> TRMM HI_SHRUB (1)
    1
    >>> calculate_trmm_surface_type(np.array([1, 17]))  # 17 = WATER_BODIES -> OCEAN (0)
    array([1, 0])
    >>> calculate_trmm_surface_type(999)
    ValueError: Cannot convert IGBP surface type value to TRMM surface type: [999]
    """
    all_surfaces = list()
    for igbp_surface_enum in IGBPSurfaceType:
        all_surfaces.append(igbp_surface_enum)
    max_igbp = max(surface.value for surface in all_surfaces) if all_surfaces else 0
    lookup = np.full(max_igbp + 1, -1, dtype=int)

    for surface_type in all_surfaces:
        lookup[surface_type.value] = surface_type.trmm_surface_type

    # Vectorized lookup with bounds checking
    result = np.where(
        (igbp_surface_type > 0) & (igbp_surface_type <= max_igbp),
        lookup[np.clip(igbp_surface_type, 0, max_igbp)],
        -1,
    )

    # Check for failed conversions and raise ValueError
    failed_mask = result == -1
    if np.any(failed_mask):
        # Extract the specific failed values
        if np.isscalar(igbp_surface_type):
            failed_values = [igbp_surface_type]
        else:
            failed_values = igbp_surface_type[failed_mask].tolist()
        raise ValueError(f"Cannot convert IGBP surface type value to TRMM surface type: {failed_values}")

    return result


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

    # Columns from input datasets
    IGBP_SURFACE_TYPE = "igbp_surface_type"
    SURFACE_WIND_U = "surface_wind_u"
    SURFACE_WIND_V = "surface_wind_v"
    CLEAR_AREA = "clear_area"
    OPTICAL_DEPTH_LOWER = "optical_depth_lower"
    OPTICAL_DEPTH_UPPER = "optical_depth_upper"
    CLOUD_FRACTION_LOWER = "cloud_fraction_lower"
    CLOUD_FRACTION_UPPER = "cloud_fraction_upper"
    CLOUD_PHASE_LOWER = "cloud_phase_lower"
    CLOUD_PHASE_UPPER = "cloud_phase_upper"

    # Viewing-geometry angles (input variables). These are measured per footprint and read directly from the CERES
    # SSF "Viewing_Angles" group. They are required classification variables for every scene definition (each scene
    # CSV carries solar_zenith_angle/viewing_zenith_angle/relative_azimuth_angle min/max columns). The string values
    # here must match those CSV variable names exactly, otherwise SceneDefinition.required_columns will not line up
    # with the dataset variables. Note the SSF file names the viewing-zenith variable "view_zenith_angle"; we rename
    # it to "viewing_zenith_angle" on extraction so the pipeline uses one consistent name.
    SOLAR_ZENITH_ANGLE = "solar_zenith_angle"
    VIEWING_ZENITH_ANGLE = "viewing_zenith_angle"
    RELATIVE_AZIMUTH_ANGLE = "relative_azimuth_angle"

    # Calculated columns
    CLOUD_FRACTION = "cloud_fraction"
    OPTICAL_DEPTH = "optical_depth"
    SURFACE_WIND = "surface_wind"
    SURFACE_TYPE = "surface_type"
    CLOUD_PHASE = "cloud_phase"


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
    ...     output_var="cloud_fraction",
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
        output_datatype=np.uint8,
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
    their classification inputs from *different* source variables (the CAM camera cloud fraction, the imager CERES
    clear coverage / RBSP CLDPIX fields, and so on). Rather than a bespoke ``_extract_data_from_fmatch`` per
    product, each reader passes a declarative ``{target FootprintVariables: _FmatchColumn}`` map and the shared
    helper applies it uniformly.

    Attributes
    ----------
    source_name : str
        Variable name to read from the FMATCH file.
    dtype : type
        NumPy dtype the emitted column is cast to (also what the SCENE-ID product definition declares).
    scale : float, optional
        Multiplicative rescale applied to the source values. Defaults to 1.0 (no rescale). Available for source
        variables whose stored units differ from the classifier/product scale.
    transform : Callable or None, optional
        Elementwise transform applied to the source array (e.g. a code remap). Applied instead of ``scale`` when
        set. Defaults to None.
    """

    source_name: str
    dtype: type
    scale: float = 1.0
    transform: Callable | None = None


# CAM / CAM-CAMTIME classification inputs. The Libera WFOV camera cloud fraction is stored in percent [0, 100] in
# the FMATCH product (fmatch_cam.yml: units "percent", matching the CF-CAM input), which is the same scale as the
# scene-definition bins and the SCENE-ID "cloud_fraction" variable, so it is ingested as-is (no rescale).
# cloud_fraction is injected directly (there is no CERES "clear_area" on the CAM products to derive it from).
_FMATCH_CAM_COLUMN_MAP: dict[FootprintVariables, _FmatchColumn] = {
    FootprintVariables.IGBP_SURFACE_TYPE: _FmatchColumn("igbp_surface_type", np.uint8),
    FootprintVariables.CLOUD_FRACTION: _FmatchColumn("cloud_fraction_camera", np.float32),
    FootprintVariables.SOLAR_ZENITH_ANGLE: _FmatchColumn("solar_zenith_angle", np.float32),
    FootprintVariables.VIEWING_ZENITH_ANGLE: _FmatchColumn("viewing_zenith_angle", np.float32),
    FootprintVariables.RELATIVE_AZIMUTH_ANGLE: _FmatchColumn("relative_azimuth_angle", np.float32),
}

# Classification inputs common to both imager scene-ID products (FLASH and IMAGER). Note two of these
# are *raw inputs* to existing derived-variable calculators rather than final classification columns, chosen so the
# imager path reuses the validated calculators instead of duplicating them:
#   * clear_area (from the CERES SSF clear coverage, already in percent) -> cloud_fraction is derived by
#     identify_scenes as 100 - clear_area (calculate_cloud_fraction); and
#   * surface_wind_u / surface_wind_v (from the ERA5 10 m wind components) -> surface_wind is derived as
#     sqrt(u^2 + v^2) (calculate_surface_wind).
# surface_type is likewise derived from igbp_surface_type. The three viewing angles are read straight through.
_FMATCH_IMAGER_COMMON_COLUMNS: dict[FootprintVariables, _FmatchColumn] = {
    FootprintVariables.IGBP_SURFACE_TYPE: _FmatchColumn("igbp_surface_type", np.uint8),
    FootprintVariables.CLEAR_AREA: _FmatchColumn("ssf_clear_coverage", np.float32),
    FootprintVariables.SURFACE_WIND_U: _FmatchColumn("era5_wind_u10", np.float32),
    FootprintVariables.SURFACE_WIND_V: _FmatchColumn("era5_wind_v10", np.float32),
    FootprintVariables.SOLAR_ZENITH_ANGLE: _FmatchColumn("solar_zenith_angle", np.float32),
    FootprintVariables.VIEWING_ZENITH_ANGLE: _FmatchColumn("viewing_zenith_angle", np.float32),
    FootprintVariables.RELATIVE_AZIMUTH_ANGLE: _FmatchColumn("relative_azimuth_angle", np.float32),
}

# FMATCH-IMAGER-FLASH: optical_depth is the single already-aggregated CERES SSF cloud optical depth (injected
# directly -- the two-layer calculate_cloud_fraction_weighted_optical_depth is bypassed because FMATCH carries one
# value per footprint, not lower/upper layers). FLASH carries NO cloud-phase source; the reader injects cloud_phase
# as all-NaN (see nan_columns in from_fmatch_imager_flash) so the phase-gated TRMM scenes fall through to unmatched.
_FMATCH_IMAGER_FLASH_COLUMN_MAP: dict[FootprintVariables, _FmatchColumn] = {
    **_FMATCH_IMAGER_COMMON_COLUMNS,
    FootprintVariables.OPTICAL_DEPTH: _FmatchColumn("ssf_cloud_optical_depth", np.float32),
}

# FMATCH-IMAGER (RBSP): prefer the native RBSP CLDPIX fields where both CLDPIX and SSF exist. The
# CLDPIX cloud particle phase code is remapped to the classifier's 1 = liquid / 2 = ice convention via
# map_cldpix_phase_to_trmm (see its TODO[LIBSDC-817] -- the code meanings are a placeholder pending the CLDPIX data dictionary).
_FMATCH_IMAGER_COLUMN_MAP: dict[FootprintVariables, _FmatchColumn] = {
    **_FMATCH_IMAGER_COMMON_COLUMNS,
    FootprintVariables.OPTICAL_DEPTH: _FmatchColumn("cldpix_cloud_optical_depth", np.float32),
    FootprintVariables.CLOUD_PHASE: _FmatchColumn(
        "cldpix_cloud_particle_phase", np.float32, transform=map_cldpix_phase_to_trmm
    ),
}


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

    Methods
    -------
    process_ssf_and_camera(ssf_path, scene_definitions)
        Process SSF and camera data to identify scenes
    process_cldpx_viirs_geos_cam_groundscene()
        Process alternative data format (not implemented)
    process_clouds_groundscene()
        Process cloud/ground scene data (not implemented)

    Notes
    -----
    This class handles the complete pipeline from raw satellite data to scene
    identification, including:
    1. Data extraction from NetCDF files
    2. Missing value handling
    3. Derived field calculation (cloud fraction, optical depth, etc.)
    4. Scene ID matching based on classification rules
    """

    def __init__(self, data: xr.Dataset):
        self._data = data

    @classmethod
    def from_ceres_ssf(cls, ssf_path: pathlib.Path):
        """Process SSF (Single Scanner Footprint) and camera data to identify scenes.

        Reads CERES SSF data, extracts relevant variables, calculates derived fields, and identifies scene
        classifications for each footprint.

        Parameters
        ----------
        ssf_path : pathlib.Path
            Path to the SSF NetCDF file (CeresSSFNOAA20FM6Ed1C format)

        Returns
        -------
        FootprintData
            Processed footprint data object containing the extracted footprint variables. Scene IDs are added
            later by :meth:`identify_scenes`.

        Raises
        ------
        FileNotFoundError
            If the SSF file cannot be found or opened

        Notes
        -----
        Processing steps:
        1. Extract variables from SSF NetCDF groups
        2. Apply maximum value thresholds to cloud properties
        3. Calculate derived fields (cloud fraction, optical depth, wind speed, etc.)
        4. Match footprints to scene IDs using provided scene definitions

        Maximum value thresholds applied:
        - Cloud fraction: 100%
        - Cloud phase: 2 (ice)
        - Optical depth: 500

        Examples
        --------
        >>> footprint_data = FootprintData.from_ceres_ssf(Path("CERES_SSF_NOAA20_2024001.nc"))
        >>> footprint_data.identify_scenes()
        """
        try:
            with nc.Dataset(ssf_path) as file:
                extracted_data = cls._extract_data_from_CeresSSFNOAA20FM6Ed1C(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to parse input file: {ssf_path}")
        footprint_data = cls(extracted_data)
        # Format extracted data
        max_cloud_fraction = 100.0
        max_cloud_phase = 2.0
        max_optical_depth = 500.0

        columns_with_max_value = [
            (FootprintVariables.CLOUD_FRACTION_LOWER, max_cloud_fraction),
            (FootprintVariables.CLOUD_FRACTION_UPPER, max_cloud_fraction),
            (FootprintVariables.CLOUD_PHASE_LOWER, max_cloud_phase),
            (FootprintVariables.CLOUD_PHASE_UPPER, max_cloud_phase),
            (FootprintVariables.OPTICAL_DEPTH_LOWER, max_optical_depth),
            (FootprintVariables.OPTICAL_DEPTH_UPPER, max_optical_depth),
        ]
        for column_name, threshold in columns_with_max_value:
            footprint_data._fill_column_above_max_value(column_name, threshold)

        return footprint_data

    @classmethod
    def from_fmatch_cam(cls, fmatch_path: pathlib.Path) -> "FootprintData":
        """Read a FMATCH-CAM product into a FootprintData (radiometer timescale).

        FMATCH-CAM is the operational input to SCENE-ID-CAM: one footprint per ``RADIOMETER_TIME``, carrying the
        Libera-camera-derived cloud fraction and the scene properties needed for classification.

        Parameters
        ----------
        fmatch_path : pathlib.Path
            Path to a Libera FMATCH-CAM NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``RADIOMETER_TIME`` dimension, ready for :meth:`identify_scenes`.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            record_dimension=RADIOMETER_TIME_DIMENSION,
            time_variable=RADIOMETER_TIME_DIMENSION,
            column_map=_FMATCH_CAM_COLUMN_MAP,
            context="SCENE-ID-CAM reader (FMATCH-CAM)",
        )
        return cls(extracted_data)

    @classmethod
    def from_fmatch_cam_camtime(cls, fmatch_path: pathlib.Path) -> "FootprintData":
        """Read a FMATCH-CAM-CAMTIME product into a FootprintData (camera timescale).

        FMATCH-CAM-CAMTIME is the operational input to SCENE-ID-CAM-CAMTIME: many pseudo-footprints per
        ``CAMERA_TIME`` (each a contiguous, possibly overlapping block of L1B WFOV camera pixels), carrying both the
        scene properties and the footprint *identifier* variables (camera pixel-index ranges
        ``camera_pixel_x``/``camera_pixel_y`` as inclusive (min, max) pairs, PSF bounding box, boresight geolocation)
        that the camera-timescale product passes straight through. Records live on the ``FOOTPRINT`` axis, with
        ``CAMERA_TIME`` carried as a non-unique coordinate on it.

        Parameters
        ----------
        fmatch_path : pathlib.Path
            Path to a Libera FMATCH-CAM-CAMTIME NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``FOOTPRINT`` dimension (one record per image subsection), with ``CAMERA_TIME`` and
            the ``camera_pixel_x``/``camera_pixel_y`` range coordinates carried through, ready for
            :meth:`identify_scenes`.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            record_dimension=FOOTPRINT_DIMENSION,
            time_variable=CAMERA_TIME_VARIABLE,
            column_map=_FMATCH_CAM_COLUMN_MAP,
            passthrough_variables=_FMATCH_CAM_CAMTIME_PASSTHROUGH_VARIABLES,
            context="SCENE-ID-CAM-CAMTIME reader (FMATCH-CAM-CAMTIME)",
        )
        return cls(extracted_data)

    @classmethod
    def from_fmatch_imager_flash(cls, fmatch_path: pathlib.Path) -> "FootprintData":
        """Read a FMATCH-IMAGER-FLASH product into a FootprintData (radiometer timescale).

        FMATCH-IMAGER-FLASH is the operational input to SCENE-ID-IMAGER-FLASH: one footprint per
        ``RADIOMETER_TIME``. Its cloud fraction is derived from the CERES SSF clear coverage and its optical depth
        from the CERES SSF cloud optical depth; it carries ERA5 winds (for surface wind) and the IGBP surface type.

        It carries **no cloud-phase source**, so ``cloud_phase`` is injected as all-NaN. TRMM is still run (per the
        product design), which classifies the clear/surface TRMM scenes that leave ``cloud_phase`` unbounded while
        leaving every phase-gated cloudy TRMM scene unmatched (scene ID 0).

        Parameters
        ----------
        fmatch_path : pathlib.Path
            Path to a Libera FMATCH-IMAGER-FLASH NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``RADIOMETER_TIME`` dimension, ready for :meth:`identify_scenes`.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            record_dimension=RADIOMETER_TIME_DIMENSION,
            time_variable=RADIOMETER_TIME_DIMENSION,
            column_map=_FMATCH_IMAGER_FLASH_COLUMN_MAP,
            # FMATCH-IMAGER-FLASH has no cloud-phase field; inject an all-NaN cloud_phase so identify_scenes does
            # not try (and fail) to derive it, and so phase-gated TRMM scenes are left unmatched.
            nan_columns=(FootprintVariables.CLOUD_PHASE,),
            context="SCENE-ID-IMAGER-FLASH reader (FMATCH-IMAGER-FLASH)",
        )
        return cls(extracted_data)

    @classmethod
    def from_fmatch_imager(cls, fmatch_path: pathlib.Path) -> "FootprintData":
        """Read a FMATCH-IMAGER product into a FootprintData (radiometer timescale).

        The FMATCH-IMAGER product is the operational input to SCENE-ID-IMAGER: one footprint per
        ``RADIOMETER_TIME``, carrying the CERES SSF clear coverage (for cloud fraction), the RBSP CLDPIX cloud
        optical depth and particle phase (for the TRMM classification), the ERA5 winds (for surface wind), and
        the IGBP surface type. Unlike SCENE-ID-CAM this supports the full TRMM classification.

        The reader validates the RBSP CLDPIX/SSF variables up front and raises a clear error if handed a file that
        lacks them (e.g. a FMATCH-IMAGER-FLASH product); see :meth:`_require_variables`.

        Parameters
        ----------
        fmatch_path : pathlib.Path
            Path to a Libera FMATCH-IMAGER NetCDF product file.

        Returns
        -------
        FootprintData
            Footprint data on the ``RADIOMETER_TIME`` dimension, ready for :meth:`identify_scenes`.
        """
        extracted_data = cls._extract_data_from_fmatch(
            fmatch_path,
            record_dimension=RADIOMETER_TIME_DIMENSION,
            time_variable=RADIOMETER_TIME_DIMENSION,
            column_map=_FMATCH_IMAGER_COLUMN_MAP,
            context=(
                "SCENE-ID-IMAGER reader (FMATCH-IMAGER); the FMATCH file lacks the RBSP ssf/cldpix variables "
                "required for scene identification"
            ),
        )
        return cls(extracted_data)

    @staticmethod
    def _require_variables(dataset: xr.Dataset, required_names: tuple[str, ...], *, context: str) -> None:
        """Raise a single, actionable error if any expected FMATCH source variable is absent.

        This turns what would otherwise be an opaque ``KeyError`` deep in extraction into one message naming every
        missing variable and the reader that expected it. Its most important use is catching a FMATCH file routed
        to the wrong reader -- e.g. a FMATCH-IMAGER-FLASH file handed to the IMAGER reader, which lacks the RBSP
        ``ssf``/``cldpix`` variables -- so this reports that clearly instead of failing on the first missing lookup.

        Parameters
        ----------
        dataset : xr.Dataset
            The opened FMATCH dataset.
        required_names : tuple of str
            Source variable names the reader's column map requires.
        context : str
            Human-readable description of the reader, included in the error message.

        Raises
        ------
        ValueError
            If any required variable is missing from ``dataset``.
        """
        missing = [name for name in required_names if name not in dataset.variables]
        if missing:
            raise ValueError(
                f"{context}: FMATCH file is missing required variable(s) {missing}. "
                f"Present data variables: {sorted(dataset.data_vars)}"
            )

    @staticmethod
    def _extract_data_from_fmatch(
        fmatch_path: pathlib.Path,
        *,
        record_dimension: str,
        time_variable: str,
        column_map: dict[FootprintVariables, "_FmatchColumn"],
        nan_columns: tuple[FootprintVariables, ...] = (),
        passthrough_variables: tuple[str, ...] = (),
        context: str = "FMATCH reader",
    ) -> xr.Dataset:
        """Extract the classification inputs (and any pass-through identifiers) from a FMATCH product.

        The radiometer- and camera-timescale FMATCH products (CAM, CAM-CAMTIME, IMAGER-FLASH, IMAGER)
        share a flat, one-value-per-footprint layout but draw their classification inputs from different source
        variables. This helper is therefore driven by a declarative ``column_map`` supplied by each concrete reader:
        it reads each mapped source variable, applies its dtype/scale/transform, and emits it under the standardized
        :class:`FootprintVariables` name the scene classifier consumes. Some emitted columns are *raw inputs* to the
        derived-variable calculators (e.g. ``clear_area`` -> ``cloud_fraction``, ``surface_wind_u``/``_v`` ->
        ``surface_wind``, ``igbp_surface_type`` -> ``surface_type``) rather than final classification columns; those
        are derived later by :meth:`identify_scenes`.

        The time coordinate is carried as a plain ``time_variable`` data variable so it rides along through scene
        identification; the runner promotes it to a coordinate via :meth:`to_time_product` before writing.

        Parameters
        ----------
        fmatch_path : pathlib.Path
            Path to a Libera FMATCH NetCDF product file.
        record_dimension : str
            The per-record dimension the classification columns are emitted on: ``RADIOMETER_TIME`` for the
            radiometer-timescale readers (where it also names the time coordinate), or ``FOOTPRINT`` for the
            camera-timescale reader (where the time coordinate is the separate, non-unique ``CAMERA_TIME`` riding on
            this axis).
        time_variable : str
            Name of the datetime time coordinate, read from the FMATCH file and emitted under the same name (it is
            the SCENE-ID product's declared time-coordinate name): ``RADIOMETER_TIME`` for SCENE-ID-CAM and
            SCENE-ID-IMAGER(-FLASH), or ``CAMERA_TIME`` for SCENE-ID-CAM-CAMTIME. Carried as a plain data variable so
            it rides through scene identification; the runner promotes it to a coordinate via
            :meth:`to_time_product` before writing.
        column_map : dict[FootprintVariables, _FmatchColumn]
            Mapping of standardized output column -> FMATCH source column spec (name, dtype, scale, transform).
        nan_columns : tuple of FootprintVariables, optional
            Standardized columns to emit as all-NaN ``float32`` because the product has no source for them (e.g.
            ``cloud_phase`` for FMATCH-IMAGER-FLASH). Present-but-NaN keeps :meth:`identify_scenes` from trying to
            derive them and leaves the corresponding footprints unmatched for any scene that bounds them.
        passthrough_variables : tuple of str, optional
            FMATCH variables to copy verbatim onto the output (e.g. the camera-timescale identifier variables). Empty
            for the radiometer-timescale products.
        context : str, optional
            Human-readable reader description used in the missing-variable error (see :meth:`_require_variables`).

        Returns
        -------
        xr.Dataset
            Footprint data on ``record_dimension`` ready for :meth:`identify_scenes`.
        """
        logger.info("Reading FMATCH product %s", fmatch_path)
        try:
            with xr.open_dataset(fmatch_path) as opened_dataset:
                fmatch_dataset = opened_dataset.load()
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to parse input file: {fmatch_path}")

        # Fail fast (with one clear, actionable message) if any expected source variable is absent -- most
        # importantly a FMATCH file routed to the wrong reader (e.g. an IMAGER-FLASH file to the IMAGER reader,
        # which lacks the RBSP columns).
        required_source_names = tuple(column.source_name for column in column_map.values()) + tuple(
            passthrough_variables
        )
        FootprintData._require_variables(fmatch_dataset, required_source_names, context=context)

        data_variables: dict[str, tuple[list[str], NDArray]] = {}
        for target, column in column_map.items():
            values = fmatch_dataset[column.source_name].to_numpy()
            # A transform (a code remap such as CLDPIX phase -> {liquid, ice}) is applied on the raw source values;
            # otherwise an optional multiplicative scale is applied when the source units differ from the target.
            if column.transform is not None:
                values = column.transform(values)
            elif column.scale != 1.0:
                values = values * column.scale
            # Cast last so the emitted column always matches the dtype the SCENE-ID product definition declares
            # (np.where in a transform can widen to float64, and the scale multiply can promote dtype).
            data_variables[target] = ([record_dimension], np.asarray(values).astype(column.dtype, copy=False))

        # Columns with no FMATCH source: emit all-NaN float32 (see nan_columns docstring above).
        number_of_footprints = fmatch_dataset.sizes[record_dimension]
        for target in nan_columns:
            data_variables[target] = (
                [record_dimension],
                np.full(number_of_footprints, np.nan, dtype=np.float32),
            )

        # The FMATCH time coordinate (RADIOMETER_TIME, or the non-unique CAMERA_TIME riding on FOOTPRINT) is decoded to
        # datetime64[ns] by xarray; carry it as a plain data variable on the record axis under the SCENE-ID product's
        # declared name (``time_variable``), so to_time_product can promote it to the product's time coordinate.
        data_variables[time_variable] = ([record_dimension], fmatch_dataset[time_variable].to_numpy())

        # Copy each identifier variable verbatim, preserving its own dimensions so a multi-dimensional passthrough
        # (the 2-D camera_pixel_x/y range coordinates on FOOTPRINT x CAMERA_PIXEL_BOUNDS) rides through unchanged, not
        # just the 1-D per-record identifiers.
        for variable_name in passthrough_variables:
            source_variable = fmatch_dataset[variable_name]
            data_variables[variable_name] = (list(source_variable.dims), source_variable.to_numpy())

        parsed_dataset = xr.Dataset(data_variables)
        logger.info("FMATCH product read successfully with %d records", parsed_dataset.sizes[record_dimension])
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
            scene occupies, as ``scene_bin_{type}_{variable}_min`` / ``_max``
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
            logger.info(f"Added scene_id_{scene_definition.type.lower()} to dataset")

    def _calculate_required_fields(self, result_fields: list[str]):
        """Calculate necessary derived fields on data from input FootprintVariables.

        Computes derived atmospheric variables needed for scene identification, handling dependencies between
        calculated fields automatically.

        Parameters
        ----------
        result_fields : list of str
            List of field names to calculate (e.g., 'cloud_fraction', 'optical_depth')

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

            # Calculate using xarray's apply_ufunc with proper output dtype specification
            result = xr.apply_ufunc(
                spec.function,
                *inputs,
                output_dtypes=[spec.output_datatype],
                keep_attrs=True,
            )
            # ``output_dtypes`` is only a *hint* for apply_ufunc (it is used for dask graph metadata and is ignored for
            # eager numpy execution), so the array's real dtype is whatever the calculation function returns. For
            # *integer* outputs we enforce the declared dtype explicitly: calculate_trmm_surface_type returns int64
            # from np.where, but surface_type must be emitted as uint8 to match the product definition -- and the
            # write-time conformance check only performs *safe* casts, so it would refuse an automatic int64->uint8
            # narrowing and raise. Float outputs are left at their naturally computed precision (narrowing them here
            # would change long-standing numerical results), and the product definition / conformance step handles any
            # float dtype reconciliation at write time.
            if np.issubdtype(np.dtype(spec.output_datatype), np.integer):
                result = result.astype(spec.output_datatype)
            self._data[spec.output_var] = result
        else:
            raise ValueError(f"Cannot calculate fields - missing dependencies {spec.input_vars}")

    def _convert_missing_values(self, input_missing_value: float):
        """Convert input missing values in footprint data to output missing values.

        This method standardizes missing value representations by converting from the input dataset's missing value
        convention to the output convention used in FootprintData processing (np.nan).

        Parameters
        ----------
        input_missing_value : float
            Missing value indicator used in input data (e.g., -999.0, 9.96921e+36)

        Notes
        -----
        Handles two cases:
        - If input_missing_value is NaN: Uses np.isnan() for comparison
        - If input_missing_value is numeric: Uses direct equality comparison

        Modifies self._data in place, replacing all occurrences of input_missing_value
        with np.nan.

        Examples
        --------
        >>> footprint._data = xr.Dataset({'temp': [20.0, -999.0, 25.0]})
        >>> footprint._convert_missing_values(-999.0)
        >>> print(footprint._data['temp'].values)
        array([20., nan, 25.])
        """
        if np.isnan(input_missing_value):
            # For NaN input missing values, use isnan
            result = self._data.where(~np.isnan(self._data), np.nan)
        else:
            # For numeric input missing values, use direct comparison
            result = self._data.where(self._data != input_missing_value, np.nan)
        self._data = result

    def _fill_column_above_max_value(self, column_name: str, threshold: float, fill_value=np.nan):
        """Replace values above threshold with fill value for specified column.

        Parameters
        ----------
        column_name : str
            Name of the column/variable to process
        threshold : float
            Maximum allowed value - values above this will be replaced
        fill_value : float, optional
            Value to use as replacement for out-of-range data. Default is NaN.

        Raises
        ------
        ValueError
            If the specified column is not found in the dataset

        Examples
        --------
        >>> footprint._data = xr.Dataset({'cloud_fraction': [50, 120, 80]})
        >>> footprint._fill_column_above_max_value('cloud_fraction', 100.0)
        >>> print(footprint._data['cloud_fraction'].values)
        array([50., nan, 80.])
        """
        if column_name not in self._data.variables:
            raise ValueError(f"Column {column_name} not found in input data")
        else:
            self._data[column_name] = self._data[column_name].where(self._data[column_name] <= threshold, fill_value)

    @staticmethod
    def _extract_data_from_CeresSSFNOAA20FM6Ed1C(dataset: nc.Dataset) -> xr.Dataset:
        """Extract data from CERES SSF file (using numpy arrays).

        Parameters
        ----------
        dataset : netCDF4.Dataset
            Open NetCDF4 dataset in CeresSSFNOAA20FM6Ed1C format

        Returns
        -------
        xr.Dataset
            Dataset containing extracted footprint variables as numpy arrays

        """

        try:
            logger.info("Reading NetCDF data...")

            # Extract 2D arrays - read to numpy first
            cloud_fraction_np = np.array(dataset.groups["Cloudy_Footprint_Area"].variables["layers_coverages"][:])
            logger.debug(f"Cloud fraction shape: {cloud_fraction_np.shape}")

            igbp_surface_type_np = np.array(dataset.groups["Surface_Map"].variables["surface_igbp_type"][:])
            logger.debug(f"IGBP surface type shape: {igbp_surface_type_np.shape}")

            cloud_phase_var = dataset.groups["Cloudy_Footprint_Area"].variables["cloud_particle_phase_37um_mean"]
            cloud_phase_np = np.array(cloud_phase_var[:])
            cloud_phase_fill_value = cloud_phase_var._FillValue if hasattr(cloud_phase_var, "_FillValue") else None
            logger.debug(f"Cloud phase shape: {cloud_phase_np.shape}")
            logger.debug(f"Cloud phase fill value: {cloud_phase_fill_value}")

            optical_depth_np = np.array(
                dataset.groups["Cloudy_Footprint_Area"].variables["cloud_optical_depth_mean"][:]
            )
            logger.debug(f"Optical depth shape: {optical_depth_np.shape}")

            # Extract 1D arrays - read to numpy first
            surface_wind_u_np = np.array(dataset.groups["Full_Footprint_Area"].variables["surface_wind_u_vector"][:])
            logger.debug(f"Surface wind U shape: {surface_wind_u_np.shape}")

            surface_wind_v_np = np.array(dataset.groups["Full_Footprint_Area"].variables["surface_wind_v_vector"][:])
            logger.debug(f"Surface wind V shape: {surface_wind_v_np.shape}")

            clear_area_np = np.array(dataset.groups["Clear_Footprint_Area"].variables["clear_coverage"][:])
            logger.debug(f"Clear area shape: {clear_area_np.shape}")

            # Viewing-geometry angles, one value per footprint, read from the CERES SSF "Viewing_Angles" group. These
            # feed the geometry classification bins on every scene definition. The SSF stores them as float32 degrees;
            # solar zenith spans 0-180, viewing zenith 0-90, relative azimuth 0-360 (see the variables' valid_range
            # attributes and https://ceres.larc.nasa.gov/data/#ssf-level-2). NOTE the SSF variable is named
            # "view_zenith_angle"; we carry it forward under the pipeline name "viewing_zenith_angle".
            viewing_angles_group = dataset.groups["Viewing_Angles"]
            solar_zenith_angle_np = np.array(viewing_angles_group.variables["solar_zenith_angle"][:])
            viewing_zenith_angle_np = np.array(viewing_angles_group.variables["view_zenith_angle"][:])
            relative_azimuth_angle_np = np.array(viewing_angles_group.variables["relative_azimuth_angle"][:])

            # The SSF marks missing angles with a large float32 sentinel (_FillValue ~ 3.4e38). Convert those to NaN
            # so they are treated as "no value" downstream: a NaN in any classification variable leaves the footprint
            # unmatched (scene ID 0) in the scene matcher (see Scene.matches / _identify_vectorized), and NetCDF
            # writing stores the NaN as-is (float variables carry no declared fill value; NaN is inherently missing).
            for angle_name, angle_array in (
                ("solar_zenith_angle", solar_zenith_angle_np),
                ("view_zenith_angle", viewing_zenith_angle_np),
                ("relative_azimuth_angle", relative_azimuth_angle_np),
            ):
                angle_variable = viewing_angles_group.variables[angle_name]
                angle_fill_value = angle_variable._FillValue if hasattr(angle_variable, "_FillValue") else None
                if angle_fill_value is not None:
                    angle_array[angle_array == angle_fill_value] = np.nan
            logger.debug("Converted viewing-angle fill values to NaN")

            # Time of observation, one value per footprint. The downstream Libera SCENE-ID product is written on
            # the same "RADIOMETER_TIME" axis as its L1B input (see the L1B_RAD product), so we carry the CERES SSF
            # observation time through the pipeline as the RADIOMETER_TIME coordinate. In the CERES SSF format the
            # time is stored as floating-point "days since 1970-01-01 00:00:00" UTC.
            # Reference: https://ceres.larc.nasa.gov/data/#ssf-level-2
            observation_time_np = np.array(dataset.groups["Time_and_Position"].variables["time"][:])
            logger.debug(f"Observation time shape: {observation_time_np.shape}")

            logger.info("NetCDF data read successfully")

        except KeyError as e:
            raise ValueError(f"Required variable or group not found in NetCDF file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading NetCDF file: {e}")

        # Slice 2D arrays to extract specific layers/estimates
        logger.info("Extracting layers from 2D arrays...")

        igbp_surface_type = igbp_surface_type_np[:, 0].astype(np.uint8)
        cloud_fraction_lower = cloud_fraction_np[:, 1]
        cloud_fraction_upper = cloud_fraction_np[:, 2]
        cloud_phase_lower = cloud_phase_np[:, 0]
        cloud_phase_upper = cloud_phase_np[:, 1]
        optical_depth_lower = optical_depth_np[:, 0]
        optical_depth_upper = optical_depth_np[:, 1]

        # Process cloud phase arrays: replace fill values with NaN
        if cloud_phase_fill_value is not None:
            cloud_phase_lower = np.where(cloud_phase_lower == cloud_phase_fill_value, np.nan, cloud_phase_lower)
            cloud_phase_upper = np.where(cloud_phase_upper == cloud_phase_fill_value, np.nan, cloud_phase_upper)

        # Force all non-NaN values to be either 1 or 2 (whichever is closer)
        # Round to nearest integer (1 or 2)
        cloud_phase_lower = np.where(
            ~np.isnan(cloud_phase_lower), np.round(np.clip(cloud_phase_lower, 1, 2)), cloud_phase_lower
        )
        cloud_phase_upper = np.where(
            ~np.isnan(cloud_phase_upper), np.round(np.clip(cloud_phase_upper, 1, 2)), cloud_phase_upper
        )

        # Convert the CERES SSF observation time ("days since 1970-01-01") into absolute datetime64[ns] values.
        # We build the timestamps by adding the (fractional) day offsets to the 1970-01-01 epoch. Using
        # timedelta64[ns] keeps sub-second precision (the SSF sampling is well under a second apart). The absolute
        # datetime representation lets write_libera_data_product re-encode it with whatever epoch/units the product
        # definition specifies (the Libera convention is "nanoseconds since 1958-01-01").
        ssf_epoch = np.datetime64("1970-01-01T00:00:00", "ns")
        nanoseconds_per_day = 24 * 60 * 60 * 1_000_000_000
        radiometer_time = ssf_epoch + (observation_time_np * nanoseconds_per_day).astype("timedelta64[ns]")

        # Create xarray Dataset with numpy arrays
        logger.info("Creating xarray Dataset...")

        parsed_dataset = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: ([RADIOMETER_TIME_DIMENSION], igbp_surface_type),
                FootprintVariables.SURFACE_WIND_U: ([RADIOMETER_TIME_DIMENSION], surface_wind_u_np),
                FootprintVariables.SURFACE_WIND_V: ([RADIOMETER_TIME_DIMENSION], surface_wind_v_np),
                FootprintVariables.CLEAR_AREA: ([RADIOMETER_TIME_DIMENSION], clear_area_np),
                FootprintVariables.OPTICAL_DEPTH_LOWER: ([RADIOMETER_TIME_DIMENSION], optical_depth_lower),
                FootprintVariables.OPTICAL_DEPTH_UPPER: ([RADIOMETER_TIME_DIMENSION], optical_depth_upper),
                FootprintVariables.CLOUD_FRACTION_LOWER: ([RADIOMETER_TIME_DIMENSION], cloud_fraction_lower),
                FootprintVariables.CLOUD_FRACTION_UPPER: ([RADIOMETER_TIME_DIMENSION], cloud_fraction_upper),
                FootprintVariables.CLOUD_PHASE_LOWER: ([RADIOMETER_TIME_DIMENSION], cloud_phase_lower),
                FootprintVariables.CLOUD_PHASE_UPPER: ([RADIOMETER_TIME_DIMENSION], cloud_phase_upper),
                FootprintVariables.SOLAR_ZENITH_ANGLE: ([RADIOMETER_TIME_DIMENSION], solar_zenith_angle_np),
                FootprintVariables.VIEWING_ZENITH_ANGLE: ([RADIOMETER_TIME_DIMENSION], viewing_zenith_angle_np),
                FootprintVariables.RELATIVE_AZIMUTH_ANGLE: ([RADIOMETER_TIME_DIMENSION], relative_azimuth_angle_np),
                # RADIOMETER_TIME's name equals its dimension, so xarray treats it as a dimension coordinate the
                # moment this Dataset is built. It rides through scene identification as that coordinate; the
                # runner's to_time_product() step is then idempotent for it.
                RADIOMETER_TIME_DIMENSION: ([RADIOMETER_TIME_DIMENSION], radiometer_time),
            }
        )

        logger.info(f"Dataset created successfully with {parsed_dataset.sizes[RADIOMETER_TIME_DIMENSION]} footprints")

        return parsed_dataset

    def to_time_product(self, time_variable: str = RADIOMETER_TIME_DIMENSION) -> xr.Dataset:
        """Return the footprint data ready to write on its per-footprint time axis.

        The scene-ID CAM/IMAGER/FLASH products contain exactly one footprint per observation time and are written
        on the same time dimension as their upstream product, so downstream consumers can align scene IDs to the
        upstream records positionally. :class:`FootprintData` already carries data on that dimension, so this
        method only promotes the time variable to a coordinate so the result is ready to hand to
        :func:`libera_utils.io.netcdf.write_libera_data_product` with the matching ``time_variable``.

        Parameters
        ----------
        time_variable : str, optional
            Name of the time variable to promote to a coordinate. Defaults to ``RADIOMETER_TIME`` (the
            radiometer-timescale products); the camera-timescale product passes ``CAMERA_TIME``.

        Returns
        -------
        xr.Dataset
            A copy of the internal dataset with the time variable promoted to a coordinate and a placeholder
            ``Quality_Flag`` added on the same dimension.

        Raises
        ------
        ValueError
            If ``time_variable`` is not present (e.g. the instance was built directly from a synthetic dataset
            that did not include observation times).
        """
        if time_variable not in self._data.variables:
            raise ValueError(
                f"Cannot build a time product: '{time_variable}' is not present in the footprint data. It is "
                "populated by the FootprintData reader (e.g. from_ceres_ssf()); datasets constructed directly "
                "must add it themselves."
            )
        # Work on a copy so callers that inspect FootprintData._data afterwards still see the internal
        # representation (set_coords otherwise mutates the shared dataset).
        product = self._data.set_coords(time_variable)
        # The per-footprint dimension is whatever axis the time variable lives on: RADIOMETER_TIME for the
        # radiometer-timescale products (where the time variable is a dimension coordinate), or FOOTPRINT for the
        # camera-timescale product (where CAMERA_TIME is a non-unique coordinate riding on the FOOTPRINT axis).
        (time_dimension,) = product[time_variable].dims
        # TODO[LIBSDC-810]: Add real quality flag to the product
        product = add_placeholder_quality_flag(product, dimension=time_dimension)
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

    def export_to_netcdf(self, netcdf_path: str | pathlib.Path) -> None:
        """Write the internal footprint dataset straight to a NetCDF file.

        This is a raw dump of the working dataset (no product-definition conformance step); use it for debugging
        or intermediate inspection. Operational products are written via
        :func:`libera_utils.io.netcdf.write_libera_data_product`.

        Parameters
        ----------
        netcdf_path : str or pathlib.Path
            Destination path for the NetCDF file. An existing file is overwritten.
        """
        self._data.to_netcdf(path=netcdf_path, mode="w")
