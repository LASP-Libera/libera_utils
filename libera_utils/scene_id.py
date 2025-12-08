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
from libera_utils.scene_definitions import SceneDefinition

logger = logging.getLogger(__name__)


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
    """Convert TRMM surface type to IGBP surface type classification.

    Parameters
    ----------
    igbp_surface_type : int or ndarray of int
        IGBP surface type codes

    Returns
    -------
    int or ndarray of int
        TRMM surface type codes

    Raises
    ------
    ValueError
        If any input values cannot be converted to a valid IGBP surface type

    Notes
    -----
    The conversion uses a lookup table derived from the TRMMSurfaceType.value property.
    Values that don't correspond to valid TRMM surface types will raise a ValueError.

    Examples
    --------
    >>> calculate_trmm_surface_type(1)
    5  # Maps IGBP HI_SHRUB back to TRMM type 5
    >>> calculate_trmm_surface_type(np.array([1, 0]))
    array([5, 17])
    >>> calculate_trmm_surface_type(999)
    ValueError: Cannot convert IGBP surface type value(s) to TRMM surface type: [999]
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
        output_datatype=int,
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
        scene_definitions : list of SceneDefinition
            List of scene definition objects to apply for classification

        Returns
        -------
        FootprintData
            Processed footprint data object containing original variables, calculated
            derived fields, and scene IDs.

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
        >>> scene_defs = [SceneDefinition(Path("trmm.csv"))]
        >>> footprint_data = FootprintData.from_ceres_ssf(
        ...     Path("CERES_SSF_NOAA20_2024001.nc"),
        ...     scene_defs
        ... )
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
    def from_cldpx_viirs_geos_cam_groundscene(cls):
        """Process cloud pixel/VIIRS/GEOS/camera/ground scene data format.

        Raises
        ------
        NotImplementedError
            This data format is not yet supported

        Notes
        -----
        TODO: LIBSDC-672 Implement processing for alternative data formats including:
        - Cloud pixel data
        - VIIRS observations
        - GEOS model data
        - Camera data
        - Ground scene classifications
        """
        raise NotImplementedError(
            "Calculating scene IDs not implemented for cldpx/viirs/geos/cam/ground scene data format."
        )

    @classmethod
    def from_clouds_groundscene(cls):
        """Process clouds/ground scene data format.

        Raises
        ------
        NotImplementedError
            This data format is not yet supported

        Notes
        -----
        TODO: LIBSDC-673 Implement processing for cloud and ground scene data formats.
        """
        raise NotImplementedError("Calculating scene IDs not implemented for clouds/ground scene data format.")

    def identify_scenes(
        self,
        scene_definitions: list[SceneDefinition] = [
            SceneDefinition(pathlib.Path(config.get("TRMM_SCENE_DEFINITION"))),
            SceneDefinition(pathlib.Path(config.get("ERBE_SCENE_DEFINITION"))),
        ],
        additional_scene_definitions_files: list[pathlib.Path] | None = None,
    ):
        """Identify and assign scene IDs to all footprints based on scene definitions.

        Applies scene classification rules from one or more SceneDefinition objects to assign scene IDs to each
        footprint in the dataset.

        Parameters
        ----------
        scene_definitions : list[SceneDefinition]
            List of SceneDefinition objects from standard libera_utils definitions
        additional_scene_definitions_files : list of pathlib.Path or None
            List of scene definition files containing classification rules for custom analysis.

        Notes
        -----
        This method modifies self._data in place by adding scene IDs for each row of footprint data.

        For each SceneDefinition provided:
        1. Validates that all required variables exist in the footprint data
        2. Matches each footprint to a scene based on variable ranges
        3. Adds a new variable to the dataset with the scene IDs

        Footprints that don't match any scene are assigned a scene ID of 0.

        TODO: LIBSDC-674 - Add unfiltering scene ID algorithm

        Examples
        --------
        >>> footprint_data = FootprintData(dataset)
        >>> footprint_data.identify_scenes()
        """
        if scene_definitions is None:
            raise ValueError("No scene definitions provided.")
        if len(scene_definitions) == 0:
            raise ValueError("Scene definitions list is empty.")

        # Calculate required fields for each scene
        required_calculated_fields = list()
        if additional_scene_definitions_files:
            for additional_scene_definition in additional_scene_definitions_files:
                scene_definitions.append(SceneDefinition(additional_scene_definition))
        for scene_definition in scene_definitions:
            required_calculated_fields += scene_definition.required_columns

        self._calculate_required_fields(required_calculated_fields)
        for scene_definition in scene_definitions:
            logger.info(f"Identifying {scene_definition.type} scenes...")
            self._data = scene_definition.identify_and_update(self._data)
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

    def _calculate_single_field_from_spec(self, spec: CalculationSpec, calculated: list[str]):
        """Calculate a single field from input FootprintVariables.

        Applies the calculation function specified in the CalculationSpec to the input variables, creating a new
        variable in the dataset.

        Parameters
        ----------
        spec : CalculationSpec
            Specification defining the calculation to perform
        calculated : list of str
            List of variable names already available in the dataset

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
            self._data[spec.output_var] = result
        else:
            raise ValueError(f"Cannot calculate fields - missing dependencies {spec.input_vars}")

    def _convert_missing_values(self, input_missing_value: float):
        """Convert input missing values in footprint data to output missing values.

        This method standardizes missing value representations by converting from the input dataset's missing value
        convention to the output convention used in FootprintData processing (np.NaN).

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
        with np.NaN.

        Examples
        --------
        >>> footprint._data = xr.Dataset({'temp': [20.0, -999.0, 25.0]})
        >>> footprint._convert_missing_values(-999.0)
        >>> print(footprint._data['temp'].values)
        array([20., nan, 25.])
        """
        if np.isnan(input_missing_value):
            # For NaN input missing values, use isnan
            result = self._data.where(~np.isnan(self._data), np.NaN)
        else:
            # For numeric input missing values, use direct comparison
            result = self._data.where(self._data != input_missing_value, np.NaN)
        self._data = result

    def _fill_column_above_max_value(self, column_name: str, threshold: float, fill_value=np.NaN):
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
        chunk_size : int, optional
            Number of footprints per chunk along the first dimension (parameter kept for compatibility but not used)

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

            logger.info("NetCDF data read successfully")

        except KeyError as e:
            raise ValueError(f"Required variable or group not found in NetCDF file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading NetCDF file: {e}")

        # Slice 2D arrays to extract specific layers/estimates
        logger.info("Extracting layers from 2D arrays...")

        igbp_surface_type = igbp_surface_type_np[:, 0]
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

        # Create xarray Dataset with numpy arrays
        logger.info("Creating xarray Dataset...")

        parsed_dataset = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], igbp_surface_type),
                FootprintVariables.SURFACE_WIND_U: (["footprint"], surface_wind_u_np),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], surface_wind_v_np),
                FootprintVariables.CLEAR_AREA: (["footprint"], clear_area_np),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], optical_depth_lower),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], optical_depth_upper),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], cloud_fraction_lower),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], cloud_fraction_upper),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], cloud_phase_lower),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], cloud_phase_upper),
            }
        )

        logger.info(f"Dataset created successfully with {len(parsed_dataset.footprint)} footprints")

        return parsed_dataset

    def export_to_netcdf(self, netcdf_path):
        self._data.to_netcdf(path=netcdf_path, mode="w")
