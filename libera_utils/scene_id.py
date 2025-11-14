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
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from libera_utils.config import config
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)

TRMM_SCENE_DEFINITION = config.get("TRMM_SCENE_DEFINITION")
ERBE_SCENE_DEFINITION = config.get("ERBE_SCENE_DEFINITION")


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
        >>> TRMMSurfaceType.TRMM_1.igbp_surface_type
        <TRMMSurfaceType.HI_SHRUB: 1>
        >>> TRMMSurfaceType.TRMM_17.igbp_surface_type
        <TRMMSurfaceType.OCEAN: 0>
        """
        IGBP_TO_TRMM_MAP = {
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
        return IGBP_TO_TRMM_MAP[self.value]


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
    >>> cloud_fraction(30.0)
    70.0
    >>> cloud_fraction(np.array([10, 25, 90]))
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
    >>> surface_wind(3.0, 4.0)
    5.0
    >>> surface_wind(np.array([3, np.nan]), np.array([4, 5]))
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


def calculate_cloud_fraction_weighted_property_for_layer(
    property_lower: float | NDArray[np.floating],
    property_upper: float | NDArray[np.floating],
    cloud_fraction_lower: float | NDArray[np.floating],
    cloud_fraction_upper: float | NDArray[np.floating],
    cloud_fraction: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Calculate cloud fraction weighted property from upper and lower layers.

    Computes a weighted average of a cloud property across two atmospheric layers, where the weights are determined by
    each layer's contribution to total cloud fraction.

    Parameters
    ----------
    property_lower : float or ndarray
        Property values for the lower cloud layer
    property_upper : float or ndarray
        Property values for the upper cloud layer
    cloud_fraction_lower : float or ndarray
        Cloud fraction for the lower cloud layer (0-100)
    cloud_fraction_upper : float or ndarray
        Cloud fraction for the upper cloud layer (0-100)
    cloud_fraction : float or ndarray
        Total cloud fraction (0-100)

    Returns
    -------
    float or ndarray
        Property weighted by cloud fraction and summed across layers,
        or np.nan if no valid data or zero total cloud fraction

    Notes
    -----
    The weighting formula is:
        result = (property_lower * cloud_fraction_lower / cloud_fraction) +
                 (property_upper * cloud_fraction_upper / cloud_fraction)

    Returns NaN when:
    - Total cloud fraction is zero or NaN
    - Both layers have invalid (NaN) property values
    - All cloud fractions are NaN

    Examples
    --------
    >>> calculate_cloud_fraction_weighted_property_for_layer(
    ...     property_lower=10.0, property_upper=20.0,
    ...     cloud_fraction_lower=30.0, cloud_fraction_upper=70.0,
    ...     cloud_fraction=100.0
    ... )
    17.0  # (10*30 + 20*70)/100
    """
    # Handle zero total cloud fraction
    valid_total = ~np.isnan(cloud_fraction) & (cloud_fraction > 0)

    # Create masks for valid data
    valid_lower = ~np.isnan(property_lower) & ~np.isnan(cloud_fraction_lower) & valid_total
    valid_upper = ~np.isnan(property_upper) & ~np.isnan(cloud_fraction_upper) & valid_total

    # Safely calculate weights by using np.divide with where parameter
    # This avoids division by zero warnings/errors
    weight_lower = np.zeros_like(cloud_fraction_lower, dtype=float)
    weight_upper = np.zeros_like(cloud_fraction_upper, dtype=float)

    # Only divide where valid_total is True (cloud_fraction > 0)
    np.divide(cloud_fraction_lower, cloud_fraction, out=weight_lower, where=valid_total)
    np.divide(cloud_fraction_upper, cloud_fraction, out=weight_upper, where=valid_total)

    # Calculate weighted values
    weighted_lower = np.where(valid_lower, weight_lower * property_lower, 0)
    weighted_upper = np.where(valid_upper, weight_upper * property_upper, 0)

    # Sum and handle cases with no valid data
    result = weighted_lower + weighted_upper
    no_valid_data = ~(valid_lower | valid_upper)

    return np.where(no_valid_data, np.nan, result)


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

    See Also
    --------
    calculate_cloud_fraction_weighted_property_for_layer : General weighting function

    Examples
    --------
    >>> optical_depth(5.0, 15.0, 40.0, 60.0, 100.0)
    11.0  # (5*40 + 15*60)/100
    """
    return calculate_cloud_fraction_weighted_property_for_layer(
        optical_depth_lower, optical_depth_upper, cloud_fraction_lower, cloud_fraction_upper, cloud_fraction
    )


def calculate_cloud_phase(
    cloud_phase_lower: float | NDArray[np.floating],
    cloud_phase_upper: float | NDArray[np.floating],
    cloud_fraction_lower: float | NDArray[np.floating],
    cloud_fraction_upper: float | NDArray[np.floating],
    cloud_fraction: float | NDArray[np.floating],
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

    Returns
    -------
    float or ndarray
        Cloud phase weighted by cloud fraction and rounded to nearest integer
        (1=liquid, 2=ice), or np.nan if no valid data

    Notes
    -----
    The weighted average is rounded to the nearest integer to provide a discrete phase classification. Values between
    1 and 2 are rounded to either 1 or 2, with 1.5 rounding to 2.

    Examples
    --------
    >>> cloud_phase(1.0, 2.0, 30.0, 70.0, 100.0)
    2.0  # (1*30 + 2*70)/100 = 1.7, rounds to 2
    >>> cloud_phase(1.0, 1.0, 50.0, 50.0, 100.0)
    1.0  # All liquid
    """
    weighted_cloud_phase = calculate_cloud_fraction_weighted_property_for_layer(
        cloud_phase_lower, cloud_phase_upper, cloud_fraction_lower, cloud_fraction_upper, cloud_fraction
    )

    # Handle both scalar and array cases
    is_nan = np.isnan(weighted_cloud_phase)
    if np.isscalar(is_nan):
        if is_nan:
            return np.nan
        return np.round(weighted_cloud_phase, 0)
    else:
        return np.where(is_nan, np.nan, np.round(weighted_cloud_phase, 0))


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

    Defines the parameters needed to calculate a derived variable from input data,  including the calculation function,
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
    ...     function=cloud_fraction,
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
        ],
        output_datatype=float,
        dependent_calculations=[FootprintVariables.CLOUD_FRACTION],
    ),
}

# Scene Definitions


@dataclass
class Scene:
    """Represents a single scene with its variable bin definitions.

    A scene defines a specific atmospheric state characterized by ranges of multiple variables (e.g., cloud fraction,
    optical depth, surface type). Data points are classified into scenes when all their variable values fall within the
    scene's defined ranges.

    Attributes
    ----------
    scene_id : int
        Unique identifier for this scene
    variable_ranges : dict of str to tuple of (float, float)
        Dictionary mapping variable names to (min, max) tuples defining
        the acceptable range for each variable. None values indicate
        unbounded ranges (no min or no max constraint).

    Methods
    -------
    matches(data_point)
        Check if a data point belongs to this scene

    Examples
    --------
    >>> scene = Scene(
    ...     scene_id=1,
    ...     variable_ranges={
    ...         "cloud_fraction": (0.0, 50.0),
    ...         "optical_depth": (0.0, 10.0)
    ...     }
    ... )
    >>> scene.matches({"cloud_fraction": 30.0, "optical_depth": 5.0})
    True
    >>> scene.matches({"cloud_fraction": 60.0, "optical_depth": 5.0})
    False
    """

    scene_id: int
    variable_ranges: dict[str, tuple[float, float]]

    def matches(self, data_point: dict[str, float]) -> bool:
        """Check if a data point falls within all variable ranges for this scene.

        Parameters
        ----------
        data_point : dict of str to float
            Dictionary of variable names to values

        Returns
        -------
        bool
            True if data point matches all variable ranges, False otherwise

        Notes
        -----
        A data point matches when:
        - All required variables are present in the data point
        - All variable values are within their specified ranges (inclusive)
        - No variable values are NaN

        Range boundaries:
        - None for min_val means no lower bound
        - None for max_val means no upper bound
        - Both bounds are inclusive when specified

        Examples
        --------
        >>> scene = Scene(
        ...     scene_id=1,
        ...     variable_ranges={"temp": (0.0, 100.0), "pressure": (None, 1000.0)}
        ... )
        >>> scene.matches({"temp": 50.0, "pressure": 900.0})
        True
        >>> scene.matches({"temp": 150.0, "pressure": 900.0})
        False
        >>> scene.matches({"temp": np.nan, "pressure": 900.0})
        False
        """
        for var_name, (min_val, max_val) in self.variable_ranges.items():
            if var_name not in data_point:
                return False

            value = data_point[var_name]

            # Handle NaN values
            if np.isnan(value):
                return False

            # Check if value is within range (inclusive on both ends)
            # Handle None for unbounded ranges
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False

        return True


class SceneDefinition:
    """Defines scenes and their classification rules from CSV configuration.

    Loads and manages scene definitions from a CSV file, providing functionality to identify which scene a given set of
    atmospheric measurements belongs to.

    Attributes
    ----------
    type : str
        Type of scene definition (e.g., 'TRMM', 'ERBE'), derived from filename
    scenes : list of Scene
        List of scene definitions with their variable ranges
    required_columns : list of str
        List of variable names required for scene identification

    Methods
    -------
    identify(data)
        Identify scene IDs for all data points in a dataset
    validate_input_data_columns(data)
        Validate that dataset contains all required variables

    Notes
    -----
    Expected CSV format:
        scene_id,variable1_min,variable1_max,variable2_min,variable2_max,...
        1,0.0,10.0,20.0,30.0,...
        2,10.0,20.0,30.0,40.0,...

    Each variable must have both a _min and _max column. NaN or empty values
    indicate unbounded ranges.

    Examples
    --------
    >>> scene_def = SceneDefinition(Path("trmm.csv"))
    >>> print(scene_def.type)
    'TRMM'
    >>> print(len(scene_def.scenes))
    42
    """

    def __init__(self, definition_path: pathlib.Path):
        """Initialize scene definition from CSV file.

        Parameters
        ----------
        definition_path : pathlib.Path
            Path to CSV file containing scene definitions

        Raises
        ------
        FileNotFoundError
            If the definition file does not exist
        ValueError
            If the CSV format is invalid or missing required columns

        Notes
        -----
        The CSV file must contain:
        - A 'scene_id' column with unique integer identifiers
        - Pairs of columns for each variable: {var}_min and {var}_max
        - At least one variable pair
        """
        self.type = definition_path.stem.upper()

        # Read CSV with scene definitions
        scene_df = pd.read_csv(definition_path)

        # TODO: LIBSDC-589 Add validation for definition file contents
        # self.validate_scene_definition_file()

        # Parse variable names from column headers
        # Columns should be: scene_id, var1_min, var1_max, var2_min, var2_max, etc.
        self.required_columns = self._extract_variable_names(scene_df.columns)

        # Create SceneBin objects for each row
        self.scenes = []
        for _, row in scene_df.iterrows():
            scene_id = int(row["scene_id"])
            variable_ranges = self._parse_row_to_ranges(row, self.required_columns)
            self.scenes.append(Scene(scene_id, variable_ranges))

        logger.info(f"Loaded {len(self.scenes)} scenes from {definition_path}")
        logger.info(f"Required variables: {self.required_columns}")

    def _extract_variable_names(self, columns: pd.Index) -> list[str]:
        """Extract unique variable names from min/max column pairs.

        Parameters
        ----------
        columns : pd.Index
            Column names from the CSV

        Returns
        -------
        list of str
            Sorted list of unique variable names

        Notes
        -----
        Variable names are extracted by removing the '_min' or '_max' suffix from column names. Only columns with these
        suffixes are considered as variable definitions.

        Examples
        --------
        >>> cols = pd.Index(['scene_id', 'temp_min', 'temp_max', 'pressure_min', 'pressure_max'])
        >>> scene_def._extract_variable_names(cols)
        ['pressure', 'temp']
        """
        variable_names = set()
        for col in columns:
            if col == "scene_id":
                continue

            # Remove _min or _max suffix to get variable name
            if col.endswith("_min"):
                var_name = col[:-4]  # Remove '_min'
                variable_names.add(var_name)
            elif col.endswith("_max"):
                var_name = col[:-4]  # Remove '_max'
                variable_names.add(var_name)

        return sorted(list(variable_names))

    def _parse_row_to_ranges(
        self, row: pd.Series, variable_names: list[str]
    ) -> dict[str, tuple[float | None, float | None]]:
        """Parse a CSV row into variable ranges.

        Parameters
        ----------
        row : pd.Series
            Row from the scene definition DataFrame containing scene_id and
            variable min/max values
        variable_names : list of str
            List of variable names to extract ranges for

        Returns
        -------
        dict of str to tuple of (float or None, float or None)
            Dictionary mapping variable names to (min, max) tuples.
            None values indicate unbounded ranges (no constraint).

        Notes
        -----
        For each variable, looks for columns named {variable}_min and
        {variable}_max. NaN values in the CSV are converted to None to
        indicate unbounded ranges.

        Examples
        --------
        >>> row = pd.Series({'scene_id': 1, 'temp_min': 0.0, 'temp_max': 100.0,
        ...                  'pressure_min': np.nan, 'pressure_max': 1000.0})
        >>> scene_def._parse_row_to_ranges(row, ['temp', 'pressure'])
        {'temp': (0.0, 100.0), 'pressure': (None, 1000.0)}
        """
        ranges = {}
        for var_name in variable_names:
            min_col = f"{var_name}_min"
            max_col = f"{var_name}_max"

            min_val = row.get(min_col, None)
            max_val = row.get(max_col, None)

            # Convert NaN to None for unbounded ranges
            if pd.isna(min_val):
                min_val = None
            else:
                min_val = float(min_val)

            if pd.isna(max_val):
                max_val = None
            else:
                max_val = float(max_val)

            ranges[var_name] = (min_val, max_val)

        return ranges

    def identify(self, data: xr.Dataset) -> xr.DataArray:
        """Identify scene IDs for all data points in the dataset.

        Classifies each data point in the dataset by finding the first scene whose variable ranges match all the data
        point's variable values.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing all required variables for scene identification

        Returns
        -------
        xr.DataArray
            Array of scene IDs with the same dimensions as the input data.
            Scene ID of -1 indicates no matching scene was found for that point.

        Raises
        ------
        ValueError
            If the dataset is missing required variables

        Notes
        -----
        - Scene matching uses first-match priority: if multiple scenes could
          match a data point, the first one in the definition list is assigned
        - Data points with NaN values in any required variable are not matched
        - The method logs statistics about matched/unmatched points and the
          distribution of scene IDs

        Examples
        --------
        >>> data = xr.Dataset({
        ...     'cloud_fraction': ([('x',)], [20.0, 60.0, 85.0]),
        ...     'optical_depth': ([('x',)], [5.0, 15.0, 25.0])
        ... })
        >>> scene_def = SceneDefinition(Path("scenes.csv"))
        >>> scene_ids = scene_def.identify(data)
        >>> print(scene_ids.values)
        array([1, 2, -1])  # Last point didn't match any scene
        """
        self.validate_input_data_columns(data)

        # Get the dimensions and shape
        dims = list(data.dims.keys())
        shape = tuple(data.dims[dim] for dim in dims)

        # Initialize result array with -1 (no match)
        scene_ids = np.full(shape, -1, dtype=np.int32)

        # Vectorized approach for better performance
        scene_ids = self._identify_vectorized(data, dims, shape)

        # Create DataArray with same dimensions as input
        result = xr.DataArray(
            scene_ids,
            dims=dims,
            coords={dim: data.coords[dim] for dim in dims if dim in data.coords},
            name=f"scene_id_{self.type.lower()}",
        )

        # Log statistics
        unique_scenes, counts = np.unique(scene_ids[scene_ids != -1], return_counts=True)
        unmatched = np.sum(scene_ids == -1)
        logger.info(f"{self.type} scene identification complete:")
        logger.info(f"  Matched: {np.sum(scene_ids != -1)} points")
        logger.info(f"  Unmatched: {unmatched} points")
        for scene_id, count in zip(unique_scenes, counts):
            logger.info(f"  Scene {scene_id}: {count} points")

        return result

    def _identify_vectorized(self, data: xr.Dataset, dims: list[str], shape: tuple[int, ...]) -> np.ndarray:
        """Vectorized scene identification for better performance.

        Uses NumPy array operations to efficiently classify all data points simultaneously rather than iterating
        point-by-point.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing all required variables
        dims : list of str
            List of dimension names
        shape : tuple of int
            Shape of the output array

        Returns
        -------
        np.ndarray
            Array of scene IDs with shape matching input dimensions

        Notes
        -----
        For each scene, creates a boolean mask identifying all matching points, then assigns the scene ID to those
        points. Earlier scenes in the list have priority for overlapping classifications.
        """
        # Initialize result array
        scene_ids = np.full(shape, -1, dtype=np.int32)

        # For each scene, create a mask of matching points
        for scene in self.scenes:
            # Start with all True mask
            mask = np.ones(shape, dtype=bool)

            # Apply each variable range constraint
            for var_name, (min_val, max_val) in scene.variable_ranges.items():
                var_data = data[var_name].values

                # Update mask with this variable's constraints
                if min_val is not None:
                    mask &= var_data >= min_val
                if max_val is not None:
                    mask &= var_data <= max_val

                # Exclude NaN values
                mask &= ~np.isnan(var_data)

            # Assign scene ID to matching points (only if not already assigned)
            # This gives priority to earlier scenes in the list
            scene_ids = np.where((mask) & (scene_ids == -1), scene.scene_id, scene_ids)

        return scene_ids

    def validate_input_data_columns(self, data: xr.Dataset):
        """Ensure input data contains all required FootprintVariables.

        Parameters
        ----------
        data : xr.Dataset
            Dataset to validate

        Raises
        ------
        ValueError
            If required variables are missing from the dataset, with a message
            listing all missing variables

        Examples
        --------
        >>> scene_def = SceneDefinition(Path("scenes.csv"))
        >>> scene_def.required_columns = ['cloud_fraction', 'optical_depth']
        >>> data = xr.Dataset({'cloud_fraction': [10, 20]})
        >>> scene_def.validate_input_data_columns(data)
        ValueError: Required columns ['optical_depth'] not in input data for TRMM scene identification.
        """
        missing_columns = []
        for column in self.required_columns:
            if column not in data.data_vars:
                missing_columns.append(column)

        if missing_columns:
            raise ValueError(
                f"Required columns {missing_columns} not in input data for {self.type} scene identification."
            )

    def validate_scene_definition_file(self):
        """Ensure scene definition file contains valid column names, bin ranges, that classification parameters are not
        duplicated across IDs, and that there are no gaps in classification bins.

        Raises
        ------
        NotImplementedError
            This validation is not yet implemented

        Notes
        -----
        TODO: LIBSDC-589 Implement validation checks for:
        - Valid column naming conventions
        - Non-overlapping scene definitions
        - Complete coverage of parameter space (no gaps)
        - Consistent min/max value ordering
        """
        raise NotImplementedError()


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
    def from_ceres_ssf(cls, ssf_path: pathlib.Path, scene_definitions: list[SceneDefinition]):
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
            Processed dataset containing original variables and calculated fields ready for scene identification.

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
        4. Match footprints to scene IDs

        Maximum value thresholds applied:
        - Cloud fraction: 100%
        - Cloud phase: 2 (ice)
        - Optical depth: 500

        Examples
        --------
        >>> footprint = FootprintData()
        >>> scene_defs = [SceneDefinition(Path("trmm.csv"))]
        >>> data = footprint.process_ssf_and_camera(
        ...     Path("CERES_SSF_NOAA20_2024001.nc"),
        ...     scene_defs
        ... )
        """
        try:
            with smart_open(ssf_path) as file:
                with nc.Dataset(file) as dataset:
                    footprint_data = cls(cls()._extract_data_from_CeresSSFNOAA20FM6Ed1C(dataset))
        except FileNotFoundError:
            raise FileNotFoundError(f"Unable to parse input file: {ssf_path}")

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

        # Calculate required fields for each scene
        required_calculated_fields = []
        for scene_definition in scene_definitions:
            required_calculated_fields.append(scene_definition.required_calculated_fields)

        footprint_data._calculate_required_fields(required_calculated_fields)
        footprint_data.identify_scenes(scene_definitions)
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

    def identify_scenes(self, additional_scene_definitions: list[pathlib.Path] | None = None):
        """Calculate scene IDs for all data points.

        This method performs the actual scene identification algorithm on the
        processed footprint data. Currently a placeholder implementation that
        should be updated with the actual scene classification logic.

        additional_scene_definitions : list of pathlib.Path or None, optional
            Additional scene definition CSV files to apply beyond the default
            TRMM and ERBE definitions. Default is None.

        Notes
        -----
        Default scene definitions used:
        - TRMM: Tropical Rainfall Measuring Mission scenes
        - ERBE: Earth Radiation Budget Experiment scenes
        TODO: LIBSDC-674 Add unfiltering scene ID algorithm

        TODO: LIBSDC-589 Implement the scene ID matching algorithm. Scene identification

        The implementation should:
        1. Assign scene IDs to footprint based on variable ranges in scene definitions (default and custom)
        2. Add scene ID variables as columns to self._data
        3. Handle unmatched footprints appropriately
        """
        # Placeholder implementation - to be replaced with actual scene ID logic. Scene Identification will be a
        # standalone function that utilizes this function once input data has been preprocessed into this FootprintData.
        pass

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
        calculated = set(self._data.data_vars.keys())

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
        # Check if all input variables are available
        if all(var in calculated for var in spec.input_vars):
            # Extract input arrays
            inputs = [self._data[var] for var in spec.input_vars]

            self._data[spec.output_var] = xr.apply_ufunc(
                spec.function,
                *inputs,
                output_dtypes=[spec.output_datatype],
            )
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
            self._data[column_name] = self._data[column_name].where(self._data[column_name] < threshold, fill_value)

    @staticmethod
    def _extract_data_from_CeresSSFNOAA20FM6Ed1C(dataset: xr.Dataset) -> xr.Dataset:
        """Extract data from CERES SSF NOAA-20 FM6 Edition 1C NetCDF file.

        Reads specific variables from the hierarchical group structure of CERES SSF (Single Scanner Footprint) files
        and organizes them into a flat xarray Dataset with standardized variable names.

        Parameters
        ----------
        dataset : xr.Dataset
            Open NetCDF dataset in CeresSSFNOAA20FM6Ed1C format

        Notes
        -----
        Data is extracted from NetCDF groups:
        - Surface_Map: Surface type information
        - Cloudy_Footprint_Area: Cloud properties (fraction, phase, optical depth)
        - Full_Footprint_Area: Wind vectors
        - Clear_Footprint_Area: Clear sky coverage

        Array indexing:
        - surface_igbp_type[:,0]: First surface type estimate
        - layers_coverages[:,1] and [:,2]: Lower and upper cloud layers
        - cloud_*[:,0] and [:,1]: Lower and upper cloud layers
        """
        cloud_fraction = np.array(dataset.groups["Cloudy_Footprint_Area"].variables["layers_coverages"])
        igbp_surface_type = np.array(dataset.groups["Surface_Map"].variables["surface_igbp_type"])
        cloud_phase = np.array(dataset.groups["Cloudy_Footprint_Area"].variables["cloud_particle_phase_37um_mean"])
        optical_depth = np.array(dataset.groups["Cloudy_Footprint_Area"].variables["cloud_optical_depth_mean"])
        surface_wind_u = np.array(dataset.groups["Full_Footprint_Area"].variables["surface_wind_u_vector"])
        surface_wind_v = np.array(dataset.groups["Full_Footprint_Area"].variables["surface_wind_v_vector"])
        clear_area = np.array(dataset.groups["Clear_Footprint_Area"].variables["clear_coverage"])

        igbp_surface_type = igbp_surface_type[:, 0]
        cloud_fraction_1 = cloud_fraction[:, 1]
        cloud_fraction_2 = cloud_fraction[:, 2]
        cloud_phase_1 = cloud_phase[:, 0]
        cloud_phase_2 = cloud_phase[:, 1]
        optical_depth_1 = optical_depth[:, 0]
        optical_depth_2 = optical_depth[:, 1]

        parsed_dataset = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: igbp_surface_type,
                FootprintVariables.SURFACE_WIND_U: surface_wind_u,
                FootprintVariables.SURFACE_WIND_V: surface_wind_v,
                FootprintVariables.CLEAR_AREA: clear_area,
                FootprintVariables.OPTICAL_DEPTH_LOWER: optical_depth_1,
                FootprintVariables.OPTICAL_DEPTH_UPPER: optical_depth_2,
                FootprintVariables.CLOUD_FRACTION_LOWER: cloud_fraction_1,
                FootprintVariables.CLOUD_FRACTION_UPPER: cloud_fraction_2,
                FootprintVariables.CLOUD_PHASE_LOWER: cloud_phase_1,
                FootprintVariables.CLOUD_PHASE_UPPER: cloud_phase_2,
            }
        )
        return parsed_dataset
