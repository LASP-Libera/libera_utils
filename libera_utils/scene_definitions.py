import itertools
import logging
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


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
    variable_ranges: dict[str, tuple[float | None, float | None]]

    def __init__(self, scene_id: int, variable_ranges: dict[str, tuple[float | None, float | None]]) -> None:
        self.scene_id = scene_id
        self.variable_ranges = variable_ranges

    def get_bounded_variables(self) -> list[str]:
        """Get list of variables that have at least one defined bound.

        Returns
        -------
        list of str
            Variable names where at least one of (min, max) is not None
        """
        bounded_vars = []
        for var_name, (min_val, max_val) in self.variable_ranges.items():
            if min_val is not None or max_val is not None:
                bounded_vars.append(var_name)
        return bounded_vars

    def matches(self, data_point: dict[str, float]) -> bool:
        """Check if a data point falls within all variable ranges for this scene.

        Only variables with at least one defined bound are checked.
        Variables with both bounds as None (unbounded) are ignored.
        """
        bounded_vars = self.get_bounded_variables()

        for var_name in bounded_vars:
            if var_name not in data_point:
                return False

            value = data_point[var_name]

            if np.isnan(value):
                return False

            min_val, max_val = self.variable_ranges[var_name]

            # Check if value is within range (inclusive on both ends)
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value >= max_val:
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
        """Initialize scene definition from CSV file."""
        self.type = definition_path.stem.lower()

        # Read CSV with scene definitions
        scene_df = pd.read_csv(definition_path, na_values=["", " ", "NaN", "nan", "NULL"])

        # Parse variable names from column headers
        self.required_columns = self._extract_variable_names(scene_df.columns)

        # Create Scene objects for each row
        self.scenes = []
        for _, row in scene_df.iterrows():
            scene_id = int(row["scene_id"])
            variable_ranges = self._parse_row_to_ranges(row, self.required_columns)
            self.scenes.append(Scene(scene_id, variable_ranges))

        # Identify which variables are actually used for classification
        # (have at least one bound defined in at least one scene)
        self.classification_variables = self._identify_classification_variables()

        logger.info(f"Loaded {len(self.scenes)} scenes from {definition_path}")
        logger.info(f"Required variables: {self.required_columns}")
        logger.info(f"Classification variables (with defined bounds): {self.classification_variables}")

        self._validate_scene_definition_file(scene_df)

    def _identify_classification_variables(self) -> list[str]:
        """Identify variables that are actually used for classification.

        A variable is used for classification if at least one scene has at least one defined bound for that variable.

        Returns
        -------
        list of str
            Variables used for classification
        """
        classification_vars = set()

        for scene in self.scenes:
            bounded_vars = scene.get_bounded_variables()
            classification_vars.update(bounded_vars)

        return sorted(list(classification_vars))

    @staticmethod
    def _extract_variable_names(columns: pd.Index) -> list[str]:
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
            if "_id" in col:
                continue

            # Remove _min or _max suffix to get variable name
            if col.endswith("_min"):
                var_name = col[:-4]  # Remove '_min'
                variable_names.add(var_name)
            elif col.endswith("_max"):
                var_name = col[:-4]  # Remove '_max'
                variable_names.add(var_name)

        return sorted(list(variable_names))

    @staticmethod
    def _parse_row_to_ranges(row: pd.Series, variable_names: list[str]) -> dict[str, tuple[float | None, float | None]]:
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

            # Convert NaN to None for unbounded
            min_val = None if pd.isna(min_val) else float(min_val)
            max_val = None if pd.isna(max_val) else float(max_val)

            ranges[var_name] = (min_val, max_val)

        return ranges

    def identify_and_update(self, data: xr.Dataset) -> xr.Dataset:
        """Identify scene IDs for all data points.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing all required variables for scene identification

        Returns
        -------
        xr.Dataset
            Input dataset with scene ID variable added as f"scene_id_{self.type.lower()}"

        """
        self._validate_footprint_data_columns_present(data)

        # Get dimensions and shape
        dims = list(data.dims.keys())
        shape = tuple(data.dims[dim] for dim in dims)

        # Vectorized scene identification with chunking
        scene_ids = self._identify_vectorized(data, shape)

        # Create scene ID variable name
        scene_id_var_name = f"scene_id_{self.type.lower()}"

        # Add scene IDs as a new variable to the dataset
        data[scene_id_var_name] = xr.DataArray(
            scene_ids,
            dims=dims,
            coords={dim: data.coords[dim] for dim in dims if dim in data.coords},
        )
        return data

    def _identify_vectorized(self, data: xr.Dataset, shape: tuple[int, ...]) -> np.ndarray:
        """Vectorized scene identification using numpy arrays."""
        # Initialize scene_ids with zeros
        scene_ids = np.zeros(shape, dtype=np.int32)

        # For each scene, create a mask and assign IDs
        for scene in self.scenes:
            mask = np.ones(shape, dtype=bool)

            bounded_vars = scene.get_bounded_variables()

            for var_name in bounded_vars:
                min_val, max_val = scene.variable_ranges[var_name]
                var_data = data[var_name].values  # Get numpy array from xarray

                is_nan = np.isnan(var_data)
                var_mask = np.ones(shape, dtype=bool)

                if min_val is not None:
                    var_mask &= (var_data >= min_val) | is_nan
                if max_val is not None:
                    var_mask &= (var_data < max_val) | is_nan

                mask &= var_mask

            # Assign scene ID (only if not already assigned)
            scene_ids = np.where((mask) & (scene_ids == 0), scene.scene_id, scene_ids)

        return scene_ids

    def _validate_scene_definition_file(self, scene_df: pd.DataFrame) -> None:
        """Validate scene definition file for complete coverage and no overlaps.

        Ensures that:
        1. Column names follow the expected format (variable_min, variable_max pairs)
        2. Min values are less than or equal to max values for all bins
        3. Every possible combination of variable values maps to exactly one scene ID
        4. There are no gaps in coverage (all value combinations are classified)
        5. There are no overlaps (no ambiguous classifications)

        Parameters
        ----------
        scene_df : pd.DataFrame
            DataFrame loaded from scene definition CSV with columns:
            scene_id, var1_min, var1_max, var2_min, var2_max, ...
        required_columns : list of str
            List of variable names that should have min/max pairs

        Raises
        ------
        ValueError
            If any validation check fails, with detailed description of the issue

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'scene_id': [1, 2],
        ...     'temp_min': [0, 50],
        ...     'temp_max': [50, 100],
        ...     'pressure_min': [900, 900],
        ...     'pressure_max': [1100, 1100]
        ... })
        >>> validate_scene_definition_file(df, ['temp', 'pressure'])
        # Passes validation

        >>> df = pd.DataFrame({
        ...     'scene_id': [1, 2],
        ...     'temp_min': [0, 40],  # Overlap at 40-50
        ...     'temp_max': [50, 100],
        ...     'pressure_min': [900, 900],
        ...     'pressure_max': [1100, 1100]
        ... })
        >>> validate_scene_definition_file(df, ['temp', 'pressure'])
        ValueError: Overlapping scenes detected...
        """

        # 1. Validate column naming
        self._validate_column_name_format(scene_df)

        # 2. Validate min/max ordering
        self._validate_min_max_ordering(scene_df)

        # 3. Validate scene_id column
        self._validate_scene_ids(scene_df)

        # 5. Check for overlaps using exact geometric intersection
        self._validate_no_overlaps()

        # 6. Check for complete coverage using space decomposition
        self._validate_complete_coverage()

    def _validate_footprint_data_columns_present(self, data: xr.Dataset):
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

    def _validate_column_name_format(self, scene_df: pd.DataFrame) -> None:
        """Validate that all required variable columns exist with _min and _max suffixes.

        Parameters
        ----------
        scene_df : pd.DataFrame
            Scene definition DataFrame
        required_columns : list of str
            List of variable names that should have min/max pairs

        Raises
        ------
        ValueError
            If scene_id column is missing or if any required variable is missing
            its _min or _max column
        """
        if "scene_id" not in scene_df.columns:
            raise ValueError("Scene definition file must contain a 'scene_id' column")

        missing_columns = []
        for var_name in self.required_columns:
            min_col = f"{var_name}_min"
            max_col = f"{var_name}_max"

            if min_col not in scene_df.columns:
                missing_columns.append(min_col)
            if max_col not in scene_df.columns:
                missing_columns.append(max_col)

        if missing_columns:
            raise ValueError(f"Scene definition file is missing required columns: {missing_columns}")

    def _validate_min_max_ordering(self, scene_df: pd.DataFrame) -> None:
        """Validate that min values are less than or equal to max values for all bins.

        Parameters
        ----------
        scene_df : pd.DataFrame
            Scene definition DataFrame
        required_columns : list of str
            List of variable names to check

        Raises
        ------
        ValueError
            If any bin has min > max for any variable
        """
        invalid_bins = []

        for idx, row in scene_df.iterrows():
            scene_id = row["scene_id"]

            for var_name in self.required_columns:
                min_val = row[f"{var_name}_min"]
                max_val = row[f"{var_name}_max"]

                # Skip if either value is NaN (unbounded)
                if pd.isna(min_val) or pd.isna(max_val):
                    continue

                if min_val > max_val:
                    invalid_bins.append(f"Scene {scene_id}: {var_name}_min ({min_val}) > {var_name}_max ({max_val})")

        if invalid_bins:
            raise ValueError(f"Invalid bin definitions (min > max):\n" + "\n".join(invalid_bins))

    @staticmethod
    def _validate_scene_ids(scene_df: pd.DataFrame) -> None:
        """Validate scene_id column contains unique integer values.

        Parameters
        ----------
        scene_df : pd.DataFrame
            Scene definition DataFrame

        Raises
        ------
        ValueError
            If scene IDs are not unique or not integer-convertible
        """
        if scene_df["scene_id"].duplicated().any():
            duplicates = scene_df[scene_df["scene_id"].duplicated()]["scene_id"].tolist()
            raise ValueError(f"Duplicate scene IDs found: {duplicates}")

        try:
            scene_df["scene_id"].astype(int)
        except (ValueError, TypeError):
            raise ValueError("Scene IDs must be integer values")

    def _validate_no_overlaps(self) -> None:
        """Validate that no two scenes in the scene definition overlap.

        Raises
        ------
        ValueError
            If any two scenes overlap, with details about the overlapping region

        Notes
        -----
        This handles unbounded ranges in the following ways:
        - None for min means -∞ (always overlaps with any max)
        - None for max means +∞ (always overlaps with any min)
        """
        overlaps = []

        for i in range(len(self.scenes)):
            for j in range(i + 1, len(self.scenes)):
                rect1 = self.scenes[i]
                rect2 = self.scenes[j]

                overlap_info = self._compute_intersection(rect1, rect2, self.classification_variables)

                if overlap_info is not None:
                    overlaps.append((rect1.scene_id, rect2.scene_id, overlap_info))

        if overlaps:
            overlap_msg = "Overlapping scenes detected:\n"
            for scene_id1, scene_id2, overlap_region in overlaps:
                overlap_msg += f"\n  Scenes {scene_id1} and {scene_id2} overlap in region:\n"
                for var_name, (min_val, max_val) in overlap_region.items():
                    min_str = f"{min_val:.6g}" if min_val is not None else "-∞"
                    max_str = f"{max_val:.6g}" if max_val is not None else "+∞"
                    overlap_msg += f"    {var_name}: [{min_str}, {max_str}]\n"

            raise ValueError(overlap_msg)

    def _validate_complete_coverage(self) -> None:
        """Validate that scenes completely cover the bounded parameter space."""
        # Determine the global bounded region for classification variables only
        global_bounds = self._compute_global_bounds(self.classification_variables)

        # Check if classification space is fully bounded
        unbounded_vars = []
        for var_name, (min_val, max_val) in global_bounds.items():
            if min_val is None or max_val is None:
                unbounded_vars.append(var_name)

        if unbounded_vars:
            logger.warning(
                f"Warning: Cannot verify complete coverage - unbounded classification variables: {unbounded_vars}"
            )
            return

        gaps = self._find_gaps(self.scenes, global_bounds, self.classification_variables)

        if gaps:
            gap_msg = "Incomplete coverage detected. The following regions are not covered by any scene:\n"
            for gap_region in gaps[:10]:
                gap_msg += "\n  Uncovered region:\n"
                for var_name, (min_val, max_val) in gap_region.items():
                    gap_msg += f"    {var_name}: [{min_val:.6g}, {max_val:.6g}]\n"

            if len(gaps) > 10:
                gap_msg += f"\n  ... and {len(gaps) - 10} more uncovered regions\n"

            raise ValueError(gap_msg)

    @staticmethod
    def _find_gaps(
        scenes: list[Scene], global_bounds: dict[str, tuple[float, float]], variables: list[str]
    ) -> list[dict[str, tuple[float, float]]]:
        """Find gaps in property ranges defined in scenes.

        Parameters
        ----------
        scenes : list of Scene
            List of scene regions
        global_bounds : dict
            Global bounding box
        variables : list of str
            List of variable names

        Returns
        -------
        list of dict
            List of uncovered regions (gaps)
        """
        boundary_values = {}

        for var_name in variables:
            boundaries = set()
            global_min, global_max = global_bounds[var_name]

            boundaries.add(global_min)
            boundaries.add(global_max)

            for rect in scenes:
                if var_name not in rect.variable_ranges:
                    continue

                scene_min, scene_max = rect.variable_ranges[var_name]

                if scene_min is not None and global_min <= scene_min <= global_max:
                    boundaries.add(scene_min)
                if scene_max is not None and global_min <= scene_max <= global_max:
                    boundaries.add(scene_max)

            boundary_values[var_name] = sorted(list(boundaries))

        gaps = []
        cells = SceneDefinition._generate_cells_from_boundaries(boundary_values, variables)

        for cell in cells:
            center_point = {}
            for var_name in variables:
                min_val, max_val = cell[var_name]
                center_point[var_name] = (min_val + max_val) / 2.0

            covered = False
            for rect in scenes:
                if SceneDefinition._point_in_scene(center_point, rect, variables):
                    covered = True
                    break

            if not covered:
                gaps.append(cell)

        return gaps

    @staticmethod
    def _generate_cells_from_boundaries(
        boundary_values: dict[str, list[float]], variables: list[str]
    ) -> list[dict[str, tuple[float, float]]]:
        """Generate all cells (hyper-rectangles) from boundary values.

        Parameters
        ----------
        boundary_values : dict
            For each variable, a sorted list of boundary values
        variables : list of str
            List of variable names

        Returns
        -------
        list of dict
            List of cells, where each cell is a dict mapping variable names
            to (min, max) tuples
        """

        # For each variable, create intervals between consecutive boundaries
        intervals_per_var = {}
        for var_name in variables:
            boundaries = boundary_values[var_name]
            intervals = []
            for i in range(len(boundaries) - 1):
                intervals.append((boundaries[i], boundaries[i + 1]))
            intervals_per_var[var_name] = intervals

        # Generate all combinations of intervals (Cartesian product)
        var_names = sorted(variables)
        interval_lists = [intervals_per_var[var_name] for var_name in var_names]

        cells = []
        for interval_combo in itertools.product(*interval_lists):
            cell = {}
            for var_name, interval in zip(var_names, interval_combo):
                cell[var_name] = interval
            cells.append(cell)

        return cells

    @staticmethod
    def _point_in_scene(point: dict[str, float], scene, variables: list[str]) -> bool:
        """Check if a point falls within a scene's bounds.

        Parameters
        ----------
        point : dict
            Point coordinates
        scene : Scene
            Scene to test
        variables : list of str
            List of variable names

        Returns
        -------
        bool
            True if point is in scene, False otherwise
        """
        for var_name in variables:
            if var_name not in scene.variable_ranges:
                continue

            scene_min, scene_max = scene.variable_ranges[var_name]

            # Skip unbounded variables
            if scene_min is None and scene_max is None:
                continue

            if var_name not in point:
                return False

            point_val = point[var_name]

            # Check bounds (min inclusive, max exclusive)
            if scene_min is not None and point_val < scene_min:
                return False
            if scene_max is not None and point_val >= scene_max:
                return False

        return True

    def _compute_global_bounds(self, variables: list[str]) -> dict[str, tuple[float | None, float | None]]:
        """Compute the global bounding box that contains all scenes.

        Parameters
        ----------
        variables : list of str
            List of variable names

        Returns
        -------
        dict of str to tuple of (float or None, float or None)
            Global bounds for each variable

        Notes
        -----
        Global min is the minimum of all scene mins (excluding None/unbounded).
        Global max is the maximum of all scene maxs (excluding None/unbounded).
        If all scenes are unbounded in a direction, returns None for that bound.
        """
        global_bounds = {}

        for var_name in variables:
            all_mins = []
            all_maxs = []

            for rect in self.scenes:
                if var_name not in rect.variable_ranges:
                    continue

                min_val, max_val = rect.variable_ranges[var_name]
                if min_val is not None:
                    all_mins.append(min_val)
                if max_val is not None:
                    all_maxs.append(max_val)

            global_min = min(all_mins) if all_mins else None
            global_max = max(all_maxs) if all_maxs else None

            global_bounds[var_name] = (global_min, global_max)

        return global_bounds

    @staticmethod
    def _compute_intersection(
        rect1: Scene, rect2: Scene, variables: list[str]
    ) -> dict[str, tuple[float | None, float | None]] | None:
        """Compute the intersection of two hyper-rectangles.

        Parameters
        ----------
        rect1, rect2 : Scene
            Rectangles to intersect
        variables : list of str
            List of variable names

        Returns
        -------
        dict or None
            Dictionary of variable bounds for the intersection region, or None if
            the rectangles don't intersect

        Notes
        -----
        For each dimension, computes the intersection of two intervals:
        - Intersection of [a1, b1] and [a2, b2] is [max(a1, a2), min(b1, b2)]
        - Intersection exists only if max(a1, a2) < min(b1, b2)
        - Special handling for unbounded (None) values
        """
        intersection = {}

        for var_name in variables:
            if var_name not in rect1.variable_ranges or var_name not in rect2.variable_ranges:
                continue

            min1, max1 = rect1.variable_ranges[var_name]
            min2, max2 = rect2.variable_ranges[var_name]

            # Skip if both scenes have this variable completely unbounded
            if (min1 is None and max1 is None) or (min2 is None and max2 is None):
                continue

            # Compute intersection bounds for this dimension
            if min1 is None and min2 is None:
                intersect_min = None
            elif min1 is None:
                intersect_min = min2
            elif min2 is None:
                intersect_min = min1
            else:
                intersect_min = max(min1, min2)

            if max1 is None and max2 is None:
                intersect_max = None
            elif max1 is None:
                intersect_max = max2
            elif max2 is None:
                intersect_max = max1
            else:
                intersect_max = min(max1, max2)

            # Check if this dimension has a valid intersection
            if intersect_min is not None and intersect_max is not None:
                if intersect_min >= intersect_max:
                    # No overlap in this dimension means no overlap overall
                    return None

            intersection[var_name] = (intersect_min, intersect_max)

        # If no bounded variables to check, no meaningful intersection
        if not intersection:
            return None

        return intersection
