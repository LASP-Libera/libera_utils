"""Unit tests for scene identification module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from libera_utils.scene_id import (
    _CALCULATED_VARIABLE_MAP,
    CalculationSpec,
    FootprintData,
    FootprintVariables,
    IGBPSurfaceType,
    Scene,
    SceneDefinition,
    TRMMSurfaceType,
    calculate_cloud_fraction,
    calculate_cloud_fraction_weighted_optical_depth,
    calculate_cloud_fraction_weighted_property_for_layer,
    calculate_cloud_phase,
    calculate_surface_wind,
    calculate_trmm_surface_type,
)


class TestIGBPSurfaceType:
    """Test cases for IGBPSurfaceType enum."""

    def test_surface_type_values(self):
        """Test that IGBPSurfaceType enum has correct values."""
        assert IGBPSurfaceType.EVERGREEN_NEEDLELEAF_FOREST.value == 1
        assert IGBPSurfaceType.SEA_ICE.value == 20

    def test_trmm_property_mapping(self):
        """Test that TRMM to IGBP mapping is correct."""
        assert IGBPSurfaceType.EVERGREEN_NEEDLELEAF_FOREST.trmm_surface_type == TRMMSurfaceType.HI_SHRUB
        assert IGBPSurfaceType.WATER_BODIES.trmm_surface_type == TRMMSurfaceType.OCEAN
        assert IGBPSurfaceType.PERMANENT_SNOW_ICE.trmm_surface_type == TRMMSurfaceType.SNOW
        assert IGBPSurfaceType.BARE_SOIL_ROCKS.trmm_surface_type == TRMMSurfaceType.BRIGHT_DESERT
        assert IGBPSurfaceType.OPEN_SHRUBLANDS.trmm_surface_type == TRMMSurfaceType.DARK_DESERT


class TestTRMMSurfaceType:
    """Test cases for TRMMSurfaceType enum."""

    def test_trmm_surface_type_values(self):
        """Test that TRMMSurfaceType enum has correct values."""
        assert TRMMSurfaceType.OCEAN.value == 0
        assert TRMMSurfaceType.HI_SHRUB.value == 1
        assert TRMMSurfaceType.LOW_SHRUB.value == 2
        assert TRMMSurfaceType.DARK_DESERT.value == 3
        assert TRMMSurfaceType.BRIGHT_DESERT.value == 4
        assert TRMMSurfaceType.SNOW.value == 5


class TestCalculationSpec:
    """Test cases for CalculationSpec dataclass."""

    def test_calculation_spec_creation(self):
        """Test creating a CalculationSpec."""
        spec = CalculationSpec(output_var="foo", function=lambda x: x * 2, input_vars=["var1"], output_datatype=float)
        assert spec.output_var == "foo"
        assert callable(spec.function)
        assert spec.input_vars == ["var1"]
        assert spec.output_datatype is float

    def test_calculation_spec_with_dependencies(self):
        """Test creating a CalculationSpec with dependencies."""
        spec = CalculationSpec(
            output_var="foo",
            function=lambda x: x * 2,
            input_vars=["var1"],
            output_datatype=float,
            dependent_calculations=["var2"],
        )
        assert spec.dependent_calculations == ["var2"]


class TestCalculatedVariableMap:
    """Test cases for _CALCULATED_VARIABLE_MAP."""

    def test_map_specs_have_correct_structure(self):
        """Test that each spec has required attributes."""
        for var_name, spec in _CALCULATED_VARIABLE_MAP.items():
            assert isinstance(spec, CalculationSpec)
            assert callable(spec.function)
            assert isinstance(spec.input_vars, list)
            assert len(spec.input_vars) > 0
            assert spec.output_datatype in (float, int)

    def test_map_contains_expected_variables(self):
        """Test that map contains all expected calculated variables."""
        expected_vars = [
            FootprintVariables.CLOUD_FRACTION,
            FootprintVariables.SURFACE_WIND,
            FootprintVariables.SURFACE_TYPE,
            FootprintVariables.OPTICAL_DEPTH,
            FootprintVariables.CLOUD_PHASE,
        ]
        for var in expected_vars:
            assert var in _CALCULATED_VARIABLE_MAP


class TestFootprintDataCalculations:
    """Test cases for FootprintData static calculation methods."""

    def test_calculate_cloud_fraction_calculation(self):
        """Test cloud fraction calculation."""
        assert calculate_cloud_fraction(60.0) == 40.0
        assert calculate_cloud_fraction(100.0) == 0.0
        assert calculate_cloud_fraction(0.0) == 100.0

    def test_calculate_cloud_fraction_array(self):
        """Test cloud fraction with array input."""
        input_arr = np.array([0.0, 50.0, 100.0])
        expected = np.array([100.0, 50.0, 0.0])
        result = calculate_cloud_fraction(input_arr)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_surface_wind_calculation(self):
        """Test surface wind calculation."""
        assert calculate_surface_wind(3.0, 4.0) == 5.0
        assert calculate_surface_wind(3.0, -4.0) == 5.0
        assert calculate_surface_wind(0.0, 0.0) == 0.0

    def test_calculate_surface_wind_with_nan(self):
        """Test surface wind calculation with NaN values."""
        assert np.isnan(calculate_surface_wind(np.nan, 4.0))
        assert np.isnan(calculate_surface_wind(3.0, np.nan))
        assert np.isnan(calculate_surface_wind(np.nan, np.nan))

    def test_calculate_surface_wind_array(self):
        """Test surface wind with array inputs."""
        u = np.array([3.0, 0.0, np.nan])
        v = np.array([4.0, 5.0, 3.0])
        result = calculate_surface_wind(u, v)

        assert result[0] == 5.0
        assert result[1] == 5.0
        assert np.isnan(result[2])

    def test_calculate_cloud_fraction_weighted_optical_depth_calculation(self):
        """Test optical depth calculation."""
        result = calculate_cloud_fraction_weighted_optical_depth(10.0, 8.0, 25.0, 15.0, 40.0)
        expected = (10.0 * 25.0 + 8.0 * 15.0) / 40.0
        assert abs(result - expected) < 1e-10

    def test_calculate_cloud_fraction_weighted_optical_depth_with_zero_cloud_fraction(self):
        """Test optical depth calculation with zero total cloud fraction."""
        result = calculate_cloud_fraction_weighted_optical_depth(10.0, 8.0, 0.0, 0.0, 0.0)
        assert np.isnan(result)

    def test_calculate_cloud_fraction_weighted_optical_depth_with_nan_values(self):
        """Test optical depth calculation with NaN values."""
        result = calculate_cloud_fraction_weighted_optical_depth(np.nan, 8.0, 25.0, 15.0, 40.0)
        expected = 8.0 * 15.0 / 40.0  # Only upper layer contributes
        assert abs(result - expected) < 1e-10

    def test_calculate_cloud_phase_calculation(self):
        """Test cloud phase calculation."""
        result = calculate_cloud_phase(1.0, 2.0, 30.0, 10.0, 40.0)
        expected = round((1.0 * 30.0 + 2.0 * 10.0) / 40.0)
        assert result == expected

    def test_calculate_cloud_phase_with_nan_values(self):
        """Test cloud phase calculation with NaN values."""
        result = calculate_cloud_phase(np.nan, np.nan, 30.0, 10.0, 40.0)
        assert np.isnan(result)

    def test_calculate_cloud_phase_rounding(self):
        """Test that cloud phase is rounded to nearest integer."""
        # Test rounding down
        result = calculate_cloud_phase(1.0, 2.0, 60.0, 10.0, 70.0)
        # (1.0 * 60 + 2.0 * 10) / 70 = 80/70 = 1.14... rounds to 1
        assert result == 1.0

        # Test rounding up
        result = calculate_cloud_phase(1.0, 2.0, 20.0, 50.0, 70.0)
        # (1.0 * 20 + 2.0 * 50) / 70 = 120/70 = 1.71... rounds to 2
        assert result == 2.0

    def test_calculate_cloud_fraction_weighted_property_for_layer(self):
        """Test the generic weighted property calculation."""
        result = calculate_cloud_fraction_weighted_property_for_layer(
            np.array([10.0, 20.0]),
            np.array([15.0, 25.0]),
            np.array([30.0, 40.0]),
            np.array([20.0, 10.0]),
            np.array([50.0, 50.0]),
        )

        expected = np.array(
            [
                (10.0 * 30.0 + 15.0 * 20.0) / 50.0,
                (20.0 * 40.0 + 25.0 * 10.0) / 50.0,
            ]
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_cloud_fraction_weighted_property_with_zero_total(self):
        """Test weighted property calculation with zero total cloud fraction."""
        result = calculate_cloud_fraction_weighted_property_for_layer(
            np.array([10.0]), np.array([15.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
        )
        assert np.isnan(result[0])

    def test_weighted_property_partial_nan(self):
        """Test weighted property with some NaN values."""
        property_lower = np.array([10.0, np.nan, 5.0])
        property_upper = np.array([np.nan, 20.0, np.nan])
        cf_lower = np.array([30.0, 40.0, 50.0])
        cf_upper = np.array([20.0, 10.0, 0.0])
        cf_total = np.array([50.0, 50.0, 50.0])

        result = calculate_cloud_fraction_weighted_property_for_layer(
            property_lower, property_upper, cf_lower, cf_upper, cf_total
        )

        # First: only lower contributes (upper is NaN)
        expected_0 = 10.0 * 30.0 / 50.0
        assert abs(result[0] - expected_0) < 1e-10

        # Second: only upper contributes (lower is NaN)
        expected_1 = 20.0 * 10.0 / 50.0
        assert abs(result[1] - expected_1) < 1e-10

        # Third: only lower contributes (upper is NaN and cf_upper is 0)
        expected_2 = 5.0 * 50.0 / 50.0
        assert abs(result[2] - expected_2) < 1e-10


class TestFootprintDataInstanceMethods:
    """Test cases for FootprintData instance methods."""

    @pytest.fixture
    def footprint_with_data(self):
        """Create FootprintData instance with sample data."""
        data = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0, 80.0, 20.0]),
                FootprintVariables.SURFACE_WIND_U: (["record"], [3.0, 4.0, 0.0]),
                FootprintVariables.SURFACE_WIND_V: (["record"], [4.0, 3.0, 5.0]),
            }
        )
        return FootprintData(data)

    def test_footprint_data_initialization(self):
        """Test FootprintData initialization."""
        data = xr.Dataset({"test_var": (["record"], [1.0, 2.0, 3.0])})
        fp = FootprintData(data)
        assert fp._data is not None
        assert "test_var" in fp._data

    def test_separate_instances_are_independent(self):
        """Test that modifying _data on one instance doesn't affect another."""
        data1 = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0, 80.0, 20.0]),
                FootprintVariables.SURFACE_WIND_U: (["record"], [3.0, 4.0, 0.0]),
            }
        )
        fp1 = FootprintData(data1)

        data2 = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0, 80.0, 20.0]),
                FootprintVariables.SURFACE_WIND_U: (["record"], [3.0, 4.0, 0.0]),
            }
        )
        fp2 = FootprintData(data2)

        # Verify initial state is the same
        np.testing.assert_array_equal(
            fp1._data[FootprintVariables.CLEAR_AREA].values, fp2._data[FootprintVariables.CLEAR_AREA].values
        )

        # Modify first instance
        fp1._data[FootprintVariables.CLEAR_AREA].values[0] = 999.0

        # Verify second instance is unchanged
        assert fp2._data[FootprintVariables.CLEAR_AREA].values[0] == 60.0
        assert fp1._data[FootprintVariables.CLEAR_AREA].values[0] == 999.0

    def test_convert_missing_values_with_nan(self, footprint_with_data):
        """Test converting missing values when input missing value is NaN."""
        footprint_with_data._data = xr.Dataset({"test_var": (["record"], [1.0, np.nan, 3.0])})
        footprint_with_data._convert_missing_values(np.nan)

        assert footprint_with_data._data["test_var"][0] == 1.0
        assert np.isnan(footprint_with_data._data["test_var"][1])
        assert footprint_with_data._data["test_var"][2] == 3.0

    def test_convert_missing_values_with_numeric(self, footprint_with_data):
        """Test converting missing values when input missing value is numeric."""
        footprint_with_data._data = xr.Dataset({"test_var": (["record"], [1.0, -999.0, 3.0])})
        footprint_with_data._convert_missing_values(-999.0)

        assert footprint_with_data._data["test_var"][0] == 1.0
        assert np.isnan(footprint_with_data._data["test_var"][1])
        assert footprint_with_data._data["test_var"][2] == 3.0

    def test_fill_column_above_max_value(self, footprint_with_data):
        """Test filling values above threshold."""
        footprint_with_data._data = xr.Dataset({"col1": (["x"], [1.0, 50.0, 100.0, 200.0])})
        footprint_with_data._fill_column_above_max_value("col1", 100.0)

        assert footprint_with_data._data["col1"][0] == 1.0
        assert footprint_with_data._data["col1"][1] == 50.0
        # Values >= 100 should be replaced (note: < threshold means keep if < 100)
        assert np.isnan(footprint_with_data._data["col1"][2])
        assert np.isnan(footprint_with_data._data["col1"][3])

    def test_fill_column_above_max_value_custom_fill(self, footprint_with_data):
        """Test filling with custom fill value."""
        footprint_with_data._data = xr.Dataset({"col1": (["x"], [1.0, 50.0, 150.0])})
        footprint_with_data._fill_column_above_max_value("col1", 100.0, fill_value=-999.0)

        assert footprint_with_data._data["col1"][2] == -999.0

    def test_fill_column_missing_column_error(self, footprint_with_data):
        """Test error when column not found."""
        with pytest.raises(ValueError, match="Column nonexistent not found"):
            footprint_with_data._fill_column_above_max_value("nonexistent", 100.0)

    def test_calculate_required_fields_single_field(self, footprint_with_data):
        """Test calculating a single field."""
        footprint_with_data._calculate_required_fields([FootprintVariables.CLOUD_FRACTION])

        assert FootprintVariables.CLOUD_FRACTION in footprint_with_data._data.data_vars
        np.testing.assert_array_equal(
            footprint_with_data._data[FootprintVariables.CLOUD_FRACTION].values, [40.0, 20.0, 80.0]
        )

    def test_calculate_required_fields_multiple_fields(self, footprint_with_data):
        """Test calculating multiple fields."""
        footprint_with_data._calculate_required_fields(
            [FootprintVariables.CLOUD_FRACTION, FootprintVariables.SURFACE_WIND]
        )

        assert FootprintVariables.CLOUD_FRACTION in footprint_with_data._data.data_vars
        assert FootprintVariables.SURFACE_WIND in footprint_with_data._data.data_vars

    def test_calculate_required_fields_with_dependencies(self):
        """Test calculating field that depends on another calculated field."""
        data = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["record"], [10.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["record"], [5.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["record"], [25.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["record"], [15.0]),
            }
        )
        fp = FootprintData(data)

        fp._calculate_required_fields([FootprintVariables.OPTICAL_DEPTH])

        # Should calculate CLOUD_FRACTION first, then OPTICAL_DEPTH
        assert FootprintVariables.CLOUD_FRACTION in fp._data.data_vars
        assert FootprintVariables.OPTICAL_DEPTH in fp._data.data_vars

    def test_calculate_required_fields_unknown_field_error(self, footprint_with_data):
        """Test error when requesting unknown calculated field."""
        with pytest.raises(ValueError, match="Unknown calculated field"):
            footprint_with_data._calculate_required_fields(["unknown_field"])

    def test_calculate_required_fields_missing_dependencies_error(self):
        """Test error when dependencies are missing."""
        data = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0]),
                # Missing OPTICAL_DEPTH_LOWER, etc.
            }
        )
        fp = FootprintData(data)

        with pytest.raises(ValueError, match="Cannot calculate fields"):
            fp._calculate_required_fields([FootprintVariables.OPTICAL_DEPTH])

    def test_identify_scenes_placeholder(self, footprint_with_data):
        """Test that identify_scenes exists and accepts parameters."""
        # Should not raise any errors - it's currently a placeholder
        footprint_with_data.identify_scenes()
        footprint_with_data.identify_scenes(additional_scene_definitions=[Path("test.csv")])


class TestFootprintDataExtraction:
    """Test cases for data extraction methods."""

    @pytest.fixture
    def mock_netcdf_dataset(self):
        """Create a mock NetCDF dataset."""
        mock_dataset = MagicMock()

        # Mock the groups and variables structure
        mock_dataset.groups = {
            "Cloudy_Footprint_Area": MagicMock(),
            "Surface_Map": MagicMock(),
            "Full_Footprint_Area": MagicMock(),
            "Clear_Footprint_Area": MagicMock(),
        }

        # Mock the variables with numpy arrays
        mock_dataset.groups["Cloudy_Footprint_Area"].variables = {
            "layers_coverages": np.random.rand(100, 3),
            "cloud_particle_phase_37um_mean": np.random.rand(100, 2),
            "cloud_optical_depth_mean": np.random.rand(100, 2),
        }

        mock_dataset.groups["Surface_Map"].variables = {
            "surface_igbp_type": np.random.randint(1, 20, size=(100, 2)),
        }

        mock_dataset.groups["Full_Footprint_Area"].variables = {
            "surface_wind_u_vector": np.random.rand(100),
            "surface_wind_v_vector": np.random.rand(100),
        }

        mock_dataset.groups["Clear_Footprint_Area"].variables = {
            "clear_coverage": np.random.rand(100) * 100,
        }

        return mock_dataset

    def test_extract_data_from_ceres_ssf(self, mock_netcdf_dataset):
        """Test extracting data from CERES SSF format."""
        result = FootprintData._extract_data_from_CeresSSFNOAA20FM6Ed1C(mock_netcdf_dataset)

        # Verify all required variables are present in _dataset
        required_vars = [
            FootprintVariables.IGBP_SURFACE_TYPE,
            FootprintVariables.SURFACE_WIND_U,
            FootprintVariables.SURFACE_WIND_V,
            FootprintVariables.CLEAR_AREA,
            FootprintVariables.OPTICAL_DEPTH_LOWER,
            FootprintVariables.OPTICAL_DEPTH_UPPER,
            FootprintVariables.CLOUD_FRACTION_LOWER,
            FootprintVariables.CLOUD_FRACTION_UPPER,
            FootprintVariables.CLOUD_PHASE_LOWER,
            FootprintVariables.CLOUD_PHASE_UPPER,
        ]

        for var in required_vars:
            assert var in result

    def test_extract_data_sets_dataset_attribute(self, mock_netcdf_dataset):
        """Test that extraction sets the _dataset attribute."""
        data = xr.Dataset()
        footprint = FootprintData(data)
        footprint._extract_data_from_CeresSSFNOAA20FM6Ed1C(mock_netcdf_dataset)

        assert hasattr(footprint, "_data")
        assert isinstance(footprint._data, xr.Dataset)


class TestProcessSSFAndCamera:
    """Test cases for class methods."""

    def test_process_cldpx_not_implemented(self):
        """Test that cldpx processing raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="cldpx/viirs/geos"):
            FootprintData.from_cldpx_viirs_geos_cam_groundscene()

    def test_from_clouds_not_implemented(self):
        """Test that clouds processing raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="clouds/ground scene"):
            FootprintData.from_clouds_groundscene()


class TestScene:
    """Test cases for Scene dataclass."""

    def test_scene_creation(self):
        """Test creating a Scene."""
        scene = Scene(
            scene_id=1,
            variable_ranges={
                "var1": (0.0, 10.0),
                "var2": (20.0, 30.0),
            },
        )
        assert scene.scene_id == 1
        assert len(scene.variable_ranges) == 2

    def test_scene_matches_within_range(self):
        """Test scene matching with values within range."""
        scene = Scene(
            scene_id=1,
            variable_ranges={
                "var1": (0.0, 10.0),
                "var2": (20.0, 30.0),
            },
        )

        data_point = {"var1": 5.0, "var2": 25.0}
        assert scene.matches(data_point) is True

    def test_scene_matches_boundary_values(self):
        """Test scene matching with boundary values (inclusive)."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0)})

        assert scene.matches({"var1": 0.0}) is True  # Lower boundary
        assert scene.matches({"var1": 10.0}) is True  # Upper boundary

    def test_scene_no_match_outside_range(self):
        """Test scene not matching when value outside range."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0)})

        assert scene.matches({"var1": -1.0}) is False
        assert scene.matches({"var1": 11.0}) is False

    def test_scene_no_match_with_nan(self):
        """Test scene not matching with NaN value."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0)})

        assert scene.matches({"var1": np.nan}) is False

    def test_scene_no_match_missing_variable(self):
        """Test scene not matching when required variable missing."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0), "var2": (0.0, 20.0)})

        assert scene.matches({"var1": 5.0}) is False  # Missing var2

    def test_scene_matches_unbounded_min(self):
        """Test scene matching with unbounded minimum."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (None, 10.0)})

        assert scene.matches({"var1": -1000.0}) is True
        assert scene.matches({"var1": 11.0}) is False

    def test_scene_matches_unbounded_max(self):
        """Test scene matching with unbounded maximum."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, None)})

        assert scene.matches({"var1": 1000.0}) is True
        assert scene.matches({"var1": -1.0}) is False

    def test_scene_matches_unbounded_both(self):
        """Test scene matching with no bounds."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (None, None)})

        assert scene.matches({"var1": 0.0}) is True
        assert scene.matches({"var1": -1000.0}) is True
        assert scene.matches({"var1": 1000.0}) is True


class TestSceneDefinition:
    """Test cases for SceneDefinition class."""

    @pytest.fixture
    def sample_csv(self):
        """Create a temporary CSV file for testing."""
        csv_content = """scene_id,surface_type_min,surface_type_max,cloud_fraction_min,cloud_fraction_max
1,0,0,0,10
2,0,0,10,50
3,1,2,0,10
4,1,2,50,100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        yield temp_path
        temp_path.unlink()

    def test_scene_definition_initialization(self, sample_csv):
        """Test initializing SceneDefinition from CSV."""
        scene_def = SceneDefinition(sample_csv)

        assert scene_def.type == sample_csv.stem.upper()
        assert len(scene_def.scenes) == 4
        assert len(scene_def.required_columns) == 2

    def test_extract_variable_names(self, sample_csv):
        """Test extracting variable names from CSV columns."""
        scene_def = SceneDefinition(sample_csv)

        assert "surface_type" in scene_def.required_columns
        assert "cloud_fraction" in scene_def.required_columns
        assert "scene_id" not in scene_def.required_columns

    def test_parse_row_to_ranges(self, sample_csv):
        """Test parsing CSV row to variable ranges."""
        scene_def = SceneDefinition(sample_csv)

        # Check first scene
        first_scene = scene_def.scenes[0]
        assert first_scene.scene_id == 1
        assert first_scene.variable_ranges["surface_type"] == (0.0, 0.0)
        assert first_scene.variable_ranges["cloud_fraction"] == (0.0, 10.0)

    def test_parse_row_with_nan_creates_none(self):
        """Test that NaN values in CSV become None (unbounded)."""
        csv_content = """scene_id,var1_min,var1_max
1,,10.0
2,5.0,
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            scene_def = SceneDefinition(temp_path)

            assert scene_def.scenes[0].variable_ranges["var1"] == (None, 10.0)
            assert scene_def.scenes[1].variable_ranges["var1"] == (5.0, None)
        finally:
            temp_path.unlink()

    def test_validate_input_data_columns_with_valid_data(self, sample_csv):
        """Test validation with valid data."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["record"], [0, 1, 2]),
                "cloud_fraction": (["record"], [5.0, 25.0, 75.0]),
            }
        )

        # Should not raise
        scene_def.validate_input_data_columns(data)

    def test_validate_input_data_columns_with_missing_columns_error(self, sample_csv):
        """Test validation error with missing columns."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["record"], [0, 1, 2]),
                # Missing cloud_fraction
            }
        )

        with pytest.raises(ValueError, match="Required columns .* not in input data"):
            scene_def.validate_input_data_columns(data)

    def test_validate_scene_definition_file_not_implemented(self, sample_csv):
        """Test that validate_scene_definition_file raises NotImplementedError."""
        scene_def = SceneDefinition(sample_csv)

        with pytest.raises(NotImplementedError):
            scene_def.validate_scene_definition_file()

    def test_identify_simple_matching(self, sample_csv):
        """Test identifying scene IDs with simple data."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["record"], [0, 0, 1]),
                "cloud_fraction": (["record"], [5.0, 25.0, 60.0]),
            }
        )

        result = scene_def.identify(data)

        assert result[0] == 1  # Scene 1: type=0, cf=0-10
        assert result[1] == 2  # Scene 2: type=0, cf=10-50
        assert result[2] == 4  # Scene 4: type=1-2, cf=50-100

    def test_identify_no_match(self, sample_csv):
        """Test that unmatched data points get -1."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["record"], [5]),  # No scene for type=5
                "cloud_fraction": (["record"], [10.0]),
            }
        )

        result = scene_def.identify(data)

        assert result[0] == -1

    def test_identify_with_nan_values(self, sample_csv):
        """Test that NaN values result in no match."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["record"], [0, np.nan]),
                "cloud_fraction": (["record"], [5.0, 5.0]),
            }
        )

        result = scene_def.identify(data)

        assert result[0] == 1
        assert result[1] == -1  # NaN doesn't match

    def test_identify_priority_ordering(self):
        """Test that earlier scenes have priority when ranges overlap."""
        csv_content = """scene_id,var1_min,var1_max
1,0,10
2,5,15
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        try:
            scene_def = SceneDefinition(temp_path)

            data = xr.Dataset(
                {
                    "var1": (["record"], [7.0]),  # Matches both scenes
                }
            )

            result = scene_def.identify(data)

            assert result[0] == 1  # First scene wins
        finally:
            temp_path.unlink()

    def test_identify_2d_data(self, sample_csv):
        """Test identifying scenes with 2D data."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["x", "y"], [[0, 0], [1, 1]]),
                "cloud_fraction": (["x", "y"], [[5.0, 25.0], [5.0, 60.0]]),
            }
        )

        result = scene_def.identify(data)

        assert result.shape == (2, 2)
        assert result[0, 0] == 1
        assert result[0, 1] == 2
        assert result[1, 0] == 3
        assert result[1, 1] == 4

    def test_identify_result_attributes(self, sample_csv):
        """Test that identify result has correct attributes."""
        scene_def = SceneDefinition(sample_csv)

        data = xr.Dataset(
            {
                "surface_type": (["record"], [0]),
                "cloud_fraction": (["record"], [5.0]),
            }
        )

        result = scene_def.identify(data)

        assert result.name == f"scene_id_{scene_def.type.lower()}"
        assert "record" in result.dims
        assert result.dtype == np.int32


class TestSceneDefinitionFileIngest:
    """Test SceneDefinition with realistic scene configurations."""

    @pytest.fixture
    def trmm_style_csv(self):
        """Create a TRMM-style scene definition."""
        csv_header = (
            f"scene_id,"
            f"{FootprintVariables.SURFACE_TYPE}_min,"
            f"{FootprintVariables.SURFACE_TYPE}_max,"
            f"{FootprintVariables.CLOUD_FRACTION}_min,"
            f"{FootprintVariables.CLOUD_FRACTION}_max,"
            f"{FootprintVariables.OPTICAL_DEPTH}_min,"
            f"{FootprintVariables.OPTICAL_DEPTH}_max,"
            f"{FootprintVariables.SURFACE_WIND}_min,"
            f"{FootprintVariables.SURFACE_WIND}_max"
        )
        csv_content = f"""{csv_header}
1,0,0,0,5,,,0,15
2,0,0,5,50,0,10,0,7
3,0,0,5,50,10,50,0,7
4,0,0,50,95,0,10,0,7
5,0,0,95,100,,,0,15
6,1,2,0,5,,,0,15
7,1,2,5,50,0,10,0,7
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix="_trmm.csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        yield temp_path
        temp_path.unlink()

    @pytest.fixture
    def erbe_style_csv(self):
        """Create an ERBE-style scene definition."""
        csv_header = (
            f"scene_id,"
            f"{FootprintVariables.SURFACE_TYPE}_min,"
            f"{FootprintVariables.SURFACE_TYPE}_max,"
            f"{FootprintVariables.CLOUD_FRACTION}_min,"
            f"{FootprintVariables.CLOUD_FRACTION}_max,"
            f"{FootprintVariables.CLOUD_PHASE}_min,"
            f"{FootprintVariables.CLOUD_PHASE}_max"
        )
        csv_content = f"""{csv_header}
1,0,0,0,10,,
2,0,0,10,50,1,1
3,0,0,10,50,2,2
4,0,0,50,100,1,1
5,0,0,50,100,2,2
6,1,5,0,100,,
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix="_erbe.csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        yield temp_path
        temp_path.unlink()

    def test_trmm_scene_identification(self, trmm_style_csv):
        """Test TRMM-style scene identification."""
        scene_def = SceneDefinition(trmm_style_csv)

        # Create test data
        data = xr.Dataset(
            {
                FootprintVariables.SURFACE_TYPE: (["record"], [0, 0, 0, 1]),
                FootprintVariables.CLOUD_FRACTION: (["record"], [2.0, 25.0, 97.0, 3.0]),
                FootprintVariables.OPTICAL_DEPTH: (["record"], [0.0, 5.0, 0.0, 0.0]),
                FootprintVariables.SURFACE_WIND: (["record"], [5.0, 5.0, 5.0, 10.0]),
            }
        )

        result = scene_def.identify(data)

        assert result[0] == 1  # Ocean, cf<5
        assert result[1] == 2  # Ocean, cf=5-50, od<10
        assert result[2] == 5  # Ocean, cf>95
        assert result[3] == 6  # Land, cf<5

    def test_erbe_scene_identification(self, erbe_style_csv):
        """Test ERBE-style scene identification."""
        scene_def = SceneDefinition(erbe_style_csv)

        # Create test data
        data = xr.Dataset(
            {
                FootprintVariables.SURFACE_TYPE: (["record"], [0, 0, 0, 1]),
                FootprintVariables.CLOUD_FRACTION: (["record"], [5.0, 30.0, 75.0, 50.0]),
                FootprintVariables.CLOUD_PHASE: (["record"], [1.0, 1.0, 2.0, 1.0]),
            }
        )

        result = scene_def.identify(data)

        assert result[0] == 1  # Ocean, cf<10, no phase constraint
        assert result[1] == 2  # Ocean, cf=10-50, liquid
        assert result[2] == 5  # Ocean, cf=50-100, ice
        assert result[3] == 6  # Land, any cf

    def test_multiple_scene_definitions(self, trmm_style_csv, erbe_style_csv):
        """Test applying multiple scene definitions to same data."""
        trmm_def = SceneDefinition(trmm_style_csv)
        erbe_def = SceneDefinition(erbe_style_csv)

        # Create data with all required variables
        data = xr.Dataset(
            {
                FootprintVariables.SURFACE_TYPE: (["record"], [0, 1]),
                FootprintVariables.CLOUD_FRACTION: (["record"], [25.0, 25.0]),
                FootprintVariables.OPTICAL_DEPTH: (["record"], [5.0, 5.0]),
                FootprintVariables.SURFACE_WIND: (["record"], [5.0, 5.0]),
                FootprintVariables.CLOUD_PHASE: (["record"], [1.0, 1.0]),
            }
        )

        trmm_ids = trmm_def.identify(data)
        erbe_ids = erbe_def.identify(data)

        # Should get different scene IDs from different definitions
        assert trmm_ids[0] == 2  # TRMM ocean scene
        assert erbe_ids[0] == 2  # ERBE ocean liquid scene


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_calculate_cloud_fraction_with_negative_clear_area(self):
        """Test cloud fraction calculation with negative clear area."""
        with pytest.raises(ValueError, match="Clear Area must be between 0 and 100. Got "):
            calculate_cloud_fraction(np.array([-10.0, 50.0, 110.0]))

    @pytest.mark.parametrize(
        ("lower", "upper", "cf_lower", "cf_upper", "cf_total", "expected"),
        [
            (10.0, 5.0, 30.0, 20.0, 50.0, ((10.0 * 30.0 + 5.0 * 20.0) / 50.0)),
            (np.nan, 20.0, 40.0, 10.0, 50.0, (20.0 * 10.0 / 50.0)),
            (5.0, np.nan, 50.0, 0.0, 50.0, (5.0 * 50.0 / 50.0)),
            (10.0, np.nan, 25.0, 0.0, 25.0, (10.0 * 25.0 / 25.0)),
        ],
    )
    def test_calculate_cloud_fraction_weighted_optical_depth_mixed_valid_invalid(
        self, lower, upper, cf_lower, cf_upper, cf_total, expected
    ):
        """Test optical depth with mix of valid and invalid data."""

        result = calculate_cloud_fraction_weighted_optical_depth(lower, upper, cf_lower, cf_upper, cf_total)
        assert result == expected

    def test_scene_matching_with_float_precision(self):
        """Test scene matching handles float precision correctly."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 1.0)})

        # Test values very close to boundaries
        assert scene.matches({"var1": 1.0 - 1e-10}) is True
        assert scene.matches({"var1": 1.0 + 1e-10}) is False
        assert scene.matches({"var1": 0.0 + 1e-10}) is True
        assert scene.matches({"var1": 0.0 - 1e-10}) is False

    def test_weighted_property_all_nan(self):
        """Test weighted property when all values are NaN."""
        result = calculate_cloud_fraction_weighted_property_for_layer(
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        )

        assert np.isnan(result[0])

    def test_calculate_trmm_surface_type_not_in_range(self):
        """Test calculating surface type not found in range."""
        with pytest.raises(ValueError, match="Cannot convert IGBP surface type value to TRMM surface type: "):
            calculate_trmm_surface_type(0)


class TestFootprintDataClass:
    """Tests for multi-step processing in FootprintData."""

    @pytest.fixture
    def complete_sample_data(self):
        """Create complete sample data for full workflow."""
        return xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["record"], [1, 17, 15]),
                FootprintVariables.SURFACE_WIND_U: (["record"], [3.0, 4.0, 0.0]),
                FootprintVariables.SURFACE_WIND_V: (["record"], [4.0, 3.0, 5.0]),
                FootprintVariables.CLEAR_AREA: (["record"], [60.0, 80.0, 20.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["record"], [10.0, 5.0, 15.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["record"], [8.0, 12.0, 10.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["record"], [25.0, 15.0, 50.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["record"], [15.0, 5.0, 30.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["record"], [1.0, 2.0, 1.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["record"], [2.0, 1.0, 2.0]),
            }
        )

    def test_calculate_all_derived_fields(self, complete_sample_data):
        """Test calculating all derived fields from raw data."""
        fp = FootprintData(complete_sample_data)

        all_calculated = [
            FootprintVariables.CLOUD_FRACTION,
            FootprintVariables.SURFACE_WIND,
            FootprintVariables.OPTICAL_DEPTH,
            FootprintVariables.CLOUD_PHASE,
        ]

        fp._calculate_required_fields(all_calculated)

        # Verify all fields are present
        for field in all_calculated:
            assert field in fp._data.data_vars

        # Verify some calculations
        np.testing.assert_array_equal(fp._data[FootprintVariables.CLOUD_FRACTION].values, [40.0, 20.0, 80.0])

        # Surface wind: sqrt(3^2 + 4^2) = 5, sqrt(4^2 + 3^2) = 5, sqrt(0^2 + 5^2) = 5
        np.testing.assert_array_almost_equal(fp._data[FootprintVariables.SURFACE_WIND].values, [5.0, 5.0, 5.0])

    def test_fill_multiple_columns_above_threshold(self, complete_sample_data):
        """Test filling multiple columns above threshold."""
        fp = FootprintData(complete_sample_data)

        # Add some out-of-range values
        fp._data[FootprintVariables.OPTICAL_DEPTH_LOWER].values = np.array([10.0, 600.0, 15.0])
        fp._data[FootprintVariables.CLOUD_FRACTION_LOWER].values = np.array([25.0, 150.0, 50.0])

        # Fill both columns
        fp._fill_column_above_max_value(FootprintVariables.OPTICAL_DEPTH_LOWER, 500.0)
        fp._fill_column_above_max_value(FootprintVariables.CLOUD_FRACTION_LOWER, 100.0)

        # Check optical depth
        assert fp._data[FootprintVariables.OPTICAL_DEPTH_LOWER].values[0] == 10.0
        assert np.isnan(fp._data[FootprintVariables.OPTICAL_DEPTH_LOWER].values[1])
        assert fp._data[FootprintVariables.OPTICAL_DEPTH_LOWER].values[2] == 15.0

        # Check cloud fraction
        assert fp._data[FootprintVariables.CLOUD_FRACTION_LOWER].values[0] == 25.0
        assert np.isnan(fp._data[FootprintVariables.CLOUD_FRACTION_LOWER].values[1])
        assert fp._data[FootprintVariables.CLOUD_FRACTION_LOWER].values[2] == 50.0

    @pytest.fixture
    def sample_scene_csv(self):
        """Create a sample scene CSV for integration testing."""
        csv_header = (
            f"scene_id,"
            f"{FootprintVariables.SURFACE_TYPE}_min,"
            f"{FootprintVariables.SURFACE_TYPE}_max,"
            f"{FootprintVariables.CLOUD_FRACTION}_min,"
            f"{FootprintVariables.CLOUD_FRACTION}_max"
        )
        csv_content = f"""{csv_header}
1,0,0,0,20
2,0,0,20,100
3,1,5,0,100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = Path(f.name)

        yield temp_path
        temp_path.unlink()

    def test_full_workflow_field_calculation_and_scene_id(self, complete_sample_data, sample_scene_csv):
        """Test complete workflow from raw data to scene IDs."""
        fp = FootprintData(complete_sample_data)

        # Calculate required fields for scene identification
        scene_def = SceneDefinition(sample_scene_csv)
        required_fields = scene_def.required_columns

        # Note: SURFACE_TYPE needs to be calculated from IGBP_SURFACE_TYPE
        # For this test, we'll add it manually or add to required_fields
        if FootprintVariables.SURFACE_TYPE in required_fields:
            fp._calculate_required_fields(required_fields)

        # For testing purposes, add surface_type if needed
        if FootprintVariables.SURFACE_TYPE not in fp._data:
            fp._data[FootprintVariables.SURFACE_TYPE] = fp._data[FootprintVariables.IGBP_SURFACE_TYPE]

        # Identify scenes
        scene_ids = scene_def.identify(fp._data)

        # Verify scene IDs were assigned
        assert len(scene_ids) == 3


class TestMemoryEfficiency:
    """Test memory-related behaviors."""

    def test_calculate_fields_modifies_in_place(self):
        """Test that _calculate_required_fields modifies data in place."""
        fp = FootprintData(
            xr.Dataset(
                {
                    FootprintVariables.CLEAR_AREA: (["record"], [60.0]),
                }
            )
        )

        # Get the id of the original dataset
        original_id = id(fp._data)

        fp._calculate_required_fields([FootprintVariables.CLOUD_FRACTION])

        # Should be the same object (modified in place)
        assert id(fp._data) == original_id

    def test_fill_column_modifies_in_place(self):
        """Test that _fill_column_above_max_value modifies in place."""
        fp = FootprintData(xr.Dataset({"col1": (["x"], [1.0, 50.0, 150.0])}))

        original_id = id(fp._data)

        fp._fill_column_above_max_value("col1", 100.0)

        # Should be the same object
        assert id(fp._data) == original_id
