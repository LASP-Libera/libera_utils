"""Unit tests for scene identification module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from libera_utils.scene_definitions import SceneDefinition
from libera_utils.scene_id import (
    _CALCULATED_VARIABLE_MAP,
    CalculationSpec,
    FootprintData,
    FootprintVariables,
    IGBPSurfaceType,
    TRMMSurfaceType,
    calculate_cloud_fraction,
    calculate_cloud_fraction_weighted_optical_depth,
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

    def test_calculate_cloud_fraction_invalid_range(self):
        """Test cloud fraction with invalid input range."""
        with pytest.raises(ValueError, match="Clear Area must be between 0 and 100"):
            calculate_cloud_fraction(-10.0)

        with pytest.raises(ValueError, match="Clear Area must be between 0 and 100"):
            calculate_cloud_fraction(110.0)

        # Test with arrays
        with pytest.raises(ValueError, match="Clear Area must be between 0 and 100"):
            calculate_cloud_fraction(np.array([10.0, -5.0, 20.0]))

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

    def test_calculate_trmm_surface_type_scalar(self):
        """Test TRMM surface type conversion with scalar input."""
        # Test valid conversions
        assert calculate_trmm_surface_type(1) == TRMMSurfaceType.HI_SHRUB
        assert calculate_trmm_surface_type(17) == TRMMSurfaceType.OCEAN
        assert calculate_trmm_surface_type(15) == TRMMSurfaceType.SNOW
        assert calculate_trmm_surface_type(16) == TRMMSurfaceType.BRIGHT_DESERT

    def test_calculate_trmm_surface_type_array(self):
        """Test TRMM surface type conversion with array input."""
        igbp_types = np.array([1, 17, 15, 16, 7])
        expected = np.array(
            [
                TRMMSurfaceType.HI_SHRUB,
                TRMMSurfaceType.OCEAN,
                TRMMSurfaceType.SNOW,
                TRMMSurfaceType.BRIGHT_DESERT,
                TRMMSurfaceType.DARK_DESERT,
            ]
        )
        result = calculate_trmm_surface_type(igbp_types)
        np.testing.assert_array_equal(result, expected)

    def test_calculate_trmm_surface_type_invalid_values(self):
        """Test TRMM surface type conversion with invalid values."""
        # Test invalid scalar
        with pytest.raises(ValueError, match="Cannot convert IGBP surface type value to TRMM surface type"):
            calculate_trmm_surface_type(0)  # Invalid: 0 is not a valid IGBP type

        with pytest.raises(ValueError, match="Cannot convert IGBP surface type value to TRMM surface type"):
            calculate_trmm_surface_type(21)  # Invalid: 21 is out of range

        # Test invalid values in array
        with pytest.raises(ValueError, match="Cannot convert IGBP surface type value to TRMM surface type"):
            calculate_trmm_surface_type(np.array([1, 0, 17]))  # Contains invalid value 0

        with pytest.raises(ValueError, match="Cannot convert IGBP surface type value to TRMM surface type"):
            calculate_trmm_surface_type(np.array([1, 25, 17]))  # Contains invalid value 25

    def test_calculate_cloud_fraction_weighted_optical_depth_calculation(self):
        """Test optical depth calculation."""
        result = calculate_cloud_fraction_weighted_optical_depth(10.0, 8.0, 25.0, 15.0, 40.0)
        expected = (10.0 * 25.0 + 8.0 * 15.0) / 40.0
        assert abs(result - expected) < 1e-10

    def test_calculate_cloud_fraction_weighted_optical_depth_with_nan_values(self):
        """Test optical depth calculation with NaN values."""
        # When lower optical depth is NaN, only upper contributes
        result = calculate_cloud_fraction_weighted_optical_depth(np.nan, 8.0, 25.0, 15.0, 40.0)
        expected = 8.0 * 15.0 / 40.0  # Only upper layer contributes
        assert abs(result - expected) < 1e-10

        # When upper optical depth is NaN, only lower contributes
        result = calculate_cloud_fraction_weighted_optical_depth(10.0, np.nan, 25.0, 15.0, 40.0)
        expected = 10.0 * 25.0 / 40.0  # Only lower layer contributes
        assert abs(result - expected) < 1e-10

        # When both are NaN, result is NaN
        result = calculate_cloud_fraction_weighted_optical_depth(np.nan, np.nan, 25.0, 15.0, 40.0)
        assert np.isnan(result)

    def test_calculate_cloud_fraction_weighted_optical_depth_array(self):
        """Test optical depth calculation with array inputs."""
        optical_lower = np.array([10.0, np.nan, 5.0, np.nan])
        optical_upper = np.array([8.0, 12.0, np.nan, np.nan])
        cf_lower = np.array([25.0, 30.0, 40.0, 20.0])
        cf_upper = np.array([15.0, 20.0, 10.0, 30.0])
        cf_total = np.array([40.0, 50.0, 50.0, 50.0])

        result = calculate_cloud_fraction_weighted_optical_depth(
            optical_lower, optical_upper, cf_lower, cf_upper, cf_total
        )

        # First element: both valid
        expected_0 = (10.0 * 25.0 + 8.0 * 15.0) / 40.0
        assert abs(result[0] - expected_0) < 1e-10

        # Second element: only upper valid
        expected_1 = 12.0 * 20.0 / 50.0
        assert abs(result[1] - expected_1) < 1e-10

        # Third element: only lower valid
        expected_2 = 5.0 * 40.0 / 50.0
        assert abs(result[2] - expected_2) < 1e-10

        # Fourth element: both NaN
        assert np.isnan(result[3])

    def test_calculate_cloud_phase_calculation(self):
        """Test cloud phase calculation."""
        # Note: cloud_phase now requires optical_depth parameters
        result = calculate_cloud_phase(1.0, 2.0, 30.0, 10.0, 40.0, 5.0, 3.0)
        expected = round((1.0 * 30.0 + 2.0 * 10.0) / 40.0)
        assert result == expected

    def test_calculate_cloud_phase_with_nan_optical_depths(self):
        """Test cloud phase calculation with NaN optical depth values."""
        # When both optical depths are NaN, result should be NaN
        result = calculate_cloud_phase(1.0, 2.0, 30.0, 10.0, 40.0, np.nan, np.nan)
        assert np.isnan(result)

        # When only one optical depth is NaN, result should still be calculated
        result = calculate_cloud_phase(1.0, 2.0, 30.0, 10.0, 40.0, np.nan, 3.0)
        assert result == 1.0

        result = calculate_cloud_phase(1.0, 2.0, 30.0, 10.0, 40.0, 5.0, np.nan)
        expected = round((1.0 * 30.0 + 2.0 * 10.0) / 40.0)
        assert result == expected

    def test_calculate_cloud_phase_with_nan_phase_values(self):
        """Test cloud phase calculation with NaN phase values."""
        # When lower phase is NaN, only upper contributes
        result = calculate_cloud_phase(np.nan, 2.0, 30.0, 60.0, 90.0, 5.0, 3.0)
        expected = round(2.0 * 60.0 / 90.0)  # Only upper contributes
        assert result == expected

        # When upper phase is NaN, only lower contributes
        result = calculate_cloud_phase(1.0, np.nan, 30.0, 10.0, 40.0, 5.0, 3.0)
        expected = round(1.0 * 30.0 / 40.0)  # Only lower contributes
        assert result == expected

        # When both phases are NaN but optical depths are valid
        result = calculate_cloud_phase(np.nan, np.nan, 30.0, 10.0, 40.0, 5.0, 3.0)
        assert np.isnan(result)

    def test_calculate_cloud_phase_rounding(self):
        """Test that cloud phase is rounded to nearest integer."""
        # Test rounding down
        result = calculate_cloud_phase(1.0, 2.0, 60.0, 10.0, 70.0, 5.0, 3.0)
        # (1.0 * 60 + 2.0 * 10) / 70 = 80/70 = 1.14... rounds to 1
        assert result == 1.0

        # Test rounding up
        result = calculate_cloud_phase(1.0, 2.0, 20.0, 50.0, 70.0, 5.0, 3.0)
        # (1.0 * 20 + 2.0 * 50) / 70 = 120/70 = 1.71... rounds to 2
        assert result == 2.0


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

    def test_convert_missing_values_large_fill_value(self, footprint_with_data):
        """Test converting missing values with large fill value like 9.96921e+36."""
        large_fill = 9.96921e36
        footprint_with_data._data = xr.Dataset({"test_var": (["record"], [1.0, large_fill, 3.0])})
        footprint_with_data._convert_missing_values(large_fill)

        assert footprint_with_data._data["test_var"][0] == 1.0
        assert np.isnan(footprint_with_data._data["test_var"][1])
        assert footprint_with_data._data["test_var"][2] == 3.0

    def test_fill_column_above_max_value(self, footprint_with_data):
        """Test filling values above threshold."""
        footprint_with_data._data = xr.Dataset({"col1": (["x"], [1.0, 50.0, 100.0, 200.0])})
        footprint_with_data._fill_column_above_max_value("col1", 100.0)

        assert footprint_with_data._data["col1"][0] == 1.0
        assert footprint_with_data._data["col1"][1] == 50.0
        assert footprint_with_data._data["col1"][2] == 100.0  # Boundary value is kept
        assert np.isnan(footprint_with_data._data["col1"][3])  # Above threshold

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

    def test_calculate_required_fields_cloud_phase_dependencies(self):
        """Test calculating cloud phase which now requires optical depth fields."""
        data = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["record"], [1.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["record"], [2.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["record"], [30.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["record"], [10.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["record"], [5.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["record"], [3.0]),
            }
        )
        fp = FootprintData(data)

        fp._calculate_required_fields([FootprintVariables.CLOUD_PHASE])

        # Should calculate CLOUD_FRACTION first, then CLOUD_PHASE
        assert FootprintVariables.CLOUD_FRACTION in fp._data.data_vars
        assert FootprintVariables.CLOUD_PHASE in fp._data.data_vars

    def test_calculate_single_field_from_spec(self):
        """Test _calculate_single_field_from_spec method."""
        data = xr.Dataset(
            {
                FootprintVariables.CLEAR_AREA: (["record"], [60.0, 80.0, 20.0]),
            }
        )
        fp = FootprintData(data)

        # Create a spec for cloud fraction calculation
        spec = CalculationSpec(
            output_var=FootprintVariables.CLOUD_FRACTION,
            function=calculate_cloud_fraction,
            input_vars=[FootprintVariables.CLEAR_AREA],
            output_datatype=float,
        )

        # Mark CLEAR_AREA as calculated/available
        calculated = [FootprintVariables.CLEAR_AREA]

        fp._calculate_single_field_from_spec(spec, calculated)

        assert FootprintVariables.CLOUD_FRACTION in fp._data
        np.testing.assert_array_equal(fp._data[FootprintVariables.CLOUD_FRACTION].values, [40.0, 20.0, 80.0])

    def test_calculate_single_field_missing_dependencies(self):
        """Test _calculate_single_field_from_spec with missing dependencies."""
        data = xr.Dataset()
        fp = FootprintData(data)

        spec = CalculationSpec(
            output_var=FootprintVariables.CLOUD_FRACTION,
            function=calculate_cloud_fraction,
            input_vars=[FootprintVariables.CLEAR_AREA],
            output_datatype=float,
        )

        # CLEAR_AREA is not in calculated list
        calculated = []

        with pytest.raises(ValueError, match="Cannot calculate fields - missing dependencies"):
            fp._calculate_single_field_from_spec(spec, calculated)


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

        # Mock 2D arrays
        cloud_frac_data = np.array([[0.0, 10.0, 20.0], [0.0, 30.0, 40.0], [0.0, 50.0, 60.0]])
        mock_dataset.groups["Cloudy_Footprint_Area"].variables = {
            "layers_coverages": MagicMock(),
            "cloud_particle_phase_37um_mean": MagicMock(),
            "cloud_optical_depth_mean": MagicMock(),
        }
        mock_dataset.groups["Cloudy_Footprint_Area"].variables["layers_coverages"].__getitem__ = (
            lambda x: cloud_frac_data
        )

        cloud_phase_data = np.array([[1.0, 2.0], [1.5, 1.8], [2.0, 1.0]])
        cloud_phase_var = mock_dataset.groups["Cloudy_Footprint_Area"].variables["cloud_particle_phase_37um_mean"]
        cloud_phase_var.__getitem__ = lambda x: cloud_phase_data
        cloud_phase_var._FillValue = -999.0  # Set fill value

        optical_depth_data = np.array([[5.0, 10.0], [15.0, 20.0], [25.0, 30.0]])
        mock_dataset.groups["Cloudy_Footprint_Area"].variables["cloud_optical_depth_mean"].__getitem__ = (
            lambda x: optical_depth_data
        )

        igbp_data = np.array([[1], [17], [15]])
        mock_dataset.groups["Surface_Map"].variables = {"surface_igbp_type": MagicMock()}
        mock_dataset.groups["Surface_Map"].variables["surface_igbp_type"].__getitem__ = lambda x: igbp_data

        # Mock 1D arrays
        mock_dataset.groups["Full_Footprint_Area"].variables = {
            "surface_wind_u_vector": MagicMock(),
            "surface_wind_v_vector": MagicMock(),
        }
        mock_dataset.groups["Full_Footprint_Area"].variables["surface_wind_u_vector"].__getitem__ = lambda x: np.array(
            [3.0, 4.0, 5.0]
        )
        mock_dataset.groups["Full_Footprint_Area"].variables["surface_wind_v_vector"].__getitem__ = lambda x: np.array(
            [4.0, 3.0, 12.0]
        )

        mock_dataset.groups["Clear_Footprint_Area"].variables = {"clear_coverage": MagicMock()}
        mock_dataset.groups["Clear_Footprint_Area"].variables["clear_coverage"].__getitem__ = lambda x: np.array(
            [80.0, 50.0, 20.0]
        )

        return mock_dataset

    def test_extract_data_missing_group_error(self):
        """Test error handling when required group is missing."""
        mock_dataset = MagicMock()
        mock_dataset.groups = {}  # Empty groups

        with pytest.raises(ValueError, match="Required variable or group not found"):
            FootprintData._extract_data_from_CeresSSFNOAA20FM6Ed1C(mock_dataset)

    def test_extract_data_missing_variable_error(self):
        """Test error handling when required variable is missing."""
        mock_dataset = MagicMock()
        mock_dataset.groups = {
            "Cloudy_Footprint_Area": MagicMock(),
            "Surface_Map": MagicMock(),
            "Full_Footprint_Area": MagicMock(),
            "Clear_Footprint_Area": MagicMock(),
        }

        # Mock incomplete variables
        mock_dataset.groups["Cloudy_Footprint_Area"].variables = {}  # Missing variables

        with pytest.raises(ValueError, match="Required variable or group not found"):
            FootprintData._extract_data_from_CeresSSFNOAA20FM6Ed1C(mock_dataset)


class TestIdentifyScenes:
    """Test cases for identify_scenes functionality."""

    @pytest.fixture
    def simple_scene_definition_csv(self, tmp_path):
        """Create a simple scene definition CSV."""
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max
1,0.0,50.0
2,50.0,100.0
"""
        csv_file = tmp_path / "test_scenes.csv"
        csv_file.write_text(csv_content)
        return csv_file

    @pytest.fixture
    def sample_footprint_data(self):
        """Create sample footprint data for testing."""
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [1, 17, 15]),
                FootprintVariables.SURFACE_WIND_U: (["footprint"], [3.0, 4.0, 5.0]),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], [4.0, 3.0, 12.0]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [80.0, 50.0, 20.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], [2.0, 5.0, 3.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], [3.0, 10.0, 7.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], [10.0, 40.0, 25.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], [10.0, 40.0, 25.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], [1.0, 1.0, 2.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], [2.0, 2.0, 2.0]),
            }
        )
        return FootprintData(data)

    def test_identify_scenes_calculates_required_fields(self, sample_footprint_data, simple_scene_definition_csv):
        """Test that identify_scenes calculates required fields before identification."""
        scene_def = SceneDefinition(simple_scene_definition_csv)

        # Mock required_calculated_fields attribute
        scene_def.required_calculated_fields = [FootprintVariables.CLOUD_FRACTION]

        with patch.object(sample_footprint_data, "_calculate_required_fields") as mock_calc:
            with patch.object(scene_def, "identify_and_update"):
                sample_footprint_data.identify_scenes([scene_def])

                # Verify the required fields were passed to calculate
                mock_calc.assert_called_once()
                called_fields = mock_calc.call_args[0][0]
                for required_field in scene_def.required_calculated_fields:
                    assert required_field in called_fields

    def test_identify_scenes_with_nan_values(self, simple_scene_definition_csv):
        """Test that identify_scenes handles NaN values in data."""
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [1, 3, 15]),
                FootprintVariables.SURFACE_WIND_U: (["footprint"], [3.0, 4.0, np.nan]),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], [4.0, 3.0, 12.0]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [80.0, 20.0, 50.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], [2.0, np.nan, 3.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], [3.0, 10.0, 7.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], [10.0, 40.0, 25.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], [10.0, 40.0, 25.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], [1.0, 1.0, 2.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], [2.0, 2.0, 2.0]),
            }
        )
        footprint_data = FootprintData(data)
        scene_def = SceneDefinition(simple_scene_definition_csv)

        # Should not raise an error
        footprint_data.identify_scenes([scene_def])
