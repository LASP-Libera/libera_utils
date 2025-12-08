"""Integration tests for scene identification module.

These tests verify end-to-end workflows and interactions between components
of the scene identification system.
"""

import pathlib

import numpy as np
import pytest
import xarray as xr

from libera_utils.config import config
from libera_utils.scene_id import (
    FootprintData,
    FootprintVariables,
    SceneDefinition,
)


class TestEndToEndSceneIdentification:
    """Integration tests for complete scene identification workflow."""

    def test_from_ceres_ssf(self, test_scene_id):
        input_file_path = test_scene_id / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"
        expected_file_path = test_scene_id / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100_identified.nc"
        fp = FootprintData.from_ceres_ssf(input_file_path)
        fp.identify_scenes()
        expected = xr.open_dataset(expected_file_path)
        xr.testing.assert_equal(fp._data, expected)

    @pytest.mark.parametrize(
        "scene_definition",
        [
            SceneDefinition(pathlib.Path(config.get("TRMM_SCENE_DEFINITION"))),
            SceneDefinition(pathlib.Path(config.get("ERBE_SCENE_DEFINITION"))),
        ],
    )
    def test_from_ceres_ssf_single_scene_definition(self, scene_definition, test_scene_id):
        input_file_path = test_scene_id / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"
        fp = FootprintData.from_ceres_ssf(input_file_path)
        fp.identify_scenes(scene_definitions=[scene_definition])
        expected_file_path = test_scene_id / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100_identified.nc"
        expected = xr.open_dataset(expected_file_path)
        id_column = f"scene_id_{scene_definition.type.lower()}"
        xr.testing.assert_equal(fp._data[id_column], expected[id_column])

    @pytest.mark.parametrize(
        ("input_file_name", "scene_definition", "expected_file_name"),
        [
            (
                "trmm_footprints.nc",
                SceneDefinition(pathlib.Path(config.get("TRMM_SCENE_DEFINITION"))),
                "trmm_footprints_identified.nc",
            ),
            (
                "erbe_footprints.nc",
                SceneDefinition(pathlib.Path(config.get("ERBE_SCENE_DEFINITION"))),
                "erbe_footprints_identified.nc",
            ),
        ],
    )
    def test_standard_scene_definitions(self, input_file_name, scene_definition, expected_file_name, test_scene_id):
        input_dataset = xr.open_dataset(test_scene_id / input_file_name)
        fp = FootprintData(input_dataset)
        fp.identify_scenes(scene_definitions=[scene_definition])
        expected = xr.open_dataset(test_scene_id / expected_file_name)
        xr.testing.assert_equal(fp._data, expected)
        # Each synthetic dataset should contain one of every scene.
        # Loop below confirms that we have coverage of all scenes for each scene type
        for scene in scene_definition.scenes:
            assert scene.scene_id in fp._data[f"scene_id_{scene_definition.type.lower()}"]


class TestSceneDefinitionBehavior:
    """Integration tests for scene definition behavior and edge cases."""

    @pytest.fixture
    def minimal_footprint_data(self):
        """Create minimal footprint data for testing."""
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [1, 17, 15]),
                FootprintVariables.SURFACE_WIND_U: (["footprint"], [3.0, 4.0, 5.0]),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], [4.0, 3.0, 12.0]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [80.0, 50.0, 20.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], [2.0, 5.0, 3.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], [3.0, 10.0, 7.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], [10.0, 25.0, 40.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], [10.0, 25.0, 40.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], [1.0, 1.0, 2.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], [2.0, 2.0, 2.0]),
            }
        )
        return FootprintData(data)

    def test_empty_scene_definition_list(self, minimal_footprint_data):
        """Test behavior with empty scene definition list."""
        # Should not raise an error
        with pytest.raises(ValueError, match="Scene definitions list is empty."):
            minimal_footprint_data.identify_scenes([])

    def test_none_scene_definition_list(self, minimal_footprint_data):
        """Test behavior with scene definition list is none."""
        # Should not raise an error
        with pytest.raises(ValueError, match="No scene definitions provided."):
            minimal_footprint_data.identify_scenes(None)


class TestDataQualityAndEdgeCases:
    """Integration tests for data quality issues and edge cases."""

    def test_mixed_valid_and_missing_data(self):
        """Test handling of partially missing data."""
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [1, 17, 15, 5, 10]),
                FootprintVariables.SURFACE_WIND_U: (["footprint"], [3.0, np.nan, 5.0, 2.0, np.nan]),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], [4.0, 3.0, np.nan, 3.0, 8.0]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [80.0, np.nan, 20.0, 60.0, 30.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], [2.0, 5.0, np.nan, 15.0, 8.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], [3.0, 10.0, 7.0, np.nan, 12.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], [10.0, 25.0, 40.0, 20.0, 35.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], [10.0, 25.0, 40.0, 20.0, 35.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], [1.0, np.nan, 2.0, 1.0, 2.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], [2.0, 2.0, np.nan, 1.0, 1.0]),
            }
        )

        footprint_data = FootprintData(data)

        # Calculate all derived fields
        footprint_data._calculate_required_fields(
            [
                FootprintVariables.CLOUD_FRACTION,
                FootprintVariables.SURFACE_WIND,
                FootprintVariables.OPTICAL_DEPTH,
                FootprintVariables.CLOUD_PHASE,
            ]
        )

        # Check cloud fraction calculation with NaN clear_area
        cloud_fraction = footprint_data._data[FootprintVariables.CLOUD_FRACTION].values
        assert cloud_fraction[0] == 20.0  # Valid calculation
        assert np.isnan(cloud_fraction[1])  # NaN input
        assert cloud_fraction[2] == 80.0  # Valid calculation

        # Check surface wind with mixed NaN values
        surface_wind = footprint_data._data[FootprintVariables.SURFACE_WIND].values
        assert surface_wind[0] == 5.0  # Both components valid
        assert np.isnan(surface_wind[1])  # U component NaN
        assert np.isnan(surface_wind[2])  # V component NaN

        # Check optical depth with NaN values
        optical_depth = footprint_data._data[FootprintVariables.OPTICAL_DEPTH].values
        # First footprint: both valid
        expected_0 = (2.0 * 10.0 + 3.0 * 10.0) / 20.0
        assert abs(optical_depth[0] - expected_0) < 1e-10

        # Third footprint: lower is NaN, only upper contributes
        expected_2 = (7.0 * 40.0) / 80.0
        assert abs(optical_depth[2] - expected_2) < 1e-10


class TestSceneDefinitionValidation:
    """Integration tests for scene definition validation and coverage."""

    def test_overlapping_scenes_detection(self, tmp_path):
        """Test detection of overlapping scene definitions."""
        # Create overlapping scene definitions
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
1,0.0,60.0,0.0,10.0
2,40.0,100.0,0.0,10.0
3,0.0,50.0,5.0,15.0
4,50.0,100.0,5.0,15.0
"""
        csv_file = tmp_path / "overlapping_scenes.csv"
        csv_file.write_text(csv_content)

        # This should log warnings about overlaps
        with pytest.raises(ValueError, match="Overlapping scenes detected:"):
            SceneDefinition(csv_file)

    def test_gap_in_coverage_detection(self, tmp_path):
        """Test detection of gaps in scene definition coverage."""
        # Create scene definitions with gaps
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
1,0.0,30.0,0.0,10.0
2,70.0,100.0,0.0,10.0
3,0.0,30.0,20.0,50.0
4,70.0,100.0,20.0,50.0
"""
        csv_file = tmp_path / "gap_scenes.csv"
        csv_file.write_text(csv_content)

        with pytest.raises(ValueError, match="Incomplete coverage detected."):
            SceneDefinition(csv_file)

    def test_unbounded_scene_definitions(self, tmp_path):
        """Test scene definitions with unbounded ranges."""
        # Create scenes with unbounded min/max values using very large numbers
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
1,0.0,50.0,0.0,10.0
2,50.0,100.0,0.0,10.0
3,0.0,50.0,10.0,9999999.0
4,50.0,100.0,10.0,9999999.0
"""
        csv_file = tmp_path / "unbounded_scenes.csv"
        csv_file.write_text(csv_content)

        scene_def = SceneDefinition(csv_file)

        # Create data with very large optical depth values
        data = xr.Dataset(
            {
                str(FootprintVariables.CLEAR_AREA): (["footprint"], [75.0, 25.0, 75.0]),
                str(FootprintVariables.OPTICAL_DEPTH_LOWER): (["footprint"], [5.0, 100.0, 10000.0]),
                str(FootprintVariables.OPTICAL_DEPTH_UPPER): (["footprint"], [5.0, 200.0, 50000.0]),
                str(FootprintVariables.CLOUD_FRACTION_LOWER): (["footprint"], [12.5, 37.5, 12.5]),
                str(FootprintVariables.CLOUD_FRACTION_UPPER): (["footprint"], [12.5, 37.5, 12.5]),
            }
        )

        footprint_data = FootprintData(data)
        footprint_data.identify_scenes([scene_def])

        scene_ids = footprint_data._data[f"scene_id_{scene_def.type}"].values

        # All should be classified
        assert scene_ids[0] == 1  # cloud=25%, optical=5
        assert scene_ids[1] == 4  # cloud=75%, optical=150 (high but within "unbounded" range)
        assert scene_ids[2] == 3  # cloud=25%, optical=30000 (very high but within "unbounded" range)


class TestErrorHandling:
    """Integration tests for error handling and recovery."""

    def test_invalid_igbp_surface_type_handling(self):
        """Test handling of invalid IGBP surface type values."""
        # Create data with invalid IGBP types
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [0, 25, -1, 10, 21]),  # Invalid values
                FootprintVariables.SURFACE_WIND_U: (["footprint"], [3.0, 4.0, 5.0, 2.0, 6.0]),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], [4.0, 3.0, 12.0, 3.0, 8.0]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [80.0, 50.0, 20.0, 60.0, 30.0]),
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], [2.0, 5.0, 3.0, 15.0, 8.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], [3.0, 10.0, 7.0, 20.0, 12.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], [10.0, 25.0, 40.0, 20.0, 35.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], [10.0, 25.0, 40.0, 20.0, 35.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], [1.0, 1.0, 2.0, 1.0, 2.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], [2.0, 2.0, 2.0, 1.0, 1.0]),
            }
        )

        footprint_data = FootprintData(data)

        # Attempting to calculate surface type should raise an error for invalid values
        with pytest.raises(ValueError, match="Cannot convert IGBP surface type"):
            footprint_data._calculate_required_fields([FootprintVariables.SURFACE_TYPE])

    def test_invalid_clear_area_range(self):
        """Test handling of clear area values outside valid range."""
        # Create data with invalid clear area percentages
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [1, 17, 15]),
                FootprintVariables.SURFACE_WIND_U: (["footprint"], [3.0, 4.0, 5.0]),
                FootprintVariables.SURFACE_WIND_V: (["footprint"], [4.0, 3.0, 12.0]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [-10.0, 110.0, 50.0]),  # Invalid values
                FootprintVariables.OPTICAL_DEPTH_LOWER: (["footprint"], [2.0, 5.0, 3.0]),
                FootprintVariables.OPTICAL_DEPTH_UPPER: (["footprint"], [3.0, 10.0, 7.0]),
                FootprintVariables.CLOUD_FRACTION_LOWER: (["footprint"], [10.0, 25.0, 40.0]),
                FootprintVariables.CLOUD_FRACTION_UPPER: (["footprint"], [10.0, 25.0, 40.0]),
                FootprintVariables.CLOUD_PHASE_LOWER: (["footprint"], [1.0, 1.0, 2.0]),
                FootprintVariables.CLOUD_PHASE_UPPER: (["footprint"], [2.0, 2.0, 2.0]),
            }
        )

        footprint_data = FootprintData(data)

        # Should raise error for invalid clear area values
        with pytest.raises(ValueError, match="Clear Area must be between 0 and 100"):
            footprint_data._calculate_required_fields([FootprintVariables.CLOUD_FRACTION])

    def test_missing_required_variables_for_calculation(self):
        """Test error handling when required variables are missing."""
        # Create incomplete data
        data = xr.Dataset(
            {
                FootprintVariables.IGBP_SURFACE_TYPE: (["footprint"], [1, 17, 15]),
                FootprintVariables.CLEAR_AREA: (["footprint"], [80.0, 50.0, 20.0]),
                # Missing: OPTICAL_DEPTH_LOWER, OPTICAL_DEPTH_UPPER, CLOUD_FRACTION_LOWER, CLOUD_FRACTION_UPPER
            }
        )

        footprint_data = FootprintData(data)

        # Should raise error when trying to calculate optical depth without required inputs
        with pytest.raises(ValueError, match="Cannot calculate fields"):
            footprint_data._calculate_required_fields([FootprintVariables.OPTICAL_DEPTH])
