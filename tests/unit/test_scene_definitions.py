"""Unit tests for scene identification module."""

import pathlib
import tempfile

import numpy as np
import pytest

from libera_utils.config import config
from libera_utils.scene_definitions import Scene, SceneDefinition


class TestScene:
    """Test cases for Scene class."""

    def test_scene_creation(self):
        """Test basic scene creation."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0), "var2": (5.0, 15.0)})

        assert scene.scene_id == 1
        assert scene.variable_ranges["var1"] == (0.0, 10.0)
        assert scene.variable_ranges["var2"] == (5.0, 15.0)

    def test_scene_match_single_variable(self):
        """Test scene matching with single variable."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0)})

        assert scene.matches({"var1": 5.0}) is True
        assert scene.matches({"var1": -1.0}) is False
        assert scene.matches({"var1": 11.0}) is False

    def test_scene_match_multiple_variables(self):
        """Test scene matching with multiple variables."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0), "var2": (5.0, 15.0)})

        assert scene.matches({"var1": 5.0, "var2": 10.0}) is True
        assert scene.matches({"var1": 5.0, "var2": 20.0}) is False
        assert scene.matches({"var1": -1.0, "var2": 10.0}) is False

    def test_scene_match_boundary_conditions(self):
        """Test scene matching at boundaries (inclusive min, exclusive max)."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0)})

        assert scene.matches({"var1": 0.0}) is True  # Lower boundary
        assert scene.matches({"var1": 10.0}) is False  # Upper boundary

    def test_scene_no_match_outside_range(self):
        """Test scene not matching when value outside range."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0)})

        assert scene.matches({"var1": -1.0}) is False
        assert scene.matches({"var1": 11.0}) is False

    def test_scene_match_with_nan(self):
        """Test scene does not allow matching with only a NaN value."""
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
    def valid_scene_csv(self):
        """Create a temporary CSV file with valid scene definitions."""
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
    1,0.0,50.0,0.0,10.0
    2,50.0,100.0,0.0,10.0
    3,0.0,50.0,10.0,20.0
    4,50.0,100.0,10.0,20.0
    """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = pathlib.Path(f.name)

        yield temp_path
        temp_path.unlink()

    @pytest.fixture
    def overlapping_scene_csv(self):
        """Create a temporary CSV file with overlapping scene definitions."""
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
    1,0.0,60.0,0.0,10.0
    2,40.0,100.0,0.0,10.0
    """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = pathlib.Path(f.name)

        yield temp_path
        temp_path.unlink()

    @pytest.fixture
    def gap_scene_csv(self):
        """Create a temporary CSV file with gaps in coverage."""
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
    1,0.0,40.0,0.0,10.0
    2,60.0,100.0,0.0,10.0
    """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = pathlib.Path(f.name)

        yield temp_path
        temp_path.unlink()

    @pytest.fixture
    def touching_boundaries_csv(self):
        """Create CSV where scenes touch exactly at boundaries (no overlap)."""
        csv_content = """scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max
    1,0.0,50.0,0.0,10.0
    2,50.0,100.0,0.0,10.0
    3,0.0,50.0,10.0,20.0
    4,50.0,100.0,10.0,20.0
    """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = pathlib.Path(f.name)

        yield temp_path
        temp_path.unlink()

    def test_point_in_scene_inside(self):
        """Test point inside scene bounds."""

        scene = Scene(1, {"cloud_fraction": (0.0, 50.0), "optical_depth": (0.0, 10.0)})
        point = {"cloud_fraction": 25.0, "optical_depth": 5.0}

        result = SceneDefinition._point_in_scene(point, scene, ["cloud_fraction", "optical_depth"])

        assert result is True

    def test_point_in_scene_outside(self):
        """Test point outside scene bounds."""

        scene = Scene(1, {"cloud_fraction": (0.0, 50.0), "optical_depth": (0.0, 10.0)})
        point = {"cloud_fraction": 75.0, "optical_depth": 5.0}

        result = SceneDefinition._point_in_scene(point, scene, ["cloud_fraction", "optical_depth"])

        assert result is False

    def test_point_in_scene_on_min_boundary(self):
        """Test point on minimum boundary (inclusive)."""

        scene = Scene(1, {"cloud_fraction": (50.0, 100.0)})
        point = {"cloud_fraction": 50.0}

        result = SceneDefinition._point_in_scene(point, scene, ["cloud_fraction"])

        assert result is True

    def test_point_in_scene_on_max_boundary(self):
        """Test point on maximum boundary (exclusive)."""
        scene = Scene(1, {"cloud_fraction": (0.0, 50.0)})
        point = {"cloud_fraction": 50.0}

        result = SceneDefinition._point_in_scene(point, scene, ["cloud_fraction"])

        assert result is False


class TestSceneDefinitionLoading:
    """Tests for loading scene definitions from config."""

    def test_load_trmm_scene_definition(self):
        """Test loading TRMM scene definition from config."""
        trmm_path = pathlib.Path(config.get("TRMM_SCENE_DEFINITION"))
        assert trmm_path.exists(), f"TRMM scene definition not found at {trmm_path}"

        scene_def = SceneDefinition(trmm_path)
        assert scene_def is not None
        assert len(scene_def.scenes) == 644

    def test_load_erbe_scene_definition(self):
        """Test loading ERBE scene definition from config."""
        erbe_path = pathlib.Path(config.get("ERBE_SCENE_DEFINITION"))
        assert erbe_path.exists(), f"ERBE scene definition not found at {erbe_path}"

        scene_def = SceneDefinition(erbe_path)
        assert scene_def is not None
        assert len(scene_def.scenes) == 11

    def test_standard_definitions_have_different_types(self):
        """Test that TRMM and ERBE definitions have different types."""
        trmm_def = SceneDefinition(pathlib.Path(config.get("TRMM_SCENE_DEFINITION")))
        erbe_def = SceneDefinition(pathlib.Path(config.get("ERBE_SCENE_DEFINITION")))

        assert trmm_def.type != erbe_def.type, "TRMM and ERBE should have different types"
