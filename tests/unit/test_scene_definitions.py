"""Unit tests for scene identification module."""

import pathlib
import tempfile

import numpy as np
import pytest

from libera_utils.config import config
from libera_utils.scene_identification.scene_definitions import Scene, SceneDefinition


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

    def test_get_bin_bounds(self):
        """Test retrieving (min, max) bounds for a variable."""
        scene = Scene(scene_id=1, variable_ranges={"var1": (0.0, 10.0), "var2": (None, 5.0)})

        assert scene.get_bin_bounds("var1") == (0.0, 10.0)
        assert scene.get_bin_bounds("var2") == (None, 5.0)
        # Variable not constrained by this scene is unbounded on both sides.
        assert scene.get_bin_bounds("missing") == (None, None)


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


class TestViewingGeometryClassificationVariables:
    """Tests that the viewing-geometry angles are required, bounded classification variables everywhere.

    The three viewing-geometry angles (solar_zenith_angle, viewing_zenith_angle, relative_azimuth_angle) were added
    to every standard scene definition with the full physical range as a placeholder bin. Because those bins are
    defined (not empty), the angles behave like any other classification variable: they are required inputs, they
    are validated, and their bin bounds are reported. These tests lock in that behavior so a future change that
    subdivides the geometry space does not accidentally drop the angles from the required/classification sets.
    """

    # The full physical range placeholder bins, shared by every scene (degrees). These match the CERES SSF
    # valid_range attributes and the values written into the scene definition CSVs.
    GEOMETRY_BINS = {
        "solar_zenith_angle": (0.0, 180.0),
        "viewing_zenith_angle": (0.0, 90.0),
        "relative_azimuth_angle": (0.0, 360.0),
    }

    @pytest.mark.parametrize(
        "config_key", ["TRMM_SCENE_DEFINITION", "ERBE_SCENE_DEFINITION", "UNFILTERING_SCENE_DEFINITION"]
    )
    def test_geometry_angles_are_required_and_classification_variables(self, config_key):
        """Every standard definition must require and classify on all three geometry angles."""
        scene_def = SceneDefinition(pathlib.Path(config.get(config_key)))
        for variable in self.GEOMETRY_BINS:
            assert variable in scene_def.required_columns
            assert variable in scene_def.classification_variables

    @pytest.mark.parametrize(
        "config_key", ["TRMM_SCENE_DEFINITION", "ERBE_SCENE_DEFINITION", "UNFILTERING_SCENE_DEFINITION"]
    )
    def test_every_scene_uses_full_range_geometry_placeholder_bins(self, config_key):
        """Each scene's geometry bins are the full physical range placeholder (no special-casing needed)."""
        scene_def = SceneDefinition(pathlib.Path(config.get(config_key)))
        for scene in scene_def.scenes:
            for variable, expected_bounds in self.GEOMETRY_BINS.items():
                assert scene.get_bin_bounds(variable) == expected_bounds

    def test_geometry_bins_reported_by_compute_property_bins(self):
        """_compute_property_bins emits bounded geometry bins for matched footprints and NaN for unmatched."""
        scene_def = SceneDefinition(pathlib.Path(config.get("ERBE_SCENE_DEFINITION")))
        scene_ids = np.array([1, 0])  # one matched scene, one unmatched footprint
        bins = scene_def._compute_property_bins(scene_ids)

        for variable, (bin_min, bin_max) in self.GEOMETRY_BINS.items():
            min_name = f"scene_bin_{scene_def.type}_{variable}_min"
            max_name = f"scene_bin_{scene_def.type}_{variable}_max"
            # Matched footprint gets the placeholder bounds; unmatched footprint gets NaN.
            np.testing.assert_array_equal(bins[min_name], [bin_min, np.nan])
            np.testing.assert_array_equal(bins[max_name], [bin_max, np.nan])


class TestPropertyBins:
    """Tests for property bin reporting (min/max bounds alongside scene IDs)."""

    @pytest.fixture
    def scene_definition(self, tmp_path):
        """Create a fully-covering 4-scene definition over two variables."""
        csv_content = (
            "scene_id,cloud_fraction_min,cloud_fraction_max,optical_depth_min,optical_depth_max\n"
            "1,0.0,50.0,0.0,10.0\n"
            "2,50.0,100.0,0.0,10.0\n"
            "3,0.0,50.0,10.0,20.0\n"
            "4,50.0,100.0,10.0,20.0\n"
        )
        csv_file = tmp_path / "bins.csv"
        csv_file.write_text(csv_content)
        return SceneDefinition(csv_file)

    def test_get_bin_bounds_for_scene_id(self, scene_definition):
        """Test looking up bin bounds for a known scene ID."""
        bounds = scene_definition.get_bin_bounds_for_scene_id(4)
        assert bounds == {"cloud_fraction": (50.0, 100.0), "optical_depth": (10.0, 20.0)}

    def test_get_bin_bounds_for_unknown_scene_id_raises(self, scene_definition):
        """Test that an unknown scene ID (including 0) raises KeyError."""
        with pytest.raises(KeyError, match="Scene ID 0 not found"):
            scene_definition.get_bin_bounds_for_scene_id(0)

    def test_compute_property_bins(self, scene_definition):
        """Test per-footprint bin bound arrays, including unmatched footprints."""
        scene_ids = np.array([1, 4, 0])
        bins = scene_definition._compute_property_bins(scene_ids)

        np.testing.assert_array_equal(bins["scene_bin_bins_cloud_fraction_min"], [0.0, 50.0, np.nan])
        np.testing.assert_array_equal(bins["scene_bin_bins_cloud_fraction_max"], [50.0, 100.0, np.nan])
        np.testing.assert_array_equal(bins["scene_bin_bins_optical_depth_min"], [0.0, 10.0, np.nan])
        np.testing.assert_array_equal(bins["scene_bin_bins_optical_depth_max"], [10.0, 20.0, np.nan])

    def test_compute_property_bins_unbounded_side_is_nan(self, tmp_path):
        """Test that an unbounded bin side is reported as NaN."""
        csv_content = (
            "scene_id,cloud_fraction_min,cloud_fraction_max\n"
            "1,,50.0\n"  # unbounded minimum
            "2,50.0,\n"  # unbounded maximum
        )
        csv_file = tmp_path / "unbounded.csv"
        csv_file.write_text(csv_content)
        scene_definition = SceneDefinition(csv_file)

        bins = scene_definition._compute_property_bins(np.array([1, 2]))
        np.testing.assert_array_equal(bins["scene_bin_unbounded_cloud_fraction_min"], [np.nan, 50.0])
        np.testing.assert_array_equal(bins["scene_bin_unbounded_cloud_fraction_max"], [50.0, np.nan])

    def test_continuous_bin_bounds_are_float32(self, scene_definition):
        """Continuous (non-categorical) bin bounds are stored as compact float32, not float64."""
        bins = scene_definition._compute_property_bins(np.array([1, 4, 0]))
        # cloud_fraction / optical_depth are continuous, so they use the default float32 storage dtype.
        assert bins["scene_bin_bins_cloud_fraction_min"].dtype == np.float32
        assert bins["scene_bin_bins_optical_depth_max"].dtype == np.float32

    def test_surface_type_bin_bounds_are_uint8(self, tmp_path):
        """surface_type bin bounds are stored as compact uint8; unmatched footprints get 0.

        surface_type is a small categorical code, so its bin bounds are kept as uint8 rather than float32 to save
        storage. There is no fill value: an unmatched footprint (scene_id 0) simply gets 0 for its bounds, and
        scene_id == 0 is the authoritative flag that the footprint was not classified.
        """
        # Two contiguous bins covering surface_type [0, 6) so the SceneDefinition coverage check passes.
        csv_content = "scene_id,surface_type_min,surface_type_max\n1,0,3\n2,3,6\n"
        csv_file = tmp_path / "surface.csv"
        csv_file.write_text(csv_content)
        scene_definition = SceneDefinition(csv_file)

        # Two matched footprints (scenes 1 and 2) and one unmatched footprint (scene_id 0).
        bins = scene_definition._compute_property_bins(np.array([1, 2, 0]))

        min_bounds = bins["scene_bin_surface_surface_type_min"]
        max_bounds = bins["scene_bin_surface_surface_type_max"]

        # Bounds are stored as compact unsigned bytes.
        assert min_bounds.dtype == np.uint8
        assert max_bounds.dtype == np.uint8
        # Matched footprints report their bounds; the unmatched footprint gets 0 (disambiguated by scene_id 0).
        np.testing.assert_array_equal(min_bounds, np.array([0, 3, 0], dtype=np.uint8))
        np.testing.assert_array_equal(max_bounds, np.array([3, 6, 0], dtype=np.uint8))
