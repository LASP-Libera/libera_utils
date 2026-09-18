"""Unit tests for the placeholder camera cloud-fraction core algorithm.

Covers both timescale builders in :mod:`libera_utils.cloud_fraction.cloud_fraction`: the in-range,
reproducible random draw shared by both, the camera-timescale grid/provenance reuse, and the
radiometer-timescale 1-D record axis.
"""

import numpy as np
import pytest

from libera_utils.cloud_fraction.cf_cam_camtime import PRODUCT_DEFINITION_PATH
from libera_utils.cloud_fraction.cloud_fraction import (
    CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE,
    CLOUD_FRACTION_VARIABLE,
    generate_placeholder_cloud_fraction_camtime,
    generate_placeholder_cloud_fraction_radiometer,
)
from libera_utils.footprint_matching._runner_common import load_l1b_camera_dataset
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.io.product_definition import LiberaDataProductDefinition
from tests.test_data.footprint_matching.fixtures import make_l1b_camera_fixture


@pytest.fixture
def camtime_definition() -> LiberaDataProductDefinition:
    """The parsed CF-CAM-CAMTIME product definition."""
    return LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)


@pytest.fixture
def camera_footprints(tmp_path):
    """Camera pseudo-footprints from a small synthetic L1B CAM file (one footprint per pixel)."""
    l1b_file = make_l1b_camera_fixture(tmp_path, n_images=2, n_pixels_x=4, n_pixels_y=4)
    dataset = load_l1b_camera_dataset(l1b_file)
    return segment_l1b_camera(dataset)


class TestCamtimeBuilder:
    """The camera-timescale builder returns the CF-CAM-CAMTIME grid with in-range placeholder values."""

    def test_returns_all_declared_variables(self, camera_footprints, camtime_definition):
        data = generate_placeholder_cloud_fraction_camtime(camera_footprints, camtime_definition)
        for name in (
            "CAMERA_TIME",
            "PSEUDOFOOTPRINT",
            "camera_pixel_x_min",
            "camera_pixel_x_max",
            "camera_pixel_y_min",
            "camera_pixel_y_max",
            CLOUD_FRACTION_VARIABLE,
            CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE,
        ):
            assert name in data

    def test_values_are_float32_and_in_range(self, camera_footprints, camtime_definition):
        data = generate_placeholder_cloud_fraction_camtime(camera_footprints, camtime_definition)
        for name in (CLOUD_FRACTION_VARIABLE, CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE):
            values = data[name]
            assert values.dtype == np.float32
            finite = values[np.isfinite(values)]
            assert finite.min() >= 0.0
        assert data[CLOUD_FRACTION_VARIABLE][np.isfinite(data[CLOUD_FRACTION_VARIABLE])].max() <= 100.0
        std = data[CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE]
        assert std[np.isfinite(std)].max() <= 15.0

    def test_grid_is_two_dimensional_on_camera_time_and_pseudofootprint(self, camera_footprints, camtime_definition):
        data = generate_placeholder_cloud_fraction_camtime(camera_footprints, camtime_definition)
        n_camera_times = len(np.unique([f.time for f in camera_footprints]))
        assert data[CLOUD_FRACTION_VARIABLE].ndim == 2
        assert data[CLOUD_FRACTION_VARIABLE].shape[0] == n_camera_times
        # camera_pixel bounds are int32 provenance carried straight from the footprint slices.
        assert data["camera_pixel_x_min"].dtype == np.int32

    def test_reproducible_with_fixed_seed(self, camera_footprints, camtime_definition):
        a = generate_placeholder_cloud_fraction_camtime(
            camera_footprints, camtime_definition, rng=np.random.default_rng(1)
        )
        b = generate_placeholder_cloud_fraction_camtime(
            camera_footprints, camtime_definition, rng=np.random.default_rng(1)
        )
        np.testing.assert_array_equal(a[CLOUD_FRACTION_VARIABLE], b[CLOUD_FRACTION_VARIABLE])

    def test_empty_footprints_raises(self, camtime_definition):
        with pytest.raises(ValueError, match="zero pseudo-footprints"):
            generate_placeholder_cloud_fraction_camtime([], camtime_definition)

    def test_out_of_range_max_std_raises(self, camera_footprints, camtime_definition):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            generate_placeholder_cloud_fraction_camtime(
                camera_footprints, camtime_definition, max_standard_deviation_percent=150.0
            )


class TestRadiometerBuilder:
    """The radiometer-timescale builder returns the CF-CAM 1-D record axis with in-range values."""

    def _times(self, n: int) -> np.ndarray:
        base = np.datetime64("2026-06-11T00:00:00", "ns")
        return base + np.arange(n, dtype="int64") * np.timedelta64(10_000_000, "ns")

    def test_returns_declared_variables_on_radiometer_time(self):
        times = self._times(8)
        data = generate_placeholder_cloud_fraction_radiometer(times)
        assert set(data) == {
            "RADIOMETER_TIME",
            CLOUD_FRACTION_VARIABLE,
            CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE,
        }
        assert data["RADIOMETER_TIME"].dtype == np.dtype("datetime64[ns]")
        assert data[CLOUD_FRACTION_VARIABLE].shape == (8,)

    def test_values_float32_and_in_range(self):
        data = generate_placeholder_cloud_fraction_radiometer(self._times(16))
        cf = data[CLOUD_FRACTION_VARIABLE]
        std = data[CLOUD_FRACTION_STANDARD_DEVIATION_VARIABLE]
        assert cf.dtype == np.float32
        assert std.dtype == np.float32
        assert cf.min() >= 0.0
        assert cf.max() <= 100.0
        assert std.min() >= 0.0
        assert std.max() <= 15.0

    def test_reproducible_with_fixed_seed(self):
        times = self._times(8)
        a = generate_placeholder_cloud_fraction_radiometer(times, rng=np.random.default_rng(7))
        b = generate_placeholder_cloud_fraction_radiometer(times, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(a[CLOUD_FRACTION_VARIABLE], b[CLOUD_FRACTION_VARIABLE])

    def test_empty_times_raises(self):
        with pytest.raises(ValueError, match="zero radiometer footprints"):
            generate_placeholder_cloud_fraction_radiometer(np.array([], dtype="datetime64[ns]"))
