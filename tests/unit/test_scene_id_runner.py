"""Runner tests for the SCENE-ID CAM-family runner and product write path.

These exercise the manifest/dropbox plumbing in
``libera_utils.scene_identification.scene_id_algorithm`` and its concrete runner configs, including the actual product
write (which is not covered by the algorithm-level tests in ``test_scene_id.py``). The happy-path test in particular
is the guard that the SCENE-ID product definitions can be written under ``strict=True`` conformance.
"""

import numpy as np
import xarray as xr

from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification.scene_id import standard_scene_definitions
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS,
    create_and_write_data_product,
    run_scene_identification,
)
from libera_utils.version import version

# Product-definition path for the CAM variant, read straight from the runner registry.
PRODUCT_DEFINITION_PATH = RUNNER_CONFIGS["cam"].product_definition_path


# Thin per-variant helpers bind the generic runner functions to a registry config so the test bodies below stay
# readable. They are the test-side stand-ins for the per-module wrappers that the runners used to expose.
def run_scene_identification_cam(path):
    """Run scene identification with the CAM config."""
    return run_scene_identification(path, RUNNER_CONFIGS["cam"])


def create_and_write_data_product_cam(footprint_data, input_file_name, output_path):
    """Write a SCENE-ID-CAM product."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIGS["cam"])


def create_and_write_data_product_cam_camtime(footprint_data, input_file_name, output_path):
    """Write a SCENE-ID-CAM-CAMTIME product."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIGS["cam-camtime"])


SSF_INPUT_NAME = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"


class TestSceneIdCamWrite:
    """The CAM runner must produce a conformant SCENE-ID-CAM product with only declared variables."""

    def test_write_data_product_is_conformant(self, scene_id_test_data_path, tmp_path):
        """A full run + write succeeds under strict conformance and re-opens."""
        input_path = scene_id_test_data_path / SSF_INPUT_NAME
        footprint_data = run_scene_identification_cam(input_path)

        # create_and_write_data_product_cam writes with strict=True; if the product definition and dataset are not
        # conformant this raises. Reaching the assertions below is itself the strict-conformance guarantee.
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        # Provenance attributes set by the runner survive the round trip.
        assert reopened.attrs["InputGranules"] == input_path.name
        # algorithm_version is sourced from the installed libera_utils version.
        assert reopened.attrs["algorithm_version"] == version()

    def test_written_product_has_no_undeclared_variables(self, scene_id_test_data_path, tmp_path):
        """Intermediate FootprintData inputs must not leak into the written product."""
        input_path = scene_id_test_data_path / SSF_INPUT_NAME
        footprint_data = run_scene_identification_cam(input_path)
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
        declared = set(definition.coordinates) | set(definition.variables)

        # Read without CF mask/scale so integer variables are not upcast and encoding is preserved as written.
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        undeclared = [name for name in reopened.variables if name not in declared]
        assert undeclared == []
        # And the intermediate scene-property inputs specifically must be gone.
        for leaked in ("surface_wind_u", "surface_wind_v", "optical_depth_lower", "cloud_phase_lower"):
            assert leaked not in reopened.variables


class TestToTimeProduct:
    """FootprintData.to_time_product prepares the dataset for writing on its time axis.

    The error-path and the FMATCH not-implemented-reader checks are pure-logic unit tests and live in
    ``tests/unit/test_scene_id.py`` / ``tests/unit/test_scene_id_algorithm.py``; this integration case runs the real
    CAM runner on real data.
    """

    def test_promotes_time_and_adds_quality_flag(self, scene_id_test_data_path):
        """to_time_product promotes the named time variable to a coordinate and adds a Quality_Flag.

        Runs the real CAM runner to get a populated FootprintData, converts it on RADIOMETER_TIME,
        and asserts the time variable is now a coordinate and a Quality_Flag data variable was added.
        """
        footprint_data = run_scene_identification_cam(scene_id_test_data_path / SSF_INPUT_NAME)
        product = footprint_data.to_time_product("RADIOMETER_TIME")

        assert "RADIOMETER_TIME" in product.coords
        assert "Quality_Flag" in product.data_vars


def _synthetic_camtime_footprint_data() -> FootprintData:
    """Build a small CAM-CAMTIME FootprintData on the 2-D ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid.

    TODO[LIBSDC-855]: this is synthetic input standing in for a real FMATCH-CAM-CAMTIME product, so the CAM-CAMTIME
    write tests below behave as unit tests. Replace it with a real FMATCH-CAM-CAMTIME fixture (and a true manifest-in /
    product-out end-to-end run) once the ``FootprintData.from_fmatch_cam_camtime`` reader is implemented.

    Mirrors the raw inputs the (unimplemented) FMATCH-CAM-CAMTIME reader will supply: the scene-property inputs the
    pipeline derives ``surface_type``/``cloud_fraction`` from, the viewing angles, the boresight geolocation + PSF
    bbox passthroughs, and the four inclusive ``camera_pixel_{x,y}_{min,max}`` pixel-block bounds.

    The grid is two images (two distinct ``CAMERA_TIME`` values) each segmented into two subsections (``PSEUDOFOOTPRINT``
    of size 2). The pixel blocks deliberately OVERLAP within an image (e.g. x = 0..2000 and 1000..2047) -- exactly
    the model the grid exists to represent. ``CAMERA_TIME`` is unique and sorted.
    """
    grid_dims = ("CAMERA_TIME", "PSEUDOFOOTPRINT")
    # Two images (unique, sorted CAMERA_TIME), each segmented into two subsections along PSEUDOFOOTPRINT.
    camera_time = np.array(["2028-02-12T00:00:00", "2028-02-12T00:00:01"], dtype="datetime64[ns]")
    latitude = np.array([[10.0, -20.0], [45.0, -60.0]], dtype=np.float32)
    longitude = np.array([[100.0, -50.0], [170.0, -179.0]], dtype=np.float32)
    dataset = xr.Dataset(
        {
            "igbp_surface_type": (grid_dims, np.array([[1, 5], [10, 17]], dtype=np.uint8)),
            # clear_area is an intermediate the pipeline inverts into cloud_fraction; it is dropped before writing.
            "clear_area": (grid_dims, np.array([[100.0, 40.0], [0.0, 75.0]], dtype=np.float32)),
            "solar_zenith_angle": (grid_dims, np.array([[10.0, 45.0], [80.0, 30.0]], dtype=np.float32)),
            "viewing_zenith_angle": (grid_dims, np.array([[5.0, 20.0], [60.0, 15.0]], dtype=np.float32)),
            "relative_azimuth_angle": (grid_dims, np.array([[30.0, 120.0], [200.0, 300.0]], dtype=np.float32)),
            "latitude": (grid_dims, latitude),
            "longitude": (grid_dims, longitude),
            "altitude": (grid_dims, np.array([[0.0, 100.0], [500.0, 1200.0]], dtype=np.float32)),
            "psf_bbox_lat_min": (grid_dims, (latitude - 1.0).astype(np.float32)),
            "psf_bbox_lat_max": (grid_dims, (latitude + 1.0).astype(np.float32)),
            "psf_bbox_lon_min": (grid_dims, (longitude - 1.0).astype(np.float32)),
            "psf_bbox_lon_max": (grid_dims, (longitude + 1.0).astype(np.float32)),
            # Inclusive pixel-block bounds; blocks overlap within each image (x = 0..2000 overlaps 1000..2047).
            "camera_pixel_x_min": (grid_dims, np.array([[0, 1000], [0, 1000]], dtype=np.int32)),
            "camera_pixel_x_max": (grid_dims, np.array([[2000, 2047], [1024, 2047]], dtype=np.int32)),
            "camera_pixel_y_min": (grid_dims, np.array([[0, 0], [0, 1000]], dtype=np.int32)),
            "camera_pixel_y_max": (grid_dims, np.array([[2047, 2047], [1024, 2047]], dtype=np.int32)),
            "CAMERA_TIME": ("CAMERA_TIME", camera_time),
        }
    )
    return FootprintData(dataset)


class TestSceneIdCamCamtimeWrite:
    """The CAM-CAMTIME runner must write a conformant product on the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid."""

    def test_write_data_product_is_conformant_with_camera_pixel_bounds(self, tmp_path):
        """A full classify + strict write succeeds; data lands on the (CAMERA_TIME, PSEUDOFOOTPRINT) grid."""
        footprint_data = _synthetic_camtime_footprint_data()
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(["erbe", "unfiltering"]))

        # Writes with strict=True; a non-conformant definition/dataset (including the 2-D grid) would raise.
        output_file = create_and_write_data_product_cam_camtime(footprint_data, "fmatch-cam-camtime.nc", tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        # Data lives on the 2-D grid; CAMERA_TIME is a unique, sorted 1-D dimension coordinate.
        assert "CAMERA_TIME" in reopened.sizes
        assert "PSEUDOFOOTPRINT" in reopened.sizes
        assert reopened["CAMERA_TIME"].dims == ("CAMERA_TIME",)
        assert not bool(reopened["CAMERA_TIME"].to_series().duplicated().any())
        # PSEUDOFOOTPRINT is a coordinate: a 0-based int32 index generated over the PSEUDOFOOTPRINT dimension.
        assert "PSEUDOFOOTPRINT" in reopened.coords
        assert reopened["PSEUDOFOOTPRINT"].dims == ("PSEUDOFOOTPRINT",)
        assert reopened["PSEUDOFOOTPRINT"].dtype == np.int32
        assert list(reopened["PSEUDOFOOTPRINT"].values) == list(range(reopened.sizes["PSEUDOFOOTPRINT"]))
        for name in ("cloud_fraction", "scene_id_erbe", "Quality_Flag"):
            assert reopened[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        for name in ("camera_pixel_x_min", "camera_pixel_x_max", "camera_pixel_y_min", "camera_pixel_y_max"):
            assert name in reopened.variables
            assert reopened[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
            assert reopened[name].dtype == np.int32
        # Inclusive (min, max): the max endpoint is never below the min, elementwise across the grid.
        assert bool(np.all(reopened["camera_pixel_x_max"].values >= reopened["camera_pixel_x_min"].values))
        assert bool(np.all(reopened["camera_pixel_y_max"].values >= reopened["camera_pixel_y_min"].values))
