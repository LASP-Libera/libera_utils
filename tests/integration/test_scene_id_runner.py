"""Integration tests for the SCENE-ID CAM-family runner and product write path.

These exercise the manifest/dropbox plumbing in
``libera_utils.scene_identification._runner`` and the concrete CAM runner, including the actual product write (which
is not covered by the algorithm-level tests in ``test_scene_id.py``). The happy-path test in particular is the guard
that the SCENE-ID product definitions can be written under ``strict=True`` conformance.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestFileRecord, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification.cam.scene_id_cam import (
    PRODUCT_DEFINITION_PATH,
    collect_ssf_input_files,
    create_and_write_data_product_cam,
    run_scene_identification_cam,
)
from libera_utils.scene_identification.cam_camtime.scene_id_cam_camtime import (
    create_and_write_data_product_cam_camtime,
)
from libera_utils.scene_identification.scene_id import standard_scene_definitions

SSF_INPUT_NAME = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"


def _libera_product_name(product_id: DataProductIdentifier) -> str:
    """Build a valid Libera data-product filename string for the given product id."""
    return LiberaDataProductFilename.from_filename_parts(
        product_name=product_id,
        version="V1-0-0",
        utc_start=datetime(2023, 1, 1, tzinfo=UTC),
        utc_end=datetime(2023, 1, 1, 23, 59, 59, tzinfo=UTC),
    ).path.name


class TestSceneIdCamWrite:
    """The CAM runner must produce a conformant SCENE-ID-CAM product with only declared variables."""

    def test_write_data_product_is_conformant(self, test_scene_id, tmp_path):
        """A full run + write succeeds under strict conformance and re-opens."""
        # Targets the CAM run + strict write; asserts the output exists, re-opens, and keeps provenance attrs.
        input_path = test_scene_id / SSF_INPUT_NAME
        footprint_data = run_scene_identification_cam(input_path)

        # create_and_write_data_product_cam writes with strict=True; if the product definition and dataset are not
        # conformant this raises. Reaching the assertions below is itself the strict-conformance guarantee.
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        # Provenance attributes set by the runner survive the round trip.
        assert reopened.attrs["InputGranules"] == input_path.name
        assert reopened.attrs["algorithm_version"] == "0.1.0"

    def test_written_product_has_no_undeclared_variables(self, test_scene_id, tmp_path):
        """Intermediate FootprintData inputs must not leak into the written product."""
        # Targets that only declared vars are written; asserts undeclared list is empty and intermediate inputs gone.
        input_path = test_scene_id / SSF_INPUT_NAME
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


class TestCollectInputFiles:
    """collect_input_files selects the right manifest entries in placeholder vs product mode."""

    # Manifest records must be absolute paths; the runner keys off the filename (basename) when parsing.
    _INPUT_DIR = "/dropbox/inputs"

    def _manifest(self, *filenames: str) -> Manifest:
        return Manifest(
            manifest_type=ManifestType.INPUT,
            files=[ManifestFileRecord(filename=f"{self._INPUT_DIR}/{name}", checksum="0") for name in filenames],
        )

    def test_placeholder_mode_keeps_non_libera_files(self):
        """The CAM runner runs in placeholder mode: keep the CERES SSF (non-Libera) file, skip Libera products."""
        # Targets placeholder-mode selection; asserts collect_ssf_input_files keeps the SSF path, drops the Libera one.
        libera_name = _libera_product_name(DataProductIdentifier.aux_fmatch_cam_camtime)
        manifest = self._manifest(SSF_INPUT_NAME, libera_name)

        selected = collect_ssf_input_files(manifest)

        assert selected == [f"{self._INPUT_DIR}/{SSF_INPUT_NAME}"]

    def test_product_mode_keeps_only_matching_product(self):
        """In Libera-product mode only files with the configured product id are kept."""
        # Targets product-mode selection; asserts collect_input_files returns only the path matching the product id.
        from libera_utils.scene_identification._runner import collect_input_files

        wanted = _libera_product_name(DataProductIdentifier.aux_fmatch_cam_camtime)
        other = _libera_product_name(DataProductIdentifier.l1b_rad)
        manifest = self._manifest(wanted, other, SSF_INPUT_NAME)

        selected = collect_input_files(manifest, DataProductIdentifier.aux_fmatch_cam_camtime)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]


class TestToTimeProduct:
    """FootprintData.to_time_product prepares the dataset for writing on its time axis."""

    def test_promotes_time_and_adds_quality_flag(self, test_scene_id):
        # Targets to_time_product; asserts the named time var becomes a coordinate and a Quality_Flag data var is added.
        footprint_data = run_scene_identification_cam(test_scene_id / SSF_INPUT_NAME)
        product = footprint_data.to_time_product("RADIOMETER_TIME")

        assert "RADIOMETER_TIME" in product.coords
        assert "Quality_Flag" in product.data_vars

    def test_missing_time_variable_raises(self):
        # Targets to_time_product's guard; asserts a missing time variable raises ValueError naming that variable.
        footprint_data = FootprintData(xr.Dataset({"cloud_fraction": ("RADIOMETER_TIME", [1.0, 2.0])}))
        with pytest.raises(ValueError, match="RADIOMETER_TIME"):
            footprint_data.to_time_product("RADIOMETER_TIME")


class TestFmatchReaders:
    """The operational FMATCH readers are not implemented yet."""

    def test_from_fmatch_cam_not_implemented(self, tmp_path):
        # Targets the unimplemented CAM FMATCH reader; asserts from_fmatch_cam raises NotImplementedError.
        with pytest.raises(NotImplementedError):
            FootprintData.from_fmatch_cam(tmp_path / "fmatch.nc")

    def test_from_fmatch_cam_camtime_not_implemented(self, tmp_path):
        # Targets the unimplemented CAM-CAMTIME reader; asserts from_fmatch_cam_camtime raises NotImplementedError.
        with pytest.raises(NotImplementedError):
            FootprintData.from_fmatch_cam_camtime(tmp_path / "fmatch.nc")


def _synthetic_camtime_footprint_data() -> FootprintData:
    """Build a small CAM-CAMTIME FootprintData on the 2-D ``(CAMERA_TIME, FOOTPRINT)`` grid.

    Mirrors the raw inputs the (unimplemented) FMATCH-CAM-CAMTIME reader will supply: the scene-property inputs the
    pipeline derives ``surface_type``/``cloud_fraction`` from, the viewing angles, the boresight geolocation + PSF
    bbox passthroughs, and the four inclusive ``camera_pixel_{x,y}_{min,max}`` pixel-block bounds.

    The grid is two images (two distinct ``CAMERA_TIME`` values) each segmented into two subsections (``FOOTPRINT``
    of size 2). The pixel blocks deliberately OVERLAP within an image (e.g. x = 0..2000 and 1000..2047) -- exactly
    the model the grid exists to represent. ``CAMERA_TIME`` is unique and sorted.
    """
    grid_dims = ("CAMERA_TIME", "FOOTPRINT")
    # Two images (unique, sorted CAMERA_TIME), each segmented into two subsections along FOOTPRINT.
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
    """The CAM-CAMTIME runner must write a conformant product on the 2-D (CAMERA_TIME, FOOTPRINT) grid."""

    def test_write_data_product_is_conformant_with_camera_pixel_bounds(self, tmp_path):
        """A full classify + strict write succeeds; data lands on the (CAMERA_TIME, FOOTPRINT) grid."""
        # Targets the CAM-CAMTIME strict write; asserts data lands on the 2-D grid with int32 camera_pixel bounds.
        footprint_data = _synthetic_camtime_footprint_data()
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(["erbe", "unfiltering"]))

        # Writes with strict=True; a non-conformant definition/dataset (including the 2-D grid) would raise.
        output_file = create_and_write_data_product_cam_camtime(footprint_data, "fmatch-cam-camtime.nc", tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        # Data lives on the 2-D grid; CAMERA_TIME is a unique, sorted 1-D dimension coordinate.
        assert "CAMERA_TIME" in reopened.sizes
        assert "FOOTPRINT" in reopened.sizes
        assert reopened["CAMERA_TIME"].dims == ("CAMERA_TIME",)
        assert not bool(reopened["CAMERA_TIME"].to_series().duplicated().any())
        for name in ("cloud_fraction", "scene_id_erbe", "Quality_Flag"):
            assert reopened[name].dims == ("CAMERA_TIME", "FOOTPRINT")
        for name in ("camera_pixel_x_min", "camera_pixel_x_max", "camera_pixel_y_min", "camera_pixel_y_max"):
            assert name in reopened.variables
            assert reopened[name].dims == ("CAMERA_TIME", "FOOTPRINT")
            assert reopened[name].dtype == np.int32
        # Inclusive (min, max): the max endpoint is never below the min, elementwise across the grid.
        assert bool(np.all(reopened["camera_pixel_x_max"].values >= reopened["camera_pixel_x_min"].values))
        assert bool(np.all(reopened["camera_pixel_y_max"].values >= reopened["camera_pixel_y_min"].values))

    def test_write_drops_the_replaced_pixel_variables(self, tmp_path):
        """The retired center_pixel / start-stop / (min,max)-pair variables must not appear in the written product."""
        # Targets the write's variable pruning; asserts retired center_pixel and start/stop pixel vars are absent.
        footprint_data = _synthetic_camtime_footprint_data()
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(["erbe", "unfiltering"]))
        output_file = create_and_write_data_product_cam_camtime(footprint_data, "fmatch-cam-camtime.nc", tmp_path)

        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        for retired in (
            "center_pixel_x",
            "center_pixel_y",
            "camera_pixel_x_start",
            "camera_pixel_x_stop",
            "camera_pixel_y_start",
            "camera_pixel_y_stop",
            # The replaced (min, max)-pair coordinates and their axis are gone.
            "camera_pixel_x",
            "camera_pixel_y",
        ):
            assert retired not in reopened.variables
        assert "CAMERA_PIXEL_BOUNDS" not in reopened.dims
