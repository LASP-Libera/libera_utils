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
from libera_utils.footprint_matching.product import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestFileRecord, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification.scene_id import standard_scene_definitions
from libera_utils.scene_identification.scene_id_cam import (
    PRODUCT_DEFINITION_PATH,
    collect_fmatch_cam_input_files,
    create_and_write_data_product_cam,
    run_scene_identification_cam,
)
from libera_utils.scene_identification.scene_id_cam_camtime import (
    create_and_write_data_product_cam_camtime,
)
from libera_utils.scene_identification.scene_id_imager import (
    PRODUCT_DEFINITION_PATH as IMAGER_PRODUCT_DEFINITION_PATH,
)
from libera_utils.scene_identification.scene_id_imager import (
    create_and_write_data_product_imager,
    run_scene_identification_imager,
)
from libera_utils.scene_identification.scene_id_imager_flash import (
    create_and_write_data_product_imager_flash,
    run_scene_identification_imager_flash,
)
from tests.test_data.footprint_matching.fixtures import make_fmatch_product_fixture


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

    def test_write_data_product_is_conformant(self, tmp_path):
        """A full run + write succeeds under strict conformance and re-opens."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM)
        footprint_data = run_scene_identification_cam(input_path)

        # create_and_write_data_product_cam writes with strict=True; if the product definition and dataset are not
        # conformant this raises. Reaching the assertions below is itself the strict-conformance guarantee.
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        # Provenance attributes set by the runner survive the round trip.
        assert reopened.attrs["InputGranules"] == input_path.name
        assert reopened.attrs["algorithm_version"] == "0.1.0"

    def test_written_product_has_no_undeclared_variables(self, tmp_path):
        """Reader-sourced FMATCH inputs the classifier does not emit must not leak into the written product."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM)
        footprint_data = run_scene_identification_cam(input_path)
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
        declared = set(definition.coordinates) | set(definition.variables)

        # Read without CF mask/scale so integer variables are not upcast and encoding is preserved as written.
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        undeclared = [name for name in reopened.variables if name not in declared]
        assert undeclared == []


SSF_INPUT_NAME = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"


class TestCollectInputFiles:
    """collect_input_files keeps only the manifest entries whose Libera product id matches the runner's input."""

    # Manifest records must be absolute paths; the runner keys off the filename (basename) when parsing.
    _INPUT_DIR = "/dropbox/inputs"

    def _manifest(self, *filenames: str) -> Manifest:
        # Distinct checksums so the manifest keeps every record (it de-duplicates on identical checksums).
        return Manifest(
            manifest_type=ManifestType.INPUT,
            files=[
                ManifestFileRecord(filename=f"{self._INPUT_DIR}/{name}", checksum=str(index))
                for index, name in enumerate(filenames)
            ],
        )

    def test_cam_runner_keeps_only_fmatch_cam(self):
        """The CAM runner keeps FMATCH-CAM files and skips other Libera products and non-Libera (CERES SSF) files."""
        wanted = _libera_product_name(DataProductIdentifier.aux_fmatch_cam)
        manifest = self._manifest(SSF_INPUT_NAME, wanted)

        selected = collect_fmatch_cam_input_files(manifest)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]

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

    def test_promotes_time_and_adds_quality_flag(self, tmp_path):
        footprint_data = run_scene_identification_cam(make_fmatch_product_fixture(tmp_path, OperationalMode.CAM))
        product = footprint_data.to_time_product("RADIOMETER_TIME")

        assert "RADIOMETER_TIME" in product.coords
        assert "Quality_Flag" in product.data_vars

    def test_missing_time_variable_raises(self):
        # Targets to_time_product's guard; asserts a missing time variable raises ValueError naming that variable.
        footprint_data = FootprintData(xr.Dataset({"cloud_fraction": ("RADIOMETER_TIME", [1.0, 2.0])}))
        with pytest.raises(ValueError, match="RADIOMETER_TIME"):
            footprint_data.to_time_product("RADIOMETER_TIME")


class TestFmatchReaders:
    """The operational FMATCH readers ingest a FMATCH product into a classifiable FootprintData."""

    def test_from_fmatch_cam_reads_classification_inputs(self, tmp_path):
        """from_fmatch_cam maps the FMATCH-CAM inputs onto the RADIOMETER_TIME classification variables."""
        footprint_data = FootprintData.from_fmatch_cam(make_fmatch_product_fixture(tmp_path, OperationalMode.CAM))
        dataset = footprint_data._data

        for required in ("igbp_surface_type", "cloud_fraction", "solar_zenith_angle", "RADIOMETER_TIME"):
            assert required in dataset.variables

    def test_from_fmatch_cam_camtime_reads_records_on_footprint_axis(self, tmp_path):
        """from_fmatch_cam_camtime reads FMATCH-CAM-CAMTIME onto the FOOTPRINT axis, carrying CAMERA_TIME and the 2-D
        camera_pixel range coordinates through (but not the FMATCH-only center pixel)."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM_CAMTIME, n_footprints=6)
        dataset = FootprintData.from_fmatch_cam_camtime(input_path)._data

        # Records live on FOOTPRINT; CAMERA_TIME rides on that axis (a plain variable pre-write).
        assert dataset.sizes["FOOTPRINT"] == 6
        assert dataset["CAMERA_TIME"].dims == ("FOOTPRINT",)
        # Classification inputs the pipeline consumes/derives from are present on the record axis.
        for required in ("igbp_surface_type", "cloud_fraction", "solar_zenith_angle"):
            assert required in dataset.variables
        # The camera pixel-index ranges pass through as 2-D coordinates; the boresight center pixel does not.
        for name in ("camera_pixel_x", "camera_pixel_y"):
            assert dataset[name].dims == ("FOOTPRINT", "CAMERA_PIXEL_BOUNDS")
        assert "center_pixel_x" not in dataset.variables

    def test_from_fmatch_imager_flash_injects_nan_cloud_phase(self, tmp_path):
        """from_fmatch_imager_flash supplies the classification inputs and an all-NaN cloud_phase (no phase source)."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_FLASH)
        dataset = FootprintData.from_fmatch_imager_flash(input_path)._data

        # clear_area feeds the derived cloud_fraction; surface_wind_u/v feed the derived surface_wind; optical_depth
        # is injected from the SSF cloud optical depth.
        for required in ("igbp_surface_type", "clear_area", "surface_wind_u", "optical_depth", "RADIOMETER_TIME"):
            assert required in dataset.variables
        # FMATCH-IMAGER-FLASH has no phase source, so cloud_phase is present-but-NaN.
        assert "cloud_phase" in dataset.variables
        assert bool(np.all(np.isnan(dataset["cloud_phase"].values)))

    def test_from_fmatch_imager_maps_inputs(self, tmp_path):
        """from_fmatch_imager maps the RBSP inputs, including a real (mapped) cloud_phase."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER)
        dataset = FootprintData.from_fmatch_imager(input_path)._data

        for required in ("igbp_surface_type", "clear_area", "surface_wind_u", "optical_depth", "cloud_phase"):
            assert required in dataset.variables
        # The fixture cycles CLDPIX phase codes 1/2, which map to the classifier's 1 (liquid) / 2 (ice).
        assert set(np.unique(dataset["cloud_phase"].values).tolist()) <= {1.0, 2.0}

    def test_from_fmatch_imager_rejects_file_without_rbsp_columns(self, tmp_path):
        """A FMATCH-IMAGER-FLASH file lacks the RBSP CLDPIX variables, so the IMAGER reader raises clearly."""
        flash_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_FLASH)
        with pytest.raises(ValueError, match="missing required variable"):
            FootprintData.from_fmatch_imager(flash_path)


def _synthetic_camtime_footprint_data() -> FootprintData:
    """Build a small CAM-CAMTIME FootprintData on the 2-D ``(CAMERA_TIME, FOOTPRINT)`` grid.

    Mirrors the raw inputs the FMATCH-CAM-CAMTIME reader supplies: the scene-property inputs the
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

    def test_end_to_end_from_written_fmatch_file(self, tmp_path):
        """End-to-end guard (replaces the retired demo generator): a written FMATCH-CAM-CAMTIME file read by the real
        reader, classified, and written as a conformant SCENE-ID-CAM-CAMTIME product on the FOOTPRINT axis."""
        fmatch_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM_CAMTIME, n_footprints=6)
        footprint_data = FootprintData.from_fmatch_cam_camtime(fmatch_path)
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(["erbe", "unfiltering"]))

        output_file = create_and_write_data_product_cam_camtime(footprint_data, fmatch_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        assert reopened.sizes["FOOTPRINT"] == 6
        assert reopened["CAMERA_TIME"].dims == ("FOOTPRINT",)
        for name in ("camera_pixel_x", "camera_pixel_y"):
            assert reopened[name].dims == ("FOOTPRINT", "CAMERA_PIXEL_BOUNDS")
        assert "scene_id_erbe" in reopened.variables
        assert "center_pixel_x" not in reopened.variables


class TestSceneIdImagerWrite:
    """The IMAGER runner produces a conformant SCENE-ID-IMAGER product that includes the TRMM classification."""

    def test_write_data_product_is_conformant_with_trmm(self, tmp_path):
        """A full IMAGER run + write succeeds under strict conformance and reports scene_id_trmm as uint16."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER)
        footprint_data = run_scene_identification_imager(input_path)

        # create_and_write_data_product_imager writes with strict=True; reaching the assertions is the conformance
        # guarantee for scene_id_imager.yml (including the uint16 scene_id_trmm).
        output_file = create_and_write_data_product_imager(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        assert reopened["scene_id_trmm"].dtype == np.uint16
        for scene_id in ("scene_id_erbe", "scene_id_unfiltering", "scene_id_trmm"):
            assert scene_id in reopened.variables

    def test_written_product_has_no_undeclared_variables(self, tmp_path):
        """Reader intermediates (clear_area, surface_wind_u/v) must not leak into the written product."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER)
        footprint_data = run_scene_identification_imager(input_path)
        output_file = create_and_write_data_product_imager(footprint_data, input_path.name, tmp_path)

        definition = LiberaDataProductDefinition.from_yaml(IMAGER_PRODUCT_DEFINITION_PATH)
        declared = set(definition.coordinates) | set(definition.variables)
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        undeclared = [name for name in reopened.variables if name not in declared]
        assert undeclared == []


class TestSceneIdImagerFlashWrite:
    """The FLASH runner produces a conformant SCENE-ID-IMAGER-FLASH product with phase-limited TRMM."""

    def test_write_data_product_is_conformant(self, tmp_path):
        """A full flash run + write succeeds under strict conformance."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_FLASH)
        footprint_data = run_scene_identification_imager_flash(input_path)

        output_file = create_and_write_data_product_imager_flash(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        assert "scene_id_trmm" in reopened.variables
        # FLASH has no cloud-phase source, so cloud_phase is entirely fill/NaN.
        cloud_phase = xr.open_dataset(output_file.path)["cloud_phase"].values
        assert bool(np.all(np.isnan(cloud_phase)))

    def test_flash_trmm_is_phase_limited(self, tmp_path):
        """FLASH still classifies the clear/surface TRMM scenes (cloud_phase unbounded) but not phase-gated ones."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_FLASH)
        footprint_data = run_scene_identification_imager_flash(input_path)
        output_file = create_and_write_data_product_imager_flash(footprint_data, input_path.name, tmp_path)

        trmm = xr.open_dataset(output_file.path, mask_and_scale=False)["scene_id_trmm"].values
        # The near-clear footprint(s) match a clear/surface TRMM scene; nothing lands in a phase-gated cloudy scene.
        assert trmm.max() > 0
        # Every matched TRMM scene here is one of the low-ID clear/surface scenes (trmm.csv scenes 1-14).
        assert set(np.unique(trmm).tolist()) <= set(range(0, 15))
