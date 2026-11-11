"""Integration tests for the SCENE-ID CAM-family runner and product write path.

These exercise the manifest/dropbox plumbing in
``libera_utils.scene_identification.scene_id_algorithm`` and its concrete runner configs, including the actual product
write (which is not covered by the algorithm-level tests in ``test_scene_id.py``). The happy-path test in particular
is the guard that the SCENE-ID product definitions can be written under ``strict=True`` conformance.
``TestSceneIdContainerRun`` runs each variant the way its container does, through the ``libera-utils scene-id`` CLI.
"""

import logging
from datetime import UTC, datetime

import netCDF4
import numpy as np
import pytest
import xarray as xr

from libera_utils import cli
from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.footprint_match_algorithm import RUNNER_CONFIGS as FMATCH_RUNNER_CONFIGS
from libera_utils.footprint_matching.product import FMATCH_INTEGER_FILL_VALUE, OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification.scene_id import standard_scene_definitions
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS,
    SCENE_ID_VARIANT_ENV,
    collect_input_files,
    create_and_write_data_product,
    run_scene_identification,
)
from libera_utils.version import version
from tests.plugins.footprint_matching_fixtures import make_fmatch_product_fixture

pytestmark = pytest.mark.integration

# Product-definition paths for the variants exercised below, read straight from the runner registry.
PRODUCT_DEFINITION_PATH = RUNNER_CONFIGS["cam"].product_definition_path
IMAGER_PRODUCT_DEFINITION_PATH = RUNNER_CONFIGS["imager"].product_definition_path
IMAGER_CAMTIME_PRODUCT_DEFINITION_PATH = RUNNER_CONFIGS["imager-camtime"].product_definition_path


# Thin per-variant helpers bind the generic runner functions to a registry config so the test bodies below stay
# readable.
def run_scene_identification_cam(path):
    """Run scene identification with the CAM config."""
    return run_scene_identification(path, RUNNER_CONFIGS["cam"])


def run_scene_identification_imager(path):
    """Run scene identification with the IMAGER config."""
    return run_scene_identification(path, RUNNER_CONFIGS["imager"])


def run_scene_identification_imager_camtime(path):
    """Run scene identification with the IMAGER-CAMTIME config."""
    return run_scene_identification(path, RUNNER_CONFIGS["imager-camtime"])


def create_and_write_data_product_cam(footprint_data, input_file_name, output_path):
    """Write a SCENE-ID-CAM product."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIGS["cam"])


def create_and_write_data_product_cam_camtime(footprint_data, input_file_name, output_path):
    """Write a SCENE-ID-CAM-CAMTIME product."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIGS["cam-camtime"])


def create_and_write_data_product_imager(footprint_data, input_file_name, output_path):
    """Write a SCENE-ID-IMAGER product."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIGS["imager"])


def create_and_write_data_product_imager_camtime(footprint_data, input_file_name, output_path):
    """Write a SCENE-ID-IMAGER-CAMTIME product."""
    return create_and_write_data_product(footprint_data, input_file_name, output_path, RUNNER_CONFIGS["imager-camtime"])


def collect_fmatch_cam_input_files(manifest):
    """Select the FMATCH-CAM input files from ``manifest``."""
    return collect_input_files(manifest, RUNNER_CONFIGS["cam"].input_product_id)


def collect_fmatch_imager_camtime_input_files(manifest):
    """Select the FMATCH-IMAGER-CAMTIME input files from ``manifest``."""
    return collect_input_files(manifest, RUNNER_CONFIGS["imager-camtime"].input_product_id)


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
        assert reopened.attrs["algorithm_version"] == version()

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
        return Manifest(manifest_type=ManifestType.INPUT, files=[f"{self._INPUT_DIR}/{name}" for name in filenames])

    def test_cam_runner_keeps_only_fmatch_cam(self):
        """The CAM runner keeps FMATCH-CAM files and skips other Libera products and non-Libera (CERES SSF) files."""
        wanted = _libera_product_name(DataProductIdentifier.aux_fmatch_cam)
        manifest = self._manifest(SSF_INPUT_NAME, wanted)

        selected = collect_fmatch_cam_input_files(manifest)

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
        footprint_data = FootprintData(xr.Dataset({"Cloud_Fraction": ("RADIOMETER_TIME", [1.0, 2.0])}))
        with pytest.raises(ValueError, match="RADIOMETER_TIME"):
            footprint_data.to_time_product("RADIOMETER_TIME")


class TestFmatchReaders:
    """The operational FMATCH readers ingest a FMATCH product into a classifiable FootprintData."""

    def test_from_fmatch_cam_reads_classification_inputs(self, tmp_path):
        """from_fmatch_cam maps the FMATCH-CAM inputs onto the RADIOMETER_TIME classification variables."""
        footprint_data = FootprintData.from_fmatch_cam(make_fmatch_product_fixture(tmp_path, OperationalMode.CAM))
        dataset = footprint_data._data

        for required in ("IGBP_Surface_Type", "Cloud_Fraction", "Solar_Zenith_Surface", "RADIOMETER_TIME"):
            assert required in dataset.variables

    def test_from_fmatch_cam_camtime_reads_records_on_camera_grid(self, tmp_path):
        """from_fmatch_cam_camtime reads FMATCH-CAM-CAMTIME onto the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid, carrying
        CAMERA_TIME and the four camera_pixel bound coordinates through (but not the FMATCH-only center pixel)."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM_CAMTIME, n_footprints=6)
        dataset = FootprintData.from_fmatch_cam_camtime(input_path)._data

        # Records live on the 2-D grid; CAMERA_TIME is the 1-D image-acquisition axis (a plain variable pre-write).
        assert dataset.sizes["CAMERA_TIME"] == 6
        assert "PSEUDOFOOTPRINT" in dataset.sizes
        assert dataset["CAMERA_TIME"].dims == ("CAMERA_TIME",)
        # Classification inputs the pipeline consumes/derives from are present on the 2-D grid.
        for required in ("IGBP_Surface_Type", "Cloud_Fraction", "Solar_Zenith_Surface"):
            assert required in dataset.variables
            assert dataset[required].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        # The four camera pixel-block bounds pass through as 2-D coordinates; the boresight center pixel does not.
        for name in ("Camera_Pixel_X_Min", "Camera_Pixel_X_Max", "Camera_Pixel_Y_Min", "Camera_Pixel_Y_Max"):
            assert dataset[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        assert "Center_Pixel_X" not in dataset.variables

    def test_from_fmatch_imager_maps_inputs(self, tmp_path):
        """from_fmatch_imager maps the FLASHFlux/CLDPIX inputs, including a real (mapped) cloud_phase."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER)
        dataset = FootprintData.from_fmatch_imager(input_path)._data

        for required in ("IGBP_Surface_Type", "clear_area", "surface_wind_u", "Optical_Depth", "Cloud_Phase"):
            assert required in dataset.variables
        # The fixture cycles CLDPIX phase codes 1/2, which map to the classifier's 1 (liquid) / 2 (ice).
        assert set(np.unique(dataset["Cloud_Phase"].values).tolist()) <= {1.0, 2.0}
        _assert_imager_sources(dataset)

    def test_from_fmatch_imager_camtime_reads_inputs_on_camera_grid_with_passthrough(self, tmp_path):
        """from_fmatch_imager_camtime maps the full FLASHFlux/CLDPIX inputs onto the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT)
        grid AND carries the boresight/PSF/camera-pixel identifier variables straight through."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_CAMTIME, n_footprints=6)
        dataset = FootprintData.from_fmatch_imager_camtime(input_path)._data

        # Records live on the 2-D grid; CAMERA_TIME is the 1-D image-acquisition axis (a plain variable pre-write).
        assert dataset.sizes["CAMERA_TIME"] == 6
        assert "PSEUDOFOOTPRINT" in dataset.sizes
        assert dataset["CAMERA_TIME"].dims == ("CAMERA_TIME",)
        # The full IMAGER classification inputs (incl. a real, mapped cloud_phase) land on the 2-D grid.
        for required in ("IGBP_Surface_Type", "clear_area", "surface_wind_u", "Optical_Depth", "Cloud_Phase"):
            assert required in dataset.variables
            assert dataset[required].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        assert set(np.unique(dataset["Cloud_Phase"].values).tolist()) <= {1.0, 2.0}
        _assert_imager_sources(dataset)
        # The camtime provenance identifiers pass through on the 2-D grid; the FMATCH-only center pixel does not.
        for name in (
            "Latitude",
            "Longitude",
            "Altitude",
            "PSF_Bbox_Lat_Min",
            "Camera_Pixel_X_Min",
            "Camera_Pixel_X_Max",
            "Camera_Pixel_Y_Min",
            "Camera_Pixel_Y_Max",
        ):
            assert dataset[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        assert "Center_Pixel_X" not in dataset.variables

    @pytest.mark.parametrize(
        ("reader", "file_mode", "missing_product_name", "expected_input"),
        [
            pytest.param(
                "from_fmatch_imager",
                OperationalMode.CAM,
                "FLASHFlux_Clear_Coverage",
                "FMATCH-IMAGER file for LIBERA_SCENE_ID_VARIANT=imager",
                id="imager-on-cam",
            ),
            pytest.param(
                "from_fmatch_imager_camtime",
                OperationalMode.CAM_CAMTIME,
                "FLASHFlux_Clear_Coverage",
                "FMATCH-IMAGER-CAMTIME file for LIBERA_SCENE_ID_VARIANT=imager-camtime",
                id="imager-camtime-on-cam-camtime",
            ),
            pytest.param(
                "from_fmatch_cam",
                OperationalMode.IMAGER,
                "Cloud_Fraction_Camera",
                "FMATCH-CAM file for LIBERA_SCENE_ID_VARIANT=cam",
                id="cam-on-imager",
            ),
            # A radiometer-timescale file has no camera pixel bounds, so the pass-through check fails too.
            pytest.param(
                "from_fmatch_imager_camtime",
                OperationalMode.IMAGER,
                "Camera_Pixel_X_Min",
                "FMATCH-IMAGER-CAMTIME file for LIBERA_SCENE_ID_VARIANT=imager-camtime",
                id="imager-camtime-on-imager",
            ),
        ],
    )
    def test_reader_rejects_a_file_of_another_mode(
        self, tmp_path, reader, file_mode, missing_product_name, expected_input
    ):
        """A reader handed another mode's FMATCH file names the missing product variable and the input it needs."""
        wrong_file = make_fmatch_product_fixture(tmp_path, file_mode, n_footprints=4)
        with pytest.raises(ValueError, match=r"missing required variable\(s\) \[.*\]") as excinfo:
            getattr(FootprintData, reader)(wrong_file)
        message = str(excinfo.value)
        assert f"'{missing_product_name}'" in message
        assert f"check that the manifest's input is a {expected_input}" in message
        assert "Present data variables" not in message

    def test_missing_input_raises_file_not_found(self, tmp_path):
        """A path that does not exist raises FileNotFoundError naming it, chained to the open's own error."""
        absent = tmp_path / "absent.nc"
        with pytest.raises(FileNotFoundError, match="Unable to open input file") as excinfo:
            FootprintData.from_fmatch_cam(absent)
        assert str(absent) in str(excinfo.value)
        assert excinfo.value.__cause__ is not None

    @pytest.mark.parametrize("as_uri", [False, True], ids=["s3path", "uri-string"])
    def test_reader_reads_an_s3_input(self, tmp_path, write_file_to_s3, as_uri):
        """An FMATCH product in S3, given as an S3Path or the manifest's URI string, reads like a local one."""
        local_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM)
        s3_path = write_file_to_s3(local_path, f"s3://fmatch-inputs/{local_path.name}")

        footprint_data = run_scene_identification(s3_path.as_uri() if as_uri else s3_path, RUNNER_CONFIGS["cam"])

        local_data = FootprintData.from_fmatch_cam(local_path)._data
        assert footprint_data._data.sizes["RADIOMETER_TIME"] == local_data.sizes["RADIOMETER_TIME"]
        np.testing.assert_array_equal(
            footprint_data._data["IGBP_Surface_Type"].values, local_data["IGBP_Surface_Type"].values
        )

    @pytest.mark.parametrize(
        ("reader", "mode"),
        [
            pytest.param("from_fmatch_cam", OperationalMode.CAM, id="cam"),
            pytest.param("from_fmatch_cam_camtime", OperationalMode.CAM_CAMTIME, id="cam-camtime"),
        ],
    )
    def test_cam_reader_warns_when_cloud_fraction_is_all_fill(self, tmp_path, caplog, reader, mode):
        """A CAM product with no finite camera cloud fraction is read, with a warning naming the file and LIBSDC-785."""
        input_path = make_fmatch_product_fixture(tmp_path, mode, n_footprints=4)
        with netCDF4.Dataset(input_path, "a") as dataset:
            dataset["Cloud_Fraction_Camera"][:] = np.nan

        with caplog.at_level(logging.WARNING, logger="libera_utils.scene_identification.scene_id"):
            footprint_data = getattr(FootprintData, reader)(input_path)

        assert not np.isfinite(footprint_data._data["Cloud_Fraction"].values).any()
        warnings = [
            record
            for record in caplog.records
            if record.name == "libera_utils.scene_identification.scene_id" and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        assert str(input_path) in warnings[0].getMessage()
        assert "LIBSDC-785" in warnings[0].getMessage()

    def test_cam_reader_does_not_warn_with_a_cloud_fraction(self, tmp_path, caplog):
        """A CAM product with a camera cloud fraction is read without the all-fill warning."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM, n_footprints=4)
        with caplog.at_level(logging.WARNING, logger="libera_utils.scene_identification.scene_id"):
            FootprintData.from_fmatch_cam(input_path)
        assert "LIBSDC-785" not in caplog.text


def _assert_imager_sources(dataset: xr.Dataset) -> None:
    """Assert each IMAGER classification input came from its own FMATCH source variable.

    ``make_fmatch_product_fixture`` writes a distinct value into each source, so a column map that reads the
    wrong source fails here.
    """
    np.testing.assert_allclose(
        dataset["clear_area"].values,
        np.linspace(0.0, 100.0, dataset["clear_area"].size, dtype=np.float32).reshape(dataset["clear_area"].shape),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(dataset["surface_wind_u"].values, 3.0)
    np.testing.assert_array_equal(dataset["surface_wind_v"].values, 4.0)
    np.testing.assert_array_equal(dataset["Optical_Depth"].values, 7.0)


def _synthetic_camtime_footprint_data() -> FootprintData:
    """Build a small CAM-CAMTIME FootprintData on the 2-D ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid.

    Mirrors the raw inputs the FMATCH-CAM-CAMTIME reader supplies: the scene-property inputs the
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
            "IGBP_Surface_Type": (grid_dims, np.array([[1, 5], [10, 17]], dtype=np.int8)),
            # clear_area is an intermediate the pipeline inverts into cloud_fraction; it is dropped before writing.
            "clear_area": (grid_dims, np.array([[100.0, 40.0], [0.0, 75.0]], dtype=np.float32)),
            "Solar_Zenith_Surface": (grid_dims, np.array([[10.0, 45.0], [80.0, 30.0]], dtype=np.float32)),
            "Viewing_Zenith_Surface": (grid_dims, np.array([[5.0, 20.0], [60.0, 15.0]], dtype=np.float32)),
            "Relative_Azimuth_Surface": (grid_dims, np.array([[30.0, 120.0], [200.0, 300.0]], dtype=np.float32)),
            "Latitude": (grid_dims, latitude),
            "Longitude": (grid_dims, longitude),
            "Altitude": (grid_dims, np.array([[0.0, 100.0], [500.0, 1200.0]], dtype=np.float32)),
            "PSF_Bbox_Lat_Min": (grid_dims, (latitude - 1.0).astype(np.float32)),
            "PSF_Bbox_Lat_Max": (grid_dims, (latitude + 1.0).astype(np.float32)),
            "PSF_Bbox_Lon_Min": (grid_dims, (longitude - 1.0).astype(np.float32)),
            "PSF_Bbox_Lon_Max": (grid_dims, (longitude + 1.0).astype(np.float32)),
            # Inclusive pixel-block bounds; blocks overlap within each image (x = 0..2000 overlaps 1000..2047).
            "Camera_Pixel_X_Min": (grid_dims, np.array([[0, 1000], [0, 1000]], dtype=np.int32)),
            "Camera_Pixel_X_Max": (grid_dims, np.array([[2000, 2047], [1024, 2047]], dtype=np.int32)),
            "Camera_Pixel_Y_Min": (grid_dims, np.array([[0, 0], [0, 1000]], dtype=np.int32)),
            "Camera_Pixel_Y_Max": (grid_dims, np.array([[2047, 2047], [1024, 2047]], dtype=np.int32)),
            "CAMERA_TIME": ("CAMERA_TIME", camera_time),
        }
    )
    return FootprintData(dataset)


class TestSceneIdCamCamtimeWrite:
    """The CAM-CAMTIME runner must write a conformant product on the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid."""

    def test_write_data_product_is_conformant_with_camera_pixel_bounds(self, tmp_path):
        """A full classify + strict write succeeds; data lands on the (CAMERA_TIME, PSEUDOFOOTPRINT) grid."""
        # Targets the CAM-CAMTIME strict write; asserts data lands on the 2-D grid with int32 camera_pixel bounds.
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
        for name in ("Cloud_Fraction", "Scene_ID_ERBE", "Quality_Flag"):
            assert reopened[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        for name in ("Camera_Pixel_X_Min", "Camera_Pixel_X_Max", "Camera_Pixel_Y_Min", "Camera_Pixel_Y_Max"):
            assert name in reopened.variables
            assert reopened[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
            assert reopened[name].dtype == np.int32
        # Inclusive (min, max): the max endpoint is never below the min, elementwise across the grid.
        assert bool(np.all(reopened["Camera_Pixel_X_Max"].values >= reopened["Camera_Pixel_X_Min"].values))
        assert bool(np.all(reopened["Camera_Pixel_Y_Max"].values >= reopened["Camera_Pixel_Y_Min"].values))

    def test_write_drops_the_replaced_pixel_variables(self, tmp_path):
        """The retired center_pixel / start-stop / (min,max)-pair variables must not appear in the written product."""
        # Targets the write's variable pruning; asserts retired center_pixel and start/stop pixel vars are absent.
        footprint_data = _synthetic_camtime_footprint_data()
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(["erbe", "unfiltering"]))
        output_file = create_and_write_data_product_cam_camtime(footprint_data, "fmatch-cam-camtime.nc", tmp_path)

        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        for retired in (
            "Center_Pixel_X",
            "Center_Pixel_Y",
            "Camera_Pixel_X_Start",
            "Camera_Pixel_X_Stop",
            "Camera_Pixel_Y_Start",
            "Camera_Pixel_Y_Stop",
            # The replaced (min, max)-pair coordinates and their axis are gone.
            "Camera_Pixel_X",
            "Camera_Pixel_Y",
        ):
            assert retired not in reopened.variables
        assert "CAMERA_PIXEL_BOUNDS" not in reopened.dims

    def test_end_to_end_from_written_fmatch_file(self, tmp_path):
        """End-to-end guard (replaces the retired demo generator): a written FMATCH-CAM-CAMTIME file read by the real
        reader, classified, and written as a conformant SCENE-ID-CAM-CAMTIME product on the 2-D grid."""
        fmatch_path = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM_CAMTIME, n_footprints=6)
        footprint_data = FootprintData.from_fmatch_cam_camtime(fmatch_path)
        footprint_data.identify_scenes(scene_definitions=standard_scene_definitions(["erbe", "unfiltering"]))

        output_file = create_and_write_data_product_cam_camtime(footprint_data, fmatch_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        assert reopened.sizes["CAMERA_TIME"] == 6
        assert reopened["CAMERA_TIME"].dims == ("CAMERA_TIME",)
        for name in ("Camera_Pixel_X_Min", "Camera_Pixel_X_Max", "Camera_Pixel_Y_Min", "Camera_Pixel_Y_Max"):
            assert reopened[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        assert reopened["Scene_ID_ERBE"].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        assert "Scene_ID_ERBE" in reopened.variables
        assert "Center_Pixel_X" not in reopened.variables


class TestSceneIdImagerWrite:
    """The IMAGER runner produces a conformant SCENE-ID-IMAGER product that includes the TRMM classification."""

    def test_write_data_product_is_conformant_with_trmm(self, tmp_path):
        """A full IMAGER run + write succeeds under strict conformance and reports Scene_ID_TRMM as uint16."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER)
        footprint_data = run_scene_identification_imager(input_path)

        # create_and_write_data_product_imager writes with strict=True; reaching the assertions is the conformance
        # guarantee for scene_id_imager.yml (including the uint16 Scene_ID_TRMM).
        output_file = create_and_write_data_product_imager(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        assert reopened["Scene_ID_TRMM"].dtype == np.uint16
        for scene_id in ("Scene_ID_ERBE", "Scene_ID_Unfiltering", "Scene_ID_TRMM"):
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


class TestSceneIdImagerCamtimeWrite:
    """The IMAGER-CAMTIME runner writes a conformant product: the full TRMM classification on the 2-D grid plus the
    pseudo-footprint provenance passed straight through."""

    def test_end_to_end_is_conformant_with_trmm_and_passthrough(self, tmp_path):
        """A full IMAGER-CAMTIME run + strict write succeeds; Scene_ID_TRMM (uint16) and the passthrough identifiers
        land on the (CAMERA_TIME, PSEUDOFOOTPRINT) grid."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_CAMTIME, n_footprints=6)
        footprint_data = run_scene_identification_imager_camtime(input_path)

        # create_and_write_data_product_imager_camtime writes with strict=True; reaching the assertions is the
        # conformance guarantee for scene_id_imager_camtime.yml on the 2-D grid (including the uint16 Scene_ID_TRMM).
        output_file = create_and_write_data_product_imager_camtime(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        # All three classifications are present and hang on the 2-D grid; TRMM is uint16.
        assert reopened["Scene_ID_TRMM"].dtype == np.uint16
        for scene_id in ("Scene_ID_ERBE", "Scene_ID_Unfiltering", "Scene_ID_TRMM"):
            assert scene_id in reopened.variables
            assert reopened[scene_id].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
        assert reopened.sizes["CAMERA_TIME"] == 6
        assert reopened["CAMERA_TIME"].dims == ("CAMERA_TIME",)
        # The pseudo-footprint provenance (incl. lat/lon) passes straight through onto the grid.
        for name in (
            "Latitude",
            "Longitude",
            "Altitude",
            "PSF_Bbox_Lat_Min",
            "PSF_Bbox_Lon_Max",
            "Camera_Pixel_X_Min",
            "Camera_Pixel_X_Max",
            "Camera_Pixel_Y_Min",
            "Camera_Pixel_Y_Max",
        ):
            assert name in reopened.variables
            assert reopened[name].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")

    def test_written_product_has_no_undeclared_variables(self, tmp_path):
        """Reader intermediates (clear_area, surface_wind_u/v) and FMATCH-only vars must not leak into the product."""
        input_path = make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER_CAMTIME, n_footprints=6)
        footprint_data = run_scene_identification_imager_camtime(input_path)
        output_file = create_and_write_data_product_imager_camtime(footprint_data, input_path.name, tmp_path)

        definition = LiberaDataProductDefinition.from_yaml(IMAGER_CAMTIME_PRODUCT_DEFINITION_PATH)
        declared = set(definition.coordinates) | set(definition.variables)
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        undeclared = [name for name in reopened.variables if name not in declared]
        assert undeclared == []

    def test_collect_keeps_only_fmatch_imager_camtime(self):
        """collect_fmatch_imager_camtime_input_files keeps only FMATCH-IMAGER-CAMTIME files, dropping other products."""
        input_dir = "/dropbox/inputs"
        wanted = _libera_product_name(DataProductIdentifier.aux_fmatch_imager_camtime)
        other = _libera_product_name(DataProductIdentifier.aux_fmatch_imager)
        manifest = Manifest(
            manifest_type=ManifestType.INPUT,
            files=[f"{input_dir}/{name}" for name in (wanted, other, SSF_INPUT_NAME)],
        )
        selected = collect_fmatch_imager_camtime_input_files(manifest)
        assert selected == [f"{input_dir}/{wanted}"]


def _with_missing_igbp(fmatch_path, tmp_path, cell):
    """Copy an FMATCH fixture with FMATCH's integer fill written into ``igbp_surface_type`` at ``cell``."""
    with xr.open_dataset(fmatch_path) as opened:
        dataset = opened.load()
    dataset["IGBP_Surface_Type"].values[cell] = FMATCH_INTEGER_FILL_VALUE
    edited_dir = tmp_path / "edited"
    edited_dir.mkdir()
    edited_path = edited_dir / fmatch_path.name
    dataset.to_netcdf(edited_path)
    return edited_path


class TestMissingIgbpCode:
    """A footprint FMATCH has no IGBP code for is left unmatched; the rest of the granule still classifies.

    FMATCH writes its integer fill for a footprint with no usable IGBP pixels and for every padded cell of a
    camera-timescale grid, so a granule routinely contains some.
    """

    @pytest.mark.parametrize(
        ("mode", "variant"),
        [
            (OperationalMode.CAM, "cam"),
            (OperationalMode.IMAGER, "imager"),
        ],
    )
    def test_radiometer_footprint_is_unmatched_and_the_rest_classify(self, tmp_path, mode, variant):
        input_path = _with_missing_igbp(make_fmatch_product_fixture(tmp_path, mode, n_footprints=4), tmp_path, 1)

        dataset = run_scene_identification(input_path, RUNNER_CONFIGS[variant])._data

        assert np.isnan(dataset["Surface_Type"].values[1])
        for scene_definition in standard_scene_definitions(RUNNER_CONFIGS[variant].scene_types):
            name = f"Scene_ID_{scene_definition.type}"
            assert dataset[name].values[1] == 0, name
        assert (dataset["Scene_ID_ERBE"].values[[0, 2, 3]] > 0).all()

    @pytest.mark.parametrize(("mode", "variant"), [(OperationalMode.CAM_CAMTIME, "cam-camtime")])
    def test_padded_camera_cell_is_unmatched(self, tmp_path, mode, variant):
        input_path = _with_missing_igbp(make_fmatch_product_fixture(tmp_path, mode, n_footprints=3), tmp_path, (2, -1))

        dataset = run_scene_identification(input_path, RUNNER_CONFIGS[variant])._data

        assert dataset["Scene_ID_ERBE"].values[2, -1] == 0
        assert (dataset["Scene_ID_ERBE"].values[0] > 0).all()

    def test_written_product_carries_minus_one_for_the_unmatched_footprint(self, tmp_path):
        """The surface-type columns carry -1, with no _FillValue, for the footprint, and scene_id 0 flags it."""
        input_path = _with_missing_igbp(
            make_fmatch_product_fixture(tmp_path, OperationalMode.IMAGER, n_footprints=4), tmp_path, 1
        )
        footprint_data = run_scene_identification(input_path, RUNNER_CONFIGS["imager"])
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        written = create_and_write_data_product(footprint_data, input_path.name, output_dir, RUNNER_CONFIGS["imager"])

        with xr.open_dataset(written.path, mask_and_scale=False) as product:
            for name in ("IGBP_Surface_Type", "Surface_Type"):
                assert product[name].dtype == np.int8
                assert "_FillValue" not in product[name].attrs
                assert product[name].values[1] == -1
            assert product["Scene_ID_TRMM"].values[1] == 0
            assert (product["IGBP_Surface_Type"].values[[0, 2, 3]] > 0).all()


class TestSceneIdContainerRun:
    """Each SCENE-ID variant runs end to end the way its container does.

    The ``libera-utils-scene-id`` image's entrypoint is ``libera-utils scene-id <manifest>``, with the processing step
    setting ``LIBERA_SCENE_ID_VARIANT`` and ``PROCESSING_PATH``. Each case writes the variant's FMATCH product, lists it
    on an input manifest, runs the CLI with that environment, and checks the dropbox holds exactly the SCENE-ID product
    and an output manifest registering it.
    """

    @pytest.mark.parametrize("variant", sorted(RUNNER_CONFIGS))
    def test_cli_writes_product_and_output_manifest(self, tmp_path, monkeypatch, variant):
        runner_config = RUNNER_CONFIGS[variant]
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        dropbox = tmp_path / "dropbox"
        dropbox.mkdir()

        # FMATCH and SCENE-ID share variant names, so the FMATCH variant of the same name writes this variant's input.
        fmatch_path = make_fmatch_product_fixture(input_dir, FMATCH_RUNNER_CONFIGS[variant].mode, n_footprints=4)
        assert LiberaDataProductFilename.from_file_path(fmatch_path).data_product_id is runner_config.input_product_id
        input_manifest = Manifest(manifest_type=ManifestType.INPUT)
        input_manifest.add_files(fmatch_path)
        input_manifest_path = input_manifest.write(input_dir)

        monkeypatch.setenv(SCENE_ID_VARIANT_ENV, variant)
        monkeypatch.setenv("PROCESSING_PATH", str(dropbox))
        cli.main(["scene-id", str(input_manifest_path)])

        output_manifest_paths = list(dropbox.glob("*.json"))
        product_paths = list(dropbox.glob("*.nc"))
        assert len(output_manifest_paths) == 1
        assert len(product_paths) == 1
        assert sorted(dropbox.iterdir()) == sorted(output_manifest_paths + product_paths)

        output_manifest = Manifest.from_file(output_manifest_paths[0])
        assert output_manifest.manifest_type is ManifestType.OUTPUT
        assert output_manifest.ulid_code == input_manifest.ulid_code
        assert output_manifest.files == [str(product_paths[0])]

        product_filename = LiberaDataProductFilename.from_file_path(product_paths[0])
        assert product_filename.data_product_id is runner_config.output_product_id

        with (
            xr.open_dataset(fmatch_path) as fmatch_product,
            xr.open_dataset(product_paths[0], mask_and_scale=False) as product,
        ):
            assert product.attrs["InputGranules"] == fmatch_path.name
            assert product.attrs["algorithm_version"] == version()
            time_variable = runner_config.time_variable
            assert time_variable in product.coords
            assert product.sizes[time_variable] == fmatch_product.sizes[time_variable]
            # The fixture gives every footprint valid classification inputs, so every footprint is matched.
            for scene_definition in standard_scene_definitions(runner_config.scene_types):
                scene_ids = product[f"Scene_ID_{scene_definition.type}"].values
                assert (scene_ids > 0).all(), scene_definition.type
