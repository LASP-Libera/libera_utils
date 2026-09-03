"""Integration tests for the FMATCH runners and their product write path.

These exercise the manifest/dropbox plumbing in ``libera_utils.footprint_matching._runner`` and the
concrete per-mode runners, including the actual product write. The happy-path tests in particular are
the guard that the FMATCH product definitions can be written under ``strict=True`` conformance from a
real L1B input, all the way from an input manifest to an output manifest.

Modelled on ``tests/integration/test_scene_id_runner.py``, which covers the same shape of workflow
for the SCENE-ID product family.
"""

from datetime import UTC, datetime

import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching._runner import ANCILLARY_PATH_ENV
from libera_utils.footprint_matching.fmatch_cam import RUNNER_CONFIG as CAM_CONFIG
from libera_utils.footprint_matching.fmatch_cam import algorithm as cam_algorithm
from libera_utils.footprint_matching.fmatch_cam_camtime import RUNNER_CONFIG as CAM_CAMTIME_CONFIG
from libera_utils.footprint_matching.fmatch_cam_camtime import algorithm as cam_camtime_algorithm
from libera_utils.footprint_matching.fmatch_imager import main as imager_main
from libera_utils.footprint_matching.fmatch_imager_camtime import (
    RUNNER_CONFIG as IMAGER_CAMTIME_CONFIG,
)
from libera_utils.footprint_matching.fmatch_imager_flash import RUNNER_CONFIG as IMAGER_FLASH_CONFIG
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestFileRecord, ManifestType
from libera_utils.scene_identification.scene_id_cam import run_scene_identification_cam
from tests.test_data.footprint_matching.fixtures import (
    make_fmatch_product_fixture,
    make_l1b_camera_fixture,
    make_l1b_radiometer_fixture,
)


def _libera_product_name(product_id: DataProductIdentifier) -> str:
    """Build a valid Libera data-product filename string for the given product id."""
    return LiberaDataProductFilename.from_filename_parts(
        product_name=product_id,
        version="V1-0-0",
        utc_start=datetime(2026, 6, 11, tzinfo=UTC),
        utc_end=datetime(2026, 6, 11, 23, 59, 59, tzinfo=UTC),
    ).path.name


def _write_input_manifest(directory, *file_paths) -> str:
    """Write an input manifest referencing ``file_paths`` and return its path."""
    manifest = Manifest(manifest_type=ManifestType.INPUT)
    manifest.add_files(*file_paths)
    return str(manifest.write(directory))


@pytest.fixture
def dropbox(tmp_path, monkeypatch):
    """A processing dropbox directory, exported as PROCESSING_PATH for the runners."""
    path = tmp_path / "dropbox"
    path.mkdir()
    monkeypatch.setenv("PROCESSING_PATH", str(path))
    return path


@pytest.fixture
def staged_ancillary(tmp_path, monkeypatch):
    """A staged ancillary tree covering every registered reader, exported as FMATCH_ANCILLARY_PATH."""
    root = tmp_path / "ancillary"
    for reader_key in ReaderRegistry.list_readers():
        reader_directory = root / reader_key
        reader_directory.mkdir(parents=True)
        (reader_directory / "granule.nc").write_text("not really a granule")
    monkeypatch.setenv(ANCILLARY_PATH_ENV, str(root))
    return root


class TestRadiometerRunnerWorkflow:
    """The FMATCH-CAM runner must go from an input manifest to a conformant product plus output manifest."""

    def test_full_workflow_writes_product_and_output_manifest(self, tmp_path, dropbox, staged_ancillary):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=12)
        manifest_path = _write_input_manifest(inputs, l1b_file)

        output_manifest_path = cam_algorithm(manifest_path)

        output_manifest = Manifest.from_file(output_manifest_path)
        assert output_manifest.manifest_type is ManifestType.OUTPUT
        assert len(output_manifest.files) == 1

        product_path = output_manifest.files[0].filename
        written = LiberaDataProductFilename.from_file_path(product_path)
        assert written.data_product_id is DataProductIdentifier.aux_fmatch_cam

        with xr.open_dataset(product_path) as product:
            assert product.sizes["RADIOMETER_TIME"] == 12
            assert product.attrs["input_files"] == l1b_file.name

    def test_l1b_geolocation_reaches_the_product(self, tmp_path, dropbox, staged_ancillary):
        """The pass-through columns must survive the whole runner path, not just assembly."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=8)
        manifest_path = _write_input_manifest(inputs, l1b_file)

        output_manifest = Manifest.from_file(cam_algorithm(manifest_path))

        with (
            xr.open_dataset(output_manifest.files[0].filename) as product,
            xr.open_dataset(l1b_file) as l1b,
        ):
            assert product["latitude"].values == pytest.approx(l1b["Latitude"].values, rel=1e-6)
            assert product["solar_zenith_angle"].values == pytest.approx(l1b["Solar_Zenith_Surface"].values, rel=1e-6)

    def test_non_finite_l1b_footprints_are_excluded(self, tmp_path, dropbox, staged_ancillary):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=10, n_invalid=4)
        manifest_path = _write_input_manifest(inputs, l1b_file)

        output_manifest = Manifest.from_file(cam_algorithm(manifest_path))

        with xr.open_dataset(output_manifest.files[0].filename) as product:
            assert product.sizes["RADIOMETER_TIME"] == 6


class TestCameraRunnerWorkflow:
    """The FMATCH-CAM-CAMTIME runner segments L1B camera images into pseudo-footprints."""

    def test_full_workflow_writes_product_and_output_manifest(self, tmp_path, dropbox, staged_ancillary):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        l1b_file = make_l1b_camera_fixture(inputs, n_images=2, n_pixels_x=4, n_pixels_y=4)
        manifest_path = _write_input_manifest(inputs, l1b_file)

        output_manifest = Manifest.from_file(cam_camtime_algorithm(manifest_path))

        assert len(output_manifest.files) == 1
        product_path = output_manifest.files[0].filename
        assert LiberaDataProductFilename.from_file_path(product_path).data_product_id is (
            DataProductIdentifier.aux_fmatch_cam_camtime
        )

        with xr.open_dataset(product_path) as product:
            # The product is a 2-D (CAMERA_TIME, FOOTPRINT) grid: one CAMERA_TIME per image (2 images), each image
            # segmented into 16 subsections at this coarse 4 x 4 fixture spacing.
            assert product.sizes["CAMERA_TIME"] == 2
            assert product.sizes["FOOTPRINT"] == 16
            assert product["CAMERA_TIME"].dims == ("CAMERA_TIME",)
            # Pixel-block provenance is real, not placeholder: a scene can be traced to its pixels. The block extent
            # is the four camera_pixel_{x,y}_{min,max} coordinates; the boresight pixel stays as center_pixel_x.
            assert "center_pixel_x" in product.variables
            for name in ("camera_pixel_x_min", "camera_pixel_x_max", "camera_pixel_y_min", "camera_pixel_y_max"):
                assert name in product.coords
                assert product[name].dims == ("CAMERA_TIME", "FOOTPRINT")

    def test_multiple_l1b_inputs_produce_multiple_products(self, tmp_path, dropbox, staged_ancillary):
        """A manifest may stage more than one L1B file; each yields its own product."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        first = make_l1b_camera_fixture(inputs, n_images=1, n_pixels_x=3, n_pixels_y=3)
        second_dir = tmp_path / "inputs2"
        second_dir.mkdir()
        second = make_l1b_camera_fixture(second_dir, n_images=2, n_pixels_x=3, n_pixels_y=3)
        manifest_path = _write_input_manifest(inputs, first, second)

        output_manifest = Manifest.from_file(cam_camtime_algorithm(manifest_path))

        assert len(output_manifest.files) == 2


class TestImagerProduct:
    """FMATCH-IMAGER writes the RBSP CLDPIX/SSF fields alongside the ERA5 pressure-level fields."""

    def _run(self, tmp_path):
        inputs = tmp_path / "inputs"
        inputs.mkdir(parents=True)
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=6)
        manifest_path = _write_input_manifest(inputs, l1b_file)
        return Manifest.from_file(imager_main([manifest_path]))

    def test_run_writes_the_rbsp_and_era5_variable_set(self, tmp_path, dropbox, staged_ancillary):
        output_manifest = self._run(tmp_path)

        with xr.open_dataset(output_manifest.files[0].filename) as product:
            names = set(product.variables)
            # The RBSP CLDPIX fields ...
            assert any(name.startswith("cldpix_") for name in names)
            # ... alongside the ERA5 pressure-level fields.
            assert any(name.startswith("era5_pressure_") for name in names)

    def test_run_writes_the_fmatch_imager_product_id(self, tmp_path, dropbox, staged_ancillary):
        output_manifest = self._run(tmp_path)

        written = LiberaDataProductFilename.from_file_path(output_manifest.files[0].filename)
        assert written.data_product_id is DataProductIdentifier.aux_fmatch_imager


class TestManifestInputSelection:
    """A runner must take only its own L1B product out of a mixed manifest."""

    _INPUT_DIR = "/dropbox/inputs"
    # A staged ancillary granule: a real, non-Libera filename that must be ignored by manifest selection.
    _ANCILLARY_NAME = "MCD12Q1.A2023001.h09v05.061.hdf"

    def _manifest(self, *filenames: str) -> Manifest:
        # Distinct checksums so the manifest keeps every record (it de-duplicates on identical checksums).
        return Manifest(
            manifest_type=ManifestType.INPUT,
            files=[
                ManifestFileRecord(filename=f"{self._INPUT_DIR}/{name}", checksum=str(index))
                for index, name in enumerate(filenames)
            ],
        )

    def test_radiometer_runner_keeps_only_l1b_rad(self):
        from libera_utils.footprint_matching._runner import select_manifest_files_by_product_id

        wanted = _libera_product_name(DataProductIdentifier.l1b_rad)
        other = _libera_product_name(DataProductIdentifier.l1b_cam)
        manifest = self._manifest(wanted, other, self._ANCILLARY_NAME)

        selected = select_manifest_files_by_product_id(manifest, CAM_CONFIG.l1b_input_product_id)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]

    def test_camera_runner_keeps_only_l1b_cam(self):
        from libera_utils.footprint_matching._runner import select_manifest_files_by_product_id

        wanted = _libera_product_name(DataProductIdentifier.l1b_cam)
        other = _libera_product_name(DataProductIdentifier.l1b_rad)
        manifest = self._manifest(other, wanted, self._ANCILLARY_NAME)

        selected = select_manifest_files_by_product_id(manifest, CAM_CAMTIME_CONFIG.l1b_input_product_id)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]

    def test_cam_runner_also_selects_the_cloud_fraction_product(self):
        """FMATCH-CAM takes an optional CF-CAM input alongside its L1B input."""
        from libera_utils.footprint_matching._runner import _collect_cloud_fraction_files

        cloud_fraction = _libera_product_name(DataProductIdentifier.l2_cf_cam)
        manifest = self._manifest(_libera_product_name(DataProductIdentifier.l1b_rad), cloud_fraction)

        selected = _collect_cloud_fraction_files(manifest, CAM_CONFIG)

        assert selected == [f"{self._INPUT_DIR}/{cloud_fraction}"]

    def test_imager_runners_take_no_cloud_fraction_input(self):
        """The IMAGER products do not declare cloud_fraction_camera, so none is selected."""
        from libera_utils.footprint_matching._runner import _collect_cloud_fraction_files

        manifest = self._manifest(
            _libera_product_name(DataProductIdentifier.l1b_rad),
            _libera_product_name(DataProductIdentifier.l2_cf_cam),
        )

        for config in (IMAGER_FLASH_CONFIG, IMAGER_CAMTIME_CONFIG):
            assert _collect_cloud_fraction_files(manifest, config) == []

    def test_fmatch_product_in_manifest_is_not_mistaken_for_an_input(self, tmp_path):
        """A FMATCH product staged alongside the inputs must not be re-ingested as one."""
        from libera_utils.footprint_matching._runner import select_manifest_files_by_product_id

        fmatch_product = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM).name
        manifest = self._manifest(fmatch_product, _libera_product_name(DataProductIdentifier.l1b_rad))

        selected = select_manifest_files_by_product_id(manifest, DataProductIdentifier.l1b_rad)

        assert len(selected) == 1
        assert "FMATCH" not in selected[0]


class TestRunnerErrorHandling:
    """Misconfiguration must fail loudly rather than writing a wrong product."""

    def test_missing_processing_path_raises(self, tmp_path, staged_ancillary, monkeypatch):
        monkeypatch.delenv("PROCESSING_PATH", raising=False)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        manifest_path = _write_input_manifest(inputs, make_l1b_radiometer_fixture(inputs))

        with pytest.raises(ValueError, match="PROCESSING_PATH"):
            cam_algorithm(manifest_path)

    def test_manifest_without_the_expected_l1b_product_raises(self, tmp_path, dropbox, staged_ancillary):
        """A CAM (radiometer) runner handed only camera L1B files has nothing to process."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        manifest_path = _write_input_manifest(inputs, make_l1b_camera_fixture(inputs, n_images=1))

        with pytest.raises(ValueError, match="No RAD-4CH input files"):
            cam_algorithm(manifest_path)

    def test_missing_ancillary_tree_does_not_block_the_run(self, tmp_path, dropbox, monkeypatch):
        """Ancillary staging is resolved non-strictly this milestone: nothing consumes it yet."""
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        manifest_path = _write_input_manifest(inputs, make_l1b_radiometer_fixture(inputs, n_footprints=4))

        output_manifest = Manifest.from_file(cam_algorithm(manifest_path))

        assert len(output_manifest.files) == 1


class TestRunnerConfiguration:
    """Each runner must be wired to the right mode and input product."""

    @pytest.mark.parametrize(
        ("config", "mode", "l1b_product", "cloud_fraction_product"),
        [
            (CAM_CONFIG, OperationalMode.CAM, DataProductIdentifier.l1b_rad, DataProductIdentifier.l2_cf_cam),
            (
                CAM_CAMTIME_CONFIG,
                OperationalMode.CAM_CAMTIME,
                DataProductIdentifier.l1b_cam,
                DataProductIdentifier.l2_cf_cam_camtime,
            ),
            (IMAGER_FLASH_CONFIG, OperationalMode.IMAGER_FLASH, DataProductIdentifier.l1b_rad, None),
            (IMAGER_CAMTIME_CONFIG, OperationalMode.IMAGER_CAMTIME, DataProductIdentifier.l1b_cam, None),
        ],
    )
    def test_runner_config(self, config, mode, l1b_product, cloud_fraction_product):
        assert config.mode is mode
        assert config.l1b_input_product_id is l1b_product
        assert config.cloud_fraction_product_id is cloud_fraction_product


class TestRunnerOutputNotYetConsumableBySceneId:
    """Runner output is intentionally non-operational end-to-end until the aggregation engine lands.

    The runners write only the L1B-derived columns with real values; the external-reader classification
    inputs (``igbp_surface_type``, ...) are placeholders (``TODO[LIBSDC-785]``). This tripwire pins that
    boundary: a *runner-written* FMATCH product is not yet consumable by SCENE-ID. It is the counterpart
    to the SCENE-ID integration tests, which use the synthetic ``make_fmatch_product_fixture`` with valid
    classification inputs.
    """

    @pytest.mark.xfail(
        raises=ValueError,
        strict=True,
        reason="TODO[LIBSDC-785]: the runner writes placeholder igbp_surface_type=0, which "
        "calculate_trmm_surface_type rejects. Real reader-derived classification inputs are not aggregated "
        "yet, so runner output is not consumable by SCENE-ID. Remove this marker when the engine lands.",
    )
    def test_runner_written_product_is_consumable_by_scene_id(self, tmp_path, dropbox, staged_ancillary):
        """A runner-written FMATCH-CAM product should classify through SCENE-ID once aggregation exists.

        Today ``run_scene_identification_cam`` raises ``ValueError`` in ``calculate_trmm_surface_type`` on the
        placeholder ``igbp_surface_type=0``; when ``TODO[LIBSDC-785]`` fills real classification inputs this
        test xpasses and (strict xfail) fails, forcing the marker's removal.
        """
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        manifest_path = _write_input_manifest(inputs, make_l1b_radiometer_fixture(inputs, n_footprints=8))

        output_manifest = Manifest.from_file(cam_algorithm(manifest_path))
        fmatch_product_path = output_manifest.files[0].filename

        footprint_data = run_scene_identification_cam(fmatch_product_path)

        scene_product = footprint_data.to_time_product("RADIOMETER_TIME")
        assert "Quality_Flag" in scene_product.data_vars
