"""Integration tests for the FMATCH runners and their product write path.

Every class but ``TestRunnerOutputThroughSceneId`` tests the FMATCH runners: what their unit tests
(``tests/unit/test_footprint_matching/test_footprint_match_algorithm.py``) do not, namely manifest input selection,
L1B geolocation reaching the written product, dropping non-finite L1B footprints and the FMATCH-IMAGER variable set.
``TestRunnerOutputThroughSceneId`` is the FMATCH to SCENE-ID chain test: it checks that a product the FMATCH runner
writes is read by SCENE-ID under the product names both declare, not the classification, since the fake readers supply
no classification inputs and every footprint is left unmatched. Ancillary aggregation runs through the fake readers
(``fake_aggregation``), so no real ancillary granule is opened.
"""

import argparse
from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.footprint_match_algorithm import (
    ANCILLARY_PATH_ENV,
    FMATCH_VARIANT_ENV,
    _collect_cloud_fraction_files,
    fmatch_cli_handler,
    select_manifest_files_by_product_id,
)
from libera_utils.footprint_matching.footprint_match_algorithm import (
    RUNNER_CONFIGS as FMATCH_RUNNER_CONFIGS,
)
from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification.scene_id import standard_scene_definitions
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS as SCENE_ID_RUNNER_CONFIGS,
)
from libera_utils.scene_identification.scene_id_algorithm import (
    create_and_write_data_product,
    run_scene_identification,
)
from tests.plugins.footprint_matching_fixtures import (
    FakeCatReader,
    FakeMeanReader,
    make_era5_pressure_netcdf_fixture,
    make_fmatch_product_fixture,
    make_l1b_radiometer_fixture,
)

pytestmark = pytest.mark.integration

# The FMATCH runners live in footprint_match_algorithm's registry. Bind the per-mode configs here so the test bodies
# below stay readable.
CAM_CONFIG = FMATCH_RUNNER_CONFIGS["cam"]
CAM_CAMTIME_CONFIG = FMATCH_RUNNER_CONFIGS["cam-camtime"]
IMAGER_CAMTIME_CONFIG = FMATCH_RUNNER_CONFIGS["imager-camtime"]
IMAGER_CONFIG = FMATCH_RUNNER_CONFIGS["imager"]


def _run_cli(variant, manifest_path, monkeypatch):
    """Run an FMATCH runner the way its container does: the CLI handler with LIBERA_FMATCH_VARIANT set."""
    monkeypatch.setenv(FMATCH_VARIANT_ENV, variant)
    return fmatch_cli_handler(argparse.Namespace(manifest=str(manifest_path)))


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
def staged_ancillary(tmp_path, monkeypatch, fake_aggregation):
    """An ancillary tree with one placeholder granule per registered reader, exported as FMATCH_ANCILLARY_PATH.

    The runner finds a granule for every reader it needs, and ``fake_aggregation`` aggregates from the fake readers
    instead of opening them.
    """
    root = tmp_path / "ancillary"
    for reader_key in ReaderRegistry.list_readers():
        reader_directory = root / reader_key
        reader_directory.mkdir(parents=True)
        (reader_directory / "granule.nc").touch()
    monkeypatch.setenv(ANCILLARY_PATH_ENV, str(root))
    return root


@pytest.fixture
def imager_uses_fake_readers(tmp_path, monkeypatch, set_mode_readers, fake_sources):
    """Run the FMATCH-IMAGER mode on the fake readers ``fake_sources`` pins for the CAM modes, plus ERA5 pressure.

    The ERA5 pressure-level reader is the one whose stacks declare the ``PRESSURE_LEVEL`` dimension FMATCH-IMAGER
    carries, so it runs for real over the plugin's ERA5 fixture; the fake readers stand in for the rest.
    """
    from libera_utils.footprint_matching import product  # noqa: PLC0415

    era5_directory = tmp_path / "era5_pressure"
    era5_directory.mkdir()
    era5_reader = ERA5PressureLevelReader([make_era5_pressure_netcdf_fixture(era5_directory)])
    set_mode_readers(
        OperationalMode.IMAGER,
        (FakeMeanReader.READER_KEY, FakeCatReader.READER_KEY, ERA5PressureLevelReader.READER_KEY),
    )
    monkeypatch.setattr(
        product,
        "build_sources",
        lambda mode, source_file_paths: (
            {**fake_sources, ERA5PressureLevelReader.READER_KEY: era5_reader}
            if mode is OperationalMode.IMAGER
            else fake_sources
        ),
    )


class TestRadiometerRunnerWorkflow:
    """The FMATCH-CAM runner carries L1B geolocation into its product and drops footprints L1B cannot geolocate."""

    def test_l1b_geolocation_reaches_the_product(self, tmp_path, dropbox, staged_ancillary, monkeypatch):
        """The pass-through columns must survive the whole runner path, not just assembly."""
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=8)
        manifest_path = _write_input_manifest(inputs, l1b_file)

        output_manifest = Manifest.from_file(_run_cli("cam", manifest_path, monkeypatch))

        with xr.open_dataset(output_manifest.files[0]) as product, xr.open_dataset(l1b_file) as l1b:
            assert product["Latitude"].values == pytest.approx(l1b["Latitude"].values, rel=1e-6)
            assert product["Solar_Zenith_Surface"].values == pytest.approx(l1b["Solar_Zenith_Surface"].values, rel=1e-6)

    def test_non_finite_l1b_footprints_are_excluded(self, tmp_path, dropbox, staged_ancillary, monkeypatch):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=10, n_invalid=4)
        manifest_path = _write_input_manifest(inputs, l1b_file)

        output_manifest = Manifest.from_file(_run_cli("cam", manifest_path, monkeypatch))

        with xr.open_dataset(output_manifest.files[0]) as product:
            assert product.sizes["RADIOMETER_TIME"] == 6


@pytest.mark.usefixtures("dropbox", "staged_ancillary", "imager_uses_fake_readers")
class TestImagerProduct:
    """FMATCH-IMAGER writes the CLDPIX/FLASHFlux fields alongside the ERA5 pressure-level fields."""

    def _run(self, tmp_path, monkeypatch):
        inputs = tmp_path / "inputs"
        inputs.mkdir(parents=True)
        l1b_file = make_l1b_radiometer_fixture(inputs, n_footprints=6)
        manifest_path = _write_input_manifest(inputs, l1b_file)
        return Manifest.from_file(_run_cli("imager", manifest_path, monkeypatch))

    def test_run_writes_the_cldpix_and_era5_variable_set(self, tmp_path, monkeypatch):
        output_manifest = self._run(tmp_path, monkeypatch)

        with xr.open_dataset(output_manifest.files[0]) as product:
            names = set(product.variables)
            assert any(name.startswith(CLDPIXReader.READER_KEY + "_") for name in names)
            assert any(name.startswith(ERA5PressureLevelReader.READER_KEY + "_") for name in names)

    def test_run_writes_the_fmatch_imager_product_id(self, tmp_path, monkeypatch):
        output_manifest = self._run(tmp_path, monkeypatch)

        written = LiberaDataProductFilename.from_file_path(output_manifest.files[0])
        assert written.data_product_id is DataProductIdentifier.aux_fmatch_imager


class TestManifestInputSelection:
    """A runner must take only its own primary input product out of a mixed manifest."""

    _INPUT_DIR = "/dropbox/inputs"
    # A staged ancillary granule: a real, non-Libera filename that must be ignored by manifest selection.
    _ANCILLARY_NAME = "MCD12Q1.A2023001.h09v05.061.hdf"

    def _manifest(self, *filenames: str) -> Manifest:
        return Manifest(manifest_type=ManifestType.INPUT, files=[f"{self._INPUT_DIR}/{name}" for name in filenames])

    def test_radiometer_runner_keeps_only_l1b_rad(self):
        wanted = _libera_product_name(DataProductIdentifier.l1b_rad)
        other = _libera_product_name(DataProductIdentifier.l1b_cam)
        manifest = self._manifest(wanted, other, self._ANCILLARY_NAME)

        selected = select_manifest_files_by_product_id(manifest, CAM_CONFIG.input_product_id)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]

    def test_camera_runner_keeps_only_its_cf_cam_camtime_input(self):
        """FMATCH-CAM-CAMTIME's primary input is the CF-CAM-CAMTIME product, not the raw L1B camera file."""
        wanted = _libera_product_name(DataProductIdentifier.l2_cf_cam_camtime)
        other = _libera_product_name(DataProductIdentifier.l1b_cam)
        manifest = self._manifest(other, wanted, self._ANCILLARY_NAME)

        selected = select_manifest_files_by_product_id(manifest, CAM_CAMTIME_CONFIG.input_product_id)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]

    def test_cam_runner_also_selects_the_cloud_fraction_product(self):
        """FMATCH-CAM takes an optional CF-CAM input alongside its L1B input."""
        cloud_fraction = _libera_product_name(DataProductIdentifier.l2_cf_cam)
        manifest = self._manifest(_libera_product_name(DataProductIdentifier.l1b_rad), cloud_fraction)

        selected = _collect_cloud_fraction_files(manifest, CAM_CONFIG)

        assert selected == [f"{self._INPUT_DIR}/{cloud_fraction}"]

    def test_imager_runners_take_no_cloud_fraction_input(self):
        """The IMAGER products do not declare Cloud_Fraction_Camera, so no cloud-fraction product is selected."""
        manifest = self._manifest(
            _libera_product_name(DataProductIdentifier.l1b_rad),
            _libera_product_name(DataProductIdentifier.l2_cf_cam),
        )

        for config in (IMAGER_CONFIG, IMAGER_CAMTIME_CONFIG):
            assert _collect_cloud_fraction_files(manifest, config) == []

    def test_fmatch_product_in_manifest_is_not_mistaken_for_an_input(self, tmp_path):
        """A FMATCH product staged alongside the inputs must not be re-ingested as one."""
        fmatch_product = make_fmatch_product_fixture(tmp_path, OperationalMode.CAM).name
        manifest = self._manifest(fmatch_product, _libera_product_name(DataProductIdentifier.l1b_rad))

        selected = select_manifest_files_by_product_id(manifest, DataProductIdentifier.l1b_rad)

        assert len(selected) == 1
        assert "FMATCH" not in selected[0]


class TestRunnerConfiguration:
    """Each runner must be wired to the right mode and input product."""

    @pytest.mark.parametrize(
        ("config", "mode", "input_product", "cloud_fraction_product"),
        [
            # Radiometer-timescale modes take an L1B RAD-4CH primary input; CAM additionally merges an optional CF-CAM.
            (CAM_CONFIG, OperationalMode.CAM, DataProductIdentifier.l1b_rad, DataProductIdentifier.l2_cf_cam),
            # Both camera-timescale modes read CF-CAM-CAMTIME (pseudo-footprints, and for CAM-CAMTIME the cloud
            # fraction, bundled upstream), so neither declares a separate cloud-fraction product.
            (
                CAM_CAMTIME_CONFIG,
                OperationalMode.CAM_CAMTIME,
                DataProductIdentifier.l2_cf_cam_camtime,
                None,
            ),
            (IMAGER_CONFIG, OperationalMode.IMAGER, DataProductIdentifier.l1b_rad, None),
            (IMAGER_CAMTIME_CONFIG, OperationalMode.IMAGER_CAMTIME, DataProductIdentifier.l2_cf_cam_camtime, None),
        ],
    )
    def test_runner_config(self, config, mode, input_product, cloud_fraction_product):
        assert config.mode is mode
        assert config.input_product_id is input_product
        assert config.cloud_fraction_product_id is cloud_fraction_product


@pytest.mark.usefixtures("dropbox", "staged_ancillary", "imager_uses_fake_readers")
class TestRunnerOutputThroughSceneId:
    """An FMATCH runner's product reads into SCENE-ID and writes a SCENE-ID product under product names.

    This checks that the names FMATCH writes line up with the names SCENE-ID reads and writes, not classification:
    the fake readers supply none of SCENE-ID's classification inputs (IGBP, FLASHFlux clear coverage, CLDPIX, ERA5
    winds), so those are fill and every footprint is left unmatched. ``tests/integration/test_scene_id.py`` covers real
    classification from an FMATCH product with valid inputs.
    """

    @pytest.mark.parametrize("variant", ["cam", "imager"])
    def test_runner_product_writes_a_scene_id_product(self, variant, tmp_path, monkeypatch):
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        manifest_path = _write_input_manifest(inputs, make_l1b_radiometer_fixture(inputs, n_footprints=8))
        fmatch_product_path = Manifest.from_file(_run_cli(variant, manifest_path, monkeypatch)).files[0]

        scene_id_config = SCENE_ID_RUNNER_CONFIGS[variant]
        footprint_data = run_scene_identification(fmatch_product_path, scene_id_config)
        output_dir = tmp_path / "scene_id"
        output_dir.mkdir()
        written = create_and_write_data_product(footprint_data, "fmatch.nc", output_dir, scene_id_config)

        definition = LiberaDataProductDefinition.from_yaml(scene_id_config.product_definition_path)
        with xr.open_dataset(written.path, mask_and_scale=False) as product:
            assert set(definition.variables) <= set(product.variables)
            for scene_definition in standard_scene_definitions(scene_id_config.scene_types):
                scene_ids = product[f"Scene_ID_{scene_definition.type}"].values
                assert np.count_nonzero(scene_ids) == 0, scene_definition.type
