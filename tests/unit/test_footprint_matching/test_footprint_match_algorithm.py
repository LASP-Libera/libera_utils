"""Unit tests for ``footprint_match_algorithm``: the LIBERA_FMATCH_VARIANT dispatch, ancillary staging, the
camera-timescale inputs and the manifest-driven runner."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching import product
from libera_utils.footprint_matching.footprint_match_algorithm import (
    ANCILLARY_PATH_ENV,
    CAM_CAMTIME_CONFIG,
    FMATCH_VARIANT_ENV,
    RUNNER_CONFIGS,
    _ancillary_source_file_paths,
    _load_cam_camtime_footprints,
    _load_imager_camtime_footprints,
    create_and_write_data_product,
    resolve_ancillary_inputs,
    resolve_runner_config_from_env,
    run_algorithm,
    run_footprint_matching,
)
from libera_utils.footprint_matching.product import MissingCamtimeVariablesError
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from tests.plugins.footprint_matching_fixtures import (
    make_cf_cam_camtime_dataset,
    make_cf_cam_camtime_fixture,
    make_l1b_radiometer_fixture,
)


def _stage_empty_tree(root: Path, mode: OperationalMode) -> Path:
    """Stage one empty granule per reader active for ``mode``, enough for ``resolve_ancillary_inputs``."""
    for reader_key in ReaderRegistry.get_readers_for_mode(mode):
        (root / reader_key).mkdir(parents=True)
        (root / reader_key / "granule.nc").touch()
    return root


@pytest.fixture
def fake_aggregation(monkeypatch, fake_sources):
    """Aggregate from the fake readers whatever granules are staged, so a run needs no real ancillary files."""
    monkeypatch.setattr(product, "build_sources", lambda mode, source_file_paths: fake_sources)


def _write_input_manifest(directory: Path, *files: Path) -> Path:
    manifest = Manifest(manifest_type=ManifestType.INPUT)
    manifest.add_files(*files)
    return Path(str(manifest.write(directory)))


class TestResolveRunnerConfigFromEnv:
    """LIBERA_FMATCH_VARIANT selects the FMATCH runner config that the CLI handler runs."""

    @pytest.mark.parametrize("variant", ["cam", "cam-camtime", "imager", "imager-camtime", "imager-flash"])
    def test_variant_selects_its_config(self, monkeypatch, variant):
        monkeypatch.setenv(FMATCH_VARIANT_ENV, variant)
        assert resolve_runner_config_from_env() is RUNNER_CONFIGS[variant]

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_unset_variant_raises(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(FMATCH_VARIANT_ENV, raising=False)
        else:
            monkeypatch.setenv(FMATCH_VARIANT_ENV, value)
        with pytest.raises(ValueError, match="not set"):
            resolve_runner_config_from_env()

    def test_unknown_variant_raises(self, monkeypatch):
        monkeypatch.setenv(FMATCH_VARIANT_ENV, "scene-id")
        with pytest.raises(ValueError, match="not a FMATCH variant"):
            resolve_runner_config_from_env()


class TestCameraTimescaleInputs:
    """Both camera-timescale runners read their pseudo-footprints from CF-CAM-CAMTIME."""

    @pytest.mark.parametrize("variant", ["cam-camtime", "imager-camtime"])
    def test_input_product_is_cf_cam_camtime(self, variant):
        assert RUNNER_CONFIGS[variant].input_product_id is DataProductIdentifier.l2_cf_cam_camtime

    def test_cam_camtime_loader_aligns_cloud_fraction_with_footprints(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=2, n_pixels_x=2, n_pixels_y=2, n_padded=3)

        footprints, cloud_fraction = _load_cam_camtime_footprints(cf_file)

        assert len(footprints) == 5
        expected = make_cf_cam_camtime_dataset(n_images=2, n_pixels_x=2, n_pixels_y=2)["cloud_fraction"].values
        np.testing.assert_allclose(cloud_fraction, expected.ravel()[:5])

    def test_imager_camtime_loader_reads_no_cloud_fraction(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=1)

        footprints, cloud_fraction = _load_imager_camtime_footprints(cf_file)

        assert len(footprints) == 9
        assert cloud_fraction is None

    def test_cam_camtime_loader_raises_without_cloud_fraction(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, with_cloud_fraction=False)

        with pytest.raises(ValueError, match="cloud_fraction"):
            _load_cam_camtime_footprints(cf_file)

    def test_camera_timescale_config_without_loader_raises(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path)
        config = dataclasses.replace(CAM_CAMTIME_CONFIG, camera_footprint_loader=None)

        with pytest.raises(ValueError, match="camera_footprint_loader"):
            run_footprint_matching(cf_file, config)

    @pytest.mark.usefixtures("fake_aggregation")
    def test_imager_camtime_writes_a_product_from_cf_cam_camtime(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=2)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        config = RUNNER_CONFIGS["imager-camtime"]
        ancillary_inputs = resolve_ancillary_inputs(config.mode, _stage_empty_tree(tmp_path / "ancillary", config.mode))

        written = create_and_write_data_product(cf_file, output_dir, config, ancillary_inputs)

        with xr.open_dataset(written.path) as dataset:
            assert dataset.sizes["CAMERA_TIME"] == 2
            assert dataset.sizes["PSEUDOFOOTPRINT"] == 9
            assert "cloud_fraction_camera" not in dataset
            assert cf_file.name in dataset.attrs["InputGranules"]

    def test_input_missing_pseudofootprint_variables_fails_the_run(self, tmp_path):
        dataset = make_cf_cam_camtime_dataset().drop_vars("psf_bbox_flags")
        cf_file = tmp_path / "cf.nc"
        dataset.to_netcdf(cf_file)

        with pytest.raises(MissingCamtimeVariablesError, match="psf_bbox_flags"):
            create_and_write_data_product(cf_file, tmp_path, RUNNER_CONFIGS["cam-camtime"], {})


class TestResolveAncillaryInputs:
    """Ancillary resolution is strict by default: an unstaged reader stops the run."""

    MODE = OperationalMode.CAM

    def test_unset_ancillary_path_raises(self, monkeypatch):
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)
        with pytest.raises(ValueError, match=ANCILLARY_PATH_ENV):
            resolve_ancillary_inputs(self.MODE)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_ancillary_inputs(self.MODE, tmp_path / "missing")

    @pytest.mark.parametrize("problem", ["missing", "empty"])
    def test_unstaged_reader_directory_raises_naming_the_reader(self, tmp_path, problem):
        root = _stage_empty_tree(tmp_path, self.MODE)
        (root / "nise" / "granule.nc").unlink()
        if problem == "missing":
            (root / "nise").rmdir()

        with pytest.raises(FileNotFoundError, match="nise"):
            resolve_ancillary_inputs(self.MODE, root)

    def test_staged_tree_maps_every_active_reader(self, tmp_path):
        root = _stage_empty_tree(tmp_path, self.MODE)

        resolved = resolve_ancillary_inputs(self.MODE, root)

        assert set(resolved) == set(ReaderRegistry.get_readers_for_mode(self.MODE))
        assert all(files == [root / key / "granule.nc"] for key, files in resolved.items())


class TestAncillarySourceFilePaths:
    """The staged inventory reaches assembly only as local granules for every reader."""

    def test_empty_inventory_raises(self):
        with pytest.raises(ValueError, match="No ancillary inputs"):
            _ancillary_source_file_paths({})

    def test_reader_without_granule_raises(self):
        with pytest.raises(ValueError, match="era5_surface"):
            _ancillary_source_file_paths({"era5_surface": []})

    def test_s3_granule_raises(self):
        granule = S3Path("s3://bucket/igbp/tile.hdf")
        with pytest.raises(ValueError, match="'igbp' is staged in S3"):
            _ancillary_source_file_paths({"igbp": [granule]})

    def test_local_granules_pass_through(self, tmp_path):
        granules = [tmp_path / "a.nc", tmp_path / "b.nc"]
        assert _ancillary_source_file_paths({"era5_surface": granules}) == {"era5_surface": granules}


class TestRunAlgorithm:
    """The manifest-driven runner: its failure paths and one product per input for every variant."""

    @pytest.fixture
    def dropbox(self, tmp_path, monkeypatch):
        path = tmp_path / "dropbox"
        path.mkdir()
        monkeypatch.setenv("PROCESSING_PATH", str(path))
        return path

    def test_raises_without_processing_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PROCESSING_PATH", raising=False)
        manifest_path = _write_input_manifest(tmp_path, make_l1b_radiometer_fixture(tmp_path))

        with pytest.raises(ValueError, match="PROCESSING_PATH"):
            run_algorithm(manifest_path, RUNNER_CONFIGS["cam"])

    @pytest.mark.usefixtures("dropbox")
    def test_raises_without_ancillary_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)
        manifest_path = _write_input_manifest(tmp_path, make_l1b_radiometer_fixture(tmp_path))

        with pytest.raises(ValueError, match=ANCILLARY_PATH_ENV):
            run_algorithm(manifest_path, RUNNER_CONFIGS["cam"])

    @pytest.mark.usefixtures("dropbox")
    def test_raises_when_manifest_has_no_input(self, tmp_path, monkeypatch):
        config = RUNNER_CONFIGS["imager"]
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", config.mode)))
        manifest_path = _write_input_manifest(tmp_path, make_cf_cam_camtime_fixture(tmp_path))

        with pytest.raises(ValueError, match="No RAD-4CH input files"):
            run_algorithm(manifest_path, config)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("variant", sorted(RUNNER_CONFIGS))
    def test_writes_one_product_per_input(self, tmp_path, monkeypatch, dropbox, variant):
        config = RUNNER_CONFIGS[variant]
        ancillary_root = _stage_empty_tree(tmp_path / "ancillary", config.mode)
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(ancillary_root))
        if config.is_camera_timescale:
            input_file = make_cf_cam_camtime_fixture(tmp_path, n_images=1)
        else:
            input_file = make_l1b_radiometer_fixture(tmp_path)
        manifest_path = _write_input_manifest(tmp_path, input_file)

        output_manifest = Manifest.from_file(run_algorithm(manifest_path, config))

        assert output_manifest.manifest_type == ManifestType.OUTPUT
        assert len(output_manifest.files) == 1
        product_path = Path(str(output_manifest.files[0]))
        assert product_path.parent == dropbox
        assert LiberaDataProductFilename.from_file_path(product_path).data_product_id.value == config.mode.value
        with xr.open_dataset(product_path) as written:
            granules = written.attrs["InputGranules"].split(",")
        assert granules[0] == input_file.name
        assert len(granules) == 1 + len(ReaderRegistry.get_readers_for_mode(config.mode))
