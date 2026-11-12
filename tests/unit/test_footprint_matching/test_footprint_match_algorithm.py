"""Unit tests for ``footprint_match_algorithm``: the processing step dispatch, the ancillary inputs, the
camera-timescale inputs and the manifest-driven runner."""

import dataclasses
import functools
import logging
import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr
from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier, ProcessingStepIdentifier
from libera_utils.footprint_matching import footprint_match_algorithm, product
from libera_utils.footprint_matching.footprint_match_algorithm import (
    ANCILLARY_INPUTS_CONFIGURATION_KEY,
    ANCILLARY_PATH_ENV,
    CAM_CAMTIME_CONFIG,
    PROCESSING_STEP_ID_CONFIGURATION_KEY,
    RUNNER_CONFIGS,
    AncillaryManifestError,
    CloudFractionTimeMismatchError,
    MissingCloudFractionInputError,
    ProcessingStepIdMismatchError,
    RadiometerCloudFraction,
    _ancillary_source_file_paths,
    _load_cam_camtime_footprints,
    _load_imager_camtime_footprints,
    _materialize_ancillary_inputs,
    ancillary_inputs_from_manifest,
    create_and_write_data_product,
    load_cf_cam_cloud_fraction,
    load_l1b_radiometer_inputs,
    resolve_ancillary_inputs,
    resolve_runner_config,
    run_algorithm,
    run_footprint_matching,
)
from libera_utils.footprint_matching.product import MissingCamtimeVariablesError
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType

# The shortened real CF-CAM granule under tests/test_data/footprint_matching; its README says what was kept.
REAL_CF_CAM_FILE = "LIBERA_L2_CF-CAM_V0-1-0_20260712T032817_20260712T032827_R26294210139_subset.nc"

CAM = ProcessingStepIdentifier.aux_fmatch_cam
CAM_CAMTIME = ProcessingStepIdentifier.aux_fmatch_cam_camtime
IMAGER = ProcessingStepIdentifier.aux_fmatch_imager
IMAGER_CAMTIME = ProcessingStepIdentifier.aux_fmatch_imager_camtime


def _stage_empty_tree(root: Path, mode: OperationalMode) -> Path:
    """Stage one empty granule per reader active for ``mode``, enough for ``resolve_ancillary_inputs``."""
    for reader_key in ReaderRegistry.get_readers_for_mode(mode):
        (root / reader_key).mkdir(parents=True)
        (root / reader_key / "granule.nc").touch()
    return root


@pytest.fixture
def fake_aggregation(monkeypatch, fake_sources):
    """Aggregate from the fake readers whatever granules are staged, so a run needs no real ancillary files.

    The fake readers produce only their own columns, so the runner's product write is made to fill the columns of
    the mode's real readers; the real-granule run in ``tests/integration/test_fmatch.py`` writes every column.
    """
    monkeypatch.setattr(product, "build_sources", lambda mode, source_file_paths: fake_sources)
    monkeypatch.setattr(
        footprint_match_algorithm,
        "write_fmatch_product",
        functools.partial(product.write_fmatch_product, allow_placeholders=True),
    )


def _step_configuration(step: ProcessingStepIdentifier) -> dict:
    return {PROCESSING_STEP_ID_CONFIGURATION_KEY: str(step)}


class TestResolveRunnerConfig:
    """The input manifest's configuration, or the command line, names the processing step whose config runs."""

    @pytest.mark.parametrize("step", list(RUNNER_CONFIGS))
    def test_manifest_step_selects_its_config(self, step):
        manifest = Manifest(manifest_type=ManifestType.INPUT, configuration=_step_configuration(step))
        assert resolve_runner_config(manifest) is RUNNER_CONFIGS[step]

    @pytest.mark.parametrize("step", [IMAGER, str(IMAGER)])
    def test_command_line_step_selects_its_config(self, step):
        manifest = Manifest(manifest_type=ManifestType.INPUT)
        assert resolve_runner_config(manifest, step) is RUNNER_CONFIGS[IMAGER]

    def test_matching_steps_select_the_config(self):
        manifest = Manifest(manifest_type=ManifestType.INPUT, configuration=_step_configuration(CAM))
        assert resolve_runner_config(manifest, str(CAM)) is RUNNER_CONFIGS[CAM]

    def test_different_steps_raise(self):
        manifest = Manifest(manifest_type=ManifestType.INPUT, configuration=_step_configuration(CAM))
        with pytest.raises(ProcessingStepIdMismatchError, match="aux-fmatch-cam.*aux-fmatch-imager"):
            resolve_runner_config(manifest, IMAGER)

    def test_no_step_raises(self):
        with pytest.raises(ValueError, match="processing_step_id"):
            resolve_runner_config(Manifest(manifest_type=ManifestType.INPUT))

    @pytest.mark.parametrize("value", [str(ProcessingStepIdentifier.aux_scene_id_cam), "cam"])
    def test_step_that_is_not_fmatch_raises(self, value):
        manifest = Manifest(
            manifest_type=ManifestType.INPUT, configuration={PROCESSING_STEP_ID_CONFIGURATION_KEY: value}
        )
        with pytest.raises(ValueError, match="not an FMATCH processing step"):
            resolve_runner_config(manifest)


class TestCameraTimescaleInputs:
    """Both camera-timescale runners read their pseudo-footprints from CF-CAM-CAMTIME."""

    @pytest.mark.parametrize("step", [CAM_CAMTIME, IMAGER_CAMTIME])
    def test_input_product_is_cf_cam_camtime(self, step):
        assert RUNNER_CONFIGS[step].input_product_id is DataProductIdentifier.l2_cf_cam_camtime

    def test_cam_camtime_loader_aligns_cloud_fraction_with_footprints(
        self, cf_cam_camtime_file_factory, cf_cam_camtime_dataset_factory
    ):
        cf_file = cf_cam_camtime_file_factory(n_images=2, n_pixels_x=2, n_pixels_y=2, n_padded=3)

        footprints, cloud_fraction = _load_cam_camtime_footprints(cf_file)

        assert len(footprints) == 5
        expected = cf_cam_camtime_dataset_factory(n_images=2, n_pixels_x=2, n_pixels_y=2)["Cloud_Fraction"].values
        np.testing.assert_allclose(cloud_fraction, expected.ravel()[:5])

    def test_imager_camtime_loader_reads_no_cloud_fraction(self, cf_cam_camtime_file_factory):
        cf_file = cf_cam_camtime_file_factory(n_images=1)

        footprints, cloud_fraction = _load_imager_camtime_footprints(cf_file)

        assert len(footprints) == 9
        assert cloud_fraction is None

    def test_cam_camtime_loader_raises_without_cloud_fraction(self, cf_cam_camtime_file_factory):
        cf_file = cf_cam_camtime_file_factory(with_cloud_fraction=False)

        with pytest.raises(MissingCloudFractionInputError, match="Cloud_Fraction"):
            _load_cam_camtime_footprints(cf_file)

    def test_camera_timescale_config_without_loader_raises(self, cf_cam_camtime_file_factory):
        cf_file = cf_cam_camtime_file_factory()
        config = dataclasses.replace(CAM_CAMTIME_CONFIG, camera_footprint_loader=None)

        with pytest.raises(ValueError, match="camera_footprint_loader"):
            run_footprint_matching(cf_file, config)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("step", [CAM_CAMTIME, IMAGER_CAMTIME])
    def test_camtime_product_raises_until_pseudofootprints_carry_ephemeris(
        self, tmp_path, step, cf_cam_camtime_file_factory
    ):
        # TODO[LIBSDC-884]: write the product once CF-CAM-CAMTIME carries the camera ephemeris.
        cf_file = cf_cam_camtime_file_factory(n_images=2)
        config = RUNNER_CONFIGS[step]
        ancillary_inputs = resolve_ancillary_inputs(config.mode, _stage_empty_tree(tmp_path / "ancillary", config.mode))

        with pytest.raises(ValueError, match="Camera pseudo-footprints carry no L1B ephemeris"):
            create_and_write_data_product(cf_file, tmp_path, config, ancillary_inputs)

    def test_input_missing_pseudofootprint_variables_fails_the_run(self, tmp_path, cf_cam_camtime_dataset_factory):
        dataset = cf_cam_camtime_dataset_factory().drop_vars("PSF_Bbox_Flags")
        cf_file = tmp_path / "cf.nc"
        dataset.to_netcdf(cf_file)

        with pytest.raises(MissingCamtimeVariablesError, match="PSF_Bbox_Flags"):
            create_and_write_data_product(cf_file, tmp_path, RUNNER_CONFIGS[CAM_CAMTIME], {})


class TestResolveAncillaryInputs:
    """Ancillary resolution is strict by default: an unstaged reader stops the run."""

    MODE = OperationalMode.CAM

    def test_unset_ancillary_path_raises(self, monkeypatch):
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)
        with pytest.raises(ValueError, match=f"{ANCILLARY_INPUTS_CONFIGURATION_KEY}.*{ANCILLARY_PATH_ENV}"):
            resolve_ancillary_inputs(self.MODE)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_ancillary_inputs(self.MODE, tmp_path / "missing")

    @pytest.mark.parametrize("problem", ["missing", "empty"])
    def test_unstaged_reader_directory_raises_naming_the_reader(self, tmp_path, problem):
        root = _stage_empty_tree(tmp_path, self.MODE)
        (root / "NISE" / "granule.nc").unlink()
        if problem == "missing":
            (root / "NISE").rmdir()

        with pytest.raises(FileNotFoundError, match="NISE"):
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
        with pytest.raises(ValueError, match="ERA5_Surface"):
            _ancillary_source_file_paths({"ERA5_Surface": []})

    def test_s3_granule_raises(self):
        granule = S3Path("s3://bucket/igbp/tile.hdf")
        with pytest.raises(ValueError, match="'IGBP' is still in S3"):
            _ancillary_source_file_paths({"IGBP": [granule]})

    def test_local_granules_pass_through(self, tmp_path):
        granules = [tmp_path / "a.nc", tmp_path / "b.nc"]
        assert _ancillary_source_file_paths({"ERA5_Surface": granules}) == {"ERA5_Surface": granules}


class TestAncillaryInputsFromManifest:
    """The input manifest can list the ancillary granules, mapped to their readers."""

    MODE = OperationalMode.CAM

    @pytest.fixture
    def granules(self, tmp_path) -> dict[str, list[str]]:
        """One empty local granule per reader active for ``MODE``."""
        mapping = {}
        for reader_key in ReaderRegistry.get_readers_for_mode(self.MODE):
            granule = tmp_path / f"{reader_key}.nc"
            granule.touch()
            mapping[reader_key] = [str(granule)]
        return mapping

    @staticmethod
    def _manifest(mapping, files) -> Manifest:
        return Manifest(
            manifest_type=ManifestType.INPUT,
            files=files,
            configuration={ANCILLARY_INPUTS_CONFIGURATION_KEY: mapping},
        )

    def test_absent_key_returns_none(self):
        assert ancillary_inputs_from_manifest(Manifest(manifest_type=ManifestType.INPUT), self.MODE) is None

    def test_mapping_lists_every_active_reader(self, granules):
        manifest = self._manifest(granules, [path for paths in granules.values() for path in paths])

        resolved = ancillary_inputs_from_manifest(manifest, self.MODE)

        assert resolved == {key: [Path(paths[0])] for key, paths in sorted(granules.items())}

    def test_s3_granule_is_kept_as_an_s3_path(self, granules):
        granules["IGBP"] = ["s3://bucket/igbp/tile.hdf"]
        manifest = self._manifest(granules, [path for paths in granules.values() for path in paths])

        assert ancillary_inputs_from_manifest(manifest, self.MODE)["IGBP"] == [S3Path("s3://bucket/igbp/tile.hdf")]

    @pytest.mark.parametrize("mapping", [["ERA5_Surface"], {"ERA5_Surface": "granule.nc"}, {"ERA5_Surface": [1]}])
    def test_malformed_mapping_raises(self, mapping):
        with pytest.raises(AncillaryManifestError, match="must map each reader key"):
            ancillary_inputs_from_manifest(self._manifest(mapping, []), self.MODE)

    def test_reader_not_active_for_the_mode_raises(self, granules, tmp_path):
        (tmp_path / "CLDPIX.nc").touch()
        granules["CLDPIX"] = [str(tmp_path / "CLDPIX.nc")]
        manifest = self._manifest(granules, [path for paths in granules.values() for path in paths])

        with pytest.raises(AncillaryManifestError, match="CLDPIX"):
            ancillary_inputs_from_manifest(manifest, self.MODE)

    @pytest.mark.parametrize("problem", ["missing", "empty"])
    def test_active_reader_without_granule_raises(self, granules, problem):
        if problem == "missing":
            del granules["NISE"]
        else:
            granules["NISE"] = []
        manifest = self._manifest(granules, [path for paths in granules.values() for path in paths])

        with pytest.raises(AncillaryManifestError, match="no granule for readers \\['NISE'\\]"):
            ancillary_inputs_from_manifest(manifest, self.MODE)

    def test_granule_not_in_files_raises(self, granules):
        manifest = self._manifest(
            granules, [path for key, paths in granules.items() if key != "IGBP" for path in paths]
        )

        with pytest.raises(AncillaryManifestError, match="not in its files"):
            ancillary_inputs_from_manifest(manifest, self.MODE)


class TestMaterializeAncillaryInputs:
    """S3 ancillary granules are copied to local disk for the readers; local granules are not copied."""

    def test_s3_granule_is_copied_and_local_granule_passes_through(
        self, tmp_path, create_mock_bucket, write_file_to_s3
    ):
        bucket = create_mock_bucket()
        source = tmp_path / "tile.hdf"
        source.write_bytes(b"igbp granule")
        s3_granule = S3Path(f"s3://{bucket.name}/igbp/tile.hdf")
        write_file_to_s3(source, s3_granule)
        local_granule = tmp_path / "era5.nc"
        local_granule.touch()
        local_dir = tmp_path / "local"

        local = _materialize_ancillary_inputs({"IGBP": [s3_granule], "ERA5_Surface": [local_granule]}, local_dir)

        assert local == {"IGBP": [local_dir / "IGBP" / "tile.hdf"], "ERA5_Surface": [local_granule]}
        assert (local_dir / "IGBP" / "tile.hdf").read_bytes() == b"igbp granule"


class TestRunAlgorithm:
    """The manifest-driven runner: its failure paths and one product per input for every variant."""

    @pytest.fixture
    def dropbox(self, tmp_path, monkeypatch):
        path = tmp_path / "dropbox"
        path.mkdir()
        monkeypatch.setenv("PROCESSING_PATH", str(path))
        return path

    def test_raises_without_processing_path(
        self, monkeypatch, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        monkeypatch.delenv("PROCESSING_PATH", raising=False)
        manifest_path = generate_input_manifest_local(
            l1b_radiometer_file_factory(), configuration=_step_configuration(CAM)
        )

        with pytest.raises(ValueError, match="PROCESSING_PATH"):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("dropbox")
    def test_raises_without_processing_step(self, generate_input_manifest_local, l1b_radiometer_file_factory):
        manifest_path = generate_input_manifest_local(l1b_radiometer_file_factory())

        with pytest.raises(ValueError, match="No FMATCH processing step"):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("dropbox")
    def test_raises_without_ancillary_inputs(
        self, monkeypatch, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)
        manifest_path = generate_input_manifest_local(
            l1b_radiometer_file_factory(), configuration=_step_configuration(CAM)
        )

        with pytest.raises(ValueError, match=ANCILLARY_PATH_ENV):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("dropbox")
    def test_raises_when_manifest_has_no_input(
        self, tmp_path, monkeypatch, generate_input_manifest_local, cf_cam_camtime_file_factory
    ):
        config = RUNNER_CONFIGS[IMAGER]
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", config.mode)))
        manifest_path = generate_input_manifest_local(
            cf_cam_camtime_file_factory(), configuration=_step_configuration(IMAGER)
        )

        with pytest.raises(ValueError, match="No RAD-4CH input files"):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("dropbox")
    def test_cam_raises_when_manifest_has_no_cf_cam(
        self, tmp_path, monkeypatch, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        config = RUNNER_CONFIGS[CAM]
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", config.mode)))
        manifest_path = generate_input_manifest_local(
            l1b_radiometer_file_factory(), configuration=_step_configuration(CAM)
        )

        with pytest.raises(MissingCloudFractionInputError, match="lists no CF-CAM product"):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("step", [CAM, IMAGER])
    def test_writes_one_product_per_input(
        self,
        tmp_path,
        monkeypatch,
        dropbox,
        step,
        generate_input_manifest_local,
        l1b_radiometer_file_factory,
        cf_cam_file_factory,
    ):
        config = RUNNER_CONFIGS[step]
        ancillary_root = _stage_empty_tree(tmp_path / "ancillary", config.mode)
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(ancillary_root))
        input_file = l1b_radiometer_file_factory()
        manifest_files = [input_file]
        if config.cloud_fraction_product_id is not None:
            with xr.open_dataset(input_file) as l1b:
                manifest_files.append(cf_cam_file_factory(l1b["radiometer_time"].values))
        manifest_path = generate_input_manifest_local(*manifest_files, configuration=_step_configuration(step))

        output_manifest = Manifest.from_file(run_algorithm(manifest_path))

        assert output_manifest.manifest_type == ManifestType.OUTPUT
        assert len(output_manifest.files) == 1
        product_path = Path(str(output_manifest.files[0]))
        assert product_path.parent == dropbox
        assert LiberaDataProductFilename.from_file_path(product_path).data_product_id.value == config.mode.value
        with xr.open_dataset(product_path) as written:
            granules = written.attrs["InputGranules"].split(",")
        assert granules[: len(manifest_files)] == [path.name for path in manifest_files]
        assert len(granules) == len(manifest_files) + len(ReaderRegistry.get_readers_for_mode(config.mode))

    @pytest.mark.usefixtures("fake_aggregation", "dropbox")
    def test_cam_writes_cf_cam_cloud_fraction_matched_by_time(
        self, tmp_path, monkeypatch, generate_input_manifest_local, l1b_radiometer_file_factory, cf_cam_file_factory
    ):
        config = RUNNER_CONFIGS[CAM]
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", config.mode)))
        input_file = l1b_radiometer_file_factory(n_footprints=8)
        with xr.open_dataset(input_file) as l1b:
            times = l1b["radiometer_time"].values
        # CF-CAM covers only some footprints, listed out of time order, and writes one as fill.
        cf_cam_file = cf_cam_file_factory(times[[5, 1, 2]], np.array([50.0, 10.0, np.nan]))
        manifest_path = generate_input_manifest_local(input_file, cf_cam_file, configuration=_step_configuration(CAM))

        output_manifest = Manifest.from_file(run_algorithm(manifest_path))

        with xr.open_dataset(Path(str(output_manifest.files[0]))) as written:
            cloud_fraction = written["Cloud_Fraction_Camera"].values
        expected = np.full(8, np.nan, dtype=np.float32)
        expected[[1, 5]] = [10.0, 50.0]
        np.testing.assert_array_equal(cloud_fraction, expected)

    @pytest.mark.usefixtures("fake_aggregation", "dropbox")
    def test_manifest_ancillary_inputs_win_over_the_staged_tree(
        self, tmp_path, monkeypatch, caplog, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        config = RUNNER_CONFIGS[IMAGER]
        # The staged tree is missing every reader, so the run can only succeed from the manifest's granules.
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path / "missing"))
        granule_root = _stage_empty_tree(tmp_path / "listed", config.mode)
        ancillary = {
            key: [str(granule_root / key / "granule.nc")] for key in ReaderRegistry.get_readers_for_mode(config.mode)
        }
        input_file = l1b_radiometer_file_factory()
        manifest_path = generate_input_manifest_local(
            input_file,
            *(path for paths in ancillary.values() for path in paths),
            configuration={**_step_configuration(IMAGER), ANCILLARY_INPUTS_CONFIGURATION_KEY: ancillary},
        )

        # Task logging replaces the root handlers, which would drop caplog's.
        monkeypatch.setattr(footprint_match_algorithm, "configure_task_logging", lambda task_id: None)
        with caplog.at_level(logging.WARNING, logger=footprint_match_algorithm.__name__):
            output_manifest = Manifest.from_file(run_algorithm(manifest_path))

        assert len(output_manifest.files) == 1
        assert any(
            isinstance(record.msg, dict) and record.msg["msg"].startswith(f"Ignoring {ANCILLARY_PATH_ENV}")
            for record in caplog.records
        )

    @pytest.mark.usefixtures("fake_aggregation", "dropbox")
    def test_input_manifest_is_logged_as_structured_data(
        self, tmp_path, monkeypatch, caplog, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", OperationalMode.IMAGER)))
        input_file = l1b_radiometer_file_factory()
        manifest_path = generate_input_manifest_local(input_file, configuration=_step_configuration(IMAGER))

        # Task logging replaces the root handlers, which would drop caplog's.
        monkeypatch.setattr(footprint_match_algorithm, "configure_task_logging", lambda task_id: None)
        with caplog.at_level(logging.INFO, logger=footprint_match_algorithm.__name__):
            run_algorithm(manifest_path)

        manifest_records = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict) and record.msg["msg"] == "Step 1: Read the input manifest"
        ]
        assert manifest_records[0]["manifest"]["files"] == [str(input_file)]


class TestLoadCfCamCloudFraction:
    """CF-CAM is read once per run, sorted by radiometer time, and must carry Cloud_Fraction."""

    TIMES = np.datetime64("2026-06-11T00:00:00", "ns") + np.arange(6) * np.timedelta64(10, "ms")

    def test_no_files_raises(self):
        with pytest.raises(MissingCloudFractionInputError, match="lists no CF-CAM product"):
            load_cf_cam_cloud_fraction([])

    def test_file_without_cloud_fraction_raises(self, cf_cam_file_factory):
        cf_cam_file = cf_cam_file_factory(self.TIMES, cloud_fraction_variable="cam_cloud_fraction")

        with pytest.raises(MissingCloudFractionInputError, match="no Cloud_Fraction variable"):
            load_cf_cam_cloud_fraction([cf_cam_file])

    def test_files_are_combined_in_time_order_with_fill_as_nan(self, tmp_path, cf_cam_file_factory):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        late = cf_cam_file_factory(self.TIMES[3:], np.array([30.0, np.nan, 50.0]), directory=tmp_path / "a")
        early = cf_cam_file_factory(self.TIMES[:3], np.array([0.0, 10.0, 20.0]), directory=tmp_path / "b")

        cloud_fraction = load_cf_cam_cloud_fraction([late, early])

        np.testing.assert_array_equal(cloud_fraction.radiometer_time, self.TIMES)
        np.testing.assert_array_equal(
            cloud_fraction.cloud_fraction, np.array([0.0, 10.0, 20.0, 30.0, np.nan, 50.0], dtype=np.float32)
        )
        assert cloud_fraction.file_names == (late.name, early.name)

    def test_duplicate_times_across_files_raise(self, tmp_path, cf_cam_file_factory):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        first = cf_cam_file_factory(self.TIMES[:4], directory=tmp_path / "a")
        second = cf_cam_file_factory(self.TIMES[3:], directory=tmp_path / "b")

        with pytest.raises(CloudFractionTimeMismatchError, match="appear more than once"):
            load_cf_cam_cloud_fraction([first, second])


class TestRadiometerCloudFractionValuesFor:
    """CF-CAM rows match L1B footprints by exact radiometer time, and a misaligned input stops the run."""

    FOOTPRINTS = np.datetime64("2026-06-11T00:00:00", "ns") + np.arange(5) * np.timedelta64(10, "ms")

    @staticmethod
    def _cloud_fraction(times, values) -> RadiometerCloudFraction:
        return RadiometerCloudFraction(
            radiometer_time=np.asarray(times, dtype="datetime64[ns]"),
            cloud_fraction=np.asarray(values, dtype=np.float32),
            file_names=("cf.nc",),
        )

    def test_unmatched_footprints_are_nan_and_rows_outside_the_granule_are_ignored(self):
        before = self.FOOTPRINTS[0] - np.timedelta64(1, "s")
        after = self.FOOTPRINTS[-1] + np.timedelta64(1, "s")
        cloud_fraction = self._cloud_fraction(
            [before, self.FOOTPRINTS[1], self.FOOTPRINTS[3], after], [99.0, 10.0, 30.0, 99.0]
        )

        values = cloud_fraction.values_for(self.FOOTPRINTS)

        np.testing.assert_array_equal(values, np.array([np.nan, 10.0, np.nan, 30.0, np.nan], dtype=np.float32))

    def test_row_inside_the_granule_matching_no_footprint_raises(self):
        offset = self.FOOTPRINTS[2] + np.timedelta64(1, "us")
        cloud_fraction = self._cloud_fraction([self.FOOTPRINTS[1], offset], [10.0, 20.0])

        with pytest.raises(CloudFractionTimeMismatchError, match="match no usable L1B footprint time"):
            cloud_fraction.values_for(self.FOOTPRINTS)

    def test_fill_row_inside_the_granule_matching_no_footprint_is_ignored(self):
        offset = self.FOOTPRINTS[2] + np.timedelta64(1, "us")
        cloud_fraction = self._cloud_fraction([self.FOOTPRINTS[1], offset], [10.0, np.nan])

        values = cloud_fraction.values_for(self.FOOTPRINTS)

        assert values[1] == 10.0
        assert np.isnan(np.delete(values, 1)).all()

    @pytest.mark.parametrize("times", [[], [np.datetime64("2026-06-12T00:00:00", "ns")]])
    def test_no_footprint_matched_raises(self, times):
        cloud_fraction = self._cloud_fraction(times, [50.0] * len(times))

        with pytest.raises(CloudFractionTimeMismatchError, match="do not cover it"):
            cloud_fraction.values_for(self.FOOTPRINTS)


class TestRealCfCamFile:
    """The CF-CAM read on a real CF-CAM granule's layout: its time encoding, fill values and variable names."""

    @pytest.fixture
    def real_cf_cam(self, fmatch_real_data_path) -> Path:
        return fmatch_real_data_path / REAL_CF_CAM_FILE

    @pytest.fixture
    def renamed_cf_cam(self, real_cf_cam, tmp_path) -> Path:
        """A copy of the real granule with ``cam_cloud_fraction`` renamed ``Cloud_Fraction`` and nothing else changed."""
        copy = Path(shutil.copy(real_cf_cam, tmp_path / real_cf_cam.name))
        with netCDF4.Dataset(copy, "a") as dataset:
            dataset.renameVariable("cam_cloud_fraction", "Cloud_Fraction")
        return copy

    @staticmethod
    def _raw(path: Path, name: str) -> np.ndarray:
        """Read a variable's stored values, with no fill masking or time decoding."""
        with netCDF4.Dataset(path) as dataset:
            dataset.set_auto_maskandscale(False)
            return dataset[name][:]

    def test_granule_naming_its_cloud_fraction_cam_cloud_fraction_is_rejected(self, real_cf_cam):
        with pytest.raises(MissingCloudFractionInputError, match="no Cloud_Fraction variable"):
            load_cf_cam_cloud_fraction([real_cf_cam])

    def test_times_decode_from_nanoseconds_since_the_ccsds_epoch(self, renamed_cf_cam):
        stored = self._raw(renamed_cf_cam, "radiometer_time")

        cloud_fraction = load_cf_cam_cloud_fraction([renamed_cf_cam])

        expected = np.datetime64("1958-01-01T00:00:00", "ns") + stored.astype("timedelta64[ns]")
        np.testing.assert_array_equal(cloud_fraction.radiometer_time, expected)
        np.testing.assert_array_equal(cloud_fraction.cloud_fraction, self._raw(renamed_cf_cam, "Cloud_Fraction"))
        assert cloud_fraction.cloud_fraction.dtype == np.float32

    def test_fill_value_reads_as_nan(self, renamed_cf_cam):
        fill = np.float32(-999.0)
        with netCDF4.Dataset(renamed_cf_cam, "a") as dataset:
            assert dataset["Cloud_Fraction"].getncattr("_FillValue") == fill
            dataset["Cloud_Fraction"][2] = fill

        values = load_cf_cam_cloud_fraction([renamed_cf_cam]).cloud_fraction

        assert np.isnan(values[2])
        assert np.isfinite(np.delete(values, 2)).all()

    def test_rows_match_footprints_by_exact_time(self, renamed_cf_cam):
        cloud_fraction = load_cf_cam_cloud_fraction([renamed_cf_cam])
        cf_times = cloud_fraction.radiometer_time
        # L1B footprints at every CF-CAM time plus one 5 ms later, which CF-CAM does not cover.
        footprint_time = np.sort(np.concatenate([cf_times, cf_times + np.timedelta64(5, "ms")]))

        values = cloud_fraction.values_for(footprint_time)

        covered = np.isin(footprint_time, cf_times)
        np.testing.assert_array_equal(values[covered], cloud_fraction.cloud_fraction)
        assert np.isnan(values[~covered]).all()

    def test_row_with_no_footprint_at_its_time_raises(self, renamed_cf_cam):
        cloud_fraction = load_cf_cam_cloud_fraction([renamed_cf_cam])

        with pytest.raises(CloudFractionTimeMismatchError, match="match no usable L1B footprint time"):
            cloud_fraction.values_for(np.delete(cloud_fraction.radiometer_time, 6))


class TestCreateAndWriteCloudFractionInput:
    """The CF-CAM cloud fraction is required for FMATCH-CAM and refused for every other mode."""

    def test_cam_without_cloud_fraction_raises(self, tmp_path, l1b_radiometer_file_factory):
        with pytest.raises(MissingCloudFractionInputError, match="requires the CF-CAM cloud fraction"):
            create_and_write_data_product(l1b_radiometer_file_factory(), tmp_path, RUNNER_CONFIGS[CAM], {})

    def test_imager_with_cloud_fraction_raises(self, tmp_path, l1b_radiometer_file_factory, cf_cam_file_factory):
        l1b_file = l1b_radiometer_file_factory()
        times = load_l1b_radiometer_inputs(l1b_file)["RADIOMETER_TIME"]
        cloud_fraction = load_cf_cam_cloud_fraction([cf_cam_file_factory(times)])

        with pytest.raises(ValueError, match="takes no CF-CAM input"):
            create_and_write_data_product(l1b_file, tmp_path, RUNNER_CONFIGS[IMAGER], {}, cloud_fraction=cloud_fraction)
