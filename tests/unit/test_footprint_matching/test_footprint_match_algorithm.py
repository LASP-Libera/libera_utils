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
    CAM_CAMTIME_CONFIG,
    PROCESSING_STEP_ID_CONFIGURATION_KEY,
    RUNNER_CONFIGS,
    CloudFractionTimeMismatchError,
    ProcessingStepIdMismatchError,
    RadiometerCloudFraction,
    _ancillary_source_file_paths,
    _load_cam_camtime_footprints,
    _load_imager_camtime_footprints,
    _materialize_ancillary_inputs,
    ancillary_inputs_from_manifest,
    check_ancillary_granule_dates,
    create_and_write_data_product,
    load_cf_cam_cloud_fraction,
    load_l1b_radiometer_inputs,
    resolve_runner_config,
    run_algorithm,
    run_footprint_matching,
    select_manifest_files_by_product_id,
)
from libera_utils.footprint_matching.product import FmatchInputError
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


def _empty_granules(root: Path, mode: OperationalMode) -> dict[str, list[str]]:
    """Write one empty granule per reader active for ``mode``, and return the ``ancillary_inputs`` mapping to them."""
    root.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for reader_key in ReaderRegistry.get_readers_for_mode(mode):
        granule = root / f"{reader_key}.nc"
        granule.touch()
        mapping[reader_key] = [str(granule)]
    return mapping


def _ancillary_manifest(generate_input_manifest_local, root: Path, step: ProcessingStepIdentifier, *input_files):
    """Write an input manifest for ``step`` listing ``input_files`` and one empty granule per active reader."""
    ancillary = _empty_granules(root, RUNNER_CONFIGS[step].mode)
    return generate_input_manifest_local(
        *input_files,
        *(path for paths in ancillary.values() for path in paths),
        configuration={**_step_configuration(step), ANCILLARY_INPUTS_CONFIGURATION_KEY: ancillary},
    )


@pytest.fixture
def fake_aggregation(monkeypatch, fake_sources):
    """Aggregate from the fake readers whatever granules are staged, so a run needs no real ancillary files.

    The fake readers produce only their own columns, so the runner's product write is made to fill the columns of
    the mode's real readers; the real-granule run in ``tests/integration/test_fmatch.py`` writes every column. The
    staged granules are empty files with no date to read, so the granule date check is skipped
    (``TestCheckAncillaryGranuleDates`` tests it on real granules).
    """
    monkeypatch.setattr(product, "build_sources", lambda mode, source_file_paths: fake_sources)
    monkeypatch.setattr(footprint_match_algorithm, "check_ancillary_granule_dates", lambda *args: None)
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

        with pytest.raises(FmatchInputError, match="Cloud_Fraction"):
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
        # TODO[LIBSDC-943]: write the product once CF-CAM-CAMTIME carries the camera ephemeris.
        cf_file = cf_cam_camtime_file_factory(n_images=2)
        config = RUNNER_CONFIGS[step]
        ancillary_inputs = {
            key: [Path(path) for path in paths]
            for key, paths in _empty_granules(tmp_path / "ancillary", config.mode).items()
        }

        with pytest.raises(ValueError, match="Camera pseudo-footprints carry no L1B ephemeris"):
            create_and_write_data_product(cf_file, tmp_path, config, ancillary_inputs)

    def test_input_missing_pseudofootprint_variables_fails_the_run(self, tmp_path, cf_cam_camtime_dataset_factory):
        dataset = cf_cam_camtime_dataset_factory().drop_vars("PSF_Bbox_Flags")
        cf_file = tmp_path / "cf.nc"
        dataset.to_netcdf(cf_file)

        with pytest.raises(FmatchInputError, match="PSF_Bbox_Flags"):
            create_and_write_data_product(cf_file, tmp_path, RUNNER_CONFIGS[CAM_CAMTIME], {})


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

    def test_absent_key_raises(self):
        with pytest.raises(FmatchInputError, match='has no configuration\\["ancillary_inputs"\\]'):
            ancillary_inputs_from_manifest(Manifest(manifest_type=ManifestType.INPUT), self.MODE)

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
        with pytest.raises(FmatchInputError, match="must map each reader key"):
            ancillary_inputs_from_manifest(self._manifest(mapping, []), self.MODE)

    def test_reader_not_active_for_the_mode_raises(self, granules, tmp_path):
        (tmp_path / "CLDPIX.nc").touch()
        granules["CLDPIX"] = [str(tmp_path / "CLDPIX.nc")]
        manifest = self._manifest(granules, [path for paths in granules.values() for path in paths])

        with pytest.raises(FmatchInputError, match="CLDPIX"):
            ancillary_inputs_from_manifest(manifest, self.MODE)

    @pytest.mark.parametrize("problem", ["missing", "empty"])
    def test_active_reader_without_granule_raises(self, granules, problem):
        if problem == "missing":
            del granules["NISE"]
        else:
            granules["NISE"] = []
        manifest = self._manifest(granules, [path for paths in granules.values() for path in paths])

        with pytest.raises(FmatchInputError, match="no granule for readers \\['NISE'\\]"):
            ancillary_inputs_from_manifest(manifest, self.MODE)

    def test_granule_not_in_files_raises(self, granules):
        manifest = self._manifest(
            granules, [path for key, paths in granules.items() if key != "IGBP" for path in paths]
        )

        with pytest.raises(FmatchInputError, match="not in its files"):
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


class TestSelectManifestFilesByProductId:
    """Libera inputs are selected by product id; a misnamed Libera file stops the run rather than vanishing."""

    @staticmethod
    def _manifest(tmp_path, *names) -> Manifest:
        files = []
        for name in names:
            path = tmp_path / name
            path.touch()
            files.append(str(path))
        return Manifest(manifest_type=ManifestType.INPUT, files=files)

    def test_selects_the_product_and_skips_third_party_names(self, tmp_path, l1b_radiometer_file_factory):
        l1b_file = l1b_radiometer_file_factory()
        manifest = self._manifest(tmp_path, "NISE_AMSR2_20260610_subset.HDFEOS")
        manifest.add_files(str(l1b_file))

        selected = select_manifest_files_by_product_id(manifest, DataProductIdentifier.l1b_rad)

        assert selected == [str(l1b_file)]

    def test_libera_name_that_does_not_parse_raises(self, tmp_path):
        manifest = self._manifest(tmp_path, REAL_CF_CAM_FILE)

        with pytest.raises(FmatchInputError, match=REAL_CF_CAM_FILE):
            select_manifest_files_by_product_id(manifest, DataProductIdentifier.l2_cf_cam)


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
    def test_raises_without_ancillary_inputs(self, generate_input_manifest_local, l1b_radiometer_file_factory):
        manifest_path = generate_input_manifest_local(
            l1b_radiometer_file_factory(), configuration=_step_configuration(CAM)
        )

        with pytest.raises(FmatchInputError, match=ANCILLARY_INPUTS_CONFIGURATION_KEY):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("dropbox")
    def test_raises_when_manifest_has_no_input(
        self, tmp_path, generate_input_manifest_local, cf_cam_camtime_file_factory
    ):
        manifest_path = _ancillary_manifest(
            generate_input_manifest_local, tmp_path / "ancillary", IMAGER, cf_cam_camtime_file_factory()
        )

        with pytest.raises(FmatchInputError, match="lists no RAD-4CH file"):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("dropbox")
    def test_cam_raises_when_manifest_has_no_cf_cam(
        self, tmp_path, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        manifest_path = _ancillary_manifest(
            generate_input_manifest_local, tmp_path / "ancillary", CAM, l1b_radiometer_file_factory()
        )

        with pytest.raises(FmatchInputError, match="lists no CF-CAM product"):
            run_algorithm(manifest_path)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("step", [CAM, IMAGER])
    def test_writes_one_product_per_input(
        self,
        tmp_path,
        dropbox,
        step,
        generate_input_manifest_local,
        l1b_radiometer_file_factory,
        cf_cam_file_factory,
    ):
        config = RUNNER_CONFIGS[step]
        input_file = l1b_radiometer_file_factory()
        manifest_files = [input_file]
        if config.cloud_fraction_product_id is not None:
            with xr.open_dataset(input_file) as l1b:
                manifest_files.append(cf_cam_file_factory(l1b["radiometer_time"].values))
        manifest_path = _ancillary_manifest(
            generate_input_manifest_local, tmp_path / "ancillary", step, *manifest_files
        )

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
        self, tmp_path, generate_input_manifest_local, l1b_radiometer_file_factory, cf_cam_file_factory
    ):
        input_file = l1b_radiometer_file_factory(n_footprints=8)
        with xr.open_dataset(input_file) as l1b:
            times = l1b["radiometer_time"].values
        # CF-CAM covers only some footprints, listed out of time order, and writes one as fill.
        cf_cam_file = cf_cam_file_factory(times[[5, 1, 2]], np.array([50.0, 10.0, np.nan]))
        manifest_path = _ancillary_manifest(
            generate_input_manifest_local, tmp_path / "ancillary", CAM, input_file, cf_cam_file
        )

        output_manifest = Manifest.from_file(run_algorithm(manifest_path))

        with xr.open_dataset(Path(str(output_manifest.files[0]))) as written:
            cloud_fraction = written["Cloud_Fraction_Camera"].values
        expected = np.full(8, np.nan, dtype=np.float32)
        expected[[1, 5]] = [10.0, 50.0]
        np.testing.assert_array_equal(cloud_fraction, expected)

    @pytest.mark.usefixtures("fake_aggregation", "dropbox")
    def test_cam_ignores_a_cf_cam_row_at_a_dropped_footprint(
        self, tmp_path, generate_input_manifest_local, l1b_radiometer_file_factory, cf_cam_file_factory
    ):
        input_file = l1b_radiometer_file_factory(n_footprints=8, n_invalid=2)
        with xr.open_dataset(input_file) as l1b:
            times = l1b["radiometer_time"].values
        # CF-CAM has a value at the first footprint, which the L1B read drops for its non-finite geolocation.
        cf_cam_file = cf_cam_file_factory(times[[0, 3]], np.array([70.0, 30.0]))
        manifest_path = _ancillary_manifest(
            generate_input_manifest_local, tmp_path / "ancillary", CAM, input_file, cf_cam_file
        )

        output_manifest = Manifest.from_file(run_algorithm(manifest_path))

        with xr.open_dataset(Path(str(output_manifest.files[0]))) as written:
            cloud_fraction = written["Cloud_Fraction_Camera"].values
        expected = np.full(6, np.nan, dtype=np.float32)
        expected[1] = 30.0
        np.testing.assert_array_equal(cloud_fraction, expected)

    @pytest.mark.usefixtures("fake_aggregation", "dropbox")
    def test_input_manifest_is_logged_as_structured_data(
        self, tmp_path, monkeypatch, caplog, generate_input_manifest_local, l1b_radiometer_file_factory
    ):
        input_file = l1b_radiometer_file_factory()
        manifest_path = _ancillary_manifest(generate_input_manifest_local, tmp_path / "ancillary", IMAGER, input_file)

        # Task logging replaces the root handlers, which would drop caplog's.
        monkeypatch.setattr(footprint_match_algorithm, "configure_task_logging", lambda task_id: None)
        with caplog.at_level(logging.INFO, logger=footprint_match_algorithm.__name__):
            run_algorithm(manifest_path)

        manifest_records = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict) and record.msg["msg"] == "Step 1: Read the input manifest"
        ]
        assert manifest_records[0]["manifest"]["files"][0] == str(input_file)


def _hours(n: float) -> np.timedelta64:
    return np.timedelta64(int(n * 3600), "s").astype("timedelta64[ns]")


# Every reader that checks its granules' dates, by the key of its real granule in fmatch_real_ancillary_granules.
_WINDOWED_READERS = [
    "CLDPIX",
    "ERA5_Pressure",
    "ERA5_Surface",
    "FLASHFlux",
    "NISE",
    "VIIRS_AOD",
    "VIIRS_BRDF",
    "VIIRS_CLDPROP_D3",
]


class TestCheckAncillaryGranuleDates:
    """A granule dated more than its reader's window from the input's footprint times is rejected before aggregation.

    The real NISE granule covers 2026-06-10 (00:00 to 00:00 the next day), so footprints on 2026-06-11 see it as the
    previous day's granule and footprints on 2026-06-12 as two days old.
    """

    @pytest.fixture
    def nise(self, fmatch_real_ancillary_granules):
        return {"NISE": [fmatch_real_ancillary_granules["NISE"]]}

    @staticmethod
    def _day(date: str) -> np.ndarray:
        """Footprint times spanning 03:00 to 21:00 UTC of ``date``."""
        return np.array([f"{date}T03:00:00", f"{date}T12:00:00", f"{date}T21:00:00"], dtype="datetime64[ns]")

    @pytest.mark.parametrize("reader_key", _WINDOWED_READERS)
    def test_granule_covering_the_footprints_passes(self, fmatch_real_ancillary_granules, reader_key):
        path = fmatch_real_ancillary_granules[reader_key]
        start, end = ReaderRegistry.get(reader_key).granule_span(path)
        check_ancillary_granule_dates({reader_key: [path]}, np.array([start + (end - start) // 2]))

    @pytest.mark.parametrize("date", ["2026-06-09", "2026-06-10", "2026-06-11"])
    def test_same_and_adjacent_day_granules_pass(self, nise, date):
        """The NISE granule of 2026-06-10 is accepted for footprints on the day before, the day itself and the day
        after, which the one-day window accepts."""
        check_ancillary_granule_dates(nise, self._day(date))

    @pytest.mark.parametrize("date", ["2026-06-08", "2026-06-12"])
    def test_granule_two_days_away_raises(self, nise, fmatch_real_ancillary_granules, date):
        with pytest.raises(FmatchInputError, match="outside their reader's window") as excinfo:
            check_ancillary_granule_dates(nise, self._day(date))
        message = str(excinfo.value)
        assert f"NISE: {fmatch_real_ancillary_granules['NISE'].name} covers 2026-06-10T00:00:00Z" in message
        assert f"footprint times, {date}T03:00:00Z to {date}T21:00:00Z" in message

    @pytest.mark.parametrize("reader_key", _WINDOWED_READERS)
    @pytest.mark.parametrize(("granule_is", "offset_hours"), [("older", 1), ("newer", -1)])
    def test_every_reader_rejects_a_granule_beyond_its_window(
        self, fmatch_real_ancillary_granules, reader_key, granule_is, offset_hours
    ):
        """A footprint an hour beyond the reader's window from its granule's span rejects the granule."""
        path = fmatch_real_ancillary_granules[reader_key]
        reader = ReaderRegistry.get(reader_key)
        start, end = reader.granule_span(path)
        edge = end + reader.GRANULE_DATE_WINDOW if granule_is == "older" else start - reader.GRANULE_DATE_WINDOW

        with pytest.raises(FmatchInputError, match=f"{reader_key}: {path.name} covers"):
            check_ancillary_granule_dates({reader_key: [path]}, np.array([edge + _hours(offset_hours)]))

    @pytest.mark.parametrize("reader_key", _WINDOWED_READERS)
    @pytest.mark.parametrize(("granule_is", "offset_hours"), [("older", -1), ("newer", 1)])
    def test_every_reader_accepts_a_granule_just_inside_its_window(
        self, fmatch_real_ancillary_granules, reader_key, granule_is, offset_hours
    ):
        """A footprint an hour inside the reader's window from its granule's span accepts the granule."""
        path = fmatch_real_ancillary_granules[reader_key]
        reader = ReaderRegistry.get(reader_key)
        start, end = reader.granule_span(path)
        edge = end + reader.GRANULE_DATE_WINDOW if granule_is == "older" else start - reader.GRANULE_DATE_WINDOW

        check_ancillary_granule_dates({reader_key: [path]}, np.array([edge + _hours(offset_hours)]))

    def test_message_names_every_granule_outside_its_window(self, fmatch_real_ancillary_granules):
        granules = {key: [fmatch_real_ancillary_granules[key]] for key in ("NISE", "VIIRS_BRDF", "VIIRS_AOD")}

        # The NISE granule (2026-06-10) is within a day; VIIRS_BRDF (2026-06-02) and VIIRS_AOD (2026-09-27) are not.
        with pytest.raises(FmatchInputError, match="^2 ancillary granule") as excinfo:
            check_ancillary_granule_dates(granules, self._day("2026-06-11"))
        message = str(excinfo.value)
        assert f"VIIRS_BRDF: {fmatch_real_ancillary_granules['VIIRS_BRDF'].name}" in message
        assert f"VIIRS_AOD: {fmatch_real_ancillary_granules['VIIRS_AOD'].name}" in message
        assert "NISE:" not in message
        assert 'configuration["ancillary_inputs"]' in message

    @pytest.mark.parametrize("date", ["2010-01-01", "2026-06-11", "2040-12-31"])
    def test_igbp_granule_of_any_year_passes(self, fmatch_real_ancillary_granules, date):
        """IGBP's annual map (here 2024) is applied to later dates, so its date is never checked."""
        check_ancillary_granule_dates({"IGBP": [fmatch_real_ancillary_granules["IGBP"]]}, self._day(date))

    def test_reader_without_a_window_is_not_read(self, tmp_path):
        """A reader whose GRANULE_DATE_WINDOW is None is skipped without reading its granule's span."""
        empty = tmp_path / "unreadable.nc"
        empty.touch()
        assert ReaderRegistry.get("_fake_agg_mean").GRANULE_DATE_WINDOW is None

        check_ancillary_granule_dates({"_fake_agg_mean": [empty]}, self._day("2026-06-11"))

    @pytest.mark.parametrize(
        "footprint_time",
        [np.array([], dtype="datetime64[ns]"), np.array(["NaT", "NaT"], dtype="datetime64[ns]")],
        ids=["empty", "all-NaT"],
    )
    def test_no_valid_footprint_time_checks_nothing(self, nise, footprint_time):
        """With no valid footprint time there is nothing to date a granule against, so no granule is rejected.

        The same granule is rejected once a footprint time two days from it is present.
        """
        check_ancillary_granule_dates(nise, footprint_time)
        with pytest.raises(FmatchInputError, match="outside their reader's window"):
            check_ancillary_granule_dates(nise, np.concatenate([footprint_time, self._day("2026-06-12")]))

    def test_nat_footprint_times_are_ignored(self, nise):
        times = np.concatenate([self._day("2026-06-11"), np.array(["NaT"], dtype="datetime64[ns]")])
        check_ancillary_granule_dates(nise, times)

        with pytest.raises(FmatchInputError, match="NISE"):
            check_ancillary_granule_dates(nise, np.concatenate([self._day("2026-06-12"), times[-1:]]))

    def test_unreadable_granule_metadata_raises(self, tmp_path):
        granule = tmp_path / "AERDB_D3_no_coverage.nc"
        with netCDF4.Dataset(granule, "w"):
            pass

        with pytest.raises(ValueError, match="time_coverage_start"):
            check_ancillary_granule_dates({"VIIRS_AOD": [granule]}, self._day("2026-06-11"))

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("step", [CAM, IMAGER_CAMTIME])
    def test_run_checks_the_input_footprint_times(
        self, tmp_path, monkeypatch, step, l1b_radiometer_file_factory, cf_cam_file_factory, cf_cam_camtime_file_factory
    ):
        """``create_and_write_data_product`` checks the granules against the radiometer times of an L1B input, or the
        pseudo-footprints' camera times of a CF-CAM-CAMTIME input, before writing the product."""
        config = RUNNER_CONFIGS[step]
        cloud_fraction = None
        if config.is_camera_timescale:
            input_file = cf_cam_camtime_file_factory(n_images=2)
            with xr.open_dataset(input_file) as camtime:
                expected_times = np.unique(camtime["CAMERA_TIME"].values)
        else:
            input_file = l1b_radiometer_file_factory()
            expected_times = load_l1b_radiometer_inputs(input_file)["RADIOMETER_TIME"]
            cloud_fraction = load_cf_cam_cloud_fraction([cf_cam_file_factory(expected_times)])
        ancillary_inputs = {
            key: [Path(path) for path in paths]
            for key, paths in _empty_granules(tmp_path / "ancillary", config.mode).items()
        }
        checked = {}

        def reject(source_file_paths, footprint_time):
            checked.update(source_file_paths=source_file_paths, footprint_time=footprint_time)
            raise FmatchInputError("dated outside")

        monkeypatch.setattr(footprint_match_algorithm, "check_ancillary_granule_dates", reject)
        monkeypatch.setattr(footprint_match_algorithm, "write_fmatch_product", pytest.fail)

        with pytest.raises(FmatchInputError, match="dated outside"):
            create_and_write_data_product(input_file, tmp_path, config, ancillary_inputs, cloud_fraction=cloud_fraction)
        assert checked["source_file_paths"] == ancillary_inputs
        np.testing.assert_array_equal(np.unique(checked["footprint_time"]), expected_times)


class TestLiberaInputsFromS3:
    """Libera inputs in S3 are read through smart_open, with no local copy made by the runner."""

    def test_l1b_input_reads_from_s3(self, l1b_radiometer_file_factory, create_mock_bucket, write_file_to_s3):
        l1b_file = l1b_radiometer_file_factory(n_footprints=8)
        bucket = create_mock_bucket()
        s3_l1b = write_file_to_s3(l1b_file, S3Path(f"s3://{bucket.name}/l1b/{l1b_file.name}"))

        np.testing.assert_array_equal(
            load_l1b_radiometer_inputs(s3_l1b)["RADIOMETER_TIME"],
            load_l1b_radiometer_inputs(l1b_file)["RADIOMETER_TIME"],
        )

    def test_cf_cam_input_reads_from_s3(self, cf_cam_file_factory, create_mock_bucket, write_file_to_s3):
        times = np.datetime64("2026-06-11T00:00:00", "ns") + np.arange(3) * np.timedelta64(10, "ms")
        cf_cam_file = cf_cam_file_factory(times, np.array([0.0, 10.0, 20.0]))
        bucket = create_mock_bucket()
        s3_cf_cam = write_file_to_s3(cf_cam_file, S3Path(f"s3://{bucket.name}/cf/{cf_cam_file.name}"))

        cloud_fraction = load_cf_cam_cloud_fraction([s3_cf_cam])

        np.testing.assert_array_equal(cloud_fraction.radiometer_time, times)
        np.testing.assert_array_equal(cloud_fraction.cloud_fraction, [0.0, 10.0, 20.0])


class TestLoadCfCamCloudFraction:
    """CF-CAM is read once per run, sorted by radiometer time, and must carry Cloud_Fraction."""

    TIMES = np.datetime64("2026-06-11T00:00:00", "ns") + np.arange(6) * np.timedelta64(10, "ms")

    def test_no_files_raises(self):
        with pytest.raises(FmatchInputError, match="lists no CF-CAM product"):
            load_cf_cam_cloud_fraction([])

    def test_file_without_cloud_fraction_raises(self, cf_cam_file_factory):
        cf_cam_file = cf_cam_file_factory(self.TIMES, cloud_fraction_variable="cam_cloud_fraction")

        with pytest.raises(FmatchInputError, match="no Cloud_Fraction variable"):
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

        values = cloud_fraction.values_for(self.FOOTPRINTS, granule_time=self.FOOTPRINTS)

        np.testing.assert_array_equal(values, np.array([np.nan, 10.0, np.nan, 30.0, np.nan], dtype=np.float32))

    def test_row_inside_the_granule_matching_no_footprint_raises(self):
        offset = self.FOOTPRINTS[2] + np.timedelta64(1, "us")
        cloud_fraction = self._cloud_fraction([self.FOOTPRINTS[1], offset], [10.0, 20.0])

        with pytest.raises(CloudFractionTimeMismatchError, match="match no L1B time"):
            cloud_fraction.values_for(self.FOOTPRINTS, granule_time=self.FOOTPRINTS)

    def test_fill_row_inside_the_granule_matching_no_footprint_is_ignored(self):
        offset = self.FOOTPRINTS[2] + np.timedelta64(1, "us")
        cloud_fraction = self._cloud_fraction([self.FOOTPRINTS[1], offset], [10.0, np.nan])

        values = cloud_fraction.values_for(self.FOOTPRINTS, granule_time=self.FOOTPRINTS)

        assert values[1] == 10.0
        assert np.isnan(np.delete(values, 1)).all()

    def test_row_at_a_dropped_footprint_is_ignored(self):
        # The L1B read dropped the footprint at FOOTPRINTS[2], so the product has no row for CF-CAM's value there.
        kept = np.delete(self.FOOTPRINTS, 2)
        cloud_fraction = self._cloud_fraction([self.FOOTPRINTS[1], self.FOOTPRINTS[2]], [10.0, 20.0])

        values = cloud_fraction.values_for(kept, granule_time=self.FOOTPRINTS)

        np.testing.assert_array_equal(values, np.array([np.nan, 10.0, np.nan, np.nan], dtype=np.float32))

    def test_rows_only_at_dropped_footprints_cover_the_granule(self):
        kept = np.delete(self.FOOTPRINTS, 2)
        cloud_fraction = self._cloud_fraction([self.FOOTPRINTS[2]], [20.0])

        values = cloud_fraction.values_for(kept, granule_time=self.FOOTPRINTS)

        assert np.isnan(values).all()

    @pytest.mark.parametrize("times", [[], [np.datetime64("2026-06-12T00:00:00", "ns")]])
    def test_no_footprint_matched_raises(self, times):
        cloud_fraction = self._cloud_fraction(times, [50.0] * len(times))

        with pytest.raises(CloudFractionTimeMismatchError, match="do not cover it"):
            cloud_fraction.values_for(self.FOOTPRINTS, granule_time=self.FOOTPRINTS)


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
        with pytest.raises(FmatchInputError, match="no Cloud_Fraction variable"):
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

        values = cloud_fraction.values_for(footprint_time, granule_time=footprint_time)

        covered = np.isin(footprint_time, cf_times)
        np.testing.assert_array_equal(values[covered], cloud_fraction.cloud_fraction)
        assert np.isnan(values[~covered]).all()

    def test_row_with_no_footprint_at_its_time_raises(self, renamed_cf_cam):
        cloud_fraction = load_cf_cam_cloud_fraction([renamed_cf_cam])

        granule_time = np.delete(cloud_fraction.radiometer_time, 6)

        with pytest.raises(CloudFractionTimeMismatchError, match="match no L1B time"):
            cloud_fraction.values_for(granule_time, granule_time=granule_time)


class TestCreateAndWriteCloudFractionInput:
    """The CF-CAM cloud fraction is required for FMATCH-CAM and refused for every other mode."""

    def test_cam_without_cloud_fraction_raises(self, tmp_path, l1b_radiometer_file_factory):
        with pytest.raises(ValueError, match="requires the CF-CAM cloud fraction"):
            create_and_write_data_product(l1b_radiometer_file_factory(), tmp_path, RUNNER_CONFIGS[CAM], {})

    def test_imager_with_cloud_fraction_raises(self, tmp_path, l1b_radiometer_file_factory, cf_cam_file_factory):
        l1b_file = l1b_radiometer_file_factory()
        times = load_l1b_radiometer_inputs(l1b_file)["RADIOMETER_TIME"]
        cloud_fraction = load_cf_cam_cloud_fraction([cf_cam_file_factory(times)])

        with pytest.raises(ValueError, match="takes no CF-CAM input"):
            create_and_write_data_product(l1b_file, tmp_path, RUNNER_CONFIGS[IMAGER], {}, cloud_fraction=cloud_fraction)
