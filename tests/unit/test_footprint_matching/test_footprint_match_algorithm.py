"""Unit tests for ``footprint_match_algorithm``: the LIBERA_FMATCH_VARIANT dispatch, ancillary staging, the
camera-timescale inputs and the manifest-driven runner."""

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cloudpathlib import S3Path

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching import footprint_match_algorithm, product
from libera_utils.footprint_matching.footprint_match_algorithm import (
    ANCILLARY_PATH_ENV,
    CAM_CAMTIME_CONFIG,
    FMATCH_VARIANT_ENV,
    RUNNER_CONFIGS,
    CloudFractionTimeMismatchError,
    MissingCloudFractionInputError,
    RadiometerCloudFraction,
    _ancillary_source_file_paths,
    _load_cam_camtime_footprints,
    _load_imager_camtime_footprints,
    create_and_write_data_product,
    load_cf_cam_cloud_fraction,
    load_l1b_radiometer_inputs,
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
    make_cf_cam_fixture,
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


def _write_input_manifest(directory: Path, *files: Path) -> Path:
    manifest = Manifest(manifest_type=ManifestType.INPUT)
    manifest.add_files(*files)
    return Path(str(manifest.write(directory)))


class TestResolveRunnerConfigFromEnv:
    """LIBERA_FMATCH_VARIANT selects the FMATCH runner config that the CLI handler runs."""

    @pytest.mark.parametrize("variant", ["cam", "cam-camtime", "imager", "imager-camtime"])
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
        expected = make_cf_cam_camtime_dataset(n_images=2, n_pixels_x=2, n_pixels_y=2)["Cloud_Fraction"].values
        np.testing.assert_allclose(cloud_fraction, expected.ravel()[:5])

    def test_imager_camtime_loader_reads_no_cloud_fraction(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=1)

        footprints, cloud_fraction = _load_imager_camtime_footprints(cf_file)

        assert len(footprints) == 9
        assert cloud_fraction is None

    def test_cam_camtime_loader_raises_without_cloud_fraction(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, with_cloud_fraction=False)

        with pytest.raises(MissingCloudFractionInputError, match="Cloud_Fraction"):
            _load_cam_camtime_footprints(cf_file)

    def test_camera_timescale_config_without_loader_raises(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path)
        config = dataclasses.replace(CAM_CAMTIME_CONFIG, camera_footprint_loader=None)

        with pytest.raises(ValueError, match="camera_footprint_loader"):
            run_footprint_matching(cf_file, config)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("variant", ["cam-camtime", "imager-camtime"])
    def test_camtime_product_raises_until_pseudofootprints_carry_ephemeris(self, tmp_path, variant):
        # TODO[LIBSDC-884]: write the product once CF-CAM-CAMTIME carries the camera ephemeris.
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=2)
        config = RUNNER_CONFIGS[variant]
        ancillary_inputs = resolve_ancillary_inputs(config.mode, _stage_empty_tree(tmp_path / "ancillary", config.mode))

        with pytest.raises(ValueError, match="Camera pseudo-footprints carry no L1B ephemeris"):
            create_and_write_data_product(cf_file, tmp_path, config, ancillary_inputs)

    def test_input_missing_pseudofootprint_variables_fails_the_run(self, tmp_path):
        dataset = make_cf_cam_camtime_dataset().drop_vars("PSF_Bbox_Flags")
        cf_file = tmp_path / "cf.nc"
        dataset.to_netcdf(cf_file)

        with pytest.raises(MissingCamtimeVariablesError, match="PSF_Bbox_Flags"):
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
        with pytest.raises(ValueError, match="'IGBP' is staged in S3"):
            _ancillary_source_file_paths({"IGBP": [granule]})

    def test_local_granules_pass_through(self, tmp_path):
        granules = [tmp_path / "a.nc", tmp_path / "b.nc"]
        assert _ancillary_source_file_paths({"ERA5_Surface": granules}) == {"ERA5_Surface": granules}


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

    @pytest.mark.usefixtures("dropbox")
    def test_cam_raises_when_manifest_has_no_cf_cam(self, tmp_path, monkeypatch):
        config = RUNNER_CONFIGS["cam"]
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", config.mode)))
        manifest_path = _write_input_manifest(tmp_path, make_l1b_radiometer_fixture(tmp_path))

        with pytest.raises(MissingCloudFractionInputError, match="lists no CF-CAM product"):
            run_algorithm(manifest_path, config)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("variant", ["cam", "imager"])
    def test_writes_one_product_per_input(self, tmp_path, monkeypatch, dropbox, variant):
        config = RUNNER_CONFIGS[variant]
        ancillary_root = _stage_empty_tree(tmp_path / "ancillary", config.mode)
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(ancillary_root))
        input_file = make_l1b_radiometer_fixture(tmp_path)
        manifest_files = [input_file]
        if config.cloud_fraction_product_id is not None:
            with xr.open_dataset(input_file) as l1b:
                manifest_files.append(make_cf_cam_fixture(tmp_path, l1b["radiometer_time"].values))
        manifest_path = _write_input_manifest(tmp_path, *manifest_files)

        output_manifest = Manifest.from_file(run_algorithm(manifest_path, config))

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
    def test_cam_writes_cf_cam_cloud_fraction_matched_by_time(self, tmp_path, monkeypatch):
        config = RUNNER_CONFIGS["cam"]
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(_stage_empty_tree(tmp_path / "ancillary", config.mode)))
        input_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)
        with xr.open_dataset(input_file) as l1b:
            times = l1b["radiometer_time"].values
        # CF-CAM covers only some footprints, listed out of time order, and writes one as fill.
        cf_cam_file = make_cf_cam_fixture(tmp_path, times[[5, 1, 2]], np.array([50.0, 10.0, np.nan]))
        manifest_path = _write_input_manifest(tmp_path, input_file, cf_cam_file)

        output_manifest = Manifest.from_file(run_algorithm(manifest_path, config))

        with xr.open_dataset(Path(str(output_manifest.files[0]))) as written:
            cloud_fraction = written["Cloud_Fraction_Camera"].values
        expected = np.full(8, np.nan, dtype=np.float32)
        expected[[1, 5]] = [10.0, 50.0]
        np.testing.assert_array_equal(cloud_fraction, expected)


class TestLoadCfCamCloudFraction:
    """CF-CAM is read once per run, sorted by radiometer time, and must carry Cloud_Fraction."""

    TIMES = np.datetime64("2026-06-11T00:00:00", "ns") + np.arange(6) * np.timedelta64(10, "ms")

    def test_no_files_raises(self):
        with pytest.raises(MissingCloudFractionInputError, match="lists no CF-CAM product"):
            load_cf_cam_cloud_fraction([])

    def test_file_without_cloud_fraction_raises(self, tmp_path):
        cf_cam_file = make_cf_cam_fixture(tmp_path, self.TIMES, cloud_fraction_variable="cam_cloud_fraction")

        with pytest.raises(MissingCloudFractionInputError, match="no Cloud_Fraction variable"):
            load_cf_cam_cloud_fraction([cf_cam_file])

    def test_files_are_combined_in_time_order_with_fill_as_nan(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        late = make_cf_cam_fixture(tmp_path / "a", self.TIMES[3:], np.array([30.0, np.nan, 50.0]))
        early = make_cf_cam_fixture(tmp_path / "b", self.TIMES[:3], np.array([0.0, 10.0, 20.0]))

        cloud_fraction = load_cf_cam_cloud_fraction([late, early])

        np.testing.assert_array_equal(cloud_fraction.radiometer_time, self.TIMES)
        np.testing.assert_array_equal(
            cloud_fraction.cloud_fraction, np.array([0.0, 10.0, 20.0, 30.0, np.nan, 50.0], dtype=np.float32)
        )
        assert cloud_fraction.file_names == (late.name, early.name)

    def test_duplicate_times_across_files_raise(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        first = make_cf_cam_fixture(tmp_path / "a", self.TIMES[:4])
        second = make_cf_cam_fixture(tmp_path / "b", self.TIMES[3:])

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


class TestCreateAndWriteCloudFractionInput:
    """The CF-CAM cloud fraction is required for FMATCH-CAM and refused for every other mode."""

    def test_cam_without_cloud_fraction_raises(self, tmp_path):
        with pytest.raises(MissingCloudFractionInputError, match="requires the CF-CAM cloud fraction"):
            create_and_write_data_product(make_l1b_radiometer_fixture(tmp_path), tmp_path, RUNNER_CONFIGS["cam"], {})

    def test_imager_with_cloud_fraction_raises(self, tmp_path):
        l1b_file = make_l1b_radiometer_fixture(tmp_path)
        times = load_l1b_radiometer_inputs(l1b_file)["RADIOMETER_TIME"]
        cloud_fraction = load_cf_cam_cloud_fraction([make_cf_cam_fixture(tmp_path, times)])

        with pytest.raises(ValueError, match="takes no CF-CAM input"):
            create_and_write_data_product(
                l1b_file, tmp_path, RUNNER_CONFIGS["imager"], {}, cloud_fraction=cloud_fraction
            )
