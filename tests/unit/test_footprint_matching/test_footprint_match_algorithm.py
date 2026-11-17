"""Unit tests for ``footprint_match_algorithm``: the LIBERA_FMATCH_VARIANT dispatch, ancillary selection from the
manifest, the camera-timescale inputs and the manifest-driven runner."""

import dataclasses
import shutil
from pathlib import Path
from unittest import mock

import netCDF4
import numpy as np
import pytest
import ulid
import xarray as xr

import libera_utils.footprint_matching.footprint_match_algorithm as footprint_match_algorithm
from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.footprint_match_algorithm import (
    CAM_CAMTIME_CONFIG,
    FMATCH_VARIANT_ENV,
    RUNNER_CONFIGS,
    CloudFractionTimeMismatchError,
    MissingAncillaryInputError,
    MissingCloudFractionInputError,
    RadiometerCloudFraction,
    _load_cam_camtime_footprints,
    _load_imager_camtime_footprints,
    create_and_write_data_product,
    load_cf_cam_cloud_fraction,
    load_l1b_radiometer_inputs,
    resolve_runner_config_from_env,
    run_algorithm,
    run_footprint_matching,
    select_ancillary_inputs,
    select_manifest_files_by_product_id,
)
from libera_utils.footprint_matching.product import MissingCamtimeVariablesError
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename, ManifestFilename
from libera_utils.io.manifest import Manifest, ManifestType
from tests.plugins.footprint_matching_fixtures import (
    ancillary_upstream_name,
    make_cf_cam_camtime_dataset,
    make_cf_cam_camtime_fixture,
    make_cf_cam_fixture,
    make_l1b_radiometer_fixture,
)


def _local_ancillary_inputs(placeholders: list[Path], mode: OperationalMode) -> dict[str, list[Path]]:
    """Select ``mode``'s ancillary inputs from local placeholders, as a run hands them to product assembly."""
    selected = select_ancillary_inputs(Manifest(manifest_type=ManifestType.INPUT, files=placeholders), mode)
    return {reader_key: [Path(path) for path in paths] for reader_key, paths in selected.items()}


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
    def test_camtime_product_raises_until_pseudofootprints_carry_ephemeris(
        self, tmp_path, variant, ancillary_placeholders
    ):
        # TODO[LIBSDC-884]: write the product once CF-CAM-CAMTIME carries the camera ephemeris.
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=2)
        config = RUNNER_CONFIGS[variant]
        ancillary_inputs = _local_ancillary_inputs(ancillary_placeholders, config.mode)

        with pytest.raises(ValueError, match="Camera pseudo-footprints carry no L1B ephemeris"):
            create_and_write_data_product(cf_file, tmp_path, config, ancillary_inputs)

    def test_input_missing_pseudofootprint_variables_fails_the_run(self, tmp_path):
        dataset = make_cf_cam_camtime_dataset().drop_vars("PSF_Bbox_Flags")
        cf_file = tmp_path / "cf.nc"
        dataset.to_netcdf(cf_file)

        with pytest.raises(MissingCamtimeVariablesError, match="PSF_Bbox_Flags"):
            create_and_write_data_product(cf_file, tmp_path, RUNNER_CONFIGS["cam-camtime"], {})


class TestSelectAncillaryInputs:
    """Each reader active for a mode is handed the manifest's files whose names parse to its ``PRODUCT_ID``."""

    @pytest.mark.parametrize("mode", list(OperationalMode))
    def test_every_active_reader_gets_its_product(self, mode):
        readers = ReaderRegistry.get_readers_for_mode(mode)
        files = [f"/inputs/{ancillary_upstream_name(reader.PRODUCT_ID)}" for reader in readers.values()]

        selected = select_ancillary_inputs(Manifest(manifest_type=ManifestType.INPUT, files=files), mode)

        assert selected == {
            key: [f"/inputs/{ancillary_upstream_name(reader.PRODUCT_ID)}"] for key, reader in readers.items()
        }

    def test_other_products_and_unparsable_names_are_ignored(self):
        mode = OperationalMode.CAM
        readers = ReaderRegistry.get_readers_for_mode(mode)
        wanted = [f"/inputs/{ancillary_upstream_name(reader.PRODUCT_ID)}" for reader in readers.values()]
        others = [
            "/inputs/" + ancillary_upstream_name(DataProductIdentifier.anc_era5_single_level),
            "/inputs/LIBERA_L1B_RAD-4CH_V1-2-3_20250102T120000_20250103T120000_R20270103T080910123456.nc",
            "/inputs/not_a_product.txt",
        ]

        selected = select_ancillary_inputs(Manifest(manifest_type=ManifestType.INPUT, files=others + wanted), mode)

        assert sorted(path for paths in selected.values() for path in paths) == sorted(wanted)

    def test_every_file_of_a_product_is_selected_in_manifest_order(self):
        mode = OperationalMode.IMAGER
        files = [
            f"/inputs/{ancillary_upstream_name(r.PRODUCT_ID)}"
            for r in ReaderRegistry.get_readers_for_mode(mode).values()
        ]
        second_hour = "/inputs/CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041016.nc"
        first_hour = "/inputs/" + ancillary_upstream_name(DataProductIdentifier.anc_ceres_noaa20_cldpix)

        selected = select_ancillary_inputs(
            Manifest(manifest_type=ManifestType.INPUT, files=[second_hour, *files]), mode
        )

        assert selected["CLDPIX"] == [second_hour, first_hour]

    _FLASH_SSF = "/inputs/FLASH_SSF_NOAA20-FM6-VIIRS_Version1B_405410.2026071200.nc"
    _CER_SSF = "/inputs/CER_SSF_NOAA20-FM6-VIIRS_alpha4_000000.2020040115.nc"

    def test_ceres_ssf_reader_takes_the_ssf_file_and_not_the_flash_ssf_file(self):
        mode = OperationalMode.IMAGER
        files = [
            f"/inputs/{ancillary_upstream_name(reader.PRODUCT_ID)}"
            for reader in ReaderRegistry.get_readers_for_mode(mode).values()
        ]

        selected = select_ancillary_inputs(
            Manifest(manifest_type=ManifestType.INPUT, files=[self._FLASH_SSF, *files]), mode
        )

        assert selected["FLASHFlux"] == [self._CER_SSF]

    def test_flash_ssf_file_alone_does_not_feed_the_ceres_ssf_reader(self):
        mode = OperationalMode.IMAGER
        files = [
            f"/inputs/{ancillary_upstream_name(reader.PRODUCT_ID)}"
            for key, reader in ReaderRegistry.get_readers_for_mode(mode).items()
            if key != "FLASHFlux"
        ]

        with pytest.raises(MissingAncillaryInputError, match=r"FLASHFlux \(CERES-NOAA20-SSF\)"):
            select_ancillary_inputs(Manifest(manifest_type=ManifestType.INPUT, files=[*files, self._FLASH_SSF]), mode)

    def test_every_missing_reader_is_named_in_one_error(self):
        mode = OperationalMode.CAM
        files = ["/inputs/" + ancillary_upstream_name(DataProductIdentifier.anc_nsidc_ice_snow)]

        with pytest.raises(MissingAncillaryInputError) as raised:
            select_ancillary_inputs(Manifest(manifest_type=ManifestType.INPUT, files=files), mode)

        message = str(raised.value)
        assert "IGBP (MODIS-IGBP)" in message
        assert "VIIRS_CLDPROP_D3 (VIIRS-NOAA20-CLOUD-PROPERTIES)" in message
        assert "NISE" not in message

    def test_active_reader_without_product_id_raises(self, set_mode_readers, monkeypatch):
        monkeypatch.setattr(ReaderRegistry.get("_fake_agg_mean"), "PRODUCT_ID", None)
        set_mode_readers(OperationalMode.CAM, ["_fake_agg_mean"])

        with pytest.raises(TypeError, match="'_fake_agg_mean' is active for FMATCH-CAM but declares no PRODUCT_ID"):
            select_ancillary_inputs(Manifest(manifest_type=ManifestType.INPUT, files=[]), OperationalMode.CAM)


class TestSelectManifestFilesByProductId:
    def test_manifest_file_in_a_manifest_is_skipped(self):
        manifest_name = str(
            ManifestFilename.from_filename_parts(ManifestType.INPUT, ulid.ULID(), basepath="/inputs").path
        )
        igbp = "/inputs/" + ancillary_upstream_name(DataProductIdentifier.anc_modis_igbp)

        selected = select_manifest_files_by_product_id(
            Manifest(manifest_type=ManifestType.INPUT, files=[manifest_name, igbp]),
            DataProductIdentifier.anc_modis_igbp,
        )

        assert selected == [igbp]


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

    def test_raises_before_reading_inputs_when_an_ancillary_product_is_missing(
        self, tmp_path, dropbox, ancillary_placeholders
    ):
        config = RUNNER_CONFIGS["cam"]
        igbp = ancillary_upstream_name(DataProductIdentifier.anc_modis_igbp)
        placeholders = [path for path in ancillary_placeholders if path.name != igbp]
        manifest_path = _write_input_manifest(tmp_path, make_l1b_radiometer_fixture(tmp_path), *placeholders)

        with (
            mock.patch.object(footprint_match_algorithm, "load_cf_cam_cloud_fraction") as load_cf_cam,
            pytest.raises(MissingAncillaryInputError, match=r"IGBP \(MODIS-IGBP\)"),
        ):
            run_algorithm(manifest_path, config)

        load_cf_cam.assert_not_called()
        assert not any(dropbox.iterdir())

    @pytest.mark.usefixtures("dropbox")
    def test_raises_when_manifest_has_no_input(self, tmp_path, ancillary_placeholders):
        config = RUNNER_CONFIGS["imager"]
        manifest_path = _write_input_manifest(tmp_path, make_cf_cam_camtime_fixture(tmp_path), *ancillary_placeholders)

        with pytest.raises(ValueError, match="No RAD-4CH input files"):
            run_algorithm(manifest_path, config)

    @pytest.mark.usefixtures("dropbox")
    def test_cam_raises_when_manifest_has_no_cf_cam(self, tmp_path, ancillary_placeholders):
        config = RUNNER_CONFIGS["cam"]
        manifest_path = _write_input_manifest(tmp_path, make_l1b_radiometer_fixture(tmp_path), *ancillary_placeholders)

        with pytest.raises(MissingCloudFractionInputError, match="lists no CF-CAM product"):
            run_algorithm(manifest_path, config)

    @pytest.mark.usefixtures("fake_aggregation")
    @pytest.mark.parametrize("variant", ["cam", "imager"])
    def test_writes_one_product_per_input(self, tmp_path, dropbox, variant, ancillary_placeholders):
        config = RUNNER_CONFIGS[variant]
        input_file = make_l1b_radiometer_fixture(tmp_path)
        manifest_files = [input_file]
        if config.cloud_fraction_product_id is not None:
            with xr.open_dataset(input_file) as l1b:
                manifest_files.append(make_cf_cam_fixture(tmp_path, l1b["radiometer_time"].values))
        manifest_path = _write_input_manifest(tmp_path, *manifest_files, *ancillary_placeholders)

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
    def test_cam_writes_cf_cam_cloud_fraction_matched_by_time(self, tmp_path, ancillary_placeholders):
        config = RUNNER_CONFIGS["cam"]
        input_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)
        with xr.open_dataset(input_file) as l1b:
            times = l1b["radiometer_time"].values
        # CF-CAM covers only some footprints, listed out of time order, and writes one as fill.
        cf_cam_file = make_cf_cam_fixture(tmp_path, times[[5, 1, 2]], np.array([50.0, 10.0, np.nan]))
        manifest_path = _write_input_manifest(tmp_path, input_file, cf_cam_file, *ancillary_placeholders)

        output_manifest = Manifest.from_file(run_algorithm(manifest_path, config))

        with xr.open_dataset(Path(str(output_manifest.files[0]))) as written:
            cloud_fraction = written["Cloud_Fraction_Camera"].values
        expected = np.full(8, np.nan, dtype=np.float32)
        expected[[1, 5]] = [10.0, 50.0]
        np.testing.assert_array_equal(cloud_fraction, expected)

    @pytest.mark.usefixtures("fake_aggregation")
    def test_s3_ancillary_granules_are_read_from_local_copies_removed_after_the_run(
        self, tmp_path, dropbox, ancillary_placeholders, write_file_to_s3
    ):
        config = RUNNER_CONFIGS["imager"]
        bucket = "s3://fmatch-ancillary-test"
        s3_placeholders = [write_file_to_s3(path, f"{bucket}/ancillary/{path.name}") for path in ancillary_placeholders]
        input_file = make_l1b_radiometer_fixture(tmp_path)
        manifest_path = _write_input_manifest(tmp_path, input_file, *s3_placeholders)
        seen: dict[str, list[Path]] = {}
        write_fmatch_product = footprint_match_algorithm.write_fmatch_product

        def _capture(*args, source_file_paths, **kwargs):
            for reader_key, paths in source_file_paths.items():
                assert all(isinstance(path, Path) and path.is_file() for path in paths), reader_key
            seen.update(source_file_paths)
            return write_fmatch_product(*args, source_file_paths=source_file_paths, **kwargs)

        with mock.patch.object(footprint_match_algorithm, "write_fmatch_product", side_effect=_capture):
            output_manifest = Manifest.from_file(run_algorithm(manifest_path, config))

        assert set(seen) == set(ReaderRegistry.get_readers_for_mode(config.mode))
        local_copies = [path for paths in seen.values() for path in paths]
        assert {path.name for path in local_copies} <= {path.name for path in s3_placeholders}
        assert not any(path.exists() for path in local_copies)
        with xr.open_dataset(Path(str(output_manifest.files[0]))) as written:
            granules = written.attrs["InputGranules"].split(",")
        assert set(granules[1:]) == {path.name for path in local_copies}


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


# The shortened real CF-CAM granule under tests/test_data/footprint_matching; its README says what was kept.
REAL_CF_CAM_FILE = "LIBERA_L2_CF-CAM_V0-1-0_20260712T032817_20260712T032827_R26294210139_subset.nc"


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
