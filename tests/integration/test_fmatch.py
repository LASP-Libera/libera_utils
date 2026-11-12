"""Integration tests for the FMATCH runners: one full run per variant, as the ``libera-utils-fmatch`` container runs it.

Each test gives the run only what a processing step gives the container: the ``PROCESSING_PATH`` environment variable
and an input manifest, passed to the ``libera-utils fmatch <manifest>`` entrypoint. The manifest names the processing
step in its configuration and lists the input products and the shortened real ancillary granules from
``tests/test_data/footprint_matching``, mapped to their readers, so every reader opens a file in its distributed format.
Nothing in the runner is patched. The manifest, every input and the dropbox are on local disk or in mocked S3; the
runner copies S3 ancillary granules to local disk for the readers.

The shortened granules come from different days and regions, so most of a product's external columns are fill. These
tests check that a run gets from manifest to product, not the aggregated values, which the product tests check.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from cloudpathlib import S3Path

from libera_utils import cli
from libera_utils.footprint_matching.footprint_match_algorithm import (
    ANCILLARY_PATH_ENV,
    L1B_TIME_VARIABLE,
    RUNNER_CONFIGS,
)
from libera_utils.footprint_matching.product import L1B_PASSTHROUGH_VARIABLES
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.io.filenaming import MANIFEST_FILE_REGEX, LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.smart_open import smart_copy_file
from libera_utils.version import version

pytestmark = pytest.mark.integration

RADIOMETER_TIMESCALE_STEPS = [step for step, config in RUNNER_CONFIGS.items() if not config.is_camera_timescale]
CAMERA_TIMESCALE_STEPS = [step for step, config in RUNNER_CONFIGS.items() if config.is_camera_timescale]

# Leading L1B samples written as NaN, as the real files mark samples whose boresight misses the Earth.
N_OFF_EARTH = 2


@pytest.fixture(params=["local", "s3"])
def processing_area(request, tmp_path) -> tuple[Path | S3Path, Path | S3Path]:
    """The ``(input directory, dropbox)`` of a run, on local disk or in a mocked S3 bucket."""
    if request.param == "local":
        input_dir, dropbox = tmp_path / "inputs", tmp_path / "dropbox"
        input_dir.mkdir()
        dropbox.mkdir()
        return input_dir, dropbox
    bucket = request.getfixturevalue("create_mock_bucket")()
    return S3Path(f"s3://{bucket.name}/inputs"), S3Path(f"s3://{bucket.name}/dropbox")


@pytest.fixture
def fmatch_environment(monkeypatch, processing_area, fmatch_input_manifest):
    """Set up a processing step's environment, returning a function that writes a step's input manifest.

    The returned function sets ``PROCESSING_PATH``, leaves ``FMATCH_ANCILLARY_PATH`` unset so the manifest is the only
    source of ancillary granules, and writes the input manifest for the step with ``fmatch_input_manifest``.
    """
    input_dir, dropbox = processing_area

    def _stage(processing_step_id, *input_products: Path) -> Path | S3Path:
        monkeypatch.setenv("PROCESSING_PATH", str(dropbox))
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)
        return fmatch_input_manifest(processing_step_id, *input_products, input_dir=input_dir)

    return _stage


@pytest.mark.parametrize("step", RADIOMETER_TIMESCALE_STEPS)
def test_radiometer_timescale_run_writes_product_and_output_manifest(
    step,
    tmp_path,
    processing_area,
    fmatch_environment,
    fmatch_real_ancillary_granules,
    l1b_radiometer_file_factory,
    cf_cam_file_factory,
):
    """An L1B RAD-4CH manifest runs through the entrypoint to one FMATCH product and an output manifest naming it."""
    config = RUNNER_CONFIGS[step]
    _, dropbox = processing_area
    generated = tmp_path / "generated"
    generated.mkdir()
    l1b_file = l1b_radiometer_file_factory(n_footprints=20, n_invalid=N_OFF_EARTH, directory=generated)
    input_files = [l1b_file]
    on_earth = slice(N_OFF_EARTH, None)
    if config.cloud_fraction_product_id is not None:
        # CF-CAM covers every other on-Earth footprint, as the camera sees only part of the radiometer track.
        with xr.open_dataset(l1b_file) as l1b:
            covered_times = l1b[L1B_TIME_VARIABLE].values[on_earth][::2]
        input_files.append(cf_cam_file_factory(covered_times, directory=generated))
    manifest_path = fmatch_environment(step, *input_files)

    cli.main(["fmatch", str(manifest_path)])

    # The dropbox holds the output manifest and the one product it registers, and nothing else.
    written = list(dropbox.iterdir())
    output_manifest_paths = [path for path in written if MANIFEST_FILE_REGEX.match(path.name)]
    assert len(output_manifest_paths) == 1
    output_manifest = Manifest.from_file(output_manifest_paths[0])
    assert output_manifest.manifest_type == ManifestType.OUTPUT
    assert len(output_manifest.files) == 1
    product_path = output_manifest.files[0]
    assert sorted(path.name for path in written) == sorted([output_manifest_paths[0].name, Path(product_path).name])
    assert LiberaDataProductFilename.from_file_path(product_path).data_product_id.value == config.mode.value

    # Copied to local disk because xarray opens a file, not an S3 object.
    local_product = smart_copy_file(product_path, tmp_path / Path(product_path).name)
    with xr.open_dataset(local_product) as product, xr.open_dataset(l1b_file) as l1b:
        # Provenance names the L1B input and any CF-CAM input, then every listed granule, so each active reader was
        # handed its file.
        input_granules = product.attrs["InputGranules"].split(",")
        assert input_granules[: len(input_files)] == [path.name for path in input_files]
        assert sorted(input_granules[len(input_files) :]) == sorted(
            fmatch_real_ancillary_granules[reader_key].name
            for reader_key in ReaderRegistry.get_readers_for_mode(config.mode)
        )
        assert product.attrs["algorithm_version"] == version()

        # One product row per on-Earth L1B footprint, its time and pass-through columns copied from the L1B.
        np.testing.assert_array_equal(product[config.time_variable].values, l1b[L1B_TIME_VARIABLE].values[on_earth])
        for name in L1B_PASSTHROUGH_VARIABLES:
            np.testing.assert_array_equal(product[name].values, l1b[name].values[on_earth])

        # Aggregation ran on the listed granules: coverage and quality are set for every footprint, and the NISE
        # granule, which spans both polar grids, reaches some of them.
        assert np.isfinite(product["PSF_Coverage_Fraction"].values).all()
        assert np.isfinite(product["Quality_Flag"].values).all()
        assert np.isfinite(product["NISE_Open_Ocean"].values).any()

        # FMATCH-CAM carries the CF-CAM value of each footprint CF-CAM covers, and fill for the rest.
        if config.cloud_fraction_product_id is not None:
            expected = np.full(product.sizes[config.time_variable], np.nan, dtype=np.float32)
            expected[::2] = np.linspace(0.0, 100.0, expected[::2].size)
            np.testing.assert_allclose(product["Cloud_Fraction_Camera"].values, expected)


@pytest.mark.parametrize("step", CAMERA_TIMESCALE_STEPS)
def test_camera_timescale_run_stops_before_writing(
    step, tmp_path, processing_area, fmatch_environment, cf_cam_camtime_file_factory
):
    """A CF-CAM-CAMTIME manifest runs through the entrypoint and stops before writing, leaving the dropbox empty."""
    # TODO[LIBSDC-884]: assert the written product and output manifest once CF-CAM-CAMTIME carries the camera
    #  ephemeris, as the radiometer-timescale test does.
    _, dropbox = processing_area
    generated = tmp_path / "generated"
    generated.mkdir()
    cf_cam_camtime_file = cf_cam_camtime_file_factory(n_images=2, directory=generated)
    manifest_path = fmatch_environment(step, cf_cam_camtime_file)

    with pytest.raises(ValueError, match="Camera pseudo-footprints carry no L1B ephemeris"):
        cli.main(["fmatch", str(manifest_path)])

    assert not list(dropbox.iterdir())
