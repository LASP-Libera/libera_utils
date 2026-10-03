"""Integration tests for scene identification module.

These tests verify end-to-end workflows and interactions between components
of the scene identification system.
"""

import argparse
import pathlib

import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.scene_identification.scene_id_algorithm import scene_id_cam_cli_handler
from libera_utils.version import version

pytestmark = pytest.mark.integration


# Name of the CERES SSF placeholder input that SCENE-ID-CAM currently runs on, relative to the test-data root.
_SSF_INPUT_RELATIVE_PATH = "scene_id/CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"


class TestSceneIdCamRunnerEndToEnd:
    """Full manifest-in / product-out end-to-end run of the SCENE-ID-CAM algorithm.

    Unlike the component unit tests (which call the reader/classifier directly), this drives the whole runner the way
    the pipeline does: build an input manifest, set ``PROCESSING_PATH``, invoke the CLI handler, then assert on the
    output manifest and the written product. It is the "scrutinized example" integration test and the guard that the
    manifest/dropbox plumbing, strict product write, and output-manifest generation all work together.

    TODO[LIBSDC-855]: extend this end-to-end coverage to the SCENE-ID-CAM-CAMTIME runner once
    ``FootprintData.from_fmatch_cam_camtime`` is implemented and a real FMATCH-CAM-CAMTIME input fixture exists.
    """

    def test_runner_writes_conformant_product_and_output_manifest(
        self, generate_input_manifest_local, tmp_path, monkeypatch
    ):
        """A manifest referencing a CERES SSF input runs end-to-end and yields a conformant SCENE-ID-CAM product."""
        # Step 1: build an input manifest pointing at the CERES SSF placeholder input.
        input_manifest_path = generate_input_manifest_local(_SSF_INPUT_RELATIVE_PATH)

        # Step 2: set the processing dropbox the runner writes products and the output manifest into.
        dropbox_path = tmp_path / "dropbox"
        dropbox_path.mkdir()
        monkeypatch.setenv("PROCESSING_PATH", str(dropbox_path))

        # Step 3: run the algorithm end-to-end through the CLI handler.
        output_manifest_path = scene_id_cam_cli_handler(argparse.Namespace(manifest=str(input_manifest_path)))

        # Step 4: the output manifest exists, is an OUTPUT manifest, and registers exactly one product file.
        assert output_manifest_path.exists()
        output_manifest = Manifest.from_file(output_manifest_path)
        assert output_manifest.manifest_type == ManifestType.OUTPUT
        assert len(output_manifest.files) == 1

        # Step 5: the registered product exists, has a valid SCENE-ID-CAM Libera filename, and reopens.
        product_path = pathlib.Path(output_manifest.files[0].filename)
        assert product_path.exists()
        product_filename = LiberaDataProductFilename.from_file_path(product_path)
        assert product_filename.data_product_id is DataProductIdentifier.aux_scene_id_cam

        product = xr.open_dataset(product_path)
        # Provenance: InputGranules names the SSF input, algorithm_version comes from the installed package version.
        assert product.attrs["InputGranules"] == pathlib.Path(_SSF_INPUT_RELATIVE_PATH).name
        assert product.attrs["algorithm_version"] == version()
        # The product carries the time axis plus the CAM classifications (ERBE + unfiltering) and the quality flag.
        assert "RADIOMETER_TIME" in product.coords
        for expected_variable in ("scene_id_erbe", "scene_id_unfiltering", "Quality_Flag"):
            assert expected_variable in product.data_vars
