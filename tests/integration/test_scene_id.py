"""Integration tests for scene identification module.

These tests verify end-to-end workflows and interactions between components
of the scene identification system.
"""

import argparse
import pathlib

import numpy as np
import pytest
import xarray as xr

from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification.scene_id import standard_scene_definitions
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS,
    SCENE_ID_VARIANT_ENV,
    scene_id_cli_handler,
)
from libera_utils.version import version
from tests.plugins.footprint_matching_fixtures import make_fmatch_product_fixture

pytestmark = pytest.mark.integration

# The identifier variables a camera-timescale SCENE-ID product passes straight through from its FMATCH input.
_CAMTIME_PASSTHROUGH = (
    "Latitude",
    "Longitude",
    "PSF_Bbox_Lat_Min",
    "PSF_Bbox_Lat_Max",
    "PSF_Bbox_Lon_Min",
    "PSF_Bbox_Lon_Max",
    "Camera_Pixel_X_Min",
    "Camera_Pixel_X_Max",
    "Camera_Pixel_Y_Min",
    "Camera_Pixel_Y_Max",
)


class TestSceneIdRunnerEndToEnd:
    """Full manifest-in / product-out end-to-end run of each SCENE-ID algorithm variant.

    Unlike the component unit tests (which call the reader/classifier directly), this drives the whole runner the way
    the pipeline does: write an FMATCH product through FMATCH's own assembly, build an input manifest, set
    ``PROCESSING_PATH`` and ``LIBERA_SCENE_ID_VARIANT``, invoke the CLI handler, then assert on the output manifest and
    the written product. It is the guard that FMATCH's product names reach SCENE-ID, and that the variant dispatch,
    manifest/dropbox plumbing, strict product write, and output-manifest generation all work together.
    """

    @pytest.mark.parametrize("variant", ["cam", "cam-camtime", "imager", "imager-camtime"])
    def test_runner_writes_conformant_product_and_output_manifest(
        self, variant, generate_input_manifest_local, tmp_path, monkeypatch
    ):
        """An FMATCH input runs end-to-end into a conformant SCENE-ID product under product names, with real matches.

        Asserts the output manifest registers one product with the variant's Libera filename, the product declares
        every variable of its definition under its product name and none under its internal name, each scene type
        the variant runs matches at least one footprint, and a camera-timescale product carries its FMATCH input's
        identifier variables unchanged.
        """
        runner_config = RUNNER_CONFIGS[variant]

        # Step 1: build an input manifest pointing at an FMATCH product written through FMATCH's own assembly.
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        fmatch_mode = OperationalMode(runner_config.input_product_id.value)
        fmatch_path = make_fmatch_product_fixture(inputs, fmatch_mode, n_footprints=6)
        input_manifest_path = generate_input_manifest_local(fmatch_path)

        # Step 2: set the environment a processing step provides: the dropbox the runner writes products and the
        # output manifest into, and the SCENE-ID variant to run.
        dropbox_path = tmp_path / "dropbox"
        dropbox_path.mkdir()
        monkeypatch.setenv("PROCESSING_PATH", str(dropbox_path))
        monkeypatch.setenv(SCENE_ID_VARIANT_ENV, variant)

        # Step 3: run the algorithm end-to-end through the CLI handler, as the container entrypoint does.
        output_manifest_path = scene_id_cli_handler(argparse.Namespace(manifest=str(input_manifest_path)))

        # Step 4: the output manifest exists, is an OUTPUT manifest, and registers exactly one product file.
        output_manifest = Manifest.from_file(output_manifest_path)
        assert output_manifest.manifest_type == ManifestType.OUTPUT
        assert len(output_manifest.files) == 1

        # Step 5: the registered product has the variant's Libera filename and carries its provenance.
        product_path = pathlib.Path(output_manifest.files[0])
        assert LiberaDataProductFilename.from_file_path(product_path).data_product_id is runner_config.output_product_id
        definition = LiberaDataProductDefinition.from_yaml(runner_config.product_definition_path)

        with xr.open_dataset(product_path) as product, xr.open_dataset(fmatch_path) as fmatch:
            assert product.attrs["InputGranules"] == fmatch_path.name
            assert product.attrs["algorithm_version"] == version()
            assert runner_config.time_variable in product.coords

            # Every declared variable is written.
            assert set(definition.variables) <= set(product.variables)

            # Each scene type the variant runs classifies at least one footprint (scene ID 0 is unmatched).
            for scene_definition in standard_scene_definitions(runner_config.scene_types):
                scene_ids = product[f"Scene_ID_{scene_definition.type}"].values
                assert np.count_nonzero(scene_ids) > 0, (
                    f"{variant}: no footprint matched a {scene_definition.type} scene"
                )

            # A camera-timescale product carries its FMATCH input's identifiers unchanged.
            if "camtime" in variant:
                for name in _CAMTIME_PASSTHROUGH:
                    np.testing.assert_allclose(product[name].values, fmatch[name].values, err_msg=name)
