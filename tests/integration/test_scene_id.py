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
from libera_utils.io.manifest import PROCESSING_STEP_ID_CONFIGURATION_KEY, Manifest, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification.scene_id import (
    _FMATCH_CAMTIME_PASSTHROUGH_VARIABLES,
    _FMATCH_GEOLOCATION_PASSTHROUGH_VARIABLES,
    standard_scene_definitions,
)
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS,
    scene_id_cli_handler,
)
from libera_utils.version import version

pytestmark = pytest.mark.integration


class TestSceneIdRunnerEndToEnd:
    """Full manifest-in / product-out end-to-end run of each SCENE-ID algorithm variant.

    Unlike the component unit tests (which call the reader/classifier directly), this drives the whole runner the way
    the pipeline does: write an FMATCH product through FMATCH's own assembly, build an input manifest, set
    ``PROCESSING_PATH``, name the processing step in the manifest, invoke the CLI handler, then assert on the output manifest and
    the written product. It is the guard that FMATCH's product names reach SCENE-ID, and that the variant dispatch,
    manifest/dropbox plumbing, strict product write, and output-manifest generation all work together.
    """

    @pytest.mark.parametrize("step", list(RUNNER_CONFIGS))
    def test_runner_writes_conformant_product_and_output_manifest(
        self, fmatch_product_file_factory, step, generate_input_manifest_local, tmp_path, monkeypatch
    ):
        """An FMATCH input runs end-to-end into a conformant SCENE-ID product under product names, with real matches.

        Asserts the output manifest registers one product with the variant's Libera filename and carries the input
        manifest's ULID, the dropbox holds only that product and the manifest, the product carries every variable its
        definition declares, every footprint matches a scene of each type the variant runs, and the product carries
        its FMATCH input's geolocation (and, on the camera timescale, its identifiers) unchanged.
        """
        runner_config = RUNNER_CONFIGS[step]

        # Step 1: build an input manifest pointing at an FMATCH product written through FMATCH's own assembly.
        inputs = tmp_path / "inputs"
        inputs.mkdir()
        fmatch_mode = OperationalMode(runner_config.input_product_id.value)
        fmatch_path = fmatch_product_file_factory(fmatch_mode, n_footprints=6, directory=inputs)
        input_manifest_path = generate_input_manifest_local(
            fmatch_path, configuration={PROCESSING_STEP_ID_CONFIGURATION_KEY: str(step)}
        )

        # Step 2: set the environment a processing step provides: the dropbox the runner writes products and the
        # output manifest into.
        dropbox_path = tmp_path / "dropbox"
        dropbox_path.mkdir()
        monkeypatch.setenv("PROCESSING_PATH", str(dropbox_path))

        # Step 3: run the algorithm end-to-end through the CLI handler, as the container entrypoint does.
        output_manifest_path = scene_id_cli_handler(
            argparse.Namespace(manifest=str(input_manifest_path), processing_step_id=None)
        )

        # Step 4: the output manifest exists, is an OUTPUT manifest, and registers exactly one product file.
        output_manifest = Manifest.from_file(output_manifest_path)
        assert output_manifest.manifest_type == ManifestType.OUTPUT
        assert len(output_manifest.files) == 1
        assert output_manifest.ulid_code == Manifest.from_file(input_manifest_path).ulid_code

        # Step 5: the registered product has the variant's Libera filename and carries its provenance.
        product_path = pathlib.Path(output_manifest.files[0])
        assert sorted(dropbox_path.iterdir()) == sorted([pathlib.Path(output_manifest_path), product_path])
        assert LiberaDataProductFilename.from_file_path(product_path).data_product_id is runner_config.output_product_id
        definition = LiberaDataProductDefinition.from_yaml(runner_config.product_definition_path)

        with xr.open_dataset(product_path) as product, xr.open_dataset(fmatch_path) as fmatch:
            assert product.attrs["InputGranules"] == fmatch_path.name
            assert product.attrs["algorithm_version"] == version()
            assert runner_config.time_variable in product.coords

            # Every declared variable is written.
            assert set(definition.variables) <= set(product.variables)

            # The fixture gives every footprint valid classification inputs, so every footprint matches a scene of each
            # type the variant runs (scene ID 0 is unmatched).
            for scene_definition in standard_scene_definitions(runner_config.scene_types):
                scene_ids = product[f"Scene_ID_{scene_definition.type}"].values
                assert (scene_ids > 0).all(), f"{step}: a footprint matched no {scene_definition.type} scene"

            # The product carries its FMATCH input's geolocation, and on the camera timescale its identifiers, unchanged.
            camtime = runner_config.time_variable == "CAMERA_TIME"
            passthrough = (
                _FMATCH_CAMTIME_PASSTHROUGH_VARIABLES if camtime else _FMATCH_GEOLOCATION_PASSTHROUGH_VARIABLES
            )
            for name in passthrough:
                np.testing.assert_allclose(product[name].values, fmatch[name].values, err_msg=name)
