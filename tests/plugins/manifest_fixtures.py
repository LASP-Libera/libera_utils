"""Plugin module for mocking S3 buckets"""

import os
from pathlib import Path

import pytest
from cloudpathlib import S3Path

from libera_utils.constants import ManifestType, ProcessingStepIdentifier
from libera_utils.footprint_matching.footprint_match_algorithm import (
    ANCILLARY_INPUTS_CONFIGURATION_KEY,
    PROCESSING_STEP_ID_CONFIGURATION_KEY,
    RUNNER_CONFIGS,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.io.manifest import Manifest
from libera_utils.io.smart_open import smart_copy_file


@pytest.fixture
def generate_input_manifest_local(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data

    The returned function takes the files to list, relative to ``test_data`` or absolute, and an optional keyword-only
    ``configuration`` dict for the manifest. It writes the manifest under ``tmp_path / "processing"``, so it can be
    called once per test.
    """

    def _generate_input_manifest_local_with_files(*filenames, configuration=None):
        files = []
        for filename in filenames:
            files.append(test_data_path / filename)

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration=configuration or {})
        input_manifest.add_files(*files)

        os.mkdir(tmp_path / "processing")
        input_manifest_file_path = input_manifest.write(out_path=tmp_path / "processing")

        return input_manifest_file_path

    return _generate_input_manifest_local_with_files


@pytest.fixture
def generate_input_manifest_s3(test_data_path, create_mock_bucket, write_file_to_s3):
    """Generating test manifest from the data in test_data

    The returned function takes the files to upload and list, relative to ``test_data``, and an optional keyword-only
    ``configuration`` dict for the manifest.
    """

    def _generate_input_manifest_s3_with_files(*filenames, configuration=None):
        r_bucket = create_mock_bucket()

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration=configuration or {})

        for filename in filenames:
            s3_file_path = f"s3://{r_bucket.name}/{filename}"
            local_path = test_data_path / filename
            write_file_to_s3(local_path, s3_file_path)
            input_manifest.add_files(s3_file_path)

        d_bucket = create_mock_bucket()
        input_manifest_file_path = input_manifest.write(out_path=f"s3://{d_bucket.name}/processing")

        return input_manifest_file_path

    return _generate_input_manifest_s3_with_files


@pytest.fixture
def fmatch_input_manifest(fmatch_real_ancillary_granules):
    """Factory writing an FMATCH input manifest as a processing step hands it to the container.

    The returned function takes the processing step, the input products and the directory (local or S3) of the run's
    inputs. It copies the input products and the real ancillary granule of every reader active for the step's mode
    into that directory, lists them all in ``files``, names the step in ``configuration["processing_step_id"]`` and
    maps each reader to its granule in ``configuration["ancillary_inputs"]``. It returns the written manifest's path.
    """

    def _write(
        processing_step_id: ProcessingStepIdentifier, *input_products: Path, input_dir: Path | S3Path
    ) -> Path | S3Path:
        mode = RUNNER_CONFIGS[processing_step_id].mode
        ancillary_inputs = {
            reader_key: [str(smart_copy_file(granule, input_dir / granule.name))]
            for reader_key, granule in fmatch_real_ancillary_granules.items()
            if reader_key in ReaderRegistry.get_readers_for_mode(mode)
        }
        manifest = Manifest(
            manifest_type=ManifestType.INPUT,
            configuration={
                PROCESSING_STEP_ID_CONFIGURATION_KEY: str(processing_step_id),
                ANCILLARY_INPUTS_CONFIGURATION_KEY: ancillary_inputs,
            },
        )
        manifest.add_files(*(smart_copy_file(path, input_dir / path.name) for path in input_products))
        manifest.add_files(*(path for paths in ancillary_inputs.values() for path in paths))
        return manifest.write(input_dir)

    return _write
