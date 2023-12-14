import time
# Installed
import pytest
from argparse import Namespace
from cloudpathlib import S3Path, AnyPath
import datetime
# Local
from libera_utils.io.manifest import Manifest


class L0MockParsedArgsNamespace(Namespace):
    """ Generates dummy parser """

    def __init__(self, manifest_filepath, short_tmp_path=None):
        super().__init__()
        self.manifest_filepath = str(manifest_filepath)
        self.outdir = str(short_tmp_path)
        self.delete = False
        self.verbose = False

@pytest.fixture
def test_type(request):
    """Used for the indirect parameterization passthrough for the setup fixture below"""
    return request.param


@pytest.fixture
def setup_l0_ingest_environment_with_manifest(test_type, generate_input_manifest_local, generate_input_manifest_s3,
                                              monkeypatch, create_mock_secret_manager):
    """A fixture that sets up the correct environment variables and arguments to call the ingest method
    for ingesting l0 files from an input manifest file"""
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3()
    else:
        input_manifest_path = generate_input_manifest_local()

    parsed_args = L0MockParsedArgsNamespace(str(input_manifest_path))
    processing_path = input_manifest_path.parent
    monkeypatch.setenv("PROCESSING_DROPBOX", str(processing_path))

    monkeypatch.setenv("DB_SECRET_NAME", "test-secret")
    create_mock_secret_manager("test-secret")

    return parsed_args

@pytest.fixture
def setup_kernel_maker_environment_with_manifest(test_type, generate_input_manifest_local, generate_input_manifest_s3,
                                                 test_pds_file_1, test_pds_file_2, test_pds_file_3, create_mock_bucket,
                                                 short_tmp_path):
    data_files = [
        str(test_pds_file_1),
        str(test_pds_file_2),
        str(test_pds_file_3)
    ]
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3(*data_files)
        bucket = create_mock_bucket()
        output_path = S3Path(f"s3://{bucket.name}/kernel_output/")
    else:
        input_manifest_path = generate_input_manifest_local(*data_files)
        output_path = short_tmp_path

    return input_manifest_path, output_path
