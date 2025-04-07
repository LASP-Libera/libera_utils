"""Fixtures for integration testing libera_utils"""
from argparse import Namespace

import pytest
from cloudpathlib import S3Path


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
