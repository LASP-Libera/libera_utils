"""Fixtures for integration testing libera_utils"""

from argparse import Namespace

import pytest
from cloudpathlib import S3Path


class L0MockParsedArgsNamespace(Namespace):
    """Generates dummy parser"""

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
def setup_jpss1_kernel_maker_environment_with_manifest(
    test_type,
    generate_input_manifest_local,
    generate_input_manifest_s3,
    test_l1a_sc_pos_product_file,
    create_mock_bucket,
    short_tmp_path,
):
    data_files = [str(test_l1a_sc_pos_product_file)]
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3(*data_files)
        bucket = create_mock_bucket()
        output_path = S3Path(f"s3://{bucket.name}/kernel_output/")
    else:
        input_manifest_path = generate_input_manifest_local(*data_files)
        output_path = short_tmp_path

    return input_manifest_path, output_path


@pytest.fixture
def setup_azel_kernel_maker_environment_with_manifest(
    test_type,
    generate_input_manifest_local,
    generate_input_manifest_s3,
    test_l1a_axis_sample_product_file,
    create_mock_bucket,
    short_tmp_path,
):
    data_files = [str(test_l1a_axis_sample_product_file)]
    if test_type == "S3":
        input_manifest_path = generate_input_manifest_s3(*data_files)
        bucket = create_mock_bucket()
        output_path = S3Path(f"s3://{bucket.name}/kernel_output/")
    else:
        input_manifest_path = generate_input_manifest_local(*data_files)
        output_path = short_tmp_path

    return input_manifest_path, output_path
