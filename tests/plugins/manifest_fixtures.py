"""Plugin module for mocking S3 buckets"""
import string
# Standard
import os
# Installed
import pytest
# Local
from libera_utils.io.manifest import Manifest, ManifestType

@pytest.fixture
def generate_input_manifest_local(tmp_path, test_data_path):
    """Generating test manifest from the data in test_data"""
    def generate_input_manifest_local_with_files(*filenames):

        if len(filenames) == 0:
            files = (test_data_path / "J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS",
                     test_data_path / "J01_G011_LZ_2021-04-09T02-00-00Z_V01.CONS",
                     test_data_path / "P1590011AAAAAAAAAAAAAT21099051420500.PDS",
                     test_data_path / "P1590011AAAAAAAAAAAAAT21099051420501.PDS")
        else:
            files =[]
            for filename in filenames:
                files.append(test_data_path / filename)

        input_manifest = Manifest(ManifestType.INPUT, files=[], configuration={})
        input_manifest.add_files(*files)

        os.mkdir(tmp_path / "processing")
        input_manifest_file_path = input_manifest.write(outpath=tmp_path / "processing",
                                                        filename='libera_input_manifest_20230102t112233.json')

        return input_manifest_file_path

    return generate_input_manifest_local_with_files

@pytest.fixture
def generate_input_manifest_s3(test_data_path, create_mock_bucket, write_file_to_s3):
    """Generating test manifest from the data in test_data"""

    def generate_input_manifest_s3_with_files(*filenames):
        r_bucket = create_mock_bucket()

        input_manifest = Manifest(ManifestType.INPUT, files=[], configuration={})

        if len(filenames) == 0:
            filenames = ("J01_G011_LZ_2021-04-09T00-00-00Z_V01.CONS",
                         "J01_G011_LZ_2021-04-09T02-00-00Z_V01.CONS",
                         "P1590011AAAAAAAAAAAAAT21099051420500.PDS",
                         "P1590011AAAAAAAAAAAAAT21099051420501.PDS")

        for filename in filenames:
            s3_file_path = f"s3://{r_bucket.name}/{filename}"
            local_path = test_data_path / filename
            write_file_to_s3(local_path, s3_file_path)
            input_manifest.add_files(s3_file_path)

        d_bucket = create_mock_bucket()
        input_manifest_file_path = input_manifest.write(outpath=f"s3://{d_bucket.name}/processing")

        return input_manifest_file_path

    return generate_input_manifest_s3_with_files
