"""Plugin module for mocking S3 buckets"""
# Standard
import os

# Installed
import pytest

# Local
from libera_utils.aws.constants import ManifestType
from libera_utils.io.manifest import Manifest


@pytest.fixture
def generate_input_manifest_local(tmp_path, test_data_path,
                                  test_construction_record_1, test_construction_record_2,
                                  test_pds_file_1, test_pds_file_2):
    """Generating test manifest from the data in test_data"""
    def _generate_input_manifest_local_with_files(*filenames):

        if len(filenames) == 0:
            files = (test_construction_record_1,
                     test_pds_file_1,
                     test_construction_record_2,
                     test_pds_file_2)
        else:
            files =[]
            for filename in filenames:
                files.append(test_data_path / filename)

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
        input_manifest.add_files(*files)

        os.mkdir(tmp_path / "processing")
        input_manifest_file_path = input_manifest.write(out_path=tmp_path / "processing")

        return input_manifest_file_path

    return _generate_input_manifest_local_with_files


@pytest.fixture
def generate_input_manifest_s3(test_data_path, create_mock_bucket, write_file_to_s3,
                               test_construction_record_1, test_construction_record_2,
                               test_pds_file_1, test_pds_file_2):
    """Generating test manifest from the data in test_data"""

    def _generate_input_manifest_s3_with_files(*filenames):
        r_bucket = create_mock_bucket()

        input_manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})

        if len(filenames) == 0:
            filenames = (test_construction_record_1.name,
                         test_pds_file_1.name,
                         test_construction_record_2.name,
                         test_pds_file_2.name)

        for filename in filenames:
            s3_file_path = f"s3://{r_bucket.name}/{filename}"
            local_path = test_data_path / filename
            write_file_to_s3(local_path, s3_file_path)
            input_manifest.add_files(s3_file_path)

        d_bucket = create_mock_bucket()
        input_manifest_file_path = input_manifest.write(out_path=f"s3://{d_bucket.name}/processing")

        return input_manifest_file_path

    return _generate_input_manifest_s3_with_files
