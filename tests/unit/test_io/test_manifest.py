"""Tests for manifest module"""
# Standard
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
from unittest import mock
from hashlib import md5
from zoneinfo import ZoneInfo
# Installed
from cloudpathlib import S3Path
import pytest
# Local
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.smart_open import smart_open


def test_manifest_from_file(test_json_manifest):
    """Test factory method for creating a manifest object from a filepath"""
    m = Manifest.from_file(test_json_manifest)
    assert m.manifest_type == ManifestType.INPUT
    assert isinstance(m.files, list)
    assert isinstance(m.configuration, dict)


def test_manifest_constructor_with_file_list(test_txt, test_construction_record_1):
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[test_txt, test_construction_record_1],
        configuration={}
    )
    m.validate_checksums()
    assert len(m.files) == 2


def test_manifest_add_relative_path_file_error():
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    with pytest.raises(ValueError):
        m.add_files(Path("relative/a_file.txt"))


def test_manifest_add_files_to_manifest_local(test_json_manifest, test_txt, test_construction_record_1):
    """Test factory method for adding files to a manifest with checksum and local paths"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    initial_list_len = len(m.files)
    m.add_files(test_json_manifest)
    m.validate_checksums()
    assert len(m.files) == initial_list_len + 1

    more_files = (test_txt, test_construction_record_1)
    m.add_files(*more_files)
    m.validate_checksums()
    assert len(m.files) == initial_list_len + 3


def test_manifest_add_files_to_manifest_s3(test_json_manifest, test_txt, test_construction_record_1,
                                           create_mock_bucket, write_file_to_s3):
    """Test factory method for adding files to a manifest with checksum with S3 paths.
        Ensures functionality for single and multiple file additions."""
    bucket = create_mock_bucket()
    manifest_path = f"s3://{bucket.name}/test_file1.json"
    text_paths = (f"s3://{bucket.name}/test_file2.txt",
                  f"s3://{bucket.name}/test_construction_record.PDS")
    write_file_to_s3(test_json_manifest, manifest_path)
    write_file_to_s3(test_txt, text_paths[0])
    write_file_to_s3(test_construction_record_1, text_paths[1])

    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    initial_list_len = len(m.files)
    m.add_files(manifest_path)
    m.validate_checksums()
    assert len(m.files) == initial_list_len + 1

    m.add_files(*text_paths)
    m.validate_checksums()
    m.validate()
    assert len(m.files) == initial_list_len + 3


def test_manifest_add_duplicate_file_to_manifest(test_json_manifest):
    """Test factory method for adding a duplicate file to a manifest"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    m.add_files(test_json_manifest)
    initial_length = len(m.files)

    # Add the same file
    with pytest.warns(UserWarning) as record:
        m.add_files(test_json_manifest)
    m.validate_checksums()
    assert len(m.files) == initial_length
    assert len(record) == 1


def test_manifest_add_desired_time_range(test_json_manifest):
    """Test factory method for adding a time range to a manifest file"""
    m = Manifest.from_file(test_json_manifest)
    start = datetime.now()
    end = start + timedelta(hours=1)
    m.add_desired_time_range(start, end)

    assert "start_time" in m.configuration.keys()
    assert "end_time" in m.configuration.keys()


def test_manifest_from_file_s3(test_json_manifest, write_file_to_s3):
    """Test loading a file from S3"""
    file_key = f"s3://test-manifest-from-file-s3-bucket/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"
    s3_path = write_file_to_s3(test_json_manifest, file_key)
    m = Manifest.from_file(s3_path)
    assert m.manifest_type == ManifestType.INPUT
    assert isinstance(m.files, list)
    assert isinstance(m.configuration, dict)


def test_manifest_write(tmp_path):
    """Test writing a manifest file from an object"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    m.write(tmp_path, 'LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json')
    with open(tmp_path / 'LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json') as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


def test_manifest_generate_filename():
    """Test generating a filename for a manifest file"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    assert m._generate_filename().filename_parts.ulid_code is not None
    m.manifest_type = ManifestType.OUTPUT
    assert m._generate_filename().filename_parts.ulid_code is not None


def test_manifest_write_s3(create_mock_bucket):
    """Test writing a manifest file from an object"""
    bucket = create_mock_bucket()
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    outpath = S3Path(f"s3://{bucket.name}")
    filename = "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"
    m.write(outpath, filename)
    with smart_open(outpath / filename) as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


def test_validate_checksums(test_json_manifest, caplog):
    """Test the method that validates checksums in a manifest file"""
    # We test by referencing the manifest file itself, so we're only dependent on one test file
    m = Manifest.from_file(test_json_manifest)
    m.files[0]['filename'] = test_json_manifest.absolute()
    m.files[1]['filename'] = test_json_manifest.absolute()
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError):  # Fake values don't validate
            m.validate_checksums()
        assert f"Checksum validation for {test_json_manifest.absolute()} failed." in caplog.records[0].message

    with test_json_manifest.open('rb') as fh:
        checksum = md5(fh.read()).hexdigest()
    m.files = [{"filename": test_json_manifest.absolute(), "checksum": checksum}]
    m.validate_checksums()


@pytest.mark.parametrize(
    'input_manifest',
    [
        (S3Path("s3://test-manifest-from-file-s3-bucket/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json")),
        (Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'
         / 'LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json'),
        (Manifest.from_file(filepath=Path(sys.modules[__name__.split('.')[0]].__file__).parent
                                     / 'test_data' / 'LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json')),
        (S3Path("s3://l0-ingest-dropbox/processing//LIBERA_OUTPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"))
    ]
)
def test_output_manifest_from_input_manifest(input_manifest, test_json_manifest, write_file_to_s3):
    """Test method that creates output manifest from input manifest filename or object"""
    if isinstance(input_manifest, S3Path):
        s3_path = write_file_to_s3(test_json_manifest, str(input_manifest))
        input_manifest_object = Manifest.from_file(filepath=s3_path)

    elif isinstance(input_manifest, Path):
        input_manifest_object = Manifest.from_file(filepath=input_manifest)

    elif isinstance(input_manifest, Manifest):
        input_manifest_object = input_manifest

    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest=input_manifest_object)
    input_time = input_manifest_object.ulid_code.datetime
    output_time = output_manifest.ulid_code.datetime

    assert input_manifest_object.manifest_type == ManifestType.INPUT
    assert output_manifest.manifest_type == ManifestType.OUTPUT
    assert input_time == output_time
    assert len(output_manifest.configuration) != 0


@pytest.mark.parametrize(
    'man_path, man_files, man_type, man_config',
    [('subfolder/LIBERA_INPUT_MANIFEST_201GDHWG4R0W8KXWY0KRDD6BZTT.json',
      [{"filename": "relative/file.txt", "checksum": "fakesum"}], ManifestType.OUTPUT, None),
     ('subfolder/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json', None, ManifestType.OUTPUT, ["config"])
     ]
)
def test_manifest_validation_failure(man_path, man_files, man_type, man_config):
    """Test manifest validation method for correct failure cases"""
    with pytest.raises(ValueError):
        m = Manifest(
            manifest_type=man_type,
            files=man_files,
            configuration=man_config,
            filename=man_path
        )


@pytest.mark.parametrize(
    'man_path, man_files, man_type, man_config',
    [(None, [{"filename": "s3://abs/file.txt", "checksum": "fakesum"}], ManifestType.OUTPUT, {"data": "description"}),
    ('subfolder/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json', None, ManifestType.OUTPUT, {"data": "description"}),
    (None, None, ManifestType.OUTPUT, {"data": "description"})]
)
def test_manifest_validation_success(man_path, man_files, man_type, man_config):
    """Test manifest validation method for correct success cases"""
    m = Manifest(
        manifest_type=man_type,
        files=man_files,
        configuration=man_config,
        filename=man_path
    )
