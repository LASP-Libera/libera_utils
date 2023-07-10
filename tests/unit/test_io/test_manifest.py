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


def test_manifest_add_file_to_manifest(test_json_manifest):
    """Test factory method for adding a file to a manifest with checksum"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    initial_list_len = len(m.files)
    m.add_files(test_json_manifest)
    m.validate_checksums()
    assert len(m.files) == initial_list_len + 1


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
    file_key = f"s3://test-manifest-from-file-s3-bucket/libera_input_manifest_20220101t112233.json"
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
    m.write(tmp_path, 'libera_input_manifest_20220101t112233.json')
    with open(tmp_path / 'libera_input_manifest_20220101t112233.json') as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


@mock.patch("libera_utils.io.manifest.datetime")
def test_manifest_generate_filename(mock_manifest_datetime):
    """Test generating a filename for a manifest file"""
    mock_manifest_datetime.utcnow.return_value = datetime(2022, 1, 1, 12, 34, 56, tzinfo=ZoneInfo("UTC"))
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    assert m._generate_filename().path.name == "libera_input_manifest_20220101t123456.json"
    m.manifest_type = ManifestType.OUTPUT
    assert m._generate_filename().path.name == "libera_output_manifest_20220101t123456.json"


def test_manifest_write_s3(create_mock_bucket):
    """Test writing a manifest file from an object"""
    bucket = create_mock_bucket()
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[],
        configuration={}
    )
    outpath = S3Path(f"s3://{bucket.name}")
    filename = "libera_input_manifest_20220101t112233.json"
    m.write(outpath, filename)
    with smart_open(outpath / filename) as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


def test_validate_checksums(test_json_manifest, caplog):
    """Test the method that validates checksums in a manifest file"""
    # We test by referencing the manifest file itself so we're only dependent on one test file
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
        (S3Path("s3://test-manifest-from-file-s3-bucket/libera_input_manifest_20220101t112233.json")),
        (Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'
         / 'libera_input_manifest_20220922t123456.json'),
        (Manifest.from_file(filepath=Path(sys.modules[__name__.split('.')[0]].__file__).parent
                                  / 'test_data' / 'libera_input_manifest_20220922t123456.json')),
        (S3Path("s3://l0-ingest-dropbox-liberasdplcs-dev/processing//libera_output_manifest_20230613t143309.json"))
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

    input_time = str(re.search(r'\d{8}t\d+', str(input_manifest_object.filename)).group(0))
    output_time = str(re.search(r'\d{8}t\d+', str(output_manifest.filename)).group(0))
    assert output_manifest.manifest_type == ManifestType.OUTPUT
    assert input_time == output_time
    assert len(output_manifest.configuration) != 0
