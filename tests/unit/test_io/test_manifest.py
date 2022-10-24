"""Tests for manifest module"""
# Standard
from datetime import datetime
import json
from unittest import mock
from hashlib import md5
import pytz
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


def test_manifest_from_file_s3(test_json_manifest, write_file_to_s3):
    """Test loading a file from S3"""
    file_key = f"s3://test-manifest-from-file-s3-bucket/test_manifest.json"
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
    m.write(tmp_path, 'test_manifest.json')
    with open(tmp_path / 'test_manifest.json') as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


@mock.patch("libera_utils.io.manifest.datetime")
def test_manifest_generate_filename(mock_pendulum_now):
    """Test generating a filename for a manifest file"""
    mock_pendulum_now.utcnow.return_value = pytz.timezone('UTC').localize(datetime.fromisoformat("2022-01-01T12:34:56"))
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
    filename = "test_manifest.json"
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
