"""Tests for manifest module"""
# Standard
import json
# Installed
from cloudpathlib import S3Path
# Local
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.io.smart_open import smart_open


def test_manifest_from_file(test_json_manifest):
    """Test factory method for creating a manifest object from a filepath"""
    m = Manifest.from_file(test_json_manifest)
    assert m.manifest_type == ManifestType.INPUT
    assert isinstance(m.inputs, list)
    assert isinstance(m.outputs, list)
    assert isinstance(m.tmp, dict)
    assert isinstance(m.logs, dict)
    assert isinstance(m.configuration, dict)


def test_manifest_from_file_s3(test_json_manifest, write_file_to_s3):
    """Test loading a file from S3"""
    s3_path = write_file_to_s3(test_json_manifest, "s3://test-bucket/input_manifest.json")
    m = Manifest.from_file(s3_path)
    assert m.manifest_type == ManifestType.INPUT
    assert isinstance(m.inputs, list)
    assert isinstance(m.outputs, list)
    assert isinstance(m.tmp, dict)
    assert isinstance(m.logs, dict)
    assert isinstance(m.configuration, dict)


def test_manifest_write(tmp_path):
    """Test writing a manifest file from an object"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        inputs=[],
        outputs=[],
        tmp={},
        logs={},
        configuration={}
    )
    m.write(tmp_path / 'test_manifest.json')
    with open(tmp_path / 'test_manifest.json') as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "inputs", "outputs", "tmp", "logs", "configuration"):
            assert element in manifest_dict


def test_manifest_write_s3(create_mock_bucket):
    """Test writing a manifest file from an object"""
    bucket = create_mock_bucket('test-bucket')
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        inputs=[],
        outputs=[],
        tmp={},
        logs={},
        configuration={}
    )
    filepath = S3Path(f's3://{bucket.name}/test_manifest.json')
    m.write(filepath)
    with smart_open(filepath) as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "inputs", "outputs", "tmp", "logs", "configuration"):
            assert element in manifest_dict
