"""Tests for manifest module"""
# Installed
import pytest
# Local
from libera_utils.io.manifest import Manifest, ManifestType


def test_manifest_from_file(test_json_manifest):
    """Test factory method for creating a manifest object from a filepath"""
    m = Manifest.from_file(test_json_manifest)
    print(m)


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
        print(f.readlines())
