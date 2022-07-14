"""Module for manifest file handling"""
# Standard
from enum import Enum
import json
from pathlib import Path
# Installed
from cloudpathlib import S3Path
# Local
from libera_utils.io.smart_open import smart_open


class ManifestError(Exception):
    """Generic exception related to manifest file handling"""
    pass


class ManifestType(Enum):
    """Enumerated legal manifest type values"""
    INPUT = 'INPUT'
    OUTPUT = 'OUTPUT'


class Manifest:
    """Object representation of a JSON manifest file"""

    __manifest_elements = (
        "manifest_type",
        "inputs",
        "outputs",
        "tmp",
        "logs",
        "configuration"
    )

    def __init__(self, manifest_type: ManifestType,
                 inputs: list, outputs: list, tmp: dict, logs: dict, configuration: dict):
        # TODO: Strive to implement structure on this Manifest object. Ideally we don't just want a bunch of
        #    dictionaries, though at least that is a lowest common denominator.
        self.manifest_type = manifest_type
        self.inputs = inputs
        self.outputs = outputs
        self.tmp = tmp
        self.logs = logs
        self.configuration = configuration

    @classmethod
    def from_file(cls, filepath: str or Path or S3Path):
        """Read a manifest file and return a Manifest object (factory method).

        Parameters
        ----------
        filepath : str or Path or S3Path
            Location of manifest file to read.

        Returns
        -------
        : Manifest
        """
        with smart_open(filepath) as manifest_file:
            contents = json.loads(manifest_file.read())
        for element in cls.__manifest_elements:
            if element not in contents:
                raise ManifestError(f"{filepath} is not a valid manifest file. Missing required element {element}.")
        return cls(ManifestType(contents['manifest_type'].upper()),
                   contents['inputs'],
                   contents['outputs'],
                   contents['tmp'],
                   contents['logs'],
                   contents['configuration'])

    def write(self, filepath: str or Path or S3Path):
        """Write a manifest file from a Manifest object.

        Parameters
        ----------
        filepath : str or Path or S3Path
            Filepath to write to.

        Returns
        -------
        : str or Path or S3Path
        """
        contents = {
            'manifest_type': self.manifest_type.value,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'tmp': self.tmp,
            'logs': self.logs,
            'configuration': self.configuration
        }
        with smart_open(filepath, 'w') as manifest_file:
            json.dump(contents, manifest_file)
        return filepath
