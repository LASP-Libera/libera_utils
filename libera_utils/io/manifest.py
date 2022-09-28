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
        "files",
        "configuration"
    )

    def __init__(self, manifest_type: ManifestType,
                 files: list, configuration: dict, filename: str = None):
        # TODO: Strive to implement structure on this Manifest object. Ideally we don't just want a bunch of
        #    dictionaries, though at least that is a lowest common denominator.
        self.manifest_type = manifest_type
        self.files = files
        self.configuration = configuration

    def to_json_dict(self):
        """Create a dict representation suitable for writing out.

        Returns
        -------
        : dict
        """
        valid_json = json.dumps({
            'manifest_type': self.manifest_type.value,
            'files': self.files,
            'configuration': self.configuration
        })
        return json.loads(valid_json)

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
                   contents['files'],
                   contents['configuration'],
                   filename=filepath)

    def write(self, filepath: str or Path or S3Path):
        """Write a manifest file from a Manifest object (self).

        Parameters
        ----------
        filepath : str or Path or S3Path
            Filepath to write to.

        Returns
        -------
        : str or Path or S3Path
        """
        with smart_open(filepath, 'x') as manifest_file:
            json.dump(self.to_json_dict(), manifest_file)
        return filepath
