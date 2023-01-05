"""Module for manifest file handling"""
# Standard
from datetime import datetime
import json
import logging
from pathlib import Path
from hashlib import md5
# Installed
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils.io.smart_open import smart_open
from libera_utils.io.filenaming import ManifestFilename, ManifestType

logger = logging.getLogger(__name__)


class ManifestError(Exception):
    """Generic exception related to manifest file handling"""
    pass


class Manifest:
    """Object representation of a JSON manifest file"""

    __manifest_elements = (
        "manifest_type",
        "files",
        "configuration"
    )

    def __init__(self, manifest_type: ManifestType,
                 files: list, configuration: dict, filename: str = None):
        self.manifest_type = manifest_type
        self.files = files
        self.configuration = configuration
        self.filename = filename

    def _generate_filename(self):
        """Generate a valid manifest filename"""
        mfn = ManifestFilename.from_filename_parts(
            manifest_type=self.manifest_type,
            created_time=datetime.utcnow()
        )
        return mfn

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

    def write(self, outpath: str or Path or S3Path, filename: str = None):
        """Write a manifest file from a Manifest object (self).

        Parameters
        ----------
        outpath : str or Path or S3Path
            Directory path to write to (directory being used loosely to refer also to an S3 bucket path).
        filename : str, Optional
            Optional filename. If not provided, the method uses the objects internal filename attribute. If that is
            not set, then a filename is automatically generated.

        Returns
        -------
        : Path or S3Path
        """
        if filename is None:
            if self.filename is None:
                filename = str(self._generate_filename())
            else:
                filename = self.filename

        filepath = AnyPath(outpath) / filename
        with smart_open(filepath, 'x') as manifest_file:
            json.dump(self.to_json_dict(), manifest_file)
        return filepath

    def validate_checksums(self):
        """Validate checksums of listed files"""
        failed_filenames = []
        for record in self.files:
            checksum_expected = record['checksum']
            filename = record['filename']
            # Validate checksums
            with smart_open(filename, 'rb') as fh:
                checksum_calculated = md5(fh.read(), usedforsecurity=False).hexdigest()
                if checksum_expected != checksum_calculated:
                    logger.error(f"Checksum validation for {filename} failed. "
                                 f"Expected {checksum_expected} but got {checksum_calculated}.")
                    failed_filenames.append(str(filename))
        if failed_filenames:
            raise ValueError(f"Files failed checksum validation: {', '.join(failed_filenames)}")

    def add_file_to_manifest(self, file_path):
        """Add a file to the manifest from filename
          Parameters
        ----------
        file_path : str or Path or S3Path
            Directory path to the file to add to the manifest.

        Returns
        -------
        None
        """
        with smart_open(file_path) as fh:
            checksum_calculated = md5(fh.read(), usedforsecurity=False).hexdigest()
        file_structure = {"filename": str(file_path),
                          "checksum": str(checksum_calculated)}
        self.files.append(file_structure)

    def add_desired_time_range(self, start_datetime: datetime, end_datetime: datetime):
        """Add a file to the manifest from filename
          Parameters
        ----------
        start_datetime : datetime
            The desired start time for the range of data in this manifest

        end_datetime : datetime
            The desired end time for the range of data in this manifest

        Returns
        -------
        None
        """
        self.configuration["start_time"] = start_datetime.strftime('%Y-%m-%d:%H:%M:%S')
        self.configuration["end_time"] = end_datetime.strftime('%Y-%m-%d:%H:%M:%S')
        