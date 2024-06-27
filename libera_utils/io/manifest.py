"""Module for manifest file handling"""
import warnings
# Standard
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from hashlib import md5
# Installed
from cloudpathlib import S3Path, AnyPath
from ulid import ULID
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
                 files: list or dict = None, configuration: dict = None, filename: str or ManifestFilename = None):
        """Constructor

        Parameters
        ----------
        manifest_type : ManifestType
            Type of manifest
        files : list, Optional
            List of dictionaries. Each entry must contain a `filename` key and a `checksum` key.
        configuration : dict, Optional
            Freeform dictionary of configuration items. It's up to the consumer to understand this JSON object.
        filename : str or ManifestFilename, Optional
            Preset filename. Must be a ManifestFilename object or a str representing a valid manifest file path.
        """
        self.manifest_type = manifest_type if isinstance(manifest_type, ManifestType) else ManifestType(manifest_type)
        self.configuration = configuration if configuration else {}
        if filename:
            self.filename = filename if isinstance(filename, ManifestFilename) else ManifestFilename(filename)
            self.ulid_code = self.filename.filename_parts.ulid_code
        else:
            self.filename = None
            self.ulid_code = None

        self.files = []
        if files:
            if isinstance(files[0], dict):
                self.files = files
            elif isinstance(files, list):
                self.add_files(*files)

        self.validate()

    def __str__(self):
        return f"<Manifest:{self.filename if self.filename else '(unnamed)'} " \
               f"{self.manifest_type} of {len(self.files)} files>"

    def _generate_filename(self):
        """Generate a valid manifest filename"""
        mfn = ManifestFilename.from_filename_parts(
            manifest_type=self.manifest_type,
            ulid_code=ULID.from_datetime(datetime.now(timezone.utc))
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
        filepath : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
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
        obj = cls(ManifestType(contents['manifest_type'].upper()),
                  contents['files'],
                  contents['configuration'],
                  filename=filepath)
        obj.validate()  # If we just read it, it should be valid
        return obj

    def write(self, outpath: str or Path or S3Path, filename: str = None):
        """Write a manifest file from a Manifest object (self).

        Parameters
        ----------
        outpath : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
            Directory path to write to (directory being used loosely to refer also to an S3 bucket path).
        filename : str, Optional
            Optional filename, must be a valid manifest filename.
            If not provided, the method uses the objects internal filename attribute. If that is
            not set, then a filename is automatically generated.

        Returns
        -------
        : pathlib.Path or cloudpathlib.s3.s3path.S3Path
        """
        if filename is None:
            if self.filename is None:
                filename = self._generate_filename()
            else:
                filename = self.filename
        else:
            filename = ManifestFilename(filename)

        filepath = AnyPath(outpath) / filename.path
        self.filename = ManifestFilename(filepath)  # Update object's filename to the filepath we just wrote
        self.validate()  # Final check before writing
        with smart_open(self.filename.path, 'x') as manifest_file:
            json.dump(self.to_json_dict(), manifest_file)
        return self.filename.path

    def validate(self):
        """Validate the contents of this manifest object"""
        if not isinstance(self.files, list):
            raise ValueError("The files attribute must be a dictionary.")
        if not isinstance(self.configuration, dict):
            raise ValueError("The configuration attribute must be a dictionary.")
        if not isinstance(self.manifest_type, ManifestType):
            raise ValueError("The manifest_type attribute must be a ManifestType object.")
        if self.filename:
            if not isinstance(self.filename, ManifestFilename):
                raise ValueError("The filename attribute must be a ManifestFilename object.")
        for filedict in self.files:
            if tuple(filedict.keys()) != ('filename', 'checksum'):
                raise ValueError("Each entry of the files attribute must be a dictionary containing the keys "
                                 f"`filename` and `checksum`. Got {tuple(filedict.keys())}.")
            if not AnyPath(filedict['filename']).is_absolute():
                raise ValueError(f"Each file path must be a absolute path, instead got: {filedict['filename']}")

    def validate_checksums(self):
        """Validate checksums of listed files"""
        # Note any gzipped file will be opened and read by smart_open so the checksum reflects the data in the zipped
        # file not the zipped file itself.
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

    def add_files(self, *files):
        """Add files to the manifest from filename

        Parameters
        ----------
        files : str or pathlib.Path or cloudpathlib.s3.s3path.S3Path
            Path to the file to add to the manifest.

        Returns
        -------
        None
        """

        for file in files:
            # S3 paths are always absolute so this is always valid for them
            if not AnyPath(file).is_absolute():
                raise ValueError(f"The file path for {AnyPath(file)} must be an absolute path.")
            if AnyPath(file).name in (AnyPath(fs['filename']).name for fs in self.files):
                warnings.warn(f"Attempting to add {file} to manifest {self} but it is already included.")
                continue
            with smart_open(file) as fh:
                checksum_calculated = md5(fh.read(), usedforsecurity=False).hexdigest()
            if checksum_calculated in (fs['checksum'] for fs in self.files):
                warnings.warn(f"Attempting to add {file} to manifest {self} but another file with "
                              f"the same checksum is already included.")
            file_structure = {"filename": str(file),
                              "checksum": str(checksum_calculated)}
            self.files.append(file_structure)

    def add_file_to_manifest(self, file):
        """Deprecated legacy method replaced by add_files"""
        warnings.warn("add_file_to_manifest(file) is deprecated. Use add_files(*files) instead",
                      category=DeprecationWarning)
        self.add_files(file)

    def add_desired_time_range(self, start_datetime: datetime, end_datetime: datetime):
        """Add a file to the manifest from filename

        Parameters
        ----------
        start_datetime : datetime.datetime
            The desired start time for the range of data in this manifest

        end_datetime : datetime.datetime
            The desired end time for the range of data in this manifest

        Returns
        -------
        None
        """
        self.configuration["start_time"] = start_datetime.strftime('%Y-%m-%d:%H:%M:%S')
        self.configuration["end_time"] = end_datetime.strftime('%Y-%m-%d:%H:%M:%S')

    @classmethod
    def output_manifest_from_input_manifest(cls, input_manifest: Path or S3Path or 'Manifest') -> 'Manifest':
        """ Create Output manifest from input manifest file path, adds input files to output manifest configuration

        Parameters
        ----------
        input_manifest : pathlib.Path or cloudpathlib.s3.s3path.S3Path or Manifest
            An S3 or regular path to an input_manifest object, or the input manifest object itself

        Returns
        -------
        output_manifest : Manifest
            The newly created output manifest
        """

        if not isinstance(input_manifest, cls):
            input_manifest = Manifest.from_file(input_manifest)

        input_manifest_ulid_code = input_manifest.filename.filename_parts.ulid_code
        manifest_filename = ManifestFilename.from_filename_parts(manifest_type=ManifestType.OUTPUT,
                                                                 ulid_code=input_manifest_ulid_code)

        input_manifest_files = input_manifest.files

        output_manifest = Manifest(manifest_type=ManifestType.OUTPUT,
                                   filename=manifest_filename,
                                   configuration={'input_manifest_files': input_manifest_files})
        output_manifest.validate()

        return output_manifest
