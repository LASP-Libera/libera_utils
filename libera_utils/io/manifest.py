"""Module for manifest file handling"""

import json
import logging
from datetime import UTC, datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Union

from cloudpathlib import AnyPath, S3Path
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from ulid import ULID

from libera_utils.aws.constants import ManifestType
from libera_utils.io.filenaming import AbstractValidFilename, ManifestFilename
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


class ManifestError(Exception):
    """Generic exception related to manifest file handling"""

    pass


def calculate_checksum(file: str | Path | S3Path) -> str:
    """Compute the checksum of the given file."""
    with smart_open(file, "rb") as fh:
        checksum_calculated = md5(fh.read(), usedforsecurity=False).hexdigest()
    return checksum_calculated


def get_ulid_code(filename: str | Path | S3Path | ManifestFilename | None) -> ULID | None:
    """Get ULID code from filename."""
    if not filename:
        return None
    if isinstance(filename, ManifestFilename):
        return filename.filename_parts.ulid_code
    return AbstractValidFilename.from_file_path(filename).filename_parts.ulid_code


class ManifestFileRecord(BaseModel):
    """Pydantic model for an individual data product file recorded within a manifest file."""

    filename: str = Field(description="Manifest file name")
    checksum: str = Field(description="Manifest file checksum, calculated if not provided")


class Manifest(BaseModel):
    """Pydantic model for a manifest file."""

    manifest_type: ManifestType = Field(description="Either INPUT or OUTPUT.")
    files: list[ManifestFileRecord] = Field(default_factory=list, description="List of ManifestFileStructure.")
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Freeform json-compatible dictionary of configuration items."
    )
    filename: ManifestFilename | None = Field(default=None, description="Preset filename, optional.")
    ulid_code: ULID | None = Field(
        default_factory=lambda data: get_ulid_code(data.get("filename")), description="ULID code from input filename."
    )

    @field_validator("filename", mode="before")  # noqa  avoid type warning
    @classmethod
    def transform_filename(cls, raw_filename: str | ManifestFilename | None) -> ManifestFilename | None:
        """Convert raw filename to ManifestFilename class if necessary."""
        if raw_filename is None:
            return None
        if isinstance(raw_filename, ManifestFilename):
            return raw_filename
        return ManifestFilename(raw_filename)

    @classmethod
    def check_file_structure(
        cls, file_structure: ManifestFileRecord, existing_names: set[str], existing_checksums: set[str]
    ) -> bool:
        """Check file structure, returning True if it is good."""
        file = file_structure.filename
        # S3 paths are always absolute so this is always valid for them
        if not AnyPath(file).is_absolute():
            raise ValueError(f"The file path for {file} must be an absolute path.")
        if file in existing_names:
            logger.warning(f"Attempting to add {file} to manifest but it is already included.")
            return False
        checksum_calculated = file_structure.checksum if file_structure.checksum else calculate_checksum(file)
        if checksum_calculated in existing_checksums:
            logger.warning(
                f"Attempting to add {file} to manifest but another file with the same checksum is already included."
            )
            return False
        return True

    @field_validator("files", mode="before")  # noqa  avoid type warning
    @classmethod
    def transform_files(
        cls, raw_list: list[dict | str | Path | S3Path | ManifestFileRecord] | None
    ) -> list[ManifestFileRecord]:
        """Allow for the incoming files list to have varying types.
        Convert to a standardized list of ManifestFileStructure."""
        result = []
        existing_names = set()
        existing_checksums = set()
        for raw_file in raw_list or []:
            if isinstance(raw_file, ManifestFileRecord):
                file_structure = raw_file
            elif isinstance(raw_file, dict):
                file_structure = ManifestFileRecord(
                    filename=raw_file.get("filename"),
                    checksum=raw_file.get("checksum") or calculate_checksum(raw_file.get("filename")),
                )
            else:
                file_structure = ManifestFileRecord(
                    filename=str(AnyPath(raw_file)), checksum=calculate_checksum(raw_file)
                )
            if cls.check_file_structure(file_structure, existing_names, existing_checksums):
                result.append(file_structure)
                existing_names.add(str(file_structure.filename))
                existing_checksums.add(file_structure.checksum)
        return result

    @field_serializer("filename")
    def serialize_filename(self, filename: str | Path | S3Path | ManifestFilename | None, _info) -> str:
        """Custom serializer for the manifest filename."""
        return str(filename)

    model_config = ConfigDict(
        # Allow using ManifestFilename as a field
        arbitrary_types_allowed=True
    )

    @classmethod
    def from_file(cls, filepath: str | Path | S3Path):
        """Read a manifest file and return a Manifest object (factory method).

        Parameters
        ----------
        filepath : Union[str, Path, S3Path]
            Location of manifest file to read.

        Returns
        -------
        Manifest
            Pydantic model built from the json of the given manifest file.
        """
        with smart_open(filepath) as manifest_file:
            contents = json.loads(manifest_file.read())
        contents["filename"] = filepath if isinstance(filepath, ManifestFilename) else ManifestFilename(filepath)
        contents["ulid_code"] = get_ulid_code(filepath)
        return Manifest.model_validate(contents)

    def add_files(self, *files: str | Path | S3Path):
        """Add files to the manifest from filename

        Parameters
        ----------
        files : Union[str, Path, S3Path]
            Path to the file to add to the manifest.

        Returns
        -------
        None
        """
        # get existing files and checksums as sets to check for duplicates
        existing_names = set()
        existing_checksums = set()
        for f in self.files:
            existing_names.add(f.filename)
            existing_checksums.add(f.checksum)

        for file in files:
            checksum_calculated = calculate_checksum(file) if AnyPath(file).exists() else None
            file_structure = ManifestFileRecord(filename=str(file), checksum=checksum_calculated)
            if self.check_file_structure(file_structure, existing_names, existing_checksums):
                self.files.append(file_structure)
                existing_names.add(str(file_structure.filename))
                existing_checksums.add(file_structure.checksum)

    def validate_checksums(self) -> None:
        """Validate checksums of listed files"""
        # Note: any gzipped file will be opened and read by smart_open so the checksum reflects the data
        # in the zipped file not the zipped file itself.
        failed_filenames = []
        for file_structure in self.files:
            checksum_expected = file_structure.checksum
            filename = file_structure.filename
            checksum_calculated = calculate_checksum(filename)
            if checksum_expected != checksum_calculated:
                logger.error(
                    f"Checksum validation for {filename} failed. "
                    f"Expected {checksum_expected} but got {checksum_calculated}."
                )
                failed_filenames.append(str(filename))
        if failed_filenames:
            raise ValueError(f"Files failed checksum validation: {', '.join(failed_filenames)}")

    def _generate_filename(self) -> ManifestFilename:
        """Generate a valid manifest filename"""
        mfn = ManifestFilename.from_filename_parts(
            manifest_type=self.manifest_type, ulid_code=ULID.from_datetime(datetime.now(UTC))
        )
        return mfn

    def write(self, out_path: str | Path | S3Path, filename: str = None) -> Path | S3Path:
        """Write a manifest file from a Manifest object (self).

        Parameters
        ----------
        out_path : Union[str, Path, S3Path]
            Directory path to write to (directory being used loosely to refer also to an S3 bucket path).
        filename : str, Optional
            must be a valid manifest filename.
            If not provided, the method uses the objects internal filename attribute. If that is
            not set, then a filename is automatically generated.

        Returns
        -------
        Union[Path, S3Path]
            The path where the manifest file is written.
        """
        if filename is None:
            filename = self._generate_filename() if self.filename is None else self.filename
        else:
            filename = ManifestFilename(filename)
        filepath = AnyPath(out_path) / filename.path

        # Update object's filename to the filepath we just wrote
        self.filename = ManifestFilename(filepath)

        with smart_open(self.filename.path, "x") as manifest_file:
            manifest_file.write(self.model_dump_json())
        return self.filename.path

    def add_desired_time_range(self, start_datetime: datetime, end_datetime: datetime):
        """Add a time range to the configuration section of the manifest.

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
        self.configuration["start_time"] = start_datetime.strftime("%Y-%m-%d:%H:%M:%S")
        self.configuration["end_time"] = end_datetime.strftime("%Y-%m-%d:%H:%M:%S")

    @classmethod
    def output_manifest_from_input_manifest(cls, input_manifest: Union[Path, S3Path, "Manifest"]) -> "Manifest":
        """Create Output manifest from input manifest file path, adds input files to output manifest configuration

        Parameters
        ----------
        input_manifest : Union[Path, S3Path, 'Manifest']
            An S3 or regular path to an input_manifest object, or the input manifest object itself

        Returns
        -------
        output_manifest : Manifest
            The newly created output manifest
        """

        if not isinstance(input_manifest, cls):
            input_manifest = Manifest.from_file(input_manifest)

        input_filename = input_manifest.filename
        input_manifest_ulid_code = input_filename.filename_parts.ulid_code

        output_filename = ManifestFilename.from_filename_parts(
            manifest_type=ManifestType.OUTPUT, ulid_code=input_manifest_ulid_code
        )

        output_manifest = Manifest(
            manifest_type=ManifestType.OUTPUT,
            filename=output_filename,
            configuration={"input_manifest_files": input_manifest.files},
        )

        return output_manifest
