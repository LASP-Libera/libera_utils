"""Module for manifest file handling"""

import json
import logging
from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Union

from cloudpathlib import AnyPath, S3Path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from ulid import ULID

from libera_utils.constants import ManifestType
from libera_utils.io.filenaming import ManifestFilename
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


class ManifestError(Exception):
    """Generic exception related to manifest file handling"""

    pass


_CHECKSUM_BLOCK_BYTES = 16 * 2**20

_MANIFEST_NAME_HELP = (
    "expected LIBERA_INPUT_MANIFEST_<ULID>.json or LIBERA_OUTPUT_MANIFEST_<ULID>.json, where <ULID> is a "
    "26-character ULID, for example LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json. Manifest.write() "
    "generates this name when out_path is a directory or S3 prefix, so it need not be given."
)


def calculate_checksum(file: str | Path | S3Path) -> str:
    """Compute the MD5 checksum of the given file, reading it in fixed-size blocks.

    The checksum is taken over the bytes as they are stored on disk. For a ``*.gz`` file that means
    the compressed bytes, so the value can be compared against the file a data provider delivered.
    Hence ``enable_gzip=False``: letting smart_open transparently decompress would checksum content
    that exists nowhere on disk, and would change if the file were ever recompressed at a different
    level. Memory use is one block regardless of file size, so a product larger than the host's
    memory can still be checksummed.

    Parameters
    ----------
    file : str, Path or S3Path
        Path of the file. A gzipped file is not decompressed, so the checksum is of its bytes on disk.

    Returns
    -------
    str
        Hex digest of the file's bytes.
    """
    digest = md5(usedforsecurity=False)
    with smart_open(file, "rb", enable_gzip=False) as fh:
        while block := fh.read(_CHECKSUM_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


class ManifestFileRecord(BaseModel):
    """Pydantic model for an individual data product file recorded within a manifest file.

    Attributes
    ----------
    filename : str
        Absolute local or S3 path of the file.
    checksum : str
        MD5 hex digest of the file's bytes as stored on disk; a gzipped file is not decompressed. Required: a null
        checksum is a validation error.
    """

    filename: str = Field(description="Absolute local or S3 path of the file")
    checksum: str = Field(description="MD5 hex digest of the file's bytes as stored")


class Manifest(BaseModel):
    """Pydantic model for a manifest file.

    Attributes
    ----------
    manifest_type : ManifestType
        INPUT or OUTPUT.
    files : list of ManifestFileRecord
        Files listed in the manifest. May be given as paths, dicts or ``ManifestFileRecord`` objects; paths, and
        dicts without a ``checksum``, are checksummed with MD5. A file that cannot be found raises
        ``FileNotFoundError``.
    configuration : dict
        Freeform JSON-compatible configuration items.
    filename : ManifestFilename or None
        Path or bare name of the manifest file. Its INPUT/OUTPUT label must match ``manifest_type``. Assignment is
        validated.
    ulid_code : ULID or None
        Read-only ULID that uniquely identifies the manifest, taken from ``filename``. As a constructor argument:
        without ``filename``, it sets a bare manifest filename carrying this ULID, which ``write()`` uses when
        ``out_path`` is a directory; with ``filename``, it must match the ULID in it. With neither, ``write()`` takes
        the ULID from a full manifest path given as ``out_path``, or generates a new one when ``out_path`` is a
        directory.
    """

    manifest_type: ManifestType = Field(description="Either INPUT or OUTPUT.")
    files: list[ManifestFileRecord] = Field(default_factory=list, description="Files listed in the manifest.")
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Freeform json-compatible dictionary of configuration items."
    )
    filename: ManifestFilename | None = Field(default=None, description="Path or bare name of the manifest file.")

    model_config = ConfigDict(
        # Allow using ManifestFilename as a field
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    @model_validator(mode="before")
    @classmethod
    def reconcile_ulid_code(cls, data: Any) -> Any:
        """Fold a ``ulid_code`` input into ``filename``.

        Parameters
        ----------
        data : Any
            Raw model input.

        Returns
        -------
        Any
            The input without ``ulid_code``, with a bare manifest filename added if there was no ``filename``.

        Raises
        ------
        ValueError
            If ``ulid_code`` is not a valid ULID or disagrees with the ULID in ``filename``.
        """
        if not isinstance(data, dict) or "ulid_code" not in data:
            return data
        data = dict(data)
        ulid_code = data.pop("ulid_code")
        if ulid_code is None:
            return data
        ulid_code = ULID.from_str(str(ulid_code))

        filename = data.get("filename")
        if filename is None:
            if data.get("manifest_type") is not None:
                data["filename"] = ManifestFilename.from_filename_parts(
                    manifest_type=ManifestType(data["manifest_type"]), ulid_code=ulid_code
                )
            return data

        filename_ulid = cls.transform_filename(filename).filename_parts.ulid_code
        if filename_ulid != ulid_code:
            raise ValueError(f"ulid_code {ulid_code} disagrees with the ULID {filename_ulid} in filename {filename}.")
        return data

    @field_validator("filename", mode="before")  # noqa  avoid type warning
    @classmethod
    def transform_filename(cls, raw_filename: str | Path | S3Path | ManifestFilename | None) -> ManifestFilename | None:
        """Convert a raw filename to a ManifestFilename.

        Parameters
        ----------
        raw_filename : str, Path, S3Path, ManifestFilename or None
            Filename to convert.

        Returns
        -------
        ManifestFilename or None
            The converted filename.

        Raises
        ------
        ValueError
            If the filename is not a valid manifest filename.
        """
        if raw_filename is None:
            return None
        if isinstance(raw_filename, ManifestFilename):
            return raw_filename
        return ManifestFilename(raw_filename)

    # The type-label checks are field validators rather than a model validator because a failed field validator
    # leaves the previous value in place on assignment, while a failed model validator keeps the new, invalid one.
    @field_validator("filename")
    @classmethod
    def check_filename_type_label(
        cls, filename: ManifestFilename | None, info: ValidationInfo
    ) -> ManifestFilename | None:
        """Reject a filename whose INPUT/OUTPUT label disagrees with ``manifest_type``.

        Parameters
        ----------
        filename : ManifestFilename or None
            Filename to check.
        info : pydantic.ValidationInfo
            Holds the already validated ``manifest_type``.

        Returns
        -------
        ManifestFilename or None
            The unchanged filename.

        Raises
        ------
        ValueError
            If the labels disagree.
        """
        manifest_type = info.data.get("manifest_type")
        if (
            filename is not None
            and manifest_type is not None
            and filename.filename_parts.manifest_type != manifest_type
        ):
            raise ValueError(f"Manifest filename {filename} is not named as a {manifest_type} manifest.")
        return filename

    @field_validator("manifest_type")
    @classmethod
    def check_manifest_type_label(cls, manifest_type: ManifestType, info: ValidationInfo) -> ManifestType:
        """Reject assigning a ``manifest_type`` that disagrees with the label of the current filename.

        Parameters
        ----------
        manifest_type : ManifestType
            Manifest type to check.
        info : pydantic.ValidationInfo
            Holds the current ``filename`` on assignment.

        Returns
        -------
        ManifestType
            The unchanged manifest type.

        Raises
        ------
        ValueError
            If the labels disagree.
        """
        filename = info.data.get("filename")
        if filename is not None and filename.filename_parts.manifest_type != manifest_type:
            raise ValueError(f"Manifest filename {filename} is not named as a {manifest_type} manifest.")
        return manifest_type

    @classmethod
    def check_file_structure(
        cls, file_structure: ManifestFileRecord, existing_names: set[str], existing_checksums: set[str]
    ) -> bool:
        """Check whether a file record can be added to a list of records.

        Parameters
        ----------
        file_structure : ManifestFileRecord
            Record to check.
        existing_names : set of str
            Filenames already in the list.
        existing_checksums : set of str
            Checksums already in the list.

        Returns
        -------
        bool
            False, with a warning, if the record duplicates an existing filename or checksum; True otherwise.

        Raises
        ------
        ValueError
            If the record's path is not absolute.
        """
        file = file_structure.filename
        # S3 paths are always absolute so this is always valid for them
        if not AnyPath(file).is_absolute():
            raise ValueError(f"The file path for {file} must be an absolute path.")
        if file in existing_names:
            logger.warning(f"Attempting to add {file} to manifest but it is already included.")
            return False
        if file_structure.checksum in existing_checksums:
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
        """Convert the incoming files to ManifestFileRecords, dropping duplicates.

        A path, or a dict with no ``checksum`` key, is checksummed with MD5.

        Parameters
        ----------
        raw_list : list of dict, str, Path, S3Path or ManifestFileRecord, or None
            Files to convert.

        Returns
        -------
        list of ManifestFileRecord
            The converted records.

        Raises
        ------
        FileNotFoundError
            If a file to be checksummed cannot be found.
        ValueError
            If a path is not absolute, or a record's checksum is null.
        """
        result = []
        existing_names = set()
        existing_checksums = set()
        for raw_file in raw_list or []:
            if isinstance(raw_file, ManifestFileRecord):
                file_structure = raw_file
            elif isinstance(raw_file, dict) and "checksum" in raw_file:
                file_structure = ManifestFileRecord(filename=raw_file.get("filename"), checksum=raw_file["checksum"])
            else:
                file = raw_file.get("filename") if isinstance(raw_file, dict) else str(AnyPath(raw_file))
                # Checked before existence so a relative path is reported as relative, not as missing
                if not AnyPath(file).is_absolute():
                    raise ValueError(f"The file path for {file} must be an absolute path.")
                if not AnyPath(file).exists():
                    raise FileNotFoundError(
                        f"File {file} cannot be found. A file must exist when it is added to a manifest so its "
                        "checksum can be calculated; check the path and that the file has been written."
                    )
                file_structure = ManifestFileRecord(filename=file, checksum=calculate_checksum(file))
            if cls.check_file_structure(file_structure, existing_names, existing_checksums):
                result.append(file_structure)
                existing_names.add(str(file_structure.filename))
                existing_checksums.add(file_structure.checksum)
        return result

    @computed_field
    @property
    def ulid_code(self) -> ULID | None:
        """ULID from ``filename``, or None if there is no filename."""
        return None if self.filename is None else self.filename.filename_parts.ulid_code

    @field_serializer("filename")
    def serialize_filename(self, filename: ManifestFilename | None, _info) -> str | None:
        """Serialize the filename as a string, or None.

        Parameters
        ----------
        filename : ManifestFilename or None
            Filename to serialize.
        _info : pydantic.SerializationInfo
            Unused.

        Returns
        -------
        str or None
            The filename's path as a string.
        """
        return None if filename is None else str(filename)

    @classmethod
    def from_file(cls, filepath: str | Path | S3Path | ManifestFilename) -> "Manifest":
        """Read a manifest file.

        Parameters
        ----------
        filepath : str, Path, S3Path or ManifestFilename
            Local or S3 path of the manifest file.

        Returns
        -------
        Manifest
            The manifest, with ``filename`` set to ``filepath``.

        Raises
        ------
        ManifestError
            If the path is not a valid manifest filename, or the filename or ULID stored in the file disagrees with
            the ULID in the path.
        pydantic_core.ValidationError
            If the contents are not a valid manifest, or their manifest_type disagrees with the INPUT/OUTPUT label in
            the path.
        """
        try:
            filename = filepath if isinstance(filepath, ManifestFilename) else ManifestFilename(filepath)
        except ValueError as err:
            raise ManifestError(
                f"Manifest file {filepath} does not have a valid manifest filename; {_MANIFEST_NAME_HELP}"
            ) from err

        with smart_open(filename.path) as manifest_file:
            contents = json.loads(manifest_file.read())

        path_ulid = filename.filename_parts.ulid_code
        stored_ulid = contents.pop("ulid_code", None)
        stored_filename = contents.pop("filename", None)
        if stored_ulid is not None and str(stored_ulid) != str(path_ulid):
            raise ManifestError(
                f"Manifest file {filepath} stores ulid_code {stored_ulid}, which disagrees with its name."
            )
        if stored_filename is not None:
            try:
                stored_filename_ulid = ManifestFilename(stored_filename).filename_parts.ulid_code
            except ValueError:
                stored_filename_ulid = None
            if stored_filename_ulid is not None and stored_filename_ulid != path_ulid:
                raise ManifestError(
                    f"Manifest file {filepath} stores filename {stored_filename}, whose ULID disagrees with its name."
                )

        contents["filename"] = filename
        return cls.model_validate(contents)

    def add_files(self, *files: str | Path | S3Path) -> None:
        """Add files to the manifest, checksumming each one.

        Parameters
        ----------
        *files : str, Path or S3Path
            Absolute local or S3 paths of the files to add.

        Raises
        ------
        FileNotFoundError
            If a file cannot be found. No file is added.
        ValueError
            If a path is not absolute.
        """
        self.files = [*self.files, *files]

    def validate_checksums(self) -> None:
        """Check every listed file against its recorded checksum.

        Raises
        ------
        ManifestError
            Listing every file that cannot be found or whose checksum differs.
        """
        failed_filenames = []
        for file_structure in self.files:
            checksum_expected = file_structure.checksum
            filename = file_structure.filename
            if not AnyPath(filename).exists():
                logger.error(f"Checksum validation for {filename} failed. The file cannot be found.")
            else:
                checksum_calculated = calculate_checksum(filename)
                if checksum_expected == checksum_calculated:
                    continue
                logger.error(
                    f"Checksum validation for {filename} failed. "
                    f"Expected {checksum_expected} but got {checksum_calculated}."
                )
            failed_filenames.append(str(filename))
        if failed_filenames:
            raise ManifestError(f"Files failed checksum validation: {', '.join(failed_filenames)}")

    def write(self, out_path: str | Path | S3Path | ManifestFilename, *, overwrite: bool = False) -> Path | S3Path:
        """Write the manifest to a file and set ``filename`` to the path written.

        Parameters
        ----------
        out_path : str, Path, S3Path or ManifestFilename
            Full path of the manifest file, or a directory or S3 prefix to write into under the name in
            ``filename`` (or a generated name with a new ULID). A path whose last component has an extension is
            taken as a file name and must be a valid manifest filename, so a directory such as ``/data/v1.2`` is
            refused; give the full manifest file path inside it instead. An S3 bucket root is always a prefix, even
            when the bucket name contains a dot.
        overwrite : bool, optional
            Whether to overwrite an existing file. Default False.

        Returns
        -------
        Path or S3Path
            The path written.

        Raises
        ------
        ManifestError
            If ``out_path`` names a manifest file whose INPUT/OUTPUT label or ULID disagrees with this manifest, or
            has an extension but is not a valid manifest filename.
        FileExistsError
            If the target exists and ``overwrite`` is False.
        """
        has_directory = self.filename is not None and str(self.filename) != self.filename.path.name
        directory = out_path.path if isinstance(out_path, ManifestFilename) else AnyPath(out_path)
        try:
            target = ManifestFilename(directory)  # out_path names the manifest file itself
        except ValueError:
            # Note: If a user provides a manifest filename without an extension, it reads as a directory
            is_bucket_root = isinstance(directory, S3Path) and not directory.key  # a bucket name may contain dots
            if directory.suffix and not is_bucket_root:
                raise ManifestError(
                    f"Refusing to write to {directory}, which has an extension but is not a valid manifest "
                    f"filename; {_MANIFEST_NAME_HELP}"
                ) from None
            if self.filename is not None:
                name = self.filename.path.name
            else:
                name = ManifestFilename.from_filename_parts(
                    manifest_type=self.manifest_type, ulid_code=ULID()
                ).path.name
            target = ManifestFilename(directory / name)

        target_parts = target.filename_parts
        if target_parts.manifest_type != self.manifest_type:
            raise ManifestError(
                f"Refusing to write a {self.manifest_type} manifest to {target}, which is named as a "
                f"{target_parts.manifest_type} manifest."
            )
        if self.ulid_code is not None and target_parts.ulid_code != self.ulid_code:
            message = (
                f"Refusing to write manifest with ULID {self.ulid_code} to {target}, which carries ULID "
                f"{target_parts.ulid_code}."
            )
            if self.manifest_type == ManifestType.OUTPUT:
                message += (
                    " You should never need to change the ULID of an output manifest; use the filename already on "
                    "the output manifest object."
                )
            raise ManifestError(message)

        text = self.model_copy(update={"filename": target}).model_dump_json()
        with smart_open(target.path, "w" if overwrite else "x") as manifest_file:
            manifest_file.write(text)

        if has_directory and self.filename.path != target.path:
            logger.warning(
                f"Internal manifest filename was updated by the latest write from {self.filename} to {target}"
            )
        self.filename = target
        return target.path

    def add_desired_time_range(self, start_datetime: datetime, end_datetime: datetime) -> None:
        """Add a time range to the configuration section of the manifest.

        Parameters
        ----------
        start_datetime : datetime.datetime
            The desired start time for the range of data in this manifest
        end_datetime : datetime.datetime
            The desired end time for the range of data in this manifest
        """
        self.configuration["start_time"] = start_datetime.strftime("%Y-%m-%d:%H:%M:%S")
        self.configuration["end_time"] = end_datetime.strftime("%Y-%m-%d:%H:%M:%S")

    @classmethod
    def output_manifest_from_input_manifest(
        cls,
        input_manifest: Union[str, Path, S3Path, ManifestFilename, "Manifest"],
        configuration: dict[str, Any] | None = None,
    ) -> "Manifest":
        """Create an output manifest with the input manifest's ULID.

        The input manifest's files are recorded in ``configuration["input_manifest_files"]``.

        Parameters
        ----------
        input_manifest : str, Path, S3Path, ManifestFilename or Manifest
            Path to an input manifest file, or an input manifest read with ``from_file``.
        configuration : dict, optional
            Additional configuration items for the output manifest. An ``input_manifest_files`` key is overwritten,
            with a warning.

        Returns
        -------
        Manifest
            The new output manifest, with a bare filename.

        Raises
        ------
        ManifestError
            If the input manifest has no ULID.
        """
        if not isinstance(input_manifest, cls):
            input_manifest = cls.from_file(input_manifest)

        if input_manifest.ulid_code is None:
            raise ManifestError(
                "The input manifest has no ULID (no filename), so an output manifest cannot preserve its lineage."
            )

        configuration = dict(configuration or {})
        if "input_manifest_files" in configuration:
            logger.warning(
                "The configuration passed to output_manifest_from_input_manifest contains input_manifest_files; it "
                "is replaced by the files of the input manifest."
            )
        configuration["input_manifest_files"] = input_manifest.files

        return cls(
            manifest_type=ManifestType.OUTPUT,
            filename=ManifestFilename.from_filename_parts(
                manifest_type=ManifestType.OUTPUT, ulid_code=input_manifest.ulid_code
            ),
            configuration=configuration,
        )
