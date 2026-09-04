"""Module for manifest file handling"""

import json
import logging
import warnings
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import md5
from pathlib import Path
from typing import Any, Union

from cloudpathlib import AnyPath, S3Path
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from ulid import ULID

from libera_utils.constants import ManifestType
from libera_utils.io.filenaming import AbstractValidFilename, ManifestFilename, PathType
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
    """Pydantic model for a manifest file.

    Notes
    -----
    - The manifest ULID lives in the filename. ``ulid_code`` is derived from ``filename`` and cannot get out of sync
      with it. Passing ``ulid_code=`` to the constructor is still accepted: without a ``filename`` it produces a bare
      ``LIBERA_<TYPE>_MANIFEST_<ULID>.json`` filename, and with one it must agree with the filename's ULID.
    - Filename validation is lenient when reading (a badly named file is read with a warning and ``filename=None``)
      and strict when writing (``write()`` and ``save()`` refuse to produce a badly named file).
    - A manifest returned by ``from_file`` is *file-backed*: it remembers where it came from (``source_path``) so it
      can be written back with ``save()``. ``copy()`` returns a detached copy for building a new manifest from it.
    - Algorithm developers normally only need ``from_file`` (read the INPUT manifest handed to the container),
      ``for_output_from_input`` (build the OUTPUT manifest with the same ULID), ``add_files`` and ``write``.
    """

    manifest_type: ManifestType = Field(description="Either INPUT or OUTPUT.")
    files: list[ManifestFileRecord] = Field(default_factory=list, description="List of ManifestFileStructure.")
    configuration: dict[str, Any] = Field(
        default_factory=dict, description="Freeform json-compatible dictionary of configuration items."
    )
    filename: ManifestFilename | None = Field(
        default=None,
        description="Preset filename, optional. May be a bare filename or a full local/S3 path. The ULID code of "
        "the manifest is taken from here.",
    )

    # Where this manifest was read from, when it was produced by ``from_file``. Not serialized.
    _source_path: PathType | None = PrivateAttr(default=None)

    model_config = ConfigDict(
        # Allow using ManifestFilename as a field
        arbitrary_types_allowed=True
    )

    # ------------------------------------------------------------------------------------------------------------
    # Validation and (de)serialization
    # ------------------------------------------------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def reconcile_ulid_code(cls, data: Any) -> Any:
        """Fold a ``ulid_code`` constructor argument into ``filename``.

        ``ulid_code`` is not a stored field (it is computed from the filename), but it is still accepted as input so
        a manifest can be given a ULID without spelling out a filename. If both are given they must agree.
        """
        if not isinstance(data, dict) or data.get("ulid_code") is None:
            return data
        data = dict(data)
        ulid_code = data.pop("ulid_code")
        if not isinstance(ulid_code, ULID):
            ulid_code = ULID.from_str(str(ulid_code))

        filename = data.get("filename")
        if filename is None:
            manifest_type = data.get("manifest_type")
            if manifest_type is None:
                return data  # Let the field validation report the missing manifest_type
            data["filename"] = ManifestFilename.from_filename_parts(
                manifest_type=ManifestType(manifest_type), ulid_code=ulid_code
            )
            return data

        filename_ulid = get_ulid_code(filename)
        if filename_ulid != ulid_code:
            raise ValueError(
                f"ulid_code {ulid_code} disagrees with the ULID {filename_ulid} in filename {filename}. "
                "The filename is the source of truth for a manifest ULID; pass only one of the two."
            )
        return data

    @field_validator("filename", mode="before")  # noqa  avoid type warning
    @classmethod
    def transform_filename(cls, raw_filename: str | Path | S3Path | ManifestFilename | None) -> ManifestFilename | None:
        """Convert raw filename to ManifestFilename class if necessary."""
        if raw_filename is None:
            return None
        if isinstance(raw_filename, ManifestFilename):
            return raw_filename
        return ManifestFilename(raw_filename)

    @model_validator(mode="after")
    def warn_on_manifest_type_mismatch(self) -> "Manifest":
        """Warn when the filename says INPUT/OUTPUT but the contents say otherwise.

        This is tolerated in memory (the file contents may still be perfectly usable) but ``write()`` and ``save()``
        refuse to produce such a file.
        """
        if self.filename is not None and self.filename.filename_parts.manifest_type != self.manifest_type:
            logger.warning(
                f"Manifest filename {self.filename} is named as a "
                f"{self.filename.filename_parts.manifest_type} manifest but its manifest_type is {self.manifest_type}."
            )
        return self

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
    def serialize_filename(self, filename: ManifestFilename | None, _info) -> str | None:
        """Custom serializer for the manifest filename."""
        return None if filename is None else str(filename)

    @computed_field(description="ULID code of the manifest, taken from its filename. None when there is no filename.")  # type: ignore[prop-decorator]
    @property
    def ulid_code(self) -> ULID | None:
        """ULID code of this manifest, derived from ``filename``."""
        return get_ulid_code(self.filename)

    # ------------------------------------------------------------------------------------------------------------
    # File-state tracking
    # ------------------------------------------------------------------------------------------------------------
    @property
    def source_path(self) -> PathType | None:
        """Path this manifest was read from by ``from_file``, or None for a manifest built programmatically."""
        return self._source_path

    @property
    def is_file_backed(self) -> bool:
        """True when this manifest was read from a file and ``save()`` can write it back there."""
        return self._source_path is not None

    # ------------------------------------------------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_file(cls, filepath: str | Path | S3Path | ManifestFilename) -> "Manifest":
        """Read a manifest file and return a file-backed Manifest object (factory method).

        The path on disk is the source of truth for the manifest filename and ULID: any ``filename`` or
        ``ulid_code`` recorded inside the JSON is ignored (with a warning if the ULID disagrees). A file whose name is
        not a valid manifest filename is still read, with a warning, but has ``filename=None`` and no ULID, so it
        cannot be saved or used to derive an output manifest until it is renamed.

        Parameters
        ----------
        filepath : Union[str, Path, S3Path, ManifestFilename]
            Location of manifest file to read.

        Returns
        -------
        Manifest
            Pydantic model built from the json of the given manifest file. ``source_path`` is set to ``filepath``.
        """
        path = filepath.path if isinstance(filepath, ManifestFilename) else AnyPath(filepath)
        with smart_open(path) as manifest_file:
            contents = json.loads(manifest_file.read())

        contents.pop("filename", None)
        stored_ulid = contents.pop("ulid_code", None)
        filename = cls._validate_filename_for_read(path)
        if (
            filename is not None
            and stored_ulid is not None
            and str(filename.filename_parts.ulid_code) != str(stored_ulid)
        ):
            logger.warning(
                f"Manifest file {path} records ulid_code {stored_ulid} in its contents but its filename carries "
                f"{filename.filename_parts.ulid_code}. Using the filename ULID."
            )
        contents["filename"] = filename

        manifest = cls.model_validate(contents)
        manifest._source_path = path
        return manifest

    @classmethod
    def for_input(
        cls,
        files: Iterable[str | Path | S3Path | ManifestFileRecord | dict] = (),
        configuration: dict[str, Any] | None = None,
        ulid_code: ULID | None = None,
    ) -> "Manifest":
        """Create a new INPUT manifest with a fresh (or given) ULID (factory method).

        The manifest gets a bare ``LIBERA_INPUT_MANIFEST_<ULID>.json`` filename immediately, so its ULID is known
        before it is written and ``write(directory)`` places it under that name.

        Parameters
        ----------
        files : Iterable[Union[str, Path, S3Path, ManifestFileRecord, dict]], Optional
            Files to record in the manifest. Checksums are calculated for paths, so they must exist.
        configuration : dict, Optional
            Freeform json-compatible configuration.
        ulid_code : ULID, Optional
            ULID to use for the manifest. Generated from the current time if not provided.

        Returns
        -------
        Manifest
        """
        filename = ManifestFilename.from_filename_parts(
            manifest_type=ManifestType.INPUT, ulid_code=ulid_code or ULID.from_datetime(datetime.now(UTC))
        )
        return cls(
            manifest_type=ManifestType.INPUT, files=list(files), configuration=configuration or {}, filename=filename
        )

    @classmethod
    def for_output_from_input(
        cls,
        input_manifest: Union[str, Path, S3Path, ManifestFilename, "Manifest"],
        files: Iterable[str | Path | S3Path | ManifestFileRecord | dict] = (),
        configuration: dict[str, Any] | None = None,
    ) -> "Manifest":
        """Create an OUTPUT manifest that carries the ULID of its INPUT manifest (factory method).

        The input manifest's file records are stored under ``configuration["input_manifest_files"]`` for lineage.

        Parameters
        ----------
        input_manifest : Union[str, Path, S3Path, ManifestFilename, Manifest]
            The input manifest object, or a path to read it from.
        files : Iterable[Union[str, Path, S3Path, ManifestFileRecord, dict]], Optional
            Output files to record. Files can also be added later with ``add_files``.
        configuration : dict, Optional
            Extra configuration entries, merged over the lineage entry.

        Returns
        -------
        Manifest
            The new output manifest, with a bare ``LIBERA_OUTPUT_MANIFEST_<ULID>.json`` filename.

        Raises
        ------
        ManifestError
            If the input manifest has no ULID (it was read from a badly named file, or built without a filename).
        """
        if not isinstance(input_manifest, cls):
            input_manifest = cls.from_file(input_manifest)

        ulid_code = input_manifest.ulid_code
        if ulid_code is None:
            raise ManifestError(
                "Cannot derive an output manifest: the input manifest has no ULID because it has no valid manifest "
                f"filename (filename={input_manifest.filename}, source_path={input_manifest.source_path}). "
                "Output manifests must carry the ULID of their input manifest for traceability."
            )
        if input_manifest.manifest_type != ManifestType.INPUT:
            logger.warning(
                f"Deriving an output manifest from a manifest whose type is {input_manifest.manifest_type}, not INPUT."
            )

        full_configuration: dict[str, Any] = {"input_manifest_files": input_manifest.files}
        full_configuration.update(configuration or {})
        return cls(
            manifest_type=ManifestType.OUTPUT,
            files=list(files),
            configuration=full_configuration,
            filename=ManifestFilename.from_filename_parts(manifest_type=ManifestType.OUTPUT, ulid_code=ulid_code),
        )

    @classmethod
    def output_manifest_from_input_manifest(
        cls, input_manifest: Union[str, Path, S3Path, ManifestFilename, "Manifest"]
    ) -> "Manifest":
        """Create Output manifest from input manifest file path, adds input files to output manifest configuration

        .. deprecated::
            Use :meth:`Manifest.for_output_from_input` instead. This method is a thin alias and will be removed.

        Parameters
        ----------
        input_manifest : Union[str, Path, S3Path, ManifestFilename, Manifest]
            An S3 or regular path to an input_manifest object, or the input manifest object itself

        Returns
        -------
        output_manifest : Manifest
            The newly created output manifest
        """
        warnings.warn(
            "Manifest.output_manifest_from_input_manifest is deprecated; use Manifest.for_output_from_input instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.for_output_from_input(input_manifest)

    def copy(self) -> "Manifest":  # type: ignore[override]  # shadows pydantic's deprecated BaseModel.copy
        """Return a detached deep copy of this manifest.

        The copy is not file-backed (``save()`` will refuse), so a manifest read from a file can be copied and
        modified freely before being written somewhere else with ``write()``.

        Returns
        -------
        Manifest
        """
        new = self.model_copy(deep=True)
        new._source_path = None
        return new

    # ------------------------------------------------------------------------------------------------------------
    # Content manipulation
    # ------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------------------------------------------------
    def _generate_filename(self) -> ManifestFilename:
        """Generate a valid manifest filename"""
        mfn = ManifestFilename.from_filename_parts(
            manifest_type=self.manifest_type, ulid_code=ULID.from_datetime(datetime.now(UTC))
        )
        return mfn

    @staticmethod
    def _validate_filename_for_read(path: PathType) -> ManifestFilename | None:
        """Lenient filename validation used when reading: warn and return None for an invalid manifest filename."""
        try:
            return ManifestFilename(path)
        except ValueError:
            logger.warning(
                f"Manifest file {path} does not have a valid manifest filename "
                "(expected LIBERA_<INPUT|OUTPUT>_MANIFEST_<ULID>.json). Reading it anyway; the resulting Manifest "
                "has no filename or ULID, so it cannot be saved or used to derive an output manifest."
            )
            return None

    def _validate_filename_for_write(self, path: PathType) -> ManifestFilename:
        """Strict filename validation used when writing: raise ManifestError for an invalid or mismatched name."""
        try:
            target = ManifestFilename(path)
        except ValueError as e:
            raise ManifestError(
                f"Refusing to write manifest to {path}: not a valid manifest filename "
                "(expected LIBERA_<INPUT|OUTPUT>_MANIFEST_<ULID>.json)."
            ) from e
        if target.filename_parts.manifest_type != self.manifest_type:
            raise ManifestError(
                f"Refusing to write a {self.manifest_type} manifest to {path}, which is named as a "
                f"{target.filename_parts.manifest_type} manifest."
            )
        return target

    def _resolve_write_path(
        self, out_path: str | Path | S3Path | ManifestFilename, filename: str | ManifestFilename | None
    ) -> ManifestFilename:
        """Work out and validate the full path ``write()`` should produce.

        ``out_path`` may be a directory (or S3 prefix) or a full manifest file path. See ``write`` for the rules.
        """
        out_path = out_path.path if isinstance(out_path, ManifestFilename) else AnyPath(out_path)

        if filename is not None:
            filename = str(filename)
            if AnyPath(filename).name != filename:
                raise ManifestError(
                    f"filename={filename!r} must be a bare manifest filename with no directory part; pass the "
                    "directory (or a full file path) as out_path instead."
                )

        try:
            full_path = ManifestFilename(out_path)  # out_path already names a manifest file
        except ValueError:
            full_path = None

        if full_path is not None:
            if filename is not None and filename != full_path.path.name:
                raise ManifestError(
                    f"out_path {out_path} already names a manifest file but a different filename {filename!r} was "
                    "also given. Pass one or the other."
                )
            return self._validate_filename_for_write(full_path.path)

        if filename is None:
            filename = (self.filename or self._generate_filename()).path.name
        return self._validate_filename_for_write(out_path / filename)

    def _dump_for_path(self, target: ManifestFilename) -> str:
        """Serialize the manifest as it should appear on disk at ``target`` without mutating ``self``.

        The on-disk ``filename`` (and hence ``ulid_code``) always reflect where the file was actually written.
        """
        return self.model_copy(update={"filename": target}).model_dump_json()

    def write(
        self, out_path: str | Path | S3Path | ManifestFilename, filename: str | ManifestFilename | None = None
    ) -> PathType:
        """Write a manifest file from a Manifest object (self).

        This has no side effects on the object: ``filename`` is not modified, so a manifest can be written to
        several locations. Writing refuses to create a file that is not a valid manifest filename or whose name
        disagrees with ``manifest_type``, and fails if the target already exists.

        Parameters
        ----------
        out_path : Union[str, Path, S3Path, ManifestFilename]
            Either a directory (or S3 prefix) to write into, or the full path of the manifest file to write.
            It is treated as a full file path when its last component is a valid manifest filename.
        filename : Union[str, ManifestFilename], Optional
            Bare filename to write within ``out_path`` (no directory part); must be a valid manifest filename.
            If not provided, the basename of the object's ``filename`` attribute is used. If that is not set either,
            a filename with a fresh ULID is generated. Must not disagree with ``out_path`` when that is a full path.

        Returns
        -------
        Union[Path, S3Path]
            The path where the manifest file is written.

        Raises
        ------
        ManifestError
            If the resulting path is not a valid manifest filename, its type disagrees with ``manifest_type``, or
            the ``out_path`` and ``filename`` arguments conflict.
        """
        target = self._resolve_write_path(out_path, filename)
        with smart_open(target.path, "x") as manifest_file:
            manifest_file.write(self._dump_for_path(target))
        return target.path

    def save(self) -> PathType:
        """Write this manifest back to the file it was read from, overwriting it.

        Only available for file-backed manifests (those produced by ``from_file``). Use ``write()`` to write a
        programmatically built manifest, or ``copy()`` to detach a file-backed one first.

        Returns
        -------
        Union[Path, S3Path]
            The path where the manifest file is written.

        Raises
        ------
        ManifestError
            If the manifest is not file-backed, or its source filename is not a valid manifest filename for its type.
        """
        if self._source_path is None:
            raise ManifestError(
                "This manifest was not read from a file, so there is nowhere to save it back to. Use write() instead."
            )
        target = self._validate_filename_for_write(self._source_path)
        with smart_open(target.path, "w") as manifest_file:
            manifest_file.write(self._dump_for_path(target))
        return target.path
