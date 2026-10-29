"""Module for manifest file handling"""

import json
import logging
from collections.abc import Iterable
from datetime import datetime
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


_MANIFEST_NAME_HELP = (
    "expected LIBERA_INPUT_MANIFEST_<ULID>.json or LIBERA_OUTPUT_MANIFEST_<ULID>.json, where <ULID> is a "
    "26-character ULID, for example LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json. Manifest.write() "
    "generates this name when out_path is a directory or S3 prefix, so it need not be given."
)


class Manifest(BaseModel):
    """Pydantic model for a manifest file.

    Attributes
    ----------
    manifest_type : ManifestType
        INPUT or OUTPUT.
    files : list of str
        Absolute local or S3 paths of the files listed in the manifest. May be given as str, Path or S3Path, and are
        stored as str. Files are never read, so files with identical content are each listed; a path listed twice is
        a validation error. Only ``add_files`` checks that a file exists and is a file.
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
    files: list[str] = Field(default_factory=list, description="Absolute paths of the files listed in the manifest.")
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
    def _normalize_files(cls, raw_files: Iterable[str | Path | S3Path], existing_names: set[str]) -> list[str]:
        """Convert paths to absolute path strings, rejecting any path listed twice.

        Files are not opened or checked for existence.

        Parameters
        ----------
        raw_files : collections.abc.Iterable[str | pathlib.Path | cloudpathlib.S3Path]
            Absolute local or S3 paths of the files.
        existing_names : set of str
            Paths already listed in the manifest.

        Returns
        -------
        list of str
            The paths as strings, in order.

        Raises
        ------
        ValueError
            If an entry is not a path, a path is not absolute, or a path is already listed or appears twice.
        """
        result = []
        seen = set(existing_names)
        for raw_file in raw_files:
            if not isinstance(raw_file, str | Path | S3Path):
                raise ValueError(
                    f"Manifest file entry {raw_file!r} is not a path. A manifest lists each file as an absolute local "
                    'or S3 path string, for example "s3://bucket/LIBERA_L1B_RAD-4CH_...nc".'
                )
            file = str(AnyPath(raw_file))
            # S3 paths are always absolute so this is always valid for them
            if not AnyPath(file).is_absolute():
                raise ValueError(f"The file path for {file} must be an absolute path.")
            if file in seen:
                raise ValueError(
                    f"The file {file} is already listed in the manifest. A manifest lists each file once; remove the "
                    "repeated path."
                )
            result.append(file)
            seen.add(file)
        return result

    @field_validator("files", mode="before")  # noqa  avoid type warning
    @classmethod
    def transform_files(cls, raw_list: list[str | Path | S3Path] | None) -> list[str]:
        """Convert the incoming paths to absolute path strings.

        Files are not opened or checked for existence.

        Parameters
        ----------
        raw_list : list of str, Path or S3Path, or None
            Absolute local or S3 paths of the files.

        Returns
        -------
        list of str
            The paths as strings.

        Raises
        ------
        ValueError
            If an entry is not a path, a path is not absolute, or a path appears twice.
        """
        return cls._normalize_files(raw_list or [], set())

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
        """Add files to the manifest after checking that each one exists and is a file.

        Files are not read, so an S3 file is not downloaded and a file larger than local disk can be listed.

        Parameters
        ----------
        *files : str, Path or S3Path
            Absolute local or S3 paths of the files to add, not directories.

        Raises
        ------
        FileNotFoundError
            If a file cannot be found. No file is added.
        IsADirectoryError
            If a path exists but is not a file, such as a local directory, an S3 bucket root or an S3 prefix. No
            file is added.
        ValueError
            If a path is not absolute, or is already listed or given twice. No file is added.
        """
        # Every path is checked before any file check, so a relative or repeated path is reported as such,
        # not as missing, and costs no request to S3
        new_files = self._normalize_files(files, set(self.files))
        for file in new_files:
            path = AnyPath(file)
            if path.is_file():
                continue
            if path.exists():
                raise IsADirectoryError(
                    f"{file} is a directory or S3 prefix, not a file. A manifest lists files only; add each file in "
                    "it by its full path."
                )
            raise FileNotFoundError(
                f"File {file} cannot be found. A file must exist when it is added to a manifest; check the path "
                "and that the file has been written."
            )
        # Extended in place: the new paths are already validated against the list, so assignment would only repeat it
        self.files.extend(new_files)

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
        # A copy, since add_files extends the input manifest's list in place
        configuration["input_manifest_files"] = list(input_manifest.files)

        return cls(
            manifest_type=ManifestType.OUTPUT,
            filename=ManifestFilename.from_filename_parts(
                manifest_type=ManifestType.OUTPUT, ulid_code=input_manifest.ulid_code
            ),
            configuration=configuration,
        )
