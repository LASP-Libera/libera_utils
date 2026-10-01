"""Tests for manifest module"""

import json
import sys
from datetime import datetime, timedelta
from hashlib import md5
from pathlib import Path

import pytest
from cloudpathlib import S3Path
from pydantic import ValidationError
from ulid import ULID

from libera_utils.constants import ManifestType
from libera_utils.io.filenaming import ManifestFilename
from libera_utils.io.manifest import Manifest, ManifestError, ManifestFileRecord
from libera_utils.io.smart_open import smart_open

VALID_ULID = "01GDHWG4R0W8KXWY0KRDD6BZTT"
OTHER_ULID = "01MBAK5DC06HX46P3PG0M6HJR0"
INPUT_NAME = f"LIBERA_INPUT_MANIFEST_{VALID_ULID}.json"
OUTPUT_NAME = f"LIBERA_OUTPUT_MANIFEST_{VALID_ULID}.json"
TEST_DATA_MANIFEST = Path(sys.modules[__name__.split(".")[0]].__file__).parent / "test_data" / INPUT_NAME


def write_manifest_json(path: Path, **overrides) -> Path:
    """Write the test data manifest's JSON to path, with top-level keys overridden"""
    contents = json.loads(TEST_DATA_MANIFEST.read_text())
    contents.update(overrides)
    path.write_text(json.dumps(contents))
    return path


class TestManifestConstruction:
    """Constructing manifests, the ulid_code property, and serialization"""

    @pytest.mark.parametrize(
        ("man_path", "man_files", "man_type", "man_config"),
        [
            (
                "subfolder/LIBERA_INPUT_MANIFEST_201GDHWG4R0W8KXWY0KRDD6BZTT.json",
                [{"filename": "relative/file.txt", "checksum": "fakesum"}],
                ManifestType.OUTPUT,
                None,
            ),
            (f"subfolder/{INPUT_NAME}", None, ManifestType.INPUT, ["config"]),
            (f"subfolder/{INPUT_NAME}", None, ManifestType.OUTPUT, None),  # INPUT name on an OUTPUT manifest
        ],
    )
    def test_validation_failure(self, man_path, man_files, man_type, man_config):
        """Test manifest validation method for correct failure cases"""
        with pytest.raises(ValidationError):
            _ = Manifest(manifest_type=man_type, files=man_files, configuration=man_config, filename=man_path)

    @pytest.mark.parametrize(
        ("man_path", "man_files", "man_type", "man_config"),
        [
            (
                None,
                [{"filename": "s3://abs/file.txt", "checksum": "fakesum"}],
                ManifestType.OUTPUT,
                {"data": "description"},
            ),
            (f"subfolder/{INPUT_NAME}", None, ManifestType.INPUT, {"data": "description"}),
            (None, None, ManifestType.OUTPUT, {"data": "description"}),
        ],
    )
    def test_validation_success(self, man_path, man_files, man_type, man_config):
        """Test manifest validation method for correct success cases"""
        _ = Manifest(manifest_type=man_type, files=man_files, configuration=man_config, filename=man_path)

    def test_ulid_code_is_read_only(self):
        """ulid_code cannot be assigned; set filename instead"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        with pytest.raises((AttributeError, ValueError)):
            m.ulid_code = ULID.from_str(OTHER_ULID)
        assert m.ulid_code == ULID.from_str(VALID_ULID)

    def test_ulid_code_argument_without_filename(self):
        """A ulid_code argument alone produces a bare manifest filename"""
        m = Manifest(manifest_type=ManifestType.OUTPUT, ulid_code=VALID_ULID)
        assert str(m.filename) == OUTPUT_NAME

    def test_ulid_code_argument_must_agree_with_filename(self):
        """A ulid_code argument given with a filename must match it"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME, ulid_code=VALID_ULID)
        assert m.ulid_code == ULID.from_str(VALID_ULID)
        with pytest.raises(ValidationError, match="disagrees"):
            Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME, ulid_code=OTHER_ULID)

    def test_serialization_without_filename(self):
        """A missing filename serializes as null, not the string 'None'"""
        dumped = json.loads(Manifest(manifest_type=ManifestType.INPUT).model_dump_json())
        assert dumped["filename"] is None
        assert dumped["ulid_code"] is None

    def test_add_desired_time_range(self, test_jpss_manifest):
        """Test adding a time range to a manifest file"""
        m = Manifest.from_file(test_jpss_manifest)
        start = datetime.now()
        end = start + timedelta(hours=1)
        m.add_desired_time_range(start, end)

        assert "start_time" in m.configuration.keys()
        assert "end_time" in m.configuration.keys()


class TestFilenameAssignment:
    """Assigning the filename attribute is validated"""

    @pytest.mark.parametrize(
        "filename",
        [
            INPUT_NAME,
            f"relative/dir/{INPUT_NAME}",
            f"/abs/dir/{INPUT_NAME}",
            f"s3://some-bucket/prefix/{INPUT_NAME}",
            ManifestFilename(f"/abs/dir/{INPUT_NAME}"),
        ],
    )
    def test_valid_assignment(self, filename):
        """Any valid manifest filename, relative or absolute, str or ManifestFilename, is accepted"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        m.filename = filename
        assert isinstance(m.filename, ManifestFilename)
        assert str(m.filename) == str(filename)
        assert m.ulid_code == ULID.from_str(VALID_ULID)

    def test_none_assignment(self):
        """Clearing the filename clears the ULID"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        m.filename = None
        assert m.ulid_code is None

    @pytest.mark.parametrize("filename", ["some/path.json", f"{INPUT_NAME}.bak", OUTPUT_NAME])
    def test_invalid_assignment_keeps_old_value(self, filename):
        """An invalid name, or a name with the wrong INPUT/OUTPUT label, raises and leaves filename unchanged"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        with pytest.raises(ValidationError):
            m.filename = filename
        assert str(m.filename) == INPUT_NAME

    def test_manifest_type_assignment_checked_against_filename(self):
        """Changing manifest_type so it disagrees with the filename raises and leaves manifest_type unchanged"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        with pytest.raises(ValidationError):
            m.manifest_type = ManifestType.OUTPUT
        assert m.manifest_type == ManifestType.INPUT

    def test_assignment_may_change_ulid(self):
        """Assigning a new filename directly is how a manifest's ULID is changed"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        m.filename = f"LIBERA_INPUT_MANIFEST_{OTHER_ULID}.json"
        assert m.ulid_code == ULID.from_str(OTHER_ULID)


class TestFromFile:
    """Reading manifest files"""

    def test_from_file(self, test_jpss_manifest):
        """Test factory method for creating a manifest object from a filepath"""
        m = Manifest.from_file(test_jpss_manifest)
        assert m.manifest_type == ManifestType.INPUT
        assert isinstance(m.files, list)
        assert isinstance(m.configuration, dict)
        assert m.filename.path == test_jpss_manifest
        assert m.ulid_code == ULID.from_str(VALID_ULID)

    def test_from_file_manifest_filename(self, test_jpss_manifest):
        """A ManifestFilename is accepted as the path"""
        m = Manifest.from_file(ManifestFilename(test_jpss_manifest))
        assert m.filename.path == test_jpss_manifest

    def test_from_file_s3(self, test_jpss_manifest, write_file_to_s3):
        """Test loading a file from S3"""
        file_key = f"s3://test-manifest-from-file-s3-bucket/{INPUT_NAME}"
        s3_path = write_file_to_s3(test_jpss_manifest, file_key)
        m = Manifest.from_file(s3_path)
        assert m.manifest_type == ManifestType.INPUT
        assert isinstance(m.files, list)
        assert isinstance(m.configuration, dict)

    def test_invalid_filename(self, tmp_path):
        """A file that is not named as a manifest is an error"""
        path = write_manifest_json(tmp_path / "not_a_manifest.json")
        with pytest.raises(ManifestError, match="does not have a valid manifest filename"):
            Manifest.from_file(path)

    @pytest.mark.parametrize(
        "stored",
        [
            {"ulid_code": OTHER_ULID},
            {"filename": f"/somewhere/LIBERA_INPUT_MANIFEST_{OTHER_ULID}.json"},
        ],
    )
    def test_stored_ulid_disagrees_with_path(self, tmp_path, stored):
        """A stored ulid_code or filename whose ULID disagrees with the path is an error"""
        path = write_manifest_json(tmp_path / INPUT_NAME, **stored)
        with pytest.raises(ManifestError, match="disagrees with its name"):
            Manifest.from_file(path)

    def test_stored_filename_in_another_directory(self, tmp_path):
        """A manifest moved to another directory reads normally; the path it is read from wins"""
        path = write_manifest_json(
            tmp_path / INPUT_NAME, filename=f"s3://old-bucket/{INPUT_NAME}", ulid_code=VALID_ULID
        )
        m = Manifest.from_file(path)
        assert m.filename.path == path

    def test_type_label_disagrees_with_contents(self, tmp_path):
        """INPUT contents in a file named as an OUTPUT manifest is an error"""
        path = write_manifest_json(tmp_path / OUTPUT_NAME)
        with pytest.raises(ValidationError):
            Manifest.from_file(path)

    def test_stored_null_checksum_is_kept(self, tmp_path, test_txt):
        """A null checksum on disk is not recomputed on read"""
        path = write_manifest_json(tmp_path / INPUT_NAME, files=[{"filename": str(test_txt), "checksum": None}])
        assert Manifest.from_file(path).files[0].checksum is None

    def test_round_trip(self, tmp_path, test_txt):
        """A written manifest reads back with the same contents and ULID, and reading leaves the file unchanged"""
        m = Manifest(manifest_type=ManifestType.INPUT, files=[test_txt], configuration={"key": "value"})
        path = m.write(tmp_path)
        before = path.read_bytes()
        read = Manifest.from_file(path)
        assert path.read_bytes() == before
        assert read.files == m.files
        assert read.configuration == m.configuration
        assert read.ulid_code == m.ulid_code


class TestFiles:
    """Adding files to manifests"""

    def test_constructor_with_file_list(self, test_txt, test_jpss1_cr_1):
        """Test constructing a manifest from a list of paths"""
        m = Manifest(
            manifest_type=ManifestType.INPUT,
            files=[test_txt, test_jpss1_cr_1],
        )
        m.validate_checksums()
        assert len(m.files) == 2

    def test_add_relative_path_file_error(self):
        """Files must be recorded with absolute paths"""
        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        with pytest.raises(ValueError, match="must be an absolute path"):
            m.add_files(Path("relative/a_file.txt"))

    def test_add_files_local(self, test_jpss_manifest, test_txt, test_jpss1_cr_1):
        """Test adding files to a manifest with checksum and local paths"""
        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        initial_list_len = len(m.files)
        m.add_files(test_jpss_manifest)
        m.validate_checksums()
        assert len(m.files) == initial_list_len + 1

        more_files = (test_txt, test_jpss1_cr_1)
        m.add_files(*more_files)
        m.validate_checksums()
        assert len(m.files) == initial_list_len + 3

    def test_add_files_s3(self, test_jpss_manifest, test_txt, test_jpss1_cr_1, create_mock_bucket, write_file_to_s3):
        """Test adding files to a manifest with checksum with S3 paths.
        Ensures functionality for single and multiple file additions."""
        bucket = create_mock_bucket()
        manifest_path = f"s3://{bucket.name}/test_file1.json"
        text_paths = (f"s3://{bucket.name}/test_file2.txt", f"s3://{bucket.name}/test_construction_record.PDS")
        write_file_to_s3(test_jpss_manifest, manifest_path)
        write_file_to_s3(test_txt, text_paths[0])
        write_file_to_s3(test_jpss1_cr_1, text_paths[1])

        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        initial_list_len = len(m.files)
        m.add_files(manifest_path)
        m.validate_checksums()
        assert len(m.files) == initial_list_len + 1

        m.add_files(*text_paths)
        m.validate_checksums()
        assert len(m.files) == initial_list_len + 3

    def test_add_duplicate_file(self, caplog, test_jpss_manifest):
        """Test adding a duplicate file to a manifest"""
        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        m.add_files(test_jpss_manifest)
        initial_length = len(m.files)

        # Add the same file
        with caplog.at_level("WARNING"):
            m.add_files(test_jpss_manifest)
        m.validate_checksums()
        assert len(m.files) == initial_length

    def test_add_missing_files(self, tmp_path, caplog):
        """Missing files are added with an empty checksum and a warning, and are not treated as duplicates"""
        missing = [tmp_path / "missing1.nc", tmp_path / "missing2.nc"]
        m = Manifest(manifest_type=ManifestType.INPUT)
        with caplog.at_level("WARNING"):
            m.add_files(*missing)
        assert [f.checksum for f in m.files] == [None, None]
        assert f"File {missing[0]} cannot be found; its checksum will be empty" in caplog.text

    @pytest.mark.parametrize("as_dict", [True, False])
    def test_constructor_with_missing_file(self, tmp_path, caplog, as_dict):
        """A missing file passed to the constructor, as a path or as a record without a checksum, is kept"""
        missing = tmp_path / "missing.nc"
        with caplog.at_level("WARNING"):
            m = Manifest(manifest_type=ManifestType.INPUT, files=[{"filename": str(missing)} if as_dict else missing])
        assert m.files[0].checksum is None
        assert "cannot be found" in caplog.text


class TestValidateChecksums:
    """Validating the checksums of the files in a manifest"""

    def test_validate_checksums(self, test_jpss_manifest, caplog):
        """Test the method that validates checksums in a manifest file"""
        # We test by referencing the manifest file itself, so we're only dependent on one test file
        m = Manifest.from_file(test_jpss_manifest)
        m.files[0].filename = test_jpss_manifest.absolute()
        m.files[1].filename = test_jpss_manifest.absolute()
        with caplog.at_level("ERROR"):
            with pytest.raises(ManifestError, match="Files failed checksum validation"):  # Fake values don't validate
                m.validate_checksums()
            assert f"Checksum validation for {test_jpss_manifest.absolute()} failed." in caplog.records[0].message
            assert len(caplog.records) == 2

        with test_jpss_manifest.open("rb") as fh:
            checksum = md5(fh.read()).hexdigest()
        m.files = [ManifestFileRecord(filename=str(test_jpss_manifest.absolute()), checksum=checksum)]
        m.validate_checksums()

    def test_missing_checksum_fails(self, test_txt):
        """A file with no recorded checksum fails validation"""
        m = Manifest(manifest_type=ManifestType.INPUT, files=[ManifestFileRecord(filename=str(test_txt))])
        with pytest.raises(ManifestError, match=str(test_txt)):
            m.validate_checksums()

    def test_missing_file_fails(self, tmp_path):
        """A file that cannot be found fails validation rather than raising FileNotFoundError"""
        missing = str(tmp_path / "missing.nc")
        m = Manifest(manifest_type=ManifestType.INPUT, files=[ManifestFileRecord(filename=missing, checksum="abc")])
        with pytest.raises(ManifestError, match=missing):
            m.validate_checksums()


class TestWrite:
    """Writing manifest files"""

    def test_write_to_directory(self, tmp_path):
        """Writing to a directory generates a filename, records it on the object and in the file"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        written = m.write(tmp_path)
        assert written.parent == tmp_path
        assert m.filename.path == written
        on_disk = json.loads(written.read_text())
        assert on_disk["filename"] == str(written)
        assert on_disk["ulid_code"] == str(m.ulid_code)
        for element in ("manifest_type", "files", "configuration"):
            assert element in on_disk

    def test_write_to_s3_directory(self, create_mock_bucket):
        """Test writing a manifest file to an S3 prefix"""
        bucket = create_mock_bucket()
        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        outpath = S3Path(f"s3://{bucket.name}")
        m.write(outpath / INPUT_NAME)
        with smart_open(outpath / INPUT_NAME) as f:
            manifest_dict = json.load(f)
            for element in ("manifest_type", "files", "configuration"):
                assert element in manifest_dict

    @pytest.mark.parametrize("out_path", ["{tmp}/" + INPUT_NAME, "path", "manifest_filename"])
    def test_write_to_full_path(self, tmp_path, out_path):
        """out_path may be the full manifest file path, as a str, Path or ManifestFilename"""
        target = tmp_path / INPUT_NAME
        out_path = {"path": target, "manifest_filename": ManifestFilename(target)}.get(
            out_path, out_path.format(tmp=tmp_path)
        )
        m = Manifest(manifest_type=ManifestType.INPUT)
        assert m.write(out_path) == target
        assert m.filename.path == target

    def test_write_overwrite_is_keyword_only(self, tmp_path):
        """A second positional argument is rejected rather than taken as overwrite"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        with pytest.raises(TypeError):
            m.write(tmp_path, INPUT_NAME)
        assert list(tmp_path.iterdir()) == []

    def test_write_refuses_mismatched_label(self, tmp_path):
        """A full path named for the other manifest type raises ManifestError and writes nothing"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        with pytest.raises(ManifestError, match="named as a OUTPUT manifest"):
            m.write(tmp_path / OUTPUT_NAME)
        assert list(tmp_path.iterdir()) == []
        assert m.filename is None

    def test_write_refuses_to_change_ulid(self, tmp_path):
        """A write cannot change the ULID of a manifest"""
        m = Manifest(manifest_type=ManifestType.OUTPUT, filename=OUTPUT_NAME)
        with pytest.raises(ManifestError, match="never need to change the ULID of an output manifest"):
            m.write(tmp_path / f"LIBERA_OUTPUT_MANIFEST_{OTHER_ULID}.json")
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("filename", [None, INPUT_NAME])
    def test_write_without_out_path_uses_cwd(self, tmp_path, monkeypatch, filename):
        """With no out_path and no directory in filename, the manifest is written to the working directory"""
        monkeypatch.chdir(tmp_path)
        m = Manifest(manifest_type=ManifestType.INPUT, filename=filename)
        written = m.write()
        assert written.resolve().parent == tmp_path.resolve()
        if filename:
            assert written.name == filename

    def test_write_without_out_path_uses_filename_directory(self, tmp_path):
        """With no out_path, a filename that includes a directory is written to that path"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=tmp_path / INPUT_NAME)
        assert m.write() == tmp_path / INPUT_NAME

    def test_write_relocation_warning(self, tmp_path, caplog):
        """Writing a written manifest somewhere else warns that its filename was updated"""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        m = Manifest(manifest_type=ManifestType.INPUT)
        with caplog.at_level("WARNING"):
            first = m.write(tmp_path / "a")
            assert "Internal manifest filename was updated" not in caplog.text
            second = m.write(tmp_path / "b")
        assert f"Internal manifest filename was updated by the latest write from {first} to {second}" in caplog.text
        assert m.filename.path == second

    def test_write_does_not_overwrite_by_default(self, tmp_path):
        """An existing target raises FileExistsError and leaves filename unchanged"""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / INPUT_NAME).write_text("existing")
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        first = m.write(tmp_path / "a")
        with pytest.raises(FileExistsError):
            m.write(tmp_path / "b")
        assert m.filename.path == first
        assert (tmp_path / "b" / INPUT_NAME).read_text() == "existing"

    def test_write_overwrite(self, tmp_path):
        """overwrite=True replaces an existing file"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        written = m.write(tmp_path)
        m.configuration["key"] = "value"
        assert m.write(overwrite=True) == written
        assert json.loads(written.read_text())["configuration"] == {"key": "value"}

    def test_cleared_filename_gets_new_ulid(self, tmp_path):
        """Clearing the filename before writing produces a new ULID"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        m.write(tmp_path)
        old_ulid = m.ulid_code
        m.filename = None
        m.write(tmp_path)
        assert m.ulid_code != old_ulid
        assert len(list(tmp_path.iterdir())) == 2


class TestOutputManifestFromInputManifest:
    """Creating an output manifest from an input manifest"""

    @pytest.mark.parametrize(
        "input_manifest",
        [
            (S3Path(f"s3://test-manifest-from-file-s3-bucket/{INPUT_NAME}")),
            TEST_DATA_MANIFEST,
            str(TEST_DATA_MANIFEST),
            Manifest.from_file(filepath=TEST_DATA_MANIFEST),
        ],
    )
    def test_output_manifest_from_input_manifest(self, input_manifest, test_jpss_manifest, write_file_to_s3):
        """Test creating an output manifest from an input manifest path or object"""
        if isinstance(input_manifest, S3Path):
            write_file_to_s3(test_jpss_manifest, str(input_manifest))
        input_manifest_object = (
            input_manifest if isinstance(input_manifest, Manifest) else Manifest.from_file(input_manifest)
        )

        output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest=input_manifest)

        assert output_manifest.manifest_type == ManifestType.OUTPUT
        assert output_manifest.ulid_code == input_manifest_object.ulid_code
        assert str(output_manifest.filename) == OUTPUT_NAME
        assert output_manifest.configuration["input_manifest_files"] == input_manifest_object.files

    def test_configuration_is_merged(self, caplog):
        """Caller configuration is kept, but input_manifest_files always comes from the input manifest"""
        input_manifest = Manifest.from_file(TEST_DATA_MANIFEST)
        with caplog.at_level("WARNING"):
            output_manifest = Manifest.output_manifest_from_input_manifest(
                input_manifest, configuration={"key": "value", "input_manifest_files": ["clobbered"]}
            )
        assert output_manifest.configuration["key"] == "value"
        assert output_manifest.configuration["input_manifest_files"] == input_manifest.files
        assert "contains input_manifest_files" in caplog.text

    def test_input_without_ulid(self):
        """An input manifest with no ULID cannot produce an output manifest"""
        with pytest.raises(ManifestError, match="has no ULID"):
            Manifest.output_manifest_from_input_manifest(Manifest(manifest_type=ManifestType.INPUT))

    def test_ulid_preserved_through_files(self, tmp_path, test_txt):
        """An input manifest written to disk and read back yields an output manifest with its ULID"""
        (tmp_path / "in").mkdir()
        (tmp_path / "out").mkdir()
        input_path = Manifest(manifest_type=ManifestType.INPUT, files=[test_txt]).write(tmp_path / "in")

        output_manifest = Manifest.output_manifest_from_input_manifest(input_path)
        output_path = output_manifest.write(tmp_path / "out")

        assert Manifest.from_file(output_path).ulid_code == Manifest.from_file(input_path).ulid_code
