"""Tests for manifest module"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from cloudpathlib import S3Path
from pydantic import ValidationError
from ulid import ULID

from libera_utils.constants import ManifestType
from libera_utils.io.filenaming import ManifestFilename
from libera_utils.io.manifest import Manifest, ManifestError

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
                ["relative/file.txt"],
                ManifestType.OUTPUT,
                None,
            ),
            (None, [{"filename": "s3://abs/file.txt", "checksum": "fakesum"}], ManifestType.OUTPUT, None),
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
                ["s3://abs/file.txt", Path("/abs/file.nc"), S3Path("s3://abs/other.nc")],
                ManifestType.OUTPUT,
                {"data": "description"},
            ),
            (f"subfolder/{INPUT_NAME}", None, ManifestType.INPUT, {"data": "description"}),
            (None, None, ManifestType.OUTPUT, {"data": "description"}),
        ],
    )
    def test_validation_success(self, man_path, man_files, man_type, man_config):
        """Test manifest validation method for correct success cases"""
        m = Manifest(manifest_type=man_type, files=man_files, configuration=man_config, filename=man_path)
        assert m.files == [str(f) for f in man_files or []]

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

    def test_equality_with_and_without_filename(self):
        """A manifest with a filename and one without compare unequal in either order without raising"""
        named = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        unnamed = Manifest(manifest_type=ManifestType.INPUT)
        assert named != unnamed
        assert unnamed != named

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
        assert m.files == json.loads(test_jpss_manifest.read_text())["files"]
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
        with pytest.raises(
            ManifestError, match=r"does not have a valid manifest filename.*LIBERA_INPUT_MANIFEST_<ULID>\.json"
        ):
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

    def test_file_record_is_invalid(self, tmp_path, test_txt):
        """A file listed as a record rather than a path string is a validation error"""
        path = write_manifest_json(tmp_path / INPUT_NAME, files=[{"filename": str(test_txt), "checksum": "fakesum"}])
        with pytest.raises(ValidationError, match="lists each file as an absolute local or S3 path string"):
            Manifest.from_file(path)

    def test_missing_files_are_read(self, tmp_path):
        """Files are not checked for existence when a manifest is read"""
        missing = str(tmp_path / "missing.nc")
        path = write_manifest_json(tmp_path / INPUT_NAME, files=[missing])
        assert Manifest.from_file(path).files == [missing]

    def test_repeated_file_is_invalid(self, tmp_path):
        """A manifest file listing the same path twice is a validation error"""
        path = write_manifest_json(tmp_path / INPUT_NAME, files=["s3://bucket/a.nc", "s3://bucket/a.nc"])
        with pytest.raises(ValidationError, match="already listed in the manifest"):
            Manifest.from_file(path)

    def test_identical_files_are_all_read(self, tmp_path):
        """Files with identical content and different names are all read, in order"""
        identical = [tmp_path / "a.nc", tmp_path / "b.nc"]
        for path in identical:
            path.write_bytes(b"identical")
        path = write_manifest_json(tmp_path / INPUT_NAME, files=[str(p) for p in identical])
        m = Manifest.from_file(path)
        assert m.files == [str(p) for p in identical]

    def test_round_trip(self, tmp_path, test_txt):
        """A written manifest reads back with the same contents and ULID, and reading leaves the file unchanged"""
        m = Manifest(manifest_type=ManifestType.INPUT, files=[test_txt], configuration={"key": "value"})
        path = m.write(tmp_path)
        before = path.read_bytes()
        read = Manifest.from_file(path)
        assert path.read_bytes() == before
        assert json.loads(before)["files"] == [str(test_txt)]
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
        assert m.files == [str(test_txt), str(test_jpss1_cr_1)]

    def test_add_relative_path_file_error(self):
        """Files must be recorded with absolute paths"""
        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        with pytest.raises(ValueError, match="must be an absolute path"):
            m.add_files(Path("relative/a_file.txt"))

    def test_add_files_local(self, test_jpss_manifest, test_txt, test_jpss1_cr_1):
        """Test adding files to a manifest with local paths"""
        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        initial_list_len = len(m.files)
        m.add_files(test_jpss_manifest)
        assert len(m.files) == initial_list_len + 1

        more_files = (test_txt, test_jpss1_cr_1)
        m.add_files(*more_files)
        assert m.files == [str(test_jpss_manifest), str(test_txt), str(test_jpss1_cr_1)]

    def test_add_files_s3(self, test_jpss_manifest, test_txt, test_jpss1_cr_1, create_mock_bucket, write_file_to_s3):
        """S3 files are added singly and several at a time without being downloaded"""
        bucket = create_mock_bucket()
        manifest_path = f"s3://{bucket.name}/test_file1.json"
        text_paths = (f"s3://{bucket.name}/test_file2.txt", f"s3://{bucket.name}/test_construction_record.PDS")
        write_file_to_s3(test_jpss_manifest, manifest_path)
        write_file_to_s3(test_txt, text_paths[0])
        write_file_to_s3(test_jpss1_cr_1, text_paths[1])

        m = Manifest(
            manifest_type=ManifestType.INPUT,
        )
        with patch.object(
            S3Path(manifest_path).client, "_download_file", side_effect=AssertionError("object was downloaded")
        ):
            m.add_files(manifest_path)
            assert m.files == [manifest_path]

            m.add_files(*text_paths)
        assert m.files == [manifest_path, *text_paths]

    def test_add_listed_file_raises(self, test_jpss_manifest, test_txt):
        """Adding a file that is already listed raises and leaves the file list unchanged"""
        m = Manifest(manifest_type=ManifestType.INPUT, files=[test_jpss_manifest])
        with pytest.raises(ValueError, match="already listed in the manifest"):
            m.add_files(test_txt, test_jpss_manifest)
        assert m.files == [str(test_jpss_manifest)]

    def test_add_same_file_twice_in_one_call_raises(self, test_txt):
        """A file given twice in one call raises and nothing is added"""
        m = Manifest(manifest_type=ManifestType.INPUT)
        with pytest.raises(ValueError, match="already listed in the manifest"):
            m.add_files(test_txt, test_txt)
        assert m.files == []

    def test_add_listed_s3_file_sends_no_existence_request(self):
        """A repeated path raises before any file check, so S3 is not queried"""
        listed = "s3://some-bucket/listed.nc"
        m = Manifest(manifest_type=ManifestType.INPUT, files=[listed])
        with patch.object(S3Path(listed).client, "_s3_file_query", side_effect=AssertionError("existence was checked")):
            with pytest.raises(ValueError, match="already listed in the manifest"):
                m.add_files(listed)

    def test_add_mixed_absolute_and_relative_raises(self, tmp_path, test_txt):
        """A batch with an absolute path and a relative path raises for the relative one and adds neither.

        Every path is checked before any file's existence, so the relative path is reported even when a missing
        file precedes it.
        """
        m = Manifest(manifest_type=ManifestType.INPUT)
        with pytest.raises(ValueError, match="must be an absolute path"):
            m.add_files(test_txt, Path("relative/a_file.txt"))
        with pytest.raises(ValueError, match="must be an absolute path"):
            m.add_files(tmp_path / "missing.nc", Path("relative/a_file.txt"))
        assert m.files == []

    def test_constructor_with_repeated_file_raises(self, test_txt):
        """A path listed twice in the constructor is a validation error"""
        with pytest.raises(ValidationError, match="already listed in the manifest"):
            Manifest(manifest_type=ManifestType.INPUT, files=[test_txt, test_txt])

    def test_add_identical_files_to_output_manifest(self, tmp_path):
        """An algorithm's two outputs with identical content and different names are both listed and written"""
        identical = [tmp_path / "a.bsp", tmp_path / "b.bsp"]
        for path in identical:
            path.write_bytes(b"identical")
        m = Manifest.output_manifest_from_input_manifest(
            Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        )
        m.add_files(*identical)
        assert m.files == [str(p) for p in identical]

        written = m.write(tmp_path)
        assert json.loads(written.read_text())["files"] == [str(p) for p in identical]

    def test_constructor_with_identical_files(self, tmp_path):
        """Files with identical content and different names passed to the constructor are all listed"""
        identical = [tmp_path / "a.nc", tmp_path / "b.nc"]
        for path in identical:
            path.write_bytes(b"identical")
        m = Manifest(manifest_type=ManifestType.INPUT, files=identical)
        assert m.files == [str(p) for p in identical]

    @pytest.mark.parametrize("on_s3", [False, True])
    def test_add_missing_file_raises(self, tmp_path, test_txt, create_mock_bucket, on_s3):
        """Adding a file that cannot be found raises FileNotFoundError and leaves the file list unchanged"""
        missing = f"s3://{create_mock_bucket().name}/missing.nc" if on_s3 else str(tmp_path / "missing.nc")
        m = Manifest(manifest_type=ManifestType.INPUT, files=[test_txt])
        before = list(m.files)
        with pytest.raises(FileNotFoundError, match=f"File {missing} cannot be found"):
            m.add_files(missing)
        assert m.files == before
        # A valid file passed in the same call is not added either
        present = tmp_path / "present.nc"
        present.write_text("present")
        with pytest.raises(FileNotFoundError):
            m.add_files(present, missing)
        assert m.files == before

    @pytest.mark.parametrize("location", ["local", "s3_prefix", "s3_bucket"])
    def test_add_directory_raises(self, tmp_path, test_txt, create_mock_bucket, write_file_to_s3, location):
        """Adding a directory, S3 prefix or S3 bucket root raises IsADirectoryError and leaves the files unchanged"""
        if location == "local":
            directory = str(tmp_path)
        else:
            bucket = create_mock_bucket().name
            write_file_to_s3(test_txt, f"s3://{bucket}/prefix/a.txt")
            directory = f"s3://{bucket}/prefix" if location == "s3_prefix" else f"s3://{bucket}"
        m = Manifest(manifest_type=ManifestType.INPUT, files=[test_txt])
        before = list(m.files)
        with pytest.raises(IsADirectoryError, match="not a file"):
            m.add_files(directory)
        assert m.files == before
        # A valid file passed in the same call is not added either
        present = tmp_path / "present.nc"
        present.write_text("present")
        with pytest.raises(IsADirectoryError):
            m.add_files(present, directory)
        assert m.files == before

    def test_constructor_with_missing_file(self, tmp_path):
        """The constructor lists a path whose file does not exist; only add_files checks existence"""
        missing = str(tmp_path / "missing.nc")
        assert Manifest(manifest_type=ManifestType.INPUT, files=[missing]).files == [missing]


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
        written = m.write(outpath)
        assert written.parent == outpath
        assert ManifestFilename(written).filename_parts.manifest_type == ManifestType.INPUT
        read = Manifest.from_file(written)
        assert read.manifest_type == ManifestType.INPUT
        assert read.ulid_code == m.ulid_code

    def test_write_to_dotted_bucket_root(self, create_mock_bucket):
        """A bucket name containing a dot is a prefix to write into, not a file name with an extension"""
        bucket = create_mock_bucket("test.dotted-bucket")
        written = Manifest(manifest_type=ManifestType.INPUT).write(f"s3://{bucket.name}")
        assert written.parent == S3Path(f"s3://{bucket.name}")

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

    def test_write_requires_out_path(self):
        """out_path is required; a manifest is never written to the working directory by default"""
        m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
        with pytest.raises(TypeError):
            m.write()

    @pytest.mark.parametrize("on_s3", [False, True])
    def test_write_refuses_invalid_name_with_extension(self, tmp_path, create_mock_bucket, on_s3):
        """An out_path with an extension that is not a valid manifest filename raises and writes nothing"""
        root = S3Path(f"s3://{create_mock_bucket().name}") if on_s3 else tmp_path / "dropbox"
        if not on_s3:
            root.mkdir()
        m = Manifest(manifest_type=ManifestType.INPUT)
        with pytest.raises(ManifestError, match=r"not a valid manifest filename.*LIBERA_INPUT_MANIFEST_<ULID>\.json"):
            m.write(root / "out.json")
        assert list(root.iterdir()) == []
        assert m.filename is None

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
        assert m.write(written, overwrite=True) == written
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

    def test_input_files_are_copied(self, test_txt, test_jpss_manifest):
        """Adding files to the input manifest afterwards leaves the output manifest's record of them unchanged"""
        input_manifest = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME, files=[test_txt])
        output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)
        input_manifest.add_files(test_jpss_manifest)
        assert output_manifest.configuration["input_manifest_files"] == [str(test_txt)]

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
