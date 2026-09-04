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


def test_manifest_from_file(test_jpss_manifest):
    """Test factory method for creating a manifest object from a filepath"""
    m = Manifest.from_file(test_jpss_manifest)
    assert m.manifest_type == ManifestType.INPUT
    assert isinstance(m.files, list)
    assert isinstance(m.configuration, dict)


def test_manifest_constructor_with_file_list(test_txt, test_jpss1_cr_1):
    m = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[test_txt, test_jpss1_cr_1],
    )
    m.validate_checksums()
    assert len(m.files) == 2


def test_manifest_add_relative_path_file_error():
    m = Manifest(
        manifest_type=ManifestType.INPUT,
    )
    with pytest.raises(ValidationError):
        m.add_files(Path("relative/a_file.txt"))


def test_manifest_add_files_to_manifest_local(test_jpss_manifest, test_txt, test_jpss1_cr_1):
    """Test factory method for adding files to a manifest with checksum and local paths"""
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


def test_manifest_add_files_to_manifest_s3(
    test_jpss_manifest, test_txt, test_jpss1_cr_1, create_mock_bucket, write_file_to_s3
):
    """Test factory method for adding files to a manifest with checksum with S3 paths.
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


def test_manifest_add_duplicate_file_to_manifest(caplog, test_jpss_manifest):
    """Test factory method for adding a duplicate file to a manifest"""
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


def test_manifest_add_desired_time_range(test_jpss_manifest):
    """Test factory method for adding a time range to a manifest file"""
    m = Manifest.from_file(test_jpss_manifest)
    start = datetime.now()
    end = start + timedelta(hours=1)
    m.add_desired_time_range(start, end)

    assert "start_time" in m.configuration.keys()
    assert "end_time" in m.configuration.keys()


def test_manifest_from_file_s3(test_jpss_manifest, write_file_to_s3):
    """Test loading a file from S3"""
    file_key = "s3://test-manifest-from-file-s3-bucket/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"
    s3_path = write_file_to_s3(test_jpss_manifest, file_key)
    m = Manifest.from_file(s3_path)
    assert m.manifest_type == ManifestType.INPUT
    assert isinstance(m.files, list)
    assert isinstance(m.configuration, dict)


def test_manifest_write(tmp_path):
    """Test writing a manifest file from an object"""
    m = Manifest(
        manifest_type=ManifestType.INPUT,
    )
    m.write(tmp_path, "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json")
    with open(tmp_path / "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json") as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


def test_manifest_generate_filename():
    """Test generating a filename for a manifest file"""
    m = Manifest(manifest_type=ManifestType.INPUT)
    assert m._generate_filename().filename_parts.ulid_code is not None
    m.manifest_type = ManifestType.OUTPUT
    assert m._generate_filename().filename_parts.ulid_code is not None
    assert m.files == []
    assert m.configuration == {}


def test_manifest_write_s3(create_mock_bucket):
    """Test writing a manifest file from an object"""
    bucket = create_mock_bucket()
    m = Manifest(
        manifest_type=ManifestType.INPUT,
    )
    outpath = S3Path(f"s3://{bucket.name}")
    filename = "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"
    m.write(outpath, filename)
    with smart_open(outpath / filename) as f:
        manifest_dict = json.load(f)
        for element in ("manifest_type", "files", "configuration"):
            assert element in manifest_dict


def test_validate_checksums(test_jpss_manifest, caplog):
    """Test the method that validates checksums in a manifest file"""
    # We test by referencing the manifest file itself, so we're only dependent on one test file
    m = Manifest.from_file(test_jpss_manifest)
    m.files[0].filename = test_jpss_manifest.absolute()
    m.files[1].filename = test_jpss_manifest.absolute()
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError, match="Files failed checksum validation"):  # Fake values don't validate
            m.validate_checksums()
        assert f"Checksum validation for {test_jpss_manifest.absolute()} failed." in caplog.records[0].message

    with test_jpss_manifest.open("rb") as fh:
        checksum = md5(fh.read()).hexdigest()
    m.files = [ManifestFileRecord(filename=str(test_jpss_manifest.absolute()), checksum=checksum)]
    m.validate_checksums()


@pytest.mark.parametrize(
    "input_manifest",
    [
        (S3Path("s3://test-manifest-from-file-s3-bucket/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json")),
        (
            Path(sys.modules[__name__.split(".")[0]].__file__).parent
            / "test_data"
            / "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"
        ),
        (
            Manifest.from_file(
                filepath=Path(sys.modules[__name__.split(".")[0]].__file__).parent
                / "test_data"
                / "LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json"
            )
        ),
        (S3Path("s3://l0-ingest-dropbox/processing//LIBERA_OUTPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json")),
    ],
)
def test_output_manifest_from_input_manifest(input_manifest, test_jpss_manifest, write_file_to_s3):
    """Test method that creates output manifest from input manifest filename or object"""
    if isinstance(input_manifest, S3Path):
        s3_path = write_file_to_s3(test_jpss_manifest, str(input_manifest))
        input_manifest_object = Manifest.from_file(filepath=s3_path)

    elif isinstance(input_manifest, Path):
        input_manifest_object = Manifest.from_file(filepath=input_manifest)

    elif isinstance(input_manifest, Manifest):
        input_manifest_object = input_manifest

    else:
        raise NotImplementedError(f"Unexpected type for input_manifest: {type(input_manifest)}")

    output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest=input_manifest_object)
    input_time = input_manifest_object.ulid_code.datetime
    output_time = output_manifest.ulid_code.datetime

    assert input_manifest_object.manifest_type == ManifestType.INPUT
    assert output_manifest.manifest_type == ManifestType.OUTPUT
    assert input_time == output_time
    assert len(output_manifest.configuration) != 0


@pytest.mark.parametrize(
    ("man_path", "man_files", "man_type", "man_config"),
    [
        (
            "subfolder/LIBERA_INPUT_MANIFEST_201GDHWG4R0W8KXWY0KRDD6BZTT.json",
            [{"filename": "relative/file.txt", "checksum": "fakesum"}],
            ManifestType.OUTPUT,
            None,
        ),
        ("subfolder/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json", None, ManifestType.OUTPUT, ["config"]),
    ],
)
def test_manifest_validation_failure(man_path, man_files, man_type, man_config):
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
        (
            "subfolder/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTT.json",
            None,
            ManifestType.OUTPUT,
            {"data": "description"},
        ),
        (None, None, ManifestType.OUTPUT, {"data": "description"}),
    ],
)
def test_manifest_validation_success(man_path, man_files, man_type, man_config):
    """Test manifest validation method for correct success cases"""
    _ = Manifest(manifest_type=man_type, files=man_files, configuration=man_config, filename=man_path)


# ----------------------------------------------------------------------------------------------------------------
# LIBSDC-653: ULID handling, write path handling, lenient read / strict write, file-state tracking, factories
# ----------------------------------------------------------------------------------------------------------------
VALID_ULID = "01GDHWG4R0W8KXWY0KRDD6BZTT"
INPUT_NAME = f"LIBERA_INPUT_MANIFEST_{VALID_ULID}.json"
OUTPUT_NAME = f"LIBERA_OUTPUT_MANIFEST_{VALID_ULID}.json"


def test_ulid_code_derived_from_filename():
    """ulid_code comes from the filename and is None without one"""
    m = Manifest(manifest_type=ManifestType.INPUT, filename=f"/some/dir/{INPUT_NAME}")
    assert m.ulid_code == ULID.from_str(VALID_ULID)
    assert Manifest(manifest_type=ManifestType.INPUT).ulid_code is None
    with pytest.raises(AttributeError):
        m.ulid_code = ULID()  # read-only, derived from the filename


def test_ulid_code_input_without_filename_produces_bare_filename():
    """Passing ulid_code with no filename yields a bare filename carrying that ULID"""
    code = ULID()
    m = Manifest(manifest_type=ManifestType.OUTPUT, ulid_code=code)
    assert m.ulid_code == code
    assert str(m.filename) == f"LIBERA_OUTPUT_MANIFEST_{code}.json"
    # String ULIDs are accepted too (e.g. from JSON)
    m2 = Manifest(manifest_type="INPUT", ulid_code=str(code))
    assert m2.ulid_code == code


def test_ulid_code_input_must_agree_with_filename():
    """An explicit ulid_code is accepted when it matches the filename and rejected when it doesn't"""
    m = Manifest(manifest_type=ManifestType.INPUT, ulid_code=VALID_ULID, filename=INPUT_NAME)
    assert m.ulid_code == ULID.from_str(VALID_ULID)
    with pytest.raises(ValidationError, match="disagrees with the ULID"):
        Manifest(manifest_type=ManifestType.INPUT, ulid_code=ULID(), filename=INPUT_NAME)


def test_ulid_code_serialized_and_filename_none_serializes_as_null():
    """ulid_code appears in the dump; a missing filename is null, not the string 'None'"""
    dumped = json.loads(Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME).model_dump_json())
    assert dumped["ulid_code"] == VALID_ULID
    assert dumped["filename"] == INPUT_NAME
    dumped = json.loads(Manifest(manifest_type=ManifestType.INPUT).model_dump_json())
    assert dumped["ulid_code"] is None
    assert dumped["filename"] is None


def test_manifest_type_mismatch_warns_on_construction(caplog):
    """A filename labelled INPUT on an OUTPUT manifest is tolerated in memory with a warning"""
    with caplog.at_level("WARNING"):
        m = Manifest(manifest_type=ManifestType.OUTPUT, filename=INPUT_NAME)
    assert m.manifest_type == ManifestType.OUTPUT
    assert any("named as a INPUT manifest" in r.message for r in caplog.records)


def test_write_multiple_times_no_side_effects(tmp_path):
    """The same manifest can be written to several directories; filename is untouched"""
    m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1 = m.write(tmp_path / "a")
    p2 = m.write(tmp_path / "b")
    assert p1 == tmp_path / "a" / INPUT_NAME
    assert p2 == tmp_path / "b" / INPUT_NAME
    assert str(m.filename) == INPUT_NAME  # not mutated
    for p in (p1, p2):
        on_disk = json.loads(p.read_text())
        assert on_disk["filename"] == str(p)  # on-disk filename reflects where it was written
        assert on_disk["ulid_code"] == VALID_ULID


def test_write_multiple_times_s3(create_mock_bucket):
    """Writing twice to S3 from one manifest works (previously raised TypeError on the second call)"""
    bucket = create_mock_bucket()
    m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
    p1 = m.write(S3Path(f"s3://{bucket.name}/one"))
    p2 = m.write(f"s3://{bucket.name}/two")
    assert p1 == S3Path(f"s3://{bucket.name}/one/{INPUT_NAME}")
    assert p2 == S3Path(f"s3://{bucket.name}/two/{INPUT_NAME}")
    with smart_open(p2) as f:
        assert json.load(f)["filename"] == str(p2)


def test_write_uses_basename_of_preset_full_path(tmp_path):
    """A manifest whose filename is a full path elsewhere is written into out_path, not back to its origin"""
    m = Manifest(manifest_type=ManifestType.INPUT, filename=f"s3://some-bucket/prefix/{INPUT_NAME}")
    written = m.write(tmp_path)
    assert written == tmp_path / INPUT_NAME
    assert written.exists()


def test_write_generates_filename_with_consistent_ulid(tmp_path):
    """With no filename anywhere, a fresh ULID is generated and the on-disk ulid_code matches it"""
    m = Manifest(manifest_type=ManifestType.OUTPUT)
    written = m.write(tmp_path)
    parts = ManifestFilename(written).filename_parts
    assert parts.manifest_type == ManifestType.OUTPUT
    on_disk = json.loads(written.read_text())
    assert on_disk["ulid_code"] == str(parts.ulid_code)
    assert m.filename is None  # still not mutated


def test_write_accepts_full_file_path(tmp_path):
    """out_path may be the full path of the manifest file"""
    m = Manifest(manifest_type=ManifestType.INPUT)
    written = m.write(tmp_path / INPUT_NAME)
    assert written == tmp_path / INPUT_NAME
    assert written.exists()
    # ... and a matching bare filename alongside it is fine
    (tmp_path / "sub").mkdir()
    written2 = m.write(tmp_path / "sub" / INPUT_NAME, filename=INPUT_NAME)
    assert written2 == tmp_path / "sub" / INPUT_NAME


def test_write_accepts_manifest_filename_objects(tmp_path):
    """ManifestFilename instances work for both out_path and filename"""
    m = Manifest(manifest_type=ManifestType.INPUT)
    written = m.write(ManifestFilename(tmp_path / INPUT_NAME))
    assert written == tmp_path / INPUT_NAME
    (tmp_path / "sub").mkdir()
    written2 = m.write(tmp_path / "sub", filename=ManifestFilename(INPUT_NAME))
    assert written2 == tmp_path / "sub" / INPUT_NAME


@pytest.mark.parametrize(
    ("manifest_type", "out_path_suffix", "filename", "match"),
    [
        (ManifestType.INPUT, "", "not_a_manifest.json", "not a valid manifest filename"),
        (ManifestType.INPUT, "", OUTPUT_NAME, "named as a OUTPUT manifest"),
        (ManifestType.OUTPUT, INPUT_NAME, None, "named as a INPUT manifest"),
        (ManifestType.INPUT, "", f"sub/{INPUT_NAME}", "no directory part"),
        (ManifestType.INPUT, INPUT_NAME, OUTPUT_NAME, "Pass one or the other"),
    ],
)
def test_write_rejects_invalid_targets(tmp_path, manifest_type, out_path_suffix, filename, match):
    """write() refuses invalid or conflicting filenames with a ManifestError and writes nothing"""
    m = Manifest(manifest_type=manifest_type)
    with pytest.raises(ManifestError, match=match):
        m.write(tmp_path / out_path_suffix if out_path_suffix else tmp_path, filename=filename)
    assert list(tmp_path.iterdir()) == []


def test_write_preset_filename_type_mismatch_rejected(tmp_path):
    """A preset filename whose type disagrees with manifest_type is tolerated in memory but not on write"""
    m = Manifest(manifest_type=ManifestType.OUTPUT, filename=INPUT_NAME)
    with pytest.raises(ManifestError, match="named as a INPUT manifest"):
        m.write(tmp_path)


def test_write_does_not_overwrite(tmp_path):
    """Writing to an existing path still fails (exclusive create); save() is the way to overwrite"""
    m = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME)
    m.write(tmp_path)
    with pytest.raises(FileExistsError):
        m.write(tmp_path)


def test_from_file_is_file_backed_and_ignores_stored_filename(tmp_path, caplog):
    """from_file tracks its source and takes filename/ULID from the path, not the JSON contents"""
    path = tmp_path / INPUT_NAME
    path.write_text(
        json.dumps(
            {
                "manifest_type": "INPUT",
                "files": [],
                "configuration": {},
                "filename": "/stale/location/LIBERA_INPUT_MANIFEST_01GDHWG4R0W8KXWY0KRDD6BZTU.json",
                "ulid_code": "01GDHWG4R0W8KXWY0KRDD6BZTU",
            }
        )
    )
    with caplog.at_level("WARNING"):
        m = Manifest.from_file(path)
    assert m.is_file_backed
    assert m.source_path == path
    assert m.filename.path == path
    assert m.ulid_code == ULID.from_str(VALID_ULID)
    assert any("Using the filename ULID" in r.message for r in caplog.records)
    # Programmatic manifests are not file-backed
    assert not Manifest(manifest_type=ManifestType.INPUT).is_file_backed
    assert Manifest(manifest_type=ManifestType.INPUT).source_path is None


def test_from_file_lenient_on_invalid_filename(tmp_path, test_jpss_manifest, caplog):
    """A badly named manifest file is read with a warning; it has no filename/ULID and cannot be saved"""
    bad_path = tmp_path / "just_a_manifest.json"
    bad_path.write_text(test_jpss_manifest.read_text())
    with caplog.at_level("WARNING"):
        m = Manifest.from_file(bad_path)
    assert any("does not have a valid manifest filename" in r.message for r in caplog.records)
    assert m.manifest_type == ManifestType.INPUT
    assert len(m.files) == 2
    assert m.filename is None
    assert m.ulid_code is None
    assert m.is_file_backed
    with pytest.raises(ManifestError, match="not a valid manifest filename"):
        m.save()
    with pytest.raises(ManifestError, match="has no ULID"):
        Manifest.for_output_from_input(m)
    # Explicit construction with an invalid filename remains a hard error
    with pytest.raises(ValidationError):
        Manifest(manifest_type=ManifestType.INPUT, filename="just_a_manifest.json")


def test_save_writes_back_to_source(tmp_path):
    """save() overwrites the file a manifest was read from"""
    path = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME).write(tmp_path)
    m = Manifest.from_file(path)
    m.configuration["comment"] = "edited"
    assert m.save() == path
    reread = Manifest.from_file(path)
    assert reread.configuration["comment"] == "edited"
    assert json.loads(path.read_text())["filename"] == str(path)


def test_save_s3(create_mock_bucket):
    """save() works for S3-backed manifests"""
    bucket = create_mock_bucket()
    path = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME).write(f"s3://{bucket.name}/processing")
    m = Manifest.from_file(path)
    m.add_desired_time_range(datetime(2026, 1, 1), datetime(2026, 1, 2))
    m.save()
    assert Manifest.from_file(path).configuration["start_time"] == "2026-01-01:00:00:00"


def test_save_requires_file_backed_manifest():
    """save() on a manifest built in code has nowhere to go"""
    with pytest.raises(ManifestError, match="not read from a file"):
        Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME).save()


def test_save_refuses_type_mismatch(tmp_path, test_jpss_manifest):
    """A file whose name says OUTPUT but whose contents say INPUT is readable but cannot be saved as-is"""
    path = tmp_path / OUTPUT_NAME
    path.write_text(test_jpss_manifest.read_text())
    m = Manifest.from_file(path)
    with pytest.raises(ManifestError, match="named as a OUTPUT manifest"):
        m.save()


def test_copy_is_detached_deep_copy(tmp_path):
    """copy() gives an independent, non-file-backed manifest"""
    path = Manifest(manifest_type=ManifestType.INPUT, filename=INPUT_NAME, configuration={"a": [1]}).write(tmp_path)
    original = Manifest.from_file(path)
    detached = original.copy()
    assert isinstance(detached, Manifest)
    assert not detached.is_file_backed
    assert detached.source_path is None
    assert detached.filename == original.filename
    detached.configuration["a"].append(2)
    assert original.configuration["a"] == [1]
    with pytest.raises(ManifestError):
        detached.save()
    (tmp_path / "elsewhere").mkdir()
    assert detached.write(tmp_path / "elsewhere") == tmp_path / "elsewhere" / INPUT_NAME


def test_for_input(test_txt, test_jpss1_cr_1):
    """for_input creates an INPUT manifest with a ULID and bare filename assigned up front"""
    m = Manifest.for_input(files=[test_txt, test_jpss1_cr_1], configuration={"k": "v"})
    assert m.manifest_type == ManifestType.INPUT
    assert m.ulid_code is not None
    assert str(m.filename) == f"LIBERA_INPUT_MANIFEST_{m.ulid_code}.json"
    assert len(m.files) == 2
    assert m.configuration == {"k": "v"}
    assert not m.is_file_backed
    m.validate_checksums()

    empty = Manifest.for_input()
    assert empty.files == []
    assert empty.configuration == {}

    code = ULID()
    assert Manifest.for_input(ulid_code=code).ulid_code == code


def test_for_input_write_uses_assigned_ulid(tmp_path):
    """The ULID chosen by for_input is the one that ends up in the written filename"""
    m = Manifest.for_input()
    written = m.write(tmp_path)
    assert written == tmp_path / f"LIBERA_INPUT_MANIFEST_{m.ulid_code}.json"


def test_for_output_from_input_object(test_jpss_manifest):
    """for_output_from_input preserves the ULID and records the input files for lineage"""
    input_manifest = Manifest.from_file(test_jpss_manifest)
    out = Manifest.for_output_from_input(input_manifest, configuration={"extra": 1})
    assert out.manifest_type == ManifestType.OUTPUT
    assert out.ulid_code == input_manifest.ulid_code
    assert str(out.filename) == f"LIBERA_OUTPUT_MANIFEST_{input_manifest.ulid_code}.json"
    assert out.configuration["input_manifest_files"] == input_manifest.files
    assert out.configuration["extra"] == 1
    assert out.files == []
    assert not out.is_file_backed


def test_for_output_from_input_path_and_files(test_jpss_manifest, test_txt, write_file_to_s3):
    """for_output_from_input accepts local and S3 paths and an initial file list"""
    out = Manifest.for_output_from_input(test_jpss_manifest, files=[test_txt])
    assert out.ulid_code == ULID.from_str(VALID_ULID)
    assert [Path(f.filename) for f in out.files] == [test_txt]

    s3_path = write_file_to_s3(test_jpss_manifest, f"s3://for-output-from-input-bucket/{INPUT_NAME}")
    out_s3 = Manifest.for_output_from_input(str(s3_path))
    assert out_s3.ulid_code == ULID.from_str(VALID_ULID)


def test_for_output_from_input_requires_ulid():
    """An input manifest without a ULID cannot produce a traceable output manifest"""
    with pytest.raises(ManifestError, match="has no ULID"):
        Manifest.for_output_from_input(Manifest(manifest_type=ManifestType.INPUT))


def test_for_output_from_input_warns_for_non_input(caplog):
    """Deriving from something that is not an INPUT manifest is allowed but flagged"""
    source = Manifest(manifest_type=ManifestType.OUTPUT, filename=OUTPUT_NAME)
    with caplog.at_level("WARNING"):
        out = Manifest.for_output_from_input(source)
    assert out.ulid_code == source.ulid_code
    assert any("not INPUT" in r.message for r in caplog.records)


def test_output_manifest_from_input_manifest_deprecated(test_jpss_manifest):
    """The old method still works and points at the replacement"""
    input_manifest = Manifest.from_file(test_jpss_manifest)
    with pytest.warns(DeprecationWarning, match="for_output_from_input"):
        out = Manifest.output_manifest_from_input_manifest(input_manifest)
    assert out.manifest_type == ManifestType.OUTPUT
    assert out.ulid_code == input_manifest.ulid_code
    assert out.configuration["input_manifest_files"] == input_manifest.files


def test_written_manifest_round_trips(tmp_path, test_txt):
    """A manifest written with write() reads back equal in content"""
    m = Manifest.for_input(files=[test_txt], configuration={"n": 1})
    path = m.write(tmp_path)
    reread = Manifest.from_file(path)
    assert reread.manifest_type == m.manifest_type
    assert reread.files == m.files
    assert reread.configuration == m.configuration
    assert reread.ulid_code == m.ulid_code
    assert reread.filename.path == path
