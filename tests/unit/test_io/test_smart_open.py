"""Tests for smart_open module"""
# Standard
import h5py as h5
import numpy as np
from pathlib import Path
# Installed
import pytest
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils.io.smart_open import smart_open, is_gzip, is_s3, smart_copy_file


@pytest.mark.parametrize(
    ("path", "expectation"),
    [
        ("s3://foobucket/barfile.txt", True),
        ("s3:/badurl/nofile.txt", False),
        ("/tmp/foofile.txt", False),
        (Path("/tmp/foofile.txt"), False),
        (S3Path("s3://foobucket/barfile.txt"), True),
        (AnyPath("/tmp/foofile.tar"), False),
        (AnyPath("s3://foobucket/barfile.txt"), True),
    ]
)
def test_is_s3(path, expectation):
    assert is_s3(path) == expectation


@pytest.mark.parametrize(
    ("path", "expectation"),
    [
        ("s3://foobucket/barfile.txt", False),
        ("s3:/badurl/nofile.txt", False),
        ("/tmp/foofile.txt", False),
        (Path("/tmp/foofile.tar"), False),
        (S3Path("s3://foobucket/barfile.txt"), False),
        ("s3://foobucket/barfile.txt.gz", True),
        ("s3:/badurl/nofile.txt.gz", True),
        ("/tmp/foofile.txt.gz", True),
        (Path("/tmp/foofile.tar.gz"), True),
        (S3Path("s3://foobucket/barfile.txt.gz"), True),
        (AnyPath("/tmp/foofile.tar"), False),
        (AnyPath("s3://foobucket/barfile.txt"), False),
    ]
)
def test_is_gzip(path, expectation):
    assert is_gzip(path) == expectation


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
def test_smart_open_s3(test_txt, test_txt_gz, create_mock_bucket, write_file_to_s3, wrapper):
    """Test smart_open on mocked S3 objects"""
    bucket = 'silly-bucket'
    create_mock_bucket(bucket)  # We could also let write_file_to_s3 do this automatically for us
    key = 'somepath'
    plain_uri = f"s3://{bucket}/{key}/test.txt"
    gz_uri = f"s3://{bucket}/{key}/test.txt.gz"
    write_file_to_s3(test_txt, plain_uri)
    write_file_to_s3(test_txt_gz, gz_uri)

    plain_wrapped = wrapper(plain_uri)
    gz_wrapped = wrapper(gz_uri)
    # Check that the contents of the files match, regardless of compression
    with smart_open(plain_wrapped) as fh_uncompressed:
        uncompressed_contents = fh_uncompressed.readlines()
    with smart_open(gz_wrapped) as fh_compressed:
        compressed_contents = fh_compressed.readlines()
    assert uncompressed_contents == compressed_contents


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
def test_smart_open_hdf5(test_hdf5, create_mock_bucket, write_file_to_s3, wrapper):
    """Test smart_open on mocked S3 objects and locally"""
    bucket = 'silly-bucket'
    create_mock_bucket(bucket)
    key = 'somepath'
    hdf5_uri = f"s3://{bucket}/{key}/test_hdf5"
    write_file_to_s3(test_hdf5, hdf5_uri)

    hdf5_wrapped = wrapper(hdf5_uri)
    # Check that the contents of the files match, regardless of s3 or local
    with h5.File(smart_open(test_hdf5), 'r') as fh:
        dataset_local = np.array(fh[list(fh.keys())[0]])
    with h5.File(smart_open(hdf5_wrapped), 'r') as fh:
        dataset_s3 = np.array(fh[list(fh.keys())[0]])
    assert dataset_local.all() == dataset_s3.all()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, S3Path, str]
)
def test_smart_open_mode(create_mock_bucket, write_file_to_s3, wrapper, test_hdf5):
    """
    Test smart_open can read in and write to hdf5.
    """
    bucket = 'silly-bucket'
    create_mock_bucket(bucket)
    key = 'somepath'
    hdf5_uri = f"s3://{bucket}/{key}/path"
    write_file_to_s3(test_hdf5, hdf5_uri)

    hdf5_wrapped = wrapper(hdf5_uri)

    with smart_open(hdf5_wrapped, 'wb') as fh:
        with h5.File(fh, 'r+') as hdf:
            hdf.create_group('new_group')
    with h5.File(smart_open(hdf5_wrapped), 'r') as fh:
        group_name = list(fh.keys())[0]
    assert group_name == 'new_group'


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, Path, str]
)
def test_smart_open_local(test_txt, test_txt_gz, wrapper):
    """Test smart_open on local files"""
    plain_wrapped = wrapper(test_txt.absolute())
    gz_wrapped = wrapper(test_txt_gz.absolute())
    # Check that the contents of the files match, regardless of compression
    with smart_open(plain_wrapped) as fh_uncompressed:
        uncompressed_contents = fh_uncompressed.readlines()
    with smart_open(gz_wrapped) as fh_compressed:
        compressed_contents = fh_compressed.readlines()
    assert uncompressed_contents == compressed_contents

@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_local_to_local_file(tmp_path, test_txt, wrapper):
    """Test smart copy for a local file to local file path"""
    # Make the local destination
    tmp_folder_path = tmp_path / "destination"
    tmp_folder_path.mkdir()

    wrapped_input_file = wrapper(test_txt)
    wrapped_output_file = wrapper(tmp_folder_path / "newfilename.txt")

    smart_copy_file(wrapped_input_file, wrapped_output_file)
    assert (tmp_folder_path / "newfilename.txt").exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_local_to_local_directory(tmp_path, test_txt, wrapper):
    """Test smart copy for a local file to local directory path"""
    # Make the local destination
    tmp_folder_path = tmp_path / "destination"
    tmp_folder_path.mkdir()

    wrapped_input = wrapper(test_txt)
    wrapped_output_dir = wrapper(tmp_folder_path)

    smart_copy_file(wrapped_input, wrapped_output_dir)
    assert (tmp_folder_path / "testtextfile.txt").exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_remote_to_local_directory(tmp_path, test_txt, wrapper, create_mock_bucket, write_file_to_s3):
    """Test smart_copy for a remote file to a local directory path"""
    local_folder_path = tmp_path / "destination"
    local_folder_path.mkdir()

    bucket = 'tmp-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    remote_file_uri = f"s3://{bucket}/{key}/internal/testtextfile.txt"
    write_file_to_s3(test_txt, remote_file_uri)

    wrapped_remote_file_path = wrapper(f"{remote_file_uri}")
    wrapped_local_destination = wrapper(local_folder_path)

    smart_copy_file(wrapped_remote_file_path, wrapped_local_destination)
    assert (local_folder_path / "testtextfile.txt").exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_remote_to_local_file(tmp_path, test_txt, wrapper, create_mock_bucket, write_file_to_s3):
    """Test smart_copy for a remote file to ta local file path"""
    local_folder_path = tmp_path / "destination"
    local_folder_path.mkdir()

    bucket = 'tmp-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    remote_file_uri = f"s3://{bucket}/{key}/internal/testtextfile.txt"
    write_file_to_s3(test_txt, remote_file_uri)

    wrapped_remote_file_path = wrapper(f"{remote_file_uri}")
    wrapped_local_file_path = wrapper(local_folder_path / "newfilename.txt")

    smart_copy_file(wrapped_remote_file_path, wrapped_local_file_path)
    assert (local_folder_path / "newfilename.txt").exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_local_to_remote_directory(test_txt, wrapper, create_mock_bucket):
    """Test smart_copy for a local file to a remote destination"""
    bucket = 'tmp-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    remote_file_uri = f"s3://{bucket}/{key}/internal/data"

    remote_path = wrapper(f"{remote_file_uri}")
    local_file_path = wrapper(test_txt)
    smart_copy_file(local_file_path, remote_path)

    remote_path = S3Path(remote_file_uri)
    assert remote_path.exists()

@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_local_to_remote_file(test_txt, wrapper, create_mock_bucket):
    """Test smart_copy for a local file to a remote file location"""
    bucket = 'tmp-bucket'
    create_mock_bucket(bucket)
    key = 'some_path'
    remote_file_uri = f"s3://{bucket}/{key}/internal/newfilename.txt"

    remote_file_path = wrapper(f"{remote_file_uri}")
    local_file_path = wrapper(test_txt)
    smart_copy_file(local_file_path, remote_file_path)

    remote_path = S3Path(remote_file_uri)
    assert remote_path.exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_remote_to_remote_directory(test_txt, wrapper, create_mock_bucket, write_file_to_s3):
    """Test smart_copy for a remote file to a different remote directory location"""
    source_bucket = 'tmp-source-bucket'
    create_mock_bucket(source_bucket)
    source_key = 'source_path'
    source_file_uri = f"s3://{source_bucket}/{source_key}/internal/sourcedata"
    write_file_to_s3(test_txt, source_file_uri)

    dest_bucket = 'tmp-dest-bucket'
    create_mock_bucket(dest_bucket)
    dest_key = 'dest_path'
    dest_file_uri = f"s3://{dest_bucket}/{dest_key}/internal/destdata"

    source_file_path = wrapper(f"{source_file_uri}")
    dest_path = wrapper(dest_file_uri)
    smart_copy_file(source_file_path, dest_path)

    dest_path = S3Path(dest_file_uri)
    assert dest_path.exists()


@pytest.mark.parametrize(
    "wrapper",
    [AnyPath, str]
)
def test_smart_copy_file_remote_to_remote_file(test_txt, wrapper, create_mock_bucket, write_file_to_s3):
    """Test smart_copy for a remote file to a different remote file location"""
    source_bucket = 'tmp-source-bucket'
    create_mock_bucket(source_bucket)
    source_key = 'source_path'
    source_file_uri = f"s3://{source_bucket}/{source_key}/internal/testtextfile.txt"
    write_file_to_s3(test_txt, source_file_uri)

    dest_bucket = 'tmp-dest-bucket'
    create_mock_bucket(dest_bucket)
    dest_key = 'dest_path'
    dest_file_uri = f"s3://{dest_bucket}/{dest_key}/internal/newfilename.txt"

    source_file_path = wrapper(f"{source_file_uri}")
    dest_file_path = wrapper(dest_file_uri)
    smart_copy_file(source_file_path, dest_file_path)

    dest_path = S3Path(dest_file_uri)
    assert dest_path.exists()

