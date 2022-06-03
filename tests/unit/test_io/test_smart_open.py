"""Tests for smart_open module"""
# Standard
from pathlib import Path
# Installed
import pytest
from cloudpathlib import S3Path, AnyPath
# Local
from libera_sdp.io.smart_open import smart_open, is_gzip, is_s3


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
