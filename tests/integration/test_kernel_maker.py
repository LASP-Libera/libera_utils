"""Tests for kernel_maker CLI module"""

from datetime import UTC, datetime
from unittest import mock

import pytest
from cloudpathlib import AnyPath, S3Path
from ulid import ULID

from libera_utils import kernel_maker
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.manifest import Manifest

# Mark test module as integration tests
pytestmark = pytest.mark.integration

_VERSION = "V3-14-159"


def _assert_kernel_written(output_dir: PathType, product_name: str, utc_start: datetime, utc_end: datetime) -> None:
    """Assert that exactly one kernel for the given product was written to output_dir with the expected time range.

    The revision part of the filename is a freshly generated ULID, so the name cannot be predicted as a string; find
    the file by product and check its parsed parts instead.
    """
    prefix = f"LIBERA_SPICE_{product_name}_{_VERSION}_"
    matches = [p for p in output_dir.iterdir() if p.name.startswith(prefix)]
    assert len(matches) == 1, f"Expected exactly one {product_name} kernel in {output_dir}, found {matches}"
    parts = LiberaDataProductFilename(matches[0]).filename_parts
    assert parts.product_name == product_name
    assert parts.version == _VERSION
    assert parts.utc_start == utc_start
    assert parts.utc_end == utc_end
    assert isinstance(parts.revision, ULID)
    assert parts.extension == ("bsp" if product_name.endswith("SPK") else "bc")


@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_jpss_spk(
    mocked_get_current_version_str,
    test_jpss1_pds_file_1,
    short_tmp_path,
    curryer_lsk,
    monkeypatch,
    spice_test_data_path,
):
    """Test creating a SPK from packets"""
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))  # added for using kernel manager
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=short_tmp_path,
    ):
        kernel_maker.create_kernel_from_packets(
            input_data_files=[str(test_jpss1_pds_file_1)],
            kernel_identifier="JPSS-SPK",
            output_dir=str(short_tmp_path),
            overwrite=False,
        )
        _assert_kernel_written(
            short_tmp_path,
            "JPSS-SPK",
            datetime(2021, 4, 9, 0, 0, 0, tzinfo=UTC),
            datetime(2021, 4, 9, 1, 59, 59, tzinfo=UTC),
        )


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_jpss_spk_aws(
    mocked_get_current_version_str,
    test_jpss1_pds_file_1,
    create_mock_bucket,
    write_file_to_s3,
    wrapper,
    curryer_lsk,
):
    """Test creating a SPK from packets stored in AWS S3"""
    bucket = create_mock_bucket()
    bucket = bucket.name
    key = "some_path"
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_jpss1_pds_file_1.name}"
    write_file_to_s3(test_jpss1_pds_file_1, kernel_uri)
    packet_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"

    kernel_maker.create_kernel_from_packets(
        input_data_files=[str(packet_s3_path)],
        kernel_identifier="JPSS-SPK",
        output_dir=str(s3_output_directory),
        overwrite=False,
    )

    _assert_kernel_written(
        S3Path(s3_output_directory),
        "JPSS-SPK",
        datetime(2021, 4, 9, 0, 0, 0, tzinfo=UTC),
        datetime(2021, 4, 9, 1, 59, 59, tzinfo=UTC),
    )


@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_jpss_ck(mocked_get_current_version_str, test_jpss1_pds_file_1, short_tmp_path, curryer_lsk):
    """Test creating a CK from packets"""
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=short_tmp_path,
    ):
        kernel_maker.create_kernel_from_packets(
            input_data_files=[str(test_jpss1_pds_file_1)],
            kernel_identifier="JPSS-CK",
            output_dir=str(short_tmp_path),
            overwrite=False,
        )
        _assert_kernel_written(
            short_tmp_path,
            "JPSS-CK",
            datetime(2021, 4, 8, 23, 59, 59, tzinfo=UTC),
            datetime(2021, 4, 9, 1, 59, 58, tzinfo=UTC),
        )


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_jpss_ck_aws(
    mocked_get_current_version_str,
    test_jpss1_pds_file_1,
    create_mock_bucket,
    write_file_to_s3,
    wrapper,
    curryer_lsk,
):
    """Test creating a CK from packets stored in AWS S3"""
    bucket = create_mock_bucket()
    bucket = bucket.name
    key = "some_path"
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_jpss1_pds_file_1.name}"
    write_file_to_s3(test_jpss1_pds_file_1, kernel_uri)
    packet_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"

    kernel_maker.create_kernel_from_packets(
        input_data_files=[str(packet_s3_path)],
        kernel_identifier="JPSS-CK",
        output_dir=str(s3_output_directory),
        overwrite=False,
    )
    _assert_kernel_written(
        S3Path(s3_output_directory),
        "JPSS-CK",
        datetime(2021, 4, 8, 23, 59, 59, tzinfo=UTC),
        datetime(2021, 4, 9, 1, 59, 58, tzinfo=UTC),
    )


@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_az_ck(
    mocked_get_current_version_str, test_ccsds_2025_218_18_37_32, short_tmp_path, curryer_lsk, monkeypatch
):
    """Test creating an Az CK from AzEl packets"""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=short_tmp_path,
    ):
        kernel_maker.create_kernel_from_packets(
            input_data_files=[str(test_ccsds_2025_218_18_37_32)],
            kernel_identifier="AZROT-CK",
            output_dir=str(short_tmp_path),
            overwrite=False,
        )
        _assert_kernel_written(
            short_tmp_path,
            "AZROT-CK",
            datetime(2025, 8, 6, 18, 37, 30, tzinfo=UTC),
            datetime(2025, 8, 6, 18, 41, 27, tzinfo=UTC),
        )


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_az_ck_aws(
    mocked_get_current_version_str,
    test_ccsds_2025_218_18_37_32,
    create_mock_bucket,
    write_file_to_s3,
    wrapper,
    curryer_lsk,
    monkeypatch,
):
    """Test creating an Az CK from AzEl packets stored in AWS S3"""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data
    bucket = create_mock_bucket()
    bucket = bucket.name
    key = "some_path"
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_ccsds_2025_218_18_37_32.name}"
    write_file_to_s3(test_ccsds_2025_218_18_37_32, kernel_uri)
    packet_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"

    kernel_maker.create_kernel_from_packets(
        input_data_files=[str(packet_s3_path)],
        kernel_identifier="AZROT-CK",
        output_dir=str(s3_output_directory),
        overwrite=False,
    )
    _assert_kernel_written(
        S3Path(s3_output_directory),
        "AZROT-CK",
        datetime(2025, 8, 6, 18, 37, 30, tzinfo=UTC),
        datetime(2025, 8, 6, 18, 41, 27, tzinfo=UTC),
    )


@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_el_ck(
    mocked_get_current_version_str, test_ccsds_2025_218_18_37_32, short_tmp_path, curryer_lsk, monkeypatch
):
    """Test creating an El CK from AzEl packets"""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=short_tmp_path,
    ):
        kernel_maker.create_kernel_from_packets(
            input_data_files=[str(test_ccsds_2025_218_18_37_32)],
            kernel_identifier="ELSCAN-CK",
            output_dir=str(short_tmp_path),
            overwrite=False,
        )
        _assert_kernel_written(
            short_tmp_path,
            "ELSCAN-CK",
            datetime(2025, 8, 6, 18, 37, 30, tzinfo=UTC),
            datetime(2025, 8, 6, 18, 41, 27, tzinfo=UTC),
        )


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_el_ck_aws(
    mocked_get_current_version_str,
    test_ccsds_2025_218_18_37_32,
    create_mock_bucket,
    write_file_to_s3,
    wrapper,
    curryer_lsk,
    monkeypatch,
):
    """Test creating an El CK from AzEl packets stored in AWS S3"""
    bucket = create_mock_bucket()
    bucket = bucket.name
    key = "some_path"
    kernel_uri = f"s3://{bucket}/{key}/test_kernel/{test_ccsds_2025_218_18_37_32.name}"
    write_file_to_s3(test_ccsds_2025_218_18_37_32, kernel_uri)
    packet_s3_path = wrapper(f"{kernel_uri}")
    s3_output_directory = f"s3://{bucket}/{key}/kernel_output/"
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data

    kernel_maker.create_kernel_from_packets(
        input_data_files=[str(packet_s3_path)],
        kernel_identifier="ELSCAN-CK",
        output_dir=str(s3_output_directory),
        overwrite=False,
    )
    _assert_kernel_written(
        S3Path(s3_output_directory),
        "ELSCAN-CK",
        datetime(2025, 8, 6, 18, 37, 30, tzinfo=UTC),
        datetime(2025, 8, 6, 18, 41, 27, tzinfo=UTC),
    )


@pytest.mark.parametrize("test_type", ["S3", "Local"], indirect=True)
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_jpss_kernels_from_manifest(
    mocked_get_current_version_str, setup_jpss1_kernel_maker_environment_with_manifest, curryer_lsk
):
    # Test that the kernels are generated when no desired range
    # is given.
    input_manifest_path, output_path = setup_jpss1_kernel_maker_environment_with_manifest

    mani_out = kernel_maker.create_kernels_from_manifest(input_manifest_path, ["JPSS-CK", "JPSS-SPK"], output_path)

    assert isinstance(mani_out, Manifest)
    assert len(mani_out.files) == 2  # Two kernel types.
    # Time ranges are real based on the input L1A packet data
    utc_start = datetime(2028, 5, 5, 4, 13, 29, tzinfo=UTC)
    utc_end = datetime(2028, 5, 5, 4, 31, 28, tzinfo=UTC)
    _assert_kernel_written(output_path, "JPSS-SPK", utc_start, utc_end)
    _assert_kernel_written(output_path, "JPSS-CK", utc_start, utc_end)
    assert len(sorted(output_path.glob("*"))) == 3  # 2 kernels + 1 manifest.


@pytest.mark.parametrize("test_type", ["S3", "Local"], indirect=True)
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_make_azel_kernels_from_manifest(
    mocked_get_current_version_str, setup_azel_kernel_maker_environment_with_manifest, curryer_lsk, monkeypatch
):
    """Test that the kernels are generated when no desired range
    is given.
    """
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data

    input_manifest_path, output_path = setup_azel_kernel_maker_environment_with_manifest

    mani_out = kernel_maker.create_kernels_from_manifest(input_manifest_path, ["AZROT-CK", "ELSCAN-CK"], output_path)

    assert isinstance(mani_out, Manifest)
    assert len(mani_out.files) == 2  # Two kernel types.
    # Time ranges are real based on the input L1A packet data
    utc_start = datetime(2025, 8, 9, 17, 17, 56, tzinfo=UTC)
    utc_end = datetime(2025, 8, 9, 17, 19, 4, tzinfo=UTC)
    _assert_kernel_written(output_path, "AZROT-CK", utc_start, utc_end)
    _assert_kernel_written(output_path, "ELSCAN-CK", utc_start, utc_end)
    assert len(sorted(output_path.glob("*"))) == 3  # 2 kernels + 1 manifest.


@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value=_VERSION)
def test_create_kernel_from_l1a_furnishes_kernels(
    mocked_get_current_version_str,
    test_l1a_sc_pos_product_file,
    short_tmp_path,
    monkeypatch,
    spice_test_data_path,
):
    """
    Test that create_kernel_from_l1a properly furnishes kernels via KernelManager: validating that kernel_maker uses
    kernel_manager to furnish required kernels before calling spice_utils.make_kernel().
    """
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))

    # Main Test: Create Kernel from existing L1A, which should internally:
    # 1) Create KernelManager
    # 2) Call km.load_static_kernels() - furnishing NAIF & static kernels
    # 3) Call km.ensure_known_kernels_are_furnished()
    # 4) Call spice_utils.make_kernel()
    output = kernel_maker.create_kernel_from_l1a(
        l1a_data=test_l1a_sc_pos_product_file, kernel_identifier="JPSS-SPK", output_dir=short_tmp_path, overwrite=True
    )

    assert output.exists(), (
        "Kernel file should exist. If this fails, kernel creation failed, "
        "likely because KernelManager didn't furnish required kernels (e.g. LSK)."
    )
    assert output.suffix == ".bsp", "Output should be an SPK file"

    # Verify it's not an empty kernel
    assert output.stat().st_size > 1024, "Kernel file should be larger than 1KB"
