"""Tests for kernel_maker CLI module"""

from datetime import datetime
from unittest import mock

import curryer.spicierpy as sp
import pytest
from cloudpathlib import AnyPath, S3Path

from libera_utils import kernel_maker
from libera_utils.constants import DataProductIdentifier
from libera_utils.io.manifest import Manifest
from libera_utils.l1a.packets import parse_packets_to_l1a_dataset

# Mark test module as integration tests
pytestmark = pytest.mark.integration


@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_jpss_spk(mocked_get_current_version_str, test_jpss1_pds_file_1, short_tmp_path, noaa20_environment, curryer_lsk):
    """Test creating a SPK from packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
        assert (
            short_tmp_path / "LIBERA_SPICE_JPSS-SPK_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp"
        ).exists()


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_jpss_spk_aws(
    mocked_get_current_version_str,
    test_jpss1_pds_file_1,
    create_mock_bucket,
    write_file_to_s3,
    wrapper,
    curryer_lsk,
):
    """Test creating a SPK from packets stored in AWS S3"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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

    s3_output_path = S3Path(s3_output_directory)
    assert (
        s3_output_path / "LIBERA_SPICE_JPSS-SPK_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp"
    ).exists()


@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_jpss_ck(mocked_get_current_version_str, test_jpss1_pds_file_1, short_tmp_path, curryer_lsk):
    """Test creating a CK from packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
        assert (
            short_tmp_path / "LIBERA_SPICE_JPSS-CK_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc"
        ).exists()


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_jpss_ck_aws(
    mocked_get_current_version_str,
    test_jpss1_pds_file_1,
    create_mock_bucket,
    write_file_to_s3,
    wrapper,
    curryer_lsk,
):
    """Test creating a CK from packets stored in AWS S3"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
    s3_output_path = S3Path(s3_output_directory)
    assert (s3_output_path / "LIBERA_SPICE_JPSS-CK_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc").exists()


@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_az_ck(
    mocked_get_current_version_str, test_ccsds_2025_218_18_37_32, short_tmp_path, curryer_lsk, monkeypatch
):
    """Test creating an Az CK from AzEl packets"""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
        assert (
            short_tmp_path / "LIBERA_SPICE_AZROT-CK_V3-14-159_20250806T183730_20250806T184127_R25056154513.bc"
        ).exists()


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
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
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
    s3_output_path = S3Path(s3_output_directory)
    assert (s3_output_path / "LIBERA_SPICE_AZROT-CK_V3-14-159_20250806T183730_20250806T184127_R25056154513.bc").exists()


@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_el_ck(
    mocked_get_current_version_str, test_ccsds_2025_218_18_37_32, short_tmp_path, curryer_lsk, monkeypatch
):
    """Test creating an El CK from AzEl packets"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
        assert (
            short_tmp_path / "LIBERA_SPICE_ELSCAN-CK_V3-14-159_20250806T183730_20250806T184127_R25056154513.bc"
        ).exists()


@pytest.mark.parametrize("wrapper", [AnyPath, S3Path, str])
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
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
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
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
    s3_output_path = S3Path(s3_output_directory)
    assert (
        s3_output_path / "LIBERA_SPICE_ELSCAN-CK_V3-14-159_20250806T183730_20250806T184127_R25056154513.bc"
    ).exists()


@pytest.mark.parametrize("test_type", ["S3", "Local"], indirect=True)
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_jpss_kernels_from_manifest(
    mocked_get_current_version_str, setup_jpss1_kernel_maker_environment_with_manifest, curryer_lsk
):
    # Test that the kernels are generated when no desired range
    # is given.
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_jpss1_kernel_maker_environment_with_manifest

    mani_out = kernel_maker.create_kernels_from_manifest(input_manifest_path, ["JPSS-CK", "JPSS-SPK"], output_path)

    assert isinstance(mani_out, Manifest)
    assert len(mani_out.files) == 2  # Two kernel types.
    # Time ranges are real based on the input L1A packet data
    assert (output_path / "LIBERA_SPICE_JPSS-SPK_V3-14-159_20280505T041329_20280505T043128_R25056154513.bsp").exists()
    assert (output_path / "LIBERA_SPICE_JPSS-CK_V3-14-159_20280505T041329_20280505T043128_R25056154513.bc").exists()
    assert len(sorted(output_path.glob("*"))) == 3  # 2 kernels + 1 manifest.


@pytest.mark.parametrize("test_type", ["S3", "Local"], indirect=True)
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_make_azel_kernels_from_manifest(
    mocked_get_current_version_str, setup_azel_kernel_maker_environment_with_manifest, curryer_lsk, monkeypatch
):
    """Test that the kernels are generated when no desired range
    is given.
    """
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")  # Set skip header bytes for ground test data

    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    input_manifest_path, output_path = setup_azel_kernel_maker_environment_with_manifest

    mani_out = kernel_maker.create_kernels_from_manifest(input_manifest_path, ["AZROT-CK", "ELSCAN-CK"], output_path)

    assert isinstance(mani_out, Manifest)
    assert len(mani_out.files) == 2  # Two kernel types.
    # Time ranges are real based on the input L1A packet data
    assert (output_path / "LIBERA_SPICE_AZROT-CK_V3-14-159_20250809T171756_20250809T171904_R25056154513.bc").exists()
    assert (output_path / "LIBERA_SPICE_ELSCAN-CK_V3-14-159_20250809T171756_20250809T171904_R25056154513.bc").exists()
    assert len(sorted(output_path.glob("*"))) == 3  # 2 kernels + 1 manifest.


@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_create_kernel_from_l1a_furnishes_kernels(
    mocked_get_current_version_str, test_jpss1_pds_file_1, short_tmp_path, noaa20_environment, monkeypatch, spice_test_data_path
):
    """
    Test that create_kernel_from_l1a properly furnishes kernels via KernelManager: validating that kernel_maker uses
    kernel_manager to furnish required kernels before calling spice_utils.make_kernel().
    """
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))

    # clear existing kernels and verify
    sp.kclear()
    initial_kernel_count = sp.ktotal("ALL")
    assert initial_kernel_count == 0, "Expected no kernels to be furnished at start of test"

    # Parse packets to L1A dataset (kernel_maker will handle kernel management internally)
    l1a_dataset = parse_packets_to_l1a_dataset(
        packet_files=[test_jpss1_pds_file_1],
        apid=DataProductIdentifier.l1a_jpss_sc_pos_decoded,
    )

    # Call create_kernel_from_l1a should internally: 1) create KernelManager, 2) load static kernels,
    # 3) ensure known kernels are furnished, and 4) call spice_utils.make_kernel()
    output = kernel_maker.create_kernel_from_l1a(
        l1a_data=l1a_dataset,
        kernel_identifier="JPSS-SPK",
        output_dir=short_tmp_path,
        overwrite=True
    )

    # Verify kernel creation
    assert output.exists(), "Output kernel should have been created"
    assert output.suffix == ".bsp", "Output should be a binary SPK file"

    # Verify that kernels were furnished during execution
    final_kernel_count = sp.ktotal("ALL")
    assert final_kernel_count > 0, (
        "KernelManager should have furnished kernels. "
        "If this fails, create_kernel_from_l1a is not using KernelManager correctly."
    )

    # Verify LSK specifically was furnished
    furnished_kernels = [sp.kdata(i, "ALL")[0] for i in range(final_kernel_count)]
    lsk_furnished = any("naif" in k.lower() and ".tls" in k.lower() for k in furnished_kernels)
    assert lsk_furnished, (
        "Leap second kernel (LSK) should be furnished by KernelManager. "
        "make_kernel requires LSK for time conversion."
    )

    sp.kclear()
