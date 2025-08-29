"""Unit tests for kernel_maker module"""

import argparse
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from cloudpathlib import AnyPath

from libera_utils import kernel_maker
from libera_utils.aws.constants import DataProductIdentifier
from libera_utils.config import config
from libera_utils.io.manifest import Manifest, ManifestFileRecord


@pytest.mark.parametrize(
    ("kernel_dpi", "config_key", "out_ext"),
    [
        ("JPSS-SPK", "LIBERA_KERNEL_SC_SPK_CONFIG", "bsp"),
        ("JPSS-CK", "LIBERA_KERNEL_SC_CK_CONFIG", "bc"),
        ("AZROT-CK", "LIBERA_KERNEL_AZ_CK_CONFIG", "bc"),
        ("ELSCAN-CK", "LIBERA_KERNEL_EL_CK_CONFIG", "bc"),
    ],
)
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="v2-5-2")
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch(
    "libera_utils.kernel_maker.preprocess_data",
    return_value=(
        pd.DataFrame(),
        (datetime.fromisoformat("2020-01-01T00:00:00"), datetime.fromisoformat("2020-01-01T23:59:59")),
    ),
)
@mock.patch("libera_utils.kernel_maker.make_kernel", return_value=AnyPath("/fake/kernel.ext"))
def test_from_args(mock_make_kernel, mock_process_data, mock_version, kernel_dpi, config_key, out_ext):
    """Test the kernel maker from args function"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    out = kernel_maker.from_args(
        input_data_files=["/fake/input.csv"],
        kernel_identifier=kernel_dpi,
        output_dir="/fake/dropbox",
        overwrite=False,
        append=True,
    )
    assert isinstance(out, AnyPath)
    assert out.name == "kernel.ext"
    kernel_maker.datetime.now.assert_called()
    mock_process_data.assert_called_once()
    mock_make_kernel.assert_called_once_with(
        config_file=config.get(config_key),
        output_kernel=Path(
            f"/fake/dropbox/LIBERA_{kernel_dpi}_V2-5-2_20200101T000000_20200101T235959_R25056154513.{out_ext}"
        ),
        input_data=mock_process_data.return_value[0],
        overwrite=False,
        append=True,
    )


@mock.patch(
    "libera_utils.io.manifest.Manifest.output_manifest_from_input_manifest", return_value=mock.MagicMock(Manifest)
)
@mock.patch(
    "libera_utils.io.manifest.Manifest.from_file",
    return_value=mock.MagicMock(Manifest, files=[mock.MagicMock(ManifestFileRecord, filename="/fake/input.csv")]),
)
@mock.patch("libera_utils.kernel_maker.from_args", return_value=AnyPath("/fake/kernel.spk"))
def test_from_manifest(mock_from_args, mock_mani, mock_pedi):
    """Test the kernel maker from manifest function"""
    pedi = kernel_maker.from_manifest(
        input_manifest="/fake/mocked_call.json",
        data_product_identifiers=[DataProductIdentifier.spice_jpss_spk],
        output_dir="/fake/dropbox",
        overwrite=True,
        append=True,
        verbose=True,
    )
    assert isinstance(pedi, Manifest)
    assert pedi is mock_pedi.return_value
    mock_mani.return_value.validate_checksums.assert_called_once()
    mock_pedi.assert_called_once_with(mock_mani.return_value)
    mock_pedi.return_value.add_files.assert_called_once_with(Path("/fake/kernel.spk"))
    mock_from_args.assert_called_once_with(
        input_data_files=["/fake/input.csv"],
        kernel_identifier=DataProductIdentifier.spice_jpss_spk,
        output_dir="/fake/dropbox",
        overwrite=True,
        append=True,
        verbose=True,
    )


@pytest.mark.parametrize(
    ("cli_handler", "dpis"),
    [
        (
            kernel_maker.jpss_kernel_cli_handler,
            [DataProductIdentifier.spice_jpss_spk, DataProductIdentifier.spice_jpss_ck],
        ),
        (kernel_maker.azel_kernel_cli_handler, [DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck]),
    ],
)
@mock.patch("libera_utils.kernel_maker.from_manifest", return_value=mock.Mock(Manifest))
def test_kernel_cli_handler(mock_from_manifest, cli_handler, dpis, monkeypatch):
    """Test the kernel maker CLI functions"""
    monkeypatch.setenv("PROCESSING_PATH", "/fake/dropbox")
    mock_parsed_args = argparse.Namespace(input_manifest="manifest.json", verbose=False)

    out = cli_handler(mock_parsed_args)
    assert isinstance(out, Manifest)
    mock_from_manifest.assert_called_once_with(
        input_manifest="manifest.json",
        data_product_identifiers=dpis,
        output_dir=Path("/fake/dropbox"),
        overwrite=False,
        append=False,
        verbose=False,
    )
