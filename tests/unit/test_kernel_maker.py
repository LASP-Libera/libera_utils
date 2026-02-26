"""Unit tests for kernel_maker module"""

import argparse
from datetime import datetime
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cloudpathlib import AnyPath

from libera_utils import kernel_maker
from libera_utils.config import config
from libera_utils.constants import DataProductIdentifier, LiberaApid
from libera_utils.io.manifest import Manifest, ManifestFileRecord
from libera_utils.l1a.l1a_packet_configs import PacketConfiguration


@pytest.mark.parametrize(
    ("kernel_dpi", "apid"),
    [
        ("JPSS-SPK", LiberaApid.jpss_sc_pos),
        ("JPSS-CK", LiberaApid.jpss_sc_pos),
        ("AZROT-CK", LiberaApid.icie_axis_sample),
        ("ELSCAN-CK", LiberaApid.icie_axis_sample),
    ],
)
@mock.patch("libera_utils.l1a.packets.parse_packets_to_l1a_dataset", return_value=xr.Dataset())
@mock.patch("libera_utils.kernel_maker.create_kernel_from_l1a", return_value=AnyPath("/fake/kernel.ext"))
def test_create_kernel_from_packets(
    mock_create_kernel_from_l1a,
    mock_parse_packets_to_l1a_dataset,
    kernel_dpi,
    apid,
):
    """Test the kernel maker create from packets function,
    mocking out the parsing of the packet data and the call to create the kernel from the L1A dataset.
    """
    out = kernel_maker.create_kernel_from_packets(
        input_data_files=["/fake/input.pkts"], kernel_identifier=kernel_dpi, output_dir="/fake/dropbox", overwrite=False
    )
    assert isinstance(out, AnyPath)
    assert out.name == "kernel.ext"
    mock_parse_packets_to_l1a_dataset.assert_called_once_with(packet_files=["/fake/input.pkts"], apid=apid)
    mock_create_kernel_from_l1a.assert_called_once_with(
        l1a_data=mock_parse_packets_to_l1a_dataset.return_value,
        kernel_identifier=kernel_dpi,
        output_dir=Path("/fake/dropbox"),
        overwrite=False,
    )


@pytest.mark.parametrize(
    ("kernel_dpi", "config_key", "out_ext"),
    [
        ("JPSS-SPK", "LIBERA_KERNEL_SC_SPK_CONFIG", "bsp"),
        ("JPSS-CK", "LIBERA_KERNEL_SC_CK_CONFIG", "bc"),
        ("AZROT-CK", "LIBERA_KERNEL_AZ_CK_CONFIG", "bc"),
        ("ELSCAN-CK", "LIBERA_KERNEL_EL_CK_CONFIG", "bc"),
    ],
)
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V2-5-2")
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch(
    "libera_utils.kernel_maker.create_kernel_dataframe_from_l1a",
    return_value=(
        pd.DataFrame(),
        (datetime.fromisoformat("2020-01-01T00:00:00"), datetime.fromisoformat("2020-01-01T23:59:59")),
    ),
)
@mock.patch("libera_utils.libera_spice.spice_utils.make_kernel", return_value=AnyPath("/fake/kernel.spk"))
@mock.patch("libera_utils.kernel_maker.KernelManager")
def test_create_kernel_from_l1a(
    mock_kernel_manager_class,
    mock_make_kernel,
    mock_create_kernel_dataframe_from_l1a,
    mock_version,
    kernel_dpi,
    config_key,
    out_ext,
):
    """Test the kernel maker create from L1A function"""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)

    # Mock KernelManager instance
    mock_km_instance = mock.Mock()
    mock_kernel_manager_class.return_value = mock_km_instance

    out = kernel_maker.create_kernel_from_l1a(
        l1a_data=xr.Dataset(),
        kernel_identifier=DataProductIdentifier(kernel_dpi),
        output_dir="/fake/dropbox",
        overwrite=False,
    )

    # Assert KernelManager was instantiated and methods were called
    mock_kernel_manager_class.assert_called_once()
    mock_km_instance.load_static_kernels.assert_called_once()
    mock_km_instance.ensure_known_kernels_are_furnished.assert_called_once()

    # Assert call to create the Curryer data frame
    mock_create_kernel_dataframe_from_l1a.assert_called_once()
    # Assert call to make the kernel (call out to Curryer)
    mock_make_kernel.assert_called_once_with(
        config_file=Path(config.get(config_key)),
        output_kernel=Path(
            f"/fake/dropbox/LIBERA_SPICE_{kernel_dpi}_V2-5-2_20200101T000000_20200101T235959_R25056154513.{out_ext}"
        ),
        input_data=mock_create_kernel_dataframe_from_l1a.return_value[0],
        overwrite=False,
        append=False,
    )
    kernel_maker.datetime.now.assert_called()

    assert isinstance(out, AnyPath)


@mock.patch(
    "libera_utils.io.manifest.Manifest.output_manifest_from_input_manifest", return_value=mock.MagicMock(Manifest)
)
@mock.patch(
    "libera_utils.io.manifest.Manifest.from_file",
    return_value=mock.MagicMock(Manifest, files=[mock.MagicMock(ManifestFileRecord, filename="/fake/fake_l1a.nc")]),
)
@mock.patch("libera_utils.kernel_maker.create_kernel_from_l1a", return_value=AnyPath("/fake/kernel.spk"))
def test_from_manifest(mock_create_kernel_from_l1a, mock_mani, mock_pedi):
    """Test the kernel maker from manifest function"""
    pedi = kernel_maker.create_kernels_from_manifest(
        input_manifest="/fake/mocked_call.json",
        kernel_product_ids=[DataProductIdentifier.spice_jpss_spk],
        output_dir="/fake/dropbox",
        overwrite=True,
    )
    assert isinstance(pedi, Manifest)
    assert pedi is mock_pedi.return_value
    mock_mani.return_value.validate_checksums.assert_called_once()
    mock_pedi.assert_called_once_with(mock_mani.return_value)
    mock_pedi.return_value.add_files.assert_called_once_with(Path("/fake/kernel.spk"))
    mock_create_kernel_from_l1a.assert_called_once_with(
        l1a_data="/fake/fake_l1a.nc",
        kernel_identifier=DataProductIdentifier.spice_jpss_spk,
        output_dir="/fake/dropbox",
        overwrite=True,
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
@mock.patch("libera_utils.kernel_maker.create_kernels_from_manifest", return_value=mock.Mock(Manifest))
def test_kernel_cli_handler(mock_from_manifest, cli_handler, dpis, monkeypatch):
    """Test the kernel maker CLI functions"""
    monkeypatch.setenv("PROCESSING_PATH", "/fake/dropbox")
    mock_parsed_args = argparse.Namespace(input_manifest="manifest.json", verbose=False)

    out = cli_handler(mock_parsed_args)
    assert isinstance(out, Manifest)
    mock_from_manifest.assert_called_once_with(
        input_manifest="manifest.json", kernel_product_ids=dpis, output_dir="/fake/dropbox", overwrite=False
    )


@mock.patch.object(kernel_maker, "create_kernel_from_l1a")
@mock.patch.object(kernel_maker, "Manifest")
def test_from_manifest_multi_kernel_processing(mock_manifest_class, mock_create_kernel_from_l1a):
    """Test that from_manifest correctly creates multiple kernels from a single manifest"""
    mock_mani = mock.Mock()
    mock_mani.files = [
        mock.Mock(filename="l1a_file.nc"),
    ]
    mock_manifest_class.from_file.return_value = mock_mani
    mock_manifest_class.output_manifest_from_input_manifest.return_value = mock.Mock()

    # Mock from_args to return a unique filename for each call
    mock_create_kernel_from_l1a.side_effect = ["output1.bsp", "output2.bsp"]

    # Call to from_manifest with two DPIs (only one input file per expectations for kernel generation)
    _ = kernel_maker.create_kernels_from_manifest(
        input_manifest="test.manifest",
        kernel_product_ids=["JPSS-SPK", "JPSS-CK"],
        output_dir="/output",
    )

    # Verify that from_args is called with the input file for each DPI
    assert mock_create_kernel_from_l1a.call_count == 2
    expected_calls = [
        mock.call(
            l1a_data="l1a_file.nc",
            kernel_identifier="JPSS-SPK",
            output_dir="/output",
            overwrite=False,
        ),
        mock.call(
            l1a_data="l1a_file.nc",
            kernel_identifier="JPSS-CK",
            output_dir="/output",
            overwrite=False,
        ),
    ]
    mock_create_kernel_from_l1a.assert_has_calls(expected_calls)


@mock.patch("libera_utils.kernel_maker.logger")
@mock.patch.object(kernel_maker, "create_kernel_from_l1a")
@mock.patch.object(kernel_maker, "Manifest")
def test_from_manifest_exception_handling(mock_manifest_class, create_kernel_from_l1a, mock_logger):
    """Test that from_manifest handles exceptions properly and continues processing"""
    mock_mani = mock.Mock()
    mock_mani.files = [mock.Mock(filename="l1a_file.nc")]
    mock_manifest_class.from_file.return_value = mock_mani
    mock_pedi = mock.Mock()
    mock_manifest_class.output_manifest_from_input_manifest.return_value = mock_pedi

    # Mock from_args to raise exception for first DPI, succeed for second
    create_kernel_from_l1a.side_effect = [Exception("Test error"), "output2.bsp"]

    with pytest.raises(
        ValueError,
        match=r"Kernel processing steps failed.*'JPSS-SPK'.*'l1a_file.nc'",
    ):
        _ = kernel_maker.create_kernels_from_manifest(
            input_manifest="test.manifest",
            kernel_product_ids=["JPSS-SPK", "JPSS-CK"],
            output_dir="/output",
        )

    # Verify exception was logged
    mock_logger.exception.assert_called_once()
    # Verify we did not make it to the manifest creation due to the error
    mock_pedi.add_files.assert_not_called()


@pytest.fixture
def mock_packet_config(test_packet_configuration: PacketConfiguration):
    """Mock get_packet_config to return test configuration."""
    with patch("libera_utils.kernel_maker.get_packet_config", return_value=test_packet_configuration):
        yield test_packet_configuration


class TestCreateKernelDataframeFromL1a:
    """Tests for create_kernel_dataframe_from_l1a function."""

    def test_basic_functionality(
        self, l1a_test_product: xr.Dataset, mock_packet_config: PacketConfiguration, curryer_lsk
    ):
        """Test basic DataFrame creation from L1A dataset."""
        # Get APID from test configuration
        apid = mock_packet_config.packet_apid

        # Call function
        df, utc_range = kernel_maker.create_kernel_dataframe_from_l1a(
            l1a_dataset=l1a_test_product, apid=apid, sample_group_name="AXIS_SAMPLE"
        )

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500  # 10 packets * 50 samples

        # Verify columns exist
        expected_columns = {
            "AXIS_SAMPLE_ICIE_ET",  # Time dimension renamed to ET
            "ICIE__AXIS_AZ_FILT",
            "ICIE__AXIS_EL_FILT",
        }
        assert set(df.columns) == expected_columns

        # Verify ET column contains floats (not datetime64)
        assert df["AXIS_SAMPLE_ICIE_ET"].dtype == np.float64

        # Verify data fields contain expected data types
        assert df["ICIE__AXIS_AZ_FILT"].dtype == np.float64
        assert df["ICIE__AXIS_EL_FILT"].dtype == np.float64

        # Verify time range
        assert isinstance(utc_range, tuple)
        assert len(utc_range) == 2
        assert isinstance(utc_range[0], datetime)
        assert isinstance(utc_range[1], datetime)
        assert utc_range[0] < utc_range[1]

    def test_missing_time_dimension_raises_error(
        self, l1a_test_product: xr.Dataset, mock_packet_config: PacketConfiguration
    ):
        """Test that missing time dimension raises KeyError."""
        apid = mock_packet_config.packet_apid

        # Remove time coordinate
        ds_missing_time = l1a_test_product.drop_vars("AXIS_SAMPLE_ICIE_TIME")

        with pytest.raises(KeyError, match="Required time dimension"):
            kernel_maker.create_kernel_dataframe_from_l1a(
                l1a_dataset=ds_missing_time, apid=apid, sample_group_name="AXIS_SAMPLE"
            )

    def test_missing_data_field_raises_error(
        self, l1a_test_product: xr.Dataset, mock_packet_config: PacketConfiguration
    ):
        """Test that missing data fields raise KeyError."""
        apid = mock_packet_config.packet_apid

        # Remove one data variable
        ds_missing_field = l1a_test_product.drop_vars("ICIE__AXIS_AZ_FILT")

        with pytest.raises(KeyError, match="Required data fields missing"):
            kernel_maker.create_kernel_dataframe_from_l1a(
                l1a_dataset=ds_missing_field, apid=apid, sample_group_name="AXIS_SAMPLE"
            )

    def test_invalid_apid_raises_error(self, l1a_test_product: xr.Dataset):
        """Test that invalid APID raises ValueError."""
        with pytest.raises(ValueError, match="Invalid APID"):
            kernel_maker.create_kernel_dataframe_from_l1a(
                l1a_dataset=l1a_test_product,
                apid=99999,  # Non-existent APID
                sample_group_name="AXIS_SAMPLE",
            )

    def test_invalid_sample_group_name_raises_error(
        self, l1a_test_product: xr.Dataset, mock_packet_config: PacketConfiguration
    ):
        """Test that invalid sample_group_name raises ValueError."""
        apid = mock_packet_config.packet_apid

        with pytest.raises(ValueError, match="Sample group.*not found"):
            kernel_maker.create_kernel_dataframe_from_l1a(
                l1a_dataset=l1a_test_product,
                apid=apid,
                sample_group_name="NONEXISTENT_GROUP",
            )

    def test_wrong_input_type_raises_error(self, mock_packet_config: PacketConfiguration):
        """Test that non-Dataset input raises TypeError."""
        apid = mock_packet_config.packet_apid

        with pytest.raises(TypeError, match="must be an xarray.Dataset"):
            kernel_maker.create_kernel_dataframe_from_l1a(
                l1a_dataset="not_a_dataset",  # Wrong type
                apid=apid,
                sample_group_name="AXIS_SAMPLE",
            )


class TestCreateKernelDataframeFromL1aNetcdf:
    """Tests for create_kernel_dataframe_from_l1a_netcdf function."""

    def test_basic_functionality(
        self, l1a_test_product: xr.Dataset, mock_packet_config: PacketConfiguration, tmp_path, curryer_lsk
    ):
        """Test that reading from NetCDF file produces correct DataFrame."""
        # Get APID from test configuration
        apid = mock_packet_config.packet_apid

        # Write test dataset to temporary NetCDF file
        netcdf_path = tmp_path / "test_l1a.nc"
        l1a_test_product.to_netcdf(netcdf_path, engine="h5netcdf")

        # Call function with NetCDF file
        df, utc_range = kernel_maker.create_kernel_dataframe_from_l1a_netcdf(
            netcdf_path=netcdf_path, apid=apid, sample_group_name="AXIS_SAMPLE"
        )

        # Verify DataFrame structure matches expected output
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500  # 10 packets * 50 samples

        # Verify columns exist
        expected_columns = {
            "AXIS_SAMPLE_ICIE_ET",  # Time dimension renamed to ET
            "ICIE__AXIS_AZ_FILT",
            "ICIE__AXIS_EL_FILT",
        }
        assert set(df.columns) == expected_columns

        # Verify ET column contains floats
        assert df["AXIS_SAMPLE_ICIE_ET"].dtype == np.float64

        # Verify time range
        assert isinstance(utc_range, tuple)
        assert len(utc_range) == 2
        assert isinstance(utc_range[0], datetime)
        assert isinstance(utc_range[1], datetime)
        assert utc_range[0] < utc_range[1]

    def test_file_not_found_raises_error(self, mock_packet_config: PacketConfiguration):
        """Test that missing file raises FileNotFoundError."""
        apid = mock_packet_config.packet_apid
        non_existent_path = Path("/nonexistent/path/to/file.nc")

        with pytest.raises(FileNotFoundError, match="NetCDF file not found"):
            kernel_maker.create_kernel_dataframe_from_l1a_netcdf(
                netcdf_path=non_existent_path, apid=apid, sample_group_name="AXIS_SAMPLE"
            )

    def test_invalid_netcdf_file_raises_error(self, mock_packet_config: PacketConfiguration, tmp_path):
        """Test that invalid NetCDF file raises ValueError."""
        apid = mock_packet_config.packet_apid

        # Create a text file (not a valid NetCDF)
        invalid_file = tmp_path / "invalid.nc"
        invalid_file.write_text("This is not a NetCDF file")

        with pytest.raises(ValueError, match="Failed to open or read NetCDF file"):
            kernel_maker.create_kernel_dataframe_from_l1a_netcdf(
                netcdf_path=invalid_file, apid=apid, sample_group_name="AXIS_SAMPLE"
            )

    def test_invalid_apid_raises_error(self, l1a_test_product: xr.Dataset, tmp_path):
        """Test that invalid APID raises ValueError when processing NetCDF."""
        # Write test dataset to temporary NetCDF file
        netcdf_path = tmp_path / "test_l1a.nc"
        l1a_test_product.to_netcdf(netcdf_path, engine="h5netcdf")

        with pytest.raises(ValueError, match="Invalid APID"):
            kernel_maker.create_kernel_dataframe_from_l1a_netcdf(
                netcdf_path=netcdf_path, apid=99999, sample_group_name="AXIS_SAMPLE"
            )


class TestCreateJpssKernelDataframeFromCsv:
    """Tests for create_jpss_kernel_dataframe_from_csv function."""

    def test_basic_functionality(self, tmp_path, curryer_lsk):
        """Test basic CSV reading and DataFrame creation."""
        # Create a test CSV file with known GPS ephemeris data
        csv_path = tmp_path / "test_ephemeris.csv"
        csv_content = """"Time (UTCG)","x (km)","y (km)","z (km)","vx (km/sec)","vy (km/sec)","vz (km/sec)","q1","q2","q3","q4"
2 Jan 2028 00:00:00.000,-2176.938701,236.025335,6861.222353,6.526358,3.180148,1.961296,0.967192,0.202807,0.145488,0.047357
2 Jan 2028 00:00:01.000,-2170.418535,239.200955,6863.177728,6.529126,3.178946,1.953983,0.967274,0.202747,0.144990,0.047467
2 Jan 2028 00:00:02.000,-2163.895608,242.375374,6865.125798,6.531887,3.177740,1.946667,0.967356,0.202687,0.144491,0.047576
"""
        csv_path.write_text(csv_content)

        # Call function
        df, utc_range = kernel_maker.create_jpss_kernel_dataframe_from_csv(csv_path=csv_path)

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

        # Verify expected columns exist
        expected_columns = {
            "ADGPS_JPSS_ET",
            "ADCFA_JPSS_ET",
            "ADGPSPOSX",
            "ADGPSPOSY",
            "ADGPSPOSZ",
            "ADGPSVELX",
            "ADGPSVELY",
            "ADGPSVELZ",
            "ADCFAQ1",
            "ADCFAQ2",
            "ADCFAQ3",
            "ADCFAQ4",
        }
        assert set(df.columns) == expected_columns

        # Verify ET columns contain floats
        assert df["ADGPS_JPSS_ET"].dtype == np.float64
        assert df["ADCFA_JPSS_ET"].dtype == np.float64

        # Verify both ET columns have identical values
        assert (df["ADGPS_JPSS_ET"] == df["ADCFA_JPSS_ET"]).all()

        # Verify position and velocity data preserved but converted from km to meters
        assert df["ADGPSPOSX"].iloc[0] == -2176.938701 * 1000
        assert df["ADGPSVELX"].iloc[1] == 6.529126 * 1000

        # Verify time range
        assert isinstance(utc_range, tuple)
        assert len(utc_range) == 2
        assert isinstance(utc_range[0], datetime)
        assert isinstance(utc_range[1], datetime)
        assert utc_range[0] <= utc_range[1]

    def test_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        non_existent_path = Path("/nonexistent/path/to/file.csv")

        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            kernel_maker.create_jpss_kernel_dataframe_from_csv(csv_path=non_existent_path)

    def test_missing_columns_raises_error(self, tmp_path):
        """Test that missing required columns raises ValueError."""
        # Create CSV with missing columns
        csv_path = tmp_path / "incomplete.csv"
        csv_content = """"Time (UTCG)","x (km)","y (km)","z (km)","vx (km/sec)","vy (km/sec)","vz (km/sec)","q1","q2","q3"
2 Jan 2028 00:00:00.000,-2176.938701,236.025335,6861.222353,6.526358,3.180148,1.961296,0.967192,0.202807,0.145488
2 Jan 2028 00:00:01.000,-2170.418535,239.200955,6863.177728,6.529126,3.178946,1.953983,0.967274,0.202747,0.144990
2 Jan 2028 00:00:02.000,-2163.895608,242.375374,6865.125798,6.531887,3.177740,1.946667,0.967356,0.202687,0.144491
"""
        csv_path.write_text(csv_content)

        with pytest.raises(ValueError, match="Missing required columns"):
            kernel_maker.create_jpss_kernel_dataframe_from_csv(csv_path=csv_path)

    def test_et_conversion_with_lsk(self, tmp_path, curryer_lsk):
        """Test that ET conversion works correctly with LSK loaded."""
        # Create a test CSV file
        csv_path = tmp_path / "test_ephemeris.csv"
        csv_content = """"Time (UTCG)","x (km)","y (km)","z (km)","vx (km/sec)","vy (km/sec)","vz (km/sec)","q1","q2","q3","q4"
2 Jan 2028 00:00:00.000,-2176.938701,236.025335,6861.222353,6.526358,3.180148,1.961296,0.967192,0.202807,0.145488,0.047357
"""
        csv_path.write_text(csv_content)

        # This should not raise an error because curryer_lsk fixture loads the LSK
        df, utc_range = kernel_maker.create_jpss_kernel_dataframe_from_csv(csv_path=csv_path)

        # Verify ET values are reasonable (should be large positive numbers)
        # ET is seconds since J2000 epoch, so should be on the order of 10^8 or higher
        assert df["ADGPS_JPSS_ET"].iloc[0] > 1e8
        assert df["ADCFA_JPSS_ET"].iloc[0] > 1e8
