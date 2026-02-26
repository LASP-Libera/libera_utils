"""Tier-1 geolocation test case.

Goal: The Geolocation Processing Tier 1 test builds on the Tier 0 test and will
validate that we can produce geolocation from kernels made from Libera Az and El
angles and simulated JPSS-4 position and angles using the SDC architecture.

"""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from curryer import meta, spicetime
from curryer import spicierpy as sp
from curryer.compute import spatial

from libera_utils import kernel_maker
from libera_utils.config import config
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

# Mark test module as integration tests
pytestmark = pytest.mark.integration


# TODO LIBSDC-703 This needs to be updated to match the production tools.
def preprocess_preliminary_data(input_data_file, nominal_time_field=None, pkt_time_fields=None, kernel_identifier=None):
    """Preprocess input CSVs provided by Jake."""
    input_data_file = Path(input_data_file)

    if "Fixed_Ephemeris" in input_data_file.name:
        # Spacecraft ephemeris and attitude csv containing time, position/velocity, and attitude quaternions.
        input_dataset, time_span = kernel_maker.create_jpss_kernel_dataframe_from_csv(input_data_file)

    elif "Libya-4_access003" in input_data_file.name:
        # Instrument azimuth angle csv containing time and azimuth angle values (and other fields we don't need).
        input_az_data = pd.read_csv(input_data_file, index_col=0)
        input_az_data = input_az_data.rename(columns={"Command Azimuth (deg)": "ICIE__AXIS_AZ_FILT"})

        az_timetags = pd.to_datetime(input_az_data.index.values)
        input_dataset = input_az_data[["ICIE__AXIS_AZ_FILT"]]
        input_dataset["ICIE__AXIS_AZ_FILT"] = np.deg2rad(input_dataset["ICIE__AXIS_AZ_FILT"].to_numpy())
        input_dataset["AXIS_SAMPLE_ICIE_ET"] = spicetime.adapt(az_timetags, "dt64", "et")

        time_span = [az_timetags[0], az_timetags[-1]]
        time_span = [time_span[0].to_pydatetime(), time_span[1].to_pydatetime()]

    elif "libera_el" in input_data_file.name:
        # Instrument elevation angle csv containing time and elevation mechanism angle values.
        input_el_data = pd.read_csv(input_data_file, index_col=0)
        input_el_data = input_el_data.rename(columns={"El Angle [deg]": "ICIE__AXIS_EL_FILT"})

        # NOTE: Also depends on the Az file (name is assumed!)
        input_az_file = input_data_file.parent / "Libya-4_access003.csv"
        input_az_data = pd.read_csv(input_az_file, index_col=0)

        az_timetags = pd.to_datetime(input_az_data.index.values)
        el_delta_sec = pd.to_timedelta(input_el_data.index.values, "sec")

        total_delta = az_timetags[-1] - az_timetags[0]
        sub_delta = el_delta_sec[-1] - el_delta_sec[0]
        n_loops = int(np.ceil(total_delta / sub_delta))

        el_timetags = []
        el_values = []
        prev_timetag = az_timetags[0]
        for ith in range(n_loops):
            el_values.append(input_el_data["ICIE__AXIS_EL_FILT"].values)
            el_timetags.append(prev_timetag + el_delta_sec)
            prev_timetag = el_timetags[-1][-1]

        el_timetags = pd.to_datetime(np.hstack(el_timetags))
        el_values = np.hstack(el_values)
        el_values = el_values[el_timetags <= az_timetags[-1]]
        el_values = np.deg2rad(el_values)
        el_timetags = el_timetags[el_timetags <= az_timetags[-1]]

        input_dataset = pd.DataFrame(
            {
                "ICIE__AXIS_EL_FILT": el_values,
                "AXIS_SAMPLE_ICIE_ET": spicetime.adapt(el_timetags, "dt64", "et"),
            }
        )

        time_span = [el_timetags[0], el_timetags[-1]]
        time_span = [time_span[0].to_pydatetime(), time_span[1].to_pydatetime()]

    else:
        raise ValueError(f"Unexpected test input file: {input_data_file}")

    return input_dataset, time_span


@mock.patch.object(kernel_maker.xr, "open_dataset", return_value=mock.MagicMock())
def test_geolocate_earth_target(
    mock_open_dataset,
    curryer_lsk,
    short_tmp_path,
    spice_test_data_path,
    test_data_path,
    monkeypatch,
):
    """Integration test for an Earth Target scenario."""
    # Point GENERIC_KERNEL_DIR at test data so load_static_kernels() can find sds_kernels.
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    # Generate static SPK offset kernels.
    generated_kernels = []
    for kernel_config_file in config.get("LIBERA_KERNEL_STATIC_CONFIGS"):
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=None))

    # Generate dynamic kernels from non-standard input files.
    input_sc_file = test_data_path / "tier1_geo" / "JPSS-4_Fixed_Ephemeris_And_Attitude.csv"
    input_az_file = test_data_path / "tier1_geo" / "Libya-4_access003.csv"
    input_el_file = test_data_path / "tier1_geo" / "libera_el_cross_track_scan_profile_72deg_2021-10.csv"

    # Generate Curryer-friendly kernel dataframes from the preliminary CSV data
    mock_sc_kernel_dataframe = preprocess_preliminary_data(input_sc_file)
    mock_az_kernel_dataframe = preprocess_preliminary_data(input_az_file)
    mock_el_kernel_dataframe = preprocess_preliminary_data(input_el_file)

    with mock.patch("libera_utils.kernel_maker.create_kernel_dataframe_from_l1a") as mock_create_l1a:
        # Mock create_kernel_dataframe_from_l1a to return the dataframes from preliminary CSV data
        mock_create_l1a.return_value = mock_sc_kernel_dataframe
        sc_spk_file = kernel_maker.create_kernel_from_l1a(input_sc_file, "JPSS-SPK", short_tmp_path)
        sc_ck_file = kernel_maker.create_kernel_from_l1a(input_sc_file, "JPSS-CK", short_tmp_path)

        mock_create_l1a.return_value = mock_az_kernel_dataframe
        az_ck_file = kernel_maker.create_kernel_from_l1a(input_az_file, "AZROT-CK", short_tmp_path)

        mock_create_l1a.return_value = mock_el_kernel_dataframe
        el_ck_file = kernel_maker.create_kernel_from_l1a(input_el_file, "ELSCAN-CK", short_tmp_path)

    generated_kernels.extend([sc_spk_file, sc_ck_file, az_ck_file, el_ck_file])

    # Load time to geolocate values onto. Email said to use 100hz cadence.
    # Offsetting AZ times by one index since EL times start at +0.001.
    az_dataset, _ = preprocess_preliminary_data(input_az_file)
    az_utc_times = spicetime.adapt(az_dataset["AXIS_SAMPLE_ICIE_ET"], "et", "dt64")
    ugps_times = spicetime.adapt(pd.date_range(az_utc_times[1], az_utc_times[-2], freq="10ms", inclusive="left"), "iso")

    # Load meta kernel details that is defined from the libera_utils package not the test data.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=config.get("GENERIC_KERNEL_DIR"),
    )

    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
        # Geolocate to the ellipsoid.
        ellips_lla_df, sc_xyz_df, ellips_qf_ds = spatial.instrument_intersect_ellipsoid(
            ugps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), geodetic=True, degrees=True
        )

        qf_counts = ellips_qf_ds.value_counts()
        per_good = qf_counts[0] / qf_counts.sum()
        assert per_good >= 0.78, per_good

        slc = slice(*spicetime.adapt(["2028-01-02 00:21:22", "2028-01-02 00:21:36"], "iso"))
        print("Mean Lat: ", np.nanmean(ellips_lla_df.loc[slc]["lat"]))
        print("Mean Lon: ", np.nanmean(ellips_lla_df.loc[slc]["lon"]))

        clon, clat = 23.39, 28.55
        hist, xedge, yedge = np.histogram2d(
            ellips_lla_df.loc[slc]["lon"],
            ellips_lla_df.loc[slc]["lat"],
            bins=(13, 13),
            range=[[clon - 3, clon + 3], [clat - 3, clat + 3]],
        )
        idx = np.where(hist == hist.max())
        ix, iy = idx[0][0], idx[1][0]

        print(f"Expected focus point:    lon=[{clon}],          lat=[{clat}]")
        print(
            f"2D histogram max between lon=[{xedge[ix]:.3f}, {xedge[ix + 1]:.3f}],"
            f" lat=[{yedge[iy]:.3f}, {yedge[iy + 1]:.3f}]"
        )
        assert ix == hist.shape[0] // 2, ix
        assert iy == hist.shape[0] // 2, iy
