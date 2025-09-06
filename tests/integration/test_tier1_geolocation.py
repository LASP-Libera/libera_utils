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

# Mark test module as integration tests
pytestmark = pytest.mark.integration


def preprocess_preliminary_data(input_data_file, nominal_time_field=None, pkt_time_fields=None, kernel_identifier=None):
    """Preprocess input CSVs provided by Jake."""
    input_data_file = Path(input_data_file)

    if "Fixed_Ephemeris" in input_data_file.name:
        input_dataset = pd.read_csv(input_data_file, index_col=0)

        input_dataset = input_dataset.rename(
            columns={
                "x (km)": "ADGPSPOSX",
                "y (km)": "ADGPSPOSY",
                "z (km)": "ADGPSPOSZ",
                "vx (km/sec)": "ADGPSVELX",
                "vy (km/sec)": "ADGPSVELY",
                "vz (km/sec)": "ADGPSVELZ",
                "q1": "ADCFAQ1",
                "q2": "ADCFAQ2",
                "q3": "ADCFAQ3",
                "q4": "ADCFAQ4",
            }
        )

        for col in ["ADGPSPOSX", "ADGPSPOSY", "ADGPSPOSZ", "ADGPSVELX", "ADGPSVELY", "ADGPSVELZ"]:
            input_dataset[col] *= 1e3  # KM to meters.

        input_dataset["SPK_ET"] = spicetime.adapt(input_dataset.index.values, "iso", "et")
        input_dataset["CK_ET"] = spicetime.adapt(input_dataset.index.values, "iso", "et")

        # For this test's purpose, it doesn't matter if we use SPK or CK here.
        time_span = spicetime.adapt([input_dataset["SPK_ET"].iloc[0], input_dataset["SPK_ET"].iloc[-1]], "et", "dt64")

    elif "Libya-4_access003" in input_data_file.name:
        input_az_data = pd.read_csv(input_data_file, index_col=0)
        input_az_data = input_az_data.rename(columns={"Command Azimuth (deg)": "AZ_ANGLE"})

        az_timetags = pd.to_datetime(input_az_data.index.values)
        input_dataset = input_az_data[["AZ_ANGLE"]]
        input_dataset["AZ_ET"] = spicetime.adapt(az_timetags, "dt64", "et")

        time_span = [az_timetags[0], az_timetags[-1]]
        time_span = [time_span[0].to_pydatetime(), time_span[1].to_pydatetime()]

    elif "libera_el" in input_data_file.name:
        input_el_data = pd.read_csv(input_data_file, index_col=0)
        input_el_data = input_el_data.rename(columns={"El Angle [deg]": "EL_ANGLE"})

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
            el_values.append(input_el_data["EL_ANGLE"].values)
            el_timetags.append(prev_timetag + el_delta_sec)
            prev_timetag = el_timetags[-1][-1]

        el_timetags = pd.to_datetime(np.hstack(el_timetags))
        el_values = np.hstack(el_values)
        el_values = el_values[el_timetags <= az_timetags[-1]]
        el_timetags = el_timetags[el_timetags <= az_timetags[-1]]

        input_dataset = pd.DataFrame(
            {
                "EL_ANGLE": el_values,
                "EL_ET": spicetime.adapt(el_timetags, "dt64", "et"),
            }
        )

        time_span = [el_timetags[0], el_timetags[-1]]
        time_span = [time_span[0].to_pydatetime(), time_span[1].to_pydatetime()]

    else:
        raise ValueError(f"Unexpected test input file: {input_data_file}")

    return input_dataset, time_span


@mock.patch.object(kernel_maker, "preprocess_data", preprocess_preliminary_data)
def test_geolocate_earth_target(curryer_lsk, short_tmp_path, spice_test_data_path, test_data_path, monkeypatch):
    """Integration test for an Earth Target scenario."""

    # Force the config gets to use alternative configuration files for JPSS-4.
    monkeypatch.setenv("LIBERA_KERNEL_DIR", "{LIBERA_UTILS_DATA_DIR}/spice/jpss4")
    monkeypatch.setenv("LIBERA_KERNEL_CLOCK", "{LIBERA_KERNEL_DIR}/jpss4_v01.fakeclock.sclk.tsc")
    monkeypatch.setenv("LIBERA_KERNEL_SC_SPK_CONFIG", "{LIBERA_KERNEL_DIR}/jpss4_sc_v01.ephemeris.spk.json")
    monkeypatch.setenv("LIBERA_KERNEL_SC_CK_CONFIG", "{LIBERA_KERNEL_DIR}/jpss4_sc_v01.attitude.ck.json")

    # Generate static SPK offset kernels.
    generated_kernels = []
    for kernel_config_file in config.get("LIBERA_KERNEL_STATIC_CONFIGS"):
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(kernel_maker.make_kernel(kernel_config_file, short_tmp_path, input_data=None))

    # Generate dynamic kernels from non-standard input files.
    input_sc_file = test_data_path / "tier1_geo" / "JPSS-4_Fixed_Ephemeris_And_Attitude.csv"
    input_az_file = test_data_path / "tier1_geo" / "Libya-4_access003.csv"
    input_el_file = test_data_path / "tier1_geo" / "libera_el_cross_track_scan_profile_72deg_2021-10.csv"

    sc_spk_file = kernel_maker.from_args([input_sc_file], "JPSS-SPK", short_tmp_path)
    sc_ck_file = kernel_maker.from_args([input_sc_file], "JPSS-CK", short_tmp_path)
    az_ck_file = kernel_maker.from_args([input_az_file], "AZROT-CK", short_tmp_path)
    el_ck_file = kernel_maker.from_args([input_el_file], "ELSCAN-CK", short_tmp_path)

    generated_kernels.extend([sc_spk_file, sc_ck_file, az_ck_file, el_ck_file])

    # Load time to geolocate values onto. Email said to use 100hz cadence.
    # Offsetting AZ times by one index since EL times start at +0.001.
    az_dataset, _ = preprocess_preliminary_data(input_az_file)
    az_utc_times = spicetime.adapt(az_dataset["AZ_ET"], "et", "dt64")
    ugps_times = spicetime.adapt(pd.date_range(az_utc_times[1], az_utc_times[-2], freq="10ms", inclusive="left"), "iso")

    # Load meta kernel details.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
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
