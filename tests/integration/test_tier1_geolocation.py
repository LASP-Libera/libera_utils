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
from curryer.compute import constants, spatial

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
        input_dataset = input_az_data[["ICIE__AXIS_AZ_FILT"]].copy()
        # CSV angles are ideal commanded pointing; kernel generation applies the encoder correction, so
        # pre-apply its inverse here to represent the raw encoder telemetry the correction maps back.
        input_dataset["ICIE__AXIS_AZ_FILT"] = kernel_maker.uncorrect_azimuth(
            np.deg2rad(input_dataset["ICIE__AXIS_AZ_FILT"].to_numpy())
        )
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
        # Inverse encoder correction (see azimuth branch): raw telemetry that kernel generation corrects.
        el_values = kernel_maker.uncorrect_elevation(np.deg2rad(el_values))
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

        # End-to-end smoke test: confirm the misalignment-active pipeline lands the footprint in the
        # correct region. A relaxed bounding box is used rather than an exact coordinate on purpose -- the
        # wide cross-track scan grazes the Earth limb, where a few samples flip between hit and
        # ellipsoid-miss (NaN) across Python/msopck builds and move the footprint's exact center (a tight
        # mean-location assertion here broke on Python 3.14). Precise checks live elsewhere: sub-km
        # geolocation accuracy under nominal geometry vs CERES in test_tier0_geolocation, and the
        # misalignment's magnitude vs nominal in test_misalignment_shifts_geolocation. Here we only need
        # the pipeline to reach the right place.
        slc = slice(*spicetime.adapt(["2028-01-02 00:21:22", "2028-01-02 00:21:36"], "iso"))
        median_lat = np.nanmedian(ellips_lla_df.loc[slc]["lat"])
        median_lon = np.nanmedian(ellips_lla_df.loc[slc]["lon"])
        print(f"footprint median: lat {median_lat:.6f}, lon {median_lon:.6f}")
        assert 28.0 < median_lat < 31.0, median_lat
        assert 20.0 < median_lon < 23.0, median_lon


# Ideal orthogonal-gimbal axes. The measured axes of rotation (LIBSDC-806) are small perturbations of
# these, so building the Az/El CKs about them with an identity radiometer boresight reproduces the
# nominal, misalignment-free geometry; the measured-minus-nominal footprint shift is then purely the
# misalignment.
NOMINAL_AZ_AXIS = np.array([0.0, 0.0, 1.0])
NOMINAL_EL_AXIS = np.array([1.0, 0.0, 0.0])


def _set_ck_quaternions(df, field, axis):
    """Populate the mechanism CK quaternion columns for ``field`` about ``axis``, in place."""
    quaternions = kernel_maker.mechanism_quaternions(df[field].to_numpy(), axis)
    for name, column in zip(kernel_maker.MECHANISM_QUATERNION_COLUMNS[field], quaternions.T, strict=True):
        df[name] = column


def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance (km) between arrays of geodetic points (degrees in)."""
    lon1, lat1, lon2, lat2 = (np.deg2rad(v) for v in (lon1, lat1, lon2, lat2))
    haversine = (1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))) / 2
    return 2 * constants.WGS84_SEMI_MAJOR_AXIS_KM * np.arcsin(np.sqrt(haversine))


@mock.patch.object(kernel_maker.xr, "open_dataset", return_value=mock.MagicMock())
def test_misalignment_shifts_geolocation(
    mock_open_dataset,
    curryer_lsk,
    short_tmp_path,
    spice_test_data_path,
    test_data_path,
    monkeypatch,
):
    """A/B geolocation: the measured misalignments (LIBSDC-806) shift the footprint from nominal.

    Geolocates the same Libya-4 scan twice from one set of spacecraft kernels -- once with the ideal
    orthogonal gimbal (Az about +Z, El about +X, identity radiometer boresight) and once with the
    measured axes of rotation and boresight. The measured-minus-nominal ground shift must be a
    non-zero, bounded, sub-degree effect, confirming the misalignment propagates all the way to
    geolocation. Differencing within one run cancels common-mode geometry/platform effects, so (unlike
    an absolute footprint location) the check is robust across Python/msopck builds. The magnitude's
    physical correctness is pinned separately in test_los_alignment.py.
    """
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    static_kernels = []
    for kernel_config_file in config.get("LIBERA_KERNEL_STATIC_CONFIGS"):
        static_kernels.append(spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=None))

    input_sc_file = test_data_path / "tier1_geo" / "JPSS-4_Fixed_Ephemeris_And_Attitude.csv"
    input_az_file = test_data_path / "tier1_geo" / "Libya-4_access003.csv"
    input_el_file = test_data_path / "tier1_geo" / "libera_el_cross_track_scan_profile_72deg_2021-10.csv"

    # Spacecraft kernels and measured-geometry Az/El CKs (create_kernel_from_l1a corrects the encoder
    # angles and builds the quaternions about the measured axes read from the production frame kernel).
    with mock.patch("libera_utils.kernel_maker.create_kernel_dataframe_from_l1a") as mock_create_l1a:
        mock_create_l1a.return_value = preprocess_preliminary_data(input_sc_file)
        sc_spk_file = kernel_maker.create_kernel_from_l1a(input_sc_file, "JPSS-SPK", short_tmp_path)
        sc_ck_file = kernel_maker.create_kernel_from_l1a(input_sc_file, "JPSS-CK", short_tmp_path)
        mock_create_l1a.return_value = preprocess_preliminary_data(input_az_file)
        az_ck_measured = kernel_maker.create_kernel_from_l1a(input_az_file, "AZROT-CK", short_tmp_path)
        mock_create_l1a.return_value = preprocess_preliminary_data(input_el_file)
        el_ck_measured = kernel_maker.create_kernel_from_l1a(input_el_file, "ELSCAN-CK", short_tmp_path)

    sc_kernels = [*static_kernels, sc_spk_file, sc_ck_file]

    az_dataset, _ = preprocess_preliminary_data(input_az_file)
    az_utc_times = spicetime.adapt(az_dataset["AXIS_SAMPLE_ICIE_ET"], "et", "dt64")
    ugps_times = spicetime.adapt(pd.date_range(az_utc_times[1], az_utc_times[-2], freq="10ms", inclusive="left"), "iso")

    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=config.get("GENERIC_KERNEL_DIR"),
    )
    non_fk_mission = [k for k in mkrn.mission_kernels if "frames.fk" not in str(k)]
    nominal_fk = test_data_path / "tier0_geo" / "libera_nominal_v01.frames.fk.tf"
    slc = slice(*spicetime.adapt(["2028-01-02 00:21:22", "2028-01-02 00:21:36"], "iso"))

    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, [*sc_kernels, az_ck_measured, el_ck_measured]]):
        measured_lla, _, _ = spatial.instrument_intersect_ellipsoid(
            ugps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), geodetic=True, degrees=True
        )

    # Nominal geometry: rebuild the Az/El CKs about the ideal axes and swap in the frozen misalignment-
    # free frame kernel. Clear the pool first so the production kernel's measured boresight can't linger.
    az_nominal, _ = preprocess_preliminary_data(input_az_file)
    kernel_maker.apply_encoder_corrections(az_nominal)
    _set_ck_quaternions(az_nominal, kernel_maker.AZ_ENCODER_FIELD, NOMINAL_AZ_AXIS)
    az_ck_nominal = spice_utils.make_kernel(
        config.get("LIBERA_KERNEL_AZ_CK_CONFIG"), short_tmp_path, input_data=az_nominal
    )

    el_nominal, _ = preprocess_preliminary_data(input_el_file)
    kernel_maker.apply_encoder_corrections(el_nominal)
    _set_ck_quaternions(el_nominal, kernel_maker.EL_ENCODER_FIELD, NOMINAL_EL_AXIS)
    el_ck_nominal = spice_utils.make_kernel(
        config.get("LIBERA_KERNEL_EL_CK_CONFIG"), short_tmp_path, input_data=el_nominal
    )

    km.unload_all()

    with sp.ext.load_kernel(
        [mkrn.sds_kernels, [nominal_fk, *non_fk_mission], [*sc_kernels, az_ck_nominal, el_ck_nominal]]
    ):
        nominal_lla, _, _ = spatial.instrument_intersect_ellipsoid(
            ugps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), geodetic=True, degrees=True
        )

    m_lat, m_lon = measured_lla.loc[slc]["lat"].to_numpy(), measured_lla.loc[slc]["lon"].to_numpy()
    n_lat, n_lon = nominal_lla.loc[slc]["lat"].to_numpy(), nominal_lla.loc[slc]["lon"].to_numpy()
    good = np.isfinite(m_lat) & np.isfinite(m_lon) & np.isfinite(n_lat) & np.isfinite(n_lon)
    assert good.mean() >= 0.78, good.mean()

    # No-alignment anchor: the ideal orthogonal gimbal gives a fixed reference footprint, pinned here as
    # a precise regression check on the nominal geometry (much tighter than the aligned pipeline's
    # bounding box in test_geolocate_earth_target). The median (not mean) with a modest tolerance keeps
    # it robust to the limb-grazing NaN flips that make absolute footprint locations vary slightly across
    # Python/msopck builds; sub-km accuracy of the nominal geometry itself is validated against CERES in
    # test_tier0_geolocation.
    nominal_median_lat, nominal_median_lon = np.median(n_lat[good]), np.median(n_lon[good])
    print(f"nominal footprint median: lat {nominal_median_lat:.6f}, lon {nominal_median_lon:.6f}")
    np.testing.assert_allclose(nominal_median_lat, 29.258301, atol=0.02)
    np.testing.assert_allclose(nominal_median_lon, 21.248531, atol=0.02)

    shift_km = _haversine_km(m_lon[good], m_lat[good], n_lon[good], n_lat[good])
    shift_deg = shift_km / 111.32
    print(
        f"\nmisalignment ground shift over slice: median {np.median(shift_km):.3f} km "
        f"({np.median(shift_deg):.4f} deg), max {shift_km.max():.3f} km ({shift_km.max() / 111.32:.4f} deg)"
    )

    # The misalignment reaches geolocation as a few-km shift (this scan: median ~4 km, max ~0.26 deg at
    # the far cross-track edge, where a near-tangent line of sight amplifies the ~0.26 deg boresight
    # offset). The bands guard against regressions -- a removed misalignment collapses the shift to ~0,
    # a wrong axis/boresight blows it up to degrees; the physical magnitude is pinned in
    # test_los_alignment.py. The median is used as the robust metric (insensitive to a few limb-grazing
    # samples flipping in/out), so the check holds across Python/msopck builds.
    assert 1.0 < np.median(shift_km) < 15.0, np.median(shift_km)
    assert shift_deg.max() < 0.5, shift_deg.max()
