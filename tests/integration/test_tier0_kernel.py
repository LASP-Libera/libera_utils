"""Tier-0 kernel creation test case.

Goal: Give known Az/El angles from CERES and SC attitude and ephemeris, generate
kernels with call to curryer to produce expected output kernels. Don't use a
manifest file (in or out) because input isn't packets. Validate against CERES
geolocation for precision tolerance.

Preparation:
    - Since this test depends on non-Libera input data (JPSS-1 (NOAA-20) &
    CERES), the data was pre-processed into easy to read / load files (CSVs)
    - The spacecraft (NOAA-20) ephemeris and attitude telemetry were decoded
    from raw packets using jpss1_geolocation_xtce_v1.xml and
    J01_G011_LZ_2021-04-09T*_V01.DAT1 files
        - For the unit test, a 2-minute chunk was extracted and saved to a CSV,
        checked into the repo
    - The azimuth and elevation angles were taken from the CERES public data
    product: CER_BDS_NOAA20-FM6_Edition1_100111.20210409.hdf
        - For the unit test, a 2-minute chunk was extracted and saved to a CSV,
        checked into the repo

"""

import shutil
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr
from curryer import meta, spicetime
from curryer import spicierpy as sp

from libera_utils import kernel_maker, time
from libera_utils.config import config
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

# Mark test module as integration tests
pytestmark = pytest.mark.integration


def _rotation(axis, angle):
    """Rotation matrix for ``angle`` radians about unit ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos, sin = np.cos(angle), np.sin(angle)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * cos + np.outer(axis, axis) * (1 - cos) + skew * sin


def _angle_about(rotation, axis):
    """Signed rotation angle (radians) of ``rotation`` about unit ``axis``."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    vee = 0.5 * np.array(
        [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]
    )
    return np.arctan2(vee @ axis, 0.5 * (np.trace(rotation) - 1.0))


@pytest.fixture
def noaa20_spacecraft_data(test_data_path):
    """Load NOAA-20 spacecraft test data."""
    input_sc_file = test_data_path / "tier0_kernel" / "J01_G011_LZ_2021-04-09.2min.csv"
    input_sc_data = pd.read_csv(input_sc_file, index_col=0)

    spk_dt64 = time.multipart_to_dt64(input_sc_data, "ADAET1DAY", "ADAET1MS", "ADAET1US")
    ck_dt64 = time.multipart_to_dt64(input_sc_data, "ADAET2DAY", "ADAET2MS", "ADAET2US")
    input_sc_data["ADGPS_JPSS_ET"] = spicetime.adapt(spk_dt64.values, "dt64", "et")
    input_sc_data["ADCFA_JPSS_ET"] = spicetime.adapt(ck_dt64.values, "dt64", "et")

    return input_sc_data


@pytest.fixture
def noaa20_azel_data(test_data_path):
    """Load NOAA-20 Az/El test data."""
    input_azel_file = test_data_path / "tier0_kernel" / "CER_BDS_NOAA20-FM6_Edition1_100111.20210409.azel.2min.csv"
    input_azel_data = pd.read_csv(input_azel_file, index_col=0)

    # Pointing timing has an offset relative to S/C timing (unknown why).
    input_azel_data["AZ_ET"] += 0.024905
    input_azel_data["EL_ET"] += 0.024905
    # Expect the times for both Az and El sampling to be the same (as they are in Libera AXIS_SAMPLE packets)
    assert all(input_azel_data["AZ_ET"].to_numpy() == input_azel_data["EL_ET"].to_numpy())
    # Drop one column and rename the other
    input_azel_data.drop(columns=["EL_ET"])
    # AXIS_SAMPLE_ICIE_ET is the time column used for generating both Az and El CKs
    input_azel_data.rename(
        columns={"AZ_ANGLE": "ICIE__AXIS_AZ_FILT", "EL_ANGLE": "ICIE__AXIS_EL_FILT", "AZ_ET": "AXIS_SAMPLE_ICIE_ET"},
        inplace=True,
    )
    # Libera axis samples come in as radians so convert for consistency
    input_azel_data["ICIE__AXIS_AZ_FILT"] = np.deg2rad(input_azel_data["ICIE__AXIS_AZ_FILT"])
    input_azel_data["ICIE__AXIS_EL_FILT"] = np.deg2rad(input_azel_data["ICIE__AXIS_EL_FILT"])

    return input_azel_data


def test_make_static_kernels(noaa20_environment, curryer_lsk, short_tmp_path, spice_test_data_path, monkeypatch):
    """Tier-0 test for creating static kernels"""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("mkspk")

    # Point GENERIC_KERNEL_DIR at test data so load_static_kernels() can find sds_kernels.
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    # Create the static kernels from the JSONs definitions.
    fixed_kernel_configs = config.get("LIBERA_KERNEL_STATIC_CONFIGS")
    assert len(fixed_kernel_configs) == 8

    generated_kernels = []
    for kernel_config_file in config.get("LIBERA_KERNEL_STATIC_CONFIGS"):
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=None))

    found_kernels = sorted(short_tmp_path.glob("*"))
    assert len(found_kernels) == 8

    # Load meta kernel details.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Assert that the expected kernel file exists, contains the correct SPICE
    # object and correct time coverage.
    static_pairings = [
        ("LIBERA_BASE", "libera_base_v01.fixed_offset.spk.bsp"),
        ("LIBERA_AZ", "libera_az_v01.fixed_offset.spk.bsp"),
        ("LIBERA_WFOV_CAM", "libera_wfov_cam_v01.fixed_offset.spk.bsp"),
        ("LIBERA_EL", "libera_el_v01.fixed_offset.spk.bsp"),
        ("LIBERA_SW_RAD", "libera_sw_rad_v01.fixed_offset.spk.bsp"),
        ("LIBERA_SSW_RAD", "libera_ssw_rad_v01.fixed_offset.spk.bsp"),
        ("LIBERA_LW_RAD", "libera_lw_rad_v01.fixed_offset.spk.bsp"),
        ("LIBERA_TOT_RAD", "libera_tot_rad_v01.fixed_offset.spk.bsp"),
    ]
    for obj_key, kernel_file in static_pairings:
        span = sp.ext.kernel_coverage(short_tmp_path / kernel_file, mkrn.mappings[obj_key], to_fmt="iso")
        assert span == ("1980-01-06 00:00:00.000000", "2080-01-06 00:00:00.000000")

    # Assert that there's no spatial offset within the spacecraft elements.
    ugps_time = spicetime.adapt("2025-01-01", "iso", "ugps")
    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        static_elements = [
            ("NOAA20_SC", "LIBERA_AZ"),
            ("LIBERA_AZ", "LIBERA_WFOV_CAM"),
            ("LIBERA_AZ", "LIBERA_EL"),
            ("LIBERA_EL", "LIBERA_SW_RAD"),
            ("LIBERA_EL", "LIBERA_SSW_RAD"),
            ("LIBERA_EL", "LIBERA_LW_RAD"),
            ("LIBERA_EL", "LIBERA_TOT_RAD"),
        ]
        for from_obj, to_obj in static_elements:
            xyz = sp.ext.query_ephemeris([ugps_time], from_obj, to_obj, ref_frame=f"{from_obj}_COORD")
            assert (xyz.values == 0).all(), (from_obj, to_obj)


def test_make_spacecraft_kernels(
    noaa20_environment,
    curryer_lsk,
    noaa20_spacecraft_data,
    short_tmp_path,
    spice_test_data_path,
    monkeypatch,
):
    """Tier-0 test for creating spacecraft kernels"""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("mkspk")
    assert shutil.which("msopck")

    # Point GENERIC_KERNEL_DIR at test data so load_static_kernels() can find sds_kernels.
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    # Create the dynamic kernel from the JSONs definition and given data.
    generated_kernels = []
    for kernel_config_file in [config.get("LIBERA_KERNEL_SC_SPK_CONFIG"), config.get("LIBERA_KERNEL_SC_CK_CONFIG")]:
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(
            spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=noaa20_spacecraft_data)
        )
    assert len(sorted(short_tmp_path.glob("*"))) == 2

    # Load meta kernel details. Includes existing static kernels.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Assert that the expected kernel file exists, contains the correct SPICE
    # object and correct time coverage.
    span = sp.ext.kernel_coverage(
        short_tmp_path / "noaa20_sc_v01.ephemeris.spk.bsp", mkrn.mappings["NOAA20_SC"], to_fmt="iso"
    )
    assert span == ("2021-04-09 12:00:06.030922", "2021-04-09 12:02:05.030923")

    # Clock kernel must be loaded to inspect CK kernels.
    with sp.ext.load_kernel(config.get("LIBERA_KERNEL_CLOCK")):
        span = sp.ext.kernel_coverage(
            short_tmp_path / "noaa20_sc_v01.attitude.ck.bc", mkrn.mappings["NOAA20_SC"], to_fmt="iso"
        )
        assert span == ("2021-04-09 12:00:05.930922", "2021-04-09 12:02:04.930923")

    # Load the kernels to verify the values match what we put in.
    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        # Position of the SC within ECEF.
        ugps_times = spicetime.adapt(noaa20_spacecraft_data["ADGPS_JPSS_ET"], "et")
        pos_data = sp.ext.query_ephemeris(ugps_times, "NOAA20_SC", "EARTH", ref_frame="ITRF93", velocity=True)
        pos_data = pos_data.values * 1e3
        exp_data = noaa20_spacecraft_data[
            ["ADGPSPOSX", "ADGPSPOSY", "ADGPSPOSZ", "ADGPSVELX", "ADGPSVELY", "ADGPSVELZ"]
        ].values
        npt.assert_allclose(exp_data, pos_data)

        # Note that the kernel definition forces the input quat signs to be
        # flipped, hence why we query for a rotation from the SC to the Earth.
        rot_data = []
        for et_time in noaa20_spacecraft_data["ADCFA_JPSS_ET"]:
            tmat = sp.pxform("NOAA20_SC_COORD", "J2000", et_time)
            rot_data.append(sp.m2q(tmat))
        rot_data = np.array(rot_data)
        exp_data = noaa20_spacecraft_data[
            [
                "ADCFAQ4",
                "ADCFAQ1",
                "ADCFAQ2",
                "ADCFAQ3",
            ]
        ].values  # Format: CXYZ
        npt.assert_allclose(exp_data, rot_data)


def test_make_spacecraft_azel_kernels(
    noaa20_environment,
    curryer_lsk,
    noaa20_azel_data,
    short_tmp_path,
    spice_test_data_path,
    monkeypatch,
):
    """Tier-0 test for creating pointing kernels"""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("msopck")

    # Point GENERIC_KERNEL_DIR at test data so load_static_kernels() can find sds_kernels.
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    # Az/El CKs are quaternion-valued; build the mechanism quaternions (about the measured axes) from
    # the raw telemetered angles for the direct make_kernel calls below.
    kernel_maker.add_mechanism_ck_quaternions(noaa20_azel_data)

    # Create the dynamic kernel from the JSONs definition and given data.
    generated_kernels = []
    for kernel_config_file in [config.get("LIBERA_KERNEL_AZ_CK_CONFIG"), config.get("LIBERA_KERNEL_EL_CK_CONFIG")]:
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(
            spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=noaa20_azel_data)
        )
    assert len(sorted(short_tmp_path.glob("*"))) == 2

    # Load meta kernel details. Includes existing static kernels.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Assert that the expected kernel file exists, contains the correct SPICE
    # object and correct time coverage.
    # Clock kernel must be loaded to inspect CK kernels.
    with sp.ext.load_kernel(config.get("LIBERA_KERNEL_CLOCK")):
        span = sp.ext.kernel_coverage(
            short_tmp_path / "libera_az_v01.attitude.ck.bc", mkrn.mappings["LIBERA_AZ"], to_fmt="iso"
        )
        assert span == ("2021-04-09 12:00:06.173775", "2021-04-09 12:02:04.964132")
        span = sp.ext.kernel_coverage(
            short_tmp_path / "libera_el_v01.attitude.ck.bc", mkrn.mappings["LIBERA_EL"], to_fmt="iso"
        )
        assert span == ("2021-04-09 12:00:06.173775", "2021-04-09 12:02:04.964132")

    # Load the kernels to verify each CK encodes rotation about the measured axis by the input angle.
    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        az_axis = kernel_maker._read_alignment_axis("LIBERA_AZ_AOR_IN_STAND")
        el_axis = kernel_maker._read_alignment_axis("LIBERA_EL_AOR_IN_STAND")
        for et_time, az, el in zip(
            noaa20_azel_data["AXIS_SAMPLE_ICIE_ET"].values,
            noaa20_azel_data["ICIE__AXIS_AZ_FILT"].values,
            noaa20_azel_data["ICIE__AXIS_EL_FILT"].values,
        ):
            npt.assert_allclose(
                sp.pxform("LIBERA_AZ_COORD", "LIBERA_BASE_COORD", et_time), _rotation(az_axis, az), atol=1e-4
            )
            npt.assert_allclose(
                sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", et_time), _rotation(el_axis, el), atol=1e-4
            )


def test_make_spacecraft_azel_kernels_apply_encoder_correction(
    noaa20_environment,
    curryer_lsk,
    noaa20_azel_data,
    short_tmp_path,
    spice_test_data_path,
    monkeypatch,
):
    """Az/El CKs built through create_kernel_from_l1a recover telemetry + correction (LIBSDC-668).

    Drives the production create_kernel_from_l1a path with raw CERES angles (mocking only the L1A read)
    so the deterministic encoder correction is applied during kernel generation, then queries the CKs
    and checks the recovered mechanism angle equals correct(raw), not the raw telemetry.
    """
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("msopck")
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))

    raw_az = noaa20_azel_data["ICIE__AXIS_AZ_FILT"].to_numpy()
    raw_el = noaa20_azel_data["ICIE__AXIS_EL_FILT"].to_numpy()
    et_times = noaa20_azel_data["AXIS_SAMPLE_ICIE_ET"].to_numpy()
    utc_range = (datetime.fromisoformat("2021-04-09T12:00:06"), datetime.fromisoformat("2021-04-09T12:02:05"))

    # Build the CKs via the production L1A path. Each call gets a fresh copy of the raw data so the
    # in-place correction is applied exactly once per kernel.
    generated_kernels = []
    with mock.patch(
        "libera_utils.kernel_maker.create_kernel_dataframe_from_l1a",
        side_effect=lambda *a, **k: (noaa20_azel_data.copy(), utc_range),
    ):
        for dpi in ("AZROT-CK", "ELSCAN-CK"):
            generated_kernels.append(
                kernel_maker.create_kernel_from_l1a(xr.Dataset(), dpi, short_tmp_path, overwrite=True)
            )

    mkrn = meta.MetaKernel.from_json(config.get("LIBERA_KERNEL_META"), relative=True, sds_dir=spice_test_data_path)

    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        az_axis = kernel_maker._read_alignment_axis("LIBERA_AZ_AOR_IN_STAND")
        el_axis = kernel_maker._read_alignment_axis("LIBERA_EL_AOR_IN_STAND")

        # Each CK encodes rotation about the mechanism's measured axis by the corrected angle (telemetry
        # + correction), not the raw telemetry.
        for et, corrected_az in zip(et_times, kernel_maker.correct_azimuth(raw_az)):
            npt.assert_allclose(
                sp.pxform("LIBERA_AZ_COORD", "LIBERA_BASE_COORD", et), _rotation(az_axis, corrected_az), atol=1e-4
            )
        for et, corrected_el in zip(et_times, kernel_maker.correct_elevation(raw_el)):
            npt.assert_allclose(
                sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", et), _rotation(el_axis, corrected_el), atol=1e-4
            )

        # The correction is genuinely present (elevation amplitude ~4.6e-4 rad, well above CK round-trip noise).
        el_recovered = np.array(
            [_angle_about(sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", et), el_axis) for et in et_times]
        )
        assert np.max(np.abs((el_recovered - raw_el + np.pi) % (2 * np.pi) - np.pi)) > 1e-4

        # Between telemetered samples the CK interpolates the corrected angles.
        mid_et = 0.5 * (et_times[0] + et_times[1])
        el_mid = _angle_about(sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", mid_et), el_axis)
        lo, hi = sorted(kernel_maker.correct_elevation(raw_el[:2]))
        assert lo - 1e-5 <= el_mid <= hi + 1e-5
