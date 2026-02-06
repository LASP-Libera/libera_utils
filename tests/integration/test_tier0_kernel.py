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
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from curryer import meta, spicetime
from curryer import spicierpy as sp

from libera_utils import kernel_maker, time
from libera_utils.config import config
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

# Mark test module as integration tests
pytestmark = pytest.mark.integration


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


def test_make_static_kernels(noaa20_environment, curryer_lsk, short_tmp_path, spice_test_data_path):
    """Tier-0 test for creating static kernels"""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("mkspk")

    # Create the static kernels from the JSONs definitions.
    fixed_kernel_configs = config.get("LIBERA_KERNEL_STATIC_CONFIGS")
    assert len(fixed_kernel_configs) == 8

    # Set up kernel manager to furnish required kernels before creating new ones
    km = KernelManager()
    km.load_naif_kernels()
    km.ensure_known_kernels_are_furnished()

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
    noaa20_environment, curryer_lsk, noaa20_spacecraft_data, short_tmp_path, spice_test_data_path
):
    """Tier-0 test for creating spacecraft kernels"""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("mkspk")
    assert shutil.which("msopck")

    # Set up kernel manager to furnish required kernels
    km = KernelManager()
    km.load_naif_kernels()

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
    noaa20_environment, curryer_lsk, noaa20_azel_data, short_tmp_path, spice_test_data_path
):
    """Tier-0 test for creating pointing kernels"""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("msopck")

    # Set up kernel manager to furnish required kernels
    km = KernelManager()
    km.load_naif_kernels()

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

    # Load the kernels to verify the values match what we put in.
    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        az_angles = []
        for et_time in noaa20_azel_data["AXIS_SAMPLE_ICIE_ET"].values:
            tmat = sp.pxform("LIBERA_BASE_COORD", "LIBERA_AZ_COORD", et_time)
            az_angles.append(sp.m2eul(tmat, 1, 2, 3)[2])
        az_angles = np.array(az_angles) + 2 * np.pi  # Range (0, 2*pi) instead of (-pi, pi).
        npt.assert_allclose(noaa20_azel_data["ICIE__AXIS_AZ_FILT"].values, az_angles)

        el_angles = []
        for et_time in noaa20_azel_data["AXIS_SAMPLE_ICIE_ET"].values:
            tmat = sp.pxform("LIBERA_AZ_COORD", "LIBERA_EL_COORD", et_time)
            el_angles.append(sp.m2eul(tmat, 1, 2, 3)[0])
        el_angles = np.array(el_angles)
        npt.assert_allclose(noaa20_azel_data["ICIE__AXIS_EL_FILT"].values, el_angles, rtol=7e-5)
