"""SPICE kernel generation, from telemetry through to the production entry points.

Test naming
-----------
Two concerns live here, told apart by what the test name names:

``test_<kernel>_...``
    What a generated kernel *contains* -- coverage spans, recovered rotation angles, quaternion
    conventions. These descend from the internal validation described below.
``test_<function>_...``
    What a production entry point *does* -- ``create_kernel_from_packets``,
    ``create_kernels_from_manifest``, ``create_kernel_from_l1a`` -- including local and S3 I/O
    and manifest handling. These are plumbing tests and assert on outputs existing and being
    named correctly, not on kernel contents.

Validation provenance
---------------------
These are the tests that satisfied the **Tier-0 kernel creation** internal validation. The
"Tier-0" label is historical and is being retired (LIBSDC-703); the tests remain the record of
that validation and must not be weakened without re-running it.

What it establishes: given known Az/El angles from CERES plus NOAA-20 spacecraft attitude and
ephemeris, the curryer-backed kernel writers produce SPKs and CKs whose contents read back as the
values that went in. No manifest is involved because the input is CSV, not packets.

This is also the only coverage of kernel generation from *real decoded spacecraft telemetry*
(``ADGPSPOS*``, ``ADCFAQ*``, and multipart ``ADAET1DAY/MS/US`` times via
:func:`libera_utils.time.multipart_to_dt64`) and of the quaternion sign-flip convention in the
spacecraft CK config. The JPSS-4 path uses simulated ephemeris and covers neither.

Source data
-----------
Both inputs are non-Libera and were pre-processed into CSVs checked into the repo, each a
2-minute extract:

- Spacecraft ephemeris and attitude decoded from ``J01_G011_LZ_2021-04-09T*_V01.DAT1`` using
  ``jpss1_geolocation_xtce_v1.xml``.
- Azimuth and elevation angles from the CERES public product
  ``CER_BDS_NOAA20-FM6_Edition1_100111.20210409.hdf``.
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
from cloudpathlib import S3Path
from curryer import meta, spicetime
from curryer import spicierpy as sp

from libera_utils import kernel_maker, time
from libera_utils.config import config
from libera_utils.io.manifest import Manifest
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager
from tests.helpers import angle_about, rotation

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


def test_static_offset_kernels_span_the_mission_with_zero_offsets(
    curryer_lsk, short_tmp_path, spice_test_data_path, monkeypatch
):
    """Each configured static offset kernel is written, spans the mission, and carries no offset.

    Runs on the default (jpss4) configuration, which is what production ships.
    """
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("mkspk")

    # Point GENERIC_KERNEL_DIR at test data so load_static_kernels() can find sds_kernels.
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    # Create the static kernels from the JSON definitions. Counts are derived from the config
    # rather than pinned: the set of structural elements changes with the frame layout (it went
    # from four radiometer channels to one in LIBSDC-815), and a hardcoded count turns that into a
    # spurious failure while proving nothing about the kernels themselves.
    fixed_kernel_configs = config.get("LIBERA_KERNEL_STATIC_CONFIGS")
    assert fixed_kernel_configs, "No static kernel configs are defined"

    generated_kernels = []
    for kernel_config_file in fixed_kernel_configs:
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=None))

    found_kernels = sorted(short_tmp_path.glob("*"))
    assert len(found_kernels) == len(fixed_kernel_configs)

    # Load meta kernel details.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Assert that the expected kernel file exists, contains the correct SPICE
    # object and correct time coverage.
    static_pairings = [
        ("LIBERA_BASE", "libera_base.fixed_offset.spk.bsp"),
        ("LIBERA_AZ", "libera_az.fixed_offset.spk.bsp"),
        ("LIBERA_WFOV_CAM", "libera_wfov_cam.fixed_offset.spk.bsp"),
        ("LIBERA_EL", "libera_el.fixed_offset.spk.bsp"),
        ("LIBERA_RAD", "libera_rad.fixed_offset.spk.bsp"),
    ]
    configured_basenames = {Path(c).name.replace(".json", ".bsp") for c in fixed_kernel_configs}
    assert {kernel_file for _, kernel_file in static_pairings} == configured_basenames, (
        "static_pairings has drifted from LIBERA_KERNEL_STATIC_CONFIGS"
    )

    for obj_key, kernel_file in static_pairings:
        span = sp.ext.kernel_coverage(short_tmp_path / kernel_file, mkrn.mappings[obj_key], to_fmt="iso")
        assert span == ("1980-01-06 00:00:00.000000", "2080-01-06 00:00:00.000000")

    # Assert that there's no spatial offset within the spacecraft elements.
    ugps_time = spicetime.adapt("2025-01-01", "iso", "ugps")
    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        static_elements = [
            ("JPSS4_SC", "LIBERA_AZ"),
            ("LIBERA_AZ", "LIBERA_WFOV_CAM"),
            ("LIBERA_AZ", "LIBERA_EL"),
            ("LIBERA_EL", "LIBERA_RAD"),
        ]
        for from_obj, to_obj in static_elements:
            xyz = sp.ext.query_ephemeris([ugps_time], from_obj, to_obj, ref_frame=f"{from_obj}_COORD")
            assert (xyz.values == 0).all(), (from_obj, to_obj)


def test_spacecraft_kernels_round_trip_the_input_state(
    noaa20_environment,
    curryer_lsk,
    noaa20_spacecraft_data,
    short_tmp_path,
    spice_test_data_path,
):
    """The spacecraft SPK and CK read back the position, velocity and attitude that went in."""
    assert not sorted(short_tmp_path.glob("*"))
    assert shutil.which("mkspk")
    assert shutil.which("msopck")

    # Reading the metakernel registers the NOAA20_SC name-to-ID mapping the kernel configs are
    # written against, so it has to happen before generation.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Create the dynamic kernel from the JSONs definition and given data.
    generated_kernels = []
    for kernel_config_file in [config.get("LIBERA_KERNEL_SC_SPK_CONFIG"), config.get("LIBERA_KERNEL_SC_CK_CONFIG")]:
        assert Path(kernel_config_file).is_file(), kernel_config_file
        generated_kernels.append(
            spice_utils.make_kernel(kernel_config_file, short_tmp_path, input_data=noaa20_spacecraft_data)
        )
    assert len(sorted(short_tmp_path.glob("*"))) == 2

    # Assert that the expected kernel file exists, contains the correct SPICE
    # object and correct time coverage.
    span = sp.ext.kernel_coverage(
        short_tmp_path / "noaa20_sc.ephemeris.spk.bsp", mkrn.mappings["NOAA20_SC"], to_fmt="iso"
    )
    assert span == ("2021-04-09 12:00:06.030922", "2021-04-09 12:02:05.030923")

    # Clock kernel must be loaded to inspect CK kernels.
    with sp.ext.load_kernel(config.get("LIBERA_KERNEL_CLOCK")):
        span = sp.ext.kernel_coverage(
            short_tmp_path / "noaa20_sc.attitude.ck.bc", mkrn.mappings["NOAA20_SC"], to_fmt="iso"
        )
        assert span == ("2021-04-09 12:00:05.930922", "2021-04-09 12:02:04.930923")

    # Load the kernels to verify the values match what we put in. The metakernel's sds_kernels supply
    # the Earth orientation data the ITRF93 query needs; no static Libera offset kernels are involved,
    # since nothing here queries a position relative to an instrument frame.
    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, generated_kernels]):
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


def test_mechanism_cks_rotate_about_the_measured_axes(
    curryer_lsk,
    noaa20_azel_data,
    short_tmp_path,
    spice_test_data_path,
    monkeypatch,
):
    """Each mechanism CK encodes rotation about the measured axis by the telemetered angle."""
    # Builds the Az/El CKs about the measured axes read from the frame kernel.
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
            short_tmp_path / "libera_az.attitude.ck.bc", mkrn.mappings["LIBERA_AZ"], to_fmt="iso"
        )
        assert span == ("2021-04-09 12:00:06.173775", "2021-04-09 12:02:04.964132")
        span = sp.ext.kernel_coverage(
            short_tmp_path / "libera_el.attitude.ck.bc", mkrn.mappings["LIBERA_EL"], to_fmt="iso"
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
                sp.pxform("LIBERA_AZ_COORD", "LIBERA_BASE_COORD", et_time), rotation(az_axis, az), atol=1e-4
            )
            npt.assert_allclose(
                sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", et_time), rotation(el_axis, el), atol=1e-4
            )


def test_mechanism_cks_apply_the_encoder_correction(
    curryer_lsk,
    noaa20_azel_data,
    short_tmp_path,
    spice_test_data_path,
    monkeypatch,
):
    """Az/El CKs built through create_kernel_from_l1a recover telemetry + correction.

    Drives the production create_kernel_from_l1a path with raw CERES angles (mocking only the L1A read)
    so the deterministic encoder correction is applied during kernel generation, then queries the CKs
    and checks the recovered mechanism angle equals correct(raw), not the raw telemetry. Runs on the
    default (jpss4) misaligned set: the CKs rotate about the measured axes read from the frame kernel.
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
                sp.pxform("LIBERA_AZ_COORD", "LIBERA_BASE_COORD", et), rotation(az_axis, corrected_az), atol=1e-4
            )
        for et, corrected_el in zip(et_times, kernel_maker.correct_elevation(raw_el)):
            npt.assert_allclose(
                sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", et), rotation(el_axis, corrected_el), atol=1e-4
            )

        # The correction is genuinely present (elevation amplitude ~4.6e-4 rad, well above CK round-trip noise).
        el_recovered = np.array(
            [angle_about(sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", et), el_axis) for et in et_times]
        )
        assert np.max(np.abs((el_recovered - raw_el + np.pi) % (2 * np.pi) - np.pi)) > 1e-4

        # Between telemetered samples the CK interpolates the corrected angles.
        mid_et = 0.5 * (et_times[0] + et_times[1])
        el_mid = angle_about(sp.pxform("LIBERA_EL_COORD", "LIBERA_AZ_COORD", mid_et), el_axis)
        lo, hi = sorted(kernel_maker.correct_elevation(raw_el[:2]))
        assert lo - 1e-5 <= el_mid <= hi + 1e-5


# ---------------------------------------------------------------------------------------------
# Production entry points: create_kernel_from_packets / create_kernels_from_manifest /
# create_kernel_from_l1a. These assert that the right file lands in the right place under the
# right name, local and on S3. Kernel *contents* are covered by the tests above.
# ---------------------------------------------------------------------------------------------

#: One case per kernel the packet path can produce. ``ground_test_header`` marks the ground-test
#: CCSDS files, which carry an 8-byte prefix ahead of the CCSDS primary header.
PACKET_KERNEL_CASES = [
    pytest.param(
        "test_jpss1_pds_file_1",
        "JPSS-SPK",
        "LIBERA_SPICE_JPSS-SPK_V3-14-159_20210409T000000_20210409T015959_R25056154513.bsp",
        False,
        id="jpss-spk",
    ),
    pytest.param(
        "test_jpss1_pds_file_1",
        "JPSS-CK",
        "LIBERA_SPICE_JPSS-CK_V3-14-159_20210408T235959_20210409T015958_R25056154513.bc",
        False,
        id="jpss-ck",
    ),
    pytest.param(
        "test_ccsds_2025_218_18_37_32",
        "AZROT-CK",
        "LIBERA_SPICE_AZROT-CK_V3-14-159_20250806T183730_20250806T184127_R25056154513.bc",
        True,
        id="az-ck",
    ),
    pytest.param(
        "test_ccsds_2025_218_18_37_32",
        "ELSCAN-CK",
        "LIBERA_SPICE_ELSCAN-CK_V3-14-159_20250806T183730_20250806T184127_R25056154513.bc",
        True,
        id="el-ck",
    ),
]


@pytest.mark.parametrize(
    ("packet_fixture", "kernel_identifier", "expected_name", "ground_test_header"), PACKET_KERNEL_CASES
)
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_create_kernel_from_packets_writes_the_expected_kernel(
    mocked_get_current_version_str,
    packet_fixture,
    kernel_identifier,
    expected_name,
    ground_test_header,
    request,
    short_tmp_path,
    curryer_lsk,
    monkeypatch,
    spice_test_data_path,
):
    """Each kernel type is built from its packet file and written under the expected name."""
    packet_file = request.getfixturevalue(packet_fixture)
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    if ground_test_header:
        monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")

    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=short_tmp_path,
    ):
        kernel_maker.create_kernel_from_packets(
            input_data_files=[str(packet_file)],
            kernel_identifier=kernel_identifier,
            output_dir=str(short_tmp_path),
            overwrite=False,
        )
    assert (short_tmp_path / expected_name).exists()


@pytest.mark.parametrize(
    ("packet_fixture", "kernel_identifier", "expected_name", "ground_test_header"), PACKET_KERNEL_CASES
)
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_create_kernel_from_packets_round_trips_through_s3(
    mocked_get_current_version_str,
    packet_fixture,
    kernel_identifier,
    expected_name,
    ground_test_header,
    request,
    create_mock_bucket,
    write_file_to_s3,
    curryer_lsk,
    generic_kernel_dir,
    monkeypatch,
):
    """The same path works with both the packet input and the kernel output living on S3."""
    packet_file = request.getfixturevalue(packet_fixture)
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    if ground_test_header:
        monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")

    bucket = create_mock_bucket().name
    packet_uri = f"s3://{bucket}/some_path/test_kernel/{packet_file.name}"
    write_file_to_s3(packet_file, packet_uri)
    output_directory = f"s3://{bucket}/some_path/kernel_output/"

    kernel_maker.create_kernel_from_packets(
        input_data_files=[packet_uri],
        kernel_identifier=kernel_identifier,
        output_dir=output_directory,
        overwrite=False,
    )
    assert (S3Path(output_directory) / expected_name).exists()


@pytest.mark.parametrize("test_type", ["S3", "Local"], indirect=True)
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_create_kernels_from_manifest_writes_jpss_kernels_and_a_manifest(
    mocked_get_current_version_str, setup_jpss1_kernel_maker_environment_with_manifest, curryer_lsk, generic_kernel_dir
):
    """An input manifest with no requested time range yields both JPSS kernels and an output manifest."""
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    input_manifest_path, output_path = setup_jpss1_kernel_maker_environment_with_manifest

    mani_out = kernel_maker.create_kernels_from_manifest(input_manifest_path, ["JPSS-CK", "JPSS-SPK"], output_path)

    assert isinstance(mani_out, Manifest)
    assert len(mani_out.files) == 2  # Two kernel types.
    # Time ranges are real, derived from the input L1A packet data.
    assert (output_path / "LIBERA_SPICE_JPSS-SPK_V3-14-159_20280505T041329_20280505T043128_R25056154513.bsp").exists()
    assert (output_path / "LIBERA_SPICE_JPSS-CK_V3-14-159_20280505T041329_20280505T043128_R25056154513.bc").exists()
    assert len(sorted(output_path.glob("*"))) == 3  # 2 kernels + 1 manifest.


@pytest.mark.parametrize("test_type", ["S3", "Local"], indirect=True)
@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_create_kernels_from_manifest_writes_mechanism_kernels_and_a_manifest(
    mocked_get_current_version_str,
    setup_azel_kernel_maker_environment_with_manifest,
    curryer_lsk,
    generic_kernel_dir,
    monkeypatch,
):
    """The same manifest path for the Az/El mechanism CKs."""
    monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    input_manifest_path, output_path = setup_azel_kernel_maker_environment_with_manifest

    mani_out = kernel_maker.create_kernels_from_manifest(input_manifest_path, ["AZROT-CK", "ELSCAN-CK"], output_path)

    assert isinstance(mani_out, Manifest)
    assert len(mani_out.files) == 2  # Two kernel types.
    assert (output_path / "LIBERA_SPICE_AZROT-CK_V3-14-159_20250809T171756_20250809T171904_R25056154513.bc").exists()
    assert (output_path / "LIBERA_SPICE_ELSCAN-CK_V3-14-159_20250809T171756_20250809T171904_R25056154513.bc").exists()
    assert len(sorted(output_path.glob("*"))) == 3  # 2 kernels + 1 manifest.


@mock.patch.object(kernel_maker, "datetime", mock.Mock(wraps=datetime))
@mock.patch("libera_utils.kernel_maker.filenaming.get_current_version_str", return_value="V3-14-159")
def test_create_kernel_from_l1a_furnishes_required_kernels(
    mocked_get_current_version_str,
    test_l1a_sc_pos_product_file,
    short_tmp_path,
    monkeypatch,
    spice_test_data_path,
):
    """``create_kernel_from_l1a`` furnishes what curryer needs before calling out to it.

    It builds a KernelManager, loads static and NAIF kernels, checks they are furnished, and only
    then calls ``spice_utils.make_kernel``. If that sequence breaks, kernel creation fails for
    want of an LSK rather than producing a wrong answer, so a written, non-trivial output file is
    the signal here.
    """
    kernel_maker.datetime.now.return_value = datetime(2025, 2, 25, 15, 45, 13)
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))

    output = kernel_maker.create_kernel_from_l1a(
        l1a_data=test_l1a_sc_pos_product_file, kernel_identifier="JPSS-SPK", output_dir=short_tmp_path, overwrite=True
    )

    assert output.exists(), (
        "Kernel file should exist. If this fails, kernel creation failed, "
        "likely because KernelManager didn't furnish required kernels (e.g. LSK)."
    )
    assert output.suffix == ".bsp", "Output should be an SPK file"
    assert output.stat().st_size > 1024, "Kernel file should be larger than 1KB"
