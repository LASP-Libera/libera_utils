"""The frozen dynamic kernels, checked step by step against their sources.

``tests/test_data/dynamic_kernels`` stands in for the output of the SPICE processing step. Each test
here checks one kernel against the data it was built from or an independent reference, to the
precision that step supports: the mechanism CKs against the measured axes of rotation and the
commanded angles, and the spacecraft SPK and CK against the reference ground track and target
geometry for the simulated JPSS-4 overpass of Libya-4.

They read checked-in kernels and run no external process, so they are unit tests. Geolocation
through the whole chain, which can only be bounded loosely, is in
``tests/integration/test_geolocation.py``.
"""

import numpy as np
import pandas as pd
import pytest
from curryer import spicetime
from curryer import spicierpy as sp
from curryer.compute import spatial

from libera_utils import kernel_maker
from libera_utils.config import config
from tests.fixture_generation import generate_dynamic_kernels as fixture_gen
from tests.plugins.spice_fixtures import LIBYA4_LAT_DEG, LIBYA4_LON_DEG, NOMINAL_AZ_AXIS, NOMINAL_EL_AXIS


def _fixed_axis_residual(rotation: np.ndarray, axis: np.ndarray) -> float:
    """How far ``axis`` moves under ``rotation``. Zero exactly when it is the rotation axis."""
    axis = axis / np.linalg.norm(axis)
    return float(np.linalg.norm(rotation @ axis - axis))


def _largest_rotation(frames: list[np.ndarray]) -> np.ndarray:
    """The rotation matrix turning through the largest angle, where the axis is best resolved."""
    return max(frames, key=lambda r: np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


@pytest.mark.parametrize(
    ("from_frame", "to_frame", "axis_keyword", "nominal_axis"),
    [
        ("LIBERA_AZ_COORD", "LIBERA_BASE_COORD", "LIBERA_AZ_AOR_IN_STAND", NOMINAL_AZ_AXIS),
        ("LIBERA_EL_COORD", "LIBERA_AZ_COORD", "LIBERA_EL_AOR_IN_STAND", NOMINAL_EL_AXIS),
    ],
    ids=["azimuth", "elevation"],
)
def test_frozen_mechanism_cks_rotate_about_measured_axes(
    frozen_kernel_manager, from_frame, to_frame, axis_keyword, nominal_axis
):
    """Each frozen mechanism CK rotates about the frame kernel's measured axis, not the nominal one.

    A CK built about the ideal coordinate axes, paired with a frame kernel that declares the
    measured axes and a misaligned radiometer boresight, describes a configuration no real
    instrument has.

    A rotation fixes its own axis, so ``R @ axis == axis`` identifies the axis directly and does
    not depend on knowing the commanded angle. Asserting the nominal axis is *not* fixed is what
    fails if the fixture is ever regenerated from a nominal-geometry path.
    """
    times = pd.date_range(fixture_gen.WINDOW_START, fixture_gen.WINDOW_END, freq="500ms", inclusive="left")
    ephemeris_times = spicetime.adapt(times, "dt64", "et")

    measured_axis = np.array(sp.gdpool(axis_keyword, 0, 3))
    rotations = [sp.pxform(from_frame, to_frame, et) for et in ephemeris_times]
    rotation = _largest_rotation(rotations)

    # The measured axis is fixed by the rotation: this is the axis the CK was built about.
    assert _fixed_axis_residual(rotation, measured_axis) < 1e-6, f"{from_frame} CK does not rotate about {axis_keyword}"

    # The nominal axis is not. The separation is the misalignment, so the threshold only has to
    # exceed CK round-trip noise while staying well under the ~0.065-0.283 deg axis tilt.
    assert _fixed_axis_residual(rotation, nominal_axis) > 1e-4, (
        f"{from_frame} CK rotates about the nominal axis -- the measured misalignment is missing"
    )


@pytest.mark.parametrize(
    ("builder_name", "from_frame", "to_frame", "axis_keyword", "stride"),
    [
        ("azimuth_frame", "LIBERA_AZ_COORD", "LIBERA_BASE_COORD", "LIBERA_AZ_AOR_IN_STAND", 1),
        # The elevation sweep is sampled at 1 kHz; every 50th sample still resolves each sweep.
        ("elevation_frame", "LIBERA_EL_COORD", "LIBERA_AZ_COORD", "LIBERA_EL_AOR_IN_STAND", 50),
    ],
    ids=["azimuth", "elevation"],
)
def test_frozen_mechanism_ck_reproduces_commanded_pointing(
    frozen_kernel_manager, builder_name, from_frame, to_frame, axis_keyword, stride
):
    """Each mechanism CK encodes the commanded angle, so it carries the encoder correction.

    Ties the fixture to the data it was generated from. The generation script synthesises raw
    telemetry by applying the inverse encoder correction to the commanded angle, so the expected
    value here is that telemetry with the forward correction applied, which is the commanded angle.
    A CK built from the raw telemetry without the correction is off by up to ~5e-4 rad, which no
    footprint bound in this module would detect.
    """
    frame, _ = getattr(fixture_gen, builder_name)()
    frame = kernel_maker.apply_encoder_corrections(frame).iloc[::stride]
    field = kernel_maker.AZ_ENCODER_FIELD if kernel_maker.AZ_ENCODER_FIELD in frame else kernel_maker.EL_ENCODER_FIELD

    axis = np.array(sp.gdpool(axis_keyword, 0, 3))
    axis = axis / np.linalg.norm(axis)

    recovered = []
    for et in frame["AXIS_SAMPLE_ICIE_ET"]:
        rotation = sp.pxform(from_frame, to_frame, et)
        vee = 0.5 * np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        recovered.append(np.arctan2(vee @ axis, 0.5 * (np.trace(rotation) - 1.0)))

    # Tolerance covers msopck's quaternion quantisation only; the correction round trip itself
    # is exact to machine precision (see TestEncoderCorrection in tests/unit/test_kernel_maker.py).
    np.testing.assert_allclose(np.array(recovered), frame[field].to_numpy(), rtol=0, atol=1e-6)


def test_frozen_spacecraft_spk_matches_the_reference_ground_track(frozen_kernel_manager, test_data_path):
    """The spacecraft SPK reproduces the reference sub-satellite ground track.

    A per-step golden check on the ephemeris alone, independent of any pointing: the SPK is built
    from ``JPSS-4_Fixed_Ephemeris_And_Attitude.csv`` (Cartesian state), and this compares it
    against ``JPSS-4_Geodetic_LLA_libya4_access03.csv`` -- the geodetic sub-satellite track for
    the same scenario, supplied alongside it.

    Isolating the ephemeris matters because every pointing test inherits it: a spacecraft state
    error would otherwise surface as a confusing pointing failure.
    """
    truth = pd.read_csv(test_data_path / "tier1_geo" / "JPSS-4_Geodetic_LLA_libya4_access03.csv")
    timestamps = pd.to_datetime(truth["Time (UTCG)"], format="%d %b %Y %H:%M:%S.%f")

    # Every 120th row: the track is smooth at 1 Hz, so 16 samples across the span exercise the
    # whole orbit arc without 1801 SPICE calls.
    sampled = truth.iloc[::120]
    ephemeris_times = spicetime.adapt(timestamps.iloc[::120].to_numpy(), "dt64", "et")
    positions = np.array([sp.spkpos("-143013", et, "ITRF93", "NONE", "EARTH")[0] for et in ephemeris_times])

    # WGS84, STK's default Earth ellipsoid for geodetic reports.
    longitude, latitude, altitude = spatial.ecef_to_geodetic(positions, meters=False, degrees=True).T
    latitude_error = latitude - sampled["Lat (deg)"].to_numpy()
    longitude_error = longitude - sampled["Lon (deg)"].to_numpy()
    altitude_error = altitude - sampled["Alt (km)"].to_numpy()

    # The reference lat/lon are quoted to 3 decimal places, so agreement is bounded below by the
    # 0.001 deg quantum regardless of how good the ephemeris is. Observed residuals sit at half a
    # quantum in both components -- including latitude, which Earth-rotation error cannot touch --
    # so this is reference rounding rather than ephemeris error.
    assert np.abs(latitude_error).max() < 1e-3, np.abs(latitude_error).max()
    assert np.abs(longitude_error).max() < 1e-3, np.abs(longitude_error).max()
    # Altitude is quoted to six decimals, so it admits a far tighter bound.
    assert np.abs(altitude_error).max() < 5e-3, np.abs(altitude_error).max()


def test_frozen_spacecraft_kernels_match_the_reference_target_geometry(frozen_kernel_manager, test_data_path):
    """The spacecraft SPK and attitude CK reproduce the reference geometry to Libya-4.

    ``Libya-4_access003.csv`` is the access report for the same simulated overpass, independent of
    anything this repository computes. Its ``Nadir Angle`` is the angle at the spacecraft between
    geodetic nadir and the site, which depends on the ephemeris alone. Its ``Target Azimuth`` is the
    site's azimuth about nadir, measured from the spacecraft body +X axis projected onto the plane
    normal to nadir, so it depends on the spacecraft yaw as well.

    The ground-track test checks the ephemeris against a reference, and the mechanism CK tests check
    the CKs against their commanded angles, but nothing else checks the spacecraft attitude against
    a source other than its own input. A yaw error in the attitude CK, or in the frame that carries
    it, moves the target azimuth by the same angle.
    """
    access = pd.read_csv(test_data_path / "tier1_geo" / "Libya-4_access003.csv")
    timestamps = pd.to_datetime(access["Time (UTC)"], format="%d %b %Y %H:%M:%S.%f")
    in_window = (timestamps >= fixture_gen.WINDOW_START) & (timestamps < fixture_gen.WINDOW_END)
    access = access[in_window.to_numpy()]
    ephemeris_times = spicetime.adapt(timestamps[in_window].to_numpy(), "dt64", "et")

    _, radii = sp.bodvrd("EARTH", "RADII", 3)
    flattening = (radii[0] - radii[2]) / radii[0]
    site = np.array(sp.georec(np.deg2rad(LIBYA4_LON_DEG), np.deg2rad(LIBYA4_LAT_DEG), 0.0, radii[0], flattening))

    target_azimuth = []
    nadir_angle = []
    for et in ephemeris_times:
        spacecraft = np.array(sp.spkpos("-143013", et, "ITRF93", "NONE", "EARTH")[0])
        # ITRF93 is the destination frame so the chain resolves through the spacecraft CK, which is
        # written relative to ITRF93, without Earth orientation data for the 2028 epochs.
        itrf_to_body = sp.pxform("JPSS4_SC_COORD", "ITRF93", et).T
        to_site = itrf_to_body @ (site - spacecraft)
        nadir = itrf_to_body @ -spacecraft
        nadir /= np.linalg.norm(nadir)
        reference_x = np.array([1.0, 0.0, 0.0]) - nadir[0] * nadir
        reference_x /= np.linalg.norm(reference_x)
        reference_y = np.cross(nadir, reference_x)
        target_azimuth.append(np.degrees(np.arctan2(to_site @ reference_y, to_site @ reference_x)))
        nadir_angle.append(np.degrees(np.arccos(to_site @ nadir / np.linalg.norm(to_site))))

    azimuth_error = np.array(target_azimuth) - access["Target Azimuth (deg)"].to_numpy()
    nadir_error = np.array(nadir_angle) - access["Nadir Angle (deg)"].to_numpy()
    # The report quotes both angles to 0.001 deg, so two quanta is reference rounding.
    assert np.abs(azimuth_error).max() < 2e-3, np.abs(azimuth_error).max()
    # The site is placed on the ellipsoid here. The reference site sits above it, at a height the
    # report does not give, which biases the nadir angle by about -0.002 deg (a site height near
    # 0.14 km removes the bias). The azimuth about nadir is nearly insensitive to that height.
    assert np.abs(nadir_error).max() < 4e-3, np.abs(nadir_error).max()


def test_frozen_mechanism_ck_coverage_matches_the_documented_window(frozen_kernel_manager, test_data_path):
    """The frozen mechanism CKs cover exactly the window the generation script documents.

    Downstream repos pin their test epochs to this window, so a silent coverage change needs to
    fail here rather than as a confusing SPICE insufficient-data error somewhere else.
    """
    kernel_dir = test_data_path / "dynamic_kernels"
    with sp.ext.load_kernel(config.get("LIBERA_KERNEL_CLOCK")):
        for pattern, body in (("*AZROT-CK*.bc", "LIBERA_AZ"), ("*ELSCAN-CK*.bc", "LIBERA_EL")):
            (kernel_file,) = kernel_dir.glob(pattern)
            start, end = sp.ext.kernel_coverage(kernel_file, sp.obj.Body(body, frame=True), to_fmt="iso")
            assert start.startswith(str(fixture_gen.WINDOW_START)), (body, start)
            assert end.startswith(str(fixture_gen.WINDOW_END)), (body, end)
