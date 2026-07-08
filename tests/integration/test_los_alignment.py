"""Goal-2 line-of-sight alignment test (LIBSDC-806).

Validates that the measured Libera frame misalignments (Az/El axes of rotation + radiometer
boresight, stored in the frame kernel) reproduce Javier Fernandez's independently-computed
line-of-sight in the LIBERA_BASE (STAND) frame over a RAP scan.

This is a partial check: Javier's computation stops at the LIBERA_BASE frame (it does not yet
reach the spacecraft/JPSS ephemeris or geolocation stages), so we compare LOS unit vectors in
LIBERA_BASE, not geolocated lat/lon.
"""

import numpy as np
import pandas as pd
import pytest
from curryer import meta, spicetime
from curryer import spicierpy as sp

from libera_utils import kernel_maker
from libera_utils.config import config
from libera_utils.libera_spice import spice_utils
from libera_utils.libera_spice.kernel_manager import KernelManager

pytestmark = pytest.mark.integration


def _rotation(axis, angle):
    """Rotation matrix for ``angle`` radians about unit ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos, sin = np.cos(angle), np.sin(angle)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * cos + np.outer(axis, axis) * (1 - cos) + skew * sin


def _angle_between(a, b):
    """Angle (degrees) between corresponding rows of two arrays of vectors."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def test_los_alignment_vs_javier(
    noaa20_environment, curryer_lsk, short_tmp_path, spice_test_data_path, test_data_path, monkeypatch
):
    """LIBERA_BASE -> radiometer LOS reproduces Javier's u_LOS_STAND over a RAP scan (LIBSDC-806)."""
    monkeypatch.setenv("GENERIC_KERNEL_DIR", str(spice_test_data_path))
    km = KernelManager()
    km.load_static_kernels()

    # RAP scan (sub-sampled to every 200th telemetry sample to keep the fixture small): already-corrected
    # Az/El mechanism angles plus Javier's computed LOS in the LIBERA_BASE (STAND) frame.
    scan = pd.read_csv(test_data_path / "los_alignment" / "sampleElAzAngles_wLosVec_RAPS_20260707T152556.csv")
    base_et = spicetime.adapt("2021-04-09T12:00:07", "iso", "et")
    et = base_et + scan["time_sec"].to_numpy()
    corrected_az = scan["measAngle_Az_rad"].to_numpy()
    corrected_el = scan["measAngle_El_rad"].to_numpy()
    javier_los = scan[["u_LOS_STAND_x", "u_LOS_STAND_y", "u_LOS_STAND_z"]].to_numpy()

    # Build the mechanism quaternions directly from the already-corrected angles (the encoder correction
    # is exercised separately in the tier0 kernel test; here the input is post-correction, as Javier's is).
    kernel_df = pd.DataFrame(
        {"AXIS_SAMPLE_ICIE_ET": et, "ICIE__AXIS_AZ_FILT": corrected_az, "ICIE__AXIS_EL_FILT": corrected_el}
    )
    kernel_maker.add_mechanism_ck_quaternions(kernel_df)

    generated_kernels = [
        spice_utils.make_kernel(config.get(cfg), short_tmp_path, input_data=kernel_df)
        for cfg in ("LIBERA_KERNEL_AZ_CK_CONFIG", "LIBERA_KERNEL_EL_CK_CONFIG")
    ]
    mkrn = meta.MetaKernel.from_json(config.get("LIBERA_KERNEL_META"), relative=True, sds_dir=spice_test_data_path)

    boresight = np.array([0.0, 0.0, 1.0])
    with sp.ext.load_kernel([mkrn.mission_kernels, generated_kernels]):
        az_axis = np.array(sp.gdpool("LIBERA_AZ_AOR_IN_STAND", 0, 3))
        el_axis = np.array(sp.gdpool("LIBERA_EL_AOR_IN_STAND", 0, 3))
        el0_z = np.array(sp.gdpool("LIBERA_EL0_Z_IN_STAND", 0, 3))

        spice_los = np.array([sp.pxform("LIBERA_TOT_RAD_COORD", "LIBERA_BASE_COORD", e) @ boresight for e in et])
        # Independent recompute of Javier's rotateVectorAboutAxis process from the same measured vectors.
        rodrigues_los = np.array(
            [_rotation(az_axis, az) @ _rotation(el_axis, el) @ el0_z for az, el in zip(corrected_az, corrected_el)]
        )

    # The misalignment is a real, non-trivial effect: Javier's LOS is well off the nominal +Z boresight.
    assert _angle_between(javier_los, np.tile(boresight, (len(et), 1))).max() > 0.1

    # Our understanding of the geometry (measured vectors + rotateVectorAboutAxis) matches his numbers.
    assert _angle_between(rodrigues_los, javier_los).max() < 1e-4

    # Our SPICE frame chain (measured-axis Az/El CKs + boresight frame kernel) reproduces his LOS. The
    # residual is dominated by the Az CK down-sample tolerance (~0.006 deg), far below the misalignment.
    assert _angle_between(spice_los, javier_los).max() < 0.02
