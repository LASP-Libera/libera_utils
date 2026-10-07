"""Geometry and SPICE-pool helpers shared across the kernel and geolocation test modules.

These are assertion tools, not fixtures: they take arguments and return values rather than
participating in pytest's dependency injection, so they live here instead of in ``tests/plugins``.
"""

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from libera_utils.libera_spice import spice_utils


def rotation(axis: npt.ArrayLike, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` radians about unit ``axis`` (Rodrigues).

    Parameters
    ----------
    axis : array_like
        Three-element rotation axis. Normalized internally, so it need not be a unit vector.
    angle : float
        Rotation angle in radians, right-handed about ``axis``.

    Returns
    -------
    numpy.ndarray
        The 3x3 rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos, sin = np.cos(angle), np.sin(angle)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * cos + np.outer(axis, axis) * (1 - cos) + skew * sin


def angle_about(matrix: np.ndarray, axis: npt.ArrayLike) -> float:
    """Signed rotation angle (radians) of ``matrix`` about unit ``axis``.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 3x3 rotation matrix.
    axis : array_like
        Three-element axis to measure the rotation about. Normalized internally.

    Returns
    -------
    float
        The signed angle in radians. Only meaningful when ``matrix`` is a rotation about ``axis``;
        for a general rotation this returns the component along ``axis``.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    vee = 0.5 * np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return float(np.arctan2(vee @ axis, 0.5 * (np.trace(matrix) - 1.0)))


def index_of_furnished_kernel_matching(pattern: str) -> int:
    """Return the SPICE kernel-pool index of the first furnished file whose name matches ``pattern``.

    Parameters
    ----------
    pattern : str
        Regular expression matched against each furnished kernel's basename.

    Returns
    -------
    int
        Position in the kernel pool. Pool order matters where coverage overlaps: SPICE prefers the
        last-loaded kernel.

    Raises
    ------
    Failed
        Calls ``pytest.fail`` when no furnished kernel matches, reporting the whole pool.
    """
    pool = spice_utils.ls_kernels()
    for index, kernel in enumerate(pool):
        if re.search(pattern, Path(kernel.file_name).name):
            return index
    pool_names = [kernel.file_name for kernel in pool]
    pytest.fail(f"No furnished kernel matching {pattern!r}. Kernel pool: {pool_names}")
