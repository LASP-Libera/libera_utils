"""Tests for time module"""
import numpy as np
import pytest

from libera_sdp import time


@pytest.mark.parametrize(
    ('ccsds_jd', 'jd'),
    [
        (23171.80139, 2459376.30139),
        (np.float64(23171.80139), np.float64(2459376.30139)),
        (np.array([23171.80139], dtype=np.float64), np.array([2459376.30139], dtype=np.float64)),
        (
                np.array([23171.80139, 23171.80139], dtype=np.float64),
                np.array([2459376.30139, 2459376.30139], dtype=np.float64)
        )
    ]
)
def test_ccsdsjd_2_jd(ccsds_jd, jd):
    """Test conversion of CCSDS JD to JD"""
    result = time.ccsdsjd_2_jd(ccsds_jd)
    if isinstance(jd, np.ndarray):
        assert np.array_equal(result, jd)
    else:
        assert result == jd


@pytest.mark.parametrize(
    ('days', 'ms', 'us', 'expected'),
    [
        (23109, 7, 137, 23109.000000082604),
        (
            np.array([23109], dtype=np.int64),
            np.array([7], dtype=np.int64),
            np.array([137], dtype=np.int64),
            np.array([23109.000000082604], dtype=np.float64)
        ),
        (
            np.array([23109, 23109], dtype=np.int64),
            np.array([7, 1005], dtype=np.int64),
            np.array([137, 176], dtype=np.int64),
            np.array([23109.000000082604, 23109.000011633983], dtype=np.float64)
        ),

    ]
)
def test_days_ms_us_2_decimal_days(days, ms, us, expected):
    """Test conversion of distinct time parts to a Julian day representation"""
    result = time.days_ms_us_2_decimal_days(days, ms, us)
    if isinstance(expected, np.ndarray):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ('s', 'jd'),
    [
        ('1873-12-29T12:04:08', 2405522.00287037),  # MSD 0 epoch
        ('1992-02-25T03:45:15', 2448677.656423611),  # Random time
        ('1958-01-01T00:00:00', 2436204.5),  # CCSDS JD epoch
        (np.array(['1873-12-29T12:04:08', '1992-02-25T03:45:15']), np.array([2405522.00287037, 2448677.656423611]))
    ])
def test_utc_2_jd(s, jd):
    """Test converting ISO timestamps to Julian day"""
    result = time.utc_2_jd(s)
    if isinstance(result, np.ndarray):
        assert np.array_equal(result, jd)
    else:
        assert result == jd


@pytest.mark.parametrize(
    ('isot', 'jd'),
    [
        ('1873-12-29T12:04:08.000', 2405522.00287037),  # MSD 0 epoch
        ('1992-02-25T03:45:15.000', 2448677.656423611),  # Random time
        ('1958-01-01T00:00:00.000', 2436204.5),  # CCSDS JD epoch
        (np.array(['1873-12-29T12:04:08.000', '1992-02-25T03:45:15.000']),
         np.array([2405522.00287037, 2448677.656423611]))
    ])
def test_jd_2_utc(isot, jd):
    """Test converting Julian day to ISO timestamps"""
    result = time.jd_2_utc(jd)
    if isinstance(result, np.ndarray):
        assert np.array_equal(result, isot)
    else:
        assert result == isot
