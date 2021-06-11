"""Module for dealing with time and time convensions"""
import re
from astropy import time as aptime
import numpy as np

_CCSDS_JD_EPOCH = np.float64(2436204.5)

ISOT_REGEX = re.compile(r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
                        r"[T|t]"
                        r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
                        r"(?:\.(?P<fractional_second>[0-9]*))?$")

PRINTABLE_TS_REGEX = re.compile(r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})"
                                r"[T|t]"
                                r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})$")


def ccsdsjd_2_jd(ccsds_jd: np.float64 or np.ndarray):
    """Convert CCSDS Julian Day (since 0h 1/1/1958) to standard Julian Day (since 12h Jan 1, 4713 BC)

    Parameters
    ----------
    ccsds_jd : np.float64 or np.ndarray
        CCSDS Julian day

    Returns
    -------
    : np.float64 or np.ndarray
        Julian day
    """
    return ccsds_jd + _CCSDS_JD_EPOCH


def days_ms_us_2_decimal_days(days: np.int64 or np.ndarray, ms: np.int64 or np.ndarray, us: np.int64 or np.ndarray):
    """Convert a tuple of days, milliseconds, microseconds to decimal days.

    Parameters
    ----------
    days: np.int64 or np.ndarray
        Days since epoch (could be any epoch, this function is agnostic).
    ms : np.int64 or np.ndarray
        Milliseconds
    us : np.int64 or np.ndarray
        Microseconds

    Returns
    -------
    : np.float64 or np.ndarray
        Julian day
    """
    ms_in_days = (np.float64(ms) * 1e-3) / 86400.
    us_in_days = (np.float64(us) * 1e-6) / 86400.
    return days + ms_in_days + us_in_days


def utc_2_jd(iso_str: str or np.ndarray):
    """Convert a UTC string to Julian day

    Parameters
    ----------
    iso_str : str or np.ndarray
        UTC ISO-T string or array of strings to convert

    Returns
    -------
    : np.float64 or np.ndarray
        JD as a float or an array of floats
    """

    if isinstance(iso_str, np.ndarray):
        return np.array([aptime.Time(s, format='isot', scale='utc').jd for s in iso_str])
    else:
        return aptime.Time(iso_str, format='isot', scale='utc').jd


def jd_2_utc(jd: np.float64 or np.ndarray):
    """Convert a Julian day value to a UTC string

    Parameters
    ----------
    jd : np.float64 or np.ndarray
        Float or an array of floats.

    Returns
    -------
    : str or np.ndarray
        UTC ISO-T string or an array of them
    """

    if isinstance(jd, np.ndarray):
        return np.array([aptime.Time(t, format='jd', scale='utc').isot for t in jd])
    else:
        return aptime.Time(jd, format='jd', scale='utc').isot

