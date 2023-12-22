"""Module for dealing with time and time conventions

Some convention for this module

1. Only decorate direct spiceypy wrapper functions with the ensure_spice decorator. They should directly call a
spiceypy function.

2. All spiceypy wrapper functions should read as <spiceypyfunc>_wrapper. We really only use these to allow array
inputs for spiceypy functions that aren't already vectorized in C and to wrap them in ensure_spice.

3. All functions should have robust type-hinting.
"""
# Standard
import re
from datetime import datetime, timedelta
from typing import Union, Collection
from zoneinfo import ZoneInfo
# Installed
import numpy as np
import spiceypy as spice
# Local
from libera_utils.config import config
from libera_utils.spice_utils import ensure_spice

ISOT_REGEX = re.compile(r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
                        r"[T|t]"
                        r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
                        r"(?:\.(?P<fractional_second>[0-9]*))?$")

PRINTABLE_TS_REGEX = re.compile(r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})"
                                r"[T|t]"
                                r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})$")

PRINTABLE_TS_FORMAT = "%Y%m%dT%H%M%S"

NUMERIC_DOY_TS_FORMAT = "%y%j%H%M%S"


def et_2_timestamp(et: Union[float, Collection[float], np.ndarray],
                   fmt: str = '%Y%m%dT%H%M%S.%f') -> Union[str, Collection[str]]:
    """
    Convert ephemeris time to a custom formatted timestamp (default is lowercase version of ISO).

    Parameters
    ----------
    et: Union[float, Collection[float], numpy.ndarray]
        Ephemeris Time to be converted.
    fmt: str, Optional
        Format string as defined by the datetime.strftime() function.

    Returns
    -------
    : Union[str, Collection[str]]
        Formatted timestamps
    """
    datetime_objs = et_2_datetime(et)

    if isinstance(datetime_objs, Collection):
        time_out = np.array([t.strftime(fmt) for t in datetime_objs])
    else:
        time_out = datetime_objs.strftime(fmt)

    return time_out


def et_2_datetime(et: Union[float, Collection[float], np.ndarray]) -> Union[datetime, np.ndarray]:
    """
    Convert ephemeris time to a python datetime object by first converting it to a UTC timestamp.

    Parameters
    ----------
    et: float or Collection or numpy.ndarray
        Ephemeris times to be converted.

    Returns
    -------
    : datetime.datetime or numpy.ndarray
        Object representation of ephemeris times.
    """
    isoc_fmt = '%Y-%m-%dT%H:%M:%S.%f'
    isoc_prec = 6

    isoc_timestamp = et2utc_wrapper(et, 'ISOC', isoc_prec)
    if isinstance(et, Collection):
        return np.array([datetime.strptime(s, isoc_fmt) for s in isoc_timestamp])

    return datetime.strptime(isoc_timestamp, isoc_fmt)


@ensure_spice(time_kernels_only=True)
def et2utc_wrapper(et: Union[float, Collection[float], np.ndarray], fmt: str, prec: int) -> Union[str, Collection[str]]:
    """
    Convert ephemeris times to UTC ISO strings.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/et2utc_c.html
    Decorated wrapper for spiceypy.et2utc that will automatically furnish the latest metakernel and retry
    if the first call raises an exception.

    Parameters
    ----------
    et: Union[float, Collection[float], numpy.ndarray]
        The ephemeris time value to be converted to UTC.
    fmt: str
        Format string defines the format of the output time string. See CSPICE docs.
    prec: int
        Number of digits of precision for fractional seconds.

    Returns
    -------
    : Union[numpy.ndarray, str]
        UTC time string(s)
    """
    return spice.et2utc(et, fmt, prec)


@ensure_spice(time_kernels_only=True)
def utc2et_wrapper(iso_str: Union[str, Collection[str]]) -> Union[float, np.ndarray]:
    """
    Convert UTC ISO strings to ephemeris times.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/utc2et_c.html
    Decorated wrapper for spiceypy.utc2et that will automatically furnish the latest metakernel and retry
    if the first call raises an exception.

    Parameters
    ----------
    iso_str: Union[str, Collection[str]]
        The UTC to convert to ephemeris time

    Returns
    -------
    : float or numpy.ndarray
        Ephemeris time
    """
    if isinstance(iso_str, str):
        return spice.utc2et(iso_str)

    return np.array([spice.utc2et(s) for s in iso_str])


@ensure_spice(time_kernels_only=True)
def scs2e_wrapper(sclk_str: Union[str, Collection[str]]) -> Union[float, np.ndarray]:
    """
    Convert SCLK strings to ephemeris time.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/scs2e_c.html
    Decorated wrapper for spiceypy.scs2e that will automatically furnish the latest metakernel and retry
    if the first call raises an exception.

    Parameters
    ----------
    sclk_str: Union[str, Collection[str]]
        Spacecraft clock string

    Returns
    -------
    : Union[float, numpy.ndarray]
        Ephemeris time
    """
    sc_id = config.get("JPSS_SC_ID")
    if isinstance(sclk_str, str):
        return spice.scs2e(sc_id, sclk_str)

    return np.array([spice.scs2e(sc_id, s) for s in sclk_str])


@ensure_spice(time_kernels_only=True)
def sce2s_wrapper(et: Union[float, Collection[float], np.ndarray]) -> Union[str, np.ndarray]:
    """
    Convert ephemeris times to SCLK string
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/sce2s_c.html
    Decorated wrapper for spiceypy.sce2s that will automatically furnish the latest metakernel and retry
    if the first call raises an exception.

    Parameters
    ----------
    et: Union[float, Collection[float], numpy.ndarray]
        Ephemeris time

    Returns
    -------
    : Union[str, Collection[str]]
        SCLK string
    """
    sc_id = config.get("JPSS_SC_ID")
    if isinstance(et, Collection):
        return np.array([spice.sce2s(sc_id, t) for t in et])

    return spice.sce2s(sc_id, et)


def convert_cds_integer_to_datetime(satellite_time: int):
    """Helper function to convert a satellite time given as an CCSDS Day Segmented Time Code (CDS) form as 8 byte
    integer to a timezone aware datetime object

    Parameters
    ----------
    satellite_time : int
        A 64-bit unsigned integer that represents CDS time

    Returns
    -------
    cds_time : datetime.datetime
     """
    byte_data = satellite_time.to_bytes(8, 'big')
    int_days = int.from_bytes([byte_data[0], byte_data[1]], byteorder="big")
    int_millisec = int.from_bytes([byte_data[2], byte_data[3],
                                    byte_data[4], byte_data[5]],
                                   byteorder="big")
    int_microsec = int.from_bytes([byte_data[6], byte_data[7]], byteorder="big")

    reference_date = datetime(1958, 1, 1, 0, 0, 0, 0, ZoneInfo("UTC"))
    cds_time = (reference_date +
                timedelta(days=int_days) +
                timedelta(milliseconds=int_millisec) +
                timedelta(microseconds=int_microsec))

    # TODO: Check with EDOS on this time conversion. The commented out below gives approximately a 70 second difference
    # TODO: to the method above.
    # TODO: satellite_time_string = f"{int_days}:{int_millisec}:{int_microsec}"
    # TODO: non_tz_datetime = et_2_datetime(scs2e_wrapper(satellite_time_string))
    # TODO: cds_time = timezone("UTC").localize(non_tz_datetime)

    return cds_time
