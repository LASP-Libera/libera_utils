"""Module for dealing with time and time conventions

Some convention for this module

1. Only decorate direct spiceypy wrapper functions with the ensure_spice decorator. They should directly call a
spiceypy function.

2. All spiceypy wrapper functions should read as <spiceypyfunc>_wrapper. We really only use these to allow array
inputs for spiceypy functions that aren't already vectorized in C and to wrap them in ensure_spice.

3. All functions should have robust type-hinting.
"""

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr


ISOT_REGEX = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"[T|t]"
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fractional_second>[0-9]*))?$"
)

PRINTABLE_TS_REGEX = re.compile(
    r"^(?P<year>[0-9]{4})(?P<month>[0-9]{2})(?P<day>[0-9]{2})"
    r"[T|t]"
    r"(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2})$"
)

PRINTABLE_TS_FORMAT = "%Y%m%dT%H%M%S"

NUMERIC_DOY_TS_FORMAT = "%y%j%H%M%S"

# Epoch recommended by CCSDS and used by JPSS4 and Libera
CCSDS_EPOCH = datetime.fromisoformat("1958-01-01")


# ============================================================================
# SPICE Time Conversion Functions
# These have been moved to libera_spice.spice_utils to avoid circular imports.
# Re-exported here for backwards compatibility.
# ============================================================================

from libera_utils.libera_spice.spice_utils import (  # noqa: E402
    et_2_datetime,
    et_2_timestamp,
    et2utc_wrapper,
    sce2s_wrapper,
    scs2e_wrapper,
    utc2et_wrapper,
)

# Explicitly export all public functions
__all__ = [
    "et_2_datetime",
    "et_2_timestamp",
    "et2utc_wrapper",
    "sce2s_wrapper",
    "scs2e_wrapper",
    "utc2et_wrapper",
    "convert_cds_integer_to_datetime",
    "multipart_to_dt64",
    "CCSDS_EPOCH",
    "ISOT_REGEX",
    "PRINTABLE_TS_REGEX",
    "PRINTABLE_TS_FORMAT",
    "NUMERIC_DOY_TS_FORMAT",
]


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
    byte_data = satellite_time.to_bytes(8, "big")
    int_days = int.from_bytes([byte_data[0], byte_data[1]], byteorder="big")
    int_millisec = int.from_bytes([byte_data[2], byte_data[3], byte_data[4], byte_data[5]], byteorder="big")
    int_microsec = int.from_bytes([byte_data[6], byte_data[7]], byteorder="big")

    reference_date = datetime(1958, 1, 1, 0, 0, 0, 0, ZoneInfo("UTC"))
    cds_time = (
        reference_date
        + timedelta(days=int_days)
        + timedelta(milliseconds=int_millisec)
        + timedelta(microseconds=int_microsec)
    )

    # TODO[LIBSDC-206]: Check with EDOS on this time conversion.
    #  The commented out below gives approximately a 70 second difference
    #  to the method above.
    #  satellite_time_string = f"{int_days}:{int_millisec}:{int_microsec}"
    #  non_tz_datetime = et_2_datetime(scs2e_wrapper(satellite_time_string))
    #  cds_time = timezone("UTC").localize(non_tz_datetime)

    return cds_time


def multipart_to_dt64(
    data: pd.DataFrame | xr.Dataset,
    day_field: str | None = None,
    ms_field: str | None = None,
    us_field: str | None = None,
    s_field: str | None = None,
    epoch: str | datetime = CCSDS_EPOCH,
) -> pd.Series:
    """Convert multipart time fields to a datetime64 time.

    Parameters
    ----------
    data : pd.DataFrame | xr.Dataset
        Any data structure containing the named subscript-able fields.
    day_field : str | None, optional
        Name of the day count field.
    ms_field : str | None, optional
        Name of the millisecond count field.
    us_field : str | None, optional
        Name of the microsecond count field.
    s_field : str | None, optional
        Name of the second count field.
    epoch : str | datetime
        Date time string of the zero-offset epoch. Default="1958-01-01"

    Returns
    -------
    pd.Series
        Pandas series of the datetime64 values.

    """
    result = pd.Timestamp(epoch)

    if day_field is not None:
        result = result + pd.to_timedelta(data[day_field], "D")
    if s_field is not None:
        result = result + pd.to_timedelta(data[s_field], "s")
    if ms_field is not None:
        result = result + pd.to_timedelta(data[ms_field], "ms")
    if us_field is not None:
        result = result + pd.to_timedelta(data[us_field], "us")

    return result
