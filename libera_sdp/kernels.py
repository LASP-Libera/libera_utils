"""Modules for SPICE kernel creation, management, and usage"""

import os
import subprocess
from pathlib import Path

import numpy as np

from libera_sdp.config import config

# Default for type 13: Hermite Spline Interpolation, unequal spacing
default_spk_setup = {
    'INPUT_DATA_TYPE': 'STATES',
    'OUTPUT_SPK_TYPE': 13,
    'OBJECT_ID': 159,  # TODO: This is the SC ID that comes down for JPSS1. What is the JPSS-3 NAIF ID?
    'CENTER_ID': -1,  # TODO: NAIF reserved code of the center of motion for the object. What is this for JPSS-3?
    'REF_FRAME_NAME': 'ITRF93',  # TODO: JPSS-1 XML says ECEF. ITRF93 is ECEF but is it the correct one?
    'PRODUCER_ID': 'Gavin Medley (for Libera SDC)',
    'DATA_ORDER': ['epoch', 'x', 'y', 'z', 'vx', 'vy', 'vz'],
    'DATA_DELIMITER': ' ',
    'LEAPSECONDS_FILE': f'{config.get("LIBSDP_DATA_DIR")}/naif0012.tls',
    # 'FRAME_DEF_FILE': 'frame definition file name',  # TODO: If we need to use a non-standard frame, we need this
    'INPUT_DATA_UNITS': {'ANGLES': 'DEGREES', 'DISTANCES': 'METERS'},
    'IGNORE_FIRST_LINE': 0,
    'LINES_PER_RECORD': 1,
    'TIME_WRAPPER': '# JD',
    'POLYNOM_DEGREE': 7,
    # 'SEGMENT_ID': 'segment identifier',
    # TODO: How do we want to handle segments if we are producing a kernel for every 24 hr period?
    'APPEND_TO_OUTPUT': 'NO'
}

default_ck_setup = {
    "LSK_FILE_NAME": f'{config.get("LIBSDP_DATA_DIR")}/naif0012.tls',
    "MAKE_FAKE_SCLK": "/tmp/fake.tsc",
    "REFERENCE_FRAME_NAME": "J2000",
    "INPUT_DATA_TYPE": "SPICE QUATERNIONS",
    "INPUT_TIME_TYPE": "UTC",  # JD in UTC scale
    "ANGULAR_RATE_PRESENT": 'MAKE UP/NO AVERAGING',
    'CHECK_TIME_ORDER': 'YES',
    "CK_TYPE": 3,
    "INSTRUMENT_ID": -100000 - config.get('JPSS1_NORAD_ID'),
    "INCLUDE_INTERVAL_TABLE": "YES",
    "PRODUCER_ID": "Gavin Medley (for Libera SDC)",
}
# TODO: Consider storing these constants in config.json


def write_kernel_input_file(data: np.ndarray, filepath: str or Path, fields: list = None, fmt: str or list = "%.16f"):
    """Write ephemeris and attitude data to MKSPK and MSOPCK input data files, respectively.
    See MSOPCK documentation here:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/msopck.html
    See MKSPK documentation here:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/mkspk.html

    Parameters
    ----------
    data : np.ndarray
        Structured array (named, with data types) of attitude or ephemeris data.
    filepath : str or Path
        Filepath to write to.
    fields : list
        Optional. List of field names to write out to the data file. If not specified, assume fields are already
        in the proper order.
    fmt : str or list
        Format specifier(s) to pass to np.savetxt. Default is to assume everything should be floats with 16 decimal
        places of precision (%.16f). If a list is passed, it must contain a format specifier for each column in data.

    Returns
    -------
    : Path
        Absolute path to written file.
    """
    if fields:
        np.savetxt(filepath, data[fields], delimiter=" ", fmt=fmt)
    else:
        np.savetxt(filepath, data[:], delimiter=" ", fmt=fmt)
    return filepath.absolute()


def write_kernel_setup_file(data: dict, filepath: Path):
    """Write an MSOPCK or MKSPK compatible setup file of key-value pairs.
    See documentation here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/ug/msopck.html#Input%20Data%20Format

    Parameters
    ----------
    data : dict
        Dictionary of key-value pairs to write to the setup file.
    filepath : Path
        Filepath to write to.

    Returns
    -------
    : Path
        Absolute path to written file.
    """
    with open(filepath, 'w') as fh:
        fh.write("\\begindata\n")
        for key, value in data.items():
            if key in ('PATH_VALUES', 'PATH_SYMBOLS', 'KERNELS_TO_LOAD'):
                inside = ", ".join([f"\n\t'{item}'" for item in value])
                value_str = f"({inside}\n)"
            elif isinstance(value, str):
                value_str = f"'{value}'"
            elif isinstance(value, list):
                list_str = " ".join(value)
                value_str = f"'{list_str}'"
            elif isinstance(value, dict):
                dict_str = " ".join([f"\n\t'{k}={v}'" for k, v in value.items()])
                value_str = f"({dict_str}\n)"
            else:
                value_str = f"{value}"
            fh.write(f"{key}={value_str}\n")
        fh.write("\\begintext\n")
    return filepath.absolute()


# TODO: Write subprocess functions to call out to msopck and mkspk, first trying internally packaged binaries
#  (based on platform) and falling back to system pathed binaries.
