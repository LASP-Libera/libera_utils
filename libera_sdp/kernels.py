"""Modules for SPICE kernel creation, management, and usage"""
# Standard
from pathlib import Path
# Installed
import numpy as np


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
