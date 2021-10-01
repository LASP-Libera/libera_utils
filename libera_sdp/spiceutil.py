"""Modules for SPICE kernel creation, management, and usage"""
# Standard
import logging
from typing import NamedTuple
# Installed
import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class KernelFileRecord(NamedTuple):
    """Tuple for keeping track of kernel files with default kernel_level"""
    kernel_type: str
    file_name: str

    def __str__(self):
        return "%06s %s" % (self.kernel_type, self.file_name)

    def __repr__(self):
        return f"KernelFileRecord({self.kernel_type}, {self.file_name})"


def ls_kernels(verbose: bool = False, log: bool = False) -> list:
    """
    List all furnished spice kernels.

    Parameters
    ----------
    verbose: bool
        If True, print to stdout also
    log: bool
        Whether or not to log the current kernel pool (this gets called a lot)

    Returns
    -------
    : list
        A list of KernelFileRecord named tuples.
    """
    count = spice.ktotal('ALL')
    if verbose:
        print(f"SPICE ktotal reports {count} kernels loaded")
    result = []
    for i in range(count):
        file, kernel_type, source, handle = spice.kdata(i, 'ALL')
        kfr = KernelFileRecord(kernel_type=kernel_type, file_name=file)
        if verbose:
            print(kfr)
        result.append(kfr)
    if log:
        formatted_kernels = "\n\t".join([str(kfr) for kfr in result])
        logger.debug(f"Kernels currently loaded:\n\t{formatted_kernels}")
    return result


def ls_spice_constants(verbose: bool = False) -> dict:
    """
    List all constants in the Spice constant pool

    Parameters
    ----------
    verbose:
        If true, print to stdout also

    Returns
    -------
    : dict
        Dictionary of kernel constants
    """
    try:
        kervars = spice.gnpool('*', 0, 1000, 81)
    except NotFoundError:  # Happens if there are no constants in the pool
        return {}

    result = {}
    for kervar in sorted(kervars):
        n, kernel_type = spice.dtpool(kervar)
        if verbose:
            print("%-50s %s %d" % (kervar, kernel_type, n))
        if type == 'N':
            values = spice.gdpool(kervar, 0, n)
            result[kervar] = values
            if verbose:
                print(values)
        elif type == 'C':
            values = spice.gcpool(kervar, 0, n, 81)
            result[kervar] = values
            if verbose:
                print(values)
    return result


def ls_kernel_coverage(kernel_type: str, verbose: bool = False) -> dict:
    """
    List time coverage of all furnished kernels of a given type

    Parameters
    ----------
    kernel_type: str
        Either 'CK' or 'SPK'
    verbose: bool
        If True, print to stdout also

    Returns
    -------
    : dict
        Key is filename, value is a list of tuples giving the start and end times in ET.
    """
    if kernel_type not in ('CK', 'SPK'):
        raise ValueError(f"Invalid kernel_type argument to ls_kernel_coverage {kernel_type}.")

    result = {}
    count = spice.ktotal(kernel_type)
    for i in range(count):
        file, _, source, handle = spice.kdata(i, kernel_type)
        result[file] = []
        if kernel_type.upper() == 'CK':
            ids = spice.ckobj(file)
        elif kernel_type.upper() == 'SPK':
            ids = spice.spkobj(file)

        for id in ids:
            cover = spice.cell_double(10000)
            if kernel_type.upper() == 'CK':
                cover = spice.ckcov(file, id, False, 'INTERVAL', 0.0, 'TDB', cover)
            else:
                cover = spice.spkcov(file, id, cover)
            card = spice.wncard(cover)
            for i_window in range(card):
                left, right = spice.wnfetd(cover, i_window)
                result[file].append((left, right))
                if verbose:
                    print("%s,%s,%d,%17.6f,%s,%17.6f,%s" % (kernel_type, file, id, left, spice.etcal(left), right, spice.etcal(right)))
    return result
