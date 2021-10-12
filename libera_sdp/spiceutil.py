"""Modules for SPICE kernel creation, management, and usage"""
# Standard
import datetime
import functools
import logging
import os
import re
from enum import Enum

import requests
from pathlib import Path
from typing import NamedTuple
# Installed
import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError, SpiceyError
# Local
from libera_sdp.config import config
from libera_sdp.io import caching

NAIF_PCK_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
NAIF_LSK_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/"
NAIF_DEVELOPMENT_EPHEMERIS_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
NAIF_HIGH_PREC_PCK_REGEX = "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc"
NAIF_LSK_REGEX = "naif[0-9]{4}.tls"
NAIF_DE_REGEX = "de[0-9]{3}.bsp"

logger = logging.getLogger(__name__)


class KernelFileCache:
    """Class for downloading and caching SPICE kernel files from NAIF.

    It attempts to find a cached kernel file in the user's cache directory (OS-specific location).
    If that file is not there or is old, it attempts to download the latest kernel from NAIF.
    If it is unable to do that, it can optionally read a fallback file included in the libera_sdp package.
    """

    def __init__(self, naif_base_url: str, kernel_file_regex: str,
                 max_cache_age: datetime.timedelta = datetime.timedelta(days=1),
                 fallback_kernel: Path = None):
        """Create a new file cache. Downloading/fallback is done on first access if necessary.
        Parameters
        ----------
        naif_base_url : str
            The base url where this type of file lives at NAIF
        kernel_file_regex : str
            Regex pattern for matching filenames in the base url at NAIF. This gets combined with "href=<regex_str>".
        max_cache_age : datetime.timedelta
            Length of time to tolerate stale kernels in the cache without forcing a redownload.
        fallback_kernel : Path
            Path pointing to a fallback kernel location. May be None, which disallows a fallback.
        """
        # Remove any trailing slash from naif_base_url (we assume no trailing slash in methods)
        self.naif_base_url = naif_base_url if naif_base_url[-1] != '/' else naif_base_url[0:-1]
        self.kernel_file_regex = kernel_file_regex
        self.max_cache_age = max_cache_age
        self.fallback_kernel = fallback_kernel

    def __str__(self):
        return str(self.cache_dir / self.kernel_file_regex)

    @property
    def kernel_link_regex(self):
        """Compiled regex for finding links to matching files for download.

        Returns
        -------
        re.Pattern
            Compiled regex pattern for finding links to downloadable files on NAIF pages
        """
        return re.compile(r'href="({0})"'.format(self.kernel_file_regex))

    @property
    def cache_dir(self):
        """Property that calls out to get the proper local cache directory

        Returns
        -------
        Path
            Path to the proper local cache for the system.
        """
        return caching.get_local_cache_dir()

    @property
    def kernel_path(self) -> Path:
        """Return the get_cached_kernels path, as set by the _get_from_naif method."""
        cached_valid_kernels = self.get_cached_kernels()
        if cached_valid_kernels:
            return cached_valid_kernels[-1]
        else:
            logger.info(f"No valid cached files for {self.kernel_file_regex} in {self.cache_dir}")
            try:
                most_recent_kernel = self.find_most_recent_kernel()
                downloaded_kernel = self.download_kernel(most_recent_kernel)
                return downloaded_kernel
            except Exception as unhandled:
                logger.exception(unhandled)
                if self.fallback_kernel:
                    logger.error(
                        f"Error finding and downloading the most recent kernel matching {self.kernel_file_regex}. "
                        f"Falling back to {self.fallback_kernel}")
                    return self.fallback_kernel
                else:
                    raise

    def furnsh(self):
        """Furnish the cached kernel"""
        spice.furnsh(str(self.kernel_path))

    def clear(self):
        """Remove all cached files matching the kernel file regex from the cache directory"""
        removed_kernels = []
        for kernel in self.get_cached_kernels(include_stale=True):
            os.remove(kernel)
            removed_kernels.append(kernel)
        return removed_kernels

    def get_cached_kernels(self, include_stale: bool = False) -> list:
        """Check the cache directory for files matching the given kernel file regex and within cache age limit.

        Parameters
        ----------
        include_stale : bool
            Default False. If True, results include kernel that are past the max age.

        Returns
        -------
        list
            List of Paths to valid cached kernel files (matching regex and optionally within age limit).
        """
        valid_kernels = []
        for file in self.cache_dir.glob("*"):
            last_modified = datetime.datetime.fromtimestamp(file.stat().st_mtime)
            if re.search(self.kernel_file_regex, file.name):
                if include_stale or datetime.datetime.now() - last_modified < self.max_cache_age:
                    valid_kernels.append(file)
        valid_kernels.sort()
        return valid_kernels

    def download_kernel(self, kernel_filename: str) -> Path:
        """

        Parameters
        ----------
        kernel_filename : str
            Filename of kernel on NAIF site, as discovered by find_most_recent_kernel

        Returns
        -------
        Path
            Location of downloaded file
        """
        download_url = self.naif_base_url + "/" + kernel_filename
        local_filepath = self.cache_dir / kernel_filename
        if not local_filepath.parent.exists():
            local_filepath.parent.mkdir(parents=True)
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        logger.info(f"Cached kernel file to {local_filepath}")
        return local_filepath

    def find_most_recent_kernel(self) -> str:
        """Retrieves the name of the most recent kernel at NAIF.

        Returns
        -------
        str
            Returns the file name of the latest kernel (e.g., "naif0012.tls")
        """
        resp = requests.get(self.naif_base_url)
        resp.raise_for_status()

        file_names = re.findall(self.kernel_link_regex, resp.text)
        if len(file_names) == 0:
            raise ValueError(f'No files were found on the NAIF page: {self.naif_base_url}')

        file_names.sort()
        logger.debug('Found files on NAIF page: %r', file_names)

        return file_names[-1]


class KernelFileRecord(NamedTuple):
    """Tuple for keeping track of kernel files with default kernel_level"""
    kernel_type: str
    file_name: str

    def __str__(self):
        return "%06s %s" % (self.kernel_type, self.file_name)

    def __repr__(self):
        return f"KernelFileRecord({self.kernel_type}, {self.file_name})"


def ensure_spice(f_py: callable = None, time_kernels_only: bool = False):
    # FIXME: revisit this interface. It works well for time kernels currently (LSK/SCLK) but we haven't figured out
    #  exactly how we want to use it for SPK and CK files.
    #  Perhaps this decorator should only be smart enough to check for generic kernels?
    """
    Before trying to understand this piece of code, read this:
    https://stackoverflow.com/questions/5929107/decorators-with-parameters/60832711#60832711

    Decorator/wrapper that tries to ensure that a metakernel is furnished in as complete a way as possible.

    ======================
    Control flow overview:
    ======================
    1. Try simply calling the wrapped function naively.
    --> SUCCESS? Great! We're done.
    --> SpiceyError? Go to step 2.

    2. Furnish metakernel at SPICE_METAKERNEL
    --> SUCCESS? Great, return the original function again (so it can be re-run).
    --> KeyError? Seems like SPICE_METAKERNEL isn't set, no problem. Go to step 3.

    ======
    Usage:
    ======
    Three ways to use this object:
    1) A decorator with no arguments
    ```
    @ensure_spice
    def my_spicey_func(a, b):
        pass
    ```
    2) A decorator with parameters. This is useful
    if we only need the latest SCLK and LSK kernels for the function involved.
    ```
    @ensure_spice(time_kernels_only=True)
    def my_spicey_time_func(a, b):
        pass
    ```
    3) An explicit wrapper function, providing a dynamically set value for parameters, e.g. time_kernels_only
    ```
    wrapped = ensure_spice(spicey_func, time_kernels_only=True)
    result = wrapped(*args, **kwargs)
    ```

    Parameters
    ----------
    f_py: callable
        The function requiring SPICE that we are going to wrap if being used explicitly,
        Otherwise None, in which case ensure_spice is being used, not as a function wrapper (see l2a_processing.py) but
        as a true decorator without an explicit function argument.
    time_kernels_only: bool, optional
        Specify that we only need to furnish time kernels
        (if SPICE_METAKERNEL is set, we still just furnish that metakernel and assume the time kernels are included.

    Returns
    -------
    : callable
        Decorated function, with spice error handling
    """
    assert callable(f_py) or f_py is None  # If this is set, it must be a callable object

    def _decorator(func):
        """This is either a decorator or a function wrapper, depending on how ensure_spice is being used"""

        @functools.wraps(func)
        def wrapper_ensure_spice(*args, **kwargs):
            """
            This function wraps the actual function that ensure_spice is wrapping/decorating. *args and **kwargs
            refer to those passed to the decorated function.
            """
            try:
                # Step 1.
                return func(*args, **kwargs)  # Naive first try. Maybe SPICE is already furnished.
            except SpiceyError as spcy_err:
                try:
                    # Step 2.
                    metakernel_path = os.environ['SPICE_METAKERNEL']
                    spice.furnsh(metakernel_path)
                except KeyError:
                    if time_kernels_only:
                        lsk = KernelFileCache(NAIF_LSK_INDEX_URL, NAIF_LSK_REGEX)
                        spice.furnsh(str(lsk.kernel_path))
                        spice.furnsh(config.get('JPSS_SCLK'))
                    else:
                        raise SpiceyError(f"When calling a function requiring SPICE, we failed to load a metakernel. "
                                          f"SPICE_METAKERNEL is not set, and time_kernels_only is not set to True"
                                          ) from spcy_err
                return func(*args, **kwargs)
        return wrapper_ensure_spice
    return _decorator(f_py) if callable(f_py) else _decorator


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
        if kernel_type == 'N':
            values = spice.gdpool(kervar, 0, n)
            result[kervar] = values
            if verbose:
                print(values)
        elif kernel_type == 'C':
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
        raise ValueError(f"Invalid kernel_type argument to ls_kernel_coverage {kernel_type}. Must be CK or SPK.")

    result = {}
    count = spice.ktotal(kernel_type)
    for i in range(count):
        file, _, source, handle = spice.kdata(i, kernel_type)
        result[file] = []
        if kernel_type.upper() == 'CK':
            ids = spice.ckobj(file)
        else:  # Must be SPK
            ids = spice.spkobj(file)

        for kernel_id in ids:
            cover = spice.cell_double(10000)
            if kernel_type.upper() == 'CK':
                cover = spice.ckcov(file, kernel_id, False, 'INTERVAL', 0.0, 'TDB', cover)
            else:  # Must be SPK
                cover = spice.spkcov(file, kernel_id, cover)
            card = spice.wncard(cover)
            for i_window in range(card):
                left, right = spice.wnfetd(cover, i_window)
                result[file].append((left, right))
                if verbose:
                    print("%s,%s,%d,%17.6f,%s,%17.6f,%s" %
                          (kernel_type, file, kernel_id, left, spice.etcal(left), right, spice.etcal(right)))
    return result


class SpiceId(NamedTuple):
    """Class that represents a unique identifier in the NAIF SPICE library"""
    strid: str
    numid: int


class SpiceBody(Enum):
    """Enum containing SPICE IDs for ephemeris bodies that we use."""
    JPSS = SpiceId('JPSS', config.get('JPSS_SC_ID'))
    SSB = SpiceId('SOLAR_SYSTEM_BARYCENTER', 0)
    SUN = SpiceId('SUN', 10)
    EARTH = SpiceId('EARTH', 399)
    EARTH_MOON_BARYCENTER = SpiceId('EARTH-MOON BARYCENTER', 3)


class SpiceInstrument(Enum):
    """Enum containing SPICE IDs for instrument geometries configured in the Instrument Kernel (IK)"""
    # TODO: We don't have an IK yet. Once we do we should add instrument names and IDs, like
    #  LIBERA_SW_RADIOMETER = SpiceId('LIBERA_SW_RADIOMETER', -143013301)
    #  Do the required reading on NAIF on how to assign IDs to instrument bodies in an IK,
    #  here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/25_ik.pdf
    #  and here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Kernel%20Types
    pass


class SpiceFrame(Enum):
    """Enum containing SPICE IDs for reference frames, possibly defined in the Frame Kernel (FK)"""
    J2000 = SpiceId('J2000', 1)
    ITRF93 = SpiceId('ITRF93', 3000)
    # EARTH_FIXED is a generic frame used only by the internals of SPICE. See docs here:
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html \
    #     #Appendix.%20High%20Precision%20Earth%20Fixed%20Frames
    # We mention it here only for consistency and documentation purposes.
    EARTH_FIXED = ITRF93
