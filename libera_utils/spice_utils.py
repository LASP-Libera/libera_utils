"""Modules for SPICE kernel creation, management, and usage"""
# Standard
import datetime
from enum import Enum
import functools
import logging
import os
from pathlib import Path
import re
from typing import NamedTuple, Callable
import time
# Installed
from cloudpathlib import S3Path
import requests
import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError, SpiceyError
# Local
from libera_utils.config import config
from libera_utils.io import caching, smart_open

NAIF_PCK_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
NAIF_LSK_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/"
NAIF_DEVELOPMENT_EPHEMERIS_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
NAIF_HIGH_PREC_PCK_REGEX = "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc"
NAIF_LSK_REGEX = "naif[0-9]{4}.tls"
NAIF_DE_REGEX = "de[0-9]{3}.bsp"

logger = logging.getLogger(__name__)


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


class KernelFileCache:
    """Class for downloading, caching, and furnishing SPICE kernel files locally.

    It attempts to find a cached kernel file in the user's cache directory (OS-specific location).
    If that file is not there or is old, it attempts to download it from the specified location.
    If it is unable to do that, it can optionally read a fallback file included in the libera_utils package but this
    is not recommended.
    """

    def __init__(self, kernel_url: str or S3Path,
                 max_cache_age: datetime.timedelta = datetime.timedelta(days=1),
                 fallback_kernel: Path = None):
        """Create a new file cache. Downloading is done on first access of kernel_path if the file is not already
        cached. Fallback occurs only after failing to download.
        Parameters
        ----------
        kernel_url : str or cloudpathlib.S3Path
            Location of kernel file as a URL or an S3Path
        max_cache_age : datetime.timedelta
            Length of time to tolerate stale kernels in the cache without forcing a redownload.
        fallback_kernel : pathlib.Path
            Path pointing to a fallback kernel location. May be None, which disallows a fallback.
        """
        # Remove any trailing slash from naif_base_url (we assume no trailing slash in methods)
        self.kernel_url = kernel_url
        self.max_cache_age = max_cache_age
        self.fallback_kernel = fallback_kernel

    def __str__(self):
        return str(self.cache_dir / self.kernel_url)

    @property
    def kernel_basename(self):
        """Base filename of the kernel.

        Returns
        -------
        str
        """
        if isinstance(self.kernel_url, S3Path):
            return self.kernel_url.name
        return os.path.basename(self.kernel_url)

    @property
    def cache_dir(self):
        """Property that calls out to get the proper local cache directory

        Returns
        -------
        pathlib.Path
            Path to the proper local cache for the system.
        """
        return caching.get_local_cache_dir()

    @property
    def kernel_path(self) -> Path:
        """Return the local path location of the kernel if it exists. If not, try downloading it. If that
        fails, return the fallback kernel, if allowed."""
        if self.is_cached():
            return self.cache_dir / self.kernel_basename

        logger.info("No valid cached file %s in %s",
                    self.kernel_basename, self.cache_dir)
        try:
            downloaded_kernel = self.download_kernel(self.kernel_url)
            return downloaded_kernel
        except Exception as unhandled:
            logger.exception(unhandled)
            if self.fallback_kernel:
                logger.error("Error finding and downloading %s. Falling back to %s",
                             self.kernel_url, self.fallback_kernel)
                return self.fallback_kernel
            raise

    def furnsh(self):
        """Furnish the cached kernel"""
        spice.furnsh(str(self.kernel_path))

    def clear(self):
        """Remove cached kernel file"""
        logger.info(f"Removing cached file (if exists): {self.kernel_basename}")
        self.kernel_path.unlink(missing_ok=True)

    def is_cached(self, include_stale: bool = False) -> bool:
        """Check the cache directory for kernel file that is within cache age limit. If present, return True.

        Parameters
        ----------
        include_stale : bool
            Default False. If True, results include kernel that are past the max age.

        Returns
        -------
        bool
            Returns True if kernel is present locally and within the age limit.
        """
        presumptive_local_file = self.cache_dir / self.kernel_basename
        if presumptive_local_file.exists():
            last_modified = datetime.datetime.fromtimestamp(presumptive_local_file.stat().st_mtime)
            if include_stale or (datetime.datetime.now() - last_modified < self.max_cache_age):
                return True
            return False
        return False

    def download_kernel(self, kernel_url: str or S3Path, allowed_attempts: int = 3) -> Path:
        """Downloads a kernel from a URL or an S3 location to the system cache location.

        Parameters
        ----------
        kernel_url : str
            Filename of kernel on NAIF site, as discovered by find_most_recent_naif_kernel
        allowed_attempts : int, Optional
            Number of allowed download times for naif kernel default = 3

        Returns
        -------
        pathlib.Path
            Location of downloaded file
        """
        kernel_name = kernel_url.name if isinstance(kernel_url, S3Path) else os.path.basename(kernel_url)
        local_filepath = self.cache_dir / kernel_name

        # If kernel_url is an S3 object location
        if smart_open.is_s3(kernel_url):
            with smart_open.smart_open(kernel_url) as s3_object:
                with local_filepath.open('wb') as local_object:
                    local_object.write(s3_object.read())
        elif isinstance(kernel_url, str):
            # Else, treat as URL string
            if not local_filepath.parent.exists():
                local_filepath.parent.mkdir(parents=True)

            attempt_number = 1
            while attempt_number <= allowed_attempts:
                try:
                    with requests.get(kernel_url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with open(local_filepath, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    break
                except requests.exceptions.RequestException as error:
                    logger.info(f"Request failed. {error}")
                    if attempt_number < allowed_attempts:
                        logger.info(f"Trying again, retries left {allowed_attempts - attempt_number}, "
                                    f"Exception: {error}")
                        time.sleep(1)
                    else:
                        logger.error(f"Failed to download file after {allowed_attempts} attempts, Final Error: {error}")
                        raise
                attempt_number += 1

            logger.info("Cached kernel file to %s", local_filepath)
        else:
            raise ValueError(f"Kernel URL must be of type S3Path or str (for URL). Got {type(kernel_url)}")
        return local_filepath


def find_most_recent_naif_kernel(naif_base_url: str, kernel_file_regex: str, allowed_attempts: int = 3) -> str:
    """Retrieves the name of the most recent kernel at NAIF.

    Parameters
    ----------
    naif_base_url : str
        URL to search for filenames matching kernel_file_regex
    kernel_file_regex : str
        Regular expression to match filenames on the naif website
    allowed_attempts : int, Optional
        Number of allowed download times for naif page default = 3

    Returns
    -------
    str
        Returns the file name of the latest kernel on the naif page (e.g., "naif0012.tls")
    """
    kernel_link_regex = re.compile(f'href="({kernel_file_regex})"')

    attempt_number = 1
    while attempt_number <= allowed_attempts:
        try:
            resp = requests.get(naif_base_url, timeout=30)
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as error:
            if attempt_number < allowed_attempts:
                logger.info(f"{error} occurred trying again, retries left {allowed_attempts - attempt_number}")
                time.sleep(1)
            else:
                logger.error(f"Failed to download file after {allowed_attempts} attempts, Final Error: {error}")
                raise
        attempt_number += 1

    file_names = re.findall(kernel_link_regex, resp.text)
    if len(file_names) == 0:
        raise ValueError(f'No files were found on the NAIF page: {naif_base_url}')

    file_names.sort()  # NAIF filenames sort properly
    logger.debug('Found files on NAIF page: %r', file_names)

    return os.path.join(naif_base_url, file_names[-1])


class KernelFileRecord(NamedTuple):
    """Tuple for keeping track of kernel files with default kernel_level"""
    kernel_type: str
    file_name: str

    def __str__(self):
        return f"{self.kernel_type:<6} {self.file_name}"

    def __repr__(self):
        return f"KernelFileRecord({self.kernel_type}, {self.file_name})"


def ensure_spice(f_py: Callable = None, time_kernels_only: bool = False):
    # TODO: revisit this interface. It works well for time kernels currently (LSK/SCLK) but we haven't figured out
    #  exactly how we want to use it for SPK and CK files.
    #  Perhaps this decorator should only be smart enough to check for generic kernels?
    """
    Before trying to understand this piece of code, read this:
    https://stackoverflow.com/questions/5929107/decorators-with-parameters/60832711#60832711

    Decorator/wrapper that tries to ensure that a metakernel is furnished in as complete a way as possible.

    **Control flow overview:**

    1. Try simply calling the wrapped function naively.
        * SUCCESS? Great! We're done.
        * SpiceyError? Go to step 2.

    2. Furnish metakernel at SPICE_METAKERNEL
        * SUCCESS? Great, return the original function again (so it can be re-run).
        * KeyError? Seems like SPICE_METAKERNEL isn't set, no problem. Go to step 3.

    **Usage:**

    Three ways to use this object

    1. A decorator with no arguments

    .. code-block:: python

        @ensure_spice
        def my_spicey_func(a, b):
            pass

    2. A decorator with parameters. This is useful
    if we only need the latest SCLK and LSK kernels for the function involved.

    .. code-block:: python

        @ensure_spice(time_kernels_only=True)
        def my_spicey_time_func(a, b):
            pass

    3. An explicit wrapper function, providing a dynamically set value for parameters, e.g. time_kernels_only

    .. code-block:: python

        wrapped = ensure_spice(spicey_func, time_kernels_only=True)
        result = wrapped(*args, **kwargs)

    Parameters
    ----------
    f_py: Callable
        The function requiring SPICE that we are going to wrap if being used explicitly,
        Otherwise None, in which case ensure_spice is being used, not as a function wrapper (see l2a_processing.py) but
        as a true decorator without an explicit function argument.
    time_kernels_only: bool, Optional
        Specify that we only need to furnish time kernels
        (if SPICE_METAKERNEL is set, we still just furnish that metakernel and assume the time kernels are included.

    Returns
    -------
    Callable
        Decorated function, with spice error handling
    """
    if f_py and not callable(f_py):
        raise ValueError(f"Received a non-callable object {f_py} as the f_py argument to ensure_spice. "
                         "f_py must be a callable object.")

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
                        lsk_url = find_most_recent_naif_kernel(NAIF_LSK_INDEX_URL, NAIF_LSK_REGEX)
                        lsk = KernelFileCache(lsk_url)
                        spice.furnsh(str(lsk.kernel_path))
                        spice.furnsh(config.get('JPSS_SCLK'))
                    else:
                        raise SpiceyError("When calling a function requiring SPICE, we failed to load a metakernel. "
                                          "SPICE_METAKERNEL is not set, and time_kernels_only is not set to True"
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
    list
        A list of KernelFileRecord named tuples.
    """
    count = spice.ktotal('ALL')
    if verbose:
        print(f"SPICE ktotal reports {count} kernels loaded")
    result = []
    for i in range(count):
        file, kernel_type, _, _ = spice.kdata(i, 'ALL')  # pylint: disable=W0632
        kfr = KernelFileRecord(kernel_type=kernel_type, file_name=file)
        if verbose:
            print(kfr)
        result.append(kfr)
    if log:
        formatted_kernels = "\n\t".join([str(kfr) for kfr in result])
        logger.debug("Kernels currently loaded:\n\t%s", formatted_kernels)
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
    dict
        Dictionary of kernel constants
    """
    try:
        kervars = spice.gnpool('*', 0, 1000, 81)
    except NotFoundError:  # Happens if there are no constants in the pool
        return {}

    result = {}
    for kervar in sorted(kervars):
        n, kernel_type = spice.dtpool(kervar)  # pylint: disable=W0632
        if verbose:
            print(f"{kervar:<50} {kernel_type} {n}")
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
    dict
        Key is filename, value is a list of tuples giving the start and end times in ET.
    """
    if kernel_type not in ('CK', 'SPK'):
        raise ValueError(f"Invalid kernel_type argument to ls_kernel_coverage {kernel_type}. Must be CK or SPK.")

    result = {}
    count = spice.ktotal(kernel_type)
    for i in range(count):
        file, _, _, _ = spice.kdata(i, kernel_type)  # pylint: disable=W0632
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
                    print(f"{kernel_type},{file},{kernel_id},"
                          f"{left:17.6f},{spice.etcal(left)},{right:17.6f},{spice.etcal(right)}")
    return result
