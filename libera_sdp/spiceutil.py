"""Modules for SPICE kernel creation, management, and usage"""
# Standard
import datetime
import logging
import os
import re
import requests
import sys
from importlib.metadata import version
from pathlib import Path
from typing import NamedTuple
# Installed
import spiceypy as spice
from spiceypy.utils.exceptions import NotFoundError

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
        self.naif_base_url = naif_base_url if naif_base_url[-1] == '/' else naif_base_url + '/'
        self.kernel_file_regex = kernel_file_regex
        self.max_cache_age = max_cache_age
        self.fallback_kernel = fallback_kernel

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
        """Determine where to cache files based on the system and installed package version.

        Returns
        -------
        : Path
            Path to the cache directory for this version of this package on the current system
        """
        system = sys.platform
        package_name = __name__.split('.', 1)[0]
        package_version = version(package_name)
        if system == 'darwin':
            path = Path('~/Library/Caches').expanduser()
            if package_name:
                path = path / package_name
        elif system.startswith('linux'):
            path = os.getenv('XDG_CACHE_HOME', Path('~/.cache').expanduser())
            if package_name:
                path = path / package_name
        else:
            raise NotImplemented("Only MacOS (darwin) and Linux (linux) platforms are currently supported. "
                                 "Unsupported platform appears to be %s", system)
        if package_name and version:
            path = path / package_version
        return path

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
                if self.fallback_kernel:
                    return self.fallback_kernel
                else:
                    raise

    def clear(self):
        """Remove all cached files matching the kernel file regex from the cache directory"""
        for kernel in self.get_cached_kernels(include_stale=True):
            os.remove(kernel)

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
        download_url = self.naif_base_url + kernel_filename
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
