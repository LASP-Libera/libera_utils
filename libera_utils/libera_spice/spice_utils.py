"""Core SPICE utilities for kernel creation, inspection, and time conversions.

This module is home to the core SPICE and kernel utilities in libera_utils, providing the ``make_kernel()``
wrapper around Curryer's KernelCreator, kernel file caching and furnishing via ``KernelFileCache``,
the ``ensure_spice`` decorator, SPICE-based time conversion wrappers, and kernel inspection helpers.
"""

import datetime
import functools
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Collection
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast, overload

import numpy as np
import pandas as pd
import requests
import spiceypy as spice
from cloudpathlib import AnyPath, CloudPath, S3Path
from curryer import kernels
from spiceypy.utils.exceptions import NotFoundError, SpiceyError

from libera_utils.config import config
from libera_utils.io import caching, smart_open

# Type alias for paths (same as filenaming.PathType but defined here to avoid circular import)
PathType = CloudPath | Path

NAIF_PCK_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"
NAIF_LSK_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/"
NAIF_DEVELOPMENT_EPHEMERIS_INDEX_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
NAIF_HIGH_PREC_PCK_REGEX = "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc"
NAIF_LSK_REGEX = "naif[0-9]{4}.tls"
NAIF_DE_REGEX = "de[0-9]{3}s.bsp"
NAIF_PCK_REGEX = "pck[0-9]{5}.tpc"

logger = logging.getLogger(__name__)


# TODO[LIBSDC-611]: Revisit idea.
class SpiceId(NamedTuple):
    """Class that represents a unique identifier in the NAIF SPICE library"""

    strid: str
    numid: int


class SpiceBody(Enum):
    """Enum containing SPICE IDs for ephemeris bodies that we use."""

    JPSS = SpiceId("JPSS", config.get("JPSS_SC_ID"))
    SSB = SpiceId("SOLAR_SYSTEM_BARYCENTER", 0)
    SUN = SpiceId("SUN", 10)
    EARTH = SpiceId("EARTH", 399)
    EARTH_MOON_BARYCENTER = SpiceId("EARTH-MOON BARYCENTER", 3)


class SpiceInstrument(Enum):
    """Enum containing SPICE IDs for instrument geometries configured in the Instrument Kernel (IK)"""

    # TODO[LIBSDC-611]: We don't have an IK yet. Once we do we should add instrument names and IDs, like
    #  LIBERA_SW_RADIOMETER = SpiceId('LIBERA_SW_RADIOMETER', -143013301)
    #  Do the required reading on NAIF on how to assign IDs to instrument bodies in an IK,
    #  here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/Tutorials/pdf/individual_docs/25_ik.pdf
    #  and here: https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/kernel.html#Kernel%20Types
    pass


class SpiceFrame(Enum):
    """Enum containing SPICE IDs for reference frames, possibly defined in the Frame Kernel (FK)"""

    J2000 = SpiceId("J2000", 1)
    ITRF93 = SpiceId("ITRF93", 3000)
    # EARTH_FIXED is a generic frame used only by the internals of SPICE. See docs here:
    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html \
    #     #Appendix.%20High%20Precision%20Earth%20Fixed%20Frames
    # We mention it here only for consistency and documentation purposes.
    EARTH_FIXED = ITRF93


class KernelFileCache:
    """Download, cache, and furnish SPICE kernel files under the user cache directory.

    On first access of :attr:`kernel_path`, a valid cached copy (younger than
    :attr:`max_cache_age`) is reused; otherwise the kernel is materialized from
    its source into the cache and that path is returned.

    Supported sources for ``kernel_url``:

    * **HTTP(S) URL** — ``str`` starting with ``http://`` or ``https://``; fetched with ``requests``.
    * **S3** — ``s3://`` string or :class:`cloudpathlib.S3Path`; read via ``smart_open``.
    * **Local file** — :class:`pathlib.Path` or non-URL ``str`` path to an existing file; copied with
      :func:`shutil.copy2`. Relative paths are resolved with
      :meth:`pathlib.Path.expanduser` and :meth:`pathlib.Path.resolve` against the
      **process current working directory at the time** :meth:`download_kernel` runs
      (typically the first uncached read of :attr:`kernel_path`), not at construction time.

    If materialization fails and ``fallback_kernel`` is set, :attr:`kernel_path` may return
    that path instead (not recommended for production).
    """

    @staticmethod
    def _resolve_local_kernel_file(kernel_url: Path | str) -> Path:
        """Return an absolute path to an existing local kernel file.

        Parameters
        ----------
        kernel_url : pathlib.Path or str
            Path to a kernel file (not a directory).

        Returns
        -------
        pathlib.Path
            Resolved path suitable as the copy source.

        Raises
        ------
        FileNotFoundError
            If the resolved path is not a regular file.
        """
        local = Path(kernel_url) if isinstance(kernel_url, str) else kernel_url
        resolved = local.expanduser().resolve(strict=False)
        if not resolved.is_file():
            msg = f"Local kernel file not found: {resolved}"
            raise FileNotFoundError(msg)
        return resolved

    def __init__(
        self,
        kernel_url: str | Path | S3Path,
        max_cache_age: datetime.timedelta = datetime.timedelta(days=1),
        fallback_kernel: Path | None = None,
    ) -> None:
        """Create a cache handle; copying or download happens on first use of :attr:`kernel_path` if needed.

        Parameters
        ----------
        kernel_url : str or pathlib.Path or cloudpathlib.S3Path
            Remote URL, S3 location, or path to a local kernel file.
        max_cache_age : datetime.timedelta
            Maximum age of a cached file before it is treated as stale and replaced.
        fallback_kernel : pathlib.Path or None
            Optional path returned if materialization from ``kernel_url`` fails.
        """
        self.kernel_url = kernel_url
        self.max_cache_age = max_cache_age
        self.fallback_kernel = fallback_kernel

    def __str__(self) -> str:
        return str(self.cache_dir / self.kernel_basename)

    @property
    def kernel_basename(self) -> str:
        """Base filename of the kernel used in the cache directory.

        Returns
        -------
        str
            Filename only (no directory components).
        """
        if isinstance(self.kernel_url, S3Path):
            return self.kernel_url.name
        if isinstance(self.kernel_url, Path):
            return self.kernel_url.name
        return os.path.basename(self.kernel_url)

    @property
    def cache_dir(self) -> Path:
        """Directory where cached kernel files are stored.

        Returns
        -------
        pathlib.Path
            User-specific cache directory for this application.
        """
        return caching.get_local_cache_dir()

    @property
    def kernel_path(self) -> Path:
        """Path to the kernel in the cache, materializing it if necessary.

        Returns
        -------
        pathlib.Path
            Path to the cached file, or ``fallback_kernel`` if materialization failed and a fallback was set.
        """
        if self.is_cached():
            return self.cache_dir / self.kernel_basename

        logger.info("No valid cached file %s in %s", self.kernel_basename, self.cache_dir)
        try:
            downloaded_kernel = self.download_kernel(self.kernel_url)
            return downloaded_kernel
        except Exception as unhandled:
            logger.exception(unhandled)
            if self.fallback_kernel:
                logger.error(
                    "Error finding and downloading %s. Falling back to %s", self.kernel_url, self.fallback_kernel
                )
                return self.fallback_kernel
            raise

    def furnsh(self) -> None:
        """Load the cached kernel into the SPICE kernel pool via :func:`spiceypy.furnsh`."""
        spice.furnsh(str(self.kernel_path))

    def clear(self) -> None:
        """Remove the cached kernel file from the cache directory if it exists."""
        logger.info("Removing cached file (if exists): %s", self.kernel_basename)
        (self.cache_dir / self.kernel_basename).unlink(missing_ok=True)

    def is_cached(self, include_stale: bool = False) -> bool:
        """Return whether a usable cached copy of the kernel exists.

        Parameters
        ----------
        include_stale : bool, optional
            If True, treat kernels older than ``max_cache_age`` as still cached.

        Returns
        -------
        bool
            True if the cache file exists and is not stale (unless ``include_stale`` is True).
        """
        presumptive_local_file = self.cache_dir / self.kernel_basename
        if presumptive_local_file.exists():
            last_modified = datetime.datetime.fromtimestamp(presumptive_local_file.stat().st_mtime)
            if include_stale or (datetime.datetime.now() - last_modified < self.max_cache_age):
                return True
            return False
        return False

    def download_kernel(self, kernel_url: str | Path | S3Path, allowed_attempts: int = 3) -> Path:
        """Copy or download a kernel into the user cache directory.

        Parameters
        ----------
        kernel_url : str or pathlib.Path or cloudpathlib.S3Path
            Same kinds of sources as :class:`KernelFileCache` (S3, HTTP(S) URL, or local file).
        allowed_attempts : int, optional
            Retries for HTTP downloads only.

        Returns
        -------
        pathlib.Path
            Path to the file in the cache directory.

        Raises
        ------
        FileNotFoundError
            If ``kernel_url`` denotes a local path that does not exist or is not a file.
        requests.exceptions.RequestException
            If the HTTP download fails after all retries.
        ValueError
            If ``kernel_url`` is not a supported type or not an HTTP(S) URL when given as ``str``.
        """
        if isinstance(kernel_url, S3Path):
            kernel_name = kernel_url.name
        elif isinstance(kernel_url, Path):
            kernel_name = kernel_url.name
        else:
            kernel_name = os.path.basename(kernel_url)
        local_filepath = self.cache_dir / kernel_name

        if smart_open.is_s3(kernel_url) or isinstance(kernel_url, S3Path):
            if not local_filepath.parent.exists():
                local_filepath.parent.mkdir(parents=True)
            with smart_open.smart_open(kernel_url) as s3_object:
                with local_filepath.open("wb") as local_object:
                    local_object.write(s3_object.read())
        elif isinstance(kernel_url, Path) or (
            isinstance(kernel_url, str) and not kernel_url.startswith(("http://", "https://"))
        ):
            resolved = self._resolve_local_kernel_file(kernel_url)
            if not local_filepath.parent.exists():
                local_filepath.parent.mkdir(parents=True)
            shutil.copy2(resolved, local_filepath)
            # New mtime so is_cached() age checks match freshly downloaded kernels (sources
            # may carry old mtimes).
            local_filepath.touch()
            logger.info("Cached local kernel file to %s", local_filepath)
        elif isinstance(kernel_url, str) and kernel_url.startswith(("http://", "https://")):
            if not local_filepath.parent.exists():
                local_filepath.parent.mkdir(parents=True)

            attempt_number = 1
            while attempt_number <= allowed_attempts:
                try:
                    with requests.get(kernel_url, stream=True, timeout=30) as r:
                        r.raise_for_status()
                        with local_filepath.open("wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    break
                except requests.exceptions.RequestException as error:
                    logger.info("Request failed. %s", error)
                    if attempt_number < allowed_attempts:
                        logger.info(
                            "Trying again, retries left %s, Exception: %s",
                            allowed_attempts - attempt_number,
                            error,
                        )
                        time.sleep(1)
                    else:
                        logger.error(
                            "Failed to download file after %s attempts, Final Error: %s",
                            allowed_attempts,
                            error,
                        )
                        raise
                attempt_number += 1

            logger.info("Cached kernel file to %s", local_filepath)
        else:
            raise ValueError(
                f"Kernel source must be S3Path, s3:// str, http(s) URL str, or local Path/str. Got {type(kernel_url)}"
            )
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
        raise ValueError(f"No files were found on the NAIF page: {naif_base_url}")

    file_names.sort()  # NAIF filenames sort properly
    logger.debug("Found files on NAIF page: %r", file_names)

    return os.path.join(naif_base_url, file_names[-1])


_F = TypeVar("_F", bound=Callable[..., Any])


class KernelFileRecord(NamedTuple):
    """Tuple for keeping track of kernel files with default kernel_level"""

    kernel_type: str
    file_name: str

    def __str__(self):
        return f"{self.kernel_type:<6} {self.file_name}"

    def __repr__(self):
        return f"KernelFileRecord({self.kernel_type}, {self.file_name})"


@overload
def ensure_spice(f_py: _F, time_kernels_only: bool = False) -> _F: ...


@overload
def ensure_spice(
    f_py: None = None,
    *,
    time_kernels_only: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...


def ensure_spice(
    f_py: Callable[..., Any] | None = None,
    time_kernels_only: bool = False,
) -> Callable[..., Any] | Callable[[Callable[..., Any]], Callable[..., Any]]:
    # TODO[LIBSDC-614] LIBSDC-687: revisit this interface. It works well for time kernels currently (LSK/SCLK)
    #  but we haven't figured out exactly how we want to use it for SPK and CK files.
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
        raise ValueError(
            f"Received a non-callable object {f_py} as the f_py argument to ensure_spice. "
            f"f_py must be a callable object."
        )

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """This is either a decorator or a function wrapper, depending on how ensure_spice is being used"""

        @functools.wraps(func)
        def wrapper_ensure_spice(*args: Any, **kwargs: Any) -> Any:
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
                    metakernel_path = os.environ["SPICE_METAKERNEL"]
                    spice.furnsh(metakernel_path)
                except KeyError:
                    if time_kernels_only:
                        lsk_url = find_most_recent_naif_kernel(NAIF_LSK_INDEX_URL, NAIF_LSK_REGEX)
                        lsk = KernelFileCache(lsk_url)
                        spice.furnsh(str(lsk.kernel_path))
                        spice.furnsh(config.get("JPSS_SCLK"))
                    else:
                        raise SpiceyError(
                            "When calling a function requiring SPICE, we failed to load a metakernel. "
                            "SPICE_METAKERNEL is not set, and time_kernels_only is not set to True"
                        ) from spcy_err
                return func(*args, **kwargs)

        return wrapper_ensure_spice

    return _decorator(f_py) if callable(f_py) else _decorator


def ls_kernels(verbose: bool = False, log: bool = False) -> list[KernelFileRecord]:
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
    count = spice.ktotal("ALL")
    if verbose:
        print(f"SPICE ktotal reports {count} kernels loaded")
    result = []
    for i in range(count):
        file, kernel_type, _, _ = spice.kdata(i, "ALL")
        kfr = KernelFileRecord(kernel_type=kernel_type, file_name=file)
        if verbose:
            print(kfr)
        result.append(kfr)
    if log:
        formatted_kernels = "\n\t".join([str(kfr) for kfr in result])
        logger.debug("Kernels currently loaded:\n\t%s", formatted_kernels)
    return result


def ls_spice_constants(verbose: bool = False) -> dict[str, Any]:
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
        kervars = spice.gnpool("*", 0, 1000, 81)
    except NotFoundError:  # Happens if there are no constants in the pool
        return {}

    result = {}
    for kervar in sorted(kervars):
        n, kernel_type = spice.dtpool(kervar)  # pylint: disable=W0632
        if verbose:
            print(f"{kervar:<50} {kernel_type} {n}")
        if kernel_type == "N":
            values = spice.gdpool(kervar, 0, n)
            result[kervar] = values
            if verbose:
                print(values)
        elif kernel_type == "C":
            values = spice.gcpool(kervar, 0, n, 81)
            result[kervar] = values
            if verbose:
                print(values)
    return result


def ls_kernel_coverage(kernel_type: str, verbose: bool = False) -> dict[str, list[tuple[float, float]]]:
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
    if kernel_type not in ("CK", "SPK"):
        raise ValueError(f"Invalid kernel_type argument to ls_kernel_coverage {kernel_type}. Must be CK or SPK.")

    result = {}
    count = spice.ktotal(kernel_type)
    for i in range(count):
        file, _, _, _ = spice.kdata(i, kernel_type)  # pylint: disable=W0632
        result[file] = []
        if kernel_type.upper() == "CK":
            ids = spice.ckobj(file)
        else:  # Must be SPK
            ids = spice.spkobj(file)

        for kernel_id in ids:
            cover = spice.cell_double(10000)
            if kernel_type.upper() == "CK":
                cover = spice.ckcov(file, kernel_id, False, "INTERVAL", 0.0, "TDB", cover)
            else:  # Must be SPK
                cover = spice.spkcov(file, kernel_id, cover)
            card = spice.wncard(cover)
            for i_window in range(card):
                left, right = spice.wnfetd(cover, i_window)
                result[file].append((left, right))
                if verbose:
                    print(
                        f"{kernel_type},{file},{kernel_id},"
                        f"{left:17.6f},{spice.etcal(left)},{right:17.6f},{spice.etcal(right)}"
                    )
    return result


def ls_all_kernel_coverage(as_datetime: bool = True, verbose: bool = False) -> dict[str, Any]:
    """
    List time coverage of all furnished kernels

    Parameters
    ----------
    verbose: bool
        If True, print to stdout also

    Returns
    -------
    dict
        Key is filename, value is a list of tuples giving the start and end times in ET.
    """
    result = {}
    for kernel_type in ("CK", "SPK"):
        result.update(ls_kernel_coverage(kernel_type, verbose))

    # Convert times to UTC strings if requested
    if as_datetime:
        for file in result:
            utc_tuples = []
            for left, right in result[file]:
                left_utc = spice.et2datetime(left)
                right_utc = spice.et2datetime(right)
                utc_tuples.append((left_utc, right_utc))
            result[file] = utc_tuples
    return result


def make_kernel(
    config_file: str | Path,
    output_kernel: str | PathType,
    input_data: pd.DataFrame | None = None,
    overwrite: bool = False,
    append: bool | int = False,
) -> PathType:
    """Create a binary SPICE kernel (CK or SPK) from a JSON configuration file and input data.

    This is a low-level utility that wraps Curryer's KernelCreator, which drives the
    NAIF command-line tools ``mkspk`` (for SPK ephemeris kernels) and ``msopck``
    (for CK attitude kernels). It is used for creating binary CK and SPK kernels,
    including fixed-offset static SPKs and dynamic CK/SPK kernels.

    This function does NOT create text kernels such as LSKs (.tls), PCKs (.tpc),
    frame kernels (.tf), instrument kernels (.ti), or clock kernels (.tsc). Those
    are plain-text files managed separately.

    Callers are responsible for ensuring required kernels (especially LSK and any
    relevant frame kernels) are furnished before calling this function.

    Parameters
    ----------
    config_file : str | Path
        JSON configuration file defining how to create the kernel.
    output_kernel : str | PathType
        Output directory or file to create the kernel. If a directory, the
        file name will be based on the config_file, but with the SPICE file
        extension.
    input_data : pd.DataFrame | None
        pd.DataFrame containing kernel input data. If not supplied, the config
        is assumed to reference an input data file.
    overwrite : bool
        Option to overwrite an existing file.
    append : bool | int
        Option to append to an existing file. Anything truthy will be treated as True.

    Returns
    -------
    PathType
        Output kernel file path

    Notes
    -----
    This function requires a leap second kernel (LSK) to be furnished by SPICE
    before it can convert times. Callers should use KernelManager.load_naif_kernels()
    or similar to ensure kernels are ready before calling this function.
    """
    output_kernel = cast(PathType, AnyPath(output_kernel))
    config_file = Path(config_file)  # This is always a local path because the configs are package data

    # Create the kernels from the JSONs definitions.
    creator = kernels.create.KernelCreator(overwrite=overwrite, append=bool(append))

    with tempfile.TemporaryDirectory(prefix="/tmp/") as tmp_dir:  # nosec B108
        tmp_path = Path(tmp_dir)
        if output_kernel.is_file():
            tmp_path = tmp_path / output_kernel.name

        out_fn = creator.write_from_json(config_file, output_kernel=tmp_path, input_data=input_data)

        # Use smart copy here to avoiding using two nested smart_open calls
        # one call would be to open the newly created file, and one to open the desired location
        if output_kernel.is_dir():
            output_kernel = output_kernel / out_fn.name
        smart_open.smart_copy_file(out_fn, output_kernel)
        logger.info("Kernel copied to %s", output_kernel)
    return output_kernel


# SPICE Time Conversion Functions (moved from time.py)


def et_2_timestamp(
    et: float | Collection[float] | np.ndarray,
    fmt: str = "%Y%m%dT%H%M%S.%f",
) -> str | np.ndarray:
    """
    Convert ephemeris time to a custom formatted timestamp (default is lowercase version of ISO).

    Parameters
    ----------
    et: float | Collection[float] | numpy.ndarray
        Ephemeris Time to be converted.
    fmt: str, Optional
        Format string as defined by the datetime.strftime() function.

    Returns
    -------
    : str | numpy.ndarray
        Formatted timestamps
    """

    datetime_objs = et_2_datetime(et)

    if isinstance(datetime_objs, Collection):
        time_out = np.array([t.strftime(fmt) for t in datetime_objs])
    else:
        time_out = datetime_objs.strftime(fmt)

    return time_out


def et_2_datetime(et: float | Collection[float] | np.ndarray) -> datetime.datetime | np.ndarray:
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

    isoc_fmt = "%Y-%m-%dT%H:%M:%S.%f"
    isoc_prec = 6

    isoc_timestamp = et2utc_wrapper(et, "ISOC", isoc_prec)
    if isinstance(et, Collection):
        return np.array([datetime.datetime.strptime(s, isoc_fmt) for s in isoc_timestamp])

    return datetime.datetime.strptime(isoc_timestamp, isoc_fmt)


@ensure_spice(time_kernels_only=True)
def et2utc_wrapper(et: float | Collection[float] | np.ndarray, fmt: str, prec: int) -> str | np.ndarray:
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
    : str or numpy.ndarray
        UTC time string(s)
    """
    return spice.et2utc(et, fmt, prec)


@ensure_spice(time_kernels_only=True)
def utc2et_wrapper(iso_str: str | Collection[str]) -> float | np.ndarray:
    """
    Convert UTC ISO strings to ephemeris times.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/utc2et_c.html
    Decorated wrapper for spiceypy.utc2et that will automatically furnish the latest metakernel and retry
    if the first call raises an exception.

    Parameters
    ----------
    iso_str: str or Collection[str]
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
def scs2e_wrapper(sclk_str: str | Collection[str]) -> float | np.ndarray:
    """
    Convert SCLK strings to ephemeris time.
    https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/scs2e_c.html
    Decorated wrapper for spiceypy.scs2e that will automatically furnish the latest metakernel and retry
    if the first call raises an exception.

    Parameters
    ----------
    sclk_str: str or Collection[str]
        Spacecraft clock string

    Returns
    -------
    : float or numpy.ndarray
        Ephemeris time
    """

    sc_id = config.get("JPSS_SC_ID")
    if isinstance(sclk_str, str):
        return spice.scs2e(sc_id, sclk_str)

    return np.array([spice.scs2e(sc_id, s) for s in sclk_str])


@ensure_spice(time_kernels_only=True)
def sce2s_wrapper(et: float | Collection[float] | np.ndarray) -> str | np.ndarray:
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
    : str or numpy.ndarray
        SCLK string
    """

    sc_id = config.get("JPSS_SC_ID")
    if isinstance(et, Collection):
        return np.array([spice.sce2s(sc_id, t) for t in et])

    return spice.sce2s(sc_id, et)
