"""Module containing code to manage local file caching"""

import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path


def get_local_cache_dir():
    """Determine where to cache files based on the system and installed package version.

    Returns
    -------
    : pathlib.Path
        Path to the cache directory for this version of this package on the current system
    """
    system = sys.platform
    package_name = __name__.split(".", 1)[0]
    package_version = version(package_name)
    if system == "darwin":
        path = Path("~/Library/Caches").expanduser()
        if package_name:
            path = path / package_name
    elif system.startswith("linux"):
        if os.getenv("XDG_CACHE_HOME"):
            path = Path(os.getenv("XDG_CACHE_HOME"))
        else:
            path = Path("~/.cache").expanduser()
        if package_name:
            path = path / package_name
    else:
        raise NotImplementedError(
            "Only MacOS (darwin) and Linux (linux) platforms are currently supported. "
            f"Unsupported platform appears to be {system}"
        )
    if package_name and package_version:
        path = path / package_version
    return path


def empty_local_cache_dir():
    """Remove all cached files in the local cache.

    Returns
    -------
    list
        List of removed files
    """
    removed_files = []
    local_cache = get_local_cache_dir()
    for cached_file in local_cache.glob("*"):
        os.remove(cached_file)
        removed_files.append(cached_file)
    return removed_files


def get_local_short_temp_dir() -> Path:
    """
    Get a short base directory for temporary files.

    Returns a platform-appropriate short path that respects
    environment variables for customization.

    Returns
    -------
    Path
        A short base path for temporary directories.
    """
    # Allow user override via environment variable
    if "LIBERA_TEMP_DIR" in os.environ:
        custom_path = Path(os.environ["LIBERA_TEMP_DIR"])
        custom_path.mkdir(parents=True, exist_ok=True)
        return custom_path

    # Platform-specific short defaults
    system = platform.system()

    if system == "Windows":
        # Use C:\Temp instead of the deep AppData path
        short_base = Path("C:/Temp")
    else:
        # Unix-like systems to use /tmp directly
        short_base = Path("/tmp/")  # noqa: S108

    # Create if it doesn't exist
    short_base.mkdir(parents=True, exist_ok=True)

    return short_base


def validate_path_length(path: Path, max_length: int = 80) -> None:
    """
    Validate that a path doesn't exceed a maximum length. This is used primarily for SPICE kernels to avoid
    issues with overly long file paths (>80 characters) that fail in the C implementation of SPICE.

    Parameters
    ----------
    path : Path
        The path to validate.
    max_length : int
        Maximum allowed path length (default: 80 for SPICE).

    Raises
    ------
    RuntimeError
        If path exceeds maximum length with helpful error message.
    """
    path_str = str(path.resolve())
    if len(path_str) > max_length:
        raise RuntimeError(
            f"Path length ({len(path_str)}) exceeds limit ({max_length}):\n"
            f"  {path_str}\n"
            f"Consider setting LIBERA_TEMP_DIR to a shorter base path."
        )
