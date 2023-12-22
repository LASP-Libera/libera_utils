"""Module containing code to manage local file caching"""
# Standard
import os
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
    package_name = __name__.split('.', 1)[0]
    package_version = version(package_name)
    if system == 'darwin':
        path = Path('~/Library/Caches').expanduser()
        if package_name:
            path = path / package_name
    elif system.startswith('linux'):
        if os.getenv('XDG_CACHE_HOME'):
            path = Path(os.getenv('XDG_CACHE_HOME'))
        else:
            path = Path('~/.cache').expanduser()
        if package_name:
            path = path / package_name
    else:
        raise NotImplementedError("Only MacOS (darwin) and Linux (linux) platforms are currently supported. "
                                  f"Unsupported platform appears to be {system}")
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
