"""Module for anything related to package versioning"""

from importlib import metadata


def version():
    """Get package version from metadata"""
    return metadata.version("libera_utils")
