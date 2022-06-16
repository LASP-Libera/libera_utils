"""Module for anything related to package versioning"""
# Standard
from importlib import metadata


def version():
    """Get package version from metadata"""
    return metadata.version('libera_utils')
