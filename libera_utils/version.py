"""Module for anything related to package versioning"""

import re
from importlib import metadata

ALGORITHM_VERSION_REGEX = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def version():
    """Get package version from metadata"""
    return metadata.version("libera_utils")
