"""Pytest fixtures"""
from pathlib import Path
import sys

import pytest


@pytest.fixture
def test_data_dir():
    """Returns the test data directory

    Returns
    -------
    : Path
    """
    return Path(sys.modules[__name__.split('.')[0]].__file__).parent / 'test_data'
