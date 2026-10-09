"""Pytest fixtures"""

import logging

import pytest

pytest_plugins = [
    "tests.plugins.data_path_fixtures",
    "tests.plugins.data_product_fixtures",
    "tests.plugins.spice_fixtures",
    "tests.plugins.aws_fixtures",
    "tests.plugins.manifest_fixtures",
    "tests.plugins.integration_test_fixtures",
    "tests.plugins.l1a_fixtures",
]


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Provides a monkeypatch that applies for an entire pytest session (saves time)"""
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture
def cleanup_loggers():
    """Ensures that root logging handlers are closed and removed after a test"""
    yield
    root = logging.getLogger()
    # Detaching a handler does not release the file it holds open. A RotatingFileHandler dropped
    # this way survives as cyclic garbage with its file still open, and whenever the collector
    # finally runs it, the ResourceWarning raised in __del__ surfaces as an unraisable exception
    # that pytest reports against whichever unrelated test happens to be running at the time.
    for handler in root.handlers:
        handler.close()
    root.handlers = []
