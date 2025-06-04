"""Pytest fixtures"""

import logging

import pytest

pytest_plugins = [
    "tests.plugins.data_path_fixtures",
    "tests.plugins.spice_fixtures",
    "tests.plugins.aws_fixtures",
    "tests.plugins.manifest_fixtures",
    "tests.plugins.integration_test_fixtures",
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
    """Ensures that root logging handlers are removed after a test"""
    yield
    root = logging.getLogger()
    root.handlers = []
