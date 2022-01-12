"""Pytest fixtures"""
# Standard
import logging
# Installed
import pytest
# Local
from libera_sdp import logutil

pytest_plugins = [
    "tests.plugins.data_path_fixtures",
    "tests.plugins.spice_fixtures",
    "tests.plugins.database_fixtures",
    "tests.plugins.s3_fixtures"
]


@pytest.fixture(scope='session')
def monkeypatch_session():
    """Provides a monkeypatch that applies for an entire pytest session (saves time)"""
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture
def setup_test_logging(tmp_path, monkeypatch):
    """Sets up a task logger for a test"""
    monkeypatch.setenv('LIBSDP_STREAM_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('LIBSDP_LOG_DIR', str(tmp_path))
    log_filepath = logutil.setup_task_logger('test_log')
    yield log_filepath
    # Remove all handlers from root logger. No other loggers should ever have handlers attached.
    logging.getLogger().handlers = []
