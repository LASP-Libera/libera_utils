"""Pytest fixtures"""
# Standard
import logging
# Installed
import pytest
# Local
from libera_utils import logutil

pytest_plugins = [
    "tests.plugins.data_path_fixtures",
    "tests.plugins.spice_fixtures",
    "tests.plugins.database_fixtures",
    "tests.plugins.aws_fixtures"
]


@pytest.fixture(scope='session')
def monkeypatch_session():
    """Provides a monkeypatch that applies for an entire pytest session (saves time)"""
    from _pytest.monkeypatch import MonkeyPatch
    m = MonkeyPatch()
    yield m
    m.undo()


@pytest.fixture
def configure_example_logging(mock_cloudwatch_context, tmp_path, test_data_path):
    """Sets up an example parameterized, configured logger and clear out all handlers afterwards"""
    logging.getLogger().handlers = []
    params = {
        "handlers": {
            "logfile": {"filename": f"{tmp_path / 'test.log'}"},
            "watchtower": {"log_group_name": "test-log-group", "log_stream_name": "test-log-stream"}
        },
        "root": {"handlers": ["console", "logfile"]}
    }
    logutil.configure_logging(test_data_path / 'example_logging_config.yml', params)
    yield
    # Remove all handlers from root logger. No other loggers should ever have handlers attached.
    logging.getLogger().handlers = []
