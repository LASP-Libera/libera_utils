"""Tests for logutil module"""
# Standard
import logging
import logging.handlers
# Installed
import pytest
import watchtower
# Local
from libera_utils import logutil


@pytest.fixture
def setup_test_logger(mock_cloudwatch_context, monkeypatch, tmp_path):
    """Set up a test task logger and clear out all the handlers afterwards

    Note: This fixes a problem with caplog that breaks caplog when loggers are instantiated
    inside a test rather than a fixture. Solution is to just instantiate loggers in a fixture like this.
    See: https://stackoverflow.com/questions/69295248
    """
    monkeypatch.setenv('LIBERA_CONSOLE_LOG_LEVEL', "info")
    monkeypatch.setenv("LIBERA_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LIBERA_LOG_GROUP", "")
    logutil.configure_task_logging('test-task-1')
    root_log = logging.getLogger()  # root logger
    yield
    root_log.handlers = []


def test_task_logging_behavior(setup_test_logger, caplog):
    """Test that log messages appear (or don't appear) as desired"""
    root_log = logging.getLogger()  # root logger
    assert root_log.propagate is True

    # Add the LogCaptureHandler to the root logger (the only logger with any handlers)
    caplog.set_level(logging.DEBUG)

    # BUT we caplog automatically changes the level of the logger, so we change it back to INFO
    # leaving the caplog handler as DEBUG but the root logger at INFO
    root_log.setLevel(logging.INFO)

    print(root_log.handlers)
    assert root_log.level == logging.INFO

    libsdp_log = logging.getLogger('libera_utils')  # top level libera_utils logger
    assert libsdp_log.level == logging.DEBUG

    libsdp_child_log = logging.getLogger('libera_utils.child')  # child libera_utils logger
    assert libsdp_child_log.level == logging.NOTSET

    # Simulates an external library that does NOT inherit from the libera_utils logger
    external_library_log = logging.getLogger('foolib.child')
    assert external_library_log.level == logging.NOTSET

    root_log.info("root info message")
    assert caplog.records[-1].message == 'root info message'

    libsdp_log.info("libsdp info message")
    assert caplog.records[-1].message == 'libsdp info message'

    libsdp_child_log.info("child info message")
    assert caplog.records[-1].message == 'child info message'

    external_library_log.info("external info message")
    assert caplog.records[-1].message == 'external info message'

    # Check that the right loggers produce debug messages
    libsdp_child_log.debug("child debug message")
    assert caplog.records[-1].message == 'child debug message'

    libsdp_log.debug("libsdp debug message")
    assert caplog.records[-1].message == 'libsdp debug message'

    # We want to exclude anything below INFO that doesn't come from a libera_utils.* logger
    external_library_log.debug("external debug message")
    for record in caplog.records:
        assert 'external debug message' not in record.message

    root_log.debug("root debug message")
    for record in caplog.records:
        assert 'root debug message' not in record.message


def test_configure_static_logging(test_data_path, cleanup_loggers, tmp_path):
    """
    Test ability to configure logging from static yaml file.
    """
    logutil.configure_static_logging(test_data_path / 'example_logging_config.yml')
    libsdp_log = logging.getLogger('libera_utils')
    assert libsdp_log.level == logging.DEBUG
    assert len(libsdp_log.handlers) == 0

    root_log = logging.getLogger()
    assert root_log.level == logging.INFO
    print(root_log.handlers)
    assert len(root_log.handlers) == 2

    filehandlers = [h for h in root_log.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(filehandlers) == 1
    assert filehandlers[0].level == logging.DEBUG

    libsdp_child_log = logging.getLogger('libera_utils.child')
    assert libsdp_child_log.level == logging.NOTSET  # Inherits from parent
    assert len(libsdp_child_log.handlers) == 0

    library_log = logging.getLogger('somelibrary')
    assert library_log.level == logging.NOTSET
    assert len(library_log.handlers) == 0
