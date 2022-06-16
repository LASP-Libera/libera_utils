"""Tests for logutil module"""
# Standard
import logging
# Local
from libera_utils import logutil


def test_logging_behavior(setup_test_logging, caplog, monkeypatch, tmp_path):
    """Test that log messages appear (or don't appear) as desired"""
    monkeypatch.setenv('LIBSDP_LOG_DIR', str(tmp_path))

    root_log = logging.getLogger()  # root logger
    assert root_log.level == logging.INFO

    libsdp_log = logging.getLogger('libera_utils')  # top level libera_utils logger
    assert libsdp_log.level == logging.DEBUG

    libsdp_child_log = logging.getLogger('libera_utils.child')  # child libera_utils logger
    assert libsdp_child_log.level == logging.NOTSET

    # Simulates an external library that does NOT inherit from the libera_utils logger
    external_library_log = logging.getLogger('foolib.child')
    assert external_library_log.level == logging.NOTSET

    # Add the caplog handler to the root logger (the only logger with any handlers)
    caplog.set_level(logging.DEBUG)
    # BUT we caplog automatically changes the level of the logger, so we change it back to INFO
    # leaving the caplog handler as DEBUG but the root logger at INFO
    root_log.setLevel(logging.INFO)

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


def test_setup_task_logger(setup_test_logging, tmpdir, monkeypatch):
    """
    Make sure that our logging setup function behaves as expected and that created loggers work as we want

    The root logger should be the only logger with handlers
    Loggers that inherit from libera_utils should be capable of logging to debug
    Loggers that don't inherit from libera_utils should never log debug messages (to exclude external library debug logs)
    The setup_test_logging feature ensures that the root logger handlers are removed after the test
    """
    monkeypatch.setenv('LIBSDP_STREAM_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('LIBSDP_LOG_DIR', str(tmpdir))

    old_log_filepath = logutil.setup_task_logger('test_log')  # run once
    log_filepath = logutil.setup_task_logger('test_log')  # run twice
    assert old_log_filepath == log_filepath

    libsdp_log = logging.getLogger('libera_utils')  #
    assert libsdp_log.level == logging.DEBUG
    assert len(libsdp_log.handlers) == 0

    root_log = logging.getLogger()
    assert root_log.level == logging.INFO
    assert len(root_log.handlers) == 2  # This will be 3 if LIBSDP_CLOUDWATCH_GROUP is truthy

    libsdp_child_log = logging.getLogger('libera_utils.child')
    assert libsdp_child_log.level == logging.NOTSET  # Inherits from parent
    assert len(libsdp_child_log.handlers) == 0
