"""Tests for logutil module"""
# Standard
import logging
import logging.handlers
# Installed
import watchtower


def test_logging_behavior(configure_example_logging, caplog):
    """Test that log messages appear (or don't appear) as desired"""

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


def test_configure_logging(configure_example_logging):
    """
    Make sure that our example logging setup function behaves as expected and that created loggers work as we want.

    The root logger should be the only logger with handlers
    Loggers that inherit from libera_utils should be capable of logging to debug
    Loggers that don't inherit from libera_utils should never log debug messages (to exclude external library debug logs)
    The setup_test_logging feature ensures that the root logger handlers are removed after the test
    """
    libsdp_log = logging.getLogger('libera_utils')
    assert libsdp_log.level == logging.DEBUG
    assert len(libsdp_log.handlers) == 0

    root_log = logging.getLogger()
    assert root_log.level == logging.INFO
    print(root_log.handlers)
    filehandlers = [h for h in root_log.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(filehandlers) == 1
    cw_handlers = [h for h in root_log.handlers if isinstance(h, watchtower.CloudWatchLogHandler)]
    assert len(cw_handlers) == 0  # didn't include this in the fixture

    libsdp_child_log = logging.getLogger('libera_utils.child')
    assert libsdp_child_log.level == logging.NOTSET  # Inherits from parent
    assert len(libsdp_child_log.handlers) == 0

    library_log = logging.getLogger('somelibrary')
    assert library_log.level == logging.NOTSET
    assert len(library_log.handlers) == 0
