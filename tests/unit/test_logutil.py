"""Tests for logutil module"""
# Standard
from copy import deepcopy
import json
import logging
import logging.handlers
from typing import Mapping
# Installed
import pytest
# Local
from libera_utils import logutil

TEST_APP_PACKAGE_NAME = 'my_test_app'


@pytest.fixture
def setup_test_logger(mock_cloudwatch_context, monkeypatch, tmp_path):
    """Set up a test task logger and clear out all the handlers afterwards

    Note: This fixes a problem with caplog that breaks caplog when loggers are instantiated
    inside a test rather than a fixture. Solution is to just instantiate loggers in a fixture like this.
    See: https://stackoverflow.com/questions/69295248
    """
    logutil.configure_task_logging('test-task-1',
                                   limit_debug_loggers=(TEST_APP_PACKAGE_NAME,),
                                   console_log_level="INFO",
                                   log_dir=tmp_path)
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

    libsdp_log = logging.getLogger(TEST_APP_PACKAGE_NAME)  # top level libera_utils logger
    assert libsdp_log.level == logging.DEBUG

    libsdp_child_log = logging.getLogger(f'{TEST_APP_PACKAGE_NAME}.child')  # child libera_utils logger
    assert libsdp_child_log.level == logging.NOTSET

    # Simulates an external library that does NOT inherit from the libera_utils logger
    external_library_log = logging.getLogger('foolib.child')
    assert external_library_log.level == logging.NOTSET

    root_log.info("(GOOD) root info message")
    assert caplog.records[-1].message == '(GOOD) root info message'

    libsdp_log.info("(GOOD) my app info message")
    assert caplog.records[-1].message == '(GOOD) my app info message'

    libsdp_child_log.info("(GOOD) child info message")
    assert caplog.records[-1].message == '(GOOD) child info message'

    external_library_log.info("(GOOD) external info message")
    assert caplog.records[-1].message == '(GOOD) external info message'

    # Check that the right loggers produce debug messages
    libsdp_child_log.debug("(GOOD) child debug message")
    assert caplog.records[-1].message == '(GOOD) child debug message'

    libsdp_log.debug("(GOOD) my app debug message")
    assert caplog.records[-1].message == '(GOOD) my app debug message'

    # We want to exclude anything below INFO that doesn't come from a libera_utils.* logger
    external_library_log.debug("(BAD) external debug message")
    for record in caplog.records:
        assert 'external debug message' not in record.message

    root_log.debug("(BAD) root debug message")
    for record in caplog.records:
        assert '(BAD) root debug message' not in record.message


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


@pytest.mark.parametrize(
    "logged_value",
    [
        "test string log message",
        ["list", "of", 5, "test", "items"],
        {"jsondict": "value", "key": 99},
        {"nested": [{'complex': "value"}, {"second": "value"}]}
    ]
)
def test_json_formatter_for_cloudwatch(caplog, cleanup_loggers, logged_value):
    """Test the JsonFormatter's ability to convert different types of events to proper JSON for easy querying with
    Cloudwatch
    """
    # So we can see log output even in pytest
    caplog.set_level(logging.DEBUG)

    original_logged_value = deepcopy(logged_value)
    logger = logging.getLogger()  # root logger
    added_attrs = ('name', 'lineno', 'funcName', 'levelname', 'created', 'module')
    formatter = logutil.JsonLogFormatter(add_log_record_attrs=added_attrs, add_asctime=True)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    logger.info(logged_value)  # Log the message and check that it doesn't mutate the input object
    # This asserts that the logging call has not mutated the input
    assert logged_value == original_logged_value

    # Now test detailed formatting behavior
    log_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="fakepath",
        lineno=10,
        msg=logged_value,
        func="somefunction",
        args=None,
        exc_info=None,
    )

    formatted_record = formatter.format(log_record)

    # Deserialize the formatted log output to make asserts against. This also verifies it is valid JSON
    reconstituted_logged_value = json.loads(formatted_record)  # This has all the extra stuff added by the formatter

    if isinstance(logged_value, Mapping):
        for key, value in logged_value.items():  # Only check the keys that are in the logged value
            assert reconstituted_logged_value[key] == value
        # Assert that the keys are exactly what we expect: a union of original keys and added attributes
        assert set(reconstituted_logged_value.keys()) == set(logged_value.keys()).union(set(added_attrs)).union({"asctime"})
    else:
        assert reconstituted_logged_value['msg'] == logged_value
        assert set(reconstituted_logged_value.keys()) == set(added_attrs).union({'msg'}).union({"asctime"})


def test_json_log_formatter_string_interpolation(caplog, cleanup_loggers):
    """Test that we can log strings with %-style string interpolation"""
    logger = logging.getLogger()  # root logger
    added_attrs = ('lineno', 'funcName', 'levelname', 'created', 'module')
    formatter = logutil.JsonLogFormatter(add_log_record_attrs=added_attrs, add_asctime=True)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    caplog.set_level(logging.DEBUG)
    logger.info("interpolate %s", "this")

    log_record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="fakepath",
        lineno=10,
        msg="interpolate %s between %s me",
        func="somefunction",
        args=("something", 99),
        exc_info=None,
    )

    formatted_record = formatter.format(log_record)
    assert json.loads(formatted_record)["msg"] == "interpolate something between 99 me"


def test_json_log_formatter_exception_logging(caplog, cleanup_loggers):
    """Test logging exceptions with tracebacks"""
    logger = logging.getLogger()  # root logger
    added_attrs = ('lineno', 'funcName', 'levelname', 'created', 'module')
    formatter = logutil.JsonLogFormatter(add_log_record_attrs=added_attrs, add_asctime=True)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    caplog.set_level(logging.DEBUG)

    try:
        raise ValueError("test error")
    except ValueError as e:
        logger.exception(e)

    assert "traceback" in caplog.records[-1].msg
    assert "ValueError: test error" in caplog.records[-1].msg['traceback']
