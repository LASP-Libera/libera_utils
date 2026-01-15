"""Tests for logutil module"""

import json
import logging
import logging.handlers
import threading
from copy import deepcopy
from datetime import date, datetime
from unittest import mock

import pytest

from libera_utils import logutil

TEST_APP_PACKAGE_NAME = "my_test_app"


@pytest.fixture
def setup_test_logger(mock_cloudwatch_context, monkeypatch, tmp_path):
    """Set up a test task logger and clear out all the handlers afterwards

    Note: This fixes a problem with caplog that breaks caplog when loggers are instantiated
    inside a test rather than a fixture. Solution is to just instantiate loggers in a fixture like this.
    See: https://stackoverflow.com/questions/69295248
    """
    logutil.configure_task_logging(
        "test-task-1", limit_debug_loggers=(TEST_APP_PACKAGE_NAME,), console_log_level="INFO", log_dir=tmp_path
    )
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

    libsdp_child_log = logging.getLogger(f"{TEST_APP_PACKAGE_NAME}.child")  # child libera_utils logger
    assert libsdp_child_log.level == logging.NOTSET

    # Simulates an external library that does NOT inherit from the libera_utils logger
    external_library_log = logging.getLogger("foolib.child")
    assert external_library_log.level == logging.NOTSET

    root_log.info("(GOOD) root info message")
    assert caplog.records[-1].message == "(GOOD) root info message"

    libsdp_log.info("(GOOD) my app info message")
    assert caplog.records[-1].message == "(GOOD) my app info message"

    libsdp_child_log.info("(GOOD) child info message")
    assert caplog.records[-1].message == "(GOOD) child info message"

    external_library_log.info("(GOOD) external info message")
    assert caplog.records[-1].message == "(GOOD) external info message"

    # Check that the right loggers produce debug messages
    libsdp_child_log.debug("(GOOD) child debug message")
    assert caplog.records[-1].message == "(GOOD) child debug message"

    libsdp_log.debug("(GOOD) my app debug message")
    assert caplog.records[-1].message == "(GOOD) my app debug message"

    # We want to exclude anything below INFO that doesn't come from a libera_utils.* logger
    external_library_log.debug("(BAD) external debug message")
    for record in caplog.records:
        assert "external debug message" not in record.message

    root_log.debug("(BAD) root debug message")
    for record in caplog.records:
        assert "(BAD) root debug message" not in record.message


def test_configure_static_logging(test_data_path, cleanup_loggers, tmp_path):
    """
    Test ability to configure logging from static yaml file.
    """
    logutil.configure_static_logging(test_data_path / "example_logging_config.yml")
    libsdp_log = logging.getLogger("libera_utils")
    assert libsdp_log.level == logging.DEBUG
    assert len(libsdp_log.handlers) == 0

    root_log = logging.getLogger()
    assert root_log.level == logging.INFO
    print(root_log.handlers)
    assert len(root_log.handlers) == 2

    filehandlers = [h for h in root_log.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(filehandlers) == 1
    assert filehandlers[0].level == logging.DEBUG

    libsdp_child_log = logging.getLogger("libera_utils.child")
    assert libsdp_child_log.level == logging.NOTSET  # Inherits from parent
    assert len(libsdp_child_log.handlers) == 0

    library_log = logging.getLogger("somelibrary")
    assert library_log.level == logging.NOTSET
    assert len(library_log.handlers) == 0


class TestJsonLogEncoder:
    """Test cases for the JsonLogEncoder class"""

    @pytest.mark.parametrize(
        ("input_obj", "serialized_result"),
        [
            ("just a string", '"just a string"'),
            (
                {"a": "b", "int": 123, "float": 1.4, "object": date.fromisoformat("2020-01-01")},
                '{"a": "b", "int": 123, "float": 1.4, "object": "2020-01-01"}',
            ),
            (datetime.fromisoformat("2021-01-01T11:22:33.1234+00:00"), '"2021-01-01T11:22:33.123400+00:00"'),
            ({1, 2, 3}, "[1, 2, 3]"),
        ],
    )
    def test_graceful_encoding_handling(self, input_obj, serialized_result):
        """Test that the JsonLogEncoder gracefully handles non-serializable objects and always returns _something_"""
        res = json.dumps(input_obj, cls=logutil.JsonLogEncoder)
        assert res == serialized_result

    def test_encoding_edge_case_objects(self):
        """Test encoding objects that have no easy serialization method"""

        def _enc(val):
            return json.dumps(val, cls=logutil.JsonLogEncoder)

        # Datetime object
        assert _enc(datetime.fromisoformat("2021-01-01T11:22:33.1234+00:00")) == '"2021-01-01T11:22:33.123400+00:00"'

        # Thread local object
        assert "_thread._local" in _enc(threading.local())

        # Random custom object
        class TestCustomObj:
            pass

        assert "TestCustomObj" in _enc(TestCustomObj())

    def test_encoding_circular_reference_dict(self):
        """Test that circular references in dicts don't cause infinite recursion"""
        d = {"key": "value", "nested": {"inner": "data"}}
        d["self"] = d  # Circular reference

        # Should not raise RecursionError
        result = json.dumps(d, cls=logutil.JsonLogEncoder)

        # Should be valid JSON
        parsed = json.loads(result)

        # Should contain indication of depth limit
        assert "max depth" in str(parsed) or "circular" in str(parsed).lower()

    def test_encoding_circular_reference_list(self):
        """Test that circular references in lists don't cause infinite recursion"""
        lst = [1, 2, 3]
        lst.append(lst)  # Circular reference

        # Should not raise RecursionError
        result = json.dumps(lst, cls=logutil.JsonLogEncoder)

        # Should be valid JSON
        parsed = json.loads(result)

        # Should handle the circular reference gracefully
        assert isinstance(parsed, list)

    def test_encoding_deeply_nested_structure(self):
        """Test that deeply nested structures (near limit) still work correctly"""
        # Create a nested dict 15 levels deep (under the 20 limit)
        obj = {"level": 0}
        current = obj
        for i in range(1, 15):
            current["nested"] = {"level": i}
            current = current["nested"]

        # Should serialize successfully
        result = json.dumps(obj, cls=logutil.JsonLogEncoder)
        parsed = json.loads(result)

        # Verify structure is intact
        assert parsed["level"] == 0
        assert parsed["nested"]["level"] == 1

    def test_encoding_exceeds_depth_limit(self):
        """Test that excessively deep nesting triggers depth limit"""
        # Create a nested dict 25 levels deep (exceeds 20 limit)
        obj = {"level": 0}
        current = obj
        for i in range(1, 25):
            current["nested"] = {"level": i}
            current = current["nested"]

        # Should serialize with depth limit message
        result = json.dumps(obj, cls=logutil.JsonLogEncoder)

        # Should contain depth limit indicator somewhere in the result
        assert "max depth" in result or "depth" in result


class TestJsonLogFormatter:
    """Tests for the JsonLogFormatter class"""

    @pytest.mark.parametrize(
        ("logged_value", "added_attrs", "expected_formatted_message_dict"),
        [
            (
                "test string log message",
                ("name", "module"),
                {"msg": "test string log message"},
            ),
            (
                "test string log message",
                None,
                {"msg": "test string log message"},
            ),
            (12345, ("levelname",), {"msg": 12345, "levelname": "INFO"}),
            (["list", "of", 5, "test", "items"], (), {"msg": ["list", "of", 5, "test", "items"]}),
            ({"jsondict": "value", "key": 99}, (), {"jsondict": "value", "key": 99}),
            (
                {"nested": [{"complex": "value"}, {"second": "value"}]},
                (),
                {"nested": [{"complex": "value"}, {"second": "value"}]},
            ),
            (
                {datetime.fromisoformat("2021-01-01T11:22:33+00:00"): "some_value"},
                (),
                {"2021-01-01T11:22:33+00:00": "some_value"},
            ),
            ({date.fromisoformat("2021-01-01"): "some_value"}, (), {"2021-01-01": "some_value"}),
            (
                {("a", "b"): "weird, normally invalid key value for json"},
                (),
                {"('a', 'b')": "weird, normally invalid key value for json"},
            ),
        ],
    )
    def test_formatting_log_messages(
        self, caplog, cleanup_loggers, logged_value, added_attrs, expected_formatted_message_dict
    ):
        """Test the JsonFormatter's ability to convert different types of events to proper JSON for easy querying with
        Cloudwatch

        The expected_formatted_message_dict is a dict of key-value pairs that should appear in the formatted log output but does NOT
        include any added attributes from the log record itself (those are tested separately).
        """
        # So we can see log output even in pytest
        caplog.set_level(logging.DEBUG)

        original_logged_value = deepcopy(logged_value)
        logger = logging.getLogger()  # root logger
        # added_attrs = ("name", "lineno", "funcName", "levelname", "created", "module")
        formatter = logutil.JsonLogFormatter(add_log_record_attrs=added_attrs, add_asctime=False)

        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        logger.info(logged_value)  # Log the message and check that it doesn't mutate the input object
        # This asserts that the logging call has not mutated the input
        assert logged_value == original_logged_value
        # Note that the == operator here only holds true if the object passed to the formatter supports equality between
        # different object instances since we've deep copied the instance above.

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

        formatted_record = formatter.format(log_record)  # This has all the extra stuff added by the formatter

        # Deserialize the formatted log output to verify it is valid JSON
        formatted_record_dict = json.loads(formatted_record)

        # Compare the loaded formatted record to the expected formatted value (does not include the attributes added by the formatter itself)
        for key, value in expected_formatted_message_dict.items():
            assert formatted_record_dict[key] == value

        # Now check that the added attributes from the log record are present
        for attr in added_attrs if added_attrs is not None else logutil.JsonLogFormatter._default_log_record_attrs:
            assert attr in formatted_record_dict

    def test_formatting_log_message_with_set(self, caplog, cleanup_loggers):
        """Test logging a set object to ensure it becomes a list in JSON output"""
        logger = logging.getLogger()  # root logger
        formatter = logutil.JsonLogFormatter()
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        caplog.set_level(logging.DEBUG)

        test_set = {"apple", "banana", "cherry"}
        logger.info(test_set)

        reconstituted_logged_value = json.loads(caplog.records[-1].msg)
        assert isinstance(reconstituted_logged_value["msg"], list)
        assert set(reconstituted_logged_value["msg"]) == test_set

    def test_formatting_log_message_non_serializable_object(self, caplog, cleanup_loggers):
        """Test logging objects that are sent to the logger that are not JSON serializable.

        This basically tests the fallback mechanism in the logger to ensure that no matter what we send in,
        the logging system doesn't raise an exception."""
        logger = logging.getLogger()  # root logger
        formatter = logutil.JsonLogFormatter()
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        caplog.set_level(logging.DEBUG)

        class NonSerializableObject:
            def __str__(self):
                return "NonSerializableObjectStringRepresentation"

        logger.info(NonSerializableObject())

        reconstituted_logged_value = json.loads(caplog.records[-1].msg)
        assert "NonSerializableObject" in reconstituted_logged_value["msg"]

        logger.info(threading.local())
        reconstituted_logged_value = json.loads(caplog.records[-1].msg)
        assert "_thread._local" in reconstituted_logged_value["msg"]

    def test_formatting_log_messages_string_interpolation(self, caplog, cleanup_loggers):
        """Test that we can log strings with %-style string interpolation"""
        logger = logging.getLogger()  # root logger
        added_attrs = ("lineno", "funcName", "levelname", "created", "module")
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

    def test_formatting_log_messages_exception_logging(self, caplog, cleanup_loggers):
        """Test logging exceptions with tracebacks"""
        logger = logging.getLogger()  # root logger
        added_attrs = ("lineno", "funcName", "levelname", "created", "module")
        formatter = logutil.JsonLogFormatter(add_log_record_attrs=added_attrs, add_asctime=True)
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        caplog.set_level(logging.DEBUG)

        try:
            raise ValueError("test error")
        except ValueError as e:
            logger.exception(e)
        assert "traceback" in json.loads(caplog.records[-1].msg)
        assert "ValueError: test error" in json.loads(caplog.records[-1].msg)["traceback"]

    @mock.patch("libera_utils.logutil.json.dumps")
    def test_log_format_failure(self, mock_json_dumps, cleanup_loggers):
        """Test what happens when json.dumps fails inside the logging formatter"""
        formatter = logutil.JsonLogFormatter()

        # Create a log record to format
        log_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="fakepath",
            lineno=10,
            msg="test message",
            func="somefunction",
            args=None,
            exc_info=None,
        )

        # Mock json.dumps to raise an exception inside the format method
        mock_json_dumps.side_effect = Exception("JSON serialization failed")

        # Format the record - should not raise an exception
        formatted_record = formatter.format(log_record)

        # Verify the fallback message is returned
        assert formatted_record == "test message"

    def test_unrepresentable_object(self):
        """Test logging a pathologically unrepresentable object"""

        class Unstringable:
            def __str__(self):
                raise ValueError("__str__ is broken")

        class Unrepresentable(Unstringable):
            def __repr__(self):
                raise ValueError("__repr__ is broken")

        formatter = logutil.JsonLogFormatter()

        # Create a log record to format
        log_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="fakepath",
            lineno=10,
            msg=Unstringable(),
            func="somefunction",
            args=None,
            exc_info=None,
        )

        # Format the record - should not raise an exception
        formatted_record = formatter.format(log_record)

        # Verify the fallback message is returned
        assert "Unstringable" in formatted_record

        # Create a log record to format
        log_record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="fakepath",
            lineno=10,
            msg=Unrepresentable(),
            func="somefunction",
            args=None,
            exc_info=None,
        )

        # Format the record - should not raise an exception
        formatted_record = formatter.format(log_record)

        # Verify the fallback message is returned
        assert formatted_record == "<unrepresentable>"

    def test_log_format_respects_raise_exceptions_true(self, cleanup_loggers, capsys):
        """Test that formatter prints traceback when logging.raiseExceptions is True"""
        # Save original value
        original_raise = logging.raiseExceptions
        try:
            logging.raiseExceptions = True

            formatter = logutil.JsonLogFormatter()

            # Create a log record that will cause an error during formatting
            log_record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="fakepath",
                lineno=10,
                msg="test message",
                func="somefunction",
                args=None,
                exc_info=None,
            )

            # Mock json.dumps to fail
            with mock.patch("libera_utils.logutil.json.dumps", side_effect=Exception("JSON failed")):
                formatted_record = formatter.format(log_record)

                # Should return fallback message
                assert formatted_record == "test message"

                # Should have printed traceback to stderr
                captured = capsys.readouterr()
                assert "JSON failed" in captured.err
                assert "Traceback" in captured.err
        finally:
            logging.raiseExceptions = original_raise

    def test_log_format_respects_raise_exceptions_false(self, cleanup_loggers, capsys):
        """Test that formatter suppresses traceback when logging.raiseExceptions is False"""
        # Save original value
        original_raise = logging.raiseExceptions
        try:
            logging.raiseExceptions = False

            formatter = logutil.JsonLogFormatter()

            # Create a log record that will cause an error during formatting
            log_record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="fakepath",
                lineno=10,
                msg="test message",
                func="somefunction",
                args=None,
                exc_info=None,
            )

            # Mock json.dumps to fail
            with mock.patch("libera_utils.logutil.json.dumps", side_effect=Exception("JSON failed")):
                formatted_record = formatter.format(log_record)

                # Should return fallback message
                assert formatted_record == "test message"

                # Should NOT have printed traceback to stderr
                captured = capsys.readouterr()
                assert "JSON failed" not in captured.err
                assert "Traceback" not in captured.err
        finally:
            logging.raiseExceptions = original_raise
