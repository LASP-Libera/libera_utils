"""Logging utilities"""

import copy
import json
import logging
import logging.config
import logging.handlers
import traceback
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import watchtower
import yaml
from cloudpathlib import AnyPath, S3Path

from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


class JsonLogEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for logging that can handle arbitrary Python objects.

    This encoder is designed to be absolutely robust and never raise exceptions,
    since it runs inside logging infrastructure. It handles:
    - datetime/date objects → ISO format strings
    - Non-string dictionary keys → converted to strings
    - Nested structures with arbitrary objects
    - Any other Python object → repr() fallback
    """

    def encode(self, o: Any) -> str:
        """
        Override encode to preprocess the entire object tree before JSON serialization.

        This is necessary to handle non-string dictionary keys, which json.dumps
        cannot handle even with a custom default() method.
        """
        try:
            return super().encode(self._preprocess(o))
        except Exception:
            # Absolute fallback - should never happen, but ensures logging never crashes
            return json.dumps({"msg": "LogEncoder failed to serialize log message", "repr": repr(o)})

    def _preprocess(self, o: Any, _depth: int = 0) -> Any:
        """
        Recursively preprocess objects to make them JSON-serializable.

        - Converts non-string dict keys to strings
        - Recursively processes nested dicts and lists
        - Converts datetime/date objects to ISO format
        - Uses repr() for any other non-serializable objects
        - Prevents infinite recursion with depth limit

        Parameters
        ----------
        o : Any
            Object to preprocess
        _depth : int
            Internal recursion depth counter (default 0)
        """
        # Prevent infinite recursion from circular references
        MAX_DEPTH = 20
        if _depth > MAX_DEPTH:
            return f"<max depth {MAX_DEPTH} exceeded, possible circular reference>"

        try:
            if isinstance(o, dict | Mapping):
                # Convert all keys to strings and recursively process values
                return {self._serialize_key(k): self._preprocess(v, _depth + 1) for k, v in o.items()}
            elif isinstance(o, list | tuple | set):
                # Recursively process list/tuple items
                return [self._preprocess(item, _depth + 1) for item in o]
            elif isinstance(o, date | datetime):
                # Convert datetime/date to ISO format string
                return o.isoformat()
            elif isinstance(o, str | int | float | bool) or o is None:
                # Already JSON-serializable
                return o
            else:
                # Fallback to repr for any other type
                return repr(o)
        except Exception:
            # Ultra-safe fallback for preprocessing errors
            return repr(o)

    def _serialize_key(self, key: Any) -> str | int | float | bool | None:
        """
        Convert a dictionary key to a string.

        Handles datetime/date keys specially to use ISO format.
        """
        if isinstance(key, str | int | float | bool | type(None)):
            return key  # These types are allowed as JSON keys by the default json serializer

        try:
            if isinstance(key, date | datetime):
                return key.isoformat()
            else:
                return str(key)
        except Exception:
            # Emergency fallback
            return repr(key)

    def default(self, o: Any) -> str:
        """
        Fallback for objects that make it past preprocessing.

        This should rarely be called since encode() preprocesses everything,
        but we keep it as an additional safety layer.
        """
        try:
            if isinstance(o, date | datetime):
                return o.isoformat()
            return repr(o)
        except Exception:
            return f"<unserializable: {type(o).__name__}>"


class JsonLogFormatter(logging.Formatter):
    """Altered version of the CloudWatchLogFormatter provided in the watchtower library"""

    _default_log_record_attrs = ("created", "name", "module", "lineno", "funcName", "levelname")

    def __init__(
        self,
        *args,
        add_log_record_attrs: tuple[str, ...] | None = None,
        add_asctime: bool = True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        add_log_record_attrs : Optional, tuple
            Tuple of log record attributes to add to the resulting structured JSON structure that comes out of the
            logging formatter. To omit all, pass a empty tuple. Default None, which adds a default set of useful attributes.
        add_asctime : bool
            If True, adds an ASCII (ISO 8601-like) timestamp to the log record. Default True.
        """
        super().__init__(*args, **kwargs)
        self.add_log_record_attrs = (
            add_log_record_attrs if add_log_record_attrs is not None else self._default_log_record_attrs
        )
        self.add_asctime = add_asctime

    def format(self, record: logging.LogRecord) -> str:
        """Format log message to a JSON string

        Parameters
        ----------
        record : logging.LogRecord
            Log record object containing the logged message, which may be a dict or a string

        Returns
        -------
        str
            JSON formatted log message string
        """
        # Ultra safe fallback to ensure we have a fallback message to present to the user
        # in the event of log formatting failure
        try:
            fallback_message_string = str(record.msg)
        except Exception:
            try:
                fallback_message_string = repr(record.msg)
            except Exception:
                fallback_message_string = "<unrepresentable>"

        try:
            if isinstance(record.msg, str) and record.args:
                # Perform %-style string interpolation before we make the message into a dict
                # This allows logging in the `log.info("%s incomplete %s", 1, "message")` style
                record.msg = record.msg % record.args
                record.args = None
                msg_dict = {"msg": record.msg}
            elif isinstance(record.msg, dict):
                # If a dict was passed in, we don't know what is in it and we don't want
                # to mutate it as a side effect so we deepcopy it before we add attributes to the JSON log message
                # This is a huge performance hit, but otherwise we are mutating our users' data and that's not cool
                try:
                    msg_dict = copy.deepcopy(record.msg)
                except Exception:
                    # If deepcopy fails for some reason, fall back to a string representation so we never mutate input
                    record.msg = {"msg": str(record.msg)}
                    return json.dumps(record.msg, cls=JsonLogEncoder)
            else:
                # If we got any other object, there's no obvious path to turning it into JSON,
                # so just convert it to string and put it in a dict so we can add attributes
                msg_dict = {"msg": record.msg}

            if self.add_asctime:
                msg_dict["asctime"] = self.formatTime(record)

            # Add additional attributes from the logging system to the msg dict
            if self.add_log_record_attrs:
                for field in self.add_log_record_attrs:
                    if field != "msg":
                        msg_dict[field] = getattr(record, field)

            # If we logged an exception, add the formatted traceback to the msg dict
            if record.exc_info:
                formatted_traceback = "".join(traceback.format_exception(*record.exc_info))
                msg_dict["traceback"] = formatted_traceback

            # Modify the record itself with the new msg dict
            record.msg = json.dumps(msg_dict, cls=JsonLogEncoder)  # Serialize the msg dict
            return record.msg
        except Exception:
            # Absolute fallback - should never happen, but ensures logging never crashes
            # Print the full traceback to stderr if configured to raise exceptions
            # This respects the standard library's logging.raiseExceptions setting
            if logging.raiseExceptions:
                traceback.print_exc()
            return fallback_message_string


def configure_static_logging(config_file: str | Path | S3Path):
    """Configure logging based on a static logging configuration yaml file.

    The yaml is interpreted as a dict configuration. There is no ability to customize this logging
    configuration at runtime.

    Parameters
    ----------
    config_file : cloudpathlib.anypath.AnyPath or str
        Location of config file.

    See Also
    --------
    configure_task_logging : Runtime modifiable logging configuration.
    """
    with smart_open(config_file) as log_config:
        config_yml = log_config.read()
        config_dict = yaml.safe_load(config_yml)
    logging.config.dictConfig(config_dict)
    logger.info(f"Logging configured statically according to {config_file}.")


def configure_task_logging(
    task_id: str,
    *,  # Only keyword arguments after this point
    limit_debug_loggers: Iterable[str] | str | None = None,
    console_log_level: str | int = logging.INFO,
    console_log_json: bool = False,
    log_dir: str | Path | S3Path | None = None,
    cloudwatch_log_group: str | None = None,
):
    """Configure logging for a specific task (e.g. a processing algorithm).

    File-based logging is always done at the DEBUG level.
    Watchtower-based cloudwatch logging is always done at the DEBUG level.
    Console logging level defaults to INFO but can be set with console_log_level.

    Examples
    --------
    Example 1: The following will configure DEBUG console-only logging for anything in your script but all
    other loggers will be limited to INFO level.

    ```python
    configure_task_logging("my-script", limit_debug_loggers=("__main__",), console_log_level=logging.DEBUG)
    ```

    Example 2: This will allow all debug messages through from all loggers
    and sets up file-based logging and a custom cloudwatch
    log group. Also console messages will be logged in serialized JSON.

    ```python
    configure_task_logging("my-script",
                           console_log_level=logging.DEBUG,
                           log_dir=Path("/tmp/my-script"),
                           console_log_json=True,
                           cloudwatch_log_group="custom-log-group")
    ```

    Parameters
    ----------
    task_id : str
        Unique identifier by which to name the log file and cloudwatch log stream.
    limit_debug_loggers : Optional[Union[Iterable[str] | str]]
        A list of logger name prefixes from which you want to allow debug messages (blocks debug from all others).
        For example, if you are working on a package called `my_app` and using module level logging,
        all your loggers will be named like `my_app.module_name.submodule_name`. By setting this to `(my_app,)`,
        all loggers that are named `my_app.*` will propagate debug messages while preventing spammy debug
        messages from installed libraries like boto3. If this is empty or None, all debug messages will propagate.
        To use this in scripts, either leave it unset or use `limit_debug_loggers=("__main__,)`.
    console_log_level : str or int, Optional
        Log level for console logging. If not specified, defaults to INFO
    console_log_json : bool, Optional
        If True, console logs will be JSON formatted. This is suitable for setting up loggers in AWS services that are
        automatically monitored by cloudwatch on stdout and stderr (e.g. Lambda or Batch)
    log_dir : str or Path or S3Path, Optional
        Log directory, which may be a local or S3Path. Default is None and results in no file-based logging.
    cloudwatch_log_group : str, Optional
        Override optional environment variable log group name. Default is None and will result in falling back to
        the LIBERA_LOG_GROUP environment variable. If that is not set, no cloudwatch JSON logging will be configured.

    Notes
    -----
    Even in the absence of cloudwatch JSON logging, all stdout/stderr messages generated by a Lambda will be logged to
    CloudWatch as string messages. Embedded JSON strings in log message text can still be queried in CloudWatch.

    See Also
    --------
    configure_static_logging : Static logging configuration based on yaml file.
    """
    handlers = {}  # Configured handlers
    setup_messages = []  # List of log messages generated during set up (these get logged after setup)
    if isinstance(limit_debug_loggers, str):
        limit_debug_loggers = (limit_debug_loggers,)

    # Set up console logging (also gets streamed to CloudWatch when running in AWS)
    # Optionally logs JSON structures or plaintext
    if isinstance(console_log_level, str):
        console_log_level = console_log_level.upper()
    console_handler = {
        "class": "logging.StreamHandler",
        "formatter": "json" if console_log_json else "plaintext",
        "level": console_log_level,
        "stream": "ext://sys.stdout",
    }
    handlers.update(console=console_handler)
    setup_messages.append(f"Console logging configured at level {console_log_level}.")

    # Set up file based logging
    if log_dir:
        log_filepath = AnyPath(log_dir) / f"{task_id}.log"
        logfile_handler = {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "plaintext",
            "level": "DEBUG",
            "filename": str(log_filepath),
            "maxBytes": 10000000,  # 10MB
            "backupCount": 3,
        }
        handlers.update(logfile=logfile_handler)
        setup_messages.append(f"File logging configured to log to {log_filepath}.")

    # Set up direct CloudWatch logging via Watchtower
    if cloudwatch_log_group:
        watchtower_handler = {
            "class": "watchtower.CloudWatchLogHandler",
            "formatter": "json",
            "level": "DEBUG",
            "log_group_name": cloudwatch_log_group,
            "log_stream_name": task_id,
            "send_interval": 10,
            "create_log_group": True,
        }
        handlers.update(watchtower=watchtower_handler)
        setup_messages.append({"cloudwatch_log_handler_config": watchtower_handler})

    # Single configuration dict made up of components configured above
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "libera_utils.logutil.JsonLogFormatter",
            },
            "plaintext": {
                "format": "%(asctime)s %(levelname)-9.9s [%(name)s:%(filename)s:%(lineno)d in %(funcName)s()]: "
                "%(message)s"
            },
        },
        "handlers": handlers,
        "root": {
            "level": "INFO" if limit_debug_loggers else "DEBUG",  # Optionally block unwanted debug messages
            "propagate": True,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            # This explicitly allows debug messages from specific loggers if configured
            logger_prefix: {"level": "DEBUG", "handlers": []}
            for logger_prefix in limit_debug_loggers
        }
        if limit_debug_loggers
        else {},
    }

    logging.config.dictConfig(config_dict)

    for message in setup_messages:
        logger.info(message)


def flush_cloudwatch_logs():
    """Force flush of all cloudwatch logging handlers.

    If you are missing the last few log messages in a log stream, this may help get those logs ingested before
    the process shuts down the logging system.

    Returns
    -------
    None
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, watchtower.CloudWatchLogHandler):
            handler.flush()
