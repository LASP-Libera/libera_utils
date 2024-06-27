"""Logging utilities"""
# Standard
import copy
from datetime import date, datetime
import json
import logging
import logging.config
import logging.handlers
from pathlib import Path
import traceback
from typing import Optional, Tuple, Mapping, Union, Any, Iterable
import yaml
# Installed
from cloudpathlib import AnyPath, S3Path
import watchtower
# Local
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def _json_serialize_default(o: Any) -> str:
    """
    A standard 'default' json serializer function.

    - Serializes datetime objects using their .isoformat() method.

    - Serializes all other objects using repr().
    """
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    return repr(o)


class JsonLogFormatter(logging.Formatter):
    """Altered version of the CloudWatchLogFormatter provided in the watchtower library"""

    _default_log_record_attrs = ('created', 'name', 'module', 'lineno', 'funcName', 'levelname')

    def __init__(
            self,
            *args,
            add_log_record_attrs: Optional[Tuple[str, ...]] = None,
            add_asctime: bool = True,
            **kwargs,
    ):
        """

        Parameters
        ----------
        add_log_record_attrs : Optional, tuple
            Tuple of log record attributes to add to the resulting structured JSON structure that comes out of the
            logging formatter.
        add_asctime : bool
            If True, adds an ASCII (ISO 8601-like) timestamp to the log record. Default True.
        """
        super().__init__(*args, **kwargs)
        self.add_log_record_attrs = add_log_record_attrs or self._default_log_record_attrs

        self.add_asctime = add_asctime

    def format(self, record: logging.LogRecord) -> str:
        """Format log message to a string

        Parameters
        ----------
        record : logging.LogRecord
            Log record object containing the logged message, which may be a dict (Mapping) or a string
        """
        # Perform %-style string interpolation before we make the message into a dict
        # This allows logging in the `log.info("%s incomplete %s", 1, "message")` style
        if isinstance(record.msg, str) and record.args:
            record.msg = record.msg % record.args
            record.args = None

        # If a dict was passed in, we don't want to mutate it as a side effect so we deepcopy it
        # This is a huge performance hit, but otherwise we are mutating our users' data and that's not cool
        msg = copy.deepcopy(record.msg) if isinstance(record.msg, Mapping) else {"msg": record.msg}

        if self.add_asctime:
            msg["asctime"] = self.formatTime(record)

        # Add additional attributes from the logging system to the msg dict
        if self.add_log_record_attrs:
            for field in self.add_log_record_attrs:
                if field != "msg":
                    msg[field] = getattr(record, field)

        # If we logged an exception, add the formatted traceback to the msg dict
        if record.exc_info:
            formatted_traceback = ''.join(traceback.format_exception(*record.exc_info))
            msg["traceback"] = formatted_traceback

        # Modify the record itself with the new msg dict
        record.msg = msg
        return json.dumps(record.msg, default=_json_serialize_default)  # Serialize the msg dict


def configure_static_logging(config_file: Union[str, Path, S3Path]):
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


def configure_task_logging(task_id: str,
                           limit_debug_loggers: Optional[Union[Iterable[str], str]] = None,
                           console_log_level: Union[str, int] = logging.INFO,
                           console_log_json: bool = False,
                           log_dir: Optional[Union[str, Path, S3Path]] = None,
                           cloudwatch_log_group: Optional[str] = None):
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
        "stream": "ext://sys.stdout"
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
            "backupCount": 3
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
            "create_log_group": True
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
                "format":
                    "%(asctime)s %(levelname)-9.9s [%(name)s:%(filename)s:%(lineno)d in %(funcName)s()]: %(message)s"
            }
        },
        "handlers": handlers,
        "root": {
            "level": "INFO" if limit_debug_loggers else "DEBUG",  # Optionally block unwanted debug messages
            "propagate": True,
            "handlers": list(handlers.keys())
        },
        "loggers": {
            # This explicitly allows debug messages from specific loggers if configured
            logger_prefix: {"level": "DEBUG", "handlers": []} for logger_prefix in limit_debug_loggers
        } if limit_debug_loggers else {}
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
