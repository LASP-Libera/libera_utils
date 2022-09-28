"""Logging utilities"""
# Standard
import logging
import logging.config
import logging.handlers
import yaml
# Installed
from cloudpathlib import AnyPath
import watchtower
# Local
from libera_utils.config import config
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def configure_static_logging(config_file: AnyPath or str):
    """Configure logging based on a static logging configuration yaml file.

    The yaml is interpreted as a dict configuration. There is no ability to customize this logging
    configuration at runtime.

    Parameters
    ----------
    config_file : AnyPath or str
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


def configure_task_logging(task_id: str):
    """Configure logging based on runtime environment variables.

    Variables that control logging are LIBERA_LOG_DIR, LIBERA_CONSOLE_LOG_LEVEL, and LIBERA_LOG_GROUP. If these
    variables are unset, only INFO level console logging will be enabled.

    Parameters
    ----------
    task_id : str
        Unique identifier by which to name the log file and cloudwatch log stream.

    See Also
    --------
    configure_static_logging : Static logging configuration based on yaml file.
    """
    def _str_bool(s: str):
        """Examines an environment variable string to determine if it is truthy or falsy"""
        if not bool(s):
            return False
        if s.lower() in ("false", "0", "none", "null"):
            return False
        return True

    import json
    handlers = {}
    setup_messages = []
    try:  # Establish console log level from config
        console_log_level = config.get("LIBERA_CONSOLE_LOG_LEVEL")
        if _str_bool(console_log_level):
            console_handler = {
                "class": "logging.StreamHandler",
                "formatter": "plaintext",
                "level": console_log_level.upper(),
                "stream": "ext://sys.stdout"
            }
            handlers.update(console=console_handler)
            setup_messages.append(f"Console logging configured at level {console_log_level}.")
    except KeyError:
        pass

    try:  # Establish log directory from config
        log_dir = config.get("LIBERA_LOG_DIR")
        if _str_bool(log_dir):
            log_filepath = AnyPath(log_dir) / f"{task_id}.log"
            logfile_handler = {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "plaintext",
                "level": "DEBUG",
                "filename": str(log_filepath),
                "maxBytes": 1000000,
                "backupCount": 3
            }
            handlers.update(logfile=logfile_handler)
            setup_messages.append(f"File logging configured to log to {log_filepath}.")
    except KeyError:
        pass

    try:  # Establish cloudwatch log group from config
        log_group = config.get("LIBERA_LOG_GROUP")
        if _str_bool(log_group):
            watchtower_handler = {
                "class": "watchtower.CloudWatchLogHandler",
                "formatter": "json",
                "level": "DEBUG",
                "log_group_name": log_group,
                "log_stream_name": task_id,
                "send_interval": 10,
                "create_log_group": False
            }
            handlers.update(watchtower=watchtower_handler)
            setup_messages.append(f"Cloudwatch logging configured for log-group/log-stream: {log_group}/{task_id}.")
    except KeyError:
        pass

    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "format": '{"time": "%(asctime)s", level": "%(levelname)s", "module": "%(filename)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
            },
            "plaintext": {
                "format": "%(asctime)s %(levelname)-9.9s [%(filename)s:%(lineno)d in %(funcName)s()]: %(message)s"
            }
        },
        "handlers": handlers,
        "root": {
            "level": "INFO",
            "propagate": True,
            "handlers": list(handlers.keys())
        },
        "loggers": {
            "libera_utils": {
                "level": "DEBUG",
                "handlers": []
            }
        }
    }

    logging.config.dictConfig(config_dict)
    for message in setup_messages:
        logger.info(message)


def flush_cloudwatch_logs():
    """Force flush of all cloudwatch logging handlers. For example at the end of a process just before it is killed.

    Returns
    -------
    None
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, watchtower.CloudWatchLogHandler):
            handler.flush()
