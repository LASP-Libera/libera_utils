"""Logging utilities"""
# Standard
import logging
from pathlib import Path
from warnings import warn
# Installed
import watchtower
# Local
from libera_utils.config import config

LOG_MESSAGE_FORMAT = "%(asctime)s %(levelname)-9.9s [%(filename)s:%(lineno)s in %(funcName)s()]: %(message)s"
standard_log_formatter = logging.Formatter(LOG_MESSAGE_FORMAT)

logger = logging.getLogger(__name__)


def setup_task_logger(task_id: str,
                      stream_log_level: str or int = None,
                      logdir: Path or str = None,
                      cloudwatch_group: str = None):
    """
    Function to set up logger for a processing task based on a log directory and a task_id for that process.
    Note: DO NOT use the process identity as the task_id. Multiprocessing frameworks often re-use workers (processes)
    and that will result in the same log file being used for multiple processing tasks.

    Parameters
    ----------
    task_id: str
        An identifier for the task (~process) being logged. Used to name Cloudwatch log streams and log files.
    stream_log_level : str or int, Optional
        If not provided, stream log level is retrieved from the config
    logdir : Path or str, Optional
        If specified, logger writes filesystem logs to this path. If not specified, setup searches for the
        LIBSDP_LOG_DIR environment variable or uses the default specified in config.json.
    cloudwatch_group : str, Optional
        If specified, logger sends logs to cloudwatch_group/task_id. If not specified, setup searches for the
        LIBSDP_CLOUDWATCH_GROUP environment variable or uses the default from config.json.
        Note: If the cloudwatch group doesn't exist, the setup will refuse to create it and raise an exception. This
        prevents accidentally creating dozens of cloudwatch groups due to accidental bad input.

    Returns
    -------
    log_filepath: Path
        Absolute path to log file location.
    """
    # Set up root logger
    root_logger = logging.getLogger()
    # This ensures any loggers inheriting from the root (i.e. library loggers) don't spam debug logs all over the place
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Remove handlers, so they don't duplicate when dask worker processes are reused

    # Set up the libera_utils "parent" logger from which all module level loggers will inherit, since all modules are
    # submodules of libera_utils
    libsdp_logger = logging.getLogger('libera_utils')
    libsdp_logger.handlers = []  # Prevent this logger from doing anything except passing logs up to the root
    libsdp_logger.setLevel(logging.DEBUG)  # Setting this level means that all child loggers will pass DEBUG messages up

    configure_stream_logging(root_logger, stream_log_level)
    log_filepath = configure_filesystem_logging(root_logger, logdir, task_id)
    configure_cloudwatch_logging(root_logger, cloudwatch_group, task_id)

    logger.info(f"Logging setup complete for {task_id}.")

    return log_filepath


def configure_filesystem_logging(root_logger: logging.Logger, logdir: Path or str, task_id: str):
    """Add file logging handler to the root logger.

    Parameters
    ----------
    root_logger : logging.Logger
        The root logger.
    logdir : Path or str
        Path to the log directory. Will be created if it doesn't exist. If logdir is None, an attempt
        will be made to retrieve the log directory from the environment.
    task_id : str
        ID for the current process. This is used to create a log filename.

    Returns
    -------

    """
    if logdir:
        logdir = Path(logdir)
    elif config.get('LIBSDP_LOG_DIR') is not None:
        logdir = Path(config.get('LIBSDP_LOG_DIR'))
    else:
        warn("Initializing logger but LIBSDP_LOG_DIR is not set. Process will not produce filesystem logs.")
        return None

    logdir.mkdir(parents=True, exist_ok=True)  # Ignores FileExistsError for parent directories but NOT for file itself
    log_level = 'DEBUG'  # Always log to filesystem at DEBUG level
    log_filepath = logdir / f"{task_id}.log"
    filehandler = logging.FileHandler(log_filepath)
    filehandler.setFormatter(standard_log_formatter)
    filehandler.setLevel(log_level)
    root_logger.addHandler(filehandler)
    logger.info(f"Set up filesystem {log_level} logging to {log_filepath}.")
    return log_filepath


def configure_stream_logging(root_logger: logging.Logger, log_level: str):
    """Add stream logging handler to the root logger.

    Parameters
    ----------
    root_logger : logging.Logger
        The root logger.
    log_level : str
        The level at which to log to the stream handler. If None, we attempt to find the log level in the environment
        or config.json.

    Returns
    -------
    None
    """
    if not log_level:
        log_level = config.get('LIBSDP_STREAM_LOG_LEVEL')  # Get default stream log level

    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(standard_log_formatter)
    streamhandler.setLevel(log_level)
    root_logger.addHandler(streamhandler)
    logger.info(f"Set up stream {log_level} logging.")


def configure_cloudwatch_logging(root_logger: logging.Logger, log_group_name: str, log_stream_name: str):
    """Add Cloudwatch logging handler to the root logger.

    Parameters
    ----------
    root_logger : logging.Logger
        The root logger
    log_group_name : str
        Name of the Cloudwatch log group to use. If None, we attempt to find the Cloudwatch configuration in the
        environment or config.json. This is analogous to a log directory path.
    log_stream_name : str
        Name of the log stream to use. This is analogous to a log file.

    Returns
    -------
    None
    """
    config_cloudwatch_group = config.get('LIBSDP_CLOUDWATCH_GROUP')
    log_level = 'DEBUG'

    if log_group_name:
        pass
    elif config_cloudwatch_group and config_cloudwatch_group not in ['False', 'false', '0', 'none', 'None']:
        log_group_name = config_cloudwatch_group
    else:
        logger.info("No Cloudwatch logging configuration found. Process will not send logs to Cloudwatch.")
        return

    cloudwatch_handler = watchtower.CloudWatchLogHandler(log_group=log_group_name, create_log_group=True,
                                                         stream_name=log_stream_name)
    # The `msg` attribute is always included but `message` is not (msg != message). If we start seeing
    # oddities, we may need to start including `message` as well.
    # See https://docs.python.org/3/library/logging.html#logrecord-attributes
    structured_formatter = watchtower.CloudWatchLogFormatter(
        add_log_record_attrs=('asctime', 'levelname',
                              'process', 'processName',
                              'filename', 'lineno', 'funcName'))
    cloudwatch_handler.setFormatter(structured_formatter)
    cloudwatch_handler.setLevel('DEBUG')  # Always log to custom cloudwatch log_stream at DEBUG level
    root_logger.addHandler(cloudwatch_handler)
    logger.info(f"Set up Cloudwatch {log_level} logging to {log_group_name}/{log_stream_name}.")


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
