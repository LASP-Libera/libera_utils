"""Logging utilities"""
# Standard
import logging
from pathlib import Path
# Local
from libera_sdp.config import config

LOG_MESSAGE_FORMAT = "%(asctime)s %(levelname)-9.9s [%(filename)s:%(lineno)s in %(funcName)s()]: %(message)s"


def setup_task_logger(task_id: str, stream_log_level: str or int = None):
    """
    Function to set up logger for a processing task based on a log directory and a task_id for that process.
    Note: DO NOT use the process identity as the task_id. Dask workers re-use workers (processes) and that will result
    in the same log file being used for multiple processing tasks.

    Parameters
    ----------
    task_id: str
        An identifier for the task (~process) being logged. Used to name Cloudwatch log streams and log files.
    stream_log_level : str or int, optional
        If not provided, stream log level is retrieved from the config

    Returns
    -------
    abs_log_filepath: Path
        Absolute path to log file location.
    """
    logdir = Path(config.get('LIBSDP_LOG_DIR'))
    if not stream_log_level:
        stream_log_level = config.get('LIBSDP_STREAM_LOG_LEVEL')

    # Set up root logger
    root_logger = logging.getLogger()
    # This ensures any loggers inheriting from the root (i.e. library loggers) don't spam debug logs all over the place
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Remove handlers so they don't duplicate when dask worker processes are reused

    standard_log_formatter = logging.Formatter(LOG_MESSAGE_FORMAT)

    # Add file logging at level DEBUG
    full_log_filepath = logdir / f"{task_id}.log"
    filehandler = logging.FileHandler(full_log_filepath)
    filehandler.setFormatter(standard_log_formatter)
    filehandler.setLevel(logging.DEBUG)  # Always log to files at DEBUG level
    root_logger.addHandler(filehandler)

    # Add stream logging at level stream_log_level
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(standard_log_formatter)
    streamhandler.setLevel(stream_log_level)
    root_logger.addHandler(streamhandler)

    # try:
    #     cloudwatch_group = config.get('LIBSDP_CLOUDWATCH_GROUP')  # This will either raise a KeyError or will be a str
    #     if cloudwatch_group in ['False', 'false', '0', 'none', 'None']:
    #         cloudwatch_group = False
    # except KeyError:
    #     cloudwatch_group = False
    #
    # if cloudwatch_group:
    #     # Add cloudwatch handler
    #     cloudwatch_handler = watchtower.CloudWatchLogHandler(log_group=cloudwatch_group, create_log_group=False,
    #                                                          stream_name=f"{os.path.basename(logdir)}/{task_id}")
    #     cloudwatch_handler.setFormatter(standard_log_formatter)
    #     cloudwatch_handler.setLevel('DEBUG')  # Always log to custom cloudwatch log_stream at DEBUG level
    #     root_logger.addHandler(cloudwatch_handler)

    # Set up the emmemuspy "parent" logger from which all loggers created with `get_emus_logger` will inherit
    libsdp_logger = logging.getLogger('libera_sdp')
    libsdp_logger.handlers = []  # Prevent this logger from doing anything except passing logs up to the root
    libsdp_logger.setLevel(logging.DEBUG)  # Setting this level means that all child loggers will pass DEBUG messages up

    return full_log_filepath
