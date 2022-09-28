"""Logging utilities"""
# Standard
import collections.abc
import logging
import logging.config
from types import SimpleNamespace
import yaml
# Installed
from cloudpathlib import AnyPath
import watchtower
# Local
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def configure_logging(config_file: AnyPath or str, params: SimpleNamespace or dict = None):
    """Configure logging based on a (possibly parameterized) logging config yaml file.

    Parameters
    ----------
    config_file : AnyPath or str
        Location of config file.
    params : dict, Optional
        Parameters used to update the logging dictConfig object before configuring. Use this to pass things
        like log filenames, logging levels, cloudwatch log group names, etc. that must be dynamically
        determined at runtime rather than hardcoded into a static log configuration. When using this, be sure to put
        your values in the correct location in the dict, or they will not be used.
    """

    def update(d, u):
        """Recursively update a nested dict."""
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    with smart_open(config_file) as log_config:
        config_yml = log_config.read()
        config_dict = yaml.safe_load(config_yml)
    update(config_dict, params)
    print(config_dict)
    logging.config.dictConfig(config_dict)
    logger.info(f"Logging configured according to {config_file}.")


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
