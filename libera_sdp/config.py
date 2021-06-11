"""Configuration reader. To modify the configuration, see file: emus_config.json"""
import json
import os
import string
from pathlib import Path

_CACHED_JSON_CONFIG = None  # Cached version of the raw, parsed JSON file to save overhead of reading it repeatedly.


class EmusEnvironmentError(Exception):
    """Exception for error in environment variable configuration"""
    def __init__(self, msg):
        super().__init__(msg)


class ConfigFormatter(string.Formatter):
    """Customize the string formatter to replace fields in a config string
    with values from the configuration dictionary. This will allow configuration
    parameters in the emus_config.json file to be based off of other configuration
    parameters by wrapping the configuration key in curly braces."""

    def get_value(self, key: str, *args, **kwargs):
        """Overrides the default get_value method in the python formatter. This will
        return the value from the emus configuration with the specified key."""
        return get_config(key)


def format_return_value(value: str):
    """Recursively formats the returned value, looking for config keys to substitute.

    Parameters
    ----------
    value : str
        String to format

    Returns
    -------
    str
    """
    if isinstance(value, str):
        formatter = ConfigFormatter()
        # if the string contains only a curly bracket keyword, return the value of get_config(keyword)
        key_iter = formatter.parse(value)
        text_and_key = next(key_iter)
        if text_and_key[0] == '' and not any(key_iter):
            return get_config(text_and_key[1])

        return formatter.format(value)
    elif isinstance(value, list):
        return [format_return_value(x) for x in value]
    else:
        return value


def get_config(key: str = None):
    """Retrieve a configuration value. First searched the config.json file for matches, then searches the
    environment

    Parameters
    ----------
    key : str
        Key for the desired configuration value.

    Returns
    -------

    """
    global _CACHED_JSON_CONFIG
    # Special case to return the emmemuspy root directory

    # Read in the emus_config.json file
    if not _CACHED_JSON_CONFIG:
        with open(Path(__file__).parent / 'config.json') as config_file:
            _CACHED_JSON_CONFIG = json.load(config_file)

    config = _CACHED_JSON_CONFIG

    if key is None:
        return config

    if key in config:
        result = config[key]
        return format_return_value(result)

    if os.getenv(key):
        result = os.getenv(key)
        return format_return_value(result)

    raise KeyError('Configuration variable {} not found.'.format(key))
