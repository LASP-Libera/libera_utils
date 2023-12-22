"""Configuration reader. To modify the configuration, see file: config.json"""
# Standard
import json
import logging
import os
from pathlib import Path
import re
import string
import sys
import warnings

logger = logging.getLogger(__name__)


class ConfigurationFormatter(string.Formatter):
    """Customize the string formatter to replace fields in a config string
    with values from the configuration dictionary. This will allow configuration
    parameters in the emus_config.json file to be based off of other configuration
    parameters by wrapping the configuration key in curly braces."""

    def get_value(self, key: str, *args, **kwargs):
        """Overrides the default get_value method in the python formatter. This will
        return the value from the emus configuration with the specified key."""
        return config.get(key)


class _ConfigurationCache:
    """Class that stores the JSON configuration and provides methods for accessing configuration information"""

    _json_config_filepath = Path(__file__).parent / 'data/config.json'

    # Should handle all floats
    FLOAT_REGEXP = re.compile(r'^[-+]?([0-9]+|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?$')
    # Example "-12.34E+56"      # sign (-)
    #     integer (12)
    #           OR
    #             int/mantissa (12.34)
    #                            exponent (E+56)

    def __init__(self):
        logger.debug("Initializing configuration cache from config.json.")
        with open(self._json_config_filepath, encoding='utf_8') as config_file:
            self._cached_json_config = json.load(config_file)
            self._known_config_variables = set(self._cached_json_config)

    def _format_return_value(self, value: str):
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
            formatter = ConfigurationFormatter()
            # if the string contains only a curly bracket keyword, return the value of get_config(keyword)
            key_iter = formatter.parse(value)
            text_and_key = next(key_iter)
            if text_and_key[0] == '' and not any(key_iter):
                return config.get(text_and_key[1])

            return formatter.format(value)

        if isinstance(value, list):
            return [self._format_return_value(x) for x in value]

        if isinstance(value, dict):
            return {k: self._format_return_value(v) for k, v in value.items()}

        return value

    def _parse_numeric_types(self, value: str):
        """Checks the final result of a config retrieval. If it is a string that can be interpreted as a float or int,
        parse it and return that.

        Parameters
        ----------
        value : any
            Final formatted value.

        Returns
        -------
        : str or float or int
        """
        if not isinstance(value, str):  # Indicates a non-string type came from the JSON config. Leave it as is.
            return value

        if self.FLOAT_REGEXP.match(value):
            f = float(value)
            if f.is_integer():
                return int(f)
            return f
        return value

    def force_reload(self):
        """Force reloading of the JSON config"""
        self.__init__()  # pylint: disable=C2801

    def get(self, key):
        """Retrieves a configuration value from either the cached JSON or from the environment

        Parameters
        ----------
        key : str
            Key for which to retrieve the configured value.

        Returns
        -------
        : any
            Resulting value
        """
        if (key != 'PKG_ROOT') and (key not in self._known_config_variables):
            warnings.warn(f"Configuration key {key} is not known to the config module. "
                          f"We will try to find it in the environment anyway but a default should be added to "
                          f"the config.json file.")

        if key is None:
            return self._cached_json_config

        if key == 'PKG_ROOT':  # Special case for root of installed package
            return str(Path(sys.modules[__name__.split('.', maxsplit=1)[0]].__file__).parent)

        # Checking the environment first allows easy override of any json config variable
        if os.getenv(key):
            result = os.getenv(key)
            return self._parse_numeric_types(self._format_return_value(result))

        if key in self._cached_json_config:
            result = self._cached_json_config[key]
            return self._parse_numeric_types(self._format_return_value(result))

        raise KeyError(f'Configuration variable {key} not found.')


config = _ConfigurationCache()
"""Singleton (one per process) accessor for :meth:`libera_utils.config._ConfigurationCache`"""
