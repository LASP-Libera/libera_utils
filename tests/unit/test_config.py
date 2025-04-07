"""Tests for config module"""
import pytest

from libera_utils.config import config


def test_get_config_test_key():
    """Test getting a real json-originated config value"""
    assert config.get('TEST_KEY') == -999


def test_get_config_no_default(monkeypatch):
    """Test setting custom variables that are not necessarily present in the json"""
    monkeypatch.setenv('TEST_ENV_VAR', 'FOOBAR')
    with pytest.warns(UserWarning):  # Check that the config module warns about non-standard config key
        assert config.get('TEST_ENV_VAR') == 'FOOBAR'


@pytest.mark.parametrize(
    ("s", "parsed", "t"),
    [
        ('-000', 0, int),
        ('-999', -999, int),
        ('3.14', 3.14, float),
        ('-3.14', -3.14, float),
        ('3e8', 3e8, int),
        ('-1e-9', -1e-9, float),
        ('astring', 'astring', str)
    ]
)
def test_get_config_numeric_typing(monkeypatch, s, parsed, t):
    """In the JSON, we can use numeric types easily but from the environment, the config module has to parse them out"""
    monkeypatch.setenv('TEST_KEY', s)  # Environment variable can only be strings
    v = config.get('TEST_KEY')
    assert v == parsed
    assert isinstance(v, t)


def test_nonexistent_config_var():
    """Demonstrate that a nonexistent config raises a KeyError"""
    with pytest.raises(KeyError):
        config.get('THIS_VARIABLE_DOES_NOT_EXIST_83632834')
