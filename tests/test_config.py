"""Tests for config module"""
# Installed
import pytest
# Local
from libera_sdp.config import config


def test_get_config(monkeypatch):
    """Test getting configuration values"""
    assert config.get('TEST_KEY') == -999

    monkeypatch.setenv('TEST_ENV_VAR', 'FOOBAR')
    with pytest.warns(UserWarning):  # Check that the config module warns about non-standard config key
        assert config.get('TEST_ENV_VAR') == 'FOOBAR'

    monkeypatch.setenv('TEST_ENV_VAR', '42')
    with pytest.warns(UserWarning):  # Check that the config module warns about non-standard config key
        assert isinstance(config.get('TEST_ENV_VAR'), str)
