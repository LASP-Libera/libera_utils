"""Tests for config module"""
from libera_sdp import config


def test_get_config(monkeypatch):
    """Test getting configuration values"""
    assert config.get_config('TEST_KEY') == -999

    monkeypatch.setenv('TEST_ENV_VAR', 'FOOBAR')
    assert config.get_config('TEST_ENV_VAR') == 'FOOBAR'

    monkeypatch.setenv('TEST_ENV_VAR', '42')
    assert isinstance(config.get_config('TEST_ENV_VAR'), str)
