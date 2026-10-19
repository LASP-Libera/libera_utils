"""Unit tests for the LIBERA_CLOUD_FRACTION_VARIANT dispatch in ``cloud_fraction_algorithm``."""

import pytest

from libera_utils.cloud_fraction.cloud_fraction_algorithm import (
    CLOUD_FRACTION_VARIANT_ENV,
    RUNNER_CONFIGS,
    resolve_runner_config_from_env,
)


class TestResolveRunnerConfigFromEnv:
    """LIBERA_CLOUD_FRACTION_VARIANT selects the cloud-fraction runner config that the CLI handler runs."""

    @pytest.mark.parametrize("variant", ["cam", "cam-camtime"])
    def test_variant_selects_its_config(self, monkeypatch, variant):
        monkeypatch.setenv(CLOUD_FRACTION_VARIANT_ENV, variant)
        assert resolve_runner_config_from_env() is RUNNER_CONFIGS[variant]

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_unset_variant_raises(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(CLOUD_FRACTION_VARIANT_ENV, raising=False)
        else:
            monkeypatch.setenv(CLOUD_FRACTION_VARIANT_ENV, value)
        with pytest.raises(ValueError, match="not set"):
            resolve_runner_config_from_env()

    def test_unknown_variant_raises(self, monkeypatch):
        monkeypatch.setenv(CLOUD_FRACTION_VARIANT_ENV, "imager")
        with pytest.raises(ValueError, match="not a cloud-fraction variant"):
            resolve_runner_config_from_env()
