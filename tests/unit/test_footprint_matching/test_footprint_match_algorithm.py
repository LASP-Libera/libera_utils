"""Unit tests for the LIBERA_FMATCH_VARIANT dispatch in ``footprint_match_algorithm``."""

import pytest

from libera_utils.footprint_matching.footprint_match_algorithm import (
    FMATCH_VARIANT_ENV,
    RUNNER_CONFIGS,
    resolve_runner_config_from_env,
)


class TestResolveRunnerConfigFromEnv:
    """LIBERA_FMATCH_VARIANT selects the FMATCH runner config that the CLI handler runs."""

    @pytest.mark.parametrize("variant", ["cam", "cam-camtime", "imager", "imager-camtime", "imager-flash"])
    def test_variant_selects_its_config(self, monkeypatch, variant):
        monkeypatch.setenv(FMATCH_VARIANT_ENV, variant)
        assert resolve_runner_config_from_env() is RUNNER_CONFIGS[variant]

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_unset_variant_raises(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(FMATCH_VARIANT_ENV, raising=False)
        else:
            monkeypatch.setenv(FMATCH_VARIANT_ENV, value)
        with pytest.raises(ValueError, match="not set"):
            resolve_runner_config_from_env()

    def test_unknown_variant_raises(self, monkeypatch):
        monkeypatch.setenv(FMATCH_VARIANT_ENV, "scene-id")
        with pytest.raises(ValueError, match="not a FMATCH variant"):
            resolve_runner_config_from_env()
