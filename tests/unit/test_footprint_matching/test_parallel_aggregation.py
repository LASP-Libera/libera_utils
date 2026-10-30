"""Tests for the process-pool path of product.aggregate_external_variables.

The pool must be invisible in the result: a run on several workers returns exactly the
arrays of the single-process run, chunk boundaries included. Workers are forked after the
sources are loaded, so they aggregate with the parent's readers; where only spawn is
available each worker builds its sources from the staged paths.
"""

from __future__ import annotations

import multiprocessing

import numpy as np
import pytest

from libera_utils.footprint_matching import product
from libera_utils.footprint_matching.product import aggregate_external_variables
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode
from tests.plugins.footprint_matching_fixtures import FakeFootprint

_FORK_AVAILABLE = "fork" in multiprocessing.get_all_start_methods()


def _footprints(n: int) -> list[FakeFootprint]:
    # Along-track footprints over the fake grid; every third one sits off the grid so the
    # NaN fill and zero coverage also have to line up across chunk boundaries.
    out = []
    for i in range(n):
        if i % 3 == 2:
            out.append(FakeFootprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5))
        else:
            lon = 0.3 + 0.001 * i
            out.append(FakeFootprint(BoundingBox(0.2, 0.8, lon - 0.3, lon + 0.3), 0.5, lon))
    return out


@pytest.mark.skipif(not _FORK_AVAILABLE, reason="the forked pool needs a platform with fork")
class TestForkedPool:
    def test_matches_single_process_run(self, fake_sources):
        footprints = _footprints(303)  # not a multiple of the chunk count
        single, single_cov = aggregate_external_variables(
            OperationalMode.CAM, footprints, fake_sources, return_coverage=True, n_workers=1
        )
        pooled, pooled_cov = aggregate_external_variables(
            OperationalMode.CAM, footprints, fake_sources, return_coverage=True, n_workers=3
        )
        assert set(pooled) == set(single)
        for name in single:
            np.testing.assert_array_equal(pooled[name], single[name])
            assert pooled[name].dtype == single[name].dtype
        np.testing.assert_array_equal(pooled_cov, single_cov)
        # The off-grid footprints really are NaN / zero, so the equality above is not vacuous.
        assert np.isnan(single["_fake_agg_mean_field_a"][2])
        assert single_cov[2] == 0.0
        assert single_cov[0] == pytest.approx(1.0)

    def test_values_only_return_shape(self, fake_sources):
        footprints = _footprints(200)
        pooled = aggregate_external_variables(OperationalMode.CAM, footprints, fake_sources, n_workers=2)
        assert all(array.shape == (200,) for array in pooled.values())

    def test_fork_sources_are_cleared_after_the_run(self, fake_sources):
        aggregate_external_variables(OperationalMode.CAM, _footprints(120), fake_sources, n_workers=2)
        assert product._FORK_SOURCES is None


class TestWorkerCount:
    def test_default_comes_from_config(self, fake_sources, monkeypatch):
        monkeypatch.setenv(product.FMATCH_WORKERS_CONFIG_KEY, "1")
        seen = []
        original = product._aggregate_footprints

        def spy(*args, **kwargs):
            seen.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(product, "_aggregate_footprints", spy)
        aggregate_external_variables(OperationalMode.CAM, _footprints(6), fake_sources)
        assert seen == [True]

    def test_few_footprints_stay_in_process(self, fake_sources, monkeypatch):
        called = []
        monkeypatch.setattr(product, "_aggregate_in_parallel", lambda *a, **k: called.append(True))
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(10), fake_sources, n_workers=4)
        assert called == []
        assert result["_fake_agg_mean_field_a"].shape == (10,)

    def test_zero_workers_rejected(self, fake_sources):
        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(3), fake_sources, n_workers=0)

    def test_spawn_without_paths_rejected(self, fake_sources, monkeypatch):
        monkeypatch.setattr(product.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
        with pytest.raises(ValueError, match="source_file_paths"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(200), fake_sources, n_workers=2)


class TestWorkerSources:
    """A worker finds its sources: the parent's after a fork, its own build after a spawn."""

    def test_forked_worker_uses_the_parent_sources(self, fake_sources, monkeypatch):
        monkeypatch.setattr(product, "_FORK_SOURCES", fake_sources)
        assert product._worker_sources(OperationalMode.CAM, None) is fake_sources

    def test_spawned_worker_builds_once_per_staged_set(self, tmp_path, fake_sources, monkeypatch):
        monkeypatch.setattr(product, "_FORK_SOURCES", None)
        monkeypatch.setattr(product, "_WORKER_SOURCES_CACHE", {})
        builds = []

        def fake_build(mode, paths):
            builds.append((mode, tuple(sorted(paths))))
            return fake_sources

        monkeypatch.setattr(product, "build_sources", fake_build)
        paths = {"era5_surface": [tmp_path / "a.nc"], "igbp": [tmp_path / "b.hdf"]}
        first = product._worker_sources(OperationalMode.CAM, paths)
        second = product._worker_sources(OperationalMode.CAM, paths)
        assert first is second
        assert len(builds) == 1
        # A different staged set replaces the cached sources.
        product._worker_sources(OperationalMode.CAM, {"era5_surface": [tmp_path / "c.nc"]})
        assert len(builds) == 2
