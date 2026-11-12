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
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, RadiometerFootprint
from tests.plugins.footprint_matching_fixtures import fake_radiometer_footprint

_FORK_AVAILABLE = "fork" in multiprocessing.get_all_start_methods()


def _footprints(n: int) -> list[RadiometerFootprint]:
    # Along-track footprints over the fake grid; every third one sits off the grid so the
    # NaN fill and zero coverage also have to line up across chunk boundaries.
    out = []
    for i in range(n):
        if i % 3 == 2:
            out.append(fake_radiometer_footprint(BoundingBox(40.2, 40.8, 40.2, 40.8), 40.5, 40.5))
        else:
            lon = 0.3 + 0.001 * i
            out.append(fake_radiometer_footprint(BoundingBox(0.2, 0.8, lon - 0.3, lon + 0.3), 0.5, lon))
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
        # 120 footprints is enough for two workers, so a configured count of 2 reaches the pool.
        monkeypatch.setenv(product.FMATCH_WORKERS_CONFIG_KEY, "2")
        seen = []

        def spy(mode, footprints, sources, source_file_paths, weigher, return_coverage, n_workers):
            seen.append(n_workers)
            return {}

        monkeypatch.setattr(product, "_aggregate_in_parallel", spy)
        aggregate_external_variables(OperationalMode.CAM, _footprints(120), fake_sources)
        assert seen == [2]

    def test_few_footprints_stay_in_process(self, fake_sources, monkeypatch):
        called = []
        monkeypatch.setattr(product, "_aggregate_in_parallel", lambda *a, **k: called.append(True))
        result = aggregate_external_variables(OperationalMode.CAM, _footprints(10), fake_sources, n_workers=4)
        assert called == []
        assert result["_fake_agg_mean_field_a"].shape == (10,)

    @pytest.mark.parametrize("raw", ["four", "0"])
    def test_bad_configured_count_names_the_key(self, fake_sources, monkeypatch, raw):
        monkeypatch.setenv(product.FMATCH_WORKERS_CONFIG_KEY, raw)
        with pytest.raises(ValueError, match="FMATCH_WORKERS must be a positive integer or 'auto', got"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(3), fake_sources)

    def test_zero_workers_rejected(self, fake_sources):
        with pytest.raises(ValueError, match="n_workers must be at least 1"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(3), fake_sources, n_workers=0)

    def test_spawn_without_paths_rejected(self, fake_sources, monkeypatch):
        monkeypatch.setattr(product.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
        with pytest.raises(ValueError, match="source_file_paths"):
            aggregate_external_variables(OperationalMode.CAM, _footprints(200), fake_sources, n_workers=2)


class TestAutoWorkerCount:
    """``FMATCH_WORKERS="auto"`` sizes the pool from the CPUs and memory a cgroup grants the process."""

    GIB = 1024**3

    @pytest.fixture
    def cgroup(self, tmp_path, monkeypatch):
        """A fake cgroup v2 tree with four CPUs and 6 GiB free, on a host with 8 CPUs and 100 GiB available."""
        root = tmp_path / "cgroup"
        root.mkdir()
        (root / "cpu.max").write_text("400000 100000\n")
        (root / "memory.max").write_text(f"{8 * self.GIB}\n")
        (root / "memory.current").write_text(f"{2 * self.GIB}\n")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(f"MemTotal: {128 * 1024**2} kB\nMemAvailable: {100 * 1024**2} kB\n")
        monkeypatch.setattr(product, "_CGROUP_ROOT", root)
        monkeypatch.setattr(product, "_PROC_MEMINFO", meminfo)
        monkeypatch.setattr(product.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)
        monkeypatch.setenv(product.FMATCH_WORKERS_CONFIG_KEY, "auto")
        return root

    def test_cpu_quota_bounds_the_pool(self, cgroup, monkeypatch):
        monkeypatch.setenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, str(self.GIB // 2))
        assert product._configured_worker_count(OperationalMode.CAM) == 4

    def test_memory_bounds_the_pool(self, cgroup, monkeypatch):
        # 6 GiB free less the 0.5 GiB reserve leaves room for two 2 GiB workers.
        monkeypatch.setenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, str(2 * self.GIB))
        assert product._configured_worker_count(OperationalMode.IMAGER) == 2

    def test_host_memory_bounds_an_unlimited_cgroup(self, cgroup, tmp_path, monkeypatch):
        (cgroup / "memory.max").write_text("max\n")
        (tmp_path / "meminfo").write_text(f"MemAvailable: {3 * 1024**2} kB\n")
        monkeypatch.setenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, str(self.GIB))
        assert product._configured_worker_count(OperationalMode.CAM) == 2

    def test_pool_is_at_least_one_worker(self, cgroup, monkeypatch):
        monkeypatch.setenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, str(64 * self.GIB))
        assert product._configured_worker_count(OperationalMode.IMAGER) == 1

    def test_no_memory_figure_runs_one_worker(self, cgroup, tmp_path, monkeypatch):
        monkeypatch.setattr(product, "_CGROUP_ROOT", tmp_path / "absent")
        monkeypatch.setattr(product, "_PROC_MEMINFO", tmp_path / "absent")
        monkeypatch.setenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, str(self.GIB))
        assert product._configured_worker_count(OperationalMode.CAM) == 1

    def test_cgroup_v1_quota_and_limit_are_read(self, tmp_path, monkeypatch):
        root = tmp_path / "v1"
        (root / "cpu").mkdir(parents=True)
        (root / "memory").mkdir()
        (root / "cpu" / "cpu.cfs_quota_us").write_text("300000\n")
        (root / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
        (root / "memory" / "memory.limit_in_bytes").write_text(f"{4 * self.GIB}\n")
        (root / "memory" / "memory.usage_in_bytes").write_text("0\n")
        monkeypatch.setattr(product.os, "sched_getaffinity", lambda pid: set(range(8)), raising=False)

        assert product._available_cpus(root) == 3
        assert product._available_memory_bytes(root, tmp_path / "absent") == 4 * self.GIB

    @pytest.mark.parametrize("mode", list(OperationalMode))
    def test_config_json_gives_every_mode_a_worker_memory(self, mode, monkeypatch):
        monkeypatch.delenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, raising=False)
        assert product._worker_memory_bytes(mode) > 0

    @pytest.mark.parametrize("raw", ["0", "lots"])
    def test_bad_worker_memory_names_the_key(self, cgroup, monkeypatch, raw):
        monkeypatch.setenv(product.FMATCH_WORKER_MEMORY_CONFIG_KEY, raw)
        with pytest.raises(ValueError, match="FMATCH_WORKER_MEMORY_BYTES must give a positive integer"):
            product._configured_worker_count(OperationalMode.CAM)


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
        paths = {"ERA5_Surface": [tmp_path / "a.nc"], "IGBP": [tmp_path / "b.hdf"]}
        first = product._worker_sources(OperationalMode.CAM, paths)
        second = product._worker_sources(OperationalMode.CAM, paths)
        assert first is second
        assert len(builds) == 1
        # A different staged set replaces the cached sources.
        product._worker_sources(OperationalMode.CAM, {"ERA5_Surface": [tmp_path / "c.nc"]})
        assert len(builds) == 2
