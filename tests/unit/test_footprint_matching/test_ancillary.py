"""Unit tests for locating staged FMATCH ancillary granules.

These cover the directory contract defined by
``libera_utils.footprint_matching.ancillary``: which reader subdirectories a run looks for (derived
from the registry, not hard-coded), and how missing inputs are treated under each ``strict`` setting.
"""

import pytest

from libera_utils.footprint_matching.ancillary import (
    ANCILLARY_PATH_ENV,
    log_ancillary_inventory,
    resolve_ancillary_inputs,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode


def _stage(root, reader_keys, *, filenames=("granule_b.nc", "granule_a.nc")):
    """Create one subdirectory per reader key under ``root``, each holding ``filenames``."""
    for key in reader_keys:
        directory = root / key
        directory.mkdir(parents=True)
        for name in filenames:
            (directory / name).write_text("not really a granule")
    return root


class TestActiveReaderSet:
    """The resolved keys must track the registry's mode gating, not a hard-coded list."""

    @pytest.mark.parametrize("mode", list(OperationalMode))
    def test_keys_match_registry_for_mode(self, tmp_path, monkeypatch, mode):
        expected = set(ReaderRegistry.get_readers_for_mode(mode))
        _stage(tmp_path, expected)
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        resolved = resolve_ancillary_inputs(mode)

        assert set(resolved) == expected

    def test_imager_resolves_rbsp_alongside_era5_pressure(self, tmp_path, monkeypatch):
        """FMATCH-IMAGER looks for the RBSP CLDPIX/SSF readers alongside ERA5 pressure levels."""
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        resolved = resolve_ancillary_inputs(OperationalMode.IMAGER)

        assert "cldpix" in resolved
        assert "ssf" in resolved
        assert "era5_pressure" in resolved


class TestFileListing:
    """Staged files are returned per reader, sorted, and nested directories are ignored."""

    def test_files_are_listed_and_sorted(self, tmp_path, monkeypatch):
        keys = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        _stage(tmp_path, keys, filenames=("z_last.nc", "a_first.nc"))
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM)

        for key in keys:
            assert [path.name for path in resolved[key]] == ["a_first.nc", "z_last.nc"]

    def test_nested_directories_are_not_listed_as_granules(self, tmp_path, monkeypatch):
        """A reader directory may gain per-day subfolders later; those must not be mistaken for files."""
        _stage(tmp_path, ["era5"], filenames=("granule.nc",))
        (tmp_path / "era5" / "2026-06-11").mkdir()
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM)

        assert [path.name for path in resolved["era5"]] == ["granule.nc"]

    def test_explicit_root_overrides_environment(self, tmp_path, monkeypatch):
        staged = _stage(tmp_path / "staged", ["era5"], filenames=("granule.nc",))
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path / "not_used"))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM, root=staged)

        assert [path.name for path in resolved["era5"]] == ["granule.nc"]


class TestMissingInputs:
    """Missing inputs warn in this milestone (nothing consumes them yet) but raise under strict."""

    def test_unset_environment_variable_yields_empty_lists(self, monkeypatch):
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)

        resolved = resolve_ancillary_inputs(OperationalMode.CAM)

        assert set(resolved) == set(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        assert all(files == [] for files in resolved.values())

    def test_unset_environment_variable_raises_when_strict(self, monkeypatch):
        monkeypatch.delenv(ANCILLARY_PATH_ENV, raising=False)

        with pytest.raises(ValueError, match=ANCILLARY_PATH_ENV):
            resolve_ancillary_inputs(OperationalMode.CAM, strict=True)

    def test_missing_root_directory_raises_when_strict(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path / "does_not_exist"))

        with pytest.raises(FileNotFoundError, match="Ancillary root directory does not exist"):
            resolve_ancillary_inputs(OperationalMode.CAM, strict=True)

    def test_missing_reader_directory_is_empty_when_not_strict(self, tmp_path, monkeypatch):
        """A partially-staged tree still resolves; the absent reader simply has no files."""
        keys = set(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        _stage(tmp_path, keys - {"igbp"})
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        resolved = resolve_ancillary_inputs(OperationalMode.CAM)

        assert resolved["igbp"] == []
        assert resolved["era5"] != []

    def test_missing_reader_directory_raises_when_strict(self, tmp_path, monkeypatch):
        keys = set(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        _stage(tmp_path, keys - {"igbp"})
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        with pytest.raises(FileNotFoundError, match="igbp"):
            resolve_ancillary_inputs(OperationalMode.CAM, strict=True)

    def test_empty_reader_directory_raises_when_strict(self, tmp_path, monkeypatch):
        keys = set(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        _stage(tmp_path, keys)
        for stale in (tmp_path / "nise").iterdir():
            stale.unlink()
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))

        with pytest.raises(FileNotFoundError, match="is empty"):
            resolve_ancillary_inputs(OperationalMode.CAM, strict=True)


class TestInventoryLogging:
    """The inventory must name every active reader, calling out the ones with nothing staged."""

    def test_logs_counts_and_flags_missing(self, tmp_path, monkeypatch, caplog):
        keys = set(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        _stage(tmp_path, keys - {"igbp"}, filenames=("granule.nc",))
        monkeypatch.setenv(ANCILLARY_PATH_ENV, str(tmp_path))
        resolved = resolve_ancillary_inputs(OperationalMode.CAM)

        with caplog.at_level("INFO", logger="libera_utils.footprint_matching.ancillary"):
            log_ancillary_inventory(resolved)

        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "'era5': 1 file(s)" in messages
        assert "'igbp': 0 file(s) -- MISSING" in messages
