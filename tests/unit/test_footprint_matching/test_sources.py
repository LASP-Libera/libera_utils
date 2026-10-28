"""Unit tests for build_sources.

Readers check at construction that their granules exist but open none of them, so every
source except the one a test queries is given an empty placeholder file that is never read.
"""

from pathlib import Path

import pytest

import libera_utils.footprint_matching.readers  # noqa: F401  (registers the built-in readers)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.sources import build_sources
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode
from tests.plugins.footprint_matching_fixtures import ERA5_FIXTURE_TIME

MODE = OperationalMode.CAM
INACTIVE_SOURCE = "cldpix"
"""A registered source the CAM product does not use."""


def _placeholder_paths(tmp_path: Path) -> dict[str, list[Path]]:
    """Map every source active in ``MODE`` to an empty granule file that is never opened."""
    paths = {key: [tmp_path / f"{key}.nc"] for key in ReaderRegistry.get_readers_for_mode(MODE)}
    for (path,) in paths.values():
        path.touch()
    return paths


class TestBuildSources:
    def test_inactive_source_is_registered_but_not_in_mode(self):
        assert INACTIVE_SOURCE in ReaderRegistry.list_readers()
        assert INACTIVE_SOURCE not in ReaderRegistry.get_readers_for_mode(MODE)

    def test_returns_one_reader_per_active_source(self, tmp_path):
        paths = _placeholder_paths(tmp_path)
        readers = build_sources(MODE, paths)
        assert set(readers) == set(ReaderRegistry.get_readers_for_mode(MODE))
        for key, reader in readers.items():
            assert type(reader) is ReaderRegistry.get(key)
            assert reader.file_paths == tuple(paths[key])

    def test_opens_no_granule_until_the_first_samples_in(self, tmp_path):
        readers = build_sources(MODE, _placeholder_paths(tmp_path))
        with pytest.raises(OSError, match="Unknown file format"):
            readers["era5_surface"].samples_in(BoundingBox(0.5, 1.5, 10.5, 11.5), ERA5_FIXTURE_TIME)


class TestBuildSourcesRejects:
    def test_missing_active_source_raises_keyerror(self, tmp_path):
        paths = _placeholder_paths(tmp_path)
        del paths["era5_surface"]
        with pytest.raises(KeyError, match="No file path supplied for active source 'era5_surface'"):
            build_sources(MODE, paths)

    def test_unused_source_raises_keyerror(self, tmp_path):
        paths = _placeholder_paths(tmp_path)
        paths[INACTIVE_SOURCE] = [tmp_path / "cldpix.nc"]
        with pytest.raises(KeyError, match=r"keys \['cldpix'\] that mode 'FMATCH-CAM' does not use"):
            build_sources(MODE, paths)

    def test_misspelled_key_is_reported_as_itself(self, tmp_path):
        """The unexpected-key check runs before the missing-key check, so a typo names itself."""
        paths = _placeholder_paths(tmp_path)
        paths["era5_surfce"] = paths.pop("era5_surface")
        with pytest.raises(KeyError, match=r"keys \['era5_surfce'\]"):
            build_sources(MODE, paths)

    def test_single_path_entry_raises_typeerror(self, tmp_path):
        paths = _placeholder_paths(tmp_path)
        paths["era5_surface"] = paths["era5_surface"][0]
        with pytest.raises(TypeError, match="Wrap it in a list"):
            build_sources(MODE, paths)

    def test_empty_entry_raises_valueerror(self, tmp_path):
        paths = _placeholder_paths(tmp_path)
        paths["era5_surface"] = []
        with pytest.raises(ValueError, match="at least one staged granule"):
            build_sources(MODE, paths)

    def test_missing_granule_raises_filenotfounderror(self, tmp_path):
        paths = _placeholder_paths(tmp_path)
        paths["era5_surface"] = [tmp_path / "absent.nc"]
        with pytest.raises(FileNotFoundError, match="ERA5SurfaceReader .* not existing files"):
            build_sources(MODE, paths)
