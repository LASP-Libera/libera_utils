"""Unit tests for AncillarySources and build_sources.

Readers open no granule at construction, so every source except the one a test queries
is given a placeholder path that is never read.
"""

from pathlib import Path

import numpy as np
import pytest

import libera_utils.footprint_matching.readers  # noqa: F401  (registers the built-in readers)
from libera_utils.footprint_matching.readers.era5_surface import ERA5SurfaceReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.sources import build_sources
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode
from tests.plugins.footprint_matching_fixtures import ERA5_FIXTURE_TIME, make_era5_surface_netcdf_fixture

MODE = OperationalMode.CAM
INACTIVE_SOURCE = "cldpix"
"""A registered source the CAM product does not use."""


def _placeholder_paths(tmp_path: Path) -> dict[str, list[Path]]:
    """Map every source active in ``MODE`` to a granule path that is never opened."""
    return {key: [tmp_path / f"{key}.nc"] for key in ReaderRegistry.get_readers_for_mode(MODE)}


def test_inactive_source_is_registered_but_not_in_mode():
    assert INACTIVE_SOURCE in ReaderRegistry.list_readers()
    assert INACTIVE_SOURCE not in ReaderRegistry.get_readers_for_mode(MODE)


def test_build_sources_builds_every_active_reader(tmp_path):
    sources = build_sources(MODE, _placeholder_paths(tmp_path))
    assert sources.active_mode is MODE
    assert sources.sources == sorted(ReaderRegistry.get_readers_for_mode(MODE))


def test_build_sources_missing_active_source_raises_keyerror(tmp_path):
    paths = _placeholder_paths(tmp_path)
    del paths["era5_surface"]
    with pytest.raises(KeyError, match="No file path supplied for active source 'era5_surface'"):
        build_sources(MODE, paths)


def test_build_sources_empty_entry_raises_valueerror(tmp_path):
    paths = _placeholder_paths(tmp_path)
    paths["era5_surface"] = []
    with pytest.raises(ValueError, match="at least one staged granule"):
        build_sources(MODE, paths)


def test_build_sources_unused_source_raises_keyerror(tmp_path):
    paths = _placeholder_paths(tmp_path)
    paths[INACTIVE_SOURCE] = [tmp_path / "cldpix.nc"]
    with pytest.raises(KeyError, match=r"keys \['cldpix'\] that mode 'FMATCH-CAM' does not use"):
        build_sources(MODE, paths)


def test_build_sources_misspelled_key_is_reported_as_itself(tmp_path):
    """The unexpected-key check runs before the missing-key check, so a typo names itself."""
    paths = _placeholder_paths(tmp_path)
    paths["era5_surfce"] = paths.pop("era5_surface")
    with pytest.raises(KeyError, match=r"keys \['era5_surfce'\]"):
        build_sources(MODE, paths)


def test_query_inactive_source_raises_keyerror(tmp_path):
    sources = build_sources(MODE, _placeholder_paths(tmp_path))
    with pytest.raises(KeyError, match="not active for mode"):
        sources.query(INACTIVE_SOURCE, BoundingBox(0.5, 1.5, 10.5, 11.5), ERA5_FIXTURE_TIME)


def test_missing_granule_raises_at_first_query_not_at_construction(tmp_path):
    sources = build_sources(MODE, _placeholder_paths(tmp_path))
    with pytest.raises(FileNotFoundError):
        sources.query("era5_surface", BoundingBox(0.5, 1.5, 10.5, 11.5), ERA5_FIXTURE_TIME)


def test_query_returns_the_readers_own_samples(tmp_path):
    era5_path = make_era5_surface_netcdf_fixture(tmp_path)
    paths = _placeholder_paths(tmp_path)
    paths["era5_surface"] = [era5_path]
    bbox = BoundingBox(0.5, 1.5, 10.5, 11.5)

    got = build_sources(MODE, paths).query("era5_surface", bbox, ERA5_FIXTURE_TIME)
    expected = ERA5SurfaceReader([era5_path]).samples_in(bbox, ERA5_FIXTURE_TIME)

    assert got.n_samples > 0
    np.testing.assert_array_equal(got.lats, expected.lats)
    np.testing.assert_array_equal(got.lons, expected.lons)
    np.testing.assert_array_equal(got.values, expected.values)
    assert got.times is None
    assert expected.times is None
    assert got.source == expected.source == "era5_surface"
    assert got.bounds == expected.bounds
    assert got.timestamp_source == expected.timestamp_source
    assert got.resolution_km == expected.resolution_km
