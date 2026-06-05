"""Unit tests for ReaderRegistry.

Tests confirm:
- All four built-in readers register when the readers package is imported
- get() returns the correct class
- get() raises KeyError for unknown keys
- list_readers() returns sorted keys
- get_readers_for_mode() filters by operational mode rank
"""
from __future__ import annotations

import pytest

# Importing the readers subpackage triggers __init_subclass__ registration
# for all four built-in readers.
import libera_utils.footprint_matching.readers  # noqa: F401
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NSIDCReader
from libera_utils.footprint_matching.readers.viirs import VIIRSL2L3Reader
from libera_utils.footprint_matching.types import OperationalMode


class TestListReaders:
    def test_all_four_readers_are_registered(self):
        keys = ReaderRegistry.list_readers()
        assert "era5" in keys
        assert "igbp" in keys
        assert "nsidc" in keys
        assert "viirs_l2l3" in keys

    def test_list_readers_is_sorted(self):
        keys = ReaderRegistry.list_readers()
        assert keys == sorted(keys)


class TestGetReader:
    def test_get_igbp_returns_igbp_class(self):
        assert ReaderRegistry.get("igbp") is IGBPReader

    def test_get_nsidc_returns_nsidc_class(self):
        assert ReaderRegistry.get("nsidc") is NSIDCReader

    def test_get_era5_returns_era5_class(self):
        assert ReaderRegistry.get("era5") is ERA5Reader

    def test_get_viirs_returns_viirs_class(self):
        assert ReaderRegistry.get("viirs_l2l3") is VIIRSL2L3Reader

    def test_get_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="not_a_reader"):
            ReaderRegistry.get("not_a_reader")

    def test_key_error_message_lists_available_readers(self):
        with pytest.raises(KeyError) as exc_info:
            ReaderRegistry.get("not_a_reader")
        # The error message should include the available reader names.
        assert "igbp" in str(exc_info.value)


class TestGetReadersForMode:
    def test_cam_mode_returns_all_four_readers(self):
        # All four built-in readers have REQUIRED_MODE=CAM (rank 0), so they
        # are all active in CAM mode.
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        assert set(readers.keys()) >= {"era5", "igbp", "nsidc", "viirs_l2l3"}

    def test_imager_mode_includes_cam_readers(self):
        # IMAGER has a higher rank than CAM, so all CAM readers are included.
        cam_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        for key in cam_readers:
            assert key in imager_readers

    def test_returns_dict_of_reader_classes(self):
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for _key, cls in readers.items():
            # Each value should be a type (class), not an instance.
            assert isinstance(cls, type)

    def test_reader_class_attributes_accessible_without_instantiation(self):
        # Verify that class-level attributes are readable on the class itself.
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for key, cls in readers.items():
            assert hasattr(cls, "READER_KEY")
            assert hasattr(cls, "RESOLUTION_KM")
            assert hasattr(cls, "REQUIRED_MODE")
            assert hasattr(cls, "VARIABLES")
            assert cls.READER_KEY == key
