"""Unit tests for ReaderRegistry.

Tests confirm:
- All built-in readers register when the readers package is imported
- get() returns the correct class
- get() raises KeyError for unknown keys
- list_readers() returns sorted keys
- get_readers_for_mode() resolves each mode to its per-product reader set (FMATCH_MODE_READERS)
"""

from __future__ import annotations

import pytest

# Importing the readers subpackage triggers __init_subclass__ registration
# for all built-in readers.
import libera_utils.footprint_matching.readers  # noqa: F401
from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.readers.registry import (
    FMATCH_MODE_READERS,
    ReaderRegistry,
)
from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.readers.viirs import VIIRSCloudReader
from libera_utils.footprint_matching.types import OperationalMode


class TestListReaders:
    def test_all_readers_are_registered(self):
        keys = ReaderRegistry.list_readers()
        for expected in (
            "cldpix",
            "era5",
            "era5_pressure",
            "igbp",
            "nise",
            "ssf",
            "viirs_aod",
            "viirs_brdf",
            "viirs_cloud",
        ):
            assert expected in keys

    def test_list_readers_is_sorted(self):
        keys = ReaderRegistry.list_readers()
        assert keys == sorted(keys)

    def test_old_keys_not_present(self):
        keys = ReaderRegistry.list_readers()
        assert "nsidc" not in keys
        assert "viirs_l2l3" not in keys


class TestGetReader:
    def test_get_igbp_returns_igbp_class(self):
        assert ReaderRegistry.get("igbp") is IGBPReader

    def test_get_nise_returns_nise_class(self):
        assert ReaderRegistry.get("nise") is NISEReader

    def test_get_era5_returns_era5_class(self):
        assert ReaderRegistry.get("era5") is ERA5Reader

    def test_get_era5_pressure_returns_pressure_class(self):
        assert ReaderRegistry.get("era5_pressure") is ERA5PressureLevelReader

    def test_get_viirs_cloud_returns_viirs_cloud_class(self):
        assert ReaderRegistry.get("viirs_cloud") is VIIRSCloudReader

    def test_get_viirs_brdf_returns_viirs_brdf_class(self):
        assert ReaderRegistry.get("viirs_brdf") is VIIRSBRDFReader

    def test_get_viirs_aod_returns_viirs_aod_class(self):
        assert ReaderRegistry.get("viirs_aod") is VIIRSAODReader

    def test_get_ssf_returns_ssf_class(self):
        assert ReaderRegistry.get("ssf") is SSFReader

    def test_get_cldpix_returns_cldpix_class(self):
        assert ReaderRegistry.get("cldpix") is CLDPIXReader

    def test_get_unknown_key_raises_key_error(self):
        with pytest.raises(KeyError, match="not_a_reader"):
            ReaderRegistry.get("not_a_reader")

    def test_key_error_message_lists_available_readers(self):
        with pytest.raises(KeyError) as exc_info:
            ReaderRegistry.get("not_a_reader")
        assert "igbp" in str(exc_info.value)


class TestGetReadersForMode:
    def test_cam_mode_returns_all_five_readers(self):
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        assert set(readers.keys()) >= {"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud"}

    def test_imager_mode_includes_cam_readers(self):
        cam_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        for key in cam_readers:
            assert key in imager_readers

    def test_climate_quality_readers_excluded_from_cam_mode(self):
        # AOD/SSF/CLDPIX are climate-quality dependencies and must not be active in
        # the CAM/NRT mode.
        cam_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for key in ("viirs_aod", "ssf", "cldpix", "era5_pressure"):
            assert key not in cam_readers

    def test_imager_mode_includes_rbsp_and_era5_readers(self):
        # The radiometer FMATCH-IMAGER product carries the RBSP CLDPIX/SSF readers
        # alongside the ERA5 (single- and pressure-level) and VIIRS AOD readers.
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        for key in ("viirs_aod", "ssf", "cldpix", "era5_pressure"):
            assert key in imager_readers

    def test_imager_camtime_excludes_era5_pressure(self):
        # The ERA5 pressure-level fields are a radiometer-timescale quantity: they are
        # carried on the radiometer FMATCH-IMAGER product but NOT on the camera-timescale
        # FMATCH-IMAGER-CAMTIME product.
        camtime = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER_CAMTIME)
        imager = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        assert "era5_pressure" not in camtime
        assert "era5_pressure" in imager

    def test_returns_dict_of_reader_classes(self):
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for _key, cls in readers.items():
            assert isinstance(cls, type)

    def test_reader_class_attributes_accessible_without_instantiation(self):
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for key, cls in readers.items():
            assert hasattr(cls, "READER_KEY")
            assert hasattr(cls, "RESOLUTION_KM")
            assert hasattr(cls, "VARIABLES")
            assert cls.READER_KEY == key


class TestReaderMembershipSets:
    """The per-product reader sets are the single source of truth for membership."""

    def test_every_named_reader_is_registered(self):
        registered = set(ReaderRegistry.list_readers())
        for mode, keys in FMATCH_MODE_READERS.items():
            assert keys <= registered, f"{mode.value} names unregistered reader(s): {keys - registered}"

    def test_product_sets_cover_exactly_the_production_readers(self):
        # The union of every product's set must be exactly the production readers --
        # no orphan reader left out of all products, and no stray key. Derived from the
        # reader classes themselves (not ReaderRegistry.list_readers(), which other test
        # modules pollute with throwaway readers like _FakeReader).
        production_keys = {
            cls.READER_KEY
            for cls in (
                VIIRSAODReader,
                VIIRSBRDFReader,
                VIIRSCloudReader,
                CLDPIXReader,
                ERA5Reader,
                ERA5PressureLevelReader,
                IGBPReader,
                NISEReader,
                SSFReader,
            )
        }
        used = set().union(*FMATCH_MODE_READERS.values())
        assert used == production_keys

    def test_base_map_covers_all_modes(self):
        assert set(FMATCH_MODE_READERS) == set(OperationalMode)

    def test_every_mode_resolves_to_registered_readers(self):
        registered = set(ReaderRegistry.list_readers())
        for mode in OperationalMode:
            readers = ReaderRegistry.get_readers_for_mode(mode)
            assert readers, f"{mode.value} resolved to an empty reader set"
            assert set(readers) <= registered


class TestReadersPublicApi:
    """The readers package must bind every name it advertises in ``__all__``."""

    def test_all_names_are_bound_at_package_root(self):
        import libera_utils.footprint_matching.readers as pkg

        missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
        assert not missing, f"names in __all__ not bound in package namespace: {missing}"

    def test_reader_classes_importable_from_package_root(self):
        # The exact usage Copilot flagged: importing a reader class straight from the
        # package root must work, not just from its submodule.
        from libera_utils.footprint_matching.readers import IGBPReader as PkgIGBP
        from libera_utils.footprint_matching.readers import SSFReader as PkgSSF

        assert PkgIGBP is IGBPReader
        assert PkgSSF is SSFReader
