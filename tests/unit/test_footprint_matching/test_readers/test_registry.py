"""Unit tests for ReaderRegistry.

Tests confirm:
- All built-in readers register when the readers package is imported
- get() returns the correct class
- get() raises KeyError for unknown keys
- list_readers() returns sorted keys
- get_readers_for_mode() filters by operational mode rank AND by the
  input-availability variant (year-one vs post-year-one)
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
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.readers.viirs import VIIRSCloudReader
from libera_utils.footprint_matching.types import FmatchVariant, OperationalMode


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
        # AOD/SSF/CLDPIX are climate-quality (post-Year-1) dependencies and must
        # not be active in the CAM/NRT mode, in either variant.
        for variant in FmatchVariant:
            cam_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM, variant)
            for key in ("viirs_aod", "ssf", "cldpix", "era5_pressure"):
                assert key not in cam_readers

    def test_imager_mode_default_variant_is_year_one(self):
        # Production default: RBSP readers (ssf, cldpix) are excluded because
        # those products do not exist in year one; the ERA5 substitutes are in.
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        assert "era5_pressure" in imager_readers
        assert "viirs_aod" in imager_readers
        for key in ("ssf", "cldpix"):
            assert key not in imager_readers

    def test_imager_mode_post_year_one_includes_rbsp_readers(self):
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER, FmatchVariant.POST_YEAR_ONE)
        for key in ("viirs_aod", "ssf", "cldpix"):
            assert key in imager_readers
        # The year-one-only ERA5 pressure-level reader is replaced by RBSP inputs.
        assert "era5_pressure" not in imager_readers

    def test_variant_gating_matches_reader_attributes(self):
        # The RBSP readers must declare POST_YEAR_ONE; the ERA5 pressure reader
        # YEAR_ONE; everything else must be variant-neutral.
        assert SSFReader.REQUIRED_VARIANT is FmatchVariant.POST_YEAR_ONE
        assert CLDPIXReader.REQUIRED_VARIANT is FmatchVariant.POST_YEAR_ONE
        assert ERA5PressureLevelReader.REQUIRED_VARIANT is FmatchVariant.YEAR_ONE
        for cls in (ERA5Reader, IGBPReader, NISEReader, VIIRSCloudReader):
            assert cls.REQUIRED_VARIANT is None

    def test_returns_dict_of_reader_classes(self):
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for _key, cls in readers.items():
            assert isinstance(cls, type)

    def test_reader_class_attributes_accessible_without_instantiation(self):
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for key, cls in readers.items():
            assert hasattr(cls, "READER_KEY")
            assert hasattr(cls, "RESOLUTION_KM")
            assert hasattr(cls, "REQUIRED_MODE")
            assert hasattr(cls, "VARIABLES")
            assert cls.READER_KEY == key
