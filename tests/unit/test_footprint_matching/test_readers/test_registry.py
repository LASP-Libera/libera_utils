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
from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.readers.registry import (
    AUXILIARY_PRODUCT_READERS,
    FMATCH_MODE_READERS,
    ReaderRegistry,
    auxiliary_products_for_reader,
    reader_key_for_auxiliary_product,
)
from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.readers.viirs import VIIRSCloudReader
from libera_utils.footprint_matching.types import OperationalMode


class TestListReaders:
    def test_all_readers_are_registered(self):
        # Targets auto-registration on import; asserts every expected built-in key appears in list_readers().
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
        # Targets the ordering contract of list_readers(); asserts the returned keys equal their sorted order.
        keys = ReaderRegistry.list_readers()
        assert keys == sorted(keys)

    def test_old_keys_not_present(self):
        # Targets removal of renamed/legacy reader keys; asserts "nsidc" and "viirs_l2l3" are absent from the registry.
        keys = ReaderRegistry.list_readers()
        assert "nsidc" not in keys
        assert "viirs_l2l3" not in keys


class TestGetReader:
    def test_get_igbp_returns_igbp_class(self):
        # Targets get() lookup for "igbp"; asserts it returns the IGBPReader class identity.
        assert ReaderRegistry.get("igbp") is IGBPReader

    def test_get_nise_returns_nise_class(self):
        # Targets get() lookup for "nise"; asserts it returns the NISEReader class identity.
        assert ReaderRegistry.get("nise") is NISEReader

    def test_get_era5_returns_era5_class(self):
        # Targets get() lookup for "era5"; asserts it returns the ERA5Reader class identity.
        assert ReaderRegistry.get("era5") is ERA5Reader

    def test_get_era5_pressure_returns_pressure_class(self):
        # Targets get() lookup for "era5_pressure"; asserts it returns the ERA5PressureLevelReader class identity.
        assert ReaderRegistry.get("era5_pressure") is ERA5PressureLevelReader

    def test_get_viirs_cloud_returns_viirs_cloud_class(self):
        # Targets get() lookup for "viirs_cloud"; asserts it returns the VIIRSCloudReader class identity.
        assert ReaderRegistry.get("viirs_cloud") is VIIRSCloudReader

    def test_get_viirs_brdf_returns_viirs_brdf_class(self):
        # Targets get() lookup for "viirs_brdf"; asserts it returns the VIIRSBRDFReader class identity.
        assert ReaderRegistry.get("viirs_brdf") is VIIRSBRDFReader

    def test_get_viirs_aod_returns_viirs_aod_class(self):
        # Targets get() lookup for "viirs_aod"; asserts it returns the VIIRSAODReader class identity.
        assert ReaderRegistry.get("viirs_aod") is VIIRSAODReader

    def test_get_ssf_returns_ssf_class(self):
        # Targets get() lookup for "ssf"; asserts it returns the SSFReader class identity.
        assert ReaderRegistry.get("ssf") is SSFReader

    def test_get_cldpix_returns_cldpix_class(self):
        # Targets get() lookup for "cldpix"; asserts it returns the CLDPIXReader class identity.
        assert ReaderRegistry.get("cldpix") is CLDPIXReader

    def test_get_unknown_key_raises_key_error(self):
        # Targets get() error handling for unknown keys; asserts it raises KeyError mentioning the bad key.
        with pytest.raises(KeyError, match="not_a_reader"):
            ReaderRegistry.get("not_a_reader")

    def test_key_error_message_lists_available_readers(self):
        # Targets the helpfulness of the KeyError message; asserts a known key ("igbp") is listed in the error text.
        with pytest.raises(KeyError) as exc_info:
            ReaderRegistry.get("not_a_reader")
        assert "igbp" in str(exc_info.value)


class TestGetReadersForMode:
    def test_cam_mode_returns_all_five_readers(self):
        # Targets the CAM-mode reader set; asserts it contains the five expected camera-timescale reader keys.
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        assert set(readers.keys()) >= {"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud"}

    def test_imager_mode_includes_cam_readers(self):
        # Targets IMAGER mode being a superset of CAM; asserts every CAM reader key is also present in IMAGER.
        cam_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        for key in cam_readers:
            assert key in imager_readers

    def test_climate_quality_readers_excluded_from_cam_mode(self):
        # AOD/SSF/CLDPIX are climate-quality dependencies and must not be active in
        # the CAM/NRT mode; asserts each such key is absent from the CAM reader set.
        cam_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for key in ("viirs_aod", "ssf", "cldpix", "era5_pressure"):
            assert key not in cam_readers

    def test_imager_mode_includes_rbsp_and_era5_readers(self):
        # The radiometer FMATCH-IMAGER product carries the RBSP CLDPIX/SSF readers
        # alongside the ERA5 (single- and pressure-level) and VIIRS AOD readers;
        # asserts each of those keys is present in the IMAGER reader set.
        imager_readers = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        for key in ("viirs_aod", "ssf", "cldpix", "era5_pressure"):
            assert key in imager_readers

    def test_imager_camtime_excludes_era5_pressure(self):
        # The ERA5 pressure-level fields are a radiometer-timescale quantity: they are
        # carried on the radiometer FMATCH-IMAGER product but NOT on the camera-timescale
        # FMATCH-IMAGER-CAMTIME product; asserts era5_pressure is in IMAGER but absent from CAMTIME.
        camtime = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER_CAMTIME)
        imager = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        assert "era5_pressure" not in camtime
        assert "era5_pressure" in imager

    def test_returns_dict_of_reader_classes(self):
        # Targets that get_readers_for_mode yields classes (not instances); asserts every mapped value is a type.
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for _key, cls in readers.items():
            assert isinstance(cls, type)

    def test_reader_class_attributes_accessible_without_instantiation(self):
        # Targets class-level reader metadata; asserts each class exposes READER_KEY/RESOLUTION_KM/VARIABLES and its
        # READER_KEY matches the mapping key, all without instantiating the reader.
        readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        for key, cls in readers.items():
            assert hasattr(cls, "READER_KEY")
            assert hasattr(cls, "RESOLUTION_KM")
            assert hasattr(cls, "VARIABLES")
            assert cls.READER_KEY == key


class TestReaderMembershipSets:
    """The per-product reader sets are the single source of truth for membership."""

    def test_every_named_reader_is_registered(self):
        # Targets referential integrity of FMATCH_MODE_READERS; asserts every key each mode names is actually registered.
        registered = set(ReaderRegistry.list_readers())
        for mode, keys in FMATCH_MODE_READERS.items():
            assert keys <= registered, f"{mode.value} names unregistered reader(s): {keys - registered}"

    def test_product_sets_cover_exactly_the_production_readers(self):
        # Targets completeness of the per-product sets: the union of every product's set must be exactly the
        # production readers -- no orphan reader left out of all products, and no stray key. Asserts that union
        # equals the keys derived from the reader classes themselves (not ReaderRegistry.list_readers(), which
        # other test modules pollute with throwaway readers like _FakeReader).
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
        # Targets full mode coverage of the map; asserts FMATCH_MODE_READERS keys equal the complete OperationalMode set.
        assert set(FMATCH_MODE_READERS) == set(OperationalMode)

    def test_every_mode_resolves_to_registered_readers(self):
        # Targets end-to-end mode resolution; asserts each mode resolves to a non-empty set of registered readers.
        registered = set(ReaderRegistry.list_readers())
        for mode in OperationalMode:
            readers = ReaderRegistry.get_readers_for_mode(mode)
            assert readers, f"{mode.value} resolved to an empty reader set"
            assert set(readers) <= registered


class TestAuxiliaryProductReaders:
    """The auxiliary product -> reader mapping is the bridge from a parsed granule to its reader."""

    def test_every_mapped_reader_key_is_registered(self):
        # Targets referential integrity of AUXILIARY_PRODUCT_READERS; asserts every reader key it
        # names is actually registered.
        registered = set(ReaderRegistry.list_readers())
        mapped = set(AUXILIARY_PRODUCT_READERS.values())
        assert mapped <= registered, f"Auxiliary mapping names unregistered reader(s): {mapped - registered}"

    def test_every_fmatch_reader_has_an_auxiliary_product(self):
        # Consistency guard: every reader that feeds a FMATCH product must have at least one
        # auxiliary product mapped to it, so a run's active reader set can always be sourced.
        fmatch_reader_keys = set().union(*FMATCH_MODE_READERS.values())
        mapped = set(AUXILIARY_PRODUCT_READERS.values())
        assert fmatch_reader_keys <= mapped, f"FMATCH readers with no auxiliary product: {fmatch_reader_keys - mapped}"

    def test_all_mapped_products_are_aux_auxiliary_members(self):
        # Targets that only the auxiliary_* AUX members are mapped; asserts each key is an AUX
        # DataProductIdentifier whose member name starts with "auxiliary_".
        for product_id in AUXILIARY_PRODUCT_READERS:
            assert isinstance(product_id, DataProductIdentifier)
            assert product_id.name.startswith("auxiliary_")

    def test_reader_key_for_auxiliary_product(self):
        # Targets the forward lookup; asserts a representative product maps to its reader key.
        assert reader_key_for_auxiliary_product(DataProductIdentifier.auxiliary_igbp_mcd12q1) == "igbp"
        assert reader_key_for_auxiliary_product(DataProductIdentifier.auxiliary_ceres_ssf) == "ssf"

    def test_reader_key_for_non_auxiliary_product_raises(self):
        # Targets error handling; asserts a Libera-produced product raises KeyError.
        with pytest.raises(KeyError, match="not an auxiliary input product"):
            reader_key_for_auxiliary_product(DataProductIdentifier.l1b_rad)

    def test_auxiliary_products_for_reader_groups_shared_reader(self):
        # viirs_brdf reads both VJ143C1 params and VJ143C3 albedo; asserts both come back in order.
        assert auxiliary_products_for_reader("viirs_brdf") == (
            DataProductIdentifier.auxiliary_viirs_brdf,
            DataProductIdentifier.auxiliary_viirs_brdf_albedo,
        )

    def test_auxiliary_products_for_reader_single_and_unknown(self):
        # Targets single-owner and unknown-key cases; asserts a one-tuple and an empty tuple respectively.
        assert auxiliary_products_for_reader("igbp") == (DataProductIdentifier.auxiliary_igbp_mcd12q1,)
        assert auxiliary_products_for_reader("does_not_exist") == ()


class TestReadersPublicApi:
    """The readers package must bind every name it advertises in ``__all__``."""

    def test_all_names_are_bound_at_package_root(self):
        # Targets the package __all__ contract; asserts every advertised name is actually bound in the package namespace.
        import libera_utils.footprint_matching.readers as pkg

        missing = [name for name in pkg.__all__ if not hasattr(pkg, name)]
        assert not missing, f"names in __all__ not bound in package namespace: {missing}"

    def test_reader_classes_importable_from_package_root(self):
        # The exact usage Copilot flagged: importing a reader class straight from the
        # package root must work, not just from its submodule; asserts the root-imported
        # classes are identical to the submodule-imported ones.
        from libera_utils.footprint_matching.readers import IGBPReader as PkgIGBP
        from libera_utils.footprint_matching.readers import SSFReader as PkgSSF

        assert PkgIGBP is IGBPReader
        assert PkgSSF is SSFReader
