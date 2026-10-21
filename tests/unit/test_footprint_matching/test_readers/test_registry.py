"""Unit tests for the reader registry.

Tests confirm:
- importing the readers package registers every gridded reader under its key
- an unknown key raises a KeyError naming the registered keys
- a second class registering a key another class holds is rejected at definition
"""

from __future__ import annotations

import pytest

import libera_utils.footprint_matching.readers  # noqa: F401  (registers the built-in readers)
from libera_utils.footprint_matching.readers import (
    ERA5PressureLevelReader,
    ERA5Reader,
    IGBPReader,
    NISEReader,
    ReaderRegistry,
    StaticReader,
    VIIRSAODReader,
    VIIRSBRDFReader,
)
from libera_utils.footprint_matching.types import OperationalMode, TimeMatch, VariableSpec

_GRIDDED = {
    "era5": ERA5Reader,
    "era5_pressure": ERA5PressureLevelReader,
    "igbp": IGBPReader,
    "nise": NISEReader,
    "viirs_aod": VIIRSAODReader,
    "viirs_brdf": VIIRSBRDFReader,
}
_SPEC = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM)


class TestLookup:
    @pytest.mark.parametrize(("key", "cls"), sorted(_GRIDDED.items()))
    def test_get_returns_the_registered_class(self, key, cls):
        assert ReaderRegistry.get(key) is cls

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="No reader registered"):
            ReaderRegistry.get("nope")

    def test_list_readers_is_sorted_and_holds_the_gridded_readers(self):
        # Test stub readers register for the whole session, so assert a subset.
        keys = ReaderRegistry.list_readers()
        assert keys == sorted(keys)
        assert set(_GRIDDED) <= set(keys)


class TestDuplicateKey:
    def test_second_class_with_a_registered_key_is_rejected(self, monkeypatch):
        monkeypatch.setattr(ReaderRegistry, "_registry", dict(ReaderRegistry._registry))

        class _First(StaticReader):
            READER_KEY = "_registry_test_duplicate"
            INSTRUMENT = "FAKE"
            RESOLUTION_KM = 1.0
            TIME_MATCH = TimeMatch.DAILY
            VARIABLES = (_SPEC,)

            def _load_native(self):
                raise NotImplementedError

        assert ReaderRegistry.get("_registry_test_duplicate") is _First
        with pytest.raises(TypeError, match="READER_KEY='_registry_test_duplicate'.*already registered"):

            class _Second(StaticReader):
                READER_KEY = "_registry_test_duplicate"
                INSTRUMENT = "FAKE"
                RESOLUTION_KM = 1.0
                TIME_MATCH = TimeMatch.DAILY
                VARIABLES = (_SPEC,)

                def _load_native(self):
                    raise NotImplementedError

        assert ReaderRegistry.get("_registry_test_duplicate") is _First
