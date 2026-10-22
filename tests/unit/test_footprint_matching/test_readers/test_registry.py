"""Unit tests for the reader registry.

Tests confirm:
- importing the readers package registers every gridded reader under its key
- an unknown key raises a KeyError naming the registered keys
- a second class registering a key another class holds is rejected at definition
"""

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
        """``ReaderRegistry.get`` returns the class registered under each built-in gridded reader's key.

        Callers obtain readers by key, so importing the readers package must register every gridded reader under the
        key it declares.

        Parametrized over the six gridded readers' keys; asserts ``get(key)`` returns that exact class.
        """
        assert ReaderRegistry.get(key) is cls

    def test_unknown_key_raises(self):
        """``ReaderRegistry.get`` raises ``KeyError`` for a key with no registered reader.

        A misspelled or unregistered key must fail at lookup rather than return something else, and the error lists the
        available keys so the caller can correct it.

        Looks up ``"nope"`` and asserts a ``KeyError`` whose message says no reader is registered under that key and
        lists a registered key.
        """
        with pytest.raises(KeyError, match="No reader registered with name 'nope'") as excinfo:
            ReaderRegistry.get("nope")
        assert "Available readers:" in str(excinfo.value)
        assert "'era5'" in str(excinfo.value)

    def test_list_readers_is_sorted_and_holds_the_gridded_readers(self):
        """``ReaderRegistry.list_readers`` returns the registered keys sorted, including every built-in gridded reader.

        The list is what the unknown-key error reports and what a caller sees when choosing a reader, so it must be
        complete and in a stable order.

        Asserts the returned list equals its sorted self and contains the six gridded keys; it checks a subset because
        stub readers defined by other tests stay registered for the session.
        """
        # Test stub readers register for the whole session, so assert a subset.
        keys = ReaderRegistry.list_readers()
        assert keys == sorted(keys)
        assert set(_GRIDDED) <= set(keys)


class TestDuplicateKey:
    def test_second_class_with_a_registered_key_is_rejected(self, monkeypatch):
        """A second reader class reusing a registered ``READER_KEY`` raises ``TypeError`` and the first class is kept.

        Reader keys must be unique: a silent overwrite would send every lookup of that key to a different reader, and
        failing at class definition reports the clash when the module is imported.

        Copies the registry with ``monkeypatch`` so nothing leaks to other tests, defines a stub ``StaticReader``
        under a test key and asserts it registers, then asserts a second stub under the same key raises a
        ``TypeError`` naming the key and that the key still resolves to the first class.
        """
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
