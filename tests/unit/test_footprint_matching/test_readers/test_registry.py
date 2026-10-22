"""Unit tests for the reader registry.

Tests confirm:
- importing the readers package registers every built-in reader under its key
- an unknown key raises a KeyError naming the registered keys
- each operational mode resolves to its per-product reader set (FMATCH_MODE_READERS)
- a second class registering a key another class holds is rejected at definition
"""

import pytest

import libera_utils.footprint_matching.readers  # noqa: F401  (registers the built-in readers)
from libera_utils.footprint_matching.readers import (
    CLDPIXReader,
    ERA5PressureLevelReader,
    ERA5Reader,
    IGBPReader,
    NISEReader,
    ReaderRegistry,
    SSFReader,
    StaticReader,
    VIIRSAODReader,
    VIIRSBRDFReader,
    VIIRSCldpropD3Reader,
)
from libera_utils.footprint_matching.readers.registry import FMATCH_MODE_READERS
from libera_utils.footprint_matching.types import OperationalMode, TimeMatch, VariableSpec

_BUILT_IN = {
    "cldpix": CLDPIXReader,
    "era5_single_level": ERA5Reader,
    "era5_pressure": ERA5PressureLevelReader,
    "igbp": IGBPReader,
    "nise": NISEReader,
    "ssf": SSFReader,
    "viirs_aod": VIIRSAODReader,
    "viirs_brdf": VIIRSBRDFReader,
    "viirs_cldprop_d3": VIIRSCldpropD3Reader,
}
_SPEC = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM)


class TestLookup:
    @pytest.mark.parametrize(("key", "cls"), sorted(_BUILT_IN.items()))
    def test_get_returns_the_registered_class(self, key, cls):
        """``ReaderRegistry.get`` returns the class registered under each built-in reader's key.

        Callers obtain readers by key, so importing the readers package must register every reader under the key it
        declares.

        Parametrized over the nine built-in readers' keys; asserts ``get(key)`` returns that exact class.
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
        assert "'era5_single_level'" in str(excinfo.value)

    def test_list_readers_is_sorted_and_holds_the_built_in_readers(self):
        """``ReaderRegistry.list_readers`` returns the registered keys sorted, including every built-in reader.

        The list is what the unknown-key error reports and what a caller sees when choosing a reader, so it must be
        complete and in a stable order.

        Asserts the returned list equals its sorted self and contains the nine built-in keys; it checks a subset because
        stub readers defined by other tests stay registered for the session.
        """
        # Test stub readers register for the whole session, so assert a subset.
        keys = ReaderRegistry.list_readers()
        assert keys == sorted(keys)
        assert set(_BUILT_IN) <= set(keys)


class TestReadersForMode:
    def test_every_mode_has_a_reader_set(self):
        """``FMATCH_MODE_READERS`` holds a reader set for every ``OperationalMode``.

        ``get_readers_for_mode`` indexes the map by mode, so a mode with no entry would fail at run time rather than
        when the map is written.

        Asserts the map's keys equal the full set of operational modes.
        """
        assert set(FMATCH_MODE_READERS) == set(OperationalMode)

    def test_the_sets_cover_exactly_the_built_in_readers(self):
        """The union of the per-product reader sets is exactly the nine built-in readers.

        A built-in reader in no set would never contribute to a product, and a key no reader registers would fail
        lookup when its product runs.

        Asserts the union of every mode's set equals the built-in keys.
        """
        assert set().union(*FMATCH_MODE_READERS.values()) == set(_BUILT_IN)

    @pytest.mark.parametrize("mode", list(OperationalMode))
    def test_get_readers_for_mode_returns_the_mode_set(self, mode):
        """``ReaderRegistry.get_readers_for_mode`` returns the registered class for each key in the mode's set.

        The product runners take their readers from this call, so it must return exactly the mode's set, as classes.

        Parametrized over every mode; asserts the returned keys equal the mode's set and each value is the class
        registered under its key.
        """
        readers = ReaderRegistry.get_readers_for_mode(mode)
        assert set(readers) == FMATCH_MODE_READERS[mode]
        assert all(readers[key] is _BUILT_IN[key] for key in readers)

    @pytest.mark.parametrize("mode", [OperationalMode.CAM, OperationalMode.CAM_CAMTIME])
    def test_camera_modes_exclude_the_climate_quality_readers(self, mode):
        """The CAM products take none of the AOD, SSF, CLDPIX or ERA5 pressure-level readers.

        Those sources are climate-quality inputs written only to the FMATCH-IMAGER products.

        Parametrized over the two CAM modes; asserts none of the four keys is in the mode's set.
        """
        assert not {"viirs_aod", "ssf", "cldpix", "era5_pressure"} & FMATCH_MODE_READERS[mode]

    def test_era5_pressure_levels_are_radiometer_timescale_only(self):
        """``era5_pressure`` contributes to FMATCH-IMAGER but not to FMATCH-IMAGER-CAMTIME.

        The ERA5 pressure-level fields are a radiometer-timescale quantity, so the camera-timescale product omits them.

        Asserts the key is in the IMAGER set and absent from the IMAGER_CAMTIME set.
        """
        assert "era5_pressure" in FMATCH_MODE_READERS[OperationalMode.IMAGER]
        assert "era5_pressure" not in FMATCH_MODE_READERS[OperationalMode.IMAGER_CAMTIME]


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
