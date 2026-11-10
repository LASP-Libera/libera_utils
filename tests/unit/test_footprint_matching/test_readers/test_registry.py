"""Unit tests for the reader registry.

Tests confirm:
- importing the readers package registers every built-in reader under its key, with its declared
  instrument, resolution, time policy and variables
- an unknown key raises a KeyError naming the registered keys
- each operational mode resolves to its reader set (FMATCH_MODE_READERS), derived from the readers'
  VariableSpec.modes, and carries exactly today's variables
- each FMATCH product definition declares exactly the ``<READER_KEY>_<spec name>`` variables its readers produce
- a second class registering a key another class holds is rejected at definition
"""

from pathlib import Path

import numpy as np
import pytest

import libera_utils
import libera_utils.footprint_matching.readers  # noqa: F401  (registers the built-in readers)
from libera_utils.footprint_matching.readers import (
    CLDPIXReader,
    ERA5PressureLevelReader,
    ERA5SurfaceReader,
    FLASHFluxReader,
    IGBPReader,
    NISEReader,
    ReaderRegistry,
    StaticReader,
    VIIRSAODReader,
    VIIRSBRDFReader,
    VIIRSCldpropD3Reader,
)
from libera_utils.footprint_matching.readers.registry import FMATCH_MODE_READERS, _derive_mode_readers
from libera_utils.footprint_matching.types import OperationalMode, TimeMatch, VariableSpec
from libera_utils.io.product_definition import LiberaDataProductDefinition

_SPEC = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=frozenset({OperationalMode.CAM}))
# The product sets the readers declare, written out independently of the reader modules.
_ALL = frozenset(OperationalMode)
_IMAGER_PAIR = frozenset({OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME})
_ONLY_IMAGER = frozenset({OperationalMode.IMAGER})
_HOUR = np.timedelta64(1, "h")
_PASS_WINDOW = np.timedelta64(15, "m")
_PASS_MAX_OFFSET = np.timedelta64(3, "h")
# The ERA5 pressure-level fields and the 37 standard levels, hPa, in the order the reader declares them.
_PRESSURE_FIELDS = ("Temperature", "Geopotential", "Ozone_Mass_Mixing_Ratio", "Specific_Humidity", "Relative_Humidity")
_PRESSURE_LEVELS_HPA = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450)
_PRESSURE_LEVELS_HPA += (500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000)


def _mean(name: str, modes: frozenset[OperationalMode]) -> tuple:
    return (name, "float32", "weighted_mean", None, modes)


def _log_mean(name: str, modes: frozenset[OperationalMode]) -> tuple:
    return (name, "float32", "weighted_log_mean", None, modes)


def _code(name: str, n_categories: int | None, modes: frozenset[OperationalMode]) -> tuple:
    return (name, "int16", "weighted_mode", n_categories, modes)


def _ranked(base: str, n_categories: int, modes: frozenset[OperationalMode]) -> tuple[tuple, ...]:
    return tuple(
        (f"{base}_{rank.capitalize()}", "int16", f"weighted_mode_{rank}", n_categories, modes)
        for rank in ("primary", "secondary", "tertiary")
    )


# The FLASHFlux fields carried by FMATCH-IMAGER alone.
_FLASHFLUX_EXTENDED = (
    *(
        _mean(f"{base}_{layer}", _ONLY_IMAGER)
        for base in (
            "Layer_Coverage",
            "Cloud_Coverage_Multilayer",
            "Cloud_Top_Pressure",
            "Cloud_Base_Pressure",
            "Cloud_Particle_Phase_37um",
        )
        for layer in ("Lower", "Upper")
    ),
    _mean("Cloud_Optical_Depth_Upper", _ONLY_IMAGER),
    _mean("Cloud_Water_Particle_Radius_Upper", _ONLY_IMAGER),
    _mean("Cloud_Ice_Particle_Radius_Upper", _ONLY_IMAGER),
    _mean("Match_AOT", _ONLY_IMAGER),
    *(_mean(f"Aerosol_Type_Percentage_Type{i}", _ONLY_IMAGER) for i in range(7)),
    _mean("Imager_Dark_Target_Land_055um_Corrected_Aerosol_Optical_Depth", _ONLY_IMAGER),
    _mean("Imager_Deep_Blue_Ocean_055um_Aerosol_Optical_Depth", _ONLY_IMAGER),
    _mean("Surface_Albedo", _ONLY_IMAGER),
    _mean("TOA_Incoming_Solar_Radiation", _ONLY_IMAGER),
)


# Each built-in reader's declared attributes, written out independently of the reader modules:
# (key, class, INSTRUMENT, RESOLUTION_KM, TIME_MATCH, PASS_WINDOW, MAX_TIME_OFFSET, VARIABLES,
#  ADDITIONAL_PRODUCT_VARIABLES), each variable as
#  (name, dtype, aggregation, n_categories, modes).
# VARIABLES is None for ERA5_Pressure, whose 185 specs are pinned separately.
_DECLARED = [
    (
        "ERA5_Surface",
        ERA5SurfaceReader,
        "ECMWF",
        28.0,  # the 0.25 degree ERA5 grid spacing at the equator
        TimeMatch.NEAREST_STEP,
        None,
        _HOUR,
        (
            _mean("Wind_U10", _ALL),
            _mean("Wind_V10", _ALL),
            _mean("Temperature_2m", _IMAGER_PAIR),
            _mean("Dew_Point_Temperature_2m", _IMAGER_PAIR),
            _mean("Surface_Pressure", _IMAGER_PAIR),
            _mean("Surface_Geopotential", _IMAGER_PAIR),
            _mean("Forecast_Albedo", _IMAGER_PAIR),
        ),
        (),
    ),
    ("ERA5_Pressure", ERA5PressureLevelReader, "ECMWF", 28.0, TimeMatch.NEAREST_STEP, None, _HOUR, None, ()),
    (
        "IGBP",
        IGBPReader,
        "MODIS",
        0.463,  # the MCD12Q1 sinusoidal grid cell, 463.3 m
        TimeMatch.DAILY,
        None,
        None,
        (_code("Surface_Type", 20, _ALL),),
        _ranked("Surface_Type", 20, _ALL),
    ),
    (
        "NISE",
        NISEReader,
        "AMSR2",
        25.0,  # the nominal NISE EASE-Grid cell
        TimeMatch.DAILY,
        None,
        None,
        tuple(
            _mean(name, _ALL)
            for name in ("Sea_Ice_Concentration", "Snow_Free_Land", "Permanent_Ice", "Snow_On_Land", "Open_Ocean")
        ),
        (),
    ),
    (
        "VIIRS_AOD",
        VIIRSAODReader,
        "NOAA20",
        111.0,  # a 1 degree cell at the equator
        TimeMatch.DAILY,
        None,
        None,
        (
            _mean("Optical_Depth_550", _IMAGER_PAIR),
            _code("Aerosol_Type", 8, _IMAGER_PAIR),
        ),
        _ranked("Aerosol_Type", 8, _IMAGER_PAIR),
    ),
    (
        "VIIRS_BRDF",
        VIIRSBRDFReader,
        "NOAA20",
        5.6,  # the 0.05 degree CMG spacing at the equator
        TimeMatch.DAILY,
        None,
        None,
        tuple(
            _mean(f"{band}_{param}", _ALL) for band in ("Shortwave", "VIS", "NIR") for param in ("Fiso", "Fvol", "Fgeo")
        ),
        (),
    ),
    (
        "VIIRS_CLDPROP_D3",
        VIIRSCldpropD3Reader,
        "NOAA20",
        111.0,  # a 1 degree cell at the equator
        TimeMatch.DAILY,
        None,
        None,
        (_mean("Cloud_Optical_Thickness", _ALL), _mean("Cloud_Top_Pressure", _ALL)),
        (),
    ),
    (
        "FLASHFlux",
        FLASHFluxReader,
        "NOAA20",
        20.0,  # the CERES footprint at nadir
        TimeMatch.NEAREST_PASS,
        _PASS_WINDOW,
        _PASS_MAX_OFFSET,
        (
            _mean("Aerosol_Optical_Depth", _IMAGER_PAIR),
            _mean("Clear_Coverage", _IMAGER_PAIR),
            _mean("Cloud_Optical_Depth_Lower", _IMAGER_PAIR),
            _mean("Cloud_Water_Particle_Radius_Lower", _IMAGER_PAIR),
            _mean("Cloud_Ice_Particle_Radius_Lower", _IMAGER_PAIR),
            _code("Cloud_Classification", None, _IMAGER_PAIR),
            _code("Shortwave_ADM_Type", None, _IMAGER_PAIR),
            _code("Longwave_ADM_Type", None, _IMAGER_PAIR),
            *_FLASHFLUX_EXTENDED,
        ),
        (),
    ),
    (
        "CLDPIX",
        CLDPIXReader,
        "NOAA20",
        1.0,  # the VIIRS imager pixel
        TimeMatch.NEAREST_PASS,
        _PASS_WINDOW,
        _PASS_MAX_OFFSET,
        (
            _log_mean("Cloud_Optical_Depth", _IMAGER_PAIR),
            _mean("Cloud_Water_Path", _IMAGER_PAIR),
            _mean("Cloud_Effective_Temperature", _IMAGER_PAIR),
            _mean("Cloud_Effective_Height", _IMAGER_PAIR),
            _mean("Cloud_Effective_Pressure", _IMAGER_PAIR),
            _mean("Cloud_Top_Height", _IMAGER_PAIR),
            _mean("Cloud_Particle_Radius", _IMAGER_PAIR),
            _code("Cloud_Particle_Phase", 5, _IMAGER_PAIR),
            _code("Cloud_Mask", 4, _IMAGER_PAIR),
        ),
        (),
    ),
]
_BUILT_IN = {row[0]: row[1] for row in _DECLARED}

# The readers each product draws on, written out independently of the readers' specs.
_CAM_READERS = frozenset({"ERA5_Surface", "IGBP", "NISE", "VIIRS_BRDF", "VIIRS_CLDPROP_D3"})
_EXPECTED_MODE_READERS = {
    OperationalMode.CAM: _CAM_READERS,
    OperationalMode.CAM_CAMTIME: _CAM_READERS,
    OperationalMode.IMAGER: _CAM_READERS | {"VIIRS_AOD", "ERA5_Pressure", "CLDPIX", "FLASHFlux"},
    OperationalMode.IMAGER_CAMTIME: _CAM_READERS | {"VIIRS_AOD", "CLDPIX", "FLASHFlux"},
}
# Each product's variable count, standard-deviation companions and additional variables included.
_EXPECTED_PRODUCT_VARIABLE_COUNTS = {
    OperationalMode.CAM: 40,
    OperationalMode.CAM_CAMTIME: 40,
    OperationalMode.IMAGER: 505,
    OperationalMode.IMAGER_CAMTIME: 85,
}


def _stub_reader(key: str, modes: frozenset[OperationalMode]) -> type[StaticReader]:
    """Define and return a stub StaticReader under ``key`` with one variable carried by ``modes``."""

    class _Stub(StaticReader):
        READER_KEY = key
        INSTRUMENT = "FAKE"
        RESOLUTION_KM = 1.0
        TIME_MATCH = TimeMatch.DAILY
        VARIABLES = (VariableSpec(name="v", dtype="int16", aggregation="weighted_mode", modes=modes),)

        def _load_native(self):
            raise NotImplementedError

    return _Stub


def _as_tuples(specs: tuple[VariableSpec, ...]) -> tuple[tuple, ...]:
    return tuple((v.name, v.dtype, v.aggregation, v.n_categories, v.modes) for v in specs)


class TestLookup:
    @pytest.mark.parametrize(
        (
            "key",
            "cls",
            "instrument",
            "resolution_km",
            "time_match",
            "pass_window",
            "max_time_offset",
            "variables",
            "additional",
        ),
        [pytest.param(*row, id=row[0]) for row in _DECLARED],
    )
    def test_get_returns_the_registered_class(
        self, key, cls, instrument, resolution_km, time_match, pass_window, max_time_offset, variables, additional
    ):
        """``ReaderRegistry.get`` returns each built-in reader under its key, with the attributes it is declared with.

        Callers obtain readers by key, and the key prefixes every output variable name. ``INSTRUMENT`` is recorded in
        each variable's ``long_name``; the query box is padded by half of ``RESOLUTION_KM`` and the weigher sizes its
        coverage measure from it; ``TIME_MATCH``, ``PASS_WINDOW`` and ``MAX_TIME_OFFSET`` decide which samples a footprint is paired
        with; and ``VARIABLES`` fixes the order of the served planes, each variable's aggregation, dtype, category
        count and the products that carry it. A change to any of them renames, mis-aggregates or drops a product
        variable, or pairs a footprint with the wrong data.

        Parametrized over the nine built-in readers; asserts ``get(key)`` is the class and that its class attributes
        equal the table's, with every variable's name, dtype, aggregation, category count and products, in order.
        """
        assert ReaderRegistry.get(key) is cls
        assert cls.READER_KEY == key
        assert cls.INSTRUMENT == instrument
        assert cls.RESOLUTION_KM == resolution_km
        assert cls.TIME_MATCH is time_match
        assert cls.MAX_TIME_OFFSET == max_time_offset
        assert cls.PASS_WINDOW == pass_window
        if variables is not None:
            assert _as_tuples(cls.VARIABLES) == variables
        assert _as_tuples(cls.ADDITIONAL_PRODUCT_VARIABLES) == additional

    def test_pressure_variables_are_every_field_at_every_level(self):
        """``ERA5PressureLevelReader`` declares five fields at 37 levels, variable-major with levels ascending.

        FMATCH products are one-dimensional per footprint, so each level of each field is its own variable; the spec
        order is the order of the planes the reader writes, so names and planes must agree.

        Asserts the 185 spec names are ``<field>_<level>hPa`` for the five fields in turn, each over the 37 levels
        ascending from 1 to 1000 hPa, and that all are continuous float32 weighted means carried by FMATCH-IMAGER alone.
        """
        variables = ERA5PressureLevelReader.VARIABLES
        assert [v.name for v in variables] == [
            f"{field}_{level}hPa" for field in _PRESSURE_FIELDS for level in _PRESSURE_LEVELS_HPA
        ]
        assert {_as_tuples((v,))[0][1:] for v in variables} == {("float32", "weighted_mean", None, _ONLY_IMAGER)}

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
        assert "'ERA5_Surface'" in str(excinfo.value)

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
        """``FMATCH_MODE_READERS`` gives the CAM products none of the AOD, FLASHFlux, CLDPIX or ERA5 pressure-level readers.

        Those sources are inputs to the Climate Quality FMATCH-IMAGER products only. The CAM products run at
        near-real-time latency from mission start; a ``modes`` entry that named a CAM product by mistake would add
        these variables to products whose definitions do not declare them.

        Parametrized over the two CAM modes; asserts none of the four keys is in the mode's set.
        """
        assert not {"VIIRS_AOD", "FLASHFlux", "CLDPIX", "ERA5_Pressure"} & FMATCH_MODE_READERS[mode]

    def test_era5_pressure_levels_are_radiometer_timescale_only(self):
        """``ERA5_Pressure`` contributes to FMATCH-IMAGER but not to FMATCH-IMAGER-CAMTIME.

        The ERA5 pressure-level fields are a radiometer-timescale quantity, so the camera-timescale product omits them;
        this is the one reader that tells the two Climate Quality products apart, so it is the case most likely to be
        mis-declared.

        Asserts the key is in the IMAGER set and absent from the IMAGER_CAMTIME set.
        """
        assert "ERA5_Pressure" in FMATCH_MODE_READERS[OperationalMode.IMAGER]
        assert "ERA5_Pressure" not in FMATCH_MODE_READERS[OperationalMode.IMAGER_CAMTIME]

    def test_derived_table_equals_the_expected_reader_sets(self):
        """``FMATCH_MODE_READERS``, derived from the readers' ``VariableSpec.modes``, holds the expected reader sets.

        Each product's readers follow from the products its variables name, so a ``modes`` set that names a product
        by mistake adds a reader to it, and one that leaves a product out drops a reader.

        Asserts the derived table equals the reader sets written out in this module.
        """
        assert FMATCH_MODE_READERS == _EXPECTED_MODE_READERS

    @pytest.mark.parametrize("mode", list(OperationalMode))
    def test_each_product_carries_the_expected_variable_count(self, mode):
        """Each product carries its expected number of variables, companions and additional variables included.

        ``_DECLARED`` pins each reader's ``VARIABLES`` and ``ADDITIONAL_PRODUCT_VARIABLES`` but not the
        standard-deviation companions, which inherit their parent's ``modes``; this count pins the totals the product
        definitions are built from.

        Parametrized over every mode; asserts the specs of the mode's readers that name the mode number as expected.
        """
        specs = [
            spec
            for reader in ReaderRegistry.get_readers_for_mode(mode).values()
            for spec in reader.product_variable_specs()
            if mode in spec.modes
        ]
        assert len(specs) == _EXPECTED_PRODUCT_VARIABLE_COUNTS[mode]

    def test_derive_puts_a_reader_in_exactly_the_modes_its_specs_name(self, monkeypatch):
        """``_derive_mode_readers`` adds a reader to exactly the products its variables name.

        A reader contributes to a product when any of its variables names it. A derivation that added a reader to
        more products than its variables name would run it for products that carry none of its variables. A stub
        with a single mode tests that rule directly, independent of how the built-in readers happen to be declared.

        Copies the registry with ``monkeypatch``, defines a stub whose one variable names FMATCH-CAM, and asserts the
        derived table lists the stub under CAM and under no other mode.
        """
        monkeypatch.setattr(ReaderRegistry, "_registry", dict(ReaderRegistry._registry))
        _stub_reader("_registry_test_cam_only", frozenset({OperationalMode.CAM}))

        derived = _derive_mode_readers(ReaderRegistry._registry)

        assert {mode for mode, keys in derived.items() if "_registry_test_cam_only" in keys} == {OperationalMode.CAM}

    def test_derive_raises_when_a_mode_has_no_reader(self):
        """``_derive_mode_readers`` raises ``ValueError`` naming each product no reader supplies.

        A product with no reader would carry no ancillary variables, so the table must not be built with one.

        Derives from the IMAGER-only ``ERA5_Pressure`` reader alone and asserts a ``ValueError`` naming FMATCH-CAM.
        """
        with pytest.raises(ValueError, match="No reader supplies a variable for FMATCH-CAM, "):
            _derive_mode_readers({"ERA5_Pressure": ERA5PressureLevelReader})

    def test_a_reader_registered_after_import_is_not_in_the_table(self, monkeypatch):
        """A reader defined after the package import joins the registry but not ``FMATCH_MODE_READERS``.

        The table is filled once, from the built-in readers, so a stub defined by a test or a script cannot add itself
        to a product.

        Copies the registry with ``monkeypatch``, defines a stub naming every product, and asserts it is registered
        but absent from every mode's reader set and from ``get_readers_for_mode``.
        """
        monkeypatch.setattr(ReaderRegistry, "_registry", dict(ReaderRegistry._registry))
        stub = _stub_reader("_registry_test_late", frozenset(OperationalMode))

        assert ReaderRegistry.get("_registry_test_late") is stub
        for mode in OperationalMode:
            assert "_registry_test_late" not in FMATCH_MODE_READERS[mode]
            assert "_registry_test_late" not in ReaderRegistry.get_readers_for_mode(mode)


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


# The shipped FMATCH product definition of each operational mode.
_PRODUCT_DEFINITIONS = Path(libera_utils.__file__).parent / "data" / "product_definitions"
_DEFINITION_FILES = {
    OperationalMode.CAM: "fmatch_cam.yml",
    OperationalMode.CAM_CAMTIME: "fmatch_cam_camtime.yml",
    OperationalMode.IMAGER: "fmatch_imager.yml",
    OperationalMode.IMAGER_CAMTIME: "fmatch_imager_camtime.yml",
}


def _reader_product_names(mode: OperationalMode) -> set[str]:
    """Return the ``<READER_KEY>_<name>`` product variables the built-in readers write for ``mode``.

    A stacked spec is written as part of its stack, so the stack's name replaces its members'.
    """
    names: set[str] = set()
    for key, reader in ReaderRegistry.get_readers_for_mode(mode).items():
        stacks = reader.product_stacks()
        stacked = {member for stack in stacks for member in stack.members}
        specs = {spec.name: spec for spec in reader.product_variable_specs()}
        names |= {f"{key}_{name}" for name, spec in specs.items() if mode in spec.modes and name not in stacked}
        names |= {f"{key}_{stack.name}" for stack in stacks if mode in specs[stack.members[0]].modes}
    return names


@pytest.mark.parametrize("mode", list(OperationalMode))
def test_definition_declares_exactly_the_reader_variables(mode):
    """Each FMATCH product definition declares exactly the variables its readers produce, under the same names.

    The product variable name is ``<READER_KEY>_<spec name>``, with no translation between the readers and the
    definition, so a reader spec renamed without its definition (or the reverse) would leave a declared variable
    unfilled or an aggregated value with nowhere to go.

    Parametrized over every mode; collects the reader-prefixed names the mode's readers produce (stacks in place of
    their members) and asserts they equal the definition's variables that start with a built-in reader key.
    """
    definition = LiberaDataProductDefinition.from_yaml(_PRODUCT_DEFINITIONS / _DEFINITION_FILES[mode])
    prefixes = tuple(f"{key}_" for key in _BUILT_IN)
    declared = {name for name in definition.variables if name.startswith(prefixes)}
    assert declared == _reader_product_names(mode)
