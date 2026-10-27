"""Unit tests for the reader registry.

Tests confirm:
- importing the readers package registers every gridded reader under its key, with its declared
  instrument, resolution, time policy and variables
- an unknown key raises a KeyError naming the registered keys
- a second class registering a key another class holds is rejected at definition
"""

import numpy as np
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
    "era5_single_level": ERA5Reader,
    "era5_pressure": ERA5PressureLevelReader,
    "igbp": IGBPReader,
    "nise": NISEReader,
    "viirs_aod": VIIRSAODReader,
    "viirs_brdf": VIIRSBRDFReader,
}
_SPEC = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM)
_CAM = OperationalMode.CAM
_IMAGER = OperationalMode.IMAGER
_HOUR = np.timedelta64(1, "h")
# The ERA5 pressure-level fields and the 37 standard levels, hPa, in the order the reader declares them.
_PRESSURE_FIELDS = ("temperature", "geopotential", "ozone_mass_mixing_ratio", "specific_humidity", "relative_humidity")
_PRESSURE_LEVELS_HPA = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450)
_PRESSURE_LEVELS_HPA += (500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000)


def _mean(name: str, mode: OperationalMode) -> tuple:
    return (name, "float32", "weighted_mean", None, mode)


def _ranked(base: str, n_categories: int, mode: OperationalMode) -> tuple[tuple, ...]:
    return tuple(
        (f"{base}_{rank}", "int16", f"weighted_mode_{rank}", n_categories, mode)
        for rank in ("primary", "secondary", "tertiary")
    )


# Each built-in reader's declared attributes, written out independently of the reader modules:
# (key, class, INSTRUMENT, RESOLUTION_KM, TIME_MATCH, MAX_TIME_OFFSET, VARIABLES,
#  ADDITIONAL_PRODUCT_VARIABLES), each variable as (name, dtype, aggregation, n_categories, required_mode).
# VARIABLES is None for era5_pressure, whose 185 specs are pinned separately.
_DECLARED = [
    (
        "era5_single_level",
        ERA5Reader,
        "ECMWF",
        28.0,  # the 0.25 degree ERA5 grid spacing at the equator
        TimeMatch.NEAREST_STEP,
        _HOUR,
        (
            _mean("wind_u10", _CAM),
            _mean("wind_v10", _CAM),
            _mean("temperature_2m", _IMAGER),
            _mean("dew_point_temperature_2m", _IMAGER),
            _mean("surface_pressure", _IMAGER),
            _mean("surface_geopotential", _IMAGER),
            _mean("forecast_albedo", _IMAGER),
        ),
        (),
    ),
    ("era5_pressure", ERA5PressureLevelReader, "ECMWF", 28.0, TimeMatch.NEAREST_STEP, _HOUR, None, ()),
    (
        "igbp",
        IGBPReader,
        "MODIS",
        0.463,  # the MCD12Q1 sinusoidal grid cell, 463.3 m
        TimeMatch.DAILY,
        None,
        (("surface_type", "int16", "weighted_mode", 20, _CAM),),
        _ranked("surface_type", 20, _CAM),
    ),
    (
        "nise",
        NISEReader,
        "AMSR2",
        25.0,  # the nominal NISE EASE-Grid cell
        TimeMatch.DAILY,
        None,
        tuple(
            _mean(name, _CAM)
            for name in ("sea_ice_concentration", "snow_free_land", "permanent_ice", "snow_on_land", "open_ocean")
        ),
        (),
    ),
    (
        "viirs_aod",
        VIIRSAODReader,
        "NOAA20",
        111.0,  # a 1 degree cell at the equator
        TimeMatch.DAILY,
        None,
        (
            ("aod_550", "float32", "weighted_log_mean", None, _IMAGER),
            ("aerosol_type", "int16", "weighted_mode", 8, _IMAGER),
        ),
        _ranked("aerosol_type", 8, _IMAGER),
    ),
    (
        "viirs_brdf",
        VIIRSBRDFReader,
        "NOAA20",
        5.6,  # the 0.05 degree CMG spacing at the equator
        TimeMatch.DAILY,
        None,
        tuple(
            _mean(f"brdf_{band}_{param}", _CAM)
            for band in ("shortwave", "vis", "nir")
            for param in ("fiso", "fvol", "fgeo")
        ),
        (),
    ),
]


def _as_tuples(specs: tuple[VariableSpec, ...]) -> tuple[tuple, ...]:
    return tuple((v.name, v.dtype, v.aggregation, v.n_categories, v.required_mode) for v in specs)


class TestLookup:
    @pytest.mark.parametrize(
        ("key", "cls", "instrument", "resolution_km", "time_match", "max_time_offset", "variables", "additional"),
        [pytest.param(*row, id=row[0]) for row in _DECLARED],
    )
    def test_get_returns_the_registered_class(
        self, key, cls, instrument, resolution_km, time_match, max_time_offset, variables, additional
    ):
        """``ReaderRegistry.get`` returns each built-in reader under its key, with the attributes it is declared with.

        Callers obtain readers by key, and the key prefixes every output variable name. ``INSTRUMENT`` is recorded in
        each variable's ``long_name``; the query box is padded by half of ``RESOLUTION_KM`` and the weigher sizes its
        coverage measure from it; ``TIME_MATCH`` and ``MAX_TIME_OFFSET`` decide which samples a footprint is paired
        with; and ``VARIABLES`` fixes the order of the served planes, each variable's aggregation, dtype, category
        count and the products that carry it. A change to any of them renames, mis-aggregates or drops a product
        variable, or pairs a footprint with the wrong data.

        Parametrized over the six gridded readers; asserts ``get(key)`` is the class and that its class attributes
        equal the table's, with every variable's name, dtype, aggregation, category count and gating mode.
        """
        assert ReaderRegistry.get(key) is cls
        assert cls.READER_KEY == key
        assert cls.INSTRUMENT == instrument
        assert cls.RESOLUTION_KM == resolution_km
        assert cls.TIME_MATCH is time_match
        assert cls.MAX_TIME_OFFSET == max_time_offset
        assert cls.PASS_WINDOW is None
        if variables is not None:
            assert _as_tuples(cls.VARIABLES) == variables
        assert _as_tuples(cls.ADDITIONAL_PRODUCT_VARIABLES) == additional

    def test_pressure_variables_are_every_field_at_every_level(self):
        """``ERA5PressureLevelReader`` declares five fields at 37 levels, variable-major with levels ascending.

        FMATCH products are one-dimensional per footprint, so each level of each field is its own variable; the spec
        order is the order of the planes the reader writes, so names and planes must agree.

        Asserts the 185 spec names are ``<field>_<level>hPa`` for the five fields in turn, each over the 37 levels
        ascending from 1 to 1000 hPa, and that all are IMAGER-gated continuous float32 weighted means.
        """
        variables = ERA5PressureLevelReader.VARIABLES
        assert [v.name for v in variables] == [
            f"{field}_{level}hPa" for field in _PRESSURE_FIELDS for level in _PRESSURE_LEVELS_HPA
        ]
        assert {_as_tuples((v,))[0][1:] for v in variables} == {("float32", "weighted_mean", None, _IMAGER)}

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
