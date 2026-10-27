"""Unit tests for the reader registry.

Tests confirm:
- importing the readers package registers every built-in reader under its key, with its declared
  instrument, resolution, time policy and variables
- an unknown key raises a KeyError naming the registered keys
- each operational mode resolves to its per-product reader set (FMATCH_MODE_READERS)
- a second class registering a key another class holds is rejected at definition
"""

import numpy as np
import pytest

import libera_utils.footprint_matching.readers  # noqa: F401  (registers the built-in readers)
from libera_utils.footprint_matching.readers import (
    CLDPIXReader,
    ERA5PressureLevelReader,
    ERA5SurfaceReader,
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

_SPEC = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM)
_CAM = OperationalMode.CAM
_FLASH = OperationalMode.IMAGER_FLASH
_IMAGER = OperationalMode.IMAGER
_HOUR = np.timedelta64(1, "h")
_PASS_WINDOW = np.timedelta64(15, "m")
_PASS_MAX_OFFSET = np.timedelta64(3, "h")
# The ERA5 pressure-level fields and the 37 standard levels, hPa, in the order the reader declares them.
_PRESSURE_FIELDS = ("temperature", "geopotential", "ozone_mass_mixing_ratio", "specific_humidity", "relative_humidity")
_PRESSURE_LEVELS_HPA = (1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450)
_PRESSURE_LEVELS_HPA += (500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000)


def _mean(name: str, mode: OperationalMode, only_modes: tuple | None = None) -> tuple:
    return (name, "float32", "weighted_mean", None, mode, only_modes)


def _log_mean(name: str, mode: OperationalMode, only_modes: tuple | None = None) -> tuple:
    return (name, "float32", "weighted_log_mean", None, mode, only_modes)


def _code(name: str, n_categories: int | None, mode: OperationalMode) -> tuple:
    return (name, "int16", "weighted_mode", n_categories, mode, None)


def _ranked(base: str, n_categories: int, mode: OperationalMode) -> tuple[tuple, ...]:
    return tuple(
        (f"{base}_{rank}", "int16", f"weighted_mode_{rank}", n_categories, mode, None)
        for rank in ("primary", "secondary", "tertiary")
    )


# The SSF fields pinned to FMATCH-IMAGER alone.
_IMAGER_ONLY = (_IMAGER,)
_SSF_IMAGER_ONLY = (
    *(
        _mean(f"{base}_{layer}", _IMAGER, _IMAGER_ONLY)
        for base in (
            "layer_coverage",
            "cloud_coverage_multilayer",
            "cloud_top_pressure",
            "cloud_base_pressure",
            "cloud_particle_phase_37um",
        )
        for layer in ("lower", "upper")
    ),
    _log_mean("cloud_optical_depth_upper", _IMAGER, _IMAGER_ONLY),
    _mean("cloud_water_particle_radius_upper", _IMAGER, _IMAGER_ONLY),
    _mean("cloud_ice_particle_radius_upper", _IMAGER, _IMAGER_ONLY),
    _log_mean("match_aot", _IMAGER, _IMAGER_ONLY),
    *(_mean(f"aerosol_type_percentage_type{i}", _IMAGER, _IMAGER_ONLY) for i in range(7)),
    _log_mean("imager_dark_target_land_055um_corrected_aerosol_optical_depth", _IMAGER, _IMAGER_ONLY),
    _log_mean("imager_deep_blue_ocean_055um_aerosol_optical_depth", _IMAGER, _IMAGER_ONLY),
    _mean("surface_albedo", _IMAGER, _IMAGER_ONLY),
    _mean("toa_incoming_solar_radiation", _IMAGER, _IMAGER_ONLY),
)


# Each built-in reader's declared attributes, written out independently of the reader modules:
# (key, class, INSTRUMENT, RESOLUTION_KM, TIME_MATCH, PASS_WINDOW, MAX_TIME_OFFSET, VARIABLES,
#  ADDITIONAL_PRODUCT_VARIABLES), each variable as
#  (name, dtype, aggregation, n_categories, required_mode, only_modes).
# VARIABLES is None for era5_pressure, whose 185 specs are pinned separately.
_DECLARED = [
    (
        "era5_surface",
        ERA5SurfaceReader,
        "ECMWF",
        28.0,  # the 0.25 degree ERA5 grid spacing at the equator
        TimeMatch.NEAREST_STEP,
        None,
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
    ("era5_pressure", ERA5PressureLevelReader, "ECMWF", 28.0, TimeMatch.NEAREST_STEP, None, _HOUR, None, ()),
    (
        "igbp",
        IGBPReader,
        "MODIS",
        0.463,  # the MCD12Q1 sinusoidal grid cell, 463.3 m
        TimeMatch.DAILY,
        None,
        None,
        (_code("surface_type", 20, _CAM),),
        _ranked("surface_type", 20, _CAM),
    ),
    (
        "nise",
        NISEReader,
        "AMSR2",
        25.0,  # the nominal NISE EASE-Grid cell
        TimeMatch.DAILY,
        None,
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
        None,
        (
            _log_mean("aod_550", _IMAGER),
            _code("aerosol_type", 8, _IMAGER),
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
        None,
        tuple(
            _mean(f"brdf_{band}_{param}", _CAM)
            for band in ("shortwave", "vis", "nir")
            for param in ("fiso", "fvol", "fgeo")
        ),
        (),
    ),
    (
        "viirs_cldprop_d3",
        VIIRSCldpropD3Reader,
        "NOAA20",
        111.0,  # a 1 degree cell at the equator
        TimeMatch.DAILY,
        None,
        None,
        (_log_mean("cloud_optical_thickness", _CAM), _mean("cloud_top_pressure", _CAM)),
        (),
    ),
    (
        "ssf",
        SSFReader,
        "NOAA20",
        20.0,  # the CERES footprint at nadir
        TimeMatch.NEAREST_PASS,
        _PASS_WINDOW,
        _PASS_MAX_OFFSET,
        (
            _log_mean("aerosol_optical_depth", _FLASH),
            _mean("clear_coverage", _FLASH),
            _log_mean("cloud_optical_depth_lower", _FLASH),
            _mean("cloud_water_particle_radius_lower", _FLASH),
            _mean("cloud_ice_particle_radius_lower", _FLASH),
            _code("cloud_classification", None, _FLASH),
            _code("shortwave_adm_type", None, _FLASH),
            _code("longwave_adm_type", None, _FLASH),
            *_SSF_IMAGER_ONLY,
        ),
        (),
    ),
    (
        "cldpix",
        CLDPIXReader,
        "NOAA20",
        1.0,  # the VIIRS imager pixel
        TimeMatch.NEAREST_PASS,
        _PASS_WINDOW,
        _PASS_MAX_OFFSET,
        (
            _log_mean("cloud_optical_depth", _IMAGER),
            _mean("cloud_water_path", _IMAGER),
            _mean("cloud_effective_temperature", _IMAGER),
            _mean("cloud_effective_height", _IMAGER),
            _mean("cloud_effective_pressure", _IMAGER),
            _mean("cloud_top_height", _IMAGER),
            _mean("cloud_particle_radius", _IMAGER),
            _code("cloud_particle_phase", 5, _IMAGER),
            _code("cloud_mask", 4, _IMAGER),
        ),
        (),
    ),
]
_BUILT_IN = {row[0]: row[1] for row in _DECLARED}


def _as_tuples(specs: tuple[VariableSpec, ...]) -> tuple[tuple, ...]:
    return tuple((v.name, v.dtype, v.aggregation, v.n_categories, v.required_mode, v.only_modes) for v in specs)


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
        count and the products that carry it (``only_modes`` pins a variable to exact products). A change to any of them renames, mis-aggregates or drops a product
        variable, or pairs a footprint with the wrong data.

        Parametrized over the nine built-in readers; asserts ``get(key)`` is the class and that its class attributes
        equal the table's, with every variable's name, dtype, aggregation, category count and gating modes, in order.
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
        ascending from 1 to 1000 hPa, and that all are IMAGER-gated continuous float32 weighted means.
        """
        variables = ERA5PressureLevelReader.VARIABLES
        assert [v.name for v in variables] == [
            f"{field}_{level}hPa" for field in _PRESSURE_FIELDS for level in _PRESSURE_LEVELS_HPA
        ]
        assert {_as_tuples((v,))[0][1:] for v in variables} == {("float32", "weighted_mean", None, _IMAGER, None)}

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
        assert "'era5_surface'" in str(excinfo.value)

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
