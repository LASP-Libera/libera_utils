"""Unit tests for the FMATCH product definitions and their loaders.

There is one SSF-style product definition per FMATCH operational mode. The
radiometer-timescale FMATCH-IMAGER carries the RBSP CLDPIX/FLASHFlux cloud fields
alongside the full ERA5 single-level and pressure-level fields and the VIIRS
imager fields. These tests confirm, for every shipped definition, that:
- The product ID is registered as an auxiliary (AUX) DataProductIdentifier and
  matches the OperationalMode value string.
- The product definition YAML loads and validates via LiberaDataProductDefinition.
- The schema declares the expected geolocation, derived-geometry, and QA variables
  on the correct (radiometer vs camera) time dimension.
- The external (reader-sourced) variables stay in sync with the reader plugins'
  VariableSpec definitions for the mode. Every reader-sourced variable is declared
  as `<READER_KEY>_<spec_name>` (e.g. ERA5_Surface_Wind_U10), or
  `<READER_KEY>_<stack_name>` for a reader's stacked specs (e.g.
  ERA5_Pressure_Temperature on PRESSURE_LEVEL). The reader's instrument is recorded in
  each variable's ``long_name`` instead (e.g. "... (ECMWF)").
- A small dummy dataset round-trips through create/enforce/check conformance.
"""

from __future__ import annotations

import numpy as np
import pytest

# Importing the readers subpackage registers all built-in readers so we can
# cross-check the product definitions against their VariableSpecs.
import libera_utils.footprint_matching.readers  # noqa: F401
from libera_utils.constants import DataLevel, DataProductIdentifier
from libera_utils.footprint_matching.product import (
    FMATCH_DEFINITION_FILENAMES,
    FMATCH_INTEGER_FILL_VALUE,
    fmatch_time_variable,
    load_fmatch_definition,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.product_definition import LiberaDataProductDefinition, LiberaVariableDefinition

# The production reader keys. Intersecting with these makes the cross-check robust
# against throwaway readers other tests register into the shared ReaderRegistry
# (e.g. _FakeReader in test_base.py).
PRODUCTION_READER_KEYS = frozenset(
    {
        "ERA5_Surface",
        "ERA5_Pressure",
        "IGBP",
        "NISE",
        "VIIRS_BRDF",
        "VIIRS_CLDPROP_D3",
        "FLASHFlux",
        "CLDPIX",
        "VIIRS_AOD",
    }
)

# Variables that always appear regardless of mode.
GEOLOCATION_VARIABLES = (
    "Latitude",
    "Longitude",
    "Altitude",
    "Solar_Zenith_Surface",
    "Viewing_Zenith_Surface",
    "Relative_Azimuth_Surface",
)
DERIVED_GEOMETRY_VARIABLES = ("Sunglint_Angle",)
COVERAGE_QA_VARIABLES = ("PSF_Coverage_Fraction", "Quality_Flag")

ALL_MODES = tuple(OperationalMode)


def _production_readers_for_mode(mode: OperationalMode) -> dict:
    """Active production readers for a mode (excludes test-injected readers)."""
    return {key: cls for key, cls in ReaderRegistry.get_readers_for_mode(mode).items() if key in PRODUCTION_READER_KEYS}


def _expected_external_variables(mode: OperationalMode) -> dict[str, tuple[str, str | None]]:
    """{variable_name: (dtype, stack dimension or None)} for every active production reader variable.

    Every reader-sourced variable is named `<READER_KEY>_<spec_name>` (e.g.
    ERA5_Surface_Wind_U10, IGBP_Surface_Type), or `<READER_KEY>_<stack_name>` for specs a
    reader stacks along a further dimension (e.g. ERA5_Pressure_Temperature on
    PRESSURE_LEVEL). The reader's instrument is not part of the
    name -- it is recorded in the variable's ``long_name`` instead (see
    ``test_reader_instrument_is_recorded_in_long_name``). A spec is carried when ``mode`` is in
    its ``modes`` (e.g. the extended FLASHFlux cloud/aerosol/albedo fields appear in
    FMATCH-IMAGER only).
    """
    expected: dict[str, tuple[str, str | None]] = {}
    for key, cls in _production_readers_for_mode(mode).items():
        # product_variable_specs() == the read VARIABLES plus derived outputs
        # (per-continuous-variable standard-deviation companions and reader-specific
        # extras such as IGBP's ranked scenes). Less the stacked members plus the stacks,
        # it is the full set that appears in the product definition, so this is what the
        # YAMLs must match.
        specs = {spec.name: spec for spec in cls.product_variable_specs()}
        stacks = cls.product_stacks()
        stacked_members = {member for stack in stacks for member in stack.members}
        for name, spec in specs.items():
            if name not in stacked_members and mode in spec.modes:
                expected[f"{key}_{name}"] = (spec.dtype, None)
        for stack in stacks:
            member = specs[stack.members[0]]
            if mode in member.modes:
                expected[f"{key}_{stack.name}"] = (member.dtype, stack.dimension)
    return expected


@pytest.fixture(scope="module")
def definitions() -> dict[OperationalMode, LiberaDataProductDefinition]:
    """All shipped FMATCH product definitions keyed by mode."""
    return {mode: load_fmatch_definition(mode) for mode in ALL_MODES}


class TestFmatchIdentifiers:
    """Every mode's product ID must be an AUX member matching the mode string."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_product_id_registered_as_aux(self, mode):
        product = DataProductIdentifier(mode.value)
        assert product.data_level is DataLevel.AUX

    def test_all_modes_have_a_definition_file(self):
        # The product module must map every operational mode to a YAML file.
        assert set(FMATCH_DEFINITION_FILENAMES) == set(ALL_MODES)


class TestImagerContent:
    """Guards the FMATCH-IMAGER content: it carries both the RBSP and the ERA5 fields."""

    def test_imager_has_rbsp_variables(self, definitions):
        definition = definitions[OperationalMode.IMAGER]
        assert any(name.startswith("CLDPIX_") for name in definition.variables)
        assert any(name.startswith("FLASHFlux_") for name in definition.variables)

    def test_imager_keeps_full_era5_field_set(self, definitions):
        definition = definitions[OperationalMode.IMAGER]
        # The winds (every product), the single-level fields, and the pressure-level
        # fields all sit in the FMATCH-IMAGER product alongside the RBSP fields
        # (spot-check one per family; full sync is covered by
        # test_external_variables_match_readers).
        assert "ERA5_Surface_Wind_U10" in definition.variables
        assert "ERA5_Surface_Temperature_2m" in definition.variables
        assert "ERA5_Surface_Forecast_Albedo" in definition.variables
        assert "ERA5_Pressure_Temperature" in definition.variables
        assert "ERA5_Pressure_Relative_Humidity_Standard_Deviation" in definition.variables

    def test_extended_flashflux_fields_are_imager_only(self, definitions):
        # The extended FLASHFlux cloud-layer / aerosol / albedo fields are carried by
        # FMATCH-IMAGER alone (their modes) and must NOT leak into FMATCH-IMAGER-CAMTIME, the
        # other product that activates the FLASHFlux reader. Spot-check one field per secondary
        # axis (layer, aerosol type, albedo band) plus a std companion.
        extended = (
            "FLASHFlux_Layer_Coverage_Upper",
            "FLASHFlux_Cloud_Optical_Depth_Upper",
            "FLASHFlux_Cloud_Top_Pressure_Lower_Standard_Deviation",
            "FLASHFlux_Match_AOT",
            "FLASHFlux_Aerosol_Type_Percentage_Type6",
            "FLASHFlux_Surface_Albedo",
        )
        imager = definitions[OperationalMode.IMAGER]
        for name in extended:
            assert name in imager.variables, f"FMATCH-IMAGER missing extended field {name}"
        camtime = definitions[OperationalMode.IMAGER_CAMTIME]
        leaked = [name for name in extended if name in camtime.variables]
        assert leaked == [], f"FMATCH-IMAGER-CAMTIME must not declare IMAGER-only FLASHFlux fields, found {leaked}"
        # The base FLASHFlux fields, by contrast, still appear in IMAGER-CAMTIME.
        assert "FLASHFlux_Cloud_Optical_Depth_Lower" in camtime.variables


class TestFmatchDefinitions:
    """Each YAML loads and declares the expected structure."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_definition_loads_with_matching_product_id(self, mode, definitions):
        definition = definitions[mode]
        assert isinstance(definition, LiberaDataProductDefinition)
        assert definition.attributes["ProductID"] == mode.value

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_time_coordinate_matches_timescale(self, mode, definitions):
        definition = definitions[mode]
        time_var = fmatch_time_variable(mode)
        assert time_var in definition.coordinates
        assert definition.coordinates[time_var].dtype == "datetime64[ns]"
        # Both timescales use a 1-D dimension coordinate (name == dimension): RADIOMETER_TIME for the radiometer
        # products, CAMERA_TIME for the camera-timescale products' 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid.
        assert definition.coordinates[time_var].dimensions == [time_var]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_common_variables_present(self, mode, definitions):
        definition = definitions[mode]
        for name in GEOLOCATION_VARIABLES + DERIVED_GEOMETRY_VARIABLES + COVERAGE_QA_VARIABLES:
            assert name in definition.variables, f"{mode.value} missing {name}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_external_variables_match_readers(self, mode, definitions):
        definition = definitions[mode]
        for name, (dtype, _) in _expected_external_variables(mode).items():
            assert name in definition.variables, f"{mode.value} missing external variable {name}"
            assert definition.variables[name].dtype == dtype, (
                f"{mode.value} dtype drift for {name}: definition has "
                f"{definition.variables[name].dtype}, reader has {dtype}"
            )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_reader_instrument_is_recorded_in_long_name(self, mode, definitions):
        # The instrument token is carried in the variable's long_name rather than the
        # variable name (e.g. ERA5_Surface_Wind_U10 has long_name "... (ECMWF)"). Guard that every
        # reader-sourced variable's long_name carries its reader's INSTRUMENT, so
        # provenance is not silently lost from the ~hundreds of hand-written long_names.
        definition = definitions[mode]
        readers = _production_readers_for_mode(mode)
        for name in _expected_external_variables(mode):
            instrument = next(cls.INSTRUMENT for key, cls in readers.items() if name.startswith(f"{key}_"))
            long_name = definition.variables[name].attributes.get("long_name", "")
            assert instrument in long_name, (
                f"{mode.value}/{name}: long_name {long_name!r} is missing the reader instrument {instrument!r}"
            )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_variables_use_mode_record_dimension(self, mode, definitions):
        # Radiometer modes hang variables on the 1-D RADIOMETER_TIME axis; camera-timescale modes hang every variable
        # on the 2-D (CAMERA_TIME, PSEUDOFOOTPRINT) grid. A reader's stacked variable adds its stack's dimension.
        time_var = fmatch_time_variable(mode)
        expected_dims = ["CAMERA_TIME", "PSEUDOFOOTPRINT"] if time_var == "CAMERA_TIME" else [time_var]
        stacked = {name: dim for name, (_, dim) in _expected_external_variables(mode).items() if dim is not None}
        for name, var_def in definitions[mode].variables.items():
            dims = expected_dims + [stacked[name]] if name in stacked else expected_dims
            assert var_def.dimensions == dims, f"{mode.value}/{name} wrong dimension"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_no_duplicate_variable_names(self, mode, definitions):
        # Sanity: collision prefixing must leave a unique variable set.
        definition = definitions[mode]
        all_names = list(definition.variables) + list(definition.coordinates)
        assert len(all_names) == len(set(all_names))

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_integer_columns_declare_the_fmatch_fill(self, mode, definitions):
        # 0 is a real category, pixel index or pixel bound, so every integer column but a bitmask and a
        # dimension index declares FMATCH_INTEGER_FILL_VALUE, the fill SCENE-ID reads back.
        definition = definitions[mode]
        entries = {**definition.coordinates, **definition.variables}
        undeclared = {
            name: entry.attributes.get("_FillValue")
            for name, entry in entries.items()
            if np.issubdtype(np.dtype(entry.dtype), np.integer)
            and "flag_masks" not in entry.attributes
            and entry.dimensions != [name]
            and entry.attributes.get("_FillValue") != FMATCH_INTEGER_FILL_VALUE
        }
        assert undeclared == {}

    def test_imager_camtime_has_no_era5_pressure_variables(self, definitions):
        # The ERA5 pressure-level fields are a radiometer-timescale quantity: the
        # camera-timescale FMATCH-IMAGER-CAMTIME product must not declare them, while
        # the radiometer FMATCH-IMAGER product still does. Standing guard against the
        # pressure block drifting back into the camtime YAML (there is no general
        # no-extra-variables check).
        camtime = definitions[OperationalMode.IMAGER_CAMTIME]
        offenders = [name for name in camtime.variables if name.startswith("ERA5_Pressure_")]
        assert offenders == [], f"FMATCH-IMAGER-CAMTIME must not declare ERA5_Pressure variables, found {offenders}"
        imager = definitions[OperationalMode.IMAGER]
        assert any(name.startswith("ERA5_Pressure_") for name in imager.variables)


class TestDerivedProductVariables:
    """Guards the rules that turn read VARIABLES into product output variables."""

    def test_continuous_variable_gets_standard_deviation_companion(self):
        # ERA5 winds are mean-aggregated (continuous), so each must gain a
        # `<name>_Standard_Deviation` companion in the product variable list.
        era5 = ReaderRegistry.get("ERA5_Surface")
        names = {spec.name for spec in era5.product_variable_specs()}
        assert "Wind_U10" in names
        assert "Wind_U10_Standard_Deviation" in names
        assert "Wind_V10_Standard_Deviation" in names

    def test_mode_aggregated_variable_has_no_standard_deviation_companion(self):
        # FLASHFlux's encoded scene-type codes have n_categories=None but are
        # weighted_mode, so a within-footprint standard deviation is meaningless
        # and must NOT be generated. This guards the mean-only rule.
        flashflux = ReaderRegistry.get("FLASHFlux")
        names = {spec.name for spec in flashflux.product_variable_specs()}
        for encoded in ("Cloud_Classification", "Shortwave_ADM_Type", "Longwave_ADM_Type"):
            assert encoded in names
            assert f"{encoded}_Standard_Deviation" not in names

    def test_igbp_reports_ranked_scenes_but_no_standard_deviation(self):
        # IGBP keeps the single aggregated Surface_Type plus three ranked-scene
        # outputs; being categorical (weighted_mode) it gets no std-dev companion.
        igbp = ReaderRegistry.get("IGBP")
        names = {spec.name for spec in igbp.product_variable_specs()}
        assert {
            "Surface_Type",
            "Surface_Type_Primary",
            "Surface_Type_Secondary",
            "Surface_Type_Tertiary",
        } <= names
        assert "Surface_Type_Standard_Deviation" not in names


class TestFmatchConformance:
    """A dummy dataset must round-trip through create/enforce/check for every definition."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_roundtrip(self, mode, definitions):
        definition = definitions[mode]
        time_var = fmatch_time_variable(mode)
        n_footprints = 4
        times = np.array(
            ["2026-06-11T00:00:00", "2026-06-11T00:00:01", "2026-06-11T00:00:02", "2026-06-11T00:00:03"],
            dtype="datetime64[ns]",
        )
        # Sizes for every dimension a coordinate/variable may reference: the radiometer record axis (RADIOMETER_TIME),
        # the camera-timescale grid axes (CAMERA_TIME, PSEUDOFOOTPRINT) and the fixed-size PRESSURE_LEVEL axis of the
        # stacked ERA5 variables. CAMERA_TIME must match the length of the times array below (it is the 1-D
        # camera-timescale dimension coordinate).
        dimension_sizes = {
            "RADIOMETER_TIME": n_footprints,
            "CAMERA_TIME": n_footprints,
            "PSEUDOFOOTPRINT": n_footprints,
            "PRESSURE_LEVEL": LiberaVariableDefinition._get_standard_dimensions()["PRESSURE_LEVEL"].size,
        }

        def _zeros(var_def):
            return np.zeros(tuple(dimension_sizes[d] for d in var_def.dimensions), dtype=var_def.dtype)

        data: dict[str, np.ndarray] = {time_var: times}
        # Every non-time coordinate (e.g. the camera_pixel_{x,y}_{min,max} coords on the camtime grid) and every
        # variable must be present for the conformance check to pass.
        for name, var_def in definition.coordinates.items():
            if name != time_var:
                data[name] = _zeros(var_def)
        for name, var_def in definition.variables.items():
            data[name] = _zeros(var_def)

        dynamic_attrs = {
            "algorithm_version": "1.0.0",
            "InputGranules": "dummy_l1b.nc",
        }
        dataset = definition.create_product_dataset(data, dynamic_product_attributes=dynamic_attrs)
        dataset = definition.enforce_dataset_conformance(dataset)
        errors = definition.check_dataset_conformance(dataset, strict=True)
        assert errors == []
