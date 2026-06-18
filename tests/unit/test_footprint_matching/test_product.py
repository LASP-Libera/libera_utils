"""Unit tests for the FMATCH product definitions and their loaders.

There is one SSF-style product definition per FMATCH operational mode. These
tests confirm, for every mode, that:
- The product ID is registered as an ancillary (ANC) DataProductIdentifier and
  matches the OperationalMode value string.
- The product definition YAML loads and validates via LiberaDataProductDefinition.
- The schema declares the expected geolocation, derived-geometry, and QA variables
  on the correct (radiometer vs camera) time dimension.
- The external (reader-sourced) variables stay in sync with the reader plugins'
  VariableSpec definitions. Every reader-sourced variable is named
  `<source_key>_<instrument>_<spec_name>` (e.g. era5_ECMWF_wind_u10,
  igbp_MODIS_surface_type, cldpix_NOAA20_cloud_mask).
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
    fmatch_time_variable,
    load_fmatch_definition,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.product_definition import LiberaDataProductDefinition

# The production reader keys. Intersecting with these makes the cross-check robust
# against throwaway readers other tests register into the shared ReaderRegistry
# (e.g. _FakeReader in test_base.py).
PRODUCTION_READER_KEYS = frozenset({"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud", "ssf", "cldpix", "viirs_aod"})

# Variables that always appear regardless of mode.
GEOLOCATION_VARIABLES = (
    "latitude",
    "longitude",
    "altitude",
    "solar_zenith_angle",
    "viewing_zenith_angle",
    "relative_azimuth_angle",
)
DERIVED_GEOMETRY_VARIABLES = ("sunglint_angle",)
COVERAGE_QA_VARIABLES = ("psf_coverage_fraction", "q_flags")

ALL_MODES = tuple(OperationalMode)


def _production_readers_for_mode(mode: OperationalMode) -> dict:
    """Active production readers for a mode (excludes test-injected readers)."""
    return {key: cls for key, cls in ReaderRegistry.get_readers_for_mode(mode).items() if key in PRODUCTION_READER_KEYS}


def _expected_external_variables(mode: OperationalMode) -> dict[str, str]:
    """{output_variable_name: dtype} for every active production reader variable.

    Mirrors the product-definition naming rule: every reader-sourced variable is
    named `<source_key>_<instrument>_<spec_name>`, where the instrument token comes
    from the reader's INSTRUMENT attribute (e.g. era5_ECMWF_wind_u10,
    igbp_MODIS_surface_type).
    """
    expected: dict[str, str] = {}
    for key, cls in _production_readers_for_mode(mode).items():
        # product_variable_specs() == the read VARIABLES plus derived outputs
        # (per-continuous-variable standard-deviation companions and reader-specific
        # extras such as IGBP's ranked scenes). It is the full set that appears in
        # the product definition, so this is what the YAMLs must match.
        for spec in cls.product_variable_specs():
            expected[f"{key}_{cls.INSTRUMENT}_{spec.name}"] = spec.dtype
    return expected


@pytest.fixture(scope="module")
def definitions() -> dict[OperationalMode, LiberaDataProductDefinition]:
    """All five FMATCH product definitions keyed by operational mode."""
    return {mode: load_fmatch_definition(mode) for mode in ALL_MODES}


class TestFmatchIdentifiers:
    """Every mode's product ID must be an ANC member matching the mode string."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_product_id_registered_as_anc(self, mode):
        product = DataProductIdentifier(mode.value)
        assert product.data_level is DataLevel.ANC

    def test_all_modes_have_a_definition_file(self):
        # The product module must map every operational mode to a YAML file.
        assert set(FMATCH_DEFINITION_FILENAMES) == set(ALL_MODES)


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
        assert definition.coordinates[time_var].dimensions == [time_var]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_common_variables_present(self, mode, definitions):
        definition = definitions[mode]
        for name in GEOLOCATION_VARIABLES + DERIVED_GEOMETRY_VARIABLES + COVERAGE_QA_VARIABLES:
            assert name in definition.variables, f"{mode.value} missing {name}"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_external_variables_match_readers(self, mode, definitions):
        definition = definitions[mode]
        for name, dtype in _expected_external_variables(mode).items():
            assert name in definition.variables, f"{mode.value} missing external variable {name}"
            assert definition.variables[name].dtype == dtype, (
                f"{mode.value} dtype drift for {name}: definition has "
                f"{definition.variables[name].dtype}, reader has {dtype}"
            )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_variables_use_mode_time_dimension(self, mode, definitions):
        time_var = fmatch_time_variable(mode)
        for name, var_def in definitions[mode].variables.items():
            assert var_def.dimensions == [time_var], f"{mode.value}/{name} wrong dimension"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_no_duplicate_variable_names(self, mode, definitions):
        # Sanity: collision prefixing must leave a unique variable set.
        definition = definitions[mode]
        all_names = list(definition.variables) + list(definition.coordinates)
        assert len(all_names) == len(set(all_names))


class TestDerivedProductVariables:
    """Guards the rules that turn read VARIABLES into product output variables."""

    def test_continuous_variable_gets_standard_deviation_companion(self):
        # ERA5 winds are mean-aggregated (continuous), so each must gain a
        # `<name>_standard_deviation` companion in the product variable list.
        era5 = ReaderRegistry.get("era5")
        names = {spec.name for spec in era5.product_variable_specs()}
        assert "wind_u10" in names
        assert "wind_u10_standard_deviation" in names
        assert "wind_v10_standard_deviation" in names

    def test_mode_aggregated_variable_has_no_standard_deviation_companion(self):
        # SSF's encoded scene-type codes have n_categories=None but are
        # weighted_mode, so a within-footprint standard deviation is meaningless
        # and must NOT be generated. This guards the mean-only rule.
        ssf = ReaderRegistry.get("ssf")
        names = {spec.name for spec in ssf.product_variable_specs()}
        for encoded in ("cloud_classification", "shortwave_adm_type", "longwave_adm_type"):
            assert encoded in names
            assert f"{encoded}_standard_deviation" not in names

    def test_igbp_reports_ranked_scenes_but_no_standard_deviation(self):
        # IGBP keeps the single aggregated surface_type plus three ranked-scene
        # outputs; being categorical (weighted_mode) it gets no std-dev companion.
        igbp = ReaderRegistry.get("igbp")
        names = {spec.name for spec in igbp.product_variable_specs()}
        assert {
            "surface_type",
            "surface_type_primary",
            "surface_type_secondary",
            "surface_type_tertiary",
        } <= names
        assert "surface_type_standard_deviation" not in names


class TestFmatchConformance:
    """A dummy dataset must round-trip through create/enforce/check for every mode."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_roundtrip(self, mode, definitions):
        definition = definitions[mode]
        time_var = fmatch_time_variable(mode)
        n_footprints = 4
        times = np.array(
            ["2026-06-11T00:00:00", "2026-06-11T00:00:01", "2026-06-11T00:00:02", "2026-06-11T00:00:03"],
            dtype="datetime64[ns]",
        )
        data: dict[str, np.ndarray] = {time_var: times}
        for name, var_def in definition.variables.items():
            data[name] = np.zeros(n_footprints, dtype=var_def.dtype)

        dynamic_attrs = {
            "algorithm_version": "1.0.0",
            "date_created": "2026-06-11T00:00:00Z",
            "input_files": "dummy_l1b.nc",
        }
        dataset = definition.create_product_dataset(data, dynamic_product_attributes=dynamic_attrs)
        dataset = definition.enforce_dataset_conformance(dataset)
        errors = definition.check_dataset_conformance(dataset, strict=True)
        assert errors == []
