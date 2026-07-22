"""Unit tests for the FMATCH product definitions and their loaders.

There is one SSF-style product definition per FMATCH operational mode, plus a
distinct post-year-one definition for FMATCH-IMAGER (the RBSP CLDPIX/SSF inputs
it uses do not exist during the first year of operation, so the production
``fmatch_imager.yml`` substitutes ERA5 fields; the RBSP-based definition is
kept as ``fmatch_imager_post_year_one.yml``). These tests confirm, for every
shipped definition, that:
- The product ID is registered as an auxiliary (AUX) DataProductIdentifier and
  matches the OperationalMode value string.
- The product definition YAML loads and validates via LiberaDataProductDefinition.
- The schema declares the expected geolocation, derived-geometry, and QA variables
  on the correct (radiometer vs camera) time dimension.
- The external (reader-sourced) variables stay in sync with the reader plugins'
  VariableSpec definitions for the definition's (mode, variant) pair. Every
  reader-sourced variable is named `<source_key>_<instrument>_<spec_name>`
  (e.g. era5_ECMWF_wind_u10, igbp_MODIS_surface_type, cldpix_NOAA20_cloud_mask).
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
    FMATCH_POST_YEAR_ONE_DEFINITION_FILENAMES,
    fmatch_time_variable,
    load_fmatch_definition,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import FmatchVariant, OperationalMode
from libera_utils.io.product_definition import LiberaDataProductDefinition

# The production reader keys. Intersecting with these makes the cross-check robust
# against throwaway readers other tests register into the shared ReaderRegistry
# (e.g. _FakeReader in test_base.py).
PRODUCTION_READER_KEYS = frozenset(
    {"era5", "era5_pressure", "igbp", "nise", "viirs_brdf", "viirs_cloud", "ssf", "cldpix", "viirs_aod"}
)

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

# Every shipped product definition, as the (mode, variant) pair whose active
# reader/spec set the YAML content must match:
# - The CAM-family products are variant-insensitive (no variant-gated readers or
#   specs at their latency rank), so YEAR_ONE covers them.
# - FMATCH-IMAGER ships BOTH variants: the ERA5-based year-one production YAML
#   and the RBSP-based post-year-one YAML.
# - FLASH and IMAGER-CAMTIME inherently require RBSP inputs (they only run
#   post-year-one), so their single YAML is checked against POST_YEAR_ONE.
DEFINITION_CASES: tuple[tuple[OperationalMode, FmatchVariant], ...] = (
    (OperationalMode.CAM, FmatchVariant.YEAR_ONE),
    (OperationalMode.CAM_CAMTIME, FmatchVariant.YEAR_ONE),
    (OperationalMode.IMAGER_FLASH, FmatchVariant.POST_YEAR_ONE),
    (OperationalMode.IMAGER, FmatchVariant.YEAR_ONE),
    (OperationalMode.IMAGER, FmatchVariant.POST_YEAR_ONE),
    (OperationalMode.IMAGER_CAMTIME, FmatchVariant.POST_YEAR_ONE),
)


def _production_readers_for_mode(mode: OperationalMode, variant: FmatchVariant) -> dict:
    """Active production readers for a (mode, variant) pair (excludes test-injected readers)."""
    return {
        key: cls
        for key, cls in ReaderRegistry.get_readers_for_mode(mode, variant).items()
        if key in PRODUCTION_READER_KEYS
    }


def _expected_external_variables(mode: OperationalMode, variant: FmatchVariant) -> dict[str, str]:
    """{output_variable_name: dtype} for every active production reader variable.

    Mirrors the product-definition naming rule: every reader-sourced variable is
    named `<source_key>_<instrument>_<spec_name>`, where the instrument token comes
    from the reader's INSTRUMENT attribute (e.g. era5_ECMWF_wind_u10,
    igbp_MODIS_surface_type). Specs are filtered by BOTH gates a spec can carry:
    its ``required_mode`` rank and its ``required_variant`` (e.g. the ERA5
    year-one substitute fields are declared on the always-active era5 reader but
    only appear in the year-one FMATCH-IMAGER product).
    """
    expected: dict[str, str] = {}
    for key, cls in _production_readers_for_mode(mode, variant).items():
        # product_variable_specs() == the read VARIABLES plus derived outputs
        # (per-continuous-variable standard-deviation companions and reader-specific
        # extras such as IGBP's ranked scenes). It is the full set that appears in
        # the product definition, so this is what the YAMLs must match.
        for spec in cls.product_variable_specs():
            if spec.required_mode.rank > mode.rank:
                continue
            if spec.required_variant is not None and spec.required_variant is not variant:
                continue
            expected[f"{key}_{cls.INSTRUMENT}_{spec.name}"] = spec.dtype
    return expected


@pytest.fixture(scope="module")
def definitions() -> dict[tuple[OperationalMode, FmatchVariant], LiberaDataProductDefinition]:
    """All shipped FMATCH product definitions keyed by (mode, variant)."""
    return {(mode, variant): load_fmatch_definition(mode, variant) for mode, variant in DEFINITION_CASES}


class TestFmatchIdentifiers:
    """Every mode's product ID must be an AUX member matching the mode string."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_product_id_registered_as_aux(self, mode):
        product = DataProductIdentifier(mode.value)
        assert product.data_level is DataLevel.AUX

    def test_all_modes_have_a_definition_file(self):
        # The product module must map every operational mode to a YAML file.
        assert set(FMATCH_DEFINITION_FILENAMES) == set(ALL_MODES)

    def test_only_imager_has_a_post_year_one_definition(self):
        # The year-one vs post-year-one split only applies to FMATCH-IMAGER.
        assert set(FMATCH_POST_YEAR_ONE_DEFINITION_FILENAMES) == {OperationalMode.IMAGER}


class TestVariantResolution:
    """load_fmatch_definition must resolve the variant to the right YAML."""

    def test_imager_variants_load_distinct_definitions(self):
        year_one = load_fmatch_definition(OperationalMode.IMAGER)
        post = load_fmatch_definition(OperationalMode.IMAGER, FmatchVariant.POST_YEAR_ONE)
        # Both encode the SAME ProductID (no separate product identity is wanted
        # for the post-year-one variant) but declare different variable sets.
        assert year_one.attributes["ProductID"] == post.attributes["ProductID"] == "FMATCH-IMAGER"
        assert set(year_one.variables) != set(post.variables)

    def test_post_year_one_falls_back_for_modes_without_distinct_yaml(self):
        # CAM has no post-year-one YAML; the variant must resolve to its single
        # definition rather than raising (the CAM product is variant-insensitive).
        default = load_fmatch_definition(OperationalMode.CAM)
        post = load_fmatch_definition(OperationalMode.CAM, FmatchVariant.POST_YEAR_ONE)
        assert set(default.variables) == set(post.variables)


class TestYearOneImagerContent:
    """Guards the year-one substitution itself: what is in and out of each IMAGER YAML."""

    def test_year_one_imager_has_no_rbsp_variables(self, definitions):
        definition = definitions[(OperationalMode.IMAGER, FmatchVariant.YEAR_ONE)]
        rbsp = [name for name in definition.variables if name.startswith(("cldpix_", "ssf_"))]
        assert rbsp == [], f"year-one fmatch_imager.yml must not declare RBSP variables, found {rbsp}"

    def test_year_one_imager_has_era5_substitutes(self, definitions):
        definition = definitions[(OperationalMode.IMAGER, FmatchVariant.YEAR_ONE)]
        # Spot-check one variable per substitute family (full sync is covered by
        # test_external_variables_match_readers).
        assert "era5_ECMWF_temperature_2m" in definition.variables
        assert "era5_ECMWF_forecast_albedo" in definition.variables
        assert "era5_pressure_ECMWF_temperature_500hPa" in definition.variables
        assert "era5_pressure_ECMWF_relative_humidity_1000hPa_standard_deviation" in definition.variables

    def test_post_year_one_imager_keeps_rbsp_and_drops_era5_substitutes(self, definitions):
        definition = definitions[(OperationalMode.IMAGER, FmatchVariant.POST_YEAR_ONE)]
        assert any(name.startswith("cldpix_") for name in definition.variables)
        assert any(name.startswith("ssf_") for name in definition.variables)
        # The winds remain (they feed every product); the year-one substitutes do not.
        assert "era5_ECMWF_wind_u10" in definition.variables
        year_one_only = [
            name
            for name in definition.variables
            if name.startswith("era5_pressure_") or name.startswith("era5_ECMWF_temperature_2m")
        ]
        assert year_one_only == []


class TestFmatchDefinitions:
    """Each YAML loads and declares the expected structure."""

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_definition_loads_with_matching_product_id(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
        assert isinstance(definition, LiberaDataProductDefinition)
        assert definition.attributes["ProductID"] == mode.value

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_time_coordinate_matches_timescale(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
        time_var = fmatch_time_variable(mode)
        assert time_var in definition.coordinates
        assert definition.coordinates[time_var].dtype == "datetime64[ns]"
        assert definition.coordinates[time_var].dimensions == [time_var]

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_common_variables_present(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
        for name in GEOLOCATION_VARIABLES + DERIVED_GEOMETRY_VARIABLES + COVERAGE_QA_VARIABLES:
            assert name in definition.variables, f"{mode.value} ({variant.value}) missing {name}"

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_external_variables_match_readers(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
        for name, dtype in _expected_external_variables(mode, variant).items():
            assert name in definition.variables, f"{mode.value} ({variant.value}) missing external variable {name}"
            assert definition.variables[name].dtype == dtype, (
                f"{mode.value} ({variant.value}) dtype drift for {name}: definition has "
                f"{definition.variables[name].dtype}, reader has {dtype}"
            )

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_all_variables_use_mode_time_dimension(self, mode, variant, definitions):
        time_var = fmatch_time_variable(mode)
        for name, var_def in definitions[(mode, variant)].variables.items():
            assert var_def.dimensions == [time_var], f"{mode.value} ({variant.value})/{name} wrong dimension"

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_no_duplicate_variable_names(self, mode, variant, definitions):
        # Sanity: collision prefixing must leave a unique variable set.
        definition = definitions[(mode, variant)]
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

    def test_standard_deviation_companion_inherits_variant_gate(self):
        # The year-one substitute fields are variant-gated; their std-dev
        # companions must carry the same gate or they would leak into the
        # post-year-one product definition.
        era5 = ReaderRegistry.get("era5")
        by_name = {spec.name: spec for spec in era5.product_variable_specs()}
        assert by_name["temperature_2m_standard_deviation"].required_variant is FmatchVariant.YEAR_ONE
        assert by_name["wind_u10_standard_deviation"].required_variant is None

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
    """A dummy dataset must round-trip through create/enforce/check for every definition."""

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_roundtrip(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
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
