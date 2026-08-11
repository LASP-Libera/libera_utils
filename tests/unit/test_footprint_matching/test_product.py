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
    _CAMTIME_SEGMENTATION_VARIABLES,
    _RADIOMETER_L1B_VARIABLES,
    FMATCH_DEFINITION_FILENAMES,
    FMATCH_POST_YEAR_ONE_DEFINITION_FILENAMES,
    _assemble_camtime_dataset,
    _assemble_radiometer_dataset,
    assemble_fmatch_dataset,
    fmatch_time_variable,
    load_fmatch_definition,
    write_fmatch_product,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import (
    FmatchVariant,
    OperationalMode,
    VariableSpec,
    with_standard_deviation_companions,
)
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

    def test_post_year_one_imager_keeps_rbsp_and_era5(self, definitions):
        definition = definitions[(OperationalMode.IMAGER, FmatchVariant.POST_YEAR_ONE)]
        # Post-year-one carries the RBSP CLDPIX/SSF fields ...
        assert any(name.startswith("cldpix_") for name in definition.variables)
        assert any(name.startswith("ssf_") for name in definition.variables)
        # ... AND the full ERA5 field set alongside them: the winds (every
        # product), the former year-one single-level substitutes, and the
        # pressure-level fields (spot-check one per family; full sync is covered by
        # test_external_variables_match_readers).
        assert "era5_ECMWF_wind_u10" in definition.variables
        assert "era5_ECMWF_temperature_2m" in definition.variables
        assert "era5_ECMWF_forecast_albedo" in definition.variables
        assert "era5_pressure_ECMWF_temperature_500hPa" in definition.variables
        assert "era5_pressure_ECMWF_relative_humidity_1000hPa_standard_deviation" in definition.variables


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
        # Radiometer time is a dimension coordinate (name == dimension); camera time is a non-unique auxiliary
        # coordinate on the FOOTPRINT record axis (name != dimension).
        record_dim = "FOOTPRINT" if time_var == "CAMERA_TIME" else time_var
        assert definition.coordinates[time_var].dimensions == [record_dim]

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
    def test_all_variables_use_mode_record_dimension(self, mode, variant, definitions):
        # Radiometer modes hang variables on RADIOMETER_TIME (name == dimension); camera-timescale modes hang them on
        # the FOOTPRINT record axis, carrying CAMERA_TIME only as a coordinate (name != dimension).
        time_var = fmatch_time_variable(mode)
        record_dim = "FOOTPRINT" if time_var == "CAMERA_TIME" else time_var
        for name, var_def in definitions[(mode, variant)].variables.items():
            assert var_def.dimensions == [record_dim], f"{mode.value} ({variant.value})/{name} wrong dimension"

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
        # A std-dev companion must carry the SAME variant gate as its parent, so a
        # gated field and its companion always land in exactly the same product
        # definitions. The production readers now gate variants at the reader level
        # rather than per spec, so exercise the companion mechanism directly with a
        # synthetic gated parent; then confirm a real variant-neutral parent (an
        # ERA5 wind) yields a variant-neutral companion.
        gated_parent = VariableSpec(
            name="demo_field",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
            required_variant=FmatchVariant.YEAR_ONE,
        )
        companions = {spec.name: spec for spec in with_standard_deviation_companions((gated_parent,))}
        assert companions["demo_field_standard_deviation"].required_variant is FmatchVariant.YEAR_ONE

        era5 = {spec.name: spec for spec in ReaderRegistry.get("era5").product_variable_specs()}
        assert era5["wind_u10_standard_deviation"].required_variant is None

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
        # Sizes for every dimension a coordinate/variable may reference: the record axis (RADIOMETER_TIME or FOOTPRINT)
        # is dynamic; CAMERA_PIXEL_BOUNDS is the fixed size-2 (min, max) pair carried by the camtime range coordinates.
        dimension_sizes = {"RADIOMETER_TIME": n_footprints, "FOOTPRINT": n_footprints, "CAMERA_PIXEL_BOUNDS": 2}

        def _zeros(var_def):
            return np.zeros(tuple(dimension_sizes[d] for d in var_def.dimensions), dtype=var_def.dtype)

        data: dict[str, np.ndarray] = {time_var: times}
        # Every non-time coordinate (e.g. the 2-D camera_pixel_x/y range coords on the camtime products) and every
        # variable must be present for the conformance check to pass.
        for name, var_def in definition.coordinates.items():
            if name != time_var:
                data[name] = _zeros(var_def)
        for name, var_def in definition.variables.items():
            data[name] = _zeros(var_def)

        dynamic_attrs = {
            "algorithm_version": "1.0.0",
            "input_files": "dummy_l1b.nc",
        }
        dataset = definition.create_product_dataset(data, dynamic_product_attributes=dynamic_attrs)
        dataset = definition.enforce_dataset_conformance(dataset)
        errors = definition.check_dataset_conformance(dataset, strict=True)
        assert errors == []


# Timescale split of DEFINITION_CASES, so the assembly tests can drive each mode with the
# right kind of input without repeating the (mode, variant) table.
RADIOMETER_CASES = tuple(
    (mode, variant) for mode, variant in DEFINITION_CASES if fmatch_time_variable(mode) == "RADIOMETER_TIME"
)
CAMTIME_CASES = tuple(
    (mode, variant) for mode, variant in DEFINITION_CASES if fmatch_time_variable(mode) == "CAMERA_TIME"
)


def _l1b_passthrough(n_footprints: int = 6) -> dict[str, np.ndarray]:
    """Build a minimal, valid L1B pass-through dict of the shape the radiometer assembler expects."""
    base_time = np.datetime64("2026-06-11T00:00:00", "ns")
    data: dict[str, np.ndarray] = {
        "RADIOMETER_TIME": base_time + np.arange(n_footprints, dtype="int64") * np.timedelta64(10, "ms"),
    }
    for name in _RADIOMETER_L1B_VARIABLES:
        data[name] = np.linspace(1.0, 40.0, n_footprints, dtype=np.float32)
    return data


def _pseudo_footprints(n_footprints: int = 6) -> list:
    """Segment a small synthetic L1B camera grid into pseudo-footprints."""
    import tempfile
    from pathlib import Path as _Path

    from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
    from libera_utils.footprint_matching.l1b_inputs import load_l1b_camera_dataset
    from tests.test_data.footprint_matching.fixtures import make_l1b_camera_fixture

    with tempfile.TemporaryDirectory() as tmpdir:
        l1b_file = make_l1b_camera_fixture(_Path(tmpdir), n_images=1, n_pixels_x=3, n_pixels_y=3)
        return segment_l1b_camera(load_l1b_camera_dataset(l1b_file))


class TestRadiometerAssembly:
    """Radiometer-timescale modes assemble from L1B pass-through arrays."""

    @pytest.mark.parametrize(("mode", "variant"), RADIOMETER_CASES)
    def test_assembles_conformant_dataset(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
        dataset = assemble_fmatch_dataset(
            mode, _l1b_passthrough(), variant=variant, algorithm_version="1.0.0", input_files="l1b.nc"
        )

        assert definition.check_dataset_conformance(dataset, strict=True) == []

    @pytest.mark.parametrize(("mode", "variant"), RADIOMETER_CASES)
    def test_l1b_columns_are_carried_through_verbatim(self, mode, variant):
        """The L1B-derived columns are real values, not placeholders."""
        l1b_inputs = _l1b_passthrough()
        dataset = assemble_fmatch_dataset(mode, l1b_inputs, variant=variant)

        for name in _RADIOMETER_L1B_VARIABLES:
            np.testing.assert_allclose(dataset[name].values, l1b_inputs[name], rtol=1e-6)
        np.testing.assert_array_equal(dataset["RADIOMETER_TIME"].values, l1b_inputs["RADIOMETER_TIME"])

    def test_cloud_fraction_camera_is_merged_when_declared(self):
        """FMATCH-CAM declares cloud_fraction_camera, so supplied values become real, not placeholder."""
        n = 6
        values = np.linspace(0.0, 1.0, n, dtype=np.float32)
        dataset = assemble_fmatch_dataset(OperationalMode.CAM, _l1b_passthrough(n), cloud_fraction_camera=values)

        np.testing.assert_allclose(dataset["cloud_fraction_camera"].values, values, rtol=1e-6)

    def test_cloud_fraction_camera_is_ignored_when_undeclared(self):
        """The IMAGER modes have no such variable; passing values must not raise or invent one."""
        n = 6
        dataset = assemble_fmatch_dataset(
            OperationalMode.IMAGER, _l1b_passthrough(n), cloud_fraction_camera=np.zeros(n, dtype=np.float32)
        )

        assert "cloud_fraction_camera" not in dataset.variables

    def test_missing_passthrough_key_names_what_is_missing(self):
        l1b_inputs = _l1b_passthrough()
        del l1b_inputs["latitude"]

        with pytest.raises(ValueError, match="missing required key"):
            assemble_fmatch_dataset(OperationalMode.CAM, l1b_inputs)

    def test_inconsistent_lengths_raise(self):
        l1b_inputs = _l1b_passthrough(6)
        l1b_inputs["longitude"] = np.zeros(5, dtype=np.float32)

        with pytest.raises(ValueError, match="inconsistent lengths"):
            assemble_fmatch_dataset(OperationalMode.CAM, l1b_inputs)

    def test_zero_footprints_raise(self):
        with pytest.raises(ValueError, match="zero footprints"):
            assemble_fmatch_dataset(OperationalMode.CAM, _l1b_passthrough(0))

    def test_camera_timescale_mode_rejects_passthrough_inputs(self):
        """Guards against handing the wrong kind of input to the wrong timescale."""
        with pytest.raises(ValueError, match="camera-timescale mode"):
            _assemble_radiometer_dataset(_l1b_passthrough(), mode=OperationalMode.CAM_CAMTIME)


class TestCamtimeAssembly:
    """Camera-timescale modes assemble from camera pseudo-footprints."""

    @pytest.mark.parametrize(("mode", "variant"), CAMTIME_CASES)
    def test_assembles_conformant_dataset(self, mode, variant, definitions):
        definition = definitions[(mode, variant)]
        dataset = assemble_fmatch_dataset(
            mode, _pseudo_footprints(), variant=variant, algorithm_version="1.0.0", input_files="l1b_cam.nc"
        )

        assert definition.check_dataset_conformance(dataset, strict=True) == []

    def test_imager_camtime_uses_the_same_segmentation_path(self):
        """CAM-CAMTIME and IMAGER-CAMTIME share the segmentation path; only the reader set differs."""
        footprints = _pseudo_footprints()
        cam = assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, footprints)
        imager = assemble_fmatch_dataset(OperationalMode.IMAGER_CAMTIME, footprints)

        assert cam.sizes["FOOTPRINT"] == imager.sizes["FOOTPRINT"] == len(footprints)
        # Every segmentation-derived column is declared by BOTH camtime products, so all come out
        # identical: same footprints, same code path.
        for name in _CAMTIME_SEGMENTATION_VARIABLES:
            np.testing.assert_array_equal(cam[name].values, imager[name].values)
        # Only the CAM product carries the Libera WFOV cloud fraction.
        assert "cloud_fraction_camera" in cam.variables
        assert "cloud_fraction_camera" not in imager.variables

    @pytest.mark.parametrize("mode", [OperationalMode.CAM_CAMTIME, OperationalMode.IMAGER_CAMTIME])
    def test_both_camtime_products_carry_pixel_block_provenance(self, mode, definitions):
        """Both camera-timescale products carry the pixel-block provenance, filled with real segmentation values.

        The block's inclusive (min, max) pixel extent is carried as the 2-D camera_pixel_x/y range COORDINATES
        (FOOTPRINT x CAMERA_PIXEL_BOUNDS); the boresight pixel is carried as the center_pixel_x/y variables. The four
        retired *_start/_stop variables must be gone. All are computed by segmentation for every camera-timescale
        mode, so both products carry them as real values rather than placeholders.
        """
        variant = FmatchVariant.YEAR_ONE if mode is OperationalMode.CAM_CAMTIME else FmatchVariant.POST_YEAR_ONE
        definition = definitions[(mode, variant)]
        footprints = _pseudo_footprints()
        dataset = assemble_fmatch_dataset(mode, footprints, variant=variant)

        # camera_pixel_x/y are 2-D range COORDINATES holding inclusive (min, max) = (slice.start, slice.stop - 1).
        for name in ("camera_pixel_x", "camera_pixel_y"):
            assert name in definition.coordinates, name
            assert name in dataset.coords, name
            assert dataset[name].dims == ("FOOTPRINT", "CAMERA_PIXEL_BOUNDS")
        np.testing.assert_array_equal(
            dataset["camera_pixel_x"].values, [(f.slice_x.start, f.slice_x.stop - 1) for f in footprints]
        )
        np.testing.assert_array_equal(
            dataset["camera_pixel_y"].values, [(f.slice_y.start, f.slice_y.stop - 1) for f in footprints]
        )

        # The boresight (center) pixel stays as FMATCH-only data variables (not carried downstream to SCENE-ID).
        for name in ("center_pixel_x", "center_pixel_y"):
            assert name in definition.variables, name
            assert name in dataset.variables, name
        np.testing.assert_array_equal(dataset["center_pixel_x"].values, [f.center_ix for f in footprints])
        np.testing.assert_array_equal(dataset["center_pixel_y"].values, [f.center_iy for f in footprints])

        # The old start/stop layout is retired from both products.
        retired = {"camera_pixel_x_start", "camera_pixel_x_stop", "camera_pixel_y_start", "camera_pixel_y_stop"}
        assert retired.isdisjoint(set(definition.variables) | set(definition.coordinates))
        assert retired.isdisjoint(set(dataset.variables))

    def test_radiometer_mode_rejects_pseudo_footprints(self):
        with pytest.raises(ValueError, match="not a camera-timescale mode"):
            _assemble_camtime_dataset(_pseudo_footprints(), mode=OperationalMode.CAM)


class TestWriteFmatchProduct:
    """Every mode must write a strictly-conformant file under a proper Libera filename."""

    @pytest.mark.parametrize(("mode", "variant"), DEFINITION_CASES)
    def test_writes_conformant_product(self, mode, variant, tmp_path, definitions):
        inputs = _pseudo_footprints() if fmatch_time_variable(mode) == "CAMERA_TIME" else _l1b_passthrough()

        # strict=True: reaching the assertions below is itself the conformance guarantee.
        written = write_fmatch_product(
            mode,
            inputs,
            tmp_path,
            variant=variant,
            algorithm_version="1.0.0",
            input_files="l1b.nc",
            strict=True,
        )

        assert written.path.exists()
        assert written.data_product_id.value == definitions[(mode, variant)].attributes["ProductID"]
