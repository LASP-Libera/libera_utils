"""Unit tests for the FMATCH product definitions and their loaders.

There is one SSF-style product definition per FMATCH operational mode. The
radiometer-timescale FMATCH-IMAGER carries the RBSP CLDPIX/SSF cloud fields
alongside the full ERA5 single-level and pressure-level fields and the VIIRS
imager fields. These tests confirm, for every shipped definition, that:
- The product ID is registered as an auxiliary (AUX) DataProductIdentifier and
  matches the OperationalMode value string.
- The product definition YAML loads and validates via LiberaDataProductDefinition.
- The schema declares the expected geolocation, derived-geometry, and QA variables
  on the correct (radiometer vs camera) time dimension.
- The external (reader-sourced) variables stay in sync with the reader plugins'
  VariableSpec definitions for the mode. Every reader-sourced variable is named
  `<source_key>_<spec_name>` (e.g. era5_wind_u10, igbp_surface_type,
  cldpix_cloud_mask); the reader's instrument is recorded in each variable's
  ``long_name`` instead (e.g. "... (ECMWF)").
- A small dummy dataset round-trips through create/enforce/check conformance.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# Importing the readers subpackage registers all built-in readers so we can
# cross-check the product definitions against their VariableSpecs.
import libera_utils.footprint_matching.readers  # noqa: F401
from libera_utils.constants import DataLevel, DataProductIdentifier
from libera_utils.footprint_matching.geometry import NOMINAL_ALTITUDE_KM
from libera_utils.footprint_matching.product import (
    _CAMTIME_SEGMENTATION_VARIABLES,
    _RADIOMETER_L1B_VARIABLES,
    FMATCH_DEFINITION_FILENAMES,
    _assemble_camtime_dataset,
    _assemble_radiometer_dataset,
    _footprint_geometry,
    assemble_fmatch_dataset,
    build_radiometer_footprints,
    compute_derived_viewing_geometry,
    fmatch_time_variable,
    load_fmatch_definition,
    write_fmatch_product,
)
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.tiling import TileManager
from libera_utils.footprint_matching.types import (
    BoundingBox,
    FmatchCoverageFlag,
    OperationalMode,
    VariableSpec,
    spec_active_in_mode,
    with_standard_deviation_companions,
)
from libera_utils.io.product_definition import LiberaDataProductDefinition
from tests.test_data.footprint_matching.fixtures import make_era5_netcdf_fixture

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


def _production_readers_for_mode(mode: OperationalMode) -> dict:
    """Active production readers for a mode (excludes test-injected readers)."""
    return {key: cls for key, cls in ReaderRegistry.get_readers_for_mode(mode).items() if key in PRODUCTION_READER_KEYS}


def _expected_external_variables(mode: OperationalMode) -> dict[str, str]:
    """{output_variable_name: dtype} for every active production reader variable.

    Mirrors the product-definition naming rule: every reader-sourced variable is
    named `<source_key>_<spec_name>` (e.g. era5_wind_u10, igbp_surface_type). The
    reader's instrument is not part of the name -- it is recorded in the
    variable's ``long_name`` instead (see
    ``test_reader_instrument_is_recorded_in_long_name``). Specs are gated by
    ``spec_active_in_mode`` -- the ``required_mode`` rank rule (e.g. the ERA5
    single-level fields carry ``required_mode=IMAGER`` and so appear in the
    FMATCH-IMAGER-family products), or an exact ``only_modes`` pin (e.g. the extended
    SSF cloud/aerosol/albedo fields appear in FMATCH-IMAGER only).
    """
    expected: dict[str, str] = {}
    for key, cls in _production_readers_for_mode(mode).items():
        # product_variable_specs() == the read VARIABLES plus derived outputs
        # (per-continuous-variable standard-deviation companions and reader-specific
        # extras such as IGBP's ranked scenes). It is the full set that appears in
        # the product definition, so this is what the YAMLs must match.
        for spec in cls.product_variable_specs():
            if not spec_active_in_mode(spec, mode):
                continue
            expected[f"{key}_{spec.name}"] = spec.dtype
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
        assert any(name.startswith("cldpix_") for name in definition.variables)
        assert any(name.startswith("ssf_") for name in definition.variables)

    def test_imager_keeps_full_era5_field_set(self, definitions):
        definition = definitions[OperationalMode.IMAGER]
        # The winds (every product), the single-level fields, and the pressure-level
        # fields all sit in the FMATCH-IMAGER product alongside the RBSP fields
        # (spot-check one per family; full sync is covered by
        # test_external_variables_match_readers).
        assert "era5_wind_u10" in definition.variables
        assert "era5_temperature_2m" in definition.variables
        assert "era5_forecast_albedo" in definition.variables
        assert "era5_pressure_temperature_500hPa" in definition.variables
        assert "era5_pressure_relative_humidity_1000hPa_standard_deviation" in definition.variables

    def test_extended_ssf_fields_are_imager_only(self, definitions):
        # The extended SSF cloud-layer / aerosol / albedo fields are pinned to
        # FMATCH-IMAGER via only_modes and must NOT leak into the other products that
        # also activate the ssf reader (FLASH, IMAGER-CAMTIME). Spot-check one field
        # per secondary axis (layer, aerosol type, albedo band) plus a std companion.
        extended = (
            "ssf_layer_coverage_upper",
            "ssf_cloud_optical_depth_upper",
            "ssf_cloud_top_pressure_lower_standard_deviation",
            "ssf_match_aot",
            "ssf_aerosol_type_percentage_type6",
            "ssf_surface_albedo",
        )
        imager = definitions[OperationalMode.IMAGER]
        for name in extended:
            assert name in imager.variables, f"FMATCH-IMAGER missing extended field {name}"
        for mode in (OperationalMode.IMAGER_FLASH, OperationalMode.IMAGER_CAMTIME):
            leaked = [name for name in extended if name in definitions[mode].variables]
            assert leaked == [], f"{mode.value} must not declare IMAGER-only ssf fields, found {leaked}"
        # The base SSF fields, by contrast, still appear in FLASH and IMAGER-CAMTIME.
        assert "ssf_cloud_optical_depth_lower" in definitions[OperationalMode.IMAGER_FLASH].variables


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
        # products, CAMERA_TIME for the camera-timescale products' 2-D (CAMERA_TIME, FOOTPRINT) grid.
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
    def test_reader_instrument_is_recorded_in_long_name(self, mode, definitions):
        # The instrument token is carried in the variable's long_name rather than the
        # variable name (e.g. era5_wind_u10 has long_name "... (ECMWF)"). Guard that every
        # reader-sourced variable's long_name carries its reader's INSTRUMENT, so
        # provenance is not silently lost from the ~hundreds of hand-written long_names.
        definition = definitions[mode]
        for key, cls in _production_readers_for_mode(mode).items():
            for spec in cls.product_variable_specs():
                if not spec_active_in_mode(spec, mode):
                    continue
                name = f"{key}_{spec.name}"
                long_name = definition.variables[name].attributes.get("long_name", "")
                assert cls.INSTRUMENT in long_name, (
                    f"{mode.value}/{name}: long_name {long_name!r} is missing the reader instrument {cls.INSTRUMENT!r}"
                )

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_all_variables_use_mode_record_dimension(self, mode, definitions):
        # Radiometer modes hang variables on the 1-D RADIOMETER_TIME axis; camera-timescale modes hang every variable
        # on the 2-D (CAMERA_TIME, FOOTPRINT) grid.
        time_var = fmatch_time_variable(mode)
        expected_dims = ["CAMERA_TIME", "FOOTPRINT"] if time_var == "CAMERA_TIME" else [time_var]
        for name, var_def in definitions[mode].variables.items():
            assert var_def.dimensions == expected_dims, f"{mode.value}/{name} wrong dimension"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_no_duplicate_variable_names(self, mode, definitions):
        # Sanity: collision prefixing must leave a unique variable set.
        definition = definitions[mode]
        all_names = list(definition.variables) + list(definition.coordinates)
        assert len(all_names) == len(set(all_names))

    def test_imager_camtime_has_no_era5_pressure_variables(self, definitions):
        # The ERA5 pressure-level fields are a radiometer-timescale quantity: the
        # camera-timescale FMATCH-IMAGER-CAMTIME product must not declare them, while
        # the radiometer FMATCH-IMAGER product still does. Standing guard against the
        # pressure block drifting back into the camtime YAML (there is no general
        # no-extra-variables check).
        camtime = definitions[OperationalMode.IMAGER_CAMTIME]
        offenders = [name for name in camtime.variables if name.startswith("era5_pressure_")]
        assert offenders == [], f"FMATCH-IMAGER-CAMTIME must not declare era5_pressure variables, found {offenders}"
        imager = definitions[OperationalMode.IMAGER]
        assert any(name.startswith("era5_pressure_") for name in imager.variables)


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


class TestSpecActiveInMode:
    """The per-spec product gate: required_mode rank rule vs. exact only_modes pin."""

    def test_rank_rule_when_only_modes_unset(self):
        # required_mode=IMAGER (rank 3) -> carried by IMAGER and the higher-ranked
        # IMAGER-CAMTIME, but not the lower-ranked FLASH.
        spec = VariableSpec(
            name="x", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.IMAGER
        )
        assert spec_active_in_mode(spec, OperationalMode.IMAGER)
        assert spec_active_in_mode(spec, OperationalMode.IMAGER_CAMTIME)
        assert not spec_active_in_mode(spec, OperationalMode.IMAGER_FLASH)

    def test_only_modes_pins_to_exact_products(self):
        # only_modes bypasses the rank rule: IMAGER only, even though IMAGER-CAMTIME
        # outranks it.
        spec = VariableSpec(
            name="x",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            only_modes=(OperationalMode.IMAGER,),
        )
        assert spec_active_in_mode(spec, OperationalMode.IMAGER)
        assert not spec_active_in_mode(spec, OperationalMode.IMAGER_CAMTIME)
        assert not spec_active_in_mode(spec, OperationalMode.IMAGER_FLASH)

    def test_standard_deviation_companion_inherits_only_modes(self):
        # A continuous spec's std companion must stay scoped to the same product(s).
        parent = VariableSpec(
            name="cloud_top_pressure_lower",
            dtype="float32",
            aggregation="weighted_mean",
            required_mode=OperationalMode.IMAGER,
            only_modes=(OperationalMode.IMAGER,),
        )
        expanded = with_standard_deviation_companions((parent,))
        companion = next(s for s in expanded if s.name.endswith("_standard_deviation"))
        assert companion.only_modes == (OperationalMode.IMAGER,)


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
        # Sizes for every dimension a coordinate/variable may reference: the radiometer record axis (RADIOMETER_TIME)
        # and the camera-timescale grid axes (CAMERA_TIME, FOOTPRINT). CAMERA_TIME must match the length of the times
        # array below (it is the 1-D camera-timescale dimension coordinate).
        dimension_sizes = {"RADIOMETER_TIME": n_footprints, "CAMERA_TIME": n_footprints, "FOOTPRINT": n_footprints}

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
            "input_files": "dummy_l1b.nc",
        }
        dataset = definition.create_product_dataset(data, dynamic_product_attributes=dynamic_attrs)
        dataset = definition.enforce_dataset_conformance(dataset)
        errors = definition.check_dataset_conformance(dataset, strict=True)
        assert errors == []


# Timescale split of the modes, so the assembly tests can drive each mode with the
# right kind of input.
RADIOMETER_MODES = tuple(mode for mode in ALL_MODES if fmatch_time_variable(mode) == "RADIOMETER_TIME")
CAMTIME_MODES = tuple(mode for mode in ALL_MODES if fmatch_time_variable(mode) == "CAMERA_TIME")


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

    from libera_utils.footprint_matching._runner import load_l1b_camera_dataset
    from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
    from tests.test_data.footprint_matching.fixtures import make_l1b_camera_fixture

    with tempfile.TemporaryDirectory() as tmpdir:
        l1b_file = make_l1b_camera_fixture(_Path(tmpdir), n_images=1, n_pixels_x=3, n_pixels_y=3)
        return segment_l1b_camera(load_l1b_camera_dataset(l1b_file))


class TestRadiometerAssembly:
    """Radiometer-timescale modes assemble from L1B pass-through arrays."""

    @pytest.mark.parametrize("mode", RADIOMETER_MODES)
    def test_assembles_conformant_dataset(self, mode, definitions):
        definition = definitions[mode]
        dataset = assemble_fmatch_dataset(mode, _l1b_passthrough(), algorithm_version="1.0.0", input_files="l1b.nc")

        assert definition.check_dataset_conformance(dataset, strict=True) == []

    @pytest.mark.parametrize("mode", RADIOMETER_MODES)
    def test_l1b_columns_are_carried_through_verbatim(self, mode):
        """The L1B-derived columns are real values, not placeholders."""
        l1b_inputs = _l1b_passthrough()
        dataset = assemble_fmatch_dataset(mode, l1b_inputs)

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

    @pytest.mark.parametrize("mode", CAMTIME_MODES)
    def test_assembles_conformant_dataset(self, mode, definitions):
        definition = definitions[mode]
        dataset = assemble_fmatch_dataset(
            mode, _pseudo_footprints(), algorithm_version="1.0.0", input_files="l1b_cam.nc"
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

        The block's inclusive pixel extent is carried as the four camera_pixel_{x,y}_{min,max} COORDINATES on the 2-D
        (CAMERA_TIME, FOOTPRINT) grid; the boresight pixel is carried as the center_pixel_x/y variables. The four
        retired *_start/_stop variables must be gone. All are computed by segmentation for every camera-timescale
        mode, so both products carry them as real values rather than placeholders.
        """
        definition = definitions[mode]
        footprints = _pseudo_footprints()
        dataset = assemble_fmatch_dataset(mode, footprints)

        # camera_pixel_{x,y}_{min,max} are four inclusive-bound COORDINATES on the 2-D grid. The single-image fixture
        # yields one CAMERA_TIME row, so the footprints fill that row along FOOTPRINT (values raveled to compare).
        expected_bounds = {
            "camera_pixel_x_min": [f.slice_x.start for f in footprints],
            "camera_pixel_x_max": [f.slice_x.stop - 1 for f in footprints],
            "camera_pixel_y_min": [f.slice_y.start for f in footprints],
            "camera_pixel_y_max": [f.slice_y.stop - 1 for f in footprints],
        }
        for name, values in expected_bounds.items():
            assert name in definition.coordinates, name
            assert name in dataset.coords, name
            assert dataset[name].dims == ("CAMERA_TIME", "FOOTPRINT")
            np.testing.assert_array_equal(dataset[name].values.ravel(), values)

        # The boresight (center) pixel stays as FMATCH-only data variables (not carried downstream to SCENE-ID).
        for name in ("center_pixel_x", "center_pixel_y"):
            assert name in definition.variables, name
            assert name in dataset.variables, name
        np.testing.assert_array_equal(dataset["center_pixel_x"].values.ravel(), [f.center_ix for f in footprints])
        np.testing.assert_array_equal(dataset["center_pixel_y"].values.ravel(), [f.center_iy for f in footprints])

        # The old start/stop layout is retired from both products.
        retired = {"camera_pixel_x_start", "camera_pixel_x_stop", "camera_pixel_y_start", "camera_pixel_y_stop"}
        assert retired.isdisjoint(set(definition.variables) | set(definition.coordinates))
        assert retired.isdisjoint(set(dataset.variables))

    def test_ragged_images_pad_short_rows_on_the_grid(self):
        """Images with fewer subsections than the widest image pad along FOOTPRINT with fill values.

        Real segmentation yields a variable number of subsections per image, but the product is a rectangular
        (CAMERA_TIME, FOOTPRINT) grid. A shorter image's trailing FOOTPRINT cells must be padded: float variables
        take NaN and the integer pixel-bound coordinates take their 0 fill.
        """
        from libera_utils.footprint_matching.camera_segmentation import CameraFootprintQualityFlag, PseudoFootprint
        from libera_utils.footprint_matching.types import BoundingBox

        def _footprint(image_index: int, subsection_index: int) -> PseudoFootprint:
            start = subsection_index * 8
            return PseudoFootprint(
                time=np.datetime64("2026-06-11T00:00:00", "ns") + np.timedelta64(image_index, "s"),
                slice_x=slice(start, start + 4),
                slice_y=slice(0, 4),
                center_ix=start + 2,
                center_iy=2,
                latitude=float(subsection_index + 1),
                longitude=float(subsection_index + 1),
                altitude=700000.0,
                solar_zenith_angle=30.0,
                viewing_zenith_angle=8.0,
                relative_azimuth_angle=15.0,
                bbox=BoundingBox(0.0, 1.0, 0.0, 1.0),
                q_flags=CameraFootprintQualityFlag(0),
            )

        # Image 0 has two subsections; image 1 has one -> a 2 x 2 grid whose (image 1, subsection 1) cell is padded.
        footprints = [_footprint(0, 0), _footprint(0, 1), _footprint(1, 0)]
        dataset = assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, footprints)

        assert dataset.sizes["CAMERA_TIME"] == 2
        assert dataset.sizes["FOOTPRINT"] == 2
        # Real cells carry the segmentation latitude; only the single padded cell is NaN.
        latitude = dataset["latitude"].values
        assert not np.isnan(latitude[[0, 0, 1], [0, 1, 0]]).any()
        assert np.isnan(latitude[1, 1])
        # Integer pixel-bound coordinates cannot hold NaN, so the padded cell takes the 0 fill.
        assert dataset["camera_pixel_x_min"].values[1, 1] == 0

    def test_radiometer_mode_rejects_pseudo_footprints(self):
        with pytest.raises(ValueError, match="not a camera-timescale mode"):
            _assemble_camtime_dataset(_pseudo_footprints(), mode=OperationalMode.CAM)


class TestWriteFmatchProduct:
    """Every mode must write a strictly-conformant file under a proper Libera filename."""

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_writes_conformant_product(self, mode, tmp_path, definitions):
        inputs = _pseudo_footprints() if fmatch_time_variable(mode) == "CAMERA_TIME" else _l1b_passthrough()

        # strict=True: reaching the assertions below is itself the conformance guarantee.
        written = write_fmatch_product(
            mode,
            inputs,
            tmp_path,
            algorithm_version="1.0.0",
            input_files="l1b.nc",
            strict=True,
        )

        assert written.path.exists()
        assert written.data_product_id.value == definitions[mode].attributes["ProductID"]


class TestDerivedViewingGeometry:
    """compute_derived_viewing_geometry produces the real sun-glint angle."""

    def test_specular_geometry_is_zero_glint(self):
        # View zenith == solar zenith and RAA == 180 is the specular direction.
        result = compute_derived_viewing_geometry(np.array([30.0]), np.array([30.0]), np.array([180.0]))
        np.testing.assert_allclose(result["sunglint_angle"], [0.0], atol=1e-6)

    def test_backscatter_reduces_to_zenith_difference(self):
        # RAA == 0 -> cos(glint) = cos(SZA)cos(VZA) - sin(SZA)sin(VZA) = cos(SZA + VZA).
        result = compute_derived_viewing_geometry(np.array([40.0]), np.array([10.0]), np.array([0.0]))
        np.testing.assert_allclose(result["sunglint_angle"], [50.0], atol=1e-4)

    def test_result_is_bounded_and_nan_propagates(self):
        result = compute_derived_viewing_geometry(
            np.array([10.0, np.nan]), np.array([20.0, 20.0]), np.array([30.0, 30.0])
        )
        glint = result["sunglint_angle"]
        assert 0.0 <= glint[0] <= 180.0
        assert np.isnan(glint[1])


class TestBuildRadiometerFootprints:
    """Radiometer footprints get a bounding box + scan reference from the L1B geometry."""

    def test_each_footprint_gets_a_box_enclosing_its_boresight(self):
        """Without a scan reference the box degrades to the boresight-centred approximation."""
        l1b = {
            "latitude": np.array([1.0, -30.0], dtype=np.float32),
            "longitude": np.array([11.0, 100.0], dtype=np.float32),
            "viewing_zenith_angle": np.array([0.0, 40.0], dtype=np.float32),
        }
        footprints = build_radiometer_footprints(l1b)

        assert len(footprints) == 2
        for footprint, lat, lon in zip(footprints, l1b["latitude"], l1b["longitude"], strict=True):
            assert footprint.bbox.lat_min <= lat <= footprint.bbox.lat_max
            assert footprint.bbox.lon_min <= lon <= footprint.bbox.lon_max
            # No scan reference supplied -> the angular weigher would fall back to nadir.
            assert footprint.subsatellite_latitude is None
            assert footprint.cone_angle_rate is None

    def test_oblique_view_gives_a_larger_box(self):
        nadir = build_radiometer_footprints(
            {"latitude": np.array([0.0]), "longitude": np.array([0.0]), "viewing_zenith_angle": np.array([0.0])}
        )[0]
        oblique = build_radiometer_footprints(
            {"latitude": np.array([0.0]), "longitude": np.array([0.0]), "viewing_zenith_angle": np.array([60.0])}
        )[0]
        assert (oblique.bbox.lat_max - oblique.bbox.lat_min) > (nadir.bbox.lat_max - nadir.bbox.lat_min)

    def test_scan_reference_uses_the_true_ray_traced_box_and_is_carried_onto_the_footprint(self):
        """With the L1B subsatellite point present, the footprint carries the scan reference."""
        l1b = {
            "latitude": np.array([1.0], dtype=np.float32),
            "longitude": np.array([11.0], dtype=np.float32),
            "viewing_zenith_angle": np.array([40.0], dtype=np.float32),
            "subsatellite_latitude": np.array([1.5], dtype=np.float64),
            "subsatellite_longitude": np.array([11.0], dtype=np.float64),
            "cone_angle_rate": np.array([12.0], dtype=np.float64),
        }
        (footprint,) = build_radiometer_footprints(l1b)

        # The box still encloses the boresight, and the scan reference rode through.
        assert footprint.bbox.lat_min <= 1.0 <= footprint.bbox.lat_max
        assert footprint.subsatellite_latitude == 1.5
        assert footprint.subsatellite_longitude == 11.0
        assert footprint.cone_angle_rate == 12.0

    def test_oblique_ray_traced_box_is_asymmetric_unlike_the_radial_box(self):
        """A ray-traced oblique footprint is skewed along the scan plane, not centred on the boresight."""
        # Boresight at the equator, satellite offset to the north along the same meridian.
        scan_inputs = {
            "latitude": np.array([0.0], dtype=np.float32),
            "longitude": np.array([0.0], dtype=np.float32),
            "viewing_zenith_angle": np.array([55.0], dtype=np.float32),
            "subsatellite_latitude": np.array([10.0], dtype=np.float64),
            "subsatellite_longitude": np.array([0.0], dtype=np.float64),
            "cone_angle_rate": np.array([np.nan], dtype=np.float64),
        }
        (ray_traced,) = build_radiometer_footprints(scan_inputs)
        # The boresight-only fallback box (no scan reference) is symmetric about the boresight.
        (radial,) = build_radiometer_footprints(
            {k: scan_inputs[k] for k in ("latitude", "longitude", "viewing_zenith_angle")}
        )

        assert ray_traced.bbox.lat_min <= 0.0 <= ray_traced.bbox.lat_max
        # The radial box is centred on the boresight (lat 0); the ray-traced box is not.
        radial_offset = abs(radial.bbox.lat_max + radial.bbox.lat_min)
        ray_traced_offset = abs(ray_traced.bbox.lat_max + ray_traced.bbox.lat_min)
        assert radial_offset < 1e-6
        assert ray_traced_offset > 1e-2
        # A fill cone-angle rate is normalized to None.
        assert ray_traced.cone_angle_rate is None

    def test_footprint_count_is_preserved_even_for_a_limb_grazing_record(self):
        """A record that fails the ray-trace limb check falls back to a box, never dropped."""
        l1b = {
            "latitude": np.array([0.0, 10.0], dtype=np.float32),
            "longitude": np.array([0.0, 0.0], dtype=np.float32),
            # Second record is past the limb (VZA >= 90) -> OffLimbError -> boresight fallback.
            "viewing_zenith_angle": np.array([20.0, 90.5], dtype=np.float32),
            "subsatellite_latitude": np.array([1.0, 11.0], dtype=np.float64),
            "subsatellite_longitude": np.array([0.0, 0.0], dtype=np.float64),
            "cone_angle_rate": np.array([5.0, 5.0], dtype=np.float64),
        }
        footprints = build_radiometer_footprints(l1b)

        assert len(footprints) == 2
        for footprint in footprints:
            assert footprint.bbox.lat_min <= footprint.bbox.lat_max
        # The on-limb record is a normal observation; the past-the-limb record is marked
        # off_limb so aggregation skips it (fill + zero coverage) instead of tiling and
        # weighting its boresight placeholder box as a real observation.
        assert footprints[0].off_limb is False
        assert footprints[1].off_limb is True


class TestFootprintGeometryAltitude:
    """The PSF geometry reads a spacecraft altitude in km, never a surface height."""

    def test_radiometer_footprint_uses_its_spacecraft_altitude_km(self):
        (footprint,) = build_radiometer_footprints(
            {"latitude": np.array([0.0]), "longitude": np.array([0.0]), "viewing_zenith_angle": np.array([0.0])}
        )
        assert _footprint_geometry(footprint).altitude_km == pytest.approx(NOMINAL_ALTITUDE_KM)

    def test_camera_surface_height_is_not_read_as_spacecraft_altitude(self):
        # Regression: a camera PseudoFootprint carries ``altitude`` as the center-pixel
        # *surface height in metres* (an output column), not a spacecraft altitude in km.
        # The geometry extractor must ignore that field and fall back to the nominal
        # orbit altitude -- otherwise 835_000 m would be read as 835_000 km, inflating
        # the PSF ground radius ~1000x.
        camera_like = SimpleNamespace(
            bbox=BoundingBox(0.0, 1.0, 0.0, 1.0),
            latitude=0.5,
            longitude=0.5,
            altitude=835_000.0,  # metres of surface height
            viewing_zenith_angle=0.0,
        )
        geom = _footprint_geometry(camera_like)
        assert geom.altitude_km == pytest.approx(NOMINAL_ALTITUDE_KM)
        assert geom.altitude_km != pytest.approx(835_000.0)


def _l1b_at(lat: float, lon: float, *, sza: float = 30.0, vza: float = 5.0, raa: float = 90.0) -> dict:
    """A one-footprint L1B pass-through dict at a chosen boresight location."""
    return {
        "RADIOMETER_TIME": np.array([np.datetime64("2026-06-11T00:00:00", "ns")]),
        "latitude": np.array([lat], dtype=np.float32),
        "longitude": np.array([lon], dtype=np.float32),
        "solar_zenith_angle": np.array([sza], dtype=np.float32),
        "viewing_zenith_angle": np.array([vza], dtype=np.float32),
        "relative_azimuth_angle": np.array([raa], dtype=np.float32),
    }


class TestAssemblyWithAggregation:
    """Supplying a TileManager makes the external columns carry real aggregated values."""

    def _era5_manager(self, tmp_path) -> TileManager:
        """A CAM-mode TileManager holding only the real ERA5 reader over a constant-wind grid.

        The grid is fine (0.1 deg) so a cell falls inside the PSF ground radius (~20 km)
        of the test boresight; u10 == 2.5 and v10 == -1.5 everywhere, so the PSF-weighted
        mean is exactly those constants regardless of the weight kernel.
        """
        era5_cls = ReaderRegistry.get("era5")
        era5_file = make_era5_netcdf_fixture(
            tmp_path, lat_min=0.0, lat_max=2.0, lon_min=10.0, lon_max=12.0, n_lat=21, n_lon=21
        )
        return TileManager({"era5": era5_cls(era5_file)}, OperationalMode.CAM)

    def test_radiometer_assembly_carries_real_era5_values(self, tmp_path):
        # Footprint sits inside the ERA5 grid so every PSF cell has data.
        l1b = _l1b_at(lat=1.0, lon=11.0)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM,
            l1b,
            tile_manager=self._era5_manager(tmp_path),
            algorithm_version="1.0.0",
            input_files="l1b.nc,era5.nc",
        )

        # Weighted mean of a constant field is that constant.
        np.testing.assert_allclose(dataset["era5_wind_u10"].values, [2.5], rtol=1e-4)
        np.testing.assert_allclose(dataset["era5_wind_v10"].values, [-1.5], rtol=1e-4)
        # A constant field has zero within-footprint spread.
        np.testing.assert_allclose(dataset["era5_wind_u10_standard_deviation"].values, [0.0], atol=1e-4)
        # Fully covered -> coverage ~1 and no coverage QA bits set.
        assert dataset["psf_coverage_fraction"].values[0] > 0.99
        assert int(dataset["q_flags"].values[0]) == 0
        # A source NOT in the subset manager (igbp) stays a placeholder.
        assert int(dataset["igbp_surface_type"].values[0]) == 0
        # The product still conforms strictly with the computed columns in place.
        assert load_fmatch_definition(OperationalMode.CAM).check_dataset_conformance(dataset, strict=True) == []

    def test_footprint_off_the_grid_is_flagged_insufficient_coverage(self, tmp_path):
        # Boresight far from the ERA5 grid -> empty tile -> zero coverage.
        l1b = _l1b_at(lat=1.0, lon=100.0)
        dataset = assemble_fmatch_dataset(OperationalMode.CAM, l1b, tile_manager=self._era5_manager(tmp_path))

        assert np.isnan(dataset["era5_wind_u10"].values[0])
        np.testing.assert_allclose(dataset["psf_coverage_fraction"].values, [0.0])
        assert int(dataset["q_flags"].values[0]) & int(FmatchCoverageFlag.INSUFFICIENT_COVERAGE)
