"""Tests for the internal-to-product variable name rule and the product definitions that follow it."""

from pathlib import Path

import pytest
import yaml

import libera_utils
from libera_utils.io import product_variable_names
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.io.product_variable_names import internal_variable_name, product_variable_name

PRODUCT_DEFINITIONS = Path(libera_utils.__file__).parent / "data" / "product_definitions"

#: The product definitions whose variable names follow the rule.
CAMEL_CASE_DEFINITIONS = (
    "fmatch_cam.yml",
    "fmatch_cam_camtime.yml",
    "fmatch_imager.yml",
    "fmatch_imager_camtime.yml",
    "scene_id_cam.yml",
    "scene_id_cam_camtime.yml",
    "scene_id_imager.yml",
    "scene_id_imager_camtime.yml",
)


class TestProductVariableName:
    """product_variable_name capitalizes each token, spells acronyms from the table and applies L1B renames."""

    @pytest.mark.parametrize(
        ("internal", "product"),
        [
            ("sunglint_angle", "Sunglint_Angle"),
            ("era5_surface_wind_u10", "ERA5_Surface_Wind_U10"),
            ("era5_surface_temperature_2m", "ERA5_Surface_Temperature_2m"),
            ("viirs_brdf_vis_fiso", "VIIRS_BRDF_VIS_Fiso"),
            ("viirs_aod_optical_depth_550", "VIIRS_AOD_Optical_Depth_550"),
            ("flashflux_cloud_particle_phase_37um_lower", "FLASHFlux_Cloud_Particle_Phase_37um_Lower"),
            ("flashflux_aerosol_type_percentage_type0", "FLASHFlux_Aerosol_Type_Percentage_Type0"),
            ("nise_snow_on_land_standard_deviation", "NISE_Snow_On_Land_Standard_Deviation"),
            ("psf_bbox_lat_min", "PSF_Bbox_Lat_Min"),
            ("scene_id_erbe", "Scene_ID_ERBE"),
            ("scene_id_unfiltering", "Scene_ID_Unfiltering"),
            ("scene_id_trmm", "Scene_ID_TRMM"),
            ("scene_bin_trmm_surface_wind_min", "Scene_Bin_TRMM_Surface_Wind_Min"),
            ("solar_zenith_angle", "Solar_Zenith_Surface"),
            ("viewing_zenith_angle", "Viewing_Zenith_Surface"),
            ("relative_azimuth_angle", "Relative_Azimuth_Surface"),
            ("q_flags", "Quality_Flag"),
            ("latitude", "Latitude"),
        ],
    )
    def test_maps_internal_to_product_name(self, internal, product):
        """Each internal name maps to the product name the L1B convention gives it, and back."""
        assert product_variable_name(internal) == product
        assert internal_variable_name(product) == internal

    @pytest.mark.parametrize("name", ["", "Latitude", "RADIOMETER_TIME", "2m_temperature", "wind__u", "wind_", "_wind"])
    def test_rejects_a_name_that_is_not_snake_case(self, name):
        """An internal name must be lower-case tokens joined by single underscores, starting with a letter."""
        with pytest.raises(ValueError, match="is not a snake_case internal name"):
            product_variable_name(name)

    def test_rejects_a_name_that_collides_with_a_rename(self):
        """``solar_zenith_surface`` would take ``solar_zenith_angle``'s product name, so it is refused."""
        with pytest.raises(ValueError, match="already the product name of 'solar_zenith_angle'"):
            product_variable_name("solar_zenith_surface")


class TestInternalVariableName:
    """internal_variable_name accepts only names the rule produces."""

    @pytest.mark.parametrize(
        "product_name",
        ["Viirs_BRDF_VIS_Fiso", "viirs_brdf_vis_fiso", "Solar_Zenith_Angle", "RADIOMETER_TIME", "Flashflux_Match_AOT"],
    )
    def test_rejects_a_name_off_the_rule(self, product_name):
        """A misspelled token, a snake_case name or an L1B spelling with no rename row is not a product name."""
        with pytest.raises(ValueError, match="does not follow the product variable naming rule; the name for"):
            internal_variable_name(product_name)

    @pytest.mark.parametrize("product_name", ["", "Wind__U", "_Wind", "Wind-Speed"])
    def test_rejects_a_name_with_no_internal_form(self, product_name):
        """A name whose lower-cased spelling is not a snake_case internal name has no internal name at all."""
        with pytest.raises(
            ValueError, match="does not follow the product variable naming rule: .* is not a snake_case"
        ):
            internal_variable_name(product_name)


class TestNameTableLoader:
    """_load_name_table rejects a malformed table when the module is imported."""

    HEADER = "kind,internal,product,source"

    @staticmethod
    def _load(tmp_path, monkeypatch, lines: list[str]):
        """Run _load_name_table against a synthetic table instead of the shipped one."""
        table = tmp_path / "product_variable_names.csv"
        table.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(product_variable_names, "PRODUCT_VARIABLE_NAMES_CSV", table)
        return product_variable_names._load_name_table()

    def test_loads_tokens_and_renames(self, tmp_path, monkeypatch):
        """A well-formed table loads into token spellings and whole-name renames."""
        tokens, renames = self._load(
            tmp_path,
            monkeypatch,
            [self.HEADER, "token,viirs,VIIRS,VIIRS", "variable,q_flags,Quality_Flag,L1B definitions"],
        )
        assert tokens == {"viirs": "VIIRS"}
        assert renames == {"q_flags": "Quality_Flag"}

    @pytest.mark.parametrize(
        ("lines", "message"),
        [
            (["kind,internal,product", "token,viirs,VIIRS"], "must have the columns"),
            (["kind,internal,product,source", "acronym,viirs,VIIRS,VIIRS"], "must be 'token' or 'variable'"),
            (["kind,internal,product,source", "token,viirs,VIIRS,"], "in 'source'"),
            (["kind,internal,product,source", "token,viirs,VIIRS,VIIRS,extra"], "expected 4 columns"),
            (["kind,internal,product,source", "token,viirs,VIIRx,VIIRS"], "letter case only"),
            (["kind,internal,product,source", "variable,q_flags,quality_flag,L1B"], "Camel_Case name"),
            (["kind,internal,product,source", "token,viirs,VIIRS,VIIRS", "token,viirs,Viirs,VIIRS"], "appears twice"),
            (
                ["kind,internal,product,source", "variable,q_flags,Quality_Flag,L1B", "variable,qf,Quality_Flag,L1B"],
                "already the product name",
            ),
        ],
    )
    def test_rejects_a_malformed_table(self, tmp_path, monkeypatch, lines, message):
        """Each malformed table raises ValueError naming what to fix."""
        with pytest.raises(ValueError, match=message):
            self._load(tmp_path, monkeypatch, lines)


def _section_keys(path: Path) -> dict[str, list[str]]:
    """Return every key under ``variables`` and ``coordinates`` as written, duplicates included."""
    document = yaml.compose(path.read_text())
    keys: dict[str, list[str]] = {}
    for section_key, section in document.value:
        if section_key.value in ("variables", "coordinates"):
            keys[section_key.value] = [key.value for key, _ in section.value]
    return keys


@pytest.mark.parametrize("definition_file", CAMEL_CASE_DEFINITIONS)
class TestProductDefinitionsFollowTheRule:
    """Every variable and non-dimension coordinate of the Camel_Case product definitions follows the rule."""

    def test_names_are_product_names(self, definition_file):
        """Each name is the product name of some internal name, so the definition and the code share one rule."""
        definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITIONS / definition_file)
        coordinates = [name for name, coord in definition.coordinates.items() if coord.dimensions != [name]]
        names = list(definition.variables) + coordinates
        for name in names:
            assert product_variable_name(internal_variable_name(name)) == name

    def test_names_are_unique(self, definition_file):
        """No name is written twice, which YAML loading would otherwise collapse silently."""
        for section, keys in _section_keys(PRODUCT_DEFINITIONS / definition_file).items():
            assert len(keys) == len(set(keys)), f"duplicate {section} key in {definition_file}"
