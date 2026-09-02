"""Unit tests for libera_utils.obsids registry."""

import csv

import pytest

from libera_utils import obsids
from libera_utils.constants import DataLevel, DataProductIdentifier, LiberaApid
from libera_utils.obsids import (
    _COLUMNS,
    _FAMILY_INPUT_COLUMNS,
    FAMILY_INPUTS,
    OBSID_REGISTRY,
    OBSID_REGISTRY_CSV,
    TRIM_FAMILIES,
    TRIM_FAMILY_INPUTS_CSV,
    NomHkObsidSource,
    ObsIdKind,
    _parse_row,
    get_family_inputs,
    get_family_specs,
    get_obsid_spec,
    iter_trim_eligible,
)

#: Families whose cal-combine ProcessingStepIdentifier is still deferred (TODO[LIBSDC-811]), and
#: which therefore have no settled input dependency set yet.
DEFERRED_FAMILIES = {
    DataProductIdentifier.l1a_icie_nom_hk_lunar_family_trimmed,
    DataProductIdentifier.l1a_icie_nom_hk_rad_viirs_lunar_family_trimmed,
    DataProductIdentifier.l1a_icie_nom_hk_wfov_viirs_lunar_family_trimmed,
    DataProductIdentifier.l1a_icie_nom_hk_ct_video_family_trimmed,
    DataProductIdentifier.l1a_icie_nom_hk_raps_video_family_trimmed,
    DataProductIdentifier.l1a_icie_nom_hk_darks_family_trimmed,
}


def read_registry_csv() -> list[dict[str, str]]:
    """Read the ObsID catalog CSV without going through the obsids loader.

    Returns
    -------
    list of dict
        Raw rows, each mapping column name to unparsed cell text.
    """
    with OBSID_REGISTRY_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestObsidRegistry:
    """Tests for OBSID_REGISTRY completeness and consistency."""

    def test_keys_match_source_and_obsid(self):
        """Registry keys must equal (spec.source, spec.obsid)."""
        for key, spec in OBSID_REGISTRY.items():
            assert key == (spec.source, spec.obsid)

    def test_rad_and_wfov_256_differ(self):
        """RAD and WFOV ObsID 256 map to different products."""
        rad = get_obsid_spec(NomHkObsidSource.RAD, 256)
        wfov = get_obsid_spec(NomHkObsidSource.WFOV, 256)
        assert rad.cal_product is DataProductIdentifier.cal_swc_365nm
        assert wfov.cal_product is DataProductIdentifier.cal_darks_of_darks
        assert rad.trimmed_product is not wfov.trimmed_product

    def test_trim_eligible_have_both_products(self):
        """Every trim-eligible entry has trimmed and cal ProductIDs."""
        for spec in iter_trim_eligible():
            assert spec.trimmed_product is not None
            assert spec.cal_product is not None
            assert spec.kind in (ObsIdKind.RAD_CAL, ObsIdKind.CAM_CAL)

    def test_science_entries_have_no_products(self):
        """Science/scan modes are catalog-only."""
        science = [s for s in OBSID_REGISTRY.values() if s.kind is ObsIdKind.SCIENCE]
        assert len(science) == 20  # 10 ObsIDs × 2 sources
        for spec in science:
            assert spec.trimmed_product is None
            assert spec.cal_product is None

    def test_science_modes_on_both_sources(self):
        """Catalog-only ObsIDs 0-2, 128, 132, and 136-140 are registered under RAD and WFOV."""
        for obsid in (*range(0, 3), 128, 132, *range(136, 141)):
            assert get_obsid_spec(NomHkObsidSource.RAD, obsid).kind is ObsIdKind.SCIENCE
            assert get_obsid_spec(NomHkObsidSource.WFOV, obsid).kind is ObsIdKind.SCIENCE

    def test_instrument_state_obsids(self):
        """ObsIDs 0-2 describe instrument states and carry no products."""
        expected = {
            0: "Instrument Boot-Up from Power Off",
            1: "Instrument Safe Mode",
            2: "Mechanisms are in the Stowed Position",
        }
        for obsid, description in expected.items():
            for source in NomHkObsidSource:
                spec = get_obsid_spec(source, obsid)
                assert spec.description == description
                assert spec.trimmed_product is None
                assert spec.cal_product is None

    def test_cross_track_and_rap_science_modes(self):
        """ObsIDs 128 and 132 are Cross Track and RAP scan science modes."""
        cross = get_obsid_spec(NomHkObsidSource.RAD, 128)
        rap = get_obsid_spec(NomHkObsidSource.WFOV, 132)
        assert cross.description == "Cross Track Scan Mode"
        assert rap.description == "RAP Scan Mode"

    def test_trimmed_products_associate_with_nom_hk_apid(self):
        """All TRIMMED DPIs associate with the NOM-HK APID."""
        for spec in iter_trim_eligible():
            assert spec.trimmed_product is not None
            assert spec.trimmed_product.associated_apid is LiberaApid.icie_nom_hk

    def test_camera_dpi_string_values(self):
        """Camera TRIMMED family / CAL ProductID strings match the catalog table."""
        expected = {
            129: ("NOM-HK-CT-VIDEO-FAMILY-TRIMMED", "CT-VIDEO-6MIN"),
            130: ("NOM-HK-CT-VIDEO-FAMILY-TRIMMED", "CT-VIDEO-12MIN"),
            131: ("NOM-HK-CT-VIDEO-FAMILY-TRIMMED", "CT-VIDEO-18MIN"),
            133: ("NOM-HK-RAPS-VIDEO-FAMILY-TRIMMED", "RAPS-VIDEO-6MIN"),
            134: ("NOM-HK-RAPS-VIDEO-FAMILY-TRIMMED", "RAPS-VIDEO-12MIN"),
            135: ("NOM-HK-RAPS-VIDEO-FAMILY-TRIMMED", "RAPS-VIDEO-18MIN"),
            256: ("NOM-HK-DARKS-FAMILY-TRIMMED", "DARKS-OF-DARKS"),
            257: ("NOM-HK-DARKS-FAMILY-TRIMMED", "LED-OF-DARK"),
            258: ("NOM-HK-DARKS-FAMILY-TRIMMED", "NOMINAL-DARKS"),
            513: ("NOM-HK-WFOV-VIIRS-LUNAR-FAMILY-TRIMMED", "WFOV-VIIRS-LUNAR-POS-START"),
            514: ("NOM-HK-WFOV-VIIRS-LUNAR-FAMILY-TRIMMED", "WFOV-VIIRS-LUNAR-NEG-START"),
        }
        for obsid, (trimmed_val, cal_val) in expected.items():
            spec = get_obsid_spec(NomHkObsidSource.WFOV, obsid)
            assert spec.trimmed_product is not None
            assert spec.cal_product is not None
            assert spec.trimmed_product.value == trimmed_val
            assert spec.cal_product.value == cal_val

    def test_rad_cal_count(self):
        """Twenty-nine radiometer cal ObsIDs (25 gain/noise/SWC/LWC/solar + 2 lunar + 2 VIIRS lunar)."""
        rad_cal = [s for s in OBSID_REGISTRY.values() if s.kind is ObsIdKind.RAD_CAL]
        assert len(rad_cal) == 29

    def test_lunar_cal_obsids(self):
        """Radiometer lunar ObsIDs 448/449 map to LUNAR-SOUTH/NORTH-POLE products."""
        lunar1 = get_obsid_spec(NomHkObsidSource.RAD, 448)
        lunar2 = get_obsid_spec(NomHkObsidSource.RAD, 449)
        assert lunar1.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lunar_family_trimmed
        assert lunar1.cal_product is DataProductIdentifier.cal_lunar_south_pole
        assert lunar2.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lunar_family_trimmed
        assert lunar2.cal_product is DataProductIdentifier.cal_lunar_north_pole
        assert "Monthly" in lunar1.description
        assert "Quarterly" in lunar2.description

    def test_rad_viirs_lunar_cal_obsids(self):
        """RAD ObsIDs 513/514 map to pos/neg-start VIIRS lunar products (distinct from WFOV cals)."""
        rad_pos = get_obsid_spec(NomHkObsidSource.RAD, 513)
        rad_neg = get_obsid_spec(NomHkObsidSource.RAD, 514)
        wfov_pos = get_obsid_spec(NomHkObsidSource.WFOV, 513)
        wfov_neg = get_obsid_spec(NomHkObsidSource.WFOV, 514)
        rad_family = DataProductIdentifier.l1a_icie_nom_hk_rad_viirs_lunar_family_trimmed
        wfov_family = DataProductIdentifier.l1a_icie_nom_hk_wfov_viirs_lunar_family_trimmed
        assert rad_pos.trimmed_product is rad_family
        assert rad_pos.cal_product is DataProductIdentifier.cal_rad_viirs_lunar_pos_start
        assert rad_neg.trimmed_product is rad_family
        assert rad_neg.cal_product is DataProductIdentifier.cal_rad_viirs_lunar_neg_start
        assert wfov_pos.trimmed_product is wfov_family
        assert wfov_pos.cal_product is DataProductIdentifier.cal_wfov_viirs_lunar_pos_start
        assert wfov_neg.trimmed_product is wfov_family
        assert wfov_neg.cal_product is DataProductIdentifier.cal_wfov_viirs_lunar_neg_start
        assert rad_pos.trimmed_product is not wfov_pos.trimmed_product
        assert rad_pos.cal_product is not wfov_pos.cal_product
        assert rad_pos.kind is ObsIdKind.RAD_CAL
        assert wfov_pos.kind is ObsIdKind.CAM_CAL

    def test_cam_cal_count(self):
        """Eleven camera cal ObsIDs are registered."""
        cam_cal = [s for s in OBSID_REGISTRY.values() if s.kind is ObsIdKind.CAM_CAL]
        assert len(cam_cal) == 11

    def test_iter_trim_eligible_filters_by_source(self):
        """Source filter restricts iter_trim_eligible."""
        rad = list(iter_trim_eligible(NomHkObsidSource.RAD))
        wfov = list(iter_trim_eligible(NomHkObsidSource.WFOV))
        assert len(rad) == 29
        assert len(wfov) == 11
        assert all(s.source is NomHkObsidSource.RAD for s in rad)
        assert all(s.source is NomHkObsidSource.WFOV for s in wfov)

    def test_trimmed_products_group_obsids_into_families(self):
        """ObsIDs share a TRIMMED product; TRIM_FAMILIES is the exact inverse of that column."""
        expected: dict[DataProductIdentifier, list[int]] = {}
        for spec in iter_trim_eligible():
            assert spec.trimmed_product is not None
            expected.setdefault(spec.trimmed_product, []).append(spec.obsid)
        assert {product: [s.obsid for s in members] for product, members in TRIM_FAMILIES.items()} == expected
        # Sharing is the point of families, so at least one product must cover several ObsIDs
        assert any(len(members) > 1 for members in TRIM_FAMILIES.values())

    def test_each_family_is_confined_to_one_source(self):
        """A family is trimmed by scanning one NOM-HK ObsID field, so it must not span both."""
        for product, members in TRIM_FAMILIES.items():
            sources = {spec.source for spec in members}
            assert len(sources) == 1, f"{product.name} spans {sources}"

    def test_get_family_specs_returns_members_and_rejects_non_families(self):
        """get_family_specs resolves a family ProductID and raises for anything else."""
        swc = get_family_specs(DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed)
        assert [spec.obsid for spec in swc] == [256, 257, 258, 259, 260, 261]
        assert all(spec.source is NomHkObsidSource.RAD for spec in swc)
        with pytest.raises(KeyError, match="l1a_icie_nom_hk_decoded"):
            get_family_specs(DataProductIdentifier.l1a_icie_nom_hk_decoded)

    def test_cal_products_are_unique_per_obsid(self):
        """No two ObsIDs share a CAL product."""
        cal = [s.cal_product for s in OBSID_REGISTRY.values() if s.cal_product is not None]
        assert len(cal) == len(set(cal))

    def test_cal_kinds_are_registered_on_their_own_source(self):
        """rad_cal entries live on the RAD ObsID field and cam_cal entries on WFOV."""
        expected = {ObsIdKind.RAD_CAL: NomHkObsidSource.RAD, ObsIdKind.CAM_CAL: NomHkObsidSource.WFOV}
        for spec in OBSID_REGISTRY.values():
            if spec.kind in expected:
                assert spec.source is expected[spec.kind]

    def test_get_obsid_spec_unknown_raises(self):
        """Unknown ObsID raises KeyError."""
        with pytest.raises(KeyError, match="99999"):
            get_obsid_spec(NomHkObsidSource.RAD, 99999)


class TestObsidRegistryCsv:
    """Audits of the ObsID catalog data file itself, independent of the loader."""

    def test_every_product_cell_names_a_real_data_product_identifier(self):
        """Every non-empty TRIMMED/CAL cell resolves to a DataProductIdentifier member."""
        rows = read_registry_csv()
        assert rows, f"{OBSID_REGISTRY_CSV.name} contains no rows"

        unresolved = []
        for line, row in enumerate(rows, start=2):  # line 1 is the header
            for column in ("trimmed_product", "cal_product"):
                name = row[column].strip()
                if not name:
                    continue
                if name not in DataProductIdentifier.__members__:
                    unresolved.append(f"{OBSID_REGISTRY_CSV.name}:{line} {column}={name!r}")
        assert not unresolved, "Cells naming no DataProductIdentifier member:\n" + "\n".join(unresolved)

    def test_product_cells_have_the_expected_data_levels(self):
        """TRIMMED cells name L1A products and CAL cells name CAL products."""
        wrong_level = []
        for line, row in enumerate(read_registry_csv(), start=2):
            for column, expected in (("trimmed_product", DataLevel.L1A), ("cal_product", DataLevel.CAL)):
                name = row[column].strip()
                if not name or name not in DataProductIdentifier.__members__:
                    continue  # existence is covered by the companion test
                level = DataProductIdentifier[name].data_level
                if level is not expected:
                    wrong_level.append(
                        f"{OBSID_REGISTRY_CSV.name}:{line} {column}={name!r} is {level}, expected {expected}"
                    )
        assert not wrong_level, "Cells naming a product at the wrong data level:\n" + "\n".join(wrong_level)

    def test_every_row_is_loaded_into_the_registry(self):
        """The loaded registry has exactly one entry per catalog row."""
        rows = read_registry_csv()
        keys = [(NomHkObsidSource[row["source"].strip()], int(row["obsid"])) for row in rows]
        assert len(keys) == len(set(keys)), "Catalog contains duplicate (source, obsid) rows"
        assert list(OBSID_REGISTRY) == keys

    def test_kinds_and_sources_are_recognized(self):
        """Every row names a valid ObsIdKind and NomHkObsidSource."""
        for line, row in enumerate(read_registry_csv(), start=2):
            where = f"{OBSID_REGISTRY_CSV.name}:{line}"
            assert row["source"].strip() in NomHkObsidSource.__members__, f"{where} bad source"
            assert row["kind"].strip() in {k.value for k in ObsIdKind}, f"{where} bad kind"
            assert row["description"].strip(), f"{where} has an empty description"


class TestObsidRegistryLoader:
    """Validation performed while parsing catalog rows."""

    @staticmethod
    def _row(**overrides: str | None) -> dict[str, str]:
        """Build a valid raw catalog row, with optional column overrides."""
        row = {
            "source": "RAD",
            "obsid": "512",
            "kind": "rad_cal",
            "trimmed_product": "l1a_icie_nom_hk_gain_family_trimmed",
            "cal_product": "cal_gain",
            "description": "Gain calibration",
        }
        row.update(overrides)
        return row

    def test_valid_row_parses(self):
        """A well-formed row parses into the expected ObsIdSpec."""
        spec = _parse_row(self._row(), 2)
        assert spec.obsid == 512
        assert spec.source is NomHkObsidSource.RAD
        assert spec.kind is ObsIdKind.RAD_CAL
        assert spec.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed
        assert spec.cal_product is DataProductIdentifier.cal_gain

    def test_unknown_product_name_raises(self):
        """A misspelled DataProductIdentifier member name is rejected."""
        with pytest.raises(ValueError, match="not a DataProductIdentifier member name"):
            _parse_row(self._row(cal_product="cal_gian"), 7)

    def test_product_value_instead_of_member_name_raises(self):
        """Cells must hold member names, not ProductID string values."""
        with pytest.raises(ValueError, match="not a DataProductIdentifier member name"):
            _parse_row(self._row(cal_product="GAIN"), 7)

    def test_unknown_source_raises(self):
        """An unrecognized source column is rejected."""
        with pytest.raises(ValueError, match="not one of RAD, WFOV"):
            _parse_row(self._row(source="RADIOMETER"), 7)

    def test_unknown_kind_raises(self):
        """An unrecognized kind column is rejected."""
        with pytest.raises(ValueError, match="column 'kind'"):
            _parse_row(self._row(kind="calibration"), 7)

    def test_non_integer_obsid_raises(self):
        """A non-numeric ObsID is rejected."""
        with pytest.raises(ValueError, match="not an integer"):
            _parse_row(self._row(obsid="0x200"), 7)

    def test_cal_kind_on_wrong_source_raises(self):
        """A radiometer cal registered on the camera ObsID field is rejected."""
        with pytest.raises(ValueError, match="must be registered on RAD, got WFOV"):
            _parse_row(self._row(source="WFOV"), 7)

    def test_cam_cal_on_wrong_source_raises(self):
        """A camera cal registered on the radiometer ObsID field is rejected."""
        row = self._row(
            kind="cam_cal",
            obsid="129",
            trimmed_product="l1a_icie_nom_hk_ct_video_family_trimmed",
            cal_product="cal_ct_video_6min",
        )
        with pytest.raises(ValueError, match="must be registered on WFOV, got RAD"):
            _parse_row(row, 7)

    def test_products_in_swapped_columns_raise(self):
        """A CAL product under trimmed_product (or vice versa) is rejected by data level."""
        with pytest.raises(ValueError, match="is a CAL product, expected L1A"):
            _parse_row(self._row(trimmed_product="cal_gain"), 7)
        with pytest.raises(ValueError, match="is a L1A product, expected CAL"):
            _parse_row(self._row(cal_product="l1a_icie_nom_hk_gain_family_trimmed"), 7)

    def test_row_with_missing_column_raises(self):
        """A row with too few columns names the file, line, and missing column."""
        row = self._row(description=None)  # what csv.DictReader hands back for a short row
        with pytest.raises(ValueError, match="expected exactly 6 columns"):
            _parse_row(row, 7)

    def test_row_with_surplus_columns_raises(self):
        """An unquoted comma in a description surfaces as an error, not a silent truncation."""
        row = self._row(description="Gain calibration")
        row["_extra_columns"] = [" second half of an unquoted description"]
        with pytest.raises(ValueError, match="Quote any cell containing a comma"):
            _parse_row(row, 7)

    def test_science_row_with_products_raises(self):
        """Science entries are catalog-only and must not name products."""
        with pytest.raises(ValueError, match="must not name TRIMMED or CAL products"):
            _parse_row(self._row(kind="science", obsid="128"), 7)

    def test_cal_row_missing_a_product_raises(self):
        """Calibration entries must name both TRIMMED and CAL products."""
        with pytest.raises(ValueError, match="must name both TRIMMED and CAL products"):
            _parse_row(self._row(cal_product=""), 7)


class TestObsidRegistryLoaderCrossRowChecks:
    """Validation that only shows up once the whole catalog has been read."""

    @staticmethod
    def _load(tmp_path, monkeypatch, rows: list[str]):
        """Run _load_registry against a synthetic catalog instead of the shipped one."""
        catalog = tmp_path / "obsid_registry.csv"
        catalog.write_text("\n".join([",".join(_COLUMNS), *rows]) + "\n", encoding="utf-8")
        monkeypatch.setattr(obsids, "OBSID_REGISTRY_CSV", catalog)
        return obsids._load_registry()

    def test_obsids_may_share_a_trimmed_family(self, tmp_path, monkeypatch):
        """Sharing a TRIMMED product is the point of families, so it must load cleanly."""
        registry, families = self._load(
            tmp_path,
            monkeypatch,
            [
                "RAD,512,rad_cal,l1a_icie_nom_hk_gain_family_trimmed,cal_gain,Gain calibration",
                "RAD,515,rad_cal,l1a_icie_nom_hk_gain_family_trimmed,cal_noise,Noise calibration",
            ],
        )
        assert len(registry) == 2
        family = families[DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed]
        assert [spec.obsid for spec in family] == [512, 515]

    def test_duplicate_cal_product_raises(self, tmp_path, monkeypatch):
        """Two ObsIDs claiming one CAL product would collide on the family step's output."""
        with pytest.raises(ValueError, match="is already claimed by ObsID 512"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "RAD,512,rad_cal,l1a_icie_nom_hk_gain_family_trimmed,cal_gain,Gain calibration",
                    "RAD,515,rad_cal,l1a_icie_nom_hk_gain_family_trimmed,cal_gain,Noise calibration",
                ],
            )

    def test_family_spanning_both_sources_raises(self, tmp_path, monkeypatch):
        """A family is trimmed off one NOM-HK ObsID field, so it must not span RAD and WFOV."""
        with pytest.raises(ValueError, match="must not span both"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "RAD,513,rad_cal,l1a_icie_nom_hk_rad_viirs_lunar_family_trimmed,"
                    "cal_rad_viirs_lunar_pos_start,VIIRS lunar positive start",
                    "WFOV,513,cam_cal,l1a_icie_nom_hk_rad_viirs_lunar_family_trimmed,"
                    "cal_wfov_viirs_lunar_pos_start,VIIRS lunar positive start",
                ],
            )

    def test_duplicate_source_obsid_raises(self, tmp_path, monkeypatch):
        """Two rows for one (source, obsid) key are still rejected."""
        with pytest.raises(ValueError, match="duplicate entry for ObsID 512 on RAD"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "RAD,512,rad_cal,l1a_icie_nom_hk_gain_family_trimmed,cal_gain,Gain calibration",
                    "RAD,512,rad_cal,l1a_icie_nom_hk_gain_family_trimmed,cal_noise,Gain calibration again",
                ],
            )


class TestFamilyInputs:
    """Tests for the per-family calibration input catalog."""

    def test_covers_exactly_the_registered_families(self):
        """A family without declared inputs, or inputs for an unregistered family, is a bug."""
        assert set(FAMILY_INPUTS) == set(TRIM_FAMILIES)

    def test_radiometer_families_declare_inputs(self):
        """The four families with deployed cal steps must name what those steps consume."""
        for family in set(TRIM_FAMILIES) - DEFERRED_FAMILIES:
            assert FAMILY_INPUTS[family], f"{family.name} has a cal step but declares no inputs"

    def test_deferred_families_declare_no_inputs(self):
        """Families whose step is deferred have an undecided dependency set, not an empty one."""
        for family in DEFERRED_FAMILIES:
            assert FAMILY_INPUTS[family] == ()

    def test_every_declared_input_is_an_l1a_product(self):
        """Cal steps consume L1A granules; anything else means a wrong column or a typo."""
        for family, inputs in FAMILY_INPUTS.items():
            for product in inputs:
                assert product.data_level is DataLevel.L1A, f"{family.name} declares {product.name}"

    def test_swc_family_inputs_match_the_historical_combined_list(self):
        """The SWC list is the pre-family `SW-COMBINED` one, less the daily NOM-HK."""
        assert get_family_inputs(DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed) == (
            DataProductIdentifier.l1a_pev_sw_stat_decoded,
            DataProductIdentifier.l1a_pec_sw_stat_decoded,
            DataProductIdentifier.l1a_icie_rad_sample_decoded,
            DataProductIdentifier.l1a_icie_cal_sample_decoded,
            DataProductIdentifier.l1a_icie_axis_sample_decoded,
        )

    def test_gain_family_inputs_match_the_historical_combined_list(self):
        """Gain and noise share one family, and the old `GAIN-COMBINED` comment's inputs."""
        assert get_family_inputs(DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed) == (
            DataProductIdentifier.l1a_icie_rad_full_decoded,
            DataProductIdentifier.l1a_icie_cal_full_decoded,
        )

    def test_no_family_requires_the_daily_nom_hk(self):
        """A family's NOM-HK arrives as its TRIMMED product, so the daily granule is redundant."""
        for family, inputs in FAMILY_INPUTS.items():
            assert DataProductIdentifier.l1a_icie_nom_hk_decoded not in inputs, (
                f"{family.name} lists the full-day NOM-HK; the trimmed family product supplies it"
            )

    def test_get_family_inputs_rejects_a_non_family_product(self):
        """Asking for the inputs of something that is not a TRIMMED family is an error."""
        with pytest.raises(KeyError, match="is not a TRIMMED calibration family ProductID"):
            get_family_inputs(DataProductIdentifier.l1a_icie_nom_hk_decoded)

    def test_every_catalog_row_is_loaded(self):
        """Every row in the shipped CSV reaches FAMILY_INPUTS."""
        with TRIM_FAMILY_INPUTS_CSV.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(FAMILY_INPUTS)
        assert {DataProductIdentifier[row["trimmed_product"]] for row in rows} == set(FAMILY_INPUTS)


class TestFamilyInputsLoader:
    """Validation performed while reading the family-inputs catalog."""

    @staticmethod
    def _load(tmp_path, monkeypatch, rows: list[str], families=None):
        """Run _load_family_inputs against a synthetic catalog instead of the shipped one."""
        catalog = tmp_path / "trim_family_inputs.csv"
        catalog.write_text("\n".join([",".join(_FAMILY_INPUT_COLUMNS), *rows]) + "\n", encoding="utf-8")
        monkeypatch.setattr(obsids, "TRIM_FAMILY_INPUTS_CSV", catalog)
        if families is None:
            families = [DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed]
        return obsids._load_family_inputs(families)

    def test_valid_catalog_parses(self, tmp_path, monkeypatch):
        """A well-formed row resolves each semicolon-separated member name."""
        inputs = self._load(
            tmp_path,
            monkeypatch,
            ["l1a_icie_nom_hk_gain_family_trimmed,l1a_icie_nom_hk_decoded;l1a_icie_rad_full_decoded"],
        )
        assert inputs == {
            DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed: (
                DataProductIdentifier.l1a_icie_nom_hk_decoded,
                DataProductIdentifier.l1a_icie_rad_full_decoded,
            )
        }

    def test_empty_inputs_cell_is_allowed(self, tmp_path, monkeypatch):
        """An undecided dependency set is expressed as an empty cell, not a missing row."""
        inputs = self._load(tmp_path, monkeypatch, ["l1a_icie_nom_hk_gain_family_trimmed,"])
        assert inputs == {DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed: ()}

    def test_family_missing_from_this_catalog_raises(self, tmp_path, monkeypatch):
        """Registering a family without declaring its inputs must not pass silently."""
        with pytest.raises(ValueError, match="Families with no declared inputs"):
            self._load(
                tmp_path,
                monkeypatch,
                ["l1a_icie_nom_hk_gain_family_trimmed,l1a_icie_nom_hk_decoded"],
                families=[
                    DataProductIdentifier.l1a_icie_nom_hk_gain_family_trimmed,
                    DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed,
                ],
            )

    def test_family_not_in_the_obsid_registry_raises(self, tmp_path, monkeypatch):
        """Inputs for a family no ObsID produces are dead weight and probably a typo."""
        with pytest.raises(ValueError, match="registered against no ObsID"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "l1a_icie_nom_hk_gain_family_trimmed,l1a_icie_nom_hk_decoded",
                    "l1a_icie_nom_hk_swc_family_trimmed,l1a_icie_nom_hk_decoded",
                ],
            )

    def test_non_l1a_input_raises(self, tmp_path, monkeypatch):
        """A CAL or L0 product in the inputs column means the wrong catalog was edited."""
        with pytest.raises(ValueError, match="is a CAL product, expected L1A"):
            self._load(tmp_path, monkeypatch, ["l1a_icie_nom_hk_gain_family_trimmed,cal_gain"])

    def test_unknown_input_name_raises(self, tmp_path, monkeypatch):
        """Cells hold DataProductIdentifier member names, not ProductID string values."""
        with pytest.raises(ValueError, match="is not a DataProductIdentifier member name"):
            self._load(tmp_path, monkeypatch, ["l1a_icie_nom_hk_gain_family_trimmed,NOM-HK-DECODED"])

    def test_duplicate_family_row_raises(self, tmp_path, monkeypatch):
        """Two rows for one family leave it ambiguous which input list wins."""
        with pytest.raises(ValueError, match="duplicate entry for TRIMMED family"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "l1a_icie_nom_hk_gain_family_trimmed,l1a_icie_nom_hk_decoded",
                    "l1a_icie_nom_hk_gain_family_trimmed,l1a_icie_rad_full_decoded",
                ],
            )

    def test_missing_family_cell_raises(self, tmp_path, monkeypatch):
        """A row has to name the family it declares inputs for."""
        with pytest.raises(ValueError, match="a family ProductID is required"):
            self._load(tmp_path, monkeypatch, [",l1a_icie_nom_hk_decoded"])

    def test_surplus_columns_raise(self, tmp_path, monkeypatch):
        """An unquoted extra comma would otherwise silently truncate the inputs list."""
        with pytest.raises(ValueError, match="expected exactly 2 columns"):
            self._load(
                tmp_path,
                monkeypatch,
                ["l1a_icie_nom_hk_gain_family_trimmed,l1a_icie_nom_hk_decoded,surplus"],
            )
