"""Unit tests for libera_utils.obsids registry."""

import csv

import pytest

from libera_utils.constants import DataLevel, DataProductIdentifier, LiberaApid
from libera_utils.obsids import (
    OBSID_REGISTRY,
    OBSID_REGISTRY_CSV,
    NomHkObsidSource,
    ObsIdKind,
    _parse_row,
    get_obsid_spec,
    iter_trim_eligible,
)


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
        """Camera TRIMMED/CAL ProductID strings match the catalog table."""
        expected = {
            129: ("NOM-HK-CT-VIDEO-6MIN-TRIMMED", "CT-VIDEO-6MIN"),
            130: ("NOM-HK-CT-VIDEO-12MIN-TRIMMED", "CT-VIDEO-12MIN"),
            131: ("NOM-HK-CT-VIDEO-18MIN-TRIMMED", "CT-VIDEO-18MIN"),
            133: ("NOM-HK-RAPS-VIDEO-6MIN-TRIMMED", "RAPS-VIDEO-6MIN"),
            134: ("NOM-HK-RAPS-VIDEO-12MIN-TRIMMED", "RAPS-VIDEO-12MIN"),
            135: ("NOM-HK-RAPS-VIDEO-18MIN-TRIMMED", "RAPS-VIDEO-18MIN"),
            256: ("NOM-HK-DARKS-OF-DARKS-TRIMMED", "DARKS-OF-DARKS"),
            257: ("NOM-HK-LED-OF-DARK-TRIMMED", "LED-OF-DARK"),
            258: ("NOM-HK-NOMINAL-DARKS-TRIMMED", "NOMINAL-DARKS"),
            513: ("NOM-HK-WFOV-VIIRS-LUNAR-POS-START-TRIMMED", "WFOV-VIIRS-LUNAR-POS-START"),
            514: ("NOM-HK-WFOV-VIIRS-LUNAR-NEG-START-TRIMMED", "WFOV-VIIRS-LUNAR-NEG-START"),
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
        assert lunar1.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lunar_south_pole_trimmed
        assert lunar1.cal_product is DataProductIdentifier.cal_lunar_south_pole
        assert lunar2.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lunar_north_pole_trimmed
        assert lunar2.cal_product is DataProductIdentifier.cal_lunar_north_pole
        assert "Monthly" in lunar1.description
        assert "Quarterly" in lunar2.description

    def test_rad_viirs_lunar_cal_obsids(self):
        """RAD ObsIDs 513/514 map to pos/neg-start VIIRS lunar products (distinct from WFOV cals)."""
        rad_pos = get_obsid_spec(NomHkObsidSource.RAD, 513)
        rad_neg = get_obsid_spec(NomHkObsidSource.RAD, 514)
        wfov_pos = get_obsid_spec(NomHkObsidSource.WFOV, 513)
        wfov_neg = get_obsid_spec(NomHkObsidSource.WFOV, 514)
        assert rad_pos.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_rad_viirs_lunar_pos_start_trimmed
        assert rad_pos.cal_product is DataProductIdentifier.cal_rad_viirs_lunar_pos_start
        assert rad_neg.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_rad_viirs_lunar_neg_start_trimmed
        assert rad_neg.cal_product is DataProductIdentifier.cal_rad_viirs_lunar_neg_start
        assert wfov_pos.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_wfov_viirs_lunar_pos_start_trimmed
        assert wfov_pos.cal_product is DataProductIdentifier.cal_wfov_viirs_lunar_pos_start
        assert wfov_neg.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_wfov_viirs_lunar_neg_start_trimmed
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

    def test_trimmed_products_are_unique_per_obsid(self):
        """No two ObsIDs share a TRIMMED product, so a trimmed file maps back to one (source, obsid)."""
        trimmed = [s.trimmed_product for s in iter_trim_eligible()]
        assert len(trimmed) == len(set(trimmed))

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
            "trimmed_product": "l1a_icie_nom_hk_gain_trimmed",
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
        assert spec.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed
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
            trimmed_product="l1a_icie_nom_hk_ct_video_6min_trimmed",
            cal_product="cal_ct_video_6min",
        )
        with pytest.raises(ValueError, match="must be registered on WFOV, got RAD"):
            _parse_row(row, 7)

    def test_products_in_swapped_columns_raise(self):
        """A CAL product under trimmed_product (or vice versa) is rejected by data level."""
        with pytest.raises(ValueError, match="is a CAL product, expected L1A"):
            _parse_row(self._row(trimmed_product="cal_gain"), 7)
        with pytest.raises(ValueError, match="is a L1A product, expected CAL"):
            _parse_row(self._row(cal_product="l1a_icie_nom_hk_gain_trimmed"), 7)

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
