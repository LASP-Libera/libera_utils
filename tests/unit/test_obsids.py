"""Unit tests for libera_utils.obsids registry."""

import csv

import pytest

from libera_utils import obsids
from libera_utils.constants import DataLevel, DataProductIdentifier, LiberaApid, ProcessingStepIdentifier
from libera_utils.obsids import (
    _COLUMNS,
    _COMBINER_INPUT_COLUMNS,
    CAL_COMBINER_INPUTS_CSV,
    COMBINER_INPUTS,
    OBSID_REGISTRY,
    OBSID_REGISTRY_CSV,
    NomHkObsidSource,
    ObsIdKind,
    ObsIdSpec,
    _parse_row,
    get_combiner_inputs,
    get_combiner_step,
    get_obsid_spec,
    get_trimmed_obsid_spec,
    is_calibration_event_product,
    iter_trim_eligible,
)

#: TRIMMED products whose combiner input dependency set is still undecided: the radiometer lunar and
#: VIIRS lunar ObsIDs and every camera calibration ObsID.
UNDECIDED_INPUTS = {
    spec.trimmed_product
    for spec in iter_trim_eligible()
    if spec.kind is ObsIdKind.CAM_CAL or (spec.source is NomHkObsidSource.RAD and spec.obsid in {448, 449, 513, 514})
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
        """Camera TRIMMED / CAL ProductID strings match the catalog table."""
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

    def test_each_calibration_obsid_has_its_own_trimmed_product(self):
        """TRIMMED products are 1:1 with calibration ObsIDs and named for their CAL product."""
        trimmed = [spec.trimmed_product for spec in iter_trim_eligible()]
        assert len(trimmed) == 40
        assert len(set(trimmed)) == len(trimmed)
        for spec in iter_trim_eligible():
            assert spec.trimmed_product.value == f"NOM-HK-{spec.cal_product.value}-TRIMMED"
            assert get_trimmed_obsid_spec(spec.trimmed_product) is spec

    def test_every_trimmed_product_is_registered(self):
        """A TRIMMED member no ObsID owns would be written by nothing and consumed by nothing."""
        registered = {spec.trimmed_product for spec in iter_trim_eligible()}
        declared = {product for product in DataProductIdentifier if product.name.endswith("_trimmed")}
        assert declared == registered

    def test_get_combiner_step_returns_the_step_producing_the_obsids_cal_product(self):
        """TRIMMED product, combiner step and CAL product are 1:1:1."""
        steps = [get_combiner_step(spec.trimmed_product) for spec in iter_trim_eligible()]
        assert len(set(steps)) == len(steps)
        for spec, step in zip(iter_trim_eligible(), steps, strict=True):
            assert step.products == [spec.cal_product]
        assert (
            get_combiner_step(DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed)
            is ProcessingStepIdentifier.cal_combine_swc_365nm
        )

    def test_lookups_by_trimmed_product_reject_other_products(self):
        """Asking for the ObsID or step of a product that is not TRIMMED is an error."""
        for lookup in (get_trimmed_obsid_spec, get_combiner_step, get_combiner_inputs):
            with pytest.raises(KeyError, match="'l1a_icie_nom_hk_decoded' is not a TRIMMED ProductID"):
                lookup(DataProductIdentifier.l1a_icie_nom_hk_decoded)

    def test_is_calibration_event_product(self):
        """TRIMMED and CAL products of calibration ObsIDs hold events; other products do not."""
        assert is_calibration_event_product(DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed)
        assert is_calibration_event_product(DataProductIdentifier.cal_gain)
        assert not is_calibration_event_product(DataProductIdentifier.l1a_icie_nom_hk_decoded)
        assert not is_calibration_event_product(DataProductIdentifier.l1b_rad)
        events = [product for product in DataProductIdentifier if is_calibration_event_product(product)]
        assert len(events) == 80

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


class TestObsidRegistryLoaderCrossRowChecks:
    """Validation that only shows up once the whole catalog has been read."""

    @staticmethod
    def _load(tmp_path, monkeypatch, rows: list[str]):
        """Run _load_registry against a synthetic catalog instead of the shipped one."""
        catalog = tmp_path / "obsid_registry.csv"
        catalog.write_text("\n".join([",".join(_COLUMNS), *rows]) + "\n", encoding="utf-8")
        monkeypatch.setattr(obsids, "OBSID_REGISTRY_CSV", catalog)
        return obsids._load_registry()

    def test_each_trimmed_product_maps_to_its_obsid(self, tmp_path, monkeypatch):
        """A valid catalog loads, and each TRIMMED product maps back to the one ObsID that owns it."""
        registry, by_trimmed = self._load(
            tmp_path,
            monkeypatch,
            [
                "RAD,512,rad_cal,l1a_icie_nom_hk_gain_trimmed,cal_gain,Gain calibration",
                "RAD,515,rad_cal,l1a_icie_nom_hk_noise_trimmed,cal_noise,Noise calibration",
            ],
        )
        assert len(registry) == 2
        assert by_trimmed[DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed].obsid == 512
        assert by_trimmed[DataProductIdentifier.l1a_icie_nom_hk_noise_trimmed].obsid == 515

    def test_duplicate_trimmed_product_raises(self, tmp_path, monkeypatch):
        """Two ObsIDs claiming one TRIMMED product could not be told apart by the combiner."""
        with pytest.raises(ValueError, match="TRIMMED product 'l1a_icie_nom_hk_gain_trimmed' is already claimed"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "RAD,512,rad_cal,l1a_icie_nom_hk_gain_trimmed,cal_gain,Gain calibration",
                    "RAD,515,rad_cal,l1a_icie_nom_hk_gain_trimmed,cal_noise,Noise calibration",
                ],
            )

    def test_duplicate_cal_product_raises(self, tmp_path, monkeypatch):
        """Two ObsIDs claiming one CAL product would collide on the combiner step's output."""
        with pytest.raises(ValueError, match="CAL product 'cal_gain' is already claimed by ObsID 512"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "RAD,512,rad_cal,l1a_icie_nom_hk_gain_trimmed,cal_gain,Gain calibration",
                    "RAD,515,rad_cal,l1a_icie_nom_hk_noise_trimmed,cal_gain,Noise calibration",
                ],
            )

    def test_duplicate_source_obsid_raises(self, tmp_path, monkeypatch):
        """Two rows for one (source, obsid) key are still rejected."""
        with pytest.raises(ValueError, match="duplicate entry for ObsID 512 on RAD"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "RAD,512,rad_cal,l1a_icie_nom_hk_gain_trimmed,cal_gain,Gain calibration",
                    "RAD,512,rad_cal,l1a_icie_nom_hk_noise_trimmed,cal_noise,Gain calibration again",
                ],
            )


class TestCombinerSteps:
    """The link from each TRIMMED product to the step that combines it."""

    def test_cal_product_with_no_step_raises(self):
        """A CAL product no step produces leaves its events with nothing to combine them."""
        spec = ObsIdSpec(
            obsid=512,
            source=NomHkObsidSource.RAD,
            kind=ObsIdKind.RAD_CAL,
            description="Gain calibration",
            trimmed_product=DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed,
            cal_product=DataProductIdentifier.l0_pds_cr,
        )
        with pytest.raises(ValueError, match="No ProcessingStepIdentifier produces 'l0_pds_cr'"):
            obsids._combiner_steps({spec.trimmed_product: spec})


class TestCombinerInputs:
    """Tests for the calibration combiner input catalog."""

    def test_covers_exactly_the_registered_trimmed_products(self):
        """A combiner without declared inputs, or inputs for an unregistered product, is a bug."""
        assert set(COMBINER_INPUTS) == {spec.trimmed_product for spec in iter_trim_eligible()}

    def test_radiometer_combiners_declare_inputs(self):
        """Combiners whose dependency set is settled must name what they consume."""
        for trimmed in set(COMBINER_INPUTS) - UNDECIDED_INPUTS:
            assert COMBINER_INPUTS[trimmed], f"{trimmed.name} declares no inputs"

    def test_undecided_combiners_declare_no_inputs(self):
        """An undecided dependency set is an empty cell, not a guessed one."""
        assert len(UNDECIDED_INPUTS) == 15
        for trimmed in UNDECIDED_INPUTS:
            assert COMBINER_INPUTS[trimmed] == ()

    def test_every_declared_input_is_an_l1a_product(self):
        """Combiners consume L1A granules; anything else means a wrong column or a typo."""
        for trimmed, inputs in COMBINER_INPUTS.items():
            for product in inputs:
                assert product.data_level is DataLevel.L1A, f"{trimmed.name} declares {product.name}"

    def test_swc_inputs_match_the_historical_combined_list(self):
        """Every SWC LED combiner takes the historical `SW-COMBINED` list, less the daily NOM-HK."""
        expected = (
            DataProductIdentifier.l1a_pev_sw_stat_decoded,
            DataProductIdentifier.l1a_pec_sw_stat_decoded,
            DataProductIdentifier.l1a_icie_rad_sample_decoded,
            DataProductIdentifier.l1a_icie_cal_sample_decoded,
            DataProductIdentifier.l1a_icie_axis_sample_decoded,
        )
        for obsid in range(256, 262):
            trimmed = get_obsid_spec(NomHkObsidSource.RAD, obsid).trimmed_product
            assert get_combiner_inputs(trimmed) == expected

    def test_gain_and_noise_inputs_match_the_historical_combined_list(self):
        """Gain and noise take the old `GAIN-COMBINED` comment's inputs."""
        expected = (DataProductIdentifier.l1a_icie_rad_full_decoded, DataProductIdentifier.l1a_icie_cal_full_decoded)
        assert get_combiner_inputs(DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed) == expected
        assert get_combiner_inputs(DataProductIdentifier.l1a_icie_nom_hk_noise_trimmed) == expected

    def test_no_combiner_requires_the_daily_nom_hk(self):
        """A combiner's NOM-HK arrives as its TRIMMED product, so the daily granule is redundant."""
        for trimmed, inputs in COMBINER_INPUTS.items():
            assert DataProductIdentifier.l1a_icie_nom_hk_decoded not in inputs, (
                f"{trimmed.name} lists the full-day NOM-HK; its TRIMMED product supplies it"
            )

    def test_every_catalog_row_is_loaded(self):
        """Every row in the shipped CSV reaches COMBINER_INPUTS."""
        with CAL_COMBINER_INPUTS_CSV.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(COMBINER_INPUTS)
        assert {DataProductIdentifier[row["trimmed_product"]] for row in rows} == set(COMBINER_INPUTS)


class TestCombinerInputsLoader:
    """Validation performed while reading the combiner-inputs catalog."""

    @staticmethod
    def _load(tmp_path, monkeypatch, rows: list[str], trimmed_products=None):
        """Run _load_combiner_inputs against a synthetic catalog instead of the shipped one."""
        catalog = tmp_path / "cal_combiner_inputs.csv"
        catalog.write_text("\n".join([",".join(_COMBINER_INPUT_COLUMNS), *rows]) + "\n", encoding="utf-8")
        monkeypatch.setattr(obsids, "CAL_COMBINER_INPUTS_CSV", catalog)
        if trimmed_products is None:
            trimmed_products = [DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed]
        return obsids._load_combiner_inputs(trimmed_products)

    def test_valid_catalog_parses(self, tmp_path, monkeypatch):
        """A well-formed row resolves each semicolon-separated member name."""
        inputs = self._load(
            tmp_path,
            monkeypatch,
            ["l1a_icie_nom_hk_gain_trimmed,l1a_icie_nom_hk_decoded;l1a_icie_rad_full_decoded"],
        )
        assert inputs == {
            DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed: (
                DataProductIdentifier.l1a_icie_nom_hk_decoded,
                DataProductIdentifier.l1a_icie_rad_full_decoded,
            )
        }

    def test_empty_inputs_cell_is_allowed(self, tmp_path, monkeypatch):
        """An undecided dependency set is expressed as an empty cell, not a missing row."""
        inputs = self._load(tmp_path, monkeypatch, ["l1a_icie_nom_hk_gain_trimmed,"])
        assert inputs == {DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed: ()}

    def test_trimmed_product_missing_from_this_catalog_raises(self, tmp_path, monkeypatch):
        """Registering a calibration ObsID without declaring its combiner's inputs must not pass silently."""
        with pytest.raises(ValueError, match="TRIMMED products with no row here"):
            self._load(
                tmp_path,
                monkeypatch,
                ["l1a_icie_nom_hk_gain_trimmed,l1a_icie_nom_hk_decoded"],
                trimmed_products=[
                    DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed,
                    DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed,
                ],
            )

    def test_trimmed_product_not_in_the_obsid_registry_raises(self, tmp_path, monkeypatch):
        """Inputs for a TRIMMED product no ObsID owns are dead weight and probably a typo."""
        with pytest.raises(ValueError, match="registered against no ObsID"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "l1a_icie_nom_hk_gain_trimmed,l1a_icie_nom_hk_decoded",
                    "l1a_icie_nom_hk_swc_365nm_trimmed,l1a_icie_nom_hk_decoded",
                ],
            )

    def test_non_l1a_input_raises(self, tmp_path, monkeypatch):
        """A CAL or L0 product in the inputs column means the wrong catalog was edited."""
        with pytest.raises(ValueError, match="is a CAL product, expected L1A"):
            self._load(tmp_path, monkeypatch, ["l1a_icie_nom_hk_gain_trimmed,cal_gain"])

    def test_unknown_input_name_raises(self, tmp_path, monkeypatch):
        """Cells hold DataProductIdentifier member names, not ProductID string values."""
        with pytest.raises(ValueError, match="is not a DataProductIdentifier member name"):
            self._load(tmp_path, monkeypatch, ["l1a_icie_nom_hk_gain_trimmed,NOM-HK-DECODED"])

    def test_duplicate_trimmed_product_row_raises(self, tmp_path, monkeypatch):
        """Two rows for one TRIMMED product leave it ambiguous which input list wins."""
        with pytest.raises(ValueError, match="duplicate entry for TRIMMED product"):
            self._load(
                tmp_path,
                monkeypatch,
                [
                    "l1a_icie_nom_hk_gain_trimmed,l1a_icie_nom_hk_decoded",
                    "l1a_icie_nom_hk_gain_trimmed,l1a_icie_rad_full_decoded",
                ],
            )

    def test_missing_trimmed_product_cell_raises(self, tmp_path, monkeypatch):
        """A row has to name the TRIMMED product it declares inputs for."""
        with pytest.raises(ValueError, match="a TRIMMED ProductID is required"):
            self._load(tmp_path, monkeypatch, [",l1a_icie_nom_hk_decoded"])

    def test_surplus_columns_raise(self, tmp_path, monkeypatch):
        """An unquoted extra comma would otherwise silently truncate the inputs list."""
        with pytest.raises(ValueError, match="expected exactly 2 columns"):
            self._load(
                tmp_path,
                monkeypatch,
                ["l1a_icie_nom_hk_gain_trimmed,l1a_icie_nom_hk_decoded,surplus"],
            )

    def test_missing_required_inputs_column_raises(self, tmp_path, monkeypatch):
        """A catalog whose header lacks a column cannot be read row by row."""
        catalog = tmp_path / "cal_combiner_inputs.csv"
        catalog.write_text("trimmed_product\nl1a_icie_nom_hk_gain_trimmed\n", encoding="utf-8")
        monkeypatch.setattr(obsids, "CAL_COMBINER_INPUTS_CSV", catalog)
        with pytest.raises(ValueError, match="cal_combiner_inputs.csv is missing column\\(s\\): required_inputs"):
            obsids._load_combiner_inputs([DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed])
