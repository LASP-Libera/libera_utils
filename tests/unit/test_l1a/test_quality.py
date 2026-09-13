"""Unit tests for the per-granule L1A quality record, its flag rollup, and its sidecar."""

import json
from pathlib import Path

import pytest
import yaml

from libera_utils.config import config
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.l1a.quality import (
    QUALITY_GLOBAL_ATTRIBUTES,
    DuplicateEvidence,
    GranuleQualityRecord,
    QualityFlag,
    QualityThresholds,
)

_PRODUCT = "LIBERA_L1A_RAD-SAMPLE-DECODED_V5-8-5_20260712T000023_20260713T000044_R26100000001.nc"


class TestQualityFlagRollup:
    def test_clean_granule_is_nominal(self):
        record = GranuleQualityRecord(n_packets=28_495, n_samples=1_424_750)
        assert record.quality_flag() is QualityFlag.NOMINAL

    def test_measured_ditl2_inversion_rate_stays_nominal(self):
        """1.8% of RAD packets invert as a matter of course; a flag that fires on that is noise.

        The ordering is corrected rather than lost, so an inversion costs the granule nothing.
        """
        record = GranuleQualityRecord(
            n_packets=28_006,
            n_samples=1_400_300,
            packet_time_inversion_count=501,
            packets_out_of_time_order_count=5_063,
            max_packet_time_inversion_microseconds=1_814_295,
        )
        assert record.quality_flag() is QualityFlag.NOMINAL

    def test_a_single_value_mismatch_is_degraded(self):
        """One duplicate whose rows disagree means one real sample was discarded."""
        record = GranuleQualityRecord(
            n_packets=28_006,
            n_samples=1_400_300,
            duplicate_sample_time_count=1,
            duplicate_value_mismatch_count=1,
        )
        assert record.quality_flag() is QualityFlag.DEGRADED

    def test_measured_ditl2_duplicate_load_is_degraded(self):
        """10,206 disagreeing duplicates out of 1.4M samples: real loss, but under 1%."""
        record = GranuleQualityRecord(
            n_packets=28_007,
            n_samples=1_400_350,
            duplicate_sample_time_count=10_206,
            duplicate_value_mismatch_count=10_206,
        )
        assert record.quality_flag() is QualityFlag.DEGRADED

    def test_heavy_value_mismatch_is_suspect(self):
        record = GranuleQualityRecord(
            n_packets=100,
            n_samples=1_000,
            duplicate_sample_time_count=500,
            duplicate_value_mismatch_count=500,
        )
        assert record.quality_flag() is QualityFlag.SUSPECT

    def test_uncorroborated_order_short_circuits_to_suspect(self):
        """If acquisition order is in doubt, every positional index in the granule is too."""
        record = GranuleQualityRecord(n_packets=1_000, n_samples=1_000, sequence_order_violation_count=1)
        assert record.quality_flag() is QualityFlag.SUSPECT

    def test_missing_packets_escalate(self):
        packets = 1_000
        assert GranuleQualityRecord(n_packets=packets, missing_packet_count=5).quality_flag() is QualityFlag.NOMINAL
        assert GranuleQualityRecord(n_packets=packets, missing_packet_count=50).quality_flag() is QualityFlag.DEGRADED
        assert GranuleQualityRecord(n_packets=packets, missing_packet_count=500).quality_flag() is QualityFlag.SUSPECT

    def test_thresholds_are_tunable(self):
        record = GranuleQualityRecord(n_packets=1_000, packet_time_inversion_count=20)
        assert record.quality_flag() is QualityFlag.NOMINAL
        strict = QualityThresholds(degraded_time_inversion_frac=0.01)
        assert record.quality_flag(strict) is QualityFlag.DEGRADED

    def test_empty_granule_does_not_divide_by_zero(self):
        assert GranuleQualityRecord().quality_flag() is QualityFlag.NOMINAL


class TestGlobalAttributes:
    def test_every_attribute_is_written_including_zeros(self):
        attributes = GranuleQualityRecord().global_attributes()
        assert set(attributes) == set(QUALITY_GLOBAL_ATTRIBUTES)
        assert attributes["QualityFlag"] == "NOMINAL"
        assert all(value == 0 for name, value in attributes.items() if name != "QualityFlag")

    def test_values_are_plain_python_ints(self):
        record = GranuleQualityRecord(n_packets=10, packet_time_inversion_count=3)
        attributes = record.global_attributes()
        assert type(attributes["PacketTimeInversionCount"]) is int

    @pytest.mark.parametrize(
        "definition_name",
        [
            "icie_axis_sample_l1a.yml",
            "icie_cal_full_l1a.yml",
            "icie_cal_sample_l1a.yml",
            "icie_crit_hk_l1a.yml",
            "icie_nom_hk_l1a.yml",
            "icie_rad_full_l1a.yml",
            "icie_rad_sample_l1a.yml",
            "icie_temp_hk_l1a.yml",
            "icie_wfov_sci_l1a.yml",
            "jpss_sc_pos_l1a.yml",
            "pec_sw_stat_l1a.yml",
            "pev_sw_stat_l1a.yml",
        ],
    )
    def test_every_l1a_product_definition_declares_the_vocabulary(self, definition_name):
        """A trending query must not need per-APID special cases.

        ``enforce_dataset_conformance`` also deletes global attributes a definition does not
        declare, so an undeclared counter would be silently dropped on write.
        """
        definitions_dir = Path(str(config.get("LIBERA_PRODUCT_DEFINITIONS_PATH")))
        definition = yaml.safe_load((definitions_dir / definition_name).read_text())
        missing = [name for name in QUALITY_GLOBAL_ATTRIBUTES if name not in definition["attributes"]]
        assert not missing, f"{definition_name} does not declare {missing}"


class TestQaReport:
    def test_round_trips_through_json(self, tmp_path):
        record = GranuleQualityRecord(
            product_id="RAD-SAMPLE-DECODED",
            apid=1036,
            n_packets=28_007,
            n_samples=1_400_350,
            duplicate_sample_time_count=10_206,
            duplicate_value_mismatch_count=10_206,
            mismatched_variables=["ICIE__RAD_SAMPLE_1", "ICIE__RAD_SAMPLE_3"],
            duplicate_evidence=[
                DuplicateEvidence(
                    coordinate="RAD_SAMPLE_FPE_TIME",
                    timestamp="2026-07-13T08:12:41.182467",
                    multiplicity=2,
                    row_indices=[163752, 163950],
                    variables={"ICIE__RAD_SAMPLE_1": [5539.2085, 5540.4404]},
                )
            ],
        )
        path = tmp_path / "report.qa.json"
        record.write_qa_report(path)
        loaded = json.loads(path.read_text())

        assert loaded["quality_flag"] == "DEGRADED"
        assert loaded["duplicate_value_mismatch_count"] == 10_206
        assert loaded["mismatched_variables"] == ["ICIE__RAD_SAMPLE_1", "ICIE__RAD_SAMPLE_3"]
        assert loaded["duplicate_evidence"][0]["multiplicity"] == 2
        assert loaded["duplicate_evidence"][0]["variables"]["ICIE__RAD_SAMPLE_1"] == [5539.2085, 5540.4404]

    def test_clean_granule_still_produces_a_record(self, tmp_path):
        """A granule with zero events must say zero, not go unreported."""
        path = tmp_path / "clean.qa.json"
        GranuleQualityRecord(product_id="RAD-SAMPLE-DECODED", n_packets=28_495).write_qa_report(path)
        loaded = json.loads(path.read_text())
        assert loaded["quality_flag"] == "NOMINAL"
        assert loaded["duplicate_sample_time_count"] == 0
        assert loaded["duplicate_evidence"] == []

    def test_has_events_distinguishes_clean_from_dirty(self):
        assert not GranuleQualityRecord(n_packets=100, n_samples=100).has_events()
        assert GranuleQualityRecord(n_packets=100, packet_time_inversion_count=1).has_events()

    def test_sidecar_sits_beside_the_product(self):
        product = LiberaDataProductFilename(_PRODUCT)
        assert product.qa_report_filename.name.endswith(".qa.json")
        assert product.qa_report_filename.name.rsplit(".")[0] == product.path.name.rsplit(".")[0]
        assert product.qa_report_filename.parent == product.path.parent


class TestConformanceOfAParsedGranule:
    """The counters must survive the write, including when every one of them is zero.

    ``enforce_dataset_conformance`` deletes global attributes a product definition does not
    declare and strict validation fails on declared attributes the Dataset does not carry, so a
    counter can only be written if the two agree. A clean granule is the case that historically
    slips through, because a zero is easy to leave unwritten.
    """

    @staticmethod
    def _parse(packet_file, apid, record):
        from libera_utils.l1a.packets import parse_packets_to_l1a_dataset

        return parse_packets_to_l1a_dataset(
            [str(packet_file)],
            apid,
            ground_data=True,
            skip_header_bytes=8,
            quality_record=record,
        )

    def test_parsed_granule_carries_every_counter(self, test_ccsds_2025_218_18_37_32):
        record = GranuleQualityRecord()
        dataset = self._parse(test_ccsds_2025_218_18_37_32, 1057, record)
        for name in QUALITY_GLOBAL_ATTRIBUTES:
            assert name in dataset.attrs, f"{name} missing from parsed granule attributes"
        assert dataset.attrs["QualityFlag"] in {"NOMINAL", "DEGRADED", "SUSPECT"}
        assert record.n_packets == dataset.sizes["PACKET"]
        assert record.apid == 1057

    def test_clean_granule_passes_strict_conformance(self, test_ccsds_2025_218_18_37_32, tmp_path):
        from libera_utils.io.netcdf import write_libera_data_product
        from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config

        record = GranuleQualityRecord()
        dataset = self._parse(test_ccsds_2025_218_18_37_32, 1057, record)
        assert record.duplicate_value_mismatch_count == 0
        assert dataset.attrs["QualityFlag"] == "NOMINAL"

        written = write_libera_data_product(
            get_l1a_product_definition_path(1057),
            dataset,
            tmp_path,
            time_variable=get_packet_config(1057).packet_time_coordinate,
            strict=True,
        )
        import xarray as xr

        with xr.open_dataset(written.path) as round_tripped:
            for name in QUALITY_GLOBAL_ATTRIBUTES:
                assert name in round_tripped.attrs, f"{name} did not survive the NetCDF write"
            assert round_tripped.attrs["QualityFlag"] == "NOMINAL"
