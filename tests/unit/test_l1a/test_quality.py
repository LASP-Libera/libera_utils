"""Unit tests for the per-granule L1A quality record and its flag rollup."""

from pathlib import Path

import pytest
import yaml

from libera_utils.config import config
from libera_utils.l1a.quality import (
    QUALITY_GLOBAL_ATTRIBUTES,
    GranuleQualityRecord,
    QualityFlag,
    QualityThresholds,
)


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


class TestConformanceOfAParsedGranule:
    """The counters must survive the write, including when every one of them is zero.

    ``enforce_dataset_conformance`` deletes global attributes a product definition does not
    declare and strict validation fails on declared attributes the Dataset does not carry, so a
    counter can only be written if the two agree. A clean granule is the case that historically
    slips through, because a zero is easy to leave unwritten.
    """

    @staticmethod
    def _parse(packet_file, apid):
        from libera_utils.l1a.packets import parse_packets_to_l1a_dataset

        return parse_packets_to_l1a_dataset(
            [str(packet_file)],
            apid,
            ground_data=True,
            skip_header_bytes=8,
        )

    def test_parsed_granule_carries_every_counter(self, test_ccsds_2025_218_18_37_32):
        dataset = self._parse(test_ccsds_2025_218_18_37_32, 1057)
        for name in QUALITY_GLOBAL_ATTRIBUTES:
            assert name in dataset.attrs, f"{name} missing from parsed granule attributes"
        assert dataset.attrs["QualityFlag"] in {"NOMINAL", "DEGRADED", "SUSPECT"}

    def test_clean_granule_passes_strict_conformance(self, test_ccsds_2025_218_18_37_32, tmp_path):
        from libera_utils.io.netcdf import write_libera_data_product
        from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config

        dataset = self._parse(test_ccsds_2025_218_18_37_32, 1057)
        assert dataset.attrs["DuplicateValueMismatchCount"] == 0
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
