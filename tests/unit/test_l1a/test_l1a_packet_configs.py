"""Unit tests for packet configurations for L1A processing"""

from datetime import timedelta

import numpy as np
import pytest

from libera_utils.constants import LiberaApid
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a import l1a_packet_configs
from libera_utils.l1a.l1a_packet_configs import (
    SampleTimeSource,
    get_l1a_product_definition_path,
    get_packet_config,
)


class TestPacketConfiguration:
    """Test the PacketConfiguration class mechanics — construction, validators, and properties."""

    def test_packet_configuration_creation(self):
        """Test that PacketConfiguration can be instantiated with required fields."""
        config = l1a_packet_configs.PacketConfiguration(
            packet_apid=LiberaApid.icie_axis_sample,
            packet_time_fields=l1a_packet_configs.TimeFieldMapping(
                day_field="TM_DAY",
                ms_field="TM_MS",
                us_field="TM_US",
            ),
            sample_groups=[
                l1a_packet_configs.SampleGroup(
                    name="TEST_SAMPLES",
                    time_field_patterns=l1a_packet_configs.TimeFieldMapping(
                        s_field="SAMPLE_TIME_SEC%i",
                        us_field="SAMPLE_TIME_USEC%i",
                    ),
                    data_field_patterns=["DATA_FIELD_%i", "OTHER_FIELD_%i"],
                    sample_count=10,
                    time_source=l1a_packet_configs.SampleTimeSource.ICIE,
                )
            ],
        )

        assert config.packet_apid == LiberaApid.icie_axis_sample
        assert len(config.packet_time_fields.multipart_kwargs) == 3
        assert len(config.sample_groups) == 1
        assert config.sample_groups[0].name == "TEST_SAMPLES"
        assert config.sample_groups[0].sample_count == 10
        assert config.sample_groups[0].time_source == l1a_packet_configs.SampleTimeSource.ICIE

    def test_sample_group_dimension_name(self):
        """Test that the sample_time_dimension property produces the correct name for each time source."""
        group_icie = l1a_packet_configs.SampleGroup(
            name="TEST_GROUP",
            sample_count=1,
            data_field_patterns=["TEST_DATA"],
            time_field_patterns=l1a_packet_configs.TimeFieldMapping(s_field="TIME_SEC"),
            time_source=l1a_packet_configs.SampleTimeSource.ICIE,
        )
        assert group_icie.sample_time_dimension == "TEST_GROUP_ICIE_TIME"

        group_fpe = l1a_packet_configs.SampleGroup(
            name="RAD_SAMPLE",
            sample_count=10,
            data_field_patterns=["RAD_DATA"],
            epoch_time_fields=l1a_packet_configs.TimeFieldMapping(s_field="START_SEC"),
            sample_period=timedelta(milliseconds=5),
            time_source=l1a_packet_configs.SampleTimeSource.FPE,
        )
        assert group_fpe.sample_time_dimension == "RAD_SAMPLE_FPE_TIME"

        group_jpss = l1a_packet_configs.SampleGroup(
            name="ADGPS",
            sample_count=1,
            data_field_patterns=["GPS_DATA"],
            time_field_patterns=l1a_packet_configs.TimeFieldMapping(us_field="TIME"),
            time_source=l1a_packet_configs.SampleTimeSource.JPSS,
        )
        assert group_jpss.sample_time_dimension == "ADGPS_JPSS_TIME"

    def test_packet_time_coordinate_property(self):
        """Test that packet_time_coordinate derives its name from packet_time_source."""
        config_icie = l1a_packet_configs.PacketConfiguration(
            packet_apid=LiberaApid.icie_nom_hk,
            packet_time_fields=l1a_packet_configs.TimeFieldMapping(day_field="DAY"),
            packet_time_source=l1a_packet_configs.SampleTimeSource.ICIE,
        )
        assert config_icie.packet_time_coordinate == "PACKET_ICIE_TIME"

        config_jpss = l1a_packet_configs.PacketConfiguration(
            packet_apid=LiberaApid.jpss_sc_pos,
            packet_time_fields=l1a_packet_configs.TimeFieldMapping(day_field="DAY"),
            packet_time_source=l1a_packet_configs.SampleTimeSource.JPSS,
        )
        assert config_jpss.packet_time_coordinate == "PACKET_JPSS_TIME"


class TestL1aConfigYaml:
    """Regression tests for the L1A packet processing YAML configurations.

    Each test covers a single APID config to catch accidental changes to field names,
    sample counts, time sources, or packet definition keys.
    """

    def test_pev_sw_stat_config(self):
        """APID 1000 — PEV software status, housekeeping-style, no sample groups."""
        cfg = get_packet_config(LiberaApid.pev_sw_stat)
        assert cfg.packet_apid == LiberaApid.pev_sw_stat
        assert cfg.packet_time_fields.day_field == "PEV__TM_DAY_SW_STAT"
        assert cfg.packet_time_fields.ms_field == "PEV__TM_MS_SW_STAT"
        assert cfg.packet_time_fields.us_field == "PEV__TM_US_SW_STAT"
        assert cfg.packet_definition_config_key == "LIBERA_PEV_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert cfg.sample_groups == []
        assert cfg.aggregation_groups == []

    def test_pec_sw_stat_config(self):
        """APID 1002 — PEC software status, housekeeping-style, no sample groups."""
        cfg = get_packet_config(LiberaApid.pec_sw_stat)
        assert cfg.packet_apid == LiberaApid.pec_sw_stat
        assert cfg.packet_time_fields.day_field == "PEC__TM_DAY_SW_STAT"
        assert cfg.packet_time_fields.ms_field == "PEC__TM_MS_SW_STAT"
        assert cfg.packet_time_fields.us_field == "PEC__TM_US_SW_STAT"
        assert cfg.packet_definition_config_key == "LIBERA_PEC_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert cfg.sample_groups == []
        assert cfg.aggregation_groups == []

    def test_icie_axis_sample_config(self):
        """APID 1048 — azimuth/elevation encoder samples, 50 samples per packet, ICIE timestamps."""
        cfg = get_packet_config(LiberaApid.icie_axis_sample)
        assert cfg.packet_apid == LiberaApid.icie_axis_sample
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_AXIS_SAMPLE"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_AXIS_SAMPLE"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_AXIS_SAMPLE"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert len(cfg.sample_groups) == 1
        grp = cfg.sample_groups[0]
        assert grp.name == "AXIS_SAMPLE"
        assert grp.sample_count == 50
        assert grp.time_source == SampleTimeSource.ICIE
        assert grp.sample_time_dimension == "AXIS_SAMPLE_ICIE_TIME"
        assert grp.time_field_patterns is not None
        assert grp.time_field_patterns.s_field == "ICIE__AXIS_SAMPLE_TM_SEC%i"
        assert grp.time_field_patterns.us_field == "ICIE__AXIS_SAMPLE_TM_SUB%i"

    def test_icie_rad_sample_config(self):
        """APID 1036 — radiometer 200 Hz samples, FPE epoch+period timestamps."""
        cfg = get_packet_config(LiberaApid.icie_rad_sample)
        assert cfg.packet_apid == LiberaApid.icie_rad_sample
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_RAD_SAMPLE"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_RAD_SAMPLE"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_RAD_SAMPLE"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert len(cfg.sample_groups) == 1
        grp = cfg.sample_groups[0]
        assert grp.name == "RAD_SAMPLE"
        assert grp.sample_count == 50
        assert grp.time_source == SampleTimeSource.FPE
        assert grp.sample_period == timedelta(microseconds=5000)
        assert grp.sample_time_dimension == "RAD_SAMPLE_FPE_TIME"
        assert grp.epoch_time_fields is not None
        assert grp.epoch_time_fields.s_field == "ICIE__RAD_SAMP_START_HI"
        assert grp.epoch_time_fields.us_field == "ICIE__RAD_SAMP_START_LO"

    def test_icie_rad_full_config(self):
        """APID 1035 — radiometer 1 kHz full-resolution samples, FPE epoch+period timestamps."""
        cfg = get_packet_config(LiberaApid.icie_rad_full)
        assert cfg.packet_apid == LiberaApid.icie_rad_full
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_RAD_FULL"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_RAD_FULL"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_RAD_FULL"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert len(cfg.sample_groups) == 1
        grp = cfg.sample_groups[0]
        assert grp.name == "RAD_FULL"
        assert grp.sample_count == 100
        assert grp.time_source == SampleTimeSource.FPE
        assert grp.sample_period == timedelta(microseconds=1000)
        assert grp.sample_time_dimension == "RAD_FULL_FPE_TIME"
        assert grp.epoch_time_fields is not None
        assert grp.epoch_time_fields.s_field == "ICIE__RAD_FULL_START_HI"
        assert grp.epoch_time_fields.us_field == "ICIE__RAD_FULL_START_LO"

    def test_icie_cal_full_config(self):
        """APID 1043 — calibration 1 kHz full-resolution samples, FPE epoch+period timestamps."""
        cfg = get_packet_config(LiberaApid.icie_cal_full)
        assert cfg.packet_apid == LiberaApid.icie_cal_full
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_CAL_FULL"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_CAL_FULL"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_CAL_FULL"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert len(cfg.sample_groups) == 1
        grp = cfg.sample_groups[0]
        assert grp.name == "CAL_FULL"
        assert grp.sample_count == 100
        assert grp.time_source == SampleTimeSource.FPE
        assert grp.sample_period == timedelta(microseconds=1000)
        assert grp.sample_time_dimension == "CAL_FULL_FPE_TIME"
        assert grp.epoch_time_fields is not None
        assert grp.epoch_time_fields.s_field == "ICIE__CAL_FULL_START_HI"
        assert grp.epoch_time_fields.us_field == "ICIE__CAL_FULL_START_LO"

    def test_icie_cal_sample_config(self):
        """APID 1044 — calibration 200 Hz downsampled, FPE epoch+period timestamps."""
        cfg = get_packet_config(LiberaApid.icie_cal_sample)
        assert cfg.packet_apid == LiberaApid.icie_cal_sample
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_CAL_SAMPLE"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_CAL_SAMPLE"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_CAL_SAMPLE"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert len(cfg.sample_groups) == 1
        grp = cfg.sample_groups[0]
        assert grp.name == "CAL_SAMPLE"
        assert grp.sample_count == 50
        assert grp.time_source == SampleTimeSource.FPE
        assert grp.sample_period == timedelta(microseconds=5000)
        assert grp.sample_time_dimension == "CAL_SAMPLE_FPE_TIME"
        assert grp.epoch_time_fields is not None
        assert grp.epoch_time_fields.s_field == "ICIE__CAL_SAMP_START_HI"
        assert grp.epoch_time_fields.us_field == "ICIE__CAL_SAMP_START_LO"

    def test_icie_wfov_sci_config(self):
        """APID 1040 — WFOV camera science data as a single aggregated binary blob per packet."""
        cfg = get_packet_config(LiberaApid.icie_wfov_sci)
        assert cfg.packet_apid == LiberaApid.icie_wfov_sci
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_WFOV_SCI"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_WFOV_SCI"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_WFOV_SCI"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert cfg.sample_groups == []
        # ICIE__WFOV_DATA is now parsed directly as a binary field by SPP (via BinaryParameterType in XTCE),
        # so no aggregation_groups are needed.
        assert cfg.aggregation_groups == []

    def test_icie_nom_hk_config(self):
        """APID 1057 — nominal housekeeping, no sample groups."""
        cfg = get_packet_config(LiberaApid.icie_nom_hk)
        assert cfg.packet_apid == LiberaApid.icie_nom_hk
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_NOM_HK"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_NOM_HK"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_NOM_HK"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert cfg.sample_groups == []
        assert cfg.aggregation_groups == []

    def test_icie_crit_hk_config(self):
        """APID 1051 — critical housekeeping, no sample groups."""
        cfg = get_packet_config(LiberaApid.icie_crit_hk)
        assert cfg.packet_apid == LiberaApid.icie_crit_hk
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_CRIT_HK"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_CRIT_HK"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_CRIT_HK"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert cfg.sample_groups == []
        assert cfg.aggregation_groups == []

    def test_icie_temp_hk_config(self):
        """APID 1060 — temperature housekeeping, no sample groups."""
        cfg = get_packet_config(LiberaApid.icie_temp_hk)
        assert cfg.packet_apid == LiberaApid.icie_temp_hk
        assert cfg.packet_time_fields.day_field == "ICIE__TM_DAY_TEMP_HK"
        assert cfg.packet_time_fields.ms_field == "ICIE__TM_MS_TEMP_HK"
        assert cfg.packet_time_fields.us_field == "ICIE__TM_US_TEMP_HK"
        assert cfg.packet_definition_config_key == "LIBERA_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_ICIE_TIME"
        assert cfg.sample_groups == []
        assert cfg.aggregation_groups == []

    def test_jpss_sc_pos_config(self):
        """APID 11 — JPSS spacecraft position/attitude, two JPSS-timestamped sample groups."""
        cfg = get_packet_config(LiberaApid.jpss_sc_pos)
        assert cfg.packet_apid == LiberaApid.jpss_sc_pos
        assert cfg.packet_time_fields.day_field == "DAYS"
        assert cfg.packet_time_fields.ms_field == "MSEC"
        assert cfg.packet_time_fields.us_field == "USEC"
        assert cfg.packet_definition_config_key == "JPSS_GEOLOCATION_PACKET_DEFINITION"
        assert cfg.packet_time_coordinate == "PACKET_JPSS_TIME"
        assert len(cfg.sample_groups) == 2

        adgps = cfg.sample_groups[0]
        assert adgps.name == "ADGPS"
        assert adgps.sample_count == 1
        assert adgps.time_source == SampleTimeSource.JPSS
        assert adgps.sample_time_dimension == "ADGPS_JPSS_TIME"
        assert adgps.time_field_patterns is not None
        assert adgps.time_field_patterns.day_field == "ADAET1DAY"
        assert adgps.time_field_patterns.ms_field == "ADAET1MS"
        assert adgps.time_field_patterns.us_field == "ADAET1US"

        adcfa = cfg.sample_groups[1]
        assert adcfa.name == "ADCFA"
        assert adcfa.sample_count == 1
        assert adcfa.time_source == SampleTimeSource.JPSS
        assert adcfa.sample_time_dimension == "ADCFA_JPSS_TIME"
        assert adcfa.time_field_patterns is not None
        assert adcfa.time_field_patterns.day_field == "ADAET2DAY"
        assert adcfa.time_field_patterns.ms_field == "ADAET2MS"
        assert adcfa.time_field_patterns.us_field == "ADAET2US"


@pytest.mark.parametrize(
    "apid",
    [
        LiberaApid.pev_sw_stat,
        LiberaApid.pec_sw_stat,
        LiberaApid.icie_axis_sample,
        LiberaApid.icie_crit_hk,
        LiberaApid.icie_nom_hk,
        LiberaApid.icie_temp_hk,
        LiberaApid.icie_wfov_sci,
        LiberaApid.icie_rad_sample,
        LiberaApid.icie_rad_full,
        LiberaApid.icie_cal_full,
        LiberaApid.icie_cal_sample,
        LiberaApid.jpss_sc_pos,
    ],
)
def test_get_l1a_product_definition_by_apid(apid):
    """Test retrieval of L1A product definitions by APID from config."""
    product_def = get_l1a_product_definition_path(apid)
    # Check that we can create a product definition object from the yaml path
    LiberaDataProductDefinition.from_yaml(product_def)
