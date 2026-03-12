"""Unit tests for packet configurations for L1A processing"""

import pytest

from libera_utils.constants import LiberaApid
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.l1a import l1a_packet_configs
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config


class TestPacketConfiguration:
    """Test the PacketConfiguration class and its pre-configured instances."""

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
        """Test that the dimension_name property works correctly for SampleGroup."""
        from datetime import timedelta

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

    def test_predefined_configs(self):
        """Test that predefined configurations are properly structured."""
        # Test AXIS_SAMPLE_CONFIG
        axis_config = get_packet_config(LiberaApid.icie_axis_sample)
        assert axis_config.packet_apid == LiberaApid.icie_axis_sample
        assert len(axis_config.sample_groups) == 1
        assert axis_config.sample_groups[0].name == "AXIS_SAMPLE"
        assert axis_config.sample_groups[0].sample_count == 50

        # Test RAD_SAMPLE_CONFIG
        rad_config = get_packet_config(LiberaApid.icie_rad_sample)
        assert rad_config.packet_apid == LiberaApid.icie_rad_sample
        assert len(rad_config.sample_groups) == 1
        assert rad_config.sample_groups[0].name == "RAD_SAMPLE"
        assert rad_config.sample_groups[0].sample_count == 50

        # Test SC_POS_CONFIG
        sc_pos_config = get_packet_config(LiberaApid.jpss_sc_pos)
        assert sc_pos_config.packet_apid == LiberaApid.jpss_sc_pos
        assert len(sc_pos_config.sample_groups) == 2
        assert sc_pos_config.sample_groups[0].name == "ADGPS"
        assert sc_pos_config.sample_groups[1].name == "ADCFA"

    def test_new_apid_configs(self):
        """Test that the new APID configurations (1035, 1043, 1044) are correctly loaded."""
        from datetime import timedelta

        # APID 1035 - icie_rad_full
        rad_full_config = get_packet_config(LiberaApid.icie_rad_full)
        assert rad_full_config.packet_apid == LiberaApid.icie_rad_full
        assert len(rad_full_config.sample_groups) == 1
        assert rad_full_config.sample_groups[0].name == "RAD_FULL"
        assert rad_full_config.sample_groups[0].sample_count == 100
        assert rad_full_config.sample_groups[0].sample_period == timedelta(microseconds=1000)

        # APID 1043 - icie_cal_full
        cal_full_config = get_packet_config(LiberaApid.icie_cal_full)
        assert cal_full_config.packet_apid == LiberaApid.icie_cal_full
        assert len(cal_full_config.sample_groups) == 1
        assert cal_full_config.sample_groups[0].name == "CAL_FULL"
        assert cal_full_config.sample_groups[0].sample_count == 100
        assert cal_full_config.sample_groups[0].sample_period == timedelta(microseconds=1000)

        # APID 1044 - icie_cal_sample
        cal_sample_config = get_packet_config(LiberaApid.icie_cal_sample)
        assert cal_sample_config.packet_apid == LiberaApid.icie_cal_sample
        assert len(cal_sample_config.sample_groups) == 1
        assert cal_sample_config.sample_groups[0].name == "CAL_SAMPLE"
        assert cal_sample_config.sample_groups[0].sample_count == 50
        assert cal_sample_config.sample_groups[0].sample_period == timedelta(microseconds=5000)

    def test_pev_pec_sw_stat_configs(self):
        """Test that pev_sw_stat (1000) and pec_sw_stat (1002) configurations are correctly loaded."""
        # APID 1000 - pev_sw_stat
        pev_config = get_packet_config(LiberaApid.pev_sw_stat)
        assert pev_config.packet_apid == LiberaApid.pev_sw_stat
        # These are housekeeping-style packets with no sample groups
        assert len(pev_config.sample_groups) == 0
        assert len(pev_config.aggregation_groups) == 0
        # Verify packet time fields are correctly configured
        assert pev_config.packet_time_fields.day_field == "PEV__TM_DAY_SW_STAT"
        assert pev_config.packet_time_fields.ms_field == "PEV__TM_MS_SW_STAT"
        assert pev_config.packet_time_fields.us_field == "PEV__TM_US_SW_STAT"
        # Uses the dedicated PEV packet definition
        assert pev_config.packet_definition_config_key == "LIBERA_PEV_PACKET_DEFINITION"
        # Packet time coordinate should follow standard ICIE naming
        assert pev_config.packet_time_coordinate == "PACKET_ICIE_TIME"

        # APID 1002 - pec_sw_stat
        pec_config = get_packet_config(LiberaApid.pec_sw_stat)
        assert pec_config.packet_apid == LiberaApid.pec_sw_stat
        # These are housekeeping-style packets with no sample groups
        assert len(pec_config.sample_groups) == 0
        assert len(pec_config.aggregation_groups) == 0
        # Verify packet time fields are correctly configured
        assert pec_config.packet_time_fields.day_field == "PEC__TM_DAY_SW_STAT"
        assert pec_config.packet_time_fields.ms_field == "PEC__TM_MS_SW_STAT"
        assert pec_config.packet_time_fields.us_field == "PEC__TM_US_SW_STAT"
        # Uses the dedicated PEC packet definition
        assert pec_config.packet_definition_config_key == "LIBERA_PEC_PACKET_DEFINITION"
        # Packet time coordinate should follow standard ICIE naming
        assert pec_config.packet_time_coordinate == "PACKET_ICIE_TIME"


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
