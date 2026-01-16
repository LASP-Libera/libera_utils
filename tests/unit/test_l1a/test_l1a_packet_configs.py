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


@pytest.mark.parametrize(
    "apid",
    [
        LiberaApid.icie_axis_sample,
        LiberaApid.icie_crit_hk,
        LiberaApid.icie_nom_hk,
        LiberaApid.icie_temp_hk,
        LiberaApid.icie_wfov_sci,
        LiberaApid.icie_rad_sample,
        LiberaApid.jpss_sc_pos,
    ],
)
def test_get_l1a_product_definition_by_apid(apid):
    """Test retrieval of L1A product definitions by APID from config."""
    product_def = get_l1a_product_definition_path(apid)
    # Check that we can create a product definition object from the yaml path
    LiberaDataProductDefinition.from_yaml(product_def)
