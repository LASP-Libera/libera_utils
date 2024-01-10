"""Tests for the models module"""
# Standard
from datetime import datetime, timedelta, timezone
# Installed
import pytest
# Local
from libera_utils.db import getdb
from libera_utils.db.models import *


@pytest.fixture
def insert_dummy_l0_relations(clean_local_db):
    """
    Insert test ORM objects for managing L0 data:

    Cr
    --> CrSscStartStops
    --> CrApid
        --> CrApidVcid
        --> CrApidSscGap
        --> CrApidEdosFillData
        --> CrApidSscLengthDiscrepancies
    --> PdsFile
        --> PdsFileApid
    """
    # Create some datetime objects with time zone info
    # Absent this tzinfo, the DB will assume that "naive" datetime objects are in whatever time zone Postgres
    # server/database/session is configured with (i.e. the output of `show timezone;`).
    # It's a reasonable assumption that our DB will be in UTC time (anything else is pure madness).
    t0 = datetime.fromisoformat("2023-01-01T11:22:33.123456").replace(tzinfo=timezone.utc)
    t1 = datetime.fromisoformat("2023-01-01T11:25:33.123456").replace(tzinfo=timezone.utc)
    pds_file_apid = PdsFileApid(
        scid=8473,
        apid=9372,
        first_packet_sc_time=0,
        last_packet_sc_time=(2**64)-1,
        first_packet_utc_time=t0,
        last_packet_utc_time=t1
    )
    pds = PdsFile(
        file_name='P2041834AAAAAAAAAAAAAA19218140452201.PDS',
        apids=[pds_file_apid]
    )

    vcid = CrApidVcid(
        scid_vcid=(2**16)-1
    )
    edos_g = CrApidEdosGeneratedFillData(
        ssc_with_generated_data=1,
        filled_byte_offset=3,
        index_to_fill_octet=5
    )
    len_discrep = CrApidSscLenDiscrepancies(
        ssc_length_discrepancy=0
    )
    scs_start_stop = CrScsStartStopTimes(
        scs_start_sc_time=0,
        scs_stop_sc_time=(2 ** 64) - 1,
        scs_start_utc_time=t0,
        scs_stop_utc_time=t1
    )
    gap = CrApidSscGap(
        first_missing_ssc=(2**16)-1,
        gap_byte_offset=(2**64)-1,
        n_missing_sscs=3,
        preceding_packet_sc_time=0,
        following_packet_sc_time=(2**64)-1,
        preceding_packet_utc_time=t0,
        following_packet_utc_time=t1,
        preceding_packet_esh_time=0,
        following_packet_esh_time=(2**64)-1
    )
    cr_apid = CrApid(
        scid=999,
        apid=999,
        byte_offset=3,
        n_vcids=1,
        vcids=[vcid],
        n_ssc_gaps=1,
        ssc_gaps=[gap],
        n_edos_generated_fill_data=1,
        edos_fill_data=[edos_g],
        count_edos_generated_octets=1,
        n_length_discrepancy_packets=1,
        ssc_length_discrepancies=[len_discrep],
        first_packet_sc_time=0,
        last_packet_sc_time=(2 ** 64) - 1,
        esh_first_packet_time=0,
        esh_last_packet_time=(2 ** 64) - 1,
        first_packet_utc_time=t0,
        last_packet_utc_time=t1,
        n_vcdu_corrected_packets=0,
        n_in_the_data_set=16,
        n_octect_in_apid=2
    )
    cr = Cr(
        file_name="P2041834AAAAAAAAAAAAAA19218140452200.PDS",
        edos_software_version=3,
        construction_record_type=1,
        test_flag=False,
        n_scs_start_stops=1,
        scs_start_stop_times=[scs_start_stop],
        n_bytes_fill_data=0,
        n_length_mismatches=0,
        first_packet_sc_time=0,
        last_packet_sc_time=(2**64)-1,
        first_packet_utc_time=t0,
        last_packet_utc_time=t1,
        first_packet_esh_time=1,
        last_packet_esh_time=(2**64)-2,
        n_rs_corrections=10,
        n_packets=1000000,
        size_bytes=42,
        n_ssc_discontinuities=0,
        completion_time=(2**64)-1,
        n_apids=1,
        n_pds_files=1,
        apids=[cr_apid],
        pds_files=[pds]
    )
    with getdb().session() as s:
        s.add(cr)


@pytest.mark.usefixtures('insert_dummy_l0_relations')  # Scoped to entire class
class TestL0Models:
    """Test class that tests general functionality of all models.
    Uses a specific set of dummy test data provided by the insert_dummy_data fixture."""

    def test_cr(self):
        """ Test that proves general functionality of the Cr object """

        with getdb().session() as s:
            all_cr = s.query(Cr).all()
            assert len(all_cr) == 1
            cr = all_cr[0]
            assert len(cr.scs_start_stop_times) == 1
            assert len(cr.pds_files) == 1
            assert len(cr.pds_files[0].apids) == 1
            assert len(cr.apids) == 1
            assert len(cr.apids[0].vcids) == 1
            assert len(cr.apids[0].ssc_gaps) == 1
            assert len(cr.apids[0].ssc_length_discrepancies) == 1
            assert len(cr.apids[0].edos_fill_data) == 1
            assert cr.apids[0].scid == 999
            assert cr.apids[0].apid == 999

    def test_pds_file(self):
        with getdb().session() as s:
            all_pds = s.query(PdsFile).all()
            assert len(all_pds) == 1
            assert hasattr(all_pds[0], 'construction_record')
            assert all_pds[0].construction_record is not None
            assert all_pds[0].apids[0].apid == 9372
            assert all_pds[0].apids[0].scid == 8473



# TODO: Test ability to retrieve latest products
# @pytest.mark.usefixtures('insert_dummy_data')  # Scoped to entire class
# class TestDataProductMixin:
#     """Test class that tests the methods provided by the DataProductMixin class"""
#
#     def test_latest(self):
#         """Test getting latest products"""
#         all_latest = L1b.latest()
