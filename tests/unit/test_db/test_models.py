"""Tests for the models module"""
# Standard
from datetime import datetime, timedelta
import pytz
# Installed
import pytest
# Local
from libera_utils.db import getdb
from libera_utils.db.models import APID, Packet, L0, L1b


@pytest.fixture
def insert_dummy_data(insert_test_data):
    """Inserts a set of dummy data into the DB"""
    records = []

    # Create test APID records
    apid = APID(apid=1111, description='Dummy APID for testing.')
    records.append(apid)

    # Create test packet records
    npkts = 5
    packets = []
    for seq_ctr in range(npkts):
        if seq_ctr == 0:
            seq_flgs = '01'
        elif seq_ctr == npkts - 1:
            seq_flgs = '10'
        else:
            seq_flgs = '00'
        packets.append(
            Packet(version_number=0, type=0, secondary_header_flag=False, apid=apid.apid,
                   sequence_flags=seq_flgs, sequence_count=1, data_length=3,
                   secondary_header=None, user_data=b'ABC123')
        )
    records += packets

    # Create test L0 records
    l0_v0 = L0(filename='libera_test_l0_v0.pkts', version=0, packets=packets[0:3],
               # Custom created time in UTC
               created=pytz.utc.localize(datetime.fromisoformat('2020-01-01T12:01:01')),
               # Custom ingested time in Mountain time
               ingested=pytz.timezone('America/Denver').localize(datetime.fromisoformat('2021-01-01T05:01:01')))
    l0_v1 = L0(filename='libera_test_l0_v1.pkts', version=1, packets=packets)
    records += [l0_v0, l0_v1]

    # Create test L1b records
    l1b_1 = L1b(filename='libera_test_l1b_1_v0.h5', version=0, packets=packets[0:2])
    l1b_2 = L1b(filename='libera_test_l1b_2_v0.h5', version=0, packets=packets[2:])
    l1b_1_v1 = L1b(filename='libera_test_l1b_1_v1.h5', version=1, packets=packets[0:2])
    l1b_2_v1 = L1b(filename='libera_test_l1b_2_v1.h5', version=1, packets=packets[2:])
    records += [l1b_1, l1b_2, l1b_1_v1, l1b_2_v1]

    insert_test_data(*records)


@pytest.mark.usefixtures('insert_dummy_data')  # Scoped to entire class
class TestModels:
    """Test class that tests general functionality of all models.
    Uses a specific set of dummy test data provided by the insert_dummy_data fixture."""

    def test_l0(self):
        """Test that proves general functionality of the L0 object"""
        with getdb().session() as s:
            all_l0 = s.query(L0).all()
            assert len(all_l0) == 2

            v0_l0 = s.query(L0).filter(L0.version == 0).all()
            assert len(v0_l0) == 1
            assert v0_l0[0].filename == 'libera_test_l0_v0.pkts'
            assert v0_l0[0].created == pytz.utc.localize(datetime.fromisoformat('2020-01-01T12:01:01'))
            assert v0_l0[0].ingested == pytz.utc.localize(datetime.fromisoformat('2021-01-01T12:01:01'))

            v1_l0 = s.query(L0).filter(L0.version == 1).all()
            assert len(v1_l0) == 1
            assert v1_l0[0].filename == 'libera_test_l0_v1.pkts'
            assert v1_l0[0].created - pytz.utc.localize(datetime.utcnow()) < timedelta(minutes=10)
            assert v1_l0[0].ingested is None

    def test_l1b(self):
        """Test that proves general functionality of the L1b object"""
        with getdb().session() as s:
            all_l1b = s.query(L1b).all()
            assert len(all_l1b) == 4

            v1_l1b = s.query(L1b).filter(L1b.version == 1).all()
            assert len(v1_l1b) == 2

            specific_l1b = s.query(L1b).filter(L1b.filename == 'libera_test_l1b_1_v1.h5').all()
            assert len(specific_l1b) == 1
            assert specific_l1b[0].version == 1
            assert specific_l1b[0].filename == 'libera_test_l1b_1_v1.h5'

    def test_packet(self):
        """Test that proves general functionality of the Packet object"""
        with getdb().session() as s:
            all_packets = s.query(Packet).all()
            assert len(all_packets) == 5
            for p in all_packets:
                assert p.apid == 1111
                assert p.user_data == b'ABC123'

    def test_packet_joins(self):
        """Test that proves joining functionality between ORM objects"""
        with getdb().session() as s:
            l0_v0 = s.query(L0).filter(L0.filename == 'libera_test_l0_v0.pkts').all()[0]
            assert len(l0_v0.packets) == 3

            l1b_1_v0 = s.query(L1b).filter(L1b.filename == 'libera_test_l1b_1_v0.h5').all()[0]
            assert len(l1b_1_v0.packets) == 2


@pytest.mark.usefixtures('insert_dummy_data')  # Scoped to entire class
class TestDataProductMixin:
    """Test class that tests the methods provided by the DataProductMixin class"""

    def test_latest(self):
        """Test getting latest products"""
        all_latest = L1b.latest()
