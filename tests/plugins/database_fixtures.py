"""Pytest plugin module for database-related fixtures"""
# pylint: disable=redefined-outer-name
# Standard
from datetime import datetime
import pytz
# Installed
import pytest
# Local
from libera_sdp.db import getdb
from libera_sdp.db.models import L0, APID, Packet, L1b


@pytest.fixture(scope='session', autouse=True)
def use_test_db(monkeypatch_session):
    """Automatically set environment variables to use the testing database"""
    monkeypatch_session.setenv('LIBERA_DB_NAME', 'sdp_test')
    monkeypatch_session.setenv('LIBERA_DB_USER', 'libera_unit_tester')
    monkeypatch_session.setenv('PGPASSWORD', 'testerpass')


@pytest.fixture
def clean_sdp_test_db():
    """Ensure the TEST database is in a clean state and return it to clean after we're done testing.
    Note: If this weren't such a costly operation, we would set this to autouse=True
    """
    db = getdb()
    db.truncate_product_tables()
    yield
    db.truncate_product_tables()


@pytest.fixture
def insert_test_data(clean_sdp_test_db):
    """Provides a function to insert test data records into the DB.
    Clears out the DB before and after the test."""

    def _insert(*record_objects):
        with getdb().session() as s:
            for record in record_objects:
                s.add(record)

    yield _insert


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
                   secondary_header=None, user_data=b'XYZW')
        )
    records += packets

    # Create test L0 records
    l0 = L0(filename='libera_test_l0_v0.pkts', version=0, packets=packets[0:3],
            created=datetime.fromisoformat('1960-07-20T20:17:40+00:00'),
            ingested=datetime.fromisoformat('1969-07-20T20:17:40+00:00'))
    # TODO: Figure out how to pass time zone info to datetime and save it correctly in the DB
    l0_v1 = L0(filename='libera_test_l0_v1.pkts', version=1, packets=packets,
               created=datetime.fromisoformat('2000-07-20T20:17:40+00:00'))
    records += [l0, l0_v1]

    # Create test L1b records
    l1b_1 = L1b(filename='libera_test_l1b_1_v0.h5', version=0, packets=packets[0:2])
    l1b_2 = L1b(filename='libera_test_l1b_2_v0.h5', version=0, packets=packets[2:])
    l1b_1_v1 = L1b(filename='libera_test_l1b_1_v1.h5', version=1, packets=packets[0:2])
    l1b_2_v1 = L1b(filename='libera_test_l1b_2_v1.h5', version=1, packets=packets[2:])
    records += [l1b_1, l1b_2, l1b_1_v1, l1b_2_v1]

    insert_test_data(*records)
