"""Tests for the models module"""
# Standard
from datetime import datetime
import pytz
# Installed
import pytest
# Local
from libera_sdp.db import getdb
from libera_sdp.db.models import L0, L1b


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
            assert v0_l0[0].created == datetime.fromisoformat('1960-07-20T20:17:40+00:00')
            # TODO: Test timezone info converting between zones
            assert v0_l0[0].ingested == datetime.fromisoformat('1969-07-20T20:17:40+00:00')

            v1_l0 = s.query(L0).filter(L0.version == 1).all()
            assert len(v1_l0) == 1
            assert v1_l0[0].filename == 'libera_test_l0_v1.pkts'
            assert v1_l0[0].created == datetime.fromisoformat('2000-07-20T20:17:40+00:00')
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
        pass
    # TODO: Implement

    def test_packet_joins(self):
        """Test that proves joining functionality between ORM objects"""
        pass
    # TODO: Implement


# TODO: Test timezone data storage with datetime input
# TODO: Test server default behaviors
# TODO: Test model relationships
