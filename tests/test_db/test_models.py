"""Tests for the models module"""
# Local
from libera_sdp.db import getdb
from libera_sdp.db.models import *


def test_level0(clean_sdp_test_db):
    """Test creation of an L0 record object"""
    l0 = Level0(filename='foofile.txt')
    db = getdb()
    with db.session() as s:
        s.add(l0)

    with db.session() as s:
        records = s.query(Level0).all()

    assert len(records) == 1
    assert records[0].filename == 'foofile.txt'
