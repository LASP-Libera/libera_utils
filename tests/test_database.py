"""Tests for database module"""
# Local
from libera_sdp.db.database import db
from libera_sdp.db.models import *


def test_db_connection(insert_test_data, clean_db):
    """Tests ability to connect to the test DB without doing anything special
    This ensures that the pytest fixture that sets the test DB is in place"""
    insert_test_data(Level0(filename='foofile.txt', ingest_complete=True))
    with db.session() as s:
        records = s.query(Level0).all()

    # TODO: Add some assertions here


def test_db_manager_concurrency(insert_test_data, clean_db):
    """Test that the module level DB manager object works in multiprocessing"""
    # TODO: USe multiprocessing pool to query the test DB for some generic data.
    pass
