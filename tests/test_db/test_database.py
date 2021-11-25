"""Tests for database module"""
# Standard
import multiprocessing
import os
# Installed
import pytest
# Local
from libera_sdp.db import getdb
from libera_sdp.db.database import DatabaseException
from libera_sdp.db.models import *


def db_testfunc(filename: str = None, ingest_complete: bool = None):
    """DB test function that retrieves some dummy data"""
    db = getdb()
    print(f"Hello from pid {os.getpid()}. db.pid={db.pid}. db.url={db.url}")
    with db.session() as s:
        records = s.query(Level0)
        if filename is not None:
            records = records.filter(Level0.filename == filename)
        if ingest_complete is not None:
            records = records.filter(Level0.ingest_complete == ingest_complete)
        return records.all()


def test_db_connection(insert_test_data, clean_sdp_test_db):
    """Tests ability to connect to the test DB without doing anything special
    This ensures that the pytest fixture that sets the test DB is in place"""
    insert_test_data(Level0(filename='foofile.txt', ingest_complete=True))
    with getdb().session() as s:
        records = s.query(Level0).all()
    assert len(records) == 1
    assert records[0].filename == 'foofile.txt'
    assert records[0].ingest_complete is True


def test_db_manager_multiprocessing(insert_test_data, clean_sdp_test_db):
    """Test that the module level DB manager object works in multiprocessing"""
    insert_test_data(
        Level0(filename='foofile.txt', ingest_complete=True),
        Level0(filename='barfile.txt', ingest_complete=True),
        Level0(filename='bazfile.txt', ingest_complete=True),
        Level0(filename='zipfile.txt', ingest_complete=False),
        Level0(filename='zapfile.txt', ingest_complete=False),
        Level0(filename='zopfile.txt', ingest_complete=False)
    )

    for method in ('spawn', 'fork'):
        with multiprocessing.get_context(method).Pool(4) as pool:
            res = pool.starmap(db_testfunc,
                               [
                                   ('foofile.txt', True),
                                   ('zipfile.txt', None),
                                   ('bazfile.txt', None),
                                   (None, False),
                                   (None, True),
                                   ('barfile.txt', True),
                                   ('barfile.txt', False)
                               ])


def test_db_manager_multiple_configs():
    """Test ability to create multiple database managers with different configs and get each predictably"""
    db1 = getdb()
    db2 = getdb(dbname='sdp_dev')
    assert db1 is not db2
    assert getdb() is db1
    assert getdb(dbname='sdp_dev') is db2


def test_db_misconfig(monkeypatch):
    """Test missing configuration"""
    monkeypatch.delenv('LIBERA_DB_NAME')
    with pytest.raises(DatabaseException):
        getdb()
