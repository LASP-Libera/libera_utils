"""Tests for database module"""
# Standard
import multiprocessing
import os
# Installed
import pytest
# Local
from libera_sdp.db import getdb
from libera_sdp.db.database import DatabaseException
from libera_sdp.db.models import L0


def db_testfunc(filename: str = None, version: int = None):
    """DB test function that retrieves some dummy data"""
    db = getdb()
    print(f"Hello from pid {os.getpid()}. db.pid={db.pid}. db.url={db.url}")
    with db.session() as s:
        records = s.query(L0)
        if filename is not None:
            records = records.filter(L0.filename == filename)
        if version is not None:
            records = records.filter(L0.version == version)
        return records.all()


def test_db_connection(insert_test_data, clean_sdp_test_db):
    """Tests ability to connect to the test DB without doing anything special
    This ensures that the pytest fixture that sets the test DB is in place"""
    insert_test_data(L0(filename='foofile.txt', version=1))
    with getdb().session() as s:
        records = s.query(L0).all()
    assert len(records) == 1
    assert records[0].filename == 'foofile.txt'
    assert records[0].version == 1


def test_db_manager_multiprocessing(insert_test_data, clean_sdp_test_db):
    """Test that the module level DB manager object works in multiprocessing"""
    insert_test_data(
        L0(filename='foofile.txt', version=1),
        L0(filename='barfile.txt', version=1),
        L0(filename='bazfile.txt', version=1),
        L0(filename='zipfile.txt', version=0),
        L0(filename='zapfile.txt', version=0),
        L0(filename='zopfile.txt', version=0)
    )

    for method in ('spawn', 'fork'):
        with multiprocessing.get_context(method).Pool(4) as pool:
            res = pool.starmap(db_testfunc,
                               [
                                   ('foofile.txt', 1),
                                   ('zipfile.txt', None),
                                   ('bazfile.txt', None),
                                   (None, 0),
                                   (None, 1),
                                   ('barfile.txt', 1),
                                   ('barfile.txt', 0)
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
