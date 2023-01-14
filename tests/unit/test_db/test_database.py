"""Tests for database module"""
# Standard
import datetime as dt
import multiprocessing
import os
# Installed
import pytest
# Local
from libera_utils.db import getdb
from libera_utils.db.database import DatabaseException
from libera_utils.db.models import *


def db_testfunc(filename: str = None, archived: int = None):
    """DB test function that retrieves some dummy data"""
    db = getdb()
    print(f"Hello from pid {os.getpid()}. db.pid={db.pid}. db.url={db.url}")
    with db.session() as s:
        records = s.query(PdsFile)
        if filename is not None:
            records = records.filter(PdsFile.file_name == filename)
        if archived is not None:
            records = records.filter(PdsFile.archived == archived)
        print(records.all())


def test_db_connection(insert_test_data, clean_local_db):
    """Tests ability to connect to the test DB without doing anything special
    This ensures that the pytest fixture that sets the test DB is in place"""
    insert_test_data(PdsFile(file_name='foofile.txt'))
    with getdb().session() as s:
        records = s.query(PdsFile).all()
    assert len(records) == 1
    assert records[0].file_name == 'foofile.txt'
    assert records[0].archived is None


def test_db_manager_multiprocessing(insert_test_data, clean_local_db):
    """Test that the module level DB manager object works in multiprocessing"""
    archived_date_1 = dt.datetime.fromisoformat("2023-01-02T23:11:22.543123")
    archived_date_2 = dt.datetime.fromisoformat("2022-02-03T11:22:33.123456")
    insert_test_data(
        PdsFile(file_name='foofile.txt', archived=archived_date_1),
        PdsFile(file_name='barfile.txt', archived=archived_date_1),
        PdsFile(file_name='bazfile.txt', archived=archived_date_1),
        PdsFile(file_name='zipfile.txt', archived=archived_date_2),
        PdsFile(file_name='zapfile.txt', archived=archived_date_2),
        PdsFile(file_name='zopfile.txt', archived=archived_date_2)
    )

    for method in ('spawn', 'fork'):
        with multiprocessing.get_context(method).Pool(4) as pool:
            pool.starmap(db_testfunc,
                         [
                            ('foofile.txt', archived_date_1),
                            ('zipfile.txt', None),
                            ('bazfile.txt', None),
                            (None, archived_date_2),
                            (None, archived_date_1),
                            ('barfile.txt', archived_date_1),
                            ('barfile.txt', archived_date_2)
                         ])


def test_db_manager_multiple_configs():
    """Test ability to create multiple database managers with different configs and get each predictably"""
    db1 = getdb()
    # Gets init values from env vars set by test DB fixture
    assert db1.database == 'libera'
    assert db1.host == 'localhost' or db1.host == 'local-db'
    assert db1.user == 'libera_unit_tester'
    db2 = getdb(dbname='foo_db_name')
    assert db1 is not db2
    assert getdb() is db1
    assert getdb(dbname='foo_db_name') is db2


def test_db_misconfig(monkeypatch):
    """Test missing configuration"""
    monkeypatch.delenv('LIBERA_DB_NAME')
    with pytest.raises(DatabaseException):
        getdb()
