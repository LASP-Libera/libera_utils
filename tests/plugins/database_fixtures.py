"""Pytest plugin module for database-related fixtures"""
# pylint: disable=redefined-outer-name
# Installed
import pytest
# Local
from libera_utils.db import getdb


@pytest.fixture(scope='session', autouse=True)
def use_test_db(monkeypatch_session):
    """Automatically set environment variables to use the testing database"""
    monkeypatch_session.setenv('LIBERA_DB_NAME', 'libera')
    monkeypatch_session.setenv('LIBERA_DB_USER', 'libera_unit_tester')
    monkeypatch_session.setenv('PGPASSWORD', 'testerpass')


@pytest.fixture
def clean_local_db():
    """Ensure the TEST database is in a clean state and return it to clean after we're done testing.
    Note: If this weren't such a costly operation, we would set this to autouse=True
    """
    db = getdb()
    db.truncate_product_tables()
    yield
    db.truncate_product_tables()


@pytest.fixture
def insert_test_data(clean_local_db):
    """Provides a function to insert test data records into the DB.
    Clears out the DB before and after the test."""

    def _insert(*record_objects):
        with getdb().session() as s:
            for record in record_objects:
                s.add(record)

    yield _insert
