"""Database module"""
# Standard
from contextlib import contextmanager
from enum import Enum
import logging
# Installed
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
# Local
from libera_sdp.config import config

logger = logging.getLogger(__name__)

Session = sessionmaker(expire_on_commit=False)


class DatabaseException(Exception):
    """Generic database related error"""
    pass


class DatabaseStates(Enum):
    """Enum of database names for dev, test, and prod"""
    PRODUCTION = 'sdp_prod'
    DEVELOPMENT = 'sdp_dev'
    TEST = 'sdp_test'


class DatabaseManager:
    """
    A class used to manage database connections corresponding to different database states (prod, test, etc.)
    """
    def __init__(self):
        """ DatabaseManager constructor"""
        dbname = config.get('LIBERA_DB_NAME')
        self.state = DatabaseStates(dbname) if dbname else None
        self.user = config.get('LIBERA_DB_USER') or None
        self.host = config.get('LIBERA_DB_HOST') or None

        if all((self.state, self.user, self.host)):
            self.engine = create_engine(self.url)
        else:
            self.engine = None

    def __str__(self):
        return f"DatabaseManager(user={self.user}, host={self.host}, db={self.state.name})"

    def refresh(self):
        """Force refresh of the database engine. This will break any existing connections so use cautiously"""
        # https://docs.sqlalchemy.org/en/14/core/connections.html#sqlalchemy.engine.Engine.dispose
        if self.engine:
            self.engine.dispose()
        self.__init__()

    @property
    def url(self):
        """JDBC connection string"""
        return self._format_url(self.state, self.user, self.host)

    @staticmethod
    def _format_url(database: DatabaseStates, user: str, host: str = 'localhost', port: int = 5432, password: str = "",
                    dialect: str = "postgresql"):
        """
        Returns a database connection url given database parameters

        Parameters
        ----------
        user : str
            DB username
        password : str, Optional
            Password. Default is an empty string and the value is usually found in environment
        database : str
            Name of database to connect to
        host : str
            Name of host machine
        dialect : str, Optional
            SQL dialect. Default is Postgres
        port: int, Optional
            Port number. Default is 5432

        Returns
        -------
        : str
            JDBC connection string
        """
        return f"{dialect}://{user}:{password}@{host}:{port}/{database.value}"

    @contextmanager
    def session(self):
        """Provide a transactional scope around a series of operations."""
        Session.configure(bind=self.engine)

        session = Session()
        try:
            yield session
            session.commit()
        except Exception as err:
            session.rollback()
            raise
        finally:
            session.close()

    def truncate_product_tables(self):
        """
        Truncates all product tables
        :return:
        """
        if self.state != DatabaseStates.TEST:
            raise ValueError(f"Refusing to truncate all tables for database state {self.state}. "
                             f"We only permit this operation for the test database.")
        meta = MetaData()
        meta.reflect(bind=self.engine)
        for table in reversed(meta.sorted_tables):
            if table.name not in ('flyway_schema_history', ):
                self.engine.execute(table.delete())


db = DatabaseManager()
