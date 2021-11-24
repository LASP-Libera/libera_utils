"""Database module"""
# Standard
from contextlib import contextmanager
import logging
import os
# Installed
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
# Local
from libera_sdp.config import config

logger = logging.getLogger(__name__)

Session = sessionmaker(expire_on_commit=False)

_db_manager_cache = {}


class DatabaseException(Exception):
    """Generic database related error"""
    pass


class _DatabaseManager:
    """
    A class used to manage database connections corresponding to different database states (prod, test, etc.)
    """
    def __init__(self, dbname: str = None, dbuser: str = None, dbhost: str = None, dbpass: str = ""):
        """ _DatabaseManager constructor"""
        self.pid = os.getpid()  # Store the PID of the process in which this object was created

        self.database = dbname or config.get('LIBERA_DB_NAME')
        if not self.database:
            raise DatabaseException(f"Missing database name.")

        self.user = dbuser or config.get('LIBERA_DB_USER')
        if not self.user:
            raise DatabaseException(f"Missing database user.")

        self.host = dbhost or config.get('LIBERA_DB_HOST')
        if not self.host:
            raise DatabaseException(f"Missing database host.")

        self.password = dbpass
        self.engine = create_engine(self.url)

    def __str__(self):
        return f"_DatabaseManager(user={self.user}, host={self.host}, db={self.database.name})"

    def __bool__(self):
        return bool(self.engine)

    def __hash__(self):
        # Used for retrieving cached manager objects instead of creating new ones
        return hash((self.pid, self.url))

    @property
    def url(self):
        """JDBC connection string"""
        return self._format_url(self.database, self.user, self.host, self.password)

    @staticmethod
    def _format_url(database: str, user: str, host: str, password: str = "",
                    port: int = 5432, dialect: str = "postgresql"):
        """
        Returns a database connection url given database parameters

        Parameters
        ----------
        database : str
            Name of database to connect to
        user : str
            DB username
        host : str
            Name of host machine
        password : str, Optional
            Password. Passing an empty string results in searching the environment for PGPASSWORD or the .pgpass file.
        port: int, Optional
            Port number. Default is 5432
        dialect : str, Optional
            SQL dialect. Default is Postgres

        Returns
        -------
        : str
            JDBC connection string
        """
        return f"{dialect}://{user}:{password}@{host}:{port}/{database}"

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
        if self.database != 'sdp_test':
            raise ValueError(f"Refusing to truncate all tables for database database {self.database}. "
                             f"We only permit this operation for the sdp_test database.")
        meta = MetaData()
        meta.reflect(bind=self.engine)
        for table in reversed(meta.sorted_tables):
            if table.name not in ('flyway_schema_history', ):
                self.engine.execute(table.delete())


def getdb(dbname: str = None, dbuser: str = None, dbhost: str = None, dbpass: str = ""):
    """Retrieve an existing DB manager from the cache if one already exists in the same PID and configuration
    (identified by hash). If no identical manager exists in this process already, create a new one and return it.
    This effectively makes our _DatabaseManager safe for use with either forked or spawned processes.

    If no parameters are passed, the _DatabaseManager object sources configuration from the environment.

    Parameters
    ----------
    dbname : str, Optional
        Database name
    dbuser : str, Optional
        User
    dbhost : str, Optional
        Host
    dbpass : str, Optional
        Password

    Returns
    -------
    : _DatabaseManager
    """
    # Create a new manager object with which to match the hash to a (possibly) cached manager object for this PID
    # and the currently specified connection parameters in the environment.
    new_db_manager = _DatabaseManager(dbname, dbuser, dbhost, dbpass)
    try:
        return _db_manager_cache[hash(new_db_manager)]
    except KeyError:
        _db_manager_cache[hash(new_db_manager)] = new_db_manager
        return new_db_manager
