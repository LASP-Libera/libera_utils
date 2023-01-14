"""Database module"""
# Standard
from contextlib import contextmanager
import logging
import os
# Installed
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
# Local
from libera_utils.config import config

logger = logging.getLogger(__name__)

Session = sessionmaker(expire_on_commit=False)


class DatabaseException(Exception):
    """Generic database related error"""
    pass


class _DatabaseManager:
    """
    A caching class used to manage database connections.

    This class should never be instantiated directly. Instead, users should use the convenience method
    libera_utils.db.database.getdb, which is an alias for the _DatabaseManager.get factory method.
    """
    _db_manager_cache = {}

    def __init__(self, dbhost: str, dbuser: str, dbpass: str, dbname: str):
        """_DatabaseManager constructor

        Parameters
        ----------
        dbhost : str
            Database host
        dbuser : str
            Database user
        dbpass : str
            Database password for user login
        dbname : str
            Database name
        """
        self.pid = os.getpid()  # Store the PID of the process in which this object was created

        self.database = dbname or config.get('LIBERA_DB_NAME')
        if not self.database:
            raise DatabaseException("Missing database name.")

        self.user = dbuser or config.get('LIBERA_DB_USER')
        if not self.user:
            raise DatabaseException("Missing database user.")

        if dbhost:  # If we were directly passed a truthy value
            self.host = dbhost
        else:  # No truthy value, then get the environment variable (or json default)
            self.host = config.get('LIBERA_DB_HOST')

        if not self.host:  # If we still don't have a host value, set localhost as a best guess
            self.host = 'localhost'

        self.password = dbpass

        self.engine = create_engine(self.url)
        logger.info(f"Initialized {self}")

    def __str__(self):
        return f"_DatabaseManager(user={self.user}, host={self.host}, db={self.database})"

    def __bool__(self):
        return bool(self.engine)

    def __hash__(self):
        # Used for retrieving cached manager objects instead of creating new ones
        return hash((self.pid, self.url))

    @property
    def url(self):
        """JDBC connection string"""
        return self._format_url(self.database, self.user, self.host, self.password)

    @classmethod
    def get(cls, dbname: str = None, dbuser: str = None, dbhost: str = None, dbpass: str = ""):
        """Cache-enabled factory method.

        Retrieve an existing DB manager from the cache if one already exists in the same PID and configuration
        (identified by hash). If no identical manager exists in this process already, create a new one and return it.
        This makes _DatabaseManager safe for use with either forked or spawned processes because we are never
        copying database engines across process boundaries.

        If no parameters are passed, the _DatabaseManager object sources configuration from the environment.

        Parameters
        ----------
        dbname : str, Optional
            Database name. If not provided, LIBERA_DB_NAME environment variable is used.
        dbuser : str, Optional
            Database user. If not provided, LIBERA_DB_USER environment variable is used.
        dbhost : str, Optional
            Database host. If not provided, 'localhost' is used.
        dbpass : str, Optional
            Database password for user. If not provided, PGPASSWORD environment variable or ~/.pgpass file is used.

        Returns
        -------
        : _DatabaseManager
        """
        # Create a new candidate _DatabaseManager to compare hash-wise to the existing cache
        new_db_manager = cls(dbhost, dbuser, dbpass, dbname)
        try:
            return cls._db_manager_cache[hash(new_db_manager)]
        except KeyError:
            cls._db_manager_cache[hash(new_db_manager)] = new_db_manager
            return new_db_manager

    @staticmethod
    def _format_url(database: str, user: str, host: str, password: str = "",  # nosec B107 # CWE-259
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
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def truncate_product_tables(self):
        """
        Truncates all tables in sdp schema except for flyway_schema_history
        :return:
        """
        if self.host not in ('localhost', 'local-db'):
            raise ValueError(f"Refusing to truncate all tables for database on host {self.host}. "
                             "We only permit this operation for local dev databases on host "
                             "'local-db' or 'localhost'.")
        meta = MetaData(schema='sdp')
        meta.reflect(bind=self.engine)
        for table in reversed(meta.sorted_tables):
            if table.name not in ('flyway_schema_history', ):
                self.engine.execute(table.delete())
