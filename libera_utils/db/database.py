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
    # Class level attribute that stores the cache of existing DatabaseManager objects
    # Changing this dictionary anywhere changes this dictionary for ALL instances of the class
    _db_manager_cache = {}

    def __init__(self, dbhost: str, dbuser: str, dbpass: str, dbname: str, dbport: int):
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
        dbport : int
            Database port
        """
        self.pid = os.getpid()  # Store the PID of the process in which this object was created

        # Resolve and set database connection parameters
        (self.host,
         self.user,
         self.password,
         self.database,
         self.port) = self._resolve_dbparams(dbhost, dbuser, dbpass, dbname, dbport)

        self.password = dbpass

        self.engine = create_engine(self.url)

        # Save this new instance to the class cache (changes for all instances)
        self._db_manager_cache[hash(self)] = self
        logger.info(f"Initialized {self}")
        n_removed = self._remove_invalid_cached_managers()
        logger.debug(f"Removed {n_removed} invalid cached manager objects that don't belong to this process.")

    @staticmethod
    def _resolve_dbparams(dbhost: str, dbuser: str, dbpass: str, dbname: str, dbport: int):
        """Resolve parameters provided against environment variables and preconfigured json values.
        Environment variables override preconfigured json values.
        Direct specification of truthy parameters overrides environment variables.

        Parameters
        ----------
        dbhost : str
            Database host. If passed value is falsy, defaults to 'localhost'
        dbuser : str
            Database user
        dbpass : str
            Database password for user login
        dbname : str
            Database name
        dbport : int
            Database port. If passed value is falsy, defaults to 5432
        """
        database = dbname or config.get('LIBERA_DB_NAME')
        if not database:
            raise DatabaseException("Missing database name.")

        user = dbuser or config.get('LIBERA_DB_USER')
        if not user:
            raise DatabaseException("Missing database user.")

        host = dbhost or config.get('LIBERA_DB_HOST')
        if not host:  # If we still don't have a host value, set localhost
            host = 'localhost'

        port = dbport or config.get('LIBERA_DB_PORT')
        if not port:
            port = 5432

        # Return resolved versions of the input variables according to resolution priority
        return host, user, dbpass, database, port

    def __str__(self):
        return f"_DatabaseManager(user={self.user}, host={self.host}, db={self.database})"

    def __bool__(self):
        return bool(self.engine)

    def __hash__(self):
        # Used for retrieving cached manager objects instead of creating new ones
        return self._calculate_hash(self.pid, self.url)

    @staticmethod
    def _calculate_hash(pid: int, url: str):
        """Standard hash calculation for this class.

        Used in __hash__ and other places where the same value is required.

        Parameters
        ----------
        pid : int
            Process ID associated with the object in question
        url : str
            Connection URL string to the database

        Returns
        -------
        : int
            The canonical hash of the database connection manager
        """
        return hash((pid, url))

    def _remove_invalid_cached_managers(self):
        """Checks the cache for manager objects that don't belong to this process and removes them.

        Returns
        -------
        : int
            Number of removed managers
        """
        # Find all currently cached managers that don't belong to the current process
        pid = os.getpid()
        keys_to_remove = [hash_ for hash_, manager in self._db_manager_cache.items() if manager.pid != pid]

        for hash_ in keys_to_remove:
            del self._db_manager_cache[hash_]

        return len(keys_to_remove)

    @property
    def url(self):
        """JDBC connection string"""
        return self._format_url(self.host, self.user, self.password, self.database, self.port)

    @classmethod
    def get(cls, dbhost: str = None, dbuser: str = None, dbpass: str = "", dbname: str = None, dbport: int = None):
        """Cache-enabled factory method.

        Retrieve an existing DB manager from the cache if one already exists in the same PID and configuration
        (identified by hash). If no identical manager exists in this process already, create a new one and return it.
        This makes _DatabaseManager safe for use with either forked or spawned processes because we are never
        copying database engines across process boundaries.

        All parameters are passed through the resolution method _resolve_dbparams, which searches the environment
        and sets default values.

        Parameters
        ----------
        dbhost : str, Optional
            Database host. If not provided, LIBERA_DB_HOST is searched. If not found, 'localhost' is used.
        dbuser : str, Optional
            Database user. If not provided, LIBERA_DB_USER environment variable is used.
        dbpass : str, Optional
            Database password for user. If not provided, PGPASSWORD environment variable or ~/.pgpass file is used.
        dbname : str, Optional
            Database name. If not provided, LIBERA_DB_NAME environment variable is used.
        dbport : int, Optional
            Database port. If not provided, LIBERA_DB_PORT is searched. If not found, 5432 is used.

        Returns
        -------
        : _DatabaseManager
        """
        # Calculate the hash of a candidate new DatabaseManager to search the existing cache
        resolved_params = cls._resolve_dbparams(dbhost, dbuser, dbpass, dbname, dbport)
        url = cls._format_url(*resolved_params)
        candidate_hash = cls._calculate_hash(pid=os.getpid(), url=url)
        try:
            cached_db_manager = cls._db_manager_cache[candidate_hash]
            logger.info(f"Found cached {cached_db_manager}")
            return cached_db_manager
        except KeyError:
            # No cached managers matched the candidate hash. Create a new manager.
            new_db_manager = cls(dbhost, dbuser, dbpass, dbname, dbport)
            return new_db_manager

    @staticmethod
    def _format_url(dbhost: str, dbuser: str, dbpass: str, dbname: str,  # nosec B107 # CWE-259
                    dbport: int = 5432):
        """
        Returns a postgres database connection url given database parameters

        Parameters
        ----------
        dbhost : str
            Name of host machine
        dbuser : str
            DB username
        dbpass : str
            Password. Passing an empty string results in searching the environment for PGPASSWORD or the .pgpass file.
        dbname : str
            Name of database to connect to
        dbport: int, Optional
            Port number. Default is 5432

        Returns
        -------
        : str
            JDBC connection string
        """
        dialect = "postgresql"
        return f"{dialect}://{dbuser}:{dbpass}@{dbhost}:{dbport}/{dbname}"

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
