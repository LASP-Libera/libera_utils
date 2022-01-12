"""ORM objects for SQLAlchemy"""
# Installed
from sqlalchemy import Column, Integer, String, DateTime, Boolean, FetchedValue
from sqlalchemy.orm import relationship, backref
# Local
from libera_sdp.db import Base


# TODO: This is basically a dummy model. It has a few unit tests that rely on it but we shuold consider this to be
#    a work in progress. Change tests as needed. Just be sure that any changes made here are first made to the DB schema
#    using a flyway migration.
class Level0(Base):
    """
    This table stores records of raw L0 files.
    """

    __tablename__ = 'level0'

    # Table fields
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    ingest_complete = Column(Boolean, server_default=FetchedValue())
    ingested_at = Column(DateTime, server_default=FetchedValue())

    def __str__(self):
        return (f'{self.__class__.__name__}('
                f'{self.id}, '
                f'{self.filename}, '
                f'{self.ingest_complete}, '
                f'{self.ingested_at})')
