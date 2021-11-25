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
    Defines a level0 file object, as well as a table to hold level0 file names. The filename key is a foreign key to
    sience packet and stim packet filename keys. This table it used to keep track of what file a packet came from, as
    well as indicate if a file has had all of it's packets imported into the database.
    """

    __tablename__ = 'level0'

    # Table fields
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    ingest_complete = Column(Boolean, server_default=FetchedValue())
    ingested_at = Column(DateTime, server_default=FetchedValue())

    def __str__(self):
        return f'Level0({self.l0_file_id}) {self.filename}'
