"""ORM objects for SQLAlchemy"""
# Installed
from sqlalchemy import Column, SmallInteger, Integer, Text, DateTime, Boolean, FetchedValue, ForeignKey, LargeBinary
from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.orm import relationship
# Local
from libera_utils.db import Base
from libera_utils.db.mixins import ReprMixin, DataProductMixin


class APID(Base, ReprMixin):
    """
    Lookup table of APID descriptions
    """

    # Table fields
    apid = Column(Integer, primary_key=True)
    description = Column(Text)


class Packet(Base, ReprMixin):
    """
    Packet contents
    """
    __repr_attrs__ = ['apid', 'sequence_count', 'data_length']

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    version_number = Column(SmallInteger)
    type = Column(Boolean)
    secondary_header_flag = Column(Boolean)
    apid = Column(SmallInteger, ForeignKey(APID.apid))
    sequence_flags = Column(BIT(2))
    sequence_count = Column(SmallInteger)
    data_length = Column(SmallInteger)
    secondary_header = Column(LargeBinary)
    user_data = Column(LargeBinary)


class L0(Base, DataProductMixin, ReprMixin):
    """
    L0 files containing packets.
    """
    __repr_attrs__ = ['filename', 'version']

    # Table fields
    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    filename = Column(Text)
    version = Column(SmallInteger)
    created = Column(DateTime, server_default=FetchedValue())
    ingested = Column(DateTime)

    packets = relationship(Packet, secondary='l0_pkt_jt')


class L0PktJt(Base):
    """
    Jointable between L0 and Packet for n:m relationship
    """
    l0_id = Column(Integer, ForeignKey(L0.id), primary_key=True)
    pkt_id = Column(Integer, ForeignKey(Packet.id), primary_key=True)


class L1b(Base, ReprMixin, DataProductMixin):
    """
    L1b data products.
    """
    __repr_attrs__ = ['filename']

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    filename = Column(Text)
    version = Column(SmallInteger)
    created = Column(DateTime, server_default=FetchedValue())

    packets = relationship(Packet, secondary='l1b_pkt_jt')


class L1bPktJt(Base):
    """
    Jointable between L1b and Packet for n:m relationship
    """
    l1b_id = Column(Integer, ForeignKey(L1b.id), primary_key=True)
    pkt_id = Column(Integer, ForeignKey(Packet.id), primary_key=True)
