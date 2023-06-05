"""ORM objects for SQLAlchemy"""
# Installed
from sqlalchemy import Column, FetchedValue, ForeignKey
from sqlalchemy.dialects.postgresql import SMALLINT, BIGINT, INTEGER, TEXT, TIMESTAMP, BOOLEAN, NUMERIC
from sqlalchemy.orm import relationship
# Local
from libera_utils.db import Base
from libera_utils.db.mixins import ReprMixin, DataProductMixin


class Cr(Base, ReprMixin):
    """
    Construction record
    """
    __repr_attrs__ = ["file_name"]

    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    file_name = Column(TEXT)
    ingested = Column(TIMESTAMP(timezone=True), server_default=FetchedValue())
    archived = Column(TIMESTAMP(timezone=True))
    edos_software_version = Column(INTEGER)
    construction_record_type = Column(INTEGER)
    test_flag = Column(BOOLEAN)
    n_scs_start_stops = Column(INTEGER)
    n_bytes_fill_data = Column(NUMERIC(20))
    n_length_mismatches = Column(BIGINT)
    first_packet_sc_time = Column(NUMERIC(20))
    last_packet_sc_time = Column(NUMERIC(20))
    first_packet_utc_time = Column(TIMESTAMP(timezone=True))
    last_packet_utc_time = Column(TIMESTAMP(timezone=True))
    first_packet_esh_time = Column(NUMERIC(20))
    last_packet_esh_time = Column(NUMERIC(20))
    n_rs_corrections = Column(BIGINT)
    n_packets = Column(BIGINT)
    size_bytes = Column(NUMERIC(20))
    n_ssc_discontinuities = Column(INTEGER)
    completion_time = Column(NUMERIC(20))
    n_apids = Column(SMALLINT)
    n_pds_files = Column(SMALLINT)

    scs_start_stop_times: list = relationship("CrScsStartStopTimes", back_populates="construction_record")
    apids: list = relationship("CrApid", back_populates="construction_record")
    pds_files: list = relationship("PdsFile", back_populates="construction_record")


class CrApid(Base, ReprMixin):  # I know...
    """
    APID data record in a CR
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_id = Column(INTEGER, ForeignKey(Cr.id))
    scid_apid = Column(BIGINT)
    byte_offset = Column(NUMERIC(20))
    n_vcids = Column(SMALLINT)
    n_ssc_gaps = Column(BIGINT)
    n_edos_generated_fill_data = Column(BIGINT)
    count_edos_generated_octets = Column(NUMERIC(20))
    n_length_discrepancy_packets = Column(BIGINT)
    first_packet_sc_time = Column(NUMERIC(20))
    last_packet_sc_time = Column(NUMERIC(20))
    first_packet_utc_time = Column(TIMESTAMP(timezone=True))
    last_packet_utc_time = Column(TIMESTAMP(timezone=True))
    esh_first_packet_time = Column(NUMERIC(20))
    esh_last_packet_time = Column(NUMERIC(20))
    n_vcdu_corrected_packets = Column(BIGINT)
    n_in_the_data_set = Column(BIGINT)
    n_octect_in_apid = Column(NUMERIC(20))

    construction_record = relationship("Cr", back_populates="apids")
    vcids: list = relationship("CrApidVcid", back_populates="apid")
    ssc_gaps: list = relationship("CrApidSscGap", back_populates="apid")
    ssc_length_discrepancies: list = relationship("CrApidSscLenDiscrepancies", back_populates="apid")
    edos_fill_data: list = relationship("CrApidEdosGeneratedFillData", back_populates="apid")


class CrApidVcid(Base, ReprMixin):
    """
    VCID included in PDS for a single APID
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_apid_id = Column(INTEGER, ForeignKey(CrApid.id))
    scid_vcid = Column(INTEGER)

    apid = relationship("CrApid", back_populates="vcids")


class CrApidSscGap(Base, ReprMixin):
    """
    SSC gap described per APID in a CR
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_apid_id = Column(INTEGER, ForeignKey(CrApid.id))
    first_missing_ssc = Column(BIGINT)
    gap_byte_offset = Column(NUMERIC(20))
    n_missing_sscs = Column(BIGINT)
    preceding_packet_sc_time = Column(NUMERIC(20))
    following_packet_sc_time = Column(NUMERIC(20))
    preceding_packet_utc_time = Column(TIMESTAMP(timezone=True))
    following_packet_utc_time = Column(TIMESTAMP(timezone=True))
    preceding_packet_esh_time = Column(NUMERIC(20))
    following_packet_esh_time = Column(NUMERIC(20))

    apid = relationship("CrApid", back_populates="ssc_gaps")


class PdsFile(Base, ReprMixin):
    """
    Record of a single file as part of a PDS
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_id = Column(INTEGER, ForeignKey(Cr.id))
    file_name = Column(TEXT)
    ingested = Column(TIMESTAMP(timezone=True))
    archived = Column(TIMESTAMP(timezone=True))

    construction_record = relationship("Cr", back_populates="pds_files")
    apids: list = relationship("PdsFileApid", back_populates="pds_file")


class PdsFileApid(Base, ReprMixin):
    """
    APID information for a single PDS file
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    pds_file_id = Column(INTEGER, ForeignKey(PdsFile.id))
    scid_apid = Column(BIGINT)
    first_packet_sc_time = Column(NUMERIC(20))
    last_packet_sc_time = Column(NUMERIC(20))
    first_packet_utc_time = Column(TIMESTAMP(timezone=True))
    last_packet_utc_time = Column(TIMESTAMP(timezone=True))

    pds_file = relationship("PdsFile", back_populates="apids")


class CrScsStartStopTimes(Base, ReprMixin):
    """
    SSC Start and Stop Time information for a Construction Record
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_id = Column(INTEGER, ForeignKey(Cr.id))
    scs_start_sc_time = Column(NUMERIC(20))
    scs_stop_sc_time = Column(NUMERIC(20))
    scs_start_utc_time = Column(TIMESTAMP(timezone=True))
    scs_stop_utc_time = Column(TIMESTAMP(timezone=True))

    construction_record = relationship("Cr", back_populates="scs_start_stop_times")


class CrApidSscLenDiscrepancies(Base, ReprMixin):
    """
    SSC length discrepancies per APID in a CR
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_apid_id = Column(INTEGER, ForeignKey(CrApid.id))
    ssc_length_discrepancy = Column(BIGINT)

    apid = relationship("CrApid", back_populates="ssc_length_discrepancies")


class CrApidEdosGeneratedFillData(Base, ReprMixin):
    """
    EDOS generated fill data per APID in a CR
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    cr_apid_id = Column(INTEGER, ForeignKey(CrApid.id))
    ssc_with_generated_data = Column(BIGINT)
    filled_byte_offset = Column(NUMERIC(20))
    index_to_fill_octet = Column(BIGINT)

    apid = relationship("CrApid", back_populates="edos_fill_data")


class SpkCkFile(Base, ReprMixin, DataProductMixin):
    """
    SPICE CK and SPK kernel file
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    file_name = Column(TEXT)
    start_sc_time = Column(NUMERIC(20))
    stop_sc_time = Column(NUMERIC(20))
    start_utc_time = Column(TIMESTAMP(timezone=True))
    stop_utc_time = Column(TIMESTAMP(timezone=True))
    revision = Column(INTEGER)
    quality_flag = Column(INTEGER)

    pds_files: list = relationship("PdsFile", secondary='sdp.spk_ck_file_pds_file_jt')


class SpkCkFilePdsFileJt(Base):
    """
    Join table between PdsFile and SpkCkFile for n:m relationship
    """
    pds_file_id = Column(INTEGER, ForeignKey(PdsFile.id), primary_key=True)
    spk_ck_file_id = Column(INTEGER, ForeignKey(SpkCkFile.id), primary_key=True)


class L1bCam(Base):
    """
    L1b Camera product files
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    file_name = Column(TEXT)
    revision = Column(INTEGER)
    quality_flag = Column(INTEGER)


class L1bRad(Base):
    """
    L1b Radiometer product files
    """
    id = Column(INTEGER, primary_key=True, server_default=FetchedValue())
    file_name = Column(TEXT)
    revision = Column(INTEGER)
    quality_flag = Column(INTEGER)
