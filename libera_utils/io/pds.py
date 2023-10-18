"""Module for manifest file handling"""
# Standard
import datetime
import logging
from pathlib import Path
from bitstring import ConstBitStream
# Installed
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils.io.smart_open import smart_open
import libera_utils.db.models as libera_db_models
from libera_utils.time import convert_cds_integer_to_datetime
from libera_utils.io.filenaming import L0Filename

logger = logging.getLogger(__name__)


class EDOSGeneratedFillDataFromAPID:
    """
    Object representation of the information pertaining to which data in a Production Data Set (PDS) are generated and
    filled by EDOS. This corresponds to a database table and connects to a Construction Record (CR). This object is
    created as part of reading in a CR and requires an open bitstream to read from.
    """
    def __init__(self, ssc_with_generated_data: int,
                 filled_byte_offset: int,
                 index_to_fill_octet: int):
        self.ssc_with_generated_data=ssc_with_generated_data
        self.filled_byte_offset=filled_byte_offset
        self.index_to_fill_octet=index_to_fill_octet

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """A function to build the python object representation of an EDOS fill data from apid object from a
            Construction Record."""
        ssc_with_generated_data = cr_bitstream.read("uint:32")
        filled_byte_offset = cr_bitstream.read("uint:64")
        index_to_fill_octet = cr_bitstream.read("uint:32")
        return cls(ssc_with_generated_data=ssc_with_generated_data,
                   filled_byte_offset=filled_byte_offset,
                   index_to_fill_octet=index_to_fill_octet)

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        return libera_db_models.CrApidEdosGeneratedFillData(
            ssc_with_generated_data=self.ssc_with_generated_data,
            filled_byte_offset=self.filled_byte_offset,
            index_to_fill_octet=self.index_to_fill_octet
        )


class SSCLengthDiscrepancy:
    """
    Object representation of the information of the length discrepancy in an SSC. This corresponds to a database table
    and connects to a Construction Record (CR). This object is created as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, ssc_length_discrepancy: int):
        self.ssc_length_discrepancy = ssc_length_discrepancy

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """A function to build the python object representation of an SSC length discrepancy object from a
                  Construction Record."""
        ssc_length_discrepancy = cr_bitstream.read("uint:32")
        return cls(ssc_length_discrepancy=ssc_length_discrepancy)

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        return libera_db_models.CrApidSscLenDiscrepancies(
            ssc_length_discrepancy=self.ssc_length_discrepancy
        )


class SCSStartStopTimes:
    """
    Object representation of the information of Spacecraft Session (SCS) start and stop times of data. This
    corresponds to a database table and connects to a Construction Record (CR). This object is created as part of
    reading in a CR and thus requires an open bitstream to read from.
    """
    def __init__(self, scs_start_sc_time: int,
                 scs_stop_time_sc_time: int):
        self.scs_start_time_sc_time = scs_start_sc_time
        self.scs_start_time_utc = convert_cds_integer_to_datetime(scs_start_sc_time)
        self.scs_stop_time_sc_time = scs_stop_time_sc_time
        self.scs_stop_time_utc = convert_cds_integer_to_datetime(scs_stop_time_sc_time)

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """A function to build the python object representation of an SCS start and stop time object from a
          Construction Record. This creates datetime objects from the read in spacecraft (sc) times."""
        scs_start_time_sc_time = cr_bitstream.read("uint:64")
        scs_stop_time_sc_time = cr_bitstream.read("uint:64")
        return cls(scs_start_sc_time=scs_start_time_sc_time,
                   scs_stop_time_sc_time=scs_stop_time_sc_time)

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        return libera_db_models.CrScsStartStopTimes(
            scs_start_sc_time=self.scs_start_time_sc_time,
            scs_stop_sc_time=self.scs_stop_time_sc_time,
            scs_start_utc_time=self.scs_start_time_utc,
            scs_stop_utc_time=self.scs_stop_time_utc
        )


class APIDFromPDSFromConstructionRecord:
    """
    Object representation of the information of Application IDs (APID) from within a Production Data Set (PDS). This
    corresponds to a database table and connects to a Construction Record (CR). This object is created as part of
    reading in a CR and thus requires an open bitstream to read from.
    """
    def __init__(self, scid_apid: int,
                 apid_first_packet_sc_time: int,
                 apid_last_packet_sc_time: int):
        self.scid_apid = scid_apid
        self.apid_first_packet_sc_time = apid_first_packet_sc_time
        self.apid_first_packet_utc = convert_cds_integer_to_datetime(apid_first_packet_sc_time)
        self.apid_last_packet_sc_time = apid_last_packet_sc_time
        self.apid_last_packet_utc = convert_cds_integer_to_datetime(apid_last_packet_sc_time)

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """A function to build the python object representation of the APID stored in a PDS file from a
        Construction Record. This creates datetime objects from the read in spacecraft (sc) times."""
        # Read unused data
        cr_bitstream.read("uint:8")
        scid_apid = cr_bitstream.read("uint:24")
        apid_first_packet_sc_time = cr_bitstream.read("uint:64")
        apid_last_packet_sc_time = cr_bitstream.read("uint:64")
        # Read unused data
        cr_bitstream.read("uint:32")
        return cls(scid_apid, apid_first_packet_sc_time, apid_last_packet_sc_time)

    @property
    def scid(self):
        """Property that contains the SCID alone"""
        bytes_scid = self.scid_apid.to_bytes(3, 'big')
        scid_read = ConstBitStream(bytes_scid)
        return scid_read.read("uint:8")

    @property
    def apid(self):
        """Property that contains the APID alone"""
        bytes_scid = self.scid_apid.to_bytes(3, 'big')
        scid_read = ConstBitStream(bytes_scid)
        # Read unused data
        scid_read.read("uint:13")
        return scid_read.read("uint:11")

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        return libera_db_models.PdsFileApid(
                scid_apid=self.scid_apid,
                first_packet_sc_time=self.apid_first_packet_sc_time,
                last_packet_sc_time=self.apid_last_packet_sc_time,
                first_packet_utc_time=self.apid_first_packet_utc,
                last_packet_utc_time=self.apid_last_packet_utc
            )


class PDSRecord:
    """
    Object representation of the information related to a Production Data Set (PDS) file. This corresponds to a
    database table and connects to a Construction Record (CR). This object is created as part of reading in a
    CR or a PDS file. The from_filepath method is used for PDS files and from_bitstream is used for creating
    from a CR. The only shared entry between these two methods is the filename entry and is expected that for a
    single PDS file both of these methods will eventually be called
    """
    def __init__(self, filename: str, apid_filecount: int = None, apids: list = None):
        path_filename = AnyPath(filename)
        self.filename = path_filename.name
        self.apid_count = apid_filecount
        self.apids = apids

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """This is called in when a construction record is being read and records of each PDS files are created."""
        pds_filename = (cr_bitstream.read("bytes:40")).decode()
        # Read unused data
        cr_bitstream.read("uint:24")

        # This is quoted in 25-4 as a "one-up" counter with values of 1 to 3. However, there is a situation when the
        # value can be 0, and then there is one entry with complete data as 0's throughout. To account for this take
        # the maximum of the value and 1 to ensure if a 0 is there at least one full entry of 0's is read.
        apid_count_this_file = max(cr_bitstream.read("uint:8"), 1)
        apids_this_file = []
        for _ in range(apid_count_this_file):
            apids_this_file.append(APIDFromPDSFromConstructionRecord.from_bitstream(cr_bitstream))

        return cls(filename=pds_filename,
                   apid_filecount=apid_count_this_file,
                   apids=apids_this_file)

    @classmethod
    def from_filename(cls, filename: str or AnyPath):
        """This is called by pds_ingest when a single pds data file is being ingested."""
        pds_filename = AnyPath(filename).name
        return cls(pds_filename)

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        if self.apid_count is None:
            # When the PDS file is not the CR
            return libera_db_models.PdsFile(
                file_name=self.filename,
                ingested=datetime.datetime.utcnow()
            )
        # When PDSRecord is part of a CR reading
        orm_apids_list = []
        for apid in self.apids:
            orm_apids_list.append(apid.to_orm())
        return libera_db_models.PdsFile(
            file_name=self.filename,
            apids=orm_apids_list
        )


class SSCGapInformationFromConstructionRecord:
    """
    Object representation of the information of Spacecraft Contact Sessions (SCS). This corresponds to a database table
    and connects to a Construction Record (CR). This object is created as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, apid_gap_first_missing_ssc_packet: int,
                 apid_gap_byte_offset: int,
                 apid_num_ssc_packets_missed: int,
                 apid_preceding_packet_sc_time: int,
                 apid_following_packet_sc_time: int,
                 apid_preceding_packet_esh_time: int,
                 apid_following_packet_esh_time: int):
        self.apid_gap_first_missing_ssc_packet = apid_gap_first_missing_ssc_packet
        self.apid_gap_byte_offset = apid_gap_byte_offset
        self.apid_num_ssc_packets_missed = apid_num_ssc_packets_missed

        # These are not labeled in the ICD document and so this is a guess based on other patterns in the ICD
        self.apid_preceding_packet_sc_time = apid_preceding_packet_sc_time
        self.apid_following_packet_sc_time = apid_following_packet_sc_time
        self.apid_preceding_packet_utc = convert_cds_integer_to_datetime(apid_preceding_packet_sc_time)
        self.apid_following_packet_utc = convert_cds_integer_to_datetime(apid_following_packet_sc_time)

        self.apid_preceding_packet_esh_time = apid_preceding_packet_esh_time
        self.apid_following_packet_esh_time = apid_following_packet_esh_time

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """Constructor wrapper method for creating during Construction record reading with a bitstream"""
        apid_gap_first_missing_ssc_packet = cr_bitstream.read("uint:32")
        apid_gap_byte_offset = cr_bitstream.read("uint:64")
        apid_num_ssc_packets_missed = cr_bitstream.read("uint:32")

        # These are not labeled in the ICD document and so this is a guess based on other patterns in the ICD
        apid_preceding_packet_sc_time = cr_bitstream.read("uint:64")
        apid_following_packet_sc_time = cr_bitstream.read("uint:64")

        apid_preceding_packet_esh_time = cr_bitstream.read("uint:64")
        apid_following_packet_esh_time = cr_bitstream.read("uint:64")

        return cls(apid_gap_first_missing_ssc_packet=apid_gap_first_missing_ssc_packet,
                   apid_gap_byte_offset=apid_gap_byte_offset,
                   apid_num_ssc_packets_missed=apid_num_ssc_packets_missed,
                   apid_preceding_packet_sc_time=apid_preceding_packet_sc_time,
                   apid_following_packet_sc_time=apid_following_packet_sc_time,
                   apid_preceding_packet_esh_time=apid_preceding_packet_esh_time,
                   apid_following_packet_esh_time=apid_following_packet_esh_time)

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        return libera_db_models.CrApidSscGap(
            first_missing_ssc=self.apid_gap_first_missing_ssc_packet,
            gap_byte_offset=self.apid_gap_byte_offset,
            n_missing_sscs=self.apid_num_ssc_packets_missed,
            preceding_packet_sc_time=self.apid_preceding_packet_sc_time,
            following_packet_sc_time=self.apid_following_packet_sc_time,
            preceding_packet_utc_time=self.apid_preceding_packet_utc,
            following_packet_utc_time=self.apid_following_packet_utc,
            preceding_packet_esh_time=self.apid_preceding_packet_esh_time,
            following_packet_esh_time=self.apid_following_packet_esh_time
        )


class VCIDFromConstructionRecord:
    """
    Object representation of the information of Virtual Channel ID (VCID). This corresponds to a database table and
    connects to a Construction Record (CR). This object is created as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, vcid_scid: int):
        self.vcid_scid = vcid_scid

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """Constructor method for creating during Construction record reading"""
        vcid_scid = cr_bitstream.read("uint:16")
        return cls(vcid_scid=vcid_scid)

    @property
    def scid(self):
        """Property that contains the SCID alone"""
        bytes_vcdu = self.vcid_scid.to_bytes(2, 'big')
        vcdu_read = ConstBitStream(bytes_vcdu)
        # Read unused data
        vcdu_read.read("uint:2")
        return vcdu_read.read("uint:8")

    @property
    def vcid(self):
        """Property that contains the APID alone"""
        bytes_vcdu = self.vcid_scid.to_bytes(2, 'big')
        vcdu_read = ConstBitStream(bytes_vcdu)
        # Read unused data
        vcdu_read.read("uint:10")
        return vcdu_read.read("uint:6")

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        return libera_db_models.CrApidVcid(
            scid_vcid=self.vcid_scid
        )


class APIDFromConstructionRecord:
    """
    Object representation of the information of Application IDs (APID) from within a Construction Record (CR). This
    corresponds to a database table and connects to a CR. This object is created as part of reading in a CR and thus
    requires an open bitstream to read from.
    """
    def __init__(self, apid_scid: int,
                 apid_byte_offset: int,
                 apid_vcid_count: int,
                 vcids_list: list,
                 apid_ssc_gap_count: int,
                 apid_ssc_gaps_list: list,
                 edos_generated_fill_data_count: int,
                 edos_generated_fill_data_list: list,
                 edos_generated_octet_count: int,
                 ssc_length_discrepancy_count: int,
                 ssc_length_discrepancy_list: list,
                 first_packet_sc_time: int,
                 last_packet_sc_time: int,
                 first_packet_esh_time: int,
                 last_packet_esh_time: int,
                 vcdu_error_packet_count: int,
                 count_in_the_data_set: int,
                 apid_size_octets: int):
        self.apid_scid = apid_scid
        self.apid_byte_offset = apid_byte_offset

        # For this APID, identify the Virtual Channel Identification (VCID(s))
        self.apid_vcid_count = apid_vcid_count
        self.vcids_list = vcids_list

        # List missing packets SSCs for the PDS
        self.apid_ssc_gap_count = apid_ssc_gap_count
        self.apid_ssc_gaps_list = apid_ssc_gaps_list

        # For this APID, list packets containing EDOS generated fill data
        self.edos_generated_fill_data_count = edos_generated_fill_data_count
        self.edos_generated_fill_data_list = edos_generated_fill_data_list

        self.edos_generated_octet_count = edos_generated_octet_count
        # For the packets with length discrepancy
        self.ssc_length_discrepancy_count = ssc_length_discrepancy_count
        self.ssc_length_discrepancy_list = ssc_length_discrepancy_list

        self.first_packet_sc_time = first_packet_sc_time
        self.last_packet_sc_time = last_packet_sc_time
        self.first_packet_esh_time = first_packet_esh_time
        self.last_packet_esh_time = last_packet_esh_time

        self.first_packet_time_utc = convert_cds_integer_to_datetime(first_packet_sc_time)
        self.last_packet_time_utc = convert_cds_integer_to_datetime(last_packet_sc_time)

        self.vcdu_error_packet_count = vcdu_error_packet_count

        # This is not well labeled in the ICD (24-17)
        self.count_in_the_data_set = count_in_the_data_set
        self.apid_size_octets = apid_size_octets

    @classmethod
    def from_bitstream(cls, cr_bitstream: ConstBitStream):
        """Constructor method for creating during Construction record reading"""
        # Read unused data
        cr_bitstream.read("uint:8")
        apid_scid = cr_bitstream.read("uint:24")

        apid_byte_offset = cr_bitstream.read("uint:64")
        # Read unused data
        cr_bitstream.read("uint:24")

        # For this APID, identify the Virtual Channel Identification (VCID(s))
        apid_vcid_count = cr_bitstream.read("uint:8")
        vcids_list = []
        for _ in range(apid_vcid_count):
            # Read unused data
            cr_bitstream.read("uint:16")
            vcids_list.append(VCIDFromConstructionRecord.from_bitstream(cr_bitstream))

        # List missing packets SSCs for the PDS
        apid_ssc_gap_count = cr_bitstream.read("uint:32")
        apid_ssc_gaps_list = []
        for _ in range(apid_ssc_gap_count):
            apid_ssc_gaps_list.append(SSCGapInformationFromConstructionRecord.from_bitstream(cr_bitstream))

        # For this APID, list packets containing EDOS generated fill data
        edos_generated_fill_data_count = cr_bitstream.read("uint:32")
        edos_generated_fill_data_list = []
        for _ in range(edos_generated_fill_data_count):
            edos_generated_fill_data_list.append(EDOSGeneratedFillDataFromAPID.from_bitstream(cr_bitstream))

        edos_generated_octet_count = cr_bitstream.read("uint:64")
        # For the packets with length discrepancy
        ssc_length_discrepancy_count = cr_bitstream.read("uint:32")
        ssc_length_discrepancy_list = []
        for _ in range(ssc_length_discrepancy_count):
            ssc_length_discrepancy_list.append(SSCLengthDiscrepancy.from_bitstream(cr_bitstream))

        first_packet_sc_time = cr_bitstream.read("uint:64")
        last_packet_sc_time = cr_bitstream.read("uint:64")
        first_packet_esh_time = cr_bitstream.read("uint:64")
        last_packet_esh_time = cr_bitstream.read("uint:64")

        vcdu_error_packet_count = cr_bitstream.read("uint:32")
        # This is not well labeled in the ICD (24-17)
        count_in_the_data_set = cr_bitstream.read("uint:32")
        apid_size_octets = cr_bitstream.read("uint:64")
        # Read unused data
        cr_bitstream.read("uint:64")
        # Call the constructor method
        return cls(apid_scid=apid_scid,
                   apid_byte_offset=apid_byte_offset,
                   apid_vcid_count=apid_vcid_count,
                   vcids_list=vcids_list,
                   apid_ssc_gap_count=apid_ssc_gap_count,
                   apid_ssc_gaps_list=apid_ssc_gaps_list,
                   edos_generated_fill_data_count=edos_generated_fill_data_count,
                   edos_generated_fill_data_list=edos_generated_fill_data_list,
                   edos_generated_octet_count=edos_generated_octet_count,
                   ssc_length_discrepancy_count=ssc_length_discrepancy_count,
                   ssc_length_discrepancy_list=ssc_length_discrepancy_list,
                   first_packet_sc_time=first_packet_sc_time,
                   last_packet_sc_time=last_packet_sc_time,
                   first_packet_esh_time=first_packet_esh_time,
                   last_packet_esh_time=last_packet_esh_time,
                   vcdu_error_packet_count=vcdu_error_packet_count,
                   count_in_the_data_set=count_in_the_data_set,
                   apid_size_octets=apid_size_octets)

    @property
    def scid(self):
        """Property that contains the SCID alone"""
        bytes_scid = self.apid_scid.to_bytes(3, 'big')
        scid_read = ConstBitStream(bytes_scid)
        return scid_read.read("uint:8")

    @property
    def apid(self):
        """Property that contains the APID alone"""
        bytes_scid = self.apid_scid.to_bytes(3, 'big')
        scid_read = ConstBitStream(bytes_scid)
        # Read unused data
        scid_read.read("uint:13")
        return scid_read.read("uint:11")

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        vcids = []
        for i in range(self.apid_vcid_count):
            vcids.append(self.vcids_list[i].to_orm())
        ssc_gaps = []
        for i in range(self.apid_ssc_gap_count):
            ssc_gaps.append(self.apid_ssc_gaps_list[i].to_orm())
        edos_fill_data = []
        for i in range(self.edos_generated_fill_data_count):
            edos_fill_data.append(self.edos_generated_fill_data_list[i].to_orm())
        ssc_length_discrep = []
        for i in range(self.ssc_length_discrepancy_count):
            ssc_length_discrep.append(self.ssc_length_discrepancy_list[i].to_orm())

        return libera_db_models.CrApid(
            scid_apid=self.apid_scid,
            byte_offset=self.apid_byte_offset,
            n_vcids=self.apid_vcid_count,
            vcids=vcids,
            n_ssc_gaps=self.apid_ssc_gap_count,
            ssc_gaps=ssc_gaps,
            n_edos_generated_fill_data=self.edos_generated_fill_data_count,
            edos_fill_data=edos_fill_data,
            count_edos_generated_octets=self.edos_generated_octet_count,
            n_length_discrepancy_packets=self.ssc_length_discrepancy_count,
            ssc_length_discrepancies=ssc_length_discrep,
            first_packet_sc_time=self.first_packet_sc_time,
            last_packet_sc_time=self.last_packet_sc_time,
            esh_first_packet_time=self.first_packet_esh_time,
            esh_last_packet_time=self.last_packet_esh_time,
            first_packet_utc_time=self.first_packet_time_utc,
            last_packet_utc_time=self.last_packet_time_utc,
            n_vcdu_corrected_packets=self.vcdu_error_packet_count,
            n_in_the_data_set=self.count_in_the_data_set,
            n_octect_in_apid=self.apid_size_octets
        )


class ConstructionRecordError(Exception):
    """Generic exception related to construction record file handling"""
    pass


class ConstructionRecord:
    """
    Object representation of a JPSS Construction Record (CR) including objects for all the other classes
    in this file to be stored in a database.
    """
    def __init__(self, file_name: str,
                 edos_version: int,
                 construction_record_type: int,
                 cr_id: str,
                 test_flag: bool,
                 scs_num_start_stop_times: int,
                 scs_start_stop_times_list: list,
                 pds_num_bytes_fill_data: int,
                 pds_packet_length_mismatch_count: int,
                 pds_first_packet_sc_time: int,
                 pds_last_packet_sc_time: int,
                 pds_first_packet_esh_time: int,
                 pds_last_packet_esh_time: int,
                 pds_rs_corrected_count: int,
                 pds_packet_count: int,
                 pds_size: int,
                 pds_discontinuities_count: int,
                 pds_completion_time_bytes: int,
                 apid_count: int,
                 apid_data_list: list,
                 pds_file_count: int,
                 pds_files_list: list):
        self.file_name = file_name
        self.edos_version = edos_version
        # Construction Record type 1 is for PDS
        self.construction_record_type = construction_record_type
        self.cr_id = cr_id
        self.test_flag = test_flag
        self.scs_num_start_stop_times = scs_num_start_stop_times
        self.scs_start_stop_times_list = scs_start_stop_times_list

        self.pds_num_bytes_fill_data = pds_num_bytes_fill_data
        self.pds_packet_length_mismatch_count = pds_packet_length_mismatch_count
        self.pds_first_packet_sc_time = pds_first_packet_sc_time
        self.pds_first_packet_utc_time = convert_cds_integer_to_datetime(self.pds_first_packet_sc_time)
        self.pds_last_packet_sc_time = pds_last_packet_sc_time
        self.pds_last_packet_utc_time = convert_cds_integer_to_datetime(self.pds_last_packet_sc_time)
        self.pds_first_packet_esh_time = pds_first_packet_esh_time
        self.pds_last_packet_esh_time = pds_last_packet_esh_time
        self.pds_rs_corrected_count = pds_rs_corrected_count
        self.pds_packet_count = pds_packet_count
        self.pds_size = pds_size
        self.pds_discontinuities_count = pds_discontinuities_count
        self.pds_completion_time_bytes = pds_completion_time_bytes

        # For the PDS, identify the APIDs and their associated information.
        self.apid_count = apid_count
        self.apid_data_list = apid_data_list

        # Identify files that store this PDS
        self.pds_file_count = pds_file_count
        self.pds_files_list = pds_files_list

    @classmethod
    def from_file(cls, filepath: str or Path or S3Path):
        """Read a construction record file and return a ConstructionRecord object (factory method).

            Parameters
            ----------
            filepath : str or Path or S3Path
                Location of construction record file to read.

            Returns
            -------
            : ConstructionRecord
        """
        with smart_open(filepath) as const_record_file:
            cr_bitstream = ConstBitStream(const_record_file)
            # Note this will validate that the filename being read in is a valid L0 name
            l0_filename = L0Filename(filepath).path.name
            # Any Posix Path will have the member 'name' so disable pylint on this line
            edos_version = cr_bitstream.read("uint:16")

            # Construction Record type 1 is for PDS
            construction_record_type = cr_bitstream.read("uint:8")
            # Read unused data
            cr_bitstream.read("uint:8")

            cr_id = (cr_bitstream.read("bytes:36")).decode()
            if f"{cr_id}.PDS" != l0_filename:
                return ConstructionRecordError(f"The filename read, {l0_filename}, does not match the cr_id in "
                                               f"side the file, {cr_id}")
            # Read unused data
            cr_bitstream.read("uint:7")
            test_flag = cr_bitstream.read("bool")
            # Read unused data
            cr_bitstream.read("uint:8")
            cr_bitstream.read("uint:64")

            scs_num_start_stop_times = cr_bitstream.read("uint:16")
            scs_start_stop_times_list = []
            for _ in range(scs_num_start_stop_times):
                scs_start_stop_times_list.append(SCSStartStopTimes.from_bitstream(cr_bitstream))

            pds_num_bytes_fill_data = cr_bitstream.read("uint:64")
            pds_packet_length_mismatch_count = cr_bitstream.read("uint:32")
            pds_first_packet_sc_time = cr_bitstream.read("uint:64")
            pds_last_packet_sc_time = cr_bitstream.read("uint:64")
            pds_first_packet_esh_time = cr_bitstream.read("uint:64")
            pds_last_packet_esh_time = cr_bitstream.read("uint:64")
            pds_rs_corrected_count = cr_bitstream.read("uint:32")
            pds_packet_count = cr_bitstream.read("uint:32")
            pds_size = cr_bitstream.read("uint:64")
            pds_discontinuities_count = cr_bitstream.read("uint:32")
            pds_completion_time_bytes = cr_bitstream.read("uint:64")
            # Read unused data
            cr_bitstream.read("uint:56")

            # For the PDS, identify the APIDs and their associated information.
            apid_count = cr_bitstream.read("uint:8")
            apid_data_list = []
            for _ in range(apid_count):
                apid_data_list.append(APIDFromConstructionRecord.from_bitstream(cr_bitstream))

            # Read unused data
            cr_bitstream.read("uint:24")
            # Identify files that store this PDS
            pds_file_count = cr_bitstream.read("uint:8")
            pds_files_list = []
            for _ in range(pds_file_count):
                pds_files_list.append(PDSRecord.from_bitstream(cr_bitstream))
            return cls(file_name=l0_filename,
                       edos_version=edos_version,
                       construction_record_type=construction_record_type,
                       cr_id=cr_id,
                       test_flag=test_flag,
                       scs_num_start_stop_times=scs_num_start_stop_times,
                       scs_start_stop_times_list=scs_start_stop_times_list,
                       pds_num_bytes_fill_data=pds_num_bytes_fill_data,
                       pds_packet_length_mismatch_count=pds_packet_length_mismatch_count,
                       pds_first_packet_sc_time=pds_first_packet_sc_time,
                       pds_last_packet_sc_time=pds_last_packet_sc_time,
                       pds_first_packet_esh_time=pds_first_packet_esh_time,
                       pds_last_packet_esh_time=pds_last_packet_esh_time,
                       pds_rs_corrected_count=pds_rs_corrected_count,
                       pds_packet_count=pds_packet_count,
                       pds_size=pds_size,
                       pds_discontinuities_count=pds_discontinuities_count,
                       pds_completion_time_bytes=pds_completion_time_bytes,
                       apid_count=apid_count,
                       apid_data_list=apid_data_list,
                       pds_file_count=pds_file_count,
                       pds_files_list=pds_files_list)

    @property
    def edos_version_major(self):
        """Property that contains the major version number of the EDOS software alone"""
        edos_bytes = self.edos_version.to_bytes(2, 'big')
        edos_read = ConstBitStream(edos_bytes)
        return edos_read.read("uint:8")

    @property
    def edos_version_release(self):
        """Property that contains the major version release number of the EDOS software alone"""
        edos_bytes = self.edos_version.to_bytes(2, 'big')
        edos_read = ConstBitStream(edos_bytes)
        edos_read.read("uint:8")
        return edos_read.read("uint:8")

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        # TODO currently we also have the filename where the file was read from as part of the object.
        # Should we also save this?
        orm_file_name = f"{self.cr_id}.PDS"
        scs_start_stops = []
        for i in range(self.scs_num_start_stop_times):
            scs_start_stops.append(self.scs_start_stop_times_list[i].to_orm())
        apids = []
        for i in range(self.apid_count):
            apids.append(self.apid_data_list[i].to_orm())
        pds_files = []
        orm_pds_file_count = self.pds_file_count
        for pds_file in self.pds_files_list:
            if pds_file.filename == orm_file_name:
                orm_pds_file_count = self.pds_file_count-1
                # Don't append this construction record to the pds list as one of the pds files.
                continue
            # This is a pds file that is not the construction record itself. Add this to the list
            pds_files.append(pds_file.to_orm())
        return libera_db_models.Cr(
            file_name=orm_file_name,
            edos_software_version=self.edos_version,
            construction_record_type=self.construction_record_type,
            test_flag=self.test_flag,
            n_scs_start_stops=self.scs_num_start_stop_times,
            scs_start_stop_times=scs_start_stops,
            n_bytes_fill_data=self.pds_num_bytes_fill_data,
            n_length_mismatches=self.pds_packet_length_mismatch_count,
            first_packet_sc_time=self.pds_first_packet_sc_time,
            last_packet_sc_time=self.pds_last_packet_sc_time,
            first_packet_utc_time=self.pds_first_packet_utc_time,
            last_packet_utc_time=self.pds_last_packet_utc_time,
            first_packet_esh_time=self.pds_first_packet_esh_time,
            last_packet_esh_time=self.pds_last_packet_esh_time,
            n_rs_corrections=self.pds_rs_corrected_count,
            n_packets=self.pds_packet_count,
            size_bytes=self.pds_size,
            n_ssc_discontinuities=self.pds_discontinuities_count,
            completion_time=self.pds_completion_time_bytes,
            n_apids=self.apid_count,
            apids=apids,
            n_pds_files=orm_pds_file_count,
            pds_files=pds_files
        )
