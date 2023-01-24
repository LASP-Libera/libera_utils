"""Module for manifest file handling"""
# Standard
import logging
from pathlib import Path
from bitstring import ConstBitStream
# Installed
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils.io.smart_open import smart_open
import libera_utils.db.models as libera_db_models
from libera_utils.time import convert_cds_integer_to_datetime

logger = logging.getLogger(__name__)


class EDOSGeneratedFillDataFromAPID:
    """
    Object representation of the information pertaining to which data in a Production Data Set (PDS) are generated and
    filled by EDOS. This corresponds to a database table and connects to a Construction Record (CR). This object is
    created as part of reading in a CR and requires an open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.ssc_with_generated_data = cr_bitstream.read("uint:32")
        self.filled_byte_offset = cr_bitstream.read("uint:64")
        self.index_to_fill_octet = cr_bitstream.read("uint:32")

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
    def __init__(self, cr_bitstream: ConstBitStream):
        self.ssc_length_discrepancy = cr_bitstream.read("uint:32")

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
    def __init__(self, cr_bitstream: ConstBitStream):
        self.scs_start_time_sc_time = cr_bitstream.read("uint:64")
        self.scs_start_time_utc = convert_cds_integer_to_datetime(self.scs_start_time_sc_time)
        self.scs_stop_time_sc_time = cr_bitstream.read("uint:64")
        self.scs_stop_time_utc = convert_cds_integer_to_datetime(self.scs_stop_time_sc_time)

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
    def __init__(self, cr_bitstream: ConstBitStream):
        # Read unused data
        cr_bitstream.read("uint:8")
        self.scid_apid = cr_bitstream.read("uint:24")
        self.apid_first_packet_sc_time = cr_bitstream.read("uint:64")
        self.apid_first_packet_utc = convert_cds_integer_to_datetime(self.apid_first_packet_sc_time)
        self.apid_last_packet_sc_time = cr_bitstream.read("uint:64")
        self.apid_last_packet_utc = convert_cds_integer_to_datetime(self.apid_last_packet_sc_time)
        # Read unused data
        cr_bitstream.read("uint:32")

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


class PDSFileFromConstructionRecord:
    """
    Object representation of the information directly related to a Production Data Set (PDS). This corresponds to a
    database table and connects to a Construction Record (CR). This object is created as part of reading in a CR and
    requires an open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.pds_filename = (cr_bitstream.read("bytes:40")).decode()
        # Read unused data
        cr_bitstream.read("uint:24")

        # This is quoted in 25-4 as a "one-up" counter with values of 1 to 3. However, there is a situation when the
        # value can be 0, and then there is one entry with complete data as 0's throughout. To account for this take
        # the maximum of the value and 1 to ensure if a 0 is there at least one full entry of 0's is read.
        self.apid_count_this_file = max(cr_bitstream.read("uint:8"), 1)
        self.apid_this_file = []
        for _ in range(self.apid_count_this_file):
            self.apid_this_file.append(APIDFromPDSFromConstructionRecord(cr_bitstream))

    def to_orm(self):
        """Convert this class instance to a corresponding ORM object for entry into the database"""
        apids_list = []
        for i in range(self.apid_count_this_file):
            apids_list.append(self.apid_this_file[i].to_orm())
        return libera_db_models.PdsFile(
            file_name=self.pds_filename,
            apids=apids_list
        )


class SSCGapInformationFromConstructionRecord:
    """
    Object representation of the information of Spacecraft Contact Sessions (SCS). This corresponds to a database table
    and connects to a Construction Record (CR). This object iscreated as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.apid_gap_first_missing_ssc_packet = cr_bitstream.read("uint:32")
        self.apid_gap_byte_offset = cr_bitstream.read("uint:64")
        self.apid_num_ssc_packets_missed = cr_bitstream.read("uint:32")

        # These are not labeled in the ICD document and so this is a guess based on other patterns in the ICD
        sc_packet_before_time = cr_bitstream.read("uint:64")
        sc_packet_after_time = cr_bitstream.read("uint:64")
        self.apid_preceding_packet_sc_time = sc_packet_before_time
        self.apid_following_packet_sc_time = sc_packet_after_time
        self.apid_preceding_packet_utc = convert_cds_integer_to_datetime(sc_packet_before_time)
        self.apid_following_packet_utc = convert_cds_integer_to_datetime(sc_packet_after_time)

        self.apid_preceding_packet_esh_time = cr_bitstream.read("uint:64")
        self.apid_following_packet_esh_time = cr_bitstream.read("uint:64")

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
    def __init__(self, cr_bitstream: ConstBitStream):
        self.vcid_scid = cr_bitstream.read("uint:16")

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
    def __init__(self, cr_bitstream: ConstBitStream):
        # Read unused data
        cr_bitstream.read("uint:8")
        self.apid_scid = cr_bitstream.read("uint:24")

        self.apid_byte_offset = cr_bitstream.read("uint:64")
        # Read unused data
        cr_bitstream.read("uint:24")

        # For this APID, identify the Virtual Channel Identification (VCID(s))
        self.apid_vcid_count = cr_bitstream.read("uint:8")
        self.vcids_list = []
        for _ in range(self.apid_vcid_count):
            # Read unused data
            cr_bitstream.read("uint:16")
            self.vcids_list.append(VCIDFromConstructionRecord(cr_bitstream))

        # List missing packets SSCs for the PDS
        self.apid_ssc_gap_count = cr_bitstream.read("uint:32")
        self.apid_ssc_gaps_list = []
        for _ in range(self.apid_ssc_gap_count):
            self.apid_ssc_gaps_list.append(SSCGapInformationFromConstructionRecord(cr_bitstream))

        # For this APID, list packets containing EDOS generated fill data
        self.edos_generated_fill_data_count = cr_bitstream.read("uint:32")
        self.edos_generated_fill_data_list = []
        for _ in range(self.edos_generated_fill_data_count):
            self.edos_generated_fill_data_list.append(EDOSGeneratedFillDataFromAPID(cr_bitstream))

        self.edos_generated_octet_count = cr_bitstream.read("uint:64")
        # For the packets with length discrepancy
        self.ssc_length_discrepancy_count = cr_bitstream.read("uint:32")
        self.ssc_length_discrepancy_list = []
        for _ in range(self.ssc_length_discrepancy_count):
            self.ssc_length_discrepancy_list.append(SSCLengthDiscrepancy(cr_bitstream))

        self.first_packet_sc_time = cr_bitstream.read("uint:64")
        self.last_packet_sc_time = cr_bitstream.read("uint:64")
        self.first_packet_esh_time = cr_bitstream.read("uint:64")
        self.last_packet_esh_time = cr_bitstream.read("uint:64")

        self.first_packet_time_utc = convert_cds_integer_to_datetime(self.first_packet_sc_time)
        self.last_packet_time_utc = convert_cds_integer_to_datetime(self.last_packet_sc_time)

        self.vcdu_error_packet_count = cr_bitstream.read("uint:32")

        # This is not well labeled in the ICD (24-17)
        self.count_in_the_data_set = cr_bitstream.read("uint:32")

        self.apid_size_octets = cr_bitstream.read("uint:64")
        # Read unused data
        cr_bitstream.read("uint:64")

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
            return cls(filepath, cr_bitstream)

    def __init__(self, filepath: str or Path or S3Path, cr_bitstream: ConstBitStream):
        path_object = AnyPath(filepath)
        # Any Posix Path will have the member 'name' so disable pylint on this line
        self.file_name = path_object.name  # pylint: disable=no-member
        self.edos_version = cr_bitstream.read("uint:16")

        # Construction Record type 1 is for PDS
        self.construction_record_type = cr_bitstream.read("uint:8")
        # Read unused data
        cr_bitstream.read("uint:8")
        self.cr_id = (cr_bitstream.read("bytes:36")).decode()
        # Read unused data
        cr_bitstream.read("uint:7")
        self.test_flag = cr_bitstream.read("bool")
        # Read unused data
        cr_bitstream.read("uint:8")
        cr_bitstream.read("uint:64")

        self.scs_num_start_stop_times = cr_bitstream.read("uint:16")
        self.scs_start_stop_times_list = []
        for _ in range(self.scs_num_start_stop_times):
            self.scs_start_stop_times_list.append(SCSStartStopTimes(cr_bitstream))

        self.pds_num_bytes_fill_data = cr_bitstream.read("uint:64")
        self.pds_packet_length_mismatch_count = cr_bitstream.read("uint:32")
        self.pds_first_packet_sc_time = cr_bitstream.read("uint:64")
        self.pds_first_packet_utc_time = convert_cds_integer_to_datetime(self.pds_first_packet_sc_time)
        self.pds_last_packet_sc_time = cr_bitstream.read("uint:64")
        self.pds_last_packet_utc_time = convert_cds_integer_to_datetime(self.pds_last_packet_sc_time)
        self.pds_first_packet_esh_time = cr_bitstream.read("uint:64")
        self.pds_last_packet_esh_time = cr_bitstream.read("uint:64")
        self.pds_rs_corrected_count = cr_bitstream.read("uint:32")
        self.pds_packet_count = cr_bitstream.read("uint:32")
        self.pds_size = cr_bitstream.read("uint:64")
        self.pds_discontinuities_count = cr_bitstream.read("uint:32")
        self.pds_completion_time_bytes = cr_bitstream.read("uint:64")
        # Read unused data
        cr_bitstream.read("uint:56")

        # For the PDS, identify the APIDs and their associated information.
        self.apid_count = cr_bitstream.read("uint:8")
        self.apid_data_list = []
        for _ in range(self.apid_count):
            self.apid_data_list.append(APIDFromConstructionRecord(cr_bitstream))

        # Read unused data
        cr_bitstream.read("uint:24")
        # Identify files that store this PDS
        self.pds_file_count = cr_bitstream.read("uint:8")
        self.pds_files_list = []
        for _ in range(self.pds_file_count):
            self.pds_files_list.append(PDSFileFromConstructionRecord(cr_bitstream))

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
        scs_start_stops = []
        for i in range(self.scs_num_start_stop_times):
            scs_start_stops.append(self.scs_start_stop_times_list[i].to_orm())
        apids = []
        for i in range(self.apid_count):
            apids.append(self.apid_data_list[i].to_orm())
        pds_files = []
        for i in range(self.pds_file_count):
            pds_files.append(self.pds_files_list[i].to_orm())
        return libera_db_models.Cr(
            file_name=self.file_name,
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
            n_pds_files=self.pds_file_count,
            pds_files=pds_files
        )
