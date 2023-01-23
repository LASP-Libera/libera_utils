"""Module for manifest file handling"""
# Standard
import logging
from pathlib import Path
from bitstring import ConstBitStream
from datetime import datetime, timedelta
import pytz
# Installed
from cloudpathlib import S3Path, AnyPath
# Local
from libera_utils.io.smart_open import smart_open

logger = logging.getLogger(__name__)


def convert_bytes_to_cds_time(satellite_time: int):
    reference_date = datetime(1958, 1, 1, 0, 0, 0, 0, pytz.UTC)
    byte_data = satellite_time.to_bytes(8, 'big')
    byte_days = int.from_bytes([byte_data[0], byte_data[1]], byteorder="big")
    byte_millisec = int.from_bytes([byte_data[2], byte_data[3],
                                    byte_data[4], byte_data[5]],
                                   byteorder="big")
    byte_microsec = int.from_bytes([byte_data[6], byte_data[7]], byteorder="big")

    cds_time = (reference_date +
                timedelta(days=byte_days) +
                timedelta(milliseconds=byte_millisec) +
                timedelta(microseconds=byte_microsec))
    return cds_time


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


class SSCLengthDiscrepancy:
    """
    Object representation of the information of the length discrepancy in an SSC. This corresponds to a database table
    and connects to a Construction Record (CR). This object is created as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.ssc_length_discrepancy = cr_bitstream.read("uint:32")


class SCSStartStopTimes:
    """
    Object representation of the information of Spacecraft Session (SCS) start and stop times of data. This
    corresponds to a database table and connects to a Construction Record (CR). This object is created as part of
    reading in a CR and thus requires an open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.SCS_start_time_sc_time = cr_bitstream.read("uint:64")
        self.SCS_start_time_utc = convert_bytes_to_cds_time(self.SCS_start_time_sc_time)
        self.SCS_stop_time_sc_time = cr_bitstream.read("uint:64")
        self.SCS_stop_time_utc = convert_bytes_to_cds_time(self.SCS_stop_time_sc_time)


class APIDFromPDSFromConstructionRecord:
    """
    Object representation of the information of Application IDs (APID) from within a Production Data Set (PDS). This
    corresponds to a database table and connects to a Construction Record (CR). This object is created as part of
    reading in a CR and thus requires an open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        fill_hold = [cr_bitstream.read("uint:8")]
        self.SCID_APID = cr_bitstream.read("uint:24")
        self.APID_first_packet_sc_time = cr_bitstream.read("uint:64")
        self.APID_first_packet_utc = convert_bytes_to_cds_time(self.APID_first_packet_sc_time)
        self.APID_last_packet_byte = cr_bitstream.read("uint:64")
        self.APID_last_packet_utc = convert_bytes_to_cds_time(self.APID_last_packet_byte)
        fill_hold.append(cr_bitstream.read("uint:32"))

    def print_scid_apid(self):
        bytes_scid = self.SCID_APID.to_bytes(3,'big')
        scid_read = ConstBitStream(bytes_scid)
        dataset_scid = scid_read.read("uint:8")
        fill_data = scid_read.read("uint:5")
        dataset_apid = scid_read.read("uint:11")
        print(f"SCID: {dataset_scid} and APID: {dataset_apid}")


class PDSFileFromConstructionRecord:
    """
    Object representation of the information directly related to a Production Data Set (PDS). This corresponds to a
    database table and connects to a Construction Record (CR). This object is created as part of reading in a CR and
    requires an open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.PDS_filename = (cr_bitstream.read("bytes:40")).decode()
        fill_hold = cr_bitstream.read("uint:24")

        # This is quoted in 25-4 as a "one-up" counter with values of 1 to 3. However, there is a situation when the
        # value can be 0, and then there is one entry with complete data as 0's throughout. To account for this take
        # the maximum of the value and 1 to ensure if a 0 is there at least one full entry of 0's is read.
        self.APID_count_this_file = max(cr_bitstream.read("uint:8"), 1)
        self.APID_this_file = []
        for count in range(self.APID_count_this_file):
            self.APID_this_file.append(APIDFromPDSFromConstructionRecord(cr_bitstream))
        return


class SSCGapInformationFromConstructionRecord:
    """
    Object representation of the information of Spacecraft Contact Sessions (SCS). This corresponds to a database table
    and connects to a Construction Record (CR). This object iscreated as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.APID_gap_first_missing_ssc_packet = cr_bitstream.read("uint:32")
        self.APID_gap_byte_offset = cr_bitstream.read("uint:64")
        self.APID_num_ssc_packets_missed = cr_bitstream.read("uint:32")

        # These are not labeled in the ICD document and so this is a guess based on other patterns in the ICD
        sc_packet_before_time = cr_bitstream.read("uint:64")
        sc_packet_after_time = cr_bitstream.read("uint:64")
        self.APID_sc_preceding_packet_esh.append(sc_packet_before_time)
        self.APID_sc_following_packet_esh.append(sc_packet_after_time)
        self.APID_preceding_packet_utc.append(convert_bytes_to_cds_time(sc_packet_before_time))
        self.APID_following_packet_utc.append(convert_bytes_to_cds_time(sc_packet_after_time))

        self.APID_ESH_preceding_packet_esh.append(cr_bitstream.read("uint:64"))
        self.APID_ESH_following_packet_esh.append(cr_bitstream.read("uint:64"))


class VCIDFromConstructionRecord:
    """
    Object representation of the information of Virtual Channel ID (VCID). This corresponds to a database table and
    connects to a Construction Record (CR). This object is created as part of reading in a CR and thus requires an
    open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.vcid_scid = cr_bitstream.read("uint:16")

    def print_scid_apid(self):
        bytes_vcdu = self.vcid_scid.to_bytes(2, 'big')
        vcdu_read = ConstBitStream(bytes_vcdu)
        fill_hold = self.vcid_scid.read("uint:2")
        scid = vcdu_read.read("uint:8")
        vcid = vcdu_read.read("uint:6")
        print(f"VCID: {vcid} and SCID: {scid}")


class APIDFromConstructionRecord:
    """
    Object representation of the information of Application IDs (APID) from within a Construction Record (CR). This
    corresponds to a database table and connects to a CR. This object is created as part of reading in a CR and thus
    requires an open bitstream to read from.
    """
    def __init__(self, cr_bitstream: ConstBitStream):
        self.APID_fill_hold = []
        self.APID_fill_hold.append(cr_bitstream.read("uint:8"))
        self.APID_SCID = cr_bitstream.read("uint:24")

        self.APID_byte_offset = cr_bitstream.read("uint:64")
        self.APID_fill_hold.append(cr_bitstream.read("uint:24"))

        # For this APID, identify the Virtual Channel Identification (VCID(s))
        self.APID_VCID_count = cr_bitstream.read("uint:8")
        self.VCIDs_list = []
        for count in range(self.APID_VCID_count):
            self.APID_fill_hold.append(cr_bitstream.read("uint:16"))
            self.VCIDs_list.append(VCIDFromConstructionRecord(cr_bitstream))

        # List missing packets SSCs for the PDS
        self.APID_SSC_gap_count = cr_bitstream.read("uint:32")
        self.APID_SSC_gaps_list = []
        for count in range(self.APID_SSC_discontinuity_count):
            self.APID_SSC_gaps_list.append(SSCGapInformationFromConstructionRecord(cr_bitstream))

        # For this APID, list packets containing EDOS generated fill data
        self.EDOS_generated_fill_data_count = cr_bitstream.read("uint:32")
        self.EDOS_generated_fill_data_list = []
        for count in range(self.EDOS_generated_fill_data_count):
            self.EDOS_generated_fill_data_list.append(EDOSGeneratedFillDataFromAPID(cr_bitstream))

        self.EDOS_generated_octet_count = cr_bitstream.read("uint:64")
        # For the packets with length discrepancy
        self.packets_with_discrepancy_count = cr_bitstream.read("uint:32")
        self.SSC_length_discrepancy_list = []
        for count in range(self.packets_with_discrepancy_count):
            self.SSC_length_discrepancy_list.append(SSCLengthDiscrepancy(cr_bitstream))

        self.first_packet_sc_time = cr_bitstream.read("uint:64")
        self.last_packet_sc_time = cr_bitstream.read("uint:64")
        self.ESH_first_packet_time = cr_bitstream.read("uint:64")
        self.ESH_last_packet_time = cr_bitstream.read("uint:64")

        self.first_packet_time_utc = convert_bytes_to_cds_time(self.first_packet_sc_time)
        self.last_packet_time_utc = convert_bytes_to_cds_time(self.last_packet_sc_time)

        self.VCDU_error_packet_count = cr_bitstream.read("uint:32")

        # This is not well labeled in the ICD (24-17)
        self.count_in_the_data_set = cr_bitstream.read("uint:32")

        self.APID_size_octets = cr_bitstream.read("uint:64")
        self.APID_fill_hold.append(cr_bitstream.read("uint:64"))

    def print_scid_apid(self):
        bytes_scid = self.APID_SCID.to_bytes(3, 'big')
        scid_read = ConstBitStream(bytes_scid)
        dataset_scid = scid_read.read("uint:8")
        fill_data = scid_read.read("uint:5")
        dataset_apid = scid_read.read("uint:11")
        print(f"SCID: {dataset_scid} and APID: {dataset_apid}")


class ConstructionRecordError(Exception):
    """Generic exception related to construction record file handling"""
    pass


class ConstructionRecord:
    """
    Object representation of a JPSS Construction Record (CR) including objects for all the other classes
    in this file to be stored in a database.
    """

    def __init__(self, filepath: str or Path or S3Path):
        self.filepath = filepath
        with smart_open(filepath) as const_record_file:
            cr_bitstream = ConstBitStream(const_record_file)
            self.EDOS_version_original = cr_bitstream.read("bytes:2")
            self.EDOS_version_int = int.from_bytes(self.EDOS_version_original, "big")
            self.EDOS_version_major = self.EDOS_version_original[0]
            self.EDOS_version_release = self.EDOS_version_original[1]
            # Construction Record type 1 is for PDS
            self.Construction_Record_type = cr_bitstream.read("uint:8")
            self.fill_hold = []
            self.fill_hold.append(cr_bitstream.read("uint:8"))
            self.CR_ID = (cr_bitstream.read("bytes:36")).decode()
            self.fill_hold.append(cr_bitstream.read("uint:7"))
            self.test_flag = cr_bitstream.read("bool")
            self.fill_hold.append(cr_bitstream.read("uint:8"))
            self.fill_hold.append(cr_bitstream.read("uint:64"))

            self.SCS_num_start_stop_times = cr_bitstream.read("uint:16")
            self.SCS_start_stop_times_list = []
            for count in range(self.SCS_num_start_stop_times):
                self.SCS_start_stop_times_list.append(SCSStartStopTimes(cr_bitstream))

            self.PDS_num_bytes_fill_data = cr_bitstream.read("uint:64")
            self.PDS_packet_length_mismatch_count = cr_bitstream.read("uint:32")
            self.PDS_first_packet_sc_time = cr_bitstream.read("uint:64")
            self.PDS_first_packet_datetime = convert_bytes_to_cds_time(self.PDS_first_packet_sc_time)
            self.PDS_last_packet_sc_time = cr_bitstream.read("uint:64")
            self.PDS_last_packet_datetime = convert_bytes_to_cds_time(self.PDS_last_packet_sc_time)
            self.PDS_ESH_first_packet_sc_time = cr_bitstream.read("uint:64")
            self.PDS_ESH_first_packet_datetime = convert_bytes_to_cds_time(self.PDS_ESH_first_packet_sc_time)
            self.PDS_ESH_last_packet_sc_time = cr_bitstream.read("uint:64")
            self.PDS_ESH_last_packet_datetime = convert_bytes_to_cds_time(self.PDS_ESH_last_packet_sc_time)
            self.PDS_rs_corrected_count = cr_bitstream.read("uint:32")
            self.PDS_packet_count = cr_bitstream.read("uint:32")
            self.PDS_size = cr_bitstream.read("uint:64")
            self.PDS_discontinuities_count = cr_bitstream.read("uint:32")
            self.PDS_completion_time_bytes = cr_bitstream.read("uint:64")
            self.fill_hold.append(cr_bitstream.read("uint:56"))

            # For the PDS, identify the APIDs and their associated information.
            self.APID_count = cr_bitstream.read("uint:8")
            self.APID_data_list = []
            for count in range(self.APID_count):
                self.APID_data_list.append(APIDFromConstructionRecord(cr_bitstream))

            # Identify files that store this PDS
            self.fill_hold.append(cr_bitstream.read("uint:24"))
            self.PDS_file_count = cr_bitstream.read("uint:8")
            self.PDS_files_list = []
            for file in range(self.PDS_file_count):
                self.PDS_files_list.append(PDSFileFromConstructionRecord(cr_bitstream))
