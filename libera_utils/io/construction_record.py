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


def convert_bytes_to_cds_time(byte_data: bytes):
    reference_date = datetime(1958, 1, 1, 0, 0, 0, 0, pytz.UTC)
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
    def __init__(self, cr_bitstream: ConstBitStream):
        self.SSC_with_generated_data = cr_bitstream.read("uint:32")
        self.generated_data_offset_list = cr_bitstream.read("uint:64")
        self.index_to_fill_generated_list = cr_bitstream.read("uint:32")


class PDSFileFromConstructionRecord:
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


class APIDFromPDSFromConstructionRecord:
    def __init__(self, cr_bitstream: ConstBitStream):
        # Should be 0 for construction record
        fill_hold = [cr_bitstream.read("uint:8")]
        # Should be 0 for construction record
        self.APID_SCID_non_CR = cr_bitstream.read("uint:24")
        # Should be 0 for construction record
        self.APID_CDS_time_code_non_CR = cr_bitstream.read("bytes:8")
        self.APID_CDS_time_code_non_CR_datetime = convert_bytes_to_cds_time(self.APID_CDS_time_code_non_CR)
        # Should be 0 for construction record
        self.APID_CCSDS_time_code_non_CR = cr_bitstream.read("bytes:8")
        self.APID_CCSDS_time_code_non_CR_datetime = convert_bytes_to_cds_time(self.APID_CCSDS_time_code_non_CR)
        # Should be 0 for construction record
        fill_hold.append(cr_bitstream.read("uint:32"))


class SSCGapInformationFromConstructionRecord:
    def __init__(self, cr_bitstream: ConstBitStream):
        self.APID_gap_first_missing_packet = cr_bitstream.read("uint:32")
        self.APID_gap_offset = cr_bitstream.read("uint:64")
        self.APID_packets_missed = cr_bitstream.read("uint:32")
        self.APID_missed_headers_1 = cr_bitstream.read("uint:64")
        self.APID_missed_headers_2 = cr_bitstream.read("uint:64")
        packet_before_time = cr_bitstream.read("bytes:8")
        packet_after_time = cr_bitstream.read("bytes:8")
        self.APID_ESH_time_packet_before.append(packet_before_time)
        self.APID_ESH_time_packet_after.append(packet_after_time)
        self.APID_ESH_time_packet_before_date.append(convert_bytes_to_cds_time(packet_before_time))
        self.APID_ESH_time_packet_after_date.append(convert_bytes_to_cds_time(packet_after_time))


class VCIDFromConstructionRecord:
    def __init__(self, cr_bitstream: ConstBitStream):
        self.VCID_SCID_bits = cr_bitstream.read("bits:16")
        fill_hold = self.VCID_SCID_bits.read("uint:2")
        self.SCID = self.VCID_SCID_bits.read("uint:8")
        self.VCID = self.VCID_SCID_bits.read("uint:6")


class PDSStartStopTimes:
    def __init__(self, cr_bitstream: ConstBitStream):
        self.SCS_start_time_original = cr_bitstream.read("bytes:8")
        self.SCS_start_time = convert_bytes_to_cds_time(self.SCS_start_time_original)
        self.SCS_stop_time_original = cr_bitstream.read("bytes:8")
        self.SCS_stop_time = convert_bytes_to_cds_time(self.SCS_stop_time_original)


class APIDFromConstructionRecord:
    """Object representation of a JPSS Construction Record"""

    def __init__(self, cr_bitstream: ConstBitStream):
        self.APID_fill_hold = []
        self.APID_fill_hold.append(cr_bitstream.read("uint:8"))
        self.APID_SCID_bits = cr_bitstream.read("bits:24")

        self.APID_byte_offset = cr_bitstream.read("uint:64")
        self.APID_fill_hold.append(cr_bitstream.read("uint:24"))

        # For this APID, identify the Virtual Channel Identification (VCID(s))
        self.APID_VCID_count = cr_bitstream.read("uint:8")
        self.VCIDs_list = []
        for count in range(self.APID_VCID_count):
            self.APID_fill_hold.append(cr_bitstream.read("uint:16"))
            self.VCIDs_list.append(VCIDFromConstructionRecord(cr_bitstream))

        # List missing packets SSCs for the PDS
        self.APID_SSC_discontinuity_count = cr_bitstream.read("uint:32")
        self.APID_SSC_discontinuities_list = []
        for count in range(self.APID_SSC_discontinuity_count):
            self.APID_SSC_discontinuities_list.append(SSCGapInformationFromConstructionRecord(cr_bitstream))

        # TODO this is not in DB? 24-7 -> 24-19
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
            self.SSC_length_discrepancy_list.append("uint:32")

        self.CCSDS_CDS_timecode_bytes = cr_bitstream.read("bytes:8")
        self.CCSDS_time_code_bytes = cr_bitstream.read("bytes:8")
        self.ESH_date_time_first_packet_bytes = cr_bitstream.read("bytes:8")
        self.ESH_date_time_last_packet_bytes = cr_bitstream.read("bytes:8")

        self.CCSDS_CDS_timecode_datetime = convert_bytes_to_cds_time(self.CCSDS_CDS_timecode_bytes)
        self.CCSDS_time_code_datetime = convert_bytes_to_cds_time(self.CCSDS_time_code_bytes)
        self.ESH_date_time_first_packet_datetime = convert_bytes_to_cds_time(self.ESH_date_time_first_packet_bytes)
        self.ESH_date_time_last_packet_datetime = convert_bytes_to_cds_time(self.ESH_date_time_last_packet_bytes)

        self.VCDU_error_packet_count = cr_bitstream.read("uint:32")
        # Check this out.... 24-17
        self.packet_count = cr_bitstream.read("uint:32")
        self.APID_size_octets = cr_bitstream.read("uint:64")
        self.APID_fill_hold.append(cr_bitstream.read("uint:64"))

    def print_scid_apid(self):
        dataset_scid = self.APID_SCID_bits.read("uint:8")
        fill_data = self.APID_SCID_bits.read("uint:5")
        dataset_apid = self.APID_SCID_bits.read("uint:11")
        print(f"SCID: {dataset_scid} and APID: {dataset_apid}")


class ConstructionRecordError(Exception):
    """Generic exception related to construction record file handling"""
    pass


class ConstructionRecord:
    """Object representation of a JPSS Construction Record"""

    def __init__(self, filepath: str or Path or S3Path):
        self.filepath = filepath
        with smart_open(filepath) as const_record_file:
            cr_bitstream = ConstBitStream(const_record_file)
            self.EDOS_version_original = cr_bitstream.read("bytes:2")
            self.EDOS_version_major = self.EDOS_version_original[0]
            self.EDOS_version_release = self.EDOS_version_original[1]
            # Construction Record type 1 is for PDS
            self.Construction_Record_type = cr_bitstream.read("uint:8")
            self.fill_hold = []
            self.fill_hold.append(cr_bitstream.read("uint:8"))
            self.PDS_ID = (cr_bitstream.read("bytes:36")).decode()
            self.fill_hold.append(cr_bitstream.read("uint:7"))
            self.test_flag = cr_bitstream.read("bool")
            self.fill_hold.append(cr_bitstream.read("uint:8"))
            self.fill_hold.append(cr_bitstream.read("uint:64"))

            # TODO Not in the DB (8)
            self.PDS_num_start_stop_times = cr_bitstream.read("uint:16")
            self.PDS_start_stop_times_list = []
            for count in range(self.PDS_num_start_stop_times):
                self.PDS_start_stop_times_list.append(PDSStartStopTimes(cr_bitstream))

            self.PDS_num_bytes_fill_data = cr_bitstream.read("uint:64")
            self.PDS_packet_length_mismatch_count = cr_bitstream.read("uint:32")
            self.PDS_CCSDS_timecode_first_bytes = cr_bitstream.read("bytes:8")
            self.PDS_CCSDS_timecode_first_datetime = convert_bytes_to_cds_time(self.PDS_CCSDS_timecode_first_bytes)
            self.PDS_CCSDS_timecode_last_bytes = cr_bitstream.read("bytes:8")
            self.PDS_CCSDS_timecode_last_datetime = convert_bytes_to_cds_time(self.PDS_CCSDS_timecode_last_bytes)
            self.PDS_EDOS_ESH_first_packet_bytes = cr_bitstream.read("bytes:8")
            self.PDS_EDOS_ESH_first_packet_datetime = convert_bytes_to_cds_time(self.PDS_EDOS_ESH_first_packet_bytes)
            self.PDS_EDOS_ESH_last_packet_bytes = cr_bitstream.read("bytes:8")
            self.PDS_EDOS_ESH_last_packet_datetime = convert_bytes_to_cds_time(self.PDS_EDOS_ESH_last_packet_bytes)
            self.PDS_rs_corrected_count = cr_bitstream.read("uint:32")
            self.PDS_packet_count = cr_bitstream.read("uint:32")
            self.PDS_size = cr_bitstream.read("uint:64")

            # TODO Missing in DB (20)
            self.PDS_discontinuities_count = cr_bitstream.read("uint:32")

            self.PDS_completion_time_bytes = cr_bitstream.read("bytes:8")
            self.PDS_completion_time_datetime = convert_bytes_to_cds_time(self.PDS_completion_time_bytes)
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
