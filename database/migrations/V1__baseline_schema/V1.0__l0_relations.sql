-- Information about PDS construction records
-- Note that all unsigned integer types have been upscaled to the next largest signed int type. e.g. uint16 -> int32
-- uint64 types with max value of 18,446,744,073,709,551,615 have been upscaled to NUMERIC(20) (arbitrary precision but slow)
CREATE TABLE sdp.cr (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    ingested TIMESTAMP WITH TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'),
    archived TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    edos_software_version INTEGER NOT NULL, --uint16
    construction_record_type INTEGER NOT NULL, --1B
    test_flag BOOLEAN NOT NULL,
    n_scs_start_stops INTEGER NOT NULL, --uint16
    n_bytes_fill_data NUMERIC(20) NOT NULL, --uint64
    n_length_mismatches BIGINT NOT NULL, --uint32
    first_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    last_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    first_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    last_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    first_packet_esh_time NUMERIC(20) NOT NULL, --64b CDS integer
    last_packet_esh_time NUMERIC(20) NOT NULL, --64b CDS integer
    n_rs_corrections BIGINT NOT NULL, --uint32
    n_packets BIGINT NOT NULL, --uint32
    size_bytes NUMERIC(20) NOT NULL, --uint64
    n_ssc_discontinuities INTEGER NOT NULL, --uint16
    completion_time NUMERIC(20) NOT NULL, --64b CDS time code
    n_apids SMALLINT NOT NULL, --1B
    n_pds_files SMALLINT NOT NULL
);
COMMENT ON TABLE sdp.cr IS 'Records of all ingested L0 data products and associated construction records from ASDC.';
COMMENT ON COLUMN sdp.cr.id IS 'PDS DB Record ID (primary key).';
COMMENT ON COLUMN sdp.cr.file_name IS '(4) Name of the associated PDS construction record.';
COMMENT ON COLUMN sdp.cr.ingested IS 'Timestamp when file was ingested.';
COMMENT ON COLUMN sdp.cr.archived IS 'Timestamp when file was moved to the archive bucket.';
COMMENT ON COLUMN sdp.cr.edos_software_version IS '(1) EDOS software version number. UINT16.';
COMMENT ON COLUMN sdp.cr.construction_record_type IS '(2) Value => 1=PDS.';
COMMENT ON COLUMN sdp.cr.test_flag IS '(6) 0 = operational data; 1 = test data';
COMMENT ON COLUMN sdp.cr.n_scs_start_stops IS '(8) Number of scheduled SCS start-stop times.';
COMMENT ON COLUMN sdp.cr.n_bytes_fill_data IS '(9) Number of bytes of EDOS generated fill data. UINT64.';
COMMENT ON COLUMN sdp.cr.n_length_mismatches IS '(10) Number of packets with discrepancy between packet length header item and actual length. UINT32.';
COMMENT ON COLUMN sdp.cr.first_packet_sc_time IS '(11) Spacecraft CDS time code out of first packet secondary header.';
COMMENT ON COLUMN sdp.cr.last_packet_sc_time IS '(12) Spacecraft CDS time code out of last packet secondary header.';
COMMENT ON COLUMN sdp.cr.first_packet_utc_time IS 'UTC representation of first packet time.';
COMMENT ON COLUMN sdp.cr.last_packet_utc_time IS 'UTC representation of last packet time.';
COMMENT ON COLUMN sdp.cr.first_packet_esh_time IS '(14) EDOS service header time code of first packet receipt time: days since 1958-01-01T00:00:00 (uint16), ms of day (uint32), us of ms (uint16).';
COMMENT ON COLUMN sdp.cr.last_packet_esh_time IS '(16) EDOS service header time code of last packet receipt time: days since 1958-01-01T00:00:00 (uint16), ms of day (uint32), us of ms (uint16).';
COMMENT ON COLUMN sdp.cr.n_rs_corrections IS '(17) Number of packets from VCDUs corrected by Reed-Solomon. UINT32.';
COMMENT ON COLUMN sdp.cr.n_packets IS '(18) Number of packets in the PDS file. UINT32.';
COMMENT ON COLUMN sdp.cr.size_bytes IS '(19) Number of total bytes in the PDS (possibly spanning multiple PDS files). UINT64.';
COMMENT ON COLUMN sdp.cr.n_ssc_discontinuities IS '(20) identify number of packets with SSC discontinuities.';
COMMENT ON COLUMN sdp.cr.completion_time IS '(22) CDS time code of actual time EDOS completed constructing the complete PDS.';
COMMENT ON COLUMN sdp.cr.n_apids IS '(24) Number of APIDs in the PDS. UINT8.';
COMMENT ON COLUMN sdp.cr.n_pds_files IS '(25-1) Number of files that this PDS resides in.';


-- Information about each APID referenced in a PDS construction record (may refer to multiple files)
CREATE TABLE sdp.cr_apid (
    id SERIAL PRIMARY KEY,
    cr_id INTEGER NOT NULL,
    FOREIGN KEY (cr_id) REFERENCES sdp.cr(id),
    scid BIGINT NOT NULL, --uint24
    apid BIGINT NOT NULL, --uint24
    byte_offset NUMERIC(20) NOT NULL, --uint64
    n_vcids SMALLINT NOT NULL, --uint8
    n_ssc_gaps BIGINT NOT NULL, --uint32
    n_edos_generated_fill_data BIGINT NOT NULL,--uint32
    count_edos_generated_octets NUMERIC(20) NOT NULL, --uint64
    n_length_discrepancy_packets BIGINT NOT NULL, --uint32
    first_packet_sc_time NUMERIC(20) NOT NULL, --64b CDS integer
    last_packet_sc_time NUMERIC(20) NOT NULL, --64b CDS integer
    first_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    last_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    esh_first_packet_time NUMERIC(20) NOT NULL, --64b CDS integer
    esh_last_packet_time NUMERIC(20) NOT NULL, --64b CDS integer
    n_vcdu_corrected_packets BIGINT NOT NULL, --uint32
    n_in_the_data_set BIGINT NOT NULL, --uint32
    n_octect_in_apid NUMERIC(20) NOT NULL --uint64
);
COMMENT ON TABLE sdp.cr_apid IS '(24) Records about individual APIDs described in a construction record.';
COMMENT ON COLUMN sdp.cr_apid.id IS 'Primary key.';
COMMENT ON COLUMN sdp.cr_apid.cr_id IS 'Foreign key to cr.id.';
COMMENT ON COLUMN sdp.cr_apid.scid IS '(24-2) SCID contained in the PDS as (SCID:uint8, fill:5b, apid:uint11).';
COMMENT ON COLUMN sdp.cr_apid.apid IS '(24-2) APID contained in the PDS as (SCID:uint8, fill:5b, apid:uint11).';
COMMENT ON COLUMN sdp.cr_apid.byte_offset IS '(24-3) Byte offset to first packet of APID within the dataset (possibly spanning multiple PDS files).';
COMMENT ON COLUMN sdp.cr_apid.n_vcids IS '(24-5) Number of VCIDs for this APID in this PDS.';
COMMENT ON COLUMN sdp.cr_apid.n_ssc_gaps IS '(24-6) Number of SSC discontinuities for this APID.';
COMMENT ON COLUMN sdp.cr_apid.n_edos_generated_fill_data IS '(24-7) Number of EDOS generated fill data for this APID.';
COMMENT ON COLUMN sdp.cr_apid.count_edos_generated_octets IS '(24-8) Count of octets of EDOS generated fill data for this APID.';
COMMENT ON COLUMN sdp.cr_apid.n_length_discrepancy_packets IS '(24-9) Number of packets that had length discrepancy between header and actual length this APID.';
COMMENT ON COLUMN sdp.cr_apid.first_packet_sc_time IS '(24-10) Spacecraft CDS time code out of first packet secondary header.';
COMMENT ON COLUMN sdp.cr_apid.last_packet_sc_time IS '(24-11) Spacecraft CDS time code out of last packet secondary header.';
COMMENT ON COLUMN sdp.cr_apid.first_packet_utc_time IS 'UTC representation of first packet time.';
COMMENT ON COLUMN sdp.cr_apid.last_packet_utc_time IS 'UTC representation of last packet time.';
COMMENT ON COLUMN sdp.cr_apid.esh_first_packet_time IS '(24-13) CDS time code for EDOS service header (ESH) time code of first packet receipt time.';
COMMENT ON COLUMN sdp.cr_apid.esh_last_packet_time IS '(24-15) CDS time code for EDOS service header (ESH) time code of last packet receipt time.';
COMMENT ON COLUMN sdp.cr_apid.n_vcdu_corrected_packets IS '(24-16) Count of packets from VCDUs with errors corrected by R-S decoding.';
COMMENT ON COLUMN sdp.cr_apid.n_in_the_data_set IS '(24-17) For this APID, in the data set.';
COMMENT ON COLUMN sdp.cr_apid.n_octect_in_apid IS '(24-18) For this APID, size (in count of packets octets)';


-- VCID associated with an APID within a CR
CREATE TABLE sdp.cr_apid_vcid (
    id SERIAL PRIMARY KEY,
    cr_apid_id INTEGER NOT NULL,
    FOREIGN KEY (cr_apid_id) REFERENCES sdp.cr_apid(id),
    scid_vcid INTEGER NOT NULL
);
COMMENT ON TABLE sdp.cr_apid_vcid IS '(24-5) The Virtual Channel Identification of an APID';
COMMENT ON COLUMN sdp.cr_apid_vcid.id IS 'Primary key.';
COMMENT ON COLUMN sdp.cr_apid_vcid.cr_apid_id IS 'Foreign key to cr_apid.id.';
COMMENT ON COLUMN sdp.cr_apid_vcid.scid_vcid IS '(24-5.2) VCDU-ID (SCID and VCID)';


-- Information about each source sequence counter gap described in a construction record
CREATE TABLE sdp.cr_apid_ssc_gap (
    id SERIAL PRIMARY KEY,
    cr_apid_id INTEGER NOT NULL,
    FOREIGN KEY (cr_apid_id) REFERENCES sdp.cr_apid(id),
    first_missing_ssc BIGINT NOT NULL,
    gap_byte_offset NUMERIC(20) NOT NULL,
    n_missing_sscs BIGINT NOT NULL,
    preceding_packet_sc_time NUMERIC(20) NOT NULL,
    following_packet_sc_time NUMERIC(20) NOT NULL,
    preceding_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    following_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    preceding_packet_esh_time NUMERIC(20) NOT NULL,
    following_packet_esh_time NUMERIC(20) NOT NULL
);
COMMENT ON TABLE sdp.cr_apid_ssc_gap IS '(24-6) Records of source sequence counter discontinuities for a specific APID within a PDS.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.id IS 'Primary key.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.cr_apid_id IS 'Foreign key to cr_apid.id.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.first_missing_ssc IS '(24-6.1) First missing SSC in gap.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.gap_byte_offset IS '(24-6.2) Byte offset pointing to packet with the same APID immediately after the SSC gap in the data set.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.n_missing_sscs IS '(24-6.3) Number of missing SSCs in the gap.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.preceding_packet_sc_time IS '(24-6.4) Spacecraft CDS time code in secondary header of packet with same APID that is immediately before the SSC gap.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.following_packet_sc_time IS '(24-6.5) Spacecraft CDS time code in secondary header of packet with same APID that is immediately after the SSC gap.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.preceding_packet_utc_time IS 'UTC representation of preceding packet sc time.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.following_packet_utc_time IS 'UTC representation of following packet sc time.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.preceding_packet_esh_time IS '(24-6.7) ESH time of packet with same APID that is immediately before the SSC gap.';
COMMENT ON COLUMN sdp.cr_apid_ssc_gap.following_packet_esh_time IS '(24-6.9) ESH time of packet with same APID that is immediately after the SSC gap.';


-- Information about PDS files which contain the actual L0 data
CREATE TABLE sdp.pds_file (
    id SERIAL PRIMARY KEY,
    cr_id INTEGER,  -- Nullable in case we get a PDS file but no construction record
    FOREIGN KEY (cr_id) REFERENCES sdp.cr(id),
    file_name TEXT UNIQUE NOT NULL,
    ingested TIMESTAMP WITH TIME ZONE DEFAULT NULL,  -- Nullable in case we get a CR for it but no the file itself.
    archived TIMESTAMP WITH TIME ZONE DEFAULT NULL
);
COMMENT ON TABLE sdp.pds_file IS '(25) Records of individual PDS files.';
COMMENT ON COLUMN sdp.pds_file.id IS 'Primary key.';
COMMENT ON COLUMN sdp.pds_file.cr_id IS 'Foreign key to cr.id.';
COMMENT ON COLUMN sdp.pds_file.file_name IS '(25-2) Name of Production Data Set file.';
COMMENT ON COLUMN sdp.pds_file.ingested IS 'Timestamp when file was first ingested. If NULL, indicates we parsed a CR containing it but never received the PDS file itself.';
COMMENT ON COLUMN sdp.pds_file.archived IS 'Timestamp when file was moved to the SDC archive bucket.';

-- Information about the APIDs contained in a single PDS file
CREATE TABLE sdp.pds_file_apid (
    id SERIAL PRIMARY KEY,
    pds_file_id INTEGER NOT NULL,
    FOREIGN KEY (pds_file_id) REFERENCES sdp.pds_file(id),
    scid BIGINT NOT NULL,
    apid BIGINT NOT NULL,
    first_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    last_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    first_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    last_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL
);
COMMENT ON TABLE sdp.pds_file_apid IS '(25-4) APID information in a specific PDS file.';
COMMENT ON COLUMN sdp.pds_file_apid.id IS 'Primary key.';
COMMENT ON COLUMN sdp.pds_file_apid.pds_file_id IS 'Foreign key to pds_file.id.';
COMMENT ON COLUMN sdp.pds_file_apid.scid IS '(25-4.2) SCID  in PDS file.';
COMMENT ON COLUMN sdp.pds_file_apid.apid IS '(25-4.2) APID in PDS file.';
COMMENT ON COLUMN sdp.pds_file_apid.first_packet_sc_time IS '(25-4.3) Spacecraft CDS time code out of first packet secondary header.';
COMMENT ON COLUMN sdp.pds_file_apid.last_packet_sc_time IS '(25-4.4) Spacecraft CDS time code out of last packet secondary header.';
COMMENT ON COLUMN sdp.pds_file_apid.first_packet_utc_time IS 'First packet UTC time.';
COMMENT ON COLUMN sdp.pds_file_apid.last_packet_utc_time IS 'Last packet UTC time.';


--Information about each Spacecraft Session (SCS) start and stop times in a PDS construction record (Matt added)
CREATE TABLE sdp.cr_scs_start_stop_times (
    id SERIAL PRIMARY KEY,
    cr_id INTEGER,
    FOREIGN KEY (cr_id) REFERENCES sdp.cr(id),
    scs_start_sc_time NUMERIC(20) NOT NULL, --64b integer
    scs_stop_sc_time NUMERIC(20) NOT NULL, --64b integer
    scs_start_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    scs_stop_utc_time TIMESTAMP WITH TIME ZONE NOT NULL
);
COMMENT ON TABLE sdp.cr_scs_start_stop_times IS '(8) SCS start and stop times in a PDS.';
COMMENT ON COLUMN sdp.cr_scs_start_stop_times.id IS 'Primary key.';
COMMENT ON COLUMN sdp.cr_scs_start_stop_times.cr_id IS 'Foreign key to cr.id.';
COMMENT ON COLUMN sdp.cr_scs_start_stop_times.scs_start_sc_time IS '(8-2)  Spacecraft CDS time code of start.';
COMMENT ON COLUMN sdp.cr_scs_start_stop_times.scs_stop_sc_time IS '(8-3) Spacecraft CDS time code of stop.';
COMMENT ON COLUMN sdp.cr_scs_start_stop_times.scs_start_utc_time IS 'Start time UTC time.';
COMMENT ON COLUMN sdp.cr_scs_start_stop_times.scs_stop_utc_time IS 'Stop time in UTC time.';


--Information about each EDOS generated fill data in an APID in a PDS construction record (Matt added)
CREATE TABLE sdp.cr_apid_edos_generated_fill_data (
    id SERIAL PRIMARY KEY,
    cr_apid_id INTEGER NOT NULL,
    FOREIGN KEY (cr_apid_id) REFERENCES sdp.cr_apid(id),
    ssc_with_generated_data BIGINT NOT NULL, --uint32
    filled_byte_offset NUMERIC(20) NOT NULL, --64b integer
    index_to_fill_octet BIGINT NOT NULL --uint32
);
COMMENT ON TABLE sdp.cr_apid_edos_generated_fill_data IS '(24-7) Records of packets with EDOS generated fill data for an APID.';
COMMENT ON COLUMN sdp.cr_apid_edos_generated_fill_data.id IS 'Primary key.';
COMMENT ON COLUMN sdp.cr_apid_edos_generated_fill_data.cr_apid_id IS 'Foreign key to cr_apid.id.';
COMMENT ON COLUMN sdp.cr_apid_edos_generated_fill_data.ssc_with_generated_data IS '(24-7.1) SSC of packets with generated fill data.';
COMMENT ON COLUMN sdp.cr_apid_edos_generated_fill_data.filled_byte_offset IS '(24-7.2) Index (byte offset) into the data set to the fill packet.';
COMMENT ON COLUMN sdp.cr_apid_edos_generated_fill_data.index_to_fill_octet IS '(24-7.3) Index to the first fill octet for the above packet.';


-- Information about SSC length discrepancy (Matt added)
CREATE TABLE sdp.cr_apid_ssc_len_discrepancies (
    id SERIAL PRIMARY KEY,
    cr_apid_id INTEGER NOT NULL,
    FOREIGN KEY (cr_apid_id) REFERENCES sdp.cr_apid(id),
    ssc_length_discrepancy INTEGER NOT NULL
);
COMMENT ON TABLE sdp.cr_apid_ssc_len_discrepancies IS '(24-9) The Virtual Channel Identification of an APID';
COMMENT ON COLUMN sdp.cr_apid_ssc_len_discrepancies.id IS 'Primary key.';
COMMENT ON COLUMN sdp.cr_apid_ssc_len_discrepancies.cr_apid_id IS 'Foreign key to cr_apid.id.';
COMMENT ON COLUMN sdp.cr_apid_ssc_len_discrepancies.ssc_length_discrepancy IS '(24-9.1) SSC of packet with length discrepancy';
