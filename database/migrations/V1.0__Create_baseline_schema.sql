-- Information about PDS construction records
-- Note that all unsigned integer types have been upscaled to the next largest signed int type. e.g. uint16 -> int32
-- uint64 types with max value of 18,446,744,073,709,551,615 have been upscaled to NUMERIC(20) (arbitrary precision but slow)
CREATE TABLE cr (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    ingested TIMESTAMP WITH TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'),
    archived TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    edos_software_version INTEGER NOT NULL, --uint16
    construction_record_type INTEGER NOT NULL, --1B
    test_flag BOOLEAN NOT NULL,
    n_bytes_fill_data NUMERIC(20) NOT NULL, --uint64
    n_length_mismatches BIGINT NOT NULL, --uint32
    first_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    last_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    first_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    last_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    first_packet_esh_time NUMERIC(20) NOT NULL, --64b CDS time code
    last_packet_esh_time NUMERIC(20) NOT NULL, --64b CDS time code
    n_rs_corrections BIGINT NOT NULL, --uint32
    n_packets BIGINT NOT NULL, --uint32
    size_bytes NUMERIC(20) NOT NULL, --uint64
    completion_time NUMERIC(20) NOT NULL, --64b CDS time code
    n_apids SMALLINT NOT NULL, --1B
    n_pds_files SMALLINT NOT NULL
);
COMMENT ON TABLE cr IS 'Records of all ingested L0 data products and associated construction records from ASDC.';
COMMENT ON COLUMN cr.id IS 'PDS file ID (primary key).';
COMMENT ON COLUMN cr.file_name IS 'Name of the associated PDS construction record.';
COMMENT ON COLUMN cr.ingested IS 'Timestamp when file was ingested.';
COMMENT ON COLUMN cr.archived IS 'Timestamp when file was moved to the archive bucket.';
COMMENT ON COLUMN cr.edos_software_version IS '(1) EDOS software version number. UINT16.';
COMMENT ON COLUMN cr.construction_record_type IS '(2) Value => 1=PDS.';
COMMENT ON COLUMN cr.test_flag IS '(6) 0 = operational data; 1 = test data';
COMMENT ON COLUMN cr.n_bytes_fill_data IS '(9) Number of bytes of EDOS generated fill data. UINT64.';
COMMENT ON COLUMN cr.n_length_mismatches IS '(10) Number of packets with discrepancy between packet length header item and actual length. UINT32.';
COMMENT ON COLUMN cr.first_packet_sc_time IS '(11) Spacecraft CDS time code out of first packet secondary header.';
COMMENT ON COLUMN cr.last_packet_sc_time IS '(12) Spacecraft CDS time code out of last packet secondary header.';
COMMENT ON COLUMN cr.first_packet_utc_time IS 'UTC representation of first packet time.';
COMMENT ON COLUMN cr.last_packet_utc_time IS 'UTC representation of last packet time.';
COMMENT ON COLUMN cr.first_packet_esh_time IS '(14) EDOS service header time code of first packet receipt time: days since 1958-01-01T00:00:00 (uint16), ms of day (uint32), us of ms (uint16).';
COMMENT ON COLUMN cr.last_packet_esh_time IS '(16) EDOS service header time code of last packet receipt time: days since 1958-01-01T00:00:00 (uint16), ms of day (uint32), us of ms (uint16).';
COMMENT ON COLUMN cr.n_rs_corrections IS '(17) Number of packets from VCDUs corrected by Reed-Solomon. UINT32.';
COMMENT ON COLUMN cr.n_packets IS '(18) Number of packets in the PDS file. UINT32.';
COMMENT ON COLUMN cr.size_bytes IS '(19) Number of total bytes in the PDS (possibly spanning multiple PDS files). UINT64.';
COMMENT ON COLUMN cr.completion_time IS '(22) CDS time code of actual time EDOS completed constructing the complete PDS.';
COMMENT ON COLUMN cr.n_apids IS '(24) Number of APIDs in the PDS. UINT8.';
COMMENT ON COLUMN cr.n_pds_files IS '(25-1) Number of files that this PDS resides in.';


-- Information about each APID referenced in a PDS construction record (may refer to multiple files)
CREATE TABLE cr_apid (
    id SERIAL PRIMARY KEY,
    cr_id INTEGER NOT NULL,
    FOREIGN KEY (cr_id) REFERENCES cr(id),
    scid_apid BIGINT NOT NULL, --uint24
    byte_offset NUMERIC(20) NOT NULL, --uint64
    n_vcids SMALLINT NOT NULL, --uint8
    n_ssc_discontinuities BIGINT NOT NULL --uint32
);
COMMENT ON TABLE cr_apid IS '(24) Records about individual APIDs described in a construction record.';
COMMENT ON COLUMN cr_apid.id IS 'Primary key.';
COMMENT ON COLUMN cr_apid.cr_id IS 'Foreign key to cr.id.';
COMMENT ON COLUMN cr_apid.scid_apid IS '(24-2) SCID and APID contained in the PDS as (SCID:uint8, fill:5b, apid:uint11).';
COMMENT ON COLUMN cr_apid.byte_offset IS '(24-3) Byte offset to first packet of APID within the dataset (possibly spanning multiple PDS files).';
COMMENT ON COLUMN cr_apid.n_vcids IS '(24-5) Number of VCIDs for this APID in this PDS.';
COMMENT ON COLUMN cr_apid.n_ssc_discontinuities IS '(24-6) Number of SSC discontinuities for this APID.';


-- VCID associated with an APID within a CR
CREATE TABLE cr_apid_vcid (
    id SERIAL PRIMARY KEY,
    cr_apid_id INTEGER NOT NULL,
    FOREIGN KEY (cr_apid_id) REFERENCES cr_apid(id),
    scid_vcid INTEGER NOT NULL
);


-- Information about each source sequence counter gap described in a construction record
CREATE TABLE cr_apid_ssc_gap (
    id SERIAL PRIMARY KEY,
    cr_apid_id INTEGER NOT NULL,
    FOREIGN KEY (cr_apid_id) REFERENCES cr_apid(id),
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
COMMENT ON TABLE cr_apid_ssc_gap IS '(24-6) Records of source sequence counter discontinuities for a specific APID within a PDS.';
COMMENT ON COLUMN cr_apid_ssc_gap.id IS 'Primary key.';
COMMENT ON COLUMN cr_apid_ssc_gap.cr_apid_id IS 'Foreign key to cr_apid.id.';
COMMENT ON COLUMN cr_apid_ssc_gap.first_missing_ssc IS '(24-6.1) First missing SSC in gap.';
COMMENT ON COLUMN cr_apid_ssc_gap.gap_byte_offset IS '(24-6.2) Byte offset pointing to packet with the same APID immediately after the SSC gap in the data set.';
COMMENT ON COLUMN cr_apid_ssc_gap.n_missing_sscs IS '(24-6.3) Number of missing SSCs in the gap.';
COMMENT ON COLUMN cr_apid_ssc_gap.preceding_packet_sc_time IS '(24-6.4) Spacecraft CDS time code in secondary header of packet with same APID that is immediately before the SSC gap.';
COMMENT ON COLUMN cr_apid_ssc_gap.following_packet_sc_time IS '(24-6.5) Spacecraft CDS time code in secondary header of packet with same APID that is immediately after the SSC gap.';
COMMENT ON COLUMN cr_apid_ssc_gap.preceding_packet_utc_time IS 'UTC representation of preceding packet sc time.';
COMMENT ON COLUMN cr_apid_ssc_gap.following_packet_utc_time IS 'UTC representation of following packet sc time.';
COMMENT ON COLUMN cr_apid_ssc_gap.preceding_packet_esh_time IS '(24-6.7) ESH time of packet with same APID that is immediately before the SSC gap.';
COMMENT ON COLUMN cr_apid_ssc_gap.following_packet_esh_time IS '(24-6.9) ESH time of packet with same APID that is immediately after the SSC gap.';


-- Information about PDS files which contain the actual L0 data
CREATE TABLE pds_file (
    id SERIAL PRIMARY KEY,
    cr_id INTEGER,  -- Nullable in case we get a PDS file but no construction record
    FOREIGN KEY (cr_id) REFERENCES cr(id),
    file_name TEXT UNIQUE NOT NULL,
    ingested TIMESTAMP WITH TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'),
    archived TIMESTAMP WITH TIME ZONE DEFAULT NULL
);
COMMENT ON TABLE pds_file IS '(25) Records of individual PDS files.';
COMMENT ON COLUMN pds_file.id IS 'Primary key.';
COMMENT ON COLUMN pds_file.cr_id IS 'Foreign key to cr.id.';
COMMENT ON COLUMN pds_file.file_name IS '(25-2) Name of Production Data Set file.';
COMMENT ON COLUMN pds_file.ingested IS 'Timestamp when file was first ingested (when the record was created).';
COMMENT ON COLUMN pds_file.archived IS 'Timestamp when file was moved to the SDC archive bucket.';


-- Information about the APIDs contained in a single PDS file
CREATE TABLE pds_file_apid (
    id SERIAL PRIMARY KEY,
    pds_file_id INTEGER NOT NULL,
    FOREIGN KEY (pds_file_id) REFERENCES pds_file(id),
    scid_apid BIGINT NOT NULL,
    first_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    last_packet_sc_time NUMERIC(20) NOT NULL, --64b integer
    first_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    last_packet_utc_time TIMESTAMP WITH TIME ZONE NOT NULL
);
COMMENT ON TABLE pds_file_apid IS '(25-4) APID information in a specific PDS file.';
COMMENT ON COLUMN pds_file_apid.id IS 'Primary key.';
COMMENT ON COLUMN pds_file_apid.pds_file_id IS 'Foreign key to pds_file.id.';
COMMENT ON COLUMN pds_file_apid.scid_apid IS '(25-4.2) SCID and APID in PDS file.';
COMMENT ON COLUMN pds_file_apid.first_packet_sc_time IS '(25-4.3) Spacecraft CDS time code out of first packet secondary header.';
COMMENT ON COLUMN pds_file_apid.last_packet_sc_time IS '(25-4.4) Spacecraft CDS time code out of last packet secondary header.';
COMMENT ON COLUMN pds_file_apid.first_packet_utc_time IS 'First packet UTC time.';
COMMENT ON COLUMN pds_file_apid.last_packet_utc_time IS 'Last packet UTC time.';


-- Generated SPICE SPK and CK spk_ck_file records.
CREATE TABLE spk_ck_file (
    id SERIAL PRIMARY KEY,
    file_name TEXT UNIQUE NOT NULL,
    start_sc_time NUMERIC(20) NOT NULL,
    stop_sc_time NUMERIC(20) NOT NULL,
    start_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    stop_utc_time TIMESTAMP WITH TIME ZONE NOT NULL,
    archived TIMESTAMP WITH TIME ZONE DEFAULT (now() AT TIME ZONE 'UTC'),
    revision INTEGER NOT NULL
);
COMMENT ON TABLE spk_ck_file IS 'Records of generated SPK (ephemeris) and CK (attitude) spk_ck_files';
COMMENT ON COLUMN spk_ck_file.id IS 'Primary key.';
COMMENT ON COLUMN spk_ck_file.file_name IS 'Time of first data point in the spk_ck_file.';
COMMENT ON COLUMN spk_ck_file.start_sc_time IS 'Time of first data point in the spk_ck_file in spacecraft time code.';
COMMENT ON COLUMN spk_ck_file.stop_sc_time IS 'Time of last data point in the spk_ck_file in spacecraft time code.';
COMMENT ON COLUMN spk_ck_file.start_utc_time IS 'Time of first data point in the spk_ck_file in UTC time.';
COMMENT ON COLUMN spk_ck_file.stop_utc_time IS 'Time of last data point in the spk_ck_file in UTC time.';
COMMENT ON COLUMN spk_ck_file.archived IS 'Time the spk_ck_file was moved to the archive bucket.';
COMMENT ON COLUMN spk_ck_file.revision IS '';


-- n:m relationship join table between spk_ck_file files and PDS files.
CREATE TABLE spk_ck_file_pds_file_jt (
    spk_ck_file_id INTEGER NOT NULL,
    FOREIGN KEY (spk_ck_file_id) REFERENCES spk_ck_file(id),
    pds_file_id INTEGER NOT NULL,
    FOREIGN KEY (pds_file_id) REFERENCES pds_file(id),
    PRIMARY KEY (spk_ck_file_id, pds_file_id)
);
COMMENT ON TABLE spk_ck_file_pds_file_jt IS 'n:m jointable between spk_ck_file files and PDS files.';
COMMENT ON COLUMN spk_ck_file_pds_file_jt.spk_ck_file_id IS 'Foreign key to spk_ck_file.id';
COMMENT ON COLUMN spk_ck_file_pds_file_jt.pds_file_id IS 'Foreign key to pds_file.id';


CREATE TABLE l1b_cam ();


CREATE TABLE l1b_rad ();
