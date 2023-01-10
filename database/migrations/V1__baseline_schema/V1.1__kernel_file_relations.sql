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
