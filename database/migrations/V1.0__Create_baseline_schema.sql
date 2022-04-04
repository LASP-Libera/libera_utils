CREATE TABLE l0 (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    -- We may want to add more fields parsed from the filename
    version SMALLINT NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT (now() AT TIME ZONE 'UTC'),
    ingested TIMESTAMPTZ
);
COMMENT ON TABLE l0 IS 'Level 0 data files.';
COMMENT ON COLUMN l0.id IS 'File ID.';
COMMENT ON COLUMN l0.filename IS 'Level 0 filename.';
COMMENT ON COLUMN l0.version IS 'Version of level 0 file.';
COMMENT ON COLUMN l0.created IS 'Timestamp of record creation.';
COMMENT ON COLUMN l0.ingested IS 'Timestamp of successful ingest. NULL indicates ingest failure.';


CREATE TABLE apid (
    apid SMALLINT PRIMARY KEY,
    description TEXT NOT NULL
);
COMMENT ON TABLE apid IS 'Description of Application Process ID packet contents.';
COMMENT ON COLUMN apid.apid IS 'Application Process ID.';
COMMENT ON COLUMN apid.description IS 'Description of packet contents.';


CREATE TABLE packet (
    id SERIAL PRIMARY KEY,
    version_number SMALLINT NOT NULL,  -- 3 bits (always 000)
    type BOOLEAN NOT NULL,  -- 1 bit
    secondary_header_flag BOOLEAN NOT NULL,  -- 1 bit
    apid SMALLINT NOT NULL, -- 11 bits
    sequence_flags BIT(2) NOT NULL,  -- 2 bits
    sequence_count SMALLINT NOT NULL,  -- 14 bits (unique within APID)
    data_length SMALLINT NOT NULL, -- 16 bits (length of packet data field in octets - 1)
    -- We can't open up the secondary header or user data unless we split packet storage into separate tables
    secondary_header BYTEA,  -- Secondary header binary data. May be null if secondary header flag is false
    user_data BYTEA NOT NULL,  -- Science data
    FOREIGN KEY (apid) REFERENCES apid(apid)
);
COMMENT ON TABLE packet IS 'Packet data.';
COMMENT ON COLUMN packet.id IS 'Surrogate primary key.';
COMMENT ON COLUMN packet.version_number IS 'Always `000` until CCSDS updates the standard.';
COMMENT ON COLUMN packet.type IS '`0` => Telemetry; `1` => Command.';
COMMENT ON COLUMN packet.secondary_header_flag IS '`0` => Absent; `1` => Present. Invariant within APID.';
COMMENT ON COLUMN packet.apid IS 'Application Process ID.';
COMMENT ON COLUMN packet.sequence_flags IS '`00` => Continuation; `01` => First segment; `10` => Last segment; `11` => Unsegmented.';
COMMENT ON COLUMN packet.sequence_count IS 'Sequential packet count within APID (may wrap for long missions or due to reset).';
COMMENT ON COLUMN packet.data_length IS 'Number of octets in Packet Data Field - 1 (secondary header and user data).';
COMMENT ON COLUMN packet.secondary_header IS 'Binary data comprising secondary header.';
COMMENT ON COLUMN packet.user_data IS 'Binary data comprising science data.';


CREATE TABLE l0_pkt_jt (
    l0_id INTEGER NOT NULL,
    pkt_id INTEGER NOT NULL,
    FOREIGN KEY (l0_id) REFERENCES l0(id) ON DELETE CASCADE,
    FOREIGN KEY (pkt_id) REFERENCES packet(id) ON DELETE CASCADE,
    CONSTRAINT l0_pkt_jt_pk PRIMARY KEY (l0_id, pkt_id)
);
COMMENT ON TABLE l0_pkt_jt IS 'Join-table for n:m relationship between packets and L0 files.';
COMMENT ON COLUMN l0_pkt_jt.l0_id IS 'Level 0 file ID.';
COMMENT ON COLUMN l0_pkt_jt.pkt_id IS 'Packet ID.';


CREATE TABLE l1b (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    version SMALLINT NOT NULL,
    created TIMESTAMPTZ NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')
    -- We will likely want to add lots of additional metadata here to help make our data searchable
);
COMMENT ON TABLE l1b IS 'Level 1b product.';
COMMENT ON COLUMN l1b.id IS 'L1b product ID.';
COMMENT ON COLUMN l1b.filename IS 'L1b filename.';
COMMENT ON COLUMN l1b.created IS 'Record creation timestamp.';


CREATE TABLE l1b_pkt_jt (
    l1b_id INTEGER NOT NULL,
    pkt_id INTEGER NOT NULL,
    FOREIGN KEY (l1b_id) REFERENCES l1b(id) ON DELETE CASCADE,
    FOREIGN KEY (pkt_id) REFERENCES packet(id) ON DELETE CASCADE
)
